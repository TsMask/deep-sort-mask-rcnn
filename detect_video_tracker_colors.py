#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import json
import argparse
import sys, os
import cv2
import numpy as np
from PIL import Image
from moxing.framework import file

# deep-sort跟踪
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections
from collections import deque

# mask-rcnn模型 - 固定类型颜色
from mrcnn.mrcnn_colors import MRCNN, isInSide

# obs桶路径
obs_path = "obs://puddings/deep-sort-mask-rcnn/cailiao"

# 输出目录
out_path = "cailiao"

# 输出目录存在需要删除里边的内容
if os.path.exists(out_path):
    file.remove(out_path, recursive=True)
os.makedirs(out_path)

# 运动轨迹
pts = [deque(maxlen=30) for _ in range(9999)]

# 跟踪统计
track_total = []

# 跟踪类型总数量
total_count = {}

# 帧数，用于通过帧数取图
frameNum = 0

# 执行参数 python detect_video_tracker_colors.py --video_file test.mp4 --min_score 0.3 --input_size 1024 --model_file model_data/train_mask_rcnn.h5 --model_feature model_data/mars-small128.pb
# 外部参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, default='test.mp4', help='data mp4 file.')
parser.add_argument('--min_score', type=float, default=0.3, help='displays the lowest tracking score.')
parser.add_argument('--input_size', type=int, default=1024, help='input pic size.')
parser.add_argument('--model_file', type=str, default='model_data/mask_rcnn_coco.h5', help='Object detection model file.')
parser.add_argument('--model_feature', type=str, default='model_data/market1501.pb', help='target tracking model file.')
ARGS = parser.parse_args()

box_size = 2        # 边框大小
font_scale = 0.4    # 字体比例大小

if __name__ == '__main__':
    # Deep SORT 跟踪器
    encoder = generate_detections.create_box_encoder(ARGS.model_feature, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", ARGS.min_score, None)
    tracker = Tracker(metric)

    # 载入模型
    mrcnn = MRCNN(ARGS.model_file, ARGS.input_size, ARGS.min_score)

    # 读取视频
    video = cv2.VideoCapture(ARGS.video_file)

    # 输出保存视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_out = cv2.VideoWriter(out_path + "/outputVideo.mp4", fourcc, fps, size)

    # 视频是否可以打开，进行逐帧识别绘制
    while video.isOpened:
        # 视频读取图片帧
        retval, frame = video.read()
        if retval:
            frame_orig = frame.copy()
        else:
            print("没有图像！尝试使用其他视频")
            break

        prev_time = time.time()

        # 识别结果
        boxes, scores, classes, masks, colors = mrcnn.detect_result(frame, min_score)

        # 特征提取和检测对象列表
        features = encoder(frame, boxes)
        detections = []
        for bbox, score, classe, mask, color, feature in zip(boxes, scores, classes, masks, colors, features):
            detections.append(Detection(bbox, score, classe, mask, color, feature))

        # 运行非最大值抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.score for d in detections])
        indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
        detections = [detections[i] for i in indices]

        # 遍历绘制检测对象信息
        detect_count = {}
        detect_temp = []
        for det in detections:
            y1, x1, y2, x2 = np.array(det.to_tlbr(), dtype=np.int32)
            caption = '{} {:.2f}'.format(det.classe, det.score) if det.classe else det.score

            frame = mrcnn.apply_mask(frame, det.mask, det.color, 0.3)         # 类别掩膜颜色透明度
            for c in range(3): det.color += (int(det.color[c]*255),)          # 颜色
            cv2.rectangle(frame, (y1, x1), (y2, x2), det.color[3:], box_size) # 绘制类别边框

            # 中心点
            point = (int((y1+y2)/2),int((x1+x2)/2))
            # cv2.circle(frame, point, 1, det.color[3:], box_size)

            # 类别文字显示
            cv2.putText(
                frame,
                caption,
                (y1, x1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, det.color[3:],
                box_size//2,
                lineType=cv2.LINE_AA
            )
            # 统计物体数
            if det.classe not in detect_count: detect_count[det.classe] = 0
            detect_count[det.classe] += 1
            detect_temp.append([det.classe, det.color[3:], point])

        # 追踪器刷新
        tracker.predict()
        tracker.update(detections)

        # 遍历绘制跟踪信息
        track_count = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            y1, x1, y2, x2 = np.array(track.to_tlbr(), dtype=np.int32)
            # cv2.rectangle(frame, (y1, x1), (y2, x2), (255, 255, 255), box_size//4)

            # 跟踪统计数量
            track_total.append(track.track_id)
            track_count += 1

            # 运动点轨迹
            point = (int((y1+y2)/2),int((x1+x2)/2))
            # cv2.circle(frame, point, 1, (255, 255, 255), box_size)
            pts[track.track_id].append(point)
            # 在识别类中标记跟踪 [ classe, color , point ]
            for d in range(len(detect_temp)):
                # 非标记目标跳过
                if not isInSide(detect_temp[d][2], track.to_tlbr()): continue

                # 总统计数量
                if detect_temp[d][0] not in total_count: total_count[detect_temp[d][0]] = [0, []]
                if track.track_id not in total_count[detect_temp[d][0]][1]:
                    total_count[detect_temp[d][0]][0] += 1
                    total_count[detect_temp[d][0]][1].append(track.track_id)
                    # 输出小图目录,不存目录需要创建
                    label_path = os.path.join(out_path, "{0}/{1}".format('imageSeg', detect_temp[d][0]))
                    if not os.path.exists(label_path): os.makedirs(label_path)
                    cv2.imwrite("{0}/{1}.jpg".format(label_path, track.track_id), frame_orig[x1:x2, y1:y2])

                # 跟踪标记号码
                cv2.putText(
                    frame, 
                    "No. " + str(track.track_id),
                    (y1, x1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255),
                    box_size//2,
                    lineType=cv2.LINE_AA
                )

                # 绘制运动路径
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None: continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame, (pts[track.track_id][j-1]), (pts[track.track_id][j]), detect_temp[d][1], thickness)

        # 跟踪统计
        trackTotalStr = 'Track Total: %s' % str(len(set(track_total)))
        cv2.putText(frame, trackTotalStr, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (244, 67, 54), 1, cv2.LINE_AA)

        # 跟踪数量
        trackCountStr = 'Track Count: %s' % str(track_count)
        cv2.putText(frame, trackCountStr, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 193, 7), 1, cv2.LINE_AA)

        # 识别类数统计
        totalStr = ""
        for k in detect_count.keys(): totalStr += '%s: %d    ' % (k, detect_count[k])
        cv2.putText(frame, totalStr, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)

        for i, label in enumerate(total_count):
            labelTotal = '%s: %d ' % (label, total_count[label][0])
            cv2.putText(frame, labelTotal, (20, 80 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 87, 34), 1, cv2.LINE_AA)

        # 绘制时间
        curr_time = time.time()
        exec_time = curr_time - prev_time
        print("识别帧：{:.0f}/{:.0f} , 识别耗时: {:.2f} ms".format(frameNum, video.get(7), 1000*exec_time))

        frameNum += 1
        # 视频输出逐帧保存
        video_out.write(frame)
        # 绘制视频显示窗
        # result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.namedWindow("video_reult", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("video_reult", result)
        # 退出窗口
        # if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 任务完成后释放所有内容
    video.release()
    video_out.release()
    # cv2.destroyAllWindows()

    # 打开文件统计后遍历物体结果数据
    totalFile = open(out_path + "/totalCount.txt","w")
    # 统计数量写入文件txt
    for label in total_count.keys():
        labelTotal = "{0}：{1} \n".format(label, total_count[label][0])
        totalFile.write(labelTotal)
    # 关闭文件统计
    totalFile.close()

    # 统计写入文件josn
    with open(out_path + "/totalCount.json", 'w') as tc:
        json.dump(total_count, tc)

    # 复制保存到桶
    file.copy_parallel(out_path, obs_path)
