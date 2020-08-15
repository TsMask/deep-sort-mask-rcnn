from mrcnn.config import Config
import mrcnn.model as modellib
import numpy as np
import cv2
import random
import base64
import colorsys
import os

# 识别80类
COCO_CLASSES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# 预测配置
class InferenceConfig(Config):
    # 可辨识的名称
    NAME = "my_inference"

    # GPU的数量和每个GPU处理的图片数量，可以根据实际情况进行调整，参考为Nvidia Tesla P100
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 物体的分类个数，COCO中共有80种物体+背景
    NUM_CLASSES = 1 + 80  # background + 80 shapes

class MRCNN(object):
    def __init__(self, model_path, image_size, min_score):
        self.gpu_num = 1
        self.image_size = image_size
        self.score = min_score
        self.class_names = COCO_CLASSES
        self.colors = self._random_colors(len(COCO_CLASSES))
        self.model = self._model_load(model_path)

    # 载入模型
    def _model_load(self, model_path="model_data/mask_rcnn_coco.h5"):
        # 配置参数
        inference_config = InferenceConfig()
        # 图片尺寸统一处理为1024，可以根据实际情况再进一步调小
        inference_config.IMAGE_MIN_DIM = self.image_size
        inference_config.IMAGE_MAX_DIM = self.image_size
        inference_config.display()

        # 模型预测对象
        inference_model = modellib.MaskRCNN(mode="inference",
                                            config=inference_config,
                                            model_dir="logs")

        # 训练权重
        inference_model.load_weights(model_path, by_name=True)
        inference_model.keras_model._make_predict_function()

        return inference_model
    
    # 随机颜色
    def _random_colors(self, N):
        hsv_tuples = [(1.0 * x / N, 1., 1.) for x in range(N)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10)
        random.shuffle(colors)
        random.seed(None)
        return colors

    # 颜色覆盖物体
    def apply_mask(self, image, mask, color, alpha=0.4):
        for n, c in enumerate(color):
            image[:, :, n] = np.where(
                mask == 1,
                image[:, :, n] * (1 - alpha) + alpha * c,
                image[:, :, n]
            )
        return image

    # 识别结果
    def detect_result(self, image, min_score=0.2):
        # 模型识别结果 rois, masks, class_ids, scores
        results = self.model.detect([image], verbose=0)[0]
        # 结果参数进行操作绘制
        boxes = results['rois']
        masks = results['masks']
        class_ids = results['class_ids']
        classes_scores = results['scores']
        
        # 数据段值
        return_boxes = []
        return_scores = []
        return_masks = []
        return_class_names = []
        return_class_color = []
        
        # 数据段值分类添加
        for i, box in enumerate(boxes):
            class_id = class_ids[i]             # 类别下标
            classes_score = classes_scores[i]   # 类别识别分数

            # 检测分小于跳过
            if classes_score < min_score: continue

            # 边框四点坐标
            y1, x1, y2, x2 = box
            return_boxes.append([x1, y1, (x2 - x1), (y2 - y1)])
            return_scores.append(classes_score)
            return_masks.append(masks[:, :, i])                          # 类别掩膜
            return_class_names.append(self.class_names[class_id])        # 类别标签名称
            return_class_color.append(self.colors[class_id])             # 类别所属颜色

        return return_boxes, return_scores, return_class_names, return_masks, return_class_color

# 点在里面
def isInSide(point, box):
    # print(box[0] <= point[0] <= box[2] , box[1] <= point[1] <= box[3])
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
