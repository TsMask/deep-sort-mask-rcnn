# Deep SORT —— Mask R-CNN 实例分割目标检测跟踪

## 介绍

项目使用华为云 `ModelArts` AI 开发平台进行训练部署，采用 `Mask R-CNN` 算法模型进行目标检测，使用 `Deep SORT` 目标跟踪算法。`Mask R-CNN` 源于2018年论文《Mask R-CNN》，是何恺明团队作品。`Mask R-CNN` 指的是在检测出图片中物体的同时为每一实例产生高质量的分割掩码（segmentation mask），算法基于 `Faster R-CNN`，在其中添加了实例掩码功能。

**运行环境**

华为云 `ModelArts` AI 开发平台开发环境 `notebook - TensorFlow-1.13.1` 进行调试，可以上传到自己的开发环境中使用。自己机器需要修改一些路径问题，Windows平台安装 `pycocotools` 依赖是个问题，建议Linux系统下安装配置环境后使用。

## 目录结构

将目录文件夹上传至已创建的桶中，文件过多可能上传失败，建议分目录选择上传。

```text
deep-sort-mask-rcnn

┌─── deep_sort                        DeepSort目标跟踪算法
│    ├── detection.py
│    ├── generate_detections.py
│    ├── iou_matching.py
│    ├── kalman_filter.py
│    ├── linear_assignment.py
│    ├── nn_matching.py
│    ├── preprocessing.py
│    ├── track.py
│    └── tracker.py
├─── model_data                       模型文件数据
│    ├── market1501.pb
│    ├── mars-small128.pb
│    ├── train_mask_rcnn.h5
│    └── README.md
├─── mrcnn                            Mask R-CNN目标实例分割
│    ├── config.py
│    ├── model.py
│    ├── mrcnn_color.py
│    ├── mrcnn_colors.py
│    └── utils.py
├─── train                            平台模型训练
│    ├── mask_rcnn.py
│    ├── instance_segmentation.tar.gz
│    ├── pip-requirements.txt
│    └── README.md
│─── detect_video_tracker_color.py6
│─── detect_video_tracker_colors.py
│─── README.md
│─── run_color.ipynb
│─── run_colors.ipynb
└─── test.mp4
```

## `ModelArts` 平台执行

目标检测模型先通过训练得到更高精度的模型，也可以直接使用官方权重模型。

[训练数据模型权重](https://github.com/TsMask/ma-mask-rcnn/releases/tag/0)
[物体特征文件pd](https://github.com/TsMask/deep-sort-yolov4/releases/tag/0)

`ModelArts` 平台训练得到 `train_mask_rcnn.h5` 模型文件。

**开发环境模型调试**

将以下目录进行 Sync Obs 同步。

```text
┌─── deep_sort                        DeepSort目标跟踪算法
├─── model_data                       模型文件数据
├─── mrcnn                            Mask R-CNN目标实例分割
│─── detect_video_tracker_color.py    color  统一颜色
│─── detect_video_tracker_colors.py   colors 随机颜色
│─── run_color.ipynb
│─── run_colors.ipynb
└─── test.mp4
```

**终端执行方式**

1. 打开 `Terminal` 后通过命令，可以看到已同步的文件

```shell
cd work

pwd

ls -lh
```

2. 先切换到 `tf-1.13.1` 环境 

```shell
source /home/ma-user/anaconda3/bin/activate TensorFlow-1.13.1
# (TensorFlow-1.13.1) sh-4.3$
```

3. 选择执行你需要的识别 py

```shell
# 掩膜统一颜色 detect_video_tracker_color.py 文件
python detect_video_tracker_color.py --video_file test.mp4 --min_score 0.3 --input_size 1024 --model_file model_data/train_mask_rcnn.h5 --model_feature model_data/mars-small128.pb

# 掩膜随机颜色 detect_video_tracker_colors.py 文件
python detect_video_tracker_colors.py --video_file test.mp4 --min_score 0.3 --input_size 1024 --model_file model_data/train_mask_rcnn.h5 --model_feature model_data/mars-small128.pb
```

## 使用平台

拥有一个华为云账号

- EI 企业智能 —— ModelArts
- 存储 —— 对象存储服务 OBS

`ModelArts` 平台需要在全局配置中添加访问密钥才能使用的自动学习、数据管理、Notebook、训练作业、模型和服务可能需要使用对象存储功能，若没有添加访问密钥，则无法使用对象存储功能。

`对象存储服务 OBS` 创建一个桶进行文件的存储。

选择服务地区：**华北-北京四**

## 创建开发环境

在 `ModelArts` 平台中使用 _开发环境>Notebook_ 创建一个工作环境（Python3 | GPU）和选择对象存储服务（OBS）桶内的已创建或已上传的文件夹进行创建，之后可以启动已创建 `notebook` 进行在线的开发调试和具体的模型训练。使用一些关联文件需要同步到开发环境 `work` 文件夹内，注意同步文件大小存在指定大小内。

## 训练模型

在 `ModelArts` 平台中使用 _训练管理>训练作业_ 创建：

1. 算法来源为常用框架（TensorFlow | TF-1.13.1-python3.6）
2. 代码目录，选择已上传的 `train` 目录
3. 启动文件，选择 `train` 目录内的 `mask_rcnn.py` 文件
4. 数据来源为数据存储位置，选择已创建桶中已上传的 `train` 数据文件夹
5. 训练输出位置，选择已上传或创建名为 `model_data` 的文件夹
6. 作业日志路径，选择已上传或创建名为 `log` 的文件夹
7. 选择 **公共资源池>GPU** 训练更佳
8. 运行参数，参考下表：

|      名称       |  类型  |                    说明                    |
| :-------------: | :----: | :----------------------------------------: |
|    data_url     | string |  已创建桶中已上传的 `train` 数据文件夹   |
|    train_url    | string | 已上传名为 `model_data` 的文件夹 |
|    image_dim    | number |     训练统一图片大小 320/480/512/1024      |
|   epochs_step   | number |       每轮训练载入图片张数，默认 100       |
| validation_step | number |        每轮训练后验证张数，默认 20         |
|   max_epochs    | number |           训练迭代次数，默认 10            |
|    num_gpus     | number |         拥有 GPU 数，默认固定为 1          |


## 推荐

`ModelArts` 平台：

- [模型包规范介绍](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0091.html)
- [ModelArts 平台常见问题](https://support.huaweicloud.com/modelarts_faq/modelarts_05_0014.html)
- [MoXing 开发指南](https://support.huaweicloud.com/moxing-devg-modelarts/modelarts_11_0001.html)

`Mask R-CNN` 算法模型：

- [Mask R-CNN 论文](http://cn.arxiv.org/pdf/1703.06870v3)
- [Mask R-CNN GitHub 仓库](https://github.com/matterport/Mask_RCNN)
- [Mask R-CNN 训练自己的数据集](https://blog.csdn.net/l297969586/article/details/79140840)
- [Mask R-CNN 算法及其实现详解](https://blog.csdn.net/remanented/article/details/79564045)
- [最新centermask2 GitHub 仓库](https://github.com/youngwanLEE/centermask2)

`Deep Sort` 目标跟踪算法：

- [多目标跟踪：SORT和Deep SORT](https://zhuanlan.zhihu.com/p/59148865)
- [关于 Deep Sort 的一些理解](https://zhuanlan.zhihu.com/p/80764724)
- [Deep Sort GitHub 仓库](https://github.com/nwojke/deep_sort)
- [Object-Detection-and-Tracking GitHub 仓库](https://github.com/yehengchen/Object-Detection-and-Tracking)
