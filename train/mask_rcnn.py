import os,sys
import numpy as np
import tensorflow as tf
from moxing.framework import file

# 外部参数配置
tf.flags.DEFINE_string('data_url', 'cache/data', 'dataset directory.')
tf.flags.DEFINE_string('train_url', 'cache/output', 'saved model directory.')
tf.flags.DEFINE_integer('image_dim', 1024, 'number of training uniform picture size.')
tf.flags.DEFINE_integer('epochs_step', 100, 'number of steps per round.')
tf.flags.DEFINE_integer('validation_step', 20, 'number of steps per round of validation.')
tf.flags.DEFINE_integer('max_epochs', 10, 'number of training iterations.')
tf.flags.DEFINE_integer('num_gpus', 1, 'gpu nums.')

FLAGS = tf.flags.FLAGS

# 环境目录
local_data_path = 'cache/data'
local_output_path = 'cache/output'
model_output_path = os.path.join(local_output_path, "model")

if not os.path.exists(local_data_path):
    os.makedirs(local_data_path)

if not os.path.exists(local_output_path):
    os.makedirs(local_output_path)

# 复制数据到环境中
file.copy_parallel(FLAGS.data_url, local_data_path)

# 使用tar命令解压资源包，复制到训练文件目录train下
train_file = os.path.join(local_data_path, "instance_segmentation.tar.gz")
os.system('tar xf %s -C %s' % (train_file, 'train'))

# 训练权重文件
COCO_DIR = 'train/data'
COCO_MODEL_PATH = "train/data/mask_rcnn_coco.h5"

# 必要依赖mrcnn的文件，先解压资源包哦
from src.mrcnn.config import Config
from src.mrcnn.coco import CocoDataset
import src.mrcnn.model as modellib

# 训练配置
class MyTrainConfig(Config):
    # 可辨识的名称
    NAME = "my_train"

    # GPU的数量和每个GPU处理的图片数量，可以根据实际情况进行调整，参考为Nvidia Tesla P100
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 物体的分类个数，COCO中共有80种物体+背景
    NUM_CLASSES = 1 + 80  # background + 80 shapes

def main(*args):

    # 配置参数
    config = MyTrainConfig()
    # 图片尺寸统一处理为1024，可以根据实际情况再进一步调小
    config.IMAGE_MIN_DIM = FLAGS.image_dim
    config.IMAGE_MAX_DIM = FLAGS.image_dim
    # 每轮训练的step数量
    config.STEPS_PER_EPOCH = FLAGS.epochs_step
    # 每轮验证的step数量
    config.VALIDATION_STEPS = FLAGS.validation_step
    config.display()

    # 生成训练集
    dataset_train = CocoDataset()
    dataset_train.load_coco(COCO_DIR, "train") #　加载训练数据集
    dataset_train.prepare()

    # 生成验证集
    dataset_val = CocoDataset()
    dataset_val.load_coco(COCO_DIR, "val") #　加载验证数据集
    dataset_val.prepare()

    # 模型实例训练
    model = modellib.MaskRCNN(mode="training", config=config, model_dir="log")

    # 训练权重
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # 模型训练
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=FLAGS.max_epochs, 
            layers='all')

    # 保存模型训练权重
    save_wrapper = os.path.join(local_output_path, "train_mask_rcnn.h5")
    model.keras_model.save_weights(save_wrapper)

    # 复制保存到桶
    file.copy_parallel(local_output_path, FLAGS.train_url)

if __name__ == '__main__':
  tf.app.run(main=main)
