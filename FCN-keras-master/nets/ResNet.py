# -*- coding: utf-8 -*-
# @File : ResNet.py
# @Author: Runist
# @Time : 2021/5/14 16:49
# @Software: PyCharm
# @Brief: ResNet Backbone

from tensorflow.keras import layers, models, utils
from tensorflow.python.keras.utils import data_utils
from nets.BilinearUpSampling import BilinearUpSampling2D
import os


def identity_block(inputs, filters, name, strides=1, training=True):
    """
    resnet卷积块
    :param inputs: 卷积块的输入
    :param filters: 卷积核的数量
    :param name: 卷积块名字相关
    :param strides: 为1时候不改变特征层宽高，为2就减半
    :param training: 是否是训练模式
    :return : x
    """
    filter1, filter2, filter3 = filters

    x = layers.Conv2D(filter1, kernel_size=1, strides=strides, name=name+'_1_conv')(inputs)
    x = layers.BatchNormalization(name=name+'_1_bn')(x, training=training)
    x = layers.ReLU(name=name + '_1_relu')(x)

    x = layers.Conv2D(filter2, kernel_size=3, strides=strides, padding='same', name=name+'_2_conv')(x)
    x = layers.BatchNormalization(name=name+'_2_bn')(x, training=training)
    x = layers.ReLU(name=name + '_2_relu')(x)

    x = layers.Conv2D(filter3, kernel_size=1, strides=strides, name=name+'_3_conv')(x)
    x = layers.BatchNormalization(name=name+'_3_bn')(x, training=training)

    x = layers.Add(name=name + '_add')([x, inputs])
    x = layers.ReLU(name=name + '_out')(x)

    return x


def conv_block(inputs, filters, name, strides=1, training=True):
    """
    bottleneck卷积块
    :param inputs: 卷积块的输入
    :param filters: 卷积核的数量
    :param name: 卷积块名字相关
    :param strides: 为1时候不改变特征层宽高，为2就减半
    :param training: 是否是训练模式
    :return : x
    """
    filter1, filter2, filter3 = filters

    shortcut = layers.Conv2D(filter3, kernel_size=1, strides=strides, name=name+'_0_conv')(inputs)
    shortcut = layers.BatchNormalization(name=name+'_0_bn')(shortcut, training=training)

    x = layers.Conv2D(filter1, kernel_size=1, strides=strides, name=name+'_1_conv')(inputs)
    x = layers.BatchNormalization(name=name+'_1_bn')(x, training=training)
    x = layers.ReLU(name=name + '_1_relu')(x)

    x = layers.Conv2D(filter2, kernel_size=3, padding='same', name=name+'_2_conv')(x)
    x = layers.BatchNormalization(name=name+'_2_bn')(x, training=training)
    x = layers.ReLU(name=name + '_2_relu')(x)

    x = layers.Conv2D(filter3, kernel_size=1, name=name+'_3_conv')(x)
    x = layers.BatchNormalization(name=name+'_3_bn')(x, training=training)

    x = layers.Add(name=name + '_add')([x, shortcut])
    x = layers.ReLU(name=name + '_out')(x)

    return x


def ResNet_stage(inputs, filters, num_block, name, strides):
    """
    ResNet中一个stage结构
    :param inputs: stage的输入
    :param filters: 每个卷积块对应卷积核的数量
    :param num_block: 卷积块重复的数量
    :param name: 该卷积块的名字
    :param strides: 步长
    :return: x
    """
    x = conv_block(inputs, filters, name=name+'_block1', strides=strides)
    for i in range(1, num_block):
        x = identity_block(x, filters, name=name+'_block'+str(i+1))

    return x


def ResNet50_backbone(inputs):
    """
    ResNet50 backbone
    :param inputs: 模型输入
    :return: x
    """
    x = layers.ZeroPadding2D((3, 3))(inputs)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, name='conv1_conv')(x)

    f1 = x
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv1_bn')(x)
    x = layers.ReLU(name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='pool1_pool')(x)

    x = ResNet_stage(x, [64, 64, 256], 3, name='conv2', strides=1)
    f2 = x
    x = ResNet_stage(x, [128, 128, 512], 4, name='conv3', strides=2)
    f3 = x
    x = ResNet_stage(x, [256, 256, 1024], 6, name='conv4', strides=2)
    f4 = x
    x = ResNet_stage(x, [512, 512, 2048], 3, name='conv5', strides=2)
    f5 = x

    model = models.Model(inputs, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    weights_path = data_utils.get_file(
        weights_path,
        'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return [f1, f2, f3, f4, f5]
