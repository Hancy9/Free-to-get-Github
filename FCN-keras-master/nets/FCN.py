# -*- coding: utf-8 -*-
# @File : FCN.py
# @Author: Runist
# @Time : 2021/6/16 10:20
# @Software: PyCharm
# @Brief: FCN的实现
from tensorflow.keras import layers, models
from nets.ResNet import ResNet50_backbone
from nets.VGG import VGG16_backbone
from nets.BilinearUpSampling import BilinearUpSampling2D
import core.config as cfg


def FCN_32(encoder, input_shape, num_classes=21):
    """
    FCN32通用结构
    :param encoder: 不同的backbone对象
    :param input_shape: 模型输入shape
    :param num_classes: 分类数量
    :return: model
    """
    input_image = layers.Input(shape=input_shape)
    feat = encoder(input_image)
    [f1, f2, f3, f4, f5] = feat
    x = f5

    # 原本全连接层转换成为卷积层
    f5 = layers.Conv2D(4096, (3, 3), padding='same', name='f5_conv1')(f5)
    f5 = layers.Dropout(0.5)(f5)
    f5 = layers.Conv2D(4096, (3, 3), padding='same', name='f5_conv2')(f5)
    f5 = layers.Dropout(0.5)(f5)
    f5 = layers.Conv2D(num_classes, (1, 1), activation='linear', padding='same', name='classifier_output')(f5)
    f5 = BilinearUpSampling2D(scale=(32, 32))(f5)

    # input_h, input_w = input_shape[:2]
    # h, w = x.shape[1:-1]
    # scale = int(input_h / h), int(input_w / w)
    #
    # x = BilinearUpSampling2D(scale=scale)(x)

    output = f5
    model = models.Model(input_image, output)

    return model


def FCN_16(encoder, input_shape, num_classes=21):
    """
    FCN16通用结构
    :param encoder: 不同的backbone对象
    :param input_shape: 模型输入shape
    :param num_classes: 分类数量
    :return: model
    """
    input_image = layers.Input(shape=input_shape)
    feat = encoder(input_image)
    [f1, f2, f3, f4, f5] = feat

    f5 = layers.Conv2D(4096, (3, 3), padding='same', name='f5_conv1')(f5)
    f5 = layers.Dropout(0.5)(f5)
    f5 = layers.Conv2D(4096, (3, 3), padding='same', name='f5_conv2')(f5)
    f5 = layers.Dropout(0.5)(f5)
    f5 = layers.Conv2D(num_classes, (1, 1), padding='same', name='classifier_f5')(f5)
    f5 = BilinearUpSampling2D(scale=(2, 2))(f5)

    f4 = layers.Conv2D(num_classes, (1, 1), padding='same', name='classifier_f4')(f4)
    f4 = layers.Add()([f4, f5])
    f4 = BilinearUpSampling2D(scale=(16, 16))(f4)

    output = f4
    model = models.Model(input_image, output)

    return model


def FCN_8(encoder, input_shape, num_classes=21):
    """
    FCN8通用结构
    :param encoder: 不同的backbone对象
    :param input_shape: 模型输入shape
    :param num_classes: 分类数量
    :return: model
    """
    input_image = layers.Input(shape=input_shape)
    feat = encoder(input_image)
    [f1, f2, f3, f4, f5] = feat

    f5 = layers.Conv2D(4096, (3, 3), padding='same', name='f5_conv1')(f5)
    f5 = layers.Dropout(0.5)(f5)
    f5 = layers.Conv2D(4096, (3, 3), padding='same', name='f5_conv2')(f5)
    f5 = layers.Dropout(0.5)(f5)
    f5 = layers.Conv2D(num_classes, (1, 1), padding='same', name='classifier_f5')(f5)
    f5 = BilinearUpSampling2D(scale=(2, 2))(f5)

    f4 = layers.Conv2D(num_classes, (1, 1), padding='same', name='classifier_f4')(f4)
    f4 = layers.Add()([f4, f5])
    f4 = BilinearUpSampling2D(scale=(2, 2))(f4)

    f3 = layers.Conv2D(num_classes, (1, 1), padding='same', name='classifier_f3')(f3)
    f3 = layers.Add()([f3, f4])
    f3 = BilinearUpSampling2D(scale=(8, 8))(f3)

    output = f3
    model = models.Model(input_image, output)

    return model


def FCN_32_VGG16(input_shape, num_classes=21):
    """
    基于vgg16的 FCN32网络
    :param input_shape: 输入shape
    :param num_classes: 分类个数
    :return: model
    """
    model = FCN_32(VGG16_backbone, input_shape, num_classes)
    model._name = 'fcn_32_vgg16'

    return model


def FCN_16_VGG16(input_shape, num_classes=21):
    """
    基于vgg16的 FCN32网络
    :param input_shape: 输入shape
    :param num_classes: 分类个数
    :return: model
    """
    model = FCN_16(VGG16_backbone, input_shape, num_classes)
    model._name = 'fcn_16_vgg16'

    return model


def FCN_8_VGG16(input_shape, num_classes=21):
    """
    基于vgg16的 FCN8网络
    :param input_shape: 输入shape
    :param num_classes: 分类个数
    :return: model
    """
    model = FCN_8(VGG16_backbone, input_shape, num_classes)
    model._name = 'fcn_8_vgg16'

    return model


def FCN_32_ResNet50(input_shape, num_classes=21):
    """
    基于ResNet50的 FCN32网络
    :param input_shape: 输入shape
    :param num_classes: 分类个数
    :return:
    """
    model = FCN_32(ResNet50_backbone, input_shape, num_classes)
    model._name = 'fcn_32_vgg16'

    return model


def FCN_16_ResNet50(input_shape, num_classes=21):
    """
    基于ResNet50的 FCN32网络
    :param input_shape: 输入shape
    :param num_classes: 分类个数
    :return:
    """
    model = FCN_16(ResNet50_backbone, input_shape, num_classes)
    model._name = 'fcn_16_vgg16'

    return model


def FCN_8_ResNet50(input_shape, num_classes=21):
    """
    基于ResNet50的 FCN8网络
    :param input_shape: 输入shape
    :param num_classes: 分类个数
    :return:
    """
    model = FCN_8(ResNet50_backbone, input_shape, num_classes)
    model._name = 'fcn_8_resnet50'

    return model
