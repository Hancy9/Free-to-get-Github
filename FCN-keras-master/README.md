# Keras  - FCN

## Part 1. Introduction

Fully Convolutional Networks is the first model to apply Convolutional Neural Network to semantic segmentation. It used common backbone like VGG, ResNet as encoder, and the decoders are upsampled layer by layer to original image size.

### Models for this repository

| Model name     | Dataset                | MIoU   | Pixel accuracy |
| -------------- | ---------------------- | ------ | -------------- |
| FCN_8_Resnet50 | VOC train dataset      | 0.4578 | 0.8993         |
|                | VOC validation dataset | 0.751  | 0.9528         |

## Part 2. Quick  Start

1. Pull this repository.

```shell
git clone https://github.com/Runist/FCN-keras.git
```

2. You need to install some dependency package.

```shell
cd FCN-keras
pip installl -r requirements.txt
```

3. Download the *[VOC](https://www.kaggle.com/huanghanchina/pascal-voc-2012)* dataset(VOC [SegmetationClassAug](http://home.bharathh.info/pubs/codes/SBD/download.html) if you need) .
4. Getting FCN weights.

```shell
wget https://github.com/Runist/FCN-keras/releases/download/v0.2/fcn_weights.h5
```

4. Run **predict.py**, you'll see the result of Fully Convolutional Networks.

```shell
python predict.py
```

Input image:

![2007_000822.jpg](https://i.loli.net/2021/07/01/BdkWGYVI4HNc2Ov.jpg)

Output image（resize to 320 x 320）:

![fcn.jpg](https://i.loli.net/2021/07/01/vEhSRBAmQWC1z3k.jpg)

## Part 3. Train your own dataset
1. You should rewrite your data pipeline, *Dateset* where in *dataset.py* is the base class, such as  *VOCdataset.py*.

```python
class VOCDataset(Dataset):
    def __init__(self, annotation_path, batch_size=4, target_size=(320, 320), num_classes=21, aug=False):
        super().__init__(target_size, num_classes)
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.annotation_path = annotation_path
        self.aug = aug
        self.read_voc_annotation()
        self.set_image_info()
```

2. Start training.

```shell
python train.py
```

3. Running *evaluate.py* to get mean iou and pixel accuracy.

```shell
python evaluate.py
--------------------------------------------------------------------------------
Total MIOU: 0.4578
Object MIOU: 0.4357
pixel acc: 0.8993
IOU:  [0.89910368 0.57704284 0.42540458 0.37512895 0.20381801 0.35467035
 0.77999658 0.63954053 0.6688376  0.22855226 0.36980252 0.25836402
 0.59642955 0.48159056 0.42752974 0.67481864 0.13373938 0.33782181
 0.30385944 0.47929764 0.39788109]
```

## Part 4. Paper and other implement

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- paper with code: [shelhamer](https://github.com/shelhamer)/**[fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org)**
- [aurora95/*Keras*-*FCN*](https://github.com/aurora95/Keras-FCN)
- [divamgupta/image-segmentation-*keras*](https://github.com/divamgupta/image-segmentation-keras)
