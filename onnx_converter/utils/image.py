# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/6 13:10
# @File     : image.py
import copy
from itertools import product
from math import ceil
import cv2
import functools
import numpy as np

# from .utils import py_cpu_nms


def process_im(data, shape):
    if len(data.shape) == 3:
        if data.shape[-1] != 3:
            data_ = np.transpose(data, (1, 2, 0))
        else:
            data_ = copy.deepcopy(data)
        data_ = cv2.resize(data_, tuple(shape)).astype(np.float32)
        data_ = np.transpose(data_, (2, 0, 1))
        data = data_[np.newaxis, ...]
    return data

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def generate_anchors():
    pass


# def


# class


if __name__ == '__main__':
    import cv2

    img = cv2.imread('/home/shiqing/Downloads/model-converter/jpeg/test.jpg')
    im = cv2.resize(img, (224, 256))
    res, ratio, d = letterbox(im)
    cv2.imshow('img', res)
    cv2.waitKey(0)
    print(res.shape)
