import sys
import cv2
sys.path.append("./")

from quantizer.data_quan import FloatSymQuan, FloatQuan
from simulator import error_factory

import numpy as np
import pytest
from extension.libs import pyops

def test_resize_int8(img_path):
    
    scale = np.float32(1.5)
    img = cv2.imread(img_path)
    img = img.transpose((2,0,1))
    img = img[np.newaxis, :]
    
    output_width = np.int32(img.shape[-1] * scale)
    output_height = np.int32(img.shape[-2] * scale)

    dst = np.zeros([img.shape[0], img.shape[1], output_height, output_width], dtype=np.int8)

    resize_op = pyops.py_resize_op_int8(np.array(img.shape), np.array(dst.shape), np.array([scale, scale], dtype=np.float32), 1, True, 4)

    quant = FloatSymQuan()
    quant.get_quan_param(img)
    quantized = quant.get_quan_data(img)

    resize_op.forward(quantized, dst, np.array(img.shape))

    dequant = quant.get_dequan_data(dst).astype(np.uint8)

    return dequant

def test_resize_integer_int8(img_path):
    
    scale = np.float32(1.5)
    img = cv2.imread(img_path)
    img = img.transpose((2,0,1))
    img = img[np.newaxis, :]
    
    output_width = np.int32(img.shape[-1] * scale)
    output_height = np.int32(img.shape[-2] * scale)

    dst = np.zeros([img.shape[0], img.shape[1], output_height, output_width], dtype=np.int8)

    resize_op = pyops.py_resize_op_int8(np.array(img.shape), np.array(dst.shape), np.array([scale, scale], dtype=np.float32), 1, True, 4)

    quant = FloatSymQuan()
    quant.get_quan_param(img)
    quantized = quant.get_quan_data(img)

    resize_op.forward(quantized, dst, np.array(img.shape))

    dequant = quant.get_dequan_data(dst)

    return dequant.astype(np.uint8)

def test_resize_uint8(img_path):
    
    scale = np.float32(1.5)
    img = cv2.imread(img_path)
    img = img.transpose((2,0,1))
    img = img[np.newaxis, :]
    
    output_width = np.int32(img.shape[-1] * scale)
    output_height = np.int32(img.shape[-2] * scale)

    dst = np.zeros([img.shape[0], img.shape[1], output_height, output_width], dtype=np.uint8)

    resize_op = pyops.py_resize_op_uint8(np.array(img.shape), np.array(dst.shape), np.array([scale, scale], dtype=np.float32), 1, True, 4)

    resize_op.forward(img, dst, np.array(img.shape))

    return dst.astype(np.uint8)

def test_resize_integer_uint8(img_path):
    
    scale = np.float32(1.5)
    img = cv2.imread(img_path)
    img = img.transpose((2,0,1))
    img = img[np.newaxis, :]
    
    output_width = np.int32(img.shape[-1] * scale)
    output_height = np.int32(img.shape[-2] * scale)

    dst = np.zeros([img.shape[0], img.shape[1], output_height, output_width], dtype=np.uint8)

    resize_op = pyops.py_resize_op_uint8(np.array(img.shape), np.array(dst.shape), np.array([scale, scale], dtype=np.float32), 1, True, 4)

    resize_op.forward(img, dst, np.array(img.shape))

    return dst.astype(np.uint8)

if __name__ == "__main__":
    img_path = "/home/shiqing/Downloads/onnx-converter/trained_models/resize-rgb888.png"
    for i in range(1000):
        int8_dequant = test_resize_int8(img_path)    
        int8_integer_dequant = test_resize_integer_int8(img_path)    
        uint8_dequant = test_resize_uint8(img_path)    
        uint8_integer_dequant = test_resize_integer_uint8(img_path)
        cv2.imwrite("/home/shiqing/Downloads/onnx-converter/trained_models/int8_resized"+".jpg", int8_dequant[0].transpose(1,2,0))
        cv2.imwrite("/home/shiqing/Downloads/onnx-converter/trained_models/int8_integer_resized"+".jpg", int8_integer_dequant[0].transpose(1,2,0))
        cv2.imwrite("/home/shiqing/Downloads/onnx-converter/trained_models/uint8_resized"+".jpg", uint8_dequant[0].transpose(1,2,0))
        cv2.imwrite("/home/shiqing/Downloads/onnx-converter/trained_models/uint8_integer_resized"+".jpg", uint8_integer_dequant[0].transpose(1,2,0))
        cosine = error_factory.get('Cosine')()(int8_dequant, int8_integer_dequant)/100,
        l1 = error_factory.get('L1')()(int8_dequant, int8_integer_dequant)/100
        l2 = error_factory.get('L2')()(int8_dequant, int8_integer_dequant)/100
        print("int8 resize and float error is: {}, {}, {}".format(cosine, l1, l2))
        cosine = error_factory.get('Cosine')()(uint8_dequant, uint8_integer_dequant)/100,
        l1 = error_factory.get('L1')()(uint8_dequant, uint8_integer_dequant)/100
        l2 = error_factory.get('L2')()(uint8_dequant, uint8_integer_dequant)/100
        print("uint8 resize and float error is: {}, {}, {}".format(cosine, l1, l2))
        cosine = error_factory.get('Cosine')()(uint8_dequant, int8_dequant)/100,
        l1 = error_factory.get('L1')()(uint8_dequant, int8_dequant)/100
        l2 = error_factory.get('L2')()(uint8_dequant, int8_dequant)/100
        print("uint8 and int8 float error is: {}, {}, {}".format(cosine, l1, l2))
        del int8_dequant
        del int8_integer_dequant
        del uint8_dequant
        del uint8_integer_dequant

