# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/8 11:22
# @File     : parse.py
import os

all_ops = ['Unsqueeze', 'MaxPool', 'Mul', 'Add', 'Expand', 'Sub', 'TopK', 'Div', 'Log', 'Reshape',
          'Sigmoid', 'Shape', 'LRN', 'Floor', 'Relu', 'Tile', 'NonMaxSuppression', 'Gather', 'Slice',
           'ScatterND', 'Resize', 'Softmax', 'Squeeze', 'Constant', 'GlobalAveragePool', 'InstanceNormalization',
           'Conv', 'ConvTranspose', 'Where', 'Flatten', 'Less', 'Range', 'ConstantOfShape', 'Clip', 'Gemm', 'LeakyRelu', 'ReduceL2',
           'AveragePool', 'Equal', 'ReduceMax', 'Concat', 'Transpose', 'Exp', 'Cast', 'RoiAlign', 'Sqrt', 'data',
           'Split', 'MatMul', 'BatchNormalization', 'Upsample', 'Dropout', 'Sum', 'ReduceMean', 'LSTM', 'GRU',
           'LayerNormalization']

dtypes = dict(undefined=0, float32=1, uint8=2, int=3, uint16=4, int16=5, int32=6, int64=7,
              string=8, bool=9, float16=10, double=11, uint32=12, uint64=13, complex64=14,
              complex128=15, bfloat16=16)

vaild_bit = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13}

shape_template = [1, 3, 256, 256]

is_simplify = True
preprocess = True

combine = dict(hwish=['add', 'relu', 'clip', 'div'],hsigmigd=['add', 'relu', 'clip', 'mul', 'div'])

try:
    import onnx_converter
    shared_librarys = [
        os.path.join(onnx_converter.__path__[0], "extension/libs/libcustom_op_library_cpu.so")    
    ]        
except:
    shared_librarys = [
        "extension/libs/libcustom_op_library_cpu.so",
    ]