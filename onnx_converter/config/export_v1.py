# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/13 14:09
# @File     : export_v1.py

# _base_ = ['./export.py']

ops_type = 15  # OPS_INT8_TO_INT8_INT_SCALE

MAC_256 = 0

MAX_SHAPE_LEN = 8
MAX_IN_OUT_LEN = 8

# when this is False, we check each layer's error rate;
ADD_FLOAT = False
RESIZE_FLOAT = False
CONCAT_FLOAT = False
RESULT_INT = True

# the config for the weight size
DPCONV_W_LEN = 8
CONV_W_LEN = 8
FC_W_LEN = 8

# the data types
"""
OPS_INT8 = 1
OPS_INT16 = 2
OPS_INT32 = 4
OPS_FP32 = 5
OPS_INT32_TO_INT8 = 10
OPS_INT8_TO_INT8 = 11
OPS_INT32_TO_FP = 12
OPS_INT8_TO_FP = 13
"""
OPS_INT8 = 1
OPS_INT16 = 2
OPS_INT32 = 4
OPS_FP32 = 5
OPS_FP64 = 6
OPS_INT32_TO_INT8 = 10
# From int8 to int8 by using a scale(float) output = (float)input*scale
OPS_INT8_TO_INT8 = 11
# From int32 to FP by using a scale(float) int32->int8->fp
OPS_INT32_TO_FP = 12
OPS_INT8_TO_FP = 13  # From int8 to FP by using a scale(float)
OPS_INT32_TO_INT8_INT_SCALE = 14  # From int8 to int8 by using a scale(int)
# From int8 to int8 by using a scale(int) output= (i8)input*(u8)scale/256
OPS_INT8_TO_INT8_INT_SCALE = 15
OPS_INT32_TO_INT8_PER_CHN_SCALE_BEFORE_BIAS = 16
OPS_INT8_TO_INT8_PER_CHN_SCALE = 17

# quan method
QUAN_SHIFT = 0
QUAN_FLOAT = 1
QUAN_FLOAT_SYM = 2
QUAN_METHOD = QUAN_FLOAT_SYM
QUAN_INT = 3
QUAN_INT_SYM = 4
QUAN_POST_SHIFT = 5  # Post shift
QUAN_POST_AND_PRE = 6  # Post shift and prepare concatenate
QUAN_PRE_CONCAT = 7
QUAN_NONE = 0xFF

INTERP_NEAREST = 0
INTERP_BILINEAR = 1
INTERP_AREA = 3
INTERP_BICUBIC = 4
INTERP_LANCZOS = 5
INTERP_NEAREST_FIXED_POINT = 6
INTERP_BILINEAR_FIXED_POINT = 7
INTERP_AREA_FIXED_POINT = 7
INTERP_BICUBIC_FIXED_POINT = 8
INTERP_LANCZOS_FIXED_POINT = 9

# use 32 bit TSME output or 8 bit output
# TSME_OUT_32 = 1
USE_SHIFT_DEPTH = 0

INT_SCALE = 8

ignore_layers = [
    "LAYER_ACTIVATION",
    "LAYER_LSTM",
    "LAYER_LN",
    "LAYER_BN",
    "LAYER_RESHAPE",
    "LAYER_CWS",
    "LAYER_TS_CONV2D",
    "LAYER_SOFTMAX",
    "LAYER_GRU",
    "LAYER_IN",	    
]
serialize_wlist = ["data", "conv", "bias", "fc", "depthwiseconv"]
valid_export_layer = ["data", "conv", "depthwiseconv", "fc"]
export_version = 1
