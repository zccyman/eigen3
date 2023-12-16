# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/06/07 19:54
# @File     : npu_layer.py
from enum import Enum, auto


class LayerFmt_t(Enum):
    FMT_NONE = 0x00
    FMT_VECTOR = 0x01
    FMT_VECTOR_TRANS = auto()
    FMT_MAT = 0x11
    FMT_MAT_TRANS = auto()
    FMT_CUBE_TSXE = 0x21
    FMT_CUBE_HWC = auto()
    FMT_CUBE_CHW = auto()
    FMT_CONV2D_W_TSXE = 0x31
    FMT_CONV2D_W_KHWC = auto()
    FMT_CONV2D_W_KCHW = auto()
    FMT_FC_W_TSXE = 0x41
    FMT_FC_W_MAT = auto()
    FMT_FC_W_TRANS = auto()


class LayerQuant_t(Enum):
    QUANT_NONE = 0x0
    QUANT_SHIFT = 0x1  # /* base shift or best shift# Y = X << shift */
    # /* Y = (X << shift) * scale << s_shift# scale(uint8)# s_shift = -8 */
    QUANT_ISCALE = 0x2
    QUANT_FSCALE = 0x3  # /* Y = (X << shift) * scale# scale is float */
    # /* Y = (X << shift) * scale << s_shift# shift(int8)# scale(uint8)# s_shift(int8)# they are per channel */
    QUANT_PER_CHN_ISCALE = 0x4
    # /* Y = (X << shift) * scale# shift(int8)# scale(float)# they are per channel */
    QUANT_PER_CHN_FSCALE = 0x5

    # /* Asymetric operations are not ready now */
    # /* Y = ((X << shift) - in_zero) * scale << s_shift + out_zero# scale is uint8# zero is int32# asymetric# s_shift = -7 */
    QUANT_ISCALE_ASY = 0x12
    # /* Y = ((X << shift) - in_zero) * scale + out_zero# scale is float# zero is int32# asymetric */
    QUANT_FSCALE_ASY = 0x13
    # /* Y = ((X << shift) - in_zero) * scale << s_shift + out_zero# scale is uint8# s_shift is int8# zero is int32# per channel# asymetric */
    QUANT_PER_CHN_ISCALE_ASY = 0x14
    # /* Y = ((X << shift) - in_zero) * scale + out_zero# scale is float# zero is int32# per channel# asymetric */
    QUANT_PER_CHN_FSCALE_ASY = 0x15

    QUANT_QUANT = 0x21
    QUANT_DEQUANT = 0x22
    QUANT_PER_CHN_QUANT = 0x23
    QUANT_PER_CHN_DEQUANT = 0x24

    QUANT_QUANT_ASY = 0x31
    QUANT_DEQUANT_ASY = 0x32
    QUANT_PER_CHN_QUANT_ASY = 0x33
    QUANT_PER_CHN_DEQUANT_ASY = 0x34
    
    QUANT_LUT8 = 0x35
    QUANT_LUT8_FP = 0x36
    QUANT_PER_CHN_LUT8_FP = auto()
    
    QUANT_SHIFT_ASY = auto()
    QUANT_PER_CHN_SHIFT_ASY = auto()
    QUANT_PER_CHN_SHIFT = auto()


class Activation_t(Enum):
    ACT_NONE = 0  # /* no activation */
    ACT_RELU = auto()
    ACT_RELU6 = auto()
    ACT_SIGMOID = auto()
    ACT_TANH = auto()
    ACT_LEAKY_RELU = auto()
    ACT_HARD_SIGMOID = auto()
    ACT_HARD_SWISH = auto()
    ACT_BRELU = auto()  # /* bounded ReLU# ReLU6 is a bounded ReLU */
    ACT_PRELU = auto()
    ACT_GELU = auto()
    ACT_SWISH = auto()
    
    ACT_LUT8 = 0xF0
    ACT_LUT8_FP = auto()
    ACT_LUT16 = auto()
    
    
class Lut_t(Enum):
    LUT_NONE = 0
    LUT_NORMAL = auto()
    LUT_PER_CHN = auto()


class Pool_t(Enum):
    POOL_NONE = 0  # /* no pooling */
    POOL_MAX = auto()
    POOL_AVG = auto()
    POOL_MIN = auto()
    POOL_GLOBAL_AVG = auto() 


class LayerType_t(Enum):
    LAYER_NORMAL_CLASS = 0x00000000
    LAYER_PLACEHOLDER = auto()
    LAYER_CONV2D = auto()
    LAYER_DW_CONV2D = auto()
    LAYER_FC = auto()
    LAYER_POOL = auto()  # /* Pool layer */
    LAYER_EWS = auto()  # /* Element-wise layer */
    LAYER_RESIZE = auto()
    LAYER_CONCAT = auto()
    LAYER_SHUFFLE = auto()
    LAYER_SPLIT = auto()
    LAYER_CONCAT_SHUFFLE_SPLIT = auto()
    LAYER_CONCAT_SPLIT = auto()
    LAYER_SLICE = auto()
    LAYER_ACTIVATION = auto()
    LAYER_CWS = auto()  # /* Channel-wise layer */
    LAYER_QUANT = auto()  # /* quantization/dequantization layer */
    LAYER_NORM = auto() 
    LAYER_LN = auto()  # /* layer norm layer */
    LAYER_IN = auto()  # /* instance norm layer */
    LAYER_BN = auto()  # /* batch norm layer */
    LAYER_GN = auto()  # /* group norm layer */
    LAYER_TS_CONV2D = auto() 
    
    LAYER_RNN_CLASS = 0x01000000  # /* RNN class */
    LAYER_LSTM = auto()  # /* lstm layer */

    LAYER_NOP_CLASS = 0x02000000  # /* no operation in this class */
    LAYER_SQUEEZE = auto()  # /* squeeze layer */
    LAYER_RESHAPE = auto()  # /* reshape layer */
    LAYER_REDUCE = auto()
    LAYER_TRANSPOSE = auto()
    LAYER_MATMUL = auto()
    LAYER_PAD = auto()
    
    LAYER_RESERVED_CLASS = 0xFF000000
    MEAN = auto()

    FSMN = auto()
    DCONV2D = auto()
    FSMN_ADD = auto()


class EleWiseType_t(Enum):
    EWS_ADD = 1
    EWS_SUB = auto()
    EWS_MUL = auto()
    EMS_DIV = auto()


class ChnWiseType_t(Enum):
    CWS_ADD = 1
    CWS_SUB = auto()
    CWS_MUL = auto()
    CMS_DIV = auto()


class SplitType_t(Enum):
    SPLIT = 1
    SPLIT_TO_SEQUENCE = auto()
    
    
class ReduceType_t(Enum):
    REDUCE_MAX = 1
    REDUCE_MIN = auto()
    REDUCE_SUM = auto()
    REDUCE_MEAN = auto()
    REDUCE_PROD = auto()
    REDUCE_SOFTMAX = auto()
    
    
class DimType_t(Enum):
    DIM_N = 0x01
    DIM_H = 0x02
    DIM_W = 0x04
    DIM_C = 0x08
        
        
class NormType_t(Enum):        
    BN = 1
    LN = auto()
    IN = auto()
    GN = auto()
    
    
class ResizeMethod_t(Enum):
    RESIZE_NEAREST = 0x01
    RESIZE_BILINEAR = auto()
    RESIZE_AREA = auto()
    RESIZE_BICUBIC = auto()
    RESIZE_LANCZOS = auto()
    RESIZE_NEAREST_FIXED_POINT = 0x11
    RESIZE_BILINEAR_FIXED_POINT = auto()
    RESIZE_AREA_FIXED_POINT = auto()
    RESIZE_BICUBIC_FIXED_POINT = auto()
    RESIZE_LANCZOS_FIXED_POINT = auto()
    RESIZE_NEAREST_LUT = 0x21
    RESIZE_BILINEAR_LUT = auto()
    

class ResizeCoordinateTransMode_t(Enum):
    RESIZE_COOR_TRANS_NONE = 0
    RESIZE_ASYMMETRIC = auto()
    RESIZE_HALF_PIXEL = auto()
    RESIZE_ALIGN_CORNERS = auto()
    RESIZE_PYTORCH_HALF_PIXEL = auto()
    RESIZE_TF_CROP_AND_RESIZE = auto()


class ResizeNearestRoundMode_t(Enum):
    NEAREST_ROUND = 0
    NEAREST_ROUND_FLOOR = auto()
    NEAREST_ROUND_CEIL = auto()
    
    LINEAR_ROUND = auto()
    LINEAR_ROUND_FLOOR = auto()
    LINEAR_ROUND_CEIL = auto()    


class LstmDir_t(Enum):
    LSTM_FORWARD = 0
    LSTM_REVERSE = auto()
    LSTM_BIDIRECTIONAL = auto()


class LstmQuant_t(Enum):
    LSTM_QUANT_I_H_SAME = 1
    LSTM_QUANT_I_H_DIFF = 2


class NpuType_t(Enum):
    NPU_INT8 = 0x01  # ,
    NPU_INT16 = 0x02  # ,
    NPU_INT32 = 0x04  # ,
    NPU_INT64 = 0x08  # ,

    NPU_UINT8 = 0x11  # ,
    NPU_UINT16 = 0x12  # ,
    NPU_UINT32 = 0x14  # ,
    NPU_UINT64 = 0x18  # ,

    NPU_FP32 = 0x24  # ,
    NPU_FP64 = 0x28  # ,

    NPU_BOOL = 0x31  # ,


npu_layer_enums = {}
for enums in [
    LayerFmt_t,
    LayerQuant_t,
    Activation_t,
    Lut_t,
    Pool_t,
    LayerType_t,
    EleWiseType_t,
    ChnWiseType_t,
    SplitType_t,
    ReduceType_t,
    DimType_t,
    NormType_t,
    ResizeMethod_t,
    ResizeCoordinateTransMode_t,
    ResizeNearestRoundMode_t,
    LstmDir_t,
    LstmQuant_t,
    NpuType_t,
]: # type: ignore
    for enum_ in enums:
        npu_layer_enums[enum_.name] = enum_.value

# print(npu_layer_enums)
