# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/13 14:09
# @File     : export.py

secret_key = "henson.zhang@timesintelli.com"

is_debug = False
export_mode_c = True
chip_type = "AT1K"  # [AT1K, AT5050,]

AT5050_C_EXTEND = chip_type != "AT1K"
# Csize, Ksize: input align size, output align size
Csize, Ksize = (16, 32) if AT5050_C_EXTEND else (8, 8)
O_Align = 4  # fc output align size
I_Align = 8  # fc input align size

fmt = 0
bits = dict(
    Csize=Csize,
    Ksize=Ksize,
    I_Align=I_Align,
    O_Align=O_Align,
    ABGR=False,
    DATA_C_EXTEND=AT5050_C_EXTEND,
    chip_type=chip_type,
    bgr_format=True,
    is_debug=is_debug,
)

layer_map = dict(
    LAYER_PLACEHOLDER=["data", "splice"],
    LAYER_CONV2D=["conv"],
    LAYER_DW_CONV2D=["depthwiseconv"],
    LAYER_TS_CONV2D=["convtranspose"],
    LAYER_FC=["fc"],
    LAYER_ACTIVATION=[
        "relu",
        "relu6",
        "relux",
        "leakyrelu",
        "prelu",
        "tanh",
        "sigmoid",
        "swish",
        "gelu",
        "hardsigmoid",
        "hardtanh",
        "hardswish",
        "hardshrink",
    ],
    LAYER_REDUCE=["reducemin", "reducemax", "reducemean", "reducesum", "reduceprod"],
    LAYER_POOL=["maxpool", "averagepool", "globalaveragepool"],
    LAYER_EWS=["add", "sub", "mul", "pmul"],
    LAYER_CWS=["cadd", "csub", "cmul"],
    LAYER_RESIZE=["resize"],
    LAYER_SOFTMAX=["softmax"],
    LAYER_CONCAT=["concat"],
    LAYER_SHUFFLE=["shuffle_only"],
    LAYER_SPLIT=["split"],
    LAYER_CONCAT_SHUFFLE_SPLIT=["shuffle"],
    LAYER_LSTM=["lstm"],
    LAYER_GRU=["gru"],
    LAYER_LN=["layernormalization"],
    LAYER_BN=["batchnormalization"],
    LAYER_IN=["instancenormalization"],
    LAYER_NORM=["batchnormalization", "layernormalization", "instancenormalization"],
    LAYER_RESHAPE=["reshape"],
    LAYER_PAD=["pad"],
    LAYER_LOG=["log"],
    LAYER_TRANSPOSE=["transpose"],
    LAYER_MATMUL=["matmul"],
)
LayerInstance = dict(
    LAYER_PLACEHOLDER="placeholder",
    LAYER_CONV2D="conv2d",
    LAYER_DW_CONV2D="conv2d",
    LAYER_TS_CONV2D="conv2d",
    LAYER_FC="fc",
    LAYER_ACTIVATION="act",
    LAYER_REDUCE="reduce",
    LAYER_POOL="pool",
    LAYER_EWS="element_wise",
    LAYER_CWS="channel_wise",
    LAYER_RESIZE="resize",
    LAYER_SOFTMAX="softmax",
    LAYER_CONCAT="concat",
    LAYER_SHUFFLE="shuffle_only",
    LAYER_SPLIT="split",
    LAYER_CONCAT_SHUFFLE_SPLIT="concat_shuffle_split",
    LAYER_LSTM="lstm",
    LAYER_GRU="gru",
    LAYER_LN="ln",
    LAYER_BN="bn",
    LAYER_IN="in",
    LAYER_NORM="norm",
    LAYER_RESHAPE="reshape",
    LAYER_PAD="pad",
    LAYER_LOG="log",
    LAYER_TRANSPOSE="transpose",
    LAYER_MATMUL="matmul",
)
NPU_DataType = {
    "": "NPU_INT8",
    0: "NPU_UINT8",
    1: "NPU_INT8",
    2: "NPU_UINT16",
    3: "NPU_INT16",
    4: "NPU_UINT32",
    5: "NPU_INT32",
    6: "NPU_UINT64",
    7: "NPU_INT64",
    8: "NPU_FP32",
    9: "NPU_FP64",
}
MatFmt = ["FMT_MAT", "FMT_MAT_TRANS"]
CubeFmt = ["FMT_CUBE_TSXE", "FMT_CUBE_HWC", "FMT_CUBE_CHW"]
ConvWFmt = ["FMT_CONV2D_W_TSXE", "FMT_CONV2D_WT_KHWC", "FMT_CONV2D_W_KCHW"]
FcWFmt = ["FMT_FC_W_TSXE", "FMT_FC_W", "FMT_FC_W_TRANS"]
ActivationType = dict(
    ACT_NONE=["act"],
    ACT_RELU=["relu"],
    ACT_LEAKY_RELU=["leakyrelu"],
    ACT_PRELU=["prelu"],
    ACT_RELU6=["relu6"],
    ACT_BRELU=["relux"],
    ACT_SIGMOID=["sigmoid"],
    ACT_SWISH=["swish"],
    ACT_GELU=["gelu"],
    ACT_TANH=["tanh"],
    ACT_HARD_SIGMOID=["hardsigmoid"],
    ACT_HARD_TANH=["hardtanh"],
    ACT_HARD_SWISH=["hardswish"],
    ACT_HARD_SHRINK=["hardshrink"],
)
ReduceType = dict(
    REDUCE_NONE=[""],
    REDUCE_MIN=["reducemin"],
    REDUCE_MAX=["reducemax"],
    REDUCE_MEAN=["reducemean"],
    REDUCE_SUM=["reducesum"],
    REDUCE_PROD=["reduceprod"],
)
PoolType = dict(
    POOL_NONE=[""],
    POOL_MAX=["maxpool"],
    POOL_AVG=["averagepool"],
    POOL_MIN=["minpool"],
    POOL_GLOBAL_AVG=["globalaveragepool"],
)
ElementWiseType = dict(
    EWS_ADD=["add"], EWS_SUB=["sub"], EWS_MUL=["mul", "pmul"], EMS_DIV=["div"]
)
ChnWiseType = dict(CWS_ADD=["cadd"], CWS_SUB=["csub"], CWS_MUL=["cmul"], CWS_DIV=["cdiv"])
ResizeMethod = dict(
    RESIZE_NONE="",
    RESIZE_BILINEAR="linear",
    RESIZE_NEAREST="nearest",
    RESIZE_BICUBIC="cubic",
)
# ResizeMethod=['RESIZE_NEAREST', 'RESIZE_BILINEAR', 'RESIZE_AREA', 'RESIZE_BICUBIC', 'RESIZE_LANCZOS',
#               'RESIZE_NEAREST_FIXED_POINT', 'RESIZE_BILINEAR_FIXED_POINT', 'RESIZE_AREA_FIXED_POINT',
#               'RESIZE_BICUBIC_FIXED_POINT', 'RESIZE_LANCZOS_FIXED_POINT']
LayerQuant = dict(
    QUANT_NONE=["", "smooth",],
    QUANT_SHIFT=["shiftscale"],
    QUANT_ISCALE=["preintscale", "intscale"],
    QUANT_FSCALE=[
        "floatscale",
        "rshiftscale",
        "ffloatscale",
        "rrshiftscale",
        "shiftfloatscale",
        "float",
    ],
    QUANT_PER_CHN_SHIFT=["perchannel_shiftscale"],
    QUANT_PER_CHN_ISCALE=["perchannel_preintscale", "perchannel_intscale"],
    QUANT_PER_CHN_FSCALE=[
        "perchannel_floatscale",
        "perchannel_rshiftscale",
        "perchannel_rrshiftscale",
        "perchannel_shiftfloatscale",
    ],
    QUANT_PER_CHN_ISCALE_ASY=["perchannel_preintscale_asy", "perchannel_intscale_asy"],
    QUANT_PER_CHN_FSCALE_ASY=[
        "perchannel_floatscale_asy",
        "perchannel_rshiftscale_asy",
        "perchannel_rrshiftscale_asy",
        "perchannel_shiftfloatscale_asy",
    ],
    QUANT_QUANT=["quant"],
    QUANT_DEQUANT=["dequant"],
    QUANT_PER_CHN_QUANT=["perchannel_quant"],
    QUANT_PER_CHN_DEQUANT=["perchannel_dequant"],
    QUANT_QUANT_ASY=["quant_asy"],
    QUANT_DEQUANT_ASY=["dequant_asy"],
    QUANT_PER_CHN_QUANT_ASY=["perchannel_quant_asy"],
    QUANT_PER_CHN_DEQUANT_ASY=["perchannel_dequant_asy"],
)
# QUANT_PER_CHN_ISCALE='', QUANT_PER_CHN_FSCALE='',
# QUANT_ISCALE_ASY='', QUANT_FSCALE_ASY='',
# QUANT_PER_CHN_ISCALE_ASY='', QUANT_PER_CHN_FSCALE_ASY='')
LayerPrePost = dict(
    q_none=[
        "QUANT_NONE",
    ],
    q_shift=["QUANT_SHIFT"],
    q_int=["QUANT_ISCALE"],
    q_fp=["QUANT_FSCALE"],
    q_pc_shift=["QUANT_PER_CHN_SHIFT"],
    q_pc_int=["QUANT_PER_CHN_ISCALE"],
    q_pc_fp=["QUANT_PER_CHN_FSCALE"],
    q_pc_int_asy=["QUANT_PER_CHN_ISCALE_ASY"],
    q_pc_fp_asy=["QUANT_PER_CHN_FSCALE_ASY"],
    q=["QUANT_QUANT"],
    dq=["QUANT_DEQUANT"],
    pc_q=["QUANT_PER_CHN_QUANT"],
    pc_dq=["QUANT_PER_CHN_DEQUANT"],
    q_asy=["QUANT_QUANT_ASY"],
    dq_asy=["QUANT_DEQUANT_ASY"],
    pc_q_asy=["QUANT_PER_CHN_QUANT_ASY"],
    pc_dq_asy=["QUANT_PER_CHN_DEQUANT_ASY"],
)

model_c = dict(
    layer_map=layer_map,
    LayerInstance=LayerInstance,
    CubeFmt=CubeFmt,
    ConvWFmt=ConvWFmt,
    MatFmt=MatFmt,
    FcWFmt=FcWFmt,
    ActivationType=ActivationType,
    ReduceType=ReduceType,
    PoolType=PoolType,
    ElementWiseType=ElementWiseType,
    ChnWiseType=ChnWiseType,
    ResizeMethod=ResizeMethod,
    LayerQuant=LayerQuant,
    LayerPrePost=LayerPrePost,
    NPU_DataType=NPU_DataType,
    fmt=fmt,
    bits=bits,
    MAX_IN_OUT_LEN=8,
    CONCAT_SHUFFLE_SPLIT_MAX_IN=4,
    CONCAT_SHUFFLE_SPLIT_MAX_OUT=4,
    SHUFFLE_MAX_IN_SECTION=8,
    CONCAT_MAX_IN=8,
    SPLIT_MAX_OUT=8,
    ELEMENT_WISE_MAX_IN=2,
    is_debug=is_debug,
)

serialize_wlist = ["data",  "splice", "conv", "bias", "fc", "depthwiseconv", "convtranspose", "table"]
valid_export_layer = [
    "data",
    "splice",
    "conv",
    "depthwiseconv",
    "convtranspose",
    "fc",
    "reducemin"
    "reducemax",
    "reducemean",
    "reducesum",
    "reduceprod",
    "matmul",
    "maxpool",
    "averagepool",
    "globalaveragepool",
    "cmul",
    "pmul",
    "mul",
    "add",
    "cadd",
    "sub",
    "csub",
    "concat",
    "shuffle_only",
    "concat_shuffle_split",
    "shuffle",
    "split",
    "softmax",
    "resize",
    "lstm",
    "gru",
    "layernormalization",
    "batchnormalization",
    "instancenormalization",
    "relu",
    "relu6",
    "relux",
    "leakyrelu",
    "prelu",
    "sigmoid",
    "swish",
    "gelu",
    "tanh",
    "hardsigmoid",
    "hardtanh",
    "hardswish",
    "hardshrink",
    "reshape",
    "pad",
    "log",
]
export_version = 2
