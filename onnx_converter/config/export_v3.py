# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/13 14:09
# @File     : export_v3.py


is_debug = False
is_voice_model = False
export_mode_c = False
chip_type = "AT5050"  # [AT1K, AT5050,]
save_placeholder_params = True

AT5050_C_EXTEND = chip_type != "AT1K"
# Csize, Ksize: input align size, output align size
Csize, Ksize = (16, 32) if AT5050_C_EXTEND else (8, 8)
O_Align, I_Align = (16, 16)  if AT5050_C_EXTEND else (4, 8) # fc output/input align size

fmt = 0
bits = dict(
    Csize=Csize,
    Ksize=Ksize,
    I_Align=I_Align,
    O_Align=O_Align,
    ABGR=True, ## if first layer is fc, ABGR=False
    DATA_C_EXTEND=AT5050_C_EXTEND,
    save_placeholder_params=save_placeholder_params,
    chip_type=chip_type,
    bgr_format=True,
    is_voice_model=is_voice_model,
    is_debug=is_debug,
)

LayerQuant = dict(
    QUANT_NONE=["", "smooth", "table", "float", "asy", "table_asy"],
    QUANT_LUT8_FP=["shiftfloatscaletable2float"],
    QUANT_PER_CHN_LUT8_FP=["perchannel_shiftfloatscaletable2float"],
    QUANT_SHIFT=[
        "rshiftscale", 
        "shiftscale", 
        "shiftfloatscaletable",
    ],
    QUANT_SHIFT_ASY=[
        "rshiftscale_asy", 
        "shiftscale_asy", 
        "shiftfloatscaletable_asy",
    ],
    QUANT_ISCALE=["preintscale", "preintscaleex", "intscale", "intscaleex"],
    QUANT_ISCALE_ASY=["preintscale_asy", "preintscaleex_asy", "intscale_asy", "intscaleex_asy"],
    QUANT_FSCALE=[
        "floatscale",
        "ffloatscale",
        "rrshiftscale",
        "shiftfloatscale",
    ],
    QUANT_FSCALE_ASY=[
        "floatscale_asy",
        "ffloatscale_asy",
        "rrshiftscale_asy",
        "shiftfloatscale_asy",
    ],    
    QUANT_PER_CHN_SHIFT=[
        "perchannel_rshiftscale", 
        "perchannel_shiftscale", 
        "perchannel_shiftfloatscaletable",
    ],
    QUANT_PER_CHN_ISCALE=["perchannel_preintscale", "perchannel_intscale"],
    QUANT_PER_CHN_FSCALE=[
        "perchannel_floatscale",
        "perchannel_ffloatscale",
        "perchannel_rrshiftscale",
        "perchannel_shiftfloatscale",
    ],
    QUANT_PER_CHN_SHIFT_ASY=[
        "perchannel_rshiftscale_asy", 
        "perchannel_shiftscale_asy", 
        "perchannel_shiftfloatscaletable_asy",
    ],
    QUANT_PER_CHN_ISCALE_ASY=["perchannel_preintscale_asy", "perchannel_intscale_asy"],
    QUANT_PER_CHN_FSCALE_ASY=[
        "perchannel_floatscale_asy",
        "perchannel_rshiftscale_asy",
        "perchannel_ffloatscale_asy",
        "perchannel_rrshiftscale_asy",
        "perchannel_shiftfloatscale_asy",
    ],
    QUANT_QUANT=["quant", "float_quant", "smooth_quant"],
    QUANT_DEQUANT=["dequant", "float_dequant"],
    QUANT_PER_CHN_QUANT=["perchannel_quant"],
    QUANT_PER_CHN_DEQUANT=["perchannel_dequant"],
    QUANT_QUANT_ASY=["quant_asy", "float_quant_asy", "smooth_quant_asy", "smooth_asy"],
    QUANT_DEQUANT_ASY=["dequant_asy"],
    QUANT_PER_CHN_QUANT_ASY=["perchannel_quant_asy"],
    QUANT_PER_CHN_DEQUANT_ASY=["perchannel_dequant_asy"],
)

export_version = 3
