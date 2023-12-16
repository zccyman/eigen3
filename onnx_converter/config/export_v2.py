# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/13 14:09
# @File     : export_v2.py


is_debug = False
is_voice_model = False
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
    is_voice_model=is_voice_model,    
    is_debug=is_debug,
)

export_version = 2
