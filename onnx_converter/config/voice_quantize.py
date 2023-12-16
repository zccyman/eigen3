# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : TIMESINETLLI TECH
# @Time     : 2022/6/10 17:25
# @File     : voice_quantize.py


bit_select, int_scale, pre_int_scale, out_type = 1, 8, 8-1, 8
# bit_select, int_scale, pre_int_scale, out_type = 3, 16, 16-1, 8
txme_saturation = 1
virtual_round = 0

is_last_layer_fuse_act = True
is_fuse_linear_act = True # if is_fuse_linear_act=True, use shiftfloattable, otherwise use intscale;
is_fuse_nonlinear_act = True # if is_fuse_nonlinear_act=True, use shiftfloattable, otherwise nonlinear act will not be fused into conv/fc;

act = []
if is_fuse_nonlinear_act:
    fuse_act = ["swish", "leakyrelu", "hardswish", "hardsigmoid", "tanh", "sigmoid"]
else:
    fuse_act = []
if is_fuse_linear_act:
    fuse_act.extend(["relu", "relu6", "relux"])
else:
    act.extend(["relu", "relu6", "relux", "act"])

normal = dict(method='floatsymquan', bit_select=bit_select)

perchannel = dict(method='perchannelfloatsymquan', bit_select=bit_select)

int16 = dict(method='floatsymquan', bit_select=3)

feat = dict(method='floatsymquan', bit_select=bit_select)

# output = dict(process_scale='ffloatscale', out_type=8)
output = dict(layer_type=['conv', 'depthwiseconv', 'gemm', 'matmul', 'fc'],
            #   weights=dict(method='floatsymquan', bit_select=bit_select),
            #   feat=dict(method='floatsymquan', bit_select=bit_select),
              process_scale='ffloatscale', out_type=8)  ###rshiftscale ffloatscale

default_setting = dict(data=dict(feat=feat, process_scale='floatscale', int_scale=int_scale, out_type=8),
                       fc=dict(weights=normal, feat=feat, process_scale='shiftfloatscaletable2float', int_scale=int_scale, out_type=8),
                       gemm=dict(weights=normal, feat=feat, process_scale='shiftfloatscaletable2float', int_scale=int_scale, out_type=8),
                       lstm=dict(weights=normal, feat=feat, process_scale='ffloatscale', int_scale=int_scale, out_type=8, hx_combine=False, wr_combine=False),
                       gru=dict(weights=normal, feat=feat, process_scale='ffloatscale', out_type=8, hx_combine=False, wr_combine=False),
                       splice=dict(weights=normal, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       mul=dict(weights=normal, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       cmul=dict(weights=normal, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       pmul=dict(weights=normal, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       concat=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       batchnormalization=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       layernormalization=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       sigmoid=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       log=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=8),
                       reshape=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=8),
                       )
