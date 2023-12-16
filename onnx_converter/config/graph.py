# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/8 11:11
# @File     : graph.py

# act=['relu', 'relu6', 'clip', 'leakyrelu']
# act=['relu', 'relu6', 'clip']
# act=['relu']
# act = ["relu"]
# act = [] #["relu", "relu6", "relux"]
data = dict(data=['data'], is_shorten=False, order_num=0)
conv = dict(conv=['conv', 'act'], is_shorten=True, order_num=0)
convtranspose = dict(convtranspose=['convtranspose', 'act'], is_shorten=True, order_num=0)
# shuffle = dict(concat=['concat', 'reshape', 'transpose', 'reshape'],
#                extra=[['slice', 'slice'], ['split']], is_shorten=False, order_num=4)
# concat_shuffle_only = dict(concat=['concat', 'reshape', 'transpose', 'reshape'],
#                            extra=[['slice', 'slice'], ['split'], []], is_shorten=False, order_num=4)
shuffle_only = dict(reshape=['reshape', 'transpose', 'reshape'], is_shorten=False, order_num=3)
# shuffle_only_split = dict(reshape=['reshape', 'transpose', 'reshape'],
#                           extra=[['slice', 'slice'], ['split'], []], is_shorten=False, order_num=3)
matmul = dict(matmul=['matmul', 'act'], is_shorten=True, order_num=0)
fc = dict(gemm=['gemm', 'reshape', 'act'], is_shorten=True, order_num=0)
gemm = dict(gemm=['gemm', 'act'], is_shorten=True, order_num=0)
swish = dict(sigmoid=['sigmoid', 'mul'], is_shorten=False, order_num=1)
gelu = dict(div=['div', 'erf', 'add', 'mul', 'mul'], is_shorten=False, order_num=4)
batchnormalization = dict(batchnormalization=['batchnormalization', 'gemm'],
                          is_shorten=True,
                          order_num=0)

especial_ops = dict(data=data,
                    conv=conv,
                    convtranspose=convtranspose,
                    # shuffle=shuffle,
                    shuffle_only=shuffle_only,
                    matmul=matmul,
                    fc=fc,
                    gemm=gemm,
                    swish=swish,
                    gelu=gelu,
                    batchnormalization=batchnormalization,
                    )

empty_ignore = ['data', 'maxpool', 'relu']

weights_ignore = []

fuse_ops = dict(gemm=['BatchNormalization'],
                matmul=['BatchNormalization'],
                fc=['BatchNormalization'],
                conv=['BatchNormalization'],
                convtranspose=["BatchNormalization"],
                depthwiseconv=['BatchNormalization'],
                batchnormalization=['Gemm']
                )

split_ops = dict(name=['conv', 'gemm', 'depthwiseconv', 'convtranspose'])

replace_ops = dict(resize=['resize', 'upsample'],
                   fc=['gemm'],
                   batchnormalization=['batchnormalization'])