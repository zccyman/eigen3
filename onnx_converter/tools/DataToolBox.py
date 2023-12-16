# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/19 14:08
# @File     : DataToolBox.py

try:
    from utils import Registry, Object
except:
    from onnx_converter.utils import Registry, Object # type: ignore

toolbox_factory = Registry('toolbox', scope='')


@toolbox_factory.register_module(name='svd')
class SVD(Object): # type: ignore
    def __init__(self, **kwargs):
        super(SVD, self).__init__(**kwargs)


@toolbox_factory.register_module(name='kl-divergence')
class KL(Object): # type: ignore
    def __init__(self, **kwargs):
        super(KL, self).__init__(**kwargs)


@toolbox_factory.register_module(name='pca')
class PCA(Object): # type: ignore
    def __init__(self, **kwargs):
        super(PCA, self).__init__(**kwargs)
        

@toolbox_factory.register_module(name='ica')
class ICA(Object): # type: ignore
    def __init__(self, **kwargs):
        super(ICA, self).__init__(**kwargs)


@toolbox_factory.register_module(name='histogram')
class Histogram(Object): # type: ignore
    def __init__(self, **kwargs):
        super(Histogram, self).__init__(**kwargs)
