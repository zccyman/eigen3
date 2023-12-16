# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/18 16:12
# @File     : Optimizer.py

try:
    from utils import Object
except:
    from onnx_converter.utils import Object # type: ignore


class OptimizerQuantize(Object): # type: ignore
    def __init__(self, **kwargs):
        super(OptimizerQuantize, self).__init__(**kwargs)
        if 'log_name' in kwargs.keys():
            self.logger = self.get_log(log_name=kwargs['log_name'], log_level=kwargs.get('log_level', 20))
        else:
            self.logger = self.get_log(log_name='onnx_process.log', log_level=kwargs.get('log_level', 20))

        self.logger.info('_________________________________________________________________________________')
        self.logger.info('start optimizer quantize network!')
