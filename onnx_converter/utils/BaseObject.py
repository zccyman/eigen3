# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/30 17:15
# @File     : BaseObject.py

import os
from .logger import Logging


class Object(object):
    def __init__(self, **kwargs):
        self.log_dir = 'work_dir'
        if 'log_dir' in kwargs.keys():
            self.log_dir = kwargs['log_dir']
        self.log_dir = os.path.join(os.getcwd(), self.log_dir)
        self.log_name = ''
        if 'log_name' in kwargs.keys():
            self.log_name = kwargs['log_name']

    def get_class_name(self):
        return self.__class__.__name__

    def remove(self):
        for hdlr in self.logger.handlers: # type: ignore
            self.logger.removeHandler(hdlr) # type: ignore

    def get_log(self, log_name, log_level, stdout=True):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, mode=0o777)
        if log_name and log_name != '':
            log_file = os.path.join(os.getcwd(), self.log_dir, log_name)
        else:
            log_file = os.path.join(os.getcwd(), self.log_dir, self.log_name)
        # if os.path.exists(log_file):
        #     os.remove(log_file)
        logger = Logging()
        logger = logger.get_root_logger(logger, name=log_file, log_file=log_file,
                                        level=log_level, stdout=stdout)
        return logger
