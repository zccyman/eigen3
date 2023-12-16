# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/27 16:20
# @File     : __init__.py.py
# @Introduce: some testing will transfer into benchmark

# from .face_recognition import get_validation_pair, evaluate

from .eval_base import Eval
from .alfw import AlfwEval
from .coco_eval import CocoEval
from .face_recognition import RecEval
from .imagenet_eval import ClsEval
from .voc_eval import PascalVOCEval