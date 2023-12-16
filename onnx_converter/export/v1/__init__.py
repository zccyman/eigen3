# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/07/06 13:29:08
# @File    : __init__.py
from uuid import NAMESPACE_URL, uuid5

__version__ = "2.1.0"


def get_version():
    # return uuid5(NAMESPACE_URL, __version__)
    return __version__
