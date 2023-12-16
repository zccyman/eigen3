# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/06/09 19:54
# @File     : test_encrypt.py
import sys  # NOQA: E402

sys.path.append("./")  # NOQA: E402

import os

import encrypt


def test_encrypt():
    oldpath = "work_dir/model.b"
    newpath = "work_dir/model_encode.b"
    newjiepath = "work_dir/model_decode.b"
    secret_key = "henson.zhang@timesintelli.com"

    print(encrypt.add(1, 2))
    encrypt.encode(oldpath, newpath, secret_key)
    encrypt.decode(newpath, newjiepath, secret_key)


if __name__ == "__main__":
    test_encrypt()
