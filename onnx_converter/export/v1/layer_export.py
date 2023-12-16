# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/1 17:06
# @File     : layer_export.py


class lExport(object):
    def __init__(self):
        self.data_channel_extension = None
        self.w_offset = None
        self.inserts = dict(is_align=True)
        self.w_offset = dict(w_offset=0, tmp_offset=[])

    def set_insert(self, inserts):
        self.inserts.update(inserts)

    def get_insert(self):
        return self.inserts

    def set_w_offset(self, offset):
        self.w_offset = offset

    def get_w_offset(self):
        return self.w_offset

    def set_data_channel_extension(self, data_channel_extension):
        self.data_channel_extension = data_channel_extension

    def get_data_channel_extension(self):
        return self.data_channel_extension
    
    def feat_export(
        self, func, feat, feat_id=0, insert=None, is_out=True, 
        layer=None, layer_name="", name="",
    ):
        kwargs = {
            "layer": layer,
            "layer_name": layer_name,
            "is_out": is_out,
            "data": feat,
            "param": self.get_insert() if insert is None else insert,
            "feat_id": feat_id,
        }
        return func(name, **kwargs)

    def weight_export(
        self, func, weights, insert=None, chip_type=None,
        layer=None, layer_name="", name="", 
        **kwargs,
    ):
        if "first_conv" in kwargs.keys():
            first_conv = kwargs["first_conv"]
        else:
            first_conv = False
        if "data_channel_extension" in kwargs.keys():
            data_channel_extension = kwargs["data_channel_extension"]
        else:
            data_channel_extension = False
        kwargs = {
            "chip_type": chip_type,
            "layer": layer,
            "layer_name": layer_name,
            "data": weights,
            "param": self.get_insert() if insert is None else insert,
            "first_conv": first_conv,
            "data_channel_extension": data_channel_extension,
        }
        return func(name, **kwargs)

    def bias_export(
        self, func, bias, is_fc_bias=False, insert=None, 
         layer=None, layer_name="", name=""
    ):
        kwargs = {
            "layer": layer,
            "layer_name": layer_name,
            "data": bias,
            "param": self.get_insert() if insert is None else insert,
            "is_fc_bias": is_fc_bias,
        }
        return func(name, **kwargs)

    def table_export(self, func, bias, is_fc_bias=False, layer_name="", name=""):
        kwargs = {
            "layer_name": layer_name,
            "data": bias,
        }
        return func(name, **kwargs)

    def batchnormal_export(self, func, var, layer_name="", name=""):
        kwargs = {"data": var, "param": self.get_insert(), "layer_name": layer_name}
        return func(name, **kwargs)

    def mean_export(self, func, mean, layer_name="", name=""):
        kwargs = {"data": mean, "param": self.get_insert(), "layer_name": layer_name}
        return func(name, **kwargs)

    def scale_export(self, func, scale, layer_name="", name=""):
        kwargs = {"data": scale, "param": self.get_insert(), "layer_name": layer_name}
        return func(name, **kwargs)

    def beta_export(self, func, beta, layer_name="", name=""):
        kwargs = {"data": beta, "param": self.get_insert(), "layer_name": layer_name}
        return func(name, **kwargs)
