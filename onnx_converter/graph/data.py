# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/8 13:24
# @File     : data.py


class ConstantData(object):
    def __init__(self):
        self.f_scale = 0
        self.w_scale = 0
        self.final_scale = 0
        self.bias_scale = 0

    def set_fscale(self, scale):
        self.f_scale = scale

    def set_wscale(self, scale):
        self.w_scale = scale

    def set_finalscale(self, scale):
        self.final_scale = scale

    def set_bscale(self, scale):
        self.bias_scale = scale

    def get_fscale(self):
        return self.f_scale

    def get_wscale(self):
        return self.w_scale

    def get_finalscale(self):
        return self.final_scale

    def get_bscale(self):
        return self.bias_scale


class CalcParam(object):
    def __init__(self):
        self.weights, self.bias, self.feature = 0
        self.qweights, self.qbias, self.qfeature = 0

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_feature(self):
        return self.feature

    def get_qweight(self):
        return self.qweights

    def get_qbias(self):
        return self.qbias

    def get_qfeature(self):
        return self.qfeature

    def set_weights(self, weights):
        self.weights = weights

    def set_qweights(self, quan_weights):
        self.quan_weights = quan_weights

    def set_bias(self, bias):
        self.bias = bias

    def set_qbias(self, quan_bias):
        self.qbias = quan_bias

    def set_feature(self, feature):
        self.feature = feature

    def set_qfeature(self, quan_feat):
        self.qfeature = quan_feat

