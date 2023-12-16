# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/13 11:32
# @File     : checkerror.py

import os
import copy
import numpy as np

import torch
import torch.nn.functional as F
try:
    from utils import Registry
except:
    from onnx_converter.utils import Registry # type: ignore

error_factory: Registry = Registry(name='error', scope='')


# L1Error, L2Error, ConsineError
# PearsonCorrelationSimilarity, SpearmanCorrelationSimilarity,
# Tanimoto, LogLikelihoodSimilarity, CityBlockSimilarity
# Euclidean Distance-based Similarity, Adjusted Cosine Similarity

@error_factory.register_module(name='L1')
class L1Simiarity(object):
    def __init__(self, **kwargs):
        super(L1Simiarity, self).__init__()
        self.eps = 1e-5

    def __call__(self, simulation_data, true_data):
        s_data = copy.deepcopy(simulation_data).astype(np.float32)
        t_data = copy.deepcopy(true_data).astype(np.float32)
        s_data, t_data = torch.from_numpy(s_data), torch.from_numpy(t_data)
        diff = t_data.reshape(-1) - s_data.reshape(-1)
        sum = torch.abs(t_data).sum()
        sum = self.eps if sum == 0 else sum
        rate = torch.abs(diff).sum() * 100 / (sum + self.eps)
        return np.float32(rate)


@error_factory.register_module(name='L2')
class L2Simiarity(object):
    def __init__(self, **kwargs):
        super(L2Simiarity, self).__init__()
        self.eps = 1e-5

    def __call__(self, simulation_data, true_data):
        s_data = copy.deepcopy(simulation_data).astype(np.float32) 
        t_data = copy.deepcopy(true_data).astype(np.float32)
        s_data, t_data = torch.from_numpy(s_data), torch.from_numpy(t_data)
        diff = t_data.reshape(-1) - s_data.reshape(-1)
        sum = torch.square(t_data).sum()
        sum = self.eps if sum == 0 else sum
        rate = torch.square(diff).sum() * 100 / (sum + self.eps)
        return np.float32(rate)


@error_factory.register_module(name='Cosine')
class CosineSimiarity(object):
    def __init__(self, **kwargs):
        super(CosineSimiarity, self).__init__()
        self.eps = 1e-5
       
    def __call__(self, simulation_data, true_data):
        s_data = copy.deepcopy(simulation_data).astype(np.float32)
        t_data = copy.deepcopy(true_data).astype(np.float32)
        s_data = torch.from_numpy(s_data.reshape(-1))
        t_data = torch.from_numpy(t_data.reshape(-1))
        normal = torch.sqrt(torch.sum(s_data * s_data) * torch.sum(t_data * t_data))
        if normal == 0:
            if torch.sum(torch.abs(s_data)) == 0 and torch.sum(torch.abs(t_data)) == 0:
                dist = torch.ones(1)
            else:
                dist = torch.zeros(1)
        else:
            dist = torch.sum(s_data * t_data) / (normal + self.eps)
        dist = (1- np.abs(dist.item())) * 100

        return np.float32(dist)



class SpearmanCorrelationSimilarity(object):
    def __init__(self, **kwargs):
        pass


class PearsonCorrelationSimilarity(object):
    def __init__(self, **kwargs):
        pass


class TanimotoSimiarity(object):
    def __init__(self, **kwargs):
        pass


class LogLikelihoodSimilarity(object):
    def __init__(self, **kwargs):
        pass


class CityBlockSimilarity(object):
    def __init__(self, **kwargs):
        pass


class EuclideanSimilarity(object):
    def __init__(self, **kwargs):
        pass


class AdjustConsineSimiarity(object):
    def __init__(self, **kwargs):
        pass
