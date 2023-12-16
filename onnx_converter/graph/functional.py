# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/27 15:40
# @File     : functional.py

import os
import copy
import numpy as np


def normal_batchnormalization(weights: list) -> dict:
    attrs = {'scale': copy.deepcopy(weights[0]['weight']),
             'bias': copy.deepcopy(weights[1]['weight']),
             'mean': copy.deepcopy(weights[2]['weight']),
             'var': copy.deepcopy(weights[3]['weight']),
             'in_c': copy.deepcopy(weights[0]['weight'].shape[0])}

    indexes = attrs['var']<weights[-1]['epsilon']
    attrs['mean'][indexes] = 0
    attrs['var'][indexes] = 1
    attrs['scale'][indexes] = 0
    attrs['bias'][indexes] = 0
    del weights[-1]
    return attrs


def normal_layernormalization(weights: list) -> dict:
    attrs = {'scale': weights[0]['weight'],
             'bias': weights[1]['weight'],
            }
    return attrs


def normal_conv(weights: list) -> dict:
    attrs = dict(in_c=weights[0]['weight'].shape[1], out_c=weights[0]['weight'].shape[0], bias=True)
    if len(weights) < 2 or weights[1]['weight'] is None:
        attrs['bias'] = False
    return attrs


def normal_convtranspose(weights: list) -> dict:
    attrs = dict(in_c=weights[0]['weight'].shape[0], out_c=weights[0]['weight'].shape[1], bias=True)
    if len(weights) < 2 or weights[1]['weight'] is None:
        attrs['bias'] = False
    return attrs


def normal_concat(weights: list) -> dict:
    return dict()


def normal_add(weights: list) -> dict:
    if len(weights) > 0:
        weight = weights[0]['weight']
    else:
        weight = None
    return dict(weights=weight, bias=False)


def normal_log(weights: list) -> dict:
    return dict()


def normal_mul(weights: list) -> dict:
    if len(weights) > 0:
            weight = weights[0]['weight']
    else:
        weight = None
    return dict(weights=weight, bias=False)


def normal_lstm(weights: list) -> dict:
    if len(weights) > 2:
        bias = weights[2]['weight'].reshape(1, 2, -1)
        weights[2] = dict(weight=bias[:, 0, :])
        weights.append(dict(weight=bias[:, 1, :]))
        # weights.update(dict(bias=True))
    # else:
        # weights.update(dict(bias=False))
    return dict(weights=weights)


def normal_gru(weights: list) -> dict:
    if len(weights) > 2:
        bias = weights[2]['weight'].reshape(1, 2, -1)
        weights[2] = dict(weight=bias[:, 0, :])
        weights.append(dict(weight=bias[:, 1, :]))
        bias=True
    else:
        bias=False
    return dict(weights=weights, bias=bias)


def normal_matmul(weights: list) -> dict:
    if len(weights) > 0:
        return dict(in_c=weights[0]['weight'].shape[1], out_c=weights[0]['weight'].shape[0], weights=weights[0]['weight'], bias=False)
    else:
        return dict(in_c=weights[0]['weight'].shape[1], out_c=weights[0]['weight'].shape[0], weights=None, bias=False)


def normal_gemm(weights: list) -> dict:
    bias = len(weights) > 1
    bias = True
    return dict(in_c=weights[0]['weight'].shape[1], out_c=weights[0]['weight'].shape[0], bias=True)


def normal_resize(weights: list) -> dict:
    attrs = dict()
    if len(weights) == 1:
        if weights[0]['weight'].dtype == 'int64':
            attrs.update(dict(roi=None, scale=None, sizes=weights[0]['weight']))
        else:
            attrs.update(dict(roi=None, scale=weights[0]['weight'], sizes=None))
    elif len(weights) == 2:
        attrs.update(dict(roi=weights[0]['weight'], scale=weights[1]['weight'], sizes=None))
    elif len(weights) == 3:
        attrs.update(dict(roi=weights[0]['weight'], scale=weights[1]['weight'], sizes=weights[2]['weight']))
    else:
        print('not enough resize parameter!')
        os._exit(-1)
    return attrs

def normal_upsample(weights: list) -> dict:
    attrs = dict()
    if len(weights) > 0:
        attrs.update(dict(scales=weights[0]['weight']))
    return attrs


def normal_softmax(weights: list) -> dict:
    attrs = dict()
    return attrs


def parse_flatten(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def normal_reducesum(weights: list) -> dict:
    return dict(axes=weights[0]['dims'])


def normal_reshape(weights: list) -> dict:
    return dict(shape=weights[0]['weight'], dims=weights[0]['dims'])


def normal_pad(weights: list) -> dict:
    return dict(pads=weights[0]['weight'], constant_value=weights[1]['weight'])


def normal_squeeze(weights: list) -> dict:
    return dict()


def normal_topk(weights: list) -> dict:
    return dict(topk=weights[0]['weight'])


def normal_gather(weights: list) -> dict:
    return dict(indices=weights[0]['weight'])


def normal_range(weights: list) -> dict:
    if len(weights) == 2:
        attrs = dict(start=weights[0]['weight'], delta=weights[1]['weight'])
    elif len(weights) == 3:
        attrs = dict(start=weights[0]['weight'], limit=weights[1]['weight'], delta=weights[2]['weight'])
    else:
        attrs = dict()
    return attrs


def normal_clip(weights: list) -> dict:
    values = np.array([item['weight'] for item in weights])
    return dict(min=min(values), max=max(values))


def normal_constantofshape(weights: list) -> dict:
    return dict(constantofshape=weights[0]['weight'])


def normal_div(weights: list) -> dict:
    return dict(dividend=weights[0]['weight'])


def normal_less(weights: list) -> dict:
    return dict(value=weights[0]['weight'])


def normal_where(weights: list) -> dict:
    return dict(value=weights[0]['weight'])


def normal_expand(weights: list) -> dict:
    return dict(value=weights[0]['weight'], dims=weights[0]['dims'])


def normal_nonmaxsuppression(weights: list) -> dict:
    attrs = dict(max_output_boxes_per_class=weights[0]['weight'])
    if len(weights) == 2:
        attrs.update(iou_threshold=weights[1]['weight'])
    elif len(weights) == 3:
        attrs.update(iou_threshold=weights[1]['weight'], score_threshold=weights[2]['weight'])
    return attrs


def normal_equal(weights: list) -> dict:
    return dict(value=weights[0]['weight'])


def normal_slice(weights: list) -> dict:
    attrs = dict(starts=weights[0]['weight'], ends=weights[1]['weight'])
    if len(weights) == 2:
        attrs.update(axes=None, steps=None)
    elif len(weights) == 3:
        attrs.update(axes=weights[2]['weight'], steps=None)
    elif len(weights) == 4:
        attrs.update(axes=weights[2]['weight'], steps=weights[3]['weight'])
    else:
        os._exit(-1)  # , print('not enough slice paramter!')
    return attrs


def normal_relu(weights: list) -> dict:
    return dict()


def normal_leakyrelu(weights: list) -> dict:
    return dict()


def normal_prelu(weights: list) -> dict:
    return dict(slope=weights[0]['weight'])


def normal_instancenormalization(weights: list) -> dict:
    attrs = {'scale': weights[0]['weight'],
             'bias': weights[1]['weight'],
             'in_c': weights[0]['weight'].shape[0],
             }
    return attrs


def normal_scatternd(weights: list) -> dict:
    if len(weights) > 0:
        return dict(indices=weights[0]['weight'])
    else:
        return dict()


def normal_split(weights: list):
    if len(weights) > 0:
        return dict(split=weights[0]['weight'])
    else:
        return dict()


def normal_tile(weights: list) -> dict: # type: ignore
    if len(weights) > 0:
        attrs = dict(dims=weights[0]['dims'], tile=weights[0]['weight'])
    else:
        return dict()


def normal_sub(weights: list) -> dict:
    if len(weights) > 0:
        weight = weights[0]['weight']
    else:
        weight = None
    return dict(weights=weight, bias=False)


def fuseBN(mean, var, gamma, beta, _weight, epsilon=0.001, _bias=np.empty(shape=[0]), is_convtranspose=False):
    fused_gamma = gamma / np.sqrt(var + epsilon)
    if len(_weight.shape) == 2: ### bn fused into fc
        new_weight = _weight * fused_gamma.reshape(-1, 1)
        out_c = _weight.shape[0]
    else:
        if is_convtranspose:
            new_weight = _weight * fused_gamma.reshape(1, -1, 1, 1)
            out_c = _weight.shape[1]
        else:
            new_weight = _weight * fused_gamma.reshape(-1, 1, 1, 1)
            out_c = _weight.shape[0]
    if len(_bias) <= 0:
        _bias = np.zeros(out_c, dtype=np.float32)
    new_bias = (_bias - mean) * fused_gamma + beta
    # zero_idx = np.where(var < epsilon)[0]
    # if zero_idx is not None:
    #     new_weight[zero_idx] = 0
    #     new_bias[zero_idx] = 0

    return new_weight, new_bias


def fuse_batchnormalization_into_conv_fc(nodes: list) -> list:
    attr, weights, bn_weights = nodes[1].get_attr(), nodes[1].get_weights(), nodes[0].get_weights()
    bn_weights_ = list()
    for weight in bn_weights:
        bn_weights_.append(weight['weight'])
    attrs = nodes[0].get_attr()
    attr = nodes[1].get_attr()
    if 'epsilon' in attrs.keys():
        epsilon = attrs['epsilon']
    else:
        epsilon = 1e-5
    weight = weights[0]['weight']
    bn_weight, bn_bias, running_mean, running_var = bn_weights_
    if attr['bias']:
        bias = weights[1]['weight']
    else:
        bias = np.zeros_like(bn_bias)

    minmax_var = [1.0, 1.0]
    if len(weight.shape) < 4:
        def fuse_bn_into_fc(weight, bias, mean, var, bn_scale, bn_bias, epsilon=0.001):
            new_scale = bn_scale / np.sqrt(var + epsilon)
            new_bias = np.matmul(bn_bias - new_scale * mean, np.transpose(weight, (1, 0))) #np.transpose(weight, (1, 0))
            new_bias += bias
            new_weight = weight * new_scale    
            return new_weight, new_bias         

        minmax_var = [running_var.min(), running_var.max()]
        weight, bias = fuse_bn_into_fc(weight, bias, running_mean, running_var, bn_weight, bn_bias,
                              epsilon)

    attr['bias'] = True
    nodes[1].set_attr(attr)
    nodes.remove(nodes[0])
    weights = list()
    weights.append(dict(name='weight', weight=weight))
    weights.append(dict(name='bias', weight=bias))
    # weights.append(dict(name='bn_weights', weight=[
    #     bn_weights[0]["weight"], 
    #     bn_weights[1]["weight"],
    #     bn_weights[2]["weight"],
    #     bn_weights[3]["weight"],
    # ]))
    nodes[0].set_weights(weights)

    return nodes, minmax_var # type: ignore


def fuse_batchnormalization(nodes: list) -> list:
    attr, weights, bn_weights = nodes[0].get_attr(), nodes[0].get_weights(), nodes[1].get_weights()
    bn_weights_ = list()
    for weight in bn_weights:
        bn_weights_.append(weight['weight'])
    attrs = nodes[1].get_attr()
    attr = nodes[0].get_attr()
    if 'epsilon' in attrs.keys():
        epsilon = attrs['epsilon']
    else:
        epsilon = 1e-5
    weight = weights[0]['weight']
    bn_weight, bn_bias, running_mean, running_var = bn_weights_
    if attr['bias']:
        bias = weights[1]['weight']
    else:
        bias = np.zeros_like(bn_bias)
    
    if not 'transB' in attr.keys() and nodes[0].get_op_type() in ["MatMul"]: #and nodes[0].get_op_type() != "MatMul":#if matmul, transpose operation + weight.reshape(1, 1, fc_in, fc_out) will incorrectly disorder weight
        weight = np.transpose(weight, (1, 0))
        attr['transB'] = True

    is_convtranspose = False
    if "convtranspose" == nodes[0].get_op_type().lower():
        is_convtranspose = True
    weight, bias = fuseBN(running_mean, running_var, bn_weight, bn_bias, weight, epsilon, bias, is_convtranspose=is_convtranspose)

    attr['bias'] = True
    nodes[0].set_attr(attr)
    nodes.remove(nodes[1])
    weights = list()
    weights.append(dict(name='weight', weight=weight))
    weights.append(dict(name='bias', weight=bias))
    nodes[0].set_weights(weights)

    return nodes


def setting_conv(setting: dict):
    pass


def setting_depthwise(setting: dict):
    pass


def setting_fc(setting: dict):
    pass


def setting_matmul(setting: dict):
    pass


def setting_gemm(setting: dict):
    pass


def setting_act(setting: dict):
    pass


def setting_maxpool(setting: dict):
    pass


def setting_avgpool(setting: dict):
    pass


def setting_shuffle(setting: dict):
    pass


def setting_globalaveragepool(setting: dict):
    pass
