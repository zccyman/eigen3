# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/18 19:33
# @File     : attribute.py

import os
import numpy as np


def parse_data(attr: list) -> dict:
    if attr['shape'] != attr['input_shape']: # type: ignore
        attr['shape'] = attr['input_shape'] # type: ignore
    return attr # type: ignore


def parse_conv(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['ints']
        if item['name'] == 'group':
            val = item['i']
        if item['name'] == 'auto_pad':
            val = str(item['mode'], 'utf-8')
        attrs[item['name']] = val
    return attrs


def parse_convtranspose(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['ints']
        if item['name'] == 'group':
            val = item['i']
        if item['name'] == 'auto_pad':
            val = str(item['mode'], 'utf-8')
        attrs[item['name']] = val
    return attrs


def parse_maxpool(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['ints']
        attrs[item['name']] = val
        if 'ceil_mode' == item['name']:
            attrs[item['name']] = item['i']
        if item['name'] == 'auto_pad':
            attrs[item['name']] = str(item['mode'], 'utf-8')
    return attrs


def parse_averagepool(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['ints']
        attrs[item['name']] = val
        if item['name'] in ['ceil_mode', 'count_include_pad']:
            attrs[item['name']] = item['i']
        if item['name'] == 'auto_pad':
            attrs[item['name']] = str(item['mode'], 'utf-8')
    return attrs


def parse_globalaveragepool(attr: list) -> dict:
    return dict()


# linear no bias
def parse_matmul(attr: list) -> dict:
    if len(attr) > 0:
        return attr[0]
    else:
        return dict()    
    # return dict(transB=True, alpha=1.0, beta=1.0)


def parse_lstm(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['i']
        attrs[item['name']] = val
    return attrs


def parse_splice(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        item_name = item['name']
        if item_name == "has_fc":
            val = item['i']
        elif item_name in ["weight", "bias"]:
            val = item['floats']
        else:
            val = item['ints']
        attrs[item_name] = val
    
    attrs["has_fc"] = 0
    if attrs["has_fc"]:
        assert len(attrs["bias"]) > 0
        if len(attrs["weight"]) and len(attrs["bias"]):
            attrs["bias"] = np.array(attrs["bias"])
            attrs["weight"] = np.array(attrs["weight"]).reshape(attrs["bias"].shape[0], -1)

    return attrs

def parse_gru(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['i']
        # if item['name'] == "activations":
        #    val = item['type']
        attrs[item['name']] = val

    return attrs


# linear has bias
def parse_gemm(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['float']
        if item['name'] == 'transB':
            val = item['i']
        attrs[item['name']] = val
    return attrs


def parse_add(attr: list) -> dict:
    if len(attr) > 0:
        return attr[0]
    else:
        return dict()


def parse_log(attr: list) -> dict:
    return dict()


def parse_relu(attr: list) -> dict:
    return dict()


def parse_sigmoid(attr: list) -> dict:
    return dict()


def parse_batchnormalization(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        type = item['type']
        if 'float' in type:
            val = item['float']
        else:
            val = item['i']
        attrs[item['name']] = val

    ### The default value comes from address:
    ### https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization
    if "epsilon" not in attrs.keys():
        attrs["epsilon"] = 1.0e-05
    if "momentum" not in attrs.keys():
        attrs["momentum"] = 0.9
    if "training_mode" not in attrs.keys():
        attrs["training_mode"] = 0

    return attrs


def parse_layernormalization(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['i']
        attrs[item['name']] = val
    return attrs


def parse_leakyrelu(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['float']
        attrs[item['name']] = val
    return attrs


def parse_prelu(attr: list) -> dict:
    return dict()


def parse_slice(attr: list) -> dict:
    return dict()


def parse_reshape(attr: list) -> dict:
    return dict()


def parse_pad(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['mode']
        attrs[item['name']] = str(val, 'utf-8')    
    return attrs


def parse_transpose(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['ints']
        attrs[item['name']] = val
    return attrs


def parse_concat(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        if "weight_idx" in item.keys():
            attrs.update(item)
            continue
        val = item['i']
        attrs[item['name']] = val
    return attrs


def parse_dropout(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['float']
    return attrs


# if coordinate_transformation_mode is "half_pixel",
# x_original = (x_resized + 0.5) / scale - 0.5,
#
# if coordinate_transformation_mode is "pytorch_half_pixel",
# x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0,
#
# if coordinate_transformation_mode is "align_corners",
# x_original = x_resized * (length_original - 1) / (length_resized - 1),
#
# if coordinate_transformation_mode is "asymmetric",
# x_original = x_resized / scale,
#
# if coordinate_transformation_mode is "tf_crop_and_resize",
# x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).
# cubic_coeff_a : float (default is -0.75)
# The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. This attribute is valid only if "mode" is "cubic".
# exclude_outside : int (default is 0)
# If set to 1, the weight of sampling locations outside the tensor will be set to 0 and the weight will be renormalized so that their sum is 1.0. The default value is 0.
# extrapolation_value : float (default is 0.0)
# When coordinate_transformation_mode is "tf_crop_and_resize" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f.
# mode : string (default is nearest)
# Three interpolation modes: nearest (default), linear and cubic. The "linear" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). The "cubic" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).
# nearest_mode : string (default is round_prefer_floor)
# Four modes: round_prefer_floor (default, as known as round half down), round_prefer_ceil (as known as round half up), floor, ceil. Only used by nearest interpolation. It indicates how to get "nearest" pixel in input tensor from x_original, so this attribute is valid only if "mode" is "nearest".
def parse_resize(attr: list) -> dict:
    attrs = dict()
    for attr_ in attr:
        name = attr_["name"]
        if name in ["cubic_coeff_a"]:
            attrs[name] = float(attr_['float'])
        elif name in [
            "coordinate_transformation_mode", 
            "mode", "nearest_mode"]:
            attrs[name] = str(attr_['mode'], 'utf-8')
        else:
            raise NotImplementedError
        
    return attrs


def parse_upsample(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = str(attr[0]['mode'], 'utf-8')
    return attrs


def parse_unsqueeze(attr: list) -> dict:
    attrs = dict()
    # attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_squeeze(attr: list) -> dict:
    attrs = dict()
    # attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_mul(attr: list) -> dict:
    if len(attr) > 0:
        return attr[0]
    else:
        return dict()


def parse_gather(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_cast(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_clip(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        attrs[item['name']] = item['float']
    return attrs


def parse_lrn(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        val = item['float']
        if item['name'] == 'size':
            val = item['i']
        attrs[item['name']] = val
    return attrs


def parse_scatter(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_scatternd(attr: list) -> dict:
    return dict()


def parse_constantofshape(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_split(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        attrs[item['name']] = item['i']
    return attrs


def parse_flatten(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_shape(attr: list) -> dict:
    return dict()


def parse_range(attr: list) -> dict:
    return dict()


def parse_where(attr: list) -> dict:
    return dict()


def parse_tile(attr: list) -> dict:
    return dict()


def parse_nonzero(attr: list) -> dict:
    return dict()


def parse_expand(attr: list) -> dict:
    return dict()


def parse_equal(attr: list) -> dict:
    return dict()


def parse_roialign(attr: list) -> dict:
    attrs = dict()
    for item in attr:
        if item['name'] == 'spatial_scale':
            attrs[item['name']] = item['float']
        else:
            attrs[item['name']] = item['i']
    return attrs


def parse_sub(attr: list) -> dict:
    if len(attr) > 0:
        return attr[0]
    else:
        return dict()


def parse_div(attr: list) -> dict:
    return dict()


def parse_sqrt(attr: list) -> dict:
    return dict()


def parse_softmax(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_topk(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_log(attr: list) -> dict:
    return dict()


def parse_nonmaxsuppression(attr: list) -> dict:
    return dict()


def parse_floor(attr: list) -> dict:
    return dict()


def parse_less(attr: list) -> dict:
    return dict()


def parse_exp(attr: list) -> dict:
    return dict()


def parse_not(attr: list) -> dict:
    return dict()


def parse_and(attr: list) -> dict:
    return dict()


def parse_greater(attr: list) -> dict:
    return dict()


def parse_constant(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_reducel2(attr: list) -> dict:
    return dict(axes=attr[0]['i'], keepdims=attr[1]['i'])


def parse_instancenormalization(attr: list) -> dict(): # type: ignore
    attrs = dict()
    for item in attr:
        val = item['float']
        attrs[item['name']] = val
    return attrs


def parse_sum(attr: list) -> dict(): # type: ignore
    return dict()


def parse_reducemean(attr: list) -> dict:
    attrs = dict()
    if 1 < len(attr) < 2:
        attrs['axes'] = attr[0]['ints']
    elif len(attr) >= 2:
        attrs['axes'] = attr[0]['ints']
        attrs['keepdims'] = attr[1]['i']
    return attrs


def parse_reducemax(attr: list) -> dict:
    attrs = dict()
    if 1 < len(attr) < 2:
        attrs['axes'] = attr[0]['ints']
    elif len(attr) >= 2:
        attrs['axes'] = attr[0]['ints']
        attrs['keepdims'] = attr[1]['i']
    return attrs


def parse_reducemin(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_reducesum(attr: list) -> dict:
    attrs = dict()
    attrs[attr[0]['name']] = attr[0]['i']
    return attrs


def parse_hardsigmoid(attr: list)-> dict:
    attrs = {}
    for item in attr:
        attrs[item['name']] = item['float']
    if "alpha" not in attrs.keys():
        attrs["alpha"] = 0.2
    if "beta" not in attrs.keys():
        attrs["beta"] = 0.5    
    return attrs


def parse_hardswish(attr: list)-> dict:
    attrs = {}
    return attrs