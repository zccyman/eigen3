# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/12/08 13:49:00
# @File    : generate_attrs_case.py
from functools import reduce
import copy
import numpy as np
import os
import json
import random

try:
    from utils import RegistryFunction, Dict2Object, generate_random
    from config import Config
except Exception:
    from onnx_converter.utils import RegistryFunction, Dict2Object, generate_random # type: ignore
    from onnx_converter.config import Config # type: ignore

GAC_C = RegistryFunction()


def parse_config(cfg_file):
    if os.path.basename(cfg_file).endswith(".json"):
        with open(cfg_file, "r") as f:
            cfg_dict = json.load(f)
    elif os.path.basename(cfg_file).endswith(".py"):
        # save_cfg = Config.fromfile(cfg_file)
        cfg_dict, _ = Config._file2dict(cfg_file)
    else:
        raise Exception("{} not supported".format(cfg_file))

    args = Dict2Object(cfg_dict)

    return args


def CombineLists(attribute_lists):
    total_cases = reduce(lambda x, y: x * y, map(len, attribute_lists))
    combined_list = []

    for i in range(total_cases):
        step = total_cases
        temporary_item = []
        for attribute_list in attribute_lists:
            step /= len(attribute_list)
            index = int(i / step % len(attribute_list))
            temporary_item.append(attribute_list[index])
        combined_list.append(tuple(temporary_item))

    return combined_list


def CombineDict(attrs, layer_type='conv'):
    combined_attributes, combined_keys = [], []
    individual_attributes, individual_keys = [], []
    for k, (v0, v1) in attrs.items():
        if v1: #combination
            combined_keys.append(k)
            combined_attributes.append(v0)
        else:
            individual_keys.append(k)
            individual_attributes.append(v0)

    # check whether the lengths of each list in individual_attributes are equal, and trigger an assertion if they are not equal
    for v in individual_attributes:
        assert len(v) == len(individual_attributes[0])

    # combined_attributes participate in combinations
    combined_attribute_sets = {f"{layer_type}_{str(i)}": dict(zip(combined_keys, values))
    for i, values in enumerate(CombineLists(combined_attributes)) # type: ignore
    }   

    ### individual_attributes does not participate in combinations
    if len(individual_attributes) > 0:
        case_id = 0
        combined_attribute_cases = dict()
        for i in range(len(individual_attributes[0])):
            values = [individual_attributes[j][i] for j, _ in enumerate(individual_keys)]
            combined_attribute_case_temp = dict(zip(individual_keys, values))
            for _, v in combined_attribute_sets.items():
                combined_attribute_case_temp.update(v)
                combined_attribute_cases[f"{layer_type}_{str(case_id)}"] = copy.deepcopy(combined_attribute_case_temp)
                case_id += 1
        combined_attribute_sets = combined_attribute_cases
        
    return combined_attribute_sets


def remove_keys_from_dictionary(attrs, remove_keys=[]):
    return {k: v for k, v in attrs.items() if k not in remove_keys}


def generate_all_combination_on_layer_attrs(kwargs):
    layer_type = kwargs["layer_type"]
    layer_attrs_combination = copy.deepcopy(kwargs["layer_attrs_combination"])
    if layer_type == "conv" and "auto_pad" in layer_attrs_combination.keys():
        # when 'auto_pad' exist, 'pads' will be removed
        if "pads" in layer_attrs_combination.keys():
            layer_attrs_combination.pop("pads")
    if layer_type == "resize" and "sizes" in layer_attrs_combination.keys():
        # when 'sizes' exist, 'scales' will be removed
        if "scale" in layer_attrs_combination.keys():
            layer_attrs_combination.pop("scale")
        if layer_attrs_combination.get("mode") != "cubic":
            # 'cubic_coeff_a' is valid only if "mode" is "cubic".
            if "cubic_coeff_a" in layer_attrs_combination.keys():
                layer_attrs_combination.pop("cubic_coeff_a")
        if layer_attrs_combination.get("mode") != "nearest":
            # 'nearest_mode' is valid only if "mode" is "nearest".
            if "nearest_mode" in layer_attrs_combination.keys():
                layer_attrs_combination.pop("nearest_mode")

    res = CombineDict(layer_attrs_combination, layer_type=layer_type)
    return [{k: v} for k, v in res.items()]


@GAC_C.register_func(name="conv") # type: ignore
def get_attrs_conv(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    feat_i = attrs["feat_i"]
    has_bias = attrs["has_bias"]
    fuse_op = attrs["fuse_op"]
    bias_method = method if has_bias else "zeros"
    attrs.update(dict(bias=True, fuse_op=fuse_op))
    xs = [
        generate_random([
            attrs["batch_size"],
            attrs["in_c"],
            feat_i[0],
            feat_i[1],
        ],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {
        "weight": [
            generate_random([
                attrs["out_c"],
                attrs["in_c"],
                attrs["kernel_shape"][0],
                attrs["kernel_shape"][1],
            ],
                            method=method,
                            range=range,
                            seed=1, 
                            is_weight=True,).astype(np.float32)
        ],
        "bias": [
            generate_random(attrs["out_c"],
                            method=bias_method,
                            range=range,
                            seed=2,
                            is_weight=True,).astype(np.float32)
        ],
    }

    # Remove values not belonging to attributes
    remove_keys = [
        "feat_i", "batch_size", "has_bias", "method", "range"
    ]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="depthwiseconv") # type: ignore
def get_attrs_depthwiseconv(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    feat_i = attrs["feat_i"]
    has_bias = attrs["has_bias"]
    fuse_op = attrs["fuse_op"]
    bias_method = "randn" if has_bias else "zeros"
    attrs.update(dict(bias=True, fuse_op=fuse_op))
    xs = [
        generate_random([
            attrs["batch_size"],
            attrs["in_c"],
            feat_i[0],
            feat_i[1],
        ],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {
        "weight": [
            generate_random([
                attrs["out_c"],
                1,
                attrs["kernel_shape"][0],
                attrs["kernel_shape"][1],
            ],
                            method=method,
                            range=range,
                            seed=1,
                            is_weight=True,).astype(np.float32)
        ],
        "bias": [
            generate_random(attrs["out_c"],
                            method=bias_method,
                            range=range,
                            seed=2,
                            is_weight=True,).astype(np.float32)
        ],
    }

    # Remove values not belonging to attributes
    remove_keys = [
        "feat_i", "batch_size", "has_bias", "method", "range"
    ]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="convtranspose") # type: ignore
def get_attrs_convtranspose(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    feat_i = attrs["feat_i"]
    has_bias = attrs["has_bias"]
    fuse_op = attrs["fuse_op"]
    bias_method = "randn" if has_bias else "zeros"
    attrs.update(dict(bias=True, fuse_op=fuse_op))
    xs = [
        generate_random([
            attrs["batch_size"],
            attrs["in_c"],
            feat_i[0],
            feat_i[1],
        ],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {
        "weight": [
            generate_random([
                attrs["in_c"],
                attrs["out_c"],
                attrs["kernel_shape"][0],
                attrs["kernel_shape"][1],
            ],
                            method=method,
                            range=range,
                            seed=1,
                            is_weight=True,).astype(np.float32)
        ],
        "bias": [
            generate_random(attrs["out_c"],
                            method=bias_method,
                            range=range,
                            seed=2,
                            is_weight=True,).astype(np.float32)
        ],
    }

    # Remove values not belonging to attributes
    remove_keys = [
        "feat_i", "batch_size", "has_bias", "method", "range"
    ]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="fc") # type: ignore
def get_attrs_fc(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    has_bias = attrs["has_bias"]
    fuse_op = attrs["fuse_op"]
    bias_method = "randn" if has_bias else "zeros"
    attrs.update(dict(bias=True, fuse_op=fuse_op))
    xs = [
        generate_random([
            attrs["batch_size"],
            attrs["in_c"],
        ],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {
        "weight": [
            generate_random([
                attrs["out_c"],
                attrs["in_c"],
            ],
                            method=method,
                            range=range,
                            seed=1,
                            is_weight=True,).astype(np.float32)
        ],
        "bias": [
            generate_random(attrs["out_c"],
                            method=bias_method,
                            range=range,
                            seed=2,
                            is_weight=True,).astype(np.float32)
        ],
    }

    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "has_bias", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="matmul") # type: ignore
def get_attrs_matmul(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    feat_i0 = attrs["feat_i0"]
    feat_i1 = attrs["feat_i1"]
    xs = [
        generate_random([
            attrs["batch_size"],
            attrs["in_c"],
            feat_i0[0], 
            feat_i0[1],
        ],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32),
        generate_random([
            attrs["batch_size"],
            attrs["in_c"],
            feat_i1[0], 
            feat_i1[1],
        ],
                        method=method,
                        range=range,
                        seed=1).astype(np.float32)        
    ]
    ws = {}

    # Remove values not belonging to attributes
    # remove_keys = ["batch_size", "method", "range", "feat_i0", "feat_i1"]
    # attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)
    attrs = dict()

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }
    

@GAC_C.register_func(name="add") # type: ignore
@GAC_C.register_func(name="sub") # type: ignore
@GAC_C.register_func(name="pmul") # type: ignore
def get_attrs_elementwise(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    channel = attrs["channel"]
    batch_size = attrs["batch_size"]
    feat_i = attrs["feat_i"]
    xs = [
        random.randint(0, 9) * generate_random([batch_size, channel, feat_i[0], feat_i[1]],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32),
        random.randint(0, 9) * generate_random([batch_size, channel, feat_i[0], feat_i[1]],
                        method=method,
                        range=range,
                        seed=1).astype(np.float32),
    ]
    ws = {
        "weight": [],
        "bias": [],
    }

    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "feat_i", "channel", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="cadd") # type: ignore
@GAC_C.register_func(name="csub") # type: ignore
@GAC_C.register_func(name="cmul") # type: ignore
def get_attrs_channelwise(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    channel = attrs["channel"]
    batch_size = attrs["batch_size"]
    feat_i0 = attrs["feat_i0"]
    feat_i1 = attrs["feat_i1"]
    xs = [
        generate_random([batch_size, channel, feat_i0[0], feat_i0[1]],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32),
        generate_random([batch_size, channel, feat_i1[0], feat_i1[1]],
                        method=method,
                        range=range,
                        seed=1).astype(np.float32),
    ]
    ws = {
        "weight": [],
        "bias": [],
    }

    # Remove values not belonging to attributes
    remove_keys = [
        "batch_size", "feat_i0", "feat_i1", "channel", "method", "range"
    ]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="lstm") # type: ignore
def get_attrs_lstm(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    sequence_lens = attrs["sequence_lens"]
    hidden_size = attrs["hidden_size"]
    has_init_h, has_init_c, has_bias = attrs["has_init_h"], attrs[
        "has_init_c"], attrs["has_bias"]
    init_h_method = method if has_init_h else "zeros"
    init_c_method = method if has_init_h else "zeros"
    bias_method = method if has_bias else "zeros"
    initial_h = generate_random([sequence_lens, batch_size, hidden_size],
                                method=init_h_method,
                                range=range,
                                seed=0).astype(np.float32)
    initial_c = generate_random([sequence_lens, batch_size, hidden_size],
                                method=init_c_method,
                                range=range,
                                seed=1).astype(np.float32)
    attrs.update(dict(
        bias=True,
        initial_h=initial_h,
        initial_c=initial_c,
    ))
    xs = [
        generate_random([sequence_lens, batch_size, attrs["in_c"]],
                        method=method,
                        range=range,
                        seed=2).astype(np.float32),
        initial_h,
        initial_c,
    ]
    ws = {
        "weight": [
            generate_random([1, hidden_size * 4, attrs["in_c"]],
                            method=method,
                            range=range,
                            seed=3,
                            is_weight=True,).astype(np.float32),
            generate_random([1, hidden_size * 4, hidden_size],
                            method=method,
                            range=range,
                            seed=4,
                            is_weight=True,).astype(np.float32),
        ],
        "bias": [
            generate_random([1, hidden_size * 4],
                            method=bias_method,
                            range=range,
                            seed=5,
                            is_weight=True,).astype(np.float32),
            generate_random([1, hidden_size * 4],
                            method=bias_method,
                            range=range,
                            seed=6,
                            is_weight=True,).astype(np.float32),
        ],
    }

    # Remove values not belonging to attributes
    remove_keys = [
        "batch_size", "has_bias", "has_init_h", "has_init_c", "method", "range"
    ]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="gru") # type: ignore
def get_attrs_gru(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    sequence_lens = attrs["sequence_lens"]
    hidden_size = attrs["hidden_size"]
    has_init_h, has_bias = attrs["has_init_h"], attrs["has_bias"]
    init_h_method = method if has_init_h else "zeros"
    bias_method = method if has_bias else "zeros"
    initial_h = generate_random([sequence_lens, batch_size, hidden_size],
                                method=init_h_method,
                                range=range,
                                seed=0).astype(np.float32)
    attrs.update(dict(bias=True, ))
    xs = [
        generate_random([sequence_lens, batch_size, attrs["in_c"]],
                        method=method,
                        range=range,
                        seed=2).astype(np.float32),
        initial_h,
    ]
    ws = {
        "weight": [
            generate_random([1, hidden_size * 3, attrs["in_c"]],
                            method=method,
                            range=range,
                            seed=3,
                            is_weight=True,).astype(np.float32),
            generate_random([1, hidden_size * 3, hidden_size],
                            method=method,
                            range=range,
                            seed=4,
                            is_weight=True,).astype(np.float32),
        ],
        "bias": [
            generate_random([1, hidden_size * 3],
                            method=bias_method,
                            range=range,
                            seed=5,
                            is_weight=True,).astype(np.float32),
            generate_random([1, hidden_size * 3],
                            method=bias_method,
                            range=range,
                            seed=6,
                            is_weight=True,).astype(np.float32),
        ],
    }

    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "has_bias", "has_init_h", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="layernormalization") # type: ignore
def get_attrs_layernormalization(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    feat_i = attrs["feat_i"]
    w_shape = feat_i[attrs["axis"]:]
    xs = [
        generate_random(feat_i, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {
        "weight": [
            generate_random(w_shape, method=method, range=range,
                            seed=1, is_weight=True,).astype(np.float32)
        ],
        "bias": [
            generate_random(w_shape, method=method, range=range,
                            seed=2, is_weight=True,).astype(np.float32)
        ],
        "epsilon":
        attrs["epsilon"],
    }
    # Remove values not belonging to attributes
    remove_keys = ["feat_i", "epsilon", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="batchnormalization") # type: ignore
def get_attrs_batchnormalization(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    w_shape = x_shape[1]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {
        "weight": [
            generate_random(w_shape, method=method, range=range,
                            seed=1, is_weight=True,).astype(np.float32)
        ],
        "bias": [
            generate_random(w_shape, method=method, range=range,
                            seed=2, is_weight=True,).astype(np.float32)
        ],
        "running_mean": [
            generate_random(w_shape, method=method, range=range,
                            seed=3, is_weight=True,).astype(np.float32)
        ],
        "running_var": [
            generate_random(w_shape, method=method, range=range,
                            seed=4, is_weight=True,).astype(np.float32)
        ],
        "epsilon":
        attrs["epsilon"],
    }
    # Remove values not belonging to attributes
    remove_keys = [
        "batch_size", "channel", "feat_i", "epsilon", "method", "range"
    ]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="instancenormalization") # type: ignore
def get_attrs_instancenormalization(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    w_shape = x_shape[1]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {
        "weight": [
            generate_random(w_shape, method=method, range=range,
                            seed=1, is_weight=True,).astype(np.float32)
        ],
        "bias": [
            generate_random(w_shape, method=method, range=range,
                            seed=2, is_weight=True,).astype(np.float32)
        ],
        "epsilon":
        attrs["epsilon"],
    }
    # Remove values not belonging to attributes
    remove_keys = [
        "batch_size", "channel", "feat_i", "epsilon", "method", "range"
    ]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="softmax") # type: ignore
def get_attrs_softmax(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "feat_i", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="resize") # type: ignore
def get_attrs_resize(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "feat_i", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="reducemin") # type: ignore
@GAC_C.register_func(name="reducemax") # type: ignore
@GAC_C.register_func(name="reducemean") # type: ignore
@GAC_C.register_func(name="reducesum") # type: ignore
@GAC_C.register_func(name="reduceprod") # type: ignore
def get_attrs_reducemax(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }
    

@GAC_C.register_func(name="transpose") # type: ignore
def get_attrs_transpose(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }
    
     
@GAC_C.register_func(name="reshape") # type: ignore
def get_attrs_reshape(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }
    
    
@GAC_C.register_func(name="pad") # type: ignore
def get_attrs_pad(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }
    
                
@GAC_C.register_func(name="averagepool") # type: ignore
def get_attrs_averagepool(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["kernel_shape"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="maxpool") # type: ignore
def get_attrs_maxpool(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    x_shape = [batch_size, channel, feat_i[0], feat_i[1]]
    xs = [
        generate_random(x_shape, method=method, range=range,
                        seed=0).astype(np.float32)
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "feat_i", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="shuffle_only") # type: ignore
def get_attrs_shuffle_only(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    attrs.update(
        dict(
            shape1=[batch_size, 2, channel // 2, feat_i[0], feat_i[1]],
            shape2=[batch_size, -1, feat_i[0], feat_i[1]],
        ))
    xs = [
        generate_random([batch_size, channel, feat_i[0], feat_i[1]],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32),
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "feat_i", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="shuffle") # type: ignore
def get_attrs_shuffle(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    input_len = attrs["input_len"]
    attrs.update(
        dict(
            shape1=[batch_size, input_len, channel, feat_i[0], feat_i[1]],
            shape2=[batch_size, -1, feat_i[0], feat_i[1]],
            split=[channel, channel],
        ))
    xs = [
        generate_random([batch_size, channel, feat_i[0], feat_i[1]],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32),
        generate_random([batch_size, channel, feat_i[0], feat_i[1]],
                        method=method,
                        range=range,
                        seed=1).astype(np.float32),
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "feat_i", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="concat") # type: ignore
def get_attrs_concat(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channels = attrs["channels"]
    feat_i = attrs["feat_i"]
    xs = []
    for seed, channel in enumerate(channels):
        xs_ = (seed + 1) * generate_random([batch_size, channel, feat_i[0], feat_i[1]],
                                method=method,
                                range=range,
                                seed=seed).astype(np.float32)
        xs.append(xs_)
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channels", "feat_i", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="split") # type: ignore
def get_attrs_split(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    feat_i = attrs["feat_i"]
    xs = [
        generate_random(
            [batch_size,
             np.array(attrs["split"]).sum(), feat_i[0], feat_i[1]],
            method=method,
            range=range,
            seed=0).astype(np.float32),
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "feat_i", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }


@GAC_C.register_func(name="leakyrelu") # type: ignore
@GAC_C.register_func(name="prelu") # type: ignore
@GAC_C.register_func(name="relux") # type: ignore
@GAC_C.register_func(name="relu6") # type: ignore
@GAC_C.register_func(name="sigmoid") # type: ignore
@GAC_C.register_func(name="gelu") # type: ignore
@GAC_C.register_func(name="relu") # type: ignore
@GAC_C.register_func(name="tanh") # type: ignore
@GAC_C.register_func(name="hardsigmoid") # type: ignore
@GAC_C.register_func(name="hardtanh") # type: ignore
@GAC_C.register_func(name="hardswish") # type: ignore
@GAC_C.register_func(name="hardshrink") # type: ignore
def get_attrs_activation(layer_attrs, layer_type):
    attrs = copy.deepcopy(layer_attrs)
    method = attrs["method"]
    range = attrs["range"]
    batch_size = attrs["batch_size"]
    channel = attrs["channel"]
    feat_i = attrs["feat_i"]
    xs = [
        generate_random([batch_size, channel, feat_i[0], feat_i[1]],
                        method=method,
                        range=range,
                        seed=0).astype(np.float32),
    ]
    ws = {}
    # Remove values not belonging to attributes
    remove_keys = ["batch_size", "channel", "feat_i", "method", "range"]
    attrs = remove_keys_from_dictionary(attrs, remove_keys=remove_keys)

    return {
        "xs": xs,
        "ws": ws,
        "attrs": attrs,
    }
