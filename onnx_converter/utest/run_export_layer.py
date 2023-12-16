# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/07/05 15:00:46
# @File    : run_export_layer.py
import sys  # NOQA: E402

sys.path.append("./")  # NOQA: E402

try:
    from utest import EXPORT_LAYER
    from utils import RegistryFunction, generate_random
except Exception:
    from onnx_converter.utest import EXPORT_LAYER # type: ignore
    from onnx_converter.utils import RegistryFunction, generate_random # type: ignore

import os

import numpy as np
import pytest
import torch
# import random

RUN_EXPORT = RegistryFunction()


@RUN_EXPORT.register_func(name="batchnormalization") # type: ignore
def run_export_batchnormalization(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys() and "ws" not in kwargs.keys():
        x_shape = [1, 3, 128, 128]
        w_shape = x_shape[1]
        xs = [generate_random(x_shape).astype(np.float32)]

        ws = {
            "weight": [generate_random(w_shape).astype(np.float32)],
            "bias": [generate_random(w_shape).astype(np.float32)],
            "running_mean": [generate_random(w_shape).astype(np.float32)],
            "running_var": [generate_random(w_shape).astype(np.float32)],
            "epsilon": 1.0e-5,
        }
    else:
        xs = kwargs["xs"]
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="layernormalization") # type: ignore
def run_export_layernormalization(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {"axis": 1}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys() and "ws" not in kwargs.keys():
        x_shape = [1, 1, 128]
        w_shape = x_shape[attrs["axis"] :]
        xs = [generate_random(x_shape).astype(np.float32)]
        ws = {
            "weight": [generate_random(w_shape).astype(np.float32)],
            "bias": [generate_random(w_shape).astype(np.float32)],
            "epsilon": 1.0e-5,
        }
    else:
        xs = kwargs["xs"]
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="instancenormalization") # type: ignore
def run_export_instancenormalization(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = dict()
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys() and "ws" not in kwargs.keys():
        x_shape = [1, 3, 112, 112]
        w_shape = x_shape[1]
        xs = [generate_random(x_shape).astype(np.float32)]
        ws = {
            "weight": [generate_random(w_shape).astype(np.float32)],
            "bias": [generate_random(w_shape).astype(np.float32)],
            "epsilon": 1.0e-5,
        }
    else:
        xs = kwargs["xs"]
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="softmax") # type: ignore
def run_export_softmax(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {"axis": 1}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        x_shape = [1, 3, 128, 128]
        xs = [generate_random(x_shape).astype(np.float32)]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="conv") # type: ignore
def run_export_conv(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "in_c": 3,
            "out_c": 32,
            "kernel_shape": [3, 3],
            "strides": [2, 2],
            "pads": [0, 0, 1, 1],
            "dilations": [1, 1],
            "group": 1,
            "fuse_op": ["relu"],
            "isolated": False,
            "bias": True,
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [generate_random([1, attrs["in_c"], 256, 320]).astype(np.float32)]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {
            "weight": [
                generate_random(
                    [
                        attrs["out_c"],
                        attrs["in_c"],
                        attrs["kernel_shape"][0],
                        attrs["kernel_shape"][1],
                    ]
                ).astype(np.float32)
            ],
            "bias": [generate_random(attrs["out_c"]).astype(np.float32)],
        }
    else:
        ws = kwargs["ws"]

    if "auto_pad" in attrs.keys() and "pads" in attrs.keys():
        attrs.pop("pads")

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="depthwiseconv") # type: ignore
def run_export_depthwiseconv(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "in_c": 3,
            "out_c": 32,
            "kernel_shape": [3, 3],
            "strides": [2, 2],
            "pads": [0, 0, 1, 1],
            "dilations": [1, 1],
            "group": 3,
            "fuse_op": ["relu"],
            "isolated": False,
            "bias": True,
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [generate_random([1, attrs["in_c"], 256, 320]).astype(np.float32)]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {
            "weight": [
                generate_random(
                    [
                        attrs["out_c"],
                        1,
                        attrs["kernel_shape"][0],
                        attrs["kernel_shape"][1],
                    ]
                ).astype(np.float32)
            ],
            "bias": [generate_random(attrs["out_c"]).astype(np.float32)],
        }
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="convtranspose") # type: ignore
def run_export_convtranspose(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "in_c": 3,
            "out_c": 32,
            "kernel_shape": [5, 5],
            "strides": [2, 2],
            "pads": [2, 2, 2, 2],
            "output_padding": [1, 1],
            "dilations": [1, 1],
            "group": 1,
            "fuse_op": ["relu"],
            "isolated": False,
            "bias": True,
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [generate_random([1, attrs["in_c"], 20, 20]).astype(np.float32)]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {
            "weight": [
                generate_random(
                    [
                        attrs["in_c"],
                        attrs["out_c"],
                        attrs["kernel_shape"][0],
                        attrs["kernel_shape"][1],
                    ]
                ).astype(np.float32)
            ],
            "bias": [generate_random(attrs["out_c"]).astype(np.float32)],
        }
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="fc") # type: ignore
def run_export_fc(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {"in_c": 512, "out_c": 512, "fuse_op": ["relu"], "isolated": False, "bias": True}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [generate_random([1, attrs["in_c"]]).astype(np.float32)]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {
            "weight": [
                generate_random([attrs["out_c"], attrs["in_c"]]).astype(np.float32)
            ],
            "bias": [generate_random(attrs["out_c"]).astype(np.float32)],
        }
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="matmul") # type: ignore
def run_export_matmul(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random([1, 3, 32, 16]).astype(np.float32),
            generate_random([1, 3, 16, 32]).astype(np.float32),
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="reducemax") # type: ignore
@RUN_EXPORT.register_func(name="reducemin") # type: ignore
@RUN_EXPORT.register_func(name="reducemean") # type: ignore
@RUN_EXPORT.register_func(name="reducesum") # type: ignore
@RUN_EXPORT.register_func(name="reduceprod") # type: ignore
def run_export_reduceops(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "axes": [1],
            "keepdims": 1,
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random(
                [1, 128, 112, 112]
            ).astype(np.float32)
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="transpose") # type: ignore
def run_export_transpose(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "perm": [0, 1, 3, 2],
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random(
                [1, 64, 24, 12]
            ).astype(np.float32)
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="log") # type: ignore
def run_export_log(kwargs):
    attrs = {}
    if "xs" not in kwargs.keys():
        xs = [
            generate_random(
                [1, 64, 32, 32]
            ).astype(np.float32)
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="reshape") # type: ignore
def run_export_reshape(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "shape": [0, 8, 16, -1],
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random(
                [1, 128, 2, 2]
            ).astype(np.float32)
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="pad") # type: ignore
def run_export_pad(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "mode": "constant",
            "pads": [0, 0, 0, 0, 0, 24, 0, 0],
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random(
                [1, 24, 64, 64]
            ).astype(np.float32)
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="globalaveragepool") # type: ignore
@RUN_EXPORT.register_func(name="averagepool") # type: ignore
@RUN_EXPORT.register_func(name="maxpool") # type: ignore
def run_export_pool(kwargs):
    if "attrs" not in kwargs.keys():
        if kwargs["arguments"]["layer_type"] in ["maxpool"]:
            attrs = {
                "ceil_mode": False,
                "kernel_shape": [3, 3],
                "pads": [1, 1, 1, 1],
                "strides": [2, 2],
            }
        else:
            attrs = {
                "ceil_mode": True,
                "kernel_shape": [7, 7],
                "pads": [0, 0, 0, 0],
                "strides": [1, 1],
            }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        if kwargs["arguments"]["layer_type"] in ["maxpool"]:
            xs = [generate_random([1, 64, 112, 112]).astype(np.float32)]
        else:
            xs = [
                generate_random(
                    [1, 512, attrs["kernel_shape"][0], attrs["kernel_shape"][1]]
                ).astype(np.float32)
            ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="relu") # type: ignore
@RUN_EXPORT.register_func(name="relu6") # type: ignore
@RUN_EXPORT.register_func(name="relux") # type: ignore
@RUN_EXPORT.register_func(name="leakyrelu") # type: ignore
@RUN_EXPORT.register_func(name="prelu") # type: ignore
@RUN_EXPORT.register_func(name="sigmoid") # type: ignore
@RUN_EXPORT.register_func(name="swish") # type: ignore
@RUN_EXPORT.register_func(name="gelu") # type: ignore
@RUN_EXPORT.register_func(name="tanh") # type: ignore
@RUN_EXPORT.register_func(name="hardsigmoid") # type: ignore
@RUN_EXPORT.register_func(name="hardtanh") # type: ignore
@RUN_EXPORT.register_func(name="hardswish") # type: ignore
@RUN_EXPORT.register_func(name="hardshrink") # type: ignore
def run_export_act(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {"isolated": True}
        layer_type = kwargs["arguments"]["layer_type"]
        if layer_type == "leakyrelu":
            attrs.update(dict(alpha=0.001)) # type: ignore
        if layer_type == "hardsigmoid":
            attrs.update(dict(alpha=0.2, beta=0.5)) # type: ignore            
        elif layer_type == "prelu":
            attrs.update(dict(slope=np.array([0.001]))) # type: ignore           
        elif layer_type == "relu6":
            attrs.update(dict(value=6.0)) # type: ignore
        elif layer_type == "relux":
            attrs.update(dict(value=12.0)) # type: ignore
    else:
        attrs = kwargs["attrs"]
    if "xs" not in kwargs.keys():
        xs = [generate_random([1, 64, 112, 112]).astype(np.float32)]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="resize") # type: ignore
def run_export_resize(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "scale": [1, 1, 2, 2],
            "mode": "linear",
            "coordinate_transformation_mode": "align_corners",
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [generate_random([1, 64, 112, 112]).astype(np.float32)]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]
    
    sizes = kwargs['attrs'].get('sizes')
    if sizes:
        batch_size = kwargs['case_attr']['batch_size']
        channel = kwargs['case_attr']['channel']
        sizes = [batch_size, channel, sizes[0], sizes[1]]
        kwargs['attrs']['sizes'] = sizes

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="lstm") # type: ignore
def run_export_lstm(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "hidden_size": 128,
            "sequence_lens": 1,
            "in_c": 257,
            "wr_combine": True,
            "hx_combine": False,
        }
    else:
        attrs = kwargs["attrs"]

    # init_h = np.load('init_h.npy') #generate_random(1, 1, attrs['hidden_size']).astype(np.float32)
    # init_c = np.load('init_c.npy') #generate_random(1, 1, attrs['hidden_size']).astype(np.float32)
    # np.save('init_h.npy', init_h)
    # np.save('init_c.npy', init_c)
    if "xs" not in kwargs.keys():
        initial_h = generate_random([1, 1, attrs["hidden_size"]]).astype(np.float32)
        initial_c = generate_random([1, 1, attrs["hidden_size"]]).astype(np.float32)
        attrs.update(dict(initial_h=initial_h, initial_c=initial_c))
        xs = [
            generate_random([1, 1, attrs["in_c"]]).astype(np.float32),
            # np.load('mag.npy'),
            initial_h,
            initial_c,
            # np.zeros([1, 1, attrs['hidden_size']]).astype(np.float32),
            # np.zeros([1, 1, attrs['hidden_size']]).astype(np.float32),
            # generate_random(1, 1, attrs['hidden_size']).astype(np.float32),
            # generate_random(1, 1, attrs['hidden_size']).astype(np.float32),
        ]
    else:
        xs = kwargs["xs"]

    # bias = np.load('98.npy')
    if "ws" not in kwargs.keys():
        ws = {
            "weight": [
                generate_random([1, attrs["hidden_size"] * 4, attrs["in_c"]]).astype(
                    np.float32
                ),
                generate_random(
                    [1, attrs["hidden_size"] * 4, attrs["hidden_size"]]
                ).astype(np.float32),
                # np.load('96.npy'),
                # np.load('97.npy')
            ],
            "bias": [
                generate_random([1, attrs["hidden_size"] * 4]).astype(np.float32),
                generate_random([1, attrs["hidden_size"] * 4]).astype(np.float32),
                # bias[:, :bias.shape[1] // 2],
                # bias[:, bias.shape[1] // 2:]
            ],
        }
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="gru") # type: ignore
def run_export_gru(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "hidden_size": 128,
            "sequence_lens": 1,
            "linear_before_reset": 0,
            "in_c": 257,
            "wr_combine": True,
            "hx_combine": False,
        }
    else:
        attrs = kwargs["attrs"]

    # init_h = np.load('init_h.npy') #generate_random(1, 1, attrs['hidden_size']).astype(np.float32)
    # np.save('init_h.npy', init_h)
    if "xs" not in kwargs.keys():
        initial_h = generate_random([1, 1, attrs["hidden_size"]]).astype(np.float32)
        attrs.update(dict(initial_h=initial_h))
        xs = [
            generate_random([1, 1, attrs["in_c"]]).astype(np.float32),
            # np.load('mag.npy'),
            initial_h,
            # np.zeros([1, 1, attrs['hidden_size']]).astype(np.float32),
            # np.zeros([1, 1, attrs['hidden_size']]).astype(np.float32),
            # generate_random(1, 1, attrs['hidden_size']).astype(np.float32),
            # generate_random(1, 1, attrs['hidden_size']).astype(np.float32),
        ]
    else:
        xs = kwargs["xs"]

    # bias = np.load('98.npy')
    if "ws" not in kwargs.keys():
        ws = {
            "weight": [
                generate_random([1, attrs["hidden_size"] * 3, attrs["in_c"]]).astype(
                    np.float32
                ),
                generate_random(
                    [1, attrs["hidden_size"] * 3, attrs["hidden_size"]]
                ).astype(np.float32),
                # np.load('96.npy'),
                # np.load('97.npy')
            ],
            "bias": [
                generate_random([1, attrs["hidden_size"] * 3]).astype(np.float32),
                generate_random([1, attrs["hidden_size"] * 3]).astype(np.float32),
                # bias[:, :bias.shape[1] // 2],
                # bias[:, bias.shape[1] // 2:]
            ],
        }
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content

@RUN_EXPORT.register_func(name="splice") # type: ignore
def run_export_splice(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "in_c":
            129,
            "out_c":
            129,
            "fuse_op": ["fc"],
            "isolated":
            False,
            "context":
            np.array([-1, 0, 1]).astype(np.int32),
            "forward_indexes":
            np.array([
                0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8,
                7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11, 12, 11, 12, 13, 12, 13,
                14, 13, 14, 15, 14, 15, 16, 15, 16, 17, 16, 17, 18, 17, 18, 19,
                18, 19, 20, 19, 20, 21, 20, 21, 22, 21, 22, 23, 22, 23, 24, 23,
                24, 25, 24, 25, 26, 25, 26, 27, 26, 27, 28, 27, 28, 29, 28, 29,
                30, 29, 30, 31, 30, 31, 32, 31, 32, 33, 32, 33, 34, 33, 34, 35,
                34, 35, 36, 35, 36, 37, 36, 37, 38, 37, 38, 39, 38, 39, 40, 39,
                40, 41, 40, 41, 42, 41, 42, 43, 42, 43, 44, 43, 44, 45, 44, 45,
                46, 45, 46, 47, 46, 47, 48, 47, 48, 49, 48, 49, 50, 49, 50, 51,
                50, 51, 52, 51, 52, 53, 52, 53, 54, 53, 54, 55, 54, 55, 56, 55,
                56, 57, 56, 57, 58, 57, 58, 59, 58, 59, 60, 59, 60, 61, 60, 61,
                62, 61, 62, 63, 62, 63, 64, 63, 64, 65, 64, 65, 66, 65, 66, 67,
                66, 67, 68, 67, 68, 69, 68, 69, 70, 69, 70, 71, 70, 71, 72, 71,
                72, 73
            ]).astype(np.int32),
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [generate_random([1, attrs["in_c"]]).astype(np.float32)]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {
            "weight": [
                generate_random([attrs["out_c"], attrs["in_c"]]).astype(np.float32)
            ],
            "bias": [generate_random(attrs["out_c"]).astype(np.float32)],
        }
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content

@RUN_EXPORT.register_func(name="add") # type: ignore
@RUN_EXPORT.register_func(name="sub") # type: ignore
@RUN_EXPORT.register_func(name="pmul") # type: ignore
def run_export_elementwise(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            1.0 * generate_random([1, 64, 112, 112], method="randn").astype(np.float32),
            0.5 * generate_random([1, 64, 112, 112], method="rand").astype(np.float32),
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    attrs.update(dict(layer_type=layer_type))
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="cadd") # type: ignore
@RUN_EXPORT.register_func(name="csub") # type: ignore
@RUN_EXPORT.register_func(name="cmul") # type: ignore
def run_export_channelwise(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random([1, 64, 112, 112]).astype(np.float32),
            generate_random([1, 64, 1, 1]).astype(np.float32),
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    attrs.update(dict(layer_type=layer_type))
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="concat") # type: ignore
def run_export_concat(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {"axis": 1, "input_len": 2}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random([1, 12, 112, 112]).astype(np.float32),
            generate_random([1, 14, 112, 112]).astype(np.float32),
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="split") # type: ignore
def run_export_split(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {"axis": 1, "split": [12, 24]}
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random([1, np.array(attrs["split"]).sum(), 112, 112]).astype(
                np.float32
            ),
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="shuffle_only") # type: ignore
def run_export_shuffle_only(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "shape1": [1, 2, 58, 40, 40],
            "perm": [0, 2, 1, 3, 4],
            "shape2": [1, -1, 40, 40],
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random([1, 116, 40, 40]).astype(np.float32),
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


@RUN_EXPORT.register_func(name="concat_shuffle_split") # type: ignore
@RUN_EXPORT.register_func(name="shuffle") # type: ignore
def run_export_shuffle(kwargs):
    if "attrs" not in kwargs.keys():
        attrs = {
            "axis": 1,
            "input_len": 2,
            "shape1": [1, 2, 58, 40, 40],
            "perm": [0, 2, 1, 3, 4],
            "shape2": [1, -1, 40, 40],
            "split": [58, 58],
        }
    else:
        attrs = kwargs["attrs"]

    if "xs" not in kwargs.keys():
        xs = [
            generate_random([1, 58, 40, 40]).astype(np.float32),
            generate_random([1, 58, 40, 40]).astype(np.float32),
        ]
    else:
        xs = kwargs["xs"]

    if "ws" not in kwargs.keys():
        ws = {"weight": [], "bias": []}
    else:
        ws = kwargs["ws"]

    layer_type = kwargs["arguments"]["layer_type"]
    kwargs.update(dict(layer_type=layer_type))
    engine = EXPORT_LAYER.get(layer_type)(**kwargs) # type: ignore
    content = engine(attrs, xs, ws)

    return content


def test_layer():
    layer_type = "layernormalization"
    feature = "sym"
    weight = ["sym", "pertensor"]
    quantize_dtype = "int8"
    in_type = "int8"
    out_type = "int8"
    process_scale = "float"
    mode = "ab"
    chip_type = "AT1K"
    export_version = 2
    weights_dir = "work_dir/test_layer"
    is_stdout = True

    export_version = "" if export_version > 1 else "_v{}".format(export_version)

    quantization_args = dict(
        type=quantize_dtype,
        in_type=in_type,
        out_type=out_type,
        process_scale=process_scale,
        method=dict(feature=feature, weight=weight),
    )
    arguments = dict(
        layer_type=layer_type,
        export_args=dict(chip_type=chip_type, mode=mode, is_acc_woffset=False, export_version=2),
        quantization_args=quantization_args,
        weights_dir=weights_dir,
    )

    # try:
    #     from onnx_converter.utils import props_with_
    #     from onnx_converter.config import quantize, voice_quantize, vision_quantize
    #     from onnx_converter.config import export, export_v1
    #     export_cfg = props_with_(export)
    #     if args.export_version == 1:
    #         export_cfg_ = props_with_(export_v1)
    #         export_cfg.update(export_cfg_)
    #     quant_cfg = props_with_(quantize)
    #     voice_quant_cfg = props_with_(voice_quantize)
    #     vision_quant_cfg = props_with_(vision_quantize)
    #     kwargs = {
    #         "weights_dir": arguments["weights_dir"],
    #         "case_name": layer_type,
    #         'log_dir': os.path.join(arguments["weights_dir"], layer_type),
    #         'log_name': 'test_layer.log',
    #         'is_stdout': is_stdout,
    #         "quant_cfg": quant_cfg,
    #         "voice_quant_cfg": voice_quant_cfg,
    #         "vision_quant_cfg": vision_quant_cfg,
    #         "export_cfg": export_cfg,
    #         "arguments": arguments,
    #     }
    # except Exception:
    if 1:
        kwargs = {
            "weights_dir": arguments["weights_dir"],
            "case_name": layer_type,
            "log_dir": os.path.join(arguments["weights_dir"], layer_type), # type: ignore
            "log_name": "test_{}.log".format(layer_type),
            "is_stdout": is_stdout,
            "quant_cfg": "config/quantize.py",
            "voice_quant_cfg": "config/voice_quantize.py",
            "vision_quant_cfg": "config/vision_quantize.py",
            "export_cfg": "config/export{}.py".format(export_version),
            "arguments": arguments,
        }

    content = RUN_EXPORT.get(layer_type)(kwargs) # type: ignore
    # print('test')


if __name__ == "__main__":
    pytest.main(["utest/run_export_layer.py::test_layer"])
