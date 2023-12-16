# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/4/18 9:58
# @File     : test_batchnorm.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import os
from typing import Any, List, Optional, Sequence, Text, Union

import numpy as np
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn
from onnx.external_data_helper import set_external_data
from onnx.numpy_helper import from_array, to_array
from onnx.onnx_pb import AttributeProto, FunctionProto, NodeProto, TypeProto

_TargetOpType = ""


def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias


def _extract_value_info(input: Union[List[Any], np.ndarray, None], name: Text, type_proto: Optional[TypeProto] = None) -> onnx.ValueInfoProto:
    if type_proto is None:
        if input is None:
            raise NotImplementedError(
                "_extract_value_info: both input and type_proto arguments cannot be None.")
        elif isinstance(input, list):
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input[0].dtype]
            shape = None
            tensor_type_proto = onnx.helper.make_tensor_type_proto(
                elem_type, shape)
            type_proto = onnx.helper.make_sequence_type_proto(
                tensor_type_proto)
        else:
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input.dtype]
            shape = input.shape
            type_proto = onnx.helper.make_tensor_type_proto(
                elem_type, shape)

    return onnx.helper.make_value_info(name, type_proto)


def create_initializer(data, name):
    return onnx.helper.make_tensor(
        name=name, data_type=onnx.TensorProto.FLOAT,
        dims=data.shape, vals=data.tobytes(), raw=True)


def test_batchnorm():
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    epsilon = 1e-2
    y = _batchnorm_test_mode(x, s, bias, mean, var, epsilon).astype(np.float32)

    inputs = [x, s, bias, mean, var]
    outputs = [y]

    node = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['x', 's', 'bias', 'mean', 'var'],
        outputs=['y'],
        epsilon=epsilon,
    )

    if _TargetOpType and node.op_type != _TargetOpType:
        return
    present_inputs = [x for x in node.input if (x != '')]
    present_outputs = [x for x in node.output if (x != '')]
    input_type_protos = [None] * len(inputs)
    output_type_protos = [None] * len(outputs)
    inputs_vi = [_extract_value_info(arr, arr_name, input_type)
                 for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)]
    outputs_vi = [_extract_value_info(arr, arr_name, output_type)
                  for arr, arr_name, output_type in zip(outputs, present_outputs, output_type_protos)]

    if not os.path.exists('work_dir'):
        os.makedirs('work_dir')

    initializers = [
        create_initializer(s, "s"),
        create_initializer(bias, "bias"),
        create_initializer(mean, "mean"),
        create_initializer(var, "var"),
    ]
    graph = onnx.helper.make_graph(
        nodes=[node],
        name='test_batchnorm',
        inputs=inputs_vi,
        outputs=outputs_vi,
        initializer=initializers)
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 12)])
    onnx.checker.check_model(model)
    onnx.save_model(model, 'work_dir/batchnorm.onnx')

    sess = rt.InferenceSession('work_dir/batchnorm.onnx')
    x_name = sess.get_inputs()[0].name
    # w_name = sess.get_inputs()[1].name
    # b_name = sess.get_inputs()[2].name
    # mean_name = sess.get_inputs()[3].name
    # var_name = sess.get_inputs()[4].name
    y_name = sess.get_outputs()[0].name

    pred_onnx = sess.run([y_name], {x_name: x,
                                    # w_name: s, b_name: bias,
                                    # mean_name: mean, var_name: var
                                    })[0]
    error = np.sum(np.abs(y - pred_onnx)) / np.sum(np.abs(y))
    print('=> error: ', error)


if __name__ == '__main__':
    test_batchnorm()
