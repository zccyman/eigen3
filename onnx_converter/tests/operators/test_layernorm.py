# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/4/18 9:58
# @File     : test_layernorm.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import os
from typing import Any, List, Optional, Sequence, Text, Union

import numpy as np
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx.onnx_pb import AttributeProto, FunctionProto, NodeProto, TypeProto

_TargetOpType = ""


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

# Layer normalization's reference implementation


def _layer_normalization(X, W, B, axis=-1, epsilon=1e-5):  # type: ignore
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        # If axis = -1 and rank of X is 4,
        # the axis is changed to -1 + 4 = 3,
        # which means the last axis.
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    # Parameter used to convert N-D tensor layer
    # normalization to equivalent 2-D matirx operations.
    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]

    # After reshaping input tensor X into a matrix,
    # layer normalization is equivalent to conducting
    # standardization on each column vector (s.t. each
    # column has zero mean and unit variance).
    x_mat = np.reshape(X, (row_number, col_number))
    # This computes mean for every x_mat's column.
    x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    # This computes variance for every x_mat's column.
    variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = np.sqrt(variance_eps)
    inv_std_dev = np.reciprocal(std_dev)
    # Standardization step. y_mat is zero-mean and unit-variance.
    y_mat = x_diff * inv_std_dev
    # Apply affine transform on normalization outcome.
    # W is linear coefficient while B is bias.
    Y = np.reshape(y_mat, X_shape) * W + B
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return Y, X_mean, X_inv_std_dev


def calculate_normalized_shape(X_shape, axis):  # type: ignore
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    return X_shape[axis:]


def create_initializer(data, name):
    return onnx.helper.make_tensor(
        name=name, data_type=onnx.TensorProto.FLOAT,
        dims=data.shape, vals=data.tobytes(), raw=True)


def test_layernorm():
    X = np.random.randn(1, 3, 224, 224).astype(np.float32)
    axis = 1
    normalized_shape = calculate_normalized_shape(X.shape, axis)
    W = np.random.randn(*normalized_shape).astype(np.float32)
    B = np.random.randn(*normalized_shape).astype(np.float32)
    Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)
    Y = Y.astype(np.float32)
    inputs = [X, W, B]
    outputs = [Y]

    x = torch.from_numpy(X)
    layer_norm = nn.LayerNorm(list(x.shape[axis:]))
    for name, param in layer_norm.named_parameters():
        if 'weight' in name:
            layer_norm.weight.data = torch.from_numpy(W)
        if 'bias' in name:
            layer_norm.bias.data = torch.from_numpy(B)
    y = layer_norm(x)
    y = y.detach().numpy()

    node = onnx.helper.make_node(
        'LayerNormalization',
        axis=axis,
        inputs=['X', 'W', 'B'],
        outputs=['Y']
    )

    conv_w = np.random.randn(3, 3, 3, 3).astype(np.float32)
    y = F.conv2d(torch.from_numpy(y), weight=torch.from_numpy(
        conv_w), bias=None, padding=(1, 1)).numpy()
    outputs = [y]
    node_with_padding = onnx.helper.make_node(
        'Conv',
        inputs=['Y', 'conv_w'],
        outputs=['output'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[1, 1, 1, 1],
    )

    if _TargetOpType and node.op_type != _TargetOpType:
        return

    present_inputs = [x for x in node.input if (x != '')]
    present_outputs = [x for x in node_with_padding.output if (x != '')]
    input_type_protos = [None] * len(inputs)
    output_type_protos = [None] * len(outputs)
    inputs_vi = [_extract_value_info(arr, arr_name, input_type)
                 for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)]
    outputs_vi = [_extract_value_info(arr, arr_name, output_type)
                  for arr, arr_name, output_type in zip(outputs, present_outputs, output_type_protos)]

    if not os.path.exists('work_dir'):
        os.makedirs('work_dir')

    initializers = [
        create_initializer(W, "W"),
        create_initializer(B, "B"),
        create_initializer(conv_w, "conv_w"),
    ]
    graph = onnx.helper.make_graph(
        nodes=[node, node_with_padding],
        name='test_layer_normalization_4d_axis',
        inputs=inputs_vi,
        outputs=outputs_vi,
        initializer=initializers)
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 16)])
    # onnx.checker.check_model(model)
    onnx.save_model(model, 'work_dir/layernorm.onnx')

    sess = rt.InferenceSession('work_dir/layernorm.onnx')
    x_name = sess.get_inputs()[0].name
    # w_name = sess.get_inputs()[1].name
    # b_name = sess.get_inputs()[2].name
    y_name = sess.get_outputs()[0].name

    pred_onx = sess.run([y_name], {
        x_name: X,
        # w_name: W, b_name: B
    })[0]
    error = np.sum(np.abs(y - pred_onx)) / np.sum(np.abs(y))
    print('=> error: ', error)


if __name__ == '__main__':
    test_layernorm()
