# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : TIMESINETLLI TECH
# @Time     : 2022/6/15 15:45
# @File     : test_speedup.py

import sys  # NOQA: E402

import os
import time
import copy
from typing import Any, List, Optional, Sequence, Text, Union

import numpy as np
from onnxmltools.utils import float16_converter
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx.onnx_pb import AttributeProto, FunctionProto, NodeProto, TypeProto

root_dir = '/home/shiqing/Downloads/onnx_converter'

_TargetOpType = ""

device = [{'device_id': 3}]
cycle_num = 100


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
        dist = torch.sum(s_data * t_data) / (normal + self.eps)
        dist = (1 - np.abs(dist.item())) * 100

        return np.float32(dist)


def _extract_value_info(input: Union[List[Any], np.ndarray, None], name: Text,
                        type_proto: Optional[TypeProto] = None) -> onnx.ValueInfoProto:
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


def create_initializer(data, name, dtype=np.float32):
    if not isinstance(data, np.ndarray):
        data = np.array(data).astype(dtype)
    if dtype == np.float32:
        tense_dtype = onnx.TensorProto.FLOAT
    else:
        tense_dtype = onnx.TensorProto.INT64
    tensor = onnx.helper.make_tensor(
        name=name, data_type=tense_dtype,
        dims=data.shape, vals=data.tobytes(), raw=True)
    return tensor


def split_ops(X, axis, layer_norm, conv_w, mul_1):
    x = torch.from_numpy(X)

    y = layer_norm(x)
    y = y.detach().numpy()

    conv_y = F.conv2d(torch.from_numpy(y), weight=torch.from_numpy(
        conv_w), bias=None, padding=(1, 1)).numpy()

    y = conv_y * mul_1

    outputs = [y]

    return outputs


def onnx_ops(inputs, outputs, axis, conv_w, W, B, mul_1, root_dir):
    node = onnx.helper.make_node(
        'LayerNormalization',
        axis=axis,
        inputs=['X', 'W', 'B'],
        outputs=['Y']
    )

    node_with_padding = onnx.helper.make_node(
        'Conv',
        inputs=['Y', 'conv_w'],
        outputs=['conv_y'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[1, 1, 1, 1],
    )
    mul = onnx.helper.make_node(
        'Mul',
        inputs=['conv_y', 'mul_1'],
        outputs=['output']
    )
    # outputs = [output]
    if _TargetOpType and node.op_type != _TargetOpType:
        return

    present_inputs = [x for x in node.input if (x != '')]
    present_outputs = [x for x in mul.output if (x != '')]
    input_type_protos = [None] * len(inputs)
    output_type_protos = [None] * len(outputs)
    inputs_vi = [_extract_value_info(arr, arr_name, input_type)
                 for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)]
    outputs_vi = [_extract_value_info(arr, arr_name, output_type)
                  for arr, arr_name, output_type in zip(outputs, present_outputs, output_type_protos)]
    if not os.path.exists('{}/work_dir'.format(root_dir)):
        os.makedirs('{}/work_dir'.format(root_dir))

    initializers = [
        create_initializer(W, "W"),
        create_initializer(B, "B"),
        create_initializer(conv_w, "conv_w"),
        create_initializer(mul_1, 'mul_1')
    ]
    graph = onnx.helper.make_graph(
        nodes=[node, node_with_padding, mul],
        name='test_layer_normalization_4d_axis',
        inputs=inputs_vi,
        outputs=outputs_vi,
        initializer=initializers)
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 16)])
    # onnx.checker.check_model(model)
    onnx.save_model(model, '{}/work_dir/layernorm.onnx'.format(root_dir))


def test_layernorm():
    X = np.random.randn(10, 3, 224, 224).astype(np.float32)
    axis = 1
    normalized_shape = calculate_normalized_shape(X.shape, axis)
    W = np.random.randn(*normalized_shape).astype(np.float32)
    B = np.random.randn(*normalized_shape).astype(np.float32)
    Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)
    inputs = [X]
    conv_w = np.random.randn(64, 3, 3, 3).astype(np.float32)
    mul_1 = np.random.randn(1, 64, 1, 1).astype(np.float32)
    # root_dir = '/home/shiqing/Downloads/onnx_converter'

    layer_norm = nn.LayerNorm(list(X.shape[axis:]))
    for name, param in layer_norm.named_parameters():
        if 'weight' in name:
            layer_norm.weight.data = torch.from_numpy(W)
        if 'bias' in name:
            layer_norm.bias.data = torch.from_numpy(B)
    outputs = split_ops(X, axis, layer_norm, conv_w, mul_1)
    onnx_ops(inputs, outputs, axis, conv_w, W, B, mul_1, root_dir)
    model = onnx.load('{}/work_dir/layernorm.onnx'.format(root_dir))
    trans_model = float16_converter.convert_float_to_float16(model, keep_io_types=True)
    onnx.save_model(trans_model, '{}/work_dir/test_net_fp16.onnx'.format(root_dir))
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
        # providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    sess = rt.InferenceSession('{}/layernorm.onnx'.format(root_dir), providers=providers)
    import time
    start_t = time.process_time()
    for _ in range(100):
        x_name = sess.get_inputs()[0].name
        # w_name = sess.get_inputs()[1].name
        # b_name = sess.get_inputs()[2].name
        y_name = sess.get_outputs()[0].name
        pred_onx = sess.run([y_name], {
            x_name: X,
            # w_name: W, b_name: B
        })[0]
    print('onnx time consume is: {}'.format((time.process_time() - start_t) / 100))
    start_t = time.process_time()
    for _ in range(100):
        split_ops(X, axis, layer_norm, conv_w, mul_1)
    print('split consume is: {}'.format((time.process_time() - start_t) / 100))
    # error = np.sum(np.abs(y - pred_onx)) / np.sum(np.abs(y))
    # print('=> error: ', error)


def build_qconv(x, output, w, b, out_shift, out_scale, int_scale, zi, zo, max_value, min_value, root_dir):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_1']
    )
    conv = onnx.helper.make_node(
        'Conv',
        inputs=['sub_1', 'w', 'b'],
        outputs=['conv_y'],
        # kernel_shape=[3, 3],
        # strides=[1, 1],
        # pads=[1, 1, 1, 1]
    )
    mul_1 = onnx.helper.make_node(
        'Mul',
        inputs=['conv_y', 'out_shift'],
        outputs=['mul_1']
    )
    floor_1 = onnx.helper.make_node(
        'Floor',
        inputs=['mul_1'],
        outputs=['floor_1']
    )
    clip_1 = onnx.helper.make_node(
        'Clip',
        inputs=['floor_1', 'min', 'max'],
        outputs=['clip_1']
    )
    mul_2 = onnx.helper.make_node(
        'Mul',
        inputs=['clip_1', 'out_scale'],
        outputs=['mul_2']
    )
    mul_3 = onnx.helper.make_node(
        'Mul',
        inputs=['mul_2', 'int_scale'],
        outputs=['mul_3']
    )
    floor_2 = onnx.helper.make_node(
        'Floor',
        inputs=['mul_3'],
        outputs=['floor_2']
    )
    add_1 = onnx.helper.make_node(
        'Add',
        inputs=['floor_2', 'zo'],
        outputs=['add_1']
    )
    clip_2 = onnx.helper.make_node(
        'Clip',
        inputs=['add_1', 'min', 'max'],
        outputs=['output']
    )
    nodes = [sub, conv, mul_1, floor_1, clip_1, mul_2, mul_3, floor_2, add_1, clip_2]
    initializers = [
        create_initializer(w, "w"),
        create_initializer(b, "b"),
        create_initializer(out_shift, "out_shift"),
        create_initializer(out_scale, "out_scale"),
        create_initializer(int_scale, "int_scale"),
        create_initializer(zi, "zi"),
        create_initializer(zo, "zo"),
        create_initializer(min_value, "min"),
        create_initializer(max_value, "max"),
    ]
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/qconv.onnx'.format(root_dir))


def build_qfc(x, output, w, b, out_shift, out_scale, int_scale, zi, zo, max_value, min_value, root_dir):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_1']
    )
    op_type = 'Gemm' if isinstance(b, np.ndarray) else 'MatMul'
    fc_inputs = ['sub_1', 'w', 'b'] if isinstance(b, np.ndarray) else ['sub_1', 'w']
    conv = onnx.helper.make_node(
        op_type,
        inputs=fc_inputs,
        outputs=['fc_y'],
        # kernel_shape=[3, 3],
        # strides=[1, 1],
        # pads=[1, 1, 1, 1]
    )
    mul_1 = onnx.helper.make_node(
        'Mul',
        inputs=['fc_y', 'out_shift'],
        outputs=['mul_1']
    )
    floor_1 = onnx.helper.make_node(
        'Floor',
        inputs=['mul_1'],
        outputs=['floor_1']
    )
    clip_1 = onnx.helper.make_node(
        'Clip',
        inputs=['floor_1', 'min', 'max'],
        outputs=['clip_1']
    )
    mul_2 = onnx.helper.make_node(
        'Mul',
        inputs=['clip_1', 'out_scale'],
        outputs=['mul_2']
    )
    mul_3 = onnx.helper.make_node(
        'Mul',
        inputs=['mul_2', 'int_scale'],
        outputs=['mul_3']
    )
    floor_2 = onnx.helper.make_node(
        'Floor',
        inputs=['mul_3'],
        outputs=['floor_2']
    )
    add_1 = onnx.helper.make_node(
        'Add',
        inputs=['floor_2', 'zo'],
        outputs=['add_1']
    )
    clip_2 = onnx.helper.make_node(
        'Clip',
        inputs=['add_1', 'min', 'max'],
        outputs=['output']
    )
    nodes = [sub, conv, mul_1, floor_1, clip_1, mul_2, mul_3, floor_2, add_1, clip_2]
    fc_w = np.transpose(w, (1, 0))

    initializers = [
        create_initializer(fc_w, "w"),
        create_initializer(out_shift, "out_shift"),
        create_initializer(out_scale, "out_scale"),
        create_initializer(int_scale, "int_scale"),
        create_initializer(zi, "zi"),
        create_initializer(zo, "zo"),
        create_initializer(min_value, "min"),
        create_initializer(max_value, "max"),
    ]
    if isinstance(b, np.ndarray):
        initializers.append(create_initializer(b, "b"))

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/qfc.onnx'.format(root_dir))


def build_concat(x, output, w, b, out_shift, out_scale, int_scale, zi, zo, max_value, min_value, axis, root_dir):
    nodes, clip_outs = [], []
    input_data = []
    initializers = [
        create_initializer(min_value, "min"),
        create_initializer(max_value, "max"),
    ]
    for idx in range(len(x)):
        data, zero_point = 'x' + str(idx), 'zi' + str(idx)
        sub_out, mul_scale = 'sub_' + str(idx), 'out_scale' + str(idx)
        mul_int_scale = 'int_scale' + str(idx)
        mul_1_out, clip_out = 'mul_1_' + str(idx), 'clip_' + str(idx)
        mul_2_out, ceil_out = 'mul_2_' + str(idx), 'ceil_' + str(idx)

        sub = onnx.helper.make_node(
            'Sub',
            inputs=[data, zero_point],
            outputs=[sub_out]
        )
        mul_1 = onnx.helper.make_node(
            'Mul',
            inputs=[sub_out, mul_scale],
            outputs=[mul_1_out]
        )
        mul_2 = onnx.helper.make_node(
            'Mul',
            inputs=[mul_1_out, mul_int_scale],
            outputs=[mul_2_out]
        )
        floor = onnx.helper.make_node(
            'Floor',
            inputs=[mul_2_out],
            outputs=[ceil_out]
        )

        initializers.extend([create_initializer(zi[idx], zero_point),
                             create_initializer(out_scale[idx], mul_scale),
                             create_initializer(1 / (2 ** int_scale[idx]), mul_int_scale)])

        clip_outs.append(ceil_out)
        input_data.append(data)
        nodes.extend([sub, mul_1, mul_2, floor])

    concat = onnx.helper.make_node(
        'Concat',
        inputs=clip_outs,
        outputs=['concat_out'],
        axis=axis
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['concat_out', 'zo'],
        outputs=['add_out']
    )
    clip = onnx.helper.make_node(
        'Clip',
        inputs=['add_out', 'min', 'max'],
        outputs=['output']
    )
    initializers.append(create_initializer(zo, 'zo'))
    nodes.extend([concat, add, clip])
    inputs = [onnx.helper.make_tensor_value_info('x' + str(idx),
                                                 onnx.TensorProto.FLOAT, x[idx].shape)
              for idx in range(len(x))]
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=inputs,
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/qconcat.onnx'.format(root_dir))


def build_add(x, output, out_scale, int_scale, zi, zo, max_value, min_value, axis, root_dir):
    nodes, clip_outs = [], []
    input_data = []
    initializers = [
        create_initializer(min_value, "min"),
        create_initializer(max_value, "max"),
    ]
    for idx in range(len(x)):
        data, zero_point = 'x' + str(idx), 'zi' + str(idx)
        sub_out, mul_scale = 'sub_' + str(idx), 'out_scale' + str(idx)
        mul_int_scale = 'int_scale' + str(idx)
        mul_1_out, clip_out = 'mul_1_' + str(idx), 'clip_' + str(idx)
        mul_2_out, ceil_out = 'mul_2_' + str(idx), 'ceil_' + str(idx)

        sub = onnx.helper.make_node(
            'Sub',
            inputs=[data, zero_point],
            outputs=[sub_out]
        )
        mul_1 = onnx.helper.make_node(
            'Mul',
            inputs=[sub_out, mul_scale],
            outputs=[mul_1_out]
        )
        mul_2 = onnx.helper.make_node(
            'Mul',
            inputs=[mul_1_out, mul_int_scale],
            outputs=[mul_2_out]
        )
        floor = onnx.helper.make_node(
            'Floor',
            inputs=[mul_2_out],
            outputs=[ceil_out]
        )

        initializers.extend([create_initializer(zi[idx], zero_point),
                             create_initializer(out_scale[idx], mul_scale),
                             create_initializer(1 / (2 ** int_scale[idx]), mul_int_scale)])

        clip_outs.append(ceil_out)
        input_data.append(data)
        nodes.extend([sub, mul_1, mul_2, floor])

    add = onnx.helper.make_node(
        'Add',
        inputs=[clip_outs[0], clip_outs[1]],
        outputs=['add_out']
    )
    add_zo = onnx.helper.make_node(
        'Add',
        inputs=['add_out', 'zo'],
        outputs=['add_zo_out']
    )
    clip = onnx.helper.make_node(
        'Clip',
        inputs=['add_zo_out', 'min', 'max'],
        outputs=['output']
    )
    initializers.append(create_initializer(zo, 'zo'))
    nodes.extend([add, add_zo, clip])
    inputs = [onnx.helper.make_tensor_value_info('x' + str(idx),
                                                 onnx.TensorProto.FLOAT, x[idx].shape)
              for idx in range(len(x))]
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=inputs,
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/qadd.onnx'.format(root_dir))


def build_split(x, output, out_scale, int_scale, zi, zo, max_value, min_value, axis, split_nums, root_dir):
    nodes, clip_outs = [], []
    input_data, output_names = [], []
    initializers = [
        create_initializer(min_value, "min"),
        create_initializer(max_value, "max"),
    ]
    split_names = ['split' + str(idx) for idx in range(len(split_nums))]
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_out']
    )
    split = onnx.helper.make_node(
        'Split',
        inputs=['sub_out', 'split_nums'],
        outputs=split_names,
        axis=axis
    )
    initializers.append(create_initializer(zi, 'zi'))
    initializers.append(create_initializer(split_nums, 'split_nums', dtype=np.int64))
    nodes.extend([sub, split])
    for idx in range(len(split_nums)):
        data, zero_point = split_names[idx], 'zo' + str(idx)
        out_name, mul_scale = 'output' + str(idx), 'out_scale' + str(idx)
        add_name, mul_int_scale = 'add_zo_' + str(idx), 'int_scale' + str(idx)
        mul_1_out, clip_out = 'mul_1_' + str(idx), 'clip_' + str(idx)
        mul_2_out, ceil_out = 'mul_2_' + str(idx), 'ceil_' + str(idx)

        mul_1 = onnx.helper.make_node(
            'Mul',
            inputs=[data, mul_scale],
            outputs=[mul_1_out]
        )
        mul_2 = onnx.helper.make_node(
            'Mul',
            inputs=[mul_1_out, mul_int_scale],
            outputs=[mul_2_out]
        )
        floor = onnx.helper.make_node(
            'Floor',
            inputs=[mul_2_out],
            outputs=[ceil_out]
        )
        add_zo = onnx.helper.make_node(
            'Add',
            inputs=[ceil_out, zero_point],
            outputs=[add_name]
        )
        clip = onnx.helper.make_node(
            'Clip',
            inputs=[add_name, 'min', 'max'],
            outputs=[out_name]
        )

        initializers.extend([create_initializer(zo[idx], zero_point),
                             create_initializer(out_scale[idx], mul_scale),
                             create_initializer(1 / (2 ** int_scale[idx]), mul_int_scale)])

        clip_outs.append(ceil_out)
        input_data.append(data)
        output_names.append(out_name)
        nodes.extend([mul_1, mul_2, floor, add_zo, clip])

    output = [onnx.helper.make_tensor_value_info(output_names[idx],
                                                 onnx.TensorProto.FLOAT, output[idx].shape)
              for idx in range(len(output))]
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)],
        outputs=output,
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/qsplit.onnx'.format(root_dir))


def build_shuffle(x, output, out_scale, int_scale, zi, zo, max_value, min_value, axis, shape, perm, split_nums,
                  root_dir):
    #############################################################################################################
    # build concat
    out_scale_concat, int_scale_concat = out_scale['concat'], int_scale['concat']
    zi_concat, zo_concat = zi['concat'], zo['concat']
    # x, output, w, b, out_shift, out_scale, int_scale, zi, zo, max_value, min_value, axis, root_dir
    nodes, clip_outs = [], []
    input_data = []
    initializers = [
        create_initializer(min_value, "min"),
        create_initializer(max_value, "max"),
    ]
    for idx in range(len(x)):
        data, zero_point = 'x' + str(idx), 'zi_concat' + str(idx)
        sub_out, mul_scale = 'sub_concat' + str(idx), 'out_scale_concat' + str(idx)
        mul_int_scale = 'int_scale_concat' + str(idx)
        mul_1_out, clip_out = 'mul_1_concat' + str(idx), 'clip_concat_' + str(idx)
        mul_2_out, ceil_out = 'mul_2_concat' + str(idx), 'ceil_concat_' + str(idx)

        sub = onnx.helper.make_node(
            'Sub',
            inputs=[data, zero_point],
            outputs=[sub_out]
        )
        mul_1 = onnx.helper.make_node(
            'Mul',
            inputs=[sub_out, mul_scale],
            outputs=[mul_1_out]
        )
        mul_2 = onnx.helper.make_node(
            'Mul',
            inputs=[mul_1_out, mul_int_scale],
            outputs=[mul_2_out]
        )
        floor = onnx.helper.make_node(
            'Floor',
            inputs=[mul_2_out],
            outputs=[ceil_out]
        )

        initializers.extend([create_initializer(zi_concat[idx], zero_point),
                             create_initializer(out_scale_concat[idx], mul_scale),
                             create_initializer(1 / (2 ** int_scale_concat[idx]), mul_int_scale)])

        clip_outs.append(ceil_out)
        input_data.append(data)
        nodes.extend([sub, mul_1, mul_2, floor])

    concat = onnx.helper.make_node(
        'Concat',
        inputs=clip_outs,
        outputs=['concat_out'],
        axis=axis['concat']
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['concat_out', 'zo_concat'],
        outputs=['add_out']
    )
    clip = onnx.helper.make_node(
        'Clip',
        inputs=['add_out', 'min', 'max'],
        outputs=['clip_output']
    )
    initializers.append(create_initializer(zo['concat'], 'zo_concat'))
    nodes.extend([concat, add, clip])
    #############################################################################################################
    # build reshape
    reshape_1 = onnx.helper.make_node(
        'Reshape',
        inputs=['clip_output', 'shape_1'],
        outputs=['reshape_1']
    )
    nodes.append(reshape_1)
    initializers.append(create_initializer(shape[0], 'shape_1', dtype=np.int64))
    #############################################################################################################
    # build transpose
    transpose = onnx.helper.make_node(
        'Transpose',
        inputs=['reshape_1'],
        outputs=['transpose_1'],
        perm=perm
    )
    nodes.append(transpose)
    #############################################################################################################
    # build reshape
    reshape_2 = onnx.helper.make_node(
        'Reshape',
        inputs=['transpose_1', 'shape_2'],
        outputs=['reshape_2']
    )
    nodes.append(reshape_2)
    initializers.append(create_initializer(shape[1], 'shape_2', dtype=np.int64))
    #############################################################################################################
    # build split
    out_scale_split, int_scale_split = out_scale['split'], int_scale['split']
    zi_split, zo_split = zi['split'], zo['split']
    input_data, output_names = [], []
    split_names = ['split' + str(idx) for idx in range(len(split_nums))]
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['reshape_2', 'zi_split'],
        outputs=['sub_split_out']
    )
    split = onnx.helper.make_node(
        'Split',
        inputs=['sub_split_out'],
        outputs=split_names,
        axis=axis['split']
    )
    initializers.append(create_initializer(zi_split, 'zi_split'))
    nodes.extend([sub, split])
    for idx in range(len(split_nums)):
        split_data, zero_point = split_names[idx], 'zo_split' + str(idx)
        out_name, mul_scale = 'output_split' + str(idx), 'out_scale_split' + str(idx)
        add_name, mul_int_scale = 'add_zo_split' + str(idx), 'int_scale_split' + str(idx)
        mul_1_out, clip_out = 'mul_1_split' + str(idx), 'clip_split' + str(idx)
        mul_2_out, ceil_out = 'mul_2_split' + str(idx), 'ceil_split' + str(idx)

        mul_1 = onnx.helper.make_node(
            'Mul',
            inputs=[split_data, mul_scale],
            outputs=[mul_1_out]
        )
        mul_2 = onnx.helper.make_node(
            'Mul',
            inputs=[mul_1_out, mul_int_scale],
            outputs=[mul_2_out]
        )
        floor = onnx.helper.make_node(
            'Ceil',
            inputs=[mul_2_out],
            outputs=[ceil_out]
        )
        add_zo = onnx.helper.make_node(
            'Add',
            inputs=[ceil_out, zero_point],
            outputs=[add_name]
        )
        clip = onnx.helper.make_node(
            'Clip',
            inputs=[add_name, 'min', 'max'],
            outputs=[out_name]
        )

        initializers.extend([create_initializer(zo_split[idx], zero_point),
                             create_initializer(out_scale_split[idx], mul_scale),
                             create_initializer(1 / (2 ** int_scale_split[idx]), mul_int_scale)])

        # clip_outs.append(ceil_out)
        output_names.append(out_name)
        nodes.extend([mul_1, mul_2, floor, add_zo, clip])

    #######################################################################
    inputs = [onnx.helper.make_tensor_value_info('x' + str(idx),
                                                 onnx.TensorProto.FLOAT, x[idx].shape)
              for idx in range(len(x))]
    output = [onnx.helper.make_tensor_value_info(output_names[idx],
                                                 onnx.TensorProto.FLOAT, output[idx].shape)
              for idx in range(len(output))]
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=inputs,
        outputs=output,
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/qshuffle.onnx'.format(root_dir))


def build_shuffle_only(x, output, shape, perm, root_dir):
    nodes, clip_outs = [], []
    initializers = []
    #############################################################################################################
    # build reshape
    reshape_1 = onnx.helper.make_node(
        'Reshape',
        inputs=['x', 'shape_1'],
        outputs=['reshape_1']
    )
    nodes.append(reshape_1)
    initializers.append(create_initializer(shape[0], 'shape_1', dtype=np.int64))
    #############################################################################################################
    # build transpose
    transpose = onnx.helper.make_node(
        'Transpose',
        inputs=['reshape_1'],
        outputs=['transpose_1'],
        perm=perm
    )
    nodes.append(transpose)
    #############################################################################################################
    # build reshape
    reshape_2 = onnx.helper.make_node(
        'Reshape',
        inputs=['transpose_1', 'shape_2'],
        outputs=['output']
    )
    nodes.append(reshape_2)
    initializers.append(create_initializer(shape[1], 'shape_2', dtype=np.int64))
    #############################################################################################################

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/qshuffle_only.onnx'.format(root_dir))


def build_quant(x, output, zero_point, scale, min_value, max_value, root_dir):

    mul = onnx.helper.make_node(
        'Mul',
        inputs=['x', 'scale'],
        outputs=['mul_output'],
    )
    round = onnx.helper.make_node(
        'Round',
        inputs=['mul_output'],
        outputs=['round_output']
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['round_output', 'zi'],
        outputs=['add_out']
    )
    clip = onnx.helper.make_node(
        'Clip',
        inputs=['add_out', 'min', 'max'],
        outputs=['output'],
    )

    nodes = [mul, add, round, clip]
    initializers = [
        create_initializer(zero_point, 'zi'),
        create_initializer(1 / scale, 'scale'),
        create_initializer(min_value, "min"),
        create_initializer(max_value, "max"),
    ]

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/quant.onnx'.format(root_dir))


def build_dequant(x, output, zero_point, scale, min_value, max_value, root_dir):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_out']
    )
    mul = onnx.helper.make_node(
        'Mul',
        inputs=['sub_out', 'scale'],
        outputs=['output'],
    )

    nodes = [sub, mul]
    initializers = [
        create_initializer(zero_point, 'zi'),
        create_initializer(scale, 'scale'),
        # create_initializer(min_value, "min"),
        # create_initializer(max_value, "max"),
    ]

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/dequant.onnx'.format(root_dir))


def build_resize(x, output, zi, zo, s1, s2, scales, min_value, max_value, root_dir):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_output']
    )
    mul_1 = onnx.helper.make_node(
        'Mul',
        inputs=['sub_output', 'scale1'],
        outputs=['mul_1_output'],
    )
    resize = onnx.helper.make_node(
        'Resize',
        mode='linear',
        inputs=['mul_1_output', 'scales'],
        outputs=['resize_output'],
        coordinate_transformation_mode='align_corners',
        cubic_coeff_a=0.75,
        nearest_mode='floor'
    )
    mul_2 = onnx.helper.make_node(
        'Mul',
        inputs=['resize_output', 'scale2'],
        outputs=['mul_2_output'],
    )
    round = onnx.helper.make_node(
        'Round',
        inputs=['mul_2_output'],
        outputs=['round_output']
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['round_output', 'zo'],
        outputs=['add_out']
    )
    clip = onnx.helper.make_node(
        'Clip',
        inputs=['add_out', 'min', 'max'],
        outputs=['output'],
    )

    nodes = [sub, mul_1, resize, mul_2, round, add, clip]
    initializers = [
        create_initializer(zi, 'zi'),
        create_initializer(zo, 'zo'),
        create_initializer(s1, 'scale1'),
        create_initializer(s2, 'scale2'),
        create_initializer(min_value, 'min'),
        create_initializer(max_value, 'max'),
        create_initializer(scales, 'max'),
    ]

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/qresize.onnx'.format(root_dir))


def build_act(x, output, act_func, zi, zo, out_scale, int_scale, min_value, max_value, root_dir):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_output']
    )
    act = onnx.helper.make_node(
        "act_func",
        inputs=['sub_output'],
        outputs=['act_output']
    )
    mul_1 = onnx.helper.make_node(
        'Mul',
        inputs=['act_output', 'out_scale'],
        outputs=['mul_1_output']
    )
    mul_2 = onnx.helper.make_node(
        'Mul',
        inputs=['mul_1_output', 'int_scale'],
        outputs=['mul_2_output']
    )
    ceil = onnx.helper.make_node(
        'Ceil',
        inputs=['mul_2_output'],
        outputs=['ceil_output']
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['ceil_output'],
        outputs=['output']
    )
    nodes = [sub, act, mul_1, mul_2, ceil, add]

    initializers = [
        create_initializer(x, 'x'),
        create_initializer(output, 'output'),
        create_initializer(zi, 'zi'),
        create_initializer(zo, 'zo'),
        create_initializer(out_scale, 'out_scale'),
        create_initializer(1/(2**int_scale), 'int_scale'),
        create_initializer(min_value, 'min'),
        create_initializer(max_value, 'max'),
    ]

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/q{}.onnx'.format(root_dir, act_func))


def build_relu6(x, output, zi, zo, out_scale, int_scale, min_value, max_value, root_dir):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_output']
    )
    relu6 = onnx.helper.make_node(
        "Clip",
        inputs=['sub_output', '0', 'relu6_max'],
        outputs=['relu6_output']
    )
    mul_1 = onnx.helper.make_node(
        'Mul',
        inputs=['relu6_output', 'out_scale'],
        outputs=['mul_1_output']
    )
    mul_2 = onnx.helper.make_node(
        'Mul',
        inputs=['mul_1_output', 'int_scale'],
        outputs=['mul_2_output']
    )
    ceil = onnx.helper.make_node(
        'Ceil',
        inputs=['mul_2_output'],
        outputs=['ceil_output']
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['ceil_output'],
        outputs=['output']
    )
    nodes = [sub, relu6, mul_1, mul_2, ceil, add]

    initializers = [
        create_initializer(x, 'x'),
        create_initializer(output, 'output'),
        create_initializer(zi, 'zi'),
        create_initializer(zo, 'zo'),
        create_initializer(out_scale, 'out_scale'),
        create_initializer(1 / (2 ** int_scale), 'int_scale'),
        create_initializer(min_value, 'min'),
        create_initializer(max_value, 'max'),
    ]

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)],
        initializer=initializers
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/q{}.onnx'.format(root_dir, "relu6"))


def build_MaxPooling():
    pooling = onnx.helper.make_node(
        'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2]
    )
    nodes = [pooling]


def build_AvgPooling(x, output, s1, s2, zi, zo, min_value, max_value, root_dir):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_output']
    )
    mul_1 = onnx.helper.make_node(
        'Mul',
        inputs=['sub_output', 's1'],
        outputs=['mul_1_output']
    )
    pooling = onnx.helper.make_node(
        "AveragePool",
        inputs=['mul_1_output'],
        outputs=['pool_output'],
        kernel_shape=[5, 5],
        pads=[0,0,0,0]
    )
    mul_2 = onnx.helper.make_node(
        'Mul',
        inputs=['pool_output'],
        outputs=['mul_2_output']
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['mul_2_output', 'zo'],
        outputs=['add_output']
    )
    round = onnx.helper.make_node(
        'Round',
        inputs=['add_output'],
        outputs=['round_output']
    )
    clip = onnx.helper.make_node(
        'Clip',
        inputs=['round_output', 'min', 'max'],
        outputs=['output']
    )
    nodes = [sub, mul_1, pooling, mul_2, add, round, clip]
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)]
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/q{}.onnx'.format(root_dir, "avgerpool"))


def build_lstm():
    pass


def build_layernorm(x, output):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_output']
    )
    mul_1 = onnx.helper.make_node(
        'Mul',
        inputs=['sub_output', 's1'],
        outputs=['mul_1_output']
    )
    bn = onnx.helper.make_node(
        'LayerNormalization',
        inputs=['mul_1_output', 's', 'bias', 'mean', 'var'],
        outputs=['bn_output'],
        epsilon=1e-5,
    )
    mul_2 = onnx.helper.make_node(
        'Mul',
        inputs=['bn_output', 's2'],
        outputs=['mul_2_output']
    )
    round = onnx.helper.make_node(
        'Round',
        inputs=['mul_2_output'],
        outputs=['round_output']
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['round_output'],
        outputs=['add_output']
    )
    clip = onnx.helper.make_node(
        'Clip',
        inputs=['add_output', 'min', 'max'],
        outputs=['output']
    )
    nodes = [sub, mul_1, bn, mul_2, round, add, clip]

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)]
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/q{}.onnx'.format(root_dir, "layernorm"))


def build_batchnorm(x, output):
    sub = onnx.helper.make_node(
        'Sub',
        inputs=['x', 'zi'],
        outputs=['sub_output']
    )
    mul_1 = onnx.helper.make_node(
        'Mul',
        inputs=['sub_output', 's1'],
        outputs=['mul_1_output']
    )
    bn = onnx.helper.make_node(
        'BatchNormalization',
        inputs=['mul_1_output', 's', 'bias', 'mean', 'var'],
        outputs=['bn_output'],
        epsilon=1e-5,
    )
    mul_2 = onnx.helper.make_node(
        'Mul',
        inputs=['bn_output', 's2'],
        outputs=['mul_2_output']
    )
    round = onnx.helper.make_node(
        'Round',
        inputs=['mul_2_output'],
        outputs=['round_output']
    )
    add = onnx.helper.make_node(
        'Add',
        inputs=['round_output'],
        outputs=['add_output']
    )
    clip = onnx.helper.make_node(
        'Clip',
        inputs=['add_output', 'min', 'max'],
        outputs=['output']
    )
    nodes = [sub, mul_1, bn, mul_2, round, add, clip]
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('output',
                                                    onnx.TensorProto.FLOAT,
                                                    output.shape)]
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    onnx.save_model(model, '{}/work_dir/q{}.onnx'.format(root_dir, "batchnorm"))


def build_maxpool():
    pool = onnx.helper.make_node(
        'Maxpool'
    )


def qconv_numpy(data, w, b, out_shift, out_scale, int_scale, zi, zo):
    conv = F.conv2d(torch.from_numpy(data) - zi,
                    weight=torch.from_numpy(w),
                    bias=torch.from_numpy(b))
    conv = conv.numpy().astype(np.int32) >> out_shift
    conv = np.clip(conv, -128, 127).astype(np.int16)
    conv = conv * out_scale
    conv = conv >> int_scale
    conv = np.clip(conv + zo, -128, 127)

    return conv


def qfc_numpy(data, w, b, out_shift, out_scale, int_scale, zi, zo):
    fc = F.linear(torch.from_numpy(data) - zi,
                  weight=torch.from_numpy(w),
                  bias=torch.from_numpy(b))
    fc = fc.numpy().astype(np.int32) >> out_shift
    fc = np.clip(fc, -128, 127)  # .astype(np.int16)
    fc = fc * out_scale
    fc = fc >> int_scale
    fc = np.clip(fc + zo, -128, 127)

    return fc


def qconcat_numpy(x, out_scale, int_scale, zi, zo, axis):
    inputs = []
    for idx in range(len(x)):
        data = (x[idx].astype(np.int16) - zi[idx]) * out_scale[idx]
        data = data.astype(np.int16) >> int_scale[idx]
        inputs.append(torch.from_numpy(data))

    if hasattr(torch, 'concat'):
        output = torch.concat(inputs, dim=axis) + zo
    else:
        output = torch.cat(inputs, dim=axis) + zo
    output = np.clip(output.numpy(), -128, 127)
    return output


def qadd_numpy(x, out_scale, int_scale, zi, zo, axis):
    inputs = []
    for idx in range(len(x)):
        data = (x[idx].astype(np.int16) - zi[idx]) * out_scale[idx]
        data = data.astype(np.int16) >> int_scale[idx]
        inputs.append(torch.from_numpy(data))
    output = inputs[0] + inputs[1] + zo
    output = np.clip(output.numpy(), -128, 127)
    return output


def qsplit_numpy(x, out_scale, int_scale, zi, zo, axis, split_nums):
    inputs = torch.split(torch.from_numpy(x) - zi, split_size_or_sections=split_nums, dim=axis)
    inputs = [data.numpy() for data in inputs]
    outputs = []
    for idx in range(len(inputs)):
        data = inputs[idx].astype(np.int16) * out_scale[idx]
        data = data.astype(np.int16) >> int_scale[idx]
        data = np.clip(data + zo[idx], -128, 127)
        outputs.append(data)
    return outputs


def qshuffle_numpy(x, out_scale, int_scale, zi, zo, max_value, min_value, axis, shape, perm, split_nums):
    concat = [torch.from_numpy(data.astype(np.int16) - zi['concat'][idx]) for idx, data in enumerate(x)]
    concat = [data * out_scale['concat'][idx] for idx, data in enumerate(concat)]
    concat = [data >> int_scale['concat'][idx] for idx, data in enumerate(concat)]
    if hasattr(torch, 'concat'):
        concat = torch.concat(concat, dim=axis['concat'])
    else:
        concat = torch.cat(concat, dim=axis['concat'])
    concat = torch.clip(concat + zo['concat'], min_value, max_value)
    reshape = concat.reshape(shape[0])
    transpose = torch.transpose(reshape, 1, 2)
    reshape = transpose.reshape(shape[1])
    split = torch.split(reshape - zi['split'], split_size_or_sections=split_nums, dim=axis['split'])
    split = [data * out_scale['split'][idx] for idx, data in enumerate(split)]
    output = [data >> int_scale['split'][idx] for idx, data in enumerate(split)]
    output = [torch.clip(data + zo['split'][idx], min_value, max_value) for idx, data in enumerate(output)]
    return [data.numpy() for data in output]


def qshuffle_only_numpy(x, shape, perm):
    reshape = torch.from_numpy(x).reshape(shape[0])
    transpose = torch.transpose(reshape, 1, 2)
    reshape = transpose.reshape(shape[1])
    return reshape.numpy()


def quant_numpy(x, min_value, max_value):
    scale = np.abs(x).max() / max_value
    zero_point = 0
    out = np.clip(np.round(x/scale), min_value, max_value)
    return out, scale, zero_point


def dequant_numpy(x, zero_point, scale):
    return (x - zero_point) * scale


def test_qconv():
    x = np.ceil(np.random.randn(3, 3, 256, 256).astype(np.float32) * 127)
    w = np.ceil(np.random.randn(64, 3, 3, 3).astype(np.float32) * 127)
    b = np.ceil(np.random.randn(64).astype(np.float32) * 127)
    out_shift = np.float32(1 / (2 ** 7))
    out_scale = np.float32(166)
    int_scale = np.float32(1 / (2 ** 8))
    zi = np.float32(-15)
    zo = np.float32(20)
    res = qconv_numpy(x, w, b, 7, 166, 8, zi, zo)
    # root_dir = '/home/shiqing/Downloads/onnx_converter'
    build_qconv(x, res, w, b, out_shift, out_scale, int_scale, zi, zo,
                max_value=np.float32(127), min_value=np.float32(-128),
                root_dir=root_dir)
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']


    # cycle_num = 100
    sess = rt.InferenceSession('{}/work_dir/qconv.onnx'.format(root_dir), providers=providers, provider_options=device)
    start_t = time.process_time()
    for _ in range(cycle_num):
        x_name = sess.get_inputs()[0].name
        # w_name = sess.get_inputs()[1].name
        # b_name = sess.get_inputs()[2].name
        y_name = sess.get_outputs()[0].name
        pred_onx = sess.run([y_name], {
            x_name: x,
            # w_name: W, b_name: B
        })[0]
    clock_t0 = (time.process_time() - start_t) / cycle_num
    print('qconv onnx time consume is: {}'.format(clock_t0))
    start_t = time.process_time()
    for _ in range(cycle_num):
        res = qconv_numpy(x, w, b, 7, 166, 8, zi, zo)
    clock_t1 = (time.process_time() - start_t) / cycle_num
    print('qconv split consume is: {}'.format(clock_t1))
    # from simulator.checkerror import L2Simiarity, CosineSimiarity
    l2_error = L2Simiarity()(res, pred_onx)
    consine_error = CosineSimiarity()(res, pred_onx)
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))
    return clock_t0, clock_t1


def test_qfc():
    dims = 1024
    x = np.ceil(np.random.randn(3, dims).astype(np.float32) * 127)
    w = np.ceil(np.random.randn(dims, dims).astype(np.float32) * 127)
    b = np.ceil(np.random.randn(dims).astype(np.float32) * 127)
    out_shift = np.float32(1 / (2 ** 7))
    out_scale = np.float32(166)
    int_scale = np.float32(1 / (2 ** 8))
    zi = np.float32(-15)
    zo = np.float32(20)
    res = qfc_numpy(x, w, b, 7, 166, 8, zi, zo)
    # root_dir = '/home/shiqing/Downloads/onnx_converter'
    build_qfc(x, res, w, b, out_shift, out_scale, int_scale, zi, zo,
              max_value=np.float32(127), min_value=np.float32(-128),
              root_dir=root_dir)
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
        # providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # cycle_num = 100
    sess = rt.InferenceSession('{}/work_dir/qfc.onnx'.format(root_dir), providers=providers, provider_options=device)
    start_t = time.process_time()
    for _ in range(cycle_num):
        x_name = sess.get_inputs()[0].name
        # w_name = sess.get_inputs()[1].name
        # b_name = sess.get_inputs()[2].name
        y_name = sess.get_outputs()[0].name
        pred_onx = sess.run([y_name], {
            x_name: x,
            # w_name: W, b_name: B
        })[0]
    clock_t0 = (time.process_time() - start_t) / cycle_num
    print('qfc onnx time consume is: {}'.format(clock_t0))
    start_t = time.process_time()
    for _ in range(cycle_num):
        res = qfc_numpy(x, w, b, 7, 166, 8, zi, zo)
    clock_t1 = (time.process_time() - start_t) / cycle_num
    print('qfc split consume is: {}'.format(clock_t1))
    # from simulator.checkerror import L2Simiarity, CosineSimiarity
    l2_error = L2Simiarity()(res, pred_onx)
    consine_error = CosineSimiarity()(res, pred_onx)
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))
    return clock_t0, clock_t1


def test_concat():
    # root_dir = '/home/shiqing/Downloads/onnx_converter'
    x = [np.round(np.random.randn(3, 16, 64, 64) * 127) for _ in range(3)]
    x = [np.clip(data, -128, 127).astype(np.float32) for data in x]
    min_value, max_value = -128, 127
    out_scale = [78, 80, 90]
    int_scale = [7, 7, 8]
    zi, zo = [-15, -20, -30], 40
    axis = 1
    output = np.concatenate(x, axis=axis)
    build_concat(x, output, None, None, None, out_scale, int_scale, zi, zo, max_value, min_value, axis, root_dir)
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
        # providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # cycle_num = 100
    sess = rt.InferenceSession('{}/work_dir/qconcat.onnx'.format(root_dir), providers=providers,
                               provider_options=device)
    start_t = time.process_time()
    for _ in range(cycle_num):
        x_name = {}
        for idx, data in enumerate(sess.get_inputs()):
            x_name[data.name] = x[idx]
        # w_name = sess.get_inputs()[1].name
        # b_name = sess.get_inputs()[2].name
        y_name = sess.get_outputs()[0].name
        pred_onx = sess.run([y_name], x_name)[0]
    clock_t0 = (time.process_time() - start_t) / cycle_num
    print('qconcat onnx time consume is: {}'.format(clock_t0))
    start_t = time.process_time()
    for _ in range(cycle_num):
        res = qconcat_numpy(x, out_scale=out_scale, int_scale=int_scale, zi=zi, zo=zo, axis=axis)
    clock_t1 = (time.process_time() - start_t) / cycle_num
    print('qconcat split consume is: {}'.format(clock_t1))
    # from simulator.checkerror import L2Simiarity, CosineSimiarity
    l2_error = L2Simiarity()(res, pred_onx)
    consine_error = CosineSimiarity()(res, pred_onx)
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))
    return clock_t0, clock_t1


def test_add():
    x = [np.round(np.random.randn(3, 16, 64, 64) * 127) for _ in range(2)]
    x = [np.clip(data, -128, 127).astype(np.float32) for data in x]
    min_value, max_value = -128, 127
    out_scale = [78, 80]
    int_scale = [7, 8]
    zi, zo = [-15, -20, -30], 40
    axis = 1
    output = x[0] + x[1]
    build_add(x, output, out_scale, int_scale, zi, zo, max_value, min_value, axis, root_dir)
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
        # providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # cycle_num = 100
    sess = rt.InferenceSession('{}/work_dir/qadd.onnx'.format(root_dir), providers=providers, provider_options=device)
    start_t = time.process_time()
    for _ in range(cycle_num):
        x_name = {}
        for idx, data in enumerate(sess.get_inputs()):
            x_name[data.name] = x[idx]
        # w_name = sess.get_inputs()[1].name
        # b_name = sess.get_inputs()[2].name
        y_name = sess.get_outputs()[0].name
        pred_onx = sess.run([y_name], x_name)[0]
    clock_t0 = (time.process_time() - start_t) / cycle_num
    print('qadd onnx time consume is: {}'.format(clock_t0))
    start_t = time.process_time()
    for _ in range(cycle_num):
        res = qadd_numpy(x, out_scale=out_scale, int_scale=int_scale, zi=zi, zo=zo, axis=axis)
    clock_t1 = (time.process_time() - start_t) / cycle_num
    print('qadd split consume is: {}'.format(clock_t1))
    # from simulator.checkerror import L2Simiarity, CosineSimiarity
    l2_error = L2Simiarity()(res, pred_onx)
    consine_error = CosineSimiarity()(res, pred_onx)
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))
    return clock_t0, clock_t1


def test_split():
    x = np.round(np.random.randn(3, 16, 64, 64) * 127).astype(np.float32)
    x = np.clip(x, -128, 127)
    min_value, max_value = -128, 127
    out_scale = [78, 80]
    int_scale = [7, 8]
    zi, zo = -15, [-15, -20]
    axis, split_nums = 1, [8, 8]
    output = torch.split(torch.from_numpy(x), dim=axis, split_size_or_sections=split_nums)
    output = [data.numpy() for data in output]
    # output, out_scale, int_scale, zi, zo, max_value, min_value, axis, split_nums, root_dir
    build_split(x, output, out_scale, int_scale, zi, zo, 127, -128, axis, split_nums, root_dir)
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
        # providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # cycle_num = 100
    sess = rt.InferenceSession('{}/work_dir/qsplit.onnx'.format(root_dir), providers=providers, provider_options=device)
    start_t = time.process_time()
    y_name = [sess_out.name for sess_out in sess.get_outputs()]
    for _ in range(cycle_num):
        x_name = sess.get_inputs()[0].name
        # y_name = sess.get_outputs()[0].name
        pred_onx = sess.run(y_name, {
            x_name: x,
            # w_name: W, b_name: B
        })
    clock_t0 = (time.process_time() - start_t) / cycle_num
    print('qsplit onnx time consume is: {}'.format(clock_t0))
    start_t = time.process_time()
    for _ in range(cycle_num):
        res = qsplit_numpy(x, out_scale=out_scale, int_scale=int_scale, zi=zi, zo=zo, axis=axis, split_nums=split_nums)
    clock_t1 = (time.process_time() - start_t) / cycle_num
    print('qsplit consume is: {}'.format(clock_t1))
    # from simulator.checkerror import L2Simiarity, CosineSimiarity
    l2_error = L2Simiarity()(np.array(res), np.array(pred_onx))
    consine_error = CosineSimiarity()(np.array(res), np.array(pred_onx))
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))
    return clock_t0, clock_t1


def test_shuffle():
    x = [np.round(np.random.randn(3, 128, 64, 64) * 127) for _ in range(2)]
    x = [np.clip(data, -128, 127).astype(np.float32) for data in x]
    min_value, max_value = -128, 127
    out_scale = dict(concat=[78, 80], split=[80, 96])
    int_scale = dict(concat=[7, 8], split=[8, 9])
    zi = dict(concat=[-15, -20], split=-30)
    zo = dict(concat=-40, split=[-30, -40])
    axis = dict(concat=1, split=1)
    split_nums = [128, 128]
    shape = [[3, 2, 128, 64, 64], [3, -1, 64, 64]]
    perm = [0, 2, 1, 3, 4]
    if hasattr(torch, 'concat'):
        output = torch.concat([torch.from_numpy(data) for data in x], dim=axis['concat'])
    else:
        output = torch.cat([torch.from_numpy(data) for data in x], dim=axis['concat'])
    output = output.reshape(shape[0])
    output = torch.transpose(output, 1, 2)
    output = output.reshape(shape[1])
    output = torch.split(output, split_size_or_sections=split_nums, dim=axis['split'])
    output = [data.numpy() for data in output]
    build_shuffle(x, output, out_scale, int_scale, zi, zo, max_value, min_value, axis, shape, perm, split_nums,
                  root_dir)
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
        # providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # cycle_num = 100
    sess = rt.InferenceSession('{}/work_dir/qshuffle.onnx'.format(root_dir), providers=providers,
                               provider_options=device)
    start_t = time.process_time()
    y_name = [sess_out.name for sess_out in sess.get_outputs()]
    x_name = {}
    for idx, data in enumerate(sess.get_inputs()):
        x_name[data.name] = x[idx]
    for _ in range(cycle_num):
        # y_name = sess.get_outputs()[0].name
        pred_onx = sess.run(y_name, x_name)
    clock_t0 = (time.process_time() - start_t) / cycle_num
    print('qshuffle onnx time consume is: {}'.format(clock_t0))
    start_t = time.process_time()
    for _ in range(cycle_num):
        res = qshuffle_numpy(x, out_scale, int_scale, zi, zo, max_value, min_value, axis, shape, perm, split_nums)
    clock_t1 = (time.process_time() - start_t) / cycle_num
    print('qshuffle consume is: {}'.format(clock_t1))
    # from simulator.checkerror import L2Simiarity, CosineSimiarity
    l2_error = L2Simiarity()(np.array(res), np.array(pred_onx))
    consine_error = CosineSimiarity()(np.array(res), np.array(pred_onx))
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))

    return clock_t0, clock_t1


def test_shuffle_only():
    x = np.round(np.random.randn(3, 256, 64, 64) * 127).astype(np.float32)

    shape = [[3, 2, 128, 64, 64], [3, -1, 64, 64]]
    perm = [0, 2, 1, 3, 4]

    output = torch.from_numpy(x).reshape(shape[0])
    output = torch.transpose(output, 1, 2)
    output = output.reshape(shape[1]).numpy()
    build_shuffle_only(x, output, shape, perm, root_dir)
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
        # providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # cycle_num = 100
    sess = rt.InferenceSession('{}/work_dir/qshuffle_only.onnx'.format(root_dir), providers=providers,
                               provider_options=device)
    start_t = time.process_time()
    for _ in range(cycle_num):
        x_name = sess.get_inputs()[0].name
        # w_name = sess.get_inputs()[1].name
        # b_name = sess.get_inputs()[2].name
        y_name = sess.get_outputs()[0].name
        pred_onx = sess.run([y_name], {
            x_name: x,
            # w_name: W, b_name: B
        })[0]
    clock_t0 = (time.process_time() - start_t) / cycle_num
    print('qshuffle_only onnx time consume is: {}'.format(clock_t0))
    start_t = time.process_time()
    for _ in range(cycle_num):
        res = qshuffle_only_numpy(x, shape, perm)
    clock_t1 = (time.process_time() - start_t) / cycle_num
    print('qshuffle_only consume is: {}'.format(clock_t1))
    # from simulator.checkerror import L2Simiarity, CosineSimiarity
    l2_error = L2Simiarity()(np.array(res), np.array(pred_onx))
    consine_error = CosineSimiarity()(np.array(res), np.array(pred_onx))
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))

    return clock_t0, clock_t1


def test_quant_dequant():
    x = np.random.randn(1,3,64,64).astype(np.float32)
    output, scale, zero_point = quant_numpy(x, -128, 127)
    build_quant(x, output, zero_point, scale, -128, 127, root_dir)
    build_dequant(x, output, zero_point, scale, -128, 127, root_dir)

    if 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider']
        # providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # cycle_num = 100
    sess = rt.InferenceSession('{}/work_dir/quant.onnx'.format(root_dir), providers=providers, provider_options=device)
    sess_dequant = rt.InferenceSession('{}/work_dir/dequant.onnx'.format(root_dir), providers=providers, provider_options=device)
    start_t = time.process_time()
    for _ in range(cycle_num):
        x_name = sess.get_inputs()[0].name
        de_x_name = sess_dequant.get_inputs()[0].name
        # w_name = sess.get_inputs()[1].name
        # b_name = sess.get_inputs()[2].name
        y_name = sess.get_outputs()[0].name
        de_y_name = sess.get_outputs()[0].name
        pred_onx = sess.run([y_name], {
            x_name: x,
            # w_name: W, b_name: B
        })[0]
        pred_onx = sess_dequant.run([de_y_name], {
            de_x_name: pred_onx,
            # w_name: W, b_name: B
        })[0]
    clock_t0 = (time.process_time() - start_t) / cycle_num
    print('qconv onnx time consume is: {}'.format(clock_t0))
    start_t = time.process_time()
    for _ in range(cycle_num):
        res, _, _ = quant_numpy(x, -128, 127)
        res = dequant_numpy(res, zero_point, scale)
    clock_t1 = (time.process_time() - start_t) / cycle_num
    print('qconv split consume is: {}'.format(clock_t1))
    # from simulator.checkerror import L2Simiarity, CosineSimiarity
    l2_error = L2Simiarity()(res, pred_onx)
    consine_error = CosineSimiarity()(res, pred_onx)
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))
    return clock_t0, clock_t1


def test_qresize():
    x = np.random.randn(1,3,64,64).astype(np.float32)
    output = np.random.randn(1,3,128,128).astype(np.float32)
    zi, zo = -15, -20
    s1, s2 = 0.0036, 0.0032
    scales = [1,1,2,2]
    min_value, max_value = -128, 127
    build_resize(x, output, zi, zo, s1, s2, scales, min_value, max_value, root_dir)


def test_act():
    # acts = ['Relu', 'Relu6', 'Sigmoid', 'LeakyRelu', 'HardSigmoid', 'HardSwish', 'Tanh']
    acts = ['Relu', 'Sigmoid', 'LeakyRelu', 'HardSigmoid', 'HardSwish', 'Tanh']
    x = np.random.randn(1,3,64,64).astype(np.float32)
    zi, zo = -15, -20
    out_scale, int_scale = 166, 7
    output = np.random.randn(1,3,64,64).astype(np.float32)
    for act in acts:
        build_act(x, output, act, zi, zo, out_scale, int_scale, -128, 127, root_dir)


def test_hardsigmoid():
    x = np.random.randn(1,3,64,64).astype(np.float32)
    act = onnx.helper.make_node(
        "HardSigmoid",
        inputs=['x'],
        outputs=['y']
    )

    graph = onnx.helper.make_graph(
        nodes=[act],
        name='test_timeintelli_qconv.onnx',
        inputs=[onnx.helper.make_tensor_value_info('x',
                                                   onnx.TensorProto.FLOAT,
                                                   x.shape)
                ],
        outputs=[onnx.helper.make_tensor_value_info('y',
                                                    onnx.TensorProto.FLOAT,
                                                    x.shape)],
        initializer=[]
    )
    model = onnx.helper.make_model(graph, producer_name='backend-test',
                                   opset_imports=[onnx.helper.make_opsetid("", 15)])
    sess = rt.InferenceSession(model.SerializeToString())
    output0 = sess.run(None, {'x':x})
    output1 = F.hardsigmoid(torch.from_numpy(x)).numpy()
    l2_error = L2Simiarity()(np.array(output0), np.array(output1))
    consine_error = CosineSimiarity()(np.array(output0), np.array(output1))
    print('=> l2_error: {}, consine_error: {}'.format(l2_error, consine_error))

if __name__ == '__main__':
    # onnx_t, numpy_t = [], []
    # conv_0, conv_1 = test_qconv()
    # fc_0, fc_1 = test_qfc()
    # concat_0, concat_1 = test_concat()
    # add_0, add_1 = test_add()
    # split_0, split_1 = test_split()
    # shuffle_0, shuffle_1 = test_shuffle()
    # shuffle_only_0, shuffle_only_1 = test_shuffle_only()
    # onnx_t = conv_0 + fc_0 + concat_0 + add_0 + split_0 + shuffle_0 + shuffle_only_0
    # numpy_t = conv_1 + fc_1 + concat_1 + add_1 + split_1 + shuffle_1 + shuffle_only_1
    # print('onnx time is: {}, numpy time is: {}, ratio is: {}'.format(onnx_t, numpy_t, numpy_t / onnx_t))
    # test_quant_dequant()
    # x = np.random.randn(1, 3, 64, 64).astype(np.float32)
    # output = copy.deepcopy(x)
    # build_layernorm(x, output)
    # build_batchnorm(x, output)
    # test_split()
    test_hardsigmoid()
