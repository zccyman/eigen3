# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/16 10:37
# @File     : serialize.py
import copy
from array import array
import os

import numpy as np

try:
    from utils import Registry
except Exception:
    from onnx_converter.utils import Registry # type: ignore

SERIALIZE: Registry = Registry("serialize", scope="")


def writeFile(data, filename=None, mode="wb", file=None):
    if file:
        array("b", data).tofile(file)
    else:
        with open(filename, mode) as file: # type: ignore
            array("b", data).tofile(file) # type: ignore


def serializeDataToInt8(data):
    """pack different data types to int8 stream, the input data must be a vector"""
    if len(data.shape) != 1:
        print("Error! Fail to serialize the data to int 8")
        print("Data shape is ", data.shape)
        os._exit(-1)

    if "u" in data.dtype.name:
        result = data.flatten().view(np.uint8)
    else:
        result = data.flatten().view(np.int8)

    return result


class Serialize(object):
    def __init__(self, **kwargs):
        super(Serialize, self).__init__()
        self.Csize = kwargs["Csize"]
        self.Ksize = kwargs["Ksize"]
        self.I_Align = kwargs["I_Align"]
        self.O_Align = kwargs["O_Align"]
        self.ABGR = kwargs["ABGR"]
        self.fc_bias_align = kwargs["fc_bias_align"]
        self.conv_bias_align = kwargs["conv_bias_align"]
        self.data_channel_extension = kwargs["DATA_C_EXTEND"]

    @staticmethod
    def get_align_channel(ch, align_size):
        return ((ch + align_size - 1) // align_size) * align_size

    def serialize(self, **kwargs):
        pass

    def write(self, result, name):
        writeFile(result, name)

    def __call__(self, name, **kwargs):
        result = self.serialize(**kwargs)
        # self.write(result, name)

        return result


@SERIALIZE.register_module(name="data")
@SERIALIZE.register_module(name="splice")
class SerializeData(Serialize):
    def __init__(self, **kwargs):
        super(SerializeData, self).__init__(**kwargs)

    def serialize(self, **kwargs):
        data = copy.deepcopy(kwargs["data"])
        param = kwargs["param"]

        if kwargs["is_out"]:
            if "split" in param.keys():
                feat_id = kwargs["feat_id"]
                out_pad = [param["split"]["out_pad"][feat_id]]
                out_pad = [[0, out_pad[0][1] - out_pad[0][0]]]
                out_align = param["split"]["out_align"][feat_id]  # used in split layer
            elif "concat" in param.keys():
                out_pad = param["concat"]["out_pad"]
                out_align = param["concat"]["out_align"][0]
            else:
                out_pad = param["out_pad"]
                if kwargs["layer"].get_layer_type() in ["shuffle_only"]:
                    out_align = np.array(param["out_align"]).sum()
                else:
                    out_align = param["out_align"][0]
        else:
            if "split" in param.keys():
                feat_id = kwargs["feat_id"]
                out_pad = [param["split"]["in_pad"][feat_id]]
                out_pad = [[0, out_pad[0][1] - out_pad[0][0]]]
                out_align = param["split"]["in_align"][feat_id]  # used in split layer
            elif "concat" in param.keys():
                out_pad = param["concat"]["in_pad"]
                out_align = param["concat"]["in_align"][0]
            else:
                out_pad = param["in_pad"]
                if kwargs["layer"].get_layer_type() in ["shuffle_only"]:
                    out_align = np.array(param["in_align"]).sum()
                else:
                    out_align = param["in_align"][0]

        result = []

        if len(data.shape) in {2, 4}:
            data = np.squeeze(data)
        else:
            raise Exception(
                "NotImplementedError, SerializeData, data shape {}".format(data.shape)
            )

        if len(data.shape) == 3:
            data = data.transpose(1, 2, 0)
            H, W, C = data.shape

            data_pad = np.zeros([H, W, out_align], dtype=data.dtype)
            start = 0
            for (a, b) in out_pad:
                delta = b - a
                data_pad[:H, :W, np.arange(a, b)] = data[:H, :W, start : start + delta]
                start = start + delta

            for Cidx in range(0, out_align, self.Csize):
                for f1 in range(0, H):
                    for f2 in range(0, W):
                        data_int8 = serializeDataToInt8(
                            data_pad[f1, f2, Cidx : Cidx + self.Csize].reshape(-1)
                        )
                        result.extend(data_int8)

        elif len(data.shape) == 1:
            data_pad = np.zeros([out_align], dtype=data.dtype)
            start = 0
            for (a, b) in out_pad:
                delta = b - a
                data_pad[np.arange(a, b)] = data[start : start + delta]
                start = start + delta

            for Cidx in range(0, out_align, self.O_Align):
                data_int8 = serializeDataToInt8(
                    data_pad[Cidx : Cidx + self.O_Align].reshape(-1)
                )
                result.extend(data_int8)

        return result


@SERIALIZE.register_module(name="conv")
@SERIALIZE.register_module(name="convtranspose")
class SerializeConv2d(Serialize):
    def __init__(self, **kwargs):
        super(SerializeConv2d, self).__init__(**kwargs)

    def serialize(self, **kwargs):
        data = copy.deepcopy(kwargs["data"])
        param = kwargs["param"]
        in_align = param["in_align"]
        out_align = param["out_align"]
        in_pad = param["in_pad"]
        out_pad = param["out_pad"]
        layer_type = kwargs["layer"].get_layer_type()
        data_channel_extension, first_conv = False, False
        if "data_channel_extension" in kwargs.keys():
            data_channel_extension = kwargs["data_channel_extension"]
        if "first_conv" in kwargs.keys():
            first_conv = kwargs["first_conv"]

        if data_channel_extension and first_conv:
            if layer_type == "conv":
                out_c, in_c, k_h, k_w = data.shape
                extend = np.zeros((out_c, 1, k_h, k_w), dtype=data.dtype)
                extend = np.concatenate((data, extend), axis=1)
            else:  # convtranspose
                in_c, out_c, k_h, k_w = data.shape
                extend = np.zeros((1, out_c, k_h, k_w), dtype=data.dtype)
                extend = np.concatenate((data, extend), axis=0)
            if self.ABGR:
                in_pad = [[1, in_c + 1]]
            else:
                in_pad = [[0, in_c]]
            if layer_type == "conv":
                in_align = [self.get_align_channel(extend.shape[1], 4)]
            else: # convtranspose
                in_align = [self.get_align_channel(extend.shape[0], 4)]                   
        
        if layer_type == "conv":
            data = data.transpose(2, 3, 1, 0)
        else: # convtranspose
            data = data.transpose(2, 3, 0, 1)
        H, W, C, N = data.shape

        result = []

        data_pad_in = np.zeros([H, W, in_align[0], N], dtype=data.dtype)
        start = 0
        for (a, b) in in_pad:
            delta = b - a
            data_pad_in[:H, :W, np.arange(a, b), :N] = data[
                :H, :W, start : start + delta, :N
            ]
            start = start + delta

        if data_channel_extension and first_conv:
            data_pad_in = np.concatenate((data_pad_in, data_pad_in, data_pad_in, data_pad_in), axis=-2)
            
        H, W, C, N = data_pad_in.shape
        data_pad_out = np.zeros([H, W, C, out_align[0]], dtype=data.dtype)
        start = 0
        for (a, b) in out_pad:
            delta = b - a
            data_pad_out[:H, :W, :C, np.arange(a, b)] = data_pad_in[
                :H, :W, :C, start : start + delta
            ]
            start = start + delta

        Ksize = self.Ksize
        layer = kwargs.get("layer")
        if layer and layer.get_scale_type() in ["floatscale", "ffloatscale"]:
            if Ksize == 32:
                Ksize = 16
        H, W, C, N = data_pad_out.shape
        for i in range(0, N // Ksize):  # output channels
            for Cidx in range(0, C, self.Csize):  # input channels
                for f1 in range(0, H):
                    for f2 in range(0, W):
                        for Nidx in range(0, Ksize):  # output channels align size
                            data_int8 = serializeDataToInt8(
                                data_pad_out[
                                    f1,
                                    f2,
                                    Cidx : Cidx + self.Csize,
                                    Nidx + i * Ksize,
                                ].reshape(-1)
                            )
                            result.extend(data_int8)

        return result


@SERIALIZE.register_module(name="depthwiseconv")
class SerializeDepthwiseConv2d(Serialize):
    def __init__(self, **kwargs):
        super(SerializeDepthwiseConv2d, self).__init__(**kwargs)

    def serialize(self, **kwargs):
        data = copy.deepcopy(kwargs["data"])
        param = kwargs["param"]
        out_align = param["out_align"]
        out_pad = param["out_pad"]

        data = data.transpose(2, 3, 1, 0)
        H, W, C, N = data.shape  # C->input N->output

        result = []

        data_pad_in = np.zeros([H, W, C, out_align[0]], dtype=data.dtype)
        start = 0
        for (a, b) in out_pad:
            delta = b - a
            data_pad_in[:H, :W, :C, np.arange(a, b)] = data[
                :H, :W, :C, start : start + delta
            ]
            start = start + delta

        H, W, C, N = data_pad_in.shape
        for i in range(0, N // self.Csize):  # output channels
            for f1 in range(0, H):
                for f2 in range(0, W):
                    for Nidx in range(0, self.Csize):  # output channels align size
                        data_int8 = serializeDataToInt8(
                            data_pad_in[f1, f2, 0, Nidx + i * self.Csize].reshape(-1)
                        )
                        result.extend(data_int8)

        return result


@SERIALIZE.register_module(name="fc")
@SERIALIZE.register_module(name="gemm")
@SERIALIZE.register_module(name="matmul")
class SerializeFC(Serialize):
    def __init__(self, **kwargs):
        super(SerializeFC, self).__init__(**kwargs)

    def serialize(self, **kwargs):
        data = copy.deepcopy(kwargs["data"])
        param = kwargs["param"]
        in_align = param["in_align"]
        out_align = param["out_align"]
        in_pad = param["in_pad"]
        out_pad = param["out_pad"]

        data = data.transpose(1, 0)
        C, N = data.shape  # C->input   N->output

        result = []

        data_pad_in = np.zeros([in_align[0], N], dtype=data.dtype)
        start = 0
        for (a, b) in in_pad:
            delta = b - a
            data_pad_in[np.arange(a, b), :N] = data[start : start + delta, :N]
            start = start + delta

        C, N = data_pad_in.shape
        data_pad_out = np.zeros([C, out_align[0]], dtype=data.dtype)
        start = 0
        for (a, b) in out_pad:
            delta = b - a
            data_pad_out[:C, np.arange(a, b)] = data_pad_in[:C, start : start + delta]
            start = start + delta

        for in_idx in range(0, C, self.I_Align):
            for out_idx in range(0, out_align[0]):
                data_int8 = serializeDataToInt8(
                    data_pad_out[in_idx : in_idx + self.I_Align, out_idx].reshape(-1)
                )
                result.extend(data_int8)

        return result


@SERIALIZE.register_module(name="bias")
@SERIALIZE.register_module(name="bn")
class SerializeBias(Serialize):
    def __init__(self, **kwargs):
        super(SerializeBias, self).__init__(**kwargs)

    def serialize(self, **kwargs):
        data = copy.deepcopy(kwargs["data"])
        param = kwargs["param"]
        out_align = param["out_align"][0]
        out_pad = param["out_pad"]
        if kwargs["is_fc_bias"]:
            align_size = self.fc_bias_align
        else:
            align_size = self.conv_bias_align

        result = []

        layer = kwargs.get("layer")
        if layer and layer.get_ops_setting()["ops_string"][-1] in ["sigmoid", "hardsigmoid"]:
            bit_select = kwargs["layer"].get_ops_setting()["setting"]["bit_select"]
            mins = kwargs["layer"].get_ops_setting()["setting"]["mins"]
            out_shift = -layer.get_scales()[-1]["out_shift"]
            if isinstance(out_shift, np.ndarray):
                qout_shift = np.zeros(out_align, dtype=out_shift.dtype)
                qout_shift[:out_shift.shape[0]] = out_shift
            else:
                qout_shift = out_shift
            data_pad =  (2**qout_shift) * mins[bit_select] * np.ones([out_align], dtype=data.dtype)
        else:
            data_pad = np.zeros([out_align], dtype=data.dtype)
        start = 0
        for (a, b) in out_pad:
            delta = b - a
            data_pad[np.arange(a, b)] = data[start : start + delta]
            start = start + delta

        for out_idx in range(0, out_align, align_size):
            data_int8 = serializeDataToInt8(
                data_pad[out_idx : out_idx + align_size].reshape(-1)
            )
            result.extend(data_int8)

        return result


@SERIALIZE.register_module(name="table")
class SerializeTable(Serialize):
    def __init__(self, **kwargs):
        super(SerializeTable, self).__init__(**kwargs)

    def serialize(self, **kwargs):
        data = copy.deepcopy(kwargs["data"])
        align_size = self.Csize

        result = []

        for out_idx in range(0, data.shape[0], align_size):
            data_int8 = serializeDataToInt8(
                data[out_idx : out_idx + align_size].reshape(-1)
            )
            result.extend(data_int8)

        return result


if __name__ == "__main__":
    pass
