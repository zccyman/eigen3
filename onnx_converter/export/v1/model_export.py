# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/1 17:06
# @File     : model_export.py
import sys

sys.path.append("./")  # NOQA: E402

import copy
import json
import os
import re

# import encrypt
import numpy as np

try:
    from export.v1 import get_version
    from utils import Object, export_perchannel
except Exception:
    from onnx_converter.utils import Object, export_perchannel # type: ignore
    from onnx_converter.export.v1 import get_version # type: ignore

from .network import NETWORK_V1 as rt_factory
from .npu_layer import npu_layer_enums
from .serialize import SERIALIZE as serialize_factory
from .serialize import serializeDataToInt8, writeFile
from .wExport import WeightExport
from .wExport import wExport as exportW_factory


class mExportBase(WeightExport):
    def __init__(self, **kwargs):
        super(mExportBase, self).__init__(**kwargs)

        self.ignore_layers = (
            kwargs["ignore_layers"] if "ignore_layers" in kwargs.keys() else []
        )
        self.valid_export_layer = kwargs["valid_export_layer"]
        self.layer_map = kwargs["model_c"]["layer_map"]
        self.secret_key = kwargs["secret_key"]
        self.export_mode_c = kwargs["export_mode_c"]
        self.is_debug = kwargs["is_debug"]

        self.serialize_data = serialize_factory.get("data")(**self.bits) # type: ignore

        if "quan_graph" in kwargs.keys():
            self.quan_graph = kwargs["quan_graph"]
            self.layers = self.quan_graph.get_layers()

        self.is_voice_model = kwargs["is_voice_model"]
        self.save_weights = []
        self.save_indata = []
        self.save_feats = {}
        self.save_model = {}
        self.w_offset = dict(w_offset=0, tmp_offset=[])
        self.debug = 0
        if self.debug:
            self.test_conv_idx = 0
            self.test_feat_idx = 0
        self.root_fd = ''

    def init_wexport(self, **kwargs):
        for m in [
            "conv",
            "depthwiseconv",
            "convtranspose",
            "fc",
            "lstm",
            "gru",
            "leakyrelu",
            "prelu",
            "sigmoid",
            "swish",
            "tanh",
            "hardsigmoid",
            "hardtanh",
            "hardswish",
            "hardshrink",
            "table",
            "batchnormalization",
            "layernormalization",
            "instancenormalization",
        ]:
            setattr(self, "wExport_{}".format(m), exportW_factory.get(m)(**kwargs)) # type: ignore

    def get_graph(self):
        return self.quan_graph

    def set_graph(self, quan_graph):
        self.quan_graph = quan_graph
        self.layers = self.quan_graph.get_layers()

    @staticmethod
    def write_model_b(contents, file):
        contents = contents.replace(";\n", "")
        contents = contents.replace(",\n", "")
        while " " in contents:
            contents = contents.replace(" ", "")
        contents = contents.replace("{", "").replace("}", "")
        contents = contents.replace("[", "").replace("]", "")
        contents = contents.split(",")

        for content in contents:
            if "=" in content:
                content = content.split("=")[-1]
            if content == "":
                continue

            if content in npu_layer_enums.keys():
                content = npu_layer_enums[content]
            else:
                content = eval(content)

            if "int" in type(content).__name__:
                content = np.int32(content)
            elif "float" in type(content).__name__:
                content = np.float32(content)
            else:
                raise Exception("{} is invalid".format(content))

            writeFile(serializeDataToInt8(np.array([content])).tolist(), file=file)

    def set_root_fd(self, root_fd):
        self.root_fd = root_fd
        if self.root_fd:
            self.weights_dir = os.path.join(self.log_dir, 'weights', root_fd)
            self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=False)
            if not os.path.exists(self.weights_dir):
                os.makedirs(self.weights_dir)
        else:
            self.weights_dir = os.path.join(self.log_dir, "weights")
            self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)

    @staticmethod ### used when test export conv param and feature after conv
    def save_qdata(data, file_name, is_float=False):
        with open(file_name, "w", encoding="utf-8") as f:
            for data_ in data:
                if is_float:
                    f.write("%.4f" % (data_) + "\n")
                else:
                    f.write(str(data_) + "\n")

    @staticmethod
    def get_align_channel(ch, align_size):
        return ((ch + align_size - 1) // align_size) * align_size

    @staticmethod
    def set_last_layer_fpdata(layer):
        ### set dequant data for last layer
        outputs = layer.get_out_data()
        ops = layer.get_ops_instance()
        ops_name = layer.get_layer_ops()["ops"]

        def dequant_scale(in_data, scales, so):
            # Extract zero point and scale
            zero_point = scales['zo']
            scale = scales.get('fscale', so['scale'])

            # Ensure scale is a NumPy array
            if not isinstance(scale, np.ndarray):
                output = in_data.astype(np.float32) - np.float32(zero_point)
                return np.float32(output * scale)

            # Handle input of shape (N, C, H, W)
            if len(in_data.shape) == 4:
                n, c, h, w = in_data.shape
                zero_point = np.full_like(scale, zero_point).reshape(n, c, 1, 1)
                output = np.int32(in_data) - np.int32(zero_point)
                return output * scale.reshape(n, c, 1, 1)

            # Handle input of shape (N, C)
            if len(in_data.shape) == 2:
                n, c = in_data.shape
                zero_point = np.full_like(scale, zero_point).reshape(n, c)
                output = np.int32(in_data) - np.int32(zero_point)
                return output * scale.reshape(n, c)

            # Raise an exception for invalid input shapes
            raise ValueError('last layer dequant_scale failed!!!')

        # consider multi-output layer, ex: split
        all_idx, so = 0, layer.get_scale()

        for op_idx, (op_name, op, data) in enumerate(zip(ops_name, ops, outputs)):
            if op_name in ["act"]:
                scales = op.get_scales()  # post-shift in postprocess
                if isinstance(scales, list):
                    fp_result = list()
                    for s_idx, scale in enumerate(scales):
                        output = copy.deepcopy(outputs[op_idx][s_idx]["output"])
                        if output.dtype not in [np.float32, np.float64]:
                            output = dequant_scale(output, scale, so[all_idx])
                        output = output.astype(np.float32)
                        fp_result.append(dict(output=output))
                else:
                    output = copy.deepcopy(outputs[op_idx]["output"])
                    if output.dtype not in [np.float32, np.float64]:
                        output = dequant_scale(output, scales, so[all_idx])
                    output = output.astype(np.float32)
                    fp_result = dict(output=output)

        outputs.append(fp_result) # type: ignore
        layer.set_out_data(outputs)

    @staticmethod
    def set_bias_data(layer):
        layer_type = layer.get_layer_type()
        if layer_type in ["conv", "depthwiseconv", "convtranspose", "fc", "gemm"]:
            has_bias = layer.get_ops_setting()["attrs"][0]["bias"]
            if has_bias:
                qbias = layer.get_qbias()
                dtype_weight = layer.get_qweight().dtype
                if dtype_weight in [np.int8, np.uint8]:
                    qbias = np.int32(qbias)
                else:
                    qbias = np.int64(qbias)
                if np.max(np.abs(qbias)) == 0:
                    layer.get_ops_setting()["attrs"][0].update(dict(bias=False))
                layer.set_qbias(qbias)
        elif layer_type in ["lstm", "gru"]:
            qbiass = []
            for qbias in layer.get_qbias():
                bit_num = re.findall(r"\d+", qbias.dtype.type.__name__)[0]
                qbias = eval("np.int" + bit_num)(qbias)
                qbiass.append(qbias)
            layer.set_qbias(qbiass)

    # @staticmethod
    def set_layer_datatype(self, layer):
        # if 1:
        in_data = layer.get_in_data()
        in_datatype = layer.get_datatype(in_data)
        layer.set_input_type(in_datatype)

        out_data = layer.get_out_data()
        datatype = layer.get_datatype(out_data)
        layer.set_output_type(datatype)
        # print(layer_type, in_datatype, datatype)

    def set_voice_model_feats(self, layer, is_voice_model=None):
        def expand_to_4d(matrix):
            if len(matrix.shape) == 3:
                matrix = matrix.squeeze(axis=0)
            while matrix.ndim < 4:
                matrix = np.expand_dims(matrix, axis=-1)
            return matrix

        in_data = layer.get_in_data()
        in_datatype = layer.get_datatype(copy.deepcopy(in_data))
        layer.set_input_type(in_datatype)

        if layer.get_layer_type() == "data":
            in_data = expand_to_4d(in_data)
            layer.set_in_data(in_data)
        elif isinstance(in_data, list):
            in_datas = copy.deepcopy(in_data)
            in_datas_ = []
            for in_data in in_datas:
                in_data = in_data["output"]
                in_data = expand_to_4d(in_data)
                in_datas_.append(dict(output=in_data))
            layer.set_in_data(in_datas_)
        else:
            os._exit(-1)

        out_data = layer.get_out_data()
        datatype = layer.get_datatype(copy.deepcopy(out_data))
        layer.set_output_type(datatype)

        if isinstance(out_data, list):
            out_datas = copy.deepcopy(out_data)
            out_datas_ = []
            for out_data in out_datas:
                out_data = out_data["output"]
                out_data = expand_to_4d(out_data)
                out_datas_.append(dict(output=out_data))
            layer.set_out_data(out_datas_)
        else:
            out_data = out_data["output"]
            out_data = expand_to_4d(out_data)
            out_data_ = layer.get_out_data()
            out_data_.update(dict(output=out_data))
            layer.set_out_data(out_data_)

            # print('test')

    @staticmethod
    def get_feature_shape(layer):
        if len(layer.get_in_data()[0]["output"].shape) == 4:
            feat_i = [
                layer.get_in_data()[0]["output"].shape[2],
                layer.get_in_data()[0]["output"].shape[3],
            ]
        elif len(layer.get_in_data()[0]["output"].shape) == 2:
            feat_i = [1, 1]
        else:
            raise Exception("get_feature_shape maybe incorrect!!!")

        if isinstance(layer.get_out_data(), list):
            out_data = layer.get_out_data()[-1]["output"]
            if len(layer.get_out_data()) > 3:
                out_data = layer.get_out_data()[-2]["output"]
            if len(out_data.shape) == 4:
                feat_o = [
                    out_data.shape[2],
                    out_data.shape[3],
                ]
            elif len(out_data.shape) == 2:
                feat_o = [1, 1]
            else:
                raise Exception("get_feature_shape maybe incorrect!!!")
        elif isinstance(layer.get_out_data(), dict):
            if len(layer.get_out_data()["output"].shape) == 4:
                feat_o = [
                    layer.get_out_data()["output"].shape[2],
                    layer.get_out_data()["output"].shape[3],
                ]
            elif len(layer.get_out_data()["output"].shape) == 2:
                feat_o = [1, 1]
            else:
                raise Exception("get_feature_shape maybe incorrect!!!")
        else:
            raise Exception("get_feature_shape maybe incorrect!!!")

        return feat_i, feat_o

    def get_pad_align(self, layer, layer_id):
        if "split" in self.layers[layer_id].get_insert().keys():
            split_id = self.layers[layer_id].get_insert()["split_ids"][layer.get_idx()]
            out_pad_ = self.layers[layer_id].get_insert()["split"]["out_pad"]
            out_align_ = self.layers[layer_id].get_insert()["split"]["out_align"]
            # if self.layers[layer_id].get_layer_name() == 'Split_cvt_5':
            #     print('test')
            if len(out_align_) == 1:
                in_pad = out_pad_
                in_align = out_align_
            else:
                in_pad = [out_pad_[split_id]]
                in_align = [out_align_[split_id]]
                in_pad = [[0, in_pad[0][1] - in_pad[0][0]]]
        else:
            in_pad = self.layers[layer_id].get_insert()["out_pad"]
            in_align = self.layers[layer_id].get_insert()["out_align"]

        return in_pad, in_align

    def get_split_ids(self, layer):
        name_ = [
            set(self.layers[id].get_onnx_input_name()) for id in layer.get_output_idx()
        ]
        name_inters = [
            set(layer.get_onnx_output_name()).intersection(key) for key in name_
        ]

        id = -1
        tmp_name = []
        split_ids = {}
        for out_idx, name in zip(layer.get_output_idx(), name_inters):
            if name not in tmp_name:
                tmp_name.append(name)
                id = id + 1
            split_ids[out_idx] = id

        return split_ids

    def recursive_down_layer(self, layer, result, result_id, split_id=0, mode="split"):
        """
        It recursively goes down the layers of the network, and returns the number of channels of the first
        convolutional layer it encounters

        :param layer: the layer to be processed
        :param result: the list of channels of the output of the layer
        :param result_id: the id of the layer that needs to be adjusted
        :param split_id: the index of the split layer, defaults to 0 (optional)
        :param mode: 'split' or 'conv_parallel_concat_elementwise', defaults to split (optional)
        :return: result is a list of channels, result_id is a list of layer_idx
        """
        # if layer.get_layer_type() in ["conv", "depthwiseconv"]:
        #     res = [layer.get_layer_ops()["attrs"][0]["in_c"]]
        #     res_id = [layer.get_idx()]
        #     result.extend(res)
        #     result_id.extend(res_id)
        if layer.get_layer_type() in ["split"] and mode == "split":
            res = layer.get_ops_setting()["attrs"][0]["split"]
            res_id = [layer.get_idx()]
            result.append(res)
            result_id.append(res_id)
            ### process depthwiseconv after split
            # for split_id, id in enumerate(layer.get_output_idx()):
            # result, result_id = self.recursive_down_layer(self.layers[id], result, result_id, split_id=split_id)
        elif (
            layer.get_layer_type() in ["add", "sub", "mul", "pmul", "cmul"]
            and mode == "conv_parallel_concat_elementwise"
        ):
            ### recursive top layer
            for id in layer.get_input_idx():
                result, result_id = self.recursive_top_layer(
                    self.layers[id], result, result_id
                )
            result_id = [layer.get_idx()]
        # elif layer.get_layer_type() in ["add", 'sigmoid']:
        #     res = [layer.get_in_data()[0]['output'].shape[1]]
        #     res_id = [layer.get_idx()]
        #     result.extend(res)
        #     result_id.extend(res_id)
        # elif layer.get_layer_type() in ["concat"]:
        #     res = [layer.get_in_data()[0]['output'].shape[1]]
        #     res_id = [layer.get_idx()]
        #     result.extend(res)
        #     result_id.extend(res_id)
        else:
            # , 'add', 'mul', 'cmul', 'pmul'
            if layer.get_layer_type() in [
                "conv",
                "fc",
                "concat",
                "shuffle",
                "shuffle_only",
            ]:
                return result, result_id
            elif layer.get_output_idx()[0] == -1:
                return result, result_id
            # elif layer.get_layer_type() in ['depthwiseconv']:
            #     # res = [layer.get_ops_setting()['attrs'][0]['out_c']]
            #     # res_id = [layer.get_idx()]
            #     # result.extend(res)
            #     # result_id.extend(res_id)
            #     if len(result) > 0:
            #         ch = layer.get_ops_setting()['attrs'][0]['out_c']
            #         result[split_id] = self.get_align_channel(ch, self.Csize)
            #         result_id[split_id] = layer.get_idx()
            #     return result, result_id
            else:
                for id in layer.get_output_idx():
                    result, result_id = self.recursive_down_layer(
                        self.layers[id], result, result_id, mode=mode
                    )
                # if self.layers[id].get_layer_type() in ['conv', 'depthwiseconv', 'split']:
                #     break
                # print(result, result_id)

        return result, result_id

    def recursive_top_layer(self, layer, result, result_id, mode="concat"):
        """
        It recursively finds the top layer of the network, and returns the number of channels of the top
        layer

        :param layer: the layer to be processed
        :param result: the number of channels of the output of the layer
        :param result_id: the index of the layer that needs to be split
        :param mode: 'concat' or 'split', defaults to concat (optional)
        :return: The result is a list of the number of channels of the input layers of the top layer.
        """
        if layer.get_layer_type() in ["concat"] and mode == "concat":
            for id in layer.get_input_idx():
                output_data = self.layers[id].get_out_data()
                if isinstance(output_data, list):
                    res = output_data[-1]["output"].shape[1]
                else:
                    res = output_data["output"].shape[1]
                # res_id = [self.layers[id].get_idx()]
                res_id = layer.get_idx()
                result.append(res)
                result_id.append(res_id)
        elif layer.get_layer_type() in ["split"] and mode == "split":
            res = layer.get_ops_setting()["attrs"][0]["split"]
            res_id = [layer.get_idx()]
            result.append(res)
            result_id.append(res_id)
        else:
            if layer.get_layer_type() in ["conv", "fc", "shuffle", "shuffle_only"]:
                return result, result_id
            elif layer.get_layer_type() in ["data"]:
                result.append(layer.get_in_data().shape[1])
                result_id.append(layer.get_idx())
                return result, result_id
            else:
                for id in layer.get_input_idx():
                    result, result_id = self.recursive_top_layer(
                        self.layers[id], result, result_id, mode=mode
                    )
                # print(result, result_id)

        return result, result_id

    def find_elementwise_layer(self, layer, is_exist=False, input_idx=[]):
        """
        > If the current layer is an elementwise layer, then check if the input layer is a concat or split
        layer. If not, then check if the output layer is a convolution layer. If so, then return True.
        Otherwise, check the output layer recursively

        :param layer: the layer to be checked
        :param is_exist: whether the elementwise layer is found, defaults to False (optional)
        :param input_idx: the input index of the elementwise layer
        """
        if layer.get_layer_type() in ["add", "sub", "mul", "cmul", "pmul"]:
            for id in input_idx:
                for mode in ["concat", "split"]:
                    result, result_id = [], []
                    result, result_id = self.recursive_top_layer(
                        self.layers[id], result, result_id, mode=mode
                    )
                    if len(result) > 0:
                        ### one input of element_wise layer is not (concat and split)
                        is_exist = True
                        break

            if not is_exist:
                mode = "conv_parallel_concat_elementwise"
                result, result_id = [], []
                for id in layer.get_output_idx():
                    result, result_id = self.recursive_down_layer(
                        self.layers[id], result, result_id, mode=mode
                    )
                if len(result) > 0:
                    is_exist = True
        else:
            if layer.get_layer_type() in [
                "conv",
                "fc",
                "concat",
                "shuffle",
                "shuffle_only",
            ]:
                return is_exist
            else:
                for id in layer.get_output_idx():
                    if self.layers[id].get_layer_type() in [
                        "add",
                        "sub",
                        "mul",
                        "cmul",
                        "pmul",
                    ]:
                        input_idx = copy.deepcopy(self.layers[id].get_input_idx())
                        if layer.get_idx() in input_idx:
                            input_idx.remove(layer.get_idx())
                    is_exist = self.find_elementwise_layer(
                        self.layers[id], is_exist=is_exist, input_idx=input_idx
                    )
        return is_exist

    def process_data(self, layer):
        channel = layer.get_in_data().shape[
            1
        ]  # layer.get_layer_ops()["attrs"][0]["shape"][1]
        if self.data_channel_extension and not self.is_voice_model:
            align_size = self.get_align_channel(channel, 4)
            if self.ABGR:
                out_pad, out_align = [[1, channel + 1]], [align_size]
            else:
                out_pad, out_align = [[0, channel]], [align_size]
        else:
            # Csizes = []
            # for out_idx in layer.get_output_idx():
            #     if self.layers[out_idx].get_layer_type() in ['fc', 'lstm']:
            #         Csizes.append(self.I_Align)
            #     else:
            #         Csizes.append(self.Csize)
            # Csize = np.max(Csizes)
            Csize = self.I_Align if self.is_voice_model else self.Csize
            align_size = self.get_align_channel(channel, Csize)
            if self.ABGR:
                out_pad, out_align = [[1, channel + 1]], [align_size]
            else:
                out_pad, out_align = [[0, channel]], [align_size]
        if len(layer.get_out_data()["output"].shape) == 4:
            feat_o = [
                layer.get_out_data()["output"].shape[2],
                layer.get_out_data()["output"].shape[3],
            ]
        elif len(layer.get_out_data()["output"].shape) == 2:
            feat_o = [[1, 1]]
        else:
            raise Exception("get_feature_shape maybe incorrect!!!")

        res = {
            "out_pad": out_pad,
            "out_align": out_align,
            "feat_o": [feat_o],
            "feat_i": [feat_o],
        }
        layer.set_insert(res)

    # conv input
    def process_conv_without_concat(self, layer):
        layer_type = layer.get_layer_type()
        if layer_type == "depthwiseconv":
            in_c = layer.get_layer_ops()["attrs"][0]["out_c"]
        else:
            in_c = layer.get_layer_ops()["attrs"][0]["in_c"]

        if self.data_channel_extension and layer.get_first_conv():
            in_align = [self.get_align_channel(in_c, 4)]
            if self.ABGR:
                in_pad = [[1, in_align[0]]]
            else:
                in_pad = [[0, in_align[0] - 1]]
        else:
            layer_id = layer.get_input_idx()[0]
            if "split" in self.layers[layer_id].get_insert().keys():
                split_id = self.layers[layer_id].get_insert()["split_ids"][
                    layer.get_idx()
                ]
                in_pad = [
                    self.layers[layer_id].get_insert()["split"]["out_pad"][split_id]
                ]
                in_align = [
                    self.layers[layer_id].get_insert()["split"]["out_align"][split_id]
                ]
                in_pad = [[0, in_pad[0][1] - in_pad[0][0]]]
            else:
                in_pad = self.layers[layer_id].get_insert()["out_pad"]
                if len(in_pad) == 1:
                    if layer.get_layer_type() in ["fc"]:
                        Csize = self.I_Align
                    else:
                        Csize = self.Csize
                    in_align = [self.get_align_channel(in_c, Csize)]
                else:
                    in_align = self.layers[layer_id].get_insert()["out_align"]

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,  # [[0, in_c]],
            "in_align": in_align,
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        layer.set_insert(res)

    # conv output
    def process_conv_without_split(self, layer):
        # if layer.get_layer_name() == 'Conv_92':
        #     print('test')

        layer_type = layer.get_layer_type()
        if layer_type in ["fc"]:
            align_size = self.O_Align
        elif layer_type == "depthwiseconv":
            align_size = self.Csize
        else:
            align_size = self.Ksize

        out_c = layer.get_layer_ops()["attrs"][0]["out_c"]
        if layer_type == "depthwiseconv":
            layer_id = layer.get_input_idx()[0]
            _, out_align = self.get_pad_align(layer, layer_id)
            res_conv = {
                "out_pad": [[0, out_c]],
                "out_align": out_align,
                "in_align": out_align,
            }
        else:
            out_align = [self.get_align_channel(out_c, align_size)]
            res_conv = {"out_pad": [[0, out_c]], "out_align": out_align}
        feat_i, feat_o = self.get_feature_shape(layer)
        res = {"feat_i": [feat_i], "feat_o": [feat_o]}
        res.update(res_conv)

        layer.set_insert(res)

    def process_concat_with_elementwise(self, layer):
        # if layer.get_layer_name() == 'Concat_386':
        #     print('test')

        in_pad = []
        channel = 0
        for split_id, id in enumerate(layer.get_input_idx()):
            if "split" in self.layers[id].get_insert().keys():
                split_id = self.layers[id].get_insert()["split_ids"][layer.get_idx()]
                real_c = self.layers[id].get_insert()["split"]["out_pad"][split_id]
                real_c = [0, real_c[1] - real_c[0]]
            else:
                real_c = self.layers[id].get_insert()["out_pad"][0]
            in_pad.append(real_c)
            channel += real_c[1]

        in_align = [self.get_align_channel(ch, self.Csize) for _, ch in in_pad]

        out_pad = [[0, channel]]
        out_align = [self.get_align_channel(channel, self.Csize)]

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": out_align,
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        res.update(dict(is_align=False))
        layer.set_insert(res)

    def process_concat(self, layer):
        # if layer.get_layer_name() == 'Concat_24':
        # print('test')

        in_pad, in_align = [], []
        for layer_id in layer.get_input_idx():
            in_pad_, in_align_ = self.get_pad_align(layer, layer_id)
            in_pad.extend(in_pad_)
            in_align.extend(in_align_)

        out_align = []
        for split_id, id in enumerate(layer.get_input_idx()):
            if "split" in self.layers[id].get_insert().keys():
                split_id = self.layers[id].get_insert()["split_ids"][layer.get_idx()]
                out_align_ = self.layers[id].get_insert()["split"]["out_align"]
                if len(out_align_) == 1:
                    out_align.append(out_align_[0])
                else:
                    out_align.append(out_align_[split_id])
            else:
                out_align_ = self.layers[id].get_insert()["out_align"]
                out_align.append(out_align_[0])

        out_align_ = [int(np.sum(out_align[:i])) for i in range(len(out_align))]
        out_pad = []
        for split_id, id in enumerate(layer.get_input_idx()):
            if "split" in self.layers[id].get_insert().keys():
                split_id = self.layers[id].get_insert()["split_ids"][layer.get_idx()]
                real_c = self.layers[id].get_insert()["split"]["out_pad"][split_id]
                real_c = [0, real_c[1] - real_c[0]]
                out_pad.append(real_c)
            else:
                out_pad.append(self.layers[id].get_insert()["out_pad"][0])

        out_pad = [
            list(np.array(pad) + align) for pad, align in zip(out_pad, out_align_)
        ]

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": [np.sum(out_align)],
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        layer.set_insert(res)

    def process_split(self, layer):
        layer_id = layer.get_input_idx()[0]
        if self.layers[layer_id].get_layer_type() in ["conv", "depthwiseconv"]:
            out_pad = [
                [0, pad[1] - pad[0]]
                for pad in self.layers[layer_id].get_insert()["out_pad"]
            ]
            out_align = [self.get_align_channel(pad[1], self.Csize) for pad in out_pad]
        else:
            result = layer.get_ops_setting()["attrs"][0]["split"]

            # if layer.get_layer_name() == 'Split_cvt_5':
            # print('test')

            out_align_ = [
                a for a, b in self.layers[layer_id].get_insert()["out_pad"][1:]
            ]
            out_align_.append(self.layers[layer_id].get_insert()["out_align"][0])
            if len(out_align_) == 1:
                out_align = [self.get_align_channel(ch, self.Csize) for ch in result]
                assert np.sum(out_align) == out_align_[0]
            else:
                out_align = [out_align_[0]]
                for i in range(1, len(out_align_)):
                    out_align.append(out_align_[i] - out_align_[i - 1])

            # out_pad = self.layers[layer_id].get_insert()["out_pad"]
            # if len(out_pad) > 1:
            out_pad = []
            for i, ch in enumerate(result):
                nonezero = [0, ch]
                if i > 0:
                    nonezero = [0, ch] + np.sum(out_align[:i])
                nonezero = list(nonezero)
                out_pad.append(nonezero)

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "out_pad": out_pad,
            "out_align": out_align,
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }

        res["in_pad"] = self.layers[layer_id].get_insert()["out_pad"]
        # in_align = 0
        # for a, b in res['in_pad']:
        #     in_align += self.get_align_channel(b - a, self.Csize)
        # res['in_align'] = [in_align]
        res["in_align"] = self.layers[layer_id].get_insert()["out_align"]

        res = dict(split=res)
        res.update(dict(split_ids=self.get_split_ids(layer)))
        layer.set_insert(res)

    def process_fc(self, layer):
        in_c = layer.get_layer_ops()["attrs"][0]["in_c"]
        out_c = layer.get_layer_ops()["attrs"][0]["out_c"]
        in_pad = [[0, in_c]]
        in_align = [self.get_align_channel(in_c, self.I_Align)]  # 8
        out_pad = [[0, out_c]]
        out_align = [self.get_align_channel(out_c, self.O_Align)]  # 4

        next_layer_id = layer.get_output_idx()[0]
        next_layer_type = self.layers[next_layer_id].get_layer_type()
        if next_layer_type == "fc" and next_layer_id > 0:
            next_in_c = self.layers[next_layer_id].get_layer_ops()["attrs"][0]["in_c"]
            next_in_align = [self.get_align_channel(next_in_c, self.I_Align)]
            if out_align[0] < next_in_align[0]:
                out_align = next_in_align

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": out_align,
        }
        layer.set_insert(res)

    def process_lstm(self, layer):
        in_pad, in_align = [], []
        for layer_id in layer.get_input_idx():
            if layer_id < 0:
                continue
            in_pad_, in_align_ = self.get_pad_align(layer, layer_id)
            in_pad.extend(in_pad_)
            in_align.extend(in_align_)

        _, in_c = in_pad[0]
        hidden_size = layer.get_ops_setting()["attrs"][0]["hidden_size"]
        for ch in [in_c, hidden_size]:
            in_pad.append([0, ch])
            in_align.append(self.get_align_channel(ch, self.I_Align))

        res_concat = {"in_pad": in_pad, "in_align": in_align}

        out_c1, out_c2 = (
            layer.get_qweight()[0].shape[1],
            layer.get_qweight()[1].shape[1],
        )
        result = [hidden_size, out_c1, out_c2]
        out_align = [self.get_align_channel(ch, self.O_Align) for ch in result]

        out_pad = [[0, ch] for ch in result]
        res_split = {"out_pad": out_pad, "out_align": out_align}

        feat_i, feat_o = self.get_feature_shape(layer)

        res = dict(split=res_split, feat_i=[feat_i], feat_o=[feat_o])
        res["split"].update(res_concat) # type: ignore
        split_ids = self.get_split_ids(layer)
        # for key in split_ids.keys():
        #     if key != -1: split_ids[key] = 2
        res.update(dict(split_ids=split_ids))
        res.update(dict(is_align=True)) # type: ignore
        layer.set_insert(res)

    def process_gru(self, layer):
        in_pad, in_align = [], []
        for layer_id in layer.get_input_idx():
            if layer_id < 0:
                continue
            in_pad_, in_align_ = self.get_pad_align(layer, layer_id)
            in_pad.extend(in_pad_)
            in_align.extend(in_align_)

        _, in_c = in_pad[0]
        hidden_size = layer.get_ops_setting()["attrs"][0]["hidden_size"]
        for ch in [in_c, hidden_size]:
            in_pad.append([0, ch])
            in_align.append(self.get_align_channel(ch, self.I_Align))

        res_concat = {"in_pad": in_pad, "in_align": in_align}

        out_c = layer.get_qweight()[0].shape[1]
        result = [hidden_size, out_c]
        out_align = [self.get_align_channel(ch, self.O_Align) for ch in result]

        out_pad = [[0, ch] for ch in result]
        res_split = {"out_pad": out_pad, "out_align": out_align}

        feat_i, feat_o = self.get_feature_shape(layer)

        res = dict(split=res_split, feat_i=[feat_i], feat_o=[feat_o])
        res["split"].update(res_concat) # type: ignore
        split_ids = self.get_split_ids(layer)
        # for key in split_ids.keys():
        #     if key != -1: split_ids[key] = 2
        res.update(dict(split_ids=split_ids))
        res.update(dict(is_align=True)) # type: ignore
        layer.set_insert(res)

    def process_shuffle(self, layer):
        # if layer.get_layer_name() == 'Concat_106':
        #     print('test')

        # in_align = []
        # for layer_id in layer.get_input_idx():
        #     in_align.extend(self.get_pad_align(layer, layer_id)[1])

        # out_align = []
        # for split_id, id in enumerate(layer.get_input_idx()):
        #     if "split" in self.layers[id].get_insert().keys():
        #         out_align.append(self.layers[id].get_insert()["split"]
        #                          ["out_align"][split_id])
        #     else:
        #         out_align.append(self.layers[id].get_insert()["out_align"][0])

        # out_align_ = [
        #     int(np.sum(out_align[:i])) for i in range(len(out_align))
        # ]
        # out_pad = []
        # for split_id, id in enumerate(layer.get_input_idx()):
        #     if "split" in self.layers[id].get_insert().keys():
        #         out_pad.append(
        #             self.layers[id].get_insert()["split"]["out_pad"][split_id])
        #     else:
        #         out_pad.append(self.layers[id].get_insert()["out_pad"][0])

        # out_pad = [
        #     list(np.array(pad) + align)
        #     for pad, align in zip(out_pad, out_align_)
        # ]
        # res_concat = {"out_pad": out_pad, "out_align": [np.sum(out_align)]}
        # res_concat = {"in_pad": out_pad, "in_align": [np.sum(out_align)]}

        in_pad, in_align = [], []
        for layer_id in layer.get_input_idx():
            in_pad_, in_align_ = self.get_pad_align(layer, layer_id)
            in_pad.extend(in_pad_)
            in_align.extend(in_align_)
        res_concat = {"in_pad": in_pad, "in_align": in_align}

        # result = [
        #     layer.get_layer_ops()["attrs"][4]["ends"][0] -
        #     layer.get_layer_ops()["attrs"][4]["starts"][0],
        #     layer.get_layer_ops()["attrs"][5]["ends"][0] -
        #     layer.get_layer_ops()["attrs"][5]["starts"][0]
        # ]
        result = layer.get_layer_ops()["attrs"][-1]["split"]
        out_align = [self.get_align_channel(ch, self.Csize) for ch in result]

        out_pad = [[0, ch] for ch in result]
        res_split = {"out_pad": out_pad, "out_align": out_align}

        feat_i, feat_o = self.get_feature_shape(layer)

        # res = dict(concat=res_concat, split=res_split, feat_i=[feat_i], feat_o=[feat_o])
        res = dict(split=res_split, feat_i=[feat_i], feat_o=[feat_o])
        res["split"].update(res_concat) # type: ignore
        res.update(dict(split_ids=self.get_split_ids(layer)))
        res.update(dict(is_align=False)) # type: ignore
        layer.set_insert(res)

    def process_shuffle_only(self, layer):
        # if layer.get_layer_name() == 'Reshape_186':
        #     print('test')

        layer_id = layer.get_input_idx()[0]
        if "split" in self.layers[layer_id].get_insert().keys():
            split_id = self.layers[layer_id].get_insert()["split_ids"][layer.get_idx()]
            in_pad = [self.layers[layer_id].get_insert()["split"]["out_pad"][split_id]]
            in_align = [
                self.layers[layer_id].get_insert()["split"]["out_align"][split_id]
            ]
        else:
            in_pad = self.layers[layer_id].get_insert()["out_pad"]
            in_align = self.layers[layer_id].get_insert()["out_align"]

        out_c = (
            layer.get_layer_ops()["attrs"][0]["shape"][1]
            * layer.get_layer_ops()["attrs"][0]["shape"][2]
        )
        out_pad = [[0, out_c]]
        out_align = [self.get_align_channel(out_c, self.Csize)]
        # out_pad = in_pad
        # out_align = in_align

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": out_align,
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        res.update(dict(is_align=False))
        layer.set_insert(res)

    def process_bn_layer(self, layer):
        self.process(layer)

    def process_elementwise_layer(self, layer):
        # if layer.get_layer_name() == 'Mul_177':
        # print('test')
        layer_id0, layer_id1 = layer.get_input_idx()
        in_pad0, in_align0 = self.get_pad_align(layer, layer_id0)
        in_pad1, in_align1 = self.get_pad_align(layer, layer_id1)

        in_align = []
        for a, b in zip(in_align0, in_align1):
            in_align.append(np.min([a, b]))

        if in_pad0 == in_pad1:
            in_pad = in_pad0
        # elif len(in_pad0) == 1:
        #     in_pad = in_pad0
        # elif len(in_pad1) == 1:
        #     in_pad = in_pad1
        else:
            raise Exception("process_elementwise_layer is error !!!")

        out_pad, out_align = in_pad, in_align

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": out_align,
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        layer.set_insert(res)

    def process_pad_layer(self, layer):
        layer_id = layer.get_input_idx()[0]
        in_pad, in_align = self.get_pad_align(layer, layer_id)

        out_c = layer.get_out_data()["output"].shape[1]
        out_pad = [[0, out_c]]
        out_align = [self.get_align_channel(out_c, self.Csize)]

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": out_align,
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        if self.layers[layer.get_input_idx()[0]].get_layer_type() in [
            "conv",
            "depthwiseconv",
        ]:
            res.update(dict(fmt="CubeFmt"))
        else:
            res.update(dict(fmt="MatFmt"))
        layer.set_insert(res)
    
    def process(self, layer):
        layer_id = layer.get_input_idx()[0]
        in_pad, in_align = self.get_pad_align(layer, layer_id)
        # if 'split' in self.layers[layer_id].get_insert().keys():
        #     split_id = self.layers[layer_id].get_insert()['split_ids'][layer.get_idx()]
        #     in_pad = [self.layers[layer_id].get_insert()['split']["out_pad"][split_id]]
        #     in_align = [self.layers[layer_id].get_insert()['split']["out_align"][split_id]]
        #     in_pad = [[0, in_pad[0][1] - in_pad[0][0]]]
        # else:
        #     in_pad = self.layers[layer_id].get_insert()["out_pad"]
        #     in_align = self.layers[layer_id].get_insert()["out_align"]

        out_pad = in_pad
        out_align = in_align

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": out_align,
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        if self.layers[layer.get_input_idx()[0]].get_layer_type() in [
            "conv",
            "depthwiseconv",
        ]:
            res.update(dict(fmt="CubeFmt"))
        else:
            res.update(dict(fmt="MatFmt"))
        layer.set_insert(res)

    def check_first_conv(self, layer):
        in_idx = layer.get_input_idx()
        if len(in_idx) > 1:
            flag = False
        else:
            if "data" == self.layers[in_idx[0]].get_layer_type():
                flag = True
            else:
                flag = False
        if layer.get_layer_type() not in ["conv", "convtranspose", "depthwiseconv"]:
            flag = False
        return flag

    def store_placeholder(self, layer, func_data):
        res = layer.feat_export(
            func_data,
            layer.get_out_data()["output"],
            layer=layer,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "indata.b"),
        )
        self.save_indata.extend(res)
    
    def export_features(self, layer):
        layer_idx = layer.get_idx()
        layer_type = layer.get_layer_type()
        # export features
        func_data = getattr(self, "serialize_{}".format("data"))
        self.save_indata = []
        if layer_type == "data":
            self.store_placeholder(layer, func_data)
            ### test export indata
            # if self.debug:
                # self.save_qdata(res, "./work_dir/feat_quant/data.txt", is_float=False)
                # fdata = layer.get_out_data()["output"].transpose(2, 3, 0, 1).reshape(-1)
                # self.save_qdata(fdata, './work_dir/feat_float/data.txt', is_float=True)

        elif layer_type == "split":
            # layer_names = layer.get_onnx_output_name()
            for feat_id, feat in enumerate(layer.get_out_data()):
                feat = feat["output"]
                # layer_name = "{}_{}_{}_{}.b".format(
                # str(layer_idx).zfill(4), layer_type, layer.get_layer_name(), layer_names[feat_id])
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4), layer_type, layer.get_idx(), feat_id
                )
                res = layer.feat_export(
                    func_data,
                    feat,
                    feat_id,
                    insert=layer.get_insert(),
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, layer_name),
                )
                self.save_feats.update({layer_name: res})
        elif layer_type == "shuffle":
            # layer_names = layer.get_onnx_output_name()
            feat_concat = layer.get_out_data()[0]["output"]
            # layer_name = "{}_{}_{}_{}.b".format(
            # str(layer_idx).zfill(4), layer_type, layer.get_layer_name(), layer_names[0])
            layer_name = "{}_{}_{}_{}.b".format(
                str(layer_idx).zfill(4), layer_type, layer.get_idx(), "concat"
            )
            res = layer.feat_export(
                func_data,
                feat_concat,
                0,
                insert=layer.get_insert(),
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, layer_name),
            )
            self.save_feats.update({layer_name: res})

            feat_split = [
                layer.get_out_data()[1]["output"],
                layer.get_out_data()[2]["output"],
            ]
            for feat_id, feat in enumerate(feat_split):
                # layer_name = "{}_{}_{}_{}.b".format(
                # str(layer_idx).zfill(4), layer_type,
                # layer.get_layer_name(),
                # layer_names[feat_id + 1])
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4), layer_type, layer.get_idx(), feat_id
                )
                res = layer.feat_export(
                    func_data,
                    feat,
                    feat_id,
                    insert=layer.get_insert(),
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, layer_name),
                )
                self.save_feats.update({layer_name: res})
        elif layer_type == "lstm":
            if isinstance(layer.get_out_data(), dict):
                layer_data = [layer.get_out_data()]
            else:
                layer_data = layer.get_out_data()
            # ops_list = ["ot", "ht", "ct", "hq", "fc1", "fc2", "fc", "fc_combine"]
            ops_list = ["ot", "ht", "ct", "hq", "fc1", "fc2", "fc"]
            layer.set_layer_ops(dict(ops=ops_list))
            for ops_id, qdata in enumerate(layer_data[: len(ops_list)]):
                if layer.get_layer_ops()["ops"][ops_id] in ["hq"]:
                    # insert = dict(out_pad=[[0, 257]], out_align=[264])
                    insert = {
                        key: [layer.get_insert()["split"][key1][0]]
                        for key1, key in zip(
                            ["in_pad", "in_align"], ["out_pad", "out_align"]
                        )
                    }
                elif layer.get_layer_ops()["ops"][ops_id] in ["fc1", "fc2", "fc"]:
                    # insert = dict(out_pad=[[0, 512]], out_align=[512])
                    insert = {
                        key: [layer.get_insert()["split"][key][-1]]
                        for key in ["out_pad", "out_align"]
                    }
                # elif layer.get_layer_ops()["ops"][ops_id] in ['fc_combine']:
                #     insert_ = {
                #         key: [layer.get_insert()["split"][key][-1]]
                #         for key in ["out_pad", "out_align"]
                #     }
                #     insert = dict()
                #     for key in insert_.keys():
                #         ins = insert_[key]
                #         item = []
                #         for iksy in ins:
                #             if isinstance(iksy, list):
                #                 item.append([i*2 for i in iksy])
                #             else:
                #                 item.append(iksy * 2)
                #         insert[key] = item

                #     print()
                else:
                    insert = {
                        key: [layer.get_insert()["split"][key][0]]
                        for key in ["out_pad", "out_align"]
                    }
                if ops_id == 0:
                    ops_id = "0"
                else:
                    ops_id = "0_{}_{}".format(
                        layer.get_layer_ops()["ops"][ops_id], layer.get_layer_name()
                    )
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4),
                    layer.get_layer_type(),
                    layer.get_idx(),
                    ops_id,
                )
                # insert = {key: [layer.get_insert()['split'][key][1]] \
                #                 for key in ['out_pad', 'out_align']}

                if qdata["output"].shape[1] > 1 and len(qdata["output"].shape) > 4:
                    qdatas = np.split(qdata["output"], qdata["output"].shape[1], axis=1)
                else:
                    qdatas = [qdata["output"]]

                res_ = []
                for qdata_ in qdatas:
                    qdata_ = np.squeeze(qdata_)
                    qdata_ = np.expand_dims(qdata_, axis=[0, 2, 3])
                    res = layer.feat_export(
                        func_data,
                        qdata_,
                        insert=insert,
                        layer_name=layer.get_layer_name(),
                        name=os.path.join(self.weights_dir, layer_name),
                    )
                    res_.extend(res)
                    self.save_feats.update({layer_name: res_})
        elif layer_type == "gru":
            if isinstance(layer.get_out_data(), dict):
                layer_data = [layer.get_out_data()]
            else:
                layer_data = layer.get_out_data()
            ops_list = ["ot", "ht", "hq", "fc1", "fc2"]
            layer.set_layer_ops(dict(ops=ops_list))
            for ops_id, qdata in enumerate(layer_data[: len(ops_list)]):
                if layer.get_layer_ops()["ops"][ops_id] in ["hq"]:
                    # insert = dict(out_pad=[[0, 257]], out_align=[264])
                    insert = {
                        key: [layer.get_insert()["split"][key1][0]]
                        for key1, key in zip(
                            ["in_pad", "in_align"], ["out_pad", "out_align"]
                        )
                    }
                elif layer.get_layer_ops()["ops"][ops_id] in ["fc1", "fc2"]:
                    # insert = dict(out_pad=[[0, 512]], out_align=[512])
                    insert = {
                        key: [layer.get_insert()["split"][key][-1]]
                        for key in ["out_pad", "out_align"]
                    }
                else:
                    insert = {
                        key: [layer.get_insert()["split"][key][0]]
                        for key in ["out_pad", "out_align"]
                    }
                if ops_id == 0:
                    ops_id = "0"
                else:
                    ops_id = "0_{}_{}".format(
                        layer.get_layer_ops()["ops"][ops_id], layer.get_layer_name()
                    )
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4),
                    layer.get_layer_type(),
                    layer.get_idx(),
                    ops_id,
                )
                # insert = {key: [layer.get_insert()['split'][key][1]] \
                #                 for key in ['out_pad', 'out_align']}

                if qdata["output"].shape[1] > 1 and len(qdata["output"].shape) > 4:
                    qdatas = np.split(qdata["output"], qdata["output"].shape[1], axis=1)
                else:
                    qdatas = [qdata["output"]]

                res_ = []
                for qdata_ in qdatas:
                    qdata_ = np.squeeze(qdata_)
                    qdata_ = np.expand_dims(qdata_, axis=[0, 2, 3])
                    res = layer.feat_export(
                        func_data,
                        qdata_,
                        insert=insert,
                        layer_name=layer.get_layer_name(),
                        name=os.path.join(self.weights_dir, layer_name),
                    )
                    res_.extend(res)
                    self.save_feats.update({layer_name: res_})
        else:
            # if layer.get_layer_name() == "Reshape_rep_squeeze_7":
            # print('test')
            # layer_names = [
            #     layer.get_onnx_output_name()[0] + "_" + ops
            #     for ops in layer.get_layer_ops()["ops"]
            # ]
            layer_names = [ops for ops in layer.get_layer_ops()["ops"]]
            fake_result_layer = False
            for out_idx in layer.get_output_idx():
                if out_idx != -1:
                    fake_result_layer = True
            if (
                layer.get_is_result_layer()
                and not fake_result_layer
                and layer_type in ["conv", "depthwiseconv", "gemm", "fc"]
            ):
                ### set dequant data for last layer
                self.set_last_layer_fpdata(layer)
                layer_names.append(layer.get_onnx_output_name()[0] + "_fp")

            if isinstance(layer.get_out_data(), dict):
                layer_data = [layer.get_out_data()]
            else:
                layer_data = layer.get_out_data()

            if layer_type in ["shuffle_only"]:
                layer_names = layer_names[-1:]
            if layer_type not in ['swish', 'gelu']:
                assert len(layer_names) == len(layer_data)

            for ops_id, qdata in enumerate(layer_data):
                # layer_name = "{}_{}_{}_{}.b".format(
                #     str(layer_idx).zfill(4), layer.get_layer_type(),
                #     layer.get_layer_name(),
                #     layer_names[ops_id])
                if layer.get_layer_type() in [
                    "conv",
                    "depthwiseconv",
                    "fc",
                    "gemm",
                    "matmul",
                ]:
                    if layer.get_is_result_layer():
                        if ops_id == len(layer_names) - 1:
                            ops_id = "0_fp"
                        elif ops_id == len(layer_names) - 2:
                            ops_id = "0"
                        else:
                            ops_id = "0_{}_{}".format(
                                layer_names[ops_id], layer.get_layer_name()
                            )
                    else:
                        if ops_id == len(layer_names) - 1:
                            ops_id = "0"
                        else:
                            ops_id = "0_{}_{}".format(
                                layer_names[ops_id], layer.get_layer_name()
                            )
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4),
                    layer.get_layer_type(),
                    layer.get_idx(),
                    ops_id,
                )
                res = layer.feat_export(
                    func_data,
                    qdata["output"],
                    insert=layer.get_insert(),
                    layer=layer,
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, layer_name),
                )
                self.save_feats.update({layer_name: res})

            ### test export feature after conv
            if self.debug:
                if layer_type in ["conv", "depthwiseconv"]:
                    self.save_qdata(
                        res, # type: ignore
                        "./work_dir/feat_quant/{}_{}.txt".format(
                            self.test_feat_idx, layer.get_layer_name()
                        ),
                        is_float=False,
                    )
                    # fdata = layer.get_in_fdata()[0].transpose(2, 3, 0, 1).reshape(-1)
                    # self.save_qdata(fdata, './work_dir/feat_float/{}_{}.txt'.format(
                    #     self.test_feat_idx, layer.get_layer_name()), is_float=True)
                    self.test_feat_idx = self.test_feat_idx + 1

    def export_weights(self):
        for layer in self.layers:
            if layer.is_extend():
                pass
            layer_type = layer.get_layer_type()
            # if layer.get_layer_name() == "prefinal-chain.affine":
            #    print("test")
            if layer_type not in self.valid_export_layer:
                if self.debug:
                    self.logger.warning(
                        "ignore layer: {}, when write weight.b".format(layer_type)
                    )
            else:
                # getattr(self, "wExport_{}".format(layer_type))(
                #     layer, self.save_weights, self.w_offset
                # )
                if layer_type in [
                    "conv",
                    "depthwiseconv",
                    "convtranspose",
                    "fc",
                    "lstm",
                    "gru",
                    "batchnormalization",
                    "layernormalization",
                    "leakyrelu",
                    "prelu",
                    "sigmoid",
                    "tanh",
                    "hardshrink",
                    "hardsigmoid",
                    "hardswish",
                    "hardtanh",
                ]:
                    getattr(self, "wExport_{}".format(layer_type))(
                        layer, self.save_weights, self.w_offset
                    )
                else:
                    w_offset = layer.get_w_offset()
                    w_offset["w_offset"] = self.w_offset["w_offset"]
                    layer.set_w_offset(w_offset)
                    # self.logger.info("{} has no weights!".format(layer_type))

    def write_weights(self):
        self.export_weights()
        writeFile(
            self.save_weights, os.path.join(self.weights_dir, "weight.b"), mode="ab"
        )
        self.logger.info("write weights done!")

    def write_features(self):
        for layer in self.layers:
            if layer.get_layer_type() != "data":
                self.export_features(layer)
        for key, feats in self.save_feats.items():
            key = key.replace("/", "-").replace(":", "-")
            writeFile(feats, os.path.join(self.weights_dir, key), mode="ab")
        self.logger.info("write features done!")

    def write_indata(self):
        for layer in self.layers:
            if layer.get_layer_type() == "data":
                self.export_features(layer)
        writeFile(
            self.save_indata, os.path.join(self.weights_dir, "indata.b"), mode="ab"
        )
        self.logger.info("write indata done!")

    def write_network(self):
        global in_data_shape

        contents = []
        input_scales = []
        ignore_layer_types = []
        for layer in self.layers:
            if layer.is_extend():
                pass
            layer_id = layer.get_idx()
            layer_type = layer.get_layer_type()
            if layer_type == "data":
                if "int" in layer.get_out_data()["output"].dtype.type.__name__:
                    input_scales.append(layer.get_scales()[-1]["out_scale"])
                else:
                    input_scales.append(1.0)
                if len(layer.get_ops_setting()["attrs"][0]["shape"]) == 4:
                    in_data_shape = layer.get_ops_setting()["attrs"][0]["shape"][2:]
                else:
                    in_data_shape = 1, 1
                if layer.get_insert()["out_align"][0] == 4:
                    in_data_align = self.Csize
                else:
                    in_data_align = layer.get_insert()["out_align"][0]

            scales = copy.deepcopy(layer.get_scales())
            if isinstance(scales, list):
                scales = scales[-1]
            zk = scales["zk"] if "zk" in scales else np.array([0])
            if isinstance(zk, list):
                zk = zk[0]
            if layer_type not in self.valid_export_layer or (
                self.export_version == 1 and isinstance(zk, np.ndarray)
            ):
                if layer_type not in ignore_layer_types:
                    ignore_layer_types.append(layer_type)
                # self.logger.warning(
                #     "ignore layer: {}, when export model.c".format(layer_type)
                # )
            else:
                content = getattr(self, "network_{}".format(layer_type)).save(layer)
                contents.append(content + "\n")

            if layer_id == len(self.layers) - 1:
                content = "{};\n".format("}")
                contents.append(content)
        if len(ignore_layer_types) > 1:
            self.logger.warning("{} export not support".format(ignore_layer_types))

        headers = copy.deepcopy(self.model_template)
        headers[-1] += "\n"
        LayerInfo_t, layers_t = re.sub("[^A-Za-z_]", " ", headers[-1]).split(" ")[:2]
        content = "i32 layers_cnt=sizeof({})/sizeof({});\n".format(
            layers_t, LayerInfo_t
        )
        headers.extend(contents)
        headers.append(content)
        content = "u32 weight_total_size={};\n".format(self.w_offset["w_offset"])
        headers.append(content)
        content = "u32 in_total_size={}*{}*{};\n".format(
            in_data_align, in_data_shape[0], in_data_shape[1] # type: ignore
        )
        headers.append(content)
        content = "u32 layer_seq[]={"
        headers.append(content)

        for layer in self.layers:
            layer_id = layer.get_idx()
            layer_type = layer.get_layer_type()
            if layer_id == len(self.layers) - 1:
                content = "{}{};\n".format(str(layer_id), "}")
                headers.append(content)
                content = "i32 layer_seq_len={};\n".format(len(self.layers))
                headers.append(content)
            else:
                content = "{}, ".format(str(layer_id))
                headers.append(content)

        #### result layers, ref_cnt
        result_layer_ids, result_scales, result_chn, ref_cnt = [], [], [], []
        for layer in self.layers:
            ref_cnt.append(max(len(layer.get_output_idx()), 0))
            is_result_layer = True
            for out_idx in layer.get_output_idx():
                if out_idx > 0:
                    is_result_layer = False
            if layer.get_is_result_layer() and is_result_layer:
                layer_id = layer.get_idx()
                layer_scale_type = layer.get_scale_type()
                if layer_scale_type in ["rshiftscale", "rrshiftscale"]:
                    scales = copy.deepcopy(layer.get_scales())
                    if isinstance(scales, list):
                        scales = scales[0]
                    scale = scales["fscale"]
                    if isinstance(scale, np.ndarray):
                        scale = 1.0
                elif layer_scale_type in ["intscale"]:
                    scale = 1.0
                elif layer_scale_type in ["ffloatscale", "float", "smooth"]:
                    scale = 1.0
                else:
                    raise Exception(
                        "Not supported : {} in last layer".format(layer_scale_type)
                    )
                # if layer.get_layer_type() in ["lstm", "gru", "reshape", "cmul", "pmul", "mul"]:
                #     out_data = layer.get_out_data()
                #     if isinstance(out_data, list):
                #         chn = out_data[0]["output"].shape[1]
                #     else:
                #         chn = out_data["output"].shape[1]
                # else:
                #     if layer.get_ops_setting()["attrs"][0] == dict():
                #         chn = layer.get_out_data()['output'].shape[1]
                #     else:
                #         chn = layer.get_ops_setting()["attrs"][0]["out_c"]
                        
                if layer.get_ops_setting()["attrs"][0] == dict() or layer.get_ops_setting()["attrs"][0] == dict():
                    out_data = layer.get_out_data()
                    if isinstance(out_data, list):
                        chn = out_data[0]["output"].shape[1]
                    else:
                        chn = out_data["output"].shape[1]
                else:
                    chn = layer.get_ops_setting()["attrs"][0]["out_c"]
                    
                result_layer_ids.append(layer_id)
                result_scales.append(scale)
                result_chn.append(chn)
                self.logger.info("fscale: {}, {}".format(layer.get_layer_name(), scale))

        content = "u32 insert_list[]={};\n"
        headers.append(content)
        content = "i32 insert_list_len=0;\n"
        headers.append(content)
        content = "i32 ref_cnt[]=%s;\n" % str(ref_cnt).replace("[", "{").replace(
            "]", "}"
        )
        headers.append(content)
        content = "i32 result_layers[]=%s;\n" % str(result_layer_ids).replace(
            "[", "{"
        ).replace("]", "}")
        headers.append(content)
        content = "float result_scales[]=%s;\n" % str(result_scales).replace(
            "[", "{"
        ).replace("]", "}")
        headers.append(content)
        content = "i32 result_chn[]=%s;\n" % str(result_chn).replace("[", "{").replace(
            "]", "}"
        )
        headers.append(content)
        content = "i32 result_layers_len={};\n".format(len(result_layer_ids))
        headers.append(content)
        content = "float input_scales[]=%s;\n" % str(input_scales).replace(
            "[", "{"
        ).replace("]", "}")
        headers.append(content)
        content = 'char version[] = "{}";'.format(get_version())
        headers.append(content)

        if self.export_mode_c or self.is_debug:
            model_c_path = os.path.join(self.log_dir, "model.c")
            with open(model_c_path, "w") as file:
                for content in headers:
                    file.write(content)
            self.logger.info("write model.c done!")
        else:
            model_b_path = os.path.join(self.log_dir, "model.b")
            with open(model_b_path, "wb") as file:
                for content in contents:
                    self.write_model_b(content, file)
            self.logger.info("write model.b done!")
            # encrypt.encode(model_b_path, model_b_path, self.secret_key)
            # self.logger.info("encrypt model.b done!")

    def export(self):
        pass

    def visualization(self):
        def rename_layer(name):
            name = name.replace("-", "_")
            name = name.replace(":", "_")
            name = name.replace("/", "_")
            name = name.replace(".", "_")
            return name

        def write_connetion(f, layer_name, layer, output_idx=0):
            for idx in layer.get_output_idx():
                if idx == -1:
                    name = "output{}".format(output_idx)
                    output_idx = output_idx + 1
                else:
                    name = self.layers[idx].get_layer_name()

                name = rename_layer(name)
                layer_name = rename_layer(layer_name)

                if "split_ids" in layer.get_insert().keys():
                    split_ids = layer.get_insert()["split_ids"]
                    f.write(layer_name + " --|> " + name + ": " + str(split_ids[idx]))
                else:
                    f.write(layer_name + " --|> " + name)
                f.write("\n")

            return output_idx

        def write_attrs(f, name, attrs):
            for k, v in attrs.items():
                name = rename_layer(name)
                f.write(name + ": " + k + " ")
                for attr in v:
                    if not isinstance(attr, str):
                        attr = str(attr)
                    f.write(attr + " ")
                f.write("\n")

        def get_feat(layer, mode="feat_i"):
            if "split" in layer.get_insert().keys():
                if mode in layer.get_insert()["split"].keys():
                    feat_i = layer.get_insert()["split"][mode]
                else:
                    feat_i = [[-1, -1]]
            else:
                if mode in layer.get_insert().keys():
                    feat_i = layer.get_insert()[mode]
                else:
                    feat_i = [[-1, -1]]

            return feat_i

        def get_align(layer, mode="out_align"):
            if "split" in layer.get_insert().keys():
                if mode in layer.get_insert()["split"].keys():
                    oc = layer.get_insert()["split"][mode]
                else:
                    oc = [-1]
            else:
                if mode in layer.get_insert().keys():
                    oc = layer.get_insert()[mode]
                else:
                    oc = [-1]

            return oc

        def get_pad(layer, mode="out_pad"):
            if "split" in layer.get_insert().keys():
                if mode in layer.get_insert()["split"].keys():
                    pad = layer.get_insert()["split"][mode]
                else:
                    pad = [[-1, -1]]
            else:
                if mode in layer.get_insert().keys():
                    pad = layer.get_insert()[mode]
                else:
                    pad = [[-1, -1]]

            return pad

        output_idx = 1

        vis_file = "work_dir/test_vis.mmd"
        vis_path, _ = os.path.split(vis_file)
        if not os.path.exists(vis_path):
            os.makedirs(vis_path, mode=0o777, exist_ok=True)

        with open(vis_file, "w") as f:
            f.write("classDiagram" + "\n")
            data_layer_idx = 0
            for layer_idx, layer in enumerate(self.layers):
                layer_type = layer.get_layer_type()
                # if layer_type == 'data':
                #     layer_name = layer.get_layer_name()
                # else:

                layer_name = layer.get_layer_name()

                # if layer.get_layer_name() == 'Conv_996':
                # break
                # if layer.get_layer_type() == 'shuffle':
                # print('test')

                feat_i = get_feat(layer, mode="feat_i")
                feat_o = get_feat(layer, mode="feat_o")
                ic = get_align(layer, mode="in_align")
                oc = get_align(layer, mode="out_align")
                in_pad = get_pad(layer, mode="in_pad")
                out_pad = get_pad(layer, mode="out_pad")
                is_align = layer.get_insert()["is_align"]

                attrs = dict(
                    layer_type=[layer.get_layer_type()],
                    layer_idx=[layer.get_idx()],
                    feat_i=feat_i,
                    feat_o=feat_o,
                    in_pad=in_pad,
                    out_pad=out_pad,
                    ic=[ic],
                    oc=[oc],
                    is_align=[is_align],
                )
                output_idx = write_connetion(
                    f, layer_name, layer, output_idx=output_idx
                )
                write_attrs(f, layer_name, attrs)

        if self.is_debug:
            self.logger.info("start export visualization")
            json_content = dict(maxTextSize=99999999)
            with open(os.path.join(vis_path, "mermaidRenderConfig.json"), "w") as f:
                json.dump(json_content, f, indent=4, ensure_ascii=False)
            json_content = dict(args=["--no-sandbox"])
            with open(os.path.join(vis_path, "puppeteer-config.json"), "w") as f:
                json.dump(json_content, f, indent=4, ensure_ascii=False)
            os.system(
                "bash benchmark/scripts/export_visualization.sh {}".format(vis_path)
            )
            self.logger.info("finish export visualization")


class mExportV1(mExportBase):
    def __init__(self, **kwargs):
        super(mExportV1, self).__init__(**kwargs)

        self.model_template = ['#include "layer.h"\n', "\n", "Layer_info layers[]={"]

        for key in self.layer_map.keys():
            if key in self.ignore_layers:
                continue
            for m in self.layer_map[key]:
                setattr(self, "network_{}".format(m), rt_factory.get(m)(**kwargs)) # type: ignore
        kwargs["version"] = "v1"
        self.init_wexport(**kwargs)

    def export(self):
        for layer in self.layers:
            try:
                self.set_bias_data(layer)
                self.set_layer_datatype(layer)
                self.set_voice_model_feats(layer)

                layer_id = layer.get_idx()
                layer_type = layer.get_layer_type()

                if layer_type in ["data"]:
                    self.process_data(layer)
                elif layer_type in ["conv", "depthwiseconv"]:
                    is_first_conv = self.check_first_conv(layer)
                    layer.set_first_conv(is_first_conv)

                    self.process_conv_without_concat(layer)
                    self.process_conv_without_split(layer)
                elif layer_type in ["concat"]:
                    is_exist = False
                    for id in layer.get_output_idx():
                        is_exist = self.find_elementwise_layer(
                            self.layers[id], is_exist=is_exist
                        )
                    if is_exist:
                        self.process_concat_with_elementwise(layer)
                    else:
                        self.process_concat(layer)
                elif layer_type in ["split"]:
                    self.process_split(layer)
                elif layer_type in ["fc"]:
                    self.process_fc(layer)
                elif layer_type in ["shuffle"]:
                    self.process_shuffle(layer)
                elif layer_type in ["shuffle_only"]:
                    self.process_shuffle_only(layer)
                elif layer_type in ["batchnormalization"]:
                    self.process_bn_layer(layer)
                else:  # if layer_type in ['averagepool']:
                    self.process(layer)
                # else:
                # self.process(layer)
                if self.debug:
                    self.logger.info(
                        "-------------------------------------------------------------------------"
                    )
                    self.logger.info(
                        "layer index is: {}/{}, input index is:{}, output index is: {}".format(
                            layer_id,
                            len(self.layers),
                            layer.get_input_idx(),
                            layer.get_output_idx(),
                        )
                    )
                    self.logger.info(
                        "layer name is: {}, layer type is: {}, export parameter is: ".format(
                            layer.get_layer_name(), layer_type
                        )
                    )
                    for k, v in sorted(
                        layer.get_insert().items(), key=lambda d: d[0], reverse=True
                    ):
                        if k in ["split", "concat"]:
                            self.logger.info("{}: ".format(k))
                            for k_, v_ in v.items():
                                self.logger.info("    {}: {}".format(k_, v_))
                        else:
                            self.logger.info("{}: {}".format(k, v))
                    self.logger.info(
                        "-------------------------------------------------------------------------"
                    )
            except:
                self.logger.error("{} export error!".format(layer.get_layer_name())) # type: ignore
                os._exit(-1)
        self.logger.info("export context done!")
