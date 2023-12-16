# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : TIMESINETLLI TECH
# @Time     : 2022/7/15 11:30
# @File     : base_export.py
import struct
import sys  # NOQA: E402

sys.path.append('./') # NOQA: E402

try:
    from __init_path import root_dir  # NOQA: E402 # type: ignore
except:
    from onnx_converter.__init_path import root_dir # NOQA: E402 # type: ignore

import copy
import json
import os
import re

# import encrypt
import numpy as np

try:
    from utils import Object, export_perchannel
except:
    from onnx_converter.utils import Object, export_perchannel # type: ignore

from .network import NETWORK as rt_factory
from .v1.serialize import SERIALIZE as serialize_factory
from .v1.serialize import writeFile


def _write(content, val, tail=','):
    return content + str(val) + tail


def float_to_hex(f):
    '''convert from the float value to the string of its hex value
        for example, 0.1234 is converted to "0x3dfcb924"
    '''
    return hex(struct.unpack('<i', struct.pack('<f', f))[0])


class mExportBase(Object): # type: ignore

    def __init__(self, **kwargs):
        super(mExportBase, self).__init__()

        self.model_template = None
        self.valid_export_layer = None
        self.Ksize = None
        self.O_Align = None
        self.Csize = None
        self.I_Align = None
        self.ABGR = None
        self.data_channel_extension = None
        self.secret_key = kwargs['secret_key']
        self.export_mode_c = kwargs['export_mode_c']
        self.is_debug = kwargs['is_debug']
        self.is_stdout = kwargs['is_stdout']
        self.log_name = kwargs["log_name"] if "log_name" in kwargs.keys(
        ) else "export.log"
        self.log_level = kwargs.get('log_level', 20)
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)
        self.weights_dir = os.path.join(self.log_dir, "weights")
        self.logger.info("rm -rf {}".format(self.weights_dir))
        os.system("rm -rf {}".format(self.weights_dir))
        os.makedirs(self.weights_dir, mode=0o777, exist_ok=True)

        if 'quan_graph' in kwargs.keys():
            self.quan_graph = kwargs['quan_graph']
            self.layers = self.quan_graph.get_layers()

        self.is_voice_model = False
        self.save_weights = []
        self.save_indata = []
        self.save_feats = {}
        self.save_model = {}
        self.w_offset = dict(w_offset=0, tmp_offset=[])
        self.debug = 0
        if self.debug:
            self.test_conv_idx = 0
            self.test_feat_idx = 0

    def get_graph(self):
        return self.quan_graph

    def set_graph(self, quan_graph):
        self.quan_graph = quan_graph
        self.layers = self.quan_graph.get_layers()

    @staticmethod  ### used when test export conv param and feature after conv
    def save_qdata(data, file_name, is_float=False):
        with open(file_name, 'w', encoding='utf-8') as f:
            for data_ in data:
                if is_float:
                    f.write('%.4f' % (data_) + '\n')
                else:
                    f.write(str(data_) + '\n')

    @staticmethod
    def calc_w_offset(weight):
        b_size = 4 if isinstance(weight[0], np.float32) else 1 # type: ignore
        w_offset = len(weight) * b_size
        return w_offset

    @staticmethod
    def get_align_channel(ch, align_size):
        return ((ch + align_size - 1) // align_size) * align_size

    @staticmethod
    def set_last_layer_fpdata(layer):
        ### set dequant data for last layer
        global fp_result
        outputs = layer.get_out_data()
        ops = layer.get_ops_instance()
        ops_name = layer.get_layer_ops()['ops']

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
            if op_name in ['act']:
                scales = op.get_scales()  # post-shift in postprocess
                if isinstance(scales, list):
                    fp_result = list()
                    for s_idx, scale in enumerate(scales):
                        output = copy.deepcopy(outputs[op_idx][s_idx]['output'])
                        output = dequant_scale(output, scale, so[all_idx])
                        output = output.astype(np.float32)
                        fp_result.append(dict(output=output))
                else:
                    output = copy.deepcopy(outputs[op_idx]['output'])
                    output = dequant_scale(output, scales, so[all_idx])
                    output = output.astype(np.float32)
                    fp_result = dict(output=output)

        outputs.append(fp_result)
        layer.set_out_data(outputs)

    @staticmethod
    def set_bias_data(layer):
        layer_type = layer.get_layer_type()
        if layer_type in ['conv', 'depthwiseconv', 'fc', 'matmul']:
            has_bias = layer.get_layer_ops()['attrs'][0]['bias']
            if has_bias:
                qbias = layer.get_qbias()
                bit_num = re.findall(r'\d+', qbias.dtype.type.__name__)[0]
                qbias = eval('np.int' + bit_num)(qbias)
                layer.set_qbias(qbias)
        elif layer_type in ['lstm']:
            qbiass = []
            for qbias in layer.get_qbias():
                bit_num = re.findall(r'\d+', qbias.dtype.type.__name__)[0]
                qbias = eval('np.int' + bit_num)(qbias)
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
        layer_type = layer.get_layer_type()
        if is_voice_model:
            self.is_voice_model = is_voice_model
        else:
            if layer_type == 'data' and not self.is_voice_model:
                if len(layer.get_in_data().shape) in [2, 3]:
                    self.is_voice_model = True

        if self.is_voice_model:

            in_data = layer.get_in_data()
            in_datatype = layer.get_datatype(copy.deepcopy(in_data))
            layer.set_input_type(in_datatype)
            if layer.get_layer_type() == 'data':
                in_data = np.squeeze(in_data)
                in_data = np.expand_dims(in_data, axis=[0, 2, 3])
                layer.set_in_data(in_data)
            elif isinstance(in_data, list):
                in_datas = copy.deepcopy(in_data)
                in_datas_ = []
                for in_data in in_datas:
                    in_data = np.squeeze(in_data['output'])
                    in_data = np.expand_dims(in_data, axis=[0, 2, 3])
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
                    out_data = np.squeeze(out_data['output'])
                    out_data = np.expand_dims(out_data, axis=[0, 2, 3])
                    out_datas_.append(dict(output=out_data))
                layer.set_out_data(out_datas_)
            else:
                out_data = np.squeeze(out_data['output'])
                out_data = np.expand_dims(out_data, axis=[0, 2, 3])
                out_data_ = layer.get_out_data()
                out_data_.update(dict(output=out_data))
                layer.set_out_data(out_data_)
            # print('test')

    @staticmethod
    def get_feature_shape(layer):
        if len(layer.get_in_data()[0]['output'].shape) == 4:
            feat_i = [
                layer.get_in_data()[0]['output'].shape[2],
                layer.get_in_data()[0]['output'].shape[3]
            ]
        elif len(layer.get_in_data()[0]['output'].shape) == 2:
            feat_i = [1, 1]
        else:
            raise Exception('get_feature_shape maybe incorrect!!!')

        if isinstance(layer.get_out_data(), list):
            if len(layer.get_out_data()[-1]['output'].shape) == 4:
                feat_o = [
                    layer.get_out_data()[-1]['output'].shape[2],
                    layer.get_out_data()[-1]['output'].shape[3]
                ]
            elif len(layer.get_out_data()[-1]['output'].shape) == 2:
                feat_o = [1, 1]
            else:
                raise Exception('get_feature_shape maybe incorrect!!!')
        elif isinstance(layer.get_out_data(), dict):
            if len(layer.get_out_data()['output'].shape) == 4:
                feat_o = [
                    layer.get_out_data()['output'].shape[2],
                    layer.get_out_data()['output'].shape[3]
                ]
            elif len(layer.get_out_data()['output'].shape) == 2:
                feat_o = [1, 1]
            else:
                raise Exception('get_feature_shape maybe incorrect!!!')
        else:
            raise Exception('get_feature_shape maybe incorrect!!!')

        return feat_i, feat_o

    def get_pad_align(self, layer, layer_id):
        if 'split' in self.layers[layer_id].get_insert().keys():
            split_id = self.layers[layer_id].get_insert()['split_ids'][layer.get_idx()]
            out_pad_ = self.layers[layer_id].get_insert()['split']["out_pad"]
            out_align_ = self.layers[layer_id].get_insert()['split']["out_align"]
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
        name_ = [set(self.layers[id].get_onnx_input_name()) for id in layer.get_output_idx()]
        name_inters = [set(layer.get_onnx_output_name()).intersection(key) for key in name_]

        id = -1
        tmp_name = []
        split_ids = {}
        for out_idx, name in zip(layer.get_output_idx(), name_inters):
            if name not in tmp_name:
                tmp_name.append(name)
                id = id + 1
            split_ids[out_idx] = id

        return split_ids

    def recursive_down_layer(self, layer, result, result_id, split_id=0, mode='split'):
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
        if layer.get_layer_type() in ["split"] and mode == 'split':
            res = layer.get_ops_setting()['attrs'][0]['split']
            res_id = [layer.get_idx()]
            result.append(res)
            result_id.append(res_id)
            ### process depthwiseconv after split
            # for split_id, id in enumerate(layer.get_output_idx()):
            # result, result_id = self.recursive_down_layer(self.layers[id], result, result_id, split_id=split_id)
        elif layer.get_layer_type() in ['add', 'sub', 'mul', 'pmul',
                                        'cmul'] and mode == 'conv_parallel_concat_elementwise':
            ### recursive top layer
            for id in layer.get_input_idx():
                result, result_id = self.recursive_top_layer(
                    self.layers[id], result, result_id)
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
            if layer.get_layer_type() in ['conv', 'fc', 'concat', 'shuffle', 'shuffle_only']:
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
                        self.layers[id], result, result_id, mode=mode)
                # if self.layers[id].get_layer_type() in ['conv', 'depthwiseconv', 'split']:
                #     break
                # print(result, result_id)

        return result, result_id

    def recursive_top_layer(self, layer, result, result_id, mode='concat'):
        """
        It recursively finds the top layer of the network, and returns the number of channels of the top
        layer

        :param layer: the layer to be processed
        :param result: the number of channels of the output of the layer
        :param result_id: the index of the layer that needs to be split
        :param mode: 'concat' or 'split', defaults to concat (optional)
        :return: The result is a list of the number of channels of the input layers of the top layer.
        """
        if layer.get_layer_type() in ["concat"] and mode == 'concat':
            for id in layer.get_input_idx():
                res = self.layers[id].get_out_data()[-1]['output'].shape[1]
                # res_id = [self.layers[id].get_idx()]
                res_id = layer.get_idx()
                result.append(res)
                result_id.append(res_id)
        elif layer.get_layer_type() in ["split"] and mode == 'split':
            res = layer.get_ops_setting()['attrs'][0]['split']
            res_id = [layer.get_idx()]
            result.append(res)
            result_id.append(res_id)
        else:
            if layer.get_layer_type() in ['conv', 'fc', 'shuffle', 'shuffle_only']:
                return result, result_id
            elif layer.get_layer_type() in ['data']:
                result.append(layer.get_in_data().shape[1])
                result_id.append(layer.get_idx())
                return result, result_id
            else:
                for id in layer.get_input_idx():
                    result, result_id = self.recursive_top_layer(
                        self.layers[id], result, result_id, mode=mode)
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
        if layer.get_layer_type() in ['add', 'sub', 'mul', 'cmul', 'pmul']:
            for id in input_idx:
                for mode in ['concat', 'split']:
                    result, result_id = [], []
                    result, result_id = self.recursive_top_layer(
                        self.layers[id], result, result_id, mode=mode)
                    if len(result) > 0:
                        ### one input of element_wise layer is not (concat and split)
                        is_exist = True
                        break

            if not is_exist:
                mode = 'conv_parallel_concat_elementwise'
                result, result_id = [], []
                for id in layer.get_output_idx():
                    result, result_id = self.recursive_down_layer(
                        self.layers[id], result, result_id, mode=mode)
                if len(result) > 0:
                    is_exist = True
        else:
            if layer.get_layer_type() in ['conv', 'fc', 'concat', 'shuffle', 'shuffle_only']:
                return is_exist
            else:
                for id in layer.get_output_idx():
                    if self.layers[id].get_layer_type() in ['add', 'sub', 'mul', 'cmul', 'pmul']:
                        input_idx = copy.deepcopy(self.layers[id].get_input_idx())
                        if layer.get_idx() in input_idx:
                            input_idx.remove(layer.get_idx())
                    is_exist = self.find_elementwise_layer(
                        self.layers[id], is_exist=is_exist, input_idx=input_idx)
        return is_exist

    def process_data(self, layer):
        channel = layer.get_in_data().shape[1]  # layer.get_layer_ops()["attrs"][0]["shape"][1]
        if self.data_channel_extension:
            align_size = self.get_align_channel(channel, 4)
            if self.ABGR:
                out_pad, out_align = [[1, 4]], [align_size]
            else:
                out_pad, out_align = [[0, 3]], [align_size]
        else:
            Csizes = []
            for out_idx in layer.get_output_idx():
                if self.layers[out_idx].get_layer_type() in ['fc', 'lstm']:
                    Csizes.append(self.I_Align)
                else:
                    Csizes.append(self.Csize)
            Csize = np.max(Csizes)
            align_size = self.get_align_channel(channel, Csize)
            out_pad, out_align = [[0, channel]], [align_size]

        if len(layer.get_out_data()['output'].shape) == 4:
            feat_o = [
                layer.get_out_data()['output'].shape[2],
                layer.get_out_data()['output'].shape[3]
            ]
        elif len(layer.get_out_data()['output'].shape) == 2:
            feat_o = [[1, 1]]
        else:
            raise Exception('get_feature_shape maybe incorrect!!!')

        res = {"out_pad": out_pad, "out_align": out_align,
               "feat_o": [feat_o], "feat_i": [feat_o]}
        layer.set_insert(res)

    def set_root_fd(self, root_fd):
        self.root_fd = root_fd
        if self.root_fd:
            self.weights_dir = os.path.join(self.weights_dir, root_fd)
        else:
            self.weights_dir = self.log_dir

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
            if 'split' in self.layers[layer_id].get_insert().keys():
                split_id = self.layers[layer_id].get_insert()['split_ids'][layer.get_idx()]
                in_pad = [self.layers[layer_id].get_insert()['split']["out_pad"][split_id]]
                in_align = [self.layers[layer_id].get_insert()['split']["out_align"][split_id]]
                in_pad = [[0, in_pad[0][1] - in_pad[0][0]]]
            else:
                in_pad = self.layers[layer_id].get_insert()["out_pad"]
                if len(in_pad) == 1:
                    if layer.get_layer_type() in ['fc']:
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
            "feat_o": [feat_o]
        }
        layer.set_insert(res)

    # conv output
    def process_conv_without_split(self, layer):
        # if layer.get_layer_name() == 'Conv_92':
        #     print('test')

        layer_type = layer.get_layer_type()
        if layer_type in ['fc']:
            align_size = self.O_Align
        elif layer_type == "depthwiseconv":
            align_size = self.Csize
        else:
            align_size = self.Ksize

        out_c = layer.get_layer_ops()["attrs"][0]["out_c"]
        if layer_type == "depthwiseconv":
            layer_id = layer.get_input_idx()[0]
            _, out_align = self.get_pad_align(layer, layer_id)
            res_conv = {"out_pad": [[0, out_c]], "out_align": out_align, "in_align": out_align}
        else:
            out_align = [self.get_align_channel(out_c, align_size)]
            res_conv = {"out_pad": [[0, out_c]], "out_align": out_align}
        feat_i, feat_o = self.get_feature_shape(layer)
        res = {'feat_i': [feat_i], 'feat_o': [feat_o]}
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
            'feat_i': [feat_i],
            'feat_o': [feat_o]
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

        out_align_ = [
            int(np.sum(out_align[:i])) for i in range(len(out_align))
        ]
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
            list(np.array(pad) + align)
            for pad, align in zip(out_pad, out_align_)
        ]

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": [np.sum(out_align)],
            'feat_i': [feat_i],
            'feat_o': [feat_o]
        }
        layer.set_insert(res)

    def process_split(self, layer):
        layer_id = layer.get_input_idx()[0]
        if self.layers[layer_id].get_layer_type() in ['conv', 'depthwiseconv']:
            out_pad = [[0, pad[1] - pad[0]]
                       for pad in self.layers[layer_id].get_insert()["out_pad"]]
            out_align = [
                self.get_align_channel(pad[1], self.Csize) for pad in out_pad
            ]
        else:
            result = layer.get_ops_setting()['attrs'][0]['split']

            # if layer.get_layer_name() == 'Split_cvt_5':
            # print('test')

            out_align_ = [a for a, b in self.layers[layer_id].get_insert()["out_pad"][1:]]
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

        res = {"out_pad": out_pad, "out_align": out_align, 'feat_i': [feat_i], 'feat_o': [feat_o]}

        res['in_pad'] = self.layers[layer_id].get_insert()['out_pad']
        # in_align = 0
        # for a, b in res['in_pad']:
        #     in_align += self.get_align_channel(b - a, self.Csize)
        # res['in_align'] = [in_align]
        res['in_align'] = self.layers[layer_id].get_insert()['out_align']

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
            next_in_c = self.layers[next_layer_id].get_layer_ops(
            )["attrs"][0]["in_c"]
            next_in_align = [self.get_align_channel(next_in_c, self.I_Align)]
            if out_align[0] < next_in_align[0]:
                out_align = next_in_align

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": out_align
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
        hidden_size = layer.get_ops_setting()['attrs'][0]['hidden_size']
        for ch in [in_c, hidden_size]:
            in_pad.append([0, ch])
            in_align.append(self.get_align_channel(ch, self.I_Align))

        res_concat = {"in_pad": in_pad, "in_align": in_align}

        out_c1, out_c2 = layer.get_qweight()[0].shape[1], layer.get_qweight()[1].shape[1]
        result = [hidden_size, out_c1, out_c2]
        out_align = [self.get_align_channel(ch, self.O_Align) for ch in result]

        out_pad = [[0, ch] for ch in result]
        res_split = {"out_pad": out_pad, "out_align": out_align}

        feat_i, feat_o = self.get_feature_shape(layer)

        res = dict(split=res_split, feat_i=[feat_i], feat_o=[feat_o])
        res['split'].update(res_concat) # type: ignore
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
        result = layer.get_layer_ops()["attrs"][-1]['split']
        out_align = [self.get_align_channel(ch, self.Csize) for ch in result]

        out_pad = [[0, ch] for ch in result]
        res_split = {"out_pad": out_pad, "out_align": out_align}

        feat_i, feat_o = self.get_feature_shape(layer)

        # res = dict(concat=res_concat, split=res_split, feat_i=[feat_i], feat_o=[feat_o])
        res = dict(split=res_split, feat_i=[feat_i], feat_o=[feat_o])
        res['split'].update(res_concat) # type: ignore
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
            in_align = [self.layers[layer_id].get_insert()["split"]["out_align"][split_id]]
        else:
            in_pad = self.layers[layer_id].get_insert()["out_pad"]
            in_align = self.layers[layer_id].get_insert()["out_align"]

        out_c = layer.get_layer_ops()["attrs"][0]["shape"][
                    1] * layer.get_layer_ops()["attrs"][0]["shape"][2]
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
            'feat_i': [feat_i],
            'feat_o': [feat_o]
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
            raise Exception('process_elementwise_layer is error !!!')

        out_pad, out_align = in_pad, in_align

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": out_align,
            'feat_i': [feat_i],
            'feat_o': [feat_o]
        }
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
            'feat_i': [feat_i],
            'feat_o': [feat_o]
        }
        if self.layers[layer.get_input_idx()[0]].get_layer_type() in ['conv', 'depthwiseconv']:
            res.update(dict(fmt='CubeFmt'))
        else:
            res.update(dict(fmt='MatFmt'))
        layer.set_insert(res)

    # def process(self, layer):
    #     pass

    def check_first_conv(self, layer):
        in_idx = layer.get_input_idx()
        if len(in_idx) > 1:
            flag = False
        else:
            if "data" == self.layers[in_idx[0]].get_layer_type():
                flag = True
            else:
                flag = False
        return flag

    def export_conv_weights(self, layer):
        layer.set_w_offset(copy.deepcopy(self.w_offset))

        # export conv weights and bias into weight.b
        layer_type = layer.get_layer_type()
        first_conv = layer.get_first_conv()  # self.check_first_conv(layer)
        func_weight = getattr(self, "serialize_{}".format(layer_type))
        func_bias = getattr(self, "serialize_{}".format("bias"))
        res = layer.weight_export(
            func_weight,
            layer.get_qweight(),
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "weight.b"),
            first_conv=first_conv,
            data_channel_extension=self.data_channel_extension)
        self.save_weights.extend(res)
        self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

        ### test export conv param
        # if self.debug:
        #     if layer_type in ['conv', 'depthwiseconv']:
        #         self.save_qdata(res, './work_dir/conv_quant/{}_{}.txt'.format(
        #             self.test_conv_idx, layer.get_layer_name()), is_float=False)
        #         fdata = layer.get_weight().transpose(2, 3, 1, 0).reshape(-1)
        #         self.save_qdata(fdata, './work_dir/conv_float/{}_{}.txt'.format(
        #             self.test_conv_idx, layer.get_layer_name()), is_float=True)
        #         self.test_conv_idx = self.test_conv_idx + 1

        if layer.get_layer_ops()["attrs"][0]["bias"]:
            res = layer.bias_export(func_bias,
                                    layer.get_qbias(),
                                    is_fc_bias=False,
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

        self.w_offset, self.save_weights = \
            export_perchannel(layer, self.weights_dir, func_bias,
                              self.save_weights, self.w_offset, self.calc_w_offset, is_fc_bias=False)

    def export_bn_weights(self, layer):
        layer.set_w_offset(copy.deepcopy(self.w_offset))

        layer_type = layer.get_layer_type()
        weights = layer.get_layer_ops()['attrs'][0]
        func_bn = getattr(self, "serialize_{}".format("bias"))
        res = layer.bias_export(func_bn,
                                weights['scale'],
                                is_fc_bias=False,
                                layer_name=layer.get_layer_name(),
                                name=os.path.join(self.weights_dir,
                                                  "weight.b"))
        self.save_weights.extend(res)
        self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
        res = layer.bias_export(func_bn,
                                weights['bias'],
                                is_fc_bias=False,
                                layer_name=layer.get_layer_name(),
                                name=os.path.join(self.weights_dir,
                                                  "weight.b"))
        self.save_weights.extend(res)
        self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
        res = layer.bias_export(func_bn,
                                weights['mean'],
                                is_fc_bias=False,
                                layer_name=layer.get_layer_name(),
                                name=os.path.join(self.weights_dir,
                                                  "weight.b"))
        self.save_weights.extend(res)
        self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
        res = layer.bias_export(func_bn,
                                weights['var'],
                                is_fc_bias=False,
                                layer_name=layer.get_layer_name(),
                                name=os.path.join(self.weights_dir,
                                                  "weight.b"))
        self.save_weights.extend(res)
        self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

    def export_ln_weights(self, layer):
        layer.set_w_offset(copy.deepcopy(self.w_offset))

        layer_type = layer.get_layer_type()
        weights = layer.get_layer_ops()['attrs'][0]
        if self.is_voice_model:
            func_bn = getattr(self, "serialize_{}".format("bias"))
            qscale = np.squeeze(weights['scale'])
            qbias = np.squeeze(weights['bias'])
            res = layer.bias_export(func_bn,
                                    qscale,
                                    is_fc_bias=False,
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
            res = layer.bias_export(func_bn,
                                    qbias,
                                    is_fc_bias=False,
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
        else:
            func_data = getattr(self, "serialize_{}".format("data"))
            qscale = np.expand_dims(weights['scale'], axis=0)
            qbias = np.expand_dims(weights['bias'], axis=0)
            res = layer.feat_export(func_data,
                                    qscale,
                                    is_out=True,
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
            res = layer.feat_export(func_data,
                                    qbias,
                                    is_out=True,
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

    def export_fc_weights(self, layer):
        layer.set_w_offset(copy.deepcopy(self.w_offset))

        # export fc weights and bias into weight.b
        layer_type = layer.get_layer_type()
        func_weight = getattr(self, "serialize_{}".format(layer_type))
        func_bias = getattr(self, "serialize_{}".format("bias"))
        res = layer.weight_export(func_weight,
                                  layer.get_qweight(),
                                  layer_name=layer.get_layer_name(),
                                  name=os.path.join(self.weights_dir,
                                                    "weight.b"))
        self.save_weights.extend(res)
        self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

        if layer.get_ops_setting()['attrs'][0]['bias']:
            res = layer.bias_export(func_bias,
                                    layer.get_qbias(),
                                    is_fc_bias=True,
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

        self.w_offset, self.save_weights = \
            export_perchannel(layer, self.weights_dir, func_bias,
                              self.save_weights, self.w_offset, self.calc_w_offset, is_fc_bias=True)

    def export_lstm_weights(self, layer):
        layer.set_w_offset(copy.deepcopy(self.w_offset))
        tmp_offset = []

        layer_type = layer.get_layer_type()
        func_table = getattr(self, "serialize_{}".format("table"))
        func_weight = getattr(self, "serialize_{}".format("fc"))
        func_bias = getattr(self, "serialize_{}".format("bias"))
        for i in range(6):
            tables = layer.get_table()
            if i < len(tables):
                table = tables[i]
                res = layer.weight_export(func_table,
                                          table,
                                          layer_name=layer.get_layer_name(),
                                          name=os.path.join(self.weights_dir,
                                                            "weight.b"))
                self.save_weights.extend(res)
                tmp_offset.append(self.w_offset['w_offset'])
                self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
            else:
                tmp_offset.append(-1)

        for idx, (qweight, qbias) in enumerate(zip(layer.get_qweight(), layer.get_qbias())):
            insert = {key: [layer.get_insert()['split'][key][idx + 1]]
                      for key in ['in_pad', 'in_align', 'out_pad', 'out_align']}
            res = layer.weight_export(func_weight,
                                      np.squeeze(qweight),
                                      insert=insert,
                                      layer_name=layer.get_layer_name(),
                                      name=os.path.join(self.weights_dir,
                                                        "weight.b"))
            self.save_weights.extend(res)
            tmp_offset.append(self.w_offset['w_offset'])
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

            res = layer.bias_export(func_bias,
                                    np.squeeze(qbias),
                                    is_fc_bias=True,
                                    insert=insert,
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

            ### write wb_off, rb_off
        tmp_offset.append(-1)
        tmp_offset.append(-1)

        ### write init_h
        init_h = layer.get_init_h()
        if np.sum(np.abs(init_h)) > 0:
            insert = {key1: [layer.get_insert()['split'][key][2]] \
                      for key1, key in zip(['out_pad', 'out_align'], ['in_pad', 'in_align'])}
            res = layer.bias_export(func_bias,
                                    np.squeeze(init_h),
                                    is_fc_bias=True,
                                    insert=insert,
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            tmp_offset.append(self.w_offset['w_offset'])
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
        else:
            tmp_offset.append(-1)

        ### write init_c
        init_c = layer.get_init_c()
        if np.sum(np.abs(init_c)) > 0:
            insert = {key1: [layer.get_insert()['split'][key][2]] \
                      for key1, key in zip(['out_pad', 'out_align'], ['in_pad', 'in_align'])}
            res = layer.bias_export(func_bias,
                                    np.squeeze(init_c),
                                    is_fc_bias=True,
                                    insert=insert,
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "weight.b"))
            self.save_weights.extend(res)
            tmp_offset.append(self.w_offset['w_offset'])
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore
        else:
            tmp_offset.append(-1)

        w_offset = layer.get_w_offset()
        w_offset['tmp_offset'] = tmp_offset
        layer.set_w_offset(copy.deepcopy(w_offset))

        # print('test')
        # self.w_offset, self.save_weights = \
        #     export_perchannel(layer, self.weights_dir, func_bias, \
        #         self.save_weights, self.w_offset, self.calc_w_offset, is_fc_bias=True)

    def export_table(self, layer):
        layer.set_w_offset(copy.deepcopy(self.w_offset))
        if layer.get_scale_type() == 'table':
            func_table = getattr(self, "serialize_{}".format("table"))
            res = layer.table_export(func_table,
                                     layer.get_table(),
                                     is_fc_bias=False,
                                     layer_name=layer.get_layer_name(),
                                     name=os.path.join(self.weights_dir,
                                                       "weight.b"))
            self.save_weights.extend(res)
            self.w_offset['w_offset'] += self.calc_w_offset(res) # type: ignore

    def export_features(self, layer):
        global res
        layer_idx = layer.get_idx()
        layer_type = layer.get_layer_type()
        # export features
        func_data = getattr(self, "serialize_{}".format("data"))
        if layer_type == "data":
            res = layer.feat_export(func_data,
                                    layer.get_out_data()["output"],
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      "indata.b"))
            self.save_indata.extend(res)

            ### test export indata
            # if self.debug:
            #     self.save_qdata(res, './work_dir/feat_quant/data.txt', is_float=False)
                # fdata = layer.get_out_data()["output"].transpose(2, 3, 0, 1).reshape(-1)
                # self.save_qdata(fdata, './work_dir/feat_float/data.txt', is_float=True)

        elif layer_type == "split":
            layer_names = layer.get_onnx_output_name()
            for feat_id, feat in enumerate(layer.get_out_data()):
                feat = feat["output"]
                # layer_name = "{}_{}_{}_{}.b".format(
                # str(layer_idx).zfill(4), layer_type, layer.get_layer_name(), layer_names[feat_id])
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4), layer_type, layer.get_idx(), feat_id)
                res = layer.feat_export(func_data,
                                        feat,
                                        feat_id,
                                        insert=layer.get_insert(),
                                        layer_name=layer.get_layer_name(),
                                        name=os.path.join(
                                            self.weights_dir, layer_name))
                self.save_feats.update({layer_name: res})
        elif layer_type == "shuffle":
            layer_names = layer.get_onnx_output_name()
            feat_concat = layer.get_out_data()[0]["output"]
            # layer_name = "{}_{}_{}_{}.b".format(
            # str(layer_idx).zfill(4), layer_type, layer.get_layer_name(), layer_names[0])
            layer_name = "{}_{}_{}_{}.b".format(
                str(layer_idx).zfill(4), layer_type, layer.get_idx(), 'concat')
            res = layer.feat_export(func_data,
                                    feat_concat,
                                    0,
                                    insert=layer.get_insert(),
                                    layer_name=layer.get_layer_name(),
                                    name=os.path.join(self.weights_dir,
                                                      layer_name))
            self.save_feats.update({layer_name: res})

            feat_split = [
                layer.get_out_data()[1]["output"],
                layer.get_out_data()[2]["output"]
            ]
            for feat_id, feat in enumerate(feat_split):
                # layer_name = "{}_{}_{}_{}.b".format(
                # str(layer_idx).zfill(4), layer_type,
                # layer.get_layer_name(),
                # layer_names[feat_id + 1])
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4), layer_type, layer.get_idx(), feat_id)
                res = layer.feat_export(func_data,
                                        feat,
                                        feat_id,
                                        insert=layer.get_insert(),
                                        layer_name=layer.get_layer_name(),
                                        name=os.path.join(
                                            self.weights_dir, layer_name))
                self.save_feats.update({layer_name: res})
        elif layer_type == "lstm":
            if isinstance(layer.get_out_data(), dict):
                layer_data = [layer.get_out_data()]
            else:
                layer_data = layer.get_out_data()
            ops_list = ['ot', 'ht', 'ct', 'hq', 'fc1', 'fc2', 'fc']
            layer.set_layer_ops(dict(ops=ops_list))
            for ops_id, qdata in enumerate(layer_data[:len(ops_list)]):
                if layer.get_layer_ops()["ops"][ops_id] in ['hq']:
                    # insert = dict(out_pad=[[0, 257]], out_align=[264])
                    insert = {key: [layer.get_insert()['split'][key1][0]] \
                              for key1, key in zip(['in_pad', 'in_align'], ['out_pad', 'out_align'])}
                elif layer.get_layer_ops()["ops"][ops_id] in ['fc1', 'fc2', 'fc']:
                    # insert = dict(out_pad=[[0, 512]], out_align=[512])
                    insert = {key: [layer.get_insert()['split'][key][-1]] \
                              for key in ['out_pad', 'out_align']}
                else:
                    insert = {key: [layer.get_insert()['split'][key][0]] \
                              for key in ['out_pad', 'out_align']}
                if ops_id == 0:
                    ops_id = '0'
                else:
                    ops_id = '0_{}_{}'.format(layer.get_layer_ops()["ops"][ops_id],
                                              layer.get_layer_name())
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4), layer.get_layer_type(), layer.get_idx(), ops_id)
                # insert = {key: [layer.get_insert()['split'][key][1]] \
                #                 for key in ['out_pad', 'out_align']}

                if qdata['output'].shape[1] > 1 and len(qdata['output'].shape) > 4:
                    qdatas = np.split(qdata["output"], qdata['output'].shape[1], axis=1)
                else:
                    qdatas = [qdata["output"]]

                res_ = []
                for qdata_ in qdatas:
                    qdata_ = np.squeeze(qdata_)
                    qdata_ = np.expand_dims(qdata_, axis=[0, 2, 3])
                    res = layer.feat_export(func_data,
                                            qdata_,
                                            insert=insert,
                                            layer_name=layer.get_layer_name(),
                                            name=os.path.join(
                                                self.weights_dir, layer_name))
                    res_.extend(res)
                    self.save_feats.update({layer_name: res_})
        else:
            # if layer.get_layer_name() == "Reshape_rep_squeeze_7":
            # print('test')
            # layer_names = [
            #     layer.get_onnx_output_name()[0] + "_" + ops
            #     for ops in layer.get_layer_ops()["ops"]
            # ]
            layer_names = [
                ops for ops in layer.get_layer_ops()["ops"]
            ]
            fake_result_layer = False
            for out_idx in layer.get_output_idx():
                if out_idx != -1:
                    fake_result_layer = True
            if layer.get_is_result_layer() and not fake_result_layer and \
                    layer_type in ['conv', 'depthwiseconv', 'gemm', 'fc']:
                ### set dequant data for last layer
                self.set_last_layer_fpdata(layer)
                layer_names.append(layer.get_onnx_output_name()[0] + '_fp')

            if isinstance(layer.get_out_data(), dict):
                layer_data = [layer.get_out_data()]
            else:
                layer_data = layer.get_out_data()

            if layer_type in ['shuffle_only']:
                layer_names = layer_names[-1:]

            assert len(layer_names) == len(layer_data)

            for ops_id, qdata in enumerate(layer_data):
                # layer_name = "{}_{}_{}_{}.b".format(
                #     str(layer_idx).zfill(4), layer.get_layer_type(),
                #     layer.get_layer_name(),
                #     layer_names[ops_id])
                if layer.get_layer_type() in ['conv', 'depthwiseconv', 'fc', 'gemm', 'matmul']:
                    if layer.get_is_result_layer():
                        if ops_id == len(layer_names) - 1:
                            ops_id = '0_fp'
                        elif ops_id == len(layer_names) - 2:
                            ops_id = '0'
                        else:
                            ops_id = '0_{}_{}'.format(layer_names[ops_id], layer.get_layer_name())
                    else:
                        if ops_id == len(layer_names) - 1:
                            ops_id = '0'
                        else:
                            ops_id = '0_{}_{}'.format(layer_names[ops_id], layer.get_layer_name())
                layer_name = "{}_{}_{}_{}.b".format(
                    str(layer_idx).zfill(4), layer.get_layer_type(), layer.get_idx(), ops_id)
                res = layer.feat_export(func_data,
                                        qdata["output"],
                                        insert=layer.get_insert(),
                                        layer_name=layer.get_layer_name(),
                                        name=os.path.join(
                                            self.weights_dir, layer_name))
                self.save_feats.update({layer_name: res})

            ### test export feature after conv
            if self.debug:
                if layer_type in ['conv', 'depthwiseconv']:
                    self.save_qdata(res, './work_dir/feat_quant/{}_{}.txt'.format(
                        self.test_feat_idx, layer.get_layer_name()), is_float=False)
                    # fdata = layer.get_in_fdata()[0].transpose(2, 3, 0, 1).reshape(-1)
                    # self.save_qdata(fdata, './work_dir/feat_float/{}_{}.txt'.format(
                    #     self.test_feat_idx, layer.get_layer_name()), is_float=True)
                    self.test_feat_idx = self.test_feat_idx + 1

    def export_weights(self):
        for layer in self.layers:
            if layer.is_extend():
                pass
            layer_type = layer.get_layer_type()
            if layer_type not in self.valid_export_layer:
                if self.debug:
                    self.logger.warning('ignore layer: {}, when write weight.b'.format(layer_type))
            else:
                if layer_type in ["conv", "depthwiseconv"]:
                    self.export_conv_weights(layer)
                elif layer_type in ["fc", "gemm"]:
                    self.export_fc_weights(layer)
                elif layer_type in ["lstm"]:
                    self.export_lstm_weights(layer)
                # elif layer_type in ['batchnormalization']:
                #     self.export_bn_weights(layer)
                elif layer_type in ['batchnormalization', 'layernormalization']:
                    self.export_ln_weights(layer)
                elif layer_type in ['relu', 'relu6', 'relux', 'leakyrelu', 'sigmoid', 'hardsigmoid', 'hardswish',
                                    'tanh']:
                    self.export_table(layer)
                else:
                    pass
                    # self.logger.warning('{} has no weights!'.format(layer_type))
                    # self.logger.fatal('{} export weights is not implement!'.format(layer_type))

    def write_weights(self):
        try:
            self.export_weights()
            writeFile(self.save_weights,
                    os.path.join(self.weights_dir, "weight.b"),
                    mode="ab")
            self.logger.info("write weights done!")
        except:
            self.logger.error("write_weights failed!")
            os._exit(-1)

    def write_features(self):
        for layer in self.layers:
            if layer.get_layer_type() != "data":
                self.export_features(layer)
        for key, feats in self.save_feats.items():
            key = key.replace('/', '-').replace(':', '-')
            writeFile(feats, os.path.join(self.weights_dir, key), mode="ab")
        self.logger.info("write features done!")

    def write_indata(self):
        for layer in self.layers:
            if layer.get_layer_type() == "data":
                self.export_features(layer)
        writeFile(self.save_indata,
                  os.path.join(self.weights_dir, "indata.b"),
                  mode="ab")
        self.logger.info("write indata done!")

    def write_network(self):
        global in_data_shape, in_data_align

        if self.export_mode_c or self.is_debug:
            file = open(os.path.join(self.log_dir, "model.c"), "w")
            file2 = None
        else:
            file = None
            file2 = open(os.path.join(self.log_dir, "model.b"), "wb")

        headers = copy.deepcopy(self.model_template)
        headers[-1] += '\n' # type: ignore
        for header in headers: # type: ignore
            if file: file.write(header)

        input_scales = []
        ignore_layer_types = []
        for layer in self.layers:
            # todo
            if layer.is_extend():
                pass
            layer_id = layer.get_idx()
            layer_type = layer.get_layer_type()
            if layer_type == 'data':
                if 'int' in layer.get_out_data()['output'].dtype.type.__name__:
                    input_scales.append(layer.get_scales()['out_scale'])
                else:
                    input_scales.append(1.0)
                if len(layer.get_ops_setting()['attrs'][0]['shape']) == 4:
                    in_data_shape = layer.get_ops_setting()['attrs'][0]['shape'][2:]
                else:
                    in_data_shape = 1, 1
                if layer.get_insert()['out_align'][0] == 4:
                    in_data_align = self.Csize
                else:
                    in_data_align = layer.get_insert()['out_align'][0]

            if layer_type not in self.valid_export_layer:
                if layer_type not in ignore_layer_types:
                    ignore_layer_types.append(layer_type)
                # self.logger.warning('export not support {}'.format(layer_type))
                # self.logger.warning('ignore layer: {}, when export model.c'.format(layer_type))
            else:
                if layer_type == "batchnormalization":
                    layer_type = "layernormalization"
                getattr(self,
                        "network_{}".format(layer_type)).save(layer, file, file2)

            if layer_id == len(self.layers) - 1:
                if file: file.write("{};\n".format("}"))
        if len(ignore_layer_types) > 1:
            self.logger.warning("{} export not support".format(ignore_layer_types))

        LayerInfo_t, layers_t = re.sub("[^A-Za-z_]", ' ',
                                       headers[-1]).split(' ')[:2] # type: ignore
        if file:
            extra_contents = \
                "i32 layers_cnt=sizeof({})/sizeof({});\n".format(layers_t, LayerInfo_t) + \
                "u32 weight_total_size={};\n".format(self.w_offset["w_offset"]) + \
                "u32 weight_total_size={};\n".format(self.w_offset["w_offset"]) + \
                "u32 in_total_size={}*{}*{};\n".format(in_data_align, in_data_shape[0], in_data_shape[1]) + \
                "u32 layer_seq[]={"
            file.write(extra_contents)
        for layer in self.layers:
            layer_id = layer.get_idx()
            layer_type = layer.get_layer_type()
            if layer_id == len(self.layers) - 1:
                if file:
                    file.write("{}{};\n i32 layer_seq_len={}\n".format(str(layer_id), "}", len(self.layers)))
            else:
                if file:
                    file.write("{}, ".format(str(layer_id)))

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
                if layer_scale_type in ['rshiftscale', 'rrshiftscale']:
                    scales = copy.deepcopy(layer.get_scales())
                    if isinstance(scales, list):
                        scales = scales[0]
                    scale = scales['fscale']
                    if isinstance(scale, np.ndarray):
                        scale = 1.0
                elif layer_scale_type in ['intscale']:
                    scale = 1.0
                elif layer_scale_type in ['ffloatscale', 'float', 'smooth']:
                    scale = 1.0
                else:
                    raise Exception('Not supported : {} in last layer'.format(layer_scale_type))
                if layer.get_layer_type() in ['lstm', 'reshape', 'cmul', 'pmul', 'mul']:
                    out_data = layer.get_out_data()
                    if isinstance(out_data, list):
                        chn = out_data[0]['output'].shape[1]
                    else:
                        chn = out_data['output'].shape[1]
                else:
                    chn = layer.get_ops_setting()['attrs'][0]['out_c']
                result_layer_ids.append(layer_id)
                result_scales.append(scale)
                result_chn.append(chn)
                self.logger.info('fscale: {}, {}'.format(layer.get_layer_name(), scale))

        if file:
            extra_contents = 'u32 insert_list[]={};\n' + 'i32 insert_list_len=0;\n' + \
                             'i32 ref_cnt[]=%s;\n' % str(ref_cnt).replace('[', '{').replace(']', '}') + \
                             'i32 result_layers[]=%s;\n' % str(result_layer_ids).replace('[', '{').replace(']', '}') + \
                             'float result_scales[]=%s;\n' % str(result_scales).replace('[', '{').replace(']', '}') + \
                             'i32 result_chn[]=%s;\n' % str(result_chn).replace('[', '{').replace(']', '}') + \
                             'i32 result_layers_len={};\n'.format(len(result_layer_ids)) + \
                             'float input_scales[]=%s;' % str(input_scales).replace('[', '{').replace(']', '}')
            file.write(extra_contents)
            file.close()
        if file2:
            file2.close()
        self.logger.info("write model.c done!")

    def export(self):
        pass

    def visualization(self):
        def rename_layer(name):
            name = name.replace('-', '_')
            name = name.replace(':', '_')
            return name

        def write_connetion(f, layer_name, layer, output_idx=0):
            for idx in layer.get_output_idx():
                if idx == -1:
                    name = 'output{}'.format(output_idx)
                    output_idx = output_idx + 1
                else:
                    name = self.layers[idx].get_layer_name()

                name = rename_layer(name)
                layer_name = rename_layer(layer_name)

                if 'split_ids' in layer.get_insert().keys():
                    split_ids = layer.get_insert()['split_ids']
                    f.write(layer_name + ' --|> ' + name + ': ' + str(split_ids[idx]))
                else:
                    f.write(layer_name + ' --|> ' + name)
                f.write('\n')

            return output_idx

        def write_attrs(f, name, attrs):
            for k, v in attrs.items():
                name = rename_layer(name)
                f.write(name + ': ' + k + ' ')
                for attr in v:
                    if not isinstance(attr, str):
                        attr = str(attr)
                    f.write(attr + ' ')
                f.write('\n')

        def get_feat(layer, mode='feat_i'):
            if 'split' in layer.get_insert().keys():
                if mode in layer.get_insert()['split'].keys():
                    feat_i = layer.get_insert()['split'][mode]
                else:
                    feat_i = [[-1, -1]]
            else:
                if mode in layer.get_insert().keys():
                    feat_i = layer.get_insert()[mode]
                else:
                    feat_i = [[-1, -1]]

            return feat_i

        def get_align(layer, mode='out_align'):
            if 'split' in layer.get_insert().keys():
                if mode in layer.get_insert()['split'].keys():
                    oc = layer.get_insert()['split'][mode]
                else:
                    oc = [-1]
            else:
                if mode in layer.get_insert().keys():
                    oc = layer.get_insert()[mode]
                else:
                    oc = [-1]

            return oc

        def get_pad(layer, mode='out_pad'):
            if 'split' in layer.get_insert().keys():
                if mode in layer.get_insert()['split'].keys():
                    pad = layer.get_insert()['split'][mode]
                else:
                    pad = [[-1, -1]]
            else:
                if mode in layer.get_insert().keys():
                    pad = layer.get_insert()[mode]
                else:
                    pad = [[-1, -1]]

            return pad

        output_idx = 1

        vis_file = '{}/work_dir/test_vis.mmd'.format(root_dir)
        vis_path, _ = os.path.split(vis_file)
        if not os.path.exists(vis_path):
            os.makedirs(vis_path, mode=0o777, exist_ok=True)

        with open(vis_file, 'w') as f:
            f.write('classDiagram' + '\n')
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

                feat_i = get_feat(layer, mode='feat_i')
                feat_o = get_feat(layer, mode='feat_o')
                ic = get_align(layer, mode='in_align')
                oc = get_align(layer, mode='out_align')
                in_pad = get_pad(layer, mode='in_pad')
                out_pad = get_pad(layer, mode='out_pad')
                is_align = layer.get_insert()['is_align']

                attrs = dict(
                    layer_type=[layer.get_layer_type()],
                    layer_idx=[layer.get_idx()],
                    feat_i=feat_i,
                    feat_o=feat_o,
                    in_pad=in_pad,
                    out_pad=out_pad,
                    ic=[ic], oc=[oc],
                    is_align=[is_align])
                output_idx = write_connetion(f, layer_name, layer, output_idx=output_idx)
                write_attrs(f, layer_name, attrs)

        # if self.is_debug:
        #     self.logger.info('start export visualization')
        #     json_content = dict(maxTextSize=99999999)
        #     with open(os.path.join(vis_path, "mermaidRenderConfig.json"), "w") as f:
        #         json.dump(json_content, f, indent=4, ensure_ascii=False)
        #     json_content = dict(args=['--no-sandbox'])
        #     with open(os.path.join(vis_path, "puppeteer-config.json"), "w") as f:
        #         json.dump(json_content, f, indent=4, ensure_ascii=False)
        #     os.system('bash {}/benchmark/scripts/export_visualization.sh {}'.format(root_dir, vis_path))
        #     self.logger.info('finish export visualization')


class NetworkBase(Object): # type: ignore

    def __init__(self, **kwargs):
        super(NetworkBase, self).__init__()
        self.kwargs = None
        self.MAX_IN_OUT_LEN = None

    @staticmethod
    def list2Cstyle(x: list):
        return str(x).replace('[', '{').replace(']', '}')

    @staticmethod
    def get_align_channel(ch, align_size):
        return ((ch + align_size - 1) // align_size) * align_size

    def invert_type(self, invert_types):
        for type in invert_types:
            tmp = dict()
            for key, value in self.kwargs[type].items(): # type: ignore
                for v in value:
                    tmp[v] = key
            setattr(self, type, tmp)

    def invert_dict_of_list(self, dict_a):
        dict_b = dict()
        for key, value in dict_a.items():
            for v in value:
                dict_b[v] = key

        return dict_b

    def get_io_len(self, layer):
        in_len = len(layer.get_input_idx())
        out_len = len(layer.get_output_idx())

        return in_len, out_len

    def get_contents(self, layer, contents):
        input_ids = copy.deepcopy(layer.get_input_idx())
        output_ids = copy.deepcopy(layer.get_output_idx())
        if layer.get_layer_type() in ['lstm']:
            in_len = len([id for id in input_ids if id >= 0])
            out_len = len([id for id in output_ids if id >= 0])
        else:
            in_len = len(input_ids)
            out_len = len(output_ids)
        while len(input_ids) < self.MAX_IN_OUT_LEN: # type: ignore
            input_ids.append(0)
        while len(output_ids) < self.MAX_IN_OUT_LEN: # type: ignore
            output_ids.append(0)

        contents = _write(contents, in_len, tail=',')
        contents = _write(contents, out_len, tail=',')

        contents += '{'
        for id, data in enumerate(input_ids):
            if id == len(input_ids) - 1:
                contents = _write(contents, data, tail='')
            else:
                contents = _write(contents, data, tail=',')
        contents += '},'

        contents += '{'
        for id, data in enumerate(output_ids):
            if id == len(output_ids) - 1:
                contents = _write(contents, data, tail='')
            else:
                contents = _write(contents, data, tail=',')
        contents += '}'

        contents = contents.replace(',}', '}')
        contents = contents.replace("'", '')

        return contents

    def save(self, layer, file, file2):
        pass

    def __call__(self, layer, file, file2):
        self.save(layer, file, file2)
