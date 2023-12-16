# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/1 17:06
# @File     : model_export.py
import copy
import os

import numpy as np

try:
    from export.v2.network import NETWORK_V2 as rt_factory
    from export.v1.model_export import mExportBase
    from export.v1.serialize import SERIALIZE as serialize_factory
    from export.v1.wExport import wExport as exportW_factory
except Exception:
    from onnx_converter.export.v2.network import NETWORK_V2 as rt_factory # type: ignore
    from onnx_converter.export.v1.model_export import mExportBase # type: ignore
    from onnx_converter.export.v1.serialize import SERIALIZE as serialize_factory # type: ignore
    from onnx_converter.export.v1.wExport import wExport as exportW_factory # type: ignore


class mExportV2(mExportBase): # type: ignore
    def __init__(self, **kwargs):
        super(mExportV2, self).__init__(**kwargs)

        self.model_template = [
            '#include "npu_layer.h"\n',
            "\n",
            "LayerInfo_t layers[]={",
        ]
        kwargs["version"] = "v2"
        self.init_wexport(**kwargs)

        model_c = self.init_net_work(rt_factory, kwargs)
    
    def init_net_work(self, rt_factory, kwargs):
        model_c = self.update_model_c(kwargs)
        for key in self.layer_map.keys():
            if key in self.ignore_layers:
                continue
            for m in self.layer_map[key]:
                setattr(
                    self, "network_{}".format(m), rt_factory.get(m)(**model_c)
                )
        return model_c        
                    
    def update_model_c(self, kwargs):
        model_c = copy.deepcopy(kwargs["model_c"])
        for key in model_c.keys():
            if key in kwargs.keys():
                kwargs["model_c"][key] = kwargs[key]
        return model_c
                
    def process_conv_with_concat(self, layer):
        result, result_id = [], []
        res_dict, res_list = {}, []

        # for id in layer.get_input_idx():
        #     result, result_id = self.recursive_top_layer(
        #         self.layers[id], result, result_id)

        # print(result, result_id)
        # for id, res in zip(result_id, result):  # remove duplication
        #     if id not in res_dict.keys():
        #         res_dict[id] = res
        #         res_list.append(res)
        # print(res_list)

        layer_id = layer.get_input_idx()[0]
        in_pad = self.layers[layer_id].get_insert()["out_pad"]
        in_align = self.layers[layer_id].get_insert()["out_align"]
        # feat_i = [
        #     layer.get_in_data()[0]['output'].shape[2],
        #     layer.get_in_data()[0]['output'].shape[3]
        # ]
        # feat_o = [
        #     layer.get_out_data()[-1]['output'].shape[2],
        #     layer.get_out_data()[-1]['output'].shape[3]
        # ]

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        layer.set_insert(res)

    def process_conv_with_split(self, layer, result):
        # result, result_id = [], []

        # for id in layer.get_output_idx():
        #     result, result_id = self.recursive_down_layer(
        #         self.layers[id], result, result_id)
        #     # print(result, result_id)

        # result = self.layers[layer.get_output_idx()[0]].get_ops_setting()['attrs'][0]['split']
        if layer.get_layer_type() in ["fc"]:
            Csize = self.I_Align
        else:
            Csize = self.Csize
        out_align = [self.get_align_channel(ch, Csize) for ch in result]
        out_pad = []
        for i, ch in enumerate(result):
            nonezero = [0, ch]
            if i > 0:
                nonezero = [0, ch] + np.sum(out_align[:i])
            nonezero = list(nonezero)
            out_pad.append(nonezero)

        res = {
            "out_pad": out_pad,
            "out_align": [self.get_align_channel(np.sum(out_align), self.Ksize)],
        }
        res.update(layer.get_insert())
        layer.set_insert(res)

    def process_conv_parallel_concat_with_elementwise(self, layer, result):
        # if layer.get_layer_name() == 'Conv_7':
        # print('test')
        if layer.get_layer_type() in ["fc"]:
            Ksize = np.max([self.O_Align, self.I_Align])
        else:
            Ksize = self.Ksize
        out_pad = [[0, ch] for ch in result]
        out_align = [self.get_align_channel(ch, Ksize) for ch in result]
        out_align_ = [int(np.sum(out_align[:i])) for i in range(len(out_align))]
        out_pad = [
            list(np.array(pad) + align) for pad, align in zip(out_pad, out_align_)
        ]

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "out_pad": out_pad,
            "out_align": [np.sum(out_align)],
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        layer.set_insert(res)

    def process_shuffle_only_with_split(self, layer, result):
        # result, result_id = [], []

        # for id in layer.get_output_idx():
        #     result, result_id = self.recursive_down_layer(
        #         self.layers[id], result, result_id)
        #     # print(result, result_id)

        # result = self.layers[layer.get_output_idx()[0]].get_ops_setting()['attrs'][0]['split']
        layer_id = layer.get_input_idx()[0]
        in_pad = self.layers[layer_id].get_insert()["out_pad"]
        in_align = self.layers[layer_id].get_insert()["out_align"]

        out_align = [self.get_align_channel(ch, self.Csize) for ch in result]
        out_pad = []
        for i, ch in enumerate(result):
            nonezero = [0, ch]
            if i > 0:
                nonezero = [0, ch] + np.sum(out_align[:i])
            nonezero = list(nonezero)
            out_pad.append(nonezero)

        feat_i, feat_o = self.get_feature_shape(layer)

        res = {
            "in_pad": in_pad,
            "in_align": in_align,
            "out_pad": out_pad,
            "out_align": [self.get_align_channel(np.sum(out_align), self.Ksize)],
            "feat_i": [feat_i],
            "feat_o": [feat_o],
        }
        res.update(layer.get_insert())
        layer.set_insert(res)

    def export(self):
        for layer in self.layers:
            self.set_bias_data(layer)
            self.set_layer_datatype(layer)
            self.set_voice_model_feats(layer)
            try:

                layer_id = layer.get_idx()
                layer_type = layer.get_layer_type()

                if layer_type in ["data"]:
                    self.process_data(layer)
                elif layer_type in ["conv", "depthwiseconv", "convtranspose", "fc"]:
                    is_first_conv = self.check_first_conv(layer)
                    layer.set_first_conv(is_first_conv)
                    # if layer.get_layer_name() == 'Conv_388':
                    # print('test')
                    flag_c = (
                        True
                        if "concat"
                        in [
                            self.layers[id].get_layer_type() for id in layer.get_input_idx()
                        ]
                        else False
                    )
                    # result, result_id = [], []
                    # for id in layer.get_input_idx():
                    #     result, result_id = self.recursive_top_layer(
                    #         self.layers[id], result, result_id)
                    # flag_c = True if len(result) > 0 else False
                    if flag_c:
                        self.process_conv_with_concat(layer)
                    else:
                        self.process_conv_without_concat(layer)

                    # if split is exist, only excute 'process_conv_with_split',
                    # if split is not exist and [(conv + concat)->elementwise] is exist,
                    # excute 'process_conv_parallel_concat_with_elementwise'
                    mode = "split"
                    result, result_id = [], []
                    for id in layer.get_output_idx():
                        result, result_id = self.recursive_down_layer(
                            self.layers[id], result, result_id, mode=mode
                        )
                    flag_s0 = True if len(result) > 0 else False
                    if flag_s0:
                        self.process_conv_with_split(layer, result=result[0])

                    mode = "conv_parallel_concat_elementwise"
                    result, result_id = [], []
                    for id in layer.get_output_idx():
                        result, result_id = self.recursive_down_layer(
                            self.layers[id], result, result_id, mode=mode
                        )
                    flag_s = True if len(result) > 0 else False
                    res_concat = copy.deepcopy(result)
                    if flag_s:
                        elementwise_layer = self.layers[result_id[0]]
                        result, result_id = [], []
                        for id in elementwise_layer.get_output_idx():
                            if id < 0:
                                continue
                            result, result_id = self.recursive_down_layer(
                                self.layers[id], result, result_id, mode=mode
                            )
                        flag_s = False if len(result) > 0 else True
                    if flag_s:
                        self.process_conv_parallel_concat_with_elementwise(
                            layer, result=res_concat
                        )

                    if not flag_s0 and not flag_s:
                        self.process_conv_without_split(layer)

                elif layer_type in ["concat"]:
                    # if layer.get_layer_name() == 'Concat_4':
                    # print('test')
                    is_exist = False
                    for id in layer.get_output_idx():
                        is_exist = self.find_elementwise_layer(
                            self.layers[id], is_exist=is_exist
                        )
                    if is_exist:
                        # if (concat + concat/split)->elementwise, is_align = False
                        self.process_concat_with_elementwise(layer)
                    else:
                        self.process_concat(layer)
                elif layer_type in ["split"]:
                    self.process_split(layer)
                elif layer_type in ["lstm"]:
                    self.process_lstm(layer)
                elif layer_type in ["gru"]:
                    self.process_gru(layer)
                elif layer_type in ["shuffle"]:
                    self.process_shuffle(layer)
                elif layer_type in ["shuffle_only"]:
                    # if layer.get_layer_name() == "Reshape_189":
                    #     print('test')
                    result, result_id = [], []
                    for id in layer.get_output_idx():
                        result, result_id = self.recursive_down_layer(
                            self.layers[id], result, result_id, mode="split"
                        )
                    flag_s = True if len(result) > 0 else False
                    if flag_s:
                        self.process_shuffle_only_with_split(layer, result=result[0])
                    else:
                        self.process_shuffle_only(layer)
                elif layer_type in ["batchnormalization", "layernormalization"]:
                    self.process_bn_layer(layer)
                elif layer_type in ["mul", "cmul", "pmul", "add", "sub"]:
                    self.process_elementwise_layer(layer)
                elif layer_type in ["pad"]:
                    self.process_pad_layer(layer)
                else:  # if layer_type in ['averagepool', 'cmul', 'pmul', 'mul', 'add', 'resize', 'sigmoid']:
                    self.process(layer)
                # else:
                # self.process(layer)
                # self.logger.fatal('{} export is not implement!'.format(layer_type))
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
        self.logger.info("export done!")
        
        ### set shuffle_only align
        for layer in self.layers:
            layer_id = layer.get_idx()
            layer_type = layer.get_layer_type() 
            if layer_type in ["shuffle_only"]:
                pre_layer = self.layers[layer.get_input_idx()[0]] 
                next_layer = self.layers[layer.get_output_idx()[0]]
                pre_insert = pre_layer.get_insert()
                next_insert = next_layer.get_insert()
                insert = layer.get_insert()
                
                if pre_layer.get_layer_type() in ["concat"]:
                    pre_key = "in_align"
                else:
                    pre_key = "out_align"
                if "split" in pre_insert.keys():
                    insert["in_pad"] = pre_insert["split"]["out_pad"]
                    insert["in_align"] = pre_insert["split"][pre_key]
                else:
                    insert["in_pad"] = pre_insert["out_pad"]
                    insert["in_align"] = pre_insert[pre_key]       
                    
                if next_layer.get_layer_type() in ["split"]:
                    next_key = "out_align"
                else:
                    next_key = "in_align"                    
                if "split" in next_insert.keys():
                    insert["out_pad"] = next_insert["split"]["in_pad"]
                    insert["out_align"] = next_insert["split"][next_key]
                else:
                    insert["out_pad"] = next_insert["in_pad"]
                    insert["out_align"] = next_insert[next_key]                                  
                
                # print("test")    
