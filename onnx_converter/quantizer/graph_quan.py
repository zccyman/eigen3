# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/8 17:21
# @File     : graph_quan.py

import glob
import os
import copy
import json
import pickle
import numpy as np
from tqdm import tqdm

try:
    from utils import Object, flatten_list, get_scale_shift
except:
    from onnx_converter.utils import Object, flatten_list, get_scale_shift # type: ignore

from .data_quan import QUANTIZE as quantize_factory

first_lstm_feat_int16_enable = False
first_lstm_weight_int16_enable = False
first_lstm_hidden_int16_enable = False

def get_quant(quan_dict, quan_factory, dataset_scales, name):
    quan_method = quan_factory.get(quan_dict['method'])(bit_select=quan_dict['bit_select'],
                                                        bits_dict=quan_dict['bits_dict'],
                                                        maxs=quan_dict['maxs'], mins=quan_dict['mins'])

    if isinstance(dataset_scales[name], np.ndarray):
        quan_method.get_quan_param(dataset_scales[name])
    elif isinstance(dataset_scales[name], dict):
        quan_method.set_scale(dataset_scales[name])
    else:
        print('Not support data scale input!')
        os._exit(-1)  # , print('nor support data scale input!')
    return quan_method


def quan_weight(layers: list, layer: object, quan_dict: dict, quantize_factory: quantize_factory, quant_param_dict = None): # type: ignore
    if len(layer.get_layer_ops()['weights']) <= 0: # type: ignore
        return
    quantizes = layer.get_quantize() # type: ignore
    quan_method = quantize_factory.get(quan_dict['method'])(
        bit_select=quan_dict['bit_select'],
        bits_dict=quan_dict['bits_dict'],
        maxs=quan_dict['maxs'], mins=quan_dict['mins'])
    # add offline quant table mode, by Nan.Qin
    if quant_param_dict is None:
        quan_method.get_quan_param(layer.get_layer_ops()['weights'][0]) # type: ignore
    else:
        quan_method.set_scale(quant_param_dict)

    w_scales = dict(scale=quan_method.scale, zero_point=quan_method.zero_point)
    layer.set_w_scale(w_scales) # type: ignore
    weights = layer.get_layer_ops()['weights'][0] # type: ignore
    quan_method.get_quan_param(weights)
    #layer.set_w_scale(quan_method.get_scale()[0])
    q_weights = quan_method.get_quan_data(weights)
    layer.set_qweights(q_weights) # type: ignore
    layer.set_weight(weights) # type: ignore
    has_bias = layer.get_layer_ops()['attrs'][0]['bias'] # type: ignore
    if has_bias:
        # todo zero-point not consider
        # print(layer_type, layer.get_idx())

        # si = layers[layer.get_input_idx()[0]].get_scale()
        # if isinstance(si, list) or isinstance(si, tuple):
        #     si = si[0]
        # qbias = layer.get_layer_ops()['weights'][1] / (si * w_scales['scale']) + 0.5
        # layer.set_qbias(qbias)
        
        bias = layer.get_layer_ops()['weights'][1] # type: ignore
        si = layers[layer.get_input_idx()[0]].get_scale()[0] # type: ignore
        qbias = bias / (si['scale'] * w_scales['scale']) + 0.5
        layer.set_qbias(qbias) # type: ignore

    quantizes.update(dict(w_quan=quan_method))
    layer.set_quantize(quantizes) # type: ignore
    # return layer


# dataset scales support two data type
# first: node array data, the other: extract datasets node array data save in dictionary
# not design feat per-channel quantize
def quan_feat(layer: object, layers: list, quan_dict: dict, dataset_scales: dict, quan_factory: quantize_factory): # type: ignore
    layer_type = layer.get_layer_type() # type: ignore

    if layer.get_scale_type() in ['smooth']: # type: ignore
        in_quantizes, in_scale = list(), list()
        if layer_type == 'data':
            # todo just for symmetric quantize, we will implement asymmetric method
            in_scale.append(1.0)
        else:
            # fixed multi in/out connect, then pre and next layer scale length gt 1
            in_scale = list()
            input_layers = [layers[idx] for idx in layer.get_input_idx()] # type: ignore
            for in_name in layer.get_onnx_input_name(): # type: ignore
                for ilayer in input_layers:
                    if in_name in ilayer.get_onnx_output_name():
                        idx = ilayer.get_onnx_output_name().index(in_name)
                        in_scale.append(ilayer.get_scale()[idx])
                        if not ilayer.is_extend():
                            in_quantizes.append(ilayer.get_quantize()['feat']['so' + str(idx)])
                        else:
                            in_quantizes.append(ilayer.get_quantize())
        # todo zero-point not implement
        len_scale = len(ilayer.get_onnx_output_name()) # type: ignore
        val = max(in_scale)
        val_ = [val for _ in range(len_scale)]
        quantizes = layer.get_quantize() # type: ignore
        quantizes['feat'] = dict()
        for idx in range(len_scale):
            quantizes['feat']['so' + str(idx)] = in_quantizes[in_scale.index(val)]
        layer.set_quantize(quantizes) # type: ignore
        layer.set_scale(val_) # type: ignore
        layer.set_in_quantize(in_quantizes) # type: ignore
        return

    # layer_type = layer.get_layer_type().lower()
    out_names = layer.get_onnx_output_name() # type: ignore
    feat_dict = dict()
    # bit_select = quan_dict['bit_select']
    scale, quantize, data_scale = list(), dict(), list()
    for idx, name in enumerate(out_names):
        key = 'so' + str(idx)
        quan_method = get_quant(quan_dict, quan_factory, dataset_scales, name)

        data_scale.append(dataset_scales[name])
        # todo zero-point not consider
        feat_dict[key] = quan_method.get_scale()
        # todo zero-point not consider
        scale.append(feat_dict[key][0])
        quantize[key] = quan_method
    layer.set_scale(scale) # type: ignore
    layer.set_data_scale(data_scale) # type: ignore
    layer.set_quantize(dict(feat=quantize)) # type: ignore

    # return layer


class BaseData(Object): # type: ignore
    def __init__(self, **kwargs):
        super(BaseData, self).__init__(**kwargs)
        self.bits_dict = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32}
        self.maxs = {0: 255, 1: 127, 2: 65535, 3: 32767, 4: 4294967295, 5: 2147483647}
        self.mins = {0: 0, 1: -128, 2: 0, 3: -32768, 4: 0, 5: -2147483648}
        self.precision, self.bit_select, self.int_scale = 0, 1, 8
        if 'log_name' in kwargs.keys():
            self.logger = self.get_log(log_name=kwargs['log_name'], log_level=kwargs.get('log_level', 20))
        else:
            self.logger = self.get_log(log_name='graph_quan.log', log_level=kwargs.get('log_level', 20))

    def default_setting(self, input_dict):
        if 'bits_dict' in input_dict.keys(): self.bits_dict = input_dict['bits_dict']
        if 'maxs' in input_dict.keys(): self.maxs = input_dict['maxs']
        if 'mins' in input_dict.keys(): self.mins = input_dict['mins']
        if 'precision' in input_dict.keys(): self.precision = input_dict['precision']
        if 'int_scale' in input_dict.keys(): self.int_scale = input_dict['int_scale']
        if 'bit_select' in input_dict.keys(): self.bit_select = input_dict['bit_select']
        # if 'bit_scale' in input_dict.keys(): self.int_scale = input_dict['int_scale']
        for key in self.bits_dict.keys():
            if isinstance(self.bits_dict[key], str):
                self.bits_dict[key] = eval(self.bits_dict[key])


def json_dump(name: str="myfile.json", data: dict = {}):
    with open(name, "w") as out_file:
        json.dump(data, out_file, indent = 6)

def json_load(name):
    pass

def pickle_dump(name: str="myfile.setting", data: dict={}):
    with open(name, "wb+") as f:
        pickle.dump(data, f)

def pickle_load(name: str):
    with open(name, "rb+") as f:
        load_context = pickle.load(f)
        return load_context

def convert_weights_to_pkl(weights: list):
    for j in range(len(weights)):
        if weights[j] == []:
            continue
        if isinstance(weights[j]['weight'], np.ndarray):
            weights[j]['weight'] = weights[j]['weight'].tolist()
        if "dims" in weights[j].keys():
            dims = weights[j]['dims']
            weights[j]['dims'] = np.array(dims).tolist()
    return weights

def convert_pkl_to_weights(weights: list):
    for j in range(len(weights)):
        if isinstance(weights[j]['weight'], list):
            weights[j]['weight'] = np.array(weights[j]['weight'])
        # if "dims" in weights[j].keys():
        #     dims = weights[j]['dims']
        #     weights[j]['dims'] = np.array(dims).tolist()
    return weights

def convert_attrs_to_pkl(attrs: list):
    for i in range(len(attrs)):
        if "weights" in attrs[i].keys():
            attrs[i]["weights"] = convert_weights_to_pkl(attrs[i]["weights"])
        else:
            for key in attrs[i].keys():
                if isinstance(attrs[i][key], np.ndarray):
                    attrs[i][key] = attrs[i][key].tolist()
                if "protobuf" in str(type(attrs[i][key])):
                    attrs[i][key] = list(attrs[i][key])
                if "dims" == key:
                   attrs[i][key] = np.array(attrs[i][key]).tolist()

    return attrs

def convert_pkl_to_attrs(attrs: list):
    for i in range(len(attrs)):
        if "weights" in attrs[i].keys():
            attrs[i]["weights"] = convert_pkl_to_weights(attrs[i]["weights"])
        else:
            for key in attrs[i].keys():
                if key in ["shape", "weight"]:
                    attrs[i][key] = np.array(attrs[i][key])

    return attrs

def dump_layer(origin_layer, fd_path):
    layer_name = origin_layer.get_layer_name()
    saved_setting_name = os.path.join(fd_path, "{}.setting".format(layer_name))
    saved_inner_setting_name = os.path.join(fd_path, "{}.inner_setting".format(layer_name))
    saved_inner_attrs_name = os.path.join(fd_path, "{}.inner_attrs".format(layer_name))
    saved_ops_nodes_name = os.path.join(fd_path, "{}.ops_nodes".format(layer_name))
    saved_nodes_name = os.path.join(fd_path, "{}.nodes".format(layer_name))
    saved_ops_instance_name = os.path.join(fd_path, "{}.ops_instance".format(layer_name))
    saved_ops_name = os.path.join(fd_path, "{}.ops".format(layer_name))
    saved_layer_name = os.path.join(fd_path, "{}.layer".format(layer_name))
    setting = origin_layer.get_ops_setting() # pickle
    inner_setting = setting.pop("setting") # pickle, json
    inner_attrs = setting.pop("attrs") # pickle, json
    convert_attrs_to_pkl(inner_attrs)
    ops_nodes = origin_layer.get_ops_nodes() # pickle
    ops = origin_layer.get_layer_ops()
    ops['attrs'] = convert_attrs_to_pkl(ops['attrs'])
    ops['weights'] = \
        [weight.tolist() if isinstance(weight, np.ndarray) else weight for weight in ops['weights']]
    for t_node in ops_nodes:
        attrs = t_node.get_attr()
        attrs = convert_attrs_to_pkl([attrs])
        t_node.set_attr(attrs[0])
        weights = t_node.get_weights()
        weights = convert_weights_to_pkl(weights)
        t_node.set_weights(weights)
    nodes = origin_layer.get_nodes() # pickle
    for t_node in nodes:
        attrs = t_node.get_attr()
        attrs = convert_attrs_to_pkl([attrs])
        t_node.set_attr(attrs[0])
        weights = t_node.get_weights()
        weights = convert_weights_to_pkl(weights)
        t_node.set_weights(weights)
    ops_instance = origin_layer.get_ops_instance() # pickle
    origin_layer.set_ops_setting({})
    origin_layer.set_ops_nodes([])
    origin_layer.set_nodes([])
    origin_layer.set_ops_instance([])
    origin_layer.clear_layer_ops()
    pickle_dump(saved_setting_name, setting)
    pickle_dump(saved_inner_setting_name, inner_setting)
    pickle_dump(saved_inner_attrs_name, inner_attrs)
    pickle_dump(saved_ops_nodes_name, ops_nodes)
    pickle_dump(saved_nodes_name, nodes)
    pickle_dump(saved_ops_instance_name, ops_instance)
    pickle_dump(saved_ops_name, ops)
    pickle_dump(saved_layer_name, origin_layer)

def reolad_layer(origin_layer, fd_path):
    layer_name = origin_layer.get_layer_name()
    saved_setting_name = os.path.join(fd_path, "{}.setting".format(layer_name))
    saved_inner_setting_name = os.path.join(fd_path, "{}.inner_setting".format(layer_name))
    saved_inner_attrs_name = os.path.join(fd_path, "{}.inner_attrs".format(layer_name))
    saved_ops_nodes_name = os.path.join(fd_path, "{}.ops_nodes".format(layer_name))
    saved_nodes_name = os.path.join(fd_path, "{}.nodes".format(layer_name))
    saved_ops_instance_name = os.path.join(fd_path, "{}.ops_instance".format(layer_name))
    saved_ops_name = os.path.join(fd_path, "{}.ops".format(layer_name))
    saved_layer_name = os.path.join(fd_path, "{}.layer".format(layer_name))
    setting = pickle_load(saved_setting_name)
    inner_setting = pickle_load(saved_inner_setting_name)
    inner_attrs = pickle_load(saved_inner_attrs_name)
    setting["attrs"] = convert_pkl_to_attrs(inner_attrs)
    setting["setting"] = inner_setting
    ops_nodes = pickle_load(saved_ops_nodes_name)
    for t_node in ops_nodes:
        attrs = t_node.get_attr()
        attrs = convert_pkl_to_attrs([attrs])
        t_node.set_attr(attrs[0])
        weights = t_node.get_weights()
        weights = convert_pkl_to_weights(weights)
        t_node.set_weights(weights)
    nodes = pickle_load(saved_nodes_name)
    for t_node in nodes:
        attrs = t_node.get_attr()
        attrs = convert_pkl_to_attrs([attrs])
        t_node.set_attr(attrs[0])
        weights = t_node.get_weights()
        weights = convert_pkl_to_weights(weights)
        t_node.set_weights(weights)
    # ops_instance = copy.deepcopy(origin_layer.get_ops_instance()) # pickle
    ops_instance = pickle_load(saved_ops_instance_name)
    ops = pickle_load(saved_ops_name)
    ops['attrs'] = convert_pkl_to_attrs(ops['attrs'])
    ops['weights'] = \
        [np.array(weight) if isinstance(weight, list) else weight for weight in ops['weights']]

    layer = pickle_load(saved_layer_name)
    layer.set_ops_setting(setting)
    layer.set_ops_nodes(ops_nodes)
    layer.set_nodes(nodes)
    layer.set_ops_instance(ops_instance)
    layer.set_layer_ops(ops)
    return layer

# parse quantize config, compose layer quantize and constant setting
# quantize length for each layer
# float/int scale setting
# if layer does not has activation ops, add virtual act after conv/fc process scale and shift
#
class GraphQuant(BaseData):
    def __init__(self, **kwargs):
        super(GraphQuant, self).__init__(**kwargs)
        # layer ops setting
        self._settings = kwargs['default_setting']
        self._graph = kwargs['graph']
        self._output = kwargs['output']
        # ['relu', 'leakyrelu', 'tanh', 'hardswish', 'prelu', 'celu']
        self._act = kwargs['act']
        self._is_quantized = False
        # self.quan_dict = kwargs['quantize']
        self._txme_saturation = kwargs['txme_saturation'] if "txme_saturation" in kwargs.keys() else 1
        self._virtual_round = kwargs['virtual_round']

        self.default_setting(kwargs)

        self.logger.info(self._settings)

        self.push_virtual_act()
        self.default_setting = dict(bits_dict=self.bits_dict, # type: ignore
                                    maxs=self.maxs, mins=self.mins, # type: ignore
                                    precision=self.precision, # type: ignore
                                    int_scale=self.int_scale) # type: ignore

        self._layer_cle_list = []

    def get_quan_stat(self):
        return self._is_quantized

    def set_graph(self, graph):
        self._graph = graph

    def get_layers(self):
        return self._graph.get_layers()

    def set_layers(self, layers):
        self._graph.set_layers(layers)

    def replace_layer(self, idx, layer):
        self.get_layers()[idx] = layer

    # insert virtual act ops when conv/depthwise/fc does not activation follows
    # quantize align process in activation ops
    def push_virtual_act(self):
        for layer in self.get_layers():
            layer_type = layer.get_layer_type().lower()
            if layer_type in ['conv', 'depthwiseconv', 'convtranspose', 'fc', 'gemm', 'matmul']:
                ops = layer.get_layer_ops()
                if len(ops['ops']) > 2:
                    continue
                act_intersection = set(ops['ops']).intersection(self._act)
                if len(act_intersection) <= 0:
                    self.logger.warning(' {} no activation in normal!'.format(layer.get_nodes()[0].get_name()))
                    if 'bias' in ops['ops']:
                        key = 'bias'
                    else:
                        key = layer_type
                    ops['ops'].insert(ops['ops'].index(key) + 1, 'act')
                    ops['attrs'].insert(ops['ops'].index(key) + 1, dict())
                    layer.set_layer_ops(ops)

    # instance quantize method and using layer set_quantize saved quantize for weights/bias
    # default setting for weight quantize method
    def quan_weights(self, weight_scale_dict = None):
        # kwargs = dict(bit_select=self.bit_select, maxs=self.maxs, mins=self.mins, bit_dict=self.bit_dict)
        for idx, layer in enumerate(self.get_layers()):
            layer_type = layer.get_layer_type().lower()
            # user define quantize weights
            if layer.is_extend():
                layer.quan_feat()
            else:
                key = layer_type if layer_type in self._settings.keys() else 'default'
                w_scale_dict = weight_scale_dict[layer.get_layer_name()] if weight_scale_dict else None
                if self._settings[layer_type]['weights']:
                    quan_weight(self.get_layers(), layer, self._settings[key]['weights'], quantize_factory,
                                w_scale_dict)
            # self.__layers[idx] = layer

    # update feat weights quantize length or method
    # just process some layer, using optimizer accuary
    # layer idx as the key
    def update_quan_weights(self, quan_dict: dict):
        for key in quan_dict.keys():
            layer = self.get_layers()[key]
            layer_type = layer.get_layer_type().lower()
            if layer.is_extend():
                layer.quan_weights()
            else:
                if layer_type in ['conv', 'depthwiseconv', 'convtranspose', 'fc']:
                    quan_weight(self.get_layers(), layer, quan_dict[key], quantize_factory)
            # self.__layers[key] = layer

    # some layer has no quantize needed
    # then we will copy quantize from prev quantize to current layer
    def quan_feats(self, dataset_scales: dict):
        # kwargs = dict(bit_select=self.bit_select, maxs=self.maxs, mins=self.mins, bit_dict=self.bit_dict)
        for idx, layer in enumerate(self.get_layers()):
            layer_type = layer.get_layer_type().lower()
            # user define quantize feature map
            key = layer_type if layer_type in self._settings.keys() else 'default'
            layer.set_output_type(self._settings[key]['out_type'])
            layer.set_scale_type(self._settings[key]['process_scale'])
            if layer.is_extend():
                layer.quan_feat()
            else:
                key = layer_type if layer_type in self._settings.keys() else 'default'
                quan_feat(layer, self.get_layers(), self._settings[key]['feat'], dataset_scales, quantize_factory)

            # else:
            #     self.__layers[idx] = layer

    # update feat quantize length or method
    # layer idx as the key
    def update_quan_feats(self, quan_dict: dict, dataset_scales: dict):
        for key in quan_dict.keys():
            layer = self.get_layers()[key]
            # layer: object, quan_dict: dict, dataset_scales: dict, quan_factory: object
            # user define quantize feature map
            if layer.is_extend():
                layer.quan_feat()
            else:
                quan_feat(layer, quan_dict[key], dataset_scales, quantize_factory) # type: ignore

    # instance ops
    # calc float/int scale data correct
    def quan_ops(self):
        default_setting = {'bits_dict': self.bits_dict, 'maxs': self.maxs, 'mins': self.mins,
                           'precision': self.precision, 'int_scale': self.int_scale}
        for idx, layer in enumerate(self.get_layers()):
            # if hasattr(layer, 'quan_weight') or hasattr(layer, 'quan_feat'):
            #     layer.quan_weight()
            #     continue
            ops = layer.get_layer_ops()
            layer_type = layer.get_layer_type().lower()
            # todo shuffle or shuffle-only scale not ready
            in_quantizes, in_scale = list(), list()
            if layer_type == 'data':
                # todo just for symmetric quantize, we will implement asymmetric method
                in_scale.append(1.0)
            else:
                # fixed multi in/out connect, then pre and next layer scale length gt 1
                in_scale = list()
                input_layers = [self.get_layers()[idx] for idx in layer.get_input_idx()]
                for in_name in layer.get_onnx_input_name():
                    for ilayer in input_layers:
                        if in_name in ilayer.get_onnx_output_name():
                            idx = ilayer.get_onnx_output_name().index(in_name)
                            in_scale.append(ilayer.get_scale()[idx])
                            if not ilayer.is_extend():
                                in_quantizes.append(ilayer.get_quantize()['feat']['so'+str(idx)])
                            else:
                                in_quantizes.append(ilayer.get_quantize())

            layer.set_in_scale(in_scale)
            # user define layer
            if layer.is_extend():
                continue

            w_scale = layer.get_w_scale()
            scale = layer.get_scale()
            setting = copy.deepcopy(default_setting)
            ops_string, attrs = ops['ops'], ops['attrs']
            key = layer_type if layer_type in self._settings.keys() else 'default'
            setting.update(self._settings[key]['feat'])
            ops_setting = dict(
                is_result_layer=False,
                process_scale=self._settings[key]['process_scale'],
                precision=self._settings[key]['precision'],
                int_scale=self._settings[key]['int_scale'],
                in_quantize=None, quantize=None,)
            setting.update(ops_setting)

            if layer.get_is_result_layer():
                if layer.get_layer_type().lower() in self._output['layer_type']:
                    setting['process_scale'] = self._output['process_scale']
                    setting['is_result_layer'] = True
            # setting.update({'in_quantize': layer.get_in_quantize(), 'quantize': layer.get_quantize()})
            # setting function
            # quantize/correct/in data type/out data type based in layer
            settings = {'in_scale': in_scale, 'w_scale': w_scale, 'scale': scale,
                        'ops_string': ops_string, 'attrs': attrs, 'setting': setting, "txme_saturation":self._txme_saturation}
            # todo zero point not implement
            layer.setting_ops(settings)
            # save ops settings
            layer.set_ops_setting(settings)
            # layer.instance_layer_ops()
            # self.__layers[idx] = layer

    # update float/int scale when feat and weight update
    # update quan setting using layer idx as the key
    def update_quan_ops(self, quan_dict: dict):
        # after update si/sk/so, must process scale
        # modify scale method, update post process parameter
        for idx, layer in enumerate(self.get_layers()):
            # update ops setting
            ops = layer.get_layer_ops()
            layer_type = layer.get_layer_type().lower()
            in_quantizes = list()
            if layer_type == 'data':
                in_scale = [1.0]
                in_quantizes = list()
                # si = dict()
            else:
                # fixed multi in/out connect, then pre and next layer scale length gt 1
                in_scale = list()
                input_layers = [self.get_layers()[idx] for idx in layer.get_input_idx()]
                for in_name in layer.get_onnx_input_name():
                    for ilayer in input_layers:
                        if in_name in ilayer.get_onnx_output_name():
                            idx = ilayer.get_onnx_output_name().index(in_name)
                            in_scale.append(ilayer.get_scale()[idx])
                            if not ilayer.is_extend():
                                in_quantizes.append(ilayer.get_quantize()['feat']['so' + str(idx)])
                            else:
                                in_quantizes.append(ilayer.get_dequan_output())

            # user define layer
            if layer.is_extend():
                layer.set_in_scale(in_scale)
                continue

            w_scale = layer.get_w_scale()
            scale = layer.get_scale()
            ops_string, attrs, setting = ops['ops'], ops['attrs'], dict()
            # update about quantize setting
            if idx in quan_dict.keys():
                setting.update(quan_dict[idx])

            # update process scale for simulations
            if 'process_scale' in quan_dict[idx].keys():
                layer.set_scale_type(quan_dict[idx]['process_scale'])
                if quan_dict[idx]['process_scale'] == 'smooth':
                    val = max(in_scale)
                    val_ = [val for _ in range(len(scale))]
                    layer.set_scale(val_)
                    scale = val_
            # update about calc quantize result setting
            setting.update({'in_scale': in_scale, 'w_scale': w_scale, 'scale': scale})

            layer.set_in_quantize(in_quantizes)
            # settings = dict(in_scale=in_scale, w_scale=w_scale, scale=scale, ops_string=ops_string,
            #                 attrs=attrs, setting=setting)
            settings = layer.get_ops_setting()
            settings.update(setting)
            layer.set_ops_setting(settings)
            layer.instance_layer_ops()
            # self.__layers[idx] = layer
            # update ops inner calculate scale for each layer

    # # high level quantize for all action
    # def quan_layer(self):
    #     self.quan_weights()
    #     self.quan_feats()
    #     # correct data shift/scale
    #     # pre and post process for all ops in layer
    #     self.quan_ops()

    # update quantize method, analysis error modify quantize method, ex: int scale -> float scale
    # some layer quantize update or all layer update
    # update one of [si, sk, so], must be update quantize ops, scale=si*sk/so
    # todo zero point not considered
    def update(self, weights_dict: dict, feat_dict: dict, ops_dict: dict):
        self.update_quan_feats(feat_dict) # type: ignore
        self.update_quan_weights(weights_dict)
        self.update_quan_ops(ops_dict)
        self._is_quantized = True

    # analysis feat and weight, optimizer quantize scale
    def analysis_weights(self, weights):
        mu = np.mean(weights)
        sigma = np.std(weights)
        n = 3
        threshold1 = mu - n * sigma
        threshold2 = mu + n * sigma

        outlier = []  # 将异常值保存
        outlier_x = []

        for i in range(0, len(weights)):
            if (weights[i] < threshold1) | (weights[i] > threshold2):
                outlier.append(weights[i])
                outlier_x.append(weights[i])
            else:
                continue

        return outlier

    # combination x, h quantize parameter
    def combine_lstm(self, data_scales):
        for layer in self.get_layers():
            if layer.get_layer_type().lower() == 'lstm':
                names = layer.get_onnx_input_name()[:2]
                scales = [data_scales[name] for name in names]
                max_value = np.max(scales[0]['max'], scales[1]['max'])
                min_value = np.min(scales[0]['min'], scales[1]['min'])
                values = {'max': max_value, 'min': min_value, 'zeros_point': 0}
                for name in names:
                    data_scales[name] = values

        return data_scales

    # quantize interface function, base or default quantize
    def quantize(self, data_scales, weight_scale_dict=None):
        # self.quan_layer()
        # data_scales = self.combine_lstm(data_scales)
        self.quan_feats(data_scales)
        self.quan_weights(weight_scale_dict=weight_scale_dict)
        # for layer in self.get_layers():
        #     print("####", layer.layer_idx)
        #     print(layer.in_scale)
        #     print(layer.w_scale)
        #     print(layer.scale)
        self.quan_ops()
        self._is_quantized = True

    # save qiantized graph to binary files
    def save(self, fd_path="work_dir/resume"):
        try:
            if not os.path.exists(fd_path):
                os.makedirs(fd_path)
                
            for origin_layer in self.get_layers():
                self.logger.info("dump layer name is: {}".format(origin_layer.get_layer_name()))
                # if origin_layer.get_layer_type() == "resize":
                #     print("test")
                dump_layer(origin_layer, fd_path)
            return True
        except:
            self.logger.error("save layer of {} failed".format(origin_layer.get_layer_name()))
            return False

    def reset(self, layer_type):
        for layer in self.get_layers():
            if layer.get_layer_type() == layer_type:
                layer.reset()

    # reload quantized graph from binary file
    # load quantize and alignment key-value to instance class
    # load parameter from binary dict to instance class
    def reload(self, fd_path="./work_dir/resume"):
        try:
            if not os.path.exists(fd_path):
                self.logger.info("resume path is not exists!")
                return False
            origin_layers = self.get_layers()
            for idx, origin_layer in enumerate(origin_layers):
                self.logger.info("reload layer name is: {}".format(origin_layer.get_layer_name()))
                origin_layers[idx] = reolad_layer(origin_layer, fd_path)
            self.set_layers(origin_layers)
            return True
        except:
            self.logger.error("reload layer of {} failed".format(origin_layer.get_layer_name()))
            return False

def get_first_special_layer_idx(layers: list, layer_type: str = "lstm"):
    layer_types = [_layer.get_layer_type() for _layer in layers]
    first_special_layer_idx = layer_types.index(layer_type) if layer_type in layer_types else -1
    return first_special_layer_idx

def get_first_special_layer_name(layers: list, layer_type: str = "lstm"):
    layer_types = [_layer.get_layer_type() for _layer in layers]
    first_special_layer_idx = layer_types.index(layer_type) if layer_type in layer_types else -1
    layer_name = layers[first_special_layer_idx].get_layer_name() if first_special_layer_idx != -1 else None
    return layer_name

# this method using new quantize method interface
# layer settings will transfer dictionary of scale in graph to layer operations
# will fixed smooth process scale transfer error status
class GrapQuantUpgrade(GraphQuant):
    def __init__(self, **kwargs):
        super(GrapQuantUpgrade, self).__init__(**kwargs)
        self.__search_smaller_sk = kwargs.get("search_smaller_sk", False)
        self.__reload_sk_params = kwargs.get("reload_sk_params", False)
        self.__sk_params_json_path = kwargs.get("sk_params_json_path", "")
        if self.__reload_sk_params:
            self.sk_params = json.load(open(self.__sk_params_json_path, "r"))  
            print("reload sk_params from: {}".format(self.__sk_params_json_path))
            print(self.sk_params)        
        self.__fuse_act = kwargs['fuse_act']
        self.__input_names = kwargs['input_names']
        self.layer_cle_list = []

    def get_graph(self):
        return self._graph

    # process feature map quantize, also smooth pre and next layer scale
    def feature_process(self, layer: object, quan_dict: dict, dataset_scales: dict, quan_factory: quantize_factory, is_first_lstm_layer: bool = False): # type: ignore
        # scale is quantize parameter of dictionary
        in_quantizes, in_scale = list(), list()
        # scale, quantize, data_scale = list(), dict(), list()
        # todo shuffle or shuffle-only scale not ready
        # if layer.get_layer_name() == 'layernorm4':
        #     print('test')
        out_layer_names = [self.get_layers()[o_idx].get_layer_name() for o_idx in layer.get_output_idx()]
        out_layer_types = [self.get_layers()[o_idx].get_layer_type() for o_idx in layer.get_output_idx()]
        if layer.get_layer_type().lower() == 'data': # type: ignore
            # support asymmetric quantize method, and interface will
            # todo just for symmetric quantize, we will implement asymmetric method

            data_quant_method = quan_factory.get('base')(bit_select=quan_dict['bit_select'])
            in_scale.append(data_quant_method.get_quant_param())
            in_quantizes.append(data_quant_method)
            # si = dict()
        else:
            # fixed multi in/out connect, then pre and next layer scale length gt 1
            in_scale = list()
            input_layers = [self.get_layers()[idx] for idx in layer.get_input_idx() if idx != -1] # type: ignore
            ilayer_names = flatten_list([ilayer.get_onnx_output_name() for ilayer in input_layers])
            for cur_idx, in_name in enumerate(layer.get_onnx_input_name()): # type: ignore
                if in_name in ilayer_names:
                    for ilayer in input_layers:
                        if in_name in ilayer.get_onnx_output_name():
                            idx = ilayer.get_onnx_output_name().index(in_name)
                            in_scale.append(ilayer.get_scale()[idx])
                            if not ilayer.is_extend():
                                in_quantizes.append(ilayer.get_quantize()['feat']['so' + str(idx)])
                            else:
                                in_quantizes.append(ilayer.get_quantize())
                else:
                    first_lstm_idx = get_first_special_layer_idx(self.get_layers(), layer_type="lstm")
                    quan_dict_ = copy.deepcopy(quan_dict)
                    if first_lstm_hidden_int16_enable and first_lstm_idx == layer.get_idx():
                        quan_dict_['bit_select'] = 3
                    # if layer.get_layer_name() == "ReplaceReshapeLikeOps_Reshape_5_0":
                    # if layer.get_layer_type() == "reshape":
                    #     quan_dict_['bit_select'] = 3

                    # if "ReplaceReshapeLikeOps_Reshape_7_0" in out_layer_names or \
                    #    "ReplaceReshapeLikeOps_Reshape_5_0" in out_layer_names or \
                    #    "ReplaceReshapeLikeOps_Reshape_3_0" in out_layer_names or \
                    #    "ReplaceReshapeLikeOps_Reshape_1_0" in out_layer_names:
                    #        quan_dict_['bit_select'] = 1
                    quan_method = get_quant(quan_dict_, quan_factory, dataset_scales, in_name)
                    init_scale = quan_method.get_scale()
                    in_scale.append(dict(scale=init_scale[0], zero_point=init_scale[1]))
                    in_quantizes.append(quan_method)

        layer.set_in_scale(in_scale) # type: ignore
        # in quantities just using in de-quantize float
        layer.set_in_quantize(in_quantizes) # type: ignore
        setting = copy.deepcopy(self.default_setting)
        layer_type = layer.get_layer_type().lower() # type: ignore
        key = layer_type if layer_type in self._settings.keys() else 'default'
        setting.update(self._settings[key]['feat'])
        len_scale = len(layer.get_onnx_output_name()) # type: ignore
        in_scale_v = [value['scale'] for value in in_scale]
        if self._settings[key]['process_scale'] == 'smooth':
            val = max(in_scale_v)
            replace_scale = [in_scale[in_scale_v.index(val)] for _ in range(len_scale)]
            quantizes, feat = layer.get_quantize(), dict() # type: ignore
            for idx in range(len_scale):
                feat['so' + str(idx)] = in_quantizes[in_scale_v.index(val)]
            quantizes['feat'] = feat
            layer.set_quantize(quantizes) # type: ignore
            layer.set_scale(replace_scale) # type: ignore
            nodes = layer.get_nodes()
            if len(nodes) > 1:
                out_name = nodes[-1].get_onnx_output()[0]
            else:
                out_name = nodes[0].get_onnx_output()[0]
            MinMaxValue = [dataset_scales[out_name]]
            layer.set_data_scale(MinMaxValue)            
        else:
            # out_names = layer.get_onnx_output_name() # type: ignore
            layer_types = [layer_.get_layer_type() for layer_ in self.get_layers()]
            # first_lstm_idx = layer_types.index('lstm') if 'lstm' in layer_types else -1
            quan_dict_ = copy.deepcopy(quan_dict)
            # if first_lstm_idx in layer.get_output_idx():
            # out_layer_names = [self.get_layers()[o_idx].get_layer_name() for o_idx in layer.get_output_idx()]
            # if "ReplaceReshapeLikeOps_Reshape_5_0" in out_layer_names or \
            #    "ReplaceReshapeLikeOps_Reshape_3_0" in out_layer_names or \
            #    "ReplaceReshapeLikeOps_Reshape_1_0" in out_layer_names:
            # if layer.get_layer_type() in ["lstm"]:
            #        quan_dict_['bit_select'] = 1
            if "lstm" in out_layer_types:
                if not self._settings["lstm"]['hx_combine'] and \
                    not is_first_lstm_layer:
                    quan_dict_["method"] = "floatquan"
                if first_lstm_feat_int16_enable:
                    if layer.get_layer_type() == "data":
                        quan_dict_['bit_select'] = 3

            out_names = layer.get_onnx_output_name()
            feat_dict = dict()
            scale, quantize, MinMaxValue = list(), dict(), list()
            for idx, name in enumerate(out_names):
                key = 'so' + str(idx)
                ## process sigmoid(x) == 0
                # diff = dataset_scales[name]["max"] - dataset_scales[name]["min"]
                # if 0 < diff < 0.5:
                #     dataset_scales[name]["min"] = dataset_scales[name]["max"] - 0.5
                # output_layer_types = []
                # for t_layer in self.get_layers():
                #     if name in t_layer.get_onnx_input_name():
                #         if t_layer.get_layer_type() in ["reshape"]:
                #             next_output_types = \
                #                 [self.get_layers()[t_idx].get_layer_type() for t_idx in t_layer.get_output_idx()]
                #             output_layer_types.extend(next_output_types)
                #         else:
                #             output_layer_types.append(t_layer.get_layer_type())
                # if "fc" in output_layer_types:
                #     quan_dict_['bit_select'] = 1
                # if layer.get_layer_name() in ["LSTM_7", "LSTM_5", "LSTM_3", "LSTM_1"]:
                #     quan_dict_['bit_select'] = 1
                quan_method = get_quant(quan_dict_, quan_factory, dataset_scales, name)

                MinMaxValue.append(dataset_scales[name])
                feat_dict[key] = quan_method.get_quant_param()
                scale.append(feat_dict[key])
                quantize[key] = quan_method

            if layer.get_layer_type() in ["conv", "depthwiseconv", "convtranspose", "fc"]: # type: ignore
                if layer.get_layer_ops()["ops"][-1] in self.__fuse_act: # type: ignore
                    nodes = layer.get_nodes()
                    if len(nodes) >= 2 and nodes[1].get_op_type() == "BatchNormalization":
                        conv_output_name = nodes[1].get_onnx_output()[0]
                    else:
                        conv_output_name = nodes[0].get_onnx_output()[0]                    
                    key = 'sc' + str(0)
                    quan_dict_cpy = copy.deepcopy(quan_dict_)
                    quan_dict_cpy['method'] = quan_dict["method"]
                    quan_method = get_quant(quan_dict_cpy, quan_factory, dataset_scales, conv_output_name)
                    quan_method.get_quant_param()
                    quantize[key] = quan_method
                    MinMaxValue.append(dataset_scales[conv_output_name])
                # print("test")

            layer.set_scale(scale) # type: ignore
            layer.set_data_scale(MinMaxValue) # type: ignore
            layer.set_quantize(dict(feat=quantize)) # type: ignore

    @staticmethod
    def quant_lstm(layer, hx_combine, wr_combine, quan_method):
        W, R, Wb, Rb = layer.get_layer_ops()['weights']
        # R_quan_method = copy.deepcopy(quan_method)

        if wr_combine:
            weights = np.column_stack([W.reshape(1, -1), R.reshape(1, -1)])
            quan_method.get_quan_param(weights, is_aciq=False)
            R_quan_method = copy.deepcopy(quan_method)
        else:
            R_quan_method = copy.deepcopy(quan_method)
            quan_method.get_quan_param(W, is_aciq=False)
            R_quan_method.get_quan_param(R)

        w_scales = [quan_method.get_quant_param(), R_quan_method.get_quant_param()]
        layer.set_w_scale(w_scales)
        si = layer.get_in_scale()
        qW, qR = quan_method.get_quan_data(W, is_squant=False), R_quan_method.get_quan_data(R, is_squant=False)

        qWB = Wb / (si[0]['scale'] * w_scales[0]['scale'])
        if not hx_combine:
            zero_point = np.zeros((1, qW.shape[2])) + si[0]["zero_point"]
            qWB -= np.matmul(zero_point, qW.squeeze(0).transpose(1,0))
        h_si = si[0]['scale'] if hx_combine else si[1]['scale']
        h_sk = w_scales[0]['scale'] if wr_combine else w_scales[1]['scale']
        qRB = Rb / (h_si * h_sk)

        qWB, qRB = np.round(qWB).astype(np.float32), np.round(qRB).astype(np.float32)
        layer.set_qbias([qWB, qRB])
        # layer.set_qbias([Wb, Rb])
        layer.set_qweights([qW, qR])
        # layer.set_qweights([W, R])
        layer.set_weight([W, R])
        quantizes =dict(w_quan=[quan_method, R_quan_method])

        return quantizes

    # add zero-point process
    def weights_process(self, layers, layer, quan_dict):
        # if layer.get_layer_type().lower() in ['fc', 'matmul', 'gemm']:
        #     print('test')
        quantizes = layer.get_quantize()
        # if layer.get_layer_type().lower() == 'depthwiseconv':
        #     print('test')
        # if layer.get_idx() == 64:
        #     print('test')
        is_result_layer = layer.get_is_result_layer()
        is_output_type = layer.get_layer_type().lower() in self._output['layer_type']
        # if is_result_layer and is_output_type:
        #     quan_dict['method'] = copy.deepcopy(self.__output['weights']['method'])
        #     quan_dict['bit_select'] = copy.deepcopy(self.__output['weights']['bit_select'])
        bit_select=quan_dict['bit_select']
        # if layer.get_layer_type() in ['lstm', 'gru']:
        #     bit_select = 3
        # if layer.get_layer_type() in ['fc']:
        #     bit_select = 1
        # if layer.get_layer_name() in ["LSTM_7", "LSTM_5", "LSTM_3", "LSTM_1"]:
        #     bit_select = 1
        # if layer.get_layer_name() in ["LSTM_1"]:
        #     bit_select = 3

        quan_method = quantize_factory.get(quan_dict['method'])( # type: ignore
            bit_select=bit_select, # type: ignore
            bits_dict=quan_dict['bits_dict'], # type: ignore
            maxs=quan_dict['maxs'], mins=quan_dict['mins']) # type: ignore

        if layer.get_layer_type() in ['lstm', 'gru']:
            lstm_quantizes = self.quant_lstm(layer, hx_combine=quan_dict['hx_combine'], wr_combine=quan_dict['wr_combine'], quan_method=quan_method)
            quantizes.update(lstm_quantizes)
        else:
            weights = layer.get_layer_ops()['weights'][0]
            if layer.get_layer_type().lower() in ['matmul', 'gemm', 'fc']:
                if 'transB' in layer.get_layer_ops()['attrs'][0].keys():
                    transB = layer.get_layer_ops()['attrs'][0]['transB']
                    if not transB:
                        weights = np.transpose(weights, (1, 0))
                        self.logger.warning('Onnx fc weights transpose!')
                else:
                    if layer.get_layer_type().lower() not in ["matmul"]:
                        weights = np.transpose(weights, (1, 0))
                    self.logger.warning('Onnx fc weights transpose!')

            # weights[weights < 1e-4] = 0
            if not isinstance(weights, np.ndarray):
                return
            
            # lower_accuracy_layers = {
            #     "/features/features.1/conv/conv.0/Conv": 17.681514739990234,
            #     "/features/features.3/conv/conv.3/Conv": 26.139562606811523,
            #     "/features/features.8/conv/conv.3/Conv": 20.732975006103516,
            #     "/features/features.9/conv/conv.3/Conv": 7.606176853179932,
            #     "/features/features.10/conv/conv.3/Conv": 8.145543098449707,
            #     "/features/features.11/conv/conv.3/Conv": 16.70775032043457,
            #     "/features/features.12/conv/conv.3/Conv": 13.934417724609375,
            #     "/features/features.13/conv/conv.3/Conv": 23.490482330322266,
            #     "/features/features.14/conv/conv.3/Conv": 30.618032455444336,
            #     "/features/features.15/conv/conv.3/Conv": 16.539987564086914,
            #     "/features/features.16/conv/conv.3/Conv": 34.749847412109375,
            # }
            # from scipy.stats import laplace, kstest
            # mu, sigma = np.mean(weights), np.std(weights)
            # layer_name = layer.get_layer_name()
            # print(layer_name, mu, sigma)
            # if layer_name in lower_accuracy_layers.keys() and lower_accuracy_layers[layer.get_layer_name()] > 30.0:
            # if sigma > 1.0:
                # mu, sigma = np.mean(weights), np.std(weights)
                # ks_statistic, ks_p_value = kstest(weights.reshape(-1), 'laplace', args=(np.mean(weights), np.std(weights)))
                # quan_method.get_quan_param(weights, is_aciq=True)
            # else:
            #     quan_method.get_quan_param(weights, is_aciq=False)
            quan_method.get_quan_param(weights, is_aciq=False)
            
            w_scales = quan_method.get_quant_param()
            layer.set_w_scale(w_scales)
            # weights = layer.get_layer_ops()['weights'][0]
            # quan_method.get_quan_param(weights)
            # layer.set_w_scale(quan_method.get_scale()[0])
            q_weights = quan_method.get_quan_data(weights)
            layer.set_qweights(q_weights)
            layer.set_weight(weights)
            has_bias = layer.get_layer_ops()['attrs'][0].get('bias')
            if has_bias:
                # zero-point transfers
                # print(layer_type, layer.get_idx())
                bias = layer.get_layer_ops()['weights'][1]
                layer.set_bias(bias)
                si = layers[layer.get_input_idx()[0]].get_scale()[0]
                qbias = bias / (si['scale'] * w_scales['scale'])
                qbias = np.round(qbias)
                layer.set_qbias(qbias.astype(np.float32))

            quantizes.update(dict(w_quan=quan_method))
            layer.set_quantize(quantizes)

    # some layer has no quantize needed
    # then we will copy quantize from prev quantize to current layer
    def quan_feats(self, dataset_scales: dict):
        # kwargs = dict(bit_select=self.bit_select, maxs=self.maxs, mins=self.mins, bit_dict=self.bit_dict)
        is_first_lstm_layer = True
        for idx, layer in enumerate(tqdm(self.get_layers(), postfix="quant feat")):
            try:
                layer.set_inputs_names(self.__input_names)
                # if layer.get_layer_type().lower() == 'depthwiseconv':
                #     print('test')
                layer_type = layer.get_layer_type().lower()

                # user define quantize feature map
                if layer.is_extend():
                    layer.quan_feat()
                else:
                    key = layer_type if layer_type in self._settings.keys() else 'default'
                    quan_dict = copy.deepcopy(self._settings[key]['feat'])
                    # todo will investigation last convolution layer quantize medthod
                    # if layer.get_is_result_layer():
                    #     if layer.get_layer_type().lower() in self.__output['layer_type']:
                    #         quan_dict['method'] = copy.deepcopy(self.__output['feat']['method'])
                    #         quan_dict['bit_select'] = copy.deepcopy(self.__output['feat']['bit_select'])
                    self.feature_process(layer, quan_dict, dataset_scales, quantize_factory, is_first_lstm_layer=is_first_lstm_layer)
                if layer_type == "lstm":
                    is_first_lstm_layer = False
            except:
                error_info = "layer of {} quantize feature map wrong!".format(layer.get_layer_name())
                print(error_info)
                os._exit(-1)

    # update feat quantize length or method
    # layer idx as the key
    def update_quan_feats(self, quan_dict: dict, dataset_scales: dict):
        for key in quan_dict.keys():
            layer = self.get_layers()[key]
            # layer: object, quan_dict: dict, dataset_scales: dict, quan_factory: object
            # user define quantize feature map
            if layer.is_extend():
                layer.quan_feat()
            else:
                self.feature_process(layer, quan_dict[key], dataset_scales, quantize_factory)

    def quan_weights(self, weight_scale_dict = None):
        # kwargs = dict(bit_select=self.bit_select, maxs=self.maxs, mins=self.mins, bit_dict=self.bit_dict)
        is_first_lstm = False
        for idx, layer in enumerate(self.get_layers()):
            try:
                layer_type = layer.get_layer_type().lower()
                # user define quantize weights
                if layer.is_extend():
                    layer.quan_weights()
                else:
                    key = layer_type if layer_type in self._settings.keys() else 'default'
                    weights_setting = copy.deepcopy(self._settings[key]['weights'])
                    if layer_type in ["splice"] and not layer.get_layer_ops()['attrs'][0]["has_fc"]:
                        weights_setting = None
                    if weights_setting:
                        # if layer.get_is_result_layer():
                        #     quan_dict = copy.deepcopy(weights_setting)
                        #     quan_dict['method'] = 'floatsymquan'
                        if layer.get_layer_type() in ["lstm", "gru"]:
                            setting_ = copy.deepcopy(self._settings[key])
                            weights_setting['hx_combine'] = setting_['hx_combine']  if 'hx_combine' in setting_.keys() else True
                            weights_setting['wr_combine'] = setting_['wr_combine']  if 'wr_combine' in setting_.keys() else True
                            if not is_first_lstm and first_lstm_weight_int16_enable:
                                weights_setting['bit_select'] = 3
                                is_first_lstm = True
                        self.weights_process(self.get_layers(), layer, weights_setting)
                    else:
                        bit_select = self._settings[key]['feat']['bit_select']
                        data_quant_method = quantize_factory.get('base')(bit_select=bit_select) # type: ignore
                        layer.set_w_scale(data_quant_method.get_quant_param())
            except:
                error_info = "layer of {} quantize weight wrong!".format(layer.get_layer_name())
                print(error_info)
                os._exit(-1)

    # instance ops
    # calc float/int scale data correct
    def quan_ops(self):
        layers = self.get_layers()
        for idx, layer in enumerate(layers):
            try:
                # if hasattr(layer, 'quan_weight') or hasattr(layer, 'quan_feat'):
                #     layer.quan_weight()
                #     continue
                ops = layer.get_layer_ops()
                # layer_type = layer.get_layer_type().lower()
                # if layer.get_idx() == 64:
                #     print('test')

                # user define layer
                if layer.is_extend():
                    continue

                # in_scale = layer.get_in_scale()
                # w_scale = layer.get_w_scale()
                # scale = layer.get_scale()
                setting = copy.deepcopy(self.default_setting)
                # ops_string, attrs = ops['ops'], ops['attrs']
                layer_type = layer.get_layer_type().lower()
                key = layer_type if layer_type in self._settings.keys() else 'default'
                setting.update(self._settings[key]['feat'])
                setting['w_bit_select'] = setting['bit_select'] # type: ignore
                quant_method = [self._settings[key]['feat']['method']]
                if self._settings[key]['weights']:
                    setting['w_bit_select'] = self._settings[key]['weights']['bit_select'] # type: ignore
                    quant_method.append(self._settings[key]['weights']['method'])

                ms = ['sym' in method for method in quant_method]
                txme_saturation = copy.deepcopy(self._txme_saturation)
                if np.sum(np.array(ms)) < len(ms):
                    txme_saturation = 0

                if layer_type in ["lstm", "gru"]:
                    setting_ = self._settings[key]
                    setting.update(dict(hx_combine=setting_['hx_combine'], wr_combine=setting_['wr_combine'])) # type: ignore

                process_scale = copy.deepcopy(self._settings[key]['process_scale'])
                if layer.get_layer_type() in ["resize"]:
                    mode = layer.get_nodes()[-1].get_attr()['mode']
                    if mode in ['nearest']:
                        process_scale = "smooth"
                        layer.set_scale_type(process_scale)

                if layer_type in ["conv", "fc", "depthwiseconv", "convtranspose"]:
                    # process_scale = "shiftfloatscaletable"
                    if ops['ops'][-1] not in self._act:
                        process_scale = "shiftfloatscaletable2float" \
                            if process_scale == "shiftfloatscaletable2float" else "shiftfloatscaletable"


                layer_name = get_first_special_layer_name(self.get_layers(), "lstm")
                is_update_quantize_from_in_data = True if layer_name == layer.get_layer_name() else False
                ops_setting = dict(
                    is_result_layer=False,
                    process_scale=process_scale,
                    out_type=self._settings[key]['out_type'],
                    int_scale=self._settings[key]['int_scale'],
                    in_quantize=None, quantize=None,
                    virtual_round=self._virtual_round,
                    is_update_quantize_from_in_data=is_update_quantize_from_in_data,
                    txme_saturation=txme_saturation)
                # 3
                setting.update(ops_setting)
                # output_type = self.__settings[key]['out_type']
                # if layer_type == 'data':### added by henson
                #     # get input data type from input data ndarray data type
                #     pass
                # else:
                #     input_idx = layer.get_input_idx()[0]
                #     layer.set_input_type(layers[input_idx].get_output_type())

                layer.set_output_type(self._settings[key]['out_type'])
                layer.set_scale_type(process_scale) # type: ignore
                if layer.get_layer_type() in ["conv", "convtranspose", "depthwiseconv"]:
                    layer.set_first_conv(False)
                    pre_layer = layers[layer.get_input_idx()[0]]
                    if pre_layer.get_layer_type() == "data":
                       layer.set_first_conv(True)

                if layer.get_is_result_layer():
                    if layer.get_layer_type().lower() in self._output['layer_type']:
                        process_scale = self._output['process_scale']

                        if layer_type in ["conv", "depthwiseconv", "convtranspose", "fc"]:
                            if ops['ops'][-1] not in self._act:
                                process_scale = "shiftfloatscaletable2float"
                        setting.update(
                            dict(is_result_layer=True,
                                 txme_saturation=txme_saturation,
                                 process_scale=process_scale),
                                 out_type=self._output['out_type'])
                        # setting['process_scale'] = self.__output['process_scale']
                        layer.set_scale_type(process_scale)

                # setting.update({'in_quantize': layer.get_in_quantize(), 'quantize': layer.get_quantize()})
                # setting function
                # quantize/correct/in data type/out data type based in layer
                settings = dict(in_scale=layer.get_in_scale(),
                                w_scale=layer.get_w_scale(),
                                scale=layer.get_scale(),
                                ops_string=ops['ops'],
                                attrs=ops['attrs'],
                                setting=setting)

                if (self.__search_smaller_sk or self.__reload_sk_params) and layer.get_layer_type() in ["conv", "depthwiseconv", "convtranspose", "fc"]: 
                    w_bit_select = settings['setting']['w_bit_select']
                    if w_bit_select == 1:
                        max_value, min_value = 2**7 - 1, -2**7
                    else:
                        max_value, min_value = 2**15 - 1, -2**15
                        
                    def get_quan_data(data, scale, zero_point=0):
                        transformed_val = data.reshape(-1) / scale + zero_point
                        quantized = np.round(transformed_val)
                        quantized = np.clip(quantized, min_value, max_value)
                        return np.reshape(quantized, data.shape)

                    def get_dequan_data(data, scale, zero_point=0):
                        dequantize = (data.reshape(-1).astype(np.float32) - zero_point) * scale
                        return np.reshape(dequantize, data.shape)
                                        
                    weight = layer.get_weight()
                    bias = layer.get_bias()                    

                    if self.__search_smaller_sk:
                        def Cosine_distance(simulation_data, true_data, eps=1.0e-5):
                            import numpy as np
                            from scipy.spatial.distance import cosine

                            dist = cosine(simulation_data.reshape(-1), true_data.reshape(-1))

                            return np.float32(dist)
                        
                        si, sk, so = layer.get_in_scale()[0]['scale'], layer.get_w_scale()['scale'], layer.get_scale()[0]['scale']
                        so = layer.get_quantize()['feat']['sc0'].get_scale()[0]
                        out_shift, out_scale = get_scale_shift(si * sk / so) # type: ignore
                        gap = sk * 0.5 / (2 * out_scale * 1000)
                        data = copy.deepcopy(weight)
                        qdata = get_quan_data(data, scale=sk, zero_point=layer.get_w_scale()['zero_point'])
                        qdata = get_dequan_data(qdata, scale=sk, zero_point=layer.get_w_scale()['zero_point'])
                        error = Cosine_distance(qdata, data)                    
                        for i in range(1000):
                            scale_ = sk * 0.5 / (2 * out_scale) + gap * i
                            data = copy.deepcopy(weight)
                            qdata = get_quan_data(data, scale=scale_, zero_point=layer.get_w_scale()['zero_point'])
                            qdata = get_dequan_data(qdata, scale=scale_, zero_point=layer.get_w_scale()['zero_point'])
                            error_ = Cosine_distance(qdata, data)
                            # out_shift_, out_scale_ = get_scale_shift(si * scale_ / so) # type: ignore
                            if error_ <= 6 * error:
                                sk = scale_
                                break
                    elif self.__reload_sk_params:
                        layer_name = layer.get_layer_name()
                        si = layer.get_in_scale()[0]['scale']
                        sk = self.sk_params[layer_name] / max_value
                    else:
                        print("please set search_smaller_sk or reload_sk_params to true!!!")
                        os._exit(-1)
                            
                    layer.set_w_scale(dict(scale=sk, zero_point=layer.get_w_scale()['zero_point']))
                    qbias = bias / (si * sk)
                    qbias = np.round(qbias)
                    layer.set_qbias(qbias.astype(np.float32))
                    qweight = get_quan_data(copy.deepcopy(weight), scale=sk, zero_point=layer.get_w_scale()['zero_point'])
                    qweight = qweight.astype(layer.get_qweight().dtype)
                    layer.set_qweights(qweight)

                # zero point already implement
                layer.setting_ops(settings)
                # save ops settings
                layer.set_ops_setting(settings)
            except:
                error_info = "layer of {} quantize alignment wrong!".format(layer.get_layer_name())
                print(error_info)
                os._exit(-1)

    # update float/int scale when feat and weight update
    # update quan setting using layer idx as the key
    def update_quan_ops(self, quan_dict: dict):
        # after update si/sk/so, must process scale
        # modify scale method, update post process parameter
        for idx, layer in enumerate(self.get_layers()):
            # update ops setting
            ops = layer.get_layer_ops()
            layer_type = layer.get_layer_type().lower()
            in_quantizes = list()
            if layer_type == 'data':
                in_scale = [1.0]
                in_quantizes = list()
                # si = dict()
            else:
                # fixed multi in/out connect, then pre and next layer scale length gt 1
                in_scale = list()
                input_layers = [self.get_layers()[idx] for idx in layer.get_input_idx()]
                for in_name in layer.get_onnx_input_name():
                    for ilayer in input_layers:
                        if in_name in ilayer.get_onnx_output_name():
                            idx = ilayer.get_onnx_output_name().index(in_name)
                            in_scale.append(ilayer.get_scale()[idx])
                            if not ilayer.is_extend():
                                in_quantizes.append(ilayer.get_quantize()['feat']['so' + str(idx)])
                            else:
                                in_quantizes.append(ilayer.get_dequan_output())

            # user define layer
            if layer.is_extend():
                layer.set_in_scale(in_scale)
                continue

            w_scale = layer.get_w_scale()
            scale = layer.get_scale()
            ops_string, attrs, setting = ops['ops'], ops['attrs'], dict()
            # update about quantize setting
            if idx in quan_dict.keys():
                setting.update(quan_dict[idx])

            # update process scale for simulations
            if 'process_scale' in quan_dict[idx].keys():
                layer.set_scale_type(quan_dict[idx]['process_scale'])
                if quan_dict[idx]['process_scale'] == 'smooth':
                    val = max(in_scale)
                    val_ = [val for _ in range(len(scale))]
                    layer.set_scale(val_)
                    scale = val_
            # update about calc quantize result setting
            setting.update({'in_scale': in_scale, 'w_scale': w_scale, 'scale': scale})

            layer.set_in_quantize(in_quantizes)
            # settings = dict(in_scale=in_scale, w_scale=w_scale, scale=scale, ops_string=ops_string,
            #                 attrs=attrs, setting=setting)
            settings = layer.get_ops_setting()
            settings.update(setting)
            layer.set_ops_setting(settings)
            layer.instance_layer_ops()
            # self.__layers[idx] = layer
            # update ops inner calculate scale for each layer

    # # high level quantize for all action
    # def quan_layer(self):
    #     self.quan_weights()
    #     self.quan_feats()
    #     # correct data shift/scale
    #     # pre and post process for all ops in layer
    #     self.quan_ops()

    # update quantize method, analysis error modify quantize method, ex: int scale -> float scale
    # some layer quantize update or all layer update
    # update one of [si, sk, so], must be update quantize ops, scale=si*sk/so
    def update(self, weights_dict: dict, feat_dict: dict, ops_dict: dict):
        self.update_quan_feats(feat_dict) # type: ignore
        self.update_quan_weights(weights_dict)
        self.update_quan_ops(ops_dict)
        self.__is_quantized = True


# 
class AlreadyGrapQuant(GrapQuantUpgrade):
    def __init__(self, **kwargs):
        super(AlreadyGrapQuant, self).__init__(**kwargs)
        # layer ops setting
        self.__settings = kwargs['default_setting']
        self.__graph = kwargs['graph']
        self.__output = kwargs['output']
        self.__act = kwargs['act']
        self.__fuse_act = kwargs['fuse_act']
        self.__is_quantized = False
        self.logger.info(self.__settings)

    def quan_weights(self, weight_scale_dict=None):
        for idx, layer in enumerate(self.get_layers()):
            try:
                layer_type = layer.get_layer_type().lower()
                # user define quantize weights
                if layer.is_extend():
                    layer.quan_feat()
                elif not layer_type.lower() in ['conv', 'depthwiseconv', 'convtranspose', 'gemm', 'fc', 'matmul']:
                    continue
                else:
                    key = layer_type if layer_type in self.__settings.keys() else 'default'
                    weights = self.__settings[key]['weights']
                    if layer.get_layer_name() in weight_scale_dict.keys(): # type: ignore
                        scale_dict = weight_scale_dict[layer.get_layer_name()] # type: ignore
                    else:
                        scale_dict = None
                    quan_weight(self.get_layers(), layer, weights, quantize_factory,
                                scale_dict)
            except:
                error_info = "layer of {} quantize wrong in AlreadyGrapQuant!".format(layer.get_layer_name())
                print(error_info)
                os._exit(-1)
