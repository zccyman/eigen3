# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/9/30 15:49
# @File     : graph.py

'''
build our private structure ops for each layer in graph from loaded checkpoint
'''
import copy
import json
import os
import pickle
import sys

from .layer import Layer
from .layer import LAYER as layer_factory
# from .functional import *
from . import functional
try:
    from utils import two_node_connect, nodes_connect, check_shuffle
    from utils import exhaustive_search, type_replace, check_nodes
    from utils import check_len, shorten_nodes, replace_types
    from utils import Object, nodes_connect_
except:
    from onnx_converter.utils import two_node_connect, nodes_connect, check_shuffle
    from onnx_converter.utils import exhaustive_search, type_replace, check_nodes
    from onnx_converter.utils import check_len, shorten_nodes, replace_types
    from onnx_converter.utils import Object, nodes_connect_

# save defined network
graph_initialized = {}


# three graph, [origin, fused, fused quantize]
# especial ops struct:
# '''
# [conv, relu]
# [conv, bn,,relu]
# [conv, relu, pool]
# [conv, bn, relu, pool]
# [concat, reshape, transpose, reshape, split, split]
# [reshape, transpose, reshape]
# [matmul/gemm, bn, relu]
# [matmul/gemm, relu]
# '''
# reshape will delete in some special structure, because this not modify address in memory
class Graph(Object): # type: ignore
    # layer_quan:['Conv2d': floay_sym]
    def __init__(self, **kwargs):
        super(Graph, self).__init__(**kwargs)
        self.__layers = []
        self.__opset_version = kwargs['opset_version']
        self.__default_setting = kwargs['default_setting']
        # self.__quantize_method = -1
        #
        self.__especial_ops, self.__fuse_ops, self.__split_ops = dict(), dict(), dict()
        self.__replace_ops = dict()
        # self.__act = ['relu', 'leakyrelu', 'tanh', 'hardswish', 'prelu', 'celu']
        self.__act = kwargs['act'] #['relu']
        self.__fuse_act = kwargs["fuse_act"] # type: ignore
        self.__is_last_layer_fuse_act = kwargs["is_last_layer_fuse_act"] # type: ignore
        self.__pool = ['maxpool', 'averagepool', 'globalaveragepool']
        self.__shuffle = [['concat', 'reshape', 'transpose', 'reshape'],
                          ['concat', 'reshape', 'transpose', 'reshape', 'split']]
        self.__empty_ignore = ['data', 'maxpool', 'relu']
        self.__weights_ignore = []
        if 'log_name' in kwargs.keys():
            self.logger = self.get_log(log_name=kwargs['log_name'], log_level=kwargs.get('log_level', 20))
        else:
            self.logger = self.get_log(log_name='graph.log', log_level=kwargs.get('log_level', 20))
        self.logger.info('opset version is: {}'.format(self.__opset_version))
        if not 'nodes' in kwargs:
            self.logger.fatal('not enough input!')
            os._exit(-1)

        # if 'quantize_method' in kwargs: self.__quantize_method = kwargs['quantize_method']
        if 'especial_ops' in kwargs:  self.__especial_ops.update(kwargs['especial_ops'])
        # if 'act' in kwargs: self.__act.extend(kwargs['act'])
        if 'fuse_ops' in kwargs: self.__fuse_ops.update(kwargs['fuse_ops'])
        if 'split_ops' in kwargs: self.__split_ops.update(kwargs['split_ops'])
        if 'replace_ops' in kwargs: self.__replace_ops.update(kwargs['replace_ops'])
        if 'weights_ignore' in kwargs: self.update_list(self.__weights_ignore, kwargs['weights_ignore'])
        if 'empty_ignore' in kwargs: self.update_list(self.__empty_ignore, kwargs['empty_ignore'])
        self.__bns = [[key, "batchnormalization"] for key in self.__fuse_ops.keys()]
        self.__act = list(set(self.__act))
        self.__nodes = kwargs['nodes']

    def update_list(self, old_values: list, values: list):
        inter = set(old_values).intersection(set(values))
        for item in values:
            if item not in inter:
                old_values.append(item)

    def get_opset_version(self):
        return self.__opset_version

    # extract some attribute from weights
    def normalized_properties(self):

        if self.__nodes is None:
            self.logger.fatal('invalid parser trained model!')
            os._exit(-1)
        try:
            # has weights
            # weights_ignore = ['conv', 'sigmoid', 'softmax', 'concat', 'gemm', 'batchnormalization', 'mul', 'matmul', 'convtranspose', 'add', 'sub']
            weights_ignore = []
            empty_ignore = ['data', 'maxpool', 'relu']
            # need_normal = ['slice', 'reshape', 'resize', 'gather', 'gathernd', 'clip', 'less', 'equal']
            # names = [node.get_name() for node in self.__nodes]
            # types = [node.get_op_type() for node in self.__nodes]
            # result_nodes = list(filter(None, [node.get_name() if node.get_result() else [] for node in self.__nodes]))
            nodes = self.get_nodes()
            self.logger.info('node len is: {}'.format(len(nodes)))
            for node in nodes:
                op_type = node.get_op_type().lower()
                if op_type in weights_ignore or op_type in empty_ignore:
                    continue
                if op_type in ["splice"] and node.get_attr()["has_fc"]:
                    weights = [dict(weight=node.get_attr()["weight"]), dict(weight=node.get_attr()["bias"])]
                    node.set_weights(weights)
                weights = node.get_weights()
                if op_type.lower() == "batchnormalization":
                    epsilon = 1e-5 if 'epsilon' not in node.get_attr() else node.get_attr()['epsilon']
                    weights.append(dict(epsilon=epsilon))
                if weights:
                    # if op_type in ["splice"] and len(weights) == 0:
                    #     import numpy as np
                    #     weight = np.random.randn(129, 129)
                    #     bias = np.random.randn(129)
                    #     node.set_weights([dict(weight=weight), dict(weight=bias)])
                    #     weights = node.get_weights()
                    if hasattr(functional, 'normal_' + op_type.lower()):
                        node.get_attr().update(getattr(functional, 'normal_' + op_type.lower())(weights))
                    else:
                        self.logger.warning('new ops type {} in normal!'.format(op_type))
                    # try:
                    #     node.get_attr().update(eval('normal_' + op_type.lower())(weights))
                    # except:
                    #     print('new ops type {} in normal!'.format(op_type))
                    # continue
                # todo relux/relu6 maybe has great function to fixed it
                if node.get_op_type().lower() == 'clip':
                    values = list(node.get_attr().values())
                    top, bottom = max(values), min(values)
                    if bottom == top:
                        if top == 0:
                            op = 'relu'
                            self.logger.warning('clip convert to relu!')
                    elif top == 6:
                        op = 'relu6'
                        self.logger.warning('clip convert to relu6!')
                    else:
                        op = 'relux'
                        self.logger.warning('clip convert to relux!')

                    node.set_op_type(op)
        except:
            self.logger.error("normalized properties failure!")
            os._exit(-1)

    def add_layer(self, layer):
        if layer is None:
            return
        self.__layers.append(layer)

    def get_nodes(self):
        return self.__nodes

    def set_layers(self, layers):
        self.__layers = layers

    def get_layers(self):
        return self.__layers

    def get_extra_nodes_old(self, index, is_shorten, order_num, nodes, ops, extra_ops=None):
        node = nodes[index]
        length = len(ops)
        # replace ['conv', 'act', 'pool'] -> ['conv', 'batchnormalization', 'act', 'pool']
        if index + length < len(nodes) + 1:
            extra_nodes = nodes[index:index + length]
            op_types = replace_types(extra_nodes, self.__act, self.__pool)
            bns = copy.deepcopy(self.__bns) #[['conv', 'batchnormalization'], ['matmul', 'batchnormalization'], ['gemm', 'batchnormalization']]
            for bn in bns:
                if set(bn).intersection(op_types) == set(bn):
                    ops.insert(1, 'batchnormalization')
                    length = len(ops)
                    break

        if index + length < len(nodes) + 1:
            extra_nodes = nodes[index:index + length]
            op_types = replace_types(extra_nodes, self.__act, self.__pool)

            if is_shorten:
                extra_nodes = shorten_nodes(extra_nodes, op_types, ops)

            # process not discontinuous struct
            else:
                order_num = order_num if order_num else len(op_types)
                op_flag = nodes_connect(nodes, list(
                    range(index, index + length)[:order_num])) if order_num > 0 else False
                extra_ops_ = list()
                output = list()
                ops_ = list()
                if op_flag and extra_ops is not None:
                    names = [node.get_name() for node in nodes]
                    for name in extra_nodes[-1].get_output():
                        index = names.index(name)
                        output.append(nodes[index])
                        ops_.append(nodes[index].get_op_type().lower())
                    assert output is not None  # , print('extra not found')
                    ops.extend(ops_)
                    for op_extra in extra_ops:
                        op_types_ = copy.deepcopy(op_types)
                        op_types_.extend(op_extra)
                        extra_ops_.append(op_types_)
                else:
                    extra_ops_.append(op_types)

                if ops in extra_ops_ and op_flag:
                    extra_nodes.extend(output)
                elif ops[:order_num + 1] in extra_ops_ and op_flag:
                    # just use for concat-shuffle-only layer
                    pass
                else:
                    extra_nodes = extra_nodes[0:1]

        else:
            if is_shorten:
                extra_nodes = nodes[index:len(nodes)]
                especial_types = replace_types(extra_nodes, self.__act, self.__pool)
                extra_nodes = shorten_nodes(extra_nodes, especial_types, ops)
            else:
                extra_nodes = [node]

        return extra_nodes

    def _instance_nodes(self, nodes, names, node, index, length):
        extra_nodes, l_idxs = [], [index]
        extra_node = copy.deepcopy(node)
        for _ in range(length):
            extra_nodes.append(extra_node)
            # todo output great than one not consider
            if extra_node.get_output()[0] in names:
                l_idx = names.index(extra_node.get_output()[0])
                l_idxs.append(l_idx)
                extra_node = nodes[l_idx]
            else:
                break
        return extra_nodes, l_idxs

    def get_extra_nodes(self, index, is_shorten, order_num, nodes, ops, extra_ops=None):
        node = nodes[index]
        length = len(ops)
        # replace ['conv', 'act', 'pool'] -> ['conv', 'batchnormalization', 'act', 'pool']
        names = [node.get_name() for node in nodes]
        if index + length < len(nodes)+1:
            # extra_nodes = nodes[index:index + length]
            extra_nodes, _ = self._instance_nodes(nodes, names, node, index, length)
            op_types = replace_types(extra_nodes, self.__act, self.__pool)
            bns = copy.deepcopy(self.__bns)
            [bn.reverse() for bn in bns]
            if op_types in bns:#[["batchnormalization", "conv"], ["batchnormalization", "matmul"], ["batchnormalization", "gemm"]]:
                ops.insert(2, 'act')
                length = len(ops)
            else:
                bns = copy.deepcopy(self.__bns)
                for bn in bns:
                    if set(bn).intersection(op_types) == set(bn):
                        ops.insert(1, 'batchnormalization')
                        length = len(ops)
                        break

        if index + length < len(nodes)+1:
            # extra_nodes = nodes[index:index + length]
            extra_nodes, _ = self._instance_nodes(nodes, names, node, index, length)
            op_types = replace_types(extra_nodes, self.__act, self.__pool)

            if is_shorten:
                extra_nodes = shorten_nodes(extra_nodes, op_types, ops)

            # process not discontinuous struct
            else:
                extra_nodes, l_idxs = self._instance_nodes(nodes, names, node, index, length)

                order_num = order_num if order_num else len(op_types)
                # op_flag = nodes_connect(nodes, list(
                #     range(index, index + length)[:order_num])) if order_num > 0 else False
                op_flag = nodes_connect_(extra_nodes[:order_num]) if order_num > 0 else False
                extra_ops_ = list()
                output = list()
                ops_ = list()
                if op_flag and extra_ops is not None:
                    # names = [node.get_name() for node in nodes]
                    for name in extra_nodes[-1].get_output():
                        if name in names:
                            index = names.index(name)
                            output.append(nodes[index])
                            ops_.append(nodes[index].get_op_type().lower())
                        else:
                            self.logger.warning("node of {} output name {} is not is node names".format(extra_nodes[-1].get_name(), name))
                            continue
                    if output is None:
                        self.logger.fatal("Build layer inner error!")
                    assert output is not None#, print('extra not found')
                    ops.extend(ops_)
                    for op_extra in extra_ops:
                        op_types_ = copy.deepcopy(op_types)
                        op_types_.extend(op_extra)
                        extra_ops_.append(op_types_)
                else:
                    extra_ops_.append(op_types)

                if ops in extra_ops_ and op_flag:
                    extra_nodes.extend(output)
                elif ops[:order_num+1] in extra_ops_ and op_flag:
                    # just use for concat-shuffle-only layer
                    pass
                else:
                    extra_nodes = extra_nodes[0:1]

        else:
            if is_shorten:
                extra_nodes = nodes[index:len(nodes)]
                especial_types = replace_types(extra_nodes, self.__act, self.__pool)
                extra_nodes = shorten_nodes(extra_nodes, especial_types, ops)
            else:
                extra_nodes = [node]

        return extra_nodes

    # done
    # replace some op type in node
    def build_layers(self):
        layer_idx = 0
        index = 0
        nodes, processed_nodes = copy.deepcopy(self.get_nodes()), list()

        # test node in processed_nodes
        # nodes[output_names.index('Conv_371')] in process_nodes
        output_names = [node.get_name() for node in nodes]
        # result_nodes = list(filter(None, [node.get_name() if node.get_result() else [] for node in nodes]))
        # shuffle = ['Concat', 'Reshape', 'Transpose', 'Reshape']
        # act = ['Relu', 'LeakyRelu', 'Tanh', 'HardSwish', 'Sigmoid']
        # pool = ['MaxPool', 'AveragePool']
        try:
            while index < len(nodes):
                node = nodes[index]
                if node in processed_nodes:
                    # print('###############, continue')
                    index += 1
                    continue

                # if node.get_name() == 'Reshape_ext_0-2':
                #     print('test')

                if not bool(self.__especial_ops):
                    # print('@@@@@@@@@@@@@@@@@@ ', node.get_name())
                    if 'data' in node.get_op_type():
                        data_nodes = list()
                        layer_type = 'data'
                        if layer_type in layer_factory.module_dict:
                            layer = layer_factory.get(layer_type)()
                        else:
                            layer = Layer()
                        layer.set_idx(layer_idx)
                        data_nodes.append(node)
                        layer.set_layer_type(layer_type)
                        layer.extend(data_nodes)
                        processed_nodes.append(node)
                        self.add_layer(layer)
                        layer_idx += 1
                        index += 1
                    # [conv, conv/relu, conv/relu/pooling]
                    elif 'Conv' in node.get_op_type():
                        conv_nodes = list()
                        # layer = Layer()
                        layer_type = 'conv'
                        if layer_type in layer_factory.module_dict:
                            layer = layer_factory.get(layer_type)()
                        else:
                            layer = Layer()
                        layer.set_layer_type(layer_type)
                        conv_nodes.append(node)
                        if check_len(nodes, index + 1):
                            if two_node_connect(nodes, index, ['batchnormalization']):
                                index += 1
                                conv_nodes.append(nodes[index])
                        if check_len(nodes, index + 1):
                            if two_node_connect(nodes, index, self.__act):
                                index += 1
                                conv_nodes.append(nodes[index])
                                # if check_len(nodes, index + 1):
                                #     if two_node_connect(nodes, index, self.__pool):
                                #         index += 1
                                #         conv_nodes.append(nodes[index])
                        processed_nodes.extend(conv_nodes)
                        layer.set_idx(layer_idx)
                        layer.extend(conv_nodes)
                        self.add_layer(layer)
                        index += 1
                        layer_idx += 1
                    # include concat and shuffle
                    elif 'Concat' in node.get_op_type():
                        concat_nodes = list()
                        layer_type = 'concat'
                        concat_nodes.append(nodes[index])
                        if index < len(nodes) - 4:
                            is_order = nodes_connect(nodes, list(range(index, index + 4 - 1)))
                            is_order = is_order and [node.get_op_type().lower() for node in
                                                    nodes[index:index + 4]] == self.__shuffle[0]
                            is_order = is_order and check_shuffle(nodes[index + 1:index + 4])
                            out_types, shuffle_nodes = list(), list()
                            if is_order:
                                out_names = nodes[index + 3].get_output()
                                shuffle_nodes.extend(nodes[index + 1:index + 4])
                                for name_ in out_names:
                                    shuffle_nodes.append(nodes[output_names.index(name_)])
                                    out_types.append(nodes[output_names.index(name_)].get_op_type())
                            if set(out_types) == set(['Slice']):
                                concat_nodes.extend(shuffle_nodes)
                                layer_type = 'shuffle'
                                index += 3
                        if index < len(nodes) - 5:
                            is_order = nodes_connect(nodes, list(range(index, index + 5 - 1)))
                            is_order = is_order and [node.get_op_type().lower() for node in
                                                    nodes[index:index + 5]] == self.__shuffle[1]
                            is_order = is_order and check_shuffle(nodes[index + 1:index + 5])
                            if is_order:
                                concat_nodes.extend(shuffle_nodes)
                                layer_type = 'shuffle'
                                index += 4
                        if layer_type in layer_factory.module_dict:
                            layer = layer_factory.get(layer_type)()
                        else:
                            layer = Layer()
                        layer.set_idx(layer_idx)
                        layer.set_layer_type(layer_type)
                        processed_nodes.extend(concat_nodes)
                        layer.extend(concat_nodes)
                        self.add_layer(layer)
                        index += 1
                        layer_idx += 1
                    # include reshape and shuffle
                    elif 'Reshape' in node.get_op_type():
                        reshape_nodes = list()
                        layer_type = 'reshape'
                        reshape_nodes.append(node)
                        # out_types, shuffle_nodes = list(), list()
                        if index < len(nodes) - 3:
                            is_order = nodes_connect(nodes, list(range(index, index + 3 - 1)))
                            is_order = is_order and [node.get_op_type().lower() for node in
                                                    nodes[index:index + 3]] == self.__shuffle[0][1:]
                            is_order = is_order and check_shuffle(nodes[index:index + 3])
                            if is_order:
                                reshape_nodes.extend(nodes[index + 1:index + 3])
                                layer_type = 'shuffle_only'
                                index += 2
                        if layer_type in layer_factory.module_dict:
                            layer = layer_factory.get(layer_type)()
                        else:
                            layer = Layer()
                        layer.set_idx(layer_idx)
                        layer.set_layer_type(layer_type)
                        processed_nodes.extend(reshape_nodes)
                        layer.extend(reshape_nodes)
                        self.add_layer(layer)
                        index += 1
                        layer_idx += 1
                    # include [fc, bn, relu]/[fc, relu]/[fc]
                    elif 'MatMul' in node.get_op_type() or 'Gemm' in node.get_op_type():
                        fc_nodes = list()
                        layer_type = node.get_op_type().lower()
                        if layer_type in layer_factory.module_dict:
                            layer = layer_factory.get(layer_type)()
                        else:
                            layer = Layer()
                        layer.set_idx(layer_idx)
                        layer.set_layer_type(layer_type)
                        fc_nodes.append(node)
                        if two_node_connect(nodes, index, ['batchnormalization']):
                            index += 1
                            fc_nodes.append(nodes[index])
                        if two_node_connect(nodes, index, self.__act):
                            index += 1
                            fc_nodes.append(nodes[index])
                            if two_node_connect(nodes, index, self.__pool):
                                index += 1
                                fc_nodes.append(nodes[index])
                        processed_nodes.extend(fc_nodes)
                        layer.extend(fc_nodes)
                        self.add_layer(layer)
                        index += 1
                        layer_idx += 1
                    # other ops:[reshape, ]
                    else:
                        extra_nodes = [node]
                        processed_nodes.append(extra_nodes)
                        layer_type = extra_nodes[0].get_op_type().lower()
                        layer_type_ = type_replace(layer_type, self.__replace_ops)
                        layer_type = layer_type_ if layer_type_ is not None else layer_type
                        if layer_type in layer_factory.module_dict:
                            layer = layer_factory.get(layer_type)()
                        else:
                            layer = Layer()
                            # layer = layer_factory.get("default")()
                            # self.logger.warning("layer {} is not exist, using default layer!!!".format(layer_type))

                        layer.set_layer_type(layer_type)
                        layer.set_layer_name(node.get_name())
                        layer.set_idx(layer_idx)
                        layer.extend(extra_nodes)
                        self.add_layer(layer)
                        layer_idx += 1
                        index += 1
                # input combine structure
                else:
                    op_type = node.get_op_type().lower()
                    # if node.get_name() == 'prefinal-chain.batchnorm':
                        # print('test')
                    # extra_nodes, layer_nodes = list(), list()
                    especial_ops = self.__especial_ops
                    vaild_key = ''
                    for key in especial_ops.keys():
                        if op_type in especial_ops[key].keys():
                            vaild_key = key
                            break
                    if vaild_key != '':
                        ops = list()
                        is_shorten = especial_ops[vaild_key]['is_shorten']
                        order_num = especial_ops[vaild_key]['order_num']
                        for key in especial_ops[vaild_key].keys():
                            if key in ['is_shorten', 'order_num', 'extra']:
                                continue
                            ops = copy.deepcopy(especial_ops[vaild_key][key])
                        extra_ops = especial_ops[vaild_key]['extra'] if 'extra' in especial_ops[vaild_key].keys() else None
                        extra_nodes = self.get_extra_nodes(index, is_shorten, order_num, nodes, ops, extra_ops)
                    else:
                        extra_nodes = [node]

                    # if not check_nodes(extra_nodes[:order_num]):
                    #     extra_nodes = extra_nodes[:1]

                    layer_type = 'unknown'
                    if len(extra_nodes) > 1:
                        layer_type = vaild_key
                    else:
                        layer_type = extra_nodes[0].get_op_type().lower()
                    layer_type_ = type_replace(layer_type, self.__replace_ops)
                    layer_type = layer_type_ if layer_type_ is not None else layer_type
                    # just use for concat-shuffle-only layer
                    if layer_type in ['shuffle']:
                        if len(extra_nodes) < 5:
                            layer_type = 'concat_shuffle_only'
                    if layer_type in ['shuffle_only']:
                        if len(extra_nodes) > 3:
                            layer_type = 'shuffle_only_split'
                        for node_ in extra_nodes[1:-1]:
                            ### check shuffle_only
                            if len(node_.get_output()) > 1:
                                layer_type = extra_nodes[0].get_op_type().lower()
                                extra_nodes = [extra_nodes[0]]

                    if layer_type in ['swish']:
                        node_in_0, node_in_1 = [node.get_input() for node in extra_nodes]
                        if len(set(node_in_0).intersection(set(node_in_1))) <= 0:
                            extra_nodes = extra_nodes[0:1]
                            layer_type = extra_nodes[0].get_op_type().lower()

                    if layer_type in ['batchnormalization'] and len(extra_nodes) > 1:
                        layer_type = extra_nodes[1].get_op_type().lower()
                        layer_name = extra_nodes[1].get_name()
                    else:
                        layer_name = node.get_name()

                    if layer_type in layer_factory.module_dict:
                        layer = layer_factory.get(layer_type)()
                    else:
                        layer = Layer()
                        # layer = layer_factory.get("default")()
                        # self.logger.warning("layer {} is not exist, using default layer!!!".format(layer_type))

                    layer.set_idx(layer_idx)
                    layer.set_layer_type(layer_type)
                    layer.extend(extra_nodes)
                    layer.set_layer_name(layer_name)
                    processed_nodes.extend(extra_nodes)
                    self.add_layer(layer)
                    layer_idx += 1
                    index += 1#len(extra_nodes)
        except:
            self.logger.error("build layer failure!")
            os._exit(-1)
        # set layer input and output
        # set layer reflect on onnx runtime
        input_names, output_names = [], []
        try:

            for layer in self.get_layers():
                # if layer.get_layer_name() in ['AveragePool_2805', 'Split_794']:
                #     print('test')
                nodes = layer.get_input_nodes()
                # input_name = list()
                if isinstance(nodes, list):
                    input_name = [node.get_name() for node in nodes]
                    input_names.extend(input_name)
                else:
                    input_name = nodes.get_name()
                    input_names.append(input_name)

            for layer in self.get_layers():
                nodes = layer.get_outout_nodes()
                # output_name = list()
                if isinstance(nodes, list):
                    output_name = [node.get_name() for node in nodes]
                    output_names.append(output_name)
                else:
                    output_name = nodes.get_name()
                    output_names.append(output_name)

            for l_idx, layer in enumerate(self.get_layers()):
                layer_input = layer.get_input_name()
                layer_output = layer.get_output_name()

                input_idx, output_idx = [], []
                if layer_input != []:
                    if isinstance(layer_input, list):
                        for input_name in layer_input:
                            index = exhaustive_search(output_names, input_name)
                            if index == -1:
                                print('invaild index {} input name'.format(index))
                                input_idx.append(index)
                            else:
                                # if index != -1:
                                input_idx.insert(layer_input.index(input_name), index)
                    else:
                        index = exhaustive_search(output_names, layer_input)
                        input_idx.append(index)
                if layer_output:
                    if isinstance(layer_output, list):
                        for out_idx, output_name in enumerate(layer_output):
                            index = exhaustive_search(input_names, output_name)
                            if index == -1:
                                output_idx.append(index)
                                self.logger.warning('invaild index {} output name'.format(index))
                            else:
                                # if index != -1:
                                # output_idx.insert(layer_output.index(output_name), out_idx)
                                output_idx.insert(out_idx, index)
                            # else:
                            #     output_idx.append(index)
                    else:
                        index = exhaustive_search(input_names, layer_output)
                        output_idx.append(index)
                # self.logger.info('input idx is: {}, output idx is: {}'.format(input_idx, output_idx))
                if layer.get_layer_type() == 'data':
                    layer.set_input_idx([-1]) ### added by henson, used in export v2
                else:
                    layer.set_input_idx(input_idx)
                layer.set_output_idx(output_idx)

            for layer in self.get_layers():
                node = layer.get_nodes()[0]
                if node.get_op_type().lower() == 'conv':
                    attr = node.get_attr()
                    in_c, out_c, group = attr['in_c'], attr['out_c'], 0
                    group = attr['group'] if 'group' in attr.keys() else 0
                    if out_c == group and in_c == 1:
                        layer.set_layer_type('depthwiseconv')
                        self.logger.info('{} layer: {} - conv convert depthwiseconv!'.format(layer.get_nodes()[0].get_name(), layer.get_idx()))
                        # attr['in_c'], attr['out_c'] = group, 1
                        node.set_attr(attr)
                        node.set_op_type('DepthwiseConv')

            # input_idxs = [layer.get_input_idx() for layer in self.get_layers()]
            # output_idxs = [layer.get_output_idx() for layer in self.get_layers()]

            # print('test!')
        except:
            self.logger.error("connection of layer failure!")
            os._exit(-1)

    # todo
    # ex: fc and bn, conv and bn
    # other fuse operator not found
    def fuse_layer_ops(self):
        fuse_ops = self.__fuse_ops
        try:
            for layer in self.get_layers():
                fuse = False
                if layer.get_layer_type().lower() in fuse_ops.keys():
                    node_types = [node.get_op_type().lower() for node in layer.get_nodes()]
                    ops = fuse_ops[layer.get_layer_type().lower()]
                    ops = [op.lower() for op in ops]
                    fuse = list(set(ops).intersection(node_types))

                if fuse:
                    for fs in fuse:
                        if fs == 'batchnormalization':
                            nodes = layer.get_nodes()
                            fuse_nodes = copy.deepcopy(nodes)
                            if fuse_nodes[0].get_op_type().lower() == "batchnormalization" and \
                                layer.get_layer_type().lower() in ["conv", "gemm", "matmul", "fc"]:
                                node_tmp, minmax_var = functional.fuse_batchnormalization_into_conv_fc(
                                    fuse_nodes)
                                layer.set_ops_nodes(node_tmp)
                                if minmax_var[0] < 1.0e-3:
                                    self.logger.warning("bn running_var less than 1.0e-3")
                                if minmax_var[1] > 1.0e3:
                                    self.logger.warning("bn running_var greater than 1.0e3")
                            else:
                                layer.set_ops_nodes(
                                    functional.fuse_batchnormalization(fuse_nodes))
                            self.logger.info('layer {} fuse op'.format(fs))
                else:
                    layer.set_ops_nodes(layer.get_nodes())
        except:
            self.logger.error("fused layer ops failure!")
            os._exit(-1)

    def get_layer_info(self, idx):
        if isinstance(idx, list):
            infos = [self.__layers[item] for item in idx]
        else:
            infos = self.__layers[idx]
        return infos

    # replace ops, ex: ['gemm', 'matmul'] replace ['fc']
    # set layer is reslut ?
    def normal_ops(self):
        try:
            for layer in self.get_layers():
                new_ops = list()
                ops = layer.get_layer_ops()
                is_reslut = True in [node.get_is_result() for node in layer.get_nodes()]
                layer.set_result_layer(is_reslut)
                for op in ops['ops']:
                    for key in self.__replace_ops.keys():
                        if op in self.__replace_ops[key]:
                            op = key
                            break
                    new_ops.append(op)
                ops['ops'] = new_ops
                layer.set_layer_ops(ops)
        except:
            self.logger.error("normal ops failure!")
            os._exit(-1)

    # ex: gemm->[matmul, bias], conv:[conv, bias]
    def split_layer_ops(self):
        try:
            for layer in self.get_layers():
                ops, attrs, weights = [], [], []
                for node in layer.get_ops_nodes():
                    if list() != node.get_weights():
                        weights.extend([weights_['weight'] for weights_ in node.get_weights()])
                        # weights.append(node.get_weights()[0]['weight'])
                    else:
                        weights.append(list())
                    attrs.append(node.get_attr())
                    if node.get_op_type().lower() in self.__split_ops['name']:
                        has_bias = node.get_attr()['bias']
                        ops.append(node.get_op_type().lower())
                        if has_bias:
                            self.logger.info('{} split bias from original layer!'.format(node.get_name()))
                            ops.append('bias')
                            weights.append(node.get_weights()[1]['weight'])
                            attrs.append(dict())
                    else:
                        ops.append(node.get_op_type().lower())
                layer.set_layer_ops(dict(ops=ops, attrs=attrs, weights=weights))
        except:
            self.logger.error("split special layer failure!")
            os._exit(-1)

    def fuse_act_into_weight_layer(self):
        try:
            layers = self.get_layers()

            in_names, out_names = [], []
            layers_after_fuse, ignor_layer_names = [], []
            for layer in layers:
                layer_name = layer.get_layer_name()
                # print(layer_name)
                # if layer_name == "Gemm_24":
                #     print("test")
                if layer_name in ignor_layer_names:
                    continue

                next_layer = None
                layer_type = layer.get_layer_type()
                if layer_type in ["conv", "fc", "depthwiseconv", "convtranspose"]:
                    layer_idx = [idx for idx in layer.get_output_idx() if idx >= 0]
                    method = self.__default_setting[layer_type]["weights"]["method"]
                    if len(layer_idx) > 0 and "perchannel" not in method:
                        next_layer = layers[layer_idx[-1]]

                if next_layer:
                    if next_layer.get_layer_type() in self.__fuse_act:
                        is_last_layer_fuse_act = True
                        if next_layer.get_is_result_layer():
                            is_last_layer_fuse_act = self.__is_last_layer_fuse_act
                        if not is_last_layer_fuse_act:
                            continue
                        nodes_next = next_layer.get_nodes()
                        nodes = layer.get_nodes()
                        nodes.extend(nodes_next)
                        layer.set_nodes(nodes)
                        ignor_layer_names.append(next_layer.get_layer_name())
                        # layer.set_scale_type("shiftfloatscaletable")
                        act_type = next_layer.get_layer_type()
                        ops = layer.get_layer_ops()
                        ops['ops'].append(act_type)
                        ops['attrs'].append(next_layer.get_layer_ops()["attrs"][0])
                        layer.set_layer_ops(ops)

                layers_after_fuse.append(layer)
                in_names.append(layer.get_onnx_input_name())
                out_names.append(layer.get_onnx_output_name())

            for layer_idx, layer in enumerate(layers_after_fuse):
                layer_name = layer.get_layer_name()
                # print(layer_name)
                layer_input_idx = []
                for input_name in layer.get_onnx_input_name():
                    for in_idx, in_name in enumerate(out_names):
                        if input_name in in_name:
                            layer_input_idx.append(in_idx)
                            break
                if len(layer_input_idx) == 0:
                    layer_input_idx.append(-1)

                layer.clear_input_idx()
                layer.set_input_idx(layer_input_idx)
                layer_output_idx = []
                for output_name in layer.get_onnx_output_name():
                    for out_idx, out_name in enumerate(in_names):
                        if output_name in out_name:
                            layer_output_idx.append(out_idx)
                if len(layer_output_idx) == 0:
                    layer_output_idx.append(-1)

                layer.clear_output_idx()
                layer.set_output_idx(layer_output_idx)
                layer.set_idx(layer_idx)
                _print_in_layer_name = []
                for idx in layer.get_input_idx():
                    if idx >= 0:
                        _print_in_layer_name.append(layers_after_fuse[idx].get_layer_name())
                _print_out_layer_name = []
                for idx in layer.get_output_idx():
                    if idx >= 0:
                        _print_out_layer_name.append(layers_after_fuse[idx].get_layer_name())
                # print(layer_idx, layer_name, layer.get_input_idx(), layer.get_output_idx())
                # print(layer_idx, layer_name, _print_in_layer_name, _print_out_layer_name)
                # print("test")
            self.set_layers(layers_after_fuse)
        except:
            self.logger.error("fuse act into weight layer failure!")
            os._exit(-1)

    def current_layer_input_map_to_pre_layer(self):
        ### set input map, output map
        layers = self.get_layers()
        for layer in layers:
            if layer.get_layer_type() == 'data':
                layer.set_input_map([0])
                continue

            layer_input_idxs = layer.get_input_idx()
            layer_output_idxs = layer.get_output_idx()
            pre_layer_output_names = []
            for idx in layer_input_idxs:
                pre_layer = layers[idx]
                pre_layer_output_name = pre_layer.get_onnx_output_name()
                pre_layer_output_names.append(pre_layer_output_name)

            next_layer_input_names = []
            for idx in layer_output_idxs:
                next_layer = layers[idx]
                next_layer_input_name = next_layer.get_onnx_input_name()
                next_layer_input_names.append(next_layer_input_name)

            current_layer_input_names = layer.get_onnx_input_name()
            # current_layer_output_names = layer.get_onnx_output_name()

            input_map = []
            for current_layer_input_name in current_layer_input_names:
                for layer_idx, pre_layer_output_name in zip(layer_input_idxs, pre_layer_output_names):
                    if current_layer_input_name in pre_layer_output_name:
                        index = pre_layer_output_name.index(current_layer_input_name)
                        input_map.append(index)
                        break
            layer.set_input_map(input_map)

            # if not layer.get_is_result_layer():
            #     output_map = [[] for _ in current_layer_output_names]
            #     for layer_idx, next_layer_input_name in zip(layer_output_idxs, next_layer_input_names):
            #         for idx, current_layer_output_name in enumerate(current_layer_output_names):
            #             index = next_layer_input_name.index(current_layer_output_name)
            #             output_map[idx].append(index)
            #     layer.set_output_map(output_map)
            # else:
            #     layer.set_output_map([0])

    # read origin network, build layer structure using chain execute target
    # merge conv/act/pooling in one layer
    # merge shuffle and shuffle-only structure, this have three style structure,
    #       first: reshape->transpose->reshape,
    #       second: concat->reshape->transpose->reshape \-> slice,
    #                                                   \-> slice,
    #       third: reshape->transpose->reshape-> \-> slice
    #                                            \-> slice
    # delete useless node,ex: [mul, div, reduce, scatter]
    #                             conv
    #                               .
    #  split group conv to [slice   . concat]
    #                             conv
    #  split gemm to fc/bias
    def build(self):
        self.normalized_properties()
        self.build_layers()
        self.fuse_layer_ops()
        self.split_layer_ops()
        self.normal_ops()
        self.fuse_act_into_weight_layer()
        self.current_layer_input_map_to_pre_layer()
        [layer.init_quantizes() for layer in self.get_layers()]
        # self.logger.info('system info: {}, {}'.format(sys._getframe(), sys._getframe().f_lineno))

    def update_layers_connetion(self, from_idx, delta=1):
        layers = self.get_layers()
        for layer in layers:
            idx = layer.get_idx()
            if idx >= from_idx:
                layer_input_idx = layer.get_input_idx()
                layer_output_idx = layer.get_output_idx()
                layer_input_idx = []
                for input_idx in layer.get_input_idx():
                    if input_idx >= 0:
                        input_idx = input_idx + delta
                        layer_input_idx.append(input_idx)
                    else:
                        layer_input_idx.append(-1)
                layer_output_idx = []
                for output_idx in layer.get_output_idx():
                    if output_idx >= 0:
                        output_idx = output_idx + delta
                        layer_output_idx.append(output_idx)
                    else:
                        layer_output_idx.append(-1)
                layer.clear_input_idx()
                layer.set_input_idx(layer_input_idx)
                layer.clear_output_idx()
                layer.set_output_idx(layer_output_idx)
                layer_idx = idx + delta
                layer.set_idx(layer_idx)

    def together_layer(self, together_layer_names):
        layers = self.get_layers()

        idxs = []
        for layer_ in layers:
            if layer_.get_layer_name() in together_layer_names:
                layer = layer_
                idx = layer.get_idx()
                idxs.append(idx)

            if len(idxs) == len(together_layer_names):
                break

        if len(idxs) != len(together_layer_names):
            os._exit(-1)

        layer = layers[idxs[0]]
        nodes = layer.get_nodes()
        ops = layer.get_layer_ops()
        ops['ops'] = ops['ops'][:-1]

        for idx in idxs[1:]:
            layer_t = layers[idx]
            node = layer_t.get_nodes()
            nodes.extend(node)
            op = layer_t.get_layer_ops()
            ops['ops'].extend(op['ops'])
            layers.pop(idx)

        layer.set_layer_ops(ops)
        layer.set_nodes(nodes)
        layer.set_ops_nodes(nodes)

        # update layer idx
        self.update_layers_connetion(idxs[-1]+1, delta=-1)
        # print("test")

    # spilt together ops from single layer
    def split_layer(self, split_layer_name, ops=None):
        layers = self.get_layers()
        layer, idx = None, None
        for layer_ in layers:
            if layer_.get_layer_name() == split_layer_name:
                layer = layer_
                idx = layer.get_idx()
                break

        if layer is None:
            os._exit(-1)

        nodes = layer.get_nodes()
        ops = layer.get_layer_ops()
        new_ops = ops['ops'][-1]
        ops['ops'][-1] = "act"
        ops['attrs'][-1] = dict()
        layer.set_layer_ops(ops)
        layer.set_nodes(nodes[:1])
        layer.set_ops_nodes(nodes[:1])

        # update layer idx
        self.update_layers_connetion(idx+1, delta=1)

        # build new layer
        node = nodes[1:]
        layer_name = node[-1].get_name()
        layer_type = new_ops
        new_layer = layer_factory.get(layer_type)()
        new_layer.set_idx(idx + 1)
        new_layer.set_layer_type(layer_type)
        new_layer.set_layer_name(layer_name)
        new_layer.set_nodes(node)
        new_layer.set_ops_nodes(node)
        ops = dict()
        ops['ops'] = [new_ops]
        ops['attrs'] = [dict()]
        new_layer.set_layer_ops(ops)
        layer_input_idx = [idx]
        layer_output_idx = [idx + 2]
        new_layer.clear_input_idx()
        new_layer.set_input_idx(layer_input_idx)
        new_layer.clear_output_idx()
        new_layer.set_output_idx(layer_output_idx)

        layers.insert(idx + 1, new_layer)


    # todo
    # name index layer
    def get_layer(self, name):
        pass
    
    # todo
    # get last node name in each layer with this graph
    def get_all_layer_output_name(self):
        return [layer.get_output_name() for layer in self.get_layers()]

    def get_all_layer_input_name(self):
        return [layer.get_input_name() for layer in self.get_layers()]

    # extend extra layer
    def Extensions(self):
        pass

    # todo
    # input data scale from dataset or single file
    def set_scale(self, data: dict):
        layers = self.get_layers()
        for key in data.keys():
            layer = layers[key]
            layer.set_data_scale(data[key])

    # todo
    # quantize method 1. all layer using one method
    #                 2. the same operations layer quantize with one
    #                 3. single layer with one
    #                 4. single op with one in layer
    def set_quantize(self, json_data=None):
        if isinstance(json_data, str):
            if os.path.exists(json_data):
                data = None
            else:
                with open(json_data, 'r') as load_f:
                    data = json.load(load_f)
        else:
            data = copy.deepcopy(json_data)

        if data is None:
            print('input quantize config is None!')
            return

    # def reset(self, layer_type):
    #     for layer in self.get_layers():
    #         if layer.get_layer_type() == layer_type:
    #             layer.reset()

    def reload(self):
        with open("my_instance.bin", "rb+") as f:
            self.process = pickle.load(f)
    
    def save(self):
        with open("my_instance.bin", "wb+") as f:
            pickle.dump(self.process, f)

# if __name__ == '__main__':
#     from checkpoint import OnnxParser
#
#     parser = OnnxParser()
#     # nodes = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/nanodet_sim.onnx')
#     # nodes = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/mobilefacenet-sim.onnx')
#     nodes = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/mobilenetv2-7.onnx')
#     # nodes = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/googlenet-3.onnx')
#     # nodes = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/MaskRCNN-10.onnx', is_simplify=True)
#     # nodes = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.onnx', is_simplify=True)
#     # nodes = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.onnx', is_simplify=True)
#
#     especial_ops = dict(data=['data'],
#                         conv=['conv', 'act', 'pool'],
#                         concat=['concat', 'reshape', 'transpose', 'reshape', 'slice', 'slice'],
#                         reshape=['reshape', 'transpose', 'reshape'],
#                         matmul=['matmul', 'batchnormalization', 'act'],
#                         gemm=['gemm', 'batchnormalization', 'act'])
#
#     graph = Graph(nodes=nodes, especial_ops=especial_ops)
#     graph.build()
