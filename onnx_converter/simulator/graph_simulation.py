# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/8 17:22
# @File     : graph_simulation.py

import os
import copy
import numpy as np
from .checkerror import error_factory  # , CosineSimiarity, L1Simiarity, L2Simiarity

try:
    from utils import Object, process_im, Similarity
    from utils import extract_scale
except:
    from onnx_converter.utils import Object, process_im, Similarity # type: ignore
    from onnx_converter.utils import extract_scale # type: ignore


# layers correct type as input of init function
# parse simulation config, compose layer hardware output type and constant setting
# aligment layer input and output data type, operators not alignment for layer data types setting
#  todo static or dynamic quantize for map
class Simulation(Object): # type: ignore
    def __init__(self, **kwargs):
        super(Simulation, self).__init__(**kwargs)
        if 'log_name' in kwargs.keys():
            self.logger = self.get_log(log_name=kwargs['log_name'], log_level=kwargs.get('log_level', 20))
        else:
            self.logger = self.get_log(log_name='simulation.log', log_level=kwargs.get('log_level', 20))
        # graph data transfer in different modules
        self.__quan_graph = None  # kwargs['quan_graph']
        self.__simulation_level = kwargs['simulation_level']
        self.logger.info('Simulation level is: {}'.format(self.__simulation_level))
        self.__layers = None  # self.__quan_graph.get_layers()
        self.__layer_outputs = dict()

    def set_graph(self, graph):
        self.__quan_graph = graph
        self.__layers = self.__quan_graph.get_layers()

    def get_graph(self):
        return self.__quan_graph

    def get_layers(self): # type: ignore
        # return self.__quan_graph.get_layers() # type: ignore
        return self.__layers

    def reset_layer(self, layer_type="lstm"):
        layer_len = len(self.__layers)
        lstm_idx = 0
        for idx in range(layer_len):
            if self.__layers[idx].get_layer_type() == layer_type:
                self.__layers[idx].reset()
                lstm_idx += 1
                # if lstm_idx > 1:
                #     break
    
    # layer ops simulation output
    # include layer output
    # if simulation level: 0 -> just output layer, 1 -> every op output
    def get_layer_output(self) -> dict:
        return self.__layer_outputs

    def set_simu_level(self, level):
        self.__simulation_level = level

    def get_simu_level(self):
        return self.__simulation_level

    def get_layer_input_info(self, layer_idx):
        layers = self.__quan_graph.get_layers() # type: ignore
        cur_layer = layers[layer_idx]
        inputs, in_data_idx = dict(), dict()
        for in_name in cur_layer.get_onnx_input_name():
            for l_idx, ilayer in enumerate(layers):
                if in_name in ilayer.get_onnx_output_name():
                    iidx = ilayer.get_onnx_output_name().index(in_name)

                    def constant_idx(layer, idx):
                        itype = layer.get_layer_type()
                        vaild_key = ['conv', 'depthwiseconv', 'convtranspose', 'fc', 'matmul', 'gemm', 'batchnormalization']
                        idx = -1 if itype in vaild_key else idx
                        return idx

                    iidx = constant_idx(ilayer, iidx)
                    inputs[l_idx] = iidx
                    if l_idx in in_data_idx.keys():
                        in_data_idx[l_idx] = in_data_idx[l_idx] + 1
                    else:
                        in_data_idx[l_idx] = 1
                    break
        return inputs, in_data_idx

    def get_layer_input(self, layer_idx):
        layers = self.__quan_graph.get_layers() # type: ignore
        (inputs, in_data_idx), in_data = self.get_layer_input_info(layer_idx), list()
        log_infos = []
        for l_idx in inputs.keys():
            ilayer, iidx = layers[l_idx], inputs[l_idx]
            outputs = ilayer.get_out_data()
            if isinstance(outputs, list):
                outputs = copy.deepcopy(outputs[iidx])
            elif isinstance(outputs['output'], list):
                outputs = dict(output=outputs['output'][iidx],
                               out_shift=outputs['out_shift'][iidx],
                               out_scale=outputs['out_scale'][iidx])
                log_info = "layer input shape: {}".format(outputs["output"].shape)
                log_infos.append(log_info)
            else:
                outputs = dict(
                    output=outputs['output'],
                    out_shift=0,
                    out_scale=1)

                log_info = "layer input shape: {}".format(outputs["output"].shape) # type: ignore
                log_infos.append(log_info)
            in_data.append(outputs)
            [in_data.append(copy.deepcopy(outputs)) for _ in range(1, in_data_idx[l_idx])]

        return in_data, log_infos

    def get_qtrue_qdata(self, layer_idx, true_outputs, acc_error, valid_idx=[]):
        layers = self.__quan_graph.get_layers() # type: ignore
        log_infos = []
        (inputs, in_data_idx), in_data = self.get_layer_input_info(layer_idx), list()
        for vidx, l_idx in enumerate(inputs.keys()):
            ilayer, iidx = layers[l_idx], inputs[l_idx]
            outputs, single_out = ilayer.get_out_data(), dict()
            if not acc_error or vidx in valid_idx:
                in_quantize = ilayer.get_quantize()
                true_out = true_outputs[ilayer.get_onnx_output_name()[iidx]]                
                # if iidx < 0:
                #     iidx += len(in_quantize['feat'])
                # print(in_quantize['feat'], iidx)
                if isinstance(outputs, list):
                    single_out = copy.deepcopy(outputs[iidx])
                    if iidx < 0:
                        iidx += np.sum([1 for t in in_quantize['feat'].keys() if "so" in t])
                    if single_out['output'].dtype in [np.float32, np.float64]:
                        single_out['output'] = true_out
                    else:
                        single_out['output'] = in_quantize['feat']['so' + str(iidx)].get_quan_data(true_out)
                    log_info = "layer input shape: {}".format(outputs[iidx]["output"].shape)
                    log_infos.append(log_info)
                else:
                    single_out = copy.deepcopy(outputs)
                    if iidx < 0:
                        iidx += np.sum([1 for t in in_quantize['feat'].keys() if "so" in t])
                    if single_out['output'].dtype in [np.float32, np.float64]:
                        single_out['output'] = true_out
                    else:
                        single_out['output'] = in_quantize['feat']['so' + str(iidx)].get_quan_data(true_out)
                    log_info = "layer input shape: {}".format(outputs["output"].shape)
                    log_infos.append(log_info)
            else:
                if isinstance(outputs, list):
                    single_out = copy.deepcopy(outputs[iidx])
                else:
                    single_out = copy.deepcopy(outputs)                
            in_data.append(single_out)
            [in_data.append(copy.deepcopy(single_out)) for _ in range(1, in_data_idx[l_idx])]

        return in_data, log_infos
    
    # normalization data, data type is float or int8
    def forward(self, in_data, acc_error=False, true_outputs=None, isp_data=False):
        layers = self.get_layers()

        layers_info_fp_input = dict(layer_names=[], layer_in_idx=[])
        for idx, layer in enumerate(layers):
            # ops_outputs = list()
            # ops_instance = layer.get_ops_instance()
            layer_type = layer.get_layer_type()

            # setting = layer.get_ops_setting()

            if layer_type == 'data':
                if isinstance(in_data, dict):
                    layer.forward(in_data[layer.get_onnx_output_name()[0]], isp_data=isp_data)
                else: 
                    first_layer = layers[layer.get_output_idx()[0]]
                    first_conv_layer = True if first_layer.get_layer_type() \
                        in ["conv", "convtranspose", "depthwiseconv"] else False
                    if first_conv_layer and not isp_data:
                        pad_t, pad_l, pad_b, pad_r = first_layer.get_ops_setting()["attrs"][0]["pads"] 
                        N, C, H, W = in_data.shape
                        H += (pad_t + pad_b)                        
                        in_data_tmp = np.zeros([N, C, H, W], dtype=np.float32)
                        in_data_tmp[:, :, pad_t:H-pad_b, :] = in_data
                        in_data = in_data_tmp
                    layer.forward(in_data, isp_data=isp_data)
            else:
                if not acc_error:
                    layers_info_fp_input = dict(layer_names=[], layer_in_idx=[])
                    inputs, info = self.get_qtrue_qdata(idx, true_outputs, acc_error)
                else:
                    layer_name = layer.get_layer_name()
                    if layer_name in layers_info_fp_input["layer_names"]:
                        idx_ = layers_info_fp_input["layer_names"].index(layer_name) 
                        valid_idx = layers_info_fp_input["layer_in_idx"][idx_]  
                        inputs, info = self.get_qtrue_qdata(idx, true_outputs, acc_error, valid_idx=valid_idx) 
                    else:                
                        inputs, info = self.get_layer_input(idx)

                # log_info = info
                # log_infos.extend(log_info)

                layer.forward(inputs, isp_data=isp_data)
            # if self.print_log:
            #     for log_info in log_infos:
            #         self.logger.info(log_info)

            # layer.set_out_data(ops_outputs)
            # if self.print_log:
            #    self.logger.info(
            #        '--------------------------------------------------------------------------------------------------')
            # self.logger.info('simulation done! Cosine similarity = {}, MAE = {}'.format(sim1, sim2))

    # de-quantize result layer
    # result layer output maybe float
    def process_output(self, true_outputs):
        tresluts, resluts = dict(), dict()
        for idx, layer in enumerate(self.__quan_graph.get_layers()): # type: ignore
            if layer.get_is_result_layer():
                layer_type = layer.get_layer_type()
                name = layer.get_onnx_output_name()
                outputs = layer.get_out_data()
                ops = layer.get_ops_instance()
                # process specical layer outputs
                # filter useless output, ex: shuffle
                # get multi-outputs, ex: split
                if layer_type in ['shuffle']:
                    name = name[-2:]
                    outputs = outputs[-2:]
                    out_ops = ops[-2:] if len(ops) > 4 else ops[-1]
                elif layer_type in ['shuffle_only_split']:
                    name = name[-2:]
                    outputs = outputs[-2:]
                    out_ops = ops[-2:] if len(ops) > 3 else ops[-1]
                elif layer_type in ['lstm', 'gru']:
                    fscales = layer.get_scale()
                    for idx, name_ in enumerate(name):
                        ### output idx is result layer, but not result branch.
                        if outputs[idx]['output'].dtype in ["int8", "int32", "int16", "int64"]:
                            resluts[name_] = (outputs[idx]['output'] * fscales[idx]['scale']).astype(np.float32)
                        else:
                            resluts[name_] = outputs[idx]['output']
                            
                    continue

                elif layer_type in ['concat', 'reshape']:
                    output = layer.get_quantize()['feat']['so0'].get_dequan_data(outputs['output'])
                    name_ = name[0]
                    tresluts[name_] = true_outputs[name_] if true_outputs else None
                    resluts[name_] = output.astype(np.float32)
                    
                    # return dict(qout=resluts, trueout=tresluts)
                    continue
                elif layer_type in ['mul', 'cmul', 'pmul']:
                    if not outputs['output'].dtype in [np.float32, np.float64]:
                        output = layer.get_quantize()['feat']['so0'].get_dequan_data(outputs['output'])
                    else:
                        output = outputs['output']
                    name_ = name[0]
                    tresluts[name_] = true_outputs[name_] if true_outputs else None
                    resluts[name_] = output.astype(np.float32)
                    continue
                else:
                    if isinstance(outputs, list):
                        outputs = outputs[-len(name):]
                        out_ops = ops[-len(name):]
                    else:
                        out_ops = [ops]
                        outputs = [outputs]

                def dequant_scale(in_data, scales, so):
                    # Extract zero point and scale
                    zero_point = scales['zo']
                    scale = scales.get('fscale', so['scale'])

                    # Ensure scale is a NumPy array
                    if not isinstance(scale, np.ndarray):
                        f_zero_point = np.float32(zero_point)
                        if "fscale" in scales.keys():
                            f_zero_point = 0#np.float32(zero_point) / (scale / so['scale'])
                            
                        output = in_data.astype(np.float32) - f_zero_point
                        return np.float32(output * scale)

                    # Handle input of shape (N, C, H, W)
                    if len(in_data.shape) == 4:
                        n, c, h, w = in_data.shape
                        f_zero_point = np.full_like(scale, zero_point).reshape(n, c, 1, 1)
                        if "fscale" in scales.keys():
                            f_zero_point = 0#np.float32(zero_point) / (scale / so['scale']).reshape(1,-1,1,1)
                        output = in_data - f_zero_point
                        return output * scale.reshape(n, c, 1, 1)

                    # Handle input of shape (N, C)
                    if len(in_data.shape) == 2:
                        n, c = in_data.shape
                        f_zero_point = np.full_like(scale, zero_point).reshape(n, c)
                        if "fscale" in scales.keys():
                            f_zero_point = 0#np.float32(zero_point) / (scale / so['scale']).reshape(1, -1)
                        output = in_data - f_zero_point
                        return output * scale.reshape(n, c)

                    # Raise an exception for invalid input shapes
                    raise ValueError('last layer dequant_scale failed!!!')

                # consider multi-output layer, ex: split
                all_idx, so = 0, layer.get_scale()

                for op_idx, op in enumerate(out_ops):
                    scales = op.get_scales()  # post-shift in postprocess
                    if true_outputs:
                        tresluts[name[all_idx]] = true_outputs[name[all_idx]]
                    else:
                        tresluts[name[all_idx]] = None

                    if isinstance(scales, list):
                        for s_idx, scale in enumerate(scales):
                            output = dequant_scale(outputs[op_idx][s_idx]['output'], scale, so[all_idx])
                            # output = outputs[op_idx][s_idx]['output']
                            resluts[name[all_idx]] = output.astype(np.float32)
                            all_idx += 1
                    else:  # post-shift
                        if outputs[op_idx]['output'].dtype in [np.float32, np.float64]:
                            output = outputs[op_idx]['output'].astype(np.float32)
                        else:
                            output = dequant_scale(outputs[op_idx]['output'], scales, so[all_idx])
                        # output = outputs[op_idx]['output']
                        resluts[name[all_idx]] = output.astype(np.float32)
                        all_idx += 1

                # if isinstance(outputs, list):
                #     outputs = outputs[-1]
                # if layer.get_layer_type() in ['conv', 'depthwiseconv', 'batchnormalization', 'fc', 'shuffle']:
                #     ops = layer.get_ops_instance()
                #     if isinstance(ops, list):
                #         op = copy.deepcopy(ops[-1])
                #     else:
                #         op = copy.deepcopy(ops)
                #     scales = op.get_scales()
                #     scale = scales['fscale'] if 'fscale' in scales.keys() else scales['out_scale']
                #     if isinstance(scale, np.ndarray):
                #         scale = scale.reshape(1, -1, 1, 1)
                #     output = outputs['output'] * scale
                # else:
                #     output = outputs['output'] * extract_scale(layer.get_scale())
                # resluts[name] = output.astype(np.float32)

        return dict(qout=resluts, trueout=tresluts)

    def print_layer_info(self):
        layers = self.__quan_graph.get_layers() # type: ignore
        for idx, layer in enumerate(layers):
            try:
                layer_type = layer.get_layer_type()
                log_infos = []

                log_info = '--------------------------------------------------------------------------------------------------\n'
                # log_infos.append('************************** Quatization graph info of layers as below ***************************\n')
                log_infos.append(log_info)
                log_info = 'layer length is: {}, current layer is: {}, layer type is: {}'.format(len(layers), idx,
                                                                                                layer_type)
                log_infos.append(log_info)
                setting = layer.get_ops_setting()
                if layer_type == 'lstm':
                    continue  # TODO
                if not layer.is_extend():
                    log_info = 'input idx: {}, output idx: {}, layer name: {}'.format(
                                layer.get_input_idx(), 
                                layer.get_output_idx(), 
                                layer.get_layer_name())
                    log_infos.append(log_info)

                    for k, v in {'si': 'in_scale', 'sk': 'w_scale', 'so': 'scale'}.items():
                        if isinstance(setting[v], list):
                            tmp = []
                            for value in setting[v]:
                                tmp.append(value['scale'])
                            log_info = '{}: {}'.format(k, tmp)
                            log_infos.append(log_info)
                        else:
                            log_info = '{}: {}'.format(k, setting[v]['scale'])
                            log_infos.append(log_info)

                    for k, v in {'zi': 'in_scale', 'zk': 'w_scale', 'zo': 'scale'}.items():
                        if isinstance(setting[v], list):
                            tmp = []
                            for value in setting[v]:
                                tmp.append(value['zero_point'])
                            log_info = '{}: {}'.format(k, tmp)
                            log_infos.append(log_info)
                        else:
                            log_info = '{}: {}'.format(k, setting[v]['zero_point'])
                            log_infos.append(log_info)

                    log_infos.append(log_info)
                    log_info = 'ops: {}'.format(setting['ops_string'])
                    log_infos.append(log_info)
                else:
                    log_info = 'layer is user define!\n'
                    log_infos.append(log_info)
                    log_info = 'input idx and output idx not get form layer define functional!\n'
                    log_infos.append(log_info)
                    sk, so = layer.get_w_scale()['scale'], layer.get_scales()['scale']
                    zk, zo = layer.get_w_scale()['zero_point'], layer.get_scales()['zero_point']
                    log_info = 'sk, so: {}, {}'.format(sk, so)
                    log_infos.append(log_info)
                    log_info = 'zk, zo: {}, {}'.format(zk, zo)
                    log_infos.append(log_info)

                for log_info in log_infos:
                    self.logger.info(log_info)
                self.logger.info(
                    '--------------------------------------------------------------------------------------------------')
            except:
                print("print {} layer error info failure!".format(layer.get_layer_name())) #type: ignore
                os._exit(-1)

    def __call__(self, in_data, **kwargs):
        acc_error = kwargs.get('acc_error', False)
        true_outputs = kwargs.get('onnx_outputs', None)
        isp_data = kwargs.get('isp_data', False)
        # if not true_outputs:
        #     self.logger.fatal('true output is not exists!')
        if 'level' in kwargs.keys():
            self.set_simu_level(kwargs['level'])
            self.logger.info('reset level is: {}'.format(self.__simulation_level))
        # data = process_im(in_data)
        self.forward(in_data, acc_error=acc_error, true_outputs=true_outputs, isp_data=isp_data)
        return dict(results=self.process_output(true_outputs))
