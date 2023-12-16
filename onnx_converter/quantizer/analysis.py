# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/14 19:57
# @File     : analysis.py

# just analysis weights, because feat not extract here, datasets feat maybe using large memory
# analysis statistics of weight or feat, decided to use different strategy
# for example: per channel quantize

import os
import numpy as np
from operator import itemgetter
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    from utils import Object
except:
    from onnx_converter.utils import Object # type: ignore
         
         
class Notice(object):
    def __init__(self):
        super(Notice, self).__init__()

        self.observers = []
        
    def attach(self, observer):
        self.observers.append(observer)
        
    def detach(self, observer):
        self.observers.remove(observer)
        
    def notify(self):
        for observer in self.observers:
            observer.update(self)
            
            
class DataNotice(Notice):
    def __init__(self):
        super(DataNotice, self).__init__()
        self.__quan_graph = list()
        
    def get_graph(self):
        return self.__quan_graph
    
    def set_graph(self, quan_graph):
        self.__quan_graph = quan_graph
        
        
class Observer(object):
    def __init__(self, **kwargs):
        super(Observer, self).__init__()
        self.is_observer = kwargs['is_observer']
        self.is_always = kwargs['is_always'] 
        self.bins = kwargs['bins']
        self.is_pertensor = kwargs['is_pertensor']
        self.check_error = kwargs['check_error'][kwargs['metric']]

    def get_histograms_perchannel(self):
        pass
    
    def get_histograms_perchtensor(self):
        pass
        
    def update(self, quant_graph, onnx_outputs):
        if self.is_observer:
            self.analysis(quant_graph, onnx_outputs) # type: ignore
            self.is_observer = self.is_always
      
                                    
class HistogramWeightObserver(Observer, Object): # type: ignore
    def __init__(self, **kwargs):
        super(HistogramWeightObserver, self).__init__(**kwargs)

        self.log_dir = kwargs['log_dir']
        self.log_name = kwargs["log_name"] if "log_name" in kwargs.keys(
        ) else "analysis_weight.log"
        self.log_level = kwargs.get('log_level', 20)
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level)
        self.output_dir = os.path.join(self.log_dir, 'analysis_weight')
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, mode=0o777, exist_ok=True)

        self.layers = list()
        self.valid_layer_type = [
            'conv', 'depthwiseconv', 'fc', 'matmul', 'gemm'
        ]
        
    def get_histograms_perchannel(self):
        pass
    
    def get_histograms_perchtensor(self):
        pass

    def get_weight_distribution(self):
        for layer in tqdm(self.layers, postfix='layers weight distribution'):
            layer_type = layer.get_layer_type()
            if layer_type in self.valid_layer_type:
                weights = [layer.get_weight(), layer.get_qweight()]
                node_name = layer.get_nodes()[0].get_name()
                layer_idx = layer.get_idx()
                bit_select = layer.get_ops_setting()['setting']['bit_select']
                sk = layer.get_w_scale()['scale']
                zk = layer.get_w_scale()['zero_point']
                qmin = layer.get_ops_setting()['setting']['mins'][bit_select]
                qmax = layer.get_ops_setting()['setting']['maxs'][bit_select]
                threshold = qmax * sk
                for i, data in enumerate(weights):
                    data = data.reshape(-1).astype(np.float32)
                    min_val, max_val = data.min(), data.max()
                    plt.figure(1)
                    plt.subplot(1, len(weights), i + 1)
                    n, bins, patches = plt.hist(data,
                                                bins=self.bins, #qmax - qmin + 1,
                                                range=(min_val, max_val),
                                                density=True)
                    mu, sigma = np.mean(data), np.sqrt(np.var(data))
                    plt.plot(bins,
                             norm.pdf(bins, mu, sigma),
                             'r-',
                             linewidth=1)
                    # plt.text(0, 0, 'min_val={}, max_val={}'.format(min_val, max_val), fontsize=1)
                image_name = str(self.sorted_layer[layer_idx]).zfill(3) + \
                            '_layer' + str(layer_idx).zfill(3) + '_' + \
                            layer_type + '_' + node_name.replace('/', '_') + '.jpg'
                plt.savefig(os.path.join(self.output_dir, image_name))
                plt.close()
            else:
                pass

    def get_weight_error(self):
        error_dict = dict()
        layer_info = dict()
        for layer in tqdm(self.layers, postfix='layers weight error'):
            layer_type = layer.get_layer_type()
            if layer_type in self.valid_layer_type:
                node_name = layer.get_nodes()[0].get_name()
                layer_idx = layer.get_idx()
                sk = layer.get_w_scale()['scale']
                zk = layer.get_w_scale()['zero_point']
                weight, qweight = layer.get_weight(), layer.get_qweight()
                mu, sigma = np.mean(weight), np.sqrt(np.var(weight))
                qweight = layer.get_quantize()['w_quan'].get_dequan_data(
                    qweight)
                error_dict[node_name] = self.check_error(qweight, weight)
                layer_info[node_name] = [layer_idx, layer_type, sk, mu, sigma]

        self.sorted_layer = dict()
        sorted_error = sorted(error_dict.items(),
                              key=itemgetter(1),
                              reverse=True)
        for idx, (node_name, error) in enumerate(sorted_error):
            layer_idx, layer_type, sk, mu, sigma = layer_info[node_name]
            self.sorted_layer[layer_idx] = idx
            self.logger.info('-------------------------------------------------------------------------')
            self.logger.info('layer_idx: {}, layer_type: {}'.format(layer_idx, layer_type))
            self.logger.info('node_name: {}, error is: {}'.format(node_name, error))
            self.logger.info('-------------------------------------------------------------------------')
            
    def analysis(self, quant_graph, onnx_outputs):
        self.layers = quant_graph.get_graph().get_layers()
        self.get_weight_error() 
        self.get_weight_distribution()


class HistogramFeatureObserver(Observer, Object): # type: ignore
    def __init__(self, **kwargs):
        super(HistogramFeatureObserver, self).__init__(**kwargs)

        self.log_dir = kwargs['log_dir']
        self.log_name = kwargs["log_name"] if "log_name" in kwargs.keys(
        ) else "analysis_feat.log"
        self.log_level = kwargs.get('log_level', 20)
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level)
        self.output_dir = os.path.join(self.log_dir, 'analysis_feature')
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, mode=0o777, exist_ok=True)
            
        self.layers = list()
        self.error_dict = dict()
        self.layer_info = dict()
        
    def get_histograms_perchannel(self):
        pass
    
    def get_histograms_perchtensor(self):
        pass

    def get_feature_distribution(self, qdata, fdata, layer, output_idx):
        layer_type = layer.get_layer_type()
        feats = [fdata, qdata]
        node_name = layer.get_nodes()[0].get_name()
        layer_idx = layer.get_idx()
        bit_select = layer.get_ops_setting()['setting']['bit_select']
        qmin = layer.get_ops_setting()['setting']['mins'][bit_select]
        qmax = layer.get_ops_setting()['setting']['maxs'][bit_select]
        for i, data in enumerate(feats):
            data = data.reshape(-1).astype(np.float32)
            min_val, max_val = data.min(), data.max()
            plt.figure(1)
            plt.subplot(1, len(feats), i + 1)
            n, bins, patches = plt.hist(data,
                                        bins=self.bins, #qmax - qmin + 1,
                                        range=(min_val, max_val),
                                        density=True)
            mu, sigma = np.mean(data), np.sqrt(np.var(data))
            plt.plot(bins,
                        norm.pdf(bins, mu, sigma),
                        'r-',
                        linewidth=1)
        image_name = str(layer_idx).zfill(3) + '_' + \
                    layer_type + '_' + node_name.replace('/', '_') + \
                    '_{}.jpg'.format(output_idx)
        plt.savefig(os.path.join(self.output_dir, image_name))
        plt.close()

    def save_feature_error(self, error, layer, output_idx):
        layer_type = layer.get_layer_type()
        node_name = layer.get_nodes()[0].get_name()
        layer_idx = layer.get_idx()
        self.error_dict[node_name + '_{}'.format(output_idx)] = error
        self.layer_info[node_name + '_{}'.format(output_idx)] = [layer_idx, layer_type]
    
    def get_feature_error(self):
        sorted_error = sorted(self.error_dict.items(),
                              key=itemgetter(1),
                              reverse=True)
        for idx, (node_name, error) in enumerate(sorted_error):
            layer_idx, layer_type = self.layer_info[node_name]
            self.logger.info('-------------------------------------------------------------------------')
            self.logger.info('layer_idx: {}, layer_type: {}'.format(layer_idx, layer_type))
            self.logger.info('node_name: {}, error is: {}'.format(node_name, error))
            self.logger.info('-------------------------------------------------------------------------')
                        
    def analysis(self, quant_graph, onnx_outputs):
        self.layers = quant_graph.get_graph().get_layers()
        for layer in self.layers:
            qout, quantize = layer.get_out_data(), layer.get_quantize()
            onnx_name = layer.get_onnx_output_name()
            qtrues, ftrues = list(), list()
            for idx in range(len(onnx_name)):
                qtrues.append(quantize['feat']['so' + str(idx)].get_quan_data(onnx_outputs[onnx_name[idx]]))
                ftrues.append(onnx_outputs[onnx_name[idx]])                

            if isinstance(qout, dict):
                output = qout['output']
                if isinstance(output, list):
                    for idx in range(len(qtrues)):
                        q_idx = idx - len(qtrues)
                        name = onnx_name[q_idx]
                        error = self.check_error(output[q_idx], qtrues[q_idx])
                        self.save_feature_error(error, layer, q_idx)
                        self.get_feature_distribution(output[q_idx], ftrues[q_idx], layer, q_idx)
                else:
                    error = self.check_error(output, qtrues[-1])
                    self.save_feature_error(error, layer, -1)
                    self.get_feature_distribution(output, ftrues[-1], layer, -1)
            else:
                for idx in range(len(qtrues)):
                    q_idx = idx - len(qtrues)
                    name = onnx_name[q_idx]
                    error = self.check_error(qout[q_idx]['output'], qtrues[q_idx])
                    self.save_feature_error(error, layer, q_idx)
                    self.get_feature_distribution(qout[q_idx]['output'], ftrues[q_idx], layer, q_idx)
        
        self.get_feature_error()