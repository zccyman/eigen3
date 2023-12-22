# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : nan.qin
# @Company  : SHIQING TECH
# @Time     : 2022/4/28 14:48
# @File     : OnnxConverter.py

import cv2
import copy
import onnx
import numpy as np
# import torch
# from scipy.special import softmax
import onnxruntime as rt
from tkinter import _flatten # type: ignore
import os

try:
    from checkpoint import OnnxParser
    from graph import Graph
    from config import Config
    #from config import cfg_quan, cfg_export, cfg_parse, cfg_graph
    from export import mExportV3, mExportV2, mExportV1
    from quantizer import GraphQuant, GrapQuantUpgrade, AlreadyGrapQuant
    from simulator import Simulation, ErrorAnalyzer, error_factory
    #from .utils import Logging
    from utils import onnxruntime_infer, add_layer_output_to_graph, Object, props_with_
    #from .tools import ModelProcess
    from tools import PostTrainingQuan, OnnxruntimeInfer
    from tools import  QuanTableParser
    from simulator import PerfAnalyzer
except:
    from onnx_converter.checkpoint import OnnxParser # type: ignore
    from onnx_converter.graph import Graph # type: ignore
    from onnx_converter.config import Config # type: ignore
    from onnx_converter.export import mExportV3, mExportV2, mExportV1 # type: ignore
    from onnx_converter.quantizer import GraphQuant, GrapQuantUpgrade, AlreadyGrapQuant # type: ignore
    from onnx_converter.simulator import Simulation, ErrorAnalyzer, error_factory # type: ignore
    from onnx_converter.utils import onnxruntime_infer, add_layer_output_to_graph, Object, props_with_ # type: ignore
    from onnx_converter.tools import PostTrainingQuan, OnnxruntimeInfer # type: ignore
    from onnx_converter.tools import QuanTableParser # type: ignore
    from onnx_converter.simulator import PerfAnalyzer # type: ignore


class OnnxConverter(Object): # type: ignore
    def __init__(self, **kwargs):
        super(OnnxConverter, self).__init__(**kwargs)
        self.kwargs = copy.deepcopy(kwargs)
        self.model_path = "test.onnx"
        self.offline_quant_mode =False
        if "use_offline_quantization" in self.kwargs['quantization_args'].keys():
            self.offline_quant_mode = self.kwargs['quantization_args']["use_offline_quantization"]
            if self.offline_quant_mode==True:
                self.offline_quan_tool = self.kwargs['quantization_args']["offline_quantization_tool"]
                self.offline_quan_table = self.kwargs['quantization_args']["quantization_table"]
        self.__export_version = "V3"
        self.__sess_options = None
        
        #self.input_names, self.output_names = self.get_io_names(self.model_path)
        # create session with all ftms as outputs
        #self.create_session(self.model_path)
        #self.ftm_names = self.get_output_name(self.sess)
        self.chip_model = self.kwargs.get('chip_model', None)
        self.bgr_format = self.kwargs.get('bgr_format', True)
        self.user_model = self.kwargs.get('user_model', False)
        self.log_name = self.kwargs['log_args']['log_name']
        self.log_level = self.kwargs['log_args']['log_level']
        self.is_stdout = self.kwargs.get('is_stdout', True)
        self.is_assert = self.kwargs.get('is_assert', True)
        self.device = self.kwargs.get('device', 'cpu')
        self.transform = self.kwargs.get("transform", lambda x:x)
        self.postprocess = self.kwargs.get("postprocess", None)

        self.__logger = self.get_log(log_name=self.log_name, log_level=self.log_level)
        self.__model_type = self.kwargs.get('model_type', None)
        self.is_simplify = self.kwargs.get('is_simplify', True)
            
        try:
            # from file import default configs
            self.__parse_cfg = 'config/parse.py'
            self.__graph_cfg = 'config/graph.py'
            self.__quantize_cfg = 'config/quantize.py'
            if self.__model_type == "AUDIO":
                self.__voice_quan_cfg ='config/quantize_voice.py'
                
            export_cfg = 'config/export_{}.py'.format(self.__export_version.lower())
            export_cfg_files = [
                os.path.join(os.path.split(export_cfg)[0], "export.py"), 
                export_cfg,
            ]    
            self.__export_dict = {}
            for export_cfg in export_cfg_files:
                cfg_dict = self.__parse_config(export_cfg)
                cfg_dict.update(dict(is_stdout=self.is_stdout)) # type: ignore
                self.__export_dict.update(cfg_dict)
            
            self.__graph_dict = self.__parse_config(self.__graph_cfg)
            self.__parse_dict = self.__parse_config(self.__parse_cfg)
            self.__quan_dict = self.__parse_config(self.__quantize_cfg)
            if self.__model_type == "AUDIO":
                self.__voice_quan_dict = self.__parse_config(self.__voice_quan_cfg)
                self.__quan_dict = Config._merge_a_into_b(self.__voice_quan_dict, self.__quan_dict)
        except:
            from onnx_converter.config import parse # type: ignore
            from onnx_converter.config import graph # type: ignore
            from onnx_converter.config import quantize # type: ignore
            from onnx_converter.config import export # type: ignore
            self.__export_dict = props_with_(export)
            if self.__model_type == "AUDIO":
                from onnx_converter.config import voice_quantize # type: ignore
            if self.__export_version == "V1":
                from onnx_converter.config import export_v1 # type: ignore
                self.__export_dict = Config._merge_a_into_b(props_with_(export_v1), self.__export_dict)       
            elif self.__export_version == "V2":
                from onnx_converter.config import export_v2 # type: ignore
                self.__export_dict = Config._merge_a_into_b(props_with_(export_v2), self.__export_dict)  
            elif self.__export_version == "V3":
                from onnx_converter.config import export_v3 # type: ignore
                self.__export_dict = Config._merge_a_into_b(props_with_(export_v3), self.__export_dict)                       
            else:
                raise NotImplemented
            self.__graph_dict = props_with_(graph)
            self.__parse_dict = props_with_(parse)
            self.__quan_dict = props_with_(quantize)
            if self.__model_type == "AUDIO":
                self.__voice_quan_dict = props_with_(voice_quantize) # type: ignore
                self.__quan_dict = Config._merge_a_into_b(self.__voice_quan_dict, self.__quan_dict)

        #update user defined quan args
        self.__parse_dict['is_simplify'] = self.is_simplify # type: ignore
        if self.kwargs['quantization_args']:
            self.__update_cfg(self.kwargs['quantization_args'])
        
        # 0--no error, 1--layer error, 2--deep error analyzer
        self.do_check_error = self.kwargs['layer_error_args'].get("do_check_error", 0)# simulation with error print
        self.acc_error = self.kwargs['layer_error_args'].get("acc_error", True)# acc error during inference entire model
         
        self.__layer_average_error = dict()
        self.error_metrics = self.kwargs['layer_error_args'].get("metrics_list", ["Cosine", "L1", "L2"])
        if self.do_check_error > 0:
            self.__check_error = dict()
            for metrics in self.error_metrics:
                self.__check_error.update({metrics : error_factory.get(metrics)()}) # type: ignore
        else:
            self.__check_error = None
        
        self.input_names = self.kwargs.get('input_names', None)
        
        self.__exporter = None
        
    def __parse_config(self, cfg_file):
        save_cfg = Config.fromfile(cfg_file)
        cfg_dict, cfg_text = save_cfg._file2dict(cfg_file)
        return cfg_dict

    def __create_session(self,model_path):
        model = onnx.load(model_path)
        for node in model.graph.node: # type: ignore
            for output in node.output:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)]) # type: ignore
        
        self.sess = rt.InferenceSession(model.SerializeToString(), sess_options=self.__sess_options, providers=["CPUExecutionProvider"])  # type: ignore

    def __update_cfg(self, quan_args):
        # Update the user defined quantization configs
        # Make sure input configs exist and valid
        out_type_dict={'uint8':0, 'int8':1, 'uint16': 2, 'int16':3, 'uint32':4, 'int32':5, 'uint64':6, 'int64':7, 'float32':8}

        assert {'out_type', 'method', 'process_scale'} <= set(quan_args.keys()),\
            'out_type, method,bit_width and process_scale should be specified, please check'
        assert quan_args['method']['feature'][-1] in [8, 16], "bit_width only support 8 or 16"
        assert quan_args['method']['weight'][-1] in [8, 16], "bit_width only support 8 or 16"
        assert quan_args['out_type'].lower() in out_type_dict.keys(),\
            'You specified invalid out_type, which should be one of {}'.format(out_type_dict.keys())
        assert quan_args['process_scale'] in ['intscale', 'intscaleex', 'floatscale', 'shiftfloatscale','ffloatscale', 'table'],\
            'You specified invalid process_scale, please check the data type dict'
        
        out_type_str = quan_args['out_type']
        out_type_id = out_type_dict[out_type_str.lower()]
        self.__quan_dict['out_type'] = out_type_id # type: ignore
        #self.quan_dict['bit_select'] = out_type_id    #### right??
         
        self.__quan_dict['txme_saturation'] = 1 # default open # type: ignore

        self.__quan_dict['process_scale'] = quan_args['process_scale'] # type: ignore
        
        feat_bitwidth = quan_args['method']['feature'][-1]
        w_bitwidth = quan_args['method']['weight'][-1]
        self.__quan_dict['int_scale'] = feat_bitwidth # type: ignore
        self.__quan_dict['pre_int_scale'] = feat_bitwidth - 1 # type: ignore
        #self.quan_dict['bit_select'] = 1 if bitwidth == 8 else 3 # type: ignore

        feat_method = quan_args['method']['feature']
        if "symm" == feat_method[0]:
            self.__quan_dict['feat']['method'] = 'floatsymquan' # type: ignore
        else:
            self.__quan_dict['feat']['method'] = 'floatquan' # type: ignore
            self.__quan_dict['txme_saturation'] = 0 # type: ignore
        self.__quan_dict['feat']['bit_select'] = 1 if feat_bitwidth == 8 else 3 # type: ignore
            
        weight_method = quan_args['method']['weight']
        self.__quan_dict['normal']['bit_select'] = 1 if w_bitwidth == 8 else 3 # type: ignore
        self.__quan_dict['perchannel']['bit_select'] = 1 if w_bitwidth == 8 else 3 # type: ignore
        if 'symm' == weight_method[0]:
            self.__quan_dict['normal']['method'] = 'floatsymquan' # type: ignore            
            self.__quan_dict['perchannel']['method'] = 'perchannelfloatsymquan' # type: ignore            
            self.__quan_dict["int16"]["method"] = 'floatsymquan' # type: ignore
        else:
            self.__quan_dict['normal']['method'] = 'floatquan' # type: ignore
            self.__quan_dict['perchannel']['method'] = 'perchannelfloatquan' # type: ignore
            self.__quan_dict["int16"]["method"] = 'floatquan' # type: ignore
            self.__quan_dict['txme_saturation'] = 0 # type: ignore
            
        self.virtual_round = self.__quan_dict["virtual_round"]
        if self.__quan_dict['process_scale'] in ["floatscale", "shiftfloatscale"]:
            self.virtual_round = 1
            self.__quan_dict["virtual_round"] = 1
        if self.virtual_round == 3 and quan_args['process_scale'] == "intscaleex":
            self.__quan_dict["int_scale"] -= 1 # type: ignore

        for key in self.__quan_dict["default_setting"].keys(): # type: ignore
            layer_info = self.__quan_dict["default_setting"][key] # type: ignore
            # layer_info["precision"]=self.quan_dict["precision"] # closed by qn
            if key in ['shuffle', 'concat_shuffle_only', 'add', 'concat', 'split', 'slice', 'sub', 
                       'resize']:
                layer_info["int_scale"]=self.__quan_dict["pre_int_scale"] # type: ignore
            else:
                layer_info["int_scale"]=self.__quan_dict["int_scale"] # type: ignore
            layer_info["out_type"] = self.__quan_dict["out_type"] # type: ignore
            layer_info["feat"] = self.__quan_dict["feat"] # type: ignore

            if key in ['conv','depthwiseconv','fc', 'gemm', 'matmul']:
                layer_info["process_scale"] = "shiftfloatscaletable" if "table" in quan_args["process_scale"] \
                    else quan_args["process_scale"]
                if quan_args['method']['weight'][1] == "per_channel":
                    layer_info['weights'] = self.__quan_dict['perchannel'] # type: ignore                    
                else:
                    layer_info['weights'] = self.__quan_dict['normal'] # type: ignore
                #layer_info['process_scale'] = quan_args['process_scale'] # closed by qn
        
        # set export chip model
        if self.chip_model == "AT5050_C_EXTEND":
            self.__export_dict['AT5050_C_EXTEND'] = True # type: ignore
            self.__export_dict['Csize'], self.__export_dict['Ksize'] = 16, 32  # type: ignore
            self.__export_dict['bits']['Csize'] = self.__export_dict['Csize'] # type: ignore
            
        else:
            self.__export_dict['Csize'], self.__export_dict['Ksize'] = 8, 8 # type: ignore
            self.__export_dict['AT5050_C_EXTEND'] = False # type: ignore
        self.__export_dict['bits']['Ksize'] = self.__export_dict['Ksize'] # type: ignore
        self.__export_dict['bits']['DATA_C_EXTEND']=self.__export_dict['AT5050_C_EXTEND'] # type: ignore
        self.__export_dict['bits']['bgr_format'] = self.bgr_format # type: ignore
        self.__export_dict['bits']['save_placeholder_params'] = not self.user_model # type: ignore

        # process floatscale
        self.__update_tsme_table()

    def __update_tsme_table(self):
        if self.__quan_dict["process_scale"] not in ["table"]: # type: ignore  
            # is_last_layer_fuse_act = False 
            is_fuse_linear_act = False
            is_fuse_nonlinear_act = False            
            act = []
            if is_fuse_nonlinear_act:
                fuse_act = ["swish", "leakyrelu", "hardswish", "hardsigmoid", "tanh", "sigmoid"]
            else:
                fuse_act = []
            if is_fuse_linear_act:
                fuse_act.extend(["relu", "relu6", "relux"])
            else:
                act.extend(["relu", "relu6", "relux"])
            self.__quan_dict.update(dict(fuse_act=fuse_act, act=act)) # type: ignore

    def __get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def __build_fp_graph(self, model_path, act=None, is_simplify=False):
        
        self.__parse_dict.update({'log_name':self.log_name, 'log_level':self.log_level}) # type: ignore
        
        if self.input_names != None:
            self.__parse_dict.update({'input_names':self.input_names}) # type: ignore
        
        parser = OnnxParser(**self.__parse_dict) # type: ignore
        
        nodes, model, self.model_input_names, self.model_output_names, self.__sess_options = parser.parse(model_path)
        self.opset_version = parser.get_opset_version()
        args = dict(
            nodes=nodes, 
            act=self.__quan_dict['act'], # type: ignore, 
            opset_version = self.opset_version,
            especial_ops = self.__graph_dict['especial_ops'],  # type: ignore
            fuse_ops = self.__graph_dict['fuse_ops'], # type: ignore
            split_ops = self.__graph_dict['split_ops'],  # type: ignore
            replace_ops = self.__graph_dict['replace_ops'], # type: ignore
            default_setting=self.__quan_dict['default_setting'], # type: ignore
            fuse_act=self.__quan_dict["fuse_act"], # type: ignore
            is_last_layer_fuse_act=self.__quan_dict["is_last_layer_fuse_act"], # type: ignore
            log_name = self.log_name,
            log_level=self.log_level,
            )
        try:
            self.graph = Graph(**args)
            self.graph.build()
            #args.pop('especial_ops')
            return self.graph, model
        except:
            self.__logger.error('timeintelli graph build failed!')
            os._exit(-1)
   
    def __create_quan_graph(self):
        act = self.__quan_dict['act'] # type: ignore
        bits_dict = self.__quan_dict['bits_dict'] # type: ignore
        # *|MARKER_CURSOR|*
        bit_select, maxs, mins = self.__quan_dict['bit_select'], self.__quan_dict['maxs'], self.__quan_dict['mins'] # type: ignore
        # precision removed by qn
        # precision, int_scale, preint_scale = self.quan_dict['precision'], self.quan_dict['int_scale'], self.quan_dict['pre_int_scale']
        int_scale, preint_scale = self.__quan_dict['int_scale'], self.__quan_dict['pre_int_scale'] # type: ignore
        txme_saturation = self.__quan_dict['txme_saturation'] # type: ignore
        out_type = self.__quan_dict['out_type'] # type: ignore
        default_setting = self.__quan_dict['default_setting'] # type: ignore
        
        output_names = [layer.get_onnx_output_name() for layer in self.graph.get_layers()]
        for layer in self.graph.get_layers():
            if layer.get_layer_type().lower() in ["conv", "depthwiseconv", "convtranspose", "fc"]:
                if layer.get_layer_ops()["ops"][-1] in self.__quan_dict["fuse_act"]: # type: ignore
                    conv_output_name = layer.get_nodes()[0].get_onnx_output()
                    if conv_output_name not in output_names:
                        output_names.append(conv_output_name)
        input_names = []
        # precision removed by qn
        # kwagrs = dict(graph=self.graph, act=act, default_setting=default_setting, bit_select=bit_select, preint_scale = preint_scale, out_type = out_type,
        #           bits_dict=bits_dict, maxs=maxs, mins=mins, precision=precision, int_scale=int_scale, txme_saturation = txme_saturation, output = self.quan_dict['output'])
        kwargs = dict(graph=self.graph, 
                      act=act, 
                      default_setting=default_setting, 
                      bit_select=bit_select, 
                      preint_scale = preint_scale, 
                      out_type = out_type,
                      bits_dict=bits_dict, 
                      maxs=maxs, 
                      mins=mins, 
                      int_scale=int_scale, 
                      txme_saturation=txme_saturation, 
                      fuse_act=self.__quan_dict["fuse_act"], # type: ignore
                      is_last_layer_fuse_act=self.__quan_dict["is_last_layer_fuse_act"], # type: ignore
                      output = self.__quan_dict['output']) # type: ignore
        
        kwargs.update(dict(log_name=self.log_name, log_level=self.log_level))
        kwargs.update({"virtual_round":self.virtual_round}) # type: ignore
        #self.quan_graph = GraphQuant(**kwagrs)
        
        # if self.offline_quan_mode:
        #     quan_graph = AlreadyGrapQuant(**kwagrs)
        for layer in self.graph.get_layers():
            layer_type = layer.get_layer_type().lower()
            if layer_type == 'data':
                input_names.extend(layer.get_onnx_output_name())
        #model_ = add_layer_output_to_graph(self.model_path, output_names, input_names)
        kwargs.update({"input_names":input_names}) # type: ignore
        quan_graph = GrapQuantUpgrade(**kwargs)
        return quan_graph, input_names, output_names

    def __model_convert(self, calib_input_list):
        if self.offline_quant_mode:
            self.quan_table_parser = QuanTableParser(self.offline_quan_tool)
            weight_scale_dict, top_scale_dict = self.quan_table_parser.parse_weight_and_top_scales(self.graph, self.offline_quan_table)
            # if top_scale_dict not None, use top_scale_dict , else use calib_input_list calculate ftm scales online
            self.quan_graph = self.__post_quan.map_quant_table(weight_scale_dict, top_scale_dict, calib_input_list)
        else:
            self.quan_graph = self.__post_quan.quan_dataset(calib_input_list)

    def __create_simulator(self):
        if self.do_check_error == 2:
            self.__simulation = ErrorAnalyzer(quan_graph=self.quan_graph, check_error = self.__check_error, simulation_level=0, error_metrics = self.error_metrics, log_name = self.log_name, log_level=self.log_level)
        else:
            self.__simulation = Simulation(quan_graph=self.quan_graph, check_error = self.__check_error, simulation_level=0, error_metrics = self.error_metrics, log_name = self.log_name, log_level=self.log_level)
        self.__simulation.set_graph(self.quan_graph)  

    def __model_export(self):
        self.__export_dict.update(dict( # type: ignore
            quan_graph=self.quan_graph, 
            log_name=self.log_name,
            log_level=self.log_level,
            is_assert = self.is_assert,
            is_stdout=self.is_stdout
            )
        )
        exporter = eval("mExport{}".format(self.__export_version))(**self.__export_dict)
        exporter.export()
        exporter.write_indata()
        exporter.write_weights()
        exporter.write_features()
        exporter.write_network()
        self.__logger.info('export done!')
        return exporter

    def __collect_layer_error(self, error_dict, layer_name):
        if layer_name not in self.__layer_average_error.keys():
            self.__layer_average_error[layer_name] = error_dict
        else:
            for metric, error in error_dict.items():
                self.__layer_average_error[layer_name][metric].extend(error)

    def __layer_error(self, layer, onnx_outputs):
        onnx_name = layer.get_onnx_output_name()
        qout, quantize = layer.get_out_data(), layer.get_quantize()

        log_infos = ['layer type is: {}, '.format(layer.get_layer_type())]
        qtrues, ftrues = list(), list()
        for idx in range(len(onnx_name)):
            if layer.get_is_result_layer():
                qtrues.append(onnx_outputs[onnx_name[idx]])
            else:
                qtrues.append(quantize['feat']['so' + str(idx)].get_quan_data(onnx_outputs[onnx_name[idx]]))
            ftrues.append(onnx_outputs[onnx_name[idx]])

        if isinstance(qout, dict):
            output = qout['output']
            if isinstance(output, list):
                for idx in range(len(qtrues)):
                    q_idx = idx - len(qtrues)
                    name = onnx_name[q_idx]
                    error_dict = dict()
                    for metric, check_error in self.__check_error.items(): # type: ignore
                        error = check_error(output[q_idx], qtrues[q_idx])
                        log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error, layer.get_layer_name())
                        log_infos.append(log_info)
                        error_dict[metric] = [error]
                        #assert np.abs(np.float64(qtrues[q_idx])).sum() > 0
                    self.__collect_layer_error(error_dict, layer.get_layer_name())
            else:
                error_dict = dict()
                for metric, check_error in self.__check_error.items(): # type: ignore
                    error = check_error(output, qtrues[-1])
                    log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(onnx_name, metric, error, layer.get_layer_name())
                    log_infos.append(log_info)
                    error_dict[metric] = [error]
                    #assert np.abs(np.float64(qtrues[-1])).sum() > 0
                self.__collect_layer_error(error_dict, layer.get_layer_name())
        else:
            for idx in range(len(qtrues)):
                q_idx = idx - len(qtrues)
                name = onnx_name[q_idx]
                error_dict = dict()
                for metric, check_error in self.__check_error.items(): # type: ignore
                    error = check_error(qout[q_idx]['output'], qtrues[q_idx])
                    log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error, layer.get_layer_name())
                    log_infos.append(log_info)
                    error_dict[metric] = [error]
                    #assert np.abs(np.float64(qtrues[q_idx])).sum() > 0
                self.__collect_layer_error(error_dict, layer.get_layer_name())
                
        return qout, ftrues, log_infos
    
    def reset_preprocess(self, preprocess):
        self.transform = preprocess
        self.__post_quan.set_transform(preprocess)
    
    def reset_postprocess(self, postprocess):
        self.postprocess = postprocess
        
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            self.__logger.error('Please check your model path and model type')
            os._exit(-1)
        self.model_path = model_path
        model = None
        try:
            _, model = self.__build_fp_graph(model_path)
        except:
            self.__logger.error('parse model failed, check your model type or contact author!')
            os._exit(-1)
          
        if model:
            self.kwargs['model_path'] = model
        else:
            os._exit(-1)
        try:
            self.quan_graph, self.input_names, self.output_names = self.__create_quan_graph()
            self.kwargs.update({
                'graph':self.quan_graph,
                'input_names':self.input_names,
                'out_names':self.output_names,
                'parse_cfg':self.__parse_cfg,
                'log_name':self.log_name,
                'log_level': self.log_level,
                'is_stdout' : self.is_stdout,
                'is_assert' : self.is_assert,
                'device': self.device,
                'sess_options': self.__sess_options,
            })
            
            self.kwargs.update(dict(transform=self.transform))        

            self.__post_quan = PostTrainingQuan(**self.kwargs)
        except:
            self.__logger.error('create timeintelli quantized graph structure failed!')
            os._exit(-1)
        try:
            # add onnx session
            self.__create_simulator()
        except:
            self.__logger.error('create timeintelli simulation structure failed!')
            os._exit(-1)

    def calibration(self, inputs):
        # calibration from datasets
        if isinstance(inputs, np.ndarray):
            if len(inputs.shape) == 3:
                self.quan_graph = self.__post_quan.quan_file(inputs)
            elif len(inputs.shape) == 4:
                self.quan_graph = self.__post_quan.quan_dataset(inputs)
            else:
                self.__logger.error("Calibration not Support input datasets!")
                os.exit(-1)
        else:
            if isinstance(inputs, list) or os.path.isdir(inputs):
                self.quan_graph = self.__post_quan.quan_dataset(inputs)
            # calibration from single file
            elif isinstance(inputs, str):
                self.quan_graph = self.__post_quan.quan_file(inputs)
            else:
                self.__logger.error("Calibration not Support input datasets!")
                os.exit(-1)
    
    def already_quant_model_weight_calibration(self, offline_quan_tool, offline_quan_table, inputs):
        assert isinstance(inputs, list), print("Single file quantion with already quantized\
            model Not support!")
        self.quan_table_parser = QuanTableParser(offline_quan_tool)
        weight_scale_dict, top_scale_dict = \
            self.quan_table_parser.parse_weight_and_top_scales(self.graph, offline_quan_table)
        # if top_scale_dict not None, use top_scale_dict , else use calib_input_list calculate ftm scales online
        self.quan_graph = self.__post_quan.map_quant_table(weight_scale_dict, top_scale_dict, inputs)
        
    def run_model_convert(self, calib_input_list):
        self.__exporter = self.__model_convert(calib_input_list)
        print("")
    
    def model_export(self, dummy_input=None):
        if isinstance(dummy_input, np.ndarray):
            self.model_simulation(dummy_input)
        self.__exporter = self.__model_export()
    
    def error_analysis(self):
        if self.do_check_error == 2:
            self.__simulation.annlyzer()
    
    def model_simulation(self, in_data, fp_result=False, isp_data=False):
        try:
            input_feed = copy.deepcopy(in_data)
            onnx_outputs = self.__post_quan.onnx_infer(input_feed) if self.do_check_error > 0 or fp_result else None

            input_feed = self.transform(input_feed) if self.transform else input_feed
            self.__quan_out = self.__simulation(input_feed, onnx_outputs=onnx_outputs, acc_error=self.acc_error, isp_data=isp_data)

            if self.do_check_error == 1:
                # print error log
                layers = self.__post_quan.get_graph().get_layers()
                if self.__check_error:
                    self.__logger.info('############ start error print one image ##############')
                    self.__logger.info('#######################################################')
                for l_idx, layer in enumerate(layers):
                    try:
                        # user define check error
                        if layer.is_extend():
                            error = layer.checkerror()
                            self.__logger.info('user define layer error is: {}'.format(error))
                            continue

                        if layer.get_layer_type() == "data":
                            self.__logger.info('skip calculation of data layer error')
                            continue

                        _, _, error_infos = self.__layer_error(layer, onnx_outputs)

                        for error_info in error_infos:
                            self.__logger.info(error_info)
                        self.__logger.info('#######################################################')

                        self.__logger.info('############ end error print one image ##############')
                    except:
                        self.__logger.error("layer of {} error failed!".format(layer.get_layer_name()))
                        os._exit(-1)

            # get onnx and quan results
            results = dict(result_converter=self.__quan_out['results']['qout'])
            if fp_result:
                result_onnx = self.__quan_out['results']['trueout']
                results.update(dict(result_onnx=result_onnx))
            return results
        except:
            self.__logger.error("timesIntelli quantized graph simulation failed!")
            os._exit(-1)
    
    def perf_analyze(self, perf_data_root = None, mem_addr = "psram"):
        if self.__exporter is None:
            self.__logger.error("Error, please run model export before you estimate model performance!")
            os._exit(-1)
        chip_model = None
        if '5050' in self.chip_model: # type: ignore
            chip_model='5050'
        elif '1k' in self.chip_model or '1K' in self.chip_model: # type: ignore
            chip_model = '1k'
        else:
            self.__logger.error("Error, the specified chip model is not supported by the performance analyse module!")
            os._exit(-1)
        try:
            perf_estimator = PerfAnalyzer(model_exporter=self.__exporter, chip_model=chip_model, \
                mem_addr=mem_addr, ref_data_dir = perf_data_root, encrypt_flag = True)
            time = perf_estimator()
            return time
        except:
            self.__logger.error("PerfAnalyzer not support!")
            os._exit(-1)

    def visualize_qparams(self, save_path=None):

        quan_graph = self.__simulation.get_graph()
        layers = quan_graph.get_layers() # type: ignore
        outputs = []
        for layer in layers:
            try:
                output = layer.export_onnx_fp(is_vis_qparams=True)
                outputs.append(output)
            except:
                self.__logger.error("layer of {} visualize_qparams failed!".format(layer.get_layer_name()))
                os._exit(-1)
        nodes, initializers = [], []
        for idx, out in enumerate(outputs):
            if None in out[0]:
                self.__logger.info(layers[idx].get_layer_name()) # type: ignore
            nodes.extend(out[0])
            initializers.extend(out[1])
        create_in_out = lambda name, dtype, shape: onnx.helper.make_tensor_value_info(name, dtype, shape) # type: ignore
        dtype_dict = {
            np.int8: onnx.TensorProto.INT8,
            np.int16: onnx.TensorProto.INT16,
            np.float32: onnx.TensorProto.FLOAT,
        }
        input_layers, result_layers = [], []
        for layer in layers:
            if layer.get_layer_type() == "data":
                input_layers.append(layer)
            if layer.get_is_result_layer():
                result_layers.append(layer)
        inputs = []
        for name, layer in zip(self.model_input_names, input_layers):
            out_type = layer.get_ops_setting()["setting"]["out_type"]
            bits_dict = layer.get_ops_setting()["setting"]["bits_dict"]
            out_dtype = bits_dict[out_type]
            input = create_in_out(name, dtype_dict[out_dtype], ['n', 'c', 'h', 'w'])
            inputs.append(input)
        outputs = []
        for name, layer in zip(self.model_output_names, result_layers):
            out_type = layer.get_ops_setting()["setting"]["out_type"]
            bits_dict = layer.get_ops_setting()["setting"]["bits_dict"]
            out_dtype = bits_dict[out_type]
            output = create_in_out(name, dtype_dict[out_dtype], ['n', 'c', 'h', 'w'])
            outputs.append(output)
        try:
            graph = onnx.helper.make_graph(nodes=nodes, name="vis_qparams", # type: ignore
                                        inputs=inputs,
                                        outputs=outputs,
                                        initializer=initializers)
            model = onnx.helper.make_model( # type: ignore
                graph, opset_imports=[onnx.helper.make_opsetid("", self.opset_version)]) # type: ignore

            if save_path is None:
                save_path = "work_dir"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_name = os.path.basename(self.model_path).replace(".onnx", "_vis_qparams.onnx")
            onnx.save_model(model, '{}/{}'.format(save_path, model_name))
        except:
            self.__logger.error("save timesintelli graph to onnx failed!")
            os._exit(-1)

def version():
    return 'TimesIntelli converter 2.0.4'
    