# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/4 13:57
# @File     : PostQuanSimulation.py
import os
import copy
import onnx
import cv2
import json
import numpy as np
import onnxruntime as rt
from tqdm import tqdm
from glob import glob
from abc import abstractmethod
from fitter import Fitter
import random
import torch
import torch.multiprocessing as mp
        
from .PostTrainingQuan import PostTrainingQuan
from .QuanTableParser import QuanTableParser


try:
    from utils import Object, flatten_list, save_txt
    from checkpoint import OnnxParser
    from config import Config
    from export import mExportV3, mExportV2, mExportV1
    from graph import Graph
    from quantizer import AlreadyGrapQuant, DataNotice, GraphQuant
    from quantizer import GrapQuantUpgrade, HistogramFeatureObserver
    from quantizer import HistogramWeightObserver
    from simulator import Simulation, ErrorAnalyzer, error_factory
    from utils import Object, flatten_list, print_chaojisaiya
    from utils import print_long, print_pikaqiu, print_safe, print_victory
    from tools.qat import train
except:
    from onnx_converter.utils import Object, flatten_list, save_txt # type: ignore
    from onnx_converter.checkpoint import OnnxParser # type: ignore
    from onnx_converter.config import Config # type: ignore
    from onnx_converter.export import mExportV3, mExportV2, mExportV1 # type: ignore
    from onnx_converter.graph import Graph # type: ignore
    from onnx_converter.quantizer import AlreadyGrapQuant, DataNotice # type: ignore
    from onnx_converter.quantizer import GraphQuant, GrapQuantUpgrade # type: ignore
    from onnx_converter.quantizer import HistogramFeatureObserver # type: ignore
    from onnx_converter.quantizer import HistogramWeightObserver # type: ignore
    from onnx_converter.simulator import Simulation, ErrorAnalyzer, error_factory # type: ignore
    from onnx_converter.utils import Object, flatten_list, print_chaojisaiya # type: ignore
    from onnx_converter.utils import print_long, print_pikaqiu, print_safe # type: ignore
    from onnx_converter.utils import print_victory # type: ignore
    from onnx_converter.tools.qat import train
    
    
class ParseKwargs(object):
    def __init__(self, **kwargs):
        self.log_name = kwargs['log_name']
        self.model_path = kwargs['model_path']
        self.is_simplify = kwargs['is_simplify']
        self.parse_cfg = kwargs['parse_cfg']
        self.graph_cfg = kwargs['graph_cfg']
        self.quan_cfg = kwargs['quan_cfg']
        self.simulation_level = kwargs['simulation_level']
        self.error_metric = kwargs['error_metric']

        self.offline_quan_mode = kwargs['offline_quan_mode']
        self.offline_quan_tool = kwargs['offline_quan_tool']
        self.quan_table_path = kwargs['quan_table_path']

        self.weight_scale_dict = None
        self.top_scale_dict = None


def add_layer_output_to_graph(model_path, input_names, out_names):
    if isinstance(model_path, str):
        model_ = onnx.load_model(model_path)
    else:
        model_ = copy.deepcopy(model_path)
    # if self.is_remove_transpose:
    #     model_ = remove_transpose(model_)

    # output_names = []
    output_names_ = flatten_list(out_names)
    input_names_ = flatten_list(input_names)
    for output in output_names_:
        if output in input_names_: continue
        model_.graph.output.extend([onnx.ValueInfoProto(name=output)]) # type: ignore
        # output_names.append(output)

    return model_


class ModelProcess(Object): # type: ignore
    def __init__(self, **kwargs):
        super(ModelProcess, self).__init__(**kwargs)

        self.is_stdout = kwargs['is_stdout']
        self.log_name = kwargs['log_name']
        self.log_level = kwargs.get('log_level', 20)
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)
        self.model_path = kwargs['model_path']
        self.parse_cfg = kwargs['parse_cfg']
        self.graph_cfg = kwargs['graph_cfg']
        self.preprocess = kwargs.get('preprocess', None)
        self.device = kwargs.get('device', 'cpu')
        if "CUDAExecutionProvider" not in rt.get_available_providers():
            self.device = 'cpu'
        if 'base_quan_cfg' in kwargs.keys():
            self.quan_cfg = [kwargs['base_quan_cfg'], kwargs['quan_cfg']]
        else:
            self.quan_cfg = kwargs['quan_cfg']

        self.fp_result = kwargs['fp_result'] if 'fp_result' in kwargs.keys() else False

        self.simulation_level = kwargs['simulation_level']
        self.error_analyzer = kwargs.get('error_analyzer', False)
        self.is_simplify = kwargs.get('is_simplify', False)
        self.error_metric = kwargs['error_metric']

        self.offline_quan_mode = kwargs['offline_quan_mode']
        self.offline_quan_tool = kwargs['offline_quan_tool']
        self.quan_table_path = kwargs['quan_table_path']

        self.weight_scale_dict = None
        self.top_scale_dict = None
        input_names = kwargs.get('input_names', None)

        self.graph, self.output_names, self.input_names, model, self.model_input_names, self.model_output_names, self.sess_options = \
            self.quan_weight(self.parse_cfg, self.graph_cfg, self.quan_cfg, input_names=input_names)

        if model:
            kwargs['model_path'] = model

        if self.offline_quan_mode:
            self.quan_table_parser = QuanTableParser(self.offline_quan_tool)
            self.weight_scale_dict, self.top_scale_dict = \
                self.quan_table_parser.parse_weight_and_top_scales(self.graph, self.quan_table_path)
        self.quan_graph = self.graph
        # print_safe(self.logger)
        self.logger.info(self.input_names)
        # print_chaojisaiya(self.logger)
        # print_victory(self.logger)
        # print_pikaqiu(self.logger)
        # self.logger.info('################################################################')
        self.logger.info(self.output_names)
        # print_pikaqiu(self.logger)
        # self.logger.info('################################################################')
        kwargs.update(
            dict(out_names=self.output_names,
                 input_names=self.input_names,
                 graph=self.graph,
                 sess_options=self.sess_options,
                 device=self.device))
        self.post_quan = PostTrainingQuan(**kwargs)
        kwargs.update(dict(quan_graph=self.graph))
        if self.error_analyzer:
            self.simulation = ErrorAnalyzer(log_name=kwargs['log_name'], simulation_level=self.simulation_level)
        else:
            self.simulation = Simulation(log_name=kwargs['log_name'], simulation_level=self.simulation_level)
        self.logger.info('post quan simulation init done!')
        self.opset_version = 15
        self.check_error = dict()
        for m in self.error_metric:
            self.check_error[m] = error_factory.get(m)() # type: ignore
        if 'transform' in kwargs.keys():
            setattr(self, 'transform', kwargs['transform'])
            self.logger.info('user define transform will using data tranform!')
        if 'postprocess' in kwargs.keys():
            setattr(self, 'postprocess', kwargs['postprocess'])
            self.logger.info('user define postprocess will using postprocess!')
        self.quan_out = None

        cfg_dict = {}
        export_cfg = kwargs['export_cfg']
        export_cfg_files = [
            os.path.join(os.path.split(export_cfg)[0], "export.py"),
            kwargs['export_cfg'],
        ]
        for export_cfg in export_cfg_files:
            save_cfg = Config.fromfile(export_cfg)
            cfg_dict_, _ = save_cfg._file2dict(export_cfg)
            cfg_dict_.update(dict(is_stdout=self.is_stdout)) # type: ignore
            cfg_dict.update(cfg_dict_)

        export_version = kwargs.get("export_version", cfg_dict['export_version'])
        export_fun = 'mExportV{}'.format(export_version)
        cfg_dict["log_level"] = self.log_level
        self.model_export = eval(export_fun)(**cfg_dict)
        self.onnx_graph = False #False if self.error_analyzer else True
        self.vis_qparams = False
        self.layer_average_error = dict()
        self.simu_sess, self.simu_sess_inputs = None, None
        self.simu_output_names, self.fonnx_sess = None, None
        self.expand_output_names = []

    def set_onnx_graph(self, onnx_graph):
        self.onnx_graph = onnx_graph

    def set_draw_hist(self, draw_hist):
        self.is_draw_hist = draw_hist

    def get_layer_types(self):
        return [layer.get_layer_type() for layer in self.post_quan.get_graph().get_layers()]

    def get_output_names(self):
        return self.output_names

    def get_input_names(self):
        return self.input_names

    def collect_layer_error(self, error_dict, layer_name):
        if layer_name not in self.layer_average_error.keys():
            self.layer_average_error[layer_name] = error_dict
        else:
            for metric, error in error_dict.items():
                self.layer_average_error[layer_name][metric].extend(error)

    def get_layer_average_error(self):
        average_error = {}
        for layer_name, error_dict in self.layer_average_error.items():
            errors = {metric: np.mean(error) for metric, error in error_dict.items()}
            average_error[layer_name] = errors

        return average_error

    def quan_weight(self, parse_cfg, graph_cfg, quan_cfg, **kwargs):
        parsed_cfg = Config.fromfile(parse_cfg)
        parse_dict, _ = parsed_cfg._file2dict(parse_cfg)
        input_names = kwargs['input_names'] if 'input_names' in kwargs.keys() else None
        parse_dict.update(dict(log_name=self.log_name, 
                               log_level=self.log_level, 
                               input_names=input_names, 
                               is_simplify=self.is_simplify)) # type: ignore
        if self.preprocess:
            parse_dict.update(dict(preprocess=self.preprocess)) # type: ignore
        parser = OnnxParser(**parse_dict) # type: ignore
        nodes, model, model_input_names, model_output_names, sess_options = parser.parse(self.model_path)
        self.opset_version = parser.get_opset_version()
        # parsed_cfg = Config.fromfile(graph_cfg)
        graph_dict, _ = parsed_cfg._file2dict(graph_cfg)
        
        # parsed_cfg = Config.fromfile(quan_cfg)
        if isinstance(quan_cfg, list):
            base_graph_quan_dict, base_graph_save = parsed_cfg._file2dict(quan_cfg[0])
            graph_quan_dict, graph_save = parsed_cfg._file2dict(quan_cfg[1])
            graph_quan_dict = parsed_cfg._merge_a_into_b(graph_quan_dict, base_graph_quan_dict)
        else:
            graph_quan_dict, graph_save = parsed_cfg._file2dict(quan_cfg)
        if graph_quan_dict["virtual_round"] == 3: # type: ignore
            graph_quan_dict["int_scale"] -= 1 # type: ignore

        args = dict(
            nodes=nodes, act=graph_quan_dict['act'], opset_version=self.opset_version, # type: ignore
            especial_ops=graph_dict['especial_ops'], fuse_ops=graph_dict['fuse_ops'], # type: ignore
            split_ops=graph_dict['split_ops'], replace_ops=graph_dict['replace_ops'], # type: ignore
            default_setting=graph_quan_dict['default_setting'], # type: ignore
            fuse_act=graph_quan_dict["fuse_act"], # type: ignore
            is_last_layer_fuse_act=graph_quan_dict["is_last_layer_fuse_act"], # type: ignore
        )
        args.update(dict(log_name=self.log_name, log_level=self.log_level))
        graph = Graph(**args)
        graph.build()
        # graph.split_layer(split_layer_name="Conv_0") #yolov5n_320_v1_simplify.onnx
        # graph.together_layer(together_layer_names=["Conv_0", "Mul_2"])
        input_names = []
        for layer in graph.get_layers():
            layer_type = layer.get_layer_type().lower()
            if layer_type == 'data':
                input_names.extend(layer.get_onnx_output_name())

        output_names = [layer.get_onnx_output_name() for layer in graph.get_layers()]
        for layer in graph.get_layers():
            if layer.get_layer_type().lower() in ["conv", "depthwiseconv", "convtranspose", "fc"]:
                if layer.get_layer_ops()["ops"][-1] in graph_quan_dict["fuse_act"]: # type: ignore
                    nodes = layer.get_nodes()
                    if len(nodes) >= 2 and nodes[1].get_op_type() == "BatchNormalization":
                        conv_output_name = nodes[1].get_onnx_output()
                    else:
                        conv_output_name = nodes[0].get_onnx_output()
                    if conv_output_name not in output_names:
                        output_names.append(conv_output_name)

        act, bits_dict = graph_quan_dict['act'], graph_quan_dict['bits_dict'] # type: ignore
        bit_select, maxs, mins = graph_quan_dict['bit_select'], graph_quan_dict['maxs'], graph_quan_dict['mins'] # type: ignore
        int_scale = graph_quan_dict['int_scale'] # type: ignore
        txme_saturation = graph_quan_dict['txme_saturation'] if 'txme_saturation' in graph_quan_dict.keys() else 1 # type: ignore
        default_setting = graph_quan_dict['default_setting'] # type: ignore
        virtual_round = graph_quan_dict['virtual_round'] # type: ignore
        output = graph_quan_dict['output'] # type: ignore
        kwagrs = dict(graph=graph, act=act, default_setting=default_setting, bit_select=bit_select, output=output,
                      input_names=input_names, bits_dict=bits_dict, maxs=maxs, mins=mins, int_scale=int_scale,
                      virtual_round=virtual_round, txme_saturation=txme_saturation,
                      fuse_act=graph_quan_dict["fuse_act"], # type: ignore
                      search_smaller_sk=graph_quan_dict["search_smaller_sk"],
                      reload_sk_params=graph_quan_dict["reload_sk_params"],
                      sk_params_json_path=graph_quan_dict["sk_params_json_path"],            
                    )
        kwagrs.update(dict(log_name=self.log_name, log_level=self.log_level))
        # quan_graph = GraphQuant(**kwagrs)
        if self.offline_quan_mode:
            quan_graph = AlreadyGrapQuant(**kwagrs)
        else:
            quan_graph = GrapQuantUpgrade(**kwagrs)

        return quan_graph, output_names, input_names, model, model_input_names, model_output_names, sess_options

    def create_quan_onnx(self):
        quan_graph = self.simulation.get_graph()
        outputs = [layer.export_onnx() for layer in quan_graph.get_layers()] # type: ignore
        nodes, initializers = [], []
        for idx, out in enumerate(outputs):
            if None in out[0]:
                print(quan_graph.get_layers()[idx].get_layer_name()) # type: ignore
            nodes.extend(out[0])
            initializers.extend(out[1])
        create_in_out = lambda name, shape: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) # type: ignore
        inputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in self.model_input_names]
        outputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in self.model_output_names]
        graph = onnx.helper.make_graph(nodes=nodes, name="speedup", # type: ignore
                                       inputs=inputs,
                                       outputs=outputs,
                                       initializer=initializers)
        model = onnx.helper.make_model( # type: ignore
            graph, opset_imports=[onnx.helper.make_opsetid("", self.opset_version)]) # type: ignore
        # onnx.checker.check_model(model)
        output_names = [item.name for item in model.graph.output]
        # onnx.save_model(model, '/home/shiqing/Downloads/onnx-converter/{}'.format(os.path.basename(self.model_path)))

        self.expand_output_names = [item+'_s' if item in self.input_names else item for item in flatten_list(self.output_names)]
        extra_outputs = []
        for node in model.graph.node:
            for name in node.output:
                if [name] not in self.expand_output_names:
                    extra_outputs.append(name)

        extra_outputs.extend(self.expand_output_names)
        # self.expand_output_names = extra_outputs
        model = add_layer_output_to_graph(model, self.input_names, self.expand_output_names)
        # onnx.checker.check_model(model)

        providers = ['CPUExecutionProvider']
        device = None
        if 'cuda' in self.device:
            if 'CUDAExecutionProvider' in rt.get_available_providers():
                providers = ['CUDAExecutionProvider']
                device = [{'device_id': int(self.device.split(":")[-1])}]

        # sess = rt.InferenceSession(model.SerializeToString(), providers=rt.get_available_providers(), provider_options=device) # type: ignore
        # sess = rt.InferenceSession(model.SerializeToString(), providers=rt.get_available_providers()) # type: ignore
        # sess = rt.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProvider']) # type: ignore
        sess = rt.InferenceSession(model.SerializeToString(), providers=providers, provider_options=device) # type: ignore
        sess_inputs = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in sess.get_inputs()}
        # output_names = [out.name for out in sess.get_outputs()]

        # fonnx_sess = rt.InferenceSession(self.model_path, providers=rt.get_available_providers(), provider_options=device) # type: ignore
        # fonnx_sess = rt.InferenceSession(self.model_path, providers=rt.get_available_providers()) # type: ignore
        #fonnx_sess = rt.InferenceSession(self.model_path, providers=['CUDAExecutionProvider']) # type: ignore
        fonnx_sess = rt.InferenceSession(self.model_path, providers=providers, provider_options=device) # type: ignore

        return sess, sess_inputs, output_names, fonnx_sess

    def visualize_qparams(self, save_path=None):
        quan_graph = self.simulation.get_graph()
        layers = quan_graph.get_layers() # type: ignore
        outputs = []
        for layer in layers:
            # print(layer.get_layer_name(), layer.get_layer_type())
            output = layer.export_onnx_fp(is_vis_qparams=True)
            outputs.append(output)
        nodes, initializers = [], []
        for idx, out in enumerate(outputs):
            if None in out[0]:
                print(layers[idx].get_layer_name()) # type: ignore
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
        for name, layer in zip(self.model_input_names, input_layers): # type: ignore
            out_type = layer.get_ops_setting()["setting"]["out_type"]
            bits_dict = layer.get_ops_setting()["setting"]["bits_dict"]
            out_dtype = bits_dict[out_type]
            input = create_in_out(name, dtype_dict[out_dtype], ['n', 'c', 'h', 'w'])
            inputs.append(input)
        outputs = []
        for name, layer in zip(self.model_output_names, result_layers): # type: ignore
            out_type = layer.get_ops_setting()["setting"]["out_type"]
            bits_dict = layer.get_ops_setting()["setting"]["bits_dict"]
            out_dtype = bits_dict[out_type]
            output = create_in_out(name, dtype_dict[out_dtype], ['n', 'c', 'h', 'w'])
            outputs.append(output)
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

    def reload_calibration(self, saved_calib_name="calibration_scales"):        
        try:
            self.quan_graph = self.post_quan.quan_dataset_with_alread_scales(None, None, saved_calib_name)
            self.simulation.set_graph(self.quan_graph)
            self.model_export.set_graph(self.quan_graph)
            self.simu_sess, self.simu_sess_inputs = None, None
            self.simu_output_names, self.fonnx_sess = None, None
            if 'lstm' in self.get_layer_types() or "splice" in self.get_layer_types() or "gru" in self.get_layer_types():
                self.onnx_graph = False
            if self.onnx_graph:
                self.simu_sess, self.simu_sess_inputs, self.simu_output_names, self.fonnx_sess = self.create_quan_onnx()
            if self.vis_qparams:
                self.visualize_qparams()
            return True
        except:
            return False
        
    
    def quantize(self, fd_path, is_dataset=False, prefix='jpg', saved_calib_name="calibration_scales", calibration_params_json_path=None):
        if not is_dataset and self.weight_scale_dict is None:
            self.quan_graph = self.post_quan.quan_file(fd_path)
            self.logger.info('quantize from single file!')
        elif self.weight_scale_dict is not None:  # offline quantize mode , add by Nan.Qin
            self.logger.info('quantize from offline generated table!')
            self.quan_graph = self.post_quan.map_quant_table(self.weight_scale_dict, self.top_scale_dict, fd_path, prefix, saved_calib_name)
        else:
            self.logger.info('quantize from datasets!')
            self.quan_graph = self.post_quan.quan_dataset(fd_path, prefix, saved_calib_name, calibration_params_json_path)

        self.simulation.set_graph(self.quan_graph)
        self.model_export.set_graph(self.quan_graph)
        self.simu_sess, self.simu_sess_inputs = None, None
        self.simu_output_names, self.fonnx_sess = None, None
        if 'lstm' in self.get_layer_types() or "splice" in self.get_layer_types() or "gru" in self.get_layer_types():
            self.onnx_graph = False
        if self.onnx_graph:
            self.simu_sess, self.simu_sess_inputs, self.simu_output_names, self.fonnx_sess = self.create_quan_onnx()
        if self.vis_qparams:
            self.visualize_qparams()

    # in data is normalized by user
    def dataflow(self, in_data, acc_error=False, onnx_outputs=None, image_name=None):
        # self.onnxgraph(in_data, image_name)
        # self.numpygraph(in_data, acc_error, onnx_outputs, image_name)
        if self.onnx_graph:
            return self.onnxgraph(in_data, acc_error, onnx_outputs, image_name)
        else:
            return self.numpygraph(in_data, acc_error, onnx_outputs, image_name)
        # return self.onnxgraph(in_data, image_name)
        # return self.numpygraph(in_data, acc_error, onnx_outputs, image_name)

    def onnxgraph(self, in_data, acc_error=False, onnx_outputs=None, image_name=None):
        if image_name:
            self.logger.info('#########################################')
            self.logger.info('model process: {}'.format(image_name))
            self.logger.info('#########################################')

        if hasattr(self, 'transform'):
            data = self.transform(in_data)
            # self.logger.info('simulation has input transform!')
        else:
            data = copy.deepcopy(in_data)
            self.logger.info('simulation does not has input transform!')

        # x_inputs = sess.get_inputs()
        if isinstance(data, dict):
            for key in data.keys():
                self.simu_sess_inputs[key] = data[key] # type: ignore
        else:
            self.simu_sess_inputs[list(self.simu_sess_inputs.keys())[0]] = data # type: ignore
        # if isinstance(data)
        qpred_onx = self.simu_sess.run(self.expand_output_names, self.simu_sess_inputs) # type: ignore
        # self.logger.info(qpred_onx[-1])
        fpred_onnx = []
        if self.fp_result:
            fpred_onnx = self.fonnx_sess.run(None, self.simu_sess_inputs) # type: ignore

        qoutputs, foutputs = {}, {}
        for idx, key in enumerate(self.simu_output_names): # type: ignore
            index = self.expand_output_names.index(key)
            qoutputs[key] = qpred_onx[index]
            if self.fp_result:
                foutputs[key] = fpred_onnx[idx]
        self.quan_out = {'qout': qoutputs, 'trueout': foutputs}

    def numpygraph(self, in_data, acc_error=False, onnx_outputs=None, image_name=None):
        if image_name:
            self.logger.info('#########################################')
            self.logger.info('model process: {}'.format(image_name))
            self.logger.info('#########################################')

        if hasattr(self, 'transform'):
            data = self.transform(in_data)
            # self.logger.info('simulation has input transform!')
        else:
            data = copy.deepcopy(in_data)
            self.logger.info('simulation does not has input transform!')

        qoutputs = self.simulation(data, acc_error=acc_error, onnx_outputs=onnx_outputs)
        # self.logger.info(qoutputs['results']['qout']['output'])
        self.quan_out = qoutputs['results']

    def get_all_outdatas(self):
        return [layer.get_out_data() for layer in self.post_quan.get_layers()]

    def get_layer(self, idx):
         return self.post_quan.get_layers()[idx]

    def get_all_onnx_output_name(self):
        return [layer.get_onnx_output_name() for layer in self.post_quan.get_layers()]

    def layer_error(self, layer, onnx_outputs):
        onnx_name = layer.get_onnx_output_name()
        qout, quantize = layer.get_out_data(), layer.get_quantize()
        # todo check every operation error
        # ops = layer.get_ops_instance()
        # if isinstance(ops, list):
        #     scales = [op.scales for op in ops]
        # else:
        #     scales = ops.scales
        # if layer.get_layer_name() == 'Gemm_5':
        #     print('test')
        log_infos = ['layer index: {}, layer type is: {}, '.format(layer.get_idx(), layer.get_layer_type())]
        log_infos.extend(['input index: {}, output index: {}, '.format(layer.get_input_idx(), layer.get_output_idx())])
        qtrues, ftrues = list(), list()
        if layer.get_layer_type().lower() in ['lstm']:
            qtrues.append(quantize['feat']['so0'].get_quan_data(onnx_outputs[onnx_name[0]]))
            qtrues.append(quantize['feat']['so1'].get_quan_data(onnx_outputs[onnx_name[1]]))
            qtrues.append(quantize['feat']['so2'].get_quan_data(onnx_outputs[onnx_name[2]]))
            for idx in range(len(onnx_name)):
                ftrues.append(onnx_outputs[onnx_name[idx]])
        else:
            for idx in range(len(onnx_name)):
                qtrues.append(quantize['feat']['so' + str(idx)].get_quan_data(onnx_outputs[onnx_name[idx]]))
                ftrues.append(onnx_outputs[onnx_name[idx]])

        def float2int(qout, quantize, q_idx):
            qout_ = copy.deepcopy(qout)
            if qout_.dtype.type.__name__ in ["float32", "float64"]:
                qout_ = quantize['feat']['so' + str(q_idx)].get_quan_data(qout_)
            return qout_

        if layer.get_layer_type().lower() in ['lstm', 'gru']:
            if isinstance(qout, list):
                for q_idx in range(len(qtrues)):
                    name = onnx_name[q_idx]
                    error_dict = dict()
                    for metric, check_error in self.check_error.items():
                        qout_ = qout[q_idx]["output"]
                        qout_ = float2int(qout_, quantize, q_idx)
                        error = check_error(qout_, qtrues[q_idx])
                        log_info = 'layer name is: {}, node of {} {} error is: {:.5f}'.format(layer.get_layer_name(),
                                                                                              name, metric, error)
                        log_infos.append(log_info)
                        error_dict[metric] = [error]
                        # assert np.abs(np.float32(qtrues[q_idx])).sum() > 0
                        if np.abs(np.float32(qtrues[q_idx])).sum() == 0:
                            self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                                layer.get_idx(), layer.get_layer_name()))
                    self.collect_layer_error(error_dict, layer.get_layer_name() + "_" + name)
        elif layer.get_is_result_layer() and layer.get_layer_type() in ['fc']:
            for idx in range(len(qtrues)):
                q_idx = idx - len(qtrues)
                name = onnx_name[q_idx]
                error_dict = dict()
                for metric, check_error in self.check_error.items():
                    if isinstance(qout[q_idx], dict):
                        qout_ = qout[q_idx]['output']
                    else:
                        qout_ = qout[q_idx]
                    qout_ = float2int(qout_, quantize, q_idx=idx)
                    error = check_error(qout_, qtrues[q_idx])
                    log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error,
                                                                                        layer.get_layer_name())
                    log_infos.append(log_info)
                    error_dict[metric] = [error]
                    # assert np.abs(np.float32(qtrues[q_idx])).sum() > 0
                    if np.abs(np.float32(qtrues[q_idx])).sum() == 0:
                        self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                            layer.get_idx(), layer.get_layer_name()))
                    self.collect_layer_error(error_dict, layer.get_layer_name())
        else:
            if isinstance(qout, dict):
                output = qout['output']
                if isinstance(output, list):
                    for idx in range(len(qtrues)):
                        q_idx = idx - len(qtrues)
                        name = onnx_name[q_idx]
                        error_dict = dict()
                        for metric, check_error in self.check_error.items():
                            qout_ = output[q_idx]
                            qout_ = float2int(qout_, quantize, q_idx)
                            error = check_error(qout_, qtrues[q_idx])
                            log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error,
                                                                                                layer.get_layer_name())
                            log_infos.append(log_info)
                            error_dict[metric] = [error]
                            # assert np.abs(np.float32(qtrues[q_idx])).sum() > 0
                            if np.abs(np.float32(qtrues[q_idx])).sum() == 0:
                                self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                                    layer.get_idx(), layer.get_layer_name()))
                        self.collect_layer_error(error_dict, layer.get_layer_name())
                else:
                    error_dict = dict()
                    for metric, check_error in self.check_error.items():
                        qout_ = output
                        qout_ = float2int(qout_, quantize, q_idx=0)
                        error = check_error(qout_, qtrues[-1])
                        log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(onnx_name, metric, error,
                                                                                            layer.get_layer_name())
                        log_infos.append(log_info)
                        error_dict[metric] = [error]
                        # assert np.abs(np.float32(qtrues[-1])).sum() > 0
                        if np.abs(np.float32(qtrues[-1])).sum() == 0:
                            self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                                layer.get_idx(), layer.get_layer_name()))
                    self.collect_layer_error(error_dict, layer.get_layer_name())
            else:
                for idx in range(len(qtrues)):
                    q_idx = idx - len(qtrues)
                    name = onnx_name[q_idx]
                    error_dict = dict()
                    for metric, check_error in self.check_error.items():
                        if isinstance(qout[q_idx], dict):
                            qout_ = qout[q_idx]['output']
                        else:
                            qout_ = qout[q_idx]
                        qout_ = float2int(qout_, quantize, q_idx=idx)
                        error = check_error(qout_, qtrues[q_idx])
                        log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error,
                                                                                            layer.get_layer_name())
                        log_infos.append(log_info)
                        error_dict[metric] = [error]
                        # assert np.abs(np.float32(qtrues[q_idx])).sum() > 0
                        if np.abs(np.float32(qtrues[q_idx])).sum() == 0:
                            self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                                layer.get_idx(), layer.get_layer_name()))
                    self.collect_layer_error(error_dict, layer.get_layer_name())

        return qout, ftrues, log_infos

    def layer_error_fp(self, layer, onnx_outputs):
        onnx_name = layer.get_onnx_output_name()
        qout, quantize = layer.get_out_data(), layer.get_quantize()
        # todo check every operation error
        # ops = layer.get_ops_instance()
        # if isinstance(ops, list):
        #     scales = [op.scales for op in ops]
        # else:
        #     scales = ops.scales
        # if layer.get_layer_name() == 'Gemm_5':
        #     print('test')
        log_infos = ['layer index: {}, layer type is: {}, '.format(layer.get_idx(), layer.get_layer_type())]
        log_infos.extend(['input index: {}, output index: {}, '.format(layer.get_input_idx(), layer.get_output_idx())])
        qtrues, ftrues = list(), list()
        if layer.get_layer_type().lower() in ['lstm']:
            qtrues.append(quantize['feat']['so0'].get_quan_data(onnx_outputs[onnx_name[0]]))
            qtrues.append(onnx_outputs[onnx_name[1]])
            qtrues.append(onnx_outputs[onnx_name[2]])
            for idx in range(len(onnx_name)):
                ftrues.append(onnx_outputs[onnx_name[idx]])
        else:
            for idx in range(len(onnx_name)):
                qtrues.append(quantize['feat']['so' + str(idx)].get_quan_data(onnx_outputs[onnx_name[idx]]))
                ftrues.append(onnx_outputs[onnx_name[idx]])

        if layer.get_layer_type().lower() in ['lstm', 'gru']:
            if isinstance(qout, list):
                for q_idx in range(len(ftrues)):
                    name = onnx_name[q_idx]
                    error_dict = dict()
                    for metric, check_error in self.check_error.items():
                        if qout[q_idx]["output"][0].dtype in [np.int8, np.int16, np.int32, np.int64]:
                            fout = quantize['feat']['so' + str(q_idx)].get_dequan_data(qout[q_idx]["output"])
                        else:
                            fout = qout[q_idx]["output"]
                        error = check_error(fout, ftrues[q_idx])

                        log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error,
                                                                                            layer.get_layer_name())
                        log_infos.append(log_info)
                        error_dict[metric] = [error]
                        # assert np.abs(np.float32(ftrues[q_idx])).sum() > 0
                        if np.abs(np.float32(ftrues[q_idx])).sum() == 0:
                            self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                                layer.get_idx(), layer.get_layer_name()))
                    self.collect_layer_error(error_dict, layer.get_layer_name() + "_" + name + "_fp")
        elif layer.get_is_result_layer() and layer.get_layer_type() in ['fc']:
            for idx in range(len(ftrues)):
                q_idx = idx - len(ftrues)
                name = onnx_name[q_idx]
                error_dict = dict()
                for metric, check_error in self.check_error.items():
                    if isinstance(qout[q_idx], dict):
                        qout_ = qout[q_idx]['output']
                    else:
                        qout_ = qout[q_idx]
                    if qout_[0].dtype in [np.int8, np.int16, np.int32, np.int64]:
                        fout = quantize['feat']['so' + str(0)].get_dequan_data(qout_)
                    else:
                        fout = qout_
                    error = check_error(fout, ftrues[q_idx])

                    log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error,
                                                                                        layer.get_layer_name())
                    log_infos.append(log_info)
                    error_dict[metric] = [error]
                    # assert np.abs(np.float32(ftrues[q_idx])).sum() > 0
                    if np.abs(np.float32(ftrues[q_idx])).sum() == 0:
                        self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                            layer.get_idx(), layer.get_layer_name()))
                    self.collect_layer_error(error_dict, layer.get_layer_name() + "_fp")
        else:
            if isinstance(qout, dict):
                output = qout['output']
                if isinstance(output, list):
                    for idx in range(len(ftrues)):
                        q_idx = idx - len(ftrues)
                        name = onnx_name[q_idx]
                        error_dict = dict()
                        for metric, check_error in self.check_error.items():
                            if output[q_idx][0].dtype in [np.int8, np.int16, np.int32, np.int64]:
                                fout = quantize['feat']['so' + str(q_idx)].get_dequan_data(output[q_idx])
                            else:
                                fout = output[q_idx]
                            error = check_error(fout, ftrues[q_idx])

                            log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error,
                                                                                                layer.get_layer_name())
                            log_infos.append(log_info)
                            error_dict[metric] = [error]
                            # assert np.abs(np.float32(ftrues[q_idx])).sum() > 0
                            if np.abs(np.float32(ftrues[q_idx])).sum() == 0:
                                self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                                    layer.get_idx(), layer.get_layer_name()))
                        self.collect_layer_error(error_dict, layer.get_layer_name() + "_fp")
                else:
                    error_dict = dict()
                    for metric, check_error in self.check_error.items():
                        if output[0].dtype in [np.int8, np.int16, np.int32, np.int64]:
                            fout = quantize['feat']['so' + str(0)].get_dequan_data(output)
                        else:
                            fout = output
                        error = check_error(fout, ftrues[-1])

                        log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(onnx_name, metric, error,
                                                                                            layer.get_layer_name())
                        log_infos.append(log_info)
                        error_dict[metric] = [error]
                        # assert np.abs(np.float32(ftrues[-1])).sum() > 0
                        if np.abs(np.float32(ftrues[-1])).sum() == 0:
                            self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                                layer.get_idx(), layer.get_layer_name()))
                    self.collect_layer_error(error_dict, layer.get_layer_name() + "_fp")
            else:
                for idx in range(len(ftrues)):
                    q_idx = idx - len(ftrues)
                    name = onnx_name[q_idx]
                    error_dict = dict()
                    for metric, check_error in self.check_error.items():
                        if isinstance(qout[q_idx], dict):
                            qout_ = qout[q_idx]['output']
                        else:
                            qout_ = qout[q_idx]
                        if qout_[0].dtype in [np.int8, np.int16, np.int32, np.int64]:
                            fout = quantize['feat']['so' + str(idx)].get_dequan_data(qout_)
                        else:
                            fout = qout_
                        error = check_error(fout, ftrues[q_idx])

                        log_info = 'node of {} {} error is: {:.5f}, layer name is: {}'.format(name, metric, error,
                                                                                            layer.get_layer_name())
                        log_infos.append(log_info)
                        error_dict[metric] = [error]
                        # assert np.abs(np.float32(ftrues[q_idx])).sum() > 0
                        if np.abs(np.float32(ftrues[q_idx])).sum() == 0:
                            self.logger.warning('layer index {}, layer {} float-output are all zero!!!'.format(
                                layer.get_idx(), layer.get_layer_name()))
                    self.collect_layer_error(error_dict, layer.get_layer_name() + "_fp")
 
        return qout, ftrues, log_infos

    def checkerror(self, in_data, acc_error=False, onnx_outputs=None, image_name=None):
        if onnx_outputs is None:
            onnx_outputs = self.post_quan.onnx_infer(in_data)
        self.dataflow(in_data, acc_error=acc_error, onnx_outputs=onnx_outputs, image_name=image_name)
        if self.onnx_graph:
            pass
        else:
            layers = self.post_quan.get_graph().get_layers()
            # self.logger.info('#########################################')
            for l_idx, layer in enumerate(layers):
                if layer.get_layer_type().lower() == "data":
                    continue
                # if layer.get_layer_type().lower() != "lstm":
                #     continue
                # root_fd = "/home/shiqing/Downloads/test_package/saturation/onnx-converter/work_dir/"
                                        
                if layer.get_first_conv():
                    continue
                        
                # user define check error
                if layer.is_extend():
                    error = layer.checkerror()
                    self.logger.info('user define layer error is: {}'.format(error))
                    continue
                error_infos = []
                info = "################################int8-int8-error################################"
                error_infos.append(info)
                qout, ftrues, error_infos0 = self.layer_error(layer, onnx_outputs)
                error_infos.extend(error_infos0)
                info = "################################float-float-error################################"
                error_infos.append(info)
                # save_txt(os.path.join(root_fd, "lstm_layer_error.txt"), "a+", error_infos)
                qout, ftrues, error_infos1 = self.layer_error_fp(layer, onnx_outputs)
                error_infos.extend(error_infos1)
                # self.logger.info("@@@@@@@@@@@@ gt min / max = %f %f"%(np.min(ftrues[0]), np.max(ftrues[0])))
                # if isinstance(qout, list):
                #    self.logger.info("@@@@@@@@@@@@ pred min / max = %f %f"%(np.min(qout[-1]['output']), np.max(qout[-1]['output'])))
                # else:
                #    self.logger.info("@@@@@@@@@@@@ pred min / max = %f %f"%(np.min(qout['output']), np.max(qout['output'])))
                for error_info in error_infos:
                    self.logger.info(error_info)
                self.logger.info('#########################################')
            # if self.is_analysis:
            #     self.analysis_feat.get_feat_error()
            # self.logger.info('check error done!')
            # print_safe(self.logger)

    def reset(self, layer_type):
        self.simulation.reset_layer(layer_type)

    # export quantized graph
    def save(self, fd_path="work_dir/resume"):
       return self.quan_graph.save(fd_path=fd_path)
    
    # load file name saved graph
    def load(self, fd_path="work_dir/resume"):
        load_success = self.quan_graph.reload(fd_path=fd_path)
        self.simulation.set_graph(self.quan_graph)
        self.model_export.set_graph(self.quan_graph)
        return load_success
    
    # def checkerror_weight(self, onnx_outputs=None):
    #     if self.__data and self.observer_weight:
    #         self.observer_weight.update(self.__data, onnx_outputs)

    # def checkerror_feature(self, onnx_outputs):
    #     if self.__data and self.observer_feature and onnx_outputs:
    #         self.observer_feature.update(self.__data, onnx_outputs)

    # detection
    def get_outputs(self):
        qout, true_out = self.quan_out['qout'], self.quan_out['trueout'] # type: ignore
        if hasattr(self, "transform"):
            trans = self.transform.get_trans()
            if hasattr(self, "postprocess"):
                qout = self.postprocess(qout, trans)  # (h, w)
                if true_out:
                    none_flag = False
                    for key in true_out.keys():
                        if true_out[key] is None:
                            none_flag = True
                            break
                    if not none_flag:
                        true_out = self.postprocess(true_out, trans)  # (h, w)

        return dict(qout=qout, true_out=true_out)

    def error_analysis(self):
        self.simulation.annlyzer()

    def export(self):
        self.model_export.set_root_fd(root_fd=None)
        self.model_export.export()
        self.model_export.visualization()
        self.model_export.write_indata()
        self.model_export.write_weights()
        self.model_export.write_features()
        self.model_export.write_network()

    def export_frame(self, root_fd=None):
        self.model_export.set_root_fd(root_fd)
        self.model_export.export()
        # self.model_export.visualization()
        self.model_export.write_indata()
        # self.model_export.write_weights()
        self.model_export.write_features()
        # self.model_export.write_network()

    def __call__(self, *args, **kwargs):
        pass


class WeightOptimization(ModelProcess):
    def __init__(self, **kwargs):
        self.is_fused_act = kwargs["is_fused_act"] if "is_fused_act" in kwargs.keys() else True
        super(WeightOptimization, self).__init__(**kwargs)
        self.onnx_graph = False
        self.layer_fp = dict()

    def quan_weight(self, parse_cfg, graph_cfg, quan_cfg, **kwargs):
        parsed_cfg = Config.fromfile(parse_cfg)
        parse_dict, _ = parsed_cfg._file2dict(parse_cfg)
        input_names = kwargs['input_names'] if 'input_names' in kwargs.keys() else None
        parse_dict.update(dict(log_name=self.log_name, log_level=self.log_level, input_names=input_names)) # type: ignore
        if self.preprocess:
            parse_dict.update(dict(preprocess=self.preprocess)) # type: ignore
        parser = OnnxParser(**parse_dict) # type: ignore
        nodes, model, model_input_names, model_output_names, sess_options = parser.parse(self.model_path)
        self.opset_version = parser.get_opset_version()
        # parsed_cfg = Config.fromfile(graph_cfg)
        graph_dict, _ = parsed_cfg._file2dict(graph_cfg)
        
        # parsed_cfg = Config.fromfile(quan_cfg)
        if isinstance(quan_cfg, list):
            base_graph_quan_dict, base_graph_save = parsed_cfg._file2dict(quan_cfg[0])
            graph_quan_dict, graph_save = parsed_cfg._file2dict(quan_cfg[1])
            graph_quan_dict = parsed_cfg._merge_a_into_b(graph_quan_dict, base_graph_quan_dict)
        else:
            graph_quan_dict, graph_save = parsed_cfg._file2dict(quan_cfg)
        if graph_quan_dict["virtual_round"] == 3: # type: ignore
            graph_quan_dict["int_scale"] -= 1 # type: ignore
                    
        args = dict(
            nodes=nodes, act=graph_quan_dict['act'], opset_version=self.opset_version, # type: ignore
            especial_ops=graph_dict['especial_ops'], fuse_ops=graph_dict['fuse_ops'], # type: ignore
            split_ops=graph_dict['split_ops'], replace_ops=graph_dict['replace_ops'], # type: ignore
            default_setting=graph_quan_dict['default_setting'], # type: ignore
            fuse_act=graph_quan_dict["fuse_act"], # type: ignore
            is_last_layer_fuse_act=graph_quan_dict["is_last_layer_fuse_act"], # type: ignore
        )
        args.update(dict(log_name=self.log_name, log_level=self.log_level))
        graph = Graph(**args)
        graph.build()
        # graph.split_layer(split_layer_name="Conv_0") #yolov5n_320_v1_simplify.onnx
        # graph.together_layer(together_layer_names=["Conv_0", "Mul_2"])         
        input_names = []
        for layer in graph.get_layers():
            layer_type = layer.get_layer_type().lower()
            if layer_type == 'data':
                input_names.extend(layer.get_onnx_output_name())

        output_names = [layer.get_onnx_output_name() for layer in graph.get_layers()]
        for layer in graph.get_layers():
            if layer.get_layer_type().lower() in ["conv", "depthwiseconv", "convtranspose", "fc"]:
                if layer.get_layer_ops()["ops"][-1] in graph_quan_dict["fuse_act"]: # type: ignore
                    conv_output_name = layer.get_nodes()[0].get_onnx_output()
                    if conv_output_name not in output_names:
                        output_names.append(conv_output_name)
                        
        act, bits_dict = graph_quan_dict['act'], graph_quan_dict['bits_dict'] # type: ignore
        bit_select, maxs, mins = graph_quan_dict['bit_select'], graph_quan_dict['maxs'], graph_quan_dict['mins'] # type: ignore
        int_scale = graph_quan_dict['int_scale'] # type: ignore
        txme_saturation = graph_quan_dict['txme_saturation'] if 'txme_saturation' in graph_quan_dict.keys() else 1 # type: ignore
        default_setting = graph_quan_dict['default_setting'] # type: ignore
        virtual_round = graph_quan_dict['virtual_round'] # type: ignore
        output = graph_quan_dict['output'] # type: ignore
        kwagrs = dict(graph=graph, act=act, default_setting=default_setting, bit_select=bit_select, output=output,
                      input_names=input_names, bits_dict=bits_dict, maxs=maxs, mins=mins, int_scale=int_scale,
                      virtual_round=virtual_round, txme_saturation=txme_saturation,
                      fuse_act=graph_quan_dict["fuse_act"], # type: ignore
                    )
        kwagrs.update(dict(log_name=self.log_name, log_level=self.log_level))
        # quan_graph = GraphQuant(**kwagrs)
        if self.offline_quan_mode:
            quan_graph = AlreadyGrapQuant(**kwagrs)
        else:
            quan_graph = GrapQuantUpgrade(**kwagrs)
                   
        return quan_graph, output_names, input_names, model, model_input_names, model_output_names, sess_options

    def save_onnx_fp(self, quan_graph, wo_method=""):
        outputs = [layer.export_onnx_fp() for layer in quan_graph.get_layers()]
        nodes, initializers = [], []
        for idx, out in enumerate(outputs):
            if None in out[0]:
                print(quan_graph.get_layers()[idx].get_layer_name())
            nodes.extend(out[0])
            initializers.extend(out[1])
        create_in_node = lambda name, shape: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) # type: ignore
        create_out_node = lambda name: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, []) # type: ignore
        fisrt_layer = quan_graph.get_layers()[0]
        inputs = [create_in_node(name, fisrt_layer.get_layer_ops()["attrs"][0]['shape']) for name in self.model_input_names]
        outputs = [create_out_node(name) for name in self.model_output_names]
        graph = onnx.helper.make_graph(nodes=nodes, name="speedup", # type: ignore
                                       inputs=inputs,
                                       outputs=outputs,
                                       initializer=initializers)
        model = onnx.helper.make_model( # type: ignore
            graph, opset_imports=[onnx.helper.make_opsetid("", self.opset_version)]) # type: ignore
        model.ir_version = 8
        # onnx.checker.check_model(model)
        export_model_path = os.path.basename(self.model_path).replace(".onnx", f"_{wo_method}.onnx")
        onnx.save_model(model, 'work_dir/{}'.format(export_model_path))
        # print("test")

    def layer_error_fp(self, layer, onnx_outputs):
        onnx_name = layer.get_onnx_output_name()
        qout, quantize = layer.get_out_data(), layer.get_quantize()
        # todo check every operation error
        # ops = layer.get_ops_instance()
        # if isinstance(ops, list):
        #     scales = [op.scales for op in ops]
        # else:
        #     scales = ops.scales
        # if layer.get_layer_name() == 'Gemm_ext_1':
        #     print('test')
        log_infos = ['layer index: {}, layer type is: {}, '.format(layer.get_idx(), layer.get_layer_type())]
        log_infos.extend(['input index: {}, output index: {}, '.format(layer.get_input_idx(), layer.get_output_idx())])
        qtrues, ftrues = list(), list()
        if layer.get_layer_type().lower() in ['lstm']:
            qtrues.append(quantize['feat']['so0'].get_quan_data(onnx_outputs[onnx_name[0]]))
            qtrues.append(onnx_outputs[onnx_name[1]])
            qtrues.append(onnx_outputs[onnx_name[2]])
            for idx in range(len(onnx_name)):
                ftrues.append(onnx_outputs[onnx_name[idx]])
        else:
            for idx in range(len(onnx_name)):
                qtrues.append(quantize['feat']['so' + str(idx)].get_quan_data(onnx_outputs[onnx_name[idx]]))
                ftrues.append(onnx_outputs[onnx_name[idx]])

        if layer.get_layer_type() in ["gemm", "fc", "conv", "depthwiseconv"]:
            qout_ = copy.deepcopy(qout[-1]['output'])
            if qout_[0].dtype not in [np.float32, np.float64]:
                qout_ = quantize['feat']['so' + str(0)].get_dequan_data(qout_)
            if layer.get_layer_name() in self.layer_fp.keys():
                self.layer_fp[layer.get_layer_name()][0] += qout_.astype(
                    np.float32)
                self.layer_fp[layer.get_layer_name()][1] += ftrues[-1].astype(
                    np.float32)
                self.layer_fp[layer.get_layer_name()][2] += 1
            else:
                self.layer_fp[layer.get_layer_name()] = [
                    qout_.astype(np.float32), ftrues[-1].astype(np.float32), 1
                ]

        return qout, ftrues, log_infos

    def get_dbias(self, in_data, acc_error=False):
        onnx_outputs = self.post_quan.onnx_infer(in_data)
        self.dataflow(in_data, acc_error=acc_error, onnx_outputs=onnx_outputs)
        if self.onnx_graph:
            pass
        else:
            layers = self.post_quan.get_graph().get_layers()
            self.logger.info('#########################################')
            for l_idx, layer in enumerate(layers):
                # user define check error
                if layer.is_extend():
                    error = layer.checkerror()
                    self.logger.info('user define layer error is: {}'.format(error))
                    continue
                qout, ftrues, error_infos = self.layer_error_fp(layer, onnx_outputs)

                for error_info in error_infos:
                    self.logger.info(error_info)
                self.logger.info('#########################################')

            self.logger.info('check error done!')

    def save_bias(self, wo_method=""):
        quan_graph = self.post_quan.get_graph()

        fp = self.layer_fp
        for layer in quan_graph.get_layers():
            if layer.get_layer_type().lower() in ["gemm", "fc", "conv", "convtranpose", "depthwiseconv"]:
                weight = copy.deepcopy(layer.get_layer_ops()['weights'][0])
                bias = copy.deepcopy(layer.get_layer_ops()['weights'][1])
                if not layer.get_first_conv(): #and not layer.get_is_result_layer():
                    dbias = (fp[layer.get_layer_name()][0] - fp[layer.get_layer_name()][1]) / fp[layer.get_layer_name()][2]
                    dbias = np.squeeze(dbias)
                    if len(dbias.shape) > 1:
                        if layer.get_layer_type().lower() in ["fc"]:
                            if layer.get_ops_setting()["attrs"][0]["transB"]:
                                dbias = np.mean(dbias, axis=0)
                            else:
                                dbias = np.mean(dbias, axis=-1)
                        else:
                            dbias = np.mean(np.mean(dbias, axis=-1), axis=-1)                    
                    layer.set_layer_ops(dict(weights=[weight, bias - dbias]))
                else:
                    layer.set_layer_ops(dict(weights=[weight, bias]))

        self.save_onnx_fp(quan_graph, wo_method=wo_method)

    def bias_correction(self, datasets):
        self.quantize(datasets, is_dataset=True)
        if isinstance(datasets, str) and os.path.isdir(datasets):
            def isimage(fn):
                return os.path.splitext(fn)[-1] in (
                    '.jpg', '.JPG', '.png', '.PNG')            
            datasets = sorted(glob(os.path.join(datasets, "*")))
            datasets = [image_path for image_path in datasets if isimage(image_path)]
        elif isinstance(datasets, np.ndarray):
            datasets = [datasets[i] for i in range(datasets.shape[0])]

        for in_data in tqdm(datasets[:], postfix='bias_correction'):
            if not isinstance(in_data, np.ndarray):
                in_data = cv2.imread(in_data)
            self.get_dbias(in_data, acc_error=False)
        self.save_bias(wo_method="bias_correction")

    def cross_layer_equalization(self, skip_layer_names=[]):
        quan_graph = self.post_quan.get_graph()
        layers = quan_graph.get_layers()
        for layer in tqdm(layers, postfix='cross_layer_equalization'):
            if layer.get_layer_type() in ["fc", "gemm", "conv", "depthwiseconv", "concat"]:
                from quantizer import cross_layer_equalization
                cross_layer_equalization(layer, layers, skip_layer_names=skip_layer_names)
            elif layer.get_layer_type() == "lstm":
                pass

        self.save_onnx_fp(quan_graph, wo_method="cross_layer_equalization")


    def qat(self, config_file):
        config_dict = json.load(open(config_file, "r"))
        
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
                
        def get_datasets(datasets):
            if isinstance(datasets, str) and os.path.isdir(datasets):
                def isimage(fn):
                    return os.path.splitext(fn)[-1] in (
                        '.jpg', '.JPG', '.png', '.PNG', '.JPEG')            
                datasets = sorted(glob(os.path.join(datasets, "*")))
                datasets = [image_path for image_path in datasets if isimage(image_path)]
            elif isinstance(datasets, np.ndarray):
                datasets = [datasets[i] for i in range(datasets.shape[0])]
            return datasets
                     
        datasets = config_dict["calibration_dataset_path"]   
        self.quantize(datasets, is_dataset=True, prefix=config_dict["image_prefix"], calibration_params_json_path=None)
        datasets_eval = get_datasets(datasets)
        datasets = get_datasets(config_dict["train_dataset_path"])
                        
        if not os.path.exists("work_dir/layers"):
            os.makedirs("work_dir/layers")
                        
        quan_graph = self.post_quan.get_graph()
        layers = quan_graph.get_layers()
        layers_num = len(layers)
        # for layer_idx, layer in enumerate(layers):
        #     layer_idx = str(layer_idx).zfill(4)
        #     if layer.get_layer_type() in ["resize"]:
        #         layer.set_ops_instance([])
        #     # elif layer.get_layer_type() in ["conv"]:
        #     #     print("test") 
                            
        #     with open(f"work_dir/layers/{layer_idx}.pkl", "wb+") as f:
        #         pickle.dump(layer, f)
        
        logger = self.get_log(log_name=config_dict["log_name"], log_level=self.log_level, stdout=self.is_stdout)     
        qat_kwargs = dict(
                logger=logger,
                preprocess=self.transform if hasattr(self, "transform") else None,
                postprocess=self.postprocess if hasattr(self, "postprocess") else None,
                post_quan=self.post_quan,
                layers=layers,
                layers_num=layers_num,
                model_input_names=list(self.model_input_names), 
                model_output_names=list(self.model_output_names), 
                export_model_path="{}/{}".format(config_dict["work_dir"], os.path.basename(self.model_path)),         
                tensorboard_dir="{}/tensorboard".format(config_dict["work_dir"]),
                sk_params_file=config_dict.get("sk_params_file", None),
                calibration_params_file=config_dict.get("calibration_params_file", None),
                max_learning_rate=config_dict["max_learning_rate"],
                min_learning_rate=config_dict["min_learning_rate"],
                weight_decay=config_dict["weight_decay"],
                batch_size=config_dict["batch_size"],
                epochs=config_dict["epochs"],
                print_log_per_epoch=config_dict["print_log_per_epoch"],
                save_onnx_interval=config_dict["save_onnx_interval"],
                update_so_by_ema=config_dict["update_so_by_ema"],
                keyword_params=config_dict["keyword_params"],
        )
        rank = config_dict["rank"]
        world_size = config_dict["world_size"]
        # mp.spawn(
        #     train, 
        #     args=(world_size, datasets, qat_kwargs),
        #     nprocs=world_size,
        # )
        train(rank, world_size, datasets, datasets_eval, qat_kwargs)    
            

class OnnxModelProcess(ModelProcess):
    def __init__(self, **kwargs):
        super(OnnxModelProcess, self).__init__(**kwargs)

    def dataflow(self, in_data, acc_error=False, onnx_outputs=None, image_name=None):
        pass

    def layer_error(self, error_dict, layer_name):
        pass

    def checkerror(self, in_data, acc_error=False, image_name=None):
        pass


if __name__ == '__main__':
    pass
