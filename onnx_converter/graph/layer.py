# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/9/30 15:50
# @File     : layer.py

# import os
import copy
import os
import onnx
import numpy as np
from abc import abstractmethod

import torch


try:
    from quantizer import DefaultQuant as default_quant
    from quantizer import QUANTIZE as quantize_factory
    from export import lExport
    from utils import Registry, flatten_list, clip
    from simulator import OPERATORS as operators_factory
except:
    from onnx_converter.quantizer import DefaultQuant as default_quant
    from onnx_converter.quantizer import QUANTIZE as quantize_factory
    from onnx_converter.export import lExport
    from onnx_converter.utils import Registry, flatten_list, clip
    from onnx_converter.simulator import OPERATORS as operators_factory

LAYER: Registry = Registry(name='layer', scope='')


def create_initializer(data, name, dtype=np.float32):
    if not isinstance(data, np.ndarray):
        data = np.array(data).astype(dtype)
    if dtype == np.float32:
        tense_dtype = onnx.TensorProto.FLOAT
    elif dtype == np.int8:
        tense_dtype = onnx.TensorProto.INT8
    elif dtype == np.int16:
        tense_dtype = onnx.TensorProto.INT16
    elif dtype == np.int32:
        tense_dtype = onnx.TensorProto.INT32
    elif dtype == np.int64:
        tense_dtype = onnx.TensorProto.INT64
    else:
        raise NotImplementedError(f"{dtype} not supported for {name}")
    tensor = onnx.helper.make_tensor(
        name=name, data_type=tense_dtype,
        dims=data.shape, vals=data.tobytes(), raw=True)
    return tensor


# export parameter from layer-data
# after simulation, setting layer-data
class LayerData(object):
    def __init__(self):
        super(LayerData, self).__init__()
        self.__in_data, self.out_data, self.weight, self.bias = [], [], [], []
        self.qout_data, self.__qweight = [], []
        self.__table = []
        self.__qbias = []
        self.__bias = []
        self.__in_scale = [] #[default_quant]
        # datasets max feature map value
        # data scale is list when quantize for per channel
        self.__data_scale = 1.0
        # quantize data scale
        self.__shift, self.__scale = 0, 1.0
        # quantize weight get parameter
        self.__w_shift, self.__w_scale = 0, dict(scale=1.0, zero_point=0)
        # just written in model.c
        self.__out_shift, self.__out_scale = 0, 0
        self.__scale = [] #[default_quant]
        self.__scales = [{'out_shift': 0, 
                          'out_scale': 0.0, 
                          'extra_value': 0, 
                          'zi': 0, 
                          'zk': 0, 
                          'zo': 0}]

    def set_in_scale(self, scale):
        self.__in_scale = scale

    def get_in_scale(self):
        return self.__in_scale

    def get_scales(self):
        return self.__scales

    def set_scales(self, scales):
        self.__scales = scales

    def get_data_scale(self):
        return self.__data_scale

    # if per-channel, data scale is ndarray
    def set_data_scale(self, data_scale):
        self.__data_scale = data_scale

    def get_scale(self):
        return self.__scale

    def set_scale(self, scale):
        self.__scale = scale

    def get_shift(self):
        return self.__shift

    def set_shift(self, shift):
        self.__shift = shift

    def get_w_shift(self):
        return self.__w_shift

    def set_w_shift(self, w_shift):
        self.__w_shift = w_shift

    def get_w_scale(self):
        return self.__w_scale

    def set_w_scale(self, w_scale):
        self.__w_scale = w_scale

    def get_out_shift(self):
        return self.__out_shift

    def set_out_shift(self, out_shift):
        self.__out_shift = out_shift

    def get_out_scale(self):
        return self.__out_scale

    def set_out_scale(self, out_scale):
        self.__out_scale = out_scale

    def set_in_data(self, in_data):
        self.__in_data = in_data

    def get_in_data(self):
        return self.__in_data

    def set_out_data(self, out_data):
        self.out_data = out_data

    def get_out_data(self):
        return self.out_data

    def set_weight(self, weight):
        self.weight = weight

    def get_weight(self):
        return self.weight

    def set_table(self, table):
        self.__table = table

    def get_table(self):
        if isinstance(self.get_ops_instance(), list):
            return self.get_ops_instance()[-1].table[:, 0]
        else:
            return self.get_ops_instance().table[:, 0]
        # return self.__table

    def get_qout_data(self):
        return self.qout_data

    def get_qweight(self):
        return self.__qweight

    def set_qweights(self, weights):
        self.__qweight = weights

    def get_bias(self):
        return self.bias

    def set_bias(self, bias):
        self.bias = bias
        
    def get_qbias(self):
        return self.__qbias

    def set_qbias(self, bias):
        self.__qbias = bias
        
    def get_bias(self):
        return self.__bias

    def set_bias(self, bias):
        self.__bias = bias


class LayerInfo(object):
    def __init__(self, **kwargs):
        super(LayerInfo, self).__init__()
        self.layer_idx, self.input_idx, self.output_idx = list(), list(), list()
        self.input_map = list()
        self.input_name, self.output_name = None, None
        self.input_type, self.output_type = 0, 1
        self.layer_type = 'Conv'
        self.__input_names = []
        self.__is_result_layer = False
        self.is_first_conv = None

    def set_first_conv(self, flag):
        self.is_first_conv = flag

    def get_first_conv(self):
        return self.is_first_conv

    def set_input_map(self, input_map):
        self.input_map = input_map

    def get_input_map(self):
        return self.input_map

    def set_inputs_names(self, input_names):
        self.__input_names = input_names

    def get_inputs_names(self):
        return self.__input_names

    def set_result_layer(self, is_result):
        self.__is_result_layer = is_result

    def get_is_result_layer(self):
        return self.__is_result_layer

    def get_input_name(self):
        return self.get_nodes()[0].get_input()

    def get_output_name(self):
        return self.get_nodes()[-1].get_output()

    def get_onnx_input_name(self):
        return self.get_nodes()[0].get_onnx_input()

    def get_onnx_output_name(self):
        return self.get_nodes()[-1].get_onnx_output()

    def get_idx(self):
        return self.layer_idx

    def set_idx(self, idx):
        self.layer_idx = idx

    def clear_input_idx(self):
        self.input_idx = []

    def set_input_idx(self, idx):
        if isinstance(idx, list):
            self.input_idx.extend(idx)
        else:
            self.input_idx.append(idx)

    def get_input_idx(self):
        return self.input_idx

    def clear_output_idx(self):
        self.output_idx = []

    def set_output_idx(self, idx):
        if isinstance(idx, list):
            self.output_idx.extend(idx)
        else:
            self.output_idx.append(idx)

    def get_output_idx(self):
        return self.output_idx

    def get_input_type(self):
        return self.input_type

    def set_input_type(self, type):
        self.input_type = type

    def get_output_type(self):
        return self.output_type

    def set_output_type(self, type):
        self.output_type = type

    def get_layer_type(self):
        return self.layer_type

    def set_layer_type(self, type):
        self.layer_type = type


class Layer(LayerInfo, LayerData, lExport):
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        # self.ops save ops type, self.ops_instance save instance of class
        # attribute/weights/shape saved in ops
        # nodes/ops_nodes save original/processed nodes, process including fused node
        # ops_instances saved instance of ops_nodes with defined in registered
        self.__ops, self.__ops_nodes, self.__ops_instances = dict(), [], []
        self.__ops_setting = dict()
        self.__extend = False
        # self.ops_setting = dict()
        self.__name = None
        self.isolated = False
        self.__quantizes = dict(feat=dict())  # input weights output
        
        if 'quantize' in kwargs.keys():
            self.__quantizes.update(quantize=kwargs['quantize'])

        self.__in_quantize = []#[default_quant]
        self.__nodes = []
        self.__scale_type = 'smooth'

        self.eps = np.float32((2 + 1e-3) / 2)
        # self.eps = np.float32(1)

        self.__adaround = []
        self.__ema = []
        
    def set_adaround(self, adaround):
        self.__adaround = adaround
        
    def get_adaround(self):
        return self.__adaround
    
    def set_ema(self, ema):
        self.__ema = ema
        
    def get_ema(self):
        return self.__ema
    
    def init_quantizes(self):
        # self.__in_quantize = []
        in_scale, scale = [], []
        in_quantizes, quantizes = [], self.get_quantize()
        for _ in range(len(self.get_onnx_input_name())):
            in_quantizes.append(quantize_factory.get("base")())
            in_scale.append(default_quant)
        for idx in range(len(self.get_onnx_output_name())):
            new_key = "so"+str(idx)
            quantizes["feat"][new_key] = quantize_factory.get("base")()
            scale.append(default_quant)
        
        if self.get_layer_type() in ["data"]:
            in_scale.append(default_quant)
        self.set_scale(scale)
        self.set_in_scale(in_scale)
        self.set_in_quantize(in_quantizes)
        self.set_quantize(quantizes)
    
    def set_nodes(self, nodes):
        self.__nodes = nodes

    def get_nodes(self):
        return self.__nodes

    def init_scales(self):
        ops = self.get_ops_instance()
        scales = []
        if isinstance(ops, list):
            scales = [op.get_scales() for op in self.get_ops_instance()]
        elif isinstance(ops.get_scales(), list):
            scales = ops.get_scales()
        else:
            scales = [ops.get_scales()]
        self.set_scales(scales)

    def set_layer_name(self, name):
        self.__name = name

    def get_layer_name(self):
        return self.__name

    # layer is extend or not for user define
    def is_extend(self):
        return self.__extend

    def get_ops_setting(self):
        return self.__ops_setting

    def set_ops_setting(self, setting):
        self.__ops_setting = setting

    def set_scale_type(self, scale_type):
        self.__scale_type = scale_type

    def get_scale_type(self):
        return self.__scale_type

    def set_in_quantize(self, in_quantize):
        self.__in_quantize = in_quantize

    def get_in_quantize(self):
        return self.__in_quantize

    def get_ops_nodes(self):
        return self.__ops_nodes

    def set_ops_nodes(self, nodes):
        self.__ops_nodes = nodes

    def get_outout_nodes(self):
        return self.__nodes[-1]

    def get_input_nodes(self):
        return self.__nodes[0]

    def set_layer_ops(self, ops):
        self.__ops.update(ops)
    
    def clear_layer_ops(self):
        self.__ops = dict()
    
    def get_layer_ops(self):
        return self.__ops

    # push node in layer stack
    def extend(self, node):
        if isinstance(node, list) or isinstance(node, tuple):
            self.__nodes.extend(node)
        else:
            self.__nodes.append(node)

    # update quan method for we need
    # setting default quantize method
    def update_quan_method(self, method: dict):
        self.__quantizes.update(method)

    # update ops postprocess
    def update_output_type(self, dtype):
        self.set_output_type(dtype)

    # mix up accuracy
    def update_input_type(self, dtype):
        self.set_input_type(dtype)

    # setting for each operator, including attribute of pre-post-process for simulation
    # include attribute/[si,sk,so]
    # select float/int scale or without correct post-process:[normal, advanced, smooth]
    # normal: [si*sk/so, sk=1], advanded: [si*sk/so, weights is not none], smooth: [so=si, sk=1]
    def setting_ops(self, setting: dict):
        instances = self.get_ops_instance()
        self.set_ops_setting(setting)
        del instances

    # get instance using string name of operations
    # init shape and si, sk, so
    def instance_layer_ops(self):
        pass
        # self.ops_instances = [self.ops_instances.append(operators_factory.get(op)(self.kwargs)) for op in self.ops_nodes]

    def get_ops_instance(self):
        return self.__ops_instances

    def set_ops_instance(self, instances):
        self.__ops_instances = instances
        self.init_scales()

    def set_quantize(self, data: dict):
        self.__quantizes.update(data)
        
    def clear_quantize(self):
        self.__quantizes = dict()

    def get_quantize(self):
        return self.__quantizes

    def float_scale(self, in_datas):
        out_datas = list()
        if isinstance(in_datas, np.ndarray):
            assert isinstance(self.__in_quantize,
                              object)  # , print('in data is array, input quantize must be single quantize!')
            out_datas = self.__in_quantize.get_dequan_data(in_datas)
        elif isinstance(in_datas, list):
            assert len(in_datas) == len(self.__in_quantize)  # , print('must be has tne same length')
            for idx, _ in range(len(in_datas)):
                out_datas.append(self.__in_quantize[idx].get_dequan_data(in_datas[idx]))
        else:
            os._exit(-1)  # , print('wrong in data type!')

        return out_datas

    def do_float_scale(self, in_datas: list):
        out_datas = None
        if isinstance(in_datas, np.ndarray):
            assert isinstance(self.__quantizes, object)  # , print(
            # 'in data is array, output quantize must be single quantize!')
            out_datas = self.__quantizes['feat']['so0'].get_quan_data(in_datas)
        elif isinstance(in_datas, list):
            assert len(in_datas) == len(self.__quantizes.keys())  # , print('must be has the same length')
            out_datas = list()
            for idx, _ in range(len(in_datas)):
                out_datas.append(self.__quantizes['feat']['so' + str(idx)].get_quan_data(in_datas[idx]))
        else:
          os._exit(-1)  # , print('wrong in data type!')

        return out_datas

    def get_datatype(self, data):

        def get_type(data):
            """
            It returns the key of the dictionary that corresponds to the data type of the input data
            
            :param data: the input data, which is a list of dictionaries or dict or numpy
            :return: The datatype of the input data.
            """
            key = -1

            for key_, value_ in self.get_ops_setting()['setting']['bits_dict'].items():
                if isinstance(data, list):
                    datatype = data[0]['output'].dtype.type
                elif isinstance(data, dict):
                    datatype = data['output'].dtype.type
                else:
                    datatype = data.dtype.type

                if isinstance(value_, str):
                    value_ = eval(value_)

                if datatype == value_:
                    key = key_
                    break

            assert key >= 0

            return key

        if isinstance(data, list):
            return [get_type(data_) for data_ in data]
        else:
            return [get_type(data)]

    def forward(self, in_data, **kwargs):
        try:
            outputs = list()
            ops_instance = self.get_ops_instance()
            layer_type = self.get_layer_type()

            self.set_in_data(in_data)

            if layer_type == 'data':
                outputs = ops_instance(in_data, **kwargs)
            else:
                in_idxs = self.get_input_idx()

                if len(in_idxs) > 1:
                    in_data = in_data
                else:
                    in_data = in_data[0]

                if isinstance(ops_instance, list):
                    for op in ops_instance:
                        in_data = op(in_data, **kwargs)
                        outputs.append(in_data)
                else:
                    outputs = ops_instance(in_data, **kwargs)
                    # if self.get_scale_type() == "table":
                    #     table = ops_instance.get_table()
                    #     self.set_table(table)

            # if not self.is_extend():
            #     if isinstance(outputs, list):
            #         layer_out = outputs[-1]
            #     else:
            #         layer_out = outputs
            # else:
            #     layer_out = outputs
            # setting = self.get_ops_setting()['setting']
            # bits_dict, mins, maxs = setting['bits_dict'], setting['mins'], setting['maxs']
            # out_type = self.get_output_type()
            # data_type, min_v, max_v = bits_dict[out_type], mins[out_type], maxs[out_type]
            # if isinstance(layer_out['output'], list):
            #     for idx, _ in enumerate(layer_out['output']):
            #         out = clip(layer_out['output'][idx], min_v, max_v)
            #         layer_out['output'][idx] = out.astype(data_type)
            # else:
            #     out = clip(layer_out['output'], min_v, max_v)
            #     layer_out['output'] = out.astype(data_type)
            # # if self.get_is_result_layer():
            #     # print('test')
            self.set_out_data(outputs)
            # if self.get_layer_name() in ['Conv_189']:
            #     import os
            #     name_0 = os.path.join('/home/shiqing/Downloads', 'QConv_189.npy')
            #     output0 = self.get_quantize()['feat']['so0'].get_dequan_data(outputs[-1]['output'])
            #     if os.path.exists(name_0):
            #         data_0 = np.load(name_0)
            #         data_0 = np.row_stack((data_0, output0))
            #     else:
            #         data_0 = output0
            #     np.save(name_0, data_0.astype(np.float32))
        except:
            error_info = "layer of {} simulation wrong!".format(self.get_layer_name())
            print(error_info)
            os._exit(-1)

    def checkerror(self):
        pass

    def create_node(self, op_type, inputs, outputs, attrs={}, node_name='', domain=None):
        if node_name == '':
            node_name = self.get_layer_name() + outputs[0]
        if domain is None:
            return onnx.helper.make_node( # type: ignore
                op_type,
                inputs=inputs,
                outputs=outputs,
                name=node_name,
                **attrs
            )
        else:
            return onnx.helper.make_node(
                op_type,
                inputs=inputs,
                outputs=outputs,
                name=node_name,
                domain=domain,
                **attrs
            )

    def get_init_c(self):
        NotImplemented

    def get_init_h(self):
        NotImplemented

    @abstractmethod
    def export_onnx(self):
        pass

    @staticmethod
    def insert_quant(self, layer_name, in_name, out_names, min_value, max_value, scale, zero_point):
        nodes, initializers = [], []
        input_name = copy.deepcopy(in_name)
        out_name = out_names[0] if isinstance(out_names, list) else out_names
        # layer_name = self.get_layer_name()
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[input_name, layer_name + '_quant_scale'],
                outputs=[layer_name + '_quant_mul_output']
            )
        ])
        tmp_name = layer_name + '_quant_mul_output'
        if zero_point != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + '_dequant_zi'],
                    outputs=[layer_name + '_quant_add_output']
                )
            )
            tmp_name = layer_name + '_quant_add_output'
            initializers.append(create_initializer(zero_point, layer_name + '_dequant_zi'))
        nodes.append(
            self.create_node(
                'Round',
                inputs=[tmp_name],
                outputs=[layer_name + '_quant_round_output']
            )
        )
        output_name = out_name if out_name else layer_name + 'quant_clip_output'
        nodes.append(
            self.create_node(
                'Clip',
                inputs=[layer_name + '_quant_round_output', layer_name + '_quant_min', layer_name + '_quant_max'],
                outputs=[output_name]
            )
        )

        initializers.extend([
            create_initializer(1 / scale, layer_name + '_quant_scale'),
            create_initializer(min_value, layer_name + '_quant_min'),
            create_initializer(max_value, layer_name + '_quant_max')
        ])

        return nodes, initializers, tmp_name

    @staticmethod
    def insert_dequant(self, layer_name, in_name, out_name, scale, zero_point):
        nodes, initializers = [], []
        input_name = copy.deepcopy(in_name)
        # layer_name = self.get_layer_name()
        if zero_point != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[input_name, layer_name + input_name + 'dequant_zi'],
                    outputs=[layer_name + input_name + 'dequant_sub_output']
                )
            )
            initializers.append(create_initializer(zero_point, layer_name + input_name + 'dequant_zi'))
            input_name = layer_name + input_name + 'dequant_sub_output'
        output_name = out_name if out_name else layer_name + input_name + 'dequant_mul_output'
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[input_name, layer_name + input_name + 'dequant_scale'],
                outputs=[output_name]
            )
        ])
        initializers.append(create_initializer(scale, layer_name + input_name + 'dequant_scale'))

        return nodes, initializers, output_name

    # reload quantized layer from read binary graph file
    # context comes from interface graph
    # context must has quantize(in/out)/datacorrect/scales/table/rnn_recycle_param
    def reload(self):
        pass
# class ActtionLayer()


# @LAYER.register_module(name='single input layer')
class SingleInputLayer(Layer):
    def __init__(self, **kwargs):
        super(SingleInputLayer, self).__init__(**kwargs)

    @abstractmethod
    def export_onnx(self):
        pass

    # this layer has no quantize process, current scale := in_scale
    def smooth_scale(self, scale):
        self.set_scale(scale)


class WeightedLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightedLayer, self).__init__(**kwargs)

    def create_initializer(self, data, name):
        param = copy.deepcopy(data)
        if isinstance(param, np.ndarray):
            param = param.reshape(1, -1)
            if self.get_layer_type().lower() in ['conv', 'depthwiseconv']:
                param = param.reshape(1, -1, 1, 1)
            if self.get_layer_type().lower() in ['conv1d', 'depthwiseconv1d']:
                param = param.reshape(1, -1, 1)
        return create_initializer(param, name)

    def insert_act(self, layer_name, in_name, out_name):
        nodes, initializers = [], []
        if self.get_ops_instance()[-1].get_class_name() != 'Act':
            out_name = out_name if out_name else layer_name + '_Relu_out'
            nodes.append(
                onnx.helper.make_node(
                    'Relu',
                    inputs=[in_name],
                    outputs=[out_name]
                )
            )
        return nodes, initializers, out_name

    def create_w(self, data, name, dtype=np.float32, trans=False):
        if trans:
            w = data.transpose((1, 0))
        else:
            w = copy.deepcopy(data)
        return create_initializer(w, name, dtype=dtype)

    def export_rshiftscale(self, op_type, layer_name, in_names, out_names, attrs, trans=False):
        zi = self.get_in_scale()[0]['zero_point']
        # zo = self.get_scale()[0]['zero_point']
        nodes, initializers = [], []
        input_name = copy.deepcopy(in_names)[0]
        out_shift = copy.deepcopy(self.get_scales()[-1]['out_shift'])
        out_scale = copy.deepcopy(self.get_scales()[-1]['fscale'])
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        if isinstance(out_shift, torch.Tensor):
            out_shift = out_shift.numpy()
        if zi != 0:
            nodes.extend([
                self.create_node(
                    'Sub',
                    inputs=[input_name, layer_name + 'zi'],
                    outputs=[layer_name + '_sub_1']
                )])
            input_name = layer_name + '_sub_1'
            initializers.append(create_initializer(zi, layer_name + "zi"))

        if isinstance(self.get_qbias(), np.ndarray):
            inputs = [input_name, layer_name + 'w', layer_name + 'b']
        else:
            inputs = [input_name, layer_name + 'w']
        nodes.append(
            self.create_node(op_type, inputs, [layer_name + '_conv_y'], attrs)
        )

        act_node, act_initializers, act_out_name = self.insert_act(layer_name, layer_name + '_conv_y', None)
        act_out_name = act_out_name if act_out_name else layer_name + '_conv_y'
        nodes.extend(act_node)
        initializers.extend(act_initializers)

        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[act_out_name, layer_name + 'out_shift'],
                outputs=[layer_name + '_mul_1']
            ),
            self.create_node(
                virtual_op_type, inputs=[layer_name + '_mul_1'], outputs=[layer_name + '_Floor_1']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_Floor_1', layer_name + 'min1', layer_name + 'max1'],
                outputs=[layer_name + '_clip_1']
            )
        ])
        act_node, act_initializers, act_out_name = self.insert_act(layer_name, layer_name + '_clip_1', None)
        act_out_name = act_out_name if act_out_name else layer_name + '_clip_1'
        nodes.extend(act_node)
        initializers.extend(act_initializers)
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[act_out_name, layer_name + 'scale_end'],
                outputs=[out_names[0]]
            )
        ])

        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min1_v, max1_v = mins[bit_select], maxs[bit_select]

        initializers.extend([
            # create_initializer(self.get_ops_instance()[0].p_weights.numpy(), layer_name + "w"),
            self.create_w(self.get_ops_instance()[0].p_weights.numpy(), layer_name + "w", trans=trans),
            self.create_initializer(2 ** np.float32(out_shift) * self.eps, layer_name + "out_shift"),
            self.create_initializer(out_scale, layer_name + 'scale_end'),
            # create_initializer(2**np.float32(out_shift), layer_name + "out_shift"),
            create_initializer(min1_v, layer_name + "min1"),
            create_initializer(max1_v, layer_name + "max1")
        ])
        if isinstance(self.get_qbias(), np.ndarray):
            initializers.append(
                create_initializer(self.get_qbias(), layer_name + 'b')
            )

        return nodes, initializers

    def export_intscale(self, op_type, layer_name, in_names, out_names, attrs, trans=False):
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        input_name = copy.deepcopy(in_names)[0]
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        nodes, initializers = [], []
        if zi != 0:
            nodes.extend([
                self.create_node(
                    'Sub',
                    inputs=[input_name, layer_name + 'zi'],
                    outputs=[layer_name + '_sub_1']
                )])
            input_name = layer_name + '_sub_1'
            initializers.append(create_initializer(zi, layer_name + "zi"))
        if isinstance(self.get_qbias(), np.ndarray):
            nodes.append(
                self.create_node(
                    op_type,
                    inputs=[input_name, layer_name + 'w', layer_name + 'b'],
                    outputs=[layer_name + '_conv_y'],
                    attrs=attrs)
            )
        else:
            nodes.append(
                self.create_node(
                    op_type,
                    inputs=[input_name, layer_name + 'w'],
                    outputs=[layer_name + '_conv_y'],
                    attrs=attrs)
            )
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[layer_name + '_conv_y', layer_name + 'out_shift'],
                outputs=[layer_name + '_mul_1']
            ),
            self.create_node(
                virtual_op_type,
                inputs=[layer_name + '_mul_1'],
                outputs=[layer_name + '_Floor_1']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_Floor_1', layer_name + 'min1', layer_name + 'max1'],
                outputs=[layer_name + '_clip_1']
            ),
        ])
        act_node, act_initializers, act_out_name = self.insert_act(layer_name, layer_name + '_clip_1', None)
        act_out_name = act_out_name if act_out_name else layer_name + '_clip_1'
        nodes.extend(act_node)
        initializers.extend(act_initializers)

        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[act_out_name, layer_name + 'out_scale'],
                outputs=[layer_name + '_mul_2']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + '_mul_2', layer_name + 'int_scale'],
                outputs=[layer_name + '_mul_3']
            ),
            self.create_node(
                virtual_op_type,
                inputs=[layer_name + '_mul_3'],
                outputs=[layer_name + '_Floor_2']
            )
        ])
        name_1 = layer_name + '_Floor_2'
        if zo != 0:
            nodes.extend([self.create_node(
                'Add',
                inputs=[name_1, layer_name + 'zo'],
                outputs=[layer_name + '_add_1'])
            ])
            name_1 = layer_name + '_add_1'
            initializers.append(create_initializer(zo, layer_name + "zo"))
        nodes.extend([
            self.create_node(
                'Clip',
                inputs=[name_1, layer_name + 'min2', layer_name + 'max2'],
                outputs=[out_names[0]])
        ])
        out_shift, out_scale = copy.deepcopy(self.get_scales()[-1]['out_shift']) \
            , copy.deepcopy(self.get_scales()[-1]['out_scale'])
        if isinstance(out_shift, torch.Tensor):
            out_shift = out_shift.numpy()
        if isinstance(out_scale, torch.Tensor):
            out_scale = out_scale.numpy()
        int_scale = self.get_scales()[-1]['int_scale'] if 'int_scale' in self.get_scales()[-1].keys(
        ) else self.get_ops_setting()['setting']['int_scale']
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min2_v, max2_v = mins[bit_select], maxs[bit_select]
        if bit_select % 2 == 0:
            base_num = 2 ** bit
            min1_v, max1_v = 0, base_num - 1
        else:
            base_num = 2 ** (bit - 1)
            min1_v, max1_v = -base_num, base_num - 1

        initializers.extend([
            # create_initializer(self.get_ops_instance()[0].p_weights.numpy(), layer_name + "w"),
            self.create_w(self.get_ops_instance()[0].p_weights.numpy(), layer_name + "w", trans=trans),
            self.create_initializer(2 ** np.float32(out_shift * self.eps), layer_name + "out_shift"),
            self.create_initializer(np.float32(out_scale), layer_name + "out_scale"),
            create_initializer(1 / (2 ** int_scale) * self.eps, layer_name + "int_scale"),
            create_initializer(min1_v, layer_name + "min1"),
            create_initializer(min2_v, layer_name + "min2"),
            create_initializer(max1_v, layer_name + "max1"),
            create_initializer(max2_v, layer_name + "max2"),
        ])
        if isinstance(self.get_qbias(), np.ndarray):
            initializers.append(
                create_initializer(self.get_qbias(), layer_name + 'b')
            )

        return nodes, initializers

    def export_floatscale(self, op_type, layer_name, in_names, out_names, attrs, trans=False):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        input_name = copy.deepcopy(in_names)[0]
        if zi != 0:
            nodes.extend([
                self.create_node(
                    'Sub',
                    inputs=[input_name, layer_name + 'zi'],
                    outputs=[layer_name + '_sub_1']
                )])
            input_name = layer_name + '_sub_1'
            initializers.append(create_initializer(zi, layer_name + "zi"))

        if isinstance(self.get_qbias(), np.ndarray):
            nodes.append(
                self.create_node(
                    op_type,
                    inputs=[input_name, layer_name + 'w', layer_name + 'b'],
                    outputs=[layer_name + '_' + op_type + '_y'],
                    attrs=attrs)
            )
        else:
            nodes.append(
                self.create_node(
                    op_type,
                    inputs=[input_name, layer_name + 'w'],
                    outputs=[layer_name + '_' + op_type + '_y'],
                    attrs=attrs)
            )
        act_node, act_initializers, act_out_name = self.insert_act(layer_name, layer_name + '_' + op_type + '_y', None)
        act_out_name = act_out_name if act_out_name else layer_name + '_' + op_type + '_y'
        nodes.extend(act_node)
        initializers.extend(act_initializers)
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[act_out_name, layer_name + 'out_scale'],
                outputs=[layer_name + '_mul_2']
            ),
            self.create_node(
                'Round',
                inputs=[layer_name + '_mul_2'],
                outputs=[layer_name + '_round_out']
            )
        ])
        name_1 = layer_name + '_round_out'
        if zo != 0:
            nodes.extend([self.create_node(
                'Add',
                inputs=[name_1, layer_name + '_zo'],
                outputs=[layer_name + '_add_1'])
            ])
            name_1 = layer_name + '_add_1'
            initializers.append(create_initializer(zo, layer_name + "_zo"))
        nodes.extend([
            self.create_node(
                'Clip',
                inputs=[name_1, layer_name + 'min1', layer_name + 'max1'],
                outputs=[out_names[0]])
        ])
        out_scale = copy.deepcopy(self.get_scales()[-1]['out_scale'])
        if isinstance(out_scale, torch.Tensor):
            out_scale = out_scale.numpy()
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min2_v, max2_v = mins[bit_select], maxs[bit_select]
        initializers.extend([
            # create_initializer(self.get_scales()[0].p_weights.numpy(), layer_name + "w"),
            self.create_w(self.get_ops_instance()[0].p_weights.numpy(), layer_name + "w", trans=trans),
            self.create_initializer(out_scale, layer_name + "out_scale"),
            # crpassword_initializer(self.get_scales()[-1]['zi'], layer_name + "zi"),
            # create_initializer(self.get_scales()[-1]['zo'], layer_name + "zo"),
            create_initializer(min2_v, layer_name + "min1"),
            create_initializer(max2_v, layer_name + "max1"),
        ])
        if isinstance(self.get_qbias(), np.ndarray):
            initializers.append(
                create_initializer(self.get_qbias(), layer_name + 'b')
            )

        return nodes, initializers

    def export_shiftfloatscale(self, op_type, layer_name, in_names, out_names, attrs, trans=False):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        input_name = copy.deepcopy(in_names)[0]
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        if zi != 0:
            nodes.extend([
                self.create_node(
                    'Sub',
                    inputs=[input_name, layer_name + 'zi'],
                    outputs=[layer_name + '_sub_1']
                )])
            input_name = layer_name + '_sub_1'
            initializers.append(create_initializer(self.get_scales()[-1]['zi'], layer_name + "zi"))
        if isinstance(self.get_qbias(), np.ndarray):
            inputs_lst = [input_name, layer_name + 'w', layer_name + 'b']
        else:
            inputs_lst = [input_name, layer_name + 'w']
        nodes.append(
            self.create_node(
                op_type,
                inputs=inputs_lst,
                outputs=[layer_name + '_' + op_type + '_y'],
                attrs=attrs
            )
        )

        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[layer_name + '_' + op_type + '_y', layer_name + 'out_shift'],
                outputs=[layer_name + '_mul_1']
            ),
            self.create_node(
                virtual_op_type,
                inputs=[layer_name + '_mul_1'],
                outputs=[layer_name + '_Floor_1']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_Floor_1', layer_name + 'min1', layer_name + 'max1'],
                outputs=[layer_name + '_clip_1']
            )
        ])
        act_node, act_initializers, act_out_name = self.insert_act(layer_name, layer_name + '_clip_1', None)
        act_out_name = act_out_name if act_out_name else layer_name + '_clip_1'
        nodes.extend(act_node)
        initializers.extend(act_initializers)
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[act_out_name, layer_name + 'out_scale'],
                outputs=[layer_name + '_mul_2']
            )
        ])
        name_1 = layer_name + '_mul_2'
        if zo != 0:
            nodes.extend([self.create_node(
                'Add',
                inputs=[name_1, layer_name + 'zo'],
                outputs=[layer_name + '_add_1'])
            ])
            name_1 = layer_name + '_add_1'
            initializers.append(create_initializer(self.get_scales()[-1]['zo'], layer_name + "zo"))
        nodes.extend([
            self.create_node(
                'Round',
                inputs=[name_1],
                outputs=[layer_name + '_round_out']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_round_out', layer_name + 'min2', layer_name + 'max2'],
                outputs=[out_names[0]])
        ])
        out_shift, out_scale = copy.deepcopy(self.get_scales()[-1]['out_shift']) \
            , copy.deepcopy(self.get_scales()[-1]['out_scale'])
        if isinstance(out_shift, torch.Tensor):
            out_shift = out_shift.numpy()
        if isinstance(out_scale, torch.Tensor):
            out_scale = out_scale.numpy()
        int_scale = self.get_ops_setting()['setting']['int_scale']
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min2_v, max2_v = mins[bit_select], maxs[bit_select]
        if bit_select % 2 == 0:
            base_num = 2 ** bit
            min1_v, max1_v = 0, base_num - 1
        else:
            base_num = 2 ** (bit - 1)
            min1_v, max1_v = -base_num, base_num - 1

        initializers.extend([
            # create_initializer(self.get_scales()[0].p_weights.numpy(), layer_name + "w"),
            self.create_w(self.get_ops_instance()[0].p_weights.numpy(), layer_name + "w", trans=trans),
            self.create_initializer(2 ** np.float32(out_shift * self.eps), layer_name + "out_shift"),
            self.create_initializer(out_scale * self.eps, layer_name + "out_scale"),
            create_initializer(1 / (2 ** int_scale), layer_name + "int_scale"),
            # create_initializer(self.get_scales()[-1]['zi'], layer_name + "zi"),
            # create_initializer(self.get_scales()[-1]['zo'], layer_name + "zo"),
            create_initializer(min1_v, layer_name + "min1"),
            create_initializer(min2_v, layer_name + "min2"),
            create_initializer(max1_v, layer_name + "max1"),
            create_initializer(max2_v, layer_name + "max2"),
        ])
        if isinstance(self.get_qbias(), np.ndarray):
            initializers.append(
                create_initializer(self.get_qbias(), layer_name + 'b')
            )

        return nodes, initializers

    def export_ffloatscale(self, op_type, layer_name, in_names, out_names, attrs, trans=False):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        input_name = copy.deepcopy(in_names)[0]
        if zi != 0:
            nodes.extend([
                self.create_node(
                    'Sub',
                    inputs=[input_name, layer_name + 'zi'],
                    outputs=[layer_name + '_sub_1']
                )])
            input_name = layer_name + '_sub_1'
            initializers.append(create_initializer(self.get_scales()[-1]['zi'], "zi"))
        if isinstance(self.get_qbias(), np.ndarray):
            nodes.append(
                self.create_node(
                    op_type,
                    inputs=[input_name, layer_name + 'w', layer_name + 'b'],
                    outputs=[layer_name + '_conv_y'],
                    attrs=attrs)
            )
        else:
            nodes.append(
                self.create_node(
                    op_type,
                    inputs=[input_name, layer_name + 'w'],
                    outputs=[layer_name + '_conv_y'],
                    attrs=attrs)
            )
        act_node, act_initializers, act_out_name = self.insert_act(layer_name, layer_name + '_conv_y', None)
        act_out_name = act_out_name if act_out_name else layer_name + '_conv_y'
        nodes.extend(act_node)
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[act_out_name, layer_name + 'out_scale'],
                outputs=[]
            ),
            self.create_node(
                'Round',
                inputs=[layer_name + '_clip_1'],
                outputs=[layer_name + '_round_out']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_round_out'],
                outputs=[out_names[0]]
            )
        ])

        out_scale = copy.deepcopy(self.get_scales()[-1]['out_scale'])
        if isinstance(out_scale, torch.Tensor):
            out_scale = out_scale.numpy()

        initializers.extend([
            # create_initializer(self.get_scales()[0].p_weights.numpy(), layer_name+"w"),
            self.create_w(self.get_ops_instance()[0].p_weights.numpy(), layer_name + "w", trans=trans),
            self.create_initializer(out_scale, layer_name + "out_scale"),
            # create_initializer(self.get_scales()[-1]['zi'], "zi")
        ])
        if isinstance(self.get_qbias(), np.ndarray):
            initializers.append(
                create_initializer(self.get_qbias(), layer_name + 'b')
            )

        return nodes, initializers

    def export_onnx(self):
        pass

    # this layer has no quantize process, current scale := in_scale
    def smooth_scale(self, scale):
        self.set_scale(scale)


class ActivationLayer(SingleInputLayer):
    def __init__(self, **kwargs):
        super(ActivationLayer, self).__init__(**kwargs)

    def export_table(self, op_type, layer_name, in_names, out_names, attrs={}):
        # clip_outs = []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]

        zi = self.get_in_scale()[0]['zero_point']

        in_name = copy.deepcopy(in_names)[0]
        # layer_name, in_name, out_name, min_value, max_value, scale, zero_point
        nodes, initializers, out_name = self.insert_dequant(self, layer_name, in_name, None,
                                                            self.get_in_scale()[0]['scale'], zi)

        nodes.extend([
            self.create_node(
                op_type,
                inputs=[out_name],
                outputs=[layer_name + op_type + '_out'],
                attrs=attrs
            )
        ])
        # layer_name, in_name, out_name, scale, zero_point
        de_nodes, de_initializer, _ = \
            self.insert_quant(self, layer_name, layer_name + op_type + '_out', out_names,
                              min_value, max_value, self.get_scale()[0]['scale'], self.get_scale()[0]['zero_point'])
        nodes.extend(de_nodes)
        initializers.extend(de_initializer)
        return nodes, initializers

    def export_float(self, op_type, layer_name, in_names, out_names, attrs={}):
        # clip_outs = []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]

        zi = self.get_in_scale()[0]['zero_point']

        in_name = copy.deepcopy(in_names)[0]
        # layer_name, in_name, out_name, min_value, max_value, scale, zero_point
        nodes, initializers, out_name = self.insert_dequant(self, layer_name, in_name, None,
                                                            self.get_in_scale()[0]['scale'], zi)

        nodes.extend([
            self.create_node(
                op_type,
                inputs=[out_name],
                outputs=[layer_name + op_type + '_out'],
                attrs=attrs
            )
        ])
        # layer_name, in_name, out_name, scale, zero_point
        de_nodes, de_initializer, _ = \
            self.insert_quant(self, layer_name, layer_name + op_type + '_out', out_names,
                              min_value, max_value, self.get_scale()[0]['scale'], self.get_scale()[0]['zero_point'])
        nodes.extend(de_nodes)
        initializers.extend(de_initializer)
        return nodes, initializers

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = dict()
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            bits_dict = self.get_ops_setting()['setting']['bits_dict']
            bit_select = self.get_ops_setting()['setting']['bit_select']
            dtype = bits_dict[bit_select]
            if self.get_scale_type() == "table":
                table = self.get_ops_instance().get_table() # type: ignore
            else:
                table = []
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            table = onnx.numpy_helper.from_array(np.array(table, dtype=dtype))
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                table=table,
            )
            attrs.update(qparam)

        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = self.get_layer_type()
        op_type = op_type[0].upper() + op_type[1:]
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, {})
        return nodes, initializers


# @LAYER.register_module(name='multi input layer')
class MultiInputLayer(Layer):
    def __init__(self, **kwargs):
        super(MultiInputLayer, self).__init__(**kwargs)
        self.has_weights = False
        self.process_scale = "preintscale"
        self.weights = None
        self.weights_idx = None

    def set_has_weghts(self, has_Weight):
        self.has_weights = has_Weight

    def has_weight(self):
        return self.has_weight

    def set_process_scale(self, process_scale):
        self.process_scale = process_scale

    def get_process_scale(self):
        return self.process_scale

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def get_weights_idx(self):
        return self.weights_idx

    def set_weights_idx(self, idx):
        self.weights_idx = idx

    def reconstruct_inputs(self, setting, op_setting, si, in_quantize):
        f_weights = None
        p_weights = None
        weight_idx = None
        # op_setting["weight_idx"] = None
        if "weight_idx" in setting['attrs'][0].keys():
            weight_idx = setting['attrs'][0]["weight_idx"]
            self.set_weights_idx(weight_idx)
            si.append(self.get_w_scale())
            f_weights = self.get_ops_setting()['attrs'][0]['weights']
            p_weights = self.get_qweight()
            self.set_has_weghts(True)
            self.set_process_scale(op_setting['process_scale'])
            if self.process_scale == "preintscale":
                self.set_weights(p_weights)
            else:
                self.set_weights(f_weights)
            op_setting["weight_idx"] = weight_idx
            w_quan = self.get_quantize()['w_quan']
            w_quan = w_quan if isinstance(w_quan, list) else [w_quan]
            [in_quantize.insert(item, w_quan[i]) for i, item in enumerate(weight_idx)]

        return f_weights, p_weights, weight_idx, in_quantize

    def export_smooth(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, clip_outs = [], []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        initializers = [
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
        ]
        for idx in range(len(in_names)):
            mul_1_out = layer_name + 'mul_1_' + str(idx)
            scale = self.get_in_scale()[idx]['scale']
            zero_point = self.get_in_scale()[idx]['zero_point']
            nodes_, initializers_, output_name = self.insert_dequant(self, layer_name, in_names[idx], mul_1_out, 1,
                                                                     zero_point)
            nodes.extend(nodes_)
            initializers.extend(initializers_)
            clip_outs.append(mul_1_out)

        nodes.extend([
            self.create_node(
                op_type,
                inputs=clip_outs,
                outputs=[layer_name + '_' + op_type + '_out'],
                attrs=attrs
            )
        ])
        quant_p = self.get_scale()[0]
        names = layer_name + '_' + op_type + '_out'
        scale, zo = quant_p['scale'], quant_p['zero_point']
        nodes_, initializers_, output_name = self.insert_quant(self, layer_name, names, out_names[0], min_value,
                                                               max_value, 1, zo)
        nodes.extend(nodes_)
        initializers.extend(initializers_)

        return nodes, initializers

    def branch_floatscale(self, nodes, initializers, clip_outs, in_name, layer_name, idx):
        data, zero_point = in_name, layer_name + 'zi' + str(idx)
        sub_out, mul_scale = layer_name + 'sub_' + str(idx), layer_name + 'out_scale' + str(idx)
        ceil_out = layer_name + 'mul_2_' + str(idx)
        zi = self.get_in_scale()[idx]['zero_point']
        if zi != 0:
            nodes.extend([
                self.create_node(
                    'Sub',
                    inputs=[data, zero_point],
                    outputs=[sub_out]
                )
            ])
            data = sub_out
            initializers.append(create_initializer(zi, zero_point))
        flag = self.get_scales()[idx]['out_scale'] == 1
        if not flag:
            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[data, mul_scale],
                    outputs=[ceil_out]
                )
            ])
            initializers.extend([create_initializer(self.get_scales()[idx]['out_scale'], mul_scale)])

            clip_outs.append(ceil_out)
        else:
            clip_outs.append(data)

    def export_floatscale(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, clip_outs = [], []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        initializers = [
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
        ]
        weights_idx = self.get_weights_idx()
        clip_outs = []
        if isinstance(weights_idx, list):
            weights = self.get_qweight()
            weights = weights if isinstance(weights, list) else [weights]
            input_idx = 0
            for idx in range(len(weights_idx) + len(in_names)):
                if idx in weights_idx:
                    scale = self.get_scales()[idx]
                    weight = (weights[idx] - scale['zi']) * scale['out_scale']
                    clip_out = layer_name + '_weight_' + str(idx)
                    initializers.extend([create_initializer(weight.astype(np.float32), clip_out)])
                    clip_outs.append(clip_out)
                else:
                   self.branch_floatscale(nodes, initializers, clip_outs, in_names[input_idx], layer_name, idx)
                   input_idx += 1
        else:
            for idx in range(len(in_names)):
                self.branch_floatscale(nodes, initializers, clip_outs, in_names[idx], layer_name, idx)
                # input_data.append(data)

        nodes.extend([
            self.create_node(
                op_type,
                inputs=clip_outs,
                outputs=[layer_name + op_type + '_out'],
                attrs=attrs
            )
        ])
        op_type_out = layer_name + op_type + '_out'
        zo = self.get_scale()[0]['zero_point']
        if zo != 0:
            nodes.extend([
                self.create_node(
                    'Add',
                    inputs=[layer_name + op_type + '_out', layer_name + 'zo'],
                    outputs=[layer_name + 'add_out']
                )
            ])
            op_type_out = layer_name + 'add_out'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.extend([
            self.create_node(
                'Clip',
                inputs=[op_type_out, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]]
            )
        ])

        return nodes, initializers

    def branch_preintscale(self, nodes, initializers, clip_outs, virtual_op_type, in_name, layer_name, idx):
        data, zero_point = in_name, layer_name + 'zi' + str(idx)
        sub_out, mul_scale = layer_name + 'sub_' + str(idx), layer_name + 'out_scale' + str(idx)
        mul_int_scale = layer_name + 'int_scale' + str(idx)
        mul_1_out, clip_out = layer_name + 'mul_1_' + str(idx), layer_name + 'clip_' + str(idx)
        mul_2_out, ceil_out = layer_name + 'mul_2_' + str(idx), layer_name + 'Floor_' + str(idx)
        zi = self.get_in_scale()[idx]['zero_point']
        if zi != 0:
            nodes.extend([
                self.create_node(
                    'Sub',
                    inputs=[data, zero_point],
                    outputs=[sub_out]
                )
            ])
            data = sub_out
            initializers.append(create_initializer(zi, zero_point))
        flag = self.get_scales()[idx]['out_scale'] == 1
        if not flag:
            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[data, mul_scale],
                    outputs=[mul_1_out]
                ),
                self.create_node(
                    'Mul',
                    inputs=[mul_1_out, mul_int_scale],
                    outputs=[mul_2_out]
                ),
                self.create_node(
                    virtual_op_type,
                    inputs=[mul_2_out],
                    outputs=[ceil_out]
                )
            ])
            initializers.extend([create_initializer(self.get_scales()[idx]['out_scale'], mul_scale),
                                create_initializer(1 / (2 ** self.get_scales()[idx]['int_scale']) * self.eps, mul_int_scale)])

            clip_outs.append(ceil_out)
        else:
            clip_outs.append(data)

    def export_preintscale(self, op_type, layer_name, in_names, out_names, attrs):

        nodes, clip_outs = [], []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        initializers = [
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
        ]
        weights_idx = self.get_weights_idx()
        clip_outs = []
        if isinstance(weights_idx, list):
            weights = self.get_qweight()
            weights = weights if isinstance(weights, list) else [weights]
            input_idx = 0
            for idx in range(len(weights_idx) + len(in_names)):
                if idx in weights_idx:
                    i = weights_idx.index(idx)
                    scale = self.get_scales()[idx]
                    weight = ((weights[i] - scale['zi']) * scale['out_scale']) >> scale['int_scale']
                    clip_out = layer_name + '_weight_' + str(idx)
                    initializers.extend([create_initializer(weight.astype(np.float32), clip_out)])
                    clip_outs.append(clip_out)
                else:
                   self.branch_preintscale(nodes, initializers, clip_outs, virtual_op_type, in_names[input_idx], layer_name, idx)
                   input_idx += 1
        else:
            for idx in range(len(in_names)):
                self.branch_preintscale(nodes, initializers, clip_outs, virtual_op_type, in_names[idx], layer_name, idx)
                # input_data.append(data)

        nodes.extend([
            self.create_node(
                op_type,
                inputs=clip_outs,
                outputs=[layer_name + op_type + '_out'],
                attrs=attrs
            )
        ])
        op_type_out = layer_name + op_type + '_out'
        zo = self.get_scale()[0]['zero_point']
        if zo != 0:
            nodes.extend([
                self.create_node(
                    'Add',
                    inputs=[layer_name + op_type + '_out', layer_name + 'zo'],
                    outputs=[layer_name + 'add_out']
                )
            ])
            op_type_out = layer_name + 'add_out'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.extend([
            self.create_node(
                'Clip',
                inputs=[op_type_out, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]]
            )
        ])

        return nodes, initializers

    def branch_float(self, nodes, initializers, clip_outs, in_name, layer_name, idx):
        data, zero_point = in_name, layer_name + 'zi' + str(idx)
        sub_out = layer_name + 'sub_' + str(idx)
        mul_1_out = layer_name + 'mul_1_' + str(idx)
        zi = self.get_in_scale()[idx]['zero_point']
        if zi != 0:
            nodes.extend([
                self.create_node(
                    'Sub',
                    inputs=[data, zero_point],
                    outputs=[sub_out]
                )
            ])
            data = sub_out
            initializers.append(create_initializer(zi, zero_point))
        flag = self.get_scales()[idx]['out_scale'] == 1
        if not flag:
            scale = self.get_in_scale()[idx]['scale']
            zero_point = self.get_in_scale()[idx]['zero_point']
            nodes_, initializers_, output_name = self.insert_dequant(self, layer_name, in_name, mul_1_out, scale,
                                                                     zero_point)
            nodes.extend(nodes_)
            initializers.extend(initializers_)
            clip_outs.append(mul_1_out)
        else:
            clip_outs.append(data)

    def export_float(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, clip_outs = [], []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        initializers = [
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
        ]
        weights_idx = self.get_weights_idx()
        clip_outs = []
        if isinstance(weights_idx, list):
            weights = self.get_qweight()
            weights = weights if isinstance(weights, list) else [weights]
            input_idx = 0
            for idx in range(len(weights_idx) + len(in_names)):
                if idx in weights_idx:
                    scale = self.get_scales()[idx]
                    weight = (weights[idx] - scale['zi']) * 1 / self.get_in_scale()[idx]['scale']
                    clip_out = layer_name + '_weight_' + str(idx)
                    initializers.extend([create_initializer(weight.astype(np.float32), clip_out)])
                    clip_outs.append(clip_out)
                else:
                   self.branch_float(nodes, initializers, clip_outs, in_names[input_idx], layer_name, idx)
                   input_idx += 1
        else:
            for idx in range(len(in_names)):
                self.branch_float(nodes, initializers, clip_outs, in_names[idx], layer_name, idx)
                # input_data.append(data)

        nodes.extend([
            self.create_node(
                op_type,
                inputs=clip_outs,
                outputs=[layer_name + '_' + op_type + '_out'],
                attrs=attrs
            )
        ])
        quant_p = self.get_scale()[0]
        names = layer_name + '_' + op_type + '_out'
        scale, zo = quant_p['scale'], quant_p['zero_point']
        nodes_, initializers_, output_name = self.insert_quant(self, layer_name, names, out_names[0], min_value,
                                                               max_value, scale, zo)
        nodes.extend(nodes_)
        initializers.extend(initializers_)

        return nodes, initializers

    # @abstractmethod
    def export_onnx(self):
        op_type = self.get_layer_type()
        op_type = op_type[0].upper() + op_type[1:]
        layer_name = self.get_layer_name()
        out_names = self.get_onnx_output_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, {})
        return nodes, initializers

    def parser(self, **kwargs):
        pass

    # def get_in_scale_idx(self):
    #     in_idx = self.get_input_idx()
    #     idx = self.get_idx()


class MultiOutputLayer(Layer):
    def __init__(self, **kwargs):
        super(MultiOutputLayer, self).__init__(**kwargs)

    def export_preintscale(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, clip_outs = [], []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        split_nums = op.split
        min_value, max_value = mins[bit_select], maxs[bit_select]
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        initializers = [
        ]
        split_names = [layer_name + 'split' + str(idx) for idx in range(len(split_nums))]
        names = in_names[0]
        zi = self.get_in_scale()[0]['zero_point']
        if zi != 0:
            nodes.append(self.create_node(
                'Sub',
                inputs=[in_names[0], layer_name + 'zi'],
                outputs=[layer_name + 'sub_out']
            ))
            names = layer_name + 'sub_out'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                op_type,
                inputs=[names, layer_name + 'split_nums'],
                outputs=split_names,
                attrs=dict(axis=self.get_ops_setting()['attrs'][0]['axis'])
            )
        ])
        # initializers.append(create_initializer(self.get_scales()[-1]['zi'], layer_name + 'zi'))
        initializers.append(create_initializer(split_nums, layer_name + 'split_nums', dtype=np.int64))
        for idx in range(len(split_nums)):
            data, zero_point = split_names[idx], layer_name + '_zo' + str(idx)
            mul_scale = layer_name + '_out_scale' + str(idx)
            add_name, mul_int_scale = layer_name + 'add_zo_' + str(idx), layer_name + '_int_scale' + str(idx)
            mul_1_out, clip_out = layer_name + '_mul_1_' + str(idx), layer_name + '_clip_' + str(idx)
            mul_2_out, ceil_out = layer_name + '_mul_2_' + str(idx), layer_name + '_ceil_' + str(idx)

            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[data, mul_scale],
                    outputs=[mul_1_out]
                ),
                self.create_node(
                    'Mul',
                    inputs=[mul_1_out, mul_int_scale],
                    outputs=[mul_2_out]
                ),
                self.create_node(
                    virtual_op_type,
                    inputs=[mul_2_out],
                    outputs=[ceil_out]
                )
            ])
            tmp_name = ceil_out
            if self.get_scale()[idx]['zero_point'] != 0:
                nodes.append(self.create_node(
                    'Add',
                    inputs=[ceil_out, zero_point],
                    outputs=[add_name]
                ))
                tmp_name = add_name
                initializers.append(create_initializer(self.get_scale()[idx]['zero_point'], zero_point))
            nodes.extend([
                self.create_node(
                    'Clip',
                    inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                    outputs=[out_names[idx]]
                )
            ])

            initializers.extend([create_initializer(self.get_scales()[idx]['out_scale'], mul_scale),
                                 create_initializer(min_value, layer_name + 'min'),
                                 create_initializer(max_value, layer_name + 'max'),
                                 create_initializer(1 / (2 ** self.get_scales()[idx]['int_scale']), mul_int_scale)])

        return nodes, initializers

    def export_float(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, clip_outs = [], []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        split_nums = op.split
        min_value, max_value = mins[bit_select], maxs[bit_select]
        initializers = [
            create_initializer(min_value, "min"),
            create_initializer(max_value, "max"),
        ]
        split_names = [layer_name + 'split' + str(idx) for idx in range(len(split_nums))]
        zi = self.get_in_scale()[0]['zero_point']
        names = in_names[0]
        if zi != 0:
            nodes.append(self.create_node(
                'Sub',
                inputs=[in_names[0], layer_name + 'zi'],
                outputs=[layer_name + 'sub_out']
            ))
            names = layer_name + 'sub_out'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[names, layer_name + 'scale'],
                outputs=[layer_name + 'Mul_out']
            ),
            self.create_node(
                op_type,
                inputs=[layer_name + 'Mul_out'],
                outputs=split_names,
                attrs=attrs)
        ])
        # initializers.append(create_initializer(self.get_scales()[-1]['zi'], layer_name + 'zi'))
        initializers.append(create_initializer(ops.si[0]['scale'], layer_name + 'scale'))

        for idx in range(len(split_nums)):
            data, zero_point = layer_name + split_names[idx], layer_name + 'zo' + str(idx)
            mul_scale = layer_name + 'out_scale' + str(idx)
            add_name, mul_int_scale = layer_name + 'add_zo_' + str(idx), layer_name + 'int_scale' + str(idx)
            mul_1_out, clip_out = layer_name + 'mul_1_' + str(idx), layer_name + 'clip_' + str(idx)
            ceil_out = layer_name + 'ceil_' + str(idx)

            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[data, mul_scale],
                    outputs=[mul_1_out]
                ),
                self.create_node(
                    'Floor',
                    inputs=[mul_1_out],
                    outputs=[ceil_out]
                )
            ])
            tmp_name = ceil_out
            if self.get_scale()[idx]['zero_point'] != 0:
                nodes.append(
                    self.create_node(
                        'Add',
                        inputs=[ceil_out, zero_point],
                        outputs=[add_name]
                    ))
                tmp_name = add_name
                initializers.append(create_initializer(self.get_scale()[idx]['zero_point'], zero_point))
            nodes.extend([
                self.create_node(
                    'Clip',
                    inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                    outputs=[out_names[idx]]
                )
            ])

            initializers.extend([create_initializer(self.get_scale()[idx]['scale'], mul_scale)])

        return nodes, initializers

    @abstractmethod
    def export_onnx(self):
        pass

    def parser(self, **kwargs):
        pass


class ShapeLayer(Layer):
    def __init__(self, **kwargs):
        super(ShapeLayer, self).__init__(**kwargs)

    def export_onnx(self):
        pass

    def parser(self, **kwargs):
        pass


class NormLayer(Layer):
    def __init__(self, **kwargs):
        super(NormLayer, self).__init__(**kwargs)

    def export_float(self, op_type, layer_name, in_names, out_names, setting, attrs):
        # nodes, initializers = [], []
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        norm_inputs = list(attrs.keys())
        norm_inputs = [layer_name + name for name in norm_inputs]

        nodes, initializers, dequant_output_name = \
            self.insert_dequant(self, layer_name, in_name, None, self.get_in_scale()[0]['scale'],
                                self.get_in_scale()[0]['zero_point'])
        norm_inputs.insert(0, dequant_output_name)
        nodes.extend([
            self.create_node(
                op_type,
                inputs=norm_inputs,
                outputs=[layer_name + op_type + '_output'],
                attrs=setting
            )
        ])
        # in_name, out_name, min_value, max_value, scale, zero_point
        quant_out = self.insert_quant(self, layer_name, layer_name + op_type + '_output', out_names,
                                      min_value, max_value, self.get_scale()[0]['scale'],
                                      self.get_scale()[0]['zero_point'])
        nodes.extend(quant_out[0])
        initializers.extend(quant_out[1])
        for key in attrs.keys():
            initializers.append(create_initializer(attrs[key], layer_name + key))

        return nodes, initializers

    def parser(self, **kwargs):
        pass


@LAYER.register_module(name='default')
class DefaultLayer(SingleInputLayer):
    def __init__(self, **kwargs):
        super(DefaultLayer, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        process_scale = op_setting["process_scale"]
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        op_setting.update(
            {'si': setting['in_scale'][0], 'sk': 1.0, 'so': self.get_scale()[0],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize,
             'out_quantize': out_quantize, "ops": ops})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('default')(**op_setting))


# this class definition make user define layer working with framework
@LAYER.register_module(name='userdefine')
class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.in_scale, self.scale = 1.0, 1.0
        self.__extend = True

    @abstractmethod
    def get_scale(self):
        return self.scale

    # @abstractmethod
    def get_w_scale(self):
        return dict(scale=1.0, zero_point=0)

    @abstractmethod
    def set_in_scale(self, scale):
        self.in_scale = scale

    @abstractmethod
    def get_out_data(self):
        pass

    # @abstractmethod
    def set_out_data(self, out_data):
        pass

    @abstractmethod
    def get_onnx_input_name(self):
        pass

    @abstractmethod
    def get_onnx_output_name(self):
        pass

    # @abstractmethod
    def quan_weights(self):
        pass

    # @abstractmethod
    def quan_feat(self):
        pass

    @abstractmethod
    def get_quantize(self):
        pass

    # fp forward need de-quantize output
    @abstractmethod
    def get_dequan_output(self):
        pass

    @abstractmethod
    def checkerror(self):
        pass

    @abstractmethod
    def export(self):
        pass
        # return s_weight

    @abstractmethod
    def forward(self, in_data, **kwargs):
        pass


@LAYER.register_module(name='data')
class Data(SingleInputLayer):
    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        return nodes, initializers

    def export_onnx(self):
        layer_name = self.get_layer_name()
        ori_in_names = copy.deepcopy(self.get_onnx_output_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in ori_in_names]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        nodes, initializers, _ = \
            self.insert_quant(self, layer_name, ori_in_names[0], in_names[0], min_value, max_value,
                              self.get_scale()[0]['scale'], self.get_scale()[0]['zero_point'])
        return nodes, initializers

    def setting_ops(self, setting: dict):
        # todo not consider zero point
        ops = setting['ops_string']
        instances = self.get_ops_instance()
        del instances
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        # todo zero point not consider
        op_setting.update(
            {'si': setting['in_scale'], 'so': setting['scale'], 'out_type': op_setting['out_type'],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': op_setting['process_scale'], 'in_quantize': in_quantize,
             'quantize': out_quantize})
        op = operators_factory.get(ops[0])(**op_setting)
        self.set_ops_instance(op)


@LAYER.register_module(name='conv')
@LAYER.register_module(name='depthwiseconv')
class Conv(WeightedLayer):
    def __init__(self, **kwargs):
        super(Conv, self).__init__(**kwargs)

    # def set_quantize(self, data: dict):
    #     pass

    def export_onnx_fp(self, is_vis_qparams=False):
        attrs = self.get_layer_ops()["attrs"][0]
        conv_attrs = dict(dilations=attrs['dilations'],
                          strides=attrs['strides'],
                          kernel_shape=attrs['kernel_shape'],
                          group=attrs['group'],
                          pads=attrs['pads'])
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            output_shift = self.get_scales()[0]["out_shift"]
            # if self.get_is_result_layer():
            #     output_scale = self.get_scales()[0]["fscale"]
            # else:
            input_zero_point = self.get_scales()[0]["zi"]
            # bits_dict = self.get_ops_setting()['setting']['bits_dict']
            output_shift = onnx.numpy_helper.from_array(np.array([output_shift], dtype=np.int8).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())

            if self.get_scale_type() in ["shiftfloatscaletable", "shiftfloatscaletable2float"]:
                qparam = dict(
                    output_shift=output_shift,
                    input_zero_point=input_zero_point,
                    table=self.get_table(),
                )
            else:
                output_scale = self.get_scales()[0]["out_scale"]
                output_zero_point = self.get_scales()[0]["zo"]
                if self.get_scale_type() in ["floatscale"]:
                    output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
                else:
                    output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.int32).squeeze())
                output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
                qparam = dict(
                    output_shift=output_shift,
                    output_scale=output_scale,
                    input_zero_point=input_zero_point,
                    output_zero_point=output_zero_point,
                )
            conv_attrs.update(qparam)
            weight = self.get_qweight()
            bias = self.get_qbias()
            weight_dtype = weight.dtype
            if weight_dtype == np.int8:
                bias_dtype = np.int32
            else:
                bias_dtype = np.int64
            bias = bias.astype(bias_dtype) # type: ignore
        else:
            weight = self.get_layer_ops()["weights"][0]
            bias = self.get_layer_ops()["weights"][1]
            weight_dtype, bias_dtype = np.float32, np.float32
        nodes, initializers = [], []
        nodes.append(
            self.create_node(
                "Conv",
                inputs=[self.get_nodes()[0].get_onnx_input()[0], 
                self.get_nodes()[0].get_weights()[0]['name'], 
                self.get_nodes()[0].get_weights()[1]['name']],
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=conv_attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )
        if len(self.get_nodes()) == 2:
            if self.get_nodes()[1].get_op_type().lower() in ["relu6"]:
                nodes.append(
                    self.create_node(
                        "Clip",
                        inputs=[
                            self.get_nodes()[1].get_onnx_input()[0],
                            self.get_nodes()[1].get_name() + '/min',
                            self.get_nodes()[1].get_name() + '/max',
                            ],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )
                initializers.append(
                    create_initializer(
                        0,
                        self.get_nodes()[1].get_name() + '/min',
                        dtype=np.float32, # type: ignore
                    )
                )
                initializers.append(
                    create_initializer(
                        6,
                        self.get_nodes()[1].get_name() + '/max',
                        dtype=np.float32, # type: ignore
                    )
                )  
            elif self.get_nodes()[1].get_op_type().lower() in ["leakyrelu"]:   
                nodes.append(
                    self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=[self.get_nodes()[1].get_onnx_input()[0]],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=self.get_nodes()[1].get_attr(),
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )                                
            else:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=[self.get_nodes()[1].get_onnx_input()[0]],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )
        elif len(self.get_nodes()) > 2:
            if self.get_layer_ops()['ops'][-1] == 'swish':
                nodes.append(
                    self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=[self.get_nodes()[1].get_onnx_input()[0]],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                self.create_node(
                        self.get_nodes()[2].get_op_type(),
                        inputs=self.get_nodes()[2].get_onnx_input(),
                        outputs=self.get_nodes()[2].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[2].get_name(),
                        domain=domain,
                    )
                )
            else:
                nodes.append(
                    self.create_node(
                        self.get_layer_ops()['ops'][-1],
                        inputs=[self.get_nodes()[-1].get_onnx_input()[0]],
                        outputs=self.get_nodes()[-1].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[-1].get_name(),
                        domain=domain,
                    )
                )  
                
        initializers.extend([
            self.create_w(
                weight,
                self.get_nodes()[0].get_weights()[0]['name'],
                dtype=weight_dtype, # type: ignore
                trans=False,
                ),
        ])
        initializers.append(
            create_initializer(
                bias,
                self.get_nodes()[0].get_weights()[1]['name'],
                dtype=bias_dtype, # type: ignore
                )
        )
        return nodes, initializers

    def export_onnx(self):
        op_type = 'Conv'
        layer_name = self.get_layer_name()
        out_names = self.get_onnx_output_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        attrs = self.get_ops_setting()['attrs'][0]
        process_scale = self.get_ops_setting()['setting']['process_scale']
        if "auto_pad" in attrs.keys():
            conv_attrs = dict(dilations=attrs['dilations'],
                            strides=attrs['strides'],
                            kernel_shape=attrs['kernel_shape'],
                            group=attrs['group'],
                            auto_pad=attrs['auto_pad'],
                        )
        else:
            conv_attrs = dict(dilations=attrs['dilations'],
                            strides=attrs['strides'],
                            kernel_shape=attrs['kernel_shape'],
                            group=attrs['group'],
                            pads=attrs['pads'],
                        )
        if process_scale == "shiftfloatscaletable":
            process_scale_ = "intscale"
        else:
            process_scale_ = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale_)(op_type, layer_name, in_names, out_names, conv_attrs)
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        conv_quantize = self.get_quantize()['feat'].get('sc0')
        is_first_conv = self.get_first_conv()

        instances = list()
        for idx, op in enumerate(ops):
            op_setting = copy.deepcopy(base_setting)
            op_setting.update(attrs[idx])
            process_scale = op_setting['process_scale']  # if not op in ['conv', 'bias'] else 'smooth'

            # todo zero point not consider
            op_setting.update(
                {'si': setting['in_scale'], 'sk': self.get_w_scale(), 'so': setting['scale'],
                 'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
                 'process_scale': process_scale, 'in_quantize': in_quantize, 'quantize': out_quantize,
                 'conv_quantize': conv_quantize,
                 'p_weights': self.get_qweight(), 'bias': self.get_qbias(), 'isolated': self.isolated,
                 'out_type': op_setting['out_type'], 'is_first_conv': is_first_conv,
                 },
                )
            instances.append(operators_factory.get(op)(**op_setting)) # type: ignore
            del op_setting
        self.set_qbias(instances[-2].bias_)
        self.set_ops_instance(instances)

@LAYER.register_module(name='convtranspose')
class ConvTranspose2d(WeightedLayer):
    def __init__(self, **kwargs):
        super(ConvTranspose2d, self).__init__(**kwargs)

    # def set_quantize(self, data: dict):
    #     pass

    def export_onnx_fp(self, is_vis_qparams=False):
        attrs = self.get_layer_ops()["attrs"][0]
        conv_attrs = dict(dilations=attrs['dilations'],
                          strides=attrs['strides'],
                          kernel_shape=attrs['kernel_shape'],
                          group=attrs['group'],
                          pads=attrs['pads'])
        if "output_padding" in attrs.keys(): conv_attrs.update(output_padding=attrs['output_padding'])
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            output_shift = self.get_scales()[0]["out_shift"]
            # if self.get_is_result_layer():
            #     output_scale = self.get_scales()[0]["fscale"]
            # else:
            input_zero_point = self.get_scales()[0]["zi"]
            # bits_dict = self.get_ops_setting()['setting']['bits_dict']
            output_shift = onnx.numpy_helper.from_array(np.array([output_shift], dtype=np.int8).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())

            if self.get_scale_type() in ["shiftfloatscaletable", "shiftfloatscaletable2float"]:
                qparam = dict(
                    output_shift=output_shift,
                    input_zero_point=input_zero_point,
                    table=self.get_table(),
                )
            else:
                output_scale = self.get_scales()[0]["out_scale"]
                output_zero_point = self.get_scales()[0]["zo"]
                if self.get_scale_type() in ["floatscale"]:
                    output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
                else:
                    output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.int32).squeeze())
                output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
                qparam = dict(
                    output_shift=output_shift,
                    output_scale=output_scale,
                    input_zero_point=input_zero_point,
                    output_zero_point=output_zero_point,
                )
            conv_attrs.update(qparam)
            weight = self.get_qweight()
            bias = self.get_qbias()
            weight_dtype = weight.dtype # type: ignore
            if weight_dtype == np.int8:
                bias_dtype = np.int32
            else:
                bias_dtype = np.int64
            bias = bias.astype(bias_dtype) # type: ignore
        else:
            weight = self.get_layer_ops()["weights"][0]
            bias = self.get_layer_ops()["weights"][1]
            weight_dtype, bias_dtype = np.float32, np.float32
        nodes, initializers = [], []
        nodes.append(
            self.create_node(
                "ConvTranspose",
                inputs=[self.get_nodes()[0].get_onnx_input()[0],
                self.get_nodes()[0].get_weights()[0]['name'],
                self.get_nodes()[0].get_weights()[1]['name']],
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=conv_attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )
        if len(self.get_nodes()) == 2:
            nodes.append(
                self.create_node(
                    self.get_nodes()[1].get_op_type(),
                    inputs=[self.get_nodes()[1].get_onnx_input()[0]],
                    outputs=self.get_nodes()[1].get_onnx_output(),
                    attrs=dict(),
                    node_name=self.get_nodes()[1].get_name(),
                    domain=domain,
                )
            )
        elif len(self.get_nodes()) > 2:
            if self.get_nodes()[1].get_op_type() in ["BatchNormalization"]:
                inputs_act = [self.get_nodes()[1].get_onnx_input()[0]]
            else:
                inputs_act = [self.get_nodes()[-1].get_onnx_input()[0]]
            nodes.append(
                self.create_node(
                    self.get_ops_setting()["ops_string"][-1],
                    inputs=inputs_act,
                    outputs=self.get_nodes()[-1].get_onnx_output(),
                    attrs=dict(),
                    node_name=self.get_nodes()[-1].get_name(),
                    domain=domain,
                )
            )

        initializers.extend([
            self.create_w(
                weight,
                self.get_nodes()[0].get_weights()[0]['name'],
                dtype=weight_dtype,
                trans=False,
                ),
        ])
        initializers.append(
            create_initializer(
                bias,
                self.get_nodes()[0].get_weights()[1]['name'],
                dtype=bias_dtype, # type: ignore
            )
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'ConvTranspose'
        layer_name = self.get_layer_name()
        out_names = self.get_onnx_output_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        attrs = self.get_ops_setting()['attrs'][0]
        process_scale = self.get_ops_setting()['setting']['process_scale']
        conv_attrs = dict(dilations=attrs['dilations'],
                          strides=attrs['strides'],
                          kernel_shape=attrs['kernel_shape'],
                          group=attrs['group'],
                          pads=attrs['pads'])
        if "output_padding" in attrs.keys(): conv_attrs.update(output_padding=attrs['output_padding'])
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, conv_attrs)
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        conv_quantize = self.get_quantize()['feat'].get('sc0')

        instances = list()
        for idx, op in enumerate(ops):
            op_setting = copy.deepcopy(base_setting)
            op_setting.update(attrs[idx])
            process_scale = op_setting['process_scale']  # if not op in ['conv', 'bias'] else 'smooth'

            # todo zero point not consider
            op_setting.update(
                {'si': setting['in_scale'], 'sk': self.get_w_scale(), 'so': setting['scale'],
                 'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
                 'process_scale': process_scale, 'in_quantize': in_quantize, 'quantize': out_quantize,
                 'conv_quantize': conv_quantize,
                 'p_weights': self.get_qweight(), 'bias': self.get_qbias(), 'isolated': self.isolated,
                 'out_type': op_setting['out_type']})
            instances.append(operators_factory.get(op)(**op_setting))
            del op_setting
        self.set_ops_instance(instances)


@LAYER.register_module(name='concat')
class Concat(MultiInputLayer):
    def __init__(self, **kwargs):
        super(Concat, self).__init__(**kwargs)

    # def set_quantize(self, data: dict):
    #     pass

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]

        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']

        f_weights, p_weights, weight_idx, in_quantize = \
            self.reconstruct_inputs(setting, op_setting, si, in_quantize)
        input_len = len(setting['in_scale'])
        if isinstance(weight_idx, list):
            input_len += len(weight_idx)
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so, 'axis': attrs[0]['axis'],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': input_len, 'in_quantize': in_quantize,
             'quantize': out_quantize, 'p_weights': p_weights, 'weight_idx': weight_idx,
             'f_weights': f_weights,'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('concat')(**op_setting))

    def export_onnx_fp(self, is_vis_qparams=False):
        attrs = self.get_layer_ops()["attrs"][0]
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scales, input_shifts = [], []
            input_zero_points, output_zero_points = [], []
            for scale in self.get_scales():
                input_scales.append(scale["out_scale"])
                input_shifts.append(scale["int_scale"])
                input_zero_points.append(scale["zi"])
                output_zero_points.append(scale["zo"])
            input_scale = onnx.numpy_helper.from_array(np.array(input_scales, dtype=np.int32).squeeze())
            input_shift = onnx.numpy_helper.from_array(np.array(input_shifts, dtype=np.int8).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array(input_zero_points, dtype=np.int32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array(output_zero_points, dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_shift=input_shift,
                input_zero_point=input_zero_point,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        nodes, initializers = [], []
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )
        return nodes, initializers

    def export_onnx(self):
        op_type = self.get_layer_type()
        op_type = op_type[0].upper() + op_type[1:]
        layer_name = self.get_layer_name()
        out_names = self.get_onnx_output_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        process_scale = self.get_ops_setting()['setting']['process_scale']
        attrs = self.get_ops_setting()['attrs'][0]
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, attrs)
        return nodes, initializers

    # def forward(self, in_data):
    #     super(Concat, self).forward(in_data)
    #     print()


@LAYER.register_module(name='shuffle')
class Shuffle(MultiInputLayer):
    def __init__(self, **kwargs):
        super(Shuffle, self).__init__(**kwargs)

    def get_outout_nodes(self):
        node_types = [node.get_op_type().lower() for node in self.get_nodes()]
        if 'split' in node_types:
            return self.get_nodes()[-1]
        else:
            return [self.get_nodes()[-1], self.get_nodes()[-2]]

    def get_input_nodes(self):
        return self.get_nodes()[0]

    def get_input_name(self):
        return self.get_nodes()[0].get_input()

    def get_output_name(self):
        node_types = [node.get_op_type().lower() for node in self.get_nodes()]
        if 'split' in node_types:
            return self.get_nodes()[-1].get_output()
        else:
            return flatten_list([self.get_nodes()[-1].get_output(), self.get_nodes()[-2].get_output()])

    def get_onnx_input_name(self):
        return self.get_nodes()[0].get_onnx_input()

    def get_onnx_output_name(self):
        node_types = [node.get_op_type().lower() for node in self.get_nodes()]
        if 'split' in node_types:
            return flatten_list([self.get_nodes()[0].get_onnx_output(), self.get_nodes()[-1].get_onnx_output()])
        else:
            return flatten_list([self.get_nodes()[0].get_onnx_output(), self.get_nodes()[-2].get_onnx_output(),
                                 self.get_nodes()[-1].get_onnx_output()])

    def do_float_scale(self, in_datas: list):
        if isinstance(in_datas, np.ndarray):
            assert isinstance(self.__quantizes, object)  # , print(
            # 'in data is array, output quantize must be single quantize!')
            out_datas = self.__quantizes['feat']['so0'].get_quan_data(in_datas)
        elif isinstance(in_datas, list):
            assert len(in_datas) == len(self.__quantizes.keys())  # , print('must be has the same length')
            out_datas = list()
            for idx, _ in range(len(in_datas)):
                # s01 so2 is output quantize
                # so0 is shuffle layer inner parameter
                out_datas.append(self.__quantizes['feat']['so' + str(idx + 1)].get_quan_data(in_datas[idx]))
        else:
            exit(-1)  # , print('wrong in data type!')

    # def set_quantize(self, data: dict):
    #     pass

    # setting for each operator
    def setting_ops(self, setting: dict):
        # self.get_in_scale_idx()
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        instances = list()
        short = True if 'split' in ops else False
        concat = self.setting_concat(setting, base_setting, attrs[0])
        reshape0 = self.setting_reshape(setting, base_setting, attrs[1])
        transpose = self.setting_transpose(setting, base_setting, attrs[2])
        reshape1 = self.setting_reshape(setting, base_setting, attrs[3])
        if short:
            split = self.setting_split(setting, base_setting, attrs[4])
        else:
            split = self.setting_slice(setting, base_setting, attrs[4:])
        self.set_ops_instance(flatten_list([concat, reshape0, transpose, reshape1, split]))

    def setting_concat(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        # todo not consider zero point
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        op_setting.update(
            {'si': setting['in_scale'], 'sk': default_quant, 'so': self.get_scale()[0],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'in_quantize': in_quantize, 'quantize': out_quantize})
        # del op_setting
        return operators_factory.get('concat')(**op_setting)

    def setting_reshape(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        process_scale = 'smooth'
        op_setting.update(
            {'si': default_quant, 'sk': default_quant, 'so': default_quant,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale})
        # del op_setting
        return operators_factory.get('reshape')(**op_setting)

    def setting_transpose(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        process_scale = 'smooth'
        op_setting.update(
            {'si': default_quant, 'sk': default_quant, 'so': default_quant,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale})
        # del op_setting
        return operators_factory.get('transpose')(**op_setting)

    def setting_split(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        in_quan, out_quan = out_quantize['feat']['so0'], [out_quantize['feat']['so1'], out_quantize['feat']['so2']]
        # todo not consider zero point
        op_setting.update(
            {'si': self.get_scale(), 'sk': default_quant, 'so': self.get_scale()[1:],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quan, 'out_quantize': out_quan,
             'out_type': op_setting['out_type']})
        # del op_setting
        return operators_factory.get('split')(**op_setting)

    def setting_slice(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)

        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        in_quan, out_quan = out_quantize['feat']['so0'], [out_quantize['feat']['so1'], out_quantize['feat']['so2']]
        # todo not consider zero point
        op_setting.update(
            {'si': self.get_scale()[0], 'sk': default_quant, 'so': self.get_scale()[1],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quan, 'quantize': out_quan[0],
             'out_type': op_setting['out_type']})
        op_setting.update(attr[0])
        slice1 = operators_factory.get('slice')(**op_setting)
        op_setting.update(attr[1])
        op_setting.update({'so': self.get_scale()[2], 'quantize': out_quan[1]})
        slice2 = operators_factory.get('slice')(**op_setting)
        del op_setting
        return [slice1, slice2]

    def forward(self, in_data, **kwargs):
        self.set_in_data(in_data)

        outputs = []
        ops_instance = self.get_ops_instance()

        setting = self.get_ops_setting()['setting']
        bits_dict, mins, maxs = setting['bits_dict'], setting['mins'], setting['maxs']
        out_type = setting['out_type'] #self.get_output_type()
        if setting['process_scale'] == "float":
            pass
        else:
            data_type, min_v, max_v = bits_dict[out_type], mins[out_type], maxs[out_type]
        for o_idx, op in enumerate(ops_instance):
            if o_idx > 3:
                break
            in_data = op(in_data, **kwargs)
            outputs.append(in_data)
        if len(ops_instance) > 5:
            outputs.extend([ops_instance[-2](in_data), ops_instance[-1](in_data)])
        else:
            out = ops_instance[-1](in_data, **kwargs)
            outputs.extend(out)
            # output, shifts, out_scales = out['output'], out['out_shifts'], out['out_scales']
            # outputs.append(dict(output=output[-2], out_shifts=shifts[-2], out_scales=out_scales[-2]))
            # outputs.append(dict(output=output[-1], out_shifts=shifts[-1], out_scales=out_scales[-1]))

        if setting['process_scale'] != "float":
            for idx, _ in enumerate(outputs):
                out = clip(outputs[idx]['output'], min_v, max_v)
                outputs[idx]['output'] = out.astype(data_type)

        self.set_out_data([outputs[0], outputs[-2], outputs[-1]])

    def export_float(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, clip_outs = [], []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        initializers = [
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
        ]
        in_scale = self.get_in_scale()
        for idx in range(len(in_names)):
            zero_point = layer_name + 'zi_concat' + str(idx)
            sub_out, mul_scale = layer_name + 'sub_concat' + str(idx), layer_name + 'out_scale_concat' + str(idx)
            mul_1_out, clip_out = layer_name + 'mul_1_concat' + str(idx), layer_name + 'clip_concat_' + str(idx)
            mul_2_out, ceil_out = layer_name + 'mul_2_concat' + str(idx), layer_name + 'ceil_concat_' + str(idx)
            in_name = copy.deepcopy(in_names[idx])
            zi = in_scale[idx]['zero_point']
            if zi != 0:
                nodes.append(self.create_node(
                    'Sub',
                    inputs=[in_name, zero_point],
                    outputs=[sub_out])
                )
                in_name = sub_out
            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[in_name, mul_scale],
                    outputs=[mul_1_out]
                )
            ])
            initializers.extend([create_initializer(zi, zero_point),
                                 create_initializer(self.get_in_scale()[idx]['scale'], mul_scale)])

            clip_outs.append(ceil_out)

        nodes.extend([
            self.create_node(
                'Concat',
                inputs=clip_outs,
                outputs=[self.get_onnx_output_name()[0]],
                attrs={'axis': attrs['concat']}
            )
        ])

        nodes.extend([
            self.create_node(
                'Reshape',
                inputs=[self.get_onnx_output_name()[0], layer_name + 'shape_1'],
                outputs=[layer_name + 'reshape_1']),
            self.create_node(
                'Transpose',
                inputs=[layer_name + 'reshape_1'],
                outputs=[layer_name + 'transpose_1'],
                attrs={'perm': attrs['perm']}
            ),
            self.create_node(
                'Reshape',
                inputs=[layer_name + 'transpose_1', layer_name + 'shape_2'],
                outputs=[layer_name + 'reshape_2']
            )
        ])
        initializers.extend([
            create_initializer(attrs['shape_1'], layer_name + 'shape_1', dtype=np.int64),
            create_initializer(attrs['shape_2'], layer_name + 'shape_2', dtype=np.int64)
        ])

        split_names = [layer_name + 'split' + str(idx) for idx in range(len(out_names[1:]))]
        split_name = layer_name + 'reshape_2'
        nodes.append(self.create_node(
            'Split',
            inputs=[split_name],
            outputs=split_names,
            attrs={'axis': attrs['split']}
        ))

        for idx in range(len(out_names[1:])):
            split_data, zero_point = split_names[idx], layer_name + 'zo_split' + str(idx)
            mul_scale = layer_name + 'out_scale_split' + str(idx)
            add_name, mul_int_scale = layer_name + 'add_zo_split' + str(idx), layer_name + 'int_scale_split' + str(idx)
            mul_1_out, clip_out = layer_name + 'mul_1_split' + str(idx), layer_name + 'clip_split' + str(idx)
            mul_2_out, ceil_out = layer_name + 'mul_2_split' + str(idx), layer_name + 'ceil_split' + str(idx)

            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[split_data, mul_scale],
                    outputs=[mul_1_out]
                ),
                self.create_node(
                    'Round',
                    inputs=[mul_2_out],
                    outputs=[ceil_out]
                )
            ])
            tmp_name = ceil_out
            if self.get_scale()[1 + idx]['zero_point'] != 0:
                nodes.append(
                    self.create_node(
                        'Add',
                        inputs=[ceil_out, zero_point],
                        outputs=[add_name])
                )
                initializers.append(create_initializer(self.get_scale()[1 + idx]['zero_point'], zero_point))
                tmp_name = add_name
            nodes.extend([
                self.create_node(
                    'Clip',
                    inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                    outputs=[out_names[1 + idx]]
                )
            ])
            idx_scale = np.float32(self.get_scale()[1 + idx]['scale'])
            initializers.append(create_initializer(idx_scale, mul_scale))

        return nodes, initializers

    def export_preintscale(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, clip_outs = [], []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        initializers = [
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
        ]
        in_scale = self.get_in_scale()
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        for idx in range(len(in_names)):
            zero_point = layer_name + 'zi_concat' + str(idx)
            sub_out, mul_scale = layer_name + 'sub_concat' + str(idx), layer_name + 'out_scale_concat' + str(idx)
            mul_int_scale = layer_name + 'int_scale_concat' + str(idx)
            mul_1_out, clip_out = layer_name + 'mul_1_concat' + str(idx), layer_name + 'clip_concat_' + str(idx)
            mul_2_out, ceil_out = layer_name + 'mul_2_concat' + str(idx), layer_name + 'ceil_concat_' + str(idx)
            in_name = copy.deepcopy(in_names[idx])
            zi = in_scale[idx]['zero_point']
            if zi != 0:
                nodes.append(self.create_node(
                    'Sub',
                    inputs=[in_name, zero_point],
                    outputs=[sub_out])
                )
                in_name = sub_out
            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[in_name, mul_scale],
                    outputs=[mul_1_out]
                ),
                self.create_node(
                    'Mul',
                    inputs=[mul_1_out, mul_int_scale],
                    outputs=[mul_2_out]
                ),
                self.create_node(
                    virtual_op_type,
                    inputs=[mul_2_out],
                    outputs=[ceil_out]
                )
            ])

            initializers.extend([create_initializer(zi, zero_point),
                                 create_initializer(self.get_scales()[0][idx]['out_scale'], mul_scale),
                                 create_initializer(1 / (2 ** self.get_scales()[0][idx]['int_scale']) * self.eps, mul_int_scale)])

            clip_outs.append(ceil_out)
        zo_concat = self.get_scale()[0]['zero_point']
        nodes.extend([
            self.create_node(
                'Concat',
                inputs=clip_outs,
                outputs=[self.get_onnx_output_name()[0]],
                attrs={'axis': attrs['concat']}
            )
        ])
        tmp_name = self.get_onnx_output_name()[0]
        if zo_concat != 0:
            nodes.append(self.create_node(
                'Add',
                inputs=[tmp_name, layer_name + 'zo_concat'],
                outputs=[layer_name + 'add_out']
            ))
            tmp_name = layer_name + 'add_out'
        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[layer_name + 'clip_output']
            )
        )

        initializers.append(create_initializer(zo_concat, layer_name + 'zo_concat'))
        nodes.extend([
            self.create_node(
                'Reshape',
                inputs=[layer_name + 'clip_output', layer_name + 'shape_1'],
                outputs=[layer_name + 'reshape_1']),
            self.create_node(
                'Transpose',
                inputs=[layer_name + 'reshape_1'],
                outputs=[layer_name + 'transpose_1'],
                attrs={'perm': attrs['perm']}
            ),
            self.create_node(
                'Reshape',
                inputs=[layer_name + 'transpose_1', layer_name + 'shape_2'],
                outputs=[layer_name + 'reshape_2']
            )
        ])
        initializers.extend([
            create_initializer(attrs['shape_1'], layer_name + 'shape_1', dtype=np.int64),
            create_initializer(attrs['shape_2'], layer_name + 'shape_2', dtype=np.int64)
        ])

        zi_split = self.get_scale()[0]['zero_point']

        split_names = [layer_name + 'split' + str(idx) for idx in range(len(out_names[1:]))]
        split_name = layer_name + 'reshape_2'
        if zi_split != 0:
            nodes.append(self.create_node(
                'Sub',
                inputs=[layer_name + 'reshape_2', layer_name + 'zi_split'],
                outputs=[layer_name + 'sub_split_out']
            ))
            initializers.append(create_initializer(zi_split, layer_name + 'zi_split'))
            split_name = layer_name + 'sub_split_out'
        nodes.append(self.create_node(
            'Split',
            inputs=[split_name, layer_name + '_shuffle_split'],
            outputs=split_names,
            attrs={'axis': attrs['split']['axis']}
        ))
        initializers.append(
            create_initializer(attrs['split']['split_num'], layer_name + '_shuffle_split', dtype=np.int64))

        for idx in range(len(out_names[1:])):
            split_data, zero_point = split_names[idx], layer_name + 'zo_split' + str(idx)
            mul_scale = layer_name + 'out_scale_split' + str(idx)
            add_name, mul_int_scale = layer_name + 'add_zo_split' + str(idx), layer_name + 'int_scale_split' + str(idx)
            mul_1_out, clip_out = layer_name + 'mul_1_split' + str(idx), layer_name + 'clip_split' + str(idx)
            mul_2_out, ceil_out = layer_name + 'mul_2_split' + str(idx), layer_name + 'ceil_split' + str(idx)

            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[split_data, mul_scale],
                    outputs=[mul_1_out]
                ),
                self.create_node(
                    'Mul',
                    inputs=[mul_1_out, mul_int_scale],
                    outputs=[mul_2_out]
                ),
                self.create_node(
                    virtual_op_type,
                    inputs=[mul_2_out],
                    outputs=[ceil_out]
                )
            ])
            tmp_name = ceil_out
            if self.get_scale()[1 + idx]['zero_point'] != 0:
                nodes.append(
                    self.create_node(
                        'Add',
                        inputs=[ceil_out, zero_point],
                        outputs=[add_name])
                )
                initializers.append(create_initializer(self.get_scale()[1 + idx]['zero_point'], zero_point))
                tmp_name = add_name
            nodes.extend([
                self.create_node(
                    'Clip',
                    inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                    outputs=[out_names[1 + idx]]
                )
            ])
            initializers.extend([create_initializer(self.get_scales()[-1][idx]['out_scale'], mul_scale),
                                 create_initializer(1 / (2 ** self.get_scales()[-1][idx]['int_scale']) * self.eps, mul_int_scale)
                                 ])

        return nodes, initializers

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            assert len(self.get_nodes()) == 5
            attrs = dict(
                axis1=self.get_nodes()[0].get_attr()["axis"],
                shape1=self.get_nodes()[1].get_attr()["shape"],
                perm=self.get_nodes()[2].get_attr()["perm"],
                shape2=self.get_nodes()[3].get_attr()["shape"],
                axis2=self.get_nodes()[4].get_attr()["axis"],
                ops=["Concat", "Reshape", "Transpose", "Reshape", "Split"],
            )
            input_scales, input_shifts = [], []
            output_scales, output_shifts = [], []
            input_zero_points, output_zero_points = [], []
            for scale in self.get_scales()[0]:
                input_scales.append(scale["out_scale"])
                input_shifts.append(scale["int_scale"])
                input_zero_points.append(scale["zi"])
            for scale in self.get_scales()[-1]:
                output_scales.append(scale["out_scale"])
                output_shifts.append(scale["int_scale"])
                output_zero_points.append(scale["zo"])
            input_scale = onnx.numpy_helper.from_array(np.array(input_scales, dtype=np.int32).squeeze())
            input_shift = onnx.numpy_helper.from_array(np.array(input_shifts, dtype=np.int8).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array(output_scales, dtype=np.int32).squeeze())
            output_shift = onnx.numpy_helper.from_array(np.array(output_shifts, dtype=np.int8).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array(input_zero_points, dtype=np.int32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array(output_zero_points, dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_shift=input_shift,
                output_scale=output_scale,
                output_shift=output_shift,
                input_zero_point=input_zero_point,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
            nodes.append(
                self.create_node(
                    "Shuffle",
                    inputs=self.get_nodes()[0].get_onnx_input(),
                    outputs=self.get_nodes()[-1].get_onnx_output(),
                    attrs=attrs,
                    node_name=self.get_nodes()[0].get_name(),
                    domain=domain,
                )
            )
        else:
            if len(self.get_nodes()) == 5:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[0].get_op_type(),
                        inputs=self.get_nodes()[0].get_onnx_input(),
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=self.get_nodes()[0].get_attr(),
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                    self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=[
                            self.get_nodes()[1].get_onnx_input()[0],
                            self.get_nodes()[1].get_weights()[0]['name']
                        ],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                    self.create_node(
                        self.get_nodes()[2].get_op_type(),
                        inputs=[self.get_nodes()[2].get_onnx_input()[0]],
                        outputs=self.get_nodes()[2].get_onnx_output(),
                        attrs=self.get_nodes()[2].get_attr(),
                        node_name=self.get_nodes()[2].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                    self.create_node(
                        self.get_nodes()[3].get_op_type(),
                        inputs=[
                            self.get_nodes()[3].get_onnx_input()[0],
                            self.get_nodes()[3].get_weights()[0]['name']
                        ],
                        outputs=self.get_nodes()[3].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[3].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                    self.create_node(
                        self.get_nodes()[4].get_op_type(),
                        inputs=[
                            self.get_nodes()[4].get_onnx_input()[0],
                            self.get_nodes()[4].get_name()
                            ],
                        outputs=self.get_nodes()[4].get_onnx_output(),
                        attrs=dict(axis=self.get_nodes()[4].get_attr()["axis"]),
                        node_name=self.get_nodes()[4].get_name(),
                        domain=domain,
                    )
                )
            else:
                raise Exception("Not implemented yet !!!")

            initializers.extend([
                ### reshape
                create_initializer(
                    self.get_nodes()[1].get_attr()["shape"],
                    self.get_nodes()[1].get_weights()[0]['name'], dtype=np.int64),
                ### reshape
                create_initializer(
                    self.get_nodes()[3].get_attr()["shape"],
                    self.get_nodes()[3].get_weights()[0]['name'], dtype=np.int64),
                ### split
                create_initializer(
                    self.get_nodes()[4].get_attr()["split"],
                    self.get_nodes()[4].get_name(), dtype=np.int64),
            ])

        return nodes, initializers
    
    def export_onnx(self):
        op_type = ''
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs']
        new_attrs = dict(
            concat=attrs[0]['axis'],
            shape_1=attrs[1]['shape'],
            perm=attrs[2]['perm'],
            shape_2=attrs[3]['shape'],
            split=dict(axis=attrs[4]['axis'], split_num=attrs[4]['split'])
        )
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, new_attrs)
        return nodes, initializers

    # def quantize(self):
    #     pass


@LAYER.register_module(name='concat_shuffle_only')
class ConcatShuffleOnly(Shuffle):
    def __init__(self, **kwargs):
        super(ConcatShuffleOnly, self).__init__(**kwargs)

    def get_outout_nodes(self):
        return self.get_nodes()[-1]

    def get_input_nodes(self):
        return self.get_nodes()[0]

    def get_input_name(self):
        return self.get_nodes()[0].get_input()

    def get_output_name(self):
        return self.get_nodes()[-1].get_output()

    def get_onnx_input_name(self):
        return self.get_nodes()[0].get_onnx_input()

    def get_onnx_output_name(self):
        return flatten_list([self.get_nodes()[0].get_onnx_output(), self.get_nodes()[-1].get_onnx_output()])

    def do_float_scale(self, in_datas: list):
        if isinstance(in_datas, np.ndarray):
            assert isinstance(self.__quantizes, object)  # , print(
            # 'in data is array, output quantize must be single quantize!')
            out_datas = self.__quantizes['feat']['so0'].get_quan_data(in_datas)
        elif isinstance(in_datas, list):
            assert len(in_datas) == len(self.__quantizes.keys())  # , print('must be has the same length')
            out_datas = list()
            for idx, _ in range(len(in_datas)):
                # s01 so2 is output quantize
                # so0 is shuffle layer inner parameter
                out_datas.append(self.__quantizes['feat']['so' + str(idx + 1)].get_quan_data(in_datas[idx]))
        else:
            os._exit(-1)  # , print('wrong in data type!')

    # def set_quantize(self, data: dict):
    #     pass

    # setting for each operator
    def setting_ops(self, setting: dict):
        # super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        instances = list()
        concat = self.setting_concat(setting, base_setting, attrs[0])
        reshape0 = self.setting_reshape(setting, base_setting, attrs[1])
        transpose = self.setting_transpose(setting, base_setting, attrs[2])
        reshape1 = self.setting_reshape(setting, base_setting, attrs[3])
        self.set_ops_instance([concat, reshape0, transpose, reshape1])

    def forward(self, in_data, **kwargs):
        outputs = list()
        ops_instance = self.get_ops_instance()

        setting = self.get_ops_setting()['setting']
        bits_dict, mins, maxs = setting['bits_dict'], setting['mins'], setting['maxs']
        out_type = self.get_output_type()
        data_type, min_v, max_v = bits_dict[out_type], mins[out_type], maxs[out_type]
        for o_idx, op in enumerate(ops_instance):
            in_data = op(in_data, **kwargs)
            outputs.append(in_data)

        for idx, _ in enumerate(outputs):
            out = clip(outputs[idx]['output'], min_v, max_v)
            outputs[idx]['output'] = out.astype(bits_dict[self.get_output_type()])
        self.set_out_data([outputs[0], outputs[-1]])

    def export_onnx_fp(self, is_vis_qparams=False):
        pass

    def export_onnx(self):
        pass


@LAYER.register_module(name='shuffle_only')
class ShuffleOnly(SingleInputLayer):
    def __init__(self, **kwargs):
        super(ShuffleOnly, self).__init__(**kwargs)

    # def set_quantize(self, data: dict):
    #     pass

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        # instances = list()
        reshape0 = self.setting_reshape(setting, base_setting, attrs[0])
        transpose = self.setting_transpose(setting, base_setting, attrs[1])
        reshape1 = self.setting_reshape(setting, base_setting, attrs[2])
        self.set_ops_instance(flatten_list([reshape0, transpose, reshape1]))

    def setting_reshape(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        process_scale = 'smooth'
        op_setting.update(
            {'si': default_quant, 'sk': default_quant, 'so': default_quant,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'out_type': op_setting['out_type']})
        # del op_setting
        return operators_factory.get('reshape')(**op_setting)

    def setting_transpose(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        process_scale = 'smooth'
        op_setting.update(
            {'si': default_quant, 'sk': default_quant, 'so': default_quant,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale})
        # del op_setting
        return operators_factory.get('transpose')(**op_setting)

    def forward(self, in_data, **kwargs):
        super().forward(in_data, **kwargs)
        outputs = copy.deepcopy(self.get_out_data())[-1:]
        self.set_out_data(outputs)

    def export_smooth(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, clip_outs = [], []

        nodes.extend([
            self.create_node(
                'Reshape',
                inputs=[in_names[0], layer_name + 'shape_1'],
                outputs=[layer_name + 'reshape_1']),
            self.create_node(
                'Transpose',
                inputs=[layer_name + 'reshape_1'],
                outputs=[layer_name + 'transpose_1'],
                attrs={'perm': attrs['perm']}
            ),
            self.create_node(
                'Reshape',
                inputs=[layer_name + 'transpose_1', layer_name + 'shape_2'],
                outputs=[out_names[0]]
            )
        ])
        initializers = [
            create_initializer(attrs['shape_1'], layer_name + 'shape_1', dtype=np.int64),
            create_initializer(attrs['shape_2'], layer_name + 'shape_2', dtype=np.int64)
        ]

        return nodes, initializers

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            assert len(self.get_nodes()) == 3
            attrs = dict(
                shape1=self.get_nodes()[0].get_attr()["shape"],
                perm=self.get_nodes()[1].get_attr()["perm"],
                shape2=self.get_nodes()[2].get_attr()["shape"],
                ops=["Reshape", "Transpose", "Reshape"],
            )
            nodes.append(
                self.create_node(
                    "ShuffleOnly",
                    inputs=self.get_nodes()[0].get_onnx_input(),
                    outputs=self.get_nodes()[-1].get_onnx_output(),
                    attrs=attrs,
                    node_name=self.get_nodes()[0].get_name(),
                    domain=domain,
                )
            )
        else:
            if len(self.get_nodes()) == 3:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[0].get_op_type(),
                        inputs=[
                            self.get_nodes()[0].get_onnx_input()[0],
                            self.get_nodes()[0].get_weights()[0]['name']
                        ],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                    self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=[self.get_nodes()[1].get_onnx_input()[0]],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=self.get_nodes()[1].get_attr(),
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                    self.create_node(
                        self.get_nodes()[2].get_op_type(),
                        inputs=[
                            self.get_nodes()[2].get_onnx_input()[0],
                            self.get_nodes()[2].get_weights()[0]['name']
                        ],
                        outputs=self.get_nodes()[2].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[2].get_name(),
                        domain=domain,
                    )
                )
            else:
                raise Exception("Not implemented yet !!!")

        initializers.extend([
            ### reshape
            create_initializer(
                self.get_nodes()[0].get_attr()["shape"], 
                self.get_nodes()[0].get_weights()[0]['name'], dtype=np.int64),
            ### reshape
            create_initializer(
                self.get_nodes()[2].get_attr()["shape"], 
                self.get_nodes()[2].get_weights()[0]['name'], dtype=np.int64),                
        ])

        return nodes, initializers

    def export_onnx(self):
        op_type = ''
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs']
        new_attrs = dict(
            shape_1=attrs[0]['shape'],
            perm=attrs[1]['perm'],
            shape_2=attrs[2]['shape']
        )
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_smooth')(op_type, layer_name, in_names, out_names, new_attrs)
        return nodes, initializers


@LAYER.register_module(name='shuffle_only_split')
class ShuffleOnlySplit(SingleInputLayer):
    def __init__(self, **kwargs):
        super(ShuffleOnlySplit, self).__init__(**kwargs)

    def get_outout_nodes(self):
        node_types = [node.get_op_type().lower() for node in self.get_nodes()]
        if 'split' in node_types:
            return self.get_nodes()[-1]
        else:
            return [self.get_nodes()[-1], self.get_nodes()[-2]]

    def get_input_nodes(self):
        return self.get_nodes()[0]

    def get_input_name(self):
        return self.get_nodes()[0].get_input()

    def get_output_name(self):
        node_types = [node.get_op_type().lower() for node in self.get_nodes()]
        if 'split' in node_types:
            return self.get_nodes()[-1].get_output()
        else:
            return flatten_list([self.get_nodes()[-1].get_output(), self.get_nodes()[-2].get_output()])

    def get_onnx_input_name(self):
        return self.get_nodes()[0].get_onnx_input()

    def get_onnx_output_name(self):
        node_types = [node.get_op_type().lower() for node in self.get_nodes()]
        if 'split' in node_types:
            return flatten_list([self.get_nodes()[-1].get_onnx_output()])
        else:
            return flatten_list([self.get_nodes()[-2].get_onnx_output(),
                                 self.get_nodes()[-1].get_onnx_output()])

    def do_float_scale(self, in_datas: list):
        if isinstance(in_datas, np.ndarray):
            assert isinstance(self.__quantizes, object)  # , print(
            # 'in data is array, output quantize must be single quantize!')
            out_datas = self.__quantizes['feat']['so0'].get_quan_data(in_datas)
        elif isinstance(in_datas, list):
            assert len(in_datas) == len(self.__quantizes.keys())  # , print('must be has the same length')
            out_datas = list()
            for idx, _ in range(len(in_datas)):
                # s01 so2 is output quantize
                # so0 is shuffle layer inner parameter
                out_datas.append(self.__quantizes['feat']['so' + str(idx + 1)].get_quan_data(in_datas[idx]))
        else:
            os._exit(-1)  # , print('wrong in data type!')

    # def set_quantize(self, data: dict):
    #     pass

    # setting for each operator
    def setting_ops(self, setting: dict):
        # self.get_in_scale_idx()
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        # instances = list()
        short = True if 'split' in ops else False
        reshape0 = self.setting_reshape(setting, base_setting, attrs[1])
        transpose = self.setting_transpose(setting, base_setting, attrs[2])
        reshape1 = self.setting_reshape(setting, base_setting, attrs[3])
        if short:
            split = self.setting_split(setting, base_setting, attrs[4])
        else:
            split = self.setting_slice(setting, base_setting, attrs[4:])
        self.set_ops_instance(flatten_list([reshape0, transpose, reshape1, split]))

    def setting_reshape(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        process_scale = 'smooth'
        op_setting.update(
            {'si': default_quant, 'sk': default_quant, 'so': default_quant,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale})
        # del op_setting
        return operators_factory.get('reshape')(**op_setting)

    def setting_transpose(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        process_scale = 'smooth'
        op_setting.update(
            {'si': default_quant, 'sk': default_quant, 'so': default_quant,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale})
        # del op_setting
        return operators_factory.get('transpose')(**op_setting)

    def setting_split(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attr)
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        in_quan, out_quan = out_quantize['feat']['so0'], [out_quantize['feat']['so1'], out_quantize['feat']['so2']]
        op_setting.update(
            {'si': self.get_scale(), 'sk': default_quant, 'so': self.get_scale()[1:],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quan, 'out_quantize': out_quan})
        # del op_setting
        return operators_factory.get('split')(**op_setting)

    def setting_slice(self, setting, base_setting, attr):
        op_setting = copy.deepcopy(base_setting)

        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        in_quan, out_quan = out_quantize['feat']['so0'], [out_quantize['feat']['so1'], out_quantize['feat']['so2']]
        op_setting.update(
            {'si': self.get_scale()[0], 'sk': default_quant, 'so': self.get_scale()[1],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quan, 'quantize': out_quan[0]})
        op_setting.update(attr[0])
        slice1 = operators_factory.get('slice')(**op_setting)
        op_setting.update(attr[1])
        op_setting.update({'so': self.get_scale()[2], 'quantize': out_quan[1]})
        slice2 = operators_factory.get('slice')(**op_setting)
        del op_setting
        return [slice1, slice2]

    def forward(self, in_data, **kwargs):
        outputs = list()
        ops_instance = self.get_ops_instance()

        setting = self.get_ops_setting()['setting']
        bits_dict, mins, maxs = setting['bits_dict'], setting['mins'], setting['maxs']
        out_type = self.get_output_type()
        data_type, min_v, max_v = bits_dict[out_type], mins[out_type], maxs[out_type]
        for o_idx, op in enumerate(ops_instance):
            if o_idx > 2:
                break
            in_data = op(in_data, **kwargs)
            outputs.append(in_data)
        if len(ops_instance) > 4:
            outputs.extend([ops_instance[-2](in_data), ops_instance[-1](in_data)])
        else:
            out = ops_instance[-1](in_data, **kwargs)
            outputs.extend(out)
            # output, shifts, out_scales = out['output'], out['out_shifts'], out['out_scales']
            # outputs.append(dict(output=output[-2], out_shifts=shifts[-2], out_scales=out_scales[-2]))
            # outputs.append(dict(output=output[-1], out_shifts=shifts[-1], out_scales=out_scales[-1]))

        for idx, _ in enumerate(outputs):
            out = clip(outputs[idx]['output'], min_v, max_v)
            outputs[idx]['output'] = out.astype(bits_dict[self.get_output_type()])
        self.set_out_data([outputs[-2], outputs[-1]])

    def export_onnx_fp(self, is_vis_qparams=False):
        pass

    def export_onnx(self):
        pass


@LAYER.register_module(name='add')
@LAYER.register_module(name='cadd')
class Add(MultiInputLayer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()
        process_scale = op_setting['process_scale']
        in_quantize = self.get_in_quantize()
        out_quantize = self.get_quantize()['feat']['so0']
        f_weights, p_weights, weight_idx, in_quantize = \
            self.reconstruct_inputs(setting, op_setting, si, in_quantize)
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so, 'bit_select': op_setting['bit_select'],
             'int_scale': op_setting['int_scale'], 'process_scale': process_scale,
             'in_quantize': in_quantize, 'quantize': out_quantize, 'p_weights': p_weights,
             'weight_idx': weight_idx, 'f_weights': f_weights, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get(ops[0])(**op_setting))

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = dict()
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            out_scales, int_scales = [], []
            input_zero_points, output_zero_points = [], []
            for scale in self.get_scales():
                out_scales.append(scale["out_scale"])
                int_scales.append(scale["int_scale"])
                input_zero_points.append(scale["zi"])
                output_zero_points.append(scale["zo"])
            input_scale = onnx.numpy_helper.from_array(np.array(out_scales, dtype=np.int32).squeeze())
            input_shift = onnx.numpy_helper.from_array(np.array(int_scales, dtype=np.int8).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array(input_zero_points, dtype=np.int32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array(output_zero_points, dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_shift=input_shift,
                input_zero_point=input_zero_point,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )
        return nodes, initializers


@LAYER.register_module(name='sub')
@LAYER.register_module(name='csub')
class Sub(Add):
    def __init__(self, **kwargs):
        super(Sub, self).__init__(**kwargs)


@LAYER.register_module(name='fc')
@LAYER.register_module(name='gemm')
class FC(WeightedLayer):
    def __init__(self, **kwargs):
        super(FC, self).__init__(**kwargs)

    def export_onnx_fp(self, is_vis_qparams=False):
        op_type = 'Gemm'
        trans = False
        vaild_keys = ['alpha', 'beta', 'transB']
        attrs = self.get_layer_ops()["attrs"][0]
        # if "transB" in attrs.keys():
            # trans = attrs["transB"]
            
        new_attrs = {}
        for key in vaild_keys:
            if key in attrs.keys():
                new_attrs[key] = attrs[key]

        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            output_shift = self.get_scales()[0]["out_shift"]
            # if self.get_is_result_layer():
            #     output_scale = self.get_scales()[0]["fscale"]
            # else:
            input_zero_point = self.get_scales()[0]["zi"]
            # bits_dict = self.get_ops_setting()['setting']['bits_dict']
            output_shift = onnx.numpy_helper.from_array(np.array([output_shift], dtype=np.int8).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())

            if self.get_scale_type() in ["shiftfloatscaletable", "shiftfloatscaletable2float"]:
                qparam = dict(
                    output_shift=output_shift,
                    input_zero_point=input_zero_point,
                    table=self.get_table(),
                )
            else:
                output_scale = self.get_scales()[0]["out_scale"]
                output_zero_point = self.get_scales()[0]["zo"]
                if self.get_scale_type() in ["floatscale"]:
                    output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
                else:
                    output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.int32).squeeze())
                output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
                qparam = dict(
                    output_shift=output_shift,
                    output_scale=output_scale,
                    input_zero_point=input_zero_point,
                    output_zero_point=output_zero_point,
                )
            new_attrs.update(qparam)
            weight = self.get_qweight()
            bias = self.get_qbias()
            weight_dtype = weight.dtype # type: ignore
            if weight_dtype == np.int8:
                bias_dtype = np.int32
            else:
                bias_dtype = np.int64
            bias = bias.astype(bias_dtype) # type: ignore
        else:
            weight = self.get_layer_ops()["weights"][0]
            bias = self.get_layer_ops()["weights"][1]
            weight_dtype, bias_dtype = np.float32, np.float32

        nodes, initializers = [], []
        node_types = [node.get_op_type().lower() for node in self.get_nodes()]
        if node_types in [
            ['batchnormalization', 'gemm', 'relu'],
            ['batchnormalization', 'matmul', 'relu']
            ]:
            nodes.append(
                self.create_node(
                    "Gemm",
                    inputs=[self.get_nodes()[0].get_onnx_input()[0], 
                    self.get_nodes()[1].get_weights()[0]['name'], 
                    self.get_nodes()[1].get_weights()[1]['name']
                    ],
                    outputs=self.get_nodes()[1].get_onnx_output(),
                    attrs=new_attrs,
                    node_name=self.get_nodes()[1].get_name(),
                    domain=domain,
                )
            )
            nodes.append(
                self.create_node(
                    self.get_nodes()[2].get_op_type(),
                    inputs=self.get_nodes()[2].get_onnx_input(),
                    outputs=self.get_nodes()[2].get_onnx_output(),
                    attrs=dict(),
                    node_name=self.get_nodes()[2].get_name(),
                    domain=domain,
                )
            )  
            initializers.extend([
                self.create_w(
                    self.get_layer_ops()["weights"][0], 
                    self.get_nodes()[1].get_weights()[0]['name'], trans=trans
                    ),
            ])
            initializers.append(
                create_initializer(self.get_layer_ops()["weights"][1], self.get_nodes()[1].get_weights()[1]['name'])
            )
        elif node_types in [
            ['batchnormalization', 'gemm'],
            ['batchnormalization', 'matmul']
            ]:   
            nodes.append(
                self.create_node(
                    "Gemm",
                    inputs=[self.get_nodes()[0].get_onnx_input()[0], 
                    self.get_nodes()[1].get_weights()[0]['name'], 
                    self.get_nodes()[1].get_weights()[1]['name']
                    ],
                    outputs=self.get_nodes()[1].get_onnx_output(),
                    attrs=new_attrs,
                    node_name=self.get_nodes()[1].get_name(),
                    domain=domain,
                )
            )
            initializers.extend([
                self.create_w(
                    self.get_layer_ops()["weights"][0], 
                    self.get_nodes()[1].get_weights()[0]['name'], trans=trans
                    ),
            ])
            initializers.append(
                create_initializer(self.get_layer_ops()["weights"][1], self.get_nodes()[1].get_weights()[1]['name'])
            )                  
        else:                      
            if node_types in [
                ['gemm', 'batchnormalization', 'relu'],
                ['matmul', 'batchnormalization', 'relu']
                ]:
                nodes.append(
                    self.create_node(
                        "Gemm",
                        inputs=[self.get_nodes()[0].get_onnx_input()[0], 
                        self.get_nodes()[0].get_weights()[0]['name'], 
                        self.get_nodes()[0].get_weights()[1]['name']
                        ],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=new_attrs,
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )    
                nodes.append(
                    self.create_node(
                        self.get_nodes()[2].get_op_type(),
                        inputs=self.get_nodes()[2].get_onnx_input(),
                        outputs=self.get_nodes()[2].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[2].get_name(),
                        domain=domain,
                    )
                )                      
            elif node_types in [
                ['gemm', 'batchnormalization'], 
                ['matmul', 'batchnormalization']
                ]:
                nodes.append(
                    self.create_node(
                        "Gemm",
                        inputs=[self.get_nodes()[0].get_onnx_input()[0], 
                        self.get_nodes()[0].get_weights()[0]['name'], 
                        self.get_nodes()[0].get_weights()[1]['name']
                        ],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=new_attrs,
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )                            
            elif node_types in [
                ['gemm', 'relu'], 
                ['matmul', 'relu']]:
                nodes.append(
                    self.create_node(
                        "Gemm",
                        inputs=[self.get_nodes()[0].get_onnx_input()[0], 
                        self.get_nodes()[0].get_weights()[0]['name'], 
                        self.get_nodes()[0].get_weights()[1]['name']
                        ],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=new_attrs,
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                    self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=[self.get_nodes()[1].get_onnx_input()[0]],
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )            
            else:
                nodes.append(
                    self.create_node(
                        "Gemm",
                        inputs=[self.get_nodes()[0].get_onnx_input()[0], 
                        self.get_nodes()[0].get_weights()[0]['name'], 
                        self.get_nodes()[0].get_weights()[1]['name']
                        ],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=new_attrs,
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
                if len(self.get_nodes()) == 2:
                    nodes.append(
                        self.create_node(
                            self.get_nodes()[1].get_op_type(),
                            inputs=[self.get_nodes()[1].get_onnx_input()[0]],
                            outputs=self.get_nodes()[1].get_onnx_output(),
                            attrs=dict(),
                            node_name=self.get_nodes()[1].get_name(),
                            domain=domain,
                        )
                    )

            initializers.extend([
                self.create_w(
                    weight,
                    self.get_nodes()[0].get_weights()[0]['name'],
                    dtype=weight_dtype, # type: ignore
                    trans=trans,
                ),
            ])
            initializers.append(create_initializer(
                    bias,
                    self.get_nodes()[0].get_weights()[1]['name'],
                    dtype=bias_dtype, # type: ignore
                )
            )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'Gemm'
        trans = False
        vaild_keys = ['alpha', 'beta', 'transB']
        attrs = self.get_ops_setting()['attrs'][0]
        new_attrs = {}
        for key in vaild_keys:
            if key in attrs.keys():
                new_attrs[key] = attrs[key]
        if not isinstance(self.get_qbias(), np.ndarray):
            op_type = 'MatMul'
            trans = True
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        if process_scale == "shiftfloatscaletable":
            process_scale_ = "intscale"
        else:
            process_scale_ = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale_)(op_type, layer_name, in_names, out_names, new_attrs, trans)
        return nodes, initializers

    # def forward(self, in_data):
    #     super(FC, self).forward(in_data)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        conv_quantize = self.get_quantize()['feat'].get('sc0')
        new_ops = copy.deepcopy(ops)
        new_attrs = copy.deepcopy(attrs)
        if len(new_ops) > 3:
            if new_ops[-2] == "reshape":
                new_ops[-2], new_attrs[-2] = ops[-1], attrs[-1]
                new_ops[-1], new_attrs[-1] = ops[-2], attrs[-2]
                ops_setting = self.get_ops_setting()
                ops_setting["ops_string"] = new_ops
                self.set_ops_setting(ops_setting)
                layer_ops = self.get_layer_ops()
                layer_ops["ops"] = new_ops
                self.set_layer_ops(layer_ops)
                
        instances = list()
        for idx, op in enumerate(new_ops):
            op_setting = copy.deepcopy(base_setting)
            op_setting.update(new_attrs[idx])
            process_scale = op_setting['process_scale']  # if not op in ['fc', 'matmul', 'gemm', 'bias'] else 'smooth'
            
            # todo zero point not consider
            op_setting.update(
                {'si': setting['in_scale'], 'sk': self.get_w_scale(), 'so': setting['scale'],
                 'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
                 'process_scale': process_scale, 'in_quantize': in_quantize, 'quantize': out_quantize,
                 'conv_quantize': conv_quantize, 'p_weights': self.get_qweight(), 'bias': self.get_qbias(),
                 'isolated': self.isolated})
            instances.append(operators_factory.get(op)(**op_setting))
        if len(new_ops) > 3:
            self.set_qbias(instances[-3].bias_) 
        else: 
            self.set_qbias(instances[-2].bias_)
        self.set_ops_instance(instances)
        
    # def forward(self, in_data, **kwargs):
    #     for op in self.get_ops_instance():
    #         op.in_quantize[0].get_quan_param(in_data[0]["output"])
    #         op.update_datacorrect(si=[op.in_quantize[0].get_quant_param()])
    #         if hasattr(op, "update_qbias"):
    #             op.update_qbias(self.get_layer_ops()['weights'][1])
    #     return super().forward(in_data, **kwargs)


@LAYER.register_module(name='batchnormalization')
class BatchNormalization(NormLayer):
    def __init__(self, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)

    def export_onnx_fp(self, is_vis_qparams=False):
        attrs = copy.deepcopy(self.get_nodes()[0].get_attr())
        epsilon = attrs["epsilon"]
        attrs = dict(epsilon=epsilon)
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        nodes, initializers = [], []
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=[self.get_nodes()[0].get_onnx_input()[0], 
                        self.get_layer_name() + "_scale",
                        self.get_layer_name() + "_bias",
                        self.get_layer_name() + "_input_mean",
                        self.get_layer_name() + "_input_var"],
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )
        initializers.append(create_initializer(self.get_layer_ops()["weights"][0], self.get_layer_name() + "_scale"))
        initializers.append(create_initializer(self.get_layer_ops()["weights"][1], self.get_layer_name() + "_bias"))
        initializers.append(create_initializer(self.get_layer_ops()["weights"][2], self.get_layer_name() + "_input_mean"))
        initializers.append(create_initializer(self.get_layer_ops()["weights"][3], self.get_layer_name() + "_input_var"))

        return nodes, initializers

    def export_onnx(self):
        op_type = 'BatchNormalization'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        bn_attrs = dict(
            s=attrs['scale'],
            bias=attrs['bias'],
            mean=attrs['mean'],
            var=attrs['var']
        )
        setting = dict(epsilon=attrs['epsilon'], momentum=attrs['momentum'], training_mode=attrs['training_mode'])
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, setting, bn_attrs)
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        si, so = setting['in_scale'], self.get_scale()[0]
        assert base_setting['process_scale'] == 'float'
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': base_setting['process_scale'], 'in_quantize': in_quantize[0],
             'quantize': out_quantize, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('batchnormalization')(**op_setting))


@LAYER.register_module(name='transpose')
class Transpose(ShapeLayer):
    def __init__(self, **kwargs):
        super(Transpose, self).__init__(**kwargs)

    def export_onnx(self):
        op_type = 'Transpose'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        # new_attrs = dict(
        #     perm=attrs['perm']
        # )
        nodes = [
            self.create_node(
                op_type,
                inputs=[in_names[0]],
                outputs=[out_names[0]],
                perm=attrs['perm']
            )
        ]
        return nodes, []

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        process_scale = op_setting["process_scale"]
        op_setting.update(
            {'si': setting['in_scale'], 'sk': default_quant, 'so': self.get_scale(),
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('transpose')(**op_setting))


@LAYER.register_module(name='split')
class Split(MultiOutputLayer):
    def __init__(self, **kwargs):
        super(Split, self).__init__(**kwargs)

    def export_onnx(self):
        op_type = self.get_layer_type()
        op_type = op_type[0].upper() + op_type[1:]
        layer_name = self.get_layer_name()
        out_names = self.get_onnx_output_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        process_scale = self.get_ops_setting()['setting']['process_scale']
        attrs = self.get_ops_setting()['attrs'][0]
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, attrs)
        return nodes, initializers

    def export_onnx_fp(self, is_vis_qparams=False):
        attrs = self.get_layer_ops()["attrs"][0]
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            output_scales, output_shifts = [], []
            input_zero_points, output_zero_points = [], []
            for scale in self.get_scales():
                output_scales.append(scale["out_scale"])
                output_shifts.append(scale["int_scale"])
                input_zero_points.append(scale["zi"])
                output_zero_points.append(scale["zo"])
            output_scale = onnx.numpy_helper.from_array(np.array(output_scales, dtype=np.int32).squeeze())
            output_shift = onnx.numpy_helper.from_array(np.array(output_shifts, dtype=np.int8).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array(input_zero_points, dtype=np.int32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array(output_zero_points, dtype=np.int32).squeeze())
            qparam = dict(
                output_scale=output_scale,
                output_shift=output_shift,
                input_zero_point=input_zero_point,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        nodes, initializers = [], []
        op_type = self.get_nodes()[0].get_op_type()
        nodes.append(
            self.create_node(
                op_type,
                inputs=[
                    self.get_nodes()[0].get_onnx_input()[0],
                    self.get_nodes()[0].get_weights()[0]['name'],
                ],
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=dict(axis=attrs['axis']),
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )
        initializers.append(
            create_initializer(attrs['split'], self.get_nodes()[0].get_weights()[0]['name'], dtype=np.int64)
        )
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        # ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        out_quantize = [out_quantize['feat']['so0'], out_quantize['feat']['so1']]
        op_setting.update(
            {'si': setting['in_scale'], 'sk': default_quant, 'so': self.get_scale(),
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize[0],
             'quantize': out_quantize, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('split')(**op_setting))

    def forward(self, in_data, **kwargs):
        outputs = list()
        ops_instance = self.get_ops_instance()
        layer_type = self.get_layer_type()
        self.set_in_data(in_data)

        in_idxs = self.get_input_idx()

        if len(in_idxs) > 1:
            in_data = in_data
        else:
            in_data = in_data[0]
        if isinstance(ops_instance, list):
            for op in ops_instance:
                in_data = op(in_data, **kwargs)
                outputs.append(in_data)
        else:
            outputs = ops_instance(in_data, **kwargs)

        setting = self.get_ops_setting()['setting']
        bits_dict, mins, maxs = setting['bits_dict'], setting['mins'], setting['maxs']
        out_type = self.get_output_type()
        data_type, min_v, max_v = bits_dict[out_type], mins[out_type], maxs[out_type]
        for idx, _ in enumerate(outputs):
            if isinstance(outputs[idx], np.ndarray):
                out = clip(outputs[idx], min_v, max_v)
                outputs[idx] = out.astype(bits_dict[self.get_output_type()])
            elif isinstance(outputs[idx], dict):
                out = clip(outputs[idx]['output'], min_v, max_v)
                outputs[idx]['output'] = out.astype(bits_dict[self.get_output_type()])
            else:
                print('split not support outputs data structure!')
        self.set_out_data(outputs)


@LAYER.register_module(name='slice')
class Slice(SingleInputLayer):
    def __init__(self, **kwargs):
        super(Slice, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        process_scale = op_setting["process_scale"]
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        op_setting.update(
            {'si': self.get_scale(), 'sk': default_quant, 'so': self.get_scale(),
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize,
             'quanize': out_quantize, 'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('slice')(**op_setting))


@LAYER.register_module(name='reshape')
class Reshape(ShapeLayer):
    def __init__(self, **kwargs):
        super(Reshape, self).__init__(**kwargs)

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=[
                    self.get_nodes()[0].get_onnx_input()[0],
                    self.get_nodes()[0].get_weights()[0]['name']
                ],
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=dict(),
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        ) 
        initializers.append(
            create_initializer(
                self.get_nodes()[0].get_attr()["shape"], 
                self.get_nodes()[0].get_weights()[0]['name'], dtype=np.int64)
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'Reshape'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        # new_attrs = dict(
        #     perm=attrs['perm']
        # )
        nodes = [
            self.create_node(
                op_type,
                inputs=[in_names[0], layer_name + 'shape'],
                outputs=[out_names[0]]
            )
        ]

        initializers = [create_initializer(attrs['shape'], layer_name + 'shape', dtype=np.int64)]
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        process_scale = op_setting['process_scale']
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('reshape')(**op_setting))


@LAYER.register_module(name='pad')
class Pad(ShapeLayer):
    def __init__(self, **kwargs):
        super(Pad, self).__init__(**kwargs)

    def export_onnx_fp(self, is_vis_qparams=False):
        attrs = copy.deepcopy(self.get_nodes()[0].get_attr())
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
            for key, dtype in zip(["pads", "constant_value"], [np.int32, np.float32]):
                if key in attrs.keys():
                    attrs.update({key:onnx.numpy_helper.from_array(attrs[key].astype(dtype))})
             
        nodes, initializers = [], []
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'Pad'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        nodes = [
            self.create_node(
                op_type,
                inputs=[in_names[0]],
                outputs=[out_names[0]],
                attrs=attrs
            )
        ]

        # initializers = [create_initializer(attrs['axes'], layer_name + 'axes', dtype=np.int64)]
        return nodes, []

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        process_scale = op_setting['process_scale']
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('pad')(**op_setting))


@LAYER.register_module(name='unsqueeze')
class Unsqueeze(ShapeLayer):
    def __init__(self, **kwargs):
        super(Unsqueeze, self).__init__(**kwargs)

    def export_onnx(self):
        op_type = 'Unsqueeze'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        # new_attrs = dict(
        #     perm=attrs['perm']
        # )
        nodes = [
            self.create_node(
                op_type,
                inputs=[in_names[0], layer_name + 'axes'],
                outputs=[out_names[0]]
            )
        ]

        initializers = [create_initializer(attrs['axes'], layer_name + 'axes', dtype=np.int64)]
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        process_scale = op_setting['process_scale']
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('unsqueeze')(**op_setting))


@LAYER.register_module(name='squeeze')
class Squeeze(ShapeLayer):
    def __init__(self, **kwargs):
        super(Squeeze, self).__init__(**kwargs)

    def export_onnx(self):
        op_type = 'Squeeze'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        # new_attrs = dict(
        #     perm=attrs['perm']
        # )
        nodes = [
            self.create_node(
                op_type,
                inputs=[in_names[0], layer_name + 'axes'],
                outputs=[out_names[0]]
            )
        ]

        initializers = [create_initializer(attrs['axes'], layer_name + 'axes', dtype=np.int64)]
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        process_scale = op_setting['process_scale']
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('squeeze')(**op_setting))


@LAYER.register_module(name='flatten')
class Flatten(ShapeLayer):
    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    def export_onnx(self):
        op_type = 'Flatten'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        # new_attrs = dict(
        #     perm=attrs['perm']
        # )
        nodes = [
            self.create_node(
                op_type,
                inputs=[in_names[0]],
                outputs=[out_names[0]],
                axis=attrs['axis']
            )
        ]

        # initializers = [create_initializer(attrs['axes'], layer_name + 'axes', dtype=np.int64)]
        return nodes, []

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        process_scale = op_setting['process_scale']
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('flatten')(**op_setting))


@LAYER.register_module(name='resize')
class Resize(SingleInputLayer):
    def __init__(self, **kwargs):
        super(Resize, self).__init__(**kwargs)
    
    def get_onnx_input_name(self):
        return [self.get_nodes()[0].get_onnx_input()[0]]

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'in_quantize': in_quantize, 'quantize': out_quantize,
             'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('resize')(**op_setting))

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = copy.deepcopy(self.get_nodes()[0].get_attr())
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        layer_name = self.get_layer_name() + "_"

        inputs = [self.get_nodes()[0].get_onnx_input()[0], '', '', '']
        if isinstance(attrs['roi'], np.ndarray):
            inputs[1] = layer_name + 'roi'
            roi = attrs['roi'] if isinstance(attrs['roi'], np.ndarray) else 0
            initializers.extend([
                create_initializer(roi, layer_name + 'roi')
            ])
        if isinstance(attrs['scale'], np.ndarray):
            inputs[2] = layer_name + 'scales'
            inputs = inputs[:3]
            scale = attrs['scale'] if isinstance(attrs['scale'], np.ndarray) else 0
            initializers.extend([
                create_initializer(np.array([]), layer_name + 'roi'),
                create_initializer(scale, layer_name + 'scales'),
            ])
        else:
            inputs[3] = layer_name + 'sizes'
            sizes = attrs['sizes'] if isinstance(attrs['sizes'], np.ndarray) else 0
            initializers.extend([
                create_initializer(0, layer_name + 'roi'),
                create_initializer(0, layer_name + 'scales'),
                create_initializer(sizes, layer_name + 'sizes', dtype=np.int64),
            ]) 
        attrs.pop("scale")        
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=inputs,
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'Resize'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        new_attrs = {}
        for key in attrs.keys():
            if key in ['scale', 'roi']:
                continue
            new_attrs[key] = attrs[key]

        inputs = dict()
        if 'scale' in attrs.keys():
            inputs = dict(scale=attrs['scale'])
        if 'roi' in attrs.keys():
            inputs.update(roi=attrs['roi'])
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, inputs, new_attrs)
        return nodes, initializers

    @staticmethod
    def remove_attr(attrs, key):
        if key in attrs.keys():
            del attrs[key]

    def export_smooth(self, op_type, layer_name, in_names, out_names, scales, attrs):
        nodes, initializers = [], []
        in_name = copy.deepcopy(in_names)[0]

        inputs = [in_name, '', '', '']
        if isinstance(scales['roi'], np.ndarray):
            inputs[1] = layer_name + 'roi'
            roi = scales['roi'] if isinstance(scales['roi'], np.ndarray) else 0
            initializers.extend([
                create_initializer(roi, layer_name + 'roi')
            ])
        if isinstance(scales['scale'], np.ndarray):
            inputs[2] = layer_name + 'scales'
            inputs = inputs[:3]
            scale = scales['scale'] if isinstance(scales['scale'], np.ndarray) else 0
            initializers.extend([
                create_initializer(np.array([]), layer_name + 'roi'),
                create_initializer(scale, layer_name + 'scales'),
            ])
        else:
            inputs[3] = layer_name + 'sizes'
            sizes = scales['sizes'] if isinstance(scales['sizes'], np.ndarray) else 0
            initializers.extend([
                create_initializer(0, layer_name + 'roi'),
                create_initializer(0, layer_name + 'scales'),
                create_initializer(sizes, layer_name + 'sizes', dtype=np.int64),
            ])
        nattrs = copy.deepcopy(attrs)
        self.remove_attr(nattrs, "sizes")
        self.remove_attr(nattrs, "scale")
        nodes.extend([
            self.create_node(
                'Resize',
                inputs=inputs,
                outputs=out_names,
                attrs=nattrs
            )
        ])

        return nodes, initializers

    def export_float(self, op_type, layer_name, in_names, out_names, scales, attrs):
        # nodes, initializers = [], []
        # zi = self.get_in_scale()[0]['zero_point']
        # zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]

        nodes, initializers, out_name = \
            self.insert_dequant(self, layer_name, in_name, None, self.get_in_scale()[0]['scale'],
                                self.get_in_scale()[0]['zero_point'])
        
        attrs = self.get_ops_setting()['attrs'][0]
        scales = dict()
        if 'scale' in attrs.keys():
            scales = dict(scale=attrs['scale'])
        else:
            scales = dict(scale=None)
        if 'roi' in attrs.keys():
            scales.update(roi=attrs['roi'])
        else:
            scales.update(roi=None)

        inputs = [out_name, '', '', '']
        if isinstance(scales['roi'], np.ndarray):
            inputs[1] = layer_name + 'roi'
            roi = scales['roi'] if isinstance(scales['roi'], np.ndarray) else 0
            initializers.extend([
                create_initializer(roi, layer_name + 'roi')
            ])
        if isinstance(scales['scale'], np.ndarray):
            inputs[2] = layer_name + 'scales'
            inputs = inputs[:3]
            scale = scales['scale'] if isinstance(scales['scale'], np.ndarray) else 0
            initializers.extend([
                create_initializer(scale, layer_name + 'scales'),
            ])
        else:
            inputs[3] = layer_name + 'sizes'
            sizes = (
                np.array(attrs["sizes"]).astype(np.int64) if "sizes" in attrs.keys() else None
            )
            initializers.extend([
                # create_initializer(0, layer_name + 'scales'),
                create_initializer(sizes, layer_name + 'sizes', dtype=np.int64),
            ])
        nattrs = copy.deepcopy(attrs)
        self.remove_attr(nattrs, "sizes")
        self.remove_attr(nattrs, "scale")
        nodes.extend([
            self.create_node(
                'Resize',
                inputs=inputs,
                outputs=[layer_name + 'resize_output'],
                attrs=nattrs
            )
        ])

        de_nodes, de_initializer, _ = \
            self.insert_quant(self, layer_name, layer_name + 'resize_output', out_names, min_value, max_value,
                              self.get_scale()[0]['scale'], self.get_scale()[0]['zero_point'])
        nodes.extend(de_nodes)
        initializers.extend(de_initializer)

        return nodes, initializers

    def export_floatscale(self, op_type, layer_name, in_names, out_names, scales, attrs):
        # nodes, initializers = [], []
        # zi = self.get_in_scale()[0]['zero_point']
        # zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        nodes, initializers = [], []

        if self.get_in_scale()[0]['zero_point'] != 0:
            nodes = [
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + '_zi'],
                    outputs=[layer_name + '_sub_0']
                )
            ]
            initializers = [
                create_initializer(self.get_in_scale()[0]['zero_point'], layer_name + '_sub_0')
            ]
            in_name = layer_name + '_sub_0'

        inputs = [in_name, '', '', '']
        if isinstance(scales['roi'], np.ndarray):
            inputs[1] = layer_name + 'roi'
            roi = scales['roi'] if isinstance(scales['roi'], np.ndarray) else 0
            initializers.extend([
                create_initializer(roi, layer_name + 'roi')
            ])
        if isinstance(scales['scale'], np.ndarray):
            inputs[2] = layer_name + 'scales'
            inputs = inputs[:3]
            scale = scales['scale'] if isinstance(scales['scale'], np.ndarray) else 0
            initializers.extend([
                create_initializer(scale, layer_name + 'scales'),
            ])
        else:
            inputs[3] = layer_name + 'sizes'
            sizes = scales['sizes'] if isinstance(scales['sizes'], np.ndarray) else 0
            initializers.extend([
                # create_initializer(0, layer_name + 'scales'),
                create_initializer(sizes, layer_name + 'sizes', dtype=np.int64),
            ])
        nattrs = copy.deepcopy(attrs)
        self.remove_attr(nattrs, "sizes")
        self.remove_attr(nattrs, "scale")
        nodes.extend([
            self.create_node(
                'Resize',
                inputs=inputs,
                outputs=[layer_name + 'resize_output'],
                attrs=nattrs
            )
        ])

        nodes.append(
            self.create_node(
                'Mul',
                inputs=[layer_name + 'resize_output', layer_name + '_mul_0'],
                outputs=[layer_name + '_mul_out']
            )
        )
        initializers.extend([
            create_initializer(self.scales[-1]['out_scale'], layer_name + '_mul_0')
        ])

        if self.get_scale()[0]['zero_point'] != 0:
            nodes.extend([
                self.create_node(
                    'Add',
                    inputs=[layer_name + '_mul_out', layer_name + '_zo'],
                    outputs=[layer_name + '_add_0']
                )
            ])
            in_name = layer_name + '_add_0'
            initializers.append(create_initializer(self.get_scale()[0]['zero_point'], layer_name + '_zo'))

        nodes.extend([
            self.create_node(
                'Round',
                inputs=[in_name],
                outputs=[layer_name + '_round']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_round', layer_name + '_min', layer_name + '_max'],
                outputs=out_names
            )]
        )
        initializers.extend([
            create_initializer(min_value, layer_name + '_min'),
            create_initializer(max_value, layer_name + '_max')])

        return nodes, initializers


@LAYER.register_module(name='sigmoid')
class Sigmoid(ActivationLayer):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        if isinstance(si, list):
            si = si[0]
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'in_quantize': in_quantize[0], 'quantize': out_quantize['feat']['so0'],
             'isolated': True, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get(self.get_layer_type())(**op_setting))


@LAYER.register_module(name='gelu')
class GELU(Sigmoid):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)


@LAYER.register_module(name='tanh')
class Tanh(ActivationLayer):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        if isinstance(si, list):
            si = si[0]
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'in_quantize': in_quantize[0], 'quantize': out_quantize['feat']['so0'],
             'isolated': True, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('tanh')(**op_setting))


@LAYER.register_module(name='softmax')
class Softmax(SingleInputLayer):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def export_onnx_fp(self, is_vis_qparams=False):
        attrs = copy.deepcopy(self.get_nodes()[0].get_attr())
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        nodes, initializers = [], []
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'Softmax'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        nodes = [
            self.create_node(
                op_type,
                inputs=[in_names[0]],
                outputs=[out_names[0]],
                attrs=attrs
            )
        ]

        return nodes, []

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        si, so = setting['in_scale'], self.get_scale()[0]
        assert base_setting['process_scale'] in ['float', 'table']
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': base_setting['process_scale'], 'in_quantize': in_quantize[0],
             'quantize': out_quantize, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('softmax')(**op_setting))


@LAYER.register_module(name='reducemax')
@LAYER.register_module(name='reducemin')
@LAYER.register_module(name='reducemean')
@LAYER.register_module(name='reducesum')
@LAYER.register_module(name='reduceprod')
class ReduceOps(SingleInputLayer):
    def __init__(self, **kwargs):
        super(ReduceOps, self).__init__(**kwargs)

    def export_onnx_fp(self, is_vis_qparams=False):
        pass

    def export_onnx(self):
        pass

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        process_scale = op_setting["process_scale"]
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        op_setting.update(
            {'si': setting['in_scale'], 'sk': default_quant, 'so': self.get_scale(),
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'out_type': op_setting['out_type'],
             'in_quantize': in_quantize[0], 'quantize': out_quantize['feat']['so0'],})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get(ops[0])(**op_setting))


@LAYER.register_module(name='maxpool')
class MaxPool(SingleInputLayer):
    def __init__(self, **kwargs):
        super(MaxPool, self).__init__(**kwargs)

    def export_onnx_fp(self, is_vis_qparams=False):
        attrs = copy.deepcopy(self.get_nodes()[0].get_attr())
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        nodes, initializers = [], []
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'MaxPool'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        nodes = [
            self.create_node(
                op_type,
                inputs=[in_names[0]],
                outputs=[out_names[0]],
                attrs=attrs
            )
        ]

        # initializers = [create_initializer(attrs['axes'], layer_name + 'axes', dtype=np.int64)]
        return nodes, []

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        process_scale = op_setting["process_scale"]
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        op_setting.update(
            {'si': setting['in_scale'], 'sk': default_quant, 'so': self.get_scale(),
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'out_type': op_setting['out_type'],
             'in_quantize': in_quantize[0], 'quantize': out_quantize['feat']['so0'],})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('maxpool')(**op_setting))


@LAYER.register_module(name='globalaveragepool')
@LAYER.register_module(name='averagepool')
class AveragePool(SingleInputLayer):
    def __init__(self, **kwargs):
        super(AveragePool, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        process_scale = op_setting["process_scale"]
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        op_setting.update(
            {'si': setting['in_scale'], 'sk': default_quant, 'so': self.get_scale(),
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'in_quantize': in_quantize[0], 'quantize': out_quantize['feat']['so0'],
             'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get(ops[0])(**op_setting))

    def export_smooth(self, op_type, layer_name, in_names, out_names, attrs):
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        in_name = copy.deepcopy(in_names)[0]
        zi, zo = self.get_in_scale()[0]['zero_point'], self.get_scale()[0]['zero_point']
        nodes, initializers = [], []
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + '_zi'],
                    outputs=[layer_name + '_sub_1']
                )
            )
            in_name = layer_name + '_sub_1'
            initializers.append(
                create_initializer(zi, layer_name + '_zi')
            )
        nodes.extend([
            self.create_node(
                op_type,
                inputs=[in_name],
                outputs=[layer_name + '_' + op_type],
                attrs=attrs
            ),
            self.create_node(
                'Round',
                inputs=[layer_name + '_' + op_type],
                outputs=[layer_name + '_Round_out']
            )
        ])
        in_name = layer_name + '_Round_out'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[in_name, layer_name + '_zo'],
                    outputs=[layer_name + '_add_1']
                )
            )
            in_name = layer_name + '_add_1'
            initializers.append(
                create_initializer(zo, layer_name + '_zo')
            )
        nodes.append(
            self.create_node(
                'Clip',
                inputs=[in_name, layer_name + '_min', layer_name + '_max'],
                outputs=[out_names[0]]
            )
        )

        initializers.extend([
            create_initializer(min_value, layer_name + '_min'),
            create_initializer(max_value, layer_name + '_max')
        ])

        return nodes, initializers

    def export_float(self, op_type, layer_name, in_names, out_names, attrs):
        zi = self.get_in_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        nodes, initializers = [], []
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub']
                )
            )
            in_name = layer_name + 'sub'
            initializers.append(create_initializer(zi, layer_name + 'zi'))

        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[in_name, layer_name + 's1'],
                outputs=[layer_name + 'mul']
            ),
            self.create_node(
                op_type,
                inputs=[layer_name + 'mul'],
                outputs=[layer_name + op_type + 'output'],
                attrs=attrs
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + op_type + 'output', layer_name + 's2'],
                outputs=[layer_name + 'mul_2']
            )
        ])
        initializers.append(create_initializer(self.get_in_scale()[0]['scale'], layer_name + 's1'))
        initializers.append(create_initializer(self.get_scale()[0]['scale'], layer_name + 's2'))
        zo = self.get_scale()[0]['zero_point']
        out_name = layer_name + 'mul_2'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[out_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add']
                )
            )
            initializers.append(create_initializer(zo, layer_name + 'zo'))
            out_name = layer_name + 'add'
        nodes.extend([
            self.create_node(
                'Round',
                inputs=[out_name],
                outputs=[layer_name + '_Round_out']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_Round_out', layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]]
            )
        ])
        initializers.extend([
            create_initializer(min_value, layer_name + 'min'),
            create_initializer(max_value, layer_name + 'max'),
        ])

        return nodes, initializers

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = copy.deepcopy(self.get_nodes()[0].get_attr())
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'AveragePool'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        if attrs == dict():
            op_type = 'GlobalAveragePool'
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, attrs)
        return nodes, initializers

    def forward(self, in_data, **kwargs):
        super().forward(in_data, **kwargs)
        ops = self.get_layer_ops()
        if ops['attrs'] == [dict()]:
            kernel_size = list(in_data[0]['output'].shape[2:])
            new_attrs = [{'ceil_mode': 0,
                          'kernel_shape': kernel_size,
                          'pads': [0, 0, 0, 0],
                          'strides': [1, 1]}]
            ops['attrs'] = new_attrs
            self.set_layer_ops(ops)


@LAYER.register_module(name='relu')
class Relu(SingleInputLayer):
    def __init__(self, **kwargs):
        super(Relu, self).__init__(**kwargs)

    def export_float(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[in_name, layer_name + 's1'],
                outputs=[layer_name + 'mul_1_output']
            ),
            self.create_node(
                op_type,
                inputs=[layer_name + 'mul_1_output'],
                outputs=[layer_name + op_type + '_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + op_type + '_output', layer_name + 's2'],
                outputs=[layer_name + 'mul_2_output']
            ),
            self.create_node(
                'Round',
                inputs=[layer_name + 'mul_2_output'],
                outputs=[layer_name + 'round_output']
            )
        ])
        tmp_name = layer_name + 'round_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
            create_initializer(self.get_in_scale()[0]['scale'], layer_name + 's1'),
            create_initializer(1 / self.get_scale()[0]['scale'], layer_name + 's2'),
        ])
        for key in attrs.keys():
            initializers.append(create_initializer(attrs[key], layer_name + key))

        return nodes, initializers

    def export_floatscale(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                op_type,
                inputs=[in_name],
                outputs=[layer_name + op_type + '_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + op_type + '_output', layer_name + 's2'],
                outputs=[layer_name + 'mul_2_output']
            ),
            self.create_node(
                'Round',
                inputs=[layer_name + 'mul_2_output'],
                outputs=[layer_name + 'round_output']
            )
        ])
        tmp_name = layer_name + 'round_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
            create_initializer(self.get_scales()[-1]["out_scale"], layer_name + 's2'),
        ])
        for key in attrs.keys():
            initializers.append(create_initializer(attrs[key], layer_name + key))

        return nodes, initializers

    def export_preintscale(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                'Relu',
                inputs=[in_name],
                outputs=[layer_name + 'act_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + 'act_output', layer_name + 'out_scale'],
                outputs=[layer_name + 'mul_1_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + 'mul_1_output', layer_name + 'int_scale'],
                outputs=[layer_name + 'mul_2_output']
            ),
            self.create_node(
                virtual_op_type,
                inputs=[layer_name + 'mul_2_output'],
                outputs=[layer_name + 'ceil_output']
            ),
        ])
        tmp_name = layer_name + 'ceil_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
            create_initializer(self.get_scales()[-1]['out_scale'], layer_name + "out_scale"),
            create_initializer(1 / (2 ** self.get_scales()[-1]['int_scale']), layer_name + "int_scale")
        ])

        return nodes, initializers

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=self.get_nodes()[0].get_attr(),
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )        
        return nodes, initializers

    def export_smooth(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]

        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                'Relu',
                inputs=[in_name],
                outputs=[layer_name + 'act_output']
            )
        ])
        tmp_name = layer_name + 'act_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max")
        ])

        return nodes, initializers

    def export_onnx(self):
        op_type = self.get_layer_type()
        op_type = 'Relu'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names)
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        process_scale = op_setting["process_scale"]
        si, so = setting['in_scale'], self.get_scale()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize,
             'quantize': out_quantize, 'isolated': True, 'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('relu')(**op_setting))

    # def forward(self, in_data, **kwargs):
    #     super().forward(in_data, **kwargs)


@LAYER.register_module(name='relu6')
class Relu6(Relu):
    def __init__(self, **kwargs):
        super(Relu6, self).__init__(**kwargs)

    def export_float(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[in_name, layer_name + 's1'],
                outputs=[layer_name + 'mul_1_output']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + 'mul_1_output', layer_name + op_type + 'min', layer_name + op_type + 'max'],
                outputs=[layer_name + op_type + '_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + op_type + '_output', layer_name + 's2'],
                outputs=[layer_name + 'mul_2_output']
            ),
            self.create_node(
                'Round',
                inputs=[layer_name + 'mul_2_output'],
                outputs=[layer_name + 'round_output']
            )
        ])
        tmp_name = layer_name + 'round_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
            create_initializer(0, layer_name + op_type + 'min'),
            create_initializer(attrs['max_value'], layer_name + op_type + 'max'),
            create_initializer(1 / self.get_in_scale()['scale'], layer_name + 's1'),
            create_initializer(self.get_scale()['scale'], layer_name + 's2'),
        ])
        for key in attrs.keys():
            initializers.append(create_initializer(attrs[key], layer_name + key))

        return nodes, initializers

    def export_floatscale(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                'Clip',
                inputs=[in_name, layer_name + op_type + 'min', layer_name + op_type + 'max'],
                outputs=[layer_name + op_type + '_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + op_type + '_output', layer_name + 's2'],
                outputs=[layer_name + 'mul_2_output']
            ),
            self.create_node(
                'Round',
                inputs=[layer_name + 'mul_2_output'],
                outputs=[layer_name + 'round_output']
            )
        ])
        tmp_name = layer_name + 'round_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
            create_initializer(0, layer_name + op_type + 'min'),
            create_initializer(attrs['max_value'], layer_name + op_type + 'max'),
            create_initializer(self.get_scales()[-1]["out_scale"], layer_name + 's2'),
        ])
        for key in attrs.keys():
            initializers.append(create_initializer(attrs[key], layer_name + key))

        return nodes, initializers

    def export_preintscale(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.append(
            self.create_node(
                "Clip",
                inputs=[in_name, layer_name + '_' + op_type + '_min', layer_name + '_' + op_type + '_max'],
                outputs=[layer_name + '_' + op_type + '_clip']
            )
        )
        in_name = layer_name + '_' + op_type + '_clip'

        if self.get_scales()[-1]['out_scale'] != 1:
            nodes.extend([
                self.create_node(
                    'Mul',
                    inputs=[layer_name + '_' + op_type + '_clip', layer_name + 'out_scale'],
                    outputs=[layer_name + 'mul_1_output']
                ),
                self.create_node(
                    'Mul',
                    inputs=[layer_name + 'mul_1_output', layer_name + 'int_scale'],
                    outputs=[layer_name + 'mul_2_output']
                ),
                self.create_node(
                    virtual_op_type,
                    inputs=[layer_name + 'mul_2_output'],
                    outputs=[layer_name + 'floor_output']
                )
            ])
            initializers.extend([
                create_initializer(self.get_scales()[-1]['out_scale'], layer_name + "out_scale"),
                create_initializer(1 / (2 ** self.get_scales()[-1]['int_scale']), layer_name + "int_scale")
            ])
            in_name = layer_name + 'floor_output'

        if zo != 0:
            nodes.extend([
                self.create_node(
                    'Add',
                    inputs=[in_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            ])
            in_name = layer_name + 'add_output'
            initializers.extend([
                create_initializer(zo, layer_name + 'zo')
            ])
        nodes.append(
            self.create_node(
                "Clip",
                inputs=[in_name, layer_name + '_min', layer_name + '_max'],
                outputs=[out_names[0]]
            )
        )
        initializers.extend([
            create_initializer(min_value, layer_name + '_min'),
            create_initializer(max_value, layer_name + '_max'),
            create_initializer(ops.min_value, layer_name + '_' + op_type + '_min'),
            create_initializer(ops.max_value, layer_name + '_' + op_type + '_max'),
        ])

        return nodes, initializers

    def export_smooth(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]

        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                "Clip",
                inputs=[in_name, layer_name + '_' + op_type + '_min', layer_name + '_' + op_type + '_max'],
                outputs=[layer_name + '_' + op_type + '_clip']
            )
        ])
        tmp_name = layer_name + 'act_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + '_min'),
            create_initializer(max_value, layer_name + '_max'),
            create_initializer(ops.min_value, layer_name + '_' + op_type + '_min'),
            create_initializer(ops.max_value, layer_name + '_' + op_type + '_max')
        ])

        return nodes, initializers

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = dict()
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            bits_dict = self.get_ops_setting()['setting']['bits_dict']
            bit_select = self.get_ops_setting()['setting']['bit_select']
            dtype = bits_dict[bit_select]
            if self.get_scale_type() == "table":
                table = self.get_ops_instance().get_table() # type: ignore
            else:
                table = []
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            table = onnx.numpy_helper.from_array(np.array(table, dtype=dtype))
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                table=table,
            )
            attrs.update(qparam)
            nodes.append(
                self.create_node(
                    "Relu6",
                    inputs=self.get_nodes()[0].get_onnx_input(),
                    outputs=self.get_nodes()[0].get_onnx_output(),
                    attrs=attrs,
                    node_name=self.get_nodes()[0].get_name(),
                    domain=domain,
                )
            )
        else:
            op_type = self.get_nodes()[0].get_op_type()
            attrs = copy.deepcopy(self.get_nodes()[0].get_attr())
            nodes.append(
                self.create_node(
                    "Clip",
                    inputs=self.get_nodes()[0].get_onnx_input(),
                    outputs=self.get_nodes()[0].get_onnx_output(),
                    attrs=attrs,
                    node_name=self.get_nodes()[0].get_name())
            )
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        process_scale = op_setting["process_scale"]
        si, so = setting['in_scale'], self.get_scale()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize,
             'quantize': out_quantize, 'isolated': True, 'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('relu6')(**op_setting))

    # def forward(self, in_data):
    #     super(Relu6, self).forward(in_data)


@LAYER.register_module(name='relux')
class Relux(Relu6):
    def __init__(self, **kwargs):
        super(Relux, self).__init__(**kwargs)

    # def setting_ops(self, setting: dict):
    #     pass


@LAYER.register_module(name='leakyrelu')
class LeakyRelu(ActivationLayer):
    def __init__(self, **kwargs):
        super(LeakyRelu, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        process_scale = op_setting["process_scale"]
        si, so = setting['in_scale'], self.get_scale()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize,
             'quantize': out_quantize['feat']['so0'], 'isolated': True,
             'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('leakyrelu')(**op_setting))

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = dict(alpha=self.get_nodes()[0].get_attr()['alpha'])
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            bits_dict = self.get_ops_setting()['setting']['bits_dict']
            bit_select = self.get_ops_setting()['setting']['bit_select']
            dtype = bits_dict[bit_select]
            if self.get_scale_type() == "table":
                table = self.get_ops_instance().get_table() # type: ignore
            else:
                table = []
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            table = onnx.numpy_helper.from_array(np.array(table, dtype=dtype))
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                table=table,
            )
            attrs.update(qparam)

        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_onnx(self):
        # op_type = self.get_layer_type()
        op_type = 'LeakyRelu'
        # attrs = self.get_ops_setting()['attrs'][0]
        attrs =dict(alpha=np.round(self.get_nodes()[0].get_attr()['alpha'], 5))
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + 'float')(op_type, layer_name, in_names, out_names, {'alpha': attrs['alpha']})
        return nodes, initializers


@LAYER.register_module(name='prelu')
class PReLU(ActivationLayer):
    def __init__(self, **kwargs):
        super(PReLU, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        process_scale = op_setting["process_scale"]
        self.set_scale_type(process_scale)
        si, so = setting['in_scale'], self.get_scale()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize,
             'quantize': out_quantize['feat']['so0'], 'isolated': True,
             'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('prelu')(**op_setting))

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = dict(alpha=self.get_nodes()[0].get_attr()['alpha'])
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            bits_dict = self.get_ops_setting()['setting']['bits_dict']
            bit_select = self.get_ops_setting()['setting']['bit_select']
            dtype = bits_dict[bit_select]
            if self.get_scale_type() == "table":
                table = self.get_ops_instance().get_table() # type: ignore
            else:
                table = []
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            table = onnx.numpy_helper.from_array(np.array(table, dtype=dtype))
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                table=table,
            )
            attrs.update(qparam)
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_table(self, op_type, layer_name, in_names, out_names, attrs={}):
        # clip_outs = []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]

        zi = self.get_in_scale()[0]['zero_point']

        in_name = copy.deepcopy(in_names)[0]
        # layer_name, in_name, out_name, min_value, max_value, scale, zero_point
        nodes, initializers, out_name = self.insert_dequant(self, layer_name, in_name, None,
                                                            self.get_in_scale()[0]['scale'], zi)
        slope_name = self.get_layer_name() + "_slope"
        nodes.extend([
            self.create_node(
                op_type,
                inputs=[out_name, slope_name],
                outputs=[layer_name + op_type + '_out'],
                attrs=dict(),
            )
        ])
        # layer_name, in_name, out_name, scale, zero_point
        de_nodes, de_initializer, _ = \
            self.insert_quant(self, layer_name, layer_name + op_type + '_out', out_names,
                              min_value, max_value, self.get_scale()[0]['scale'], self.get_scale()[0]['zero_point'])
        nodes.extend(de_nodes)
        initializers.extend(de_initializer)
        initializers.extend([
            create_initializer(attrs["slope"], slope_name),
        ])
        return nodes, initializers

    def export_float(self, op_type, layer_name, in_names, out_names, attrs={}):
        # clip_outs = []
        ops = copy.deepcopy(self.get_ops_instance())
        op = ops[-1] if isinstance(ops, list) else ops
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]

        zi = self.get_in_scale()[0]['zero_point']

        in_name = copy.deepcopy(in_names)[0]
        # layer_name, in_name, out_name, min_value, max_value, scale, zero_point
        nodes, initializers, out_name = self.insert_dequant(self, layer_name, in_name, None,
                                                            self.get_in_scale()[0]['scale'], zi)
        slope_name = self.get_layer_name() + "_slope"
        nodes.extend([
            self.create_node(
                op_type,
                inputs=[out_name, slope_name],
                outputs=[layer_name + op_type + '_out'],
                attrs=dict(),
            )
        ])
        # layer_name, in_name, out_name, scale, zero_point
        de_nodes, de_initializer, _ = \
            self.insert_quant(self, layer_name, layer_name + op_type + '_out', out_names,
                              min_value, max_value, self.get_scale()[0]['scale'], self.get_scale()[0]['zero_point'])
        nodes.extend(de_nodes)
        initializers.extend(de_initializer)
        initializers.extend([
            create_initializer(attrs["slope"], slope_name),
        ])
        return nodes, initializers

    def export_onnx(self):
        # op_type = self.get_layer_type()
        op_type = 'PRelu'
        # attrs = self.get_ops_setting()['attrs'][0]
        attrs =dict(slope=np.round(self.get_nodes()[0].get_attr()['slope'], 5))
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, {'slope': attrs['slope']})
        return nodes, initializers


@LAYER.register_module(name='splice')
class Splice(SingleInputLayer):
    def __init__(self, **kwargs):
        super(Splice, self).__init__(**kwargs)

    # def set_quantize(self, data: dict):
    #     pass

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        process_scale = op_setting["process_scale"]
        si, so = setting['in_scale'], self.get_scale()
        sk = setting['w_scale']
        op_setting.update(
            {'si': si, 'sk': sk, 'so': so,
             'context': attrs[0]['context'],
             'forward_indexes': attrs[0]['forward_indexes'],
             "has_fc": attrs[0]['has_fc'],
             "isolated": False,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize, 'quantize': out_quantize,
             'p_weights': self.get_qweight(), 'bias': self.get_qbias(), 'out_type': op_setting['out_type']})
        # op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('splice')(**op_setting))

    def forward(self, in_data, **kwargs):
        self.set_in_data(in_data)
        op_instance = self.get_ops_instance()

        outputs = op_instance(in_data, **kwargs)

        self.set_out_data(outputs)


@LAYER.register_module(name='log')
class Log(SingleInputLayer):
    def __init__(self, **kwargs):
        super(Log, self).__init__(**kwargs)

    def export_float(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                'Mul',
                inputs=[in_name, layer_name + 's1'],
                outputs=[layer_name + 'mul_1_output']
            ),
            self.create_node(
                op_type,
                inputs=[layer_name + 'mul_1_output'],
                outputs=[layer_name + op_type + '_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + op_type + '_output', layer_name + 's2'],
                outputs=[layer_name + 'mul_2_output']
            ),
            self.create_node(
                'Round',
                inputs=[layer_name + 'mul_2_output'],
                outputs=[layer_name + 'round_output']
            )
        ])
        tmp_name = layer_name + 'round_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
            create_initializer(self.get_in_scale()[0]['scale'], layer_name + 's1'),
            create_initializer(1 / self.get_scale()[0]['scale'], layer_name + 's2'),
        ])
        for key in attrs.keys():
            initializers.append(create_initializer(attrs[key], layer_name + key))

        return nodes, initializers

    def export_preintscale(self, op_type, layer_name, in_names, out_names, attrs={}):
        nodes, initializers = [], []
        zi = self.get_in_scale()[0]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        in_name = copy.deepcopy(in_names)[0]
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min_value, max_value = mins[bit_select], maxs[bit_select]
        if zi != 0:
            nodes.append(
                self.create_node(
                    'Sub',
                    inputs=[in_name, layer_name + 'zi'],
                    outputs=[layer_name + 'sub_output'])
            )
            in_name = layer_name + 'sub_output'
            initializers.append(create_initializer(zi, layer_name + 'zi'))
        nodes.extend([
            self.create_node(
                'Relu',
                inputs=[in_name],
                outputs=[layer_name + 'act_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + 'act_output', layer_name + 'out_scale'],
                outputs=[layer_name + 'mul_1_output']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + 'mul_1_output', layer_name + 'int_scale'],
                outputs=[layer_name + 'mul_2_output']
            ),
            self.create_node(
                'Floor',
                inputs=[layer_name + 'mul_2_output'],
                outputs=[layer_name + 'ceil_output']
            ),
        ])
        tmp_name = layer_name + 'ceil_output'
        if zo != 0:
            nodes.append(
                self.create_node(
                    'Add',
                    inputs=[tmp_name, layer_name + 'zo'],
                    outputs=[layer_name + 'add_output']
                )
            )
            tmp_name = layer_name + 'add_output'
            initializers.append(create_initializer(zo, layer_name + 'zo'))

        nodes.append(
            self.create_node(
                'Clip',
                inputs=[tmp_name, layer_name + 'min', layer_name + 'max'],
                outputs=[out_names[0]])
        )
        initializers.extend([
            create_initializer(min_value, layer_name + "min"),
            create_initializer(max_value, layer_name + "max"),
            create_initializer(self.get_scales()[-1]['out_scale'], layer_name + "out_scale"),
            create_initializer(1 / (2 ** self.get_scales()[-1]['int_scale']), layer_name + "int_scale")
        ])

        return nodes, initializers

    def export_onnx_fp(self):
        nodes, initializers = [], []
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=self.get_nodes()[0].get_attr(),
                node_name=self.get_nodes()[0].get_name())
        )
        return nodes, initializers

    def export_onnx(self):
        op_type = self.get_layer_type()
        op_type = 'Relu'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names)
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        process_scale = op_setting["process_scale"]
        si, so = setting['in_scale'], self.get_scale()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize,
             'quantize': out_quantize, 'isolated': True, 'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('log')(**op_setting))

    # def forward(self, in_data):
    #     super().forward(in_data)


@LAYER.register_module(name='lstm')
class Lstm(MultiInputLayer):
    def __init__(self, **kwargs):
        super(Lstm, self).__init__(**kwargs)
        self.wr_combine, self.hx_combine = True, True
        self.__is_update_quantize_from_in_data = False

    def get_table(self):
        return self.get_ops_instance().table
    
    def get_init_c(self):
        return self.get_ops_instance().get_init_c()

    def get_init_h(self):
        return self.get_ops_instance().get_init_h()

    def get_init_h(self):
        return copy.deepcopy(self.get_ops_instance().get_init_h())

    def get_init_c(self):
        return copy.deepcopy(self.get_ops_instance().get_init_c())

    def set_ops_setting(self, setting):
        super().set_ops_setting(setting)
        if 'hx_combine' in setting.keys():
            self.set_hx_combine(setting['hx_combine'])
        if 'wr_combine' in setting.keys():
            self.set_wr_combine(setting['wr_combine'])

    def set_hx_combine(self, hx_combine):
        self.hx_combine = hx_combine

    def get_hx_combine(self):
        return copy.deepcopy(self.hx_combine)

    def set_wr_combine(self, wr_combine):
        self.wr_combine = wr_combine

    def get_wr_combine(self):
        return copy.deepcopy(self.wr_combine)

    def reset(self):
        ops = self.get_ops_instance()
        if isinstance(ops, list):
            for op in ops:
                if hasattr(op, "reset"):
                    op.reset()
        else:
            ops.reset()
        self.set_ops_instance(ops)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        self.__is_update_quantize_from_in_data = setting["setting"].get("is_update_quantize_from_in_data", False)
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        process_scale = op_setting["process_scale"]
        si, so = setting['in_scale'], self.get_scale()
        sk = setting['w_scale']
        self.set_hx_combine(op_setting['hx_combine'])
        self.set_wr_combine(op_setting['wr_combine'])
        op_setting.update(
            {'si': si, 'sk': sk, 'so': so, 'hidden_size': attrs[0]['hidden_size'],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize, 'quantize': out_quantize,
             'p_weights': self.get_qweight(), 'bias': self.get_qbias(), 'out_type': op_setting['out_type']})
        # op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('lstm')(**op_setting))

    def set_quantize_from_in_data(self, quantize_from_in_data):
        self.__is_update_quantize_from_in_data = quantize_from_in_data
        
    def get_quantize_from_in_data(self):
        return self.__is_update_quantize_from_in_data
    
    def forward(self, in_data, **kwargs):
        self.set_in_data(in_data)
        # for op in self.get_ops_instance():
        #     op.in_quantize[0].get_quan_param(in_data[0]["output"])
        #     op.update_datacorrect(si=[op.in_quantize[0].get_quant_param()])
        #     if hasattr(op, "update_qbias"):
        #         op.update_qbias(self.get_layer_ops()['weights'][1])
        op_instance = self.get_ops_instance()
        kwargs["is_update_quantize_from_in_data"] = kwargs.get("is_update_quantize_from_in_data",
                                                               self.__is_update_quantize_from_in_data)
        outputs = op_instance(in_data, **kwargs)

        self.set_out_data(outputs)


@LAYER.register_module(name='gru')
class Gru(MultiInputLayer):
    def __init__(self, **kwargs):
        super(Gru, self).__init__(**kwargs)
        self.wr_combine, self.hx_combine = True, True
        
    def get_init_c(self):
        return self.get_ops_instance().get_init_c()

    def get_init_h(self):
        return self.get_ops_instance().get_init_h()

    def set_hx_combine(self, hx_combine):
        self.hx_combine = hx_combine

    def get_hx_combine(self):
        return copy.deepcopy(self.hx_combine)

    def set_wr_combine(self, wr_combine):
        self.wr_combine = wr_combine

    def get_wr_combine(self):
        return copy.deepcopy(self.wr_combine)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        process_scale = op_setting["process_scale"]
        si, so = setting['in_scale'], self.get_scale()
        sk = setting['w_scale']
        self.set_hx_combine(op_setting['hx_combine'])
        self.set_wr_combine(op_setting['wr_combine'])        
        op_setting.update(
            {'si': si, 'sk': sk, 'so': so, 'hidden_size': attrs[0]['hidden_size'],
             'linear_before_reset': attrs[0]['linear_before_reset'],
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'in_quantize': in_quantize, 'quantize': out_quantize,
             'p_weights': self.get_qweight(), 'bias': self.get_qbias(), 'out_type': op_setting['out_type']})
        # op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get('gru')(**op_setting))


@LAYER.register_module(name='mul')
@LAYER.register_module(name='cmul')
@LAYER.register_module(name='pmul')
class CMul(MultiInputLayer):
    def __init__(self, **kwargs):
        super(CMul, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)

        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        process_scale = op_setting["process_scale"]
        si, so = setting['in_scale'], self.get_scale()
        f_weights, p_weights, weight_idx, in_quantize = \
            self.reconstruct_inputs(setting, op_setting, si, in_quantize)
        op_setting.update(
            {'si': si[0], 'sk': si[1], 'so': so, 'bit_select': op_setting['bit_select'],
             'int_scale': op_setting['int_scale'], 'process_scale': process_scale,
             'in_quantize': in_quantize, 'quantize': out_quantize['feat']['so0'], 'p_weights': p_weights,
             'weight_idx': weight_idx, 'f_weights': f_weights, 'out_type': op_setting['out_type']})
        op_setting.update(attrs[0])
        self.set_ops_instance(operators_factory.get(ops[0])(**op_setting))

    def process_constant_input(self, nodes, initializers, input_names, zi0, zi1, layer_name):
        if len(input_names) < 2:
            weights_idx = self.get_weights_idx()
            weights = self.get_qweight().astype(np.float32)
            if weights_idx[0] == 0:
                input_names.insert(0, layer_name + '_sub_1_0')
                initializers.append(create_initializer(weights - zi0, layer_name + "_sub_1_0"))
                if zi1 != 0:
                    nodes.extend([
                        self.create_node(
                            'Sub',
                            inputs=[input_names[1], layer_name + '_zi1'],
                            outputs=[layer_name + '_sub_1_1']
                        )])
                    input_names[1] = layer_name + '_sub_1_1'
                    initializers.append(create_initializer(zi1, layer_name + "_zi1"))
            if weights_idx[0] == 1:
                input_names.insert(1, layer_name + '_sub_1_0')
                initializers.append(create_initializer(weights - zi0, layer_name + "_sub_1_0"))
                if zi0 != 0:
                    nodes.extend([
                        self.create_node(
                            'Sub',
                            inputs=[input_names[0], layer_name + 'zi0'],
                            outputs=[layer_name + '_sub_1_0']
                        )])
                    initializers.append(create_initializer(zi0, layer_name + "zi0"))
                    input_names[0] = layer_name + '_sub_1_0'
        else:
            if zi0 != 0:
                nodes.extend([
                    self.create_node(
                        'Sub',
                        inputs=[input_names[0], layer_name + 'zi0'],
                        outputs=[layer_name + '_sub_1_0']
                    )])
                initializers.append(create_initializer(zi0, layer_name + "zi0"))
                input_names[0] = layer_name + '_sub_1_0'

            if zi1 != 0:
                nodes.extend([
                    self.create_node(
                        'Sub',
                        inputs=[input_names[1], layer_name + '_zi1'],
                        outputs=[layer_name + '_sub_1_1']
                    )])
                input_names[1] = layer_name + '_sub_1_1'
                initializers.append(create_initializer(zi1, layer_name + "_zi1"))

    def export_intscale(self, op_type, layer_name, in_names, out_names, attrs):
        zi0, zi1 = self.get_in_scale()[0]['zero_point'], self.get_in_scale()[1]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        input_names = copy.deepcopy(in_names)
        nodes, initializers = [], []

        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'

        self.process_constant_input(nodes, initializers, input_names, zi0, zi1, layer_name)

        nodes.extend([
            self.create_node(
                    op_type,
                    inputs=[input_names[0], input_names[1]],
                    outputs=[layer_name + '_' + op_type + '_y'],
                    attrs=attrs
                ),
            self.create_node(
                'Mul',
                inputs=[layer_name + '_' + op_type + '_y', layer_name + '_out_shift'],
                outputs=[layer_name + '_mul_1']
            ),
            self.create_node(
                virtual_op_type,
                inputs=[layer_name + '_mul_1'],
                outputs=[layer_name + '_floor_1']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_floor_1', layer_name + '_' + op_type + '_min1',
                        layer_name + '_' + op_type + '_max1'],
                outputs=[layer_name + '_clip_1']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + '_clip_1', layer_name + '_out_scale'],
                outputs=[layer_name + '_mul_2']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + '_mul_2', layer_name + '_int_scale'],
                outputs=[layer_name + '_mul_3']
            ),
            self.create_node(
                virtual_op_type,
                inputs=[layer_name + '_mul_3'],
                outputs=[layer_name + '_floor_2']
            )
        ])
        name_1 = layer_name + '_floor_2'
        if zo != 0:
            nodes.extend([self.create_node(
                'Add',
                inputs=[name_1, layer_name + '_zo'],
                outputs=[layer_name + '_add_1'])
            ])
            name_1 = layer_name + '_add_1'
            initializers.append(create_initializer(zo, layer_name + "_zo"))
        nodes.extend([
            self.create_node(
                'Clip',
                inputs=[name_1, layer_name + '_min2', layer_name + '_max2'],
                outputs=[out_names[0]])
        ])
        out_shift = copy.deepcopy(self.get_scales()[-1]['out_shift'])
        out_scale = copy.deepcopy(self.get_scales()[-1]['out_scale'])
        if isinstance(out_shift, torch.Tensor):
            out_shift = out_shift.numpy()
        if isinstance(out_scale, torch.Tensor):
            out_scale = out_scale.numpy()
        int_scale = self.get_ops_setting()['setting']['int_scale']
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min2_v, max2_v = mins[bit_select], maxs[bit_select]
        if bit_select % 2 == 0:
            base_num = 2 ** bit
            min1_v, max1_v = 0, base_num - 1
        else:
            base_num = 2 ** (bit - 1)
            min1_v, max1_v = -base_num, base_num - 1

        initializers.extend([
            create_initializer(2 ** np.float32(out_shift) * self.eps, layer_name + "_out_shift"),
            create_initializer(out_scale, layer_name + "_out_scale"),
            create_initializer(1 / (2 ** int_scale) * self.eps, layer_name + "_int_scale"),
            create_initializer(min1_v, layer_name + '_' + op_type + "_min1"),
            create_initializer(max1_v, layer_name + '_' + op_type + "_max1"),
            create_initializer(min2_v, layer_name + "_min2"),
            create_initializer(max2_v, layer_name + "_max2"),
        ])
        return nodes, initializers

    def export_floatscale(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, initializers = [], []
        zi0, zi1 = self.get_in_scale()[0]['zero_point'], self.get_in_scale()[1]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        input_names = copy.deepcopy(in_names)
        self.process_constant_input(nodes, initializers, input_names, zi0, zi1, layer_name)
        nodes.extend([
            self.create_node(
                op_type,
                inputs=input_names,
                outputs=[layer_name + '_' + op_type + '_y'],
                attrs=attrs
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + '_' + op_type + '_y', layer_name + '_out_scale'],
                outputs=[layer_name + '_mul_2']
            )
        ])
        name_1 = layer_name + '_mul_2'
        if zo != 0:
            nodes.extend([self.create_node(
                'Add',
                inputs=[layer_name + '_mul_2', layer_name + '_o'],
                outputs=[layer_name + '_add_1'])
            ])
            name_1 = layer_name + '_add_1'
            initializers.append(create_initializer(zo, layer_name + "_zo"))
        nodes.extend([
            self.create_node(
                'Round',
                inputs=[name_1],
                outputs=[layer_name + '_round_out']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_round_out', layer_name + '_min1', layer_name + '_max1'],
                outputs=[out_names[0]])
        ])
        out_scale = copy.deepcopy(self.get_scales()[-1]['out_scale'])
        if isinstance(out_scale, torch.Tensor):
            out_scale = out_scale.numpy()
        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min2_v, max2_v = mins[bit_select], maxs[bit_select]

        initializers.extend([
            create_initializer(out_scale, layer_name + "_out_scale"),
            # create_initializer(self.get_scales()[-1]['zi'], layer_name + "zi"),
            # create_initializer(self.get_scales()[-1]['zo'], layer_name + "zo"),
            create_initializer(min2_v, layer_name + "_min1"),
            create_initializer(max2_v, layer_name + "_max1"),
        ])
        return nodes, initializers

    def export_shiftfloatscale(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, initializers = [], []
        zi0, zi1 = self.get_in_scale()[0]['zero_point'], self.get_in_scale()[1]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        input_names = copy.deepcopy(in_names)
        virtual_op_type = 'Floor'
        if self.get_ops_setting()['setting']['virtual_round']:
            virtual_op_type = 'Round'
        self.process_constant_input(nodes, initializers, input_names, zi0, zi1, layer_name)
        nodes.extend([
            self.create_node(
                op_type,
                inputs=[input_names[0], input_names[1]],
                outputs=[layer_name + '_' + op_type + '_y'],
                attrs=attrs
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + '_' + op_type + '_y', layer_name + '_out_shift'],
                outputs=[layer_name + '_mul_1']
            ),
            self.create_node(
                virtual_op_type,
                inputs=[layer_name + '_mul_1'],
                outputs=[layer_name + '_floor_1']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_floor_1', layer_name + '_min1', layer_name + '_max1'],
                outputs=[layer_name + '_clip_1']
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + '_clip_1', layer_name + '_out_scale'],
                outputs=[layer_name + '_mul_2']
            )
        ])
        name_1 = layer_name + '_mul_2'
        if zo != 0:
            nodes.extend([self.create_node(
                'Add',
                inputs=[layer_name + '_mul_2', layer_name + '_zo'],
                outputs=[layer_name + '_add_1'])
            ])
            name_1 = layer_name + '_add_1'
            initializers.append(create_initializer(self.get_scales()[-1]['zo'], layer_name + "_zo"))
        nodes.extend([
            self.create_node(
                'Round',
                inputs=[name_1],
                outputs=[layer_name + '_round_out']
            ),
            self.create_node(
                'Clip',
                inputs=[layer_name + '_round_out', layer_name + '_min2', layer_name + '_max2'],
                outputs=[out_names[0]])
        ])

        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        bit = op.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min2_v, max2_v = mins[bit_select], maxs[bit_select]
        out_shift = copy.deepcopy(self.get_scales()[-1]['out_shift'])
        out_scale = copy.deepcopy(self.get_scales()[-1]['out_scale'])
        if bit_select % 2 == 0:
            base_num = 2 ** bit
            min1_v, max1_v = 0, base_num - 1
        else:
            base_num = 2 ** (bit - 1)
            min1_v, max1_v = -base_num, base_num - 1

        initializers.extend([
            create_initializer(2 ** np.float32(out_shift) * self.eps, layer_name + "_out_shift"),
            create_initializer(out_scale, layer_name + "_out_scale"),
            create_initializer(min1_v, layer_name + "_min1"),
            create_initializer(min2_v, layer_name + "_min2"),
            create_initializer(max1_v, layer_name + "_max1"),
            create_initializer(max2_v, layer_name + "_max2"),
        ])

        return nodes, initializers

    def export_ffloatscale(self, op_type, layer_name, in_names, out_names, attrs):
        zi0, zi1 = self.get_in_scale()[0]['zero_point'], self.get_in_scale()[1]['zero_point']
        zo = self.get_scale()[0]['zero_point']
        input_names = copy.deepcopy(in_names)
        nodes, initializers = [], []
        self.process_constant_input(nodes, initializers, input_names, zi0, zi1, layer_name)
        nodes.extend([
            self.create_node(
                op_type,
                inputs=[input_names[0], input_names[1]],
                outputs=[layer_name + '_conv_y'],
                attrs=attrs
            ),
            self.create_node(
                'Mul',
                inputs=[layer_name + '_clip_1', layer_name + 'out_scale'],
                outputs=[out_names[0]]
            )
        ])

        out_scale = copy.deepcopy(self.get_scales()[-1]['out_scale'])
        if isinstance(out_scale, torch.Tensor):
            out_scale = out_scale.numpy()

        initializers.extend([
            create_initializer(out_scale, layer_name + "out_scale")
        ])

        return nodes, initializers

    def export_float(self, op_type, layer_name, in_names, out_names, attrs):
        nodes, initializers = [], []
        zi0, zi1 = self.get_in_scale()[0]['zero_point'], self.get_in_scale()[1]['zero_point']
        s0, s1 = self.get_in_scale()[0]['scale'], self.get_in_scale()[1]['scale']
        input_names = copy.deepcopy(in_names)
        self.process_constant_input(nodes, initializers, input_names, zi0, zi1, layer_name)
        nodes.extend([
            self.create_node(
                op_type,
                inputs=[layer_name + input_names[0] + '_01', layer_name + input_names[1] + '_10'],
                outputs=[layer_name + '_' + op_type + '_y'],
                attrs=attrs
            )
        ])

        ops = self.get_ops_instance()
        op = ops[-1] if isinstance(ops, list) else ops
        # bit = ops.get_bit_saturation()
        bit_select = op.bit_select
        mins, maxs = op.mins, op.maxs
        min2_v, max2_v = mins[bit_select], maxs[bit_select]
        q_nodes, q_initializers, _ = \
            self.insert_quant(self, layer_name, layer_name + '_' + op_type + '_y', out_names[0], min2_v, max2_v,
                              self.get_scale()[0]['scale'], self.get_scale()[0]['zero_point'])
        nodes.extend(q_nodes)
        initializers.extend(q_initializers)
        return nodes, initializers

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = copy.deepcopy(self.get_nodes()[0].get_attr())
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            output_shift = self.get_scales()[0]["out_shift"]
            output_scale = self.get_scales()[0]["out_scale"]
            input_zero_point = self.get_scales()[0]["zi"]
            output_zero_point = self.get_scales()[0]["zo"]
            # bits_dict = self.get_ops_setting()['setting']['bits_dict']
            output_shift = onnx.numpy_helper.from_array(np.array([output_shift], dtype=np.int8).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.int32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            qparam = dict(
                output_shift=output_shift,
                output_scale=output_scale,
                input_zero_point=input_zero_point,
                output_zero_point=output_zero_point,
            )
            attrs.update(qparam)
        nodes.append(
            self.create_node(
                self.get_nodes()[0].get_op_type(),
                inputs=self.get_nodes()[0].get_onnx_input(),
                outputs=self.get_nodes()[0].get_onnx_output(),
                attrs=attrs,
                node_name=self.get_nodes()[0].get_name(),
                domain=domain,
            )
        )

        return nodes, initializers

    def export_onnx(self):
        op_type = 'Mul'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, {})
        return nodes, initializers

    def forward(self, in_data, **kwargs):
        super().forward(in_data, **kwargs)
        if len(in_data) < 2:
            if self.get_weights_idx()[0] == 0:
                in_data0, in_data1 = self.get_qweight(), in_data[0]["output"]
            else:
                in_data1, in_data0 = self.get_qweight(), in_data[0]["output"]
        else:
            in_data0, in_data1 = in_data[0]['output'], in_data[1]['output']

        if self.get_layer_type() not in ["matmul"]:
            if in_data0.shape == in_data1.shape:
                self.set_layer_type('pmul')
            else:
                self.set_layer_type('cmul')


@LAYER.register_module(name='matmul')
class MatMul(CMul):
    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    # def forward(self, in_data):
    #     super().forward(in_data)

    def export_onnx(self):
        op_type = 'MatMul'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, {})
        return nodes, initializers


@LAYER.register_module(name='transformer')
class Transformer(MultiInputLayer):
    def __init__(self, **kwargs):
        super(Transformer, self).__init__(**kwargs)

    # def set_quantize(self, data: dict):
    #     pass

    def setting_ops(self, setting: dict):
        pass


@LAYER.register_module(name='gather')
class Gather(MultiInputLayer):
    def __init__(self, **kwargs):
        super(Gather, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()[0]
        process_scale = op_setting['process_scale']
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale'])})
        self.set_ops_instance(operators_factory.get('gather')(**op_setting))


@LAYER.register_module(name='embedlayernormalization')
class EmbedLayerNormalization:
    pass


@LAYER.register_module(name='layernormalization')
class LayerNormalization(NormLayer):
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    def export_onnx(self):
        op_type = 'LayerNormalization'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        bn_attrs = dict(
            s=attrs['scale'],
            bias=attrs['bias'],
        )
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, attrs['epsilon'],
                                                     bn_attrs)
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        si, so = setting['in_scale'], self.get_scale()[0]
        assert base_setting['process_scale'] == 'float'
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': base_setting['process_scale'], 'in_quantize': in_quantize[0],
             'quantize': out_quantize, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('layernormalization')(**op_setting))


@LAYER.register_module(name='instancenormalization')
class InstanceNormalization(NormLayer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    def export_onnx(self):
        op_type = 'InstanceNormalization'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        attrs = self.get_ops_setting()['attrs'][0]
        bn_attrs = dict(
            s=attrs['scale'],
            bias=attrs['bias'],
            # mean=attrs['mean'],
            # var=attrs['var']
        )
        setting = dict(epsilon=attrs['epsilon'])
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, setting, bn_attrs)
        return nodes, initializers

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()['feat']['so0']
        si, so = setting['in_scale'], self.get_scale()[0]
        assert base_setting['process_scale'] == 'float'
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': base_setting['process_scale'], 'in_quantize': in_quantize[0],
             'quantize': out_quantize, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('instancenormalization')(**op_setting))


@LAYER.register_module(name='multiheadattention')
class MultiHeadAttention:
    pass


@LAYER.register_module(name='skiplayernormalization')
class SkipLayerNormalization:
    pass


@LAYER.register_module(name='hardswish')
class Hswish(ActivationLayer):
    def __init__(self, **kwargs):
        super(Hswish, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'in_quantize': in_quantize, 'quantize': out_quantize['feat']['so0'],
             'isolated': True, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('hardswish')(**op_setting))

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = dict()
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            bits_dict = self.get_ops_setting()['setting']['bits_dict']
            bit_select = self.get_ops_setting()['setting']['bit_select']
            dtype = bits_dict[bit_select]
            if self.get_scale_type() == "table":
                table = self.get_ops_instance().get_table() # type: ignore
            else:
                table = []
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            table = onnx.numpy_helper.from_array(np.array(table, dtype=dtype))
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                table=table,
            )
            attrs.update(qparam)
            nodes.append(
                self.create_node(
                    self.get_nodes()[0].get_op_type(),
                    inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                    outputs=self.get_nodes()[0].get_onnx_output(),
                    attrs=attrs,
                    node_name=self.get_nodes()[0].get_name(),
                    domain=domain,
                )
            )
        else:
            if len(self.get_nodes()) == 1:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[0].get_op_type(),
                        inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=attrs,
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
            elif len(self.get_nodes()) == 2:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[0].get_op_type(),
                        inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=attrs,
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=self.get_nodes()[1].get_onnx_input(),
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=attrs,
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )
            else:
                raise Exception("Not implemented !!!")

        return nodes, initializers

    def export_onnx(self):
        # op_type = self.get_layer_type()
        op_type = 'HardSwish'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + process_scale)(op_type, layer_name, in_names, out_names, {})
        return nodes, initializers


@LAYER.register_module(name='swish')
class Swish(ActivationLayer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        # ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'in_quantize': in_quantize, 'quantize': out_quantize['feat']['so0'],
             'isolated': True, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('swish')(**op_setting))

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = dict()
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            bits_dict = self.get_ops_setting()['setting']['bits_dict']
            bit_select = self.get_ops_setting()['setting']['bit_select']
            dtype = bits_dict[bit_select]
            if self.get_scale_type() == "table":
                table = self.get_ops_instance().get_table() # type: ignore
            else:
                table = []
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            table = onnx.numpy_helper.from_array(np.array(table, dtype=dtype))
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                table=table,
                ops=["Sigmoid", "Mul"],
            )
            attrs.update(qparam)
            nodes.append(
                self.create_node(
                    "Swish",
                    inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                    outputs=self.get_nodes()[1].get_onnx_output(),
                    attrs=attrs,
                    node_name=self.get_nodes()[1].get_name(),
                    domain=domain,
                )
            )
        else:
            if len(self.get_nodes()) == 1:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[0].get_op_type(),
                        inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=attrs,
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
            elif len(self.get_nodes()) == 2:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[0].get_op_type(),
                        inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=attrs,
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=self.get_nodes()[1].get_onnx_input(),
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=attrs,
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )
            else:
                raise Exception("Not implemented !!!")

        return nodes, initializers

    def export_onnx(self):
        # op_type = self.get_layer_type()
        op_type = 'swish'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        attrs = self.get_ops_setting()['attrs'][0]
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        # layer_name, in_name, out_name, scale, zero_point
        in_scale = self.get_in_scale()[0]
        nodes, initializers, output_name = self.insert_dequant(self, layer_name, in_names[0], None, in_scale['scale'],
                                                               in_scale['zero_point'])

        nodes.extend([
            self.create_node(
                'Sigmoid',
                inputs=[output_name],
                outputs=[layer_name + '_sigmoid']
            ),
            self.create_node(
                'Mul',
                inputs=[output_name, layer_name + '_sigmoid'],
                outputs=[layer_name + '_mul']
            )
        ])

        bit = self.get_ops_setting()['setting']['bit_select']
        mins, maxs = self.get_ops_setting()['setting']['mins'], self.get_ops_setting()['setting']['maxs']
        min_value, max_value = mins[bit], maxs[bit]
        scale, zo = self.get_scale()[0]['scale'], self.get_scale()[0]['zero_point']
        # layer_name, in_name, out_names, min_value, max_value, scale, zero_point
        qnodes, qinitializers, output_name = \
            self.insert_quant(self, layer_name, layer_name + '_mul', out_names[0], min_value, max_value, scale, zo)
        nodes.extend(qnodes)
        initializers.extend(qinitializers)
        return nodes, initializers


@LAYER.register_module(name='hardsigmoid')
class Hsigmoid(ActivationLayer):
    def __init__(self, **kwargs):
        super(Hsigmoid, self).__init__(**kwargs)

    def setting_ops(self, setting: dict):
        super().setting_ops(setting)
        # ops = setting['ops_string']
        base_setting, attrs = setting['setting'], setting['attrs']
        op_setting = copy.deepcopy(base_setting)
        op_setting.update(attrs[0])
        si, so = setting['in_scale'], self.get_scale()
        process_scale = op_setting['process_scale']
        in_quantize, out_quantize = self.get_in_quantize(), self.get_quantize()
        op_setting.update(
            {'si': si, 'sk': default_quant, 'so': so,
             'bit_select': op_setting['bit_select'], 'int_scale': op_setting['int_scale'],
             'process_scale': process_scale, 'input_len': len(setting['in_scale']),
             'in_quantize': in_quantize, 'quantize': out_quantize['feat']['so0'],
             'isolated': True, 'out_type': op_setting['out_type']})
        self.set_ops_instance(operators_factory.get('hardsigmoid')(**op_setting))

    def export_onnx_fp(self, is_vis_qparams=False):
        nodes, initializers = [], []
        attrs = dict()
        domain = None
        if is_vis_qparams:
            domain = "timesintelli.com"
            input_scale = self.get_in_scale()[0]["scale"]
            input_zero_point = self.get_in_scale()[0]["zero_point"]
            output_scale = self.get_scale()[0]["scale"] # type: ignore
            output_zero_point = self.get_scale()[0]["zero_point"] # type: ignore
            bits_dict = self.get_ops_setting()['setting']['bits_dict']
            bit_select = self.get_ops_setting()['setting']['bit_select']
            dtype = bits_dict[bit_select]
            if self.get_scale_type() == "table":
                table = self.get_ops_instance().get_table() # type: ignore
            else:
                table = []
            input_scale = onnx.numpy_helper.from_array(np.array([input_scale], dtype=np.float32).squeeze())
            input_zero_point = onnx.numpy_helper.from_array(np.array([input_zero_point], dtype=np.int32).squeeze())
            output_scale = onnx.numpy_helper.from_array(np.array([output_scale], dtype=np.float32).squeeze())
            output_zero_point = onnx.numpy_helper.from_array(np.array([output_zero_point], dtype=np.int32).squeeze())
            table = onnx.numpy_helper.from_array(np.array(table, dtype=dtype))
            qparam = dict(
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                output_scale=output_scale,
                output_zero_point=output_zero_point,
                table=table,
            )
            attrs.update(qparam)
            nodes.append(
                self.create_node(
                    self.get_nodes()[0].get_op_type(),
                    inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                    outputs=self.get_nodes()[0].get_onnx_output(),
                    attrs=attrs,
                    node_name=self.get_nodes()[0].get_name(),
                    domain=domain,
                )
            )
        else:
            if len(self.get_nodes()) == 1:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[0].get_op_type(),
                        inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=self.get_nodes()[0].get_attr(),
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
            elif len(self.get_nodes()) == 2:
                nodes.append(
                    self.create_node(
                        self.get_nodes()[0].get_op_type(),
                        inputs=[self.get_nodes()[0].get_onnx_input()[0]],
                        outputs=self.get_nodes()[0].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[0].get_name(),
                        domain=domain,
                    )
                )
                nodes.append(
                self.create_node(
                        self.get_nodes()[1].get_op_type(),
                        inputs=self.get_nodes()[1].get_onnx_input(),
                        outputs=self.get_nodes()[1].get_onnx_output(),
                        attrs=dict(),
                        node_name=self.get_nodes()[1].get_name(),
                        domain=domain,
                    )
                )
            else:
                raise Exception("Not implemented !!!")

        return nodes, initializers

    def export_onnx(self):
        # op_type = self.get_layer_type()
        op_type = 'HardSigmoid'
        layer_name = self.get_layer_name()
        in_names = copy.deepcopy(self.get_onnx_input_name())
        attrs = self.get_ops_setting()['attrs'][0]
        in_names = [name + '_s' if name in self.get_inputs_names() else name for name in in_names]
        out_names = self.get_onnx_output_name()
        process_scale = self.get_ops_setting()['setting']['process_scale']
        nodes, initializers = \
            getattr(self, 'export_' + 'float')(op_type, layer_name, in_names, out_names, attrs=attrs)
        return nodes, initializers

    # def forward(self, in_data):
    #     super(Hsigmoid, self).forward(in_data)
# if __name__ == '__main__':
#     layer = LAYER
#     print('test')
