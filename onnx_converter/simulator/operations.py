# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/9/30 15:48
# @File     : operations.py

import os
import torch
import onnx
import copy
import numpy as np
import onnxruntime as rt
from abc import abstractmethod
import torch.nn.functional as F
import torch.nn as nn
from onnx.defs import ONNX_DOMAIN, AI_ONNX_PREVIEW_TRAINING_DOMAIN

import onnx
import onnxruntime as rt

try:
    from quantizer import DATACORRECT as datacorrect_factoy
    from utils import Registry, invert_dict, get_same_padding, shift_data, scale_data, extract_scale
    from utils import extract_scale, clip_values, shift_1d
    from extension import pyops
except:
    from onnx_converter.quantizer import DATACORRECT as datacorrect_factoy # type: ignore
    from onnx_converter.utils import Registry, invert_dict, get_same_padding, shift_data, scale_data, extract_scale # type: ignore
    from onnx_converter.utils import extract_scale, clip_values, shift_1d # type: ignore
    from onnx_converter.extension import pyops # type: ignore

np.set_printoptions(4)
int_shift_abs = True

np.set_printoptions(4)
# init_shift_abs: 1--fp32-simu-int, 2--data>>int_scale if data > 0 else data>>-int_scale
# other value: the same as 2

int_shift_abs = 0#[0,1,2]
OPERATORS: Registry = Registry('ops', scope='')

select_type = lambda values, value: values.index(value)
# select_op = lambda x: not isinstance(x, np.ndarray) and x == 0
select_op = lambda x: not isinstance(x, np.ndarray) and x == 0
bits_dict = {0: 'np.uint8', 1: 'np.int8', 2: 'np.uint16', 3: 'np.int16', 4: 'np.uint32',
             5: 'np.int32', 6: 'np.uint64', 7: 'np.int64', 8: 'np.float32', 9: 'np.float64'}
maxs = {0: 255, 1: 127, 2: 65535, 3: 32767, 4: 4294967295, 5: 2147483647, 6: 1844674407370955161,
        7: 9223372036854775807}
mins = {0: 0, 1: -128, 2: 0, 3: -32768, 4: 0, 5: -2147483648, 6: 0, 7: -9223372036854775808}


# per-channel will using input/weight/output quantized parameter
# self.bits_dict = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: uint64, 7: int64}
class BaseOps(object):
    # todo mix-up quantize multi-input will be alignment, maybe they has different length of data
    def __init__(self, **kwargs):
        super(BaseOps, self).__init__()
        self.bits_dict = kwargs.get("bits_dict", bits_dict)
        self.maxs = kwargs.get("maxs", maxs)
        self.mins = kwargs.get("mins", mins)
        self.process_scale = kwargs.get("process_scale", "smooth")
        self.bit_select = kwargs.get("bit_select", 1)
        self.w_bit_select = kwargs.get("w_bit_select", 1)
        self.int_scale = kwargs.get("int_scale", 8)
        self.virtual_round = kwargs.get("virtual_round", 3)
        self.scale_percent = 1.0 if self.virtual_round != 2 else 0.9
        if self.process_scale in ["float", "floatscale", "ffloatscale"]:
            self.precision = 1
        else:
            self.precision = 0

        self.out_type = self.bits_dict[kwargs.get('out_type', 1)]
        self.datacorrect = datacorrect_factoy.get(self.process_scale)(**kwargs)
        self.scales = self.datacorrect() # type: ignore
        # self.zi, self.zk, self.zo = self.scales['zi'], self.scales['zk'], self.scales['zo']
        self.si = kwargs.get("si", [{"scale": 1.0, "zero_point": 0}])
        self.so = kwargs.get("so", [{"scale": 1.0, "zero_point": 0}])
        self.sk = kwargs.get("sk", {"scale": 1.0, "zero_point": 0})
        self.setting = {'bits_dict': self.bits_dict, 'maxs': self.maxs, 'mins': self.mins,
                        'bit_select': self.bit_select, 'int_scale': self.int_scale}
        assert "in_quantize" in kwargs.keys()
        assert "quantize" in kwargs.keys()
        self.in_quantize = kwargs['in_quantize']
        self.out_quantize = kwargs['quantize']
        self.txme_saturation = kwargs.get('txme_saturation', 8)
        bit_num = max(self.bit_select, self.w_bit_select)
        self.bit_saturation = self.extract_bit(self.bits_dict[bit_num])
        self.bit_saturation = self.bit_saturation + 2 if not self.txme_saturation else self.bit_saturation

        self.table = []
        self.eps = np.float32((2 + 1e-3) / 2)
        # self.eps = np.float32(1)
        self.extra_value = 0
        self.table, self.out_shift, self.out_scale = [], None, None

        self.ir_version_default, self.opset_version = 8, 15
    
    def update_datacorrect(self, **kwargs):
        self.si = kwargs.get("si", self.si)
        self.so = kwargs.get("so", self.so)
        self.sk = kwargs.get("sk", self.sk)
        self.scales = self.datacorrect.update_quantize(**kwargs) # type: ignore
    
    def reset_correct(self, correct):
        self.process_scale = correct
        self.scales = datacorrect_factoy.get(self.process_scale)(**self.args)() # type: ignore

    def get_precision(self):
        return self.precision

    def get_out_shift(self):
        return self.out_shift

    def get_out_scale(self):
        return self.out_scale

    def get_table(self):
        return self.table

    def get_class_name(self):
        return self.__class__.__name__

    def update_scale(self, scale):
        pass

    def get_scales(self):
        return self.scales
    
    def set_scales(self, scales):
        self.scales = scales
        
    def get_quantize(self):
        return [self.in_quantize, self.out_quantize]
    
    def set_quantize(self, quantize: dict):
        self.in_quantize = quantize["in_quantize"]
        self.out_quantize = quantize["out_quantize"]
    
    def get_datacorrect(self):
        return self.datacorrect
    
    def set_datacorrect(self, datacorrect):
        self.datacorrect = datacorrect

    def get_bit_saturation(self):
        return self.bit_saturation

    def dequan(self, inputs, in_quantize: object):
        in_data, output = inputs['output'], list()
        if isinstance(in_quantize, list):
            for idx in len(in_quantize): # type: ignore
                output.append(in_quantize.get_dequan_data(in_data[idx])) # type: ignore
        else:
            output = in_quantize.get_dequan_data(in_data) # type: ignore
        return output

    def quan(self, inputs, quantize):
        in_data, output = inputs['output'], list()
        if isinstance(quantize, list):
            for idx in len(quantize): # type: ignore
                output.append(quantize.get_dequan_data(in_data[idx])) # type: ignore
        else:
            output = quantize.get_dequan_data(in_data)
        return output

    def clip(self, data):
        min_v, max_v = self.mins[self.bit_select], self.maxs[self.bit_select]
        data = np.clip(data, min_v, max_v)
        return data

    def align_bits(self, data):
        min_v, max_v = self.mins[self.bit_select], self.maxs[self.bit_select]
        dtype = self.bits_dict[self.bit_select]
        output = np.clip(data, min_v, max_v)
        return output.astype(dtype)

    @staticmethod
    def clip_int_bit(data, bit=9):
        base_num = 2 ** (bit - 1)
        min_v, max_v = -base_num, base_num - 1
        return np.clip(data, min_v, max_v)

    @staticmethod
    def clip_uint_bit(data, bit=9):
        base_num = 2 ** bit
        min_v, max_v = 0, base_num - 1
        return np.clip(data, min_v, max_v)

    # @abstractmethod
    def preprocess(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            in_data = copy.deepcopy(inputs['output'])
        else:
            in_data = copy.deepcopy(inputs)
        return dict(output=in_data)

    @abstractmethod
    def forward(self, inputs, **kwargs):
        pass

    @staticmethod
    def extract_bit(dtype):
        return np.int8(dtype.__name__.split('int')[-1])

    @staticmethod
    def upgrade_type(data):
        if isinstance(data, list):
            data = data[0]

        if data.dtype.type in [np.uint8, np.int8]:
            output = data.astype(np.int16)
        elif data.dtype.type in [np.uint16, np.int16]:
            output = data.astype(np.int32)
        # elif data.dtype.type in [np.uint32, np.int32]:
        #     output = data.astype(np.int64)
        else:
            output = data
        return output

    @staticmethod
    def high_bits_calc(bit_select):
        # for example: Qconv(int4, int4) -> Res(int16)
        # for example: Qconv(int8, int8) -> Res(int32)
        # for example: Qconv(int8, int16) -> Res(int64)
        # for example: Qconv(int16, int8) -> Res(int64)
        # for example: Qconv(int16, int16) -> Res(int64)
        return (bit_select // 2 * 2) + 5

    @staticmethod
    def lower_bits_calc(bit_select):
        # for example: Qconv(int4, int4) -> Res(int8)
        # for example: Qconv(int8, int8) -> Res(int16)
        # for example: Qconv(int16, int16) -> Res(int32)
        return (bit_select // 2 * 2) + 3

    @staticmethod
    def boardcast(out0, out1):
        shape_diff = len(out0.shape) - len(out1.shape)
        if shape_diff > 0:
            for i in range(shape_diff):
                if isinstance(out1, np.ndarray):
                    out1 = np.expand_dims(out1, axis=-1)
                else:
                    out1 = torch.unsqueeze(out1, -1)
        else:
            for i in range(-shape_diff):
                if isinstance(out1, np.ndarray):
                    out0 = np.expand_dims(out0, axis=-1)
                else:
                    out0 = torch.unsqueeze(out0, -1)
        return out0, out1

    @staticmethod
    def int_mapping_f(data, int_scale, virtual_round, eps):
        if int_scale != 0:
            test_int_scale = np.float32(2 ** (-int_scale))
            output = data * (test_int_scale * eps)
            if virtual_round:
                output = np.round(output)
        else:
            output = data
        return output

    @staticmethod
    def neg_mapping_pos(data, int_scale, virtual_round):
        # virtual_round = 0
        output = copy.deepcopy(data)
        if int_scale != 0:
            extra_value = 2**(int_scale - 1)
            if virtual_round == 1:
                # extra_value = 2**(int_scale - 1)
                output[output > 0] = output[output > 0] + extra_value
                output[output < 0] = output[output < 0] - extra_value
            elif virtual_round == 2:
                # extra_value = 2**(int_scale - 1)
                output = output + extra_value
                # output = output >> int_scale
            else:
                pass
                # output = output >> int_scale
            if int_shift_abs:
                output = (np.abs(output) >> int_scale)
                output[data < 0] = -output[data < 0]
            else:
                output = output >> int_scale

        return output

    @staticmethod
    def int_shift(data, out_scale, extra_value_, int_scale, virtual_round, scale_percent):
        # virtual_round = 2
        output = copy.deepcopy(data)
        if int_scale != 0:
            extra_value = data.dtype.type(extra_value_ * out_scale)
            if scale_percent < 1 and out_scale > scale_percent * 2**int_scale:
                extra_value += data.dtype.type(1 / np.float32(out_scale / 2**int_scale))
            if isinstance(extra_value, np.ndarray):
                if len(output.shape) == 2:
                    extra_value = extra_value.reshape(1,-1)
                elif len(output.shape) == 3:
                    extra_value = extra_value.reshape(1,-1,1)
                elif len(output.shape) == 4:
                    extra_value = extra_value.reshape(1,-1,1,1)
                else:
                    pass
            if virtual_round == 1:
                # extra_value = 2**(int_scale - 1)
                output[output > 0] = output[output > 0] + extra_value
                output[output < 0] = output[output < 0] - extra_value
                # output = np.abs(output) >> int_scale
                # output[data < 0] = -output[data < 0]
            elif virtual_round in [2, 3]:
                # extra_value = 2**(int_scale - 1)
                output = output + extra_value
                # output = output >> int_scale
            else:
                pass
                # output = output >> int_scale
            if int_shift_abs:
                output = np.abs(output) >> int_scale
                output[data < 0] = -output[data < 0]
            else:
                output = output >> int_scale

        return output

    @staticmethod
    def process_shift(data, shift_num, out_scale, int_scale, virtual_round, int_shift_abs, scale_percent):
        out = copy.deepcopy(data)
        if shift_num != 0:
            extra_value = data.dtype.type(2**(shift_num - 1) if shift_num > 0 else 2**(-shift_num - 1))
            if scale_percent < 1 and out_scale > scale_percent * 2**int_scale:
                extra_value += data.dtype.type(1 / np.float32(out_scale / 2**int_scale))

            if virtual_round == 1:
                out[out > 0] = out[out > 0] + extra_value
                out[out < 0] = out[out < 0] - extra_value
            elif virtual_round in [2, 3]:
                # out= out + extra_value
                pass
            else:
                pass
            if int_shift_abs:
                out = (np.abs(out) >> -shift_num) if shift_num < 0 else (out << shift_num)
                out[data < 0] = -out[data < 0]
            else:
                out = (out >> -shift_num) if shift_num < 0 else (out << shift_num)
        return out

    @staticmethod
    def create_in_out(name, shape):
        return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) # type: ignore

    @staticmethod
    def create_initializer(data, name):
        return onnx.helper.make_tensor( # type: ignore
            name=name, data_type=onnx.TensorProto.FLOAT, # type: ignore
            dims=data.shape, vals=data.tobytes(), raw=True) # type: ignore

    # @abstractmethod
    def postprocess(self, inputs, **kwargs):
        in_data, zero_point = inputs['output'], self.scales['zo']
        out_shift, out_scale = self.scales['out_shift'], self.scales['out_scale']
        output = self.Intscale(in_data)
        output = output + zero_point  # tspe
        return dict(output=output, out_shift=out_shift, out_scale=out_scale)

    # just calculation for no weight layer
    def Intscale(self, data, valid_key=None):
        if self.scales['out_scale'] == 1:
            return data
        output = copy.deepcopy(data)
        # output = self.upgrade_type(output)
        # dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
        # output = output.astype(dtype)
        # if isinstance(self.scales['out_scale'], np.ndarray) and \
        #         self.scales['out_scale'].dtype.type in [np.float32, np.float64]:
        #     output = output.astype(self.scales['out_scale'].dtype)
        output = output * self.scales['out_scale']
        if valid_key is None:
            valid_key = ['intscale', 'preintscale', 'smooth', 'preintscaleex']

        if self.process_scale in valid_key:
            int_scale = self.scales['int_scale'] if 'int_scale' in self.scales.keys() else self.int_scale

            if int_shift_abs == 1:
                output = self.int_mapping_f(output, int_scale, self.virtual_round, self.eps)
            # elif int_shift_abs == 2:
            #     output = self.neg_mapping_pos(output, int_scale, self.virtual_round)
            else:

                out_scale, extra_value = self.scales['out_scale'], self.scales['extra_value']
                output = self.int_shift(output, out_scale, extra_value, int_scale, self.virtual_round, self.scale_percent)
        if self.process_scale in ['floatscale', 'float']:
            output = np.round(output)
        return output.astype(data.dtype)

    # just calculation for no weight layer
    def MultiIntscale(self, data, idx=0, valid_key=None):

        if self.scales[idx]["out_scale"] == 1:
            return data

        output = copy.deepcopy(data)
        # output = self.upgrade_type(output)
        # dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
        # output = output.astype(dtype)
        if valid_key is None:
            valid_key = ['intscale', 'preintscale', 'preintscaleex']
        # if isinstance(self.scales[idx]['out_scale'], np.ndarray) and \
        #         self.scales[idx]['out_scale'].dtype.type in [np.float32, np.float64]:
        #     output = output.astype(self.scales[idx]['out_scale'].dtype)
        output = output * self.scales[idx]['out_scale']
        if self.process_scale in valid_key:
            int_scale = self.scales[idx]['int_scale'] if 'int_scale' in self.scales[idx].keys() else self.int_scale
            if int_shift_abs == 1:
                output = self.int_mapping_f(output, int_scale, self.virtual_round, self.eps)
            # elif int_shift_abs == 2:
            #     output = self.neg_mapping_pos(output, int_scale, self.virtual_round)
            else:
                out_scale, extra_value = self.scales[idx]['out_scale'], self.scales[idx]['extra_value']
                output = self.int_shift(output, out_scale, extra_value, int_scale, self.virtual_round, self.scale_percent)
            output = output.astype(data.dtype)
        if self.process_scale in ['floatscale', 'float']:
            output = np.round(output)
        return output

    @staticmethod
    def weight_insert_into_inputs(inputs, p_weights, weight_idx):
        inputs = [inputs] if isinstance(inputs, dict) else inputs
        for i, idx in enumerate(weight_idx):
            inputs.insert(idx, dict(output=p_weights[i]))
        return inputs

    def __call__(self, inputs, **kwargs):
        if 1:
            outputs = self.preprocess(inputs, **kwargs)
            outputs = self.forward(outputs, **kwargs)
            return self.postprocess(outputs, **kwargs)
        # except:
        #     error_info = "operation of {} simulation wrong!".format(self.get_class_name())
        #     print(error_info)
        #     os._exit(-1)


class BaseTable(BaseOps):
    def __init__(self, **kwargs):
        super(BaseTable, self).__init__(**kwargs)

        if self.process_scale in ['shiftfloatscaletable', 'shiftfloatscaletable2float']:
            setting = copy.deepcopy(self.setting)
            self.conv_quantize = kwargs.get("conv_quantize")
            if getattr(self, "conv_quantize"):
                so = dict(zip(["scale", "zero_point"], self.conv_quantize.get_scale())) # type: ignore
            else:
                so = self.so
            setting.update(dict(sk=self.sk, si=self.si, so=so))
            self.datacorrect = datacorrect_factoy.get(self.process_scale)(**setting)
            self.scales = self.datacorrect() # type: ignore

    def activation(self, x):
        return x

    def init_table(self, **kwargs):
        if self.process_scale in ['table', 'shiftfloatscaletable', 'shiftfloatscaletable2float']:
            if self.isolated:
                bit_num = self.extract_bit(self.bits_dict[self.bit_select])
                self.lower_bound = -2 ** (bit_num - 1)
                self.upper_bound = 2 ** (bit_num - 1) - 1
            else:
                self.lower_bound = -2 ** (self.bit_saturation - 1)
                self.upper_bound = 2 ** (self.bit_saturation - 1) - 1
            self.table = self.calc_table()
            self.table_align()

    def table_align(self):
        if self.process_scale not in ['shiftfloatscaletable2float']:
            self.table[self.table < self.mins[self.bit_select]] = self.mins[self.bit_select]
            self.table[self.table > self.maxs[self.bit_select]] = self.maxs[self.bit_select]
            self.table = self.table.astype(self.bits_dict[self.bit_select])
        else:
            pass

    def get_act_table(self, si, so, zi, zo):
        table = np.arange(self.lower_bound, self.upper_bound+1)
        table = torch.from_numpy(table).type(torch.float32)
        if isinstance(si, np.ndarray):
            table = table.expand(si.shape[0], table.shape[0]).permute(1, 0) # type: ignore
        else:
            table = table.expand(1, table.shape[0]).permute(1, 0)

        table = (table - zi) * si
        table = self.activation(table).numpy()

        if self.process_scale == 'shiftfloatscaletable2float':
            return table
        else:
            return np.round(table / so + zo)

    def calc_table(self, idx=0):
        zo = self.out_quantize.get_scale()[1]
        so = extract_scale(self.so)

        if self.process_scale in ['shiftfloatscaletable', 'shiftfloatscaletable2float']:
            si = self.scales['out_scale']
            zi = 0
            # if getattr(self, "conv_quantize"):
            #     # zi = copy.deepcopy(self.conv_quantize.get_scale()[1]) # type: ignore
            #     zi = 0
            # else:
            #     zi = self.so[0]["zero_point"]
        else:
            si = extract_scale(self.si) # type: ignore
            zi = self.scales['zi']

        return self.get_act_table(si, so, zi, zo)

    @staticmethod
    def lookup(in_data, table, lower_bound):
        output = np.zeros_like(in_data, dtype=table.dtype)
        if table.shape[1] == 1:
            output = table[in_data - lower_bound, 0]
        else:
            for ch in range(in_data.shape[1]):
                output[:, ch] = table[in_data[:, ch] - lower_bound, ch]
        return output


@OPERATORS.register_module(name='default')
class DefaultOps(BaseOps):
    def __init__(self, **kwargs):
        super(DefaultOps, self).__init__(**kwargs)
        self.ops = kwargs["ops"][0]
        self.kwargs = kwargs

    def forward(self, inputs, **kwargs):
        output = inputs['output']
        output = self.kwargs["in_quantize"][0].get_dequan_data(output)

        if self.ops == "leakyrelu":
            output = F.leaky_relu(torch.from_numpy(output.astype(np.float32)), self.kwargs["alpha"]).numpy()
        elif self.ops == "sigmod":
            output = F.sigmoid(torch.from_numpy(output.astype(np.float32))).numpy()
        else:
            pass

        output = self.kwargs["out_quantize"].get_quan_data(output)
        return dict(output=output)


@OPERATORS.register_module(name='data')
class Data(BaseOps):
    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)
        # self.datacorrect = datacorrect_factoy.get(self.process_scale)(**kwargs) # type: ignore
        self.test_op = True
        # self.shape = kwargs['shape']

    # def preprocess(self, **kwargs):
    #     pass

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']

        return dict(output=in_data)

    def postprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])
        # in_data = self.upgrade_type(in_data)
        # output = (output * 255).astype(np.int16)
        # if self.test_op:
        #     self.scales_op.so = np.float32(255/self.scales['out_scale'])
        #     self.scales = self.scales_op.update_scale()
        #     self.test_op = False

        out_shift, out_scale = self.scales['out_shift'], self.scales['out_scale']
        # if isinstance(self.scales['out_scale'], np.ndarray) and \
        #         self.scales['out_scale'].dtype.type in [np.float32, np.float64]:
        #     in_data = in_data.astype(self.scales['out_scale'].dtype)
        isp_data = kwargs.get("isp_data", False)
        if self.out_type not in [np.float32, np.float64] and not isp_data:
            if self.process_scale in ['floatscale', 'preintscale', 'preintscaleex']:
                zi, zo = self.scales["zi"], self.scales['zo']
                output = output * out_scale
                if self.process_scale in ["floatscale"]:
                    output = np.round(output)
                else:
                    int_scale = self.scales['int_scale'] if 'int_scale' in self.scales.keys() else self.int_scale
                    out_scale, extra_value = self.scales['out_scale'], self.scales['extra_value']
                    dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
                    output = self.int_shift(output.astype(dtype), out_scale, extra_value, int_scale, self.virtual_round, self.scale_percent)
                output = self.align_bits(output + zo)  # tspe
            elif self.process_scale in ["smooth"]:
                '''inputs must be int8 value'''
                pass
            else:
                output = self.out_quantize.get_quan_data(data=output)

        return dict(output=output, out_shift=out_shift, out_scale=out_scale)


@OPERATORS.register_module(name='conv')
@OPERATORS.register_module(name='depthwiseconv')
class Conv2d(BaseOps):
    # layer ops_nodes attribute
    # after quantize
    def __init__(self, **kwargs):
        super(Conv2d, self).__init__(**kwargs)
        self.stride = kwargs['strides']
        self.dilation, self.group, self.pads = [1, 1], 1, (0, 0, 0, 0)
        self.kernel_shape = kwargs['kernel_shape']
        if 'dilations' in kwargs.keys(): self.dilation = kwargs['dilations']
        if 'group' in kwargs.keys(): self.group = kwargs['group']
        if 'pads' in kwargs.keys(): self.pads = kwargs['pads']
        self.weights = kwargs['p_weights']
        self.p_weights = torch.from_numpy(np.array(self.weights, dtype=np.float32))
        if isinstance(self.scales['zk'], np.ndarray):
            _, zk = self.boardcast(self.p_weights, self.scales['zk'])
            self.p_weights -= torch.from_numpy(zk)
        else:
            self.p_weights -= self.scales['zk']
        if 'pads' in kwargs.keys():
            pads = kwargs['pads'] ### pad_t, pad_l, pad_b, pad_r
            self.pads = [pads[1], pads[3], pads[0], pads[2]]
        self.auto_pad = kwargs.get("auto_pad")
        self.is_first_conv = kwargs.get("is_first_conv")

    def preprocess(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs['output'])
        if in_data.dtype in [np.float32, np.float64]:
            in_data = self.in_quantize[0].get_quan_data(in_data)
        p_inputs = np.array(in_data, dtype=np.float32)
        # p_weights = np.array(self.weights, dtype=np.float32)
        return dict(p_inputs=p_inputs)

    # def fuse_zero_point(self, in_data, w_conv, dtype):
    #     n, c,h,w = in_data.shape
    #     shape = (n, -1, 1, 1)
    #     fuse_in_zero_point = torch.sum(w_conv, dim=(1, 2, 3)).view(shape) * self.zi
    #     f = lambda zi, zk, dtype: self.zi.astype(dtype) * self.zk.astype(dtype)
    #     fuse_zi_zk = f(self.zi, self.zk, dtype) * w_conv.shape[1] * w_conv.shape[2] * w_conv.shape[3]
    #     zk_f = torch.ones_like(w_conv)
    #     if isinstance(self.zk, np.ndarray):
    #         fuse_zi_zk = fuse_zi_zk.reshape(shape)
    #         zk_f *= torch.from_numpy(self.zk.astype(dtype).reshape(-1, 1, 1, 1))
    #     else:
    #         zk_f *= self.zk
    #     fuse_w_zero_point = F.conv2d(input=in_data, weight=zk_f, bias=None, stride=tuple(self.stride),
    #                                  padding=(0, 0), dilation=tuple(self.dilation), groups=self.group)  # bias
    #     fused_zero_point = (-fuse_in_zero_point - fuse_w_zero_point + fuse_zi_zk).numpy()
    #
    #     return fused_zero_point

    def forward(self, inputs, **kwargs):
        #
        in_data = torch.from_numpy(inputs['p_inputs'])
        in_data -= self.scales['zi']

        if self.auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            input_shape = in_data.shape
            pad_h = get_same_padding(input_shape[2], self.kernel_shape[0], self.stride[0], auto_pad=self.auto_pad)# type: ignore
            pad_w = get_same_padding(input_shape[3], self.kernel_shape[1], self.stride[1], auto_pad=self.auto_pad)# type: ignore
            self.pads = pad_h + pad_w
        elif self.auto_pad in ["VALID"]:
            self.pads = [0, 0, 0, 0]
        isp_data = kwargs.get("isp_data", False)
        pads = copy.deepcopy(self.pads) ### [pad_l, pad_r, pad_t, pad_b]
        if self.is_first_conv:
            pads = [pads[0], pads[1], 0, 0]

        padding = nn.ZeroPad2d(tuple(pads))
        in_data = padding(in_data)
        # if isinstance(self.scales['zk'], np.ndarray):
        #     _, zk = self.boardcast(inputs['p_weights'], copy.deepcopy(self.scales['zk']))
        #     w_conv -= torch.from_numpy(zk)
        # else:
        #     w_conv -= self.scales['zk']
        output = F.conv2d(input=in_data, weight=self.p_weights, bias=None, stride=tuple(self.stride),
                          padding=(0, 0), dilation=tuple(self.dilation), groups=self.group)
        # dtype = inputs['p_inputs'].dtype
        # if [self.zi, self.zk, self.zo] != [0, 0, 0]:
        #     fused_zero_point = self.fuse_zero_point(in_data, w_conv, dtype)
        # else:
        #     fused_zero_point = dtype.type(0)
        output = output.detach().numpy()
        return dict(output=output, out_shift=1, out_scale=1)

    def postprocess(self, inputs, **kwargs):
        output = inputs['output']
        # consider uint4/int4 select upgrade data type to calc
        bit_num = max(self.bit_select, self.w_bit_select)
        dtype = self.bits_dict[self.high_bits_calc(bit_num)]
        # output = np.array(output, np.int64) if output.dtype.type == np.float64 else np.array(output, np.int32)
        # fused_zero_point = np.int64(fused_zero_point) if output.dtype.type == np.float64 else np.int32(fused_zero_point)
        return dict(output=output.astype(dtype))

    
@OPERATORS.register_module(name='convtranspose')
class ConvTranspose2d(BaseOps):
    # layer ops_nodes attribute
    # after quantize
    def __init__(self, **kwargs):
        super(ConvTranspose2d, self).__init__(**kwargs)
        self.stride = kwargs['strides']
        self.dilation, self.group, self.pads = [1, 1], 1, (0, 0, 0, 0)
        self.kernel_shape = kwargs['kernel_shape']
        if 'dilations' in kwargs.keys(): self.dilation = kwargs['dilations']
        if 'group' in kwargs.keys(): self.group = kwargs['group']
        if 'pads' in kwargs.keys(): self.pads = kwargs['pads']
        if 'output_padding' in kwargs.keys(): self.output_padding = kwargs['output_padding']
        self.weights = kwargs['p_weights']
        self.p_weights = torch.from_numpy(np.array(self.weights, dtype=np.float32))
        if isinstance(self.scales['zk'], np.ndarray):
            _, zk = self.boardcast(self.p_weights, self.scales['zk'])
            self.p_weights -= torch.from_numpy(zk)
        else:
            self.p_weights -= self.scales['zk']
        if 'pads' in kwargs.keys():
            pads = kwargs['pads']
            self.pads = [pads[1], pads[3], pads[0], pads[2]]
        # self.weights = 0

        self.sess = self.get_session(
            attrs=kwargs,
            weight=self.p_weights.numpy().astype(np.float32),
            bias=None,
            opset_version=14,
        )

    def get_session(self, attrs, weight, bias, opset_version=14):
        def create_initializer(data, name): return onnx.helper.make_tensor(# type: ignore
            name=name, data_type=onnx.TensorProto.FLOAT,# type: ignore
            dims=data.shape, vals=data.tobytes(), raw=True)# type: ignore
        def create_in_out(name, shape): return onnx.helper.make_tensor_value_info(# type: ignore
            name, onnx.TensorProto.FLOAT, shape)# type: ignore

        in_c = attrs["in_c"]
        out_c = attrs["out_c"]
        kernel_shape = attrs["kernel_shape"]
        strides = attrs["strides"]
        pads = attrs["pads"]
        dilations = attrs["dilations"]
        if "output_padding" in attrs.keys():
            output_padding = attrs["output_padding"]
        else:
            output_padding = [0, 0]
            
        node = onnx.helper.make_node(# type: ignore
            "ConvTranspose", 
            inputs=["X", "W", "B"], 
            outputs=["Y"], 
            kernel_shape=kernel_shape,
            strides=strides, 
            pads=pads,
            dilations=dilations,
            output_padding=output_padding,
            # auto_pad="SAME_UPPER",
        )

        if isinstance(weight, np.ndarray):
            W = weight.astype(np.float32)
        else:
            W = np.random.randn([in_c, out_c, kernel_shape[0], kernel_shape[1]]).astype(np.float32)# type: ignore
        if isinstance(bias, np.ndarray):
            B = bias.astype(np.float32)
        else:
            B = np.zeros([out_c]).astype(np.float32)    

        initializers = [
            create_initializer(W, "W"),
            create_initializer(B, "B"),
        ]
        inputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["X"]]
        outputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["Y"]]

        graph = onnx.helper.make_graph(# type: ignore
            nodes=[node],
            name='test_convtranspose',
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            )
        opsets = [onnx.helper.make_operatorsetid(ONNX_DOMAIN, 12),
                  onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)]
        model = onnx.helper.make_model(graph, producer_name='backend-ConvTranspose2d', opset_imports=opsets)# type: ignore
        model.ir_version = self.ir_version_default
        # onnx.save(model, "test_convtranspose.onnx")

        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        return sess

    def preprocess(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs['output'])
        if in_data.dtype in [np.float32, np.float64]:
            in_data = self.in_quantize[0].get_quan_data(in_data)
        p_inputs = np.array(in_data, dtype=np.float32)
        # p_weights = np.array(self.weights, dtype=np.float32)
        return dict(p_inputs=p_inputs)

    # def fuse_zero_point(self, in_data, w_conv, dtype):
    #     n, c,h,w = in_data.shape
    #     shape = (n, -1, 1, 1)
    #     fuse_in_zero_point = torch.sum(w_conv, dim=(1, 2, 3)).view(shape) * self.zi
    #     f = lambda zi, zk, dtype: self.zi.astype(dtype) * self.zk.astype(dtype)
    #     fuse_zi_zk = f(self.zi, self.zk, dtype) * w_conv.shape[1] * w_conv.shape[2] * w_conv.shape[3]
    #     zk_f = torch.ones_like(w_conv)
    #     if isinstance(self.zk, np.ndarray):
    #         fuse_zi_zk = fuse_zi_zk.reshape(shape)
    #         zk_f *= torch.from_numpy(self.zk.astype(dtype).reshape(-1, 1, 1, 1))
    #     else:
    #         zk_f *= self.zk
    #     fuse_w_zero_point = F.conv2d(input=in_data, weight=zk_f, bias=None, stride=tuple(self.stride),
    #                                  padding=(0, 0), dilation=tuple(self.dilation), groups=self.group)  # bias
    #     fused_zero_point = (-fuse_in_zero_point - fuse_w_zero_point + fuse_zi_zk).numpy()
    #
    #     return fused_zero_point

    def forward(self, inputs, **kwargs):
        #
        in_data = torch.from_numpy(inputs['p_inputs'])
        in_data -= self.scales['zi']
        # padding = nn.ZeroPad2d(tuple(self.pads))
        # in_data = padding(in_data)
        # if isinstance(self.scales['zk'], np.ndarray):
        #     _, zk = self.boardcast(inputs['p_weights'], copy.deepcopy(self.scales['zk']))
        #     w_conv -= torch.from_numpy(zk)
        # else:
        #     w_conv -= self.scales['zk']

        output = self.sess.run(None, {"X": in_data.numpy().astype(np.float32)})[0]
        output = torch.from_numpy(output).type(torch.float32)
        
        # output = F.conv_transpose2d(input=in_data, weight=self.p_weights, bias=None, stride=tuple(self.stride),
        #                   padding=tuple(self.pads[:2]), dilation=tuple(self.dilation), 
        #                   output_padding=tuple(self.output_padding), groups=self.group)
        
        # dtype = inputs['p_inputs'].dtype
        # if [self.zi, self.zk, self.zo] != [0, 0, 0]:
        #     fused_zero_point = self.fuse_zero_point(in_data, w_conv, dtype)
        # else:
        #     fused_zero_point = dtype.type(0)
        output = output.detach().numpy()
        return dict(output=output, out_shift=1, out_scale=1)

    def postprocess(self, inputs, **kwargs):
        output = inputs['output']
        # consider uint4/int4 select upgrade data type to calc
        bit_num = max(self.bit_select, self.w_bit_select)
        dtype = self.bits_dict[self.high_bits_calc(bit_num)]
        # output = np.array(output, np.int64) if output.dtype.type == np.float64 else np.array(output, np.int32)
        # fused_zero_point = np.int64(fused_zero_point) if output.dtype.type == np.float64 else np.int32(fused_zero_point)
        return dict(output=output.astype(dtype))
    

class Conv1d(BaseOps):
    def __init__(self, **kwargs):
        super(Conv1d, self).__init__(**kwargs)


@OPERATORS.register_module(name='bias')
class BiasAdd(BaseOps):
    def __init__(self, **kwargs):
        super(BiasAdd, self).__init__(**kwargs)
        # process bias value to int32

        out_shift = self.scales["out_shift"]
        self.bias_ = kwargs['bias']
        if self.virtual_round in [2, 3]:
            out_scale = self.scales["out_scale"]
            if isinstance(out_shift, np.ndarray):
                for i, shift_num in enumerate(out_shift):
                    if shift_num == 0:
                        continue
                    extra_value = 2**(shift_num - 1) if shift_num > 0 else 2**(-shift_num - 1)
                    out_scale_ = out_scale[i] if isinstance(out_scale, np.ndarray) else out_scale
                    if self.scale_percent < 1 and out_scale_ > self.scale_percent * 2**self.int_scale:
                        extra_value += np.floor(1 / np.float32(out_scale_ / 2**self.int_scale))
                    self.bias_[i] += extra_value
            else:
                if out_shift != 0:
                    extra_value = 2**(out_shift - 1) if out_shift > 0 else 2**(-out_shift - 1)
                    if self.scale_percent < 1 and out_scale > self.scale_percent * 2**self.int_scale:
                        extra_value += np.floor(1 / np.float32(out_scale / 2**self.int_scale))
                    self.bias_ += extra_value

    def update_qbias(self, bias):
        self.bias_ = bias / (self.si[0]['scale'] * self.sk['scale']) + 0.5
    
    def preprocess(self, inputs, **kwargs):
        dtype = inputs['output'].dtype.type
        # fused_zero_point = inputs['fused_zero_point'] if 'fused_zero_point' in \
        #                                                  inputs.keys() else dtype(0)
        num_bits = invert_dict(self.bits_dict)
        min_value, max_value = self.mins[num_bits[dtype]], self.maxs[num_bits[dtype]]
        bias_ = np.clip(self.bias_, a_min=min_value, a_max=max_value)
        bias_ = dtype(bias_)
        # in_data = copy.deepcopy(inputs)
        return dict(output=inputs['output'], bias_=bias_)

    def forward(self, inputs, **kwargs):
        in_data, bias_ = inputs['output'], inputs['bias_']
        # fused_zero_point = inputs['fused_zero_point']
        batch, out_c = in_data.shape[:2]
        bias_ = np.array(bias_, dtype=in_data.dtype).reshape(1, out_c, -1)
        # bias_ = np.repeat(np.reshape(bias_, (1, out_c, 1)), batch, axis=0)
        # pytoch implement
        # bias_ = bias_.repeat(batch).unsqueeze(-1)
        output = in_data.reshape(batch, out_c, -1) + bias_
        return dict(output=output.reshape(in_data.shape))

    def postprocess(self, inputs, **kwargs):
        return inputs


@OPERATORS.register_module(name='act')
class Act(BaseTable):
    # fp/int scale
    def __init__(self, **kwargs):
        super(Act, self).__init__(**kwargs)
        # self.process_scale, self.bit_select = kwargs['process_scale'], kwargs['bit_select']
        # self.precision, self.int_scale = kwargs['precision'], kwargs['int_scale']
        # self.scales = datacorrect_factoy.get(self.process_scale)(bit_select=self.bit_select, **kwargs)
        # self.si, self.sk, self.so = kwargs['si'], kwargs['sk'], kwargs['so']
        self.bits_dict = kwargs['bits_dict']
        self.compare_value = 0
        self.isolated = kwargs['isolated']
        self.out_shift = self.scales['out_shift']
        self.out_scale = self.scales['out_scale']
        if isinstance(self.scales['out_shift'], np.ndarray):
            # self.scales['out_shift'] = torch.from_numpy(2 ** self.scales['out_shift'].astype(np.float32)).unsqueeze(0)
            self.out_shift = torch.from_numpy(
                2 ** self.scales['out_shift'].astype(np.float32)).unsqueeze(0)
        else:
            self.out_shift = np.float32(2 ** np.float32(self.scales['out_shift']))
        if isinstance(self.scales['out_scale'], np.ndarray):
            # self.scales['out_scale'] = torch.from_numpy(self.scales['out_scale'].astype(np.float32)).unsqueeze(0)
            self.out_scale = torch.from_numpy(
                self.scales['out_scale'].astype(np.float32)).unsqueeze(0)
        self.vaild_out_scale = ["floatscale", "rshiftscale", "rrshiftscale", "intscale",\
                                "intscaleex", "preintscale", "shiftfloatscale", "ffloatscale"]
        self.table_out_scale = ["shiftfloatscaletable", "shiftfloatscaletable2float", "table"]
        self.init_table(**kwargs)

    def shift_data(self, data, out_shift):
        out = copy.deepcopy(data)
        if int_shift_abs == 1:
            if isinstance(out_shift, np.ndarray) or isinstance(out_shift, torch.Tensor):
                # test_input = copy.deepcopy(data)
                if isinstance(out_shift, np.ndarray):
                    out_shift = torch.from_numpy(out_shift).unsqueeze(0)
                test_input, test_shift = self.boardcast(torch.from_numpy(data), out_shift)
                out = test_input * test_shift * self.eps
                if self.virtual_round:
                    # out = torch.round(test_input * test_shift).numpy() # .astype(data.dtype)
                    out = torch.round(out).numpy()
                else:
                    out = torch.floor(out).numpy() # .astype(data.dtype)
                # out = torch.floor(out).numpy()
            else:
                out = data * (out_shift * self.eps)
                if self.virtual_round:
                    out = np.round(out)
                else:
                    out = np.floor(out) # shift_data(data, out_shift)
                # out = np.floor(out)
        # elif int_shift_abs == 2:
        #     if isinstance(out_shift, np.ndarray) or isinstance(out_shift, torch.Tensor):
        #         for i in range(self.scales["out_shift"].shape[0]):
        #             shift_num = self.scales["out_shift"][i]
        #             out_scale = self.scales["out_scale"][i] if isinstance(self.scales["out_scale"][i], np.ndarray) else self.scales["out_scale"]
        #             out[:,i] = self.process_shift(out[:, i], shift_num, out_scale, self.int_scale, self.virtual_round, int_shift_abs, scale_percent)
        #     else:
        #         shift_num = self.scales["out_shift"]
        #         out = self.process_shift(out, shift_num, self.scales["out_scale"], self.int_scale, self.virtual_round, int_shift_abs, scale_percent)
        else:
            virtual_round = 0 if self.virtual_round > 1 else self.virtual_round
            if isinstance(out_shift, np.ndarray) or isinstance(out_shift, torch.Tensor):
                for i in range(self.scales["out_shift"].shape[0]):
                    shift_num = self.scales["out_shift"][i]
                    out_scale = self.scales["out_scale"][i] if isinstance(self.scales["out_scale"], np.ndarray) else self.scales["out_scale"]
                    out[:,i] = self.process_shift(out[:, i], shift_num, out_scale, self.int_scale, virtual_round, int_shift_abs, self.scale_percent)
            else:
                shift_num = self.scales["out_shift"]
                out = self.process_shift(out, shift_num, self.scales["out_scale"], self.int_scale,  virtual_round, int_shift_abs, self.scale_percent)

        return out.astype(data.dtype)

    def scale_data(self, data, out_scale):

        if not isinstance(out_scale, np.ndarray) and out_scale == 1:
            return data

        def setting_dtype(data):
            dtype = data.dtype
            if self.out_type in [np.float32, np.float64]:
                dtype = np.float32
            return dtype

        if isinstance(out_scale, np.ndarray) or isinstance(out_scale, torch.Tensor):
            test_out_scale = copy.deepcopy(out_scale).reshape(-1, out_scale.shape[0])
            if isinstance(out_scale, np.ndarray):
                test_out_scale = torch.from_numpy(test_out_scale)  # .unsqueeze(0)
            test_input = torch.from_numpy(data)
            _, test_scale = self.boardcast(test_input, test_out_scale)

            output = (test_input * test_scale).numpy()
        else:
            output = data * out_scale

        if self.process_scale in ['intscale', "intscaleex", 'preintscale', 'preintscaleex']:
            dtype = setting_dtype(data)
            int_scale = self.scales['int_scale'] if self.isolated else self.int_scale
            if int_shift_abs == 1:
                output = self.int_mapping_f(output, int_scale, self.virtual_round, self.eps)
                # output = self.int_shift(output.astype(dtype), int_scale, self.virtual_round)
            # elif int_shift_abs == 2:
            #     output = self.neg_mapping_pos(output.astype(dtype), int_scale, self.virtual_round)
            else:
                # output = self.int_shift(output.astype(dtype), int_scale, self.virtual_round)
                virtual_round = self.virtual_round if self.virtual_round != 2 else 0
                out_scale, extra_value = self.scales['out_scale'], self.scales['extra_value']
                output = self.int_shift(output.astype(dtype), out_scale, extra_value, int_scale,
                                        virtual_round, self.scale_percent)

            # output = output.astype(dtype)
            # output = output.astype(dtype)
        elif self.process_scale in ['smooth', 'rshiftscale', 'rrshiftscale']:
            # nothing calc in here
            pass
        elif self.process_scale in ['floatscale', 'shiftfloatscale', 'float']:
            output = np.round(output)
        else:
            assert self.process_scale in ['ffloatscale']
            output = output.astype(np.float32)

        return output

    # process no activation after convolution
    def preprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])

        if self.isolated:
            # output = sub_zero_point(output, self.scales, self.upgrade_type)
            if self.process_scale not in ["float"]:
                if output.dtype in [np.float32, np.float64]:
                    output = self.in_quantize[0].get_quan_data(output)
                dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
                if self.process_scale not in ["table"]:
                    output = output.astype(dtype) - self.scales['zi']
        else:
            if not self.precision:
                output = self.shift_data(output, self.out_shift)
                if self.bit_select % 2:
                    output = self.clip_int_bit(output, bit=self.bit_saturation) # type: ignore
                else:
                    output = self.clip_uint_bit(output, bit=self.bit_saturation) # type: ignore

        return dict(output=output)

    @staticmethod
    def post_shift(process_scale, output, out_type, out_scale, zero_point, out_quantize, scale_data, align_bits):
        output = scale_data(output, out_scale)
        if process_scale in ["intscale", "intscaleex", "shiftfloatscale", "floatscale"]:
            output = output + zero_point  # tspe
            if not out_type in [np.float32, np.float64]:
                output = align_bits(output)  # tspe
            else:
                if isinstance(out_quantize, list):
                    output = out_quantize[0].get_dequan_data(output)
                else:
                    output = out_quantize.get_dequan_data(output)
                output = output.astype(out_type)
        return output

    def forward(self, inputs, **kwargs):
        return inputs

    def tsme_actvation(self, output):
        output = self.activation(output)

    def float_activation(self, output):
        if output.dtype not in [np.float32, np.float64]:
            in_quantize = self.in_quantize[0] if isinstance(self.in_quantize, list) else\
                self.in_quantize
            output = in_quantize.get_dequan_data(output)
        output = self.activation(torch.from_numpy(output))
        if isinstance(output, torch.Tensor): output = output.numpy()
        if output.dtype in [np.float32, np.float64]:
            out_quantize = self.out_quantize[0] if isinstance(self.out_quantize, list) else\
                self.out_quantize
            output = out_quantize.get_quan_data(output)                  
        return output

    def postprocess(self, inputs, **kwargs):
        output, zero_point = inputs['output'], self.scales['zo']

        out_shift, out_scale = self.scales['out_shift'], self.scales['out_scale']

        if self.process_scale in self.vaild_out_scale:
            output = self.post_shift(self.process_scale, output, self.out_type, out_scale,
                                     zero_point, self.out_quantize, self.scale_data, self.align_bits)
        elif self.process_scale in ["rshiftscale"]:
            output = self.align_bits(output)
        elif self.process_scale in self.table_out_scale:
            output = self.lookup(output, self.table, self.lower_bound)
        else:
            # dequant->fp32-act->quant
            # process_scale in ["float", "ffloatscale"]
            output = self.float_activation(output)

        return dict(output=output, out_shift=out_shift, out_scale=out_scale)


@OPERATORS.register_module(name='relu')
class Relu(Act):
    def __init__(self, **kwargs):
        self.min_value = 0
        self.qmin_value = self.min_value
        super(Relu, self).__init__(**kwargs)

    def activation(self, x):
        x[x<self.min_value] = float(self.min_value)
        return x

    # def preprocess(self, **kwargs):
    #     pass

    def forward(self, inputs, **kwargs):
        output = inputs['output']
        # dtype = inputs['output'].dtype.type
        # fused_zero_point = inputs['fused_zero_point'] if 'fused_zero_point' in \
        #                                                  inputs.keys() else dtype(0)

        if self.process_scale in self.vaild_out_scale:
            output[output < self.min_value] = self.min_value
        # self.clip(output)
        return dict(output=output)

    # def postprocess(self, **kwargs):
    #     pass


@OPERATORS.register_module(name='relu6')
class Relu6(Relu):
    def __init__(self, **kwargs):
        self.max_value = kwargs.get("value", np.float32(6))
        self.qmax_value = self.max_value
        super(Relu6, self).__init__(**kwargs)
        # self.min_value = 0
        # self.si, self.sk, self.so = kwargs['si'], kwargs['sk'], kwargs['so']
        # self.bits_dict = kwargs['bits_dict']
        # self.type_max_values = kwargs['maxs']
        if self.process_scale not in ["float"]:
            self.upgrade_max_value()

    def activation(self, x):
        x[x<0] = 0
        x[x>self.max_value] = float(self.max_value)
        return x

    def upgrade_max_value(self):
        if isinstance(self.si, list):
            si = self.si[0]
        else:
            si = self.si
        self.qmax_value = np.round(self.max_value / (si['scale'] * self.sk['scale']))# type: ignore
        # out_shift = self.scales['out_shift']
        # if not isinstance(self.sk['scale'], np.ndarray):
        #     out_shift = np.float32(1/2**-out_shift) if out_shift < 0 else 2**out_shift
        if isinstance(self.out_shift, torch.Tensor):
            self.qmax_value = np.round((torch.from_numpy(self.qmax_value) * self.out_shift.squeeze(0)).numpy())
        else:
            self.qmax_value = np.round(self.qmax_value * self.out_shift)
        if not self.precision or self.isolated:
            if self.txme_saturation:
                self.qmax_value = np.clip(self.qmax_value, 0, self.maxs[self.bit_select])
            else:
                if self.bit_select % 2:
                    self.qmax_value = self.clip_int_bit(self.qmax_value, bit=self.bit_saturation)# type: ignore
                else:
                    self.qmax_value = self.clip_uint_bit(self.qmax_value, bit=self.bit_saturation)# type: ignore

    def forward(self, inputs, **kwargs):
        output = inputs['output']
        # out_shift, out_scale = self.scales['out_shift'], self.scales['out_scale']
        # if not self.precision:
        #     in_data = shift_data(in_data, out_shift)

        if self.process_scale in self.vaild_out_scale:
            output[output < self.qmin_value] = self.qmin_value
            clip_values(output, self.qmax_value.astype(output.dtype))

        return dict(output=output)


@OPERATORS.register_module(name='relux')
class Relux(Relu6):
    def __init__(self, **kwargs):
        super(Relux, self).__init__(**kwargs)


@OPERATORS.register_module(name='clip')
class Clip(Relux):
    def __init__(self, **kwargs):
        self.min_value = kwargs["min_value"]
        self.qmin_value = self.min_value
        super(Clip, self).__init__(**kwargs)

    def activation(self, x):
        x[x<self.min_value] = float(self.min_value)
        x[x>self.max_value] = float(self.max_value)
        return x

    def upgrade_max_value(self):
        super(Clip, self).upgrade_max_value()
        if isinstance(self.si, list):
            si = self.si[0]
        else:
            si = self.si
        self.qmin_value = np.round(self.min_value / (si['scale'] * self.sk['scale']))# type: ignore
        # out_shift = self.scales['out_shift']
        # if not isinstance(self.sk['scale'], np.ndarray):
        #     out_shift = np.float32(1/2**-out_shift) if out_shift < 0 else 2**out_shift
        if isinstance(self.out_shift, torch.Tensor):
            self.qmin_value = np.round((torch.from_numpy(self.qmin_value) * self.out_shift.squeeze(0)).numpy())
        else:
            self.qmin_value = np.round(self.qmin_value * self.out_shift)
        if not self.precision or self.isolated:
            min_v, max_v = self.mins[self.bit_select], self.maxs[self.bit_select]
            if self.txme_saturation:
                self.qmin_value = np.clip(self.qmin_value, min_v, max_v)
            else:
                if self.bit_select % 2:
                    self.qmin_value = self.clip_int_bit(self.qmin_value, bit=self.bit_saturation)# type: ignore
                else:
                    self.qmin_value = self.clip_uint_bit(self.qmin_value, bit=self.bit_saturation)# type: ignore

    def forward(self, inputs, **kwargs):
        output = inputs['output']
        # out_shift, out_scale = self.scales['out_shift'], self.scales['out_scale']
        # if not self.precision:
        #     in_data = shift_data(in_data, out_shift)

        if self.process_scale in self.vaild_out_scale:
            clip_values(output, self.qmin_value.astype(output.dtype))
            clip_values(output, self.qmax_value.astype(output.dtype))

        return dict(output=output)


@OPERATORS.register_module(name='sigmoid')
class Sigmoid(Act):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)
        # self.process = torch.sigmoid  # torch.nn.Sigmoid()
        # self.process = kwargs['process'] if 'process' in kwargs.keys() else self.process

        # if self.process_scale in ['table', 'shiftfloatscaletable']:
        #     if self.bit_select == 1:
        #         self.bit_saturation = 8
        #     else:
        #         self.bit_saturation = 16
        #     self.bit_saturation = kwargs.get("bit_saturation", self.bit_saturation)
        #     self.lower_bound = -2 ** (self.bit_saturation - 1)
        #     self.upper_bound = 2 ** (self.bit_saturation - 1) - 1
        #     self.table = self.get_table()
        #     self.table_align()

    @staticmethod
    def clip_int_bit(data, bit=9):
        base_num = 2 ** (bit - 1)
        min_v, max_v = -base_num, base_num - 1
        return np.clip(data, min_v, max_v)

    @staticmethod
    def clip_uint_bit(data, bit=9):
        base_num = 2 ** bit
        min_v, max_v = 0, base_num - 1
        return np.clip(data, min_v, max_v)


    def activation(self, x):
        return torch.sigmoid(x)

    def table_align(self):
        super(Sigmoid, self).table_align()
        if self.process_scale not in ['shiftfloatscaletable2float']:
            self.table[0] = 0
        else:
            self.table[0] = 0.0
    
    def forward(self, inputs, **kwargs):
        return super().forward(inputs, **kwargs)


@OPERATORS.register_module(name='gelu')
class GELU(Act):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def activation(self, x):
        return F.gelu(x)


@OPERATORS.register_module(name='tanh')
class Tanh(Act):
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def activation(self, x):
        return torch.tanh(x)


@OPERATORS.register_module(name='leakyrelu')
class LeakyRelu(Act):
    def __init__(self, **kwargs):
        self.alpha = kwargs['alpha']
        super(LeakyRelu, self).__init__(**kwargs)

    def activation(self, x):
        return F.leaky_relu(x, negative_slope=self.alpha)


@OPERATORS.register_module(name='prelu')
class PReLU(Act):
    def __init__(self, **kwargs):
        self.slope = kwargs['slope'].squeeze().astype(np.float32)
        if isinstance(self.slope, np.ndarray):
            self.slope = torch.from_numpy(self.slope)
        elif isinstance(self.slope, np.float32):# type: ignore
            self.slope = torch.Tensor([self.slope])
        super(PReLU, self).__init__(**kwargs)

    def activation(self, x):
        return F.prelu(x, weight=self.slope)

    def get_act_table(self, si, so, zi, zo):
        table = np.arange(self.lower_bound, self.upper_bound+1)
        table = torch.from_numpy(table).type(torch.float32)
        if isinstance(si, np.ndarray):
            table = table.expand(si.shape[0], table.shape[0]).permute(1, 0) # type: ignore
        else:
            table = table.expand(self.slope.shape[0], table.shape[0]).permute(1, 0)

        table = (table - zi) * si
        table = self.activation(table).numpy()
        return np.round(table / so + zo)

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']
        out_shift, out_scale = self.scales['out_shift'], self.scales['out_scale']

        if self.process_scale == 'float':
            if not in_data.dtype in [np.float32, np.float64]:
                if isinstance(self.in_quantize, list):
                    output = self.in_quantize[0].get_dequan_data(in_data)
                else:
                    output = self.in_quantize.get_dequan_data(in_data)
            else:
                output = copy.deepcopy(in_data)
            output = torch.from_numpy(output.astype(np.float32))
            output = self.activation(output).numpy()
            if not self.out_type in [np.float32, np.float64]:
                output = self.out_quantize.get_quan_data(output)
        elif self.process_scale == 'table':
            # look up table
            if in_data.dtype in [np.float32, np.float64]:
                if isinstance(self.in_quantize, list):
                    output = self.in_quantize[0].get_quan_data(in_data)
                else:
                    output = self.in_quantize.get_quan_data(in_data)
            else:
                output = in_data
            if self.table.shape[1] == 1:
                output = self.table[output - self.lower_bound, 0]
            else:
                for ch in range(output.shape[1]):
                    output[:, ch] = self.table[output[:, ch] - self.lower_bound, ch]
            output = output.astype(self.table.dtype)
        elif self.process_scale in ['intscale', 'preintscale', 'preintscaleex']:
            pass
        else:
            pass

        return dict(output=output, out_shift=out_shift, out_scale=out_scale)# type: ignore


@OPERATORS.register_module(name='swish')
class Swish(Act):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def activation(self, x):
        return x * torch.sigmoid(x)


@OPERATORS.register_module(name='hardsigmoid')
class Hsigmoid(Sigmoid):
    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.2)
        self.beta = kwargs.get("beta", 0.5)
        # kwargs['bit_saturation'] = 10
        super(Hsigmoid, self).__init__(**kwargs)

    def activation(self, x):
        return torch.clamp(self.alpha * x + self.beta, 0, 1)


@OPERATORS.register_module(name='hardswish')
class Hswish(Act):
    def __init__(self, **kwargs):
        # kwargs['bit_saturation'] = 10
        super(Hswish, self).__init__(**kwargs)

    def activation(self, x):
        return F.hardswish(x)


@OPERATORS.register_module(name='hardtanh')
class HTanh(Act):
    def __init__(self, **kwargs):
        super(HTanh, self).__init__(**kwargs)

    def activation(self, x):
        return F.hardtanh(x)


@OPERATORS.register_module(name='hardshrink')
class Hshrink(Act):
    def __init__(self, **kwargs):
        super(Hshrink, self).__init__(**kwargs)

    def activation(self, x):
        return F.hardshrink(x)


@OPERATORS.register_module(name='globalaveragepool')
class GlobalAveragePooling(BaseOps):
    def __init__(self, **kwargs):
        super(GlobalAveragePooling, self).__init__(**kwargs)
        self.kwargs = copy.deepcopy(kwargs)
        self.stride, self.kernel_size, self.group = 1, 0, 0
        self.weights = 0

        self.si, self.sk, self.so = self.kwargs['si'], self.kwargs['sk'], self.kwargs['so']
        self.min_val, self.max_val = self.mins[self.bit_select], self.maxs[self.bit_select]
        self.dtype = self.bits_dict[self.bit_select]

        if 'strides' in kwargs.keys():
            self.stride = kwargs['strides']
        if 'kernel_shape' in kwargs.keys():
            self.kernel_size = kwargs['kernel_shape']
        self.pads = kwargs['pads'] if 'pads' in kwargs.keys() else (0, 0, 0, 0)

        if 'ceil_mode' in kwargs.keys():
            self.ceil_mode = bool(kwargs['ceil_mode'])
        else:
            self.ceil_mode = True
        if isinstance(self.stride, int):
            self.stride = tuple([self.stride, self.stride])
        else:
            self.stride = tuple(self.stride)
        if isinstance(self.kernel_size, int):
            self.kernel_size = tuple([self.kernel_size, self.kernel_size])
        else:
            self.kernel_size = tuple(self.kernel_size)

        if np.sum(self.kernel_size) == 0:
            attrs = dict()
        else:
            attrs = dict(ceil_mode=self.ceil_mode, kernel_shape=self.kernel_size, pads=self.pads, strides=self.stride)# type: ignore

        self.avgpool_sess = self.get_session(
            attrs=attrs,
            opset_version=self.opset_version,
        )

        self.init_scales = False

    def pool_align_bits(self, in_data, zero_point):
        output = self.Intscale(in_data) + zero_point  # tspe
        output = self.align_bits(output)
        return output

    def get_session(self, attrs, opset_version=15):

        initializers = []

        if attrs == dict():
            self.op_type = "GlobalAveragePool"
        else:
            self.op_type = "AveragePool"

        node = onnx.helper.make_node(# type: ignore
            self.op_type,
            inputs=['X'],
            outputs=["Y"],
            **attrs
        )

        inputs = [self.create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["X"]]
        outputs = [self.create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["Y"]]

        graph = onnx.helper.make_graph(# type: ignore
            nodes=[node],
            name='test_{}'.format(self.op_type),
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            )

        opsets = [onnx.helper.make_operatorsetid(ONNX_DOMAIN, self.opset_version),
                  onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)]
        model = onnx.helper.make_model(graph, producer_name='backend-{}'.format(self.op_type), opset_imports=opsets)# type: ignore
        model.ir_version = self.ir_version_default
        # onnx.save(model, "./test_{}.onnx".format(self.op_type))

        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        return sess

    def process(self, output):
        if self.op_type == "AveragePool":
            output = self.do_averagepool(output.astype(np.float32))
        else:
            output = self.avgpool_sess.run(None, {"X": output.astype(np.float32)})[0]
        return output

    def do_averagepool(self, data):
        in_data = torch.from_numpy(data)
        max_value = torch.max(in_data) + 1.0
        c_padding = nn.ConstantPad2d(tuple(self.pads), value=max_value)# type: ignore
        z_padding = nn.ZeroPad2d(tuple(self.pads))
        # x_ = (c_padding(in_data) < max_value).float()
        # x_w = F.avg_pool2d(
        #     input=x_, kernel_size=self.kernel_size, stride=self.stride, padding=(0, 0),
        #     ceil_mode=self.ceil_mode, divisor_override=1,
        # )
        out_data = F.avg_pool2d(
            input=z_padding(in_data), kernel_size=self.kernel_size,
            stride=self.stride, padding=(0, 0), ceil_mode=self.ceil_mode,
            divisor_override=1,
        )
        out_data = (out_data / (self.kernel_size[0] * self.kernel_size[1])).numpy()  # type: ignore

        return out_data

    def forward(self, inputs, **kwargs):
        in_data = inputs['output'].astype(np.float32)
        if not self.init_scales and self.process_scale in ["intscale", "smooth"]:
            n, c, h, w = in_data.shape
            self.sk = dict(scale=1.0/h/w, zero_point=0)
            self.setting.update(dict(sk=self.sk, si=self.si, so=self.so))
            self.datacorrect = datacorrect_factoy.get("intscale")(**self.setting)
            self.scales = self.datacorrect() # type: ignore
            self.init_scales = True

        if self.process_scale in ['intscale', 'floatscale', 'preintscale', 'preintscaleex', 'smooth']:
            output = in_data - self.scales["zi"]
        elif self.process_scale in ['float']:
            output = self.in_quantize.get_dequan_data(in_data)
        else:
            output = in_data

        if self.process_scale in ['floatscale', 'preintscale', 'preintscaleex', 'intscale', 'smooth', 'intscale', 'smooth']:
            dtype = self.bits_dict[self.high_bits_calc(self.bit_select)]
            output = output.astype(dtype)
            output = np.sum(output, axis=(2, 3), keepdims=True).astype(dtype)
            output = output >> -self.scales["out_shift"]
        else:
            output = self.process(output)

        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        in_data, zero_point = inputs['output'], self.scales["zo"]
        if self.process_scale in ['float']:
            output = self.out_quantize.get_quan_data(in_data)
        elif self.process_scale in ['floatscale', 'preintscale', 'preintscaleex', 'intscale', 'smooth']:
            output = self.pool_align_bits(in_data, zero_point)
        else:
            output = in_data

        return dict(output=output, out_shift=self.scales['out_shift'], out_scale=self.scales['out_scale'])

def clip_intN(data, N=15):
    min_v, max_v = -2**(N-1), 2**(N-1) - 1
    data = np.clip(data, a_min=min_v, a_max=max_v)
    return data

@OPERATORS.register_module(name='averagepool')
class AvgPooling(GlobalAveragePooling):
    def __init__(self, **kwargs):
        super(AvgPooling, self).__init__(**kwargs)
        align_param = self.special_alignment()
        self.scales = align_param

    def pool_align_bits(self, in_data, zero_point):
        dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
        output = self.Intscale(in_data.astype(dtype)) + zero_point  # tspe
        output = self.align_bits(output)
        return output

    def special_alignment(self):
        x_w = self.kernel_size[0] * self.kernel_size[1] # type: ignore
        bit_num = self.bit_saturation-3 if not self.txme_saturation else self.bit_saturation-1
        hw_out_scale, out_shift, out_scale = 1,0,1
        scale = self.si[0]['scale']*(1/x_w)/self.so[0]['scale']
        hw_out_scale = np.int32(np.round(2**bit_num*scale))
        scales = {'out_shift': 0, 'out_scale': 1,
                  'int_scale': 0, 'hw_out_scale': hw_out_scale,
                  'extra_value': 0, 'bit_num': bit_num,
                  'zi': self.si[0]['zero_point'], 'zk': 0, 'zo': self.so[0]['zero_point']}
        if hw_out_scale < 1:
            hw_out_scale = np.int32(1)
            out_shift, out_scale = bit_num, np.int32(np.round((2**bit_num*scale)*(2**bit_num-1)))
            scales.update(dict(int_scale=out_shift, out_scale=out_scale, hw_out_scale=hw_out_scale))
        return scales

    @staticmethod
    def clip_hw(inputs, output, hw_out_scale, is_asym):
        if inputs['output'].dtype in [np.int8, np.uint8]:
            N = 15 if is_asym else 30
            output = clip_intN(output * hw_out_scale, N=N)
            output = output.astype(np.int16)
        else:
            N = 31 if is_asym else 62
            output = clip_intN(output * hw_out_scale, N=N)
            output = output.astype(np.int32)
        return output

    @staticmethod
    def hw_aligment(output, align_bits, bit_shift):
        output = align_bits(output >> bit_shift)
        return output

    def forward(self, inputs, **kwargs):
        # inputs['output'] = np.load("input.npy").astype(np.int8)
        # myoutput = np.load("output.npy").astype(np.int8)
        # self.kernel_size = 3
        # self.stride = 1
        # self.ceil_mode = True
        # self.pads = [0, 0, 0, 0]

        in_data = inputs['output'].astype(np.float32)

        if self.process_scale in ['intscale', 'intscaleex', 'floatscale', 'preintscale', 'preintscaleex', 'smooth']:
            output = in_data - self.scales['zi']
            data = torch.from_numpy(output)
            z_padding = nn.ZeroPad2d(tuple(self.pads))
            output = z_padding(data)
            output = F.avg_pool2d(
                input=output, kernel_size=self.kernel_size, stride=self.stride,
                padding=(0, 0), ceil_mode=self.ceil_mode, divisor_override=1,
            ).numpy()

            is_asym = True if "sym" not in self.in_quantize.get_class_name() else False
            output = self.clip_hw(inputs, output, self.scales['hw_out_scale'], is_asym)

            output = self.hw_aligment(output, self.align_bits, self.scales['bit_num'])
        elif self.process_scale in ['float']:
            output = self.in_quantize.get_dequan_data(in_data)
        else:
            output = in_data

        if self.process_scale not in ['floatscale','intscale', 'intscaleex', 'smooth', 'preintscale', 'preintscaleex']:
            output = self.process(output)

        return dict(output=output)


@OPERATORS.register_module(name='maxpool')
class MaxPooling(BaseOps):
    def __init__(self, **kwargs):
        super(MaxPooling, self).__init__(**kwargs)
        self.kernel_size, self.stride = tuple(kwargs['kernel_shape']), kwargs['strides']

        pads = kwargs['pads'] if 'pads' in kwargs.keys() else (0, 0, 0, 0)
        self.pads = [pads[1], pads[3], pads[0], pads[2]]

        if 'ceil_mode' in kwargs.keys():
            self.ceil_mode = bool(kwargs['ceil_mode'])
        else:
            self.ceil_mode = True

    def forward(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs['output'])

        in_data = in_data.astype(np.float32)

        in_data = torch.from_numpy(in_data)
        min_v = np.min(inputs['output'])
        # padding = nn.ConstantPad2d(tuple(self.pads), np.inf)
        padding = nn.ConstantPad2d(tuple(self.pads), value=min_v - 1)
        output = F.max_pool2d(input=padding(in_data), kernel_size=self.kernel_size,
                              stride=tuple(self.stride), padding=(0, 0), ceil_mode=self.ceil_mode).numpy()
        if self.process_scale not in ["float", "ffloatscale"]:
            output = output.astype(inputs['output'].dtype)
        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = inputs['output']
        # dtype = self.bits_dict[self.high_bits_calc(self.bit_select)]
        out_shift, out_scale = self.scales['out_shift'], self.scales['out_scale']
        if self.process_scale not in ["smooth"]:
            output = self.Intscale(output)
        output = self.align_bits(output)

        return dict(output=output, out_shift=out_shift, out_scale=out_scale)


@OPERATORS.register_module(name='batchnormalization')
class BatchNormal(BaseOps):
    def __init__(self, **kwargs):
        super(BatchNormal, self).__init__(**kwargs)
        self.mean, self.var = torch.from_numpy(kwargs['mean']), torch.from_numpy(kwargs['var'])
        self.bias, self.scale = torch.from_numpy(kwargs['bias']), torch.from_numpy(kwargs['scale'])
        if 'epsilon' in kwargs.keys():
            self.epsilon = kwargs['epsilon']
        else:
            self.epsilon = 1.0e-5
        self.in_quantize, self.out_quantize = kwargs['in_quantize'], kwargs['quantize']

    def preprocess(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs['output'])
        return dict(output=in_data)

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']

        assert self.process_scale == 'float'

        if not in_data.dtype in [np.float32, np.float64]:
            in_data = self.in_quantize.get_dequan_data(in_data)

        out_data = F.batch_norm(torch.from_numpy(in_data), running_mean=self.mean, running_var=self.var,
                                weight=self.scale, bias=self.bias, eps=self.epsilon).numpy()

        if not self.out_type in [np.float32, np.float64]:
            out_data = self.out_quantize.get_quan_data(out_data)

        return dict(output=out_data)

    def postprocess(self, inputs, **kwargs):
        in_data = inputs['output']
        return dict(output=in_data, out_shift=0, out_scale=1)


@OPERATORS.register_module(name='log')
class Log(BaseOps):
    def __init__(self, **kwargs):
        super(Log, self).__init__(**kwargs)
        # if 'axis' in kwargs.keys():
        #     self.axis = kwargs['axis']
        # else:
        #     self.axis = 1
        self.in_quantize, self.out_quantize = kwargs['in_quantize'], kwargs['quantize']

    def preprocess(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs['output'])
        return dict(output=in_data)

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']

        assert self.process_scale == 'float'

        if not in_data.dtype in [np.float32, np.float64]:
            in_data = self.in_quantize.get_dequan_data(in_data)

        out_data = torch.log2(torch.from_numpy(in_data)).numpy()

        if not self.out_type in [np.float32, np.float64]:
            out_data = self.out_quantize.get_quan_data(out_data)

        return dict(output=out_data)

    def postprocess(self, inputs, **kwargs):
        in_data = inputs['output']
        return dict(output=in_data, out_shift=0, out_scale=1)


@OPERATORS.register_module(name='softmax')
class Softmax(BaseOps):
    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        if 'axis' in kwargs.keys():
            self.axis = kwargs['axis']
        else:
            self.axis = 1
        # self.high_precision = False
        self.in_quantize, self.out_quantize = kwargs['in_quantize'], kwargs['quantize']
        self.table = self.get_table()
        self.table_align()

    def table_align(self):
        self.table[self.table < self.mins[self.bit_select]] = self.mins[self.bit_select]
        self.table[self.table > self.maxs[self.bit_select]] = self.maxs[self.bit_select]
        self.table = self.bits_dict[self.bit_select](self.table)

    def get_table(self):
        if self.bits_dict[self.bit_select] == np.int8:
            data_type = np.int32
        else:
            data_type = np.int64
        zo = self.out_quantize.get_scale()[1]
        si = self.si[0]['scale']
        so = self.so['scale']
        table = np.arange(self.mins[self.bit_select], self.maxs[self.bit_select] + 1)
        max_value = np.max(table)
        qn = table - max_value
        a, b, c = 0.3585, 1.353, 0.344
        si = extract_scale(self.si)
        qb = np.floor(b / si)
        qc = np.floor(c / a / si / si)
        qln2 = np.floor(np.log(2) / si)
        z = self.bits_dict[self.bit_select](np.floor(-qn / qln2))
        qn = qn + z * qln2
        qn = (qn + qb) ** 2 + qc
        qn = qn.astype(data_type)
        table = np.round(np.right_shift(qn, z) * a * si)

        # self.max_value = self.maxs[self.bit_select]
        # tables = []
        # for j in range(self.max_value + 1):
        #     if j > 0:
        #         table_ = np.round(1.0 * (table * self.max_value + zo * j) / j)
        #         tables.append(table_)
        #     else:
        #         tables.append(table)
        # table = np.stack(tables, axis=0)
        # if self.high_precision:
        #     table = table[:1, :]

        return table

    def ISOFTMAX(self, q):
        if self.bits_dict[self.bit_select] == np.int8:
            data_type = np.int32
        else:
            data_type = np.int64
        si = self.si[0]['scale']
        so = self.so['scale']
        zi = self.scales['zi']
        zo = self.scales['zo']
        qn = np.array(q - self.mins[self.bit_select] - zi).astype(data_type)
        # qn = self.table[0, qn]
        qn = self.table[qn]

        #### https://arxiv.org/pdf/2101.01321v3.pdf
        # a, b, c = 0.3585, 1.353, 0.344
        # qln2 = np.floor(np.log(2) / si)
        # qb = np.floor(b / si)
        # qc = np.floor(c / a / si / si)

        # qn = q - np.max(q) # qn <= 0
        # z = self.bits_dict[self.bit_select](np.floor(-qn / qln2))
        # qn = qn + z * qln2
        # qn = (qn + qb) ** 2 + qc
        # qn = qn.astype(data_type)
        # qn = np.right_shift(qn, z)


        # if not self.high_precision:
        #     index = np.ceil(self.max_value * np.sum(qn, axis=self.axis) * so)
        #     index = self.bits_dict[self.bit_select](index)
        #     qn = self.table[qn, index]
        # else:
        qn = np.round(qn / (np.sum(qn, axis=self.axis, keepdims=True) * so) + zo)
        qn = qn.clip(self.mins[self.bit_select], self.maxs[self.bit_select])
        qn = self.bits_dict[self.bit_select](qn)

        return qn

    def preprocess(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs['output'])
        return dict(output=in_data)

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']

        assert self.process_scale in ['float', 'table']

        if self.process_scale == 'float':
            if not in_data.dtype in [np.float32, np.float64]:
                in_data = self.in_quantize.get_dequan_data(in_data)

            out_data = F.softmax(torch.from_numpy(in_data), dim=self.axis).numpy()

            if not self.out_type in [np.float32, np.float64]:
                out_data = self.out_quantize.get_quan_data(out_data)
        else:
            # look up table
            if in_data.dtype in [np.float32, np.float64]:
                if isinstance(self.in_quantize, list):
                    out_data = self.in_quantize[0].get_quan_data(in_data)
                else:
                    out_data = self.in_quantize.get_quan_data(in_data)
            else:
                out_data = in_data

            out_data = self.ISOFTMAX(out_data.astype(np.float32))

        return dict(output=out_data)

    def postprocess(self, inputs, **kwargs):
        in_data = inputs['output']
        return dict(output=in_data, out_shift=0, out_scale=1)


@OPERATORS.register_module(name='reshape')
class Reshape(BaseOps):
    def __init__(self, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        # self.shape, self.dims = kwargs['shape'], kwargs['dims']
        self.shape = kwargs['shape']

    def preprocess(self, inputs, **kwargs):
        return inputs

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']
        for idx, shape in enumerate(self.shape):
            if shape == 0:
                self.shape[idx] = in_data.shape[idx]
        return dict(output=np.reshape(in_data, self.shape))

    def postprocess(self, inputs, **kwargs):
        return dict(output=inputs['output'], out_shift=np.int32(0), out_scale=np.int32(1))


@OPERATORS.register_module(name='pad')
class Pad(BaseOps):
    def __init__(self, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.mode = kwargs['mode']
        self.pads = kwargs['pads']
        
    def preprocess(self, inputs, **kwargs):
        return inputs

    def forward(self, inputs, **kwargs):
        in_data = torch.from_numpy(inputs['output'])
        if self.mode == "constant":
            output = F.pad(in_data, pad=tuple(self.pads), mode=self.mode).numpy() 
        else:
            raise Exception("Not supported!!!")
        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        return dict(output=inputs['output'], out_shift=np.int32(0), out_scale=np.int32(1))


@OPERATORS.register_module(name='flatten')
class Flatten(BaseOps):
    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.axis = kwargs['axis']

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']
        shape = in_data.shape[:self.axis + 1]
        return dict(output=np.reshape(in_data, shape))

    def postprocess(self, inputs, **kwargs):
        return inputs


@OPERATORS.register_module(name='transpose')
class Transpose(BaseOps):
    def __init__(self, **kwargs):
        super(Transpose, self).__init__(**kwargs)
        self.perm = kwargs['perm']

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']
        return dict(output=np.transpose(in_data, self.perm))

    def postprocess(self, inputs, **kwargs):
        return inputs


@OPERATORS.register_module(name='fc')
@OPERATORS.register_module(name='gemm')
class FC(BaseOps):
    def __init__(self, **kwargs):
        super(FC, self).__init__(**kwargs)
        self.transB, self.transA = False, False
        if 'transB' in kwargs.keys(): self.transB = kwargs['transB']
        if 'transA' in kwargs.keys(): self.transA = kwargs['transA']
        self.weights = copy.deepcopy(kwargs['p_weights'])
        # if not 'transB' in kwargs.keys():
        #     self.weights = np.transpose(copy.deepcopy(kwargs['p_weights']), (1, 0))
        # else:
        #     self.weights = copy.deepcopy(kwargs['p_weights'])

        self.p_weights = torch.from_numpy(np.array(self.weights, dtype=np.float32))
        if isinstance(self.scales['zk'], np.ndarray):
            zk = np.reshape(self.scales['zk'], (-1, 1))
            self.p_weights = self.p_weights - zk
        else:
            self.p_weights = self.p_weights - self.scales['zk']

    def preprocess(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs['output'])
        if in_data.dtype in [np.float32, np.float64]:
            in_data = self.in_quantize[0].get_quan_data(in_data)

        if self.transA:
            in_data = np.transpose(in_data, (1, 0))
        p_inputs = np.array(in_data, dtype=np.float32)
        # p_weights = np.array(self.weights, dtype=np.float32)

        return dict(p_inputs=p_inputs)

    def forward(self, inputs, **kwargs):
        in_data = inputs['p_inputs'] - self.scales['zi']
        # if isinstance(self.scales['zk'], np.ndarray):
        #     zk = np.reshape(self.scales['zk'], (-1, 1))
        #     w_conv = w_conv - zk
        # else:
        #     w_conv = w_conv - self.scales['zk']
        # output = np.matmul(in_data, w_conv)
        # output = torch.matmul(torch.from_numpy(in_data.astype(np.float32)), torch.from_numpy(w_conv.astype(np.float32)))
        output = F.linear(torch.from_numpy(in_data), self.p_weights)
        # output = torch.matmul(torch.from_numpy(in_data), torch.transpose(self.p_weights, 1, 0))
        output = output.numpy()  # .astype(in_data.dtype)
        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = inputs['output']
        # fused_zero_point = inputs['fused_zero_point']
        bit_num = max(self.bit_select, self.w_bit_select)
        dtype = self.bits_dict[self.high_bits_calc(bit_num)]
        # output = np.array(output, np.int64) if output.dtype.type == np.float64 else np.array(output, np.int32)
        # fused_zero_point = np.int64(fused_zero_point) if output.dtype.type == np.float64 else np.int32(fused_zero_point)
        return dict(output=output.astype(dtype))


@OPERATORS.register_module(name='add')
@OPERATORS.register_module(name='cadd')
class Add(BaseTable):
    # fp/int scale
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)
        setting = copy.deepcopy(self.setting)
        self.si, self.sk, self.so = kwargs['si'], kwargs['sk'], kwargs['so']
        setting.update(dict(si=self.si[0], sk=dict(scale=1.0, zero_point=0), so=self.so))
        self.datacorrect = [datacorrect_factoy.get(self.process_scale)(**setting)]
        scales0 = self.datacorrect[0]()# type: ignore
        if len(kwargs['si']) > 1:
            setting.update(dict(si=kwargs['si'][1]))
        self.datacorrect.append(datacorrect_factoy.get(self.process_scale)(**setting))
        scales1 = self.datacorrect[1]()# type: ignore
        self.scales = [scales0, scales1]

        self.p_weights = kwargs.get('p_weights')
        self.f_weights = kwargs.get('f_weights')
        self.weight_idx = kwargs.get('weight_idx')
        self.p_weights =self.p_weights if isinstance(self.p_weights, list) else [self.p_weights]

        self.init_table(**kwargs)

    def init_table(self, **kwargs):
        if self.process_scale in ['shiftfloatscaletable']:
            self.lower_bound = -128
            self.upper_bound = 127
            self.table = self.calc_table()
            self.table_align()

    def calc_table(self, idx=0):
        zo = self.scales[0]['zo']
        so = extract_scale(self.so)
        si0 = extract_scale(self.si[0]) # type: ignore
        zi0 = self.scales[0]['zi']
        si1 = extract_scale(self.si[1]) # type: ignore
        zi1 = self.scales[1]['zi']

        x = np.linspace(self.lower_bound, self.upper_bound, self.upper_bound-self.lower_bound+1)
        y = np.linspace(self.lower_bound, self.upper_bound, self.upper_bound-self.lower_bound+1)
        if self.bit_select == 3:
            x *= (2**8)
            y *= (2**8)

        xv, yv = np.meshgrid(x, y)

        a, b = (yv - zi0) * si0 / so, (xv - zi1) * si1 / so
        table = self.process(a, b) + zo
        table = np.round(table)

        return table

    def reset_correct(self, correct):
        pass

    def process(self, in_data_0, in_data_1):
        return in_data_0 + in_data_1

    def forward(self, inputs, **kwargs):
        if isinstance(self.weight_idx, list):
            in_data0, in_data1 = self.weight_insert_into_inputs(inputs, self.p_weights, self.weight_idx)
        else:
            in_data0, in_data1 = inputs['output']
        in_data0, in_data1 = in_data0['output'], in_data1['output']

        if self.process_scale in ['float', 'ffloatscale']:
            if in_data0.dtype in [np.float32, np.float64]:
                output0 = copy.deepcopy(in_data0)
            else:
                output0 = self.in_quantize[0].get_dequan_data(in_data0)
            if in_data1.dtype in [np.float32, np.float64]:
                output1 = copy.deepcopy(in_data1)
            else:
                output1 = self.in_quantize[1].get_dequan_data(in_data1)
        elif self.process_scale in ['shiftfloatscaletable']:
            output0 = copy.deepcopy(in_data0)
            output1 = copy.deepcopy(in_data1)
            if self.bit_select == 3:
                output0 = (output0 >> 8)
                output1 = (output1 >> 8)
            output = self.table[output0 - self.lower_bound, output1 - self.lower_bound]
        # elif self.process_scale in ['floatscale']:
        #     output0 = in_data0.astype(np.float32) - self.si[0]['zero_point']
        #     output1 = in_data1.astype(np.float32) - self.si[1]['zero_point']
        #     output0, output1 = self.MultiIntscale(output0, idx=0), self.MultiIntscale(output1, idx=1)
        else:
            # assert self.process_scale in ['intscale', 'preintscale', 'preintscaleex]
            # types = {in_data0.dtype.type, in_data1.dtype.type}
            # inter = types.intersection({np.int8, np.uint8, np.int16, np.uint16})
            # if len(inter) > 0:
            #     dtype = np.int32
            # else:
            #     dtype = np.int64
            dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
            output0, output1 = in_data0.astype(dtype), in_data1.astype(dtype)
            output0 = output0 - dtype(self.si[0]['zero_point'])
            output1 = output1 - dtype(self.si[1]['zero_point'])

            if self.process_scale not in ['smooth']:
                output0 = self.MultiIntscale(output0, idx=0)
                output1 = self.MultiIntscale(output1, idx=1)

        if self.process_scale not in ['shiftfloatscaletable']:
            output = self.process(output0, output1)

        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = inputs['output']
        out_shift = []
        out_scale = []
        if self.process_scale in ['float', 'ffloatscale']:
            if self.out_type not in [np.float32, np.float64]:
                output = self.out_quantize.get_quan_data(output)
        elif self.process_scale in ['shiftfloatscaletable']:
            output = self.align_bits(output)
        else:
            output = self.align_bits(output + self.so[-1]['zero_point'])

        return {'output': output, 'out_shift': out_shift, 'out_scale': out_scale}


@OPERATORS.register_module(name='sub')
@OPERATORS.register_module(name='csub')
class Sub(Add):
    def __init__(self, **kwargs):
        super(Sub, self).__init__(**kwargs)

    def process(self, in_data_0, in_data_1):
        return in_data_0 - in_data_1


class ReduceOps(BaseOps):
    def __init__(self, **kwargs):
        super(ReduceOps, self).__init__(**kwargs)
        self.kwargs = copy.deepcopy(kwargs)
        self.dims = self.kwargs["axes"]
        self.keepdims = True if self.kwargs["keepdims"] else False

    def process(self, data):
        return data

    def squeeze_data(self, data):
        if not self.keepdims:
            for idx, dim in enumerate(self.dims):
                dim -= idx
                data = torch.squeeze(data, dim)
        return data

    def preprocess(self, inputs, **kwargs):
        in_datas = copy.deepcopy(inputs)
        return in_datas

    def forward(self, inputs, **kwargs):
        output = inputs["output"]
        if self.process_scale == 'float':
            output = self.in_quantize.get_dequan_data(output)

        output = torch.from_numpy(output).type(torch.float32)
        output = self.process(output).numpy()

        if self.process_scale == 'float':
            output = self.out_quantize.get_quan_data(output)

        output = self.align_bits(output)

        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = inputs["output"]
        return dict(output=output)


@OPERATORS.register_module(name='reducemax')
class ReduceMax(ReduceOps):
    def __init__(self, **kwargs):
        super(ReduceMax, self).__init__(**kwargs)

    def process(self, data):
        for dim in self.dims:
            data = torch.max(data, dim=dim, keepdim=True)[0] # type: ignore
        return self.squeeze_data(data)


@OPERATORS.register_module(name='reducemin')
class ReduceMin(ReduceOps):
    def __init__(self, **kwargs):
        super(ReduceMin, self).__init__(**kwargs)

    def process(self, data):
        for dim in self.dims:
            data = torch.min(data, dim=dim, keepdim=True)[0] # type: ignore
        return self.squeeze_data(data)


@OPERATORS.register_module(name='reducemean')
class ReduceMean(ReduceOps):
    def __init__(self, **kwargs):
        super(ReduceMean, self).__init__(**kwargs)

    def process(self, data):
        for dim in self.dims:
            data = torch.mean(data, dim=dim, keepdim=True)     # type: ignore
        return self.squeeze_data(data)


@OPERATORS.register_module(name='reducesum')
class ReduceSum(ReduceOps):
    def __init__(self, **kwargs):
        super(ReduceSum, self).__init__(**kwargs)

    def process(self, data):
        for dim in self.dims:
            data = torch.sum(data, dim=dim, keepdim=True)
        return self.squeeze_data(data)


@OPERATORS.register_module(name='reduceprod')
class ReduceProd(ReduceOps):
    def __init__(self, **kwargs):
        super(ReduceProd, self).__init__(**kwargs)

    def process(self, data):
        for dim in self.dims:
            data = torch.prod(data, dim=dim, keepdim=True)
        return self.squeeze_data(data)


@OPERATORS.register_module(name='concat')
class Concat(BaseTable):
    # fp/int scale
    def __init__(self, **kwargs):
        super(Concat, self).__init__(**kwargs)

        self.si = kwargs['si']
        # self.process_scale, self.bit_select = kwargs['process_scale'], kwargs['bit_select']
        # self.precision, self.int_scale = kwargs['precision'], kwargs['int_scale']
        self.axis = kwargs['axis']
        self.scales = list()
        # self.zi = [item['zero_point'] for item in si]
        # if self.process_scale in ['smooth', 'float']:
        #     self.zi = [0 for item in si]

        setting = copy.deepcopy(self.setting)
        self.input_len = kwargs['input_len']
        self.out_shifts, self.out_scales = list(), list()
        self.datacorrect = []
        for idx in range(self.input_len):
            setting.update(dict(si=kwargs['si'][idx], so=kwargs['so'], sk=dict(scale=1.0, zero_point=0)))
            self.datacorrect.append(datacorrect_factoy.get(self.process_scale)(**setting))# type: ignore
            self.scales.append(self.datacorrect[idx]())

        self.init_table(**kwargs)

    def init_table(self, **kwargs):
        if self.process_scale in ['shiftfloatscaletable']:
            self.bit_saturation = kwargs.get("bit_saturation", self.bit_saturation)
            self.lower_bound = -2 ** (self.bit_saturation - 1)
            self.upper_bound = 2 ** (self.bit_saturation - 1) - 1
            tables = []
            for idx in range(self.input_len):
                table = self.calc_table(idx=idx)
                tables.append(table)
            self.table = np.concatenate(tables, axis=1)
            self.table_align()

    def get_act_table(self, si, so, zi, zo):
        table = np.arange(self.lower_bound, self.upper_bound+1)
        table = torch.from_numpy(table).type(torch.float32)
        if isinstance(si, np.ndarray):
            table = table.expand(si.shape[0], table.shape[0]).permute(1, 0) # type: ignore
        else:
            table = table.expand(1, table.shape[0]).permute(1, 0)

        table = (table - zi) * si
        table = (table).numpy()
        return np.round(table / so + zo)

    def calc_table(self, idx=0):
        zo = self.out_quantize.get_scale()[1]
        so = extract_scale(self.so)
        si = extract_scale(self.si[idx]) # type: ignore
        zi = self.in_quantize[idx].get_scale()[1]
        return self.get_act_table(si, so, zi, zo)

    def forward(self, inputs, **kwargs):
        in_datas = inputs['output']
        out_datas, out_shifts, out_scales = list(), list(), list()
        for idx in range(self.input_len):
            out_shifts.append(self.scales[idx]['out_shift'])
            out_scales.append(self.scales[idx]['out_scale'])
            out_data = in_datas[idx]['output']
            if self.process_scale in ['float', 'ffloatscale']:
                if out_data.dtype not in [np.float32, np.float64]:
                    out_data = self.in_quantize[idx].get_dequan_data(out_data)
            elif self.process_scale in ['shiftfloatscaletable']:
                out_data = self.table[out_data - self.lower_bound, idx]
            else:  # ['intscale', 'preintscale', 'preintscaleex]
                dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
                out_data = out_data.astype(dtype) - self.si[idx]['zero_point']
                # out_data = shift_data(out_data, self.scales[idx]['out_shift'])
                if self.process_scale not in ['smooth']:
                    out_data = self.MultiIntscale(out_data, idx)

            out_datas.append(torch.from_numpy(out_data))
        if hasattr(torch, 'concat'):
            output = torch.concat(out_datas, dim=self.axis).numpy()
        else:
            output = torch.cat(out_datas, dim=self.axis).numpy()

        if self.process_scale in ['float', 'ffloatscale']:
            if self.out_type not in [np.float32, np.float64]:
                output = self.out_quantize.get_quan_data(output)
        elif self.process_scale in ['shiftfloatscaletable']:
            output = self.align_bits(output)
        else:
            output = self.align_bits(output + self.so['zero_point'])

        return dict(output=output, out_shift=out_shifts, out_scale=out_scales)

    def postprocess(self, inputs, **kwargs):
        return inputs


@OPERATORS.register_module(name='split')
class Split(BaseTable):
    def __init__(self, **kwargs):
        super(Split, self).__init__(**kwargs)

        self.scales = list()
        setting = copy.deepcopy(self.setting)

        self.so, self.si = kwargs["so"], kwargs["si"]
        self.split, self.axis = kwargs['split'], kwargs['axis']
        self.output_len = len(self.split)
        self.out_shifts, self.out_scales = list(), list()
        self.zo = [item['zero_point'] for item in self.so]
        self.datacorrect = []
        if self.process_scale in ['smooth']:
            self.zo = [self.si['zero_point'] for item in self.so]
        for idx, _ in enumerate(self.split):
            setting.update(dict(si=self.si, sk=kwargs['sk'], so=self.so[idx]))
            self.datacorrect.append(datacorrect_factoy.get(self.process_scale)(**setting)) # type: ignore
            self.scales.append(self.datacorrect[idx]())

        self.init_table(**kwargs)

    def init_table(self, **kwargs):
        if self.process_scale in ['shiftfloatscaletable']:
            self.bit_saturation = kwargs.get("bit_saturation", self.bit_saturation)
            self.lower_bound = -2 ** (self.bit_saturation - 1)
            self.upper_bound = 2 ** (self.bit_saturation - 1) - 1
            tables = []
            for idx in range(self.output_len):
                table = self.calc_table(idx=idx)
                tables.append(table)
            self.table = np.concatenate(tables, axis=1)
            self.table_align()

    def get_act_table(self, si, so, zi, zo):
        table = np.arange(self.lower_bound, self.upper_bound+1)
        table = torch.from_numpy(table).type(torch.float32)
        if isinstance(si, np.ndarray):
            table = table.expand(si.shape[0], table.shape[0]).permute(1, 0) # type: ignore
        else:
            table = table.expand(1, table.shape[0]).permute(1, 0)

        table = (table - zi) * si
        table = (table).numpy()
        return np.round(table / so + zo)

    def calc_table(self, idx=0):
        zo = self.so[idx]["zero_point"]
        so = extract_scale(self.so[idx])
        si = extract_scale(self.si) # type: ignore
        zi = self.si[0]["zero_point"]
        return self.get_act_table(si, so, zi, zo)

    # def preprocess(self, inputs, **kwargs):
    #     in_datas = copy.deepcopy(inputs)
    #     return dict(output=in_datas)

    def forward(self, inputs, **kwargs):
        in_datas = inputs['output']
        # if self.process_scale in ['float', 'ffloatscale']:
            # if self.out_type not in [np.float32, np.float64]:
                # in_datas = self.in_quantize.get_dequan_data(in_datas)
        out_datas, out_shifts, out_scales = [], [], []
        outputs = torch.split(torch.from_numpy(in_datas), tuple(self.split), dim=self.axis)
        for idx, data in enumerate(outputs):
            out_data = data.numpy()
            out_shifts.append(self.scales[idx]['out_shift'])
            out_scales.append(self.scales[idx]['out_scale'])
            if self.process_scale in ['float']:
                if out_data.dtype not in [np.float32, np.float64]:
                    out_data = self.out_quantize[idx].get_dequan_data(out_data)
            elif self.process_scale in ['ffloatscale']:
                out_data = self.out_quantize[idx].get_dequan_data(out_data)
            elif self.process_scale in ['shiftfloatscaletable']:
                out_data = self.table[out_data - self.lower_bound, idx]
            else:
                dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
                out_data = out_data.astype(dtype) - self.scales[idx]['zi']
                # out_data = shift_data(out_data, self.scales[idx]['out_shift'])
                out_data = self.MultiIntscale(out_data, idx) + self.zo[idx]
                out_data = self.align_bits(out_data)
            out_datas.append(dict(output=out_data, out_shift=self.scales[idx]['out_shift'],
                                  out_scale=self.scales[idx]['out_scale']))

        # for idx in range(len(self.split)):
        #     start = int(np.sum(self.split[:idx]))
        #     end = int(np.sum(self.split[:idx + 1]))
        #     out_data = in_datas[:, start:end] #- self.zo[idx]
        #     # out_shifts.append(self.scales[idx]['out_shift'])
        #     # out_scales.append(self.scales[idx]['out_scale'])
        #     if self.process_scale == 'float':
        #         out_data = self.out_quantize[idx].get_quan_data(out_data)
        #     else:
        #
        #         if out_data.dtype.type in [np.int8, np.uint8]:
        #             dtype = np.int32
        #         else:
        #             dtype = np.int64
        #         out_data = out_data.astype(dtype) - self.scales[idx]['zi']
        #         out_data = shift_data(out_data, self.scales[idx]['out_shift'])
        #         out_data = self.MultiIntscale(out_data, idx) + self.zo[idx]
        #
        #     out_datas.append(dict(output=out_data, out_shift=self.scales[idx]['out_shift'],
        #                           out_scale=self.scales[idx]['out_scale']))
        return out_datas

    def postprocess(self, inputs, **kwargs):
        # output = inputs['output']
        # # for idx in range(len(output)):
        # #     output[idx] = output[idx].astype(self.bits_dict[self.bit_select])
        # inputs['output'] = output
        return inputs


@OPERATORS.register_module(name='slice')
class Slice(BaseOps):
    def __init__(self, **kwargs):
        super(Slice, self).__init__(**kwargs)
        # setting = copy.deepcopy(self.setting)
        self.axes = kwargs['axes']
        self.starts, self.ends = kwargs['starts'], kwargs['ends']
        # self.scales = datacorrect_factoy.get(self.process_scale)(**setting)()

    def preprocess(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs['output'])
        if 'in_data' in inputs.keys():
            in_data = inputs['in_data']
        output = copy.deepcopy(in_data[:, self.starts[0]:self.ends[0]])
        return dict(output=output, in_data=in_data)

    def forward(self, inputs, **kwargs):
        output = inputs['output']
        if self.process_scale not in ['float', 'ffloatscale']:
            if self.out_type not in [np.float32, np.float64]:
                output = self.out_quantize.get_quan_data(output)
        else:
            dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
            output = output.astype(dtype) - self.scales['zi']
            output = self.Intscale(output) + self.scales['zo']

            output = self.align_bits(output)

        return dict(output=output, in_data=inputs['in_data'], out_shift=self.scales['out_shift'],
                    out_scale=self.scales['out_scale'])

    def postprocess(self, inputs, **kwargs):
        return inputs


@OPERATORS.register_module(name='resize')
class Resize(BaseOps):
    def __init__(self, **kwargs):
        super(Resize, self).__init__(**kwargs)
        self.resize_int = None
        self.scales_, self.sizes_ = kwargs['scale'], kwargs['sizes']
        self.resize_int_output = None

        self.resize_sess = self.get_session(
            attrs=kwargs,
            opset_version=self.opset_version,
        )

    def get_session(self, attrs, opset_version=15):
        def create_in_out(name, shape, data_type): return onnx.helper.make_tensor_value_info(# type: ignore
            name, data_type, shape)
        def create_initializer(data, name, data_type): return onnx.helper.make_tensor(# type: ignore
            name=name, data_type=data_type,
            dims=data.shape, vals=data.tobytes(), raw=True)

        new_attrs = {}
        for key in attrs.keys():
            if key in ['scale', 'roi']:
                continue
            if key not in ['coordinate_transformation_mode', 'cubic_coeff_a', 'mode', 'nearest_mode']:
                continue
            new_attrs[key] = attrs[key]

        scales = dict()
        if 'scale' in attrs.keys():
            scale = attrs['scale']
            scale = np.array(scale) if isinstance(scale, list) else scale
            scales = dict(scale=scale)
        else:
            scales = dict(scale=None)

        if 'roi' in attrs.keys():
            roi = attrs['roi']
            roi = np.array(roi) if isinstance(roi, list) else roi
            scales.update(roi=roi)# type: ignore
        else:
            scales.update(roi=None)

        initializers = []

        inputs = ['X', '', '', '']
        outputs = [create_in_out(name, ['n', 'c', 'h', 'w'], onnx.TensorProto.FLOAT) for name in ["Y"]]# type: ignore
        if isinstance(scales['roi'], np.ndarray):
            inputs[1] = 'roi'
            roi = scales['roi'] if isinstance(scales['roi'], np.ndarray) else 0
            initializers.extend([
                create_initializer(roi, 'roi', onnx.TensorProto.FLOAT)# type: ignore
            ])
        if isinstance(scales['scale'], np.ndarray):
            inputs[2] = 'scales'
            inputs = inputs[:3]
            scale = scales['scale'] if isinstance(scales['scale'], np.ndarray) else 0
            initializers.extend([
                create_initializer(scale.astype(np.float32), 'scales', onnx.TensorProto.FLOAT),# type: ignore
            ])
            self.scales_ = copy.deepcopy(scales['scale'][-2:])
        else:
            inputs[3] = 'sizes'
            sizes = (
                np.array(attrs["sizes"]).astype(np.int64) if "sizes" in attrs.keys() else None
            )
            size_o = [int(s) for s in sizes]# type: ignore
            outputs = [create_in_out(name, size_o, onnx.TensorProto.FLOAT) for name in ["Y"]]# type: ignore
            initializers.extend([
                create_initializer(np.array(size_o), "sizes", onnx.TensorProto.INT64),# type: ignore
                ])
            self.sizes_ = copy.deepcopy(np.array(attrs["sizes"])[-2:])

        node = onnx.helper.make_node(# type: ignore
            "Resize",
            inputs=inputs,
            outputs=["Y"],
            **new_attrs
        )

        inputs = [create_in_out(name, ['n', 'c', 'h', 'w'], onnx.TensorProto.FLOAT) for name in ["X"]]# type: ignore

        graph = onnx.helper.make_graph(# type: ignore
            nodes=[node],
            name='test_resize',
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            )
        # opsets = [onnx.helper.make_operatorsetid(ONNX_DOMAIN, 12),
        #           onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)]
        # model = onnx.helper.make_model(graph, producer_name='backend-resize', opset_imports=opsets)# type: ignore
        # model.ir_version = self.ir_version_default
        # self.model.opset_import[0].version = self.opset_version
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset_version)])# type: ignore
        model.ir_version = self.ir_version_default
        # onnx.save(model, "test_resize.onnx")

        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        return sess


    # def preprocess(self, inputs, **kwargs):
    #     return inputs

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']
        output_w, output_h, scales = 0, 0, np.array([1.0, 1.0], dtype=np.float32)
        in_shape, out_shape = np.array(in_data.shape), np.array(in_data.shape)
        if self.process_scale in ['preintscale', 'preintscaleex'] and not self.resize_int:

            if isinstance(self.scales_, np.ndarray):
                output_h = np.int32(in_data.shape[2] * self.scales_[0])
                output_w = np.int32(in_data.shape[3] * self.scales_[1])
                scales = self.scales_
            else:
                output_h = np.int32(self.sizes_[0]) # type: ignore
                output_w = np.int32(self.sizes_[1])# type: ignore
                scales = np.array([self.sizes_[0]/in_data.shape[2], self.sizes_[1]/in_data.shape[3]], dtype=np.float32)# type: ignore

            out_shape[2], out_shape[3] = output_h, output_w
            if self.bit_select < 2:
                self.resize_int = pyops.py_resize_op_int8(in_shape, out_shape, scales, 1, True, 4)
            else:
                self.resize_int = pyops.py_resize_op_int16(in_shape, out_shape, scales, 1, True, 4)

        assert self.process_scale in ['float', 'smooth', 'floatscale', 'preintscale', 'preintscaleex']
        if self.process_scale == 'float':
            if in_data.dtype not in [np.float32, np.float64]:
                if isinstance(self.in_quantize, list):
                    in_data = self.in_quantize[0].get_dequan_data(in_data)
                else:
                    in_data = self.in_quantize.get_dequan_data(in_data)
        else:
            # in_data = torch.from_numpy(in_data.astype(np.float32) - self.scales['zi'])
            in_data = in_data.astype(np.float32) - self.scales['zi']# type: ignore

        if self.process_scale not in ['preintscale', 'preintscaleex']:
            align_out = self.resize_sess.run(None, {"X": in_data.astype(np.float32)})[0]
        else:
            if self.bit_select < 2:
                in_data = in_data.astype(np.int8)
                if not isinstance(self.resize_int_output, np.ndarray):
                    self.resize_int_output = np.zeros(out_shape.tolist(), dtype=np.int8)
            else:
                in_data = in_data.astype(np.int16)
                if not isinstance(self.resize_int_output, np.ndarray):
                    self.resize_int_output = np.zeros(out_shape.tolist(), dtype=np.int16)
            self.resize_int.forward(in_data, self.resize_int_output, in_shape)# type: ignore
            align_out = copy.deepcopy(self.resize_int_output)

        if self.process_scale == 'float':
            if self.out_type not in [np.float32, np.float64]:
                align_out = self.out_quantize.get_quan_data(align_out)# type: ignore
        else:
            dtype = self.bits_dict[self.lower_bits_calc(self.bit_select)]
            align_out = self.Intscale(align_out.astype(dtype)) + self.scales['zo']  # type: ignore

        if self.out_type not in [np.float32, np.float64]:
            align_out = self.align_bits(align_out)

        return dict(output=align_out, out_shift=self.scales['out_shift'], out_scale=self.scales['out_scale'])

    def postprocess(self, inputs, **kwargs):
        return inputs


@OPERATORS.register_module(name='mean')
class Mean(BaseOps):
    def __init__(self, **kwargs):
        super(Mean, self).__init__(**kwargs)

    def preprocess(self, inputs, **kwargs):
        pass

    def forward(self, inputs, **kwargs):
        pass

    def postprocess(self, inputs, **kwargs):
        pass


@OPERATORS.register_module(name='unsqueeze')
class Unsqueeze(BaseOps):
    def __init__(self, **kwargs):
        super(Unsqueeze, self).__init__(**kwargs)
        self.axis = 1

    def preprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])

        return dict(output=output)

    def forward(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])
        output = np.expand_dims(output, axis=self.axis)

        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])

        return dict(output=output, out_shift=0, out_scale=1)


@OPERATORS.register_module(name='squeeze')
class Squeeze(BaseOps):
    def __init__(self, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis = 1

    def preprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])

        return dict(output=output)

    def forward(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])
        output = np.squeeze(output, axis=self.axis)

        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])

        return dict(output=output, out_shift=0, out_scale=1)


@OPERATORS.register_module(name='gather')
class Gather(BaseOps):
    '''
    Ni, Nk = a.shape[:axis], a.shape[axis+1:]
    Nj = indices.shape
    for ii in ndindex(Ni):
        for jj in ndindex(Nj):
            for kk in ndindex(Nk):
                out[ii + jj + kk] = a[ii + (indices[jj],) + kk]
    '''
    def __init__(self, **kwargs):
        super(Gather, self).__init__(**kwargs)
        self.axis = kwargs["axis"]
        self.indices = kwargs["indices"]

    def preprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])

        return dict(output=output)

    def forward(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])
        output = np.take(output, indices=self.indices, axis=self.axis)

        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])

        return dict(output=output, out_shift=0, out_scale=1)


@OPERATORS.register_module(name='splice')
class Splice(Act):
    def __init__(self, **kwargs):
        super(Splice, self).__init__(**kwargs)
        self.context = kwargs['context']
        self.forward_indexes = kwargs['forward_indexes']
        self.has_fc = kwargs['has_fc']
        if self.has_fc:
            self.weights = copy.deepcopy(kwargs['p_weights'])
            self.p_weights = torch.from_numpy(np.array(self.weights, dtype=np.float32))
            if isinstance(self.scales['zk'], np.ndarray):
                zk = np.reshape(self.scales['zk'], (-1, 1))
                self.p_weights = self.p_weights - zk
            else:
                self.p_weights = self.p_weights - self.scales['zk']
            self.bias = torch.from_numpy(kwargs['bias']).type(torch.float32)

    def preprocess(self, inputs, **kwargs):
        if isinstance(inputs, list):
            output = copy.deepcopy(inputs[0]['output'])
        else:
            output = copy.deepcopy(inputs['output'])

        return dict(output=output)

    def forward(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])
        # if output.dtype in [np.float32, np.float64]:
        #     output = self.in_quantize[0].get_quan_data(output)

        output = pyops.pyops.py_cpu_splice(
            output,
            np.array(self.context).astype(np.int32),
            np.array(self.forward_indexes).astype(np.int32))

        # output = self.in_quantize[0].get_dequan_data(output)

        # if output.dtype in [np.float32, np.float64]:
        #     output = self.in_quantize[0].get_quan_data(output)

        # output = torch.from_numpy(output).type(torch.float32)
        # output = F.linear(output, self.p_weights.squeeze(0), bias=self.bias.squeeze(0))
        # output = output.numpy()
        # output = output.astype(np.float32)
        # return dict(output=output)

        # if self.has_fc:
            # if output.dtype in [np.float32, np.float64]:
            #     output = self.in_quantize[0].get_quan_data(output)
            # ### process fc
            # output = torch.from_numpy(output.astype(np.float32))
            # output = F.linear(output, self.p_weights.squeeze(0), bias=self.bias.squeeze(0))
            # # output = F.linear(output, self.p_weights.squeeze(0), bias=None)
            # output = output.numpy()
            # bit_num = max(self.bit_select, self.w_bit_select)
            # dtype = self.bits_dict[self.high_bits_calc(bit_num)]
            # output = output.astype(dtype)
            # if self.process_scale in ["ffloatscale"]:
            #     output = output * self.si['scale'] * self.sk['scale'] #+ self.bias.numpy().astype(np.float32)
            #     output = output.astype(np.float32)
            # else:
            #     if not self.precision:
            #         output = self.shift_data(output, self.out_shift)
            #         output = self.clip_int_bit(output, bit=self.bit_saturation) if self.bit_select % 2 \
            #             else self.clip_uint_bit(output, bit=self.bit_saturation)
            #     output = self.scale_data(output, self.out_scale)
            #     if self.process_scale not in ['ffloatscale']:
            #         output = output + self.scales['zo']  # tspe
            #         if not self.out_type in [np.float32, np.float64]:
            #             output = self.align_bits(output)  # tspe
            #     else:
            #         if isinstance(self.out_quantize, list):
            #             output = self.out_quantize[0].get_dequan_data(output)
            #         else:
            #             output = self.out_quantize['feat']['so0'].get_dequan_data(output)
            #         output = output.astype(self.out_type)

        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])

        return dict(output=output, out_shift=0, out_scale=1)


@OPERATORS.register_module(name='lstm')
class Lstm(BaseOps):
    def __init__(self, **kwargs):
        super(Lstm, self).__init__(**kwargs)

        self.table = []
        self.hidden_size = kwargs['hidden_size']
        self.hx_combine, self.wr_combine = kwargs['hx_combine'], kwargs['wr_combine']

        self.weights = [torch.from_numpy(weight).type(torch.float32) for weight in kwargs['p_weights']]
        self.bias = [torch.from_numpy(bias).type(torch.float32) for bias in kwargs['bias']]
        # self.si, self.sk, self.so = kwargs['si'], kwargs['sk'], kwargs['so']
        self.f_init_h = kwargs['initial_h'].squeeze(0) if 'initial_h' in kwargs.keys() else \
            np.zeros((1, self.hidden_size), dtype=np.float32)
        self.f_init_c = kwargs['initial_c'].squeeze(0) if 'initial_c' in kwargs.keys() else \
            np.zeros((1, self.hidden_size), dtype=np.float32)
        if self.hx_combine:
            self.init_h = self.in_quantize[0].get_quan_data(self.f_init_h)
        else:
            self.init_h = self.in_quantize[1].get_quan_data(self.f_init_h)
        self.h, self.c = torch.from_numpy(self.init_h.astype(np.float32)), torch.from_numpy(self.f_init_c)
        self.init_h_pre, self.init_c_pre = copy.deepcopy(self.init_h), copy.deepcopy(self.f_init_c)

    def reset(self):
        # self.h, self.c = torch.from_numpy(self.init_h.astype(np.float32)), torch.from_numpy(self.f_init_c)
        # self.init_h_pre, self.init_c_pre = copy.deepcopy(self.init_h), copy.deepcopy(self.f_init_c)
        self.c = torch.from_numpy(self.f_init_c)

    def get_init_h(self):
        return self.f_init_h

    def get_init_c(self):
        return self.f_init_c

    def get_scales(self):
        return self.scales

    def get_table(self):
        return self.table

    def fp_lstm_cell(self, feat):
        if not feat.dtype in [np.float32, np.float64]:
            f_data = self.in_quantize[0].get_dequan_data(feat)
        else:
            f_data = copy.deepcopy(feat)
        data_1 = torch.from_numpy(f_data).type(torch.float32)
        # w1, w2 = self.weights[0] * self.sk['scale'], self.weights[1] * self.sk['scale']
        # b1, b2 = self.bias[0] * (self.si[0]['scale'] * self.sk['scale']), self.bias[1] * (self.si[1]['scale'] * self.sk['scale'])
        xw = F.linear(data_1, self.weights[0].squeeze(0), self.bias[0].squeeze(0))
        xr = F.linear(self.h, self.weights[1].squeeze(0), self.bias[1].squeeze(0))
        y = xw + xr
        # it, ft, ct, ot = torch.chunk(y, 4, dim=1)
        it, ot, ft, ct = torch.chunk(y, 4, dim=1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        ct = torch.tanh(ct)
        ot = torch.sigmoid(ot)

        Ct = ft * self.c + it * ct
        ht = ot * torch.tanh(Ct)
        self.h = ht
        self.c = Ct

        return ht

    def lstm_cell(self, feat, is_update_quantize_from_in_data=False):
        if feat.dtype in [np.float32, np.float64]:
            if is_update_quantize_from_in_data:
                self.in_quantize[0].get_quan_param(feat)
                self.si[0] = self.in_quantize[0].get_quant_param()
            f_data = self.in_quantize[0].get_quan_data(feat)
        else:
            f_data = copy.deepcopy(feat)
        hq_copy = copy.deepcopy(f_data)
        data_type = np.int32 if hq_copy.dtype == np.int8 else np.int64
        # data_type = np.int32  # if hq_copy.dtype == np.int8 else np.int64
        data_1 = torch.from_numpy(f_data).type(torch.float32)
        xw = F.linear(data_1, self.weights[0].squeeze(0), self.bias[0].squeeze(0))
        # xw = torch.clip(xw, -2147483648, 2147483647)
        fc1_copy = copy.deepcopy(xw).detach().numpy().astype(data_type)
        xr = F.linear(self.h, self.weights[1].squeeze(0), self.bias[1].squeeze(0))
        # xr = torch.clip(xr, -2147483648, 2147483647)
        fc2_copy = copy.deepcopy(xr).detach().numpy().astype(data_type)
        si_xr = self.si[0]['scale'] if self.hx_combine else self.si[1]['scale']
        sk_xr = self.sk[0]['scale'] if self.wr_combine else self.sk[1]['scale']
        xw = xw * self.si[0]['scale'] * self.sk[0]['scale']
        xr = xr * si_xr * sk_xr
        y = xw + xr
        fc_copy = copy.deepcopy(y).detach().numpy().astype(np.float32)
        # it, ft, ct, ot = torch.chunk(y, 4, 1)
        it, ot, ft, ct = torch.chunk(y, 4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        ct = torch.tanh(ct)
        ot = torch.sigmoid(ot)

        Ct = ft * self.c + it * ct
        ht = ot * torch.tanh(Ct)
        if self.hx_combine:
            h = self.in_quantize[0].get_quan_data(ht.numpy())
        else:
            h = self.in_quantize[1].get_quan_data(ht.numpy())
        self.h = torch.from_numpy(h).type(torch.float32)
        self.c = Ct
        self.init_h_pre = copy.deepcopy(self.h.detach().numpy())
        self.init_c_pre = copy.deepcopy(self.c.detach().numpy())

        return ht, hq_copy, fc1_copy, fc2_copy, fc_copy

    def preprocess(self, inputs, **kwargs):
        in_data = copy.deepcopy(inputs[0]['output'])

        return dict(p_inputs=in_data)

    def forward(self, inputs, **kwargs):
        # self.init_h = self.init_h_pre
        # self.init_c = self.init_c_pre
        is_update_quantize_from_in_data = kwargs.get("is_update_quantize_from_in_data", False)
        time_steps = inputs['p_inputs'].shape[0]
        outputs = []
        for time in range(time_steps):
            data = inputs['p_inputs'][time]
            ht, hq_copy, fc1_copy, fc2_copy, fc_copy = \
                self.lstm_cell(data, is_update_quantize_from_in_data=is_update_quantize_from_in_data)
            # ht = self.fp_lstm_cell(data)
            outputs.append(ht.unsqueeze(0))

        if hasattr(torch, 'concat'):
            outputs = torch.concat(outputs, dim=0).numpy()
        else:
            outputs = torch.cat(outputs, dim=0).numpy()

        # np.save('output1_quant.npy', output[0] * self.so[0])
        # fp = np.load('output1_float.npy')
        # quant = np.load('output1_quant.npy')
        # error = np.sum(np.abs(fp - quant)) / np.sum(np.abs(fp))
        # print('fp: ', fp.min(), fp.max())
        # print('quant: ', quant.min(), quant.max())
        # print('error: ', error, self.so[0])
        
        # if self.c.abs().max() > 100:
        #     self.reset()

        return dict(output=outputs, hq_copy=hq_copy, fc1_copy=fc1_copy, fc2_copy=fc2_copy, fc_copy=fc_copy)# type: ignore

    def postprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])
        if not self.out_type in [np.float32, np.float64]:
            output = self.out_quantize['feat']['so0'].get_quan_data(output)
        if self.hx_combine:
            h = self.in_quantize[0].get_dequan_data(self.h.numpy())
        else:
            h = self.in_quantize[1].get_dequan_data(self.h.numpy())
        return [dict(output=output),
                dict(output=h),
                dict(output=self.c.numpy()),
                dict(output=inputs['hq_copy']),
                dict(output=inputs['fc1_copy']),
                dict(output=inputs['fc2_copy']),
                dict(output=inputs['fc_copy']),
                # dict(output=inputs['fc1_copy']+inputs['fc2_copy']),
                ]


@OPERATORS.register_module(name='gru')
class Gru(BaseOps):
    def __init__(self, **kwargs):
        super(Gru, self).__init__(**kwargs)

        self.table = []
        self.hidden_size = kwargs['hidden_size']
        self.hx_combine, self.wr_combine = kwargs['hx_combine'], kwargs['wr_combine']
        self.linear_before_reset = kwargs['linear_before_reset']

        self.weights = [torch.from_numpy(weight).type(torch.float32) for weight in kwargs['p_weights']]
        self.bias = [torch.from_numpy(bias).type(torch.float32) for bias in kwargs['bias']]
        self.init_h = None
        self.h = np.zeros([1, 1, self.hidden_size], dtype=np.float32)
        self.h = kwargs.get("init_h", self.h).squeeze(0)
        self.__reset = True

    def reset(self):
        self.__reset = True

    def get_init_h(self):
        return self.init_h

    def get_scales(self):
        return self.scales

    def get_table(self):
        return self.table

    def fp_gru_cell(self, feat):
        if not feat.dtype in [np.float32, np.float64]:
            f_data = self.in_quantize[0].get_dequan_data(feat)
        else:
            f_data = copy.deepcopy(feat)

        x = torch.from_numpy(f_data).type(torch.float32)
        self.h = torch.from_numpy(self.h).type(torch.float32)

        W, R = self.weights
        Wb, Rb = self.bias

        gate_x = F.linear(x, W.squeeze(dim=0), Wb.squeeze(dim=0))
        gate_h = F.linear(self.h, R.squeeze(dim=0), Rb.squeeze(dim=0))

        ir, iz, ih = gate_x.chunk(3, 1)
        hr, hz, hh = gate_h.chunk(3, 1)

        rt = F.sigmoid(ir + hr)
        zt = F.sigmoid(iz + hz)
        if self.linear_before_reset != 0: ### pytorch default is 1
            ht = F.tanh(ih + (rt * hh)) 
        else: ### onnx default is 0
            tmp = rt * self.h
            Rh = R.chunk(3, dim=1)[-1]
            Rbh = Rb.chunk(3, dim=1)[-1]
            tmp = F.linear(tmp, Rh.squeeze(dim=0), Rbh.squeeze(dim=0))
            ht = F.tanh(ih + tmp)
        
        Ht = (1 - zt) * ht + zt * self.h
        self.h = Ht.numpy().astype(np.float32)

        return Ht

    def gru_cell(self, feat):
        if feat.dtype in [np.float32, np.float64]:
            f_data = self.in_quantize[0].get_quan_data(feat)
        else:
            f_data = copy.deepcopy(feat)
        hq_copy = copy.deepcopy(f_data)
        data_type = np.int32 if hq_copy.dtype == np.int8 else np.int64
        x = torch.from_numpy(f_data).type(torch.float32)

        #### self.h is np.float32/np.float64, h is np.int8/np.int16
        if self.h.dtype in [np.float32, np.float64]:
            h = self.in_quantize[1].get_quan_data(self.h)
        else:
            h = copy.deepcopy(self.h)
        h = torch.from_numpy(h).type(torch.float32)
        self.h = torch.from_numpy(self.h).type(torch.float32)

        if self.hx_combine:
            si_0, si_1 = self.si[0]['scale'], self.si[0]['scale']
        else:
            si_0, si_1 = self.si[0]['scale'], self.si[1]['scale']
        if self.wr_combine:
            sk_0, sk_1 = self.sk[0]['scale'], self.sk[0]['scale']# type: ignore
        else:
            sk_0, sk_1 = self.sk[0]['scale'], self.sk[1]['scale']# type: ignore

        W = self.weights[0]
        R = self.weights[1]
        Wb = self.bias[0]
        Rb = self.bias[1]

        gate_x = F.linear(x, W.squeeze(0), Wb.squeeze(0))
        fc1_copy = copy.deepcopy(gate_x).detach().numpy().astype(data_type)
        gate_h = F.linear(h, R.squeeze(0), Rb.squeeze(0))
        fc2_copy = copy.deepcopy(gate_h).detach().numpy().astype(data_type)

        gate_x = gate_x * si_0 * sk_0
        gate_h = gate_h * si_1 * sk_1

        iz, ir, ih = gate_x.chunk(3, 1)
        hz, hr, hh = gate_h.chunk(3, 1)

        rt = F.sigmoid(ir + hr)
        zt = F.sigmoid(iz + hz)
        if self.linear_before_reset != 0: ### pytorch default is 1
            ht = F.tanh(ih + (rt * hh)) 
        else: ### onnx default is 0
            tmp = rt * self.h
            tmp = tmp.numpy().astype(np.float32)
            tmp = self.in_quantize[1].get_quan_data(tmp)
            tmp = torch.from_numpy(tmp).type(torch.float32)
            Rh = self.weights[1].chunk(3, dim=1)[-1]
            Rbh = self.bias[1].chunk(3, dim=1)[-1]
            tmp = F.linear(tmp, Rh.squeeze(dim=0), Rbh.squeeze(dim=0))
            tmp = tmp * si_1 * sk_1
            ht = F.tanh(ih + tmp)

        Ht = (1 - zt) * ht + zt * self.h
        self.h = Ht.numpy().astype(np.float32)

        return Ht, hq_copy, fc1_copy, fc2_copy

    def preprocess(self, inputs, **kwargs):
        if isinstance(inputs, list):
            in_data = copy.deepcopy(inputs[0]['output'])
            h = copy.deepcopy(inputs[1]['output']).squeeze(0)
        else:
            in_data = copy.deepcopy(inputs['output'])
            h = copy.deepcopy(self.h)

        if self.init_h is None:
            self.init_h = copy.deepcopy(h)

        if self.__reset:
            self.h = copy.deepcopy(self.init_h)
            self.__reset = False
        else:
            if self.h is None:
                self.h = copy.deepcopy(h)

        return dict(p_inputs=in_data)

    def forward(self, inputs, **kwargs):
        time_steps = inputs['p_inputs'].shape[0]
        outputs = []
        for time in range(time_steps):
            data = inputs['p_inputs'][time]
            ht, hq_copy, fc1_copy, fc2_copy = self.gru_cell(data)
            # ht = self.fp_gru_cell(data)
            outputs.append(ht.unsqueeze(0))

        if hasattr(torch, 'concat'):
            outputs = torch.concat(outputs, dim=0).numpy()
        else:
            outputs = torch.cat(outputs, dim=0).numpy()

        # np.save('output1_quant.npy', output[0] * self.so[0])
        # fp = np.load('output1_float.npy')
        # quant = np.load('output1_quant.npy')
        # error = np.sum(np.abs(fp - quant)) / np.sum(np.abs(fp))
        # print('fp: ', fp.min(), fp.max())
        # print('quant: ', quant.min(), quant.max())
        # print('error: ', error, self.so[0])

        return dict(output=outputs, hq_copy=hq_copy, fc1_copy=fc1_copy, fc2_copy=fc2_copy)# type: ignore
        # return dict(output=outputs)

    def postprocess(self, inputs, **kwargs):
        output = copy.deepcopy(inputs['output'])
        if not self.out_type in [np.float32, np.float64]:
            output = self.out_quantize['feat']['so0'].get_quan_data(output)
        return [
            dict(output=output),
            dict(output=self.h),
            dict(output=inputs['hq_copy']),
            dict(output=inputs['fc1_copy']),
            dict(output=inputs['fc2_copy']),
        ]


@OPERATORS.register_module(name='layernormalization')
class LayerNormalization(BaseOps):
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = kwargs['axis'] if 'axis' in kwargs else -1
        self.bias, self.scale = torch.from_numpy(kwargs['bias']), torch.from_numpy(kwargs['scale'])
        self.epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1.0e-5
        self.in_quantize, self.out_quantize = kwargs['in_quantize'], kwargs['quantize']
        self.out_type = eval(self.out_type) if isinstance(self.out_type, str) else self.out_type

    def preprocess(self, inputs, **kwargs):
        # in_data = copy.deepcopy(inputs['output'])
        return inputs

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']
        assert self.process_scale == 'float'
        # in_data = self.in_quantize.get_dequan_data(in_data) # close by qn
        if not in_data.dtype in [np.float32, np.float64]:
            in_data = self.in_quantize.get_dequan_data(in_data)

        in_data = torch.from_numpy(in_data.astype(np.float32))

        # if self.axis < 0:
        #     normalized_shape = in_data.shape[self.axis - 1:]
        # else:
        #     normalized_shape = in_data.shape[self.axis:]
        out_data = F.layer_norm(in_data, normalized_shape=self.scale.shape,
                                weight=self.scale, bias=self.bias, eps=self.epsilon).numpy()
        if not self.out_type in [np.float32, np.float64]:
            out_data = self.out_quantize.get_quan_data(out_data)
        return dict(output=out_data)

    def postprocess(self, inputs, **kwargs):
        in_data = inputs['output']
        # output = self.out_quantize.get_quan_data(in_data) ## close by qn
        return dict(output=in_data, out_shift=0, out_scale=1)


@OPERATORS.register_module(name='instancenormalization')
class InstanceNormalization(BaseOps):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = kwargs['axis'] if 'axis' in kwargs else -1
        self.bias, self.scale = torch.from_numpy(kwargs['bias']), torch.from_numpy(kwargs['scale'])
        self.epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1.0e-5
        self.in_quantize, self.out_quantize = kwargs['in_quantize'], kwargs['quantize']
        self.out_type = eval(self.out_type) if isinstance(self.out_type, str) else self.out_type

    def preprocess(self, inputs, **kwargs):
        # in_data = copy.deepcopy(inputs['output'])
        return inputs

    def forward(self, inputs, **kwargs):
        in_data = inputs['output']
        assert self.process_scale == 'float'
        # in_data = self.in_quantize.get_dequan_data(in_data) # close by qn
        if not in_data.dtype in [np.float32, np.float64]:
            in_data = self.in_quantize.get_dequan_data(in_data)

        in_data = torch.from_numpy(in_data.astype(np.float32))

        # if self.axis < 0:
        #     normalized_shape = in_data.shape[self.axis - 1:]
        # else:
        #     normalized_shape = in_data.shape[self.axis:]
        out_data = F.instance_norm(in_data, weight=self.scale, bias=self.bias, eps=self.epsilon).numpy()
        if not self.out_type in [np.float32, np.float64]:
            out_data = self.out_quantize.get_quan_data(out_data)
        return dict(output=out_data)

    def postprocess(self, inputs, **kwargs):
        in_data = inputs['output']
        # output = self.out_quantize.get_quan_data(in_data) ## close by qn
        return dict(output=in_data, out_shift=0, out_scale=1)


# channel attention operation
# pixel attention operation
@OPERATORS.register_module(name='mul')
@OPERATORS.register_module(name='cmul')
@OPERATORS.register_module(name='pmul')
class CMul(Act):
    # fp/int scale
    def __init__(self, **kwargs):
        kwargs['isolated'] = False
        super(CMul, self).__init__(**kwargs)
        # setting = copy.deepcopy(kwargs)
        # self.si, self.sk, self.so = kwargs['si'], kwargs['sk'], kwargs['so']

        # self.zk = extract_scale(kwargs['sk'], 'zero_point', np.int32(0))

        self.p_weights = kwargs.get('p_weights')
        self.f_weights = kwargs.get('f_weights')
        self.weight_idx = kwargs.get('weight_idx')
        self.p_weights = self.p_weights if isinstance(self.p_weights, list) else [self.p_weights]

    def process(self, output0, output1):
        return output0 * output1

    def preprocess(self, inputs, **kwargs):
        return BaseOps.preprocess(self, inputs)
        # in_datas = copy.deepcopy(inputs)
        # return dict(output=in_datas)

    def forward(self, inputs, **kwargs):
        if isinstance(self.weight_idx, list):
            in_data0, in_data1 = self.weight_insert_into_inputs(inputs, self.p_weights, self.weight_idx)
        else:
            in_data0, in_data1 = inputs['output']

        in_data0, in_data1 = in_data0['output'], in_data1['output']
        # self.process_scale = "float"
        if self.process_scale in ['float']:
            if in_data0.dtype in [np.float32, np.float64]:
                output0 = in_data0
            else:
                output0 = self.in_quantize[0].get_dequan_data(in_data0)
            if in_data1.dtype in [np.float32, np.float64]:
                output1 = in_data1
            else:
                output1 = self.in_quantize[1].get_dequan_data(in_data1)
        else:
            # assert self.process_scale in ['floatscale', 'intscale', 'intscaleex',
            #                               'rshiscale', 'rrshiftscale',
            #                               'shiftfloatscaletable']
            is_asym = True
            if "sym" in self.in_quantize[0].get_class_name() and "sym" in \
                self.in_quantize[1].get_class_name():
                is_asym = False

            dtype = self.bits_dict[self.high_bits_calc(self.bit_select)] if is_asym else\
                self.bits_dict[self.lower_bits_calc(self.bit_select)]
            # if True:
            #     if in_data0.dtype in [np.float32, np.float64]:
            #         self.in_quantize[0].get_quan_param(in_data0)
            #         self.si = self.in_quantize[0].get_quant_param()
            #         in_data0 = self.in_quantize[0].get_quan_data(in_data0)
            #     if in_data1.dtype in [np.float32, np.float64]:
            #         # quant = copy.deepcopy(self.in_quantize[1])
            #         self.in_quantize[1].reset_bit_select(1)
            #         self.in_quantize[1].get_quan_param(in_data1)
            #         self.sk = self.in_quantize[1].get_quant_param()
            #         in_data1 = self.in_quantize[1].get_quan_data(in_data1)
            #     self.datacorrect = \
            #         datacorrect_factoy.get(self.process_scale)(maxs=self.maxs,
            #                                                    mins=self.mins,
            #                                                    bits_dict=self.bits_dict,
            #                                                    bit_select=self.bit_select,
            #                                                    int_scale=self.int_scale,
            #                                                    si=self.si, sk=self.sk, so=self.so)
            #     self.scales = self.datacorrect() # type: ignore
            output0 = in_data0.astype(dtype) - self.scales['zi']
            output1 = in_data1.astype(dtype) - self.scales['zk']
        if output0.shape[-1] == output1.shape[-1] and len(output0.shape) != len(output1.shape):
            if len(output0.shape) < len(output1.shape):
                output0 = output0.reshape(-1)
            else:
                output1 = output1.reshape(-1)
        else:
            output0, output1 = self.boardcast(output0, output1)

        output = self.process(output0, output1)

        return dict(output=output)

    def postprocess(self, inputs, **kwargs):
        output = inputs['output']
        # out_shift, out_scale = self.scales['out_shift'], self.scales['out_scale']
        # print(self.out_shift)
        if not self.process_scale in ['float', 'floatscale', 'ffloatscale']:
            output = self.shift_data(output, self.out_shift)
            if self.bit_select % 2:
                output = self.clip_int_bit(output, bit=self.bit_saturation) # type: ignore
            else:
                output = self.clip_uint_bit(output, bit=self.bit_saturation)# type: ignore

        if self.process_scale in ['float', 'ffloatscale']:
            if not self.out_type in [np.float32, np.float64]:
                output = self.out_quantize.get_quan_data(output)
                output = self.align_bits(output)
        elif self.process_scale in ['shiftfloatscaletable']:
            output = self.lookup(output, self.table, self.lower_bound)
        else:
            output = self.scale_data(output, self.out_scale) + self.scales['zo']
            output = self.align_bits(output)

        return {'output': output, 'out_shift': self.out_shift, 'out_scale': self.out_scale}


@OPERATORS.register_module(name='matmul')
class MatMul(CMul):
    # fp/int scale
    def __init__(self, **kwargs):
        kwargs['isolated'] = False
        super(MatMul, self).__init__(**kwargs)
        
    def process_(self, output0, output1):
        return np.matmul(output0, output1)

    def forward(self, inputs, **kwargs):
        if isinstance(self.weight_idx, list):
            in_data0, in_data1 = self.weight_insert_into_inputs(inputs, self.p_weights, self.weight_idx)
        else:
            in_data0, in_data1 = inputs['output']
        in_data0, in_data1 = in_data0['output'], in_data1['output']
        if self.process_scale in ['float', 'ffloatscale']:
            if in_data0.dtype in [np.float32, np.float64]:
                output0 = in_data0
            else:
                output0 = self.in_quantize[0].get_dequan_data(in_data0)
            if in_data1.dtype in [np.float32, np.float64]:
                output1 = in_data1
            else:
                output1 = self.in_quantize[1].get_dequan_data(in_data1)
        else:
            assert self.process_scale in ['floatscale', 'intscale', 'rshiscale', 'rrshiftscale']
            dtype = self.bits_dict[self.high_bits_calc(self.bit_select)]
            output0 = in_data0.astype(dtype) - self.scales['zi']
            output1 = in_data1.astype(dtype) - self.scales['zk']

        output = self.process_(output0, output1)

        return dict(output=output)

#    if __name__ == '__main__':
#     ops_types = ['Conv2d', 'DepthwiseConv2d', 'Conv2d']
#     ops_instance = [OPERATORS.get(item) for item in ops_types]
#     registry_class = OPERATORS._module_dict
#     print('test done!')
