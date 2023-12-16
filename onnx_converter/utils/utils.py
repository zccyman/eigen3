# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/6 13:35
# @File     : attribute.py
import copy
import itertools
import math
import os
import os.path as osp
import struct
import warnings
from importlib import import_module
from tkinter import _flatten # type: ignore

import cv2
import numpy as np
import onnx
import onnxruntime as rt
import torch

bits_dict = {8: np.int8, 16: np.int16, 32: np.int32}

process_shift = lambda data, shift: data >> shift if shift >= 0 else np.right_shift(data, -shift)
process_lr_shift = lambda data, shift: data << shift if shift >= 0 else np.right_shift(data, -shift)


fliter_perchannel = lambda out_shift: isinstance(out_shift, np.ndarray)

def save_txt(file_name, mode, context):
    write_context = context
    if isinstance(context, np.ndarray):
        write_context = context.reshape(-1).tolist()
    elif isinstance(context, list):
        write_context = list(_flatten(context))
    elif isinstance(context, tuple):
        write_context = list(_flatten(list(context)))
    elif isinstance(context, np.float32) or isinstance(context, np.int32) or \
         isinstance(context, np.float64):
        write_context = [context]
    elif isinstance(context, str):
        write_context = [context]
    else:
        print("data format not support!")
        os._exit(-1)
    with open(file_name, mode) as f:
        for item in write_context:
            f.write(str(item) + ",\n")

def props_with_(obj):
    params = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            params[name] = value
    return params


def export_perchannel(layer, weights_dir, func_bias, save_weights, w_offset, calc_w_offset, is_fc_bias=False):
    scales = layer.get_scales()[-1]
    out_shift, out_scale, extra_value = scales['out_shift'], scales['out_scale'], scales['extra_value']
    if isinstance(out_shift, np.ndarray):
        tmp_offset = [w_offset['w_offset']]
        
        out_shift = out_shift.astype(np.int8)
        # if 'int' in out_scale.dtype.name:
        if 'intscale' in layer.get_scale_type():
            if isinstance(out_scale, np.ndarray):
                out_scale = out_scale.astype(np.int32)
            else:
                out_scale = np.int32(out_scale)
        else:
            if isinstance(out_scale, np.ndarray):
                out_scale = out_scale.astype(np.float32)
            else:
                out_scale = np.float32(out_scale)
            
        write_s = lambda data, func_bias, name: layer.bias_export(func_bias,
                                                            data,
                                                            is_fc_bias=is_fc_bias,
                                                            name=name)
        out_fscale = scales['fscale'] if 'fscale' in scales.keys() else 0
        int_scale = int(list(filter(str.isdigit, layer.get_out_data()[-1]['output'].dtype.type.__name__))[0])
        int_scale = scales['int_scale'] if 'int_scale' in scales.keys() else int_scale
        name = os.path.join(weights_dir, "weight.b")
        
        res = write_s(out_shift, func_bias, name)
        save_weights.extend(res)
        w_offset['w_offset'] += calc_w_offset(res)

        if np.sum(np.abs(extra_value)) > 0:
            res = write_s(-extra_value, func_bias, name)
            save_weights.extend(res)
            w_offset['w_offset'] += calc_w_offset(res)
                
        if layer.get_is_result_layer():
            if layer.get_scale_type() in ["rshiftscale", "rrshiftscale"]:
                res = write_s(out_fscale, func_bias, name)
            else:
                res = write_s(out_scale, func_bias, name)
        else:
            res = write_s(out_scale, func_bias, name)
        save_weights.extend(res)
        w_offset['w_offset'] += calc_w_offset(res)

        ### write intscale array into model.c, when intscale
        if layer.get_scale_type() in ['intscale']:
            int_scale = int_scale * np.ones(out_shift.shape).astype(np.int8)
            res = write_s(int_scale, func_bias, name)
            save_weights.extend(res)
            w_offset['w_offset'] += calc_w_offset(res)
            
        is_asymquan = 'asymquan' in layer.get_ops_setting()['setting']['method']
        if is_asymquan:
            in_zero, out_zero = scales['zi'].astype(np.int32), scales['zo'].astype(np.int32)
            for zero_point in [in_zero, out_zero]:
                res = write_s(zero_point, func_bias, name)
                save_weights.extend(res)
                w_offset['w_offset'] += calc_w_offset(res)
    else:
        tmp_offset = [-1]
    
    w_offset['tmp_offset'].extend(tmp_offset)     
    layer.set_w_offset(copy.deepcopy(w_offset))
                
    return w_offset, save_weights


def get_last_layer_quant(layer, is_shift=True):
    scales = copy.deepcopy(layer.get_scales())
    if isinstance(scales, list):
        scales = scales[0]

    is_perchannel = isinstance(scales['out_shift'], np.ndarray)
    is_asymquan = 'asymquan' in layer.get_ops_setting()['setting']['method']
    layer_scale_type = layer.get_scale_type()
    if layer_scale_type in ['rshiftscale', 'rrshiftscale']:
        if is_shift:
            if is_perchannel: #perchannel
                quant = 'QUANT_PER_CHN_SHIFT_ASY' if is_asymquan else 'QUANT_PER_CHN_SHIFT'
            else:
                quant = 'QUANT_SHIFT'
        else:
            if is_perchannel: #perchannel
                quant = 'QUANT_PER_CHN_FSCALE_ASY' if is_asymquan else 'QUANT_PER_CHN_FSCALE'
            else:
                quant = 'QUANT_FSCALE'
    elif layer_scale_type in ['intscale']:
        if is_perchannel: #perchannel
            quant = 'QUANT_PER_CHN_ISCALE_ASY' if is_asymquan else 'QUANT_PER_CHN_ISCALE'
        else:
            quant = 'QUANT_ISCALE_ASY' if is_asymquan else 'QUANT_ISCALE'
    elif layer_scale_type in ['ffloatscale']:
        quant = 'QUANT_FSCALE'
    else:
        raise Exception('Not supported : {} in {} layer'.format(
            layer_scale_type, layer.get_layer_type()))

    return quant


def get_scale_param(layer, quant):
    scales = copy.deepcopy(layer.get_scales())
    if isinstance(scales, list):
        scales = scales[-2] if len(scales) > 3 else scales[-1]
    out_shift, out_scale = scales['out_shift'], scales['out_scale']
    if layer.get_is_result_layer():
        out_fscale = scales['fscale'] if 'fscale' in scales.keys() else 0
    else:
        out_fscale = scales['out_scale'] if 'out_scale' in scales.keys() else 0
    layer_out_data = layer.get_out_data()
    if isinstance(layer_out_data, list):
        int_scale = int(list(filter(str.isdigit, layer_out_data[-1]['output'].dtype.type.__name__))[0])
    else:
        int_scale = int(list(filter(str.isdigit, layer_out_data['output'].dtype.type.__name__))[0])
    int_scale = scales['int_scale'] if 'int_scale' in scales.keys() else int_scale
    int_scale = -int_scale
    if quant == 'QUANT_FSCALE':
        qparam = [0, 1.0] if fliter_perchannel(out_shift) else [out_shift, out_fscale]
    elif quant == 'QUANT_SHIFT':
        qparam = [0] if fliter_perchannel(out_shift) else [out_shift]
    elif quant == 'QUANT_ISCALE':
        qparam = [0, 0, 0] if fliter_perchannel(out_shift) else [out_shift, out_scale, int_scale]
    else:
        raise Exception('Not support quantize data correct!')
    
    is_asymquan = 'symquan' not in layer.get_ops_setting()['setting']['method']
    if is_asymquan:
        in_zero, out_zero = scales['zi'], scales['zo']
        qparam.extend([in_zero, out_zero])

    return qparam


def extract_scale(input, key='scale', default_value=1.0):
    # scale = 1.0
    if isinstance(input, list):
        scale = extract_scale(input[0], key=key, default_value=default_value)
    elif isinstance(input, dict):
        scale = input[key]
    elif isinstance(input, float):
        scale = input
    elif isinstance(input, np.ndarray):
        scale = input
    else:
        scale = default_value

    return scale


def clip(data, min_v, max_v):
    data[data < min_v] = min_v
    data[data > max_v] = max_v
    return data


def add_features_to_output(m: onnx.ModelProto) -> None:
    for node in m.graph.node: # type: ignore
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)]) # type: ignore


def add_layer_output_to_graph(model, output_names, input_names):
    if isinstance(model, str):
        model_ = onnx.load_model(model)
    else:
        model_ = copy.deepcopy(model)
    output_names_ = flatten_list(output_names)
    input_names_ = flatten_list(input_names)
    for output in output_names_:
        if output in input_names_: continue
        model_.graph.output.extend([onnx.ValueInfoProto(name=output)]) # type: ignore

    return model_


def onnxruntime_infer(model_path, output_names, input_names, img=None, shape=(640, 640)):
    rt.set_default_logger_severity(3)
    if isinstance(model_path, str):
        sess = rt.InferenceSession(model_path)
    else:
        sess = rt.InferenceSession(model_path.SerializeToString())

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    output_names = [out.name for out in outputs]
    input_name = inputs[0].name
    mean = [0.14300402, 0.1434545, 0.14277956],
    std = [0.10050353, 0.100842826, 0.10034215]
    if img is not None:
        input = cv2.imread(img)
    else:
        input = np.random.randn(320, 320, 3)
        input = (input-np.min(input)) / (np.max(input)-np.min(input))
        input = (input - mean) / std
    try:
        input = cv2.resize(input, tuple(inputs[0].shape[2:]))
        input = (input - np.min(input)) / (np.max(input) - np.min(input))
        input = (input - mean) / std
        input = np.transpose(input, (2, 1, 0)).reshape(1, 3, inputs[0].shape[2], inputs[0].shape[3])
    except:
        input = cv2.resize(input, shape)
        input = (input - np.min(input)) / (np.max(input) - np.min(input))
        input = (input - mean) / std
        input = np.transpose(input, (2, 1, 0)).reshape(1, 3, shape[1], shape[0])
    input = input.astype(np.float32)
    preds = sess.run(output_names=output_names, input_feed={input_name: input})
    results = dict()
    for idx, _ in enumerate(output_names):
        results[output_names[idx]] = preds[idx]
    results[flatten_list(input_names)[0]] = input
    return results


def shift_data(inputs, shifts):
    if isinstance(shifts, np.ndarray):
        for idx, shift in enumerate(shifts):
            inputs[:, idx] = process_shift(inputs[:, idx], shift)
            # inputs[:, idx] = process_lr_shift(inputs[:, idx], shift)
    else:
        inputs = process_shift(inputs, shifts)
        # inputs = process_lr_shift(inputs, shifts)

    return inputs


def shift_1d(inputs, shifts):
    if isinstance(shifts, np.ndarray):
        for idx, shift in enumerate(shifts):
            inputs[idx] = process_shift(inputs[idx], shift)
    else:
        inputs = process_shift(inputs, shifts)

    return inputs


def clip_values(inputs, values, is_gt=True):
    # clip_upper = lambda x, value: x[x>value]
    # clip_lower = lambda x, value: x[x<value]
    if isinstance(values, np.ndarray):
        for idx, item in enumerate(values):
            if is_gt:
                inputs[:,idx,:,:][inputs[:,idx,:,:] > item] = item
            else:
                inputs[:,idx,:,:][inputs[:,idx,:,:] < item] = item
    else:
        if is_gt:
            inputs[inputs > values] = values
        else:
            inputs[inputs < values] = values


def scale_data(inputs, scales):
    # if inputs.dtype.name in ['int8', 'uint8', 'int16', 'uint16']:
    #     inputs = inputs.astype(np.int32)
    if isinstance(scales, np.ndarray):
        for idx, scale in enumerate(scales):
            inputs[:, idx] = np.round(inputs[:, idx] * scale)
    else:
        inputs = inputs * scales
    return inputs


def invert_dict(dct):
    return dict(zip(dct.values(), dct.keys()))


def flatten_list(inputs):
    return list(_flatten(inputs))


check_connect = lambda node1, node2: node1.get_name() in node2.get_output()


def get_scale_shift(scale, bit=64, lower=0.5):
    if isinstance(scale, np.ndarray):
        shape = scale.shape
        scale = scale.reshape(-1)
        shifts, scales = np.zeros_like(scale, dtype=np.int32), np.zeros_like(scale)
        for idx, s in enumerate(scale):
            for shift in range(-bit, bit):
                out_scale = s * (2 ** (-shift))
                if lower <= out_scale <= 1:
                    shifts[idx] = shift
                    scales[idx] = out_scale
                    break
        shifts = shifts.reshape(shape)
        scales = scales.reshape(shape)
        return shifts, scales
    else:
        for shift in range(-bit, bit):
            out_scale = scale * (2 ** (-shift))
            if lower <= out_scale <= 1:
                return np.int32(shift), out_scale
        print('Error! Can not get the shift for scale %f' % scale)
        os._exit(-1)


def exhaustive_search(values, value):
    if value in values:
        return values.index(value)
    for index, v in enumerate(values):
        if isinstance(v, list):
            if value in v:
                return index
    return -1


def check_len(inputs, index):
    flag = False if index > len(inputs) - 1 else True
    return flag


def two_node_connect(nodes: list, index, types):
    connect = False
    if check_len(nodes, index + 1) and nodes[index + 1].get_op_type().lower() in types:
        if check_connect(nodes[index + 1], nodes[index]):
            connect = True

    return connect


def type_replace(layer_type, replace_ops):
    if layer_type in list(_flatten([replace_ops[key] for key in replace_ops.keys()])):
        for key in replace_ops.keys():
            if layer_type in replace_ops[key]:
                return key
    return None


def nodes_connect(nodes: list, indexes: list):
    for index, idx in enumerate(indexes):
        if idx >= len(nodes) - 1:
            break
        if not check_connect(nodes[idx + 1], nodes[idx]):
            return False
    return True


def nodes_connect_(nodes: list):
    for idx in range(len(nodes) - 1):
        if not check_connect(nodes[idx + 1], nodes[idx]):
            return False
    return True


def replace_types(nodes, acts, pools):
    types = [nn.get_op_type().lower() for nn in nodes]
    for ity, ty in enumerate(types):
        if ty in acts: types[ity] = 'act'
        if ty in pools: types[ity] = 'pool'
    return types


def shorten_nodes(nodes, op_types, ops):
    type_lambda = lambda especial_types, especial_ops: set(especial_types).issubset(
        especial_ops) and len(set(especial_types)) == len(especial_types)
    shorten_idx = 0
    for i in range(len(op_types)):
        flag = type_lambda(op_types[:len(op_types) - i], ops[:len(ops) - i])
        flag = flag and check_nodes(nodes[:len(op_types) - i])
        if flag:
            shorten_idx = i
            break
    return nodes[:len(nodes) - shorten_idx]


def check_nodes(nodes):
    length = len(nodes)
    flag = False
    if length > 1:
        for i in range(length - 1):
            if not check_connect(nodes[i + 1], nodes[i]):
                return False
        flag = True
    else:
        flag = True
    return flag


def check_shuffle(nodes: list):
    attrs = list()
    for node in nodes:
        if not hasattr(node, 'get_attr'):
          print('node error')
          os._exit(-1)
        attrs.append(node.get_attr())

    return len(attrs[0]['shape']) == 5 and len(attrs[1]['perm']) == 5 and len(attrs[2]['shape']) == 4


def py_cpu_nms(boxes, scores, thresh, max_output_size):
    """Pure Python NMS baseline."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    keep = keep[:max_output_size] if len(keep) > max_output_size else keep

    return keep


'''
transpose data plan
ex: nchw->nhwc, nhwc->nhwc
'''


class DataTranspose(object):
    '''
    pytorch feat: [batch, channels, height, width]
    tensorflow feat: [batch, height, width, channels]
    '''

    def __init__(self, format='nchw'):
        pass

    def transpose(self):
        pass

    def detranspose(self):
        pass


class WeightTranspose(object):
    '''
    pytorch weight: [cout, cin, kernel_h, kernel_w]
    tensorflow weight: [kernel_h, kernel_w, cin, cout]
    '''

    def __init__(self, format='cout_cin_kh_kw'):
        pass

    def transpose(self):
        pass

    def detranspose(self):
        pass


class Int2float(object):
    def __init__(self, bits=8):
        self.upper = (2 ** (bits - 1)) - 1
        self.lower = -(2 ** (bits - 1))

    def __call__(self, data):
        data[data > self.upper] = self.upper
        data[data < self.lower] = self.lower

        return np.array(data, dtype=np.float32)


def toInt8asFloat(data):
    # saturate
    data[data > 127] = 127
    data[data < -128] = -128
    out = data.astype(np.int8)
    out = out.astype(np.float) # type: ignore
    return out


def toInt16asFloat(data):
    # saturate
    data[data > 65535] = 65535
    data[data < -65535] = -65535
    out = data.astype(np.int16)
    out = out.astype(np.float) # type: ignore
    return out


def toInt32asFloat(data):
    # saturate
    '''data[data>65535]=65535
  data[data<-65535]=-65535'''
    out = data.astype(np.int32)
    out = out.astype(np.float) # type: ignore
    return out


def toIntasFloat(data, size=8):
    # satureate
    upperbound = (2 ** (size - 1)) - 1
    lowbound = -(2 ** (size - 1))
    data[data > upperbound] = upperbound
    data[data < lowbound] = lowbound
    if size in bits_dict.keys():
        out = data.astype(bits_dict[size])
    else:
        print('Error!Wrong input size value %d', size)
        os._exit(-1)

    out = out.astype(np.float64)
    return out


def to_bytes(value, dtype=np.uint16):
    if isinstance(value, list):
        value = np.array(value, dtype=dtype)
    elif isinstance(value, np.str) or isinstance(value, np.ndarray): # type: ignore
        pass
    else:
        value = np.array([value], dtype=dtype)
        
    if dtype in [np.int8]:
        value = struct.pack("b" * len(value), *value)
    elif dtype in [np.uint8]:
        value = struct.pack("B" * len(value), *value)
    elif dtype in [np.str]: # type: ignore
        value = value.encode("utf-8") # type: ignore
    elif dtype in [np.int16]:
        value = struct.pack("h" * len(value), *value)
    elif dtype in [np.uint16]:
        value = struct.pack("H" * len(value), *value)
    elif dtype in [np.int32]:
        value = struct.pack("i" * len(value), *value)    
    elif dtype in [np.uint32]:
        value = struct.pack("I" * len(value), *value)   
    elif dtype in [np.int64]:
        value = struct.pack("q" * len(value), *value)    
    elif dtype in [np.uint64]:
        value = struct.pack("Q" * len(value), *value)            
    elif dtype in [np.float32]:
        value = struct.pack("f" * len(value), *value)     
    elif dtype in [np.float64]:
        value = struct.pack("d" * len(value), *value)           
    else:
        raise NotImplemented
    
    return value


def from_bytes(value, dtype=np.uint16):
    if dtype in [np.int8]:
        value = struct.unpack("b" * len(value), value)
    elif dtype in [np.uint8]:
        value = struct.unpack("B" * len(value), value)
    elif dtype in [np.str]: # type: ignore
        value = value.decode('utf-8')
    elif dtype in [np.int16]:
        value = struct.unpack("H" * (len(value) // 2), value)
    elif dtype in [np.uint16]:
        value = struct.unpack("h" * (len(value) // 2), value)
    elif dtype in [np.int32]:
        value = struct.unpack("i" * (len(value) // 4), value)    
    elif dtype in [np.uint32]:
        value = struct.unpack("I" * (len(value) // 4), value)     
    elif dtype in [np.int64]:
        value = struct.unpack("q" * (len(value) // 8), value)    
    elif dtype in [np.uint64]:
        value = struct.unpack("Q" * (len(value) // 8), value)         
    elif dtype in [np.float32]:
        value = struct.unpack("f" * (len(value) // 4), value)     
    elif dtype in [np.float64]:
        value = struct.unpack("d" * (len(value) // 8), value)           
    else:
        raise NotImplemented
    
    return value


def find_key_by_value_in_enum(value, enum_data):
    for k, v in enum_data.__members__.items(): # type: ignore
        if v.value == value:
            return k
    return 0
    
    
def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class Dict2Object(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


def generate_random(x, method="randn", seed=0, range=[-1, 1], is_weight=False):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    def single_batch_random(x, method):
        res = eval("torch." + method)(x)
        if method not in ["zeros", "ones"]:
            min_value, max_value = torch.min(res), torch.max(res)
            res = (range[1] - range[0]) * (res - min_value) / (max_value - min_value) + range[0]
        return res
    
    if not is_weight: 
        if isinstance(x, list) or isinstance(x, np.ndarray):    
            batch_size = x[0]
            x[0] = 1
            
            res_list = []
            for _ in np.arange(batch_size):
                res = single_batch_random(x, method)
                res_list.append(res)
            res = torch.cat(res_list, dim=0)        
        else:
            res = single_batch_random(x, method)
    else:
        res = single_batch_random(x, method)
        
    return res.numpy()


def get_same_padding(in_size, kernel_size, stride, auto_pad="SAME_UPPER"):
    new_size = int(math.ceil(in_size * 1.0 / stride))
    pad_size = (new_size - 1) * stride + kernel_size - in_size
    if auto_pad == "SAME_UPPER":
        pad0 = int(pad_size / 2)
    else:
        pad0 = int((pad_size + 1) / 2)
    pad1 = pad_size - pad0
    return [pad0, pad1]

class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # x, = ctx.saved_tensors
        return grad_output
    
    
class FloorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x)
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        # x, = ctx.saved_tensors
        return grad_output
    
        
    
class ClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        ctx.save_for_backward(x, min_val, max_val)
        return torch.clamp(x, min_val, max_val)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, min_val, max_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < min_val) | (x > max_val)] = 0
        return grad_input, None, None

