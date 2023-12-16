# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/19 10:34
# @File     : preprocess.py
# functional: process some special node, ex: hswish/hsigmoid
import os
import copy
import onnx
# import sys
# sys.path.append("/home/qinnan/workspace/onnx_converter/")
try:
    from utils import Object
    from utils import Registry
except:
    from onnx_converter.utils import Object # type: ignore
    from onnx_converter.utils import Registry # type: ignore
import onnx
import onnxruntime as rt
from onnx import TensorProto
from functools import reduce
import struct
import numpy as np
import copy

NODE_OPERATION: Registry = Registry('node_operation', scope='')

data_type_dict = {
    'int8': TensorProto.INT8,
    'int16': TensorProto.INT16,
    'int32': TensorProto.INT32,
    'int64': TensorProto.INT64,
    'uint8': TensorProto.UINT8,
    'uint16': TensorProto.UINT16,
    'uint32': TensorProto.UINT32,
    'uint64': TensorProto.UINT64,
    'float32': TensorProto.FLOAT,
    'float64': TensorProto.DOUBLE,
}
data_format_dict = {
    'int8': 'b',
    'int16': 'h',
    'int32': 'i',
    'int64': 'l',
    'uint8': 'B',
    'uint16': 'H',
    'uint32': 'I',
    'uint64': 'L',
    'float32': 'f',
    'float64': 'd',
}
data_bytes_dict = {
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8,
    'uint8': 1,
    'uint16': 2,
    'uint32': 4,
    'uint64': 8,
    'float32': 4,
    'float64': 8,
}


online_operation_pipeline = [
    # replace reshape-like nodes with reshape
    {"name": "RepalceSqueeze", "method": "ReplaceReshapeLikeOps", "add_node_types": ["Reshape"], "delete_node_types": ["Squeeze"]},
    {"name": "RepalceUnsqueeze", "method": "ReplaceReshapeLikeOps", "add_node_types": ["Reshape"], "delete_node_types": ["Unsqueeze"]},
    {"name": "RepalceFlatten", "method": "ReplaceReshapeLikeOps", "add_node_types": ["Reshape"], "delete_node_types": ["Flatten"]},
    {"name": "RepalcePad", "method": "ReplaceReshapeLikeOps", "add_node_types": ["Reshape"], "delete_node_types": ["Pad"]},
    {"name":"RepalceTranspose","method":"ReplaceReshapeLikeOps", "add_node_types":["Reshape"], "delete_node_types":["Transpose"]},
    # merge nodes
    {"name": "MergeHardswish", "method": "MergeHardswish", 'add_node_types': ['HardSwish'], "delete_node_types": ['Add', 'Relu', 'Clip', 'Mul', 'Div']},
    {"name": "MergeHardswish2", "method": "MergeHardswish", 'add_node_types': ['HardSwish'], "delete_node_types": ['Add', 'Clip', 'Div', 'Mul']},
    {"name": "MergeHardsigmoid", "method": "MergeHardsigmoid", 'add_node_types': ['HardSigmoid'], "delete_node_types": ['Add', 'Relu', 'Clip', 'Div']},
    # {"name": "FuseConvBn", "method": "FuseConvBn", 'add_node_types': ['Conv'], "delete_node_types": ['Conv', 'BatchNormalization']},
    # {"name": "FuseConvTransposeBn", "method": "FuseConvBn", 'add_node_types': ['ConvTranspose'], "delete_node_types": ['ConvTranspose', 'BatchNormalization']},
    # replace some unsupport ops
    {"name": "MergeLayerNormalization", "method": "MergeLayerNormalization", 'add_node_types': ['LayerNormalization'], "delete_node_types": \
        ['ReduceMean', 'Sub', 'Mul', 'ReduceMean', 'Add', 'Sqrt', 'Div', 'Mul', 'Add']},
    {"name": "MergeLayerNormalization2", "method": "MergeLayerNormalization", 'add_node_types': ['LayerNormalization'], "delete_node_types": \
        ['ReduceMean', 'Sub', 'Pow', 'ReduceMean', 'Add', 'Sqrt', 'Div', 'Mul', 'Add']}, 
    {"name": "ReplaceMatmulReshapeAdd", "method": "ReplaceMatmulReshapeAdd", 'add_node_types': ["Gemm", "Reshape"], "delete_node_types": ["MatMul", "Reshape", "Add"]},       
    {"name": "ReplaceMatmulAdd", "method": "ReplaceMatmulAdd", 'add_node_types': ["Reshape", "Gemm", "Reshape"], "delete_node_types": ["MatMul", "Add"]},    
    {"name": "ReplaceMatmul", "method": "ReplaceMatmul", 'add_node_types': ["Gemm"], "delete_node_types": ["MatMul"]},
    {"name": "ReplaceConv1d", "method": "ReplaceConv1dGemm", "add_node_types": ["Reshape", "Gemm", "Reshape"], "delete_node_types": ["Conv"]},
    {"name": "ReplaceDepthwiseConv2d", "method": "ReplaceDepthwiseConv2d", "add_node_types": ["Mul", "Add"], "delete_node_types": ["Conv"]},    
    {"name": "ReplaceGemmReshapeRelu", "method": "ReplaceGemmReshapeRelu", "add_node_types": ["Gemm", "Relu", "Reshape"], "delete_node_types": ["Gemm", "Reshape", "Relu"]},
    {"name": "ReplaceConv1dWithConv2d", "method": "ReplaceConv1dWithConv2d", "add_node_types": ["Reshape", "Conv", "Reshape"], "delete_node_types": ["Conv"]},
    {"name": "ReplaceSliceOps", "method": "ReplaceSliceOps","add_node_types":["Split"], "delete_node_types":['Slice']},   
    # specific nodes operation
    {"name": "RepalceReduceMean", "method": "ReplaceReshapeLikeOps", "add_node_types": ["Reshape"], "delete_node_types": ["ReduceMean"]},
    {"name": "DeleteExpand", "method": "DeleteOps", 'add_node_types': [], 'delete_node_types': ['Expand']}, # TODO, define DeleteReshapes
    # Multiple reshape merge and redundent reshape delelet
    {"name": "MergeReshapeOps", "method": "MergeReshapeOps", "add_node_types": [], "delete_node_types": ["Reshape"]},
    {'name': "DeleteRedundentReshape", "method": "DeleteRedundentReshapeOps", "add_node_types": [], "delete_node_types": ["Reshape"]},
    {'name': "MergeOps2Weight0", "method": "MergeOps2Weight", "add_node_types": [], "delete_node_types": ["Div"]},
    {'name': "MergeOps2Weight1", "method": "MergeOps2Weight", "add_node_types": [], "delete_node_types": ["Mul"]},
    {'name': "MergeOps2Weight2", "method": "MergeOps2Weight", "add_node_types": [], "delete_node_types": ["Add"]},
    {'name': "MergeOps2Weight3", "method": "MergeOps2Weight", "add_node_types": [], "delete_node_types": ["Sub"]},
]

offline_operation_pipeline = [
    {"name": "RepalceSqueeze", "method": "ReplaceReshapeLikeOps", "add_node_types": ["Reshape"], "delete_node_types": ["Squeeze"]},
    {"name": "DeleteInputTranspose", "method": "DeleteInputOps", "add_node_types": [], "delete_node_types": ['Transpose']},
    {"name": "DeleteOutputSplit", "method": "DeleteOutputOps", "add_node_types": [], "delete_node_types": ['Split']},
    {"name": "DeleteOutputSlice", "method": "DeleteOutputOps", "add_node_types": [], "delete_node_types": ['Slice']},
    {"name": "DeleteOutputConcat", "method": "DeleteOutputOps", "add_node_types": [], "delete_node_types": ['Concat']},
    {"name": "DeleteOutputReshape", "method": "DeleteOutputOps", "add_node_types": [], "delete_node_types": ['Reshape']},
    {"name": "DeleteOutputTranspose", "method": "DeleteOutputOps", "add_node_types": [], "delete_node_types": ['Transpose']}
]

class BaseNodeOperation():
    def __init__(self, **args):
        assert 'model' in args.keys(), "Error, onnx model not found in args"
        assert 'featuremap_shape_dict' in args.keys(),\
            "Error, featuremap_shape_dict not found, which is necessary for onnx model process"
        assert 'delete_node_types' in args.keys() or 'delete_node_names' in args.keys(),\
            "Error, 'delete_node_types' or by 'delete_node_names' not found, Please specify delete nodes either by types or by names"
        
        self.find_ops_by_type = True if 'delete_node_types' in args.keys() else False
        self.delete_node_types = None
        self.delete_node_names = None
        if 'delete_node_types' in args.keys():
            self.delete_node_types = args['delete_node_types']
            self.add_node_types = args['add_node_types'] if 'add_node_types' in args.keys() else None
        else:
            self.delete_node_names = args['delete_node_names']

        self.logger = args["logger"]
        self.node_operate_list = list()
        self.model = args['model']
        self.nodes = self.model.graph.node
        self.featuremap_shape_dict = args['featuremap_shape_dict']
        self.initializer_dict = self.make_initializer_dict()
        #self.add_node_num = len(self.add_node_types) if self.add_node_types else 0
        self.delete_node_num = len(self.delete_node_types) if self.delete_node_types else 0
        self.inputs = self.model.graph.input
        self.outputs = self.model.graph.output
        self.data_type_dict = {
            "0": np.float16, "1": np.float32, "2": np.float64,
            "3": np.uint32, "4": np.uint64, "5": np.int32,
            "6": np.int64,
        }
        self.attribute_data_type_dict = {
            "7": "INTS", "2": "INT",
        }
        
    def make_initializer_dict(self):
        initializer_dict = dict()
        for x in self.model.graph.initializer:
            initializer_dict.update({x.name:x})
        return initializer_dict

    def create_initializer(self, **args):
        tensor = TensorProto()
        tensor.name = args['name']
        data_type = args['data_type']
        shape = args['shape']
        assert isinstance(shape, list), "error: shape should be a list!"
        assert data_type in data_type_dict.keys(), 'error: data type invalid!'
        tensor.data_type = data_type_dict[data_type]
        tensor.dims.MergeFrom(shape) # type: ignore
        data = args['data']
        n = len(data)
        format = data_format_dict[data_type]
        tensor.raw_data = struct.pack("%d%s" % (n, format), *data)
        return tensor

    def unpack_initializer(self, initializer):
        raw_data = initializer.raw_data
        data_len = len(raw_data)
        data_type = initializer.data_type
        assert data_type in [1, 7], 'only support float32 and int64'
        dims = initializer.dims
        shape = [x for x in dims]
        if data_type == 1:
            format = "%d%s" % (np.product(shape), 'f')
        else:
            format = "%d%s" % (np.product(shape), 'l')
        data = np.array(struct.unpack(format, raw_data))
        data = data.reshape(shape)
        return data

    def transpose_tesnor_of_bytes(self, data, shape, data_type):
        assert data_type in data_bytes_dict, 'data_type not valid'
        assert len(shape) == 2 and len(data) == shape[0] * shape[1] * data_bytes_dict[
            data_type], 'input shape not match input data'

        format = "%d%s" % (shape[0] * shape[1], data_format_dict[data_type])
        data_unpack = np.array(struct.unpack(format, data))
        data_unpack = data_unpack.reshape(shape)
        data_unpack_trans = list(np.transpose(data_unpack, (1, 0)).flatten())
        return struct.pack(format, *data_unpack_trans)

    def get_initializer_by_name(self, name):
        target = -1
        for (i, init) in enumerate(self.model.graph.initializer):
            if name == init.name:
                target = i
        assert target != -1, "Error, find initializer by name fail"
        return self.model.graph.initializer[target]
    
    def get_initializer_shape_by_name(self, name):
        target = -1
        for (i, init) in enumerate(self.model.graph.initializer):
            if name == init.name:
                target = i
        assert target != -1, "Error, find initializer by name fail"
        return self.model.graph.initializer[target].dims

    def get_source_nodes(self, node):
        res = list()
        input_names = node.input
        for (i, layer) in enumerate(self.nodes):
            output_names = layer.output
            for output in output_names:
                if output in input_names:
                    res.append(i)
                    break
        return res

    def get_sink_nodes(self, node):
        res = list()
        output_names = node.output
        for (i, layer) in enumerate(self.nodes):
            input_names = layer.input
            for input in input_names:
                if input in output_names:
                    res.append(i)
        return res

    def is_initializer(self, name):
        if name in self.initializer_dict.keys():
            return True
        else:
            return False

    def get_layer_ftm_inputs(self, node):
        input_list = []
        for input in node.input:
            if not self.is_initializer(input):
                input_list.append(input)
        return input_list

    def get_layer_initializer_inputs(self, node):
        input_list = []
        for input in node.input:
            if self.is_initializer(input):
                input_list.append(input)
        return input_list

    def is_input_layer(self, node):
        input_names = node.input

        res = False
        for input in self.inputs:
            if self.is_initializer(input.name):
                continue
            if input.name in input_names:
                res = True
        return res

    def is_output_layer(self, node):
        output_names = node.output
        res = False
        for output in self.outputs:
            if output.name in output_names:
                res = True
        return res

    def input_to_node_idx(self, input):
        if self.is_initializer(input):
            return []
        layers = list()
        for (i,layer) in enumerate(self.nodes):
            if input in layer.input:
                layers.append(i)
        return layers
    
    def output_to_node_idx(self, output):
        layers = list()
        for (i,layer) in enumerate(self.nodes):
            if output in layer.output:
                layers.append(i)
        return layers

    def make_delete_ops_index_list(self):
        # find nodes to be modified either by types or by names
        delete_ops_index_list = list()
        nodes = self.nodes
        all_op_num = len(nodes)

        if self.find_ops_by_type:
            sub_op_num = len(self.delete_node_types) # type: ignore
            for i in range(all_op_num):
                if i + sub_op_num > all_op_num:
                    break
                if delete_ops_index_list and i < delete_ops_index_list[-1][0] + sub_op_num:
                    continue
                success = True
                op_list = []
                for j in range(sub_op_num):
                    if nodes[i + j].op_type != self.delete_node_types[j]: # type: ignore
                        success = False
                        break
                    op_list.append(i+j)
                if success:
                    delete_ops_index_list.append(op_list)
        else:
            for (i, node) in enumerate(self.nodes):
                op_list = []
                if node.name in self.delete_node_names:
                    op_list.append(i)
                    delete_ops_index_list.append(op_list)

        return delete_ops_index_list

    def filter_truly_deleted_nodes(self, delet_ops_idx_list):
        return delet_ops_idx_list    

    def forward(self):
        self.node_operate_list

@NODE_OPERATION.register_module(name="ReplaceOps")
class ReplaceOps(BaseNodeOperation):
    def __init__(self, **args):
        super(ReplaceOps, self).__init__(**args)

    def make_add_node_io_list(self, idx_lsit):
        return list(), list()

    def make_add_node_attributes_list(self, idx_list):
        return list()

    def make_add_node_name_list(self, idx_list):
        start_idx = idx_list[0]
        classname = self.__class__.__name__
        node_name_list = list()
        for (i, op_type) in enumerate(self.add_node_types): # type: ignore
            name = "%s_%s_%d_%d" % (classname, op_type, start_idx, i)
            node_name_list.append(name)
        return node_name_list

    def forward(self):
        delete_ops_idx_list = self.make_delete_ops_index_list()
        delete_ops_idx_list = self.filter_truly_deleted_nodes(delete_ops_idx_list)

        for ops_idx in delete_ops_idx_list:
            input_list, output_list = self.make_add_node_io_list(ops_idx)
            attr_list = self.make_add_node_attributes_list(ops_idx)
            node_name_list = self.make_add_node_name_list(ops_idx)
            add_node_list = list()
            for (i, layer_type) in enumerate(self.add_node_types): # type: ignore
                attr = attr_list[i] if attr_list else dict()
                new_node = onnx.helper.make_node( # type: ignore
                    name=node_name_list[i],
                    op_type=layer_type,
                    inputs=input_list[i],
                    outputs=output_list[i],
                    **attr
                )
                add_node_list.append(new_node)
            self.node_operate_list.append(dict(idx=ops_idx[0], add_node_types=add_node_list, delete_idx_list=ops_idx))
        return self.node_operate_list   

@NODE_OPERATION.register_module(name="DeleteOps")
class DeleteOps(BaseNodeOperation):
    def __init__(self, **args):
        super(DeleteOps, self).__init__(**args)

    def modify_input_shape(self, input_name, shape):
        target_idx = -1
        for (idx, input) in enumerate(self.model.graph.input):
            if input.name == input_name:
                target_idx = idx
                break
        if target_idx != -1:
            self.model.graph.input.pop(target_idx)
            self.model.graph.input.insert(target_idx, onnx.helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)) # type: ignore

    def modify_output_shape(self, output_name, shape):
        target_idx = -1
        for (idx, output) in enumerate(self.model.graph.output):
            if output.name == output_name:
                target_idx = idx
                break
        if target_idx != -1:
            self.model.graph.output.pop(target_idx)
            self.model.graph.output.insert(target_idx, onnx.helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)) # type: ignore
    def delete_input(self, input_name):
        for (i, input) in enumerate(self.model.graph.input):
            if input.name == input_name:
                self.model.graph.input.pop(i)
                break
    
    def delete_output(self, output_name):
        for (i, output) in enumerate(self.model.graph.output):
            if output.name == output_name:
                self.model.graph.output.pop(i)
                break        

    def get_input_idx(self, name):
        for (i, input) in enumerate(self.model.graph.input):
            if input.name == name:
                return i
    
    def get_output_idx(self, name):
        for (i, output) in enumerate(self.model.graph.output):
            if output.name == name:
                return i

    def connect_nodes(self, op_idx):
        # should consider if the deleted op is input or output layer
        delete_node = self.nodes[op_idx]
        is_input_node = self.is_input_layer(delete_node)
        is_output_node = self.is_output_layer(delete_node)
        input_names = self.get_layer_ftm_inputs(delete_node)
        output_names = delete_node.output
        if is_input_node:
            if len(input_names)==1 and len(output_names)==1:
                n = len(self.input_to_node_idx(input_names[0]))
                if n > 1 and list(self.featuremap_shape_dict[input_names[0]]) != list(self.featuremap_shape_dict[output_names[0]]):
                    self.logger.error("%s cannot be removed due to the input tensor is shared by other layers"%delete_node.name)
                    return False
                else:
                    sink_node_idx_list = self.get_sink_nodes(delete_node)
                    node_idx = sink_node_idx_list[0]
                    i = np.argwhere(np.array(self.nodes[node_idx].input)==output_names[0])[0][0]
                    self.nodes[node_idx].input.pop(i)
                    self.nodes[node_idx].input.insert(i, input_names[0])
                    input_shape_new = self.featuremap_shape_dict[output_names[0]]
                    self.modify_input_shape(input_names[0], input_shape_new)
            else:
                self.logger.error("%s cannot be removed due to it is not a single inpit/output layer"%delete_node.name)
                return False
        elif is_output_node:
            if len(input_names)==1 and len(output_names)==1:
                source_node_idx_list =self.get_source_nodes(delete_node)
                node_idx = source_node_idx_list[0]
                i = np.argwhere(np.array(self.nodes[node_idx].output) == input_names[0])[0][0]
                self.nodes[node_idx].output.pop(i)
                self.nodes[node_idx].output.insert(i, output_names[0])
                output_shape_new = self.featuremap_shape_dict[input_names[0]]
                self.modify_output_shape(output_names[0], output_shape_new)
            else:
                i = self.get_output_idx(output_names[0])
                for output in output_names:
                    self.delete_output(output)
                
                for input in input_names[::-1]:
                    shape = self.featuremap_shape_dict[input]
                    self.model.graph.output.insert(i, onnx.helper.make_tensor_value_info(input, TensorProto.FLOAT, shape)) # type: ignore
        else:
            if len(input_names)==1 and len(output_names)==1:
                sink_node_idx_list = self.get_sink_nodes(delete_node)
                node_idx = sink_node_idx_list[0]
                i = np.argwhere(np.array(self.nodes[node_idx].input)==output_names[0])[0][0]
                self.nodes[node_idx].input.pop(i)
                self.nodes[node_idx].input.insert(i, input_names[0]) 
            else:
                self.logger.error("%s cannot be removed due to it is not a single input/output layer"%delete_node.name)
                return False               
        return True    

    def forward(self):
        delete_ops_idx_list = self.make_delete_ops_index_list()
        delete_ops_idx_list = self.filter_truly_deleted_nodes(delete_ops_idx_list)

        for ops_idx in delete_ops_idx_list:
            success = self.connect_nodes(ops_idx[0])
            if success:
                self.node_operate_list.append(dict(idx=ops_idx[0], add_node_types=[], delete_idx_list=ops_idx))
        return self.node_operate_list

@NODE_OPERATION.register_module(name="MergeHardswish")
class MergeHardswish(ReplaceOps):
    def __init__(self, **args):
        super(MergeHardswish, self).__init__(**args)

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        input_tensor_name = self.nodes[start_idx].input[0]
        output_tensor_name = self.nodes[start_idx + self.delete_node_num - 1].output[0]
        input_list = [[input_tensor_name]]
        output_list = [[output_tensor_name]]
        return input_list, output_list

@NODE_OPERATION.register_module(name="MergeHardsigmoid")
class MergeHardsigmoid(ReplaceOps):
    def __init__(self, **args):
        super(MergeHardsigmoid, self).__init__(**args)

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        input_tensor_name = self.nodes[start_idx].input[0]
        output_tensor_name = self.nodes[start_idx + self.delete_node_num - 1].output[0]
        input_list = [[input_tensor_name]]
        output_list = [[output_tensor_name]]
        return input_list, output_list

    def make_add_node_attributes_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        mul_node = self.nodes[start_idx + self.delete_node_num - 1]
        b = self.get_initializer_by_name(mul_node.input[1])
        alpha = self.unpack_initializer(b)
        alpha = 1 / alpha
        return [dict(alpha=alpha)]


@NODE_OPERATION.register_module(name="FuseConvBn")
class FuseConvBn(ReplaceOps):
    def __init__(self, **args):
        super(FuseConvBn, self).__init__(**args)

    def make_add_node_io_list(self, idx_lsit):
        idx0, idx1 = idx_lsit
        input_tensor_name = self.nodes[idx0].input[0]
        output_tensor_name = self.nodes[idx0 + self.delete_node_num - 1].output[0]
        
        conv_node = self.nodes[idx0]
        is_convtranspose = True if conv_node.op_type == "ConvTranspose" else False
        weight_name = conv_node.input[1]
        _weight = self.unpack_initializer(self.get_initializer_by_name(weight_name))
        if len(conv_node.input) == 2:
            if is_convtranspose:
                out_c = _weight.shape[1]
            else:
                out_c = _weight.shape[0]
            _bias = np.zeros(out_c, dtype="float32")
            bias_name = weight_name.replace(".weight", ".bias")
        else:
            bias_name = conv_node.input[2]
            _bias = self.unpack_initializer(self.get_initializer_by_name(bias_name))
        bn_node = self.nodes[idx1]
        delete_initializer_names = bn_node.input[1:5]
        gamma = self.unpack_initializer(self.get_initializer_by_name(bn_node.input[1]))
        beta = self.unpack_initializer(self.get_initializer_by_name(bn_node.input[2]))
        mean = self.unpack_initializer(self.get_initializer_by_name(bn_node.input[3]))
        var = self.unpack_initializer(self.get_initializer_by_name(bn_node.input[4]))
        for attribute_ in bn_node.attribute:
            if attribute_.name == "epsilon":
                epsilon = bn_node.attribute[0].f  
            else:
                epsilon = 1.0e-3 
        fused_gamma = gamma / np.sqrt(var + epsilon)
        if is_convtranspose:
            new_weight = _weight * fused_gamma.reshape(1, -1, 1, 1)
            out_c = _weight.shape[1]
        else:
            new_weight = _weight * fused_gamma.reshape(-1, 1, 1, 1)
            out_c = _weight.shape[0]
        new_bias = (_bias - mean) * fused_gamma + beta     
        new_bias = new_bias.astype(np.float32)
        new_weight = new_weight.astype(np.float32)
        
        # make bias initializer
        bias_size = np.prod(new_bias.shape)
        args_bias = {
            'name': bias_name,
            "data_type": "float32",
            "shape": list(new_bias.shape),
            'data': [0] * bias_size,
        }
        bias_tensor = self.create_initializer(**args_bias)
        self.model.graph.initializer.append(bias_tensor)
        bias = self.get_initializer_by_name(bias_name)
        bias.raw_data = new_bias.tobytes()
                    
        # make weight initializer
        weight_size = np.prod(new_weight.shape)
        args_weight = {
            'name': weight_name,
            "data_type": "float32",
            "shape": list(new_weight.shape),
            'data': [0] * weight_size,
        }
        weight_tensor = self.create_initializer(**args_weight)
        self.model.graph.initializer.append(weight_tensor)
        weight = self.get_initializer_by_name(weight_name)
        weight.raw_data = new_weight.tobytes()
                
        ### delete bn_weight, bn_bias, running_mean, running_var from initializer
        for i, delete_initializer_name in enumerate(delete_initializer_names):
            for idx, weight in enumerate(self.model.graph.initializer):
                if weight.name == delete_initializer_name:
                    self.model.graph.initializer.pop(idx)
                    break
                                    
        input_list = [[input_tensor_name, weight_name, bias_name]]
        output_list = [[output_tensor_name]]
        return input_list, output_list

    def make_add_node_attributes_list(self, idx_lsit):
        attributes = dict()
        
        idx0 = idx_lsit[0]
        conv_node = self.nodes[idx0]
        for item in conv_node.attribute:
            attribute_data_type = self.attribute_data_type_dict[str(item.type)]
            if attribute_data_type == "INTS":
                attributes.update(
                    {item.name: item.ints}
                ) 
            elif attribute_data_type == "INT":
                attributes.update(
                    {item.name: item.i}
                )
                    
        return [attributes]
    
    
@NODE_OPERATION.register_module(name='ReplaceReshapeLikeOps')
class ReplaceReshapeLikeOps(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceReshapeLikeOps, self).__init__(**args)
        
    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        delete_ops_idx_list_filter = list()
        if not self.delete_node_types[0] in ['Transpose', 'Flatten', 'Pad', 'Squeeze', 'ReduceMean', 'Unsqueeze']: # type: ignore
            return delete_ops_idx_list_filter
        
        for idx_list in delete_ops_idx_list:
            idx = idx_list[0]
            node = self.model.graph.node[idx]
            in_shape = self.featuremap_shape_dict[node.input[0]]
            out_shape = self.featuremap_shape_dict[node.output[0]]
            if node.op_type=="Transpose":
                perm = list(node.attribute[0].ints)
                out_shape_reduce, perm_reduce = list(), list()
                for i in range(len(out_shape)):
                    if out_shape[i] != 1:
                        out_shape_reduce.append(out_shape[i])
                        perm_reduce.append(perm[i])
                if len(out_shape_reduce) == 1:
                    delete_ops_idx_list_filter.append(idx_list)
                else:
                    perm_reduce_copy = perm_reduce.copy()
                    perm_reduce.sort()
                    if perm_reduce_copy == perm_reduce:
                        delete_ops_idx_list_filter.append(idx_list)
            else:
                if np.product(in_shape) == np.product(out_shape):
                    delete_ops_idx_list_filter.append(idx_list)
        return delete_ops_idx_list_filter

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        input_tensor_name = self.nodes[start_idx].input[0]
        output_tensor_name = self.nodes[start_idx + self.delete_node_num - 1].output[0]
        output_shape = self.featuremap_shape_dict[output_tensor_name]
        # create shape initializer
        self.delete_op_type = self.delete_node_types[0] # type: ignore
        initializer_name = 'shape_replace_%s_%d' % (self.delete_op_type, start_idx)
        args = dict(name=initializer_name,
                    data_type="int64",
                    shape=[len(output_shape)],
                    data=output_shape
                    )
        shape_initializer = self.create_initializer(**args)
        self.model.graph.initializer.append(shape_initializer)
        input_list = [[input_tensor_name, initializer_name]]
        output_list = [[output_tensor_name]]
        return input_list, output_list

@NODE_OPERATION.register_module(name='ReplaceConv1dGemm')
class ReplaceConv1dGemm(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceConv1dGemm, self).__init__(**args)

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_delete_ops_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[0]
            node = self.nodes[idx]
            inputs = node.input
            for name in inputs:
                if self.is_initializer(name):
                    weight_name = name
                else:
                    input_tensor_name = name
            input_shape = self.featuremap_shape_dict[input_tensor_name] # type: ignore
            weight_shape = self.get_initializer_by_name(weight_name).dims # type: ignore
            if len(input_shape) == 3 and input_shape[-1] == 1 and weight_shape[
                2] == 1:  # input (NCH) H==1, and ksize==1
                filter_delete_ops_idx_list.append(idx_list)
        return filter_delete_ops_idx_list

    def make_add_node_attributes_list(self, idx_lsit):
        return [dict(), dict(transB=1), dict()]

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        input_names = self.nodes[start_idx].input
        has_bias = len(input_names) == 3
        for name in input_names:
            if self.is_initializer(name):
                dims = self.initializer_dict[name].dims
                if len(dims)>1:
                    weight_name = name
                else:
                    bias_name = name                   
            else:
                input_ftm_name = name
        
        weight = self.get_initializer_by_name(weight_name) # type: ignore
        output_name = self.nodes[start_idx].output[0]
        weight.dims.pop(-1)

        input_shape = self.featuremap_shape_dict[input_ftm_name] # type: ignore
        output_shape = self.featuremap_shape_dict[output_name]
        output_size = np.product(output_shape)

        if not has_bias:  # make bias initializer
            args_bias = dict()
            bias_name = "gemm_ext_bias_%d" % start_idx
            args_bias.update(
                {'name': bias_name,
                 "data_type": "float32",
                 "shape": [output_size],
                 'data': [0] * output_size
                 },
            )
            bias_tensor = self.create_initializer(**args_bias)
            self.model.graph.initializer.append(bias_tensor)
            
        args1 = dict()
        args1.update(
            {'name': "reshape_ext_%d_upper" % start_idx,
             "data_type": "int64",
             "shape": [2],
             'data': [input_shape[0], input_shape[1]]
             },
        )
        shape_initializer_1 = self.create_initializer(**args1)
        self.model.graph.initializer.append(shape_initializer_1)
        reshape_upper_output_name = "reshape_ext_%d_upper_output" % start_idx

        args2 = dict()
        args2.update(
            {'name': "reshape_ext_%d_lower" % start_idx,
             "data_type": "int64",
             "shape": [3],
             'data': output_shape
             },
        )
        shape_initializer_2 = self.create_initializer(**args2)
        self.model.graph.initializer.append(shape_initializer_2)
        reshape_lower_input_name = "reshape_ext_%d_lower_input" % start_idx

        input_list = [[input_ftm_name, shape_initializer_1.name], [reshape_upper_output_name, weight_name, bias_name], # type: ignore
                      [reshape_lower_input_name, shape_initializer_2.name]]
        output_list = [[reshape_upper_output_name], [reshape_lower_input_name], [output_name]]
        return input_list, output_list

@NODE_OPERATION.register_module(name='ReplaceDepthwiseConv2d')
class ReplaceDepthwiseConv2dWithMulAdd(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceDepthwiseConv2dWithMulAdd, self).__init__(**args)

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_delete_ops_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[0]
            node = self.nodes[idx]
            
            pads, strides = None, None
            for item in node.attribute:
                if item.name == "pads":
                    pads = item.ints    
                if item.name == "strides":
                    strides = item.ints
                         
            weight_name = None       
            inputs = node.input
            for name in inputs:
                if self.is_initializer(name):
                    dims = self.initializer_dict[name].dims
                    if len(dims) > 1:                    
                        weight_name = name
                
            if weight_name and pads and strides:
                weight_shape = self.get_initializer_by_name(weight_name).dims # type: ignore
                if weight_shape[1] == 1 and weight_shape[2] == 1 and weight_shape[3] == 1 \
                    and np.array(pads).sum() == 0 and strides == [1, 1]: ### depthwiseconv
                    filter_delete_ops_idx_list.append(idx_list)
                    
        return filter_delete_ops_idx_list

    def make_add_node_attributes_list(self, idx_lsit):
        return [dict(), dict()]

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        input_names = self.nodes[start_idx].input
        bias_name = None
        for name in input_names:
            if self.is_initializer(name):
                dims = self.initializer_dict[name].dims
                if len(dims) > 1:
                    weight_name = name
                else:
                    bias_name = name                   
            else:
                input_ftm_name = name
        
        weight = self.get_initializer_by_name(weight_name) # type: ignore
        if bias_name:
            bias = self.get_initializer_by_name(bias_name) # type: ignore
        output_name = self.nodes[start_idx].output[0]

        input_shape = self.featuremap_shape_dict[input_ftm_name] # type: ignore
        # output_shape = self.featuremap_shape_dict[output_name]

        weight = np.frombuffer(weight.raw_data, dtype=np.float32).reshape(weight.dims)
        bias = np.frombuffer(bias.raw_data, dtype=np.float32).reshape(bias.dims)
        weight = np.squeeze(weight)
        weight = np.broadcast_to(np.expand_dims(np.expand_dims(np.expand_dims(weight, axis=0), axis=-1), axis=-1), input_shape)
        input_shape[-2:] = 1, 1
        bias = np.squeeze(bias)
        bias = np.broadcast_to(np.expand_dims(np.expand_dims(np.expand_dims(bias, axis=0), axis=-1), axis=-1), input_shape)
        weight_size = np.product(weight.shape)
        bias_size = np.product(bias.shape)
        
        args1 = dict()
        args1.update(
            {'name': "mul_ext_%d_upper" % start_idx,
             "data_type": "float32",
             "shape": list(weight.shape),
             "data": [0] * weight_size,
             },
        )
        shape_initializer_1 = self.create_initializer(**args1)
        self.model.graph.initializer.append(shape_initializer_1)
        reshape_upper_output_name = "mul_ext_%d_upper_output" % start_idx

        args2 = dict()
        args2.update(
            {'name': "add_ext_%d_lower" % start_idx,
             "data_type": "float32",
             "shape": list(bias.shape),
             "data": [0] * bias_size,
             },
        )
        shape_initializer_2 = self.create_initializer(**args2)
        self.model.graph.initializer.append(shape_initializer_2)

        input_list = [[input_ftm_name, shape_initializer_1.name],  # type: ignore
                      [reshape_upper_output_name, shape_initializer_2.name]]
        output_list = [[reshape_upper_output_name], [output_name]]
        return input_list, output_list
        
@NODE_OPERATION.register_module(name='ReplaceConv1dWithConv2d')
class ReplaceConv1dWithConv2d(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceConv1dWithConv2d, self).__init__(**args)
        self.attrs = dict()

    def get_node_attrs(self, node, is_insert=False):
        attrs = dict()
        kernel_shape = None
        for item in node.attribute:
            if item.name == "kernel_shape":
                kernel_shape = item.ints
        if len(kernel_shape) > 1: # type: ignore
            return attrs

        for attr in node.attribute:
            if attr.name in ['dilations', 'strides', 'kernel_shape']:
                if is_insert:
                    attr.ints.insert(1, 1)
                attrs[attr.name] = attr.ints
            elif attr.name in ['group']:
                attrs[attr.name] = attr.i
            elif attr.name in ['pads']:
                if is_insert:
                    attr.ints.insert(2, 0)
                    attr.ints.insert(3, 0)
                attrs[attr.name] = attr.ints
        return attrs

    def get_node_input_and_weight_name(self, node):
        input_ftm_name, weight_name, bias_name = None, None, None
        inputs = node.input
        for name in inputs:
            if self.is_initializer(name):
                dims = self.initializer_dict[name].dims
                if len(dims)>1:
                    weight_name = name
                else:
                    bias_name = name
            else:
                input_ftm_name = name
        return input_ftm_name, (weight_name, bias_name)

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_delete_ops_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[0]
            node = self.nodes[idx]
            input_ftm_name, (weight_name, bias_name) = self.get_node_input_and_weight_name(node)
            input_shape = self.featuremap_shape_dict[input_ftm_name]
            weight_shape = self.get_initializer_by_name(weight_name).dims
            attrs = self.get_node_attrs(node, is_insert=False)
            if attrs == {}:
                continue
            if not (len(input_shape) == 3 and input_shape[-1] == 1 and weight_shape[
                2] == 1 and attrs['strides'][0] == 1):  # input (NCH) H==1, and ksize==1
                filter_delete_ops_idx_list.append(idx_list)
        return filter_delete_ops_idx_list

    def make_add_node_attributes_list(self, idx_lsit):
        return [dict(), self.attrs, dict()]

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        node = self.nodes[start_idx]
        input_names = node.input
        has_bias = len(input_names) == 3
        self.attrs = self.get_node_attrs(node, is_insert=True)
        input_ftm_name, (weight_name, bias_name) = self.get_node_input_and_weight_name(node)
        weight = self.get_initializer_by_name(weight_name)
        output_name = node.output[0]
        weight.dims.append(1)

        input_shape = self.featuremap_shape_dict[input_ftm_name]
        input_shape = list(input_shape) + [1]
        output_shape = self.featuremap_shape_dict[output_name]
        output_size = np.product(output_shape)

        if not has_bias:  # make bias initializer
            args_bias = dict()
            bias_name = "conv2d_ext_bias_%d" % start_idx
            args_bias.update(
                {'name': bias_name,
                 "data_type": "float32",
                 "shape": [output_size],
                 'data': [0] * output_size
                 },
            )
            bias_tensor = self.create_initializer(**args_bias)
            self.model.graph.initializer.append(bias_tensor)

        args1 = dict()
        args1.update(
            {'name': "reshape_ext_%d_upper" % start_idx,
             "data_type": "int64",
             "shape": [len(input_shape)],
             'data': input_shape,
             },
        )
        shape_initializer_1 = self.create_initializer(**args1)
        self.model.graph.initializer.append(shape_initializer_1)
        reshape_upper_output_name = "reshape_ext_%d_upper_output" % start_idx

        args2 = dict()
        args2.update(
            {'name': "reshape_ext_%d_lower" % start_idx,
             "data_type": "int64",
             "shape": [len(output_shape)],
             'data': output_shape,
             },
        )
        shape_initializer_2 = self.create_initializer(**args2)
        self.model.graph.initializer.append(shape_initializer_2)
        reshape_lower_input_name = "reshape_ext_%d_lower_input" % start_idx

        input_list = [[input_ftm_name, shape_initializer_1.name], [reshape_upper_output_name, weight_name, bias_name],
                      [reshape_lower_input_name, shape_initializer_2.name]]
        output_list = [[reshape_upper_output_name], [reshape_lower_input_name], [output_name]]
        return input_list, output_list

@NODE_OPERATION.register_module(name='ReplaceGemmReshapeRelu')
class ReplaceGemmReshapeRelu(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceGemmReshapeRelu, self).__init__(**args)

    def make_add_node_attributes_list(self, idx_lsit):
        return [dict(transB=1), dict(), dict()]

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_delete_ops_idx_list = list()
        for idx_list in delete_ops_idx_list:
            filter_delete_ops_idx_list.append(idx_list)
        return filter_delete_ops_idx_list

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        gemm_inputs = self.nodes[start_idx].input
        weight_bias_names = []
        for x in gemm_inputs:
            if self.is_initializer(x):
                weight_bias_names.append(x)
            else:
                gemm_input_name = x
                input_shape = self.featuremap_shape_dict[x]

        gemm_output_name = self.nodes[start_idx].output[0]

        end_idx = idx_lsit[1]
        relu_output_name = self.nodes[end_idx].output[0]
        # relu_output_shape = self.featuremap_shape_dict[relu_output_name]
        for i, relu_output in enumerate(self.nodes[end_idx].output):
            for j, value_info in enumerate(self.model.graph.value_info):
                if value_info.name == relu_output:
                    relu_output_dims = self.model.graph.value_info[j].type.tensor_type.shape.dim
                    if len(relu_output_dims) > 2:
                        dims = [d.dim_value for d in relu_output_dims]
                        relu_output_dims[0].dim_value = dims[0]
                        relu_output_dims[1].dim_value = np.product(dims[1:])
                        while len(relu_output_dims) > 2:
                            relu_output_dims.pop(-1)
                    # print("test")
                    break

        end_idx = idx_lsit[2]
        reshape_output_name = self.nodes[end_idx].output[0]
        reshape_output_shape = self.featuremap_shape_dict[reshape_output_name]

        reshape_lower_name = "_".join([reshape_output_name, str(end_idx)])
        args2 = dict()
        args2.update(
            {'name': reshape_lower_name,
             "data_type": "int64",
             "shape": [len(reshape_output_shape)],
             'data': reshape_output_shape,
             },
        )
        shape_initializer_2 = self.create_initializer(**args2)
        self.model.graph.initializer.append(shape_initializer_2)

        weight_name, bias_name = weight_bias_names[0], weight_bias_names[1]
        input_list = [
                    [gemm_input_name, weight_name, bias_name], # type: ignore
                    [gemm_output_name],
                    [relu_output_name, shape_initializer_2.name],
                    ]
        output_list = [
                [gemm_output_name],
                [relu_output_name],
                [reshape_output_name],
                ]
        return input_list, output_list


@NODE_OPERATION.register_module(name='ReplaceMatmulReshapeAdd')
class ReplaceMatmulReshapeAdd(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceMatmulReshapeAdd, self).__init__(**args)
        self.initializer_names = [initializer.name for initializer in self.model.graph.initializer]

    def make_add_node_attributes_list(self, idx_lsit):
        return [dict(transB=1), dict()]

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_delete_ops_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[-1]
            inputs = self.nodes[idx].input
            is_skip = True
            for input in inputs:
                if input in self.initializer_names:
                    is_skip = False
                    break
            if is_skip:
                continue
            # input_shape = self.featuremap_shape_dict[inputs[0]]
            # if input_shape[1] == 1:
            filter_delete_ops_idx_list.append(idx_list)
        return filter_delete_ops_idx_list

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        matmul_inputs = self.nodes[start_idx].input
        for x in matmul_inputs:
            if self.is_initializer(x):
                weight_name = x
            else:
                matmul_input_name = x
                input_shape = self.featuremap_shape_dict[x]

        matmul_output_name = self.nodes[start_idx].output[0]

        end_idx = idx_lsit[2]
        reshape_output_name = self.nodes[end_idx].output[0]
        output_shape = self.featuremap_shape_dict[reshape_output_name]
        for x in self.nodes[end_idx].input:
            if self.is_initializer(x):
                bias_name = x

        weight = self.get_initializer_by_name(weight_name) # type: ignore
        weight_shape = weight.dims
        raw_data = weight.raw_data
        # print(show_unpack_data(raw_data,weight_shape,'float32' ))
        raw_data_trans = self.transpose_tesnor_of_bytes(raw_data, weight_shape, 'float32')
        # print(show_unpack_data(raw_data_trans,weight_shape,'float32' ))

        weight.raw_data = raw_data_trans
        weight.dims.reverse()

        reshape_lower_name = "_".join([reshape_output_name, str(end_idx)])
        args2 = dict()
        args2.update(
            {'name': reshape_lower_name,
             "data_type": "int64",
             "shape": [3],
             'data': output_shape
             },
        )
        shape_initializer_2 = self.create_initializer(**args2)
        self.model.graph.initializer.append(shape_initializer_2)

        input_list = [
                    [matmul_input_name, weight_name, bias_name], # type: ignore
                    [matmul_output_name, shape_initializer_2.name],
                    ]
        output_list = [
                [matmul_output_name], 
                [reshape_output_name],
                ]
        return input_list, output_list
    
@NODE_OPERATION.register_module(name='ReplaceMatmulAdd')
class ReplaceMatmulAdd(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceMatmulAdd, self).__init__(**args)
        self.initializer_names = [initializer.name for initializer in self.model.graph.initializer]

    def make_add_node_attributes_list(self, idx_lsit):
        return [dict(), dict(transB=1), dict()]

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_delete_ops_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[-1]
            inputs = self.nodes[idx].input
            is_skip = True
            for input in inputs:
                if input in self.initializer_names:
                    is_skip = False
                    break
            if is_skip:
                continue
            # input_shape = self.featuremap_shape_dict[inputs[0]]
            # if input_shape[1] == 1:
            filter_delete_ops_idx_list.append(idx_list)
        return filter_delete_ops_idx_list

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        matmul_inputs = self.nodes[start_idx].input
        for x in matmul_inputs:
            if self.is_initializer(x):
                weight_name = x
            else:
                matmul_input_name = x
                input_shape = self.featuremap_shape_dict[x]

        add_inputs = self.nodes[start_idx + 1].input 
        for x in add_inputs:
            if self.is_initializer(x):
                bias_name = x
        outputs = self.nodes[start_idx + self.delete_node_num - 1].output
        output_shape = self.featuremap_shape_dict[outputs[0]]
        weight = self.get_initializer_by_name(weight_name) # type: ignore
        weight_shape = weight.dims
        raw_data = weight.raw_data
        # print(show_unpack_data(raw_data,weight_shape,'float32' ))
        raw_data_trans = self.transpose_tesnor_of_bytes(raw_data, weight_shape, 'float32')
        # print(show_unpack_data(raw_data_trans,weight_shape,'float32' ))

        weight.raw_data = raw_data_trans
        weight.dims.reverse()

        args1 = dict()
        shape_upper_name = "shape_rep_matmul_%d_upper" % start_idx

        fisrt_dim = reduce(lambda x, y: x * y, input_shape[:-1]) # type: ignore

        args1.update(
            {'name': shape_upper_name,
             "data_type": "int64",
             "shape": [2],
             'data': [fisrt_dim, input_shape[-1]]  # squeeze channel # type: ignore
             },
        )
        shape_initializer_1 = self.create_initializer(**args1)
        self.model.graph.initializer.append(shape_initializer_1)

        reshape_lower_name = "shape_rep_matmul_%d_lower" % start_idx
        args2 = dict()
        args2.update(
            {'name': reshape_lower_name,
             "data_type": "int64",
             "shape": [3],
             'data': output_shape
             },
        )
        shape_initializer_2 = self.create_initializer(**args2)
        self.model.graph.initializer.append(shape_initializer_2)

        reshape_upper_output_name = "shape_rep_matmul_%d_upper_output" % start_idx
        reshape_lower_input_name = "shape_rep_matmul_%d_lower_input" % start_idx

        input_list = [[matmul_input_name, shape_initializer_1.name], [reshape_upper_output_name, weight_name, bias_name], # type: ignore
                      [reshape_lower_input_name, shape_initializer_2.name]]
        output_list = [[reshape_upper_output_name], [reshape_lower_input_name], [outputs[0]]]
        return input_list, output_list

@NODE_OPERATION.register_module(name='ReplaceMatmul')
class ReplaceMatmul(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceMatmul, self).__init__(**args)

    def make_add_node_attributes_list(self, idx_lsit):
        return [dict(transB=1, alpha=1.0, beta=1.0)]

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_delete_ops_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[0]
            weight_name = None
            for x in self.nodes[idx].input:
                if self.is_initializer(x):
                    weight_name = x
            if weight_name:
                weight = self.get_initializer_by_name(weight_name)
                if len(weight.dims) == 2:
                    filter_delete_ops_idx_list.append(idx_list)
        return filter_delete_ops_idx_list

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        outputs = self.nodes[start_idx + self.delete_node_num - 1].output

        weight_name, matmul_input_name = None, None
        for x in self.nodes[start_idx].input:
            if self.is_initializer(x):
                weight_name = x
            else:
                matmul_input_name = x

        input_list, output_list = [], []
        if weight_name:
            weight = self.get_initializer_by_name(weight_name)
            weight_shape = weight.dims
            raw_data = weight.raw_data
            raw_data_trans = self.transpose_tesnor_of_bytes(raw_data, weight_shape, 'float32')

            weight.raw_data = raw_data_trans
            weight.dims.reverse()

            bias_name = weight_name + "_bias"
            bias = np.zeros(weight.dims[0], dtype=np.float32)
            ### push bias_name into initializer
            args_bias = dict()
            args_bias.update(
                {'name': bias_name,
                "data_type": "float32",
                "shape": [weight.dims[0]],
                'data': bias,
                },
            )
            shape_initializer_bias = self.create_initializer(**args_bias)
            self.model.graph.initializer.append(shape_initializer_bias)

            input_list = [[matmul_input_name, weight_name, bias_name],]
            output_list = [[outputs[0]]]
        return input_list, output_list

@NODE_OPERATION.register_module(name='MergeLayerNormalization')
class MergeLayerNormalization(ReplaceOps):
    def __init__(self, **args):
        super(MergeLayerNormalization, self).__init__(**args)
        
    def make_delete_ops_index_list(self):
        delete_ops_index_list = super(MergeLayerNormalization, self).make_delete_ops_index_list()
        if len(delete_ops_index_list) > 0:
            self.model.opset_import[0].version = 15
        return delete_ops_index_list

    def get_node_attribute_dict(self, node_idx):
        attributes = self.nodes[node_idx].attribute
        attr_dict = dict()
        for attr in attributes:
            attr_dict.update({attr.name: attr})
        return attr_dict

    def make_add_node_attributes_list(self, idx_list):
        start_idx = idx_list[0]
        input_names = self.nodes[start_idx].input
        input_shape = self.featuremap_shape_dict[input_names[0]]
        # set normalize axis
        attr = self.get_node_attribute_dict(start_idx)
        if 'axes' in attr.keys():
            axes = attr['axes'].ints
        else:
            axes = None

        axis = 0 if axes is None else axes[0]
        attr = dict(axis=axis, epsilon=1e-5)
        return [attr]

    def make_add_node_io_list(self, idx_lsit):
        start_idx = idx_lsit[0]
        input_names = self.nodes[start_idx].input
        input_shape = self.featuremap_shape_dict[input_names[0]]

        gamma = self.nodes[start_idx + self.delete_node_num - 2].input[1]
        beta = self.nodes[start_idx + self.delete_node_num - 1].input[1]

        input_names.extend([gamma, beta])
        output_names = self.nodes[start_idx + self.delete_node_num - 1].output
        return [input_names], [output_names]

@NODE_OPERATION.register_module(name="ReplaceSliceOps")
class ReplaceSliceOps(ReplaceOps):
    def __init__(self, **args):
        super(ReplaceSliceOps, self).__init__(**args) 

    def make_delete_ops_index_list(self):
        nodes = self.nodes
        all_op_num = len(nodes)
        ops_idx_list = list()
        for i in range(all_op_num):
            if nodes[i].op_type == "Slice":
                ftm_input = self.get_layer_ftm_inputs(nodes[i])[0]
                ops_idx_list.append([i,ftm_input])

        if len(ops_idx_list) > 0:
            self.model.opset_import[0].version = 15
        
        same_input_ops = dict()
        cnt=0
        for x in ops_idx_list:
            if not x[1] in same_input_ops.keys():
                same_input_ops.update({x[1]:{'idx':cnt, "ops_list":[x[0]]}})
                cnt+=1
            else:
                same_input_ops[x[1]]['ops_list'].append(x[0])
        
        delete_ops_index_list=[0]*cnt
        for k in same_input_ops.keys():
            idx = same_input_ops[k]['idx']
            delete_ops_index_list[idx] = same_input_ops[k]['ops_list']
        return delete_ops_index_list
        
    def make_add_node_io_list(self, ops_idx):
        inputs = list()
        outputs = list()
        for input in self.nodes[ops_idx[0]].input:
            if not self.is_initializer(input):
                inputs.append(input)
                break
        split = list()
        starts = list()
        for idx in ops_idx:
            start_initializer = self.get_initializer_by_name(self.nodes[idx].input[1])
            start = self.unpack_initializer(start_initializer)[0]
            end_initializer = self.get_initializer_by_name(self.nodes[idx].input[2])
            end = self.unpack_initializer(end_initializer)[0] 
            axes_initializer = self.get_initializer_by_name(self.nodes[idx].input[3])
            axis = self.unpack_initializer(axes_initializer)[0]
            dim = self.featuremap_shape_dict[self.nodes[idx].input[0]][axis]
            if start < 0: start = dim + start
            if end > dim: end = dim
            if end < 0: end = dim + end
            starts.append(start)
            split.append(end - start)
        for idx in ops_idx:
            outputs.append(self.nodes[idx].output[0])

        split = np.array(split)
        starts = np.array(starts)
        ord = np.argsort(starts) # sort slice ops with their start index
        split=split[ord]        # split array reorder
        # if split in node as input
        args = dict(name="split_replace_slice_%d_%d"%(ops_idx[0], ops_idx[1]),
            data_type="int64",
            shape=[len(ops_idx)],
            data=split
            )
        split_initializer = self.create_initializer(**args)
        self.model.graph.initializer.append(split_initializer)
        inputs.append(split_initializer.name)
        outputs=list(np.array(outputs)[ord])
        return [inputs], [outputs]
    
    def filter_truly_deleted_nodes(self, delet_ops_idx_list):
        filterd_list = list()
        for idx_list in delet_ops_idx_list:
            split = list()
            starts = list()
            axes_initializer = self.get_initializer_by_name(self.nodes[idx_list[0]].input[3])
            axis = self.unpack_initializer(axes_initializer)[0]
            dim = self.featuremap_shape_dict[self.nodes[idx_list[0]].input[0]][axis] 
            for idx in idx_list:
                start_initializer = self.get_initializer_by_name(self.nodes[idx].input[1])
                start = self.unpack_initializer(start_initializer)[0]
                end_initializer = self.get_initializer_by_name(self.nodes[idx].input[2])
                end = self.unpack_initializer(end_initializer)[0] 
                if start < 0: 
                    start = dim + start
                if end > dim: 
                    end = dim
                if end < 0: 
                    end = dim + end
                starts.append(start)
                split.append(end - start)
            if np.sum(split) != dim or len(split) == 1: 
                continue
            if len(self.nodes[idx].input)==5: # has step # type: ignore
                continue
            filterd_list.append(idx_list)
        return filterd_list        

    def make_add_node_attributes_list(self, idx_list):
        # split = list()
        # starts = list()
        axes_initializer = self.get_initializer_by_name(self.nodes[idx_list[0]].input[3])
        axis = self.unpack_initializer(axes_initializer)[0]
        # dim = self.featuremap_shape_dict[self.nodes[idx_list[0]].input[0]][axis]  

        # for idx in idx_list:
        #     start_initializer = self.get_initializer_by_name(self.nodes[idx].input[1])
        #     start = self.unpack_initializer(start_initializer)[0]
        #     end_initializer = self.get_initializer_by_name(self.nodes[idx].input[2])
        #     end = self.unpack_initializer(end_initializer)[0] 
        #     if start < 0: start = dim + start
        #     if end > dim: end = dim
        #     if end < 0: end = dim + end
        #     starts.append(start)
        #     split.append(end - start)
        # split = np.array(split).astype(np.int64)
        # starts = np.array(starts)
        # ord = np.argsort(starts) # sort slice ops with their start index
        # split=split[ord]        # split array reorder
        #return [dict(axis=axis, split=split)]
        return [dict(axis=axis)]

@NODE_OPERATION.register_module(name='MergeReshapeOps')
class MergeReshapeOps(ReplaceOps):
    def __init__(self, **args):
        super(MergeReshapeOps, self).__init__(**args)

    def find_multi_reshapes(self):
        delete_ops_idx_list = list()
        for (i, node) in enumerate(self.nodes):
            if delete_ops_idx_list and i <= delete_ops_idx_list[-1][0] + len(delete_ops_idx_list[-1]):
                continue
            cnt = 0
            while self.nodes[i + cnt].op_type == "Reshape":
                cnt += 1
                if i + cnt >= len(self.nodes):
                    break
            if cnt > 1:
                delete_ops_idx_list.append(list(range(i, i + cnt)))
            i += cnt
        return delete_ops_idx_list

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_list = list()
        for idx_list in delete_ops_idx_list:
            start_idx = idx_list[0]
            size = len(idx_list)
            flag = True
            for i in range(size - 1):  # reshape
                node = self.nodes[start_idx + i]
                output_tensor_name = node.output[0]
                next_node = self.nodes[start_idx + 1 + i]
                input_tensor_name = next_node.input
                if not output_tensor_name in input_tensor_name:
                    flag = False
                    break
            if flag:
                for i in range(size - 1):
                    node = self.nodes[start_idx + i]
                    output_tensor_name = node.output[0]
                    next_node_num = 0
                    for j in range(len(self.nodes)):
                        if output_tensor_name in self.nodes[j].input:
                            next_node_num += 1
                    if next_node_num > 1:
                        flag = False
                        break
            if flag:
                filter_list.append(idx_list)
        return filter_list

    def forward(self):
        delete_ops_idx_list = self.find_multi_reshapes()
        delete_ops_idx_list = self.filter_truly_deleted_nodes(delete_ops_idx_list)
        modify_operations = list()
        for idx_list in delete_ops_idx_list:
            start_idx = idx_list[0]
            size = len(idx_list)

            input_names = self.nodes[start_idx].input
            final_output_names = self.nodes[start_idx + size - 1].output
            final_output_shape = self.featuremap_shape_dict[final_output_names[0]]
            self.nodes[start_idx].output.pop()
            self.nodes[start_idx].output.MergeFrom(final_output_names)
            reshape_initializer = self.get_initializer_by_name(input_names[1])
            reshape_initializer.dims.pop()
            reshape_initializer.dims.MergeFrom([len(final_output_shape)])
            reshape_initializer.raw_data = struct.pack("%dl" % (len(final_output_shape)), *final_output_shape)
            modify_operations.append({"idx": start_idx + 1, "add_node_types": [], "delete_num": size - 1})
        return modify_operations

@NODE_OPERATION.register_module(name='DeleteInputOps')
class DeleteInputOps(DeleteOps):
    def __init__(self, **args):
        super(DeleteInputOps, self).__init__(**args)

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        new_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[0]
            if self.is_input_layer(self.nodes[idx]):
                new_idx_list.append(idx_list)
        return new_idx_list

@NODE_OPERATION.register_module(name='DeleteOutputOps')
class DeleteOutputOps(DeleteOps):
    def __init__(self, **args):
        super(DeleteOutputOps, self).__init__(**args)

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        new_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[0]
            if not self.is_output_layer(self.nodes[idx]):
                continue
            output = self.nodes[idx].output[0]
            flag = True
            for node in self.nodes:
                if output in node.input and not self.is_output_layer(node):
                    flag = False
                    break   
            if flag:
                new_idx_list.append(idx_list)
        return new_idx_list

@NODE_OPERATION.register_module(name='MergeOps2Weight')
class MergeOps2Weight(ReplaceOps):
    def __init__(self, **args):
        super(MergeOps2Weight, self).__init__(**args)

    @staticmethod
    def find_node_id(input_name_or_output_name, input_or_output_list, node_id):
        if input_name_or_output_name in input_or_output_list:
            return node_id
        else:
            return -1

    def find_previous_node(self, node, outputs):
        input_names = node.input
        for input_name in input_names:
            for node_id, output in enumerate(outputs):
                node_id = self.find_node_id(input_name, output, node_id)
                if node_id >= 0:
                    return self.nodes[node_id], node_id

        return node, -1

    def find_next_node(self, node, inputs):
        output_names = node.output
        for output_name in output_names:
            for node_id, input in enumerate(inputs):
                node_id = self.find_node_id(output_name, input, node_id)
                if node_id >= 0:
                    return self.nodes[node_id], node_id

        return node, -1

    def find_pairs(self):
        delete_ops_idx_list = list()
        inputs = [node.input for i, node in enumerate(self.nodes)]
        outputs = [node.output for i, node in enumerate(self.nodes)]
        for (i, node) in enumerate(self.nodes):
            previous_node = copy.deepcopy(node)
            if node.op_type in self.delete_node_types:
                pair_ops_idx_list = [i]
                while True:
                    previous_node, node_id = self.find_previous_node(previous_node, outputs)
                    previous_node_real_input = []
                    for previous_node_input in previous_node.input:
                        if previous_node_input not in self.initializer_dict.keys():
                            previous_node_real_input.append(previous_node_input)

                    pair_ops_idx_list.insert(0, node_id)
                    if len(previous_node_real_input) != 1 or previous_node.op_type.lower() in ["gemm", "conv"]:
                        delete_ops_idx_list.append(pair_ops_idx_list)
                        break

        return delete_ops_idx_list

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        filter_list = list()
        is_filter = False
        for idx_list in delete_ops_idx_list:
            for idx in idx_list[1:-1]:
                if self.nodes[idx_list[-1]].op_type.lower() in ["div", "mul"]:
                    if self.nodes[idx].op_type.lower() not in ['reshape', 'transpose', 'relu']:
                        is_filter = True
                elif self.nodes[idx_list[-1]].op_type.lower() in ["add", 'sub']:
                    if self.nodes[idx].op_type.lower() not in ['reshape', 'transpose']:
                        is_filter = True
                        
            if len(idx_list) == 2:
                node_out = set(self.nodes[idx_list[0]].output)
                node_in = set(self.nodes[idx_list[1]].input)
                non_intersection = node_in.symmetric_difference(node_out)
                non_intersection_list = list(non_intersection)
                if self.nodes[idx_list[0]].op_type.lower() in ["conv"] \
                    and self.nodes[idx_list[1]].op_type.lower() in ["mul"] \
                    and non_intersection_list[0] in self.initializer_dict.keys():
                    is_filter = False
                else:
                    is_filter = True    
                
            if not is_filter:
                filter_list.append([idx_list[0], idx_list[-1]])
            else:
                filter_list.append([idx_list[-1]])
        return filter_list

    def forward(self):
        inputs = [node.input for i, node in enumerate(self.nodes)]
        delete_ops_idx_list = self.find_pairs()
        delete_ops_idx_list = self.filter_truly_deleted_nodes(delete_ops_idx_list)
        modify_operations = list()
        for idx_list in delete_ops_idx_list:
            if len(idx_list) == 2:
                start_idx = idx_list[0]
                end_idx = idx_list[1]

                weight_name, input_name = None, None
                for x in self.nodes[end_idx].input:
                    if self.is_initializer(x):
                        weight_name = x
                    else:
                        input_name = x
                if weight_name and input_name:
                    weight = self.get_initializer_by_name(weight_name)
                    shape = self.get_initializer_shape_by_name(weight_name)
                    data_type = self.data_type_dict[str(weight.data_type)]
                    weight = np.frombuffer(weight.raw_data, dtype=data_type)
                    if weight.shape[0] != 1: # only support div/mul/add/sub weight shape is [1]
                        break
                    for x in self.nodes[start_idx].input:
                        if self.is_initializer(x):
                            gemm_weight = self.get_initializer_by_name(x)
                            gemm_weight_raw_data = gemm_weight.raw_data
                            data_type = self.data_type_dict[str(gemm_weight.data_type)]
                            gemm_weight_fdata = np.frombuffer(gemm_weight_raw_data, dtype=data_type)
                            if self.nodes[end_idx].op_type.lower() in ["div"]:
                                gemm_weight_fdata = gemm_weight_fdata / weight
                            elif self.nodes[end_idx].op_type.lower() in ["mul"]:
                                gemm_weight_fdata = gemm_weight_fdata * weight
                            elif self.nodes[end_idx].op_type.lower() in ["add"]:
                                gemm_weight_fdata = gemm_weight_fdata + weight
                            elif self.nodes[end_idx].op_type.lower() in ["sub"]:
                                gemm_weight_fdata = gemm_weight_fdata - weight
                            for (i, initializer) in enumerate(self.model.graph.initializer):
                                if initializer.name == x:
                                    self.model.graph.initializer[i].raw_data = gemm_weight_fdata.tobytes()
                                    break

                    next_node, _ = self.find_next_node(self.nodes[end_idx], inputs)
                    for i, next_node_input in enumerate(next_node.input):
                        if next_node_input == self.nodes[end_idx].output[0]:
                            next_node.input[i] = input_name
                            
                    out_names = self.nodes[end_idx].output
                    for out_name in out_names:
                        for (i, model_output) in enumerate(self.model.graph.output):
                            if model_output.name == out_name:
                                # print(out_name)
                                self.model.graph.output.remove(model_output)
                                break
                    self.nodes[end_idx].output.pop()

                    modify_operations.append({"idx": end_idx, "add_node_types": [], "delete_num": 1})
            elif len(idx_list) == 1:
                end_idx = idx_list[0]
                if self.nodes[end_idx].op_type.lower() in ["div"]: #repalce div with mul
                    self.nodes[end_idx].op_type = "Mul"
                    for x in self.nodes[end_idx].input:
                        if self.is_initializer(x):
                            gemm_weight = self.get_initializer_by_name(x)
                            gemm_weight_raw_data = gemm_weight.raw_data
                            data_type = self.data_type_dict[str(gemm_weight.data_type)]
                            gemm_weight_fdata = np.frombuffer(gemm_weight_raw_data, dtype=data_type)
                            gemm_weight_fdata = 1.0 / gemm_weight_fdata
                            for (i, initializer) in enumerate(self.model.graph.initializer):
                                if initializer.name == x:
                                    self.model.graph.initializer[i].raw_data = gemm_weight_fdata.tobytes()
                                    break
            else:
                pass
        return modify_operations

@NODE_OPERATION.register_module(name='DeleteRedundentReshapeOps')
class DeleteRedundentReshapeOps(DeleteOps):
    def __init__(self, **args):
        super(DeleteRedundentReshapeOps, self).__init__(**args)

    def filter_truly_deleted_nodes(self, delete_ops_idx_list):
        new_idx_list = list()
        for idx_list in delete_ops_idx_list:
            idx = idx_list[0]
            if self.is_output_layer(self.nodes[idx]):
                new_idx_list.append(idx_list)
            else:
                input = self.get_layer_ftm_inputs(self.nodes[idx])[0]
                output = self.nodes[idx].output[0]
                if self.featuremap_shape_dict[input] == self.featuremap_shape_dict[output]:
                    new_idx_list.append(idx_list)
        return new_idx_list


class OnnxProcess(Object): # type: ignore
    def __init__(self, **kwargs):
        super(OnnxProcess, self).__init__(**kwargs)
        # self.is_simplify = kwargs['is_simplify'] if 'is_simplify' in kwargs.keys() else False
        #self.is_remove_transpose = kwargs['is_remove_transpose'] if 'is_remove_transpose' in kwargs.keys() else False
        #self.is_slice2split = kwargs['is_slice2split'] if 'is_slice2split' in kwargs.keys() else True

        # self.logger.info('_________________________________________________________________________________')
        # self.logger.info('start check and process onnx checkpoint has must be combination structure or not!')

        # self.model = self.del_transpose_slice(kwargs['model'])
        self.log_name = kwargs.get('log_name', 'onnx_preprocess.log')
        self.log_level = kwargs.get('log_level', 20)
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level)         
        self.model = kwargs['model']
        self.model_raw = copy.copy(self.model)
        self.logger.info([x.op_type for x in self.model.graph.node])
        self.ir_version_default = 8
        self.opset_version_default = 15
        self.make_operation_pipline(**kwargs)
        self.save_path = kwargs['save_path'] if 'save_path' in kwargs.keys() else None

    def make_operation_pipline(self, **kwargs):
        self.node_operation_pipeline = online_operation_pipeline.copy()

    # def del_transpose_slice(self, model_path):
    #     if isinstance(model_path, str):
    #         model = onnx.load(model_path)
    #     else:
    #         model = copy.deepcopy(model_path)

    #     if self.is_remove_transpose:
    #         model = remove_transpose(model)
    #     if self.is_slice2split:
    #         model = slice2split(model, opset_version=self.opset_version)

    #     return model

    def check(self):
        self.logger.info("---------------- Check modified model ----------------") # type: ignore
        # sess1 = rt.InferenceSession(self.model.SerializeToString(), None)
        # sess2 = rt.InferenceSession(self.model_raw.SerializeToString(), None)
        def get_outputs(output_names, outs):
            outputs = {}
            for k, v in zip(output_names, outs):
                outputs[k] = v
            return outputs
                
        # compare all featuremaps
        sess1 = self.create_session(copy.deepcopy(self.model)) # type: ignore
        sess2 = self.create_session(copy.deepcopy(self.model_raw)) # type: ignore
        output_names1 = [x.name for x in sess1.get_outputs()]
        output_names2 = [x.name for x in sess2.get_outputs()]

        input_dict1 = self.make_dummy_inputs(self.model)
        input_dict2 = self.make_dummy_inputs(self.model_raw)

        outs1 = sess1.run(output_names1, input_dict1)
        outs2 = sess2.run(output_names2, input_dict2)

        model_outputs = get_outputs(output_names1, outs1)
        model_raw_outputs = get_outputs(output_names2, outs2)

        for name in model_outputs.keys():
            try:
                if name in model_raw_outputs.keys():
                    value1 = model_outputs[name]
                    value2 = model_raw_outputs[name]
                    diff = np.abs(value1.reshape(-1) - value2.reshape(-1)).max()
                    if diff > 1.0e-6:
                        self.logger.warning(f'check error fail in node: {name}, diff: {diff}')
            except:
                self.logger.warning('name raw check failed in node')
        self.logger.info("---------------- Check modified model Done ----------------")

    def create_session(self, model):
        # for node in model.graph.node:
        #     for output in node.output:
        #         model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        return sess

    # def check_duplicate_name(self):
    #     node_name_list = []
    #     initailizer_name_list = []
    #     for node in self.model.graph.node:
    #         if not node.name in node_name_list:
    #             node_name_list.append(node.name)
    #         else:
    #             print("node name %s duplicated"%node.name)
        
    #     for init in self.model.graph.initializer:
    #         if not init.name in initailizer_name_list:
    #             initailizer_name_list.append(init.name)
    #         else:
    #             print("initializer name %s duplicated"%init.name)

    def make_featuremap_shape_dict(self):
        # create session with all featuremaps calculated
        # do not use self.model, because the next opertions will modify model graph
        model = copy.deepcopy(self.model)
        for node in model.graph.node:
            for output in node.output:
                out_names = [out.name for out in model.graph.output]
                if output in out_names:
                    continue
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        # make network input data
        input_dict = self.make_dummy_inputs(model)
        featuremap_names = [x.name for x in sess.get_outputs()]

        outs = sess.run(featuremap_names, input_dict)
        ftm_shape_dict = dict()
        for (i, name) in enumerate(featuremap_names):
            ftm_shape_dict.update({name: list(outs[i].shape)})

        # add input tensor shape to the dict
        for key in input_dict.keys():
            ftm_shape_dict.update({key: input_dict[key].shape})
        return ftm_shape_dict

    def make_initializer_dict(self):
        initializer_dict=dict()
        for initializer in self.model.graph.initializer:
            initializer_dict.update({initializer.name:initializer})
        return initializer_dict

    def make_dummy_inputs(self, model):
        rng = np.random.RandomState(1234) # type: ignore
        input_dict = dict()
        initializer_name_list = [x.name for x in model.graph.initializer]
        for input in model.graph.input:
            if input.name in initializer_name_list:
                continue
            input_shape = list()
            dims = input.type.tensor_type.shape.dim
            input_shape = [x.dim_value for x in dims]
            if input_shape[0]==0:
                input_shape[0] = 1
            input_size = np.product(np.array(input_shape))
            input_data = rng.randn(input_size)
            input_data = np.reshape(input_data, input_shape).astype(np.float32)
            input_dict.update({input.name: input_data})
        return input_dict

    def save_model(self, save_model_path):
        onnx.save(self.model, save_model_path)

    def delete_constant_node(self):
        for (i, node) in enumerate(self.model.graph.node):
            op_type = node.op_type
            if op_type == "Constant":
                self.model.graph.node.remove(node)

    def add_and_delete_node_types(self, modify_list):
        for event in modify_list[::-1]:
            idx = event['idx']
            add_ops = event['add_node_types']
            delete_ops_list = list()
            if 'delete_idx_list' in event.keys():
                delete_ops_list = event['delete_idx_list']
            else:
                delete_num = event['delete_num']
                delete_ops_list = list(range(idx, idx + delete_num))
            
            for idx in delete_ops_list[::-1]:
                for node_in_name in self.model.graph.node[idx].input:
                    if node_in_name not in self.initializer_dict.keys():
                        for node in self.model.graph.node:
                            if node_in_name in node.output and node.op_type.lower() in [
                                'mul', 'add', 'sub', 'div',
                            ]:
                                self.model.graph.output.extend([onnx.ValueInfoProto(name=node_in_name)])
                                
                self.model.graph.node.remove(self.model.graph.node[idx])

            # delate_ops = self.model.graph.node[idx: idx + delete_num]
            # for op in delate_ops[::-1]:
            #     self.model.graph.node.remove(op)

            for (i, op) in enumerate(add_ops):
                self.model.graph.node.insert(idx + i, op)
        self.delete_constant_node()

    def get_onnx_version(self):
        return self.model.ir_version, self.model.opset_import[0].version

    # def create_initializer(self, **args):
    #     tensor = TensorProto()
    #     tensor.name = args['name']
    #     data_type = args['data_type']
    #     shape = args['shape']
    #     assert isinstance(shape, list), "error: shape should be a list!"
    #     assert data_type in data_type_dict.keys(), 'error: data type invalid!'
    #     tensor.data_type = data_type_dict[data_type]
    #     tensor.dims.MergeFrom(shape)
    #     data = args['data']
    #     n = len(data)
    #     format = data_format_dict[data_type]
    #     tensor.raw_data = struct.pack("%d%s" % (n, format), *data)
    #     return tensor

    def upgrade_ops(self):
        for (i,node) in enumerate(self.model.graph.node):
            op_type = node.op_type
            op_name = node.name
            inputs, outputs = node.input, node.output
            
            if op_type in ['Squeeze', 'Unsqueeze']:
                if len(node.attribute) == 0:
                    continue
                axes = None
                for attr in node.attribute:
                    if attr.name=='axes':            
                        axes = list(node.attribute[0].ints)
                        axes_initializer = TensorProto()
                        axes_initializer.name = "%s_axes"%(op_name)
                        axes_initializer.data_type=TensorProto.INT64
                        n = len(axes)
                        axes_initializer.dims.MergeFrom([n]) # type: ignore
                        axes_initializer.raw_data = struct.pack("%dl" %n, *axes)
                        self.model.graph.initializer.append(axes_initializer)
                        inputs.append(axes_initializer.name)
                self.model.graph.node.pop(i)
                new_node = onnx.helper.make_node( # type: ignore
                    name=op_name,
                    op_type=op_type,
                    inputs = inputs,
                    outputs = outputs
                )
                self.model.graph.node.insert(i, new_node)

            if op_type == 'Split':
                if len(node.attribute)==0:
                    continue
                split = None
                axis = 0 # default
                for x in node.attribute:
                    if x.name == 'split':
                        split = list(x.ints)
                    if x.name == 'axis':
                        axis = x.i
                if split:
                    initializer_split = TensorProto()
                    initializer_split.name = "%s_split"%op_name
                    initializer_split.data_type=TensorProto.INT64
                    n = len(split)
                    initializer_split.dims.MergeFrom([n]) # type: ignore
                    initializer_split.raw_data = struct.pack("%dl" %n, *split)
                    self.model.graph.initializer.append(initializer_split)
                    self.model.graph.node.pop(i)
                    inputs.append(initializer_split.name)

                new_node = onnx.helper.make_node( # type: ignore
                    name=op_name,
                    op_type=op_type,
                    inputs = inputs,
                    outputs = outputs,
                    axis = axis
                )
                self.model.graph.node.insert(i, new_node)

            if op_type == 'Upsample' or op_type == 'Resize':
                self.model.graph.node[i].op_type='Resize'
                if len(self.model.graph.node[i].input) < 3:
                    roi = TensorProto()
                    roi.name = "%s_roi_empty"%op_name
                    roi.data_type=TensorProto.FLOAT
                    roi.dims.MergeFrom([0]) # type: ignore
                    self.model.graph.node[i].input.insert(1, roi.name)
                    self.model.graph.initializer.append(roi)
                if self.model.graph.node[i].attribute[0].s.decode('utf-8') == "linear":
                    attr = onnx.helper.make_attribute("coordinate_transformation_mode", "asymmetric") # type: ignore
                    self.model.graph.node[i].attribute.insert(0, attr)

            if op_type=='Dropout':
                input=node.input
                attributes = node.attribute
                # ratio = None
                # if len(attributes) > 0:
                #     for attr in attributes:
                #         if attr.name == 'ratio':
                #             ratio = attr.f
                # if ratio:
                #     ratio_initializer = TensorProto()
                #     ratio_initializer.name = "%s_ratio"%op_name
                #     ratio_initializer.data_type=TensorProto.FLOAT
                #     ratio_initializer.dims.MergeFrom([1])
                #     ratio_initializer.raw_data = struct.pack("1f", ratio)  
                #     model.graph.initializer.append(ratio_initializer)             
                #     inputs.append(ratio_initializer.name)

                output=node.output
                self.model.graph.node.pop(i)
                new_node = onnx.helper.make_node( # type: ignore
                    op_type='Dropout',
                    name=op_name,
                    inputs=input,
                    outputs=output
                )
                self.model.graph.node.insert(i, new_node)

            if op_type=='Pad':
                inputs = node.input
                outputs = node.output
                if len(inputs) > 1:# pads and value in inputs
                    continue
            
                mode, pads, value = 'constant', None, None
                for attr in node.attribute:
                    if attr.name == 'mode':
                        mode=attr.s
                    if attr.name == 'pads':
                        pads = list(attr.ints)
                    if attr.name == 'value':
                        value=attr.f
                if pads:
                    pads_initializer = TensorProto()
                    pads_initializer.name = "%s_pads"%op_name
                    pads_initializer.data_type=TensorProto.INT64
                    n = len(pads)
                    pads_initializer.dims.MergeFrom([n]) # type: ignore
                    pads_initializer.raw_data = struct.pack("%dl" %n, *pads)
                    self.model.graph.initializer.append(pads_initializer)
                    inputs.append(pads_initializer.name)
                if value:
                    value_initializer = TensorProto()
                    value_initializer.name = "%s_value"%op_name
                    value_initializer.data_type=TensorProto.FLOAT
                    #value_initializer.dims.MergeFrom([1])
                    value_initializer.raw_data = struct.pack("1f", value)  
                    self.model.graph.initializer.append(value_initializer)   
                    inputs.append(value_initializer.name)

                self.model.graph.node.pop(i)
                new_node = onnx.helper.make_node( # type: ignore
                    name=op_name,
                    op_type='Pad',
                    inputs = inputs,
                    outputs = outputs,
                    mode = mode
                )
                self.model.graph.node.insert(i, new_node)    

            if op_type=='BatchNormalization':
                attrs = node.attribute
                delete_list = []
                for (k,attr) in enumerate(attrs):
                    if not attr.name in ['epsilon', 'momentum','training_mode']:
                        delete_list.append(k)
                for idx in delete_list[::-1]:
                    self.model.graph.node[i].attribute.pop(idx)
        
            if op_type=='Clip':
                attributes=node.attribute
                if len(attributes)==0:
                    continue
                min_val = None
                max_val = None
                for attr in attributes:
                    if attr.name == 'min':
                        min_val = attr.f
                    if attr.name == 'max':
                        max_val = attr.f
                min_value_initializer=None
                max_value_initializer=None
                min_value_initializer = TensorProto()
                min_value_initializer.name = "%s_min"%op_name
                min_value_initializer.data_type=TensorProto.FLOAT
                #min_value_initializer.dims.MergeFrom([1])
                min_value_initializer.raw_data = struct.pack("1f", min_val)  
                self.model.graph.initializer.append(min_value_initializer) 

                max_value_initializer = TensorProto()
                max_value_initializer.name = "%s_max"%op_name
                max_value_initializer.data_type=TensorProto.FLOAT
                #max_value_initializer.dims.MergeFrom([1])
                max_value_initializer.raw_data = struct.pack("1f", max_val)  
                self.model.graph.initializer.append(max_value_initializer) 
                
                for k in range(2):
                    self.model.graph.node[i].attribute.pop(0)
                self.model.graph.node[i].input.insert(1, min_value_initializer.name)
                self.model.graph.node[i].input.insert(2, max_value_initializer.name)        
    
    def make_cell_hidden_states_as_inputs(self):
        delete_initializer_names_dict, add_output_names_dict = dict(), dict()
        initializer_names = [initializer.name for initializer in self.model.graph.initializer]
        for (i, node) in enumerate(self.model.graph.node):
            op_type = node.op_type
            inputs, outputs = node.input, node.output
            node_name = node.name
            if op_type in ['GRU']:
                delete_idx_list = [-1] # [init_h]
            elif op_type in ['LSTM']:
                delete_idx_list = [-3, -2] # [init_h, init_c]
            else:
                delete_idx_list = []
            if op_type in ['GRU', 'LSTM']:
                delete_initializer_names_t = []
                for delete_idx in delete_idx_list:
                    input = inputs[delete_idx]
                    if input in initializer_names:
                        delete_initializer_names_t.append(input)
                        # print("test")
                    inputs[delete_idx] = node_name.lower() + "_" + input
                delete_initializer_names_dict[node_name] = delete_initializer_names_t
                add_output_names_dict[node_name] = outputs[1:]
                outputs = outputs[:1]

        def find_node_names_with_same_init_h(init_h_name, delete_initializer_names_dict):
            node_names_with_same_init_h = []
            for node_name, delete_initializer_names in delete_initializer_names_dict.items():
                if init_h_name in delete_initializer_names:
                    node_names_with_same_init_h.append(node_name)

            return node_names_with_same_init_h

        ### when multiple 'init_h' exists, remove 'init_h' from initializer.
        for _, delete_initializer_names in delete_initializer_names_dict.items():
            for i, delete_initializer_name in enumerate(delete_initializer_names):
                for idx, weight in enumerate(self.model.graph.initializer):
                    if weight.name == delete_initializer_name:
                        # data = np.frombuffer(weight.raw_data, dtype=np.float32)
                        # weight_data = np.reshape(data, weight.dims)
                        # weight.ClearField('raw_data')
                        node_names_with_same_init_h = find_node_names_with_same_init_h(
                            delete_initializer_name, delete_initializer_names_dict,
                        )
                        for node_name in node_names_with_same_init_h:
                            new_name = node_name.lower() + "_" + delete_initializer_name
                            input_info = onnx.helper.make_tensor_value_info(new_name, TensorProto.FLOAT, weight.dims) # type: ignore
                            self.model.graph.input.insert(len(self.model.graph.input), input_info)
                            new_name = add_output_names_dict[node_name][i]
                            output_info = onnx.helper.make_tensor_value_info(new_name, TensorProto.FLOAT, weight.dims) # type: ignore
                            self.model.graph.output.insert(len(self.model.graph.output), output_info)
                        self.model.graph.initializer.pop(idx)
                        break
        # print("test")

    def set_version(self):
        ir_version, opset_version = self.get_onnx_version()
        if opset_version < 15:
            self.upgrade_ops()
        self.model.ir_version = self.ir_version_default
        self.model.opset_import[0].version = self.opset_version_default      

    def rename_node(self):
        """
        when node.name is same as model.graph.output, rename node.name.
        """
        out_names = [output.name for output in self.model.graph.output]
        for (i, node) in enumerate(self.model.graph.node):
            if node.name in out_names:
                node.name = node.name + ".cvt.timesintelli"
    #         # print("test")

    def process(self):
        self.logger.info('-------------------- Start model preprocess --------------------')
        # upgrade model op-set version
        self.set_version()
        # self.rename_node()

        for (i, operation) in enumerate(self.node_operation_pipeline):
            try:
                class_name = operation['method']
                # print(operation['name'], len(self.model.graph.output))
                # if "ReplaceSliceOps" == operation['name']:
                    # print("test")
                self.featuremap_shape_dict = self.make_featuremap_shape_dict()
                self.initializer_dict = self.make_initializer_dict()
                args = dict(model=self.model, featuremap_shape_dict=self.featuremap_shape_dict)
                args.update(operation)
                args.update(dict(logger=self.logger)) # type: ignore
                # create instance
                operator = NODE_OPERATION.get(class_name)(**args) # type: ignore
                change_list = operator.forward()
                self.add_and_delete_node_types(change_list)
                # for test
                if self.save_path:
                    self.save_model(self.save_path)
            except:
                self.logger.error("{} of preprocess failure!".format(operation['method']))
                os._exit(-1) # type: ignore

        self.compare_model_nodes()
        self.check()
        self.remove_unused_initializer()
        if self.save_path:
            self.save_model(self.save_path)

        self.logger.info("---------------- Model Preprocess Done ----------------")
        
        return self.model
    
    def remove_unused_initializer(self):
        input_list_all = list()
        for node in self.model.graph.node:
            for input in node.input:
                if not input in input_list_all:
                    input_list_all.append(input)
        unused_initializer_idx_list = list()
        for (i, init) in enumerate(self.model.graph.initializer):
            if not init.name in input_list_all:
                unused_initializer_idx_list.append(i)
        for idx in unused_initializer_idx_list[::-1]:
            deleted_name = self.model.graph.initializer[idx].name
            self.model.graph.initializer.pop(idx)
            # Note: if deleted initializer saved in graph.input but not deleted, it will be returned by calling sess.get_inputs()
            for i,input in enumerate(self.model.graph.input):
                if input.name == deleted_name:
                    self.model.graph.input.pop(i)
            for i,input in enumerate(self.model.graph.value_info):
                if input.name == deleted_name:
                    self.model.graph.value_info.pop(i)

    def compare_model_nodes(self):
        old_model_nodes = dict()
        new_model_nodes = dict()
        for node in self.model.graph.node:
            if node.op_type in new_model_nodes.keys():
                new_model_nodes[node.op_type] +=1
            else:
                new_model_nodes.update({node.op_type:1})
        for node in self.model_raw.graph.node:
            if node.op_type in old_model_nodes.keys():
                old_model_nodes[node.op_type] +=1
            else:
                old_model_nodes.update({node.op_type:1})  
        for key in old_model_nodes.keys():
            if not key in new_model_nodes.keys():
                new_model_nodes.update({key:0})  
        for key in new_model_nodes.keys():
            if not key in old_model_nodes.keys():
                old_model_nodes.update({key:0})
        self.logger.info("| ---------------------- Compare model nodes ---------------------- |\n")
        self.logger.info("| -- node type -- | -- before simplified -- | -- after simplified --|\n")
        for key in old_model_nodes.keys():
            self.logger.info("| %s |  %d  |  %d  |\n"%(key, old_model_nodes[key], new_model_nodes[key]))

class OnnxProcessOffLine(OnnxProcess):
    def __init__(self, **kwargs):
        super(OnnxProcessOffLine, self).__init__(**kwargs)
        self.save_path = kwargs['save_path'] if 'save_path' in kwargs.keys() else None

    def make_operation_pipline(self, **kwargs):
        self.node_operation_pipeline = offline_operation_pipeline.copy()
        if 'delete_node_names' in kwargs.keys():
            self.node_operation_pipeline.extend([{'method':'DeleteOps', 'delete_node_names':kwargs['delete_node_names']}])

    def process(self):
        self.logger.info('-------------------- Start model preprocess --------------------')

        self.make_cell_hidden_states_as_inputs()

        # upgrade model op-set version
        for (i,operation) in enumerate(self.node_operation_pipeline):
            try:
                class_name = operation['method']
                self.featuremap_shape_dict = self.make_featuremap_shape_dict()
                self.initializer_dict = self.make_initializer_dict()
                args = dict(model=self.model, featuremap_shape_dict=self.featuremap_shape_dict)
                args.update(operation)
                args.update(dict(logger=self.logger)) # type: ignore
                # create instance
                operator = NODE_OPERATION.get(class_name)(**args) # type: ignore
                change_list = operator.forward()
                self.add_and_delete_node_types(change_list)
                # for test
                if self.save_path:
                    self.save_model(self.save_path)
            except:
                self.logger.error("{} of preprocess failure!".format(operation['method']))
                os._exit(-1) # type: ignore

        self.compare_model_nodes()
        #self.check() # offline mode may be check fail due to delete input transpose node
        self.remove_unused_initializer()

        if self.save_path:
            self.save_model(self.save_path)

        self.logger.info("---------------- Model Preprocess Done ----------------")
        
        return self.model





# below are test demo
if False:
    import os
    def get_all_models(root, all_models):
        # recusively find all onnx models under root
        for file in os.listdir(root):
            if not os.path.isdir(os.path.join(root,file)):
                if file.endswith('onnx'):
                    all_models.append(os.path.join(root,file))
            else:
                get_all_models(os.path.join(root,file), all_models)
        return

    def demo():
        # test_cases = \
        # [
        #     {
        #         'model_path' : "/home/qinnan/models/voice/voice_denoise_p1.onnx",
        #         'save_path1' : '/home/qinnan/models/model_simplify_result/voice_denoise_p1_offline.onnx',
        #         'save_path2' : '/home/qinnan/models/model_simplify_result/voice_denoise_p1_final.onnx',
        #         "delete_node_names": ['Transpose_0','Transpose_5', 'Reshape_11', 'Reshape_13', 'Reshape_16'],
        #     },
        #     {
        #         'model_path' : "/home/qinnan/models/voice/voice_denoise_p2.onnx",
        #         'save_path1' : '/home/qinnan/models/model_simplify_result/voice_denoise_p2_offline.onnx',
        #         'save_path2' : '/home/qinnan/models/model_simplify_result/voice_denoise_p2_final.onnx',
        #         "delete_node_names": ['Transpose_12', 'Transpose_17', 'Reshape_23', 'Reshape_25'],
        #     },
        # ]

        ## test all models
        test_cases = list()
        model_root='/home/ts300026/workspace/trained_models/'
        save_root='/home/ts300026/workspace/model_transformer/results/'
        model_path_list = list()
        get_all_models(model_root, model_path_list)
        save_path_list = [ x.split('/')[-1].split('\\')[-1].split('//')[-1] for x in model_path_list]
        save_path_list = [os.path.join(save_root, x) for x in save_path_list]
        for i in range(len(save_path_list)):
            test_cases.append({'model_path':model_path_list[i], "save_path1":save_path_list[i], "save_path2":save_path_list[i]})

        ## test one model
        # test_cases = \
        # [
        #     {
        #         'model_path' : "/home/ts300026/workspace/trained_models/nanodet-sim.onnx",
        #         'save_path1' : '/home/ts300026/workspace/model_transformer/results/nanodet-sim.onnx',
        #         'save_path2' : '/home/ts300026/workspace/model_transformer/results/nanodet-sim.onnx',                
        #     }
        # ]

        for (i, case) in enumerate(test_cases[50:]):
            print("~~~~~~~~~~~~~~~~~~~~~~~~ the %d -th case ~~~~~~~~~~~~~~~~~~~~~~~"%i)
            model=onnx.load(case['model_path'])
            offline_operations = case['delete_node_names'] if 'delete_node_names' in case.keys() else []
            engine = OnnxProcessOffLine(model=model, delete_node_names=offline_operations, save_path=case['save_path1'])
            model = engine.process()
            del engine
            engine = OnnxProcess(model=model, save_path=case['save_path2'])
            engine.process()
            del engine


    if __name__ == '__main__':
        demo()
