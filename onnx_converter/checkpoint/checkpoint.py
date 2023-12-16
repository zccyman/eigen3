# Copyright (c) shiqing. All rights reserved.
import copy
import os
import numpy as np
from typing import List, Any

import onnx

from .onnxsim import simplify, get_input_names

# from .attribute import *
from . import attribute
try:
    from utils import Object
    from utils import Registry
    from checkpoint.preprocess import OnnxProcess
except:
    from onnx_converter.utils import Object # type: ignore
    from onnx_converter.utils import Registry # type: ignore
    from onnx_converter.checkpoint.preprocess import OnnxProcess # type: ignore

CHECKPOINTS: Registry = Registry('checkpoint', scope='')

'''
process already quantize push our framework which using other company toolkits
'''


class Node(object):
    def __init__(self, name='', weights=None, input=None, output=None, attr=None, op_type='', is_result=False):
        """

        :type is_result: object
        """
        if attr is None:
            attr = dict()
        if output is None:
            output = list()
        if input is None:
            input = list()
        if weights is None:
            weights = list()
        self.__name = name
        self.__input = input
        self.__onnx_input = input
        self.__output = output
        self.__onnx_output = output
        self.__weights = weights
        self.__attr = attr
        self.__is_reslut = is_result
        self.__op_type = op_type
        # convert onnx model is
        self.__is_dynamic = False

    def set_dynamic(self, dynamic):
        self.__is_dynamic = dynamic

    def is_dynamic(self):
        return self.__is_dynamic

    def get_onnx_input(self):
        return self.__onnx_input

    def set_onnx_input(self, value):
        """

        :type value: object
        """
        self.__onnx_input = value

    def set_onnx_output(self, output):
        self.__onnx_output = output

    def get_onnx_output(self):
        return self.__onnx_output

    def set_input(self, value, idx=0):
        if not value:
            self.__input = []
        else:
            # input_ = value if isinstance(value, list) else [value]
            # self.__input.extend(input_)
            self.__input.insert(idx, value)

    def get_input(self):
        return self.__input

    def get_output(self):
        return self.__output

    def set_output(self, output, idx=0):
        if not output:
            self.__output = []
        else:
            if isinstance(output, list):
                for i, o in enumerate(output):
                    self.__output.insert(idx+i, o)
            else:
                self.__output.insert(idx, output)

    def get_name(self):
        return self.__name

    def set_name(self, name):
        self.__name = name

    def get_weights(self):
        return self.__weights

    def set_weights(self, weights):
        self.__weights = weights

    def get_attr(self):
        return self.__attr

    def set_attr(self, attr):
        self.__attr = attr

    def get_is_result(self):
        return self.__is_reslut

    def set_result_node(self, is_result: bool):
        self.__is_reslut = is_result

    def get_op_type(self):
        return self.__op_type

    def set_op_type(self, op_type):
        self.__op_type = op_type


class QuantizedParser(object):
    def __init__(self):
        pass


@CHECKPOINTS.register_module(name='onnxparser')
class OnnxParser(Object): # type: ignore
    # todo remove onnx model constant node
    # constant maybe conflict with node has constant array setting with fused batch normal
    def __init__(self, **kwargs):
        super(OnnxParser, self).__init__(**kwargs)
        self.all_ops = kwargs['all_ops']
        self.model = None  # onnx.onnx_ml_pb2.ModelProto()
        self.sess_options = self.get_sess_options(shared_librarys=kwargs["shared_librarys"])
        # self.graph = onnx.load_model(infile)
        # self.infile = infile
        # self.parser(infile)
        self.__nodes = []
        self.__inputs = dict()
        self.__out_names = []
        self.__weight_names = []
        self._is_dynamic = False
        self.input_names = kwargs['input_names'] if 'input_names' in kwargs.keys() else None

        self.__weights_rawdata, self.__weights_data = dict(), dict()
        self.__weights_dims, self.__weights_dtype = dict(), dict()

        # UNDEFINED = 0 FLOAT = 1 UINT8 = 2 INT8 = 3 UINT16 = 4 INT16 = 5 INT32 = 6
        # INT64 = 7 STRING = 8 BOOL = 9 FLOAT16 = 10  DOUBLE = 11 UINT32 = 12
        # UINT64 = 13 COMPLEX64 = 14 COMPLEX128 = 15 BFLOAT16 = 16
        self.__dtypes = kwargs['dtypes']
        self.__vaild_bit = kwargs['vaild_bit']
        self.__dtypes_ = {value: key for key, value in self.__dtypes.items()}
        self.__shape_template = kwargs['shape_template']
        self.log_name = kwargs.get('log_name', 'parse.log')
        self.log_level = kwargs.get('log_level', 20)
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level)
        self.is_simplify = kwargs.get('is_simplify', True)
        self.preprocess = kwargs['preprocess'] if 'preprocess' in kwargs.keys() else True

    def _simplify(self):
        try:
            self.logger.info('onnx simplify!')
            inputs = copy.deepcopy(self.__inputs)
            # todo config input shape, dynamic shape has included str
            for key in inputs:
                shape = inputs[key]
                while True in [isinstance(s, str) for s in shape]:
                    idx = [isinstance(s, str) for s in shape].index(True)
                    shape[idx] = self.__shape_template[idx]

            self.logger.info('onnx model dynamic {}'.format(self._is_dynamic))

            self.model, status = simplify(self.model, input_shapes=inputs, dynamic_input_shape=self._is_dynamic) # type: ignore
            if not status:
                self.logger.fatal('simplify error')
                os._exit(-1)
        except:
            self.logger.fatal('simplify error')
            os._exit(-1)

    def get_opset_version(self):
        return self.model.opset_import[0].version # type: ignore

    def get_sess_options(self, shared_librarys):
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        for shared_library in shared_librarys:
            sess_options.register_custom_ops_library(shared_library)
        return sess_options

    # process network input shape and attr
    def _parse_input(self):
        try:
            # def process(dims, idx, value=512):
            #     for ix in idx:
            #         dims[ix] = value if dims[ix] == 0 else dims[ix]
            input_names = get_input_names(self.model) # type: ignore
            if self.input_names:
                input_names = list(set(self.input_names).intersection(set(input_names)))
            self.logger.info('onnx graph input is: \n {}'.format(self.model.graph.input)) # type: ignore
            for index, value in enumerate(self.model.graph.input): # type: ignore
                # is_input = 'input' in value.name or 'image' in value.name or 'data' in value.name
                # is_input = value.name in input_names
                # if not is_input:
                #     continue
                # if value.name in self.__weight_names:
                #     continue
                if value.name not in input_names:
                    continue
                dims: List[Any] = list()
                for item in value.type.tensor_type.shape.dim:
                    dim_param = item.dim_param
                    dim_value = item.dim_value
                    if dim_value == 0 and dim_param != '':
                        dims.append(dim_param)
                        self._is_dynamic = True
                    else:
                        dims.append(dim_value)
                self.logger.info('input index {}, dims is: {}'.format(index, dims))
                self.__inputs[value.name] = dims
        except:
            print("parse model input failure!")
            os._exit(-1)

    # parser interface
    def parse(self, infile):
        self.logger.info('onnx model name is {}'.format(infile))
        if infile is None or not os.path.exists(infile):
            self.logger.fatal('invalid model path')
            os._exit(-1)
        self.model = onnx.load_model(infile)
        
        if self.is_simplify:
            self._simplify()
        if self.preprocess:
            try:
                processor = OnnxProcess(model=self.model, opset_version=self.get_opset_version(), log_name=self.log_name, log_level=self.log_level)
                self.model = processor.process()
                # processor.check()
                # if False:
                #     processor.save_model('work_dir/modified_model.onnx')
            except:
                self.logger.error('onnx converter onnx model preprocess failed!')
                os._exit(-1)
        try:
            self._parse_input()
            self._parse_output()
            self._get_nodes()
            self._connect_node()
            self._set_input_nodes()
            self._set_results()
            self._parse_attr()
        except:
            self.logger.error('parse onnx model preprocess to timeintelli model structure failed!')
            os._exit(-1)
        self.logger.info('parse onnx model done!')

        return self.__nodes, self.model, get_input_names(self.model), self.__out_names, self.sess_options

    def _parse_output(self):
        self.__out_names = [output.name for output in self.model.graph.output] # type: ignore
        self.logger.info('onnx out name is: {}'.format(self.__out_names))

    def _parser_weight(self):
        
        for weight in self.model.graph.initializer: # type: ignore
            self.__weights_dims[weight.name] = weight.dims
            self.__weights_dtype[weight.name] = weight.data_type
            if weight.data_type == 3:
                self.__weights_dtype[weight.name] = 6
            assert self.__weights_dtype[weight.name] in self.__vaild_bit, 'unsupported data type'            
            
            if len(weight.float_data) > 0:
                self.__weights_rawdata[weight.name] = weight.float_data
                data = np.asarray(weight.float_data, dtype=eval('np.' + self.__dtypes_[self.__weights_dtype[weight.name]]))
            else:
                self.__weights_rawdata[weight.name] = weight.raw_data
                data = np.frombuffer(weight.raw_data, dtype=eval('np.' + self.__dtypes_[self.__weights_dtype[weight.name]]))

            if data.size == 0:
                self.__weights_data[weight.name] = None
            else:
                self.__weights_data[weight.name] = np.reshape(data, weight.dims)
                self.__weights_data[weight.name] = np.reshape(data, weight.dims)

    def _set_node_inout_name(self, index, idx, in_idx=0, out_idx=0):
        idx_name = self.__nodes[idx].get_name()
        index_name = self.__nodes[index].get_name()
        # self.nodes[index]['output'].extend([self.nodes[idx]['name']])
        self.__nodes[index].set_output(idx_name, out_idx)
        # self.nodes[idx]['input'].extend([self.nodes[index]['name']])
        self.__nodes[idx].set_input(index_name)

    def _connect_name(self):
        pass

    def _connect_node(self):
        try:
            in_ = [copy.deepcopy(node.get_input()) for node in self.__nodes]
            out_ = [copy.deepcopy(node.get_output()) for node in self.__nodes]
            [node.set_input([]) for node in self.__nodes[1:]]
            [node.set_output([]) for node in self.__nodes]
            # nodes = self.__nodes
            for idx, ii in enumerate(in_):
                # if idx < 1:
                #     continue
                # single input
                if 0 < len(ii) < 2:
                    if not isinstance(ii, list):
                        ii = [ii]
                    # try branch is single output
                    try:
                        index = out_.index(ii)
                        self._set_node_inout_name(index, idx)
                    except:
                        for out_i, out_1 in enumerate(out_):
                            if len(out_1) < 1:
                                continue
                            if isinstance(ii, list) and ii[0] in out_1:
                                index = out_i
                            elif isinstance(ii, str):
                                index = out_i
                            else:
                                continue
                            self._set_node_inout_name(index, idx, out_idx=out_1.index(ii[0]))
                            '''
                            idx_name = self.nodes[idx].get_name()
                            index_name = self.nodes[index].get_name()
                            #self.nodes[index]['output'].extend([self.nodes[idx]['name']])
                            self.nodes[index].set_output(idx_name)
                            # self.nodes[idx]['input'].extend([self.nodes[index]['name']])
                            self.nodes[index].set_input(index_name)
                            '''
                # todo multi input, try-exception is reasonable or not
                elif len(ii) >= 2:
                    for iii in ii:
                        out_idx = None
                        # if not isinstance(iii, list):
                        #     iii = [iii]
                        try:
                            index = out_.index([iii])
                            out_idx = index
                        except:
                            for oii, o in enumerate(out_):
                                if isinstance(o, list):
                                    if iii in o:
                                        index = oii
                                        out_idx = o.index(iii)
                                else:
                                    if iii in o:
                                        index = oii
                        # if out_idx is None:
                            # self.__nodes[idx].set_input(iii)
                        if out_idx is not None:
                            self._set_node_inout_name(index, idx, out_idx=out_idx) # type: ignore
                        '''
                        self.nodes[index]['output'].extend([self.nodes[idx]['name']])
                        self.nodes[idx]['input'].extend([self.nodes[index]['name']])
                        '''
                else:
                    raise Exception('_connect_node appears error !!!')
                    #os._exit(-1

            # set output node
            for index, out in enumerate(out_):
                if isinstance(out, list):
                    for out_1 in out:
                        idx = out.index(out_1)
                        if out_1 in self.__out_names:
                            self.__nodes[index].set_output(out_1, idx=idx)
                elif isinstance(out, str) and out in self.__out_names:
                    self.__nodes[index].set_output(out)
        except:
            self.logger.error("switch our data structure failure!")
            os._exit(-1)

    # set network output node
    def _set_results(self):
        try:
            for node in self.__nodes:
                if set(node.get_onnx_output()).intersection(set(self.__out_names)):
                    node.set_result_node(True)
        except:
            self.logger.error("output layer setting failure!")

    # set network input node
    def _set_input_nodes(self):
        input_names = get_input_names(self.model) # type: ignore
        if self.input_names:
            input_names = list(set(self.input_names).intersection(set(input_names)))
        for value in self.model.graph.input: # type: ignore
            # is_input = 'input' in value.name or 'image' in value.name or 'data' in value.name
            # is_input = value.name in input_names
            # if not is_input:
            #     continue
            # if value.name in self.__weight_names:
            #     continue
            if value.name not in input_names:
                continue
            # dims: List[Any] = list()
            # if not value.name in self.__weight_names:
            dims = [item.dim_value for item in value.type.tensor_type.shape.dim]
            dim_param = [item.dim_param for item in value.type.tensor_type.shape.dim]
            inode = Node()
            is_dynamic = np.sum([dim == '' for dim in dim_param]) > 0
            inode.set_dynamic(is_dynamic)
            for node in self.__nodes:
                inputs = node.get_onnx_input()
                if value.name in inputs:
                    inode.set_name(value.name)
                    inode.set_input([])
                    inode.set_op_type('data')
                    inode.set_output(node.get_name())
                    if len(node.get_onnx_input()) > 1:
                        onnx_idx_i = node.get_onnx_input().index(value.name)
                        inode.set_onnx_output([node.get_onnx_input()[onnx_idx_i]])
                        if not value.name in node.get_input():
                            node.set_input(inode.get_name(), onnx_idx_i)
                    else:
                        inode.set_onnx_output(node.get_onnx_input())
                        if not node.get_input():
                            node.set_input(node.get_onnx_input()[0])
                    # if not node.get_input():
                    # if True:
                    #     node.set_input(node.get_onnx_input())
                    inode.set_attr(dict(shape=dims))

            self.__nodes.insert(0, inode)

    def _get_nodes(self):
        try:

            lambda_att = lambda node: [
                dict(name=item.name, ints=item.ints, i=item.i, float=item.f, mode=item.s, type=self.__dtypes_[item.type])
                for
                item in node.attribute]

            lambda_weight = lambda node, weights_name: [input if input in weights_name else None for input in node.input]

            lambda_node_input = lambda node, wname: [None if item in wname else item for item in node.input]

            self.__weight_names = [weight.name for weight in self.model.graph.initializer] # type: ignore

            self._parser_weight()

            for n_idx, gnode in enumerate(self.model.graph.node): # type: ignore
                self.logger.info('node idx {}, name {}, type {}'.format(n_idx, gnode.name, gnode.op_type))
                wname = list(filter(None, lambda_weight(gnode, self.__weight_names)))
                input = list(filter(None, lambda_node_input(gnode, wname)))
                weights_ = []
                for name in wname:
                    # weights_.append(
                    #     dict(name=name, weight=self.__weights_data[name], weight_raw=self.__weights_rawdata[name],
                    #         dtype=self.__dtypes_[self.__weights_dtype[name]], dims=self.__weights_dims[name]))
                    weights_.append(
                        dict(name=name, weight=self.__weights_data[name], 
                             dtype=self.__dtypes_[self.__weights_dtype[name]], dims=self.__weights_dims[name]))
                name, op_type, output, attr = gnode.name, gnode.op_type, list(gnode.output), lambda_att(gnode)
                if gnode.op_type.lower() in ["add", "sub", "mul", "matmul", "concat"] and len(wname) > 0:
                    i_list, w_list = [], []
                    for i_idx, g_input in enumerate(list(gnode.input)):
                        if g_input in wname:
                            w_list.append(i_idx)
                        else:
                            i_list.append(i_idx)
                    attr.append(dict(weight_idx=w_list, input_idx=i_list))
                    # print("test")
                # if op_type == 'Clip':
                #     if len(attr) >= 2:
                #         top = attr[0]['float']
                #         bottom = attr[1]['float']
                #     else:
                #         try:
                #             value = [float(weights_[0]['weight']), float(weights_[1]['weight'])]
                #             top, bottom = max(value), min(value)
                #         except:
                #             top, bottom = 0, 0
                #
                #     if bottom == 0 and top == 6: op_type = 'Relu6'

                if not op_type in self.all_ops:
                    self.logger.warning('new ops type {} in onnx model!'.format(op_type))

                if not self.is_simplify and op_type == 'Constant':
                    continue

                name = name if name != '' else op_type + '__' + str(n_idx)
                if op_type in ["Gemm", "Conv", "ConvTranspose"]:
                    if len(weights_) == 1:
                        if op_type == "MatMul":
                            out_c = weights_[0]['weight'].shape[1]
                            weights_[0]['weight'] = np.transpose(weights_[0]['weight'], (1,0))
                        elif op_type == "ConvTranspose":
                            out_c = weights_[0]['weight'].shape[1]
                        else:
                            out_c = weights_[0]['weight'].shape[0]
                        weights_.append(
                            dict(
                            name=weights_[0]["name"] + ".bias",
                            weight=np.zeros(out_c, dtype=np.float32),
                            weight_raw=None,
                            dtype=weights_[0]['dtype'],
                            dims=[out_c],
                            )
                        )
                self.__nodes.append(
                    Node(name=name, weights=weights_, input=input, output=output, attr=attr, op_type=op_type))
        except:
            self.logger.error("parse weights failure!")
            os._exit(-1)

    def _parse_attr(self):
        try:
            for node in self.__nodes:
                attr = node.get_attr()
                op_type = node.get_op_type()
                # # if op_type.lower() in ['mul', 'slice', 'gather', 'div', 'shape', 'extend', 'scatter', 'add']:
                #
                # if not op_type.lower() in ['gather', 'conv', 'relu', 'mul', 'add', 'convtranspose', 'gemm'] and \
                #         not op_type.lower() in ['reshape', 'concat', 'slice', 'reducemin', 'unsqueeze', 'shape', 'equal', 'cast', 'roialign', 'squeeze',
                #                                 'transpose', 'nonmaxsuppression', 'constantofshape', 'nonzero', 'sub', 'less', 'topk', 'div']:
                #     print(node.get_weights(), node.get_op_type(), node.get_name())
                if op_type == 'data':
                    continue
                if hasattr(attribute, 'parse_' + op_type.lower()):
                    node.set_attr(getattr(attribute, 'parse_' + op_type.lower())(attr))
                    if op_type.lower() == "hardsigmoid":
                        alpha, beta = node.get_attr().get("alpha"), node.get_attr().get("beta")
                        if alpha and beta:
                            max_value = alpha * 3 + beta
                            min_value = alpha * (-3) + beta
                            if max_value > 1.0:
                                self.logger.warning("hardsigmoid max value is: {}".format(max_value))
                            if min_value < 0.0:
                                self.logger.warning("hardsigmoid min value is: {}".format(min_value))
                else:
                    self.logger.warning('not support ops to parse attribute! {}'.format(op_type))
                attrs = node.get_attr()
                for val in attrs.keys():
                    if "google" in str(type(attrs[val])):
                        attrs[val] = list(attrs[val])
                node.set_attr(attrs)
                # try:
                #     node.set_attr(eval('parse_' + op_type.lower())(attr))
                # except:
                #     print('not support ops to parse attribute! {}'.format(op_type))
                #     continue
        except:
            self.logger.error("parse attribute failure!")
            os._exit(-1)


@CHECKPOINTS.register_module(name='pytorchparser')
class PytorchParser(OnnxParser):
    pass


class TensorflowParser(OnnxParser):
    pass

# if __name__ == '__main__':
#     parser = OnnxParser()
#     res = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/nanodet_sim.onnx')
#     # res = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/mobilefacenet-sim.onnx')
#     # res = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/mobilefacenet-gemm.onnx')
#     # res = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/googlenet-3.onnx')
#     # res = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/retinanet-9.onnx')
#     # res = parser.parse('/home/shiqing/Downloads/model_tools/trained_mdoels/MaskRCNN-10.onnx', is_simplify=True)
#     print('done')
