import onnx
import numpy as np
from onnx import AttributeProto
import copy
import onnxruntime as rt
from onnx import TensorProto, ValueInfoProto
import struct

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


class VoiceModelModifier():
    def __init__(self, model_path, save_path, splice_weight_txt=None):
        self.model = onnx.load(model_path)
        self.save_path = save_path
        self.initializer_names = [x.name for x in self.model.graph.initializer]
        self.tensor_names = [x.name for x in self.model.graph.value_info]
        self.graph_inputs, self.graph_outputs = self.get_graph_input_output()
        self.input_tensor = self.graph_inputs[0]
        self.output_tensor = self.graph_outputs[0]
        self.custom_op_domain = 'timesintelli.com'
        self.custom_op_list = ['Splice']

        self.weight = None
        self.bias = None
        first_splice_fc_weights = list()
        if splice_weight_txt:
            f = open(splice_weight_txt, 'r')
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split()
                first_splice_fc_weights.append(np.array([float(x) for x in line]))
            first_splice_fc_weights = np.array(first_splice_fc_weights).astype(np.float32)
            first_splice_fc_weights = first_splice_fc_weights.transpose(1, 0)
            f.close()
            self.weight = list(first_splice_fc_weights[:-1, :].flatten())
            self.bias = list(first_splice_fc_weights[-1, :].flatten())

    def splice_op_modify(self):
        has_fc = 1
        for node in self.model.graph.node:
            if node.op_type == 'Splice':
                node.attribute.append(AttributeProto(name='has_fc', i=has_fc, type=2))
                if has_fc and self.weight:
                    node.attribute.append(AttributeProto(name="weight", floats=self.weight, type=6))
                    node.attribute.append(AttributeProto(name="bias", floats=self.bias, type=6))
                has_fc = 0

    def create_initializer(self, **args):
        tensor = TensorProto()
        tensor.name = args['name']
        data_type = args['data_type']
        shape = args['shape']
        assert isinstance(shape, list), "error: shape should be a list!"
        assert data_type in data_type_dict.keys(), 'error: data type invalid!'
        tensor.data_type = data_type_dict[data_type]
        tensor.dims.MergeFrom(shape)
        data = args['data']
        n = len(data)
        format = data_format_dict[data_type]
        tensor.raw_data = struct.pack("%d%s" % (n, format), *data)
        return tensor

    def get_graph_input_output(self):
        graph_input = list()
        for input in self.model.graph.input:
            name = input.name
            if not name in self.initializer_names:
                graph_input.append(input)
        graph_output = self.model.graph.output
        return graph_input, graph_output

    def get_dim(self, x):
        dims = x.type.tensor_type.shape.dim
        return [x.dim_value for x in dims]

    def get_tensor_by_name(self, name):
        if name in self.tensor_names:
            tensor = self.model.graph.value_info[np.argwhere(np.array(self.tensor_names) == name)[0][0]]
        else:
            tensor = self.output_tensor
        return tensor

    def modify_domain(self):
        self.model.opset_import[0].version = 15
        self.model.opset_import[0].domain = ""
        for layer in self.model.graph.node:
            if layer.op_type in self.custom_op_list:
                layer.domain = self.custom_op_domain
            else:
                layer.domain = ''

    def replace_batchnorm(self):
        for i in range(len(self.model.graph.node) - 1, 0, -1):
            layer = self.model.graph.node[i]
            if layer.op_type == 'BatchNorm':
                eps = 0
                target_rms = 1
                for attr in layer.attribute:
                    if attr.name == "target_rms":
                        target_rms = attr.f
                input_shape = self.get_dim(
                    self.model.graph.value_info[np.argwhere(np.array(self.tensor_names) == layer.input[0])[0][0]])
                output_shape = self.get_dim(
                    self.model.graph.value_info[np.argwhere(np.array(self.tensor_names) == layer.output[0])[0][0]])
                attrs = layer.attribute
                for attr in attrs:
                    if attr.name == "epsilon":
                        eps = attr.f

                inputs = layer.input
                scale = self.create_initializer(name='%s_scale' % layer.name, shape=[input_shape[1]],
                                                data_type='float32', data=[target_rms] * input_shape[1])
                bias = self.create_initializer(name='%s_bias' % layer.name, shape=[input_shape[1]], data_type='float32',
                                               data=[0] * input_shape[1])
                inputs.insert(1, scale.name)
                inputs.insert(2, bias.name)

                self.model.graph.initializer.append(scale)
                self.model.graph.initializer.append(bias)
                node = onnx.helper.make_node(
                    'BatchNormalization',
                    name=layer.name,
                    inputs=inputs,
                    outputs=layer.output,
                    epsilon=eps,
                    # training_mode=training_mode
                )
                self.model.graph.node.pop(i)
                self.model.graph.node.insert(i, node)

    def modify_tensor_dims(self):
        # get batch
        batch = self.get_dim(self.input_tensor)[1]
        self.batch = batch
        # modify input tensor shape
        dims = self.input_tensor.type.tensor_type.shape.dim
        dims.pop(0)
        input_value_info = self.get_tensor_by_name(self.input_tensor.name)
        dims = input_value_info.type.tensor_type.shape.dim
        dims.pop(0)

        for layer in self.model.graph.node:
            output_name = layer.output[0]
            output_tensor = self.get_tensor_by_name(output_name)
            shape = self.get_dim(output_tensor)
            batch_0 = shape[1]
            output_tensor.type.tensor_type.shape.dim.pop(0)
            output_tensor.type.tensor_type.shape.dim[0].dim_value = batch
            attrs = layer.attribute
            for attr in attrs:
                if attr.name == 'input_dim':
                    attrs.remove(attr)
            for attr in attrs:
                if attr.name == 'output_dim':
                    attrs.remove(attr)
            if False:
                new_attr = AttributeProto(name='original_batch', i=batch_0, type=2)
                attr.append(new_attr)
        print('Finish modify tensor dims')

    def forward(self):
        self.modify_domain()
        self.modify_tensor_dims()
        self.replace_batchnorm()
        self.splice_op_modify()
        onnx.save(self.model, self.save_path)


def build_submodel(model_path, node_idx_section, new_input_names, new_output_names, save_path):
    model = onnx.load(model_path)
    new_model = copy.deepcopy(model)
    num_layers = len(model.graph.node)
    initializer_names = np.array([x.name for x in model.graph.initializer])
    tensor_names = np.array([x.name for x in model.graph.value_info])
    input_names = np.array([x.name for x in model.graph.input])
    output_names = np.array([x.name for x in model.graph.output])

    # new_input_name = model.graph.node[node_idx_section[0]].inputs[0]
    # new_output_name = model.graph.node[node_idx_section[-1]].outputs[0]
    new_inputs, new_outputs = list(), list()
    for input_name in new_input_names:
        assert input_name in tensor_names or input_name in input_names, 'Error: new_input_name %s not found' % (
            input_name)
        if input_name in tensor_names:
            new_inputs.append(model.graph.value_info[np.argwhere(tensor_names == input_name)[0][0]])
        else:
            new_inputs.append(model.graph.input[np.argwhere(input_names == input_name)[0][0]])
    for output_name in new_output_names:
        assert output_name in tensor_names or output_name in output_names, 'Error: new_output_name %s not found' % (
            output_name)
        if output_name in tensor_names:
            new_outputs.append(model.graph.value_info[np.argwhere(tensor_names == output_name)[0][0]])
        else:
            new_outputs.append(model.graph.output[np.argwhere(output_names == output_name)[0][0]])

    new_layers = list()
    for i in node_idx_section:
        new_layers.append(model.graph.node[i])

    new_initializers = list()
    new_tensors = list()
    for layer in new_layers:
        input_names = layer.input
        for input_name in input_names:
            if input_name in initializer_names:
                idx = np.argwhere(initializer_names == input_name)[0][0]
                new_initializers.append(model.graph.initializer[idx])
            elif input_name in tensor_names:
                idx = np.argwhere(tensor_names == input_name)[0][0]
                new_tensors.append(model.graph.value_info[idx])
    # remove
    for x in model.graph.node:
        new_model.graph.node.remove(x)
    for x in model.graph.input:
        new_model.graph.input.remove(x)
    for x in model.graph.output:
        new_model.graph.output.remove(x)
    for x in model.graph.initializer:
        new_model.graph.initializer.remove(x)
    for x in model.graph.value_info:
        new_model.graph.value_info.remove(x)
    # add
    for x in new_layers:
        new_model.graph.node.append(x)
    for x in new_inputs:
        new_model.graph.input.append(x)
    for x in new_outputs:
        new_model.graph.output.append(x)
    for x in new_initializers:
        new_model.graph.initializer.append(x)
    for x in new_tensors:
        new_model.graph.value_info.append(x)

    onnx.save(new_model, save_path)


def test_model(model_path, input_file_path, output_file_path):
    model = onnx.load(model_path)
    f1 = open(input_file_path, 'r')
    f2 = open(output_file_path, 'r')
    input_gt, output_gt = list(), list()
    for line in f1.readlines():
        data = line.strip("\n").split()
        input_gt.append(np.array([float(x) for x in data]))
    f1.close()
    for line in f2.readlines():
        data = line.strip("\n").split()
        output_gt.append(np.array([float(x) for x in data]))
    f2.close()

    input_shape = [x.dim_value for x in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [x.dim_value for x in model.graph.output[0].type.tensor_type.shape.dim]

    input_gt = np.array(input_gt).astype(np.float32)
    output_gt = np.array(output_gt).astype(np.float32)
    assert input_gt.shape[1] == input_shape[1] and output_gt.shape[1] == output_shape[1], 'error, data cols mismatch'
    input_gt = input_gt[:input_shape[0], :input_shape[1]]
    output_gt = output_gt[:output_shape[0], :output_shape[1]]

    # input = None
    # if inputs is None:
    #     input = np.random.randn(n).reshape(shape).astype(np.float32)
    # else:
    #     input = inputs
    onnx.checker.check_model(model)
    sess = rt.InferenceSession(model.SerializeToString(), None)
    outs = sess.run([x.name for x in model.graph.output], {model.graph.input[0].name: input_gt})
    outs = outs[0].flatten()
    output_gt = output_gt.flatten()

    diff = np.mean(np.abs(outs - output_gt))
    print("The mean abs diff between model output and gt is %f" % diff)
    if diff < 1e-6:
        print("Model test pass!")
    else:
        print("Model test fail!")


# test case
if __name__ == "__main__":
    model_dir = '/home/ts300026/workspace/kaldi_model/'
    model_path = model_dir + 'decode_v11.onnx'
    save_path = model_dir + 'decode_v11_adj.onnx'
    first_splice_fc_weights = model_dir + "splice_first_mat.txt"

    modifier = VoiceModelModifier(model_path, save_path, first_splice_fc_weights)
    modifier.forward()

    # submodel tests
    sub_model_dir = model_dir + 'partiation/'
    data_dir = sub_model_dir + "data/"
    submodel_cases = \
        [
            {"model_name": "bn1.onnx", "layer_list": [3], "input_name": ['tdnn1.relu'],
             "output_name": ["tdnn1.batchnorm"], \
             "input_file": data_dir + "ouput_tdnn1.relu.txt", "output_file": data_dir + "ouput_tdnn1.batchnorm.txt"},
            {"model_name": "bn2.onnx", "layer_list": [7], "input_name": ['tdnn2.relu'],
             "output_name": ["tdnn2.batchnorm"], \
             "input_file": data_dir + "ouput_tdnn2.relu.txt", "output_file": data_dir + "ouput_tdnn2.batchnorm.txt"},
            {"model_name": "bn3.onnx", "layer_list": [11], "input_name": ['tdnn3.relu'],
             "output_name": ["tdnn3.batchnorm"], \
             "input_file": data_dir + "ouput_tdnn3.relu.txt", "output_file": data_dir + "ouput_tdnn3.batchnorm.txt"},
            {"model_name": "bn4.onnx", "layer_list": [15], "input_name": ['tdnn4.relu'],
             "output_name": ["tdnn4.batchnorm"], \
             "input_file": data_dir + "ouput_tdnn4.relu.txt", "output_file": data_dir + "ouput_tdnn4.batchnorm.txt"},
            {"model_name": "bn5.onnx", "layer_list": [19], "input_name": ['tdnn5.relu'],
             "output_name": ["tdnn5.batchnorm"], \
             "input_file": data_dir + "ouput_tdnn5.relu.txt", "output_file": data_dir + "ouput_tdnn5.batchnorm.txt"},
            {"model_name": "bn6.onnx", "layer_list": [23], "input_name": ['tdnn6.relu'],
             "output_name": ["tdnn6.batchnorm"], \
             "input_file": data_dir + "ouput_tdnn6.relu.txt", "output_file": data_dir + "ouput_tdnn6.batchnorm.txt"},
            {"model_name": "bn7.onnx", "layer_list": [26], "input_name": ['prefinal-chain.relu'],
             "output_name": ["prefinal-chain.batchnorm"],
             "input_file": data_dir + "output_prefinal-chain.relu.txt",
             "output_file": data_dir + "output_prefinal-chain.batchnorm.txt"}
        ]
    for case in submodel_cases:
        submodel_path = sub_model_dir + case['model_name']
        build_submodel(save_path, case['layer_list'], case['input_name'], case['output_name'], submodel_path)
        print("test submodel : %s" % case['model_name'])
        test_model(submodel_path, case['input_file'], case['output_file'])
