import sys

sys.path.append("./")  # NOQA: E402

import os

import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras
from tensorflow.keras import layers
from utils import Registry  # NOQA: E402

SIMULATION_OPS: Registry = Registry('simulation_ops')  # NOQA: E402

from test_build_model import SIMULATION_OPS as myops  # NOQA: E402


def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def nchw2nhwc(data):
    return data.transpose(0, 2, 3, 1)


def nhwc2nchw(data):
    return data.transpose(0, 3, 1, 2)


def to_numpy(data):
    return data.detach().numpy()


def to_torch(data):
    return torch.from_numpy(data)


class MyFCModel(nn.Module):
    def __init__(self):
        super(MyFCModel, self).__init__()

        self.middle_channel = 7
        self.conv_base = nn.Conv2d(
            3, self.middle_channel, kernel_size=3, stride=1, padding=1)

        self.N = 6
        for i in range(self.N):
            conv = nn.Conv2d(self.middle_channel, self.middle_channel,
                             kernel_size=3, stride=2, padding=1, groups=self.middle_channel)
            setattr(self, "conv_{}".format(i), conv)

        self.fc0 = nn.Linear(self.middle_channel, 25)
        self.fc1 = nn.Linear(25, 5)

    def forward(self, x):
        x = self.conv_base(x)

        for i in range(self.N):
            x = getattr(self, "conv_{}".format(i))(x)

        out = F.avg_pool2d(x, (5, 5))
        out = out.view(out.size(0), -1)
        out = self.fc0(out)
        out = self.fc1(out)

        return out


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # self.conv_a = nn.Conv2d(3, 11, kernel_size=3, stride=1, padding=1)
        # self.conv_b = nn.Conv2d(3, 11, kernel_size=3, stride=1, padding=1)
        # self.conv_c = nn.Conv2d(3, 11, kernel_size=3, stride=1, padding=1)
        # self.conv_d = nn.Conv2d(3, 13, kernel_size=3, stride=1, padding=1)

        # self.conv = nn.Conv2d(46, 42, kernel_size=3, stride=1, padding=1)

        # out_c = 33
        # self.conv0 = nn.Conv2d(12, out_c, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(13, out_c, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(10, out_c, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(5, out_c, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(2, out_c, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(out_c, out_c, kernel_size=3,
        #                        stride=1, padding=1, groups=out_c)
        # self.conv6 = nn.Conv2d(
        #     out_c, out_c, kernel_size=3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(out_c)

        # self.conv_x = nn.Conv2d(22, 22, kernel_size=3, stride=1, padding=1)
        # self.conv_xx = nn.Conv2d(5, 11, kernel_size=3, stride=1, padding=1)
        # self.conv_xxx = nn.Conv2d(11, 11, kernel_size=3, stride=1, padding=1)

        self.conv_1 = nn.Conv2d(3, 22, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(11, 11, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(11, 11, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(22, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_1(x)
        x_a, x_b = torch.split(x, [11, 11], dim=1)
        x_a = self.conv_2(x_a)
        x_b = self.conv_3(x_b)
        x = torch.cat([x_a, x_b], dim=1)
        x = self.conv_4(x)
        
        # x_a = self.conv_a(x)
        # x_b = self.conv_b(x)
        # x_c = self.conv_c(x)
        # x_d = self.conv_d(x)

        # x = x_a * x_b
        # x = self.conv_x(x)

        # x = torch.concat([x_a, x_b], dim=1)
        # x = self.conv_x(x)
        # x1, x2 = torch.split(x_a, [5, 6], dim=1)
        # x1, x2 = self.conv_xx(x1), self.conv_xxx(x2)
        # x = torch.concat([x1, x2], dim=1)
        # x = self.conv_xx(x1)

        # x = torch.concat([x_a+x_b, x_b, x_a+x_c, x_d], dim=1)
        # x = self.conv(x)

        # x1, x2, x3, x4, x5 = torch.split(x, [12, 13, 10, 5, 2], dim=1)
        # x = F.max_pool2d(self.conv0(x1), kernel_size=3) + \
        #     F.max_pool2d(self.conv1(x2), kernel_size=3) + \
        #     F.max_pool2d(self.conv2(x3), kernel_size=3) + \
        #     F.max_pool2d(self.conv3(x4), kernel_size=3) + \
        #     F.max_pool2d(self.conv4(x5), kernel_size=3)
        # x = self.conv5(x)
        # x = self.bn(self.conv6(x))

        return x


class Base(object):
    def __init__(self, **kwargs):
        self.op_type = kwargs["op_type"]
        self.model_path = kwargs["model_path"]
        self.onnx_file = os.path.join(self.model_path, kwargs["onnx_file"])
        # print("rm -rf {}".format(self.model_path))
        os.system("rm -rf {}".format(self.model_path))
        makedir(self.model_path)

    def build_net(self):
        pass

    def infer_net(self, x):
        self.net.eval()

        x = to_torch(nhwc2nchw(x))
        res = self.net(x)
        res = to_numpy(res)

        return res

    def export_onnx(self, x, opset_version=12):
        torch.onnx.export(self.net,
                          to_torch(nhwc2nchw(x)),
                          self.onnx_file,
                          export_params=True,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=['input.1'],
                          output_names=['output.1'])
        os.system(
            f"python -m onnxsim {self.onnx_file} {self.onnx_file} --input-shape 1,3,320,320"
        )

    def __call__(self, x):
        self.net = self.build_net()
        res = self.infer_net(x)
        self.export_onnx(x)

        return res


class BaseTF(Base):
    def __init__(self, **kwargs):
        super(BaseTF, self).__init__(**kwargs)

        self.tf_version = tf.__version__
        self.tf_version = int(self.tf_version.split(".")[0])

    def infer_net(self, x):
        res = self.net(x)

        if self.tf_version > 1:
            res = res.numpy()
        else:
            with tf.Session() as sess:
                res = res.eval(session=sess)

        return res

    def export_onnx(self, x, opset_version=12):
        if self.tf_version > 1:
            tf.keras.models.save_model(self.net, self.model_path)
        else:
            tf.keras.experimental.export_saved_model(self.net, self.model_path)
        os.system(
            f"python -m tf2onnx.convert --saved-model {self.model_path} --inputs-as-nchw input:0 --opset {opset_version} --output {self.onnx_file}"
        )
        os.system(
            f"python -m onnxsim {self.onnx_file} {self.onnx_file} --input-shape 1,512"
        )
        # pb_file = "model.pb" #self.onnx_file.replace(".onnx", ".pb")
        # os.system(f"onnx-tf convert -i {self.onnx_file} -o {pb_file}")
        # from onnx_tf.backend import prepare
        # onnx_model = onnx.load(self.onnx_file)  # load onnx model
        # tf_rep = prepare(onnx_model)  # prepare tf representation
        # tf_rep.export_graph("abc")  # export the model


@SIMULATION_OPS.register_module(name="conv_torch")
class ConvTorch(Base):
    def __init__(self, **kwargs):
        super(ConvTorch, self).__init__(**kwargs)

        self.in_c = kwargs["in_c"]
        self.out_c = kwargs["out_c"]
        self.kernel_size = kwargs["kernel_size"]
        self.stride = kwargs["stride"]
        self.padding = kwargs["padding"]
        self.dilation = kwargs["dilation"]
        self.groups = kwargs["groups"]
        self.bias = kwargs["bias"]
        self.relu = kwargs["relu"]

    def build_net(self):
        # model = nn.Conv2d(self.in_c,
        #                   self.out_c,
        #                   kernel_size=self.kernel_size,
        #                   stride=self.stride,
        #                   padding=self.padding,
        #                   dilation=self.dilation,
        #                   groups=self.groups,
        #                   bias=self.bias)
        # from eval.tests.ghost_net import ghost_net
        # from eval.tests.hrnet import HighResolutionNet, hyper_parameters
        # hp = hyper_parameters()
        # model = HighResolutionNet(hp.blocks, hp.num_channels, hp.num_modules, hp.num_branches, hp.num_blocks, hp.fuse_method)
        model = MyNet()  # MyNet MyFCModel

        return model


@SIMULATION_OPS.register_module(name="fc_tf")
class FCTF(BaseTF):
    def __init__(self, **kwargs):
        super(FCTF, self).__init__(**kwargs)

    def build_net(self):
        inputs = layers.Input(shape=(512, ), name="input")
        outputs = layers.Dense(1024, use_bias=None, name="output")(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        return model


if __name__ == '__main__':
    data = np.random.random([1, 320, 320, 3]).astype(np.float32)
    # data = np.random.random([2, 512]).astype(np.float32)

    # ["Conv", "BatchNormalization", "MaxPool", "AveragePool", "Resize"]
    test_op_type = "Conv"
    deepframe = "Torch"  # Torch
    model_path = "work_dir"
    onnx_file = "model.onnx"
    TorchConfig = {
        "op_type": test_op_type,
        "in_c": data.shape[-1],
        "out_c": 12,
        "kernel_size": (3, 3),
        "padding": (1, 1),
        "stride": (1, 1),
        "dilation": (1, 1),
        "groups": 1,
        "bias": True,
        "relu": "nn.ReLU",
        "align_corners": False,
        "resize_mode": "bilinear",  # nearest bilinear bicubic
        "output_size": (64, 64),
        "model_path": model_path,
        "onnx_file": onnx_file
    }

    TFConfig = {
        "op_type": test_op_type,
        "kernel_size": (3, 3),
        "padding": "same",  # "same"
        "strides": (1, 1),
        "input_shape": data.shape[1:],
        "use_bias": False,
        "align_corners": True,
        "resize_mode": "bilinear",  # nearest bilinear bicubic
        "input_size": (320, 320),
        "output_size": (64, 64),
        "model_path": model_path,
        "onnx_file": onnx_file
    }

    Config = eval(deepframe + "Config")
    res = myops.get("conv_torch")(**Config)(data)
    print("=>>>>>>>>>>>>>> res:", res.shape)
