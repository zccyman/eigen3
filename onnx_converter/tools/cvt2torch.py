import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from utils import Registry, RoundFunction, FloorFunction, ClampFunction
except Exception:
    from onnx_converter.utils import Registry, RoundFunction, FloorFunction, ClampFunction

CVT2TORCH: Registry = Registry("cvt2torch", scope="")


class TORCH_BASE(nn.Module):

    def __init__(self, **kwargs):
        super(TORCH_BASE, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.layer = kwargs.get("layer", None)
        self.process_scale = self.layer.get_scale_type()
        # self.device_master = kwargs.get("device_master", "cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device_master = torch.device(self.device_master)
        
        self.is_tranning = True
        
    def set_train_mode(self, is_trainning):
        self.is_tranning = is_trainning if self.process_scale != 'smooth' else False
                
    def set_device(self, device):
        self.device = torch.device(device)

    def get_device(self):
        return self.device
        
        
@CVT2TORCH.register_module(name="data")
class TORCH_PLACEHOLDER(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_PLACEHOLDER, self).__init__(**kwargs)

    def forward(self, in_data):
        # data = copy.deepcopy(in_data[0]).to(device=self.device)
        data = in_data[0].to(device=self.device)
        return [data]


@CVT2TORCH.register_module(name="conv")
@CVT2TORCH.register_module(name="convtranspose")
@CVT2TORCH.register_module(name="depthwiseconv")
@CVT2TORCH.register_module(name="fc")
class TORCH_CONV2D(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_CONV2D, self).__init__(**kwargs)
        self.requires_grad = kwargs.get("requires_grad", False)
        self.qat_nofakequant = kwargs.get("qat_nofakequant", False)
        self.layer = kwargs["layer"]
        self.weight = copy.deepcopy(self.layer.get_layer_ops()['weights'][0])
        self.weight = torch.from_numpy(self.weight)
        self.bias = copy.deepcopy(self.layer.get_layer_ops()['weights'][1])
        self.bias = torch.from_numpy(self.bias)
        self.weight = nn.Parameter(self.weight,
                                   requires_grad=self.requires_grad)
        self.bias = nn.Parameter(self.bias, requires_grad=self.requires_grad)
        if not self.requires_grad:
            self.weight_origin = copy.deepcopy(self.weight)
            self.bias_origin = copy.deepcopy(self.bias)
        # else:
        #     self.sk = nn.Parameter(torch.max(torch.abs(self.weight) / 127.0), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            mu, sigma = copy.deepcopy(self.weight).mean(), copy.deepcopy(self.weight).std()
            neg_w = torch.abs(mu - 3 * sigma)
            pos_w = torch.abs(mu + 3 * sigma)
            dmax_init = torch.max(neg_w, pos_w)
            dmax_init = torch.max(torch.abs(self.weight))
            self.sk_dmax = nn.Parameter(dmax_init, requires_grad=self.requires_grad)
            sk = torch.abs(self.sk_dmax) / 127.0
            qw, qw_floor = self.weight / sk, FloorFunction.apply(self.weight / sk)
            hv_init = (qw - qw_floor)
            self.hv = nn.Parameter(hv_init, requires_grad=self.requires_grad)
            self.alpha = nn.Parameter(torch.zeros(self.weight.shape), requires_grad=self.requires_grad)
            self.beta = nn.Parameter(torch.zeros(self.bias.shape), requires_grad=self.requires_grad)
            # self.init_beta_success = False
        
        self.attrs = self.layer.get_layer_ops()['attrs'][0]
        if "pads" in self.attrs.keys():
            pads = copy.deepcopy(self.attrs["pads"])
            self.padding = [pads[1], pads[3], pads[0], pads[2]]
        node_types = [
            node.get_op_type().lower() for node in self.layer.get_nodes()
        ]
        # self.has_relu = True if 'relu' in node_types else False
        # if self.has_relu:
        #     self.relu = torch.nn.ReLU()
        self.act_type = self.layer.get_layer_ops()['ops'][-1]
        swish = lambda x: x * torch.sigmoid(x)
        act = lambda x: x
        if self.act_type == "leakyrelu":
            negative_slope = self.layer.get_layer_ops()['attrs'][-1]['alpha']
        else:
            negative_slope = 0.01
        self.myactivations = {
            "act": act,
            "relu": torch.nn.ReLU(),
            "relu6": torch.nn.ReLU6(),
            "relux": torch.nn.ReLU6(),  #lamda x: ClampFunction.apply(x, 0, 6),
            "leakyrelu": torch.nn.LeakyReLU(negative_slope=negative_slope),
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
            "swish": swish,
            "hardsigmoid": F.hardsigmoid,
            "hardtanh": F.hardtanh,
            "hardswish": F.hardswish,
            "hardshrink": F.hardshrink,
        }
        
    def apply_fake_quant(self, weight, is_training=True):
        sk = torch.abs(self.sk_dmax).to(self.device) / 127.0
        weight = self.apply_only_quant(weight, is_training) 
        return weight * sk
       
    def apply_only_quant(self, weight, is_training=True):
        sk = torch.abs(self.sk_dmax).to(self.device) / 127.0
        weight = weight.to(self.device)
        weight = RoundFunction.apply(weight / sk)
        
        # weight = FloorFunction.apply(weight / sk)
        # if is_training:
        #     weight += ClampFunction.apply(
        #         self.hv.to(self.device), 
        #         # RoundFunction.apply(self.hv.to(self.device)), 
        #         torch.Tensor([0.0]).to(self.device), 
        #         torch.Tensor([1.0]).to(self.device),
        #     )
        # else:
        #     weight += ClampFunction.apply(
        #         RoundFunction.apply(self.hv.to(self.device)), 
        #         torch.Tensor([0.0]).to(self.device), 
        #         torch.Tensor([1.0]).to(self.device),
        #     )
                        
        weight = ClampFunction.apply(
            weight, 
            torch.Tensor([-128.0]).to(self.device), 
            torch.Tensor([127.0]).to(self.device),
        )
        
        return weight
           
    def weight_add_offset(self, weight):
        weight = torch.add(weight, self.alpha.to(weight.device))
        return weight
                
    def bias_add_offset(self, bias):
        bias = torch.add(bias, self.beta.to(bias.device))
        # out_c = weight.shape[0]
        # bias += torch.sum(weight.reshape(out_c, -1), dim=-1) * self.beta        
        return bias

    def infer(self, in_data):
        if self.requires_grad:
            # self.sk_dmax = nn.Parameter(torch.max(torch.abs(self.weight)), requires_grad=False)
            self.sk = torch.abs(self.sk_dmax).to(self.device) / 127.0
            weight = self.weight.to(self.device)
            weight = self.weight_add_offset(weight)
            weight = self.apply_fake_quant(weight)
            # weight = self.apply_only_quant(weight)
            
            # weight = RoundFunction.apply(weight / self.sk)
            # weight = ClampFunction.apply(
            #     weight,
            #     torch.Tensor([-2.0**7]).to(self.device),
            #     torch.Tensor([2.0**7 - 1]).to(self.device),
            # ) * self.sk
            # bias = RoundFunction.apply(self.bias / self.si / self.sk)
            # bias = ClampFunction.apply(
            #     bias,
            #     torch.Tensor([-2.0**31]).to(self.device),
            #     torch.Tensor([2.0**31 - 1]).to(self.device),
            # ) * self.si * self.sk
            bias = self.bias.to(self.device)
            bias = self.bias_add_offset(bias)
            bias = self.fake_quant(bias, self.si * self.sk, num_bits=32)
        else:
            if self.qat_nofakequant:
                weight = self.weight.to(self.device)
                bias = self.bias.to(self.device)
            else:
                weight = self.weight_origin.to(self.device)
                bias = self.bias_origin.to(self.device)

        # if self.requires_grad:
        #     if not self.init_beta_success:
        #         self.beta.data = torch.min(in_data)
        #         self.init_beta_success = True
        #     bias = self.bias_add_offset(weight, bias)

        if self.layer.get_layer_type() in ["conv"]:
            out_data = F.conv2d(input=in_data,
                                weight=weight,
                                bias=bias,
                                stride=tuple(self.attrs["strides"]),
                                padding=(0, 0),
                                dilation=tuple(self.attrs["dilations"]),
                                groups=1)
        elif self.layer.get_layer_type() in ["depthwiseconv"]:
            out_data = F.conv2d(input=in_data,
                                weight=weight,
                                bias=bias,
                                stride=tuple(self.attrs["strides"]),
                                padding=(0, 0),
                                dilation=tuple(self.attrs["dilations"]),
                                groups=self.attrs["out_c"])
        elif self.layer.get_layer_type() in ["fc", "gemm", "matmul"]:
            out_data = F.linear(in_data, weight=weight, bias=bias)
        else:
            raise Exception("Not supported layer type")
        
        # if self.requires_grad:
        #     bias_tmp = bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        #     bias_tmp = bias_tmp.expand(out_data.shape)
        #     # si = self.si.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #     # si = si.expand(out_data.shape)
        #     bias_tmp = self.fake_quant(bias_tmp, self.si * self.sk, num_bits=32)
        #     out_data += bias_tmp
            
        return out_data

    def get_scale_shift(self, scale, bit=32, lower=0.5):
        for shift in range(-bit, bit):
            out_scale = scale * (2**(-shift))
            if lower < out_scale < 1:
                return np.int32(shift), out_scale
        return np.int32(0), scale
    
        # if isinstance(scale, np.ndarray):
        #     shifts, scales = np.zeros_like(scale,
        #                                    dtype=np.int32), np.zeros_like(scale)
        #     for idx, s in enumerate(scale.reshape(-1)):
        #         for shift in range(-bit, bit):
        #             out_scale = s * (2**(-shift))
        #             if lower < out_scale < 1:
        #                 shifts[idx] = shift
        #                 scales[idx] = out_scale
        #                 break
        #     return shifts, scales
        # else:
        #     for shift in range(-bit, bit):
        #         out_scale = scale * (2**(-shift))
        #         if lower < out_scale < 1:
        #             return np.int32(shift), out_scale

    def round_and_clamp(self, data, min_value=-128.0, max_value=127.0):
        # data = RoundFunction.apply(data)
        data = ClampFunction.apply(
            data,
            torch.Tensor([min_value]).to(data.device),
            torch.Tensor([max_value]).to(data.device),
        )
        return data

    def get_txme_clip(self):
        return self.txme_clip
    
    def get_txme_scale(self):
        return dict(out_shift=self.out_shift, out_scale=self.out_scale, scale=self.scale_)
    
    def forward(self, in_data, si=1.0, sk=1.0, so=1.0, fake_quant=None, quant=None, dequant=None):
        self.si, self.so, self.fake_quant, self.only_quant, self.only_dequant = si, so, fake_quant, quant, dequant

        data = in_data[0].to(device=self.device)
        # if self.requires_grad:
            # data = self.only_quant(data, self.si)
        
        # w_bit_select = 1
        # if w_bit_select == 1:
        #     max_value, min_value = 2**7 - 1, -2**7
        # else:
        #     max_value, min_value = 2**15 - 1, -2**15
        # def get_quan_data(data, scale, zero_point=0):
        #     transformed_val = data.reshape(-1) / scale + zero_point
        #     quantized = torch.round(transformed_val)
        #     quantized = torch.clip(quantized, min_value, max_value)
        #     return torch.reshape(quantized, data.shape)

        # def get_dequan_data(data, scale, zero_point=0):
        #     dequantize = (data.reshape(-1) - zero_point) * scale
        #     return torch.reshape(dequantize, data.shape)
        
        # data = get_dequan_data(get_quan_data(data, self.si), self.si)
        
        if "pads" in self.attrs.keys():
            data = F.pad(data, self.padding, "constant", 0)

        pred = self.infer(data)

        # pred = get_dequan_data(get_quan_data(pred, self.so), self.so)
        
        # if self.has_relu:
            # pred = self.relu(pred)
            
        if self.requires_grad:
            ema_sc = self.layer.get_ema()[-1]
            if len(ema_sc) > 0:
                if self.is_tranning: self.ema_sc[0](pred) ### update scale
                sc = ema_sc[0].get_scale().to(self.device)
                self.scale_ = self.si * self.sk / sc
                # scale_ = torch.min(scale_, torch.tensor(1.0).to(self.device))
                shift, out_scale = self.get_scale_shift(self.scale_)
                self.out_shift, self.out_scale = float(shift), float(out_scale.item())
                pred = (2.0 ** shift) * RoundFunction.apply(pred / (self.si * self.sk))
                # pred = (2.0 ** shift) * pred
                pred_before_clip = RoundFunction.apply(pred)
                pred = ClampFunction.apply(
                    pred_before_clip,
                    torch.Tensor([-2.0**7]).to(self.device),
                    torch.Tensor([2.0**7-1.0]).to(self.device),) # int8
                diff_x = torch.abs(pred - pred_before_clip)
                if torch.sum(diff_x) > 0:
                    self.txme_clip = torch.mean(diff_x[diff_x != 0])
                else:
                    self.txme_clip = 0.0 * torch.sum(diff_x)
                # if self.out_shift == 0:
                #    self.txme_clip += F.relu(out_scale - torch.Tensor([1.0]).to(self.device))
                pred = pred * out_scale * sc
                pred = self.fake_quant(pred, sc)
                
            # print(self.act_type, len(ema_sc))
                
        pred = self.myactivations[self.act_type](pred)
        
        # if self.requires_grad:
            # pred = self.only_dequant(pred, self.so)
            
        # if self.requires_grad:
        #     pred = ClampFunction.apply(
        #         pred,
        #         torch.Tensor([-2.0**31]).to(pred.device),
        #         torch.Tensor([2.0**31-1.0]).to(pred.device),) # int32
        #     with torch.no_grad():
        #         scale = self.si * self.sk / self.so
        #         out_shift, out_scale = self.get_scale_shift(scale=scale)
        #     pred = pred * (2.0 ** float(out_shift))
        #     pred = ClampFunction.apply(
        #         pred,
        #         torch.Tensor([-2.0**7]).to(pred.device),
        #         torch.Tensor([2.0**7-1.0]).to(pred.device),) # int8
        #     pred = pred * FloorFunction.apply(out_scale * 256.0) # int8 * uint8
        #     pred = ClampFunction.apply(
        #         pred,
        #         torch.Tensor([-2.0**15]).to(pred.device),
        #         torch.Tensor([2.0**15-1.0]).to(pred.device),)# int16
        #     pred = pred * (2.0 ** (-8))
        #     pred = ClampFunction.apply(
        #         pred,
        #         torch.Tensor([-2.0**7]).to(pred.device),
        #         torch.Tensor([2.0**7-1.0]).to(pred.device),) # int8

        # if self.requires_grad:
        #     self.weight_origin = self.weight.data.to(self.device)
        #     self.bias_origin = self.bias.data.to(self.device)
            
        return [pred]


@CVT2TORCH.register_module(name="concat")
class TORCH_CONCAT(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_CONCAT, self).__init__(**kwargs)


    def forward(self, in_data, si=[], so=1.0):
        data_list = []
        for idx in range(len(in_data)):
            tmp = in_data[idx]
            # if len(si) > 0:
            # self.si, self.so = si[idx], so
            # tmp = tmp * self.si / self.so

            # out_scale, int_scale = self.update_scale()
            # tmp = tmp * out_scale * (2 ** (-int_scale))
            # tmp = ClampFunction.apply(
            #     tmp,
            #     torch.Tensor([-2**15]).to(tmp.device),
            #     torch.Tensor([2**15 - 1]).to(tmp.device),
            # )
            data_list.append(tmp.to(device=self.device))
        data = torch.concat(data_list, axis=1)
        # if len(si) > 0:
        #     data = ClampFunction.apply(
        #         data,
        #         torch.Tensor([-2**7]).to(data.device),
        #         torch.Tensor([2**7 - 1]).to(data.device),
        #     )

        return [data]


@CVT2TORCH.register_module(name="split")
class TORCH_SPLIT(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_SPLIT, self).__init__(**kwargs)
        self.layer = kwargs["layer"]
        self.attrs = self.layer.get_layer_ops()['attrs'][0]        
        self.axis = self.attrs['axis']
        self.split = self.attrs['split']
        
    def forward(self, in_data, si=[], so=1.0):
        data = torch.split(in_data[0], tuple(self.split), self.axis)
        return list(data)
    
    
@CVT2TORCH.register_module(name="add")
class TORCH_ADD(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_ADD, self).__init__(**kwargs)

    def forward(self, in_data, si=[], so=1.0):
        # self.si, self.sk, self.so = si, 1.0, so
        # data0 = copy.deepcopy(in_data[0]).to(device=self.device)
        # data1 = copy.deepcopy(in_data[1]).to(device=self.device)
        # data0 = in_data[0].to(device=self.device)
        # data1 = in_data[1].to(device=self.device)
        # data = data0 + data1

        data = 0.0
        for idx in range(len(in_data)):
            tmp = in_data[idx]
            if len(si) > 0:
                self.si, self.so = si[idx], so
                # out_scale, int_scale = self.update_scale()
                # print("idx, out_scale, intscale: ", idx, out_scale, int_scale)
                # tmp = tmp * out_scale
                # tmp = tmp * (2 ** (-int_scale))
                # tmp = tmp * self.si / self.so
                # tmp = ClampFunction.apply(
                #     tmp,
                #     torch.Tensor([-2**15]).to(tmp.device),
                #     torch.Tensor([2**15 - 1]).to(tmp.device),
                # )
            data += tmp.to(device=self.device)
        # if len(si) > 0:
        #     data = ClampFunction.apply(
        #         data,
        #         torch.Tensor([-2**7]).to(data.device),
        #         torch.Tensor([2**7 - 1]).to(data.device),
        #     )

        return [data]


@CVT2TORCH.register_module(name="mul")
class TORCH_MUL(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_MUL, self).__init__(**kwargs)

    def forward(self, in_data):
        # data0 = copy.deepcopy(in_data[0]).to(device=self.device)
        # data1 = copy.deepcopy(in_data[1]).to(device=self.device)
        data0 = in_data[0].to(device=self.device)
        data1 = in_data[1].to(device=self.device)
        data = data0 * data1
        return [data]


@CVT2TORCH.register_module(name="resize")
class TORCH_RESIZE(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_RESIZE, self).__init__(**kwargs)
        self.layer = kwargs["layer"]
        self.attrs = self.layer.get_layer_ops()['attrs'][0]

        self.mode = self.attrs['mode']
        self.sizes = self.attrs['sizes'] if 'sizes' in self.attrs.keys() else 0
        self.scale = self.attrs['scale'] if 'scale' in self.attrs.keys() else 0
        # todo will fixed in node parse
        if isinstance(self.scale, int) and isinstance(self.sizes, int):
            if isinstance(self.attrs['roi'], np.ndarray):
                self.scale = self.attrs['roi']
        # assert isinstance(self.sizes, int) or isinstance(self.scale, int)
        self.coor_trans_mode = self.attrs['coordinate_transformation_mode']
        if self.coor_trans_mode == 'pytorch_half_pixel':
            self.coor_trans_mode = 'asymmetric'
        valid = [
                    ['nearest', 'asymmetric'], ['bilinear', 'align_corners'],
                    ['bilinear', 'asymmetric'], ['bilinear', 'half_pixel'],
                ]
        if self.mode in ['linear', 'cubic']:
            self.mode = 'bi' + self.mode
        assert [self.mode, self.coor_trans_mode] in valid
        self.align_corners = True if self.coor_trans_mode == 'align_corners' else False

    def forward(self, in_data):
        # data = copy.deepcopy(in_data[0]).to(device=self.device)
        data = in_data[0].to(device=self.device)
        if self.align_corners:
            if isinstance(self.sizes, np.ndarray) or isinstance(
                    self.sizes, list):
                data = F.interpolate(
                    data,
                    size=tuple(self.sizes[2:]),
                    mode=self.mode,
                    align_corners=self.align_corners,
                )
            elif isinstance(self.scale, np.ndarray) or isinstance(
                    self.scale, list):
                data = F.interpolate(
                    data,
                    scale_factor=tuple(self.scale[2:]),
                    mode=self.mode,
                    align_corners=self.align_corners,
                )
            else:
                exit(-1)  # , print('wrong output size')
        else:
            if [self.mode, self.coor_trans_mode] == ['bilinear', 'half_pixel']:
                data = F.interpolate(data,
                                     scale_factor=tuple(self.scale[2:]),
                                     mode=self.mode,
                                     align_corners=self.align_corners, 
                                     recompute_scale_factor=True)
            elif isinstance(self.sizes, np.ndarray) or isinstance(
                    self.sizes, list):
                data = F.interpolate(data,
                                     size=tuple(self.sizes[2:]),
                                     mode=self.mode)
            elif isinstance(self.scale, np.ndarray) or isinstance(
                    self.scale, list):
                data = F.interpolate(data,
                                     scale_factor=tuple(self.scale[2:]),
                                     mode=self.mode)
            else:
                print('resize simulation has wrong resize shape!')
                exit(-1)

        return [data]


@CVT2TORCH.register_module(name="maxpool")
class TORCH_MAXPOOL(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_MAXPOOL, self).__init__(**kwargs)
        self.layer = kwargs["layer"]
        self.attrs = self.layer.get_layer_ops()['attrs'][0]

        self.kernel_size, self.stride = tuple(
            self.attrs['kernel_shape']), self.attrs['strides']

        pads = self.attrs['pads'] if 'pads' in self.attrs.keys() else (
            0,
            0,
            0,
            0,
        )
        self.pads = [pads[1], pads[3], pads[0], pads[2]]

        if 'ceil_mode' in self.attrs.keys():
            self.ceil_mode = bool(self.attrs['ceil_mode'])
        else:
            self.ceil_mode = True

    def forward(self, in_data):
        # data = copy.deepcopy(in_data[0]).to(device=self.device)
        data = in_data[0].to(device=self.device)
        if "cuda" == self.device.type:
            min_value = torch.min(data).detach().cpu().numpy()
        else:
            min_value = torch.min(data).detach().numpy()
        padding = nn.ConstantPad2d(tuple(self.pads), value=min_value - 1)
        data = F.max_pool2d(input=padding(data),
                            kernel_size=self.kernel_size,
                            stride=tuple(self.stride),
                            padding=(0, 0),
                            ceil_mode=self.ceil_mode)
        return [data]


@CVT2TORCH.register_module(name='averagepool')
@CVT2TORCH.register_module(name='globalaveragepool')
class TORCH_AVERAGEPOOL(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_AVERAGEPOOL, self).__init__(**kwargs)
        self.layer = kwargs["layer"]
        self.attrs = self.layer.get_layer_ops()['attrs'][0]

        self.stride, self.kernel_size, self.group = 1, 0, 0
        if 'strides' in self.attrs.keys():
            self.stride = self.attrs['strides']
        if 'kernel_shape' in self.attrs.keys():
            self.kernel_size = self.attrs['kernel_shape']
        self.pads = self.attrs['pads'] if 'pads' in self.attrs.keys() else (
            0,
            0,
            0,
            0,
        )

        if 'ceil_mode' in self.attrs.keys():
            self.ceil_mode = bool(self.attrs['ceil_mode'])
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

    def forward(self, in_data):
        # data = copy.deepcopy(in_data[0]).to(device=self.device)
        data = in_data[0].to(device=self.device)
        if self.kernel_size == (0, 0):
            self.kernel_size = tuple(data.shape[2:])
        if "cuda" == self.device.type:
            max_value = torch.max(data).detach().cpu().numpy() + 1.0
        else:
            max_value = torch.max(data).detach().numpy() + 1.0
        c_padding = nn.ConstantPad2d(tuple(self.pads), value=max_value)
        z_padding = nn.ZeroPad2d(tuple(self.pads))
        x_ = (c_padding(data) < max_value).float()
        x_w = F.avg_pool2d(
            input=x_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
            ceil_mode=self.ceil_mode,
        )
        data = F.avg_pool2d(
            input=z_padding(data),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
            ceil_mode=self.ceil_mode,
        )
        data = (data / x_w)

        return [data]


@CVT2TORCH.register_module(name="relu")
@CVT2TORCH.register_module(name="relu6")
@CVT2TORCH.register_module(name="swish")
@CVT2TORCH.register_module(name="sigmoid")
@CVT2TORCH.register_module(name="hardswish")
@CVT2TORCH.register_module(name="hardsigmoid")
class TORCH_ACTIVATION(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_ACTIVATION, self).__init__(**kwargs)
        self.layer = kwargs["layer"]
        self.attrs = self.layer.get_layer_ops()['attrs'][0]
        swish = lambda x: x * torch.sigmoid(x)
        activations = {
            "relu": torch.nn.ReLU(),
            "relu6": torch.nn.ReLU6(),
            "relux": torch.nn.ReLU6(),  #lamda x: ClampFunction.apply(x, 0, 6),
            "leakyrelu": torch.nn.LeakyReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
            "swish": swish,
            "hardsigmoid": F.hardsigmoid,
            "hardtanh": F.hardtanh,
            "hardswish": F.hardswish,
            "hardshrink": F.hardshrink,
        }
        self.act = activations[self.layer.get_layer_type()]

    def forward(self, in_data):
        # data = copy.deepcopy(in_data[0]).to(device=self.device)
        data = in_data[0].to(device=self.device)
        data = self.act(data)
        return [data]


@CVT2TORCH.register_module(name="reshape")
class TORCH_RESHAPE(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_RESHAPE, self).__init__(**kwargs)
        self.layer = kwargs["layer"]
        self.attrs = self.layer.get_layer_ops()['attrs'][0]
        self.shape = copy.deepcopy(self.attrs['shape'])

    def forward(self, in_data):
        # data = copy.deepcopy(in_data[0]).to(device=self.device)
        data = in_data[0].to(device=self.device)
        self.shape[0] = data.shape[0]
        data = torch.reshape(data, tuple(self.shape))
        return [data]


@CVT2TORCH.register_module(name="shuffle_only")
class TORCH_SHUFFLE_ONLY(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_SHUFFLE_ONLY, self).__init__(**kwargs)
        self.layer = kwargs["layer"]
        self.attrs = self.layer.get_layer_ops()['attrs']
        self.shape1 = copy.deepcopy(self.attrs[0]['shape'])
        self.perm = copy.deepcopy(self.attrs[1]['perm'])
        self.shape2 = copy.deepcopy(self.attrs[2]['shape'])
        
    def forward(self, in_data):
        data = in_data[0].to(device=self.device)
        self.shape2[0] = self.shape1[0] = data.shape[0]
        data = torch.reshape(data, tuple(self.shape1))
        data = torch.permute(data, tuple(self.perm))
        data = torch.reshape(data, tuple(self.shape2))
        return [data]
    
    
@CVT2TORCH.register_module(name="batchnormalization")
class TORCH_BATCHNORMALIZATION(TORCH_BASE):

    def __init__(self, **kwargs):
        super(TORCH_BATCHNORMALIZATION, self).__init__(**kwargs)
        self.layer = kwargs["layer"]
        self.attrs = self.layer.get_layer_ops()['attrs'][0]
        self.mean, self.var = torch.from_numpy(
            self.attrs['mean']).to(device=self.device), torch.from_numpy(
                self.attrs['var']).to(device=self.device)
        self.bias, self.scale = torch.from_numpy(
            self.attrs['bias']).to(device=self.device), torch.from_numpy(
                self.attrs['scale']).to(device=self.device)
        if 'epsilon' in self.attrs.keys():
            self.epsilon = self.attrs['epsilon']
        else:
            self.epsilon = 1.0e-5

    def forward(self, in_data):
        # data = copy.deepcopy(in_data[0]).to(device=self.device)
        data = in_data[0].to(device=self.device)
        data = F.batch_norm(
            data,
            running_mean=self.mean,
            running_var=self.var,
            weight=self.scale,
            bias=self.bias,
            eps=self.epsilon,
        )
        return [data]
