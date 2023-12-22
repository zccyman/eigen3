import copy
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from utils import Registry, RoundFunction, ClampFunction, FunLSQ
except Exception:
    from onnx_converter.utils import Registry, RoundFunction, ClampFunction, FunLSQ

from tools.cvt2torch import *
from simulator import error_factory

TORCH2QAT: Registry = Registry("torch2qat", scope="")

torch.manual_seed(0)

from torch.autograd import Function
import torch


# class FakeQuantize(Function):

#     @staticmethod
#     def forward(ctx, x, qparam):
#         x = qparam.quantize_tensor(x)
#         x = qparam.dequantize_tensor(x)
#         return x

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None
    
    
# def clamp(data, num_bits=8):
#     min_val = -2 ** (num_bits - 1)
#     max_val = 2 ** (num_bits - 1) - 1
#     data = ClampFunction.apply(
#         data,
#         torch.Tensor([min_val]).to(data.device),
#         torch.Tensor([max_val]).to(data.device),
#     )
#     return data


# def quant(data, scale, zp=0, num_bits=8):
#     data = RoundFunction.apply(data / scale.to(data.device))
#     data = clamp(data, num_bits=num_bits)
#     return data


# def dequant(data, scale, zp=0):
#     data = data * scale.to(data.device)
#     return data


def fake_quant(data, scale, num_bits=8):
    # data = quant(data, scale, num_bits=num_bits)
    # data = dequant(data, scale)
    QN = -2 ** (num_bits - 1)
    QP = 2 ** (num_bits - 1) - 1   
    g = 1.0 / data.numel() # 1 / np.sqrt(data.numel() * QP)
    data = FunLSQ.apply(data, scale, g, QN, QP)
    
    return data


def calc_scale(data, num_bits=8):
    max_val = 2 ** (num_bits - 1) - 1
    return torch.max(torch.abs(data)) / max_val


class EMA(TORCH_BASE):

    def __init__(self, **kwargs):
        super(EMA, self).__init__(**kwargs)

        # if kwargs.get("dmin"):
        #     self.dmin = torch.from_numpy(np.array(kwargs.get("dmin"))).to(self.device_master)
        # else:
        #     self.dmin = None
            
        # if kwargs.get("dmax"):
        #     self.dmax = torch.from_numpy(np.array(kwargs.get("dmax"))).to(self.device_master)
        # else:
        #     self.dmax = None
            
        # dmin = torch.from_numpy(np.array(kwargs.get("dmin")))
        # dmax = torch.from_numpy(np.array(kwargs.get("dmax")))
        dmin = torch.ones(1, dtype=torch.float32) * kwargs.get("dmin")
        dmax = torch.ones(1, dtype=torch.float32) * kwargs.get("dmax")
        dmin = dmin.to(self.device)
        dmax = dmax.to(self.device)
        self.dmin = nn.Parameter(dmin, requires_grad=False)
        self.dmax = nn.Parameter(dmax, requires_grad=False)
        self.reset_ema_param_sucess = True
        # self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
           
        ema_decay = torch.ones(1, dtype=torch.float32) * 0.99 #kwargs.get("ema_decay", 0.99)
        self.ema_decay = nn.Parameter(ema_decay, requires_grad=False)
        self.lock = threading.Lock()
        
    def get_ema_params(self):
        return [self.dmax, self.dmin]
    
    def init_ema_params(self, data):
        if not self.reset_ema_param_sucess:
            self.dmax.data = torch.max(data.reshape(-1))
            self.dmin.data = torch.min(data.reshape(-1))
            self.reset_ema_param_sucess = True
            
    def get_maxminvalue(self, x):
        batch_size = x.shape[0]
        dmin, _ = torch.min(x.reshape(batch_size, -1), dim=1)
        dmax, _ = torch.max(x.reshape(batch_size, -1), dim=1)
        dmin = dmin.to(self.device)
        dmax = dmax.to(self.device)

        return dmin, dmax

    def update(self, dmin, dmax):
        self.ema_decay.data = self.ema_decay.data.to(self.device)

        for i, (min_val, max_val) in enumerate(zip(dmin, dmax)):
            if self.dmin.data is None or self.dmax.data is None:
                self.dmin.data = min_val.to(self.device)
                self.dmax.data = max_val.to(self.device)
            else:
                self.dmin.data = self.ema_decay.data * self.dmin.data + (
                    1 - self.ema_decay.data
                ) * min_val.to(self.device)
                self.dmax.data = self.ema_decay.data * self.dmax.data + (
                    1 - self.ema_decay.data
                ) * max_val.to(self.device)   
                     
    def get_scale(self, num_bits=8):
        max_val = 2 ** (num_bits - 1) - 1
        dmin = torch.abs(self.dmin)
        dmax = torch.abs(self.dmax)
        value = torch.max(dmin, dmax)
        return value / max_val

    def forward(self, data):
        self.lock.acquire()
        try:
            dmin, dmax = self.get_maxminvalue(data)
            self.update(dmin, dmax)
            # min_val, max_val = torch.min(data), torch.max(data)
            # min_val, max_val = torch.median(dmin), torch.median(dmax)
            # self.dmin.data = self.ema_decay.data.to(self.device) * self.dmin.data.to(self.device) + (
            #         1 - self.ema_decay.data.to(self.device)) * min_val.to(self.device)
            # self.dmax.data = self.ema_decay.data.to(self.device) * self.dmax.data.to(self.device) + (
            #         1 - self.ema_decay.data.to(self.device)) * max_val.to(self.device)
            scale = self.get_scale()
        finally:
            self.lock.release()
        return scale


class QAT_BASE(object):

    def __init__(self, **kwargs):
        super(QAT_BASE, self).__init__(**kwargs)
        self.layer = kwargs.get("layer")
        self.ema_si, self.ema_so, self.ema_sc = self.layer.get_ema()
        self.writer = kwargs.get("writer")
        self.process_scale = self.layer.get_scale_type()
        
    #     self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #     self.device = torch.device(self.device)
    #     u_param = torch.tensor(20.0, dtype=torch.float32).to(self.device)
    #     u_param = nn.Parameter(u_param, requires_grad=True)
    #     self.u_params = len(self.ema_so) * [u_param]
    #     self.init_pact_param_sucess = False
        
        self.is_trainning = True
        self.smooth_mode = False
        
        
    def set_train_mode(self, is_trainning):
        self.is_trainning = is_trainning if self.process_scale != 'smooth' else False
                
    # def pact(self, in_data):
    #     if not self.init_pact_param_sucess:
    #         for idx, data in enumerate(in_data):
    #             # self.u_params[idx].data = 0.99 * data.reshape(-1).abs().max()
    #             dmin, dmax = self.ema_so[idx].dmin, self.ema_so[idx].dmax
    #             self.u_params[idx].data = torch.max(torch.abs(dmin), torch.abs(dmax))
    #         self.init_pact_param_sucess = True
            
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         part_a = F.relu(data - self.u_params[idx])
    #         part_b = F.relu(-self.u_params[idx] - data)
    #         in_data[idx] = data - part_a + part_b
        
    #     return in_data
    
    # def get_pact_params(self):
    #     return self.u_params
    
    def pact_before(self, in_data):
        for idx, data in enumerate(in_data):
            dmax, dmin = self.ema_si[idx].dmax, self.ema_si[idx].dmin
            data = data.to(self.device)
            part_a = F.relu(data - dmax)
            part_b = F.relu(dmin - data)
            in_data[idx] = data - part_a + part_b
        
        return in_data
        
    def pact_after(self, in_data):
        for idx, data in enumerate(in_data):
            dmax, dmin = self.ema_so[idx].dmax, self.ema_so[idx].dmin
            data = data.to(self.device)
            part_a = F.relu(data - dmax)
            part_b = F.relu(dmin - data)
            in_data[idx] = data - part_a + part_b
        
        return in_data
            
    def get_params(self):
        return []

    def get_weights(self):
        return None, None

    def draw_histogram(self, iter, feat=dict()):
        layer_idx = self.layer.get_idx()
        layer_idx = str(layer_idx).zfill(4) + "_"
        if feat:
            feat_name = list(feat)[0]
            feat_data = feat[feat_name]
            if feat_data.device.type == "cuda":
                feat_data = feat_data.detach().cpu().numpy()
            else:
                feat_data = feat_data.detach().numpy()
            self.writer.add_histogram("feat/" + layer_idx + feat_name, feat_data, iter)
            feat_min = np.min(feat_data, axis=0).mean()
            self.writer.add_scalar('feat/-min/' + layer_idx + feat_name, -feat_min, iter)  
            feat_max = np.max(feat_data, axis=0).mean()
            self.writer.add_scalar('feat/max/' + layer_idx + feat_name, feat_max, iter) 
            feat_range = np.max(feat_data, axis=0) - np.min(feat_data, axis=0)
            feat_range = feat_range.mean()
            self.writer.add_scalar('feat/range/' + layer_idx + feat_name, feat_range, iter)           
            self.writer.add_text(
                "feat/" + layer_idx + feat_name, '[{:.4f}, {:.4f}]'.format(
                    feat_min,
                    feat_max,
                ), iter)
        else:
            layer_name = self.layer.get_layer_name()
            weight_data = self.get_params()[0].data
            if weight_data.device.type == "cuda":
                weight_data = weight_data.detach().cpu().numpy()
            else:
                weight_data = weight_data.detach().numpy()
            self.writer.add_histogram("weight/" + layer_idx + layer_name, weight_data, iter)
            self.writer.add_scalar('weight/-min/' + layer_idx + layer_name, -weight_data.min(), iter)  
            self.writer.add_scalar('weight/max/' + layer_idx + layer_name, weight_data.max(), iter)  
            self.writer.add_scalar('weight/range/' + layer_idx + layer_name, weight_data.max() - weight_data.min(), iter)           
            self.writer.add_text(
                "weight/" + layer_idx + layer_name, '[{:.4f}, {:.4f}]'.format(
                    weight_data.min(),
                    weight_data.max(),
                ), iter)

    # def quant(self, in_data, num_bits=8):
    #     # QN = -2 ** (num_bits - 1)
    #     # QP = 2 ** (num_bits - 1) - 1         
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         scale = self.ema_si[idx].get_scale().to(self.device)
    #         in_data[idx] = fake_quant(data, scale, num_bits=num_bits)
            
    #     return in_data

    def quant(self, in_data):
        return in_data
    
    def dequant(self, in_data, num_bits=8):
        # QN = -2 ** (num_bits - 1)
        # QP = 2 ** (num_bits - 1) - 1           
        for idx, data in enumerate(in_data):
            data = data.to(self.device)
            if self.is_trainning and not self.smooth_mode: self.ema_so[idx](data) ### update scale
            scale = self.ema_so[idx].get_scale().to(self.device)
            in_data[idx] = fake_quant(data, scale, num_bits=num_bits)
        
        # num_bits = 8
        # for idx, data in enumerate(in_data):
        #     data = data.to(self.device)
        #     max_val = 2 ** (num_bits - 1) - 1
        #     scale = self.u_params[idx].to(self.device) / max_val
        #     in_data[idx] = fake_quant(data, scale)      
          
        return in_data

    def forward(self, in_data, si=1.0, sk=1.0, so=1.0):
        return in_data

    def run(self, in_data):
        # in_data = self.pact_before(in_data)
        in_data = self.quant(in_data)
        in_data = self.forward(in_data)
        in_data = self.dequant(in_data)
        # in_data = self.pact_after(in_data)
        return in_data


@TORCH2QAT.register_module(name="data")
class QAT_PLACEHOLDER(QAT_BASE, TORCH_PLACEHOLDER):

    def __init__(self, **kwargs):
        super(QAT_PLACEHOLDER, self).__init__(**kwargs)

    def quant(self, in_data):
        return in_data
    
    # def dequant(self, in_data):
    #     return in_data

    def forward(self, in_data):
        return TORCH_PLACEHOLDER.forward(self, in_data)


@TORCH2QAT.register_module(name="conv")
@TORCH2QAT.register_module(name="depthwiseconv")
@TORCH2QAT.register_module(name="fc")
class QAT_CONV2D(QAT_BASE, TORCH_CONV2D):

    def __init__(self, **kwargs):
        super(QAT_CONV2D, self).__init__(**kwargs)

    def get_params(self):
        weight = self.weight
        bias = self.bias     
        return [weight, bias]

    def set_sk_params(self, sk_dmax):
        self.sk_dmax.data = sk_dmax
        
    def get_sk_params(self):
        return [self.sk_dmax]
    
    def get_hv_params(self):
        return [self.hv]
       
    def get_alpha_params(self):
        return [self.alpha]
            
    def get_beta_params(self):
        return [self.beta]
    
    def set_broadcast_weights(self, weight, bias):
        self.weight.data = torch.from_numpy(weight)
        self.bias.data = torch.from_numpy(bias)
    
    def get_weights(self):
        weight = self.weight
        bias = self.bias 
        # weight = self.weight_add_offset(self.weight)
        # bias = self.bias_add_offset(self.bias)
        # weight = self.apply_fake_quant(self.weight, is_training=False)
        # bias = self.bias_add_offset(weight, self.bias)
        if weight.device.type == "cuda":
            weight = weight.to(device='cpu')
        if bias.device.type == "cuda":
            bias = bias.to(device='cpu')
        return weight.detach().numpy(), bias.detach().numpy()

    # def quant(self, in_data):
    #     # for idx, data in enumerate(in_data):
    #     #     data = data.to(self.device)
    #     #     scale = self.ema_si[idx].get_scale().to(self.device)
    #     #     in_data[idx] = quant(data, scale)
            
    #     return in_data

    # def dequant(self, in_data):
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         if self.is_tranning: self.ema_so[idx](data) ### update scale
    #     #     scale = self.ema_so[idx].get_scale().to(self.device)
    #     #     in_data[idx] = dequant(data, scale)
        
    #     return in_data
        
    # def quant(self, in_data):
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         with torch.no_grad():
    #             scale = self.ema_si[idx].get_scale()
    #             # dmin, dmax = self.ema_si[idx].get_maxminvalue(data)
    #             # scale = torch.max(torch.abs(dmin), torch.abs(dmax)) / 127.0
    #             # scale = scale.max()
    #             # self.ema_si[idx].scale = scale
    #             # scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #             # scale = scale.expand(data.shape)
    #         in_data[idx] = fake_quant(data, scale)
    #     return in_data

    # def dequant(self, in_data):
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         with torch.no_grad():
    #             scale = self.ema_so[idx].get_scale()
    #             # dmin, dmax = self.ema_so[idx].get_maxminvalue(data)
    #             # scale = torch.max(torch.abs(dmin), torch.abs(dmax)) / 127.0
    #             # scale = scale.max()
    #             # self.ema_so[idx].scale = scale                
    #         in_data[idx] = fake_quant(data, scale)
    #     return in_data

    def forward(self, in_data):
        si = self.ema_si[0].get_scale().to(self.device)
        so = self.ema_so[0].get_scale().to(self.device)
        with torch.no_grad():
            sk = calc_scale(self.weight).to(self.device)
        # return TORCH_CONV2D.forward(self, in_data, sk=sk, si=si, so=so)
        return TORCH_CONV2D.forward(self, in_data, sk=sk, si=si, so=so, fake_quant=fake_quant, is_trainning=self.is_trainning)


@TORCH2QAT.register_module(name="add")
class QAT_ADD(QAT_BASE, TORCH_ADD):

    def __init__(self, **kwargs):
        super(QAT_ADD, self).__init__(**kwargs)
        self.int_scale = 7
        
    def update_scale(self, out_scale):
        max_value = (2 ** self.int_scale) - 1 
        for int_scale in range(self.int_scale, 0, -1):
            scale = RoundFunction.apply(out_scale * (2 ** int_scale))
            if scale <= max_value:
                break
        return scale, -int_scale
       
    # def get_scale_shift(self):
    #     return dict(scale=self.scale, shift=self.shift) 
    
    # def quant(self, in_data):
    #     self.scale, self.shift = [], []
    #     so = self.ema_so[0].get_scale().to(self.device)
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         si = self.ema_si[idx].get_scale().to(self.device)
    #         data = quant(data, si)

    #         scale, shift = self.update_scale(si / so)
    #         self.scale.append(scale)
    #         self.shift.append(shift)
            
    #         data = clamp(data * scale, num_bits=16) ### int8 * int8, int16
    #         data = clamp(
    #             RoundFunction.apply(data * (2 ** shift)), 
    #             num_bits=16,
    #         ) ### int16

    #         in_data[idx] = data
            
    #     return in_data

    # def dequant(self, in_data):
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         if self.is_tranning: self.ema_so[idx](data) ### update scale
    #         scale = self.ema_so[idx].get_scale().to(self.device)
    #         data = clamp(data, num_bits=8) ### int8
    #         in_data[idx] = dequant(data, scale)
                  
    #     return in_data
    
    def forward(self, in_data):
        return TORCH_ADD.forward(self, in_data)


@TORCH2QAT.register_module(name="concat")
class QAT_CONCAT(QAT_ADD, TORCH_CONCAT):

    def __init__(self, **kwargs):
        super(QAT_CONCAT, self).__init__(**kwargs)
        
    def forward(self, in_data):
        return TORCH_CONCAT.forward(self, in_data)
    
    
@TORCH2QAT.register_module(name="split")
class QAT_SPLIT(QAT_ADD, TORCH_SPLIT):

    def __init__(self, **kwargs):
        super(QAT_SPLIT, self).__init__(**kwargs)
        
    # def quant(self, in_data):
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         scale = self.ema_si[idx].get_scale().to(self.device)
    #         in_data[idx] = quant(data, scale)
                  
    #     return in_data
            
    # def dequant(self, in_data):
    #     self.scale, self.shift = [], []
    #     si = self.ema_si[0].get_scale().to(self.device)
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         if self.is_tranning: self.ema_so[idx](data) ### update scale
    #         so = self.ema_so[idx].get_scale().to(self.device)
            
    #         scale, shift = self.update_scale(si / so)
    #         self.scale.append(scale)
    #         self.shift.append(shift)
                        
    #         data = clamp(data * scale, num_bits=16) ### int8 * int8, int16
    #         data = clamp(
    #             RoundFunction.apply(data * (2 ** shift)), 
    #             num_bits=8,
    #         ) ### int8
    #         data = dequant(data, so)
            
    #         in_data[idx] = data
            
    #     return in_data

    def forward(self, in_data):
        return TORCH_SPLIT.forward(self, in_data)
    
        
@TORCH2QAT.register_module(name="resize")
class QAT_RESIZE(QAT_BASE, TORCH_RESIZE):

    def __init__(self, **kwargs):
        super(QAT_RESIZE, self).__init__(**kwargs)

    def forward(self, in_data):
        return TORCH_RESIZE.forward(self, in_data)


@TORCH2QAT.register_module(name="maxpool")
class QAT_MAXPOOL(QAT_BASE, TORCH_MAXPOOL):

    def __init__(self, **kwargs):
        super(QAT_MAXPOOL, self).__init__(**kwargs)
        self.smooth_mode = True
        
    def forward(self, in_data):
        return TORCH_MAXPOOL.forward(self, in_data)


@TORCH2QAT.register_module(name="relu")
@TORCH2QAT.register_module(name="swish")
class QAT_ACTIVATION(QAT_BASE, TORCH_ACTIVATION):

    def __init__(self, **kwargs):
        super(QAT_ACTIVATION, self).__init__(**kwargs)

    def forward(self, in_data):
        return TORCH_ACTIVATION.forward(self, in_data)


@TORCH2QAT.register_module(name="reshape")
class QAT_RESHAPE(QAT_BASE, TORCH_RESHAPE):

    def __init__(self, **kwargs):
        super(QAT_RESHAPE, self).__init__(**kwargs)
        self.smooth_mode = True
        
    def forward(self, in_data):
        return TORCH_RESHAPE.forward(self, in_data)
    
@TORCH2QAT.register_module(name="shuffle_only")
class QAT_SHUFFLE_ONLY(QAT_BASE, TORCH_SHUFFLE_ONLY):

    def __init__(self, **kwargs):
        super(QAT_SHUFFLE_ONLY, self).__init__(**kwargs)
        self.smooth_mode = True
        
    def forward(self, in_data):
        return TORCH_SHUFFLE_ONLY.forward(self, in_data)    
    
    
@TORCH2QAT.register_module(name="globalaveragepool")
class QAT_AVERAGEPOOL(QAT_BASE, TORCH_AVERAGEPOOL):

    def __init__(self, **kwargs):
        super(QAT_AVERAGEPOOL, self).__init__(**kwargs)
        self.smooth_mode = False
        
    def forward(self, in_data):
        return TORCH_AVERAGEPOOL.forward(self, in_data)    
    

@TORCH2QAT.register_module(name="mul")
class QAT_MUL(QAT_BASE, TORCH_MUL):

    def __init__(self, **kwargs):
        super(QAT_MUL, self).__init__(**kwargs)
        self.int_scale = 8
        
    def update_scale(self, scale, bit=32, lower=0.5):
        for shift in range(-bit, bit):
            out_scale = scale * (2**(-shift))
            if lower < out_scale < 1:
                return np.int32(shift), out_scale
        return np.int32(0), scale
    
    # def quant(self, in_data):
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         scale = self.ema_si[idx].get_scale().to(self.device)
    #         in_data[idx] = quant(data, scale)
                  
    #     return in_data
            
    # def dequant(self, in_data):
    #     self.scale, self.shift = [], []
    #     si = self.ema_si[0].get_scale().to(self.device)
    #     sk = self.ema_si[1].get_scale().to(self.device)
    #     for idx, data in enumerate(in_data):
    #         data = data.to(self.device)
    #         if self.is_tranning: self.ema_so[idx](data) ### update scale
    #         so = self.ema_so[idx].get_scale().to(self.device)
    #         scale = si * sk / so
            
    #         out_scale, out_shift = self.update_scale(scale)
    #         out_scale = torch.tensor(out_scale * (2 ** self.int_scale), dtype=torch.float32).to(self.device)
    #         out_scale = RoundFunction.apply(out_scale)
    #         out_scale = ClampFunction.apply(
    #             out_scale, 
    #             torch.Tensor([0.0]).to(self.device),
    #             torch.Tensor([(2 ** self.int_scale) - 1]).to(self.device), ### uint8
    #         )
    #         self.scale.append(out_scale)
    #         self.shift.append(out_shift)
                        
    #         data = RoundFunction.apply(data * (2 ** out_shift))
    #         data = clamp(data, num_bits=8)
            
    #         data = RoundFunction.apply(data * out_scale)
    #         data = clamp(data, num_bits=16)
            
    #         data = clamp(
    #             RoundFunction.apply(data * (2 ** (-self.int_scale))), 
    #             num_bits=8,
    #         ) ### int8
    #         data = dequant(data, so)
            
    #         in_data[idx] = data
            
    #     return in_data
        
    def forward(self, in_data):
        return TORCH_MUL.forward(self, in_data)    