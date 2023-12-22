import copy
import os
import pickle
import threading
import cv2
import numpy as np
import onnx
import torch
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import shutil
import random
import json
    
from .cvt2torch import CVT2TORCH
from .torch2qat import TORCH2QAT, EMA, fake_quant

import torch.nn.functional as F

try:
    from utils import Registry, RoundFunction, FloorFunction, ClampFunction, FunLSQ
except Exception:
    from onnx_converter.utils import Registry, RoundFunction, FloorFunction, ClampFunction, FunLSQ
    
    
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# import albumentations as A
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.GaussNoise(p=0.2),
#     A.Blur(blur_limit=3, p=0.1),
#     A.RandomBrightnessContrast(p=0.2),
#     A.HueSaturationValue(p=0.2),
# ])
    
def is_subset(a, b):
    return all(elem in b for elem in a)

    
def batchify_list(lst, batch_size, is_trainning=True):
    new_list = [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

    if len(new_list[-1]) < batch_size and len(new_list) > 1:
        num_to_add = batch_size - len(new_list[-1])
        if is_trainning:
            new_list[-1] = new_list[-1] + lst[0:num_to_add]
    
    # new_list = new_list[:-1]
        
    return new_list


def norm_feat(featuremap, max_val=None, min_val=None): 
    batch_size = featuremap.shape[0] 
    dims = len(featuremap.shape)
    if min_val is None:  
        min_val = torch.min(featuremap.reshape(batch_size, -1), dim=-1)[0]
        for _ in range(dims):
            min_val  = min_val.unsqueeze(-1)
    if max_val is None:
        max_val = torch.max(featuremap.reshape(batch_size, -1), dim=-1)[0]
        for _ in range(dims):
            max_val  = max_val.unsqueeze(-1)
    normalized_featuremap = (featuremap - min_val) / (max_val - min_val)
    return normalized_featuremap, max_val, min_val
    
    
class FeatureMapLoss(nn.Module):

    def __init__(self, loss_type='gram', hard_ratio=0.5):
        super(FeatureMapLoss, self).__init__()
        self.loss_type = loss_type
        self.hard_ratio = hard_ratio
        
    def forward(self, input_feature, target_feature):
        if self.loss_type in ['mse', 'l1']:
            target_feature, max_val, min_val = norm_feat(target_feature)
            input_feature, _, _ = norm_feat(input_feature, max_val, min_val)            
            # 计算均方误差损失
            if self.loss_type == 'mse':
                mse_loss = F.mse_loss(input_feature, target_feature, reduction='none')
            else:
                mse_loss = F.l1_loss(input_feature, target_feature, reduction='none')
                
            if self.hard_ratio == 1.0:
                return torch.mean(mse_loss)
            
            # 计算每个样本的损失值
            sample_losses = torch.mean(mse_loss, dim=(1, 2, 3))  # 在通道维度上求平均
            
            # 对损失值进行排序
            sorted_losses, sorted_indices = torch.sort(sample_losses, descending=True)
            
            # 选择困难样本
            num_hard_samples = int(self.hard_ratio * len(sorted_indices))
            hard_indices = sorted_indices[:num_hard_samples]
            
            # 使用困难样本计算损失
            hard_loss = torch.mean(mse_loss[hard_indices])
                
            return hard_loss
        elif self.loss_type == 'l1w':
            l1_loss = F.l1_loss(input_feature, target_feature, reduction='none')
            if self.hard_ratio == 1.0:
                return torch.mean(l1_loss)
                        
            sample_losses = l1_loss.reshape(-1)
            sorted_losses, sorted_indices = torch.sort(sample_losses, descending=True)

            num_hard_samples = int(self.hard_ratio * len(sorted_indices))
            hard_indices = sorted_indices[:num_hard_samples]

            hard_loss = torch.mean(sample_losses[hard_indices])

            return hard_loss            
        elif self.loss_type == 'ssim':
            return 1 - torchvision.transforms.functional.ssim(
                input_feature, target_feature)
        elif self.loss_type == 'cosine':
            return 1 - F.cosine_similarity(
                input_feature.view(input_feature.size(0), -1),
                target_feature.view(target_feature.size(0), -1),
            ).mean()
        elif self.loss_type == 'mse_cosine':
            a = F.mse_loss(input_feature, target_feature)
            b = 1 - F.cosine_similarity(
                input_feature.view(input_feature.size(0), -1),
                target_feature.view(target_feature.size(0), -1),
            ).mean()
            return (a + b) / 2.0
        elif self.loss_type == 'mse_l1':
            a = F.mse_loss(input_feature, target_feature)
            b = F.l1_loss(input_feature, target_feature)
            return (a + b) / 2.0
        elif self.loss_type == 'mse_l1_cosine':
            a = F.mse_loss(input_feature, target_feature)
            b = F.l1_loss(input_feature, target_feature)
            c = 1 - F.cosine_similarity(
                input_feature.view(input_feature.size(0), -1),
                target_feature.view(target_feature.size(0), -1),
            ).mean()
            return (a + b + c) / 3.0
        else:
            raise NotImplementedError(
                'Unsupported feature map loss type: {}'.format(self.loss_type))


class QAT_Module(nn.Module):
    def __init__(self, **kwargs):
        super(QAT_Module, self).__init__()
           
        # self.tensorboard_dir = kwargs.get('tensorboard_dir')
        # if os.path.exists(self.tensorboard_dir):
        #     shutil.rmtree(self.tensorboard_dir)
        # self.writer = SummaryWriter(self.tensorboard_dir)   
        self.logger = kwargs["logger"]
        self.rank = kwargs.get('rank')        
        self.layers = kwargs.get('layers')
        self.max_learning_rate = kwargs.get("max_learning_rate")
        self.min_learning_rate = kwargs.get("min_learning_rate")
        ema_learning_rate = self.max_learning_rate
        weight_decay = kwargs.get("weight_decay") #0.0001
        self.preprocess = kwargs.get("preprocess")
        self.postprocess = kwargs.get("postprocess")
        self.post_quan = kwargs.get("post_quan")
        self.update_so_by_ema = kwargs.get("update_so_by_ema")
        self.already_debug_one_time = False
        
        self.sk_params_file = kwargs.get("sk_params_file")
        self.calibration_params_file = kwargs.get("calibration_params_file")
        if self.sk_params_file:
            sk_params = json.load(open(self.sk_params_file, "r"))
            self.logger.info("reload sk_params")
        else:
            sk_params = None
        if self.calibration_params_file:
            calibration_params = json.load(open(self.calibration_params_file, "r"))
            self.logger.info("reload calibration_params")
        else:
            calibration_params = None
                                
        swish = lambda x: x * torch.sigmoid(x)
        act = lambda x: x
        activations = {
            "act": act,
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
          
        layer_num, layer_idx = -1, -1
        params = []
        for layer in self.layers:
            layer_type = layer.get_layer_type()
            layer_name = layer.get_layer_name()
            process_scale = layer.get_scale_type()
            self.logger.info("rank {} initializing layer: {}, {}".format(self.rank, layer_type, layer_name))
                
            instance_devices, device_id = [], 0
            device_string = "cpu"
            activation = layer.get_layer_ops()["ops"][-1]
            ema_sc = []
            if layer_type in ["conv", "depthwiseconv", "convtranspose", "fc"]:
                if activation != "act":
                    if calibration_params:
                        conv_output_name = layer.get_nodes()[0].get_onnx_output()[0]
                        dmax = calibration_params[conv_output_name]["max"] #/ np.sqrt(127.0)
                        dmin = calibration_params[conv_output_name]["min"] #/ np.sqrt(127.0)
                    else:                            
                        dmax = layer.get_data_scale()[1]['max'].astype(np.float32) #/ np.sqrt(127.0)
                        dmin = layer.get_data_scale()[1]['min'].astype(np.float32) #/ np.sqrt(127.0)
                    ema_sc = [EMA(dmax=dmax, dmin=dmin, device=device_string)]  
            ema_so = []
            for idx, onnx_output_name in enumerate(layer.get_onnx_output_name()):
                if calibration_params:
                    dmax = calibration_params[onnx_output_name]["max"] #/ np.sqrt(127.0)
                    dmin = calibration_params[onnx_output_name]["min"] #/ np.sqrt(127.0)
                else:                      
                    dmax = layer.get_data_scale()[idx]['max'].astype(np.float32) #/ np.sqrt(127.0)
                    dmin = layer.get_data_scale()[idx]['min'].astype(np.float32) #/ np.sqrt(127.0)
                ema_so.extend([EMA(dmax=dmax, dmin=dmin, device=device_string)])
                                        
            if layer_type == "data":
                ema_si = ema_so
            else:
                # if layer.get_layer_name() in ["/backbone/stage2/stage2.1/branch2/branch2.0/Conv"]:
                    # print("test")
                ema_si = []
                prelayers = [
                    self.layers[layer_idx]
                    for layer_idx in layer.get_input_idx()
                ]
                for in_name in layer.get_onnx_input_name():
                    for prelayer in prelayers:
                        prelayer_out_names = prelayer.get_onnx_output_name()
                        if in_name in prelayer_out_names:
                            idx_selected = prelayer_out_names.index(in_name)
                            ema_si_tmp = prelayer.get_ema()[1][idx_selected]
                            ema_si.append(ema_si_tmp)
                            
            if process_scale in ["smooth"] and layer_type not in ["globalaveragepool"]:
                ema_so = ema_si
            else:
                layer_num += 1
                layer_idx += 1                
                for instance in ema_so:
                    for j, param in enumerate(instance.get_ema_params()):
                        params.append({
                            "params": param,
                            "lr": ema_learning_rate,
                            "weight_decay": weight_decay,
                            "name": "so",
                            "layer_num": layer_num,
                            "layer_idx": layer_idx,
                            "layer_name": layer_name,
                            "layer_type": layer_type, 
                            "process_scale": process_scale,
                            "is_result_layer": layer.get_is_result_layer(),                                   
                        })                     
            if len(ema_sc) == 0:
                ema_sc = ema_so    
            else:
                for j, param in enumerate([ema_sc[0].dmin, ema_sc[0].dmax]):
                    params.append({
                        "params": param,
                        "lr": ema_learning_rate,
                        "weight_decay": weight_decay,
                        "name": "sc",
                        "layer_num": layer_num,
                        "layer_idx": layer_idx,
                        "layer_name": layer_name,
                        "layer_type": layer_type,
                        "process_scale": process_scale,
                        "is_result_layer": layer.get_is_result_layer(),
                    })                                     
            layer.set_ema(ema=[ema_si, ema_so, ema_sc])
            
            torch_instance = CVT2TORCH.get(layer_type)(
                layer=layer,
                requires_grad=False,
                qat_nofakequant=False,
                device=device_string,
            )
            qat_instance = TORCH2QAT.get(layer_type)(
                layer=layer,
                requires_grad=True,
                qat_nofakequant=False,
                device=device_string,
            )
            if layer_type in [
                    "conv",
                    "depthwiseconv",
                    "convtranspose",
                    "fc",
                    "gemm",
                    "matmul",
            ]:
                if 0 == device_id:
                    param = qat_instance.get_params()[:1]
                    for j, p in enumerate(param):
                        params.append({
                            "params": p,
                            "lr": self.max_learning_rate,
                            "weight_decay": weight_decay,
                            "name": "weight",
                            "layer_num": layer_num,
                            "layer_idx": layer_idx,
                            "layer_name": layer_name,
                            "layer_type": layer_type, 
                            "process_scale": process_scale,
                            "is_result_layer": layer.get_is_result_layer(),
                        }
                    )
                    
                    param = qat_instance.get_params()[1:]
                    for j, p in enumerate(param):
                        params.append({
                            "params": p,
                            "lr": self.max_learning_rate,
                            "weight_decay": weight_decay,
                            "name": "bias",
                            "layer_num": layer_num,
                            "layer_idx": layer_idx,
                            "layer_name": layer_name,
                            "layer_type": layer_type, 
                            "process_scale": process_scale,
                            "is_result_layer": layer.get_is_result_layer(),                                
                        }
                    )
                        
                    param = qat_instance.get_alpha_params()
                    for j, p in enumerate(param):
                        params.append({
                            "params": p,
                            "lr": ema_learning_rate,
                            "weight_decay": weight_decay,
                            "name": "alpha",
                            "layer_num": layer_num,
                            "layer_idx": layer_idx,
                            "layer_name": layer_name,
                            "layer_type": layer_type,  
                            "process_scale": process_scale,
                            "is_result_layer": layer.get_is_result_layer(),                               
                        }
                    )
                    
                    param = qat_instance.get_beta_params()
                    for j, p in enumerate(param):
                        params.append({
                            "params": p,
                            "lr": ema_learning_rate,
                            "weight_decay": weight_decay,
                            "name": "beta",
                            "layer_num": layer_num,
                            "layer_idx": layer_idx,
                            "layer_name": layer_name,
                            "layer_type": layer_type,  
                            "process_scale": process_scale,
                            "is_result_layer": layer.get_is_result_layer(),                               
                        }
                    )
                                                                            
                    param = qat_instance.get_hv_params()
                    for j, p in enumerate(param):
                        params.append({
                            "params": p,
                            "lr": ema_learning_rate,
                            "weight_decay": 0.0,
                            "name": "hv",
                            "layer_num": layer_num,
                            "layer_idx": layer_idx,
                            "layer_name": layer_name,
                            "layer_type": layer_type,  
                            "process_scale": process_scale,
                            "is_result_layer": layer.get_is_result_layer(),                               
                        }
                    )
                    
                    if sk_params:
                        sk_dmax = sk_params.get(layer_name)
                        sk_dmax = torch.tensor(sk_dmax)
                        qat_instance.set_sk_params(sk_dmax)                        
                    param = qat_instance.get_sk_params()
                    for j, p in enumerate(param):
                        params.append({
                            "params": p,
                            "lr": ema_learning_rate,
                            "weight_decay": weight_decay,
                            "name": "sk",
                            "layer_num": layer_num,
                            "layer_idx": layer_idx,
                            "layer_name": layer_name,
                            "layer_type": layer_type, 
                            "process_scale": process_scale,
                            "is_result_layer": layer.get_is_result_layer(),                               
                        }
                    )

            instance = [torch_instance, qat_instance]
            instance_devices.extend(instance)
            layer.set_adaround(instance_devices)

        self.qat_params = params
                    
    def forward(self, in_data_per_gpu, device, is_trainning=True):
        if is_trainning:
            hard_ratio = 1.0
        else:
            hard_ratio = 1.0
            
        # out_data_torch = out_data_qat = in_data_per_gpu
        out_data_torch = copy.deepcopy(in_data_per_gpu)
        out_data_torch = torch.from_numpy(out_data_torch).to(device)
        out_data_qat = copy.deepcopy(in_data_per_gpu)
        out_data_qat = torch.from_numpy(out_data_qat).to(device)
                
        input_name = []
        for layer in self.layers:
            if layer.get_layer_type() == "data":
                input_name = layer.get_onnx_output_name()

        torch_fearuremap = {input_name[0]: out_data_torch}
        qat_fearuremap = {input_name[0]: out_data_qat}
            
        is_debug = False
        if out_data_torch.shape[0] == 1 and not self.already_debug_one_time:
            is_debug = True
        if is_debug:    
            ort_in_data = out_data_torch.cpu().detach().numpy()[0].transpose(1, 2, 0)
            self.post_quan.set_transform(transform=None)
            true_outputs = self.post_quan.onnx_infer(ort_in_data)
            self.post_quan.set_transform(transform=self.preprocess)
        
        txme_clip_dict = dict()
        txme_clip_list = []
        txme_clip_scale_list = []
        result_layer_loss, middle_layer_loss, loss_w_l1, loss_hv_regular = 0.0, 0.0, 0.0, 0.0
        middle_layer_num = 0.0
        result_layer_num = 0.0
        weight_layer_num = 0.0
        for layer in self.layers:
            layer_name = layer.get_layer_name()
            layer_type = layer.get_layer_type()
            
            torch_instance, qat_instance = layer.get_adaround(
            )
            torch_instance.set_device(device)
            qat_instance.set_device(device)
            if self.update_so_by_ema:
                qat_instance.set_train_mode(is_trainning=is_trainning)
            else:
                qat_instance.set_train_mode(is_trainning=False)
            
            if layer_type == "data":
                layer_input_names = layer.get_onnx_output_name()
            else:
                layer_input_names = layer.get_onnx_input_name()
            layer_output_names = layer.get_onnx_output_name()

            in_data_torch = []
            for input_name in layer_input_names:
                in_data_torch.append(torch_fearuremap[input_name])
                
            with torch.no_grad():
                out_data_torch = torch_instance(in_data_torch)
            for idx, output_name in enumerate(layer_output_names):
                torch_fearuremap[output_name] = out_data_torch[idx]

            in_data_qat = []
            for idx, input_name in enumerate(layer_input_names):
                # if is_trainning:
                #    qat_instance.ema_si[idx](qat_fearuremap[input_name]) 
                in_data_qat.append(qat_fearuremap[input_name])
            out_data_qat = qat_instance.run(in_data_qat)
            for idx, output_name in enumerate(layer_output_names):
                qat_fearuremap[output_name] = out_data_qat[idx]
                # if is_trainning:
                    # qat_instance.ema_so[idx](qat_fearuremap[output_name].detach())
                if is_debug:
                    self.logger.info("debug: {}, {}, {}".format(output_name, torch_fearuremap[output_name].sum().item(), true_outputs[output_name].sum().item()))

                # middle_layer_loss += F.mse_loss(
                #     layer.get_ema()[1][idx].get_scale(), 
                #     (2 * torch.mean(torch.abs(qat_fearuremap[output_name])) / 127.0).detach(),
                #     reduction='mean',
                # )
                if layer.get_is_result_layer():
                    # loss_fn = FeatureMapLoss(loss_type='mse', hard_ratio=hard_ratio)
                    loss_weight = 1
                    result_layer_loss += loss_weight * F.mse_loss(
                        qat_fearuremap[output_name],
                        torch_fearuremap[output_name].detach(),
                        reduction="mean",
                    )
                    result_layer_num += 1.0  
                else:
                # elif is_trainning:
                #     loss_weight = 1
                #     loss_fn = FeatureMapLoss(loss_type='l1', hard_ratio=hard_ratio)
                #     middle_layer_loss += loss_weight * loss_fn(
                #         qat_fearuremap[output_name],
                #         torch_fearuremap[output_name],
                #     )
                    middle_layer_num += 1.0
                
            #### not layer.get_is_result_layer() and 
            if layer_type in ["conv", "depthwiseconv", "convtranspose", "fc"]:
                # txme_clip = qat_instance.get_txme_clip()
                # if layer_name in txme_clip_dict.keys():
                #     txme_clip_dict[layer_name] += txme_clip
                # else:
                #     txme_clip_dict[layer_name] = txme_clip
                # txme_clip_list.append(txme_clip)
                weight_layer_num += 1.0
                
                # txme_clip_scale = qat_instance.get_txme_scale()["scale"]
                # txme_clip_scale_list.append(txme_clip_scale)
                
                device = qat_instance.get_device()
                # param = qat_instance.get_hv_params()[0].to(device)
                # param = ClampFunction.apply(
                #         param, 
                #         torch.Tensor([-1.0]).to(device), 
                #         torch.Tensor([1.0]).to(device),
                # )
                # loss_hv_regular += (-4.0 * param.abs() * (param.abs() - 1)).mean()
                # loss_hv_regular += (-4.0 * param * (param - 1) * (param >= 0) - 4.0 * param * (param + 1) * (param < 0)).mean()
                # loss_hv_regular += (16.0 / 3.0) * ((param ** 2) - (param ** 4)).mean()
                # loss_hv_regular += (-4 * param * param + 4 * param).mean()
                # loss_hv_regular += (-1.0 * param * param + 1.0).mean()
                # loss_hv_regular += ((param ** 2 - 1) ** 2).mean()
                # loss_hv_regular += torch.argmax(F.softmax(param, dim=-1), dim=-1).max()
                # loss_hv_regular += torch.exp(param).max()
                
                # weight = torch.add(qat_instance.get_params()[0].to(device), qat_instance.get_alpha_params()[0].to(device))
                weight = qat_instance.get_params()[0].to(device)
                # sk = qat_instance.get_sk_params()[0] / 127.0
                # qweight = fake_quant(weight, scale=sk, num_bits=8)
                # qweight = qat_instance.apply_fake_quant(weight, is_training=False)
                # sk = qat_instance.get_sk_params()[0] / 127.0
                # qw = quant(weight, scale=sk, num_bits=8)   
                # qw = torch.round(weight / sk).clamp(-128, 127)             
                # loss_fn_l1 = FeatureMapLoss(loss_type='l1w', hard_ratio=hard_ratio)
                # loss_var = 0.0 
                # for v in range(-128, 128):
                #     selected_indices = torch.where(qw == v)
                #     v_selected = weight[selected_indices]
                #     v_selected_q = qweight[selected_indices]                    
                #     if v_selected.numel() > 1:
                #         loss_var += v_selected.var() ### (v-v_mean)**2 / N
                #         loss_var += F.mse_loss(v_selected_q, v_selected, reduction='mean').mean()
                # loss_w_l1 += loss_var / 256.0
                # loss_w_l1 += F.mse_loss(
                #     torch.max(torch.abs(weight)) / 127.0, 
                #     (2 * torch.mean(torch.abs(weight)) / 127.0).detach(),
                #     reduction='mean')
                # loss_w_l1 += F.mse_loss(
                #     qat_instance.get_sk_params()[0].to(device) / 127.0, 
                #     (2 * torch.mean(torch.abs(weight)) / 127.0).detach(), 
                #     reduction='mean')
                # loss_w_l1 += F.mse_loss(qweight, weight, reduction='mean')
                loss_w_l1 += (weight ** 2).mean()
                bias = qat_instance.get_params()[1].to(device)
                loss_w_l1 += (bias ** 2).mean()
              
        result_layer_loss /= result_layer_num  
        # middle_layer_loss /= (middle_layer_num + result_layer_num)
                            
        # if is_trainning:
        #     middle_layer_loss /= middle_layer_num
        # else:
        middle_layer_loss = 0.0 * result_layer_loss
        
        # middle_layer_loss = 0.0 * result_layer_loss
        # txme_clip = 0.0 * result_layer_loss
        # loss_w_l1 = 0.0 * result_layer_loss
        loss_hv_regular = 0.0 * result_layer_loss
        
        loss_w_l1 = 1.0 * loss_w_l1 / weight_layer_num
        # loss_hv_regular = 1.0 * loss_hv_regular / weight_layer_num
        # txme_clip /= 1.0 * weight_layer_num 
        
        txme_clip_list_tmp, txme_clip_scale_list_tmp = [], []
        for a, b in zip(txme_clip_list, txme_clip_scale_list):
            if a > 0.0:
                txme_clip_list_tmp.append(a)
                txme_clip_scale_list_tmp.append(b)
        txme_clip = torch.tensor(0) # torch.stack(txme_clip_list_tmp).mean()    
        txme_clip_scale = torch.tensor(0) # torch.stack(txme_clip_scale_list_tmp).mean()
        
        if is_debug:
            self.already_debug_one_time = True
            
        return result_layer_loss, middle_layer_loss, loss_w_l1, loss_hv_regular, txme_clip, txme_clip_scale, txme_clip_dict
    
                
def save_onnx_fp(
    layers, model_input_names, 
    model_output_names, 
    export_model_path, 
    wo_method, opset_version=15,
):
    outputs = [layer.export_onnx_fp(is_vis_qparams=False) for layer in layers]
    nodes, initializers = [], []
    for idx, out in enumerate(outputs):
        if None in out[0]:
            print(layers[idx].get_layer_name())
        nodes.extend(out[0])
        initializers.extend(out[1])
    create_in_node = lambda name, shape: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) # type: ignore
    create_out_node = lambda name: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, []) # type: ignore
    fisrt_layer = layers[0]
    inputs = [create_in_node(name, fisrt_layer.get_layer_ops()["attrs"][0]['shape']) for name in model_input_names]
    outputs = [create_out_node(name) for name in model_output_names]
    graph = onnx.helper.make_graph(nodes=nodes, name="speedup", # type: ignore
                                    inputs=inputs,
                                    outputs=outputs,
                                    initializer=initializers)
    model = onnx.helper.make_model( # type: ignore
        graph, opset_imports=[onnx.helper.make_opsetid("", opset_version)]) # type: ignore
    model.ir_version = 8
    onnx.save_model(model, export_model_path.replace(".onnx", wo_method))
    return export_model_path.replace(".onnx", wo_method)
                
def save_model(
    layers, model_input_names, 
    model_output_names, 
    export_model_path, 
    epochs, epoch=0,
    txme_clip_dict=None,
    is_save_onnx=True):
    layer_types_with_weights = [
                "conv",
                "depthwiseconv",
                "convtranspose",
                "fc",
                "gemm",
                "matmul",
            ]

    calibration_dmaxdmin = dict()
    sk_params = dict()
    txme_params = dict()
    for layer in tqdm(layers, postfix='export layers'):
        torch_instance, qat_instance = layer.get_adaround(
        )        
        ema_si, ema_so, ema_sc = layer.get_ema()
        layer_name = layer.get_layer_name()
        layer_type = layer.get_layer_type()
        txme_param = dict()
        txme_param["layer_type"] = layer_type
        txme_param["process_scale"] = layer.get_scale_type()
        txme_param["layer_idx"] = layer.get_idx()
        txme_param["input_idx"] = layer.get_input_idx()
        txme_param["input_name"] = layer.get_input_name()
        txme_param["output_idx"] = layer.get_output_idx()
        txme_param["output_name"] = layer.get_output_name()        
        if layer_type in layer_types_with_weights:
            weight, bias = qat_instance.get_weights()
            layer.set_layer_ops(dict(weights=[weight, bias]))
            sk_params[layer_name] = torch.abs(qat_instance.get_sk_params()[0]).detach().cpu().numpy().tolist()
            
            # txme_param["alpha"] = qat_instance.get_alpha_params()[0].max().item()
            # txme_param["beta"] = qat_instance.get_beta_params()[0].max().item()
            # txme_param["txme_clip"] = txme_clip_dict[layer_name].item() #qat_instance.get_txme_clip().item()
            txme_param["si"] = ema_si[0].get_scale().item() * 127.0
            txme_param["sk"] = sk_params[layer_name]
            txme_param["sc"] = ema_sc[0].get_scale().item() * 127.0
            txme_param["so"] = ema_so[0].get_scale().item() * 127.0
            # txme_param["out_shift"] = qat_instance.get_txme_scale()["out_shift"]
            # txme_param["out_scale"] = qat_instance.get_txme_scale()["out_scale"]
            txme_params[layer_name] = txme_param
        elif layer_type in ["add", "concat", "split"]:
            txme_param["si"] = [s.get_scale().item() * 127.0 for s in ema_si]
            txme_param["so"] = [s.get_scale().item() * 127.0 for s in ema_so]             
            # txme_param["scale"] = [v.item() for v in qat_instance.get_scale_shift()["scale"]]
            # txme_param["int_scale"] = [v for v in qat_instance.get_scale_shift()["shift"]]
            txme_params[layer_name] = txme_param
        else:
            txme_param["si"] = [s.get_scale().item() * 127.0 for s in ema_si]
            txme_param["so"] = [s.get_scale().item() * 127.0 for s in ema_so]            
            txme_params[layer_name] = txme_param
                                  
        if len(ema_sc) > 0:
            conv_output_name = layer.get_nodes()[0].get_onnx_output()[0]
            dmax = ema_sc[0].dmax.detach().cpu().numpy().tolist()[0]
            dmin = ema_sc[0].dmin.detach().cpu().numpy().tolist()[0]
            calibration_dmaxdmin[conv_output_name] = \
                dict(max=dmax, min=dmin) 
                            
        for idx, instance in enumerate(ema_so):
            onnx_output_names = layer.get_onnx_output_name()
            dmax, dmin = instance.get_ema_params()
            dmax = dmax.detach().cpu().numpy().tolist()[0]
            dmin = dmin.detach().cpu().numpy().tolist()[0]
            calibration_dmaxdmin[onnx_output_names[idx]] = \
                dict(max=dmax, min=dmin)                    
             
    qat_save_path = os.path.split(export_model_path)[0]
    if not os.path.exists(qat_save_path):
        os.makedirs(qat_save_path)
    calibration_json_path = f"{qat_save_path}/calibration_{epoch}.json"  
    sk_params_json_path = f"{qat_save_path}/sk_params_{epoch}.json"
    txme_params_json_path = f"{qat_save_path}/txme_params_{epoch}.json"
    if not is_save_onnx: 
        calibration_json_path = calibration_json_path.replace(".json", "_tmp.json")    
        sk_params_json_path = sk_params_json_path.replace(".json", "_tmp.json")
        txme_params_json_path = txme_params_json_path.replace(".json", "_tmp.json")
                   
    with open(calibration_json_path, "w") as outfile: 
        json.dump(calibration_dmaxdmin, outfile, indent=4)    
        
    with open(sk_params_json_path, "w") as outfile: 
        json.dump(sk_params, outfile, indent=4) 
             
    with open(txme_params_json_path, "w") as outfile: 
        json.dump(txme_params, outfile, indent=4) 
                     
    if is_save_onnx:           
        saved_model_path = save_onnx_fp(
            layers, model_input_names, 
            model_output_names, 
            export_model_path, 
            wo_method=f"_qat_{epoch}.onnx")
        print("save {} success !!!".format(saved_model_path))                                       
                  
def get_datasets(datasets, preprocess):
    in_data_list = []
    for data_idx in tqdm(range(len(datasets)), postfix='read data to cache'):
        in_data = copy.deepcopy(datasets[data_idx])
        if not isinstance(in_data, np.ndarray):
            in_data = cv2.imread(in_data)
            # transformed = transform(image=in_data)
            # in_data = transformed["image"]
        in_data = preprocess(copy.deepcopy(in_data))
        in_data_list.append(in_data)
    return in_data_list  
                 
                                                            
def train(rank, world_size, datasets, datasets_eval, kwargs):
    # dist.init_process_group(
    #     backend="nccl",
    #     init_method="tcp://localhost:23456",
    #     rank=rank,
    #     world_size=world_size,
    # )
    # torch.cuda.set_device(rank)
            
    logger = kwargs["logger"]        
    preprocess = kwargs["preprocess"]  
    max_learning_rate = kwargs["max_learning_rate"] 
    min_learning_rate = kwargs["min_learning_rate"]
    model_input_names = kwargs["model_input_names"]
    model_output_names = kwargs["model_output_names"]
    export_model_path = kwargs["export_model_path"]
    layers = kwargs["layers"]
    epochs = kwargs["epochs"]
    batch_size = kwargs["batch_size"]
    tensorboard_dir = kwargs["tensorboard_dir"]  
    print_log_per_epoch = kwargs["print_log_per_epoch"]
    save_onnx_interval = kwargs["save_onnx_interval"]
    keyword_params = kwargs["keyword_params"]
    
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
        
    qat_save_path = os.path.split(export_model_path)[0]
    if not os.path.exists(qat_save_path):
        os.makedirs(qat_save_path)

    # layers = []
    # layers_num = kwargs["layers_num"]       
    # for layer_idx in range(layers_num):
    #     layer_idx = str(layer_idx).zfill(4)
    #     with open(f"work_dir/layers/{layer_idx}.pkl", "rb+") as f:
    #         layer = pickle.load(f)
    #         layers.append(layer)   
    # kwargs.update(dict(layers=layers))    
    
    kwargs.update(dict(rank=rank))             
    model = QAT_Module(**kwargs)  
    model = model.cuda(rank)
    # model = DDP(model, device_ids=[rank], output_device=rank)
    
    qat_params = []
    for i, qat_param in enumerate(model.qat_params):
        qat_param['params'].requires_grad = True
        layer_num = qat_param["layer_num"]
        qat_params.append(qat_param)
            
    optimizer = torch.optim.Adam(
        qat_params, lr=max_learning_rate,
    )
    # model = torch.nn.DataParallel(model)
    # model = DDP(model, device_ids=[rank], output_device=rank)
                     
    in_data_list_eval = get_datasets(datasets_eval, preprocess)
    if datasets_eval == datasets:
        in_data_list = in_data_list_eval
    else:
        in_data_list = get_datasets(datasets, preprocess)
    
    epoch_per_layer = 1
    # epochs = epoch_per_layer * (layer_num + 2)
    layer_idx = 0
    min_avg_loss = dict(loss=1000, epoch=0)                                                                   
    for epoch in tqdm(range(epochs), postfix='current epoch index of qat'): 
        # in_data_list = []
        # for data_idx in tqdm(range(len(datasets)), postfix='read data to cache'):
        #     in_data = copy.deepcopy(datasets[data_idx])
        #     if not isinstance(in_data, np.ndarray):
        #         in_data = cv2.imread(in_data)
        #         # transformed = transform(image=in_data)
        #         # in_data = transformed["image"]
        #     in_data = preprocess(copy.deepcopy(in_data))
        #     in_data_list.append(in_data)
                    
        #### evaluation and save onnx model
        # if (save_onnx_interval == 1) or (epoch == 0 or epoch % save_onnx_interval == 0 or epoch == epochs - 1):
        if epoch == epochs - 1 or epoch == 0 or epoch % 10 == 0:
        # if 0:
            logger.info("start evaluation ...")
            
            avg_loss = []
            avg_txme_clip = []
            txme_clip_dict = dict()
            
            with torch.no_grad():
                random.seed(0)  
                indices = list(range(len(datasets_eval)))
                indices = batchify_list(indices, batch_size, is_trainning=False)   
                for iter, indice in enumerate(indices):
                    in_data = [in_data_list_eval[ix] for ix in indice]
                    in_data = np.concatenate(in_data, axis=0)
                    result_layer_loss, middle_layer_loss, loss_w_l1, loss_hv_regular, txme_clip, txme_clip_scale, txme_clip_dict_tmp = model(
                        in_data, device=torch.device(f"cuda:{rank}"), is_trainning=False)
                    avg_loss.append(result_layer_loss.mean())   
                    # avg_txme_clip.append(txme_clip.mean())
                    
                    for key in txme_clip_dict_tmp.keys():
                        if key in txme_clip_dict.keys():
                            txme_clip_dict[key] += txme_clip_dict_tmp[key]
                        else:
                            txme_clip_dict[key] = txme_clip_dict_tmp[key]
                    
                        if iter == len(indices) - 1:
                            txme_clip_dict[key] /= len(indices)
                            
            avg_loss_tmp = torch.stack(avg_loss).mean()
            # avg_txme_clip_tmp = torch.stack(avg_txme_clip).mean()
            if avg_loss_tmp < min_avg_loss["loss"]:
                min_avg_loss["loss"] = avg_loss_tmp
                min_avg_loss["epoch"] = epoch
                is_best_model = True
            else:
                is_best_model = False
            # max_txme_clip = 0.0
            # layer_name = ''
            # for layer_name_ in txme_clip_dict.keys():
            #     if txme_clip_dict[layer_name_] > max_txme_clip:
            #         layer_name = layer_name_ 
            #         max_txme_clip = txme_clip_dict[layer_name_]  
            # layer_name, max_txme_clip = sorted(txme_clip_dict.items(), key=lambda x: x[1], reverse=True)[0]
                                        
            logger.info(
                "test, rank: {}, epoch: {}, avg_loss: {:.8f}, min_avg_loss: {:.8f}, best_result_epoch: {}\n"
                .format(
                    rank,
                    epoch,
                    avg_loss_tmp, # type: ignore
                    min_avg_loss["loss"],
                    min_avg_loss["epoch"],
                    # avg_txme_clip_tmp,
                    # max_txme_clip,
                    # layer_name,
                )
            )
                                                 
            save_model(
                layers, model_input_names, 
                model_output_names, 
                export_model_path, 
                epochs, 
                epoch=epoch,
                txme_clip_dict=txme_clip_dict,
                is_save_onnx=True,
            )
            logger.info("finish evaluation ...")
            if epoch == epochs - 1: 
                break
        #### evaluation and save onnx model
                  
        random.seed(epoch)  
        indices = list(range(len(datasets)))
        random.shuffle(indices)        
        indices = batchify_list(indices, batch_size, is_trainning=True)
                 
        if epoch == 0:
            total_iters = len(indices) * epochs
            frequency_pring_log = total_iters // (print_log_per_epoch * epochs)
            ### https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_iters,  # Maximum number of iterations.
                eta_min=min_learning_rate,  # Minimum learning rate. # type: ignore
            )
              
        keyword_params_ = {
            "0": ["sk"],
            "1": ["si", "sc", "so"],
            "2": ["alpha"],
            "3": ["beta"],
            "4": ["ema_decay"],
        } 
        k = 0
        keyword_params_cpy = dict()
        for key in keyword_params_.keys():
            if is_subset(keyword_params_[key], keyword_params):
                keyword_params_cpy[str(k)] = keyword_params_[key]
                k += 1
        keyword_params_ = keyword_params_cpy
            
        if epoch % epoch_per_layer == 0 and epoch > 0:
            layer_idx += 1
        
        print("layer_idx: ", layer_idx, "epoch: ", epoch)    
        # continue
                                     
        j = 0
        avg_loss = []
        for iter, indice in enumerate(indices):
            iters = epoch * len(indices) + iter
            # if iter > 1:
                # break
            
            # if len(keyword_params_.keys()) == 1:
            #     keyword_params_tmp = keyword_params_[str(0)]
            # else:
            #     keyword_params_tmp = keyword_params_[str(iters % len(keyword_params_.keys()))]

            requires_grad_sk = True
            for qat_param in qat_params:
                if qat_param["name"] in keyword_params:# and qat_param["layer_idx"] == layer_idx:
                    for param in qat_param["params"]:
                        param.requires_grad = requires_grad_sk
                else:
                    for param in qat_param["params"]:
                        param.requires_grad = not requires_grad_sk
                                    
            # weight_layer_num = len([qat_param["params"] for qat_param in qat_params if qat_param["name"] == "sk"])
            # for i, qat_param in enumerate(qat_params):
            #     if qat_param["name"] in ["sk"]:
            #         for param in qat_param["params"]:
            #             if i == j:
            #                 param.requires_grad = True
            #             else:
            #                 param.requires_grad = False
            #     else:
            #         for param in qat_param["params"]:
            #             param.requires_grad = False
            # j = j + 1
            # if j == weight_layer_num - 1:
            #     j = 0
                     
            # requires_grad_sk = True
            # layer_num = qat_params[0]["layer_num"]
            # for qat_param in qat_params:
            #     if qat_param["name"] in keyword_params and qat_param["layer_idx"] == j:
            #         for param in qat_param["params"]:
            #             param.requires_grad = requires_grad_sk
            #     else:
            #         for param in qat_param["params"]:
            #             param.requires_grad = not requires_grad_sk
            # j += 1
            # if layer_num == j:
            #     j = 0
                                                                                    
            total_loss = 0.0
            in_data = [in_data_list[ix] for ix in indice]
            in_data = np.concatenate(in_data, axis=0)
            result_layer_loss, middle_layer_loss, loss_w_l1, loss_hv_regular, txme_clip, txme_clip_scale, txme_clip_dict_tmp = model(
                in_data, device=torch.device(f"cuda:{rank}"), is_trainning=True)
            avg_loss.append(result_layer_loss.mean())   
                      
            # layer_name, max_txme_clip = sorted(txme_clip_dict_tmp.items(), key=lambda x: x[1], reverse=True)[0]
                                                                                                        
            total_loss += result_layer_loss.mean()
            # total_loss += middle_layer_loss.mean()
            total_loss += loss_w_l1.mean() * 1.0e-3
            # total_loss += loss_hv_regular.mean() * 1.0
            # total_loss += txme_clip.mean() * 1.0
            # total_loss += max_txme_clip.mean() * 1.0
            # total_loss += txme_clip_scale.mean() * 1.0
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # scheduler.step()
            
            # if iters % 100 == 0:
            #     writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], iters)
            #     writer.add_scalar("result_layer_loss", result_layer_loss.mean().item(), iters)
            #     writer.add_scalar("loss_hv_regular", loss_hv_regular.mean().item(), iters)
                
            if iters % frequency_pring_log == 0:
                # for qat_param in qat_params:
                #     for param in qat_param["params"]:
                #         if param.requires_grad:
                #             logger.info("iter: {}, layer_idx: {}, j: {}, requires_grad: true, {}, {}".format(iter, qat_param["layer_idx"], j - 1, qat_param["layer_name"], qat_param["name"]))
                learning_rate = scheduler.get_last_lr()[0]
                logger.info(
                    "train, rank: {}, epoch: {}, iter: {}, iters: {}, batch_size: {}, result_layer_loss: {:.8f}, middle_layer_loss: {:.8f}, loss_w_l1: {:.8f}, loss_hv_regular: {:.8f}, learning_rate: {:.8f}"
                    .format(
                        rank,
                        epoch,
                        iter,
                        iters,
                        batch_size,
                        result_layer_loss, # type: ignore
                        middle_layer_loss,
                        loss_w_l1,
                        loss_hv_regular,
                        # txme_clip,
                        # txme_clip_scale,
                        # max_txme_clip,
                        # layer_name,                        
                        learning_rate,
                    )
                )
                    
        avg_loss_tmp = torch.stack(avg_loss).mean()  
        logger.info(
            "train, rank: {}, epoch: {}, avg_loss: {:.8f}\n"
            .format(
                rank,
                epoch,
                avg_loss_tmp, # type: ignore
            )
        )
        
    # dist.destroy_process_group()