# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/12/1 17:06
# @File     : model_export.py
import copy
import json
import os
import re
import struct
import time
import zlib

import numpy as np

try:
    from utils import (Registry, get_last_layer_quant, get_scale_param,
                       invert_dict, to_bytes, from_bytes, find_key_by_value_in_enum)
    from export.v1 import get_version
    from export.v1.npu_layer import *
    from export.v2.model_export import mExportV2
    from export.v3.network import NETWORK_V3 as rt_factory
    from export.interface_model_b import QUANT_TYPE, LAYER_TYPE, LAYER_WEIGHT    
except Exception:
    from onnx_converter.utils import (Registry, get_last_layer_quant, get_scale_param, # type: ignore
                       invert_dict, to_bytes, from_bytes, find_key_by_value_in_enum)
    from onnx_converter.export.v1 import get_version # type: ignore
    from onnx_converter.export.v1.npu_layer import * # type: ignore
    from onnx_converter.export.v2.model_export import mExportV2 # type: ignore
    from onnx_converter.export.v3.network import NETWORK_V3 as rt_factory # type: ignore
    from onnx_converter.export.interface_model_b import QUANT_TYPE, LAYER_TYPE, LAYER_WEIGHT # type: ignore
    
    
class mExportV3(mExportV2): # type: ignore
    def __init__(self, **kwargs):
        super(mExportV3, self).__init__(**kwargs)
        kwargs["version"] = "v3"
        self.init_wexport(**kwargs)

        self.bgr_format = kwargs["bits"]["bgr_format"]
        self.save_placeholder_params = kwargs["bits"]["save_placeholder_params"]
        model_c = self.init_net_work(rt_factory, kwargs)
        self.network_inputblock = rt_factory.get("input")(**model_c) # type: ignore        
        self.network_outputblock = rt_factory.get("output")(**model_c) # type: ignore               
                                              
    def parse_model_b(self, contents, layers):
        json_contents = dict()
        
        # Head Block
        ptr = 0
        logo = from_bytes(contents[ptr:ptr+12], dtype=np.str) # type: ignore   
        ptr += 12
        json_contents["logo"] = logo
            
        version = "V"
        for i in range(4):
            v = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore   
            ptr += 1
            version += "{}".format(v)
            if i != 3:
                version += "."
            
        json_contents["version"] = version[:-2]
            
        date = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore   
        ptr += 4
        date = time.localtime(date)
        json_contents["date"] = time.strftime('%Y-%m-%d %H:%M:%S', date)
        
        input_block_offset = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore   
        ptr += 4
        output_block_offset = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore   
        ptr += 4
        sequence_block_offset = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore   
        ptr += 4
        layer_block_offset = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore   
        ptr += 4
        weight_block_offset = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore   
        ptr += 4
        tail_block_offset = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore   
        ptr += 4
        
        json_contents["input_block_offset"] = input_block_offset
        json_contents["output_block_offset"] = output_block_offset
        json_contents["sequence_block_offset"] = sequence_block_offset
        json_contents["layer_block_offset"] = layer_block_offset
        json_contents["weight_block_offset"] = weight_block_offset
        json_contents["tail_block_offset"] = tail_block_offset
           
        # Input Block
        ptr = input_block_offset
        input_cnt = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0]
        ptr += 2
        json_contents["input_cnt"] = input_cnt
        if input_cnt > 0:
            ptr += 2 ### reserved
            for i in range(input_cnt):
                json_contents[f"input_{i}"] = dict()
                input_layer_idx = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0] # type: ignore 
                ptr += 2
                json_contents[f"input_{i}"]["input_layer_idx"] = input_layer_idx                
                input_data_type = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore 
                ptr += 1
                json_contents[f"input_{i}"]["input_data_type"] = find_key_by_value_in_enum(input_data_type, NpuType_t) # type: ignore 
                input_data_fmt = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore 
                ptr += 1
                json_contents[f"input_{i}"]["input_data_fmt"] = find_key_by_value_in_enum(input_data_fmt, LayerFmt_t) 
                input_N, input_H, input_W, input_C = from_bytes(contents[ptr:ptr+8], dtype=np.uint16)
                ptr += 8
                json_contents[f"input_{i}"]["input_N"] = input_N 
                json_contents[f"input_{i}"]["input_H"] = input_H 
                json_contents[f"input_{i}"]["input_W"] = input_W 
                json_contents[f"input_{i}"]["input_C"] = input_C 
                
                # input_pre
                pre = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore   
                ptr += 1
                input_pre_quant_type = find_key_by_value_in_enum(pre, LayerQuant_t)
                pre_quant = QUANT_TYPE.get(input_pre_quant_type)() # type: ignore   
                input_pre_quant_param, ptr = pre_quant(contents, ptr)
                json_contents[f"input_{i}"]["pre_quant_type"] = input_pre_quant_type   
                json_contents[f"input_{i}"]["pre_quant_param"] = input_pre_quant_param  
                         
        # Output Block
        ptr = output_block_offset
        output_layer_cnt = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0]
        ptr += 2
        json_contents["output_layer_cnt"] = output_layer_cnt
        if output_layer_cnt > 0:
            ptr += 2 ### reserved
            for i in range(output_layer_cnt):
                json_contents[f"output_{i}"] = dict()
                output_layer_idx = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0] # type: ignore 
                ptr += 2
                json_contents[f"output_{i}"]["output_layer_idx"] = output_layer_idx
                output_data_type = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore 
                ptr += 1
                json_contents[f"output_{i}"]["output_data_type"] = find_key_by_value_in_enum(output_data_type, NpuType_t) # type: ignore 
                output_data_fmt = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore 
                ptr += 1
                json_contents[f"output_{i}"]["output_data_fmt"] = find_key_by_value_in_enum(output_data_fmt, LayerFmt_t) 
                output_N, output_H, output_W, output_C, real_c, _ = from_bytes(contents[ptr:ptr+12], dtype=np.uint16)
                ptr += 12
                json_contents[f"output_{i}"]["output_N"] = output_N 
                json_contents[f"output_{i}"]["output_H"] = output_H 
                json_contents[f"output_{i}"]["output_W"] = output_W 
                json_contents[f"output_{i}"]["output_C"] = output_C
                json_contents[f"output_{i}"]["real_c"] = real_c  
                            
                # input_pre
                pre = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore   
                ptr += 1
                output_pre_quant_type = find_key_by_value_in_enum(pre, LayerQuant_t)
                pre_quant = QUANT_TYPE.get(output_pre_quant_type)() # type: ignore   
                output_pre_quant_param, ptr = pre_quant(contents, ptr)
                json_contents[f"output_{i}"]["pre_quant_type"] = output_pre_quant_type   
                json_contents[f"output_{i}"]["pre_quant_param"] = output_pre_quant_param                               
                                
        # Sequence Block
        ptr = sequence_block_offset
        layer_seq_len = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0]
        ptr += 2
        if layer_seq_len % 2 == 0:
            layer_seq = from_bytes(contents[ptr:layer_block_offset-2], dtype=np.uint16)
        else:
            layer_seq = from_bytes(contents[ptr:layer_block_offset], dtype=np.uint16)
        layer_seq = list(layer_seq)    
        json_contents["layer_seq_len"] = layer_seq_len
        json_contents["layer_seq"] = layer_seq
                    
        # Model Block
        ptr = layer_block_offset
        layer_cnt = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0]
        ptr += 2
        reserved = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0]
        ptr += 2
        layer_types = [] 
        process_params = []       
        for layer_idx in range(layer_cnt):
            # print(layers[layer_idx].get_layer_name(), ptr)
            # if layers[layer_idx].get_is_result_layer():
            #     print("test")
            layer = layers[layer_idx]
            layer_type = layer.get_layer_type()
            layer_name = layer.get_layer_name()
               
            json_contents[f"layer_{layer_idx}"] = dict()
            json_contents[f"layer_{layer_idx}"]["layer_name"] = layer_name
            layer_in_cnt = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0]
            ptr += 2  
            json_contents[f"layer_{layer_idx}"]["layer_in_cnt"] = layer_in_cnt      
            layer_out_cnt = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0]
            ptr += 2   
            json_contents[f"layer_{layer_idx}"]["layer_out_cnt"] = layer_out_cnt   
            layer_in_from = from_bytes(contents[ptr:ptr+4*layer_in_cnt], dtype=np.uint16)
            ptr += 4*layer_in_cnt
            layer_in_from = list(layer_in_from)
            layer_in_from_layer = [layer_in_from[i] for i in range(len(layer_in_from)) if i % 2 == 0]
            layer_in_from_layer_out = [layer_in_from[i] for i in range(len(layer_in_from)) if i % 2 != 0]
            json_contents[f"layer_{layer_idx}"]["layer_in_from_layer"] = layer_in_from_layer
            json_contents[f"layer_{layer_idx}"]["layer_in_from_layer_out"] = layer_in_from_layer_out
            layer_out_to_layer_cnt = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0]
            ptr += 2
            json_contents[f"layer_{layer_idx}"]["layer_out_to_layer_cnt"] = layer_out_to_layer_cnt
            layer_out_to_layer = from_bytes(contents[ptr:ptr+2*layer_out_to_layer_cnt], dtype=np.uint16)
            layer_out_to_layer = list(layer_out_to_layer)
            json_contents[f"layer_{layer_idx}"]["layer_out_to_layer"] = layer_out_to_layer
            ptr += 2*layer_out_to_layer_cnt
            if ptr % 4 != 0:
                reserved_size = ptr - 4 * (ptr // 4)
                ptr += reserved_size
                                            
            ### pre: one input->one pre
            pre_quant_types = []
            pre_quant_params = []
            for layer_in_idx in layer_in_from_layer:
                pre = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore   
                ptr += 1
                pre_quant_type = find_key_by_value_in_enum(pre, LayerQuant_t)
                pre_quant_types.append(pre_quant_type)
                pre_quant = QUANT_TYPE.get(pre_quant_types[-1])() # type: ignore   
                pre_quant_param, ptr = pre_quant(contents, ptr)
                pre_quant_param["qi_type"] = pre_quant_param.pop("qio_type")
                pre_quant_param.update(quant_type=pre_quant_type)
                pre_quant_params.append(pre_quant_param)
                
            ### process
            layer_type = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore   
            layer_type = find_key_by_value_in_enum(layer_type, LayerType_t)
            ptr += 4
            process = LAYER_TYPE.get(layer_type)() # type: ignore   
            process_param, ptr = process(contents, ptr)
            process_param.update(layer_type=layer_type)
            layer_types.append(layer_type)
            process_params.append(process_param)
            
            ### post: one output->one post
            post_quant_types = []
            post_quant_params = []
            for _ in layer.get_onnx_output_name():
                post = from_bytes(contents[ptr:ptr+1], dtype=np.uint8)[0] # type: ignore   
                ptr += 1
                post_quant_type = find_key_by_value_in_enum(post, LayerQuant_t)
                post_quant_types.append(post_quant_type)
                post_quant = QUANT_TYPE.get(post_quant_types[-1])() # type: ignore   
                post_quant_param, ptr = post_quant(contents, ptr)
                post_quant_param["qo_type"] = post_quant_param.pop("qio_type")
                post_quant_param.update(quant_type=post_quant_type)
                post_quant_params.append(post_quant_param)
                
            json_contents[f"layer_{layer_idx}"]["pre_quant_params"] = pre_quant_params
            json_contents[f"layer_{layer_idx}"]["process_param"] = process_param
            json_contents[f"layer_{layer_idx}"]["post_quant_params"] = post_quant_params
                                                       
        # Weight Block
        ptr = weight_block_offset
        for layer_idx, (layer, layer_type, params) in enumerate(zip(layers, layer_types, process_params)):
            if layer.get_layer_type() in ["add", "concat"]:
                continue

            # if layer.get_layer_type() == "data":
            #     out_data = layer.get_out_data()['output']
            #     _, _, h, w = out_data.shape
            #     last_s = "0.b"            
            #     feature_output = np.fromfile(
            #         os.path.join(
            #             self.log_dir, "weights",
            #             "_".join([str(layer_idx).zfill(4), layer.get_layer_type(), str(layer_idx), last_s])
            #     ), dtype=out_data.dtype)               
            #     feature_output = feature_output.reshape(-1, h, w, self.Csize).transpose(0, 3, 1, 2).reshape(1, -1, h, w)
            #     feature_output = feature_output[:, 1:4, :, :]
            #     diff = np.abs(out_data - feature_output).sum()
            #     assert(diff == 0)
                                
            # if layer_type in LAYER_WEIGHT.module_dict and len(layer.get_insert()['out_pad']):  
            #     if len(layer.get_insert()['out_pad']) != 1:
            #         continue                                                    
            #     out_data = layer.get_out_data()[-1]['output']
            #     _, c, h, w = out_data.shape
            #     if layer.get_is_result_layer():
            #         last_s = "0_fp.b"
            #     else:
            #         last_s = "0.b"            
            #     feature_output = np.fromfile(
            #         os.path.join(
            #             self.log_dir, "weights",
            #             "_".join([str(layer_idx).zfill(4), layer.get_layer_type(), str(layer_idx), last_s])
            #     ), dtype=out_data.dtype)               
            #     feature_output = feature_output.reshape(-1, h, w, self.Csize).transpose(0, 3, 1, 2).reshape(1, -1, h, w)
            #     feature_output = feature_output[:, :c, :, :]
            #     diff = np.abs(out_data - feature_output).sum()
            #     assert(diff == 0)
            
            # if layer_type in LAYER_WEIGHT.module_dict:
            #     data_channel_extension, first_conv = layer.get_data_channel_extension(), layer.get_first_conv()
            #     instance = LAYER_WEIGHT.get(layer_type)(data_channel_extension, first_conv) # type: ignore   
            #     weight_dict = instance(params, contents, ptr, layer.get_layer_type())
            #     for k, v in weight_dict.items():
            #         # json_contents[f"layer_{layer_idx}"][k] = v.tolist()
            #         if k == "weight":
            #             qweight = layer.get_qweight()
            #             out_c, in_c, kh, kw = qweight.shape
            #             if layer.get_layer_type() in ["depthwiseconv"]:
            #                 diff = weight_dict[k].reshape(2, kh, kw, 1, self.Csize).transpose(0, 4, 3, 1, 2).reshape(-1, 1, kh, kw)[:out_c, :, :, :] - qweight
            #             else:
            #                 if first_conv:
            #                     diff = weight_dict[k].reshape(kh, kw, self.Ksize, self.Csize).transpose(2, 3, 0, 1)[:, 1:4, :, :] - qweight
            #                 else:
            #                     diff = weight_dict[k].reshape(kh, kw, self.Ksize, self.Csize).transpose(2, 3, 0, 1)[:out_c, :in_c, :, :] - qweight
            #         elif k == "bias":
            #             qbias = layer.get_qbias()
            #             out_c = qbias.shape[0]
            #             diff = weight_dict[k][:out_c] - qbias
            #         diff = np.abs(diff).sum()
            #         print("test, {}, {}".format(k, diff))
            
            # if layer.get_is_result_layer():
            #     dtype = np.float32
            #     last_s = "0_fp.b"
            # else:
            #     dtype = np.int8
            #     last_s = "0.b"
            # feature_output = np.fromfile(
            #     os.path.join(
            #         self.log_dir, "weights",
            #         "_".join([str(layer_idx).zfill(4), layer.get_layer_type(), str(layer_idx), last_s])
            # ), dtype=dtype)               
            # feature_output = feature_output.tolist()
            # if layer_idx == 0:
            #     OH = json_contents[f"layer_{layer_idx}"]["process_param"]["H"]
            #     OW = json_contents[f"layer_{layer_idx}"]["process_param"]["W"]
            #     C = json_contents[f"layer_{layer_idx}"]["process_param"]["C"]
            # else:
            #     if layer.get_layer_type() in ["fc"]:
            #         OH = 1
            #         OW = 1
            #         C = json_contents[f"layer_{layer_idx}"]["process_param"]["N"]
            #     else:
            #         H = json_contents[f"layer_{layer_idx}"]["process_param"].get("H")
            #         if H: 
            #             OH = H
            #         else:
            #             OH = json_contents[f"layer_{layer_idx}"]["process_param"]["OH"]
            #         W = json_contents[f"layer_{layer_idx}"]["process_param"].get("W")
            #         if W:
            #             OW = W
            #         else:
            #             OW = json_contents[f"layer_{layer_idx}"]["process_param"]["OW"]
            #         OC = json_contents[f"layer_{layer_idx}"]["process_param"].get("OC")
            #         if OC:
            #             C = OC
            #         else:
            #             C = json_contents[f"layer_{layer_idx}"]["process_param"]["C"]
            # mem_size = OH * OW * C
            # json_contents[f"feature_layer_{layer_idx}"] = feature_output[:mem_size]
            # json_contents[f"feature_size_layer_{layer_idx}"] = mem_size
        
        # Tail Block
        ptr = tail_block_offset
        crc32 = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore       
        json_contents["crc32"] = crc32
               
        json_str = json.dumps(json_contents, indent=4, ensure_ascii=False)
        with open('work_dir/parse_model_b.json', 'w') as json_file:
            json_file.write(json_str)                       
        
    def store_placeholder(self, layer, func_data):
        in_data = layer.get_in_data()
        out_data = layer.get_out_data()["output"]
        if not self.bgr_format:
            in_data = in_data[:, [2, 1, 0], :, :]
            out_data = out_data[:, [2, 1, 0], :, :]
            
        if self.save_placeholder_params:
            in_data = out_data
            
        res = layer.feat_export(
                func_data,
                in_data,
                layer=layer,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "indata.b"),
            )
        self.save_indata.extend(res)

        layer_idx = layer.get_idx()
        layer_type = layer.get_layer_type()
        layer_name = "{}_{}_{}_{}.b".format(
            str(layer_idx).zfill(4), layer_type, layer.get_idx(), 0
        )
        res = layer.feat_export(
            func_data,
            out_data,
            layer=layer,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, layer_name),
        )
        self.save_feats.update({layer_name:res})

    def write_placeholder_params(self):
        if self.save_placeholder_params:
            placeholder_param_file = os.path.join(self.weights_dir, "placeholder_params.txt")
            if os.path.exists(placeholder_param_file):
                os.remove(placeholder_param_file)
            f = open(placeholder_param_file, "a")
            f.write("scale, zero_point\n")
            for layer in self.layers:
                if layer.get_layer_type() == "data":
                    scale, zero_point = layer.get_scale()[0]["scale"], layer.get_scale()[0]["zero_point"]
                    scale, zero_point = scale.astype(np.float32), zero_point.astype(np.int32)
                    f.write(str(scale) + ", " + str(zero_point) + "\n")
            f.close()  
                                   
    def write_weights(self):
        self.write_placeholder_params()
        self.export_weights()
        
        w_offset = -1
        for layer in self.layers:
            if layer.get_is_result_layer():
                w_offset_tmp = self.layers[-1].get_w_offset()["w_offset"]
                if w_offset_tmp > w_offset:
                    w_offset = w_offset_tmp
                    
        for layer in self.layers:
            if layer.get_is_result_layer():
                scales = copy.deepcopy(layer.get_scales()[-1])
                zo = scales["zo"]
                if "fscale" in scales.keys():
                    fscale = scales['fscale']
                else:
                    fscale = scales['out_scale']
                if isinstance(fscale, np.ndarray):
                    out_align = layer.get_insert()["out_align"][0]
                    real_c = layer.get_insert()["out_pad"][0][-1]
                    fscale_ = np.zeros(out_align, dtype=np.float32)
                    fscale_[:real_c] = np.array(fscale)
                    res = bytearray()
                    res += to_bytes(fscale_, dtype=np.float32) # type: ignore
                    if isinstance(zo, np.ndarray):
                        res += to_bytes(zo, dtype=np.int32) # type: ignore
                    self.save_weights.extend(res)
                    w_offset_tmp = copy.deepcopy(layer.get_w_offset())
                    w_offset_tmp["fscale_w_offset"] = [w_offset, w_offset + len(res)]
                    layer.set_w_offset(w_offset_tmp)
                    
                    
    def write_network(self):
        try:
            reserved = 0
            layer_num = len(self.layers)
            contents = bytearray()
            
            # Head
            logo = 'TimesIntelli' # char
            version = [int(v) for v in get_version().split(".")]
            version = version + [reserved] # uint8
            # print( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(date/1000)) )
            date = int(round(time.time())) # uint32
            for content, dtype in zip( # type: ignore
                [logo, version, date],
                [np.str, np.uint8, np.uint32], # type: ignore
            ):
                contents += to_bytes(content, dtype=dtype) # type: ignore
            head_addr_replaced = len(contents) ### start from input_block_offset when replaced
            input_block_offset = reserved
            output_block_offset = reserved
            sequence_block_offset = reserved
            layer_block_offset = reserved
            weight_block_offset = reserved
            tail_block_offset = reserved
            for content in [
                    input_block_offset, output_block_offset,
                    sequence_block_offset, layer_block_offset,
                    weight_block_offset, tail_block_offset,
                ]:
                contents += to_bytes(content, dtype=np.uint32) # type: ignore

            input_block_offset = len(contents)
            # Input
            input_cnt, data_layers = 0, []
            for layer in self.layers:
                if layer.get_layer_type() == "data":
                    input_cnt += 1
                    data_layers.append(layer)
            input_cnt = [input_cnt, reserved] # uint16
            contents += to_bytes(input_cnt, dtype=np.uint16)
            for layer in data_layers:
                layer_type = layer.get_layer_type()
                contents += getattr(self, "network_inputblock").save(layer)
                            
            output_block_offset = len(contents)
            # Output
            output_cnt, result_layers = 0, []
            for layer in self.layers:
                if layer.get_is_result_layer():
                    output_cnt += 1   
                    result_layers.append(layer)     
            output_layer_cnt = [output_cnt, reserved] # uint16
            contents += to_bytes(output_layer_cnt, dtype=np.uint16)
            for layer in result_layers:
                layer_type = layer.get_layer_type()
                contents += getattr(self, "network_outputblock").save(layer)
                            
            sequence_block_offset = len(contents)
            # Sequence
            layer_seq_len = layer_num # uint16
            contents += to_bytes(layer_seq_len, dtype=np.uint16)
            layer_seq = [i for i in range(layer_seq_len)]
            contents += to_bytes(layer_seq, dtype=np.uint16)
            if layer_seq_len % 2 == 0: # reserve 2 bytes at last layer_seq_len is even
                contents += to_bytes(reserved, dtype=np.uint16)

            layer_block_offset = len(contents)
            layer_cnt = layer_num # uint16
            for content in [layer_cnt, reserved]:
                contents += to_bytes(content, dtype=np.uint16)
            layer_in_tmp, layer_out_tmp = -1, -1
            for layer_idx, layer in enumerate(self.layers):
                try:
                    # print('export: ', layer.get_layer_name(), len(contents))
                    # if layer.get_is_result_layer():
                    #     print("test")  
                    
                    layer_type = layer.get_layer_type()
                    if layer_type == "data":
                        layer_in_from_layer = []
                        for idx in layer.get_input_idx():
                            if idx == -1:
                                layer_in_from_layer.append(layer_in_tmp)
                                layer_in_tmp -= 1
                    else:
                        layer_in_from_layer = layer.get_input_idx()
                    
                    if layer.get_is_result_layer():
                        layer_out_to_layer = []
                        for idx in layer.get_output_idx():
                            if idx == -1:
                                layer_out_to_layer.append(layer_out_tmp)
                                layer_out_tmp -= 1               
                    else:
                        layer_out_to_layer = layer.get_output_idx()
                    
                    if layer_type == "data":
                        layer_in_cnt = 1
                        layer_out_cnt = 1
                    else:                
                        layer_in_cnt = len(layer.get_onnx_input_name())
                        layer_out_cnt = len(layer.get_onnx_output_name())
                    for content in [layer_in_cnt, layer_out_cnt]:
                        contents += to_bytes(content) # uint16
                                    
                    layer_in_from_layer_out = layer.get_input_map()
                    layer_out_to_layer_cnt = len(layer_out_to_layer)
                    for content0, content1 in zip(layer_in_from_layer, layer_in_from_layer_out):
                        content = [content0, content1]
                        contents += to_bytes(content) # uint16
                    contents += to_bytes(layer_out_to_layer_cnt) # uint16
                    contents += to_bytes(layer_out_to_layer) # uint16
                    if len(contents) % 4 != 0:
                        reserved_size = len(contents) - 4 * (len(contents) // 4)
                        contents += reserved_size * to_bytes(reserved, dtype=np.uint8) # type: ignore
                        
                    # pre, process, post
                    contents += getattr(self, "network_{}".format(layer_type)).save(layer)
                except:
                    self.logger.error('layer of: {} convert weights failed!'.format(layer.get_layer_name()))
                    os._exit(-1)

            weight_block_offset = len(contents)
            # write weights, 4 bytes aligned
            contents += to_bytes(self.save_weights, dtype=np.int8) # type: ignore

            tail_block_offset = len(contents)
            # write tail, 4 bytes aligned
            crc32 = zlib.crc32(contents) # uint32
            contents += to_bytes(crc32, dtype=np.uint32) # type: ignore
            
            contents = bytearray(contents)
            # replace
            head_addr_replaced_tmp = head_addr_replaced
            for content in [
                    input_block_offset, output_block_offset,
                    sequence_block_offset, layer_block_offset,
                    weight_block_offset, tail_block_offset,
                ]:
                content = to_bytes(content, dtype=np.uint32) # type: ignore
                offset_size = len(content)
                contents[head_addr_replaced_tmp : head_addr_replaced_tmp + offset_size] = content # type: ignore
                head_addr_replaced_tmp += offset_size
                        
            # write contents into model.b
            model_b_path = os.path.join(self.log_dir, "model.b")
            with open(model_b_path, "wb") as f:
                f.write(contents)      
            self.logger.info("write model.b done!")
            try:
                self.parse_model_b(contents, layers=self.layers)
                self.logger.info("parse model.b done!!!")
            except:
                self.logger.error("check convert weight failed!")
                os._exit(-1)
        except:
            self.logger.error("convert network header or binary failed!")
            os._exit(-1)
