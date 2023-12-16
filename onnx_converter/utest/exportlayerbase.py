# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/07/05 15:01:04
# @File    : layerbase.py
import json
import sys
import time
import zlib

import numpy as np


sys.path.append("./")  # NOQA: E402

try:
    from export.interface_model_b import QUANT_TYPE, LAYER_TYPE, LAYER_WEIGHT
    from export import get_version, mExportV3
    from export.v1.npu_layer import *
    from utest import RUN_EXPORT
    from utest.generate_attrs_case import GAC_C
    from utils import to_bytes, from_bytes, find_key_by_value_in_enum
except Exception:
    from onnx_converter.export.interface_model_b import QUANT_TYPE, LAYER_TYPE, LAYER_WEIGHT # type: ignore
    from onnx_converter.export import get_version, mExportV3 # type: ignore
    from onnx_converter.export.v1.npu_layer import * # type: ignore
    from onnx_converter.utest import RUN_EXPORT # type: ignore
    from onnx_converter.utest.generate_attrs_case import GAC_C # type: ignore
    from onnx_converter.utils import to_bytes, from_bytes, find_key_by_value_in_enum # type: ignore
    

import os
import copy
import shutil

import pytest


def write_commit_id(model_c_path):
    try:
        import onnx_converter # type: ignore
        commit_id_file = os.path.join(
            onnx_converter.__path__[0], "utest/data/commit_id.txt") # type: ignore   
    except:
        commit_id_file = 'utest/data/commit_id.txt'
    if os.path.isfile(commit_id_file):
        shutil.copyfile(commit_id_file, os.path.join(model_c_path, "commit_id.txt")) 


class Base(object):
    def setup_class_(self):
        self.contents = []
             
             
    @staticmethod            
    def parse_model_b(contents, layers, model_c_path):
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
        layer_seq_len = from_bytes(contents[ptr:ptr+2], dtype=np.uint16)[0] # type: ignore
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
            # for _ in layer.get_onnx_output_name():
            for _ in layer_out_to_layer:
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
            json_contents[f"layer_{layer_idx}"]["ops_string"] = layer.get_ops_setting()["ops_string"]
                                                       
        # Weight Block
        ptr = weight_block_offset
        for layer_idx, (layer, layer_type, params) in enumerate(zip(layers, layer_types, process_params)):
            if layer_type in LAYER_WEIGHT.module_dict:
                instance = LAYER_WEIGHT.get(layer_type)() # type: ignore   
                weight_dict = instance(params, contents, ptr, layer.get_layer_type())
                # for k, v in weight_dict.items():
                    # json_contents[f"layer_{layer_idx}"][k] = v.tolist()
                
        # parse feature
        # feature_output = np.fromfile(os.path.join(model_c_path, "0000_concat_0_0.b"), dtype=np.int8)
        # feature_output = feature_output.tolist()
        # mem_ptr = 0
        # for i in range(output_layer_cnt):
        #     output_H = json_contents[f"output_{i}"]["output_H"]
        #     output_W = json_contents[f"output_{i}"]["output_W"]
        #     output_C = json_contents[f"output_{i}"]["output_C"]
        #     mem_size = output_H * output_W * output_C
        #     json_contents[f"feature_output_{i}"] = feature_output[mem_ptr:mem_ptr+mem_size]
        #     json_contents[f"feature_size_{i}"] = mem_size
        #     mem_ptr += mem_size          

        # Tail Block
        ptr = tail_block_offset
        crc32 = from_bytes(contents[ptr:ptr+4], dtype=np.uint32)[0] # type: ignore       
        json_contents["crc32"] = crc32
        
        json_str = json.dumps(json_contents, indent=4, ensure_ascii=False)
        with open(os.path.join(model_c_path, "parse_model_b.json"), 'w') as json_file:
            json_file.write(json_str)   
            
                                                                               
    def teardown_class_(self, weights_dir, case_name, is_export_model_c=True):
        model_c_path = os.path.join(weights_dir, case_name, "weights")
        if not os.path.exists(model_c_path):
            os.makedirs(model_c_path, exist_ok=True, mode=0o777)
        if is_export_model_c:
            modelc_file = "model.c"
            mode = "w"
            with open(os.path.join(model_c_path, modelc_file), mode) as f:
                f.write('#include "npu_layer.h"\n\n')
                f.write("LayerInfo_t layers[]={\n")
                for content in self.contents:
                    f.write(content + "\n")
                f.write("};\n\n")
                content = 'char version[] = "{}";\n'.format(get_version())
                f.write(content)            
        else:
            modelc_file = "model.b"
            mode = "wb"
            contents = bytearray()
                    
            layer_num = len(self.contents)
            
            # Head
            reserved = 0
            logo = 'TimesIntelli' # char
            version = [int(v) for v in get_version().split(".")]
            version = version + [reserved] # uint8
            # print( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(date)) )
            date = int(round(time.time())) # uint32
            for content, dtype in zip(
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
            input_cnt = [0, reserved] # uint16
            for (_, _, layer) in self.contents:
                if layer.get_layer_type() in ["gru", "lstm"]:
                    continue
                input_idx = layer.get_input_idx()
                input_cnt[0] += len(input_idx)          
            contents += to_bytes(input_cnt, dtype=np.uint16)
            NPU_DataType = {
                "": "NPU_INT8",
                0: "NPU_UINT8",
                1: "NPU_INT8",
                2: "NPU_UINT16",
                3: "NPU_INT16",
                4: "NPU_UINT32",
                5: "NPU_INT32",
                6: "NPU_UINT64",
                7: "NPU_INT64",
                8: "NPU_FP32",
                9: "NPU_FP64",
            }  
            MatFmt = ["FMT_MAT", "FMT_MAT_TRANS"]
            CubeFmt = ["FMT_CUBE_TSXE", "FMT_CUBE_HWC", "FMT_CUBE_CHW"]                      
            input_layer_idx = 0
            for (_, _, layer) in self.contents:
                if layer.get_layer_type() in ["gru", "lstm"]:
                    continue                
                input_types = layer.get_input_type()
                for idx, input_idx in enumerate(layer.get_input_idx()):
                    input_layer_idx -= 1
                    input_type = input_types[-1]
                    i_type = NPU_DataType[input_type]
                    input_data_type = NpuType_t[i_type].value # type: ignore
                    if "split" in layer.get_insert():
                        input_H, input_W  = layer.get_insert()["split"]["feat_i"][0]   
                    else:
                        input_H, input_W  = layer.get_insert()["feat_i"][0]   
                    if [input_H, input_W] == [1, 1]:
                        i_fmt = MatFmt[1]
                    else:
                        i_fmt = CubeFmt[1]   
                    input_data_fmt = LayerFmt_t[i_fmt].value # type: ignore  
                    if "split" in layer.get_insert():
                        input_N, input_C = 1, layer.get_insert()["split"]["in_align"][idx] 
                    else:                
                        input_N, input_C = 1, layer.get_insert()["in_align"][idx]   
                    input_shape = [input_N, input_H, input_W, input_C]
                    for content, dtype in zip(
                        [
                            input_layer_idx,
                            input_data_type, input_data_fmt,
                            input_shape, 
                        ],
                        [
                            np.uint16,
                            np.uint8, np.uint8,
                            np.uint16, 
                        ],
                    ):
                        contents += to_bytes(content, dtype=dtype)         # type: ignore
                    inpu_pre = bytearray()
                    inpu_pre += to_bytes(LayerQuant_t["QUANT_NONE"].value, dtype=np.uint8) # type: ignore
                    inpu_pre += to_bytes(NpuType_t[i_type].value, dtype=np.uint8) # type: ignore
                    inpu_pre += 2 * to_bytes(reserved, dtype=np.uint8) # type: ignore
                    contents += inpu_pre
                
            output_block_offset = len(contents)
            # Output
            output_layer_cnt = [0, reserved] # uint16
            for (_, _, layer) in self.contents:
                if layer.get_layer_type() in ["gru", "lstm"]:
                    continue                
                output_idx = layer.get_output_idx()
                output_layer_cnt[0] += len(output_idx)            
            contents += to_bytes(output_layer_cnt, dtype=np.uint16)
            output_layer_idx = 0
            for (_, _, layer) in self.contents:
                if layer.get_layer_type() in ["gru", "lstm"]:
                    continue                
                output_types = layer.get_output_type()
                for idx, output_idx in enumerate(layer.get_output_idx()):
                    output_layer_idx -= 1
                    output_type = output_types[-1]
                    o_type = NPU_DataType[output_type]
                    output_data_type = NpuType_t[o_type].value # type: ignore
                    if "split" in layer.get_insert():
                        output_H, output_W  = layer.get_insert()["split"]["feat_o"][0]   
                    else:
                        output_H, output_W  = layer.get_insert()["feat_o"][0]   
                    if [output_H, output_W] == [1, 1]:
                        o_fmt = MatFmt[1]
                    else:
                        o_fmt = CubeFmt[1]   
                    output_data_fmt = LayerFmt_t[o_fmt].value # type: ignore    
                    if "split" in layer.get_insert():
                        output_N, output_C = 1, layer.get_insert()["split"]["out_align"][idx]  
                        real_c = layer.get_insert()["split"]["out_pad"][idx][1]
                    else:              
                        output_N, output_C = 1, layer.get_insert()["out_align"][idx]   
                        if layer.get_insert()["is_align"]:
                            real_c = output_C
                        else:
                            real_c = 0
                            if layer.get_layer_type() in ["concat"]:
                                for out_pad in layer.get_insert()["out_pad"]:
                                    real_c += (out_pad[1] - out_pad[0])
                    output_shape = [output_N, output_H, output_W, output_C]
                    for content, dtype in zip(
                        [
                            output_layer_idx,
                            output_data_type, output_data_fmt,
                            output_shape, real_c, reserved,
                        ],
                        [
                            np.uint16,
                            np.uint8, np.uint8,
                            np.uint16, np.uint16, np.uint16, 
                        ],
                    ):
                        contents += to_bytes(content, dtype=dtype)         # type: ignore
                    inpu_pre = bytearray()
                    inpu_pre += to_bytes(LayerQuant_t["QUANT_NONE"].value, dtype=np.uint8) # type: ignore
                    inpu_pre += to_bytes(NpuType_t[o_type].value, dtype=np.uint8) # type: ignore
                    inpu_pre += 2 * to_bytes(reserved, dtype=np.uint8) # type: ignore
                    contents += inpu_pre
                
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
            for layer_idx, (content, binary_weight, layer) in enumerate(self.contents):
                content_cpy = copy.deepcopy(content)
                
                layer_type = layer.get_layer_type()
                if 1: #layer_type == "data":
                    layer_in_from_layer = []
                    for idx in layer.get_input_idx():
                        if idx == -1:
                            layer_in_from_layer.append(layer_in_tmp)
                            layer_in_tmp -= 1
                else:
                    layer_in_from_layer = layer.get_input_idx()
                
                if 1: #layer.get_is_result_layer():
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
                elif layer_type in ["concat"]:
                    layer_in_cnt = layer.get_layer_ops()["attrs"][0]["input_len"]
                    layer_out_cnt = 1
                elif layer_type in ["add", "sub", "cadd", "csub", "pmul", "cmul", "matmul"]:
                    layer_in_cnt = 2
                    layer_out_cnt = 1
                elif layer_type in ["split"]:
                    layer_in_cnt = 1
                    layer_out_cnt = len(layer.get_output_idx())
                else:                
                    layer_in_cnt = 1
                    layer_out_cnt = 1
                for content in [layer_in_cnt, layer_out_cnt]:
                    contents += to_bytes(content) # uint16
                
                layer_in_from_layer_out = [0] * layer_in_cnt
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
                contents += content_cpy

            weight_block_offset = len(contents)
            # write weights, 4 bytes aligned
            layers = [layer for _, _, layer in self.contents]
            for layer_idx, (content, binary_weight, layer) in enumerate(self.contents):
                contents += binary_weight

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
                            
            with open(os.path.join(model_c_path, modelc_file), mode) as f:
                f.write(contents)
            
            self.parse_model_b(contents=contents, layers=layers, model_c_path=model_c_path.split("weights")[0])
                                            
        write_commit_id(model_c_path)


class LayerBase(Base):
    def run(
        self,
        params,
        layer_type,
        feature,
        weight,
        quantize_dtype,
        in_type,
        out_type,
        process_scale,
        chip_type,
        export_version,
        is_export_model_c,
        virtual_round,        
        weights_dir,
    ):
        quantization_args = dict(
            type=quantize_dtype,
            in_type=in_type,
            out_type=out_type,
            process_scale=process_scale,
            method=dict(feature=feature, weight=weight),
            virtual_round=virtual_round,
        )
        arguments = dict(
            layer_type=layer_type,
            export_args=dict(
                chip_type=chip_type,
                mode="wb",
                is_acc_woffset=False,
                export_version=export_version,
            ),
            quantization_args=quantization_args,
        )

        case_name = feature
        case_name += "_" + str(weight[0]) # type: ignore
        case_name += "_" + str(weight[1]) # type: ignore
        case_name += "_" + str(quantize_dtype) # type: ignore
        case_name += "_" + str(process_scale) # type: ignore
        case_name += "_" + in_type
        case_name += "_" + out_type
        case_name += "_" + layer_type
        case_name += "_" + chip_type

        # if in_type in ['int16'] or out_type in ['int16']:
        #     pytest.skip(case_name)

        self.setup_class_() # type: ignore

        try:
            from onnx_converter.config import (export, export_v1, export_v2, export_v3, quantize, # type: ignore
                                               vision_quantize, voice_quantize) # type: ignore
            from onnx_converter.utils import props_with_ # type: ignore

            export_cfg = props_with_(export)
            if export_version == 1:
                export_cfg_ = props_with_(export_v1)
            elif export_version == 2:
                export_cfg_ = props_with_(export_v2)
            elif export_version == 3:
                export_cfg_ = props_with_(export_v3)    
            else:
                export_cfg_ = export_cfg            
            export_cfg.update(export_cfg_)
            quant_cfg = props_with_(quantize)
            voice_quant_cfg = props_with_(voice_quantize)
            vision_quant_cfg = props_with_(vision_quantize)
            kwargs = {
                "weights_dir": weights_dir,
                "case_name": case_name,
                "log_dir": os.path.join(weights_dir, case_name),
                "log_name": "test_{}.log".format(case_name),
                "is_stdout": False,
                "quant_cfg": quant_cfg,
                "voice_quant_cfg": voice_quant_cfg,
                "vision_quant_cfg": vision_quant_cfg,
                "export_cfg": export_cfg,
                "arguments": arguments,
            }
        except Exception:
            # if 1:
            kwargs = {
                "weights_dir": weights_dir,
                "case_name": case_name,
                "log_dir": os.path.join(weights_dir, case_name),
                "log_name": "test_{}.log".format(case_name),
                "is_stdout": False,
                "quant_cfg": "config/quantize.py",
                "voice_quant_cfg": "config/voice_quantize.py",
                "vision_quant_cfg": "config/vision_quantize.py",
                "export_cfg": "config/export_v{}.py".format(
                    export_version
                ),
                "arguments": arguments,
            }

        kwargs.update(params)
        
        if kwargs.get("attrs"):
            fuse_op = kwargs['attrs'].get("fuse_op")
            if fuse_op and len(fuse_op) > 0:
                flag1 = False #fuse_op[0] in ["relu", "relu6", "relux"] and process_scale == "shiftfloatscaletable"
                flag2 = fuse_op[0] not in ["relu", "relu6", "relux"] and process_scale not in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
                # flag3 = kwargs["arguments"]["quantization_args"]["method"]["feature"] == "asym" and process_scale == "shiftfloatscaletable"
                flag3 = fuse_op[0] in ["relu6", "relux"] and process_scale not in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
                if flag1 or flag2 or flag3:
                    pytest.skip(case_name)
            # else:
            #     if process_scale == "shiftfloatscaletable":
            #         pytest.skip(case_name)
                           
        content = RUN_EXPORT.get(layer_type)(kwargs)
        self.contents.append(content)

        self.teardown_class_(
            weights_dir=weights_dir, 
            case_name=case_name, 
            is_export_model_c=is_export_model_c,
        )


class MultiLayerBase(Base):
    def run(
        self,
        quantize_dtype,
        quantize_method_process_scale_layer_type_in_type_out_type,
        input_settings_combination,
        chip_type,
    ):
        (
            quantize_method_f,
            quantize_method_w,
            process_scale,
            layer_type,
            in_type,
            out_type,
        ) = quantize_method_process_scale_layer_type_in_type_out_type.split("/")

        if self.use_input_settings_combination: # type: ignore
            attr_case_id = list(input_settings_combination.keys())[0].split("_")[-1]
            attr_layer_type = list(input_settings_combination.keys())[0].split("_{}".format(attr_case_id))[0]
        else:
            attr_tmp = input_settings_combination.split("_")
            attr_case_id = attr_tmp[-1]
            attr_layer_type = '_'.join(attr_tmp[:-1])
            
        if layer_type not in self.valid_layer_types or attr_layer_type != layer_type: # type: ignore
            pytest.skip("skipped {} layer export".format(layer_type))

        feature = "asym" if "asym" in quantize_method_f else "sym"
        weight = [
            "asym" if "asym" in quantize_method_w else "sym",
            "perchannel" if "perchannel" in quantize_method_w else "pertensor",
        ]
        quantization_args = dict(
            type=quantize_dtype,
            in_type=in_type,
            out_type=out_type,
            process_scale=process_scale,
            method=dict(feature=feature, weight=weight),
            virtual_round=self.virtual_round, # type: ignore
        )
        arguments = dict(
            layer_type=layer_type,
            export_args=dict(
                chip_type=chip_type,
                mode=self.mode, # type: ignore
                is_acc_woffset=False,
                export_version=self.export_version, # type: ignore
            ),
            quantization_args=quantization_args,
            weights_dir=self.weights_dir, # type: ignore
        )

        # basename = os.path.basename(cfg_file)
        # json_file = cfg_file.replace(basename, 'arguments.json')
        # save_json(copy.deepcopy(arguments), json_file=json_file)
        # arguments = json.load(open(json_file, 'r'))

        case_name = feature
        case_name += "_" + str(weight[0])
        case_name += "_" + str(weight[1])
        case_name += "_" + str(quantize_dtype)
        case_name += "_" + str(process_scale)
        case_name += "_" + in_type
        case_name += "_" + out_type
        case_name += "_" + layer_type
        case_name += "_" + attr_case_id
        case_name += "_" + chip_type

        if ("int" in in_type and in_type != quantize_dtype) or (
            "int" in out_type and out_type != quantize_dtype
        ):
            pytest.skip(case_name)

        self.setup_class_()

        try:
            from onnx_converter.config import (export, export_v1, export_v2, export_v3, quantize, # type: ignore
                                               vision_quantize, voice_quantize) # type: ignore
            from onnx_converter.utils import props_with_ # type: ignore

            export_cfg = props_with_(export)
            if self.export_version == 1: # type: ignore
                export_cfg_ = props_with_(export_v1)
            elif self.export_version == 2: # type: ignore
                export_cfg_ = props_with_(export_v2)
            elif self.export_version == 3: # type: ignore
                export_cfg_ = props_with_(export_v3)    
            else:
                export_cfg_ = export_cfg            
            export_cfg.update(export_cfg_)
            quant_cfg = props_with_(quantize)
            voice_quant_cfg = props_with_(voice_quantize)
            vision_quant_cfg = props_with_(vision_quantize)
            kwargs = {
                "weights_dir": self.weights_dir, # type: ignore
                "case_name": case_name,
                "log_dir": os.path.join(self.weights_dir, case_name), # type: ignore
                "log_name": "test_{}.log".format(case_name),
                "is_stdout": False,
                "is_single_layer": False,
                "quant_cfg": quant_cfg,
                "voice_quant_cfg": voice_quant_cfg,
                "vision_quant_cfg": vision_quant_cfg,
                "export_cfg": export_cfg,
                "arguments": arguments,
            }
        except Exception:
            # if 1:
            kwargs = {
                "weights_dir": self.weights_dir, # type: ignore
                "case_name": case_name,
                "log_dir": os.path.join(self.weights_dir, case_name), # type: ignore
                "log_name": "test_{}.log".format(case_name),
                "is_stdout": False,
                "is_single_layer": False,
                "quant_cfg": "config/quantize.py",
                "voice_quant_cfg": "config/voice_quantize.py",
                "vision_quant_cfg": "config/vision_quantize.py",
                "export_cfg": "config/export_v{}.py".format(
                    self.export_version # type: ignore
                ),
                "arguments": arguments,
            }

        kwargs.update(attr_case_id=attr_case_id)
        if self.use_input_settings: # type: ignore
            kwargs.update(self.input_settings[layer_type]) # type: ignore

        if self.use_input_settings_combination: # type: ignore
            layer_attrs = copy.deepcopy(input_settings_combination[layer_type + "_" + attr_case_id])
            gac_c = GAC_C.get(layer_type)(layer_attrs=layer_attrs, layer_type=layer_type)
            kwargs.update(gac_c)
            kwargs.update(case_attr=layer_attrs)

        if kwargs.get("attrs"):
            fuse_op = kwargs['attrs'].get("fuse_op")
            if fuse_op and len(fuse_op) > 0:
                flag1 = False #fuse_op[0] in ["relu", "relu6", "relux"] and process_scale == "shiftfloatscaletable"
                flag2 = fuse_op[0] not in ["relu", "relu6", "relux"] and process_scale not in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
                # flag3 = kwargs["arguments"]["quantization_args"]["method"]["feature"] == "asym" and process_scale == "shiftfloatscaletable"
                flag3 = fuse_op[0] in ["relu6", "relux"] and process_scale not in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
                if flag1 or flag2 or flag3:
                    pytest.skip("skipped {} layer export".format(layer_type))
            # else:
            #     if process_scale == "shiftfloatscaletable":
            #         pytest.skip("skipped {} layer export".format(layer_type))
                
        content = RUN_EXPORT.get(layer_type)(kwargs)
        self.contents.append(content)

        self.teardown_class_(
            weights_dir=self.weights_dir, # type: ignore
            case_name=case_name, 
            is_export_model_c=self.is_export_model_c, # type: ignore
        )


class SingleLayerBase(Base):
    def teardown_class(self):
        self.teardown_class_(self,
            weights_dir=self.weights_dir, # type: ignore
            case_name=self.layer_type, # type: ignore
            is_export_model_c=self.is_export_model_c, # type: ignore
        )               

    def run(
        self,
        quantize_dtype,
        quantize_method_process_scale_layer_type_in_type_out_type,
        input_settings_combination,
        chip_type,
    ):
        (
            quantize_method_f,
            quantize_method_w,
            process_scale,
            layer_type_,
            in_type,
            out_type,
        ) = quantize_method_process_scale_layer_type_in_type_out_type.split("/")

        if self.use_input_settings_combination: # type: ignore
            attr_case_id = list(input_settings_combination.keys())[0].split("_")[-1]
            attr_layer_type = list(input_settings_combination.keys())[0].split("_{}".format(attr_case_id))[0]
        else:
            attr_tmp = input_settings_combination.split("_")
            attr_case_id = attr_tmp[-1]
            attr_layer_type = '_'.join(attr_tmp[:-1])
            
        pytest.assume(len(self.valid_layer_types) == 1) # type: ignore
        if layer_type_ != self.layer_type or attr_layer_type != self.layer_type: # type: ignore
            pytest.skip("skipped {} layer export".format(layer_type_))

        feature = "asym" if "asym" in quantize_method_f else "sym"
        weight = [
            "asym" if "asym" in quantize_method_w else "sym",
            "perchannel" if "perchannel" in quantize_method_w else "pertensor",
        ]
        quantization_args = dict(
            type=quantize_dtype,
            in_type=in_type,
            out_type=out_type,
            process_scale=process_scale,
            method=dict(feature=feature, weight=weight),
            virtual_round=self.virtual_round, # type: ignore
        )
        arguments = dict(
            layer_type=self.layer_type, # type: ignore
            export_args=dict(
                chip_type=chip_type,
                mode=self.mode, # type: ignore
                is_acc_woffset=self.is_acc_woffset, # type: ignore
                export_version=self.export_version, # type: ignore
            ),
            quantization_args=quantization_args,
            weights_dir=self.weights_dir, # type: ignore
        )

        # basename = os.path.basename(cfg_file)
        # json_file = cfg_file.replace(basename, 'arguments.json')
        # save_json(copy.deepcopy(arguments), json_file=json_file)
        # arguments = json.load(open(json_file, 'r'))

        case_name = feature
        case_name += "_" + str(weight[0])
        case_name += "_" + str(weight[1])
        case_name += "_" + str(quantize_dtype)
        case_name += "_" + str(process_scale)
        case_name += "_" + in_type
        case_name += "_" + out_type
        case_name += "_" + self.layer_type # type: ignore
        case_name += "_" + attr_case_id
        case_name += "_" + chip_type

        case_name = self.layer_type # type: ignore
        if ("int" in in_type and in_type != quantize_dtype) or (
            "int" in out_type and out_type != quantize_dtype
        ):
            pytest.skip(case_name)

        try:
            from onnx_converter.config import (export, export_v1, export_v2, export_v3, quantize, # type: ignore
                                               vision_quantize, voice_quantize) # type: ignore
            from onnx_converter.utils import props_with_ # type: ignore

            export_cfg = props_with_(export)
            if self.export_version == 1: # type: ignore
                export_cfg_ = props_with_(export_v1)
            elif self.export_version == 2: # type: ignore
                export_cfg_ = props_with_(export_v2)
            elif self.export_version == 3: # type: ignore
                export_cfg_ = props_with_(export_v3)    
            else:
                export_cfg_ = export_cfg            
            export_cfg.update(export_cfg_)
            quant_cfg = props_with_(quantize)
            voice_quant_cfg = props_with_(voice_quantize)
            vision_quant_cfg = props_with_(vision_quantize)
            kwargs = {
                "weights_dir": self.weights_dir, # type: ignore
                "case_name": case_name,
                "log_dir": os.path.join(self.weights_dir, case_name), # type: ignore
                "log_name": "test_{}.log".format(case_name),
                "is_stdout": False,
                "is_single_layer": True,
                "quant_cfg": quant_cfg,
                "voice_quant_cfg": voice_quant_cfg,
                "vision_quant_cfg": vision_quant_cfg,
                "export_cfg": export_cfg,
                "arguments": arguments,
            }
        except Exception:
            # if 1:
            kwargs = {
                "weights_dir": self.weights_dir, # type: ignore
                "case_name": case_name,
                "log_dir": os.path.join(self.weights_dir, case_name), # type: ignore
                "log_name": "test_{}.log".format(case_name),
                "is_stdout": False,
                "is_single_layer": True,
                "quant_cfg": "config/quantize.py",
                "voice_quant_cfg": "config/voice_quantize.py",
                "vision_quant_cfg": "config/vision_quantize.py",
                "export_cfg": "config/export_v{}.py".format(
                    self.export_version # type: ignore
                ),
                "arguments": arguments,
            }

        kwargs.update(attr_case_id=attr_case_id)
        if self.use_input_settings: # type: ignore
            kwargs.update(self.input_settings[self.layer_type]) # type: ignore

        if self.use_input_settings_combination: # type: ignore
            layer_attrs = copy.deepcopy(input_settings_combination[self.layer_type + "_" + attr_case_id]) # type: ignore
            gac_c = GAC_C.get(self.layer_type)(layer_attrs=layer_attrs, layer_type=self.layer_type) # type: ignore
            kwargs.update(gac_c)
            kwargs.update(case_attr=layer_attrs)

        if kwargs.get("attrs"):              
            fuse_op = kwargs['attrs'].get("fuse_op")
            if fuse_op and len(fuse_op) > 0:
                flag1 = False #fuse_op[0] in ["relu", "relu6", "relux"] and process_scale == "shiftfloatscaletable"
                flag2 = fuse_op[0] not in ["relu", "relu6", "relux"] and process_scale not in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
                # flag3 = kwargs["arguments"]["quantization_args"]["method"]["feature"] == "asym" and process_scale == "shiftfloatscaletable"
                flag3 = fuse_op[0] in ["relu6", "relux"] and process_scale not in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
                if flag1 or flag2 or flag3:
                    pytest.skip("skipped {} layer export".format(layer_type_))
            # else:
            #     if process_scale == "shiftfloatscaletable":
            #         pytest.skip("skipped {} layer export".format(layer_type_))
                           
        content = RUN_EXPORT.get(self.layer_type)(kwargs) # type: ignore
        self.contents.append(content) # type: ignore
