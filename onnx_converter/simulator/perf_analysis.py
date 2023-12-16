# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : nan.qin
# @Company  : SHIQING TECH
# @Time     : 2022/09/01 14:28
# @File     : error_analysis.py
import sys
sys.path.append("./")

from abc import abstractmethod
import os
import csv
import copy
import numpy as np
import binascii
from pyDes import des, CBC, PAD_PKCS5

try:
    from utils import Registry
except:
    from onnx_converter.utils import Registry# type: ignore

from secrets import token_bytes
import json
PARAM_PARSER = Registry('param_parser',scope='')
TPLT_MATCH = Registry('template_match',scope='')

class ParamParserBase():
    def __init__(self):
        pass
    @abstractmethod    
    def run(self):
        pass

@PARAM_PARSER.register_module(name='conv2d')
class Conv2dParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = itype 0, wtype 1, otype 2, h 3, w 4, c 5, fh 6, fw 7, k 8, sh 9, sw 10, hasbias 11, t 12, gops/s 13
        # key = "%s-%s-%s-%d-%d-%d-%d-%d"%(params[0], params[1], params[2], int(params[6]), int(params[7]), \
        #     int(params[9]), int(params[10]), int(params[11]))
        key = "%s-%s-%s-%d"%(params[0], params[1], params[2], int(params[11]))        
        value = None
        if int(params[8]) == 1:
            key = "dw_%s"%key
            # value = [int(params[3]), int(params[4]), int(params[5]), int(params[6]),int(params[7]), int(params[9]),\
            #     int(params[10])]
        else:
            key = "norm_%s"%key
        
        value = [int(params[3]), int(params[4]), int(params[5]), int(params[6]),int(params[7]), int(params[8]), \
                int(params[9]),int(params[10])]
        return key, value

@PARAM_PARSER.register_module(name='fc')
class FcParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = itype 0, wtype 1, otype 2, M 3 N 4 K 5, hasbias 6
        key = "%s-%s-%s-%d"%(params[0], params[1], params[2], int(params[6]))
        value = [int(params[3]), int(params[4]), int(params[5])]
        return key, value

@PARAM_PARSER.register_module(name='quant')#pre and post
class QuantParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = quant 0, i-type 1, o-type 2, size 3
        key = "%s-%s-%s"%(params[0], params[1], params[2])
        value = [int(params[3])]
        return key, value

# @PARAM_PARSER.register_module(name='post')
# class PostParamParser(ParamParserBase):
#     def run(self, params):
#         # PARAMS = quant 0, qo_type 1, size 3
#         key = "%s-%s"%(params[0], params[1])
#         value = [int(params[2])]
#         return key, value

@PARAM_PARSER.register_module(name='act')
class ActParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, h 2, w 3, c 4, act 5, lut 6
        key = "%s-%s-%s-%s"%(params[0], params[1], params[5], params[6])
        value = [int(params[2]), int(params[3]), int(params[4])]
        return key, value  

@PARAM_PARSER.register_module(name='pool')
class PoolParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = itype 0, otype 1, pool 2, h 3, w 4, c 5, fh 6, fw 7, sh 8, sw 9
        key = "%s-%s-%s"%(params[0], params[1], params[2])
        value = [params[3], params[4], params[5], int(params[6]), int(params[7]),\
             int(params[8]), int(params[9])]
        return key, value  

@PARAM_PARSER.register_module(name='element_wise')
class ElemwiseParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, operation 2, len 3
        key = "%s-%s-%s"%(params[0], params[1], params[2])
        value = [int(params[3])]
        return key, value  

@PARAM_PARSER.register_module(name='channel_wise')
class ChannelwiseParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, operation 2, len 3
        key = "%s-%s-%s"%(params[0], params[1], params[2])
        value = [int(params[3])]
        return key, value

@PARAM_PARSER.register_module(name='resize')
class ResizeParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, ih 2, iw 3, c 4, oh 5, ow 6, method 7
        key = "%s-%s-%s"%(params[0], params[1], params[7])
        value = [int(params[2]), int(params[3]), int(params[4]), int(params[5]), int(params[6])]
        return key, value

@PARAM_PARSER.register_module(name='shuffle_only')
class ShuffleOnlyParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, h 2, w 3, c 4, sec_num 5, oc 6
        key = "%s-%s-%s"%(params[0], params[1], params[5])
        value = [int(params[2]), int(params[3]), int(params[4]), int(params[6])]
        return key, value
 
@PARAM_PARSER.register_module(name='ln')
class LnParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, h 2, w 3, c 4
        key = "%s-%s"%(params[0], params[1])
        value = [int(params[2]), int(params[3]), int(params[4])]
        return key, value

@PARAM_PARSER.register_module(name='bn')
class BnParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, h 2, w 3, c 4
        key = "%s-%s"%(params[0], params[1])
        value = [int(params[2]), int(params[3]), int(params[4])]
        return key, value

@PARAM_PARSER.register_module(name='reshape')
class ReshapeParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, h 2, w 3, c 4
        key = "%s-%s"%(params[0], params[1])
        value = [int(params[2]), int(params[3]), int(params[4])]
        return key, value

@PARAM_PARSER.register_module(name='concat')
class ConcatParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, h 2, w 3, ic_list(list) 4, oc 5
        #in_num = len(params) - 5
        ic_list = []
        if isinstance(params[4], str):
            ic_list = [int(x) for x in params[4][1:-2].replace(',',' ').split()]
        elif isinstance(params[4], list):
            ic_list = [int(x) for x in params[4]]
        else:
            pass
        in_num = len(ic_list)
        key = "%s-%s-%d"%(params[0], params[1], in_num)
        value = [int(params[2]), int(params[3]), int(params[-1])]
        return key, value           
    
@PARAM_PARSER.register_module(name='split')
class SplitParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = i_type 0, o_type 1, h 2, w 3, c 4, oc_list(list) 5, out_num 6
        oc_list = []
        if isinstance(params[5], str):
            oc_list = [int(x) for x in params[5][1:-2].replace(',',' ').split()]
        elif isinstance(params[5], list):
            oc_list = [int(x) for x in params[5]]
        else:
            pass
        out_num = len(oc_list)
        key = "%s-%s-%d"%(params[0], params[1], out_num)
        value = [int(params[2]), int(params[3]), int(params[4])]
        return key, value                 

@PARAM_PARSER.register_module(name='concat_shuffle_split')
class ConcatShuffleSplitParamParser(ParamParserBase):
    def run(self, params):
        # PARAMS = in_num 0, out_num 1, i_type 2, o_type 3, h 4, w 5, ic_list 6(list), sec_num 7, oc_list(list) 8
        # in_num, out_num = int(params[0]), int(params[1])
        # sec_num = int(params[6 + in_num])
        # c = sum([int(x) for x in params[6 : 6 + in_num]])
        ic_list, oc_list = list(), list()
        if isinstance(params[6], str):
            ic_list = [int(x) for x in params[6][1:-2].replace(',',' ').split()]
        elif isinstance(params[6], list):
            ic_list = [int(x) for x in params[6]]
        else:
            pass
        if isinstance(params[8], str):
            oc_list = [int(x) for x in params[8][1:-2].replace(',',' ').split()]
        elif isinstance(params[8], list):
            oc_list = [int(x) for x in params[8]]
        else:
            pass
        in_num = len(ic_list)
        out_num = len(oc_list)
        sec_num = int(params[7])
        ic = sum(ic_list)
        oc = sum(oc_list)
        h, w= int(params[4]), int(params[5])
        key = "%d-%d-%s-%s-%d"%(in_num, out_num, params[2], params[3], sec_num)
        value = [h, w, ic, oc]
        return key, value 

@PARAM_PARSER.register_module(name='lstm')
class LstmParamParser(ParamParserBase):
    def run(self, params):
        pass

def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        raw_data= list()
        for row in reader:
            line =[row[x].strip(" ") for x in row.keys() if x != '']
            if not '' in line:
                raw_data.append(line)
    return raw_data

def parse_decrypt_data(data):
    raw_data= list()
    rows = data.split('\n')
    for row in rows[1:]:
        row = row.split(',')
        line =[x.strip(" ") for x in row if x != '']
        if not '' in line and len(line):
            raw_data.append(line)
    return raw_data


# def print_res(content, x):
#     print(content[x['start']+1:x['end']])
#     if 'children' in x.keys():
#         for y in x['children']:
#             print_res(content, y)
#     return

def split_str_with_brace(content):
    stack = []
    result = []
    for idx, x in enumerate(content):
        if x == "{":
            res = dict(start=idx)
            stack.append(res)
        if x=='}':
            stack[-1].update({"end":idx})
            v = stack.pop()
            if len(stack):
                if not 'children' in stack[-1].keys():
                    stack[-1].update({"children":list()})
                stack[-1]['children'].append(v)
            else:
                result.append(v)
    return result

class Match():
    def __init__(self):
        pass
    @abstractmethod
    def run(self, query, template):
        pass

    def get_idx(self, val:list, template:np.ndarray):
        pass    

    
@TPLT_MATCH.register_module(name='conv2d')
class Conv2dOpMatch(Match):
    def __init__(self):
        self.ord = np.array([6,7,3,4,2,5,0,1,8])

    def get_calculation_scale(self, params):
        [h,w,c,fh,fw,k,sh,sw]=params[:8]
        return 2*(h / sh * w / sw * fh * fw * k * c)

    def run(self, query, template):
        # h, w, c, fh, fw, k, sh, sw, t
        qurey_t = copy.deepcopy(query)
        qurey_t.append(0) 
        # to new ord: [sh, sw, fh, fw, c, k, h, w, t, idx]
        qurey_t = copy.deepcopy(np.array(qurey_t))[self.ord]
        template_t = copy.deepcopy(template)[:, self.ord]
        # sh, sw, fh, fw, C, K, H, W
        abs_diff_arr = np.abs(qurey_t - template_t)
        abs_diff_list = [list(x) + [i] for i, x in enumerate(abs_diff_arr)]
        abs_diff_list.sort()
        res = dict()
        matched_time = abs_diff_list[0][8]
        target_idx = abs_diff_list[0][-1]
        target = template[target_idx]
        matched_size = self.get_calculation_scale(target)
        query_size = self.get_calculation_scale(query)
        time = matched_time * query_size / matched_size
        res.update({'time':time})
        res.update({'matched': target})
        return res

@TPLT_MATCH.register_module(name='quant')
class QuantOpMatch(Match):
    def run(self, query, template):
        # param: [size]
        abs_diff_arr = np.abs(np.array(query) - template[:, :-1]).flatten()
        idx = np.argsort(abs_diff_arr)[0]
        time = template[idx][-1] / template[idx][0] * query[0]  
        res = dict()
        res.update({'time': time})
        res.update({'matched' : int(template[idx][0])})
        return res      
  

@TPLT_MATCH.register_module(name='fc')
class FcOpMatch(Match):
    def __init__(self):
        # M, N, K
        self.ord = np.array([0,1,2])

    def get_calculation_scale(self, params):
        [M, N, K]=params[:3]
        return 2*(M * N *K)

    def run(self, query, template):
        qurey_t = copy.deepcopy(query)
        qurey_t.append(0) # add 0 as time
        qurey_t = copy.deepcopy(np.array(qurey_t))[self.ord]
        template_t = copy.deepcopy(template)[:, self.ord]        
        abs_diff_arr = np.abs(qurey_t - template_t)
        abs_diff_list = [list(x)+[i] for i, x in enumerate(abs_diff_arr)]
        res = dict()
        matched_time = abs_diff_list[0][3]
        target_idx = abs_diff_list[0][-1]
        target = template[target_idx]
        matched_size = self.get_calculation_scale(target)
        query_size = self.get_calculation_scale(query)

        time = matched_time * query_size / matched_size
        res.update({'time':time})
        res.update({'matched': target})
        return res     


class PerfAnalyzer():
    def __init__(self, **kwargs):
        self.exporter = kwargs['model_exporter']
        chip_model = kwargs['chip_model']
        try:
            import onnx_converter # type: ignore
            self.ref_data_dir = os.path.join(
                onnx_converter.__path__[0], "perf_data")# type: ignore
        except:
            if 'ref_data_dir' in kwargs.keys() and kwargs['ref_data_dir'] is not None:
                self.ref_data_dir = kwargs['ref_data_dir']
            else:
                self.ref_data_dir = "perf_data"
        print("=>>> perf_data directory: ", self.ref_data_dir)
        mem_addr = kwargs['mem_addr'].lower()
        ref_data_path = os.path.join(self.ref_data_dir, chip_model)
        self.refernce_data_table = dict()
        encrypt_flag = kwargs['encrypt_flag'] if "encrypt_flag" in kwargs.keys() else False
        self.key_file_mode = False
        self.refernce_data_table = self.read_reference_data(ref_data_path, mem_addr, encrypt_flag)
        self.layer_computational_params_list = self.calc_layer_computational_scale()
        self.log_path='work_dir/perf_analysis_result.txt'
        if not os.path.exists("work_dir/"):
            os.mkdir("work_dir/")

    def read_reference_data(self, root, mem_addr, encrypt_flag):
        refernce_data_table = dict()
        if encrypt_flag:
            data_format = '.dat'
        else:
            data_format = 'csv' 
        secret_tool = Secret()
        for file_name in os.listdir(root):
            if not file_name.endswith(data_format):
                continue
            if not 'quant' in file_name and not mem_addr in file_name:
                continue
            if file_name.startswith('quant'):
                layer_name='quant'
            else:
                layer_name = file_name.split(".")[0].split("-")[0].lower()  # get layer name
            file_path = os.path.join(root, file_name)
            ref_data = None
            
            if encrypt_flag: # decrypt dat file and parse decrypted data
                encrypt_file_path = os.path.join(root, file_name)
                decrypt_data = None
                if self.key_file_mode:
                    key_file = encrypt_file_path.replace('.dat', '.key')
                    decrypt_data = secret_tool.decrypt_file(encrypt_file_path, key_file)
                else:
                    decrypt_data = secret_tool.decrypt_file(encrypt_file_path)
                ref_data = parse_decrypt_data(decrypt_data)
            else:# parse csv file
                ref_data = read_csv(file_path)
            refernce_data_table.update({layer_name : ref_data})

        # combine condition-type params as key and rearrange data
        for layer_type in refernce_data_table.keys():
            new_data_dict = dict()
            parser = PARAM_PARSER.get(layer_type)()# type: ignore
            for data in refernce_data_table[layer_type]:
                key, params = parser.run(data[:-2])# not include time and GOPS/s
                t = float(data[-2]) # time
                params.append(t)
                if not key in new_data_dict.keys():
                    new_data_dict[key] = list()
                new_data_dict[key].append(np.array(params))
            for key in new_data_dict.keys():
                new_data_dict[key]=np.array(new_data_dict[key])
            refernce_data_table[layer_type] = new_data_dict
        return refernce_data_table

    def parse_export_info(self, layer_type_rt, layer_export_info):
        partition_idx = split_str_with_brace(layer_export_info)
        pre_infos, post_infos = list(), list()
        #num = len(partition_idx[0]['children'])
        pre_section = partition_idx[0]['children'][0]
        post_section = partition_idx[0]['children'][1]
        pre_num , post_num = 1, 1
        if 'children' in pre_section:
            pre_num = len(pre_section['children'])
        if 'children' in post_section:
            post_num = len(post_section['children'])
        #print("### test : layer_type = %s, pre_num = %d, post_num = %d"%(layer_type_rt, pre_num, post_num))
        # split pre and post
        # pre_num = 1 
        # for i in range(num-1):
        #     if partition_idx[0]['children'][i]['end']+2 == partition_idx[0]['children'][i+1]['start']:
        #         pre_num += 1
        #     else:
        #         break
        # post_num = num - pre_num 
        #pre_num, post_num = self.get_layer_pre_and_post_num(layer_type_rt, num)
        
        
        if pre_num == 1:
            pre_infos.append(layer_export_info[partition_idx[0]['children'][0]['start']+1:partition_idx[0]['children'][0]['end']])
        else:
            for i in range(pre_num):
                pre_infos.append(layer_export_info[partition_idx[0]['children'][0]['children'][i]['start']+1:partition_idx[0]['children'][0]['children'][i]['end']])
        pre_infos = [x.split(',')[:2] for x in pre_infos]
        if post_num == 1:
            post_infos.append(layer_export_info[partition_idx[0]['children'][1]['start']+1:partition_idx[0]['children'][1]['end']])        
        else:
            for i in range(post_num):   
                post_infos.append(layer_export_info[partition_idx[0]['children'][1]['children'][i]['start']+1:partition_idx[0]['children'][1]['children'][i]['end']])
        post_infos = [x.split(',')[:2] for x in post_infos]
        #print(pre_infos, post_infos)
        # layer info is between pre and post
        layer_info = layer_export_info[partition_idx[0]['children'][0]['end']+2:partition_idx[0]['children'][1]['start']-1]
        layer_info = layer_info.split(',')

        return pre_infos, post_infos, layer_info

    def calc_layer_computational_scale(self):
        layer_computaional_scale = list()
        for layer_idx, layer in enumerate(self.exporter.layers):
            layer_type = layer.get_layer_type()
            
            if not layer_type in self.exporter.valid_export_layer:
                layer_computaional_scale.append({"type":layer_type, "perf_params": None})
                continue
            layer_attr = getattr(self.exporter, "network_{}".format(layer_type))
            # get layer export contents
            layer_export_info = layer_attr.save(layer)
            # get layer name used by runtime
            npu_layer_type = layer_attr.layer_map_inv[layer.get_layer_type()]
            layer_type_rt = layer_attr.LayerInstance[npu_layer_type]
            # parse export contents, get pre_info, post_info and layer_info
            layer_export_info = layer_export_info.split('{%s,.layer.%s='%(npu_layer_type, layer_type_rt))[1][:-2]
            pre_info_list, post_info_list, layer_info = self.parse_export_info(layer_type_rt, layer_export_info)
            
            # qi_type = layer_attr.NPU_DataType[layer.get_input_type()[0]]   # pre op in type (layer in)
            # qo_type = layer_attr.NPU_DataType[layer.get_output_type()[-1]] # post op out type (layer out)

            layer_params,pre_params,post_params = list(),list(),list()
            ops_list=list()
            if layer_type_rt == 'placeholder':
                i_type, o_type = layer_info[0], layer_info[1]
                h, w, c = int(layer_info[4]),int(layer_info[5]),int(layer_info[6])
                #layer_params.extend([i_type, o_type, h,  w,  c])
                input_size = h * w * c
                output_size = input_size
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not pre_quan_type == 'QUANT_NONE':
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size])
                
            elif layer_type_rt == 'conv2d':
                i_type, w_type, o_type = layer_info[0], layer_info[1], layer_info[2]
                h, w, c = int(layer_info[6]),int(layer_info[7]),int(layer_info[8])
                fh, fw = int(layer_info[9]), int(layer_info[10])
                k, sh, sw = int(layer_info[11]), int(layer_info[12]), int(layer_info[13])
                has_bias = int(layer_info[20])
                act = layer_info[21]
                oh, ow = int(layer_info[18]), int(layer_info[19])
                layer_params.extend([i_type, w_type, o_type, h, w, c, fh, fw, k, sh, sw, has_bias])
                input_size = h * w * c
                output_size = oh * ow * k
                if layer_type == 'depthwiseconv':
                    output_size = oh * ow * c
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not post_quan_type in ['QUANT_NONE', 'QUANT_SHIFT']:
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size])

            elif layer_type_rt == 'fc':
                i_type, w_type, o_type = layer_info[0], layer_info[1], layer_info[2]
                m, k, n = int(layer_info[6]),int(layer_info[7]),int(layer_info[8])
                has_bias = int(layer_info[9])
                #act = layer_info[10]
                layer_params.extend([i_type, w_type, o_type, m, k, n, has_bias])
                input_size = m * k
                output_size = m * k
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':                
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not post_quan_type in ['QUANT_NONE', 'QUANT_SHIFT']:
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size])

            elif layer_type_rt == 'act':
                i_type, o_type = layer_info[0], layer_info[1]
                h, w, c = int(layer_info[4]),int(layer_info[5]),int(layer_info[6])
                act, lut = layer_info[7], layer_info[9]
                layer_params.extend([i_type, o_type, h, w, c, act, lut])
                input_size = h*w*c
                output_size = input_size
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not post_quan_type == 'QUANT_NONE':
                   post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size])

            elif layer_type_rt == 'pool':
                i_type, o_type = layer_info[0], layer_info[1]
                pool, h, w, c, fh, fw, sh, sw = layer_info[4], int(layer_info[5]),int(layer_info[6]),\
                    int(layer_info[7]),int(layer_info[8]),int(layer_info[9]), int(layer_info[10]),int(layer_info[11])
                oh, ow = int(layer_info[16]),int(layer_info[17])
                layer_params.extend([i_type, o_type, pool, h,w,c,fh,fw, sh, sw])
                input_size = h * w *c
                output_size = c * oh * ow
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not post_quan_type == 'QUANT_NONE':
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size])  

            elif layer_type_rt == 'element_wise' or layer_type_rt == 'channel_wise':                
                i_type, o_type = layer_info[0], layer_info[1]
                operation, size = layer_info[2], int(layer_info[3])
                layer_params.extend([i_type, o_type, operation, size])
                # sum all pre ops and count together 
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, size])
                if not post_quan_type == "QUANT_NONE":
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], size]) 
            
            elif layer_type_rt == 'resize':
                i_type, o_type = layer_info[0], layer_info[1]
                ih, iw, c = int(layer_info[4]), int(layer_info[5]), int(layer_info[6])
                oh, ow = int(layer_info[7]), int(layer_info[8]),
                method=layer_info[9]
                layer_params.extend([i_type, o_type,ih, iw, c, oh, ow, method])
                input_size = ih * iw * c
                output_size = oh * ow * c
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not post_quan_type == "QUANT_NONE":
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size]) 

            elif layer_type_rt == 'shuffle_only':
                i_type, o_type = layer_info[0], layer_info[1]
                h, w, c = int(layer_info[4]),int(layer_info[5]),int(layer_info[6])
                sec_num, oc = int(layer_info[-2]),int(layer_info[-1])
                input_size = h * w * c
                output_size = h * w * oc
                layer_params.extend([i_type, o_type, h, w, c, sec_num, oc])
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not post_quan_type == "QUANT_NONE":
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size]) 

            elif layer_type_rt == 'ln' or layer_type_rt == 'bn':
                i_type, o_type = layer_info[0], layer_info[1]
                h, w, c = int(layer_info[4]),int(layer_info[5]),int(layer_info[6]) 
                input_size = h * w * c
                output_size = input_size     
                layer_params.extend([i_type, o_type, h, w, c])
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not post_quan_type == "QUANT_NONE":
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size]) 
            
            elif layer_type_rt == 'reshape':
                i_type, o_type = layer_info[0], layer_info[1]
                h, w, c = int(layer_info[4]),int(layer_info[5]),int(layer_info[6]) 
                input_size = h * w * c
                output_size = input_size     
                layer_params.extend([i_type, o_type, h, w, c])
                pre_quan_type, post_quan_type = pre_info_list[0][0], post_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                if not post_quan_type == "QUANT_NONE":
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size])                 

            elif layer_type_rt == 'concat':
                num_in = len(pre_info_list)
                i_type, o_type = layer_info[0], layer_info[1]
                h,w = int(layer_info[4]),int(layer_info[5])
                ic_list = [int(x) for x in layer_info[6 : 6+ num_in]]
                oc = int(layer_info[-2])
                input_size = [h * w * x for x in ic_list]
                output_size =  h * w * oc
                layer_params.extend([i_type, o_type, h, w, ic_list, oc])
                for i, pre_info in enumerate(pre_info_list):
                    pre_quan_type = pre_info_list[i][0]
                    if not pre_quan_type == "QUANT_NONE":
                        pre_params.append([pre_info[0], pre_info[1], i_type, input_size[i]])
                post_quan_type = post_info_list[0][0]
                if not post_quan_type == "QUANT_NONE":
                    post_params.append([post_info_list[0][0], o_type, post_info_list[0][1], output_size])
                               
            elif layer_type_rt == 'split':
                num_out = len(post_info_list)
                i_type, o_type = layer_info[0], layer_info[1]
                h, w, c = int(layer_info[4]),int(layer_info[5]),int(layer_info[6])
                oc_list = [int(x) for x in layer_info[8:8+num_out]]
                input_size = h * w * c
                output_size = [h * w * x for x in oc_list]
                layer_params.extend([i_type, o_type, h, w, c, oc_list])
                pre_quan_type = pre_info_list[0][0]
                if not pre_quan_type == 'QUANT_NONE':
                    pre_params.append([pre_info_list[0][0], pre_info_list[0][1], i_type, input_size])
                for i, post_info in enumerate(post_info_list):
                    post_quan_type = post_info_list[i][0]
                    if not post_quan_type == 'QUANT_NONE':
                        post_params.append([post_info[0], o_type, post_info[1], output_size[i]])   

            elif layer_type_rt == 'concat_shuffle_split':#H, W, [C], [4,5,6:6+num_in]	H, W, [OC], [4,5,-2-num_out:-2]
                in_num = len(pre_info_list)
                out_num = len(post_info_list)
                i_type, o_type = layer_info[0], layer_info[1]
                h, w = int(layer_info[4]),int(layer_info[5])
                ic_list = [int(x) for x in layer_info[6 : 6 + in_num]]
                sec_num = int(layer_info[-3 - out_num])
                oc_list = [int(x) for x in layer_info[-2 - out_num : -2]]
                layer_params.extend([in_num, out_num, i_type, o_type, h, w, ic_list, sec_num, oc_list])
                for i,pre_info in enumerate(pre_info_list):
                    pre_quan_type = pre_info_list[i][0]
                    if not pre_quan_type == "QUANT_NONE":
                        pre_params.append([pre_info[0], pre_info[1], i_type, input_size[i]])# type: ignore
                for i,post_info in enumerate(post_info_list):
                    post_quan_type = post_info_list[i][0]
                    if not post_quan_type == 'QUANT_NONE':
                        post_params.append([post_info[0], o_type, post_info[1], output_size[i]])# type: ignore
                
            elif layer_type_rt == 'lstm':
                layer_computaional_scale.append({"type":layer_type_rt, "perf_params": None})
                continue
            else:
                continue
            
            
            for pre_param in pre_params:
                ops_list.append({"type":"quant", "perf_params": pre_param})
            if len(layer_params):
                ops_list.append({"type":layer_type_rt, "perf_params": layer_params})
            for post_param in post_params:
                ops_list.append({"type":"quant", "perf_params": post_param})            
            layer_computaional_scale.append({"layer_idx":layer_idx, "layer_name":layer.get_layer_name(), "layer_type":layer_type, "ops":ops_list})
        return layer_computaional_scale

        
    def __call__(self, debug = False):

        total_time = 0
        unsupported_layers = []
        f = open(self.log_path, 'w')
        for idx, layer_params in enumerate(self.layer_computational_params_list):
            try:
                one_layer_time = 0
                for ops in layer_params['ops']:
                    layer_type = ops['type']
                    if ops['perf_params'] == None or layer_type not in self.refernce_data_table.keys():
                        unsupported_layers.append(layer_type)
                        continue
                    key_query, params_query = PARAM_PARSER.get(layer_type)().run(ops['perf_params'])# type: ignore

                    params_ref = None
                    if key_query in self.refernce_data_table[layer_type].keys():
                        params_ref = self.refernce_data_table[layer_type][key_query]
                    else:
                        if layer_type in ['conv2d', 'fc']:
                            if key_query[-1] == '1' :
                                key_query = key_query[:-1] + '0'
                            else:
                                key_query = key_query[:-1] + '1'
                        if key_query in self.refernce_data_table[layer_type].keys():
                            params_ref = self.refernce_data_table[layer_type][key_query]
                        else:
                            #print("## Worning: %s query key %s not found"%(layer_type, key_query))
                            pass
                    if params_ref is None:
                        continue
                    # find best matches
                    if not layer_type in TPLT_MATCH.module_dict.keys():
                        #print("## Warning : layer type %s match and interplolation not support yes"%layer_type)
                        continue
                    matcher = TPLT_MATCH.get(layer_type)()# type: ignore
                    res = matcher.run(params_query, params_ref)
                    t = res['time']
                    matched = res['matched']
                    if debug:
                        f.write("---- op_type = %s, condition = %s, parmas = %s, matched = %s, time cost = %f us\n"\
                            %(layer_type, key_query, params_query, matched, t))
                    one_layer_time += t

                f.write("#### Layer idx: %d, layer name: %s, layer_type = %s, time cost = %f us\n"%(layer_params['layer_idx'], layer_params['layer_name'],\
                    layer_params['layer_type'], one_layer_time))
                total_time += one_layer_time
            except:
                print("Layer idx: %d, layer name: %s, layer_type = %s perf analysis failure\n"%(layer_params['layer_idx'], layer_params['layer_name']))
                os._exit(-1)
        unsupported_layers=list(set(unsupported_layers))
        
        f.write("The network total time cost is %f ms\n"%(total_time/1000))    
        if len(unsupported_layers) > 0:
            f.write("Warning: The time cost dose not include the following layer types : ")
            f.write("%s"%unsupported_layers)
        print("*** Performance analysis information has been writen to 'work_dir/perf_analysis_result.txt'***")
        f.close()
        return total_time

class Secret():
    def __init__(self):
        self.secret_key = "19831983" # Todo, add by C.C.Zhang

    def random_key(self, length):
        key=token_bytes(length)
        key_int=int.from_bytes(key, 'big')
        return key_int

    # generate encryped file and key file
    def encrypt_file_and_gen_key(self, raw):                                  
        raw_bytes = raw.encode()                         
        raw_int = int.from_bytes(raw_bytes, 'big')       
        key_int = self.random_key(len(raw_bytes))           
        return raw_int ^ key_int, key_int  
    
    # Todo , complete by C.C.Zhang
    # generate encrypted file using existing key
    def encrypt_file_use_exist_key(self, raw):
        iv = self.secret_key
        k = des(self.secret_key, CBC, iv, pad=None, padmode=PAD_PKCS5)
        en = k.encrypt(raw, padmode=PAD_PKCS5)
        return binascii.b2a_hex(en) # type: ignore
    
    # decrypt file and key file
    def decrypt_use_key_file(self, f_encrypted, f_key):    
        encrypted = json.load(f_encrypted)     
        key_int = json.load(f_key)                  
        decrypted = encrypted ^ key_int                         
        length = (decrypted.bit_length() + 7) // 8              
        decrypted_bytes = int.to_bytes(decrypted, length, 'big') 
        return decrypted_bytes.decode()  

    # Todo,  completed by C.C.Zhang
    def decryt_without_key_file(self, encrypted, encoding='utf-8'):
        iv = self.secret_key
        k = des(self.secret_key, CBC, iv, pad=None, padmode=PAD_PKCS5)
        decrypted = k.decrypt(binascii.a2b_hex(encrypted), padmode=PAD_PKCS5)
        decrypted = decrypted.decode()
        return decrypted

    def encrypt_file(self, raw_path, save_encrypted_path, save_key_path = None, encoding='utf-8'):
        save_key = True if save_key_path else False
        f1 = open(raw_path, 'rt', encoding=encoding)
        f2 = open(save_encrypted_path, 'wt', encoding=encoding)
        f3 = None
        if save_key:
            f3 = open(save_key_path, 'wt', encoding=encoding)# type: ignore
        
        if save_key:
            encrypted, key = self.encrypt_file_and_gen_key(f1.read())
            json.dump(encrypted, f2)
            json.dump(key, f3)# type: ignore
        else:
            encrypted = self.encrypt_file_use_exist_key(f1.read())
            encrypted = str(encrypted, encoding)
            json.dump(encrypted, f2)
        
        f1.close()
        f2.close()
        if save_key:
            f3.close()# type: ignore

    def decrypt_file(self, path_encrypted, key_path=None, encoding='utf-8'):
        decrypted = None
        if key_path:
            f1 = open(path_encrypted, 'rt', encoding=encoding)
            f2 = open(key_path, 'rt', encoding=encoding)
            decrypted = self.decrypt_use_key_file(f1, f2)
            f1.close()
            f2.close()
        else:
            f = open(path_encrypted, 'rt', encoding=encoding)
            encrypted = json.load(f)
            encrypted = bytes(encrypted, encoding)
            decrypted = self.decryt_without_key_file(encrypted)
            f.close()
        return decrypted            

def encryption_perf_data(ref_data_dir, save_key=False):
    secret_tool = Secret()
    for model_chip in os.listdir(ref_data_dir):
        sub_folder = os.path.join(ref_data_dir, model_chip)
        files = os.listdir(sub_folder)
        csv_files = [x for x in files if x.endswith('.csv')]
        for csv_name in csv_files:
            csv_path = os.path.join(sub_folder, csv_name)
            encrypt_file_path = csv_path.replace(".csv", ".dat")
            if save_key:
                key_path = csv_path.replace(".csv", ".key")
                secret_tool.encrypt_file(csv_path, encrypt_file_path, key_path)
            else:
                secret_tool.encrypt_file(csv_path, encrypt_file_path)
    print("Finish encrypt csv files!")

# demo for encrypt csv file to dat and keys
if __name__ == '__main__':
    encryption_perf_data('perf_data/')