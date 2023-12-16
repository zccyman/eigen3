# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/20 11:32
# @File     : network.py
import copy
import re
from abc import abstractmethod

import numpy as np

try:
    from export.v1.network import NetworkBase, _write
    from export.v1.npu_layer import *
    from utils import (Registry, get_last_layer_quant, get_scale_param,
                       invert_dict, to_bytes)
except Exception:
    from onnx_converter.export.v1.network import NetworkBase, _write # type: ignore
    from onnx_converter.export.v1.npu_layer import * # type: ignore
    from onnx_converter.utils import (Registry, get_last_layer_quant, # type: ignore
                                      get_scale_param, invert_dict, to_bytes)

NETWORK_V3: Registry = Registry("network_v3", scope="")

global reserved
reserved = 0

# export special layer in model.c
# written network structure in binary file
class NetworkV3(NetworkBase): # type: ignore
    def __init__(self, **kwargs):
        super(NetworkV3, self).__init__()

        self.kwargs = kwargs
        self.is_debug = kwargs["is_debug"]
        self.layer_map = kwargs["layer_map"]
        self.layer_map_inv = self.invert_dict_of_list(self.layer_map)
        self.LayerInstance = kwargs["LayerInstance"]
        self.LayerQuant = self.invert_dict_of_list(kwargs["LayerQuant"])
        self.NPU_DataType = kwargs["NPU_DataType"]
        self.fmt = kwargs["fmt"]
        self.CubeFmt = kwargs["CubeFmt"]
        self.ConvWFmt = kwargs["ConvWFmt"]
        self.MatFmt = kwargs["MatFmt"]
        self.FcWFmt = kwargs["FcWFmt"]
        self.bits = kwargs["bits"]
        self.data_channel_extension = self.bits["DATA_C_EXTEND"]
        self.save_placeholder_params = self.bits["save_placeholder_params"]
        self.Csize = self.bits["Csize"]
        self.Ksize = self.bits["Ksize"]
        self.LayerPrePost = self.invert_dict_of_list(kwargs["LayerPrePost"])
        self.ActivationType = self.invert_dict_of_list(kwargs["ActivationType"])
        self.invert_type(["ReduceType", "PoolType", "ElementWiseType", "ChnWiseType"])
        self.ResizeMethod = invert_dict(kwargs["ResizeMethod"])
        self.ELEMENT_WISE_MAX_IN = kwargs["ELEMENT_WISE_MAX_IN"]
        self.MAX_IN_OUT_LEN = kwargs["MAX_IN_OUT_LEN"]
        self.CONCAT_SHUFFLE_SPLIT_MAX_IN = kwargs["CONCAT_SHUFFLE_SPLIT_MAX_IN"]
        self.CONCAT_SHUFFLE_SPLIT_MAX_OUT = kwargs["CONCAT_SHUFFLE_SPLIT_MAX_OUT"]
        self.SHUFFLE_MAX_IN_SECTION = kwargs["SHUFFLE_MAX_IN_SECTION"]
        self.CONCAT_MAX_IN = kwargs["CONCAT_MAX_IN"]
        self.SPLIT_MAX_OUT = kwargs["SPLIT_MAX_OUT"]

    @abstractmethod
    def get_process(layer, i_type, o_type, w_type):
        NotImplemented

    def get_quant_mode(self, i_type, o_type):
        if i_type in ['NPU_FP32', 'NPU_FP64'] and \
            o_type in ['NPU_INT8', 'NPU_INT16', 'NPU_UINT8', 'NPU_UINT16']:
            mode = 'quant'
        elif i_type in ['NPU_INT8', 'NPU_INT16', 'NPU_UINT8', 'NPU_UINT16'] and \
            o_type in ['NPU_FP32', 'NPU_FP64']:
            mode = 'dequant'
        else:
            mode = ''

        return mode


    def get_pre_quant(self, layer, qi_type, i_type, idx=0):
        # mode in ["", "quant", "dequant"]
        mode = self.get_quant_mode(qi_type, i_type)
        
        quant = mode
        is_asymquan = self.get_asymquant(layer)  
                  
        extra_value = layer.get_scales()[idx]['extra_value']
        is_vaild_extra = False
        if not isinstance(extra_value, np.ndarray):
            if extra_value > 0:
                is_vaild_extra = True
        else:
            if np.sum(np.abs(extra_value)) > 0:
                is_vaild_extra = True   
        if is_asymquan or is_vaild_extra:
            if mode != "":
                quant += "_asy"
            else:
                quant = "asy"
                    
        return quant
    
    
    def get_post_quant(self, layer, o_type, qo_type, idx=0):
        # mode in ["", "quant", "dequant"]
        quant = self.get_pre_quant(layer, o_type, qo_type, idx)
        
        process_scale = layer.get_scale_type()
        if quant != "":
            quant = process_scale + "_" + quant
        else:
            quant = process_scale
            
        # is_perchannel = isinstance(layer.get_scales()[0]['out_shift'], np.ndarray)
        is_perchannel = self.get_perchannelquant(layer)
        if is_perchannel:
            quant = 'perchannel_' + quant        

        if process_scale in ["shiftfloatscaletable2float"]:
            quant = "shiftfloatscaletable2float"
            if is_perchannel:
               quant = 'perchannel_' + quant

        return quant
    
    
    def get_pre_qparams(self, layer, quant, scales):
        qparams = dict()
                        
        layer_quant = self.LayerQuant[quant]
        
        if "ISCALE" in layer_quant:
            qparams["shift"] = 0
            qparams["scale"] = 0
            qparams["s_shift"] = 0
        elif layer_quant in [
            "QUANT_QUANT", "QUANT_DEQUANT",
            "QUANT_QUANT_ASY", "QUANT_DEQUANT_ASY",
            "QUANT_SMOOTH",
        ]:
            si = layer.get_in_scale()
            if isinstance(si, list):
                si = si[-1]
            if quant == "asy":
                qparams["scale"] = 1.0
            else:
                qparams["scale"] = si["scale"]
            if layer_quant in ["QUANT_QUANT_ASY", "QUANT_DEQUANT_ASY"]:
                qparams["zero"] = si["zero_point"]
        else:
            qparams = dict()
        
        if "ISCALE" in layer_quant:
            qparams["in_zero"] = scales["zi"]
            qparams["out_zero"] = scales["zo"]
        
        return qparams, layer_quant
    
    
    def get_post_qparams(self, layer, quant, scales):
        qparams = dict()
        
        layer_quant = self.LayerQuant[quant]
        
        if "PER_CHN" in layer_quant:
            qparams["offset"] = layer.get_w_offset()['tmp_offset'][2]
        elif "SHIFT" in layer_quant:
            qparams["shift"] = scales["out_shift"]
        elif "LUT8_FP" in layer_quant:
            qparams["shift"] = scales["out_shift"]
            qparams["offset"] = layer.get_w_offset()['tmp_offset'][2]
        elif "ISCALE" in layer_quant:
            qparams["shift"] = scales["out_shift"]
            qparams["scale"] = scales["out_scale"]
            if "int_scale" in scales.keys():
                qparams["s_shift"] = -scales["int_scale"]
            else:
                qparams["s_shift"] = -layer.get_ops_setting()["setting"]["int_scale"]
        elif "FSCALE" in layer_quant:
            qparams["shift"] = scales["out_shift"]
            qparams["scale"] = scales["out_scale"]
        elif layer_quant in [
            "QUANT_QUANT", "QUANT_DEQUANT",
            "QUANT_QUANT_ASY", "QUANT_DEQUANT_ASY",
            "QUANT_SMOOTH",
        ]:
            so = copy.deepcopy(layer.get_scale())
            if isinstance(so, list):
                so = so[-1]            
            qparams["scale"] = so["scale"]
            if layer_quant in ["QUANT_QUANT_ASY", "QUANT_DEQUANT_ASY"]:
                qparams["zero"] = so["zero_point"]
        else:
            qparams = dict()
        
        if "ASY" in layer_quant and "PER_CHN" not in layer_quant:
            qparams["in_zero"] = scales["zi"]
            qparams["out_zero"] = scales["zo"]
            if scales["extra_value"]:
                qparams["in_zero"] = -scales["extra_value"]
        
        return qparams, layer_quant
    

    def get_pre_post(self, layer, quant, qi_qo_type, qparams):
        contents = bytearray()

        contents += to_bytes(LayerQuant_t[quant].value, dtype=np.uint8) # type: ignore
        contents += to_bytes(NpuType_t[qi_qo_type].value, dtype=np.uint8) # type: ignore
                    
        if quant in [
            "QUANT_PER_CHN_SHIFT", "QUANT_PER_CHN_SHIFT_ASY",
            "QUANT_NONE", "QUANT_PER_CHN_ISCALE", "QUANT_PER_CHN_FSCALE", 
            "QUANT_PER_CHN_ISCALE_ASY", "QUANT_PER_CHN_FSCALE_ASY", 
            "QUANT_PER_CHN_QUANT", "QUANT_PER_CHN_DEQUANT", 
            "QUANT_PER_CHN_QUANT_ASY", "QUANT_PER_CHN_DEQUANT_ASY",
            "QUANT_PER_CHN_LUT8_FP",
        ]:
            contents += to_bytes(reserved, dtype=np.uint8) # type: ignore
            contents += to_bytes(reserved, dtype=np.uint8) # type: ignore
            if quant in [
                "QUANT_PER_CHN_SHIFT", "QUANT_PER_CHN_SHIFT_ASY",
                "QUANT_PER_CHN_ISCALE", "QUANT_PER_CHN_FSCALE",
                "QUANT_PER_CHN_ISCALE_ASY", "QUANT_PER_CHN_FSCALE_ASY",
                "QUANT_PER_CHN_QUANT", "QUANT_PER_CHN_DEQUANT", 
                "QUANT_PER_CHN_QUANT_ASY", "QUANT_PER_CHN_DEQUANT_ASY",
                "QUANT_PER_CHN_LUT8_FP",
            ]:
                offset = qparams["offset"]
                contents += to_bytes(offset, dtype=np.uint32) # type: ignore
        elif quant in [
            "QUANT_SHIFT", "QUANT_SHIFT_ASY",
            "QUANT_ISCALE", "QUANT_FSCALE",
            "QUANT_ISCALE_ASY", "QUANT_FSCALE_ASY",
            "QUANT_LUT8_FP",
        ]:
            shift = qparams["shift"]
            contents += to_bytes(shift, dtype=np.int8) # type: ignore
            if "QUANT_ISCALE" in quant:
                scale = qparams["scale"]
                s_shift = qparams["s_shift"]
                contents += to_bytes(scale, dtype=np.uint8) # type: ignore
                contents += to_bytes(s_shift, dtype=np.int8) # type: ignore
                contents += to_bytes(reserved, dtype=np.int8) * 3 # type: ignore
            elif "QUANT_FSCALE" in quant:
                contents += to_bytes(reserved, dtype=np.uint8) # type: ignore
                scale = qparams["scale"]
                contents += to_bytes(scale, dtype=np.float32) # type: ignore
            elif "QUANT_SHIFT" in quant:
                contents += to_bytes(reserved, dtype=np.int8) # type: ignore
            elif "QUANT_LUT8_FP" in quant:
                contents += to_bytes(reserved, dtype=np.uint8) # type: ignore
                
            if "QUANT_LUT8_FP" not in quant:
                if "ASY" in quant:
                    in_zero, out_zero = qparams["in_zero"], qparams["out_zero"]
                    contents += to_bytes(in_zero, dtype=np.int32) # type: ignore
                    if layer.get_scale_type() == "shiftfloatscaletable":
                        out_zero = 0
                    contents += to_bytes(out_zero, dtype=np.int32) # type: ignore
            else:
                offset = qparams["offset"] #layer.get_w_offset()["tmp_offset"][2]
                contents += to_bytes(offset, dtype=np.uint32) # type: ignore
        elif quant in ["QUANT_QUANT", "QUANT_DEQUANT", "QUANT_QUANT_ASY", "QUANT_DEQUANT_ASY"]:
            contents += to_bytes(reserved, dtype=np.uint8) * 2# type: ignore
            if quant in ["QUANT_QUANT", "QUANT_DEQUANT", "QUANT_QUANT_ASY", "QUANT_DEQUANT_ASY"]:
                scale = qparams["scale"]
                contents += to_bytes(scale, dtype=np.float32) # type: ignore
                if quant in ["QUANT_QUANT_ASY", "QUANT_DEQUANT_ASY"]:
                    zero = qparams["zero"]
                    contents += to_bytes(zero, dtype=np.int32) # type: ignore
        elif quant in ["QUANT_SMOOTH_ASY"]:
            contents += to_bytes(reserved, dtype=np.uint8) * 2# type: ignore
            zero = 0 #qparams["zero"]
            contents += to_bytes(zero, dtype=np.int32) # type: ignore            
        else:
            raise NotImplemented

        return contents


    def get_scales(self, layer, idx=-1):
        scales = copy.deepcopy(layer.get_scales())
        if isinstance(scales, list):
            scales = scales[idx]
        return scales
    
    
    def get_op(self, layer):
        op = layer.get_ops_instance()
        if isinstance(op, list):
            op = op[0]
        return op
    
    
    def get_asymquant(self, layer):
        zo = np.sum(np.abs(layer.get_scales()[-1]["zo"]))
        
        if zo > 0:
            is_asymquant = True
        else:
            is_asymquant = False
            
        is_asymquant = is_asymquant or "floatquan" in layer.get_ops_setting()["setting"]["method"]    
            
        return is_asymquant
    
    
    def get_perchannelquant(self, layer):
        is_perchannel = isinstance(layer.get_scales()[-1]['out_shift'], np.ndarray)
        is_perchannel = is_perchannel or isinstance(layer.get_scales()[-1]['out_scale'], np.ndarray)
        return is_perchannel
    
    
    def reset_something(self, layer):
        if layer.get_layer_type() in ["globalaveragepool"]:
            if layer.get_scale_type() == "smooth":
                layer.set_scale_type("intscale")
    

    def save(self, layer):
        self.reset_something(layer)
        
        qi_type, i_type, o_type, w_type = self.get_types(layer)
        scales = self.get_scales(layer)
                    
        quant = self.get_pre_quant(layer, qi_type, i_type)
        qparams, quant = self.get_pre_qparams(layer, quant, scales)
        LayerPre = self.get_pre_post(layer, quant=quant, qi_qo_type=qi_type, qparams=qparams)
        
        LayerProcess = self.get_process(layer, i_type, o_type, w_type)
        
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        quant = self.get_post_quant(layer, o_type, qo_type)
        qparams, quant = self.get_post_qparams(layer, quant, scales)
        LayerPost = self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                        
        contents = LayerPre + LayerProcess + LayerPost
        
        return contents
    
@NETWORK_V3.register_module(name="input")
class LAYER_INPUTBLOCK(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_INPUTBLOCK, self).__init__(**kwargs)

    def save(self, layer):
        if self.save_placeholder_params:
            i_type = self.NPU_DataType[layer.get_output_type()[0]] 
        else:
            i_type = self.NPU_DataType[layer.get_input_type()[0]]
            
        if len(layer.get_insert()["feat_o"][0]) == 2:
            H, W = layer.get_insert()["feat_o"][0]
            if [H, W] == [1, 1]:
                o_fmt = self.MatFmt[self.fmt]
            else:
                o_fmt = self.CubeFmt[self.fmt]
        else:
            H, W = 1, 1
            o_fmt = self.MatFmt[self.fmt]

        _, C = layer.get_insert()["out_pad"][0]
        OC = layer.get_insert()["out_align"][0]
        OH, OW = H, W

        input_shape = [1, OH, OW, OC] # uint16
        input_layer_idx = layer.get_idx() # uint16
                
        input_data_type = NpuType_t[i_type].value # type: ignore
        input_data_fmt = LayerFmt_t[o_fmt].value # type: ignore 
        
        contents = bytearray()
        
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

        if self.save_placeholder_params:
            qi_qo_type = i_type
        else:
            qi_qo_type = "NPU_FP32"
        contents += self.get_pre_post(layer, quant="QUANT_NONE", qi_qo_type=qi_qo_type, qparams=dict())
                           
        return contents


@NETWORK_V3.register_module(name="output")
class LAYER_OUPUTBLOCK(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_OUPUTBLOCK, self).__init__(**kwargs)

    def save(self, layer):
        process_scale = layer.get_scale_type()
        is_asymquan = self.get_asymquant(layer)
        is_perchannel = self.get_perchannelquant(layer)
        
        o_fmt = self.CubeFmt[self.fmt]
        
        real_c = layer.get_ops_setting()["attrs"][0]["out_c"]
        OH, OW = layer.get_insert()["feat_o"][0]
        OC = layer.get_insert()["out_align"][0]

        i_type = self.NPU_DataType[layer.get_output_type()[-1]]
        o_type = "NPU_FP32"
        if i_type in ["NPU_INT8", "NPU_INT16"]:
            if is_perchannel:
                perchannel = "_PER_CHN"
            else:
                perchannel = ""
            if is_asymquan:
                quant = "QUANT{}_DEQUANT_ASY".format(perchannel)
            else:
                quant = "QUANT{}_DEQUANT".format(perchannel)
            if process_scale in ["rshiftscale", "rrshiftscale"]:
                output_scale = layer.get_scales()[-1]["fscale"]
            else:
                output_scale = layer.get_scales()[-1]["out_scale"]
            output_zero = layer.get_scales()[-1]["zo"] 
            # dequant with scale: ["rshiftscale", "rrshiftscale"]
            # quant with scale: ["floatscale", "shiftfloatscale", "intscale", "ffloatscale"]
            # if process_scale not in ["rshiftscale", "rrshiftscale"]:
                # output_scale = 1.0 / output_scale                           
        else:
            # if is_asymquan:
            #     quant = self.LayerQuant["asy"]
            # else:
            quant = self.LayerQuant[""]
                       
        output_shape = [1, OH, OW, OC] # uint16 
        output_layer_idx = layer.get_idx() # uint16 
        
        output_data_type = NpuType_t[i_type].value # type: ignore
        output_data_fmt = LayerFmt_t[o_fmt].value # type: ignore
                
        contents = bytearray()
        
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
            contents += to_bytes(content, dtype=dtype) # type: ignore
                       
        if quant in ["QUANT_PER_CHN_DEQUANT", "QUANT_PER_CHN_DEQUANT_ASY"]:
            contents += to_bytes(LayerQuant_t[quant].value, dtype=np.uint8) # type: ignore
            contents += to_bytes(NpuType_t[o_type].value, dtype=np.uint8) # type: ignore
            contents += 2 * to_bytes(reserved, dtype=np.uint8) # type: ignore
            offset = layer.get_w_offset()["fscale_w_offset"][0]
            contents += to_bytes(offset, dtype=np.float32) # type: ignore      
        elif quant in ["QUANT_DEQUANT", "QUANT_DEQUANT_ASY"]:
            contents += to_bytes(LayerQuant_t[quant].value, dtype=np.uint8) # type: ignore
            contents += to_bytes(NpuType_t[o_type].value, dtype=np.uint8) # type: ignore
            contents += 2 * to_bytes(reserved, dtype=np.uint8) # type: ignore
            contents += to_bytes(output_scale, dtype=np.float32) # type: ignore
            if "ASY" in quant:
                contents += to_bytes(output_zero, dtype=np.int32) # type: ignore
        else:
            contents += to_bytes(LayerQuant_t[quant].value, dtype=np.uint8) # type: ignore
            contents += to_bytes(NpuType_t[o_type].value, dtype=np.uint8) # type: ignore
            contents += 2 * to_bytes(reserved, dtype=np.uint8) # type: ignore
                                    
        return contents
    

@NETWORK_V3.register_module(name="splice")
@NETWORK_V3.register_module(name="data")
class LAYER_PLACEHOLDER(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_PLACEHOLDER, self).__init__(**kwargs)


    def get_types(self, layer, idx=0):
        if self.save_placeholder_params:
            qi_type = self.NPU_DataType[layer.get_output_type()[idx]] 
        else:
            qi_type = self.NPU_DataType[layer.get_input_type()[idx]] 
        o_type = i_type = qi_type
        
        # op = self.get_op(layer)
        
        # o_type = self.NPU_DataType[op.bit_select]
            
        return qi_type, i_type, o_type, None
    
    
    def get_post_quant(self, layer, o_type, qo_type):
        # mode in ["", "quant", "dequant"]
        quant = self.get_pre_quant(layer, o_type, qo_type)
        
        # is_perchannel = self.get_perchannelquant(layer)
        # if is_perchannel:
        #     quant = 'perchannel_' + quant        
                         
        return quant
    
        
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
    
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        N = 1
        H, W = layer.get_insert()["feat_o"][0]
        C = layer.get_insert()["out_align"][0]
            
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip( # type: ignore
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                N, H, W, C, 
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16, np.uint16,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
                        
        return LayerProcess
    
    
    def save(self, layer):
        qi_type, i_type, o_type, w_type = self.get_types(layer)
        scales = self.get_scales(layer)
                    
        quant = self.get_pre_quant(layer, qi_type, i_type)
        qparams, quant = self.get_pre_qparams(layer, quant, scales)
        LayerPre = self.get_pre_post(layer, quant=quant, qi_qo_type=qi_type, qparams=qparams)
        
        LayerProcess = self.get_process(layer, i_type, o_type, w_type)
        
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        quant = self.get_post_quant(layer, o_type, qo_type)
        qparams, quant = self.get_post_qparams(layer, quant, scales)
        LayerPost = self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                                
        contents = LayerPre + LayerProcess + LayerPost
        
        return contents
    
            
@NETWORK_V3.register_module(name="conv")
@NETWORK_V3.register_module(name="depthwiseconv")
@NETWORK_V3.register_module(name="convtranspose")
class LAYER_CONV2D(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_CONV2D, self).__init__(**kwargs)
    
    
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
    
        op = self.get_op(layer)
        
        i_fmt = self.CubeFmt[self.fmt]
        w_fmt = self.ConvWFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        H, W = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        FH, FW = layer.get_ops_setting()["attrs"][0]["kernel_shape"]
        if layer.get_layer_type() == "depthwiseconv":
            K = 1
        else:
            K = layer.get_insert()["out_align"][0]

        SH, SW = layer.get_ops_setting()["attrs"][0]["strides"]
        # pad_t, pad_b, pad_l, pad_r = layer.get_ops_setting()['attrs'][0]['pads']
        auto_pad = layer.get_ops_setting()["attrs"][0].get("auto_pad")
        if auto_pad in ["SAME_UPPER", "SAME_LOWER", "VALID"]:
            # op.pads: left, right, top, bottom -> pad_t, pad_l, pad_b, pad_r
            pad_t, pad_l, pad_b, pad_r = op.pads[2], op.pads[0], op.pads[3], op.pads[1]
        else:
            pad_t, pad_l, pad_b, pad_r = layer.get_ops_setting()["attrs"][0]["pads"]

        OH, OW = layer.get_insert()["feat_o"][0]
        if layer.get_first_conv() and self.data_channel_extension:
            pad_t, pad_b = 0, 0
        else:
            delta_h = (OH - 1) * SH + FH - H
            if delta_h > 0:
                pad_b = delta_h - pad_t

        delta_w = (OW - 1) * SW + FW - W
        if delta_w > 0:
            pad_r = delta_w - pad_l

        has_bias = 1 #int(layer.get_ops_setting()["attrs"][0]["bias"])
        split_chn = 0 # means no split in channel axis.
        w_off = layer.get_w_offset()["tmp_offset"][1]
        process_scale = layer.get_scale_type()
        use_table = process_scale in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
        if use_table:
            if o_type == "NPU_INT8":
                act_type = "ACT_LUT8"
                if process_scale in ["shiftfloatscaletable2float"]:
                    act_type += "_FP"
            elif o_type == "NPU_INT16":
                act_type = "ACT_LUT16"
                if process_scale in ["shiftfloatscaletable2float"]:
                    act_type += "_FP"                
            else:
                act_type = "ACT_NONE"
        else:
            act = self.ActivationType[layer.get_ops_setting()["ops_string"][-1]]
            if act in ["ACT_RELU", "ACT_BRELU", "ACT_RELU6"]:
                act_type = act
            else:
                act_type = "ACT_NONE"
            
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        w_type = NpuType_t[w_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        w_fmt = LayerFmt_t[w_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip( # type: ignore
            [
                type, i_type, w_type, o_type,
                i_fmt, w_fmt, o_fmt, 
                H, W, C, 
                FH, FW, K,
                SH, SW,
                pad_t, pad_b, pad_l, pad_r,
                OH, OW, has_bias, split_chn,
                w_off,
            ],
            [
                np.uint32, np.uint8, np.uint8, np.uint8,
                np.uint8, np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16, np.uint8, np.uint8,
                np.uint32,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
            
        ## activation
        # type = Activation_t[act_type].value
        # for value, dtype in zip(
        #     [
        #         type, o_type, o_type,
        #         o_fmt, o_fmt, 
        #         H, W, C, reserved, 
        #     ],
        #     [
        #         np.uint32, np.uint8, np.uint8,
        #         np.uint8, np.uint8,
        #         np.uint16, np.uint16, np.uint16, np.uint16,
        #     ],
        # ):
        #     LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
        if process_scale in ["shiftfloatscaletable2float"]:
            LayerProcess += to_bytes(Activation_t["ACT_NONE"].value, dtype=np.uint8) # type: ignore
            LayerProcess += to_bytes(reserved, dtype=np.uint8) * 3 # type: ignore
        else:
            LayerProcess += to_bytes(Activation_t[act_type].value, dtype=np.uint8) # type: ignore
            LayerProcess += to_bytes(reserved, dtype=np.uint8) * 3 # type: ignore
            if act_type == "ACT_LEAKY_RELU":
                alpha = layer.get_ops_setting()["attrs"][-1]["alpha"]
                LayerProcess += to_bytes(alpha, dtype=np.float32) # type: ignore
            elif act_type == "ACT_BRELU":
                bound = layer.get_ops_setting()["attrs"][-1]["value"]
                LayerProcess += to_bytes(bound, dtype=np.float32) # type: ignore
            # elif act_type == "ACT_RELU6":
                # LayerProcess += to_bytes(6.0, dtype=np.float32) # type: ignore
            elif act_type == "ACT_HARD_SIGMOID":
                alpha = layer.get_ops_setting()["attrs"][-1]["alpha"]
                beta = layer.get_ops_setting()["attrs"][-1]["beta"]
                LayerProcess += to_bytes(alpha, dtype=np.float32) # type: ignore
                LayerProcess += to_bytes(beta, dtype=np.float32) # type: ignore
                                            
            if use_table:
                offset = layer.get_w_offset()["tmp_offset"][2]
                LayerProcess += to_bytes(offset, dtype=np.uint32) # type: ignore
        ## activation      
            
        return LayerProcess

    
    def get_types(self, layer, idx=0):
        process_scale = layer.get_scale_type()
        if layer.get_is_result_layer():
            assert process_scale in [
                "intscale", "intscaleex", "floatscale", "shiftfloatscale",
                "shiftfloatscaletable", "shiftfloatscaletable2float", "ffloatscale",
                "rshiftscale", "rrshiftscale",
            ]
        else:
            assert process_scale in [
                "intscale", "intscaleex", "floatscale", "shiftfloatscale", 
                "shiftfloatscaletable", "shiftfloatscaletable2float", "ffloatscale",
            ]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        
        op = self.get_op(layer)
        
        i_type = self.NPU_DataType[op.bit_select]
        w_type = str(layer.get_qweight().dtype)
        w_type = "NPU_" + w_type.replace("u", "U").replace("int", "INT")
        # o_type = self.NPU_DataType[op.high_bits_calc(op.bit_select)]
        if not op.get_precision():  # layer.get_ops_setting()['setting']['precision']:
            o_type = i_type  # int8 | int16
        else:
            o_type = (
                "NPU_INT32" if i_type == "NPU_INT8" else "NPU_INT64"
            )  # int32 | int64
            
        return qi_type, i_type, o_type, w_type       
    
                
@NETWORK_V3.register_module(name="fc")
class LAYER_FC(LAYER_CONV2D):
    def __init__(self, **kwargs):
        super(LAYER_FC, self).__init__(**kwargs)


    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]

        i_fmt = self.MatFmt[self.fmt]
        w_fmt = self.FcWFmt[self.fmt]
        o_fmt = self.MatFmt[self.fmt]
        M, K, N = (
            1,
            layer.get_insert()["in_align"][0],
            layer.get_insert()["out_align"][0],
        )
        has_bias = 1 #int(layer.get_ops_setting()["attrs"][0]["bias"])
        w_off = layer.get_w_offset()["tmp_offset"][1]
        process_scale = layer.get_scale_type()
        use_table = process_scale in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
        if use_table:
            if o_type == "NPU_INT8":
                act_type = "ACT_LUT8"
                if process_scale in ["shiftfloatscaletable2float"]:
                    act_type += "_FP"
            elif o_type == "NPU_INT16":
                act_type = "ACT_LUT16"
                if process_scale in ["shiftfloatscaletable2float"]:
                    act_type += "_FP"                
            else:
                act_type = "ACT_NONE"
        else:
            act = self.ActivationType[layer.get_ops_setting()["ops_string"][-1]]
            if act in ["ACT_RELU", "ACT_BRELU", "ACT_RELU6"]:
                act_type = act
            else:
                act_type = "ACT_NONE"
                            
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        w_type = NpuType_t[w_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        w_fmt = LayerFmt_t[w_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, w_type, o_type,
                i_fmt, w_fmt, o_fmt, 
                M, K, N, 
                has_bias, reserved, reserved, reserved,
                w_off,
            ],
            [
                np.uint32, np.uint8, np.uint8, np.uint8,
                np.uint8, np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16,
                np.uint8, np.uint8, np.uint8, np.uint8,
                np.uint32,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore

        ## activation
        type = Activation_t[act_type].value
        # for value, dtype in zip(
        #     [
        #         type, o_type, o_type,
        #         o_fmt, o_fmt, 
        #         1, 1, N, reserved, 
        #     ],
        #     [
        #         np.uint32, np.uint8, np.uint8,
        #         np.uint8, np.uint8,
        #         np.uint16, np.uint16, np.uint16, np.uint16,
        #     ],
        # ):
        #     LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
        if process_scale in ["shiftfloatscaletable2float"]:
            LayerProcess += to_bytes(Activation_t["ACT_NONE"].value, dtype=np.uint8) # type: ignore
            LayerProcess += to_bytes(reserved, dtype=np.uint8) * 3 # type: ignore
        else:        
            LayerProcess += to_bytes(Activation_t[act_type].value, dtype=np.uint8) # type: ignore
            LayerProcess += to_bytes(reserved, dtype=np.uint8) * 3 # type: ignore
            if act_type == "ACT_LEAKY_RELU":
                alpha = layer.get_ops_setting()["attrs"][-1]["alpha"]
                LayerProcess += to_bytes(alpha, dtype=np.float32) # type: ignore
            elif act_type == "ACT_BRELU":
                bound = layer.get_ops_setting()["attrs"][-1]["value"]
                LayerProcess += to_bytes(bound, dtype=np.float32) # type: ignore
            # elif act_type == "ACT_RELU6":
                # LayerProcess += to_bytes(6.0, dtype=np.float32) # type: ignore
            elif act_type == "ACT_HARD_SIGMOID":
                alpha = layer.get_ops_setting()["attrs"][-1]["alpha"]
                beta = layer.get_ops_setting()["attrs"][-1]["beta"]
                LayerProcess += to_bytes(alpha, dtype=np.float32) # type: ignore
                LayerProcess += to_bytes(beta, dtype=np.float32) # type: ignore
                                            
            if use_table:
                offset = layer.get_w_offset()["tmp_offset"][2]
                LayerProcess += to_bytes(offset, dtype=np.uint32) # type: ignore      
        ## activation
        
        return LayerProcess
    
          
@NETWORK_V3.register_module(name="relu")
@NETWORK_V3.register_module(name="relu6")
@NETWORK_V3.register_module(name="relux")
@NETWORK_V3.register_module(name="leakyrelu")
@NETWORK_V3.register_module(name="prelu")
@NETWORK_V3.register_module(name="sigmoid")
@NETWORK_V3.register_module(name="swish")
@NETWORK_V3.register_module(name="gelu")
@NETWORK_V3.register_module(name="tanh")
@NETWORK_V3.register_module(name="hardsigmoid")
@NETWORK_V3.register_module(name="hardtanh")
@NETWORK_V3.register_module(name="hardswish")
@NETWORK_V3.register_module(name="hardshrink")
class LAYER_ACTIVATION(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_ACTIVATION, self).__init__(**kwargs)


    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        
        i_fmt = self.MatFmt[self.fmt]
        o_fmt = self.MatFmt[self.fmt]
        H, W = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        N = 1
        real_c = layer.get_insert()["in_pad"][0][-1]
        
        act_type = "ACT_NONE"
        if layer.get_scale_type() == "table":
            if o_type == "NPU_INT8":
                act_type = "ACT_LUT8"
            elif o_type == "NPU_INT16":
                act_type = "ACT_LUT16"
        else:
            act_type = self.ActivationType[layer.get_layer_type()]
    
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, o_type, o_type,
                o_fmt, o_fmt, 
                N, H, W, C, 
                real_c, reserved,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16, 
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
        LayerProcess += to_bytes(Activation_t[act_type].value, dtype=np.uint8) # type: ignore
        LayerProcess += to_bytes(reserved, dtype=np.uint8) * 3 # type: ignore
        if act_type in ["ACT_LUT8", "ACT_LUT16"]:
            offset = layer.get_w_offset()["w_offset"]
            LayerProcess += to_bytes(offset, dtype=np.uint32) # type: ignore
        else:
            if layer.get_layer_type() in ["relux"]:
                bound = layer.get_ops_setting()['attrs'][0]['max']
                LayerProcess += to_bytes(bound, dtype=np.float32) # type: ignore
            elif layer.get_layer_type() == "leakyrelu":
                alpha = layer.get_ops_setting()["attrs"][0]["alpha"]
                LayerProcess += to_bytes(alpha, dtype=np.float32) # type: ignore
            elif layer.get_layer_type() == "hardsigmoid":
                alpha = layer.get_ops_setting()["attrs"][0]["alpha"]
                beta = layer.get_ops_setting()["attrs"][0]["beta"]
                LayerProcess += to_bytes(alpha, dtype=np.float32) # type: ignore
                LayerProcess += to_bytes(beta, dtype=np.float32) # type: ignore
                    
        return LayerProcess
        
        
    def get_types(self, layer, idx=0):
        process_scale = layer.get_scale_type()
        layer_type = layer.get_layer_type()
        if layer_type in ["relu", "relux", "relu6"]:
            assert process_scale in ["float", "table", "preintscale"]
        else:
            assert process_scale in ["float", "table"]
            
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        
        if  process_scale == "float":
            i_type = o_type = "NPU_FP32"
        elif process_scale in ["table", "preintscale"]:
            i_type = qi_type
            o_type = self.NPU_DataType[
                layer.get_output_type()[-1]
            ]
        else:
            raise NotImplemented
        
        return qi_type, i_type, o_type, None
    
    
    def save(self, layer):
        qi_type, i_type, o_type, w_type = self.get_types(layer)
        scales = self.get_scales(layer)
                    
        scale_type = layer.get_scale_type()
        if scale_type in ["preintscale"]:
            quant = self.get_post_quant(layer, qi_type, i_type)
            qparams, quant = self.get_post_qparams(layer, quant, scales)
        else:
            quant = self.get_pre_quant(layer, qi_type, i_type)
            qparams, quant = self.get_pre_qparams(layer, quant, scales)
        LayerPre = self.get_pre_post(layer, quant=quant, qi_qo_type=qi_type, qparams=qparams)
        
        LayerProcess = self.get_process(layer, i_type, o_type, w_type)
        
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        if scale_type in ["preintscale"]:
            quant = self.get_pre_quant(layer, o_type, qo_type)
            qparams, quant = self.get_pre_qparams(layer, quant, scales)
        else:
            quant = self.get_post_quant(layer, o_type, qo_type)
            qparams, quant = self.get_post_qparams(layer, quant, scales)            
        LayerPost = self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                        
        contents = LayerPre + LayerProcess + LayerPost
        
        return contents    


@NETWORK_V3.register_module(name="reducemax")
@NETWORK_V3.register_module(name="reducemin")
@NETWORK_V3.register_module(name="reducemean")
@NETWORK_V3.register_module(name="reducesum")
@NETWORK_V3.register_module(name="reduceprod")
class LAYER_REDUCE(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_REDUCE, self).__init__(**kwargs)
        
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
    
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]

        N = 1
        C = layer.get_insert()["in_align"][0]
        H, W = layer.get_insert()["feat_o"][0]
        operation = ReduceType_t[layer.get_layer_type().upper().replace("REDUCE", "REDUCE_")].value
        axis = layer.get_ops_setting()["attrs"][0]["axes"][0]
        if axis == 1:
            axis = DimType_t["DIM_C"].value
        else:
            axis = DimType_t["DIM_N"].value
        keepdims = True if layer.get_ops_setting()["attrs"][0]["keepdims"] else False
        
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                N, H, W, C, 
                operation, axis, keepdims, reserved,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint8, np.uint8, np.uint8, np.uint8,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
                
        return LayerProcess
          
            
    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["float"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        i_type = o_type = "NPU_FP32"
    
        return qi_type, i_type, o_type, None
            
            
@NETWORK_V3.register_module(name="transpose")
class LAYER_TRANSPOSE(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_TRANSPOSE, self).__init__(**kwargs)
        
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
    
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]

        N = 1
        C = layer.get_insert()["in_align"][0]
        H, W = layer.get_insert()["feat_o"][0]
        perm = layer.get_ops_setting()["attrs"][0]["perm"]
        real_c = layer.get_insert()["in_pad"][0][-1]
        
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                N, H, W, C, 
                real_c, reserved,
                perm,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16,
                np.uint8,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
                
        return LayerProcess
          
            
    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["smooth"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        i_type = o_type = qi_type
    
        return qi_type, i_type, o_type, None
    
    
@NETWORK_V3.register_module(name="reshape")
class LAYER_RESHAPE(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_RESHAPE, self).__init__(**kwargs)
        
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
    
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]

        N = 1
        C = layer.get_insert()["in_align"][0]
        H, W = layer.get_insert()["feat_o"][0]
        shape = layer.get_ops_setting()["attrs"][0]["shape"]
        if len(shape) == 1:
            ON = shape
            OC, OH, OW = 1, 1, 1        
        elif len(shape) == 2:
            ON, OC = shape
            OH, OW = 1, 1
            o_fmt = self.MatFmt[self.fmt]
        elif len(shape) == 3:
            ON, OC, OH = shape
            OW = 1
        else:
            ON, OH, OW, OC = shape
            
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                N, H, W, C, 
                ON, OH, OW, OC,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16, np.uint16, np.uint16,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
                
        return LayerProcess
          
            
    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["smooth"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        i_type = o_type = qi_type
    
        return qi_type, i_type, o_type, None
        
                
@NETWORK_V3.register_module(name="pad")
class LAYER_PAD(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_PAD, self).__init__(**kwargs)
        
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
    
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]

        N = 1
        C = layer.get_insert()["in_align"][0]
        H, W = layer.get_insert()["feat_o"][0]
        pads = layer.get_ops_setting()["attrs"][0]["pads"]
        # shape = layer.get_ops_setting()["attrs"][0]["shape"]
        # if len(shape) == 1:
        #     ON = shape
        #     OC, OH, OW = 1, 1, 1        
        # elif len(shape) == 2:
        #     ON, OC = shape
        #     OH, OW = 1, 1
        #     o_fmt = self.MatFmt[self.fmt]
        # elif len(shape) == 3:
        #     ON, OC, OH = shape
        #     OW = 1
        # else:
        #     ON, OH, OW, OC = shape
            
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                N, H, W, C, 
                pads,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint16,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
                
        return LayerProcess
          
            
    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["smooth"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        i_type = o_type = qi_type
    
        return qi_type, i_type, o_type, None
    
                    
@NETWORK_V3.register_module(name="maxpool")
@NETWORK_V3.register_module(name="averagepool")
@NETWORK_V3.register_module(name="globalaveragepool")
class LAYER_POOL(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_POOL, self).__init__(**kwargs)

    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        pool_type = self.PoolType[layer.get_layer_type()]
        pool_type = Pool_t[pool_type].value
        H, W = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]

        if layer.get_layer_type() == "globalaveragepool":
            FH, FW = layer.get_in_data()[0]["output"].shape[2:]
        else:
            FH, FW = layer.get_ops_setting()["attrs"][0]["kernel_shape"]

        auto_pad = layer.get_ops_setting()["attrs"][0].get("auto_pad")
        if layer.get_layer_type() == "globalaveragepool":
            SH, SW = 1, 1
            pad_l, pad_t, pad_r, pad_b = 0, 0, 0, 0
        else:
            SH, SW = layer.get_ops_setting()["attrs"][0]["strides"]
            # pad_t, pad_b, pad_l, pad_r = layer.get_ops_setting()['attrs'][0]['pads']
            if auto_pad in ["NOTSET"]:
                pad_l, pad_t, pad_r, pad_b = [0, 0, 0, 0]
            else:
                pad_l, pad_t, pad_r, pad_b = layer.get_ops_setting()["attrs"][0]["pads"]
        OH, OW = layer.get_insert()["feat_o"][0]

        delta_h = (OH - 1) * SH + FH - H
        if delta_h > 0:
            pad_b = delta_h - pad_t

        delta_w = (OW - 1) * SW + FW - W
        if delta_w > 0:
           pad_r = delta_w - pad_l

        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        pool_scale = reserved
        if layer.get_layer_type() == "averagepool":
            pool_scale = layer.get_scales()[0]["hw_out_scale"]
        
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt,
                pool_type, pool_scale, 
                H, W, C, 
                FH, FW,
                SH, SW,
                pad_t, pad_b, pad_l, pad_r,
                OH, OW,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16,
                np.uint16, np.uint16,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
            
        return LayerProcess
                    
    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["float", "smooth", "intscale"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        
        op = self.get_op(layer)
        i_type = self.NPU_DataType[op.bit_select]
        process_scale = layer.get_scale_type()
        if process_scale == "float":
            i_type = o_type = "NPU_FP32"
        elif process_scale in ["smooth", "intscale", "preintscale"]:
            i_type = qi_type
            o_type = self.NPU_DataType[
                layer.get_output_type()[-1]
            ]
        else:
            raise NotImplemented
        
        return qi_type, i_type, o_type, None
    
    def save(self, layer):
        scales = self.get_scales(layer)
        if "hw_out_scale" in scales.keys():
            if scales["hw_out_scale"] <= 1:
                layer.set_scale_type("intscale")
            else:
                layer.set_scale_type("smooth")
        return NetworkV3.save(self, layer)
        
        
@NETWORK_V3.register_module(name="mul")
@NETWORK_V3.register_module(name="pmul")
@NETWORK_V3.register_module(name="add")
@NETWORK_V3.register_module(name="sub")
class LAYER_EWS(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_EWS, self).__init__(**kwargs)

    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["preintscale", "preintscaleex", "intscale", "floatscale", "float", "shiftfloatscaletable"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        if layer.get_scale_type() == "float":
            i_type = o_type = "NPU_FP32"
        else:
            i_type = o_type = qi_type
        
        return qi_type, i_type, o_type, None
    
    
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
                    
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        operation = self.ElementWiseType[layer.get_layer_type()]
        operation = EleWiseType_t[operation].value
        H, W = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        input_len = H * W * C
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                operation, reserved, input_len, 
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8, np.uint32,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
            
        return LayerProcess
    
                
    def save(self, layer):
        qi_type, i_type, o_type, w_type = self.get_types(layer)
        input_len = len(layer.get_in_data())
        
        layer_type = layer.get_layer_type()
        LayerPre = bytearray()
        for idx in range(input_len):
            if 'pmul' in layer_type:  
                scales = dict()
                quant = self.get_pre_quant(layer, qi_type, i_type)
                qparams, quant = self.get_pre_qparams(layer, quant, scales)
            else:
                scales = self.get_scales(layer, idx=idx)
                quant = self.get_post_quant(layer, qi_type, i_type, idx)
                qparams, quant = self.get_post_qparams(layer, quant, scales)
            LayerPre += self.get_pre_post(layer, quant=quant, qi_qo_type=qi_type, qparams=qparams)
        
        LayerProcess = self.get_process(layer, i_type, o_type, w_type)
        
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        if 'pmul' in layer_type:
            scales = self.get_scales(layer, idx=0)  
            quant = self.get_post_quant(layer, o_type, qo_type)
            qparams, quant = self.get_post_qparams(layer, quant, scales)
        else:
            scales = dict()
            quant = self.get_pre_quant(layer, o_type, qo_type)
            qparams, quant = self.get_pre_qparams(layer, quant, scales)
        LayerPost = self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                        
        contents = LayerPre + LayerProcess + LayerPost
        
        return contents
    

@NETWORK_V3.register_module(name="cmul")
@NETWORK_V3.register_module(name="cadd")
@NETWORK_V3.register_module(name="csub")
class LAYER_CWS(LAYER_EWS):
    def __init__(self, **kwargs):
        super(LAYER_CWS, self).__init__(**kwargs)


    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
         
        c_type = i_type
        operation = self.ChnWiseType[layer.get_layer_type()]
        operation = ChnWiseType_t[operation].value
        H, W = layer.get_insert()["feat_o"][0]
        C = layer.get_insert()["out_align"][0]                    
        i_fmt = self.CubeFmt[self.fmt]
        c_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
                
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        c_type = NpuType_t[c_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        c_fmt = LayerFmt_t[c_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, c_type, o_type,
                i_fmt, c_fmt, o_fmt, 
                H, W, C,
                operation, 
                reserved, reserved, reserved,
            ],
            [
                np.uint32, np.uint8, np.uint8, np.uint8,
                np.uint8, np.uint8, np.uint8, 
                np.uint16, np.uint16, np.uint16, 
                np.uint8, 
                np.uint8, np.uint8, np.uint8, 
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
            
        return LayerProcess
    
                
    def save(self, layer):
        qi_type, i_type, o_type, w_type = self.get_types(layer)
        input_len = len(layer.get_in_data())
        
        layer_type = layer.get_layer_type()
        LayerPre = bytearray()
        for idx in range(input_len):
            if 'cmul' in layer_type:  
                scales = dict()
                quant = self.get_pre_quant(layer, qi_type, i_type)
                qparams, quant = self.get_pre_qparams(layer, quant, scales)
            else:
                scales = self.get_scales(layer, idx=idx)
                quant = self.get_post_quant(layer, qi_type, i_type, idx)
                qparams, quant = self.get_post_qparams(layer, quant, scales)
            LayerPre += self.get_pre_post(layer, quant=quant, qi_qo_type=qi_type, qparams=qparams)
        
        LayerProcess = self.get_process(layer, i_type, o_type, w_type)
        
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        if 'cmul' in layer_type:
            scales = self.get_scales(layer, idx=0)  
            quant = self.get_post_quant(layer, o_type, qo_type)
            qparams, quant = self.get_post_qparams(layer, quant, scales)
        else:
            scales = dict()
            quant = self.get_pre_quant(layer, o_type, qo_type)
            qparams, quant = self.get_pre_qparams(layer, quant, scales)
        LayerPost = self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                        
        contents = LayerPre + LayerProcess + LayerPost
        
        return contents
    

@NETWORK_V3.register_module(name="matmul")
class LAYER_MATMUL(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_MATMUL, self).__init__(**kwargs)

    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["preintscale", "intscale", "floatscale", "float", "shiftfloatscaletable"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        if layer.get_scale_type() == "float":
            i_type = o_type = "NPU_FP32"
        else:
            i_type = o_type = qi_type
        
        return qi_type, i_type, o_type, None
    
    
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
                    
        type = LayerType_t[LayerType].value
        i_type0 = i_type1 = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        i_fmt0 = i_fmt1 = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        
        N0, N1 = 1, 1
        H0, W0 = layer.get_insert()["feat_i"][0]
        H1, W1 = layer.get_insert()["feat_i"][1]
        C0 = C1 = layer.get_insert()["out_align"][0] 
        
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type0, i_type1, o_type,
                i_fmt0, i_fmt1, o_fmt, 
                N0, H0, W0, C0,
                N1, H1, W1, C1,
                reserved,
            ],
            [
                np.uint32, np.uint8, np.uint8, np.uint8,
                np.uint8, np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint16,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
            
        return LayerProcess
    
                
    def save(self, layer):
        qi_type, i_type, o_type, w_type = self.get_types(layer)
        input_len = len(layer.get_in_data())
        
        layer_type = layer.get_layer_type()
        LayerPre = bytearray()
        for idx in range(input_len):
            scales = self.get_scales(layer, idx=-1)
            quant = self.get_pre_quant(layer, qi_type, i_type)
            qparams, quant = self.get_pre_qparams(layer, quant, scales)
            LayerPre += self.get_pre_post(layer, quant=quant, qi_qo_type=qi_type, qparams=qparams)
        
        LayerProcess = self.get_process(layer, i_type, o_type, w_type)
        
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        scales = self.get_scales(layer)  
        quant = self.get_post_quant(layer, o_type, qo_type)
        qparams, quant = self.get_post_qparams(layer, quant, scales)
        LayerPost = self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                        
        contents = LayerPre + LayerProcess + LayerPost
        
        return contents
    
    
@NETWORK_V3.register_module(name="concat")
class LAYER_CONCAT(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_CONCAT, self).__init__(**kwargs)

    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["preintscale", "preintscaleex", "shiftfloatscaletable", "float"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        if layer.get_scale_type() == "float":
            i_type = o_type = "NPU_FP32"
        else:
            i_type = o_type = qi_type
        
        return qi_type, i_type, o_type, None
    
    
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
         
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        H, W = layer.get_insert()["feat_i"][0]

        C = [a for a in layer.get_insert()["in_align"]]
        while len(C) < self.CONCAT_MAX_IN:
            C.append(0)

        real_c = [b for a, b in layer.get_insert()["in_pad"]]
        while len(real_c) < self.CONCAT_MAX_IN:
            real_c.append(0)

        OC = layer.get_insert()["out_align"][0]
        real_oc = np.sum(real_c)

        if layer.get_insert()["is_align"]:
            real_c, real_oc = C, OC
                            
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                H, W,
                C, real_c,
                OC, real_oc,                
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8, 
                np.uint16, np.uint16, 
                np.uint16, np.uint16, 
                np.uint16, np.uint16, 
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore

        return LayerProcess
    
                
    def save(self, layer):
        qi_type, i_type, o_type, w_type = self.get_types(layer)
        input_len = len(layer.get_in_data())
        
        layer_type = layer.get_layer_type()
        LayerPre = bytearray()
        for idx in range(input_len):
            scales = self.get_scales(layer, idx=idx)
            quant = self.get_post_quant(layer, qi_type, i_type, idx)
            if scales["out_scale"] == 1:
                quant = ""
            qparams, quant = self.get_post_qparams(layer, quant, scales)
            LayerPre += self.get_pre_post(layer, quant=quant, qi_qo_type=qi_type, qparams=qparams)
        
        LayerProcess = self.get_process(layer, i_type, o_type, w_type)
        
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        scales = dict()
        quant = self.get_pre_quant(layer, o_type, qo_type)
        qparams, quant = self.get_pre_qparams(layer, quant, scales)
        LayerPost = self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                
        contents = LayerPre + LayerProcess + LayerPost

        return contents


@NETWORK_V3.register_module(name="shuffle_only")
class LAYER_SHUFFLE(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_SHUFFLE, self).__init__(**kwargs)

    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
    
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]

        # OC = layer.get_insert()["out_align"][0]
        # real_oc = np.sum(real_c)
        
        axis = 1
        N = 1
        real_c = [b for a, b in layer.get_insert()["in_pad"]]
        segments = len(real_c)
        while len(real_c) < 4:
            real_c.append(0)
        C = [a for a in layer.get_insert()["in_align"]]
        C = [np.sum(C[:i + 1]) for i in range(len(C))]
        while len(C) < 4:
            C.append(0)
        H, W = layer.get_insert()["feat_o"][0]

        real_oc = [b for a, b in layer.get_insert()["out_pad"]]
        while len(real_oc) < 4:
            real_oc.append(0)
        OC = [a for a in layer.get_insert()["out_align"]]
        OC = [np.sum(OC[:i + 1]) for i in range(len(OC))]
        while len(OC) < 4:
            OC.append(0)
                                    
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                N, H, W, 
                C, real_c,
                OC, real_oc,
                axis, segments,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16,
                np.uint16, np.uint16,
                np.uint8, np.uint8,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
                
        return LayerProcess
          
            
    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["smooth"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        i_type = o_type = qi_type
    
        return qi_type, i_type, o_type, None


@NETWORK_V3.register_module(name="split")
class LAYER_SPLIT(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_SPLIT, self).__init__(**kwargs)

    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["preintscale", "preintscaleex", "shiftfloatscaletable", "float"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        if layer.get_scale_type() == "float":
            i_type = o_type = "NPU_FP32"
        else:
            i_type = o_type = qi_type
        
        return qi_type, i_type, o_type, None
    
    
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
         
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        N = 1
        H, W = layer.get_insert()["split"]["feat_i"][0]
        operation = 1 # 1:"SPLIT" | 2:"SPLIT_TO_SEQUENCE"
        axis = 1
        num = self.output_len
        pad_value = 0
        
        C = layer.get_insert()["split"]["in_align"][0]
        real_c = np.sum([b - a for a, b in layer.get_insert()["split"]["in_pad"]])

        # split_ids = [v for _, v in layer.get_insert()['split_ids'].items()]

        split = OC = layer.get_insert()["split"]["out_align"]
        # OC = [OC[id] for id in split_ids]
        # while len(OC) < self.SPLIT_MAX_OUT:
            # OC.append(0)

        real_oc = [b - a for a, b in layer.get_insert()["split"]["out_pad"]]
        # real_oc = [real_oc[id] for id in split_ids]
        # while len(real_oc) < self.SPLIT_MAX_OUT:
            # real_oc.append(0)
                            
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                N, H, W, C,
                operation, axis,
                num, pad_value, split,                
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,  
                np.uint16, np.uint16, np.uint16, np.uint16, 
                np.uint8, np.uint8, 
                np.uint16, np.uint32, np.uint16, 
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
            
        return LayerProcess
    
                
    def save(self, layer):
        qi_type, i_type, o_type, w_type = self.get_types(layer)
        self.output_len = len(layer.get_out_data())
        
        scales = dict()
        quant = self.get_pre_quant(layer, qi_type, i_type)
        qparams, quant = self.get_pre_qparams(layer, quant, scales)
        LayerPre = self.get_pre_post(layer, quant=quant, qi_qo_type=qi_type, qparams=qparams)
        
        LayerProcess = self.get_process(layer, i_type, o_type, w_type)
        
        LayerPost = bytearray()
        for idx in range(self.output_len):
            qo_type = self.NPU_DataType[layer.get_output_type()[idx]]
            scales = self.get_scales(layer, idx=idx)
            quant = self.get_post_quant(layer, o_type, qo_type, idx)
            if scales["out_scale"] == 1:
                quant = ""
            qparams, quant = self.get_post_qparams(layer, quant, scales)
            LayerPost += self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                                            
        contents = LayerPre + LayerProcess + LayerPost
        
        return contents


@NETWORK_V3.register_module(name="shuffle")
class LAYER_CONCAT_SHUFFLE_SPLIT(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_CONCAT_SHUFFLE_SPLIT, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = self.get_io_len(layer)

        qparams = []
        scales = copy.deepcopy(layer.get_scales()[0])
        for param in scales:
            qparam = []
            for k, v in param.items():
                if k in ["out_shift", "out_scale", "int_scale"]:
                    qparam.append(v)
            qparams.append(qparam)

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]

        LayerPre = "{" + str(in_len) + "," + str(out_len) + ","
        LayerPre += "{"
        for i in range(self.CONCAT_SHUFFLE_SPLIT_MAX_IN):
            if i < in_len:
                quant = self.LayerQuant[layer.get_scale_type()]
                qi_type = self.NPU_DataType[layer.get_input_type()[i]]
                quant_u = self.LayerPrePost[quant]
                i_type = o_type = qi_type
                LayerPre += "{"
                for content in [
                    quant,
                    qi_type,
                    ".quant_u.{}={}".format(quant_u, qparams[i]),
                ]:
                    LayerPre = _write(LayerPre, content)
                LayerPre += "},"
            else:
                quant = self.LayerQuant[""]
                qi_type = self.NPU_DataType[""]
                quant_u = self.LayerPrePost[quant]
                LayerPre += "{"
                for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
                    LayerPre = _write(LayerPre, content)
                LayerPre += "},"
        LayerPre += "}"

        # i_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_input_type(
        #     )].__name__  # self.NPU_DataType[layer.get_input_type()]
        # o_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_output_type(
        #     )].__name__  # self.NPU_DataType[layer.get_output_type()]
        # i_type = 'NPU_' + i_type.replace('u', 'U').replace('int', 'INT')
        # o_type = 'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        H, W = layer.get_insert()["feat_i"][0]

        IC = layer.get_insert()["split"]["in_align"]
        sec_num = len(IC)
        while len(IC) < self.CONCAT_SHUFFLE_SPLIT_MAX_IN:
            IC.append(0)

        real_ic = [b for a, b in layer.get_insert()["split"]["in_pad"]]
        while len(real_ic) < self.CONCAT_SHUFFLE_SPLIT_MAX_IN:
            real_ic.append(0)

        OC = layer.get_insert()["split"]["out_align"]
        while len(OC) < self.CONCAT_SHUFFLE_SPLIT_MAX_OUT:
            OC.append(0)

        real_oc = [b for a, b in layer.get_insert()["split"]["out_pad"]]
        while len(real_oc) < self.CONCAT_SHUFFLE_SPLIT_MAX_OUT:
            real_oc.append(0)

        Layer = ""
        for content in [
            i_type, # type: ignore
            o_type, # type: ignore
            i_fmt,
            o_fmt,
            H,
            W,
            self.list2Cstyle(IC),
            self.list2Cstyle(real_ic),
            sec_num,
            self.list2Cstyle(OC),
            self.list2Cstyle(real_oc),
        ]:
            Layer = _write(Layer, content)

        qparams = []
        scales = copy.deepcopy(layer.get_scales()[-1])
        for param in scales:
            qparam = []
            for k, v in param.items():
                if k in ["out_shift", "out_scale", "int_scale"]:
                    qparam.append(v)
            qparams.append(qparam)

        split_ids = layer.get_insert()["split_ids"]

        LayerPost = "{"
        for i in range(self.CONCAT_SHUFFLE_SPLIT_MAX_OUT):
            if i < out_len:
                param_id = split_ids[layer.get_output_idx()[i]]

                quant = self.LayerQuant[layer.get_scale_type()]
                qo_type = self.NPU_DataType[layer.get_output_type()[param_id]]
                quant_u = self.LayerPrePost[quant]
                LayerPost += "{"
                for content in [
                    quant,
                    qo_type,
                    ".quant_u.{}={}".format(quant_u, qparams[param_id]),
                ]:
                    LayerPost = _write(LayerPost, content)
                LayerPost += "},"
            else:
                quant = self.LayerQuant[""]
                qo_type = self.NPU_DataType[""]
                quant_u = self.LayerPrePost[quant]
                LayerPost += "{"
                for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, 0)]:
                    LayerPost = _write(LayerPost, content)
                LayerPost += "},"
        LayerPost += "},"
        LayerPost += "}"

        contents = bytearray()

        return contents


@NETWORK_V3.register_module(name="resize")
class LAYER_RESIZE(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_RESIZE, self).__init__(**kwargs)
    
    def get_process(self, layer, i_type, o_type, w_type):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
    
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]

        IH, IW = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        OH, OW = layer.get_insert()["feat_o"][0]
        mode = layer.get_ops_setting()["attrs"][0]["mode"]
        method = self.ResizeMethod[mode]  # 'RESIZE_BILINEAR'
        # if "INT" in i_type:
        #     method += "_FIXED_POINT"
        trans_mode = layer.get_ops_setting()["attrs"][0].get("coordinate_transformation_mode")
        if trans_mode:
            trans_mode = "RESIZE_{}".format(trans_mode) 
        trans_mode = trans_mode.upper()    
        # trans_mode = ResizeCoordinateTransMode_t[trans_mode].value
        round_mode_ = layer.get_ops_setting()["attrs"][0].get("nearest_mode")
        round_mode = mode + "_ROUND"
        if round_mode_:
            round_mode += "_{}".format(round_mode_)
        round_mode = round_mode.upper()
        
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                IH, IW, C, 
                OH, OW, C,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16,
                np.uint16, np.uint16, np.uint16,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
        
        LayerProcess += to_bytes(ResizeMethod_t[method].value, dtype=np.uint8) # type: ignore
        LayerProcess += to_bytes(ResizeCoordinateTransMode_t[trans_mode].value, dtype=np.uint8) # type: ignore
        LayerProcess += to_bytes(ResizeNearestRoundMode_t[round_mode].value, dtype=np.uint8) # type: ignore
        LayerProcess += to_bytes(reserved, dtype=np.uint8) # type: ignore
        exclude_outside = 0 # default
        extrapolation_value = 0 # default
        LayerProcess += to_bytes(exclude_outside, dtype=np.int32) # type: ignore
        LayerProcess += to_bytes(extrapolation_value, dtype=np.float32) # type: ignore
        
        return LayerProcess
          
            
    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["float", "smooth", "preintscale", "preintscaleex"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        if layer.get_scale_type() in ["float"]:
            i_type = o_type = "NPU_FP32"
        elif layer.get_scale_type() in ["smooth", "preintscale", "preintscaleex"]:
            i_type = o_type = qi_type

        return qi_type, i_type, o_type, None
    
        
@NETWORK_V3.register_module(name="softmax")
class LAYER_SOFTMAX(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_SOFTMAX, self).__init__(**kwargs)

    def save(self, layer):
        # LayerType = self.layer_map_inv[layer.get_layer_type()]
        # LayerInfo = self.LayerInstance[LayerType]
        # qi_type = self.NPU_DataType[layer.get_input_type()[0]]

        # # if qi_type in ["NPU_INT8", "NPU_INT16"] and layer.get_scale_type() == "float":
        # #     # is_perchannel = isinstance(layer.get_scales()[0]['out_shift'], np.ndarray)
        # #     is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        # #     quant = ""
        # #     # if is_perchannel:
        # #     # quant += 'perchannel_'
        # #     quant += "dequant"
        # #     if is_asymquan:
        # #         quant += "_asy"
        # #     quant = self.LayerQuant[quant]
        # #     # if is_perchannel:
        # #     #     qparams = [layer.get_w_offset()['tmp_offset'][0]] #offset
        # #     # else:
        # #     if is_asymquan:
        # #         qparams = [
        # #             layer.get_in_scale()["scale"],
        # #             layer.get_in_scale()["zero_point"],
        # #         ]  # scale, zero
        # #     else:
        # #         qparams = [layer.get_in_scale()["scale"]]  # scale
        # #     qparams = self.list2Cstyle(qparams)
        # # else:
        # #     quant = self.LayerQuant[""]
        # #     qparams = 0
        # op = layer.get_ops_instance()
        # op = op[0] if isinstance(op, list) else op
        # i_type = o_type = "NPU_FP32"
        # mode = self.get_quant_mode(qi_type, i_type)
        # if mode != "":
        #     quant, qparams = self.get_quant(layer, layer.get_in_scale(), mode)
        # else:
        #     quant = self.LayerQuant[""]
        #     qparams = 0
        # quant_u = self.LayerPrePost[quant]

        # LayerPre = bytearray()

        # # i_type = layer.get_ops_setting()['setting']['bits_dict'][
        # #     layer.get_input_type(
        # #     )].__name__  # self.NPU_DataType[layer.get_input_type()]
        # # o_type = layer.get_ops_setting()['setting']['bits_dict'][
        # #     layer.get_output_type(
        # #     )].__name__  # self.NPU_DataType[layer.get_output_type()]
        # # i_type = 'NPU_' + i_type.replace('u', 'U').replace('int', 'INT')
        # # o_type = 'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')
        # i_fmt = self.CubeFmt[self.fmt]
        # o_fmt = self.CubeFmt[self.fmt]

        # IH, IW = layer.get_insert()["feat_i"][0]
        # C = layer.get_insert()["in_align"][0]
        # axis = layer.get_ops_setting()["attrs"][0]["axis"]

        # LayerProcess = bytearray()

        # qo_type = self.NPU_DataType[layer.get_output_type()[0]]
        # if layer.get_is_result_layer():
        #     quant = get_last_layer_quant(layer)
        # else:
        #     if (
        #         qo_type in ["NPU_INT8", "NPU_INT16"]
        #         and layer.get_scale_type() == "float"
        #     ):
        #         quant = self.LayerQuant["quant"]
        #         qparams = [layer.get_scale()[0]["scale"]]
        #         qparams = self.list2Cstyle(qparams)
        #     else:
        #         quant = self.LayerQuant[""]
        #         qparams = 0
        # quant_u = self.LayerPrePost[quant]

        # LayerPost = self.get_pre_post(layer, quant=quant, qi_qo_type=qo_type, qparams=qparams)
                
        # contents = LayerPre + LayerProcess + LayerPost
        contents = bytearray()
        
        return contents


# @NETWORK_V3.register_module(name="reshape")
# class LAYER_RESHAPE(NetworkV3):
#     def __init__(self, **kwargs):
#         super(LAYER_RESHAPE, self).__init__(**kwargs)


#     def get_process(self, layer, i_type, o_type, w_type):
#         pass


#     def get_types(self, layer, idx=0):
#         pass

    
#     def save(self, layer):
#         in_len, out_len = self.get_io_len(layer)

#         qparams = []

#         LayerType = self.layer_map_inv[layer.get_layer_type()]
#         LayerInfo = self.LayerInstance[LayerType]

#         LayerPre = "{"
#         LayerPre += "{"
#         quant = self.LayerQuant[""]
#         qi_type = self.NPU_DataType[layer.get_input_type()[0]]
#         quant_u = self.LayerPrePost[quant]
#         for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
#             LayerPre = _write(LayerPre, content)
#         LayerPre += "}"

#         i_type = self.NPU_DataType[layer.get_input_type()[0]]
#         o_type = self.NPU_DataType[layer.get_output_type()[0]]
#         i_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]
#         o_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]

#         # _, C, H, W = layer.get_in_data()[0]['output'].shape
#         if "split" in layer.get_insert().keys():
#             C = layer.get_insert()["split"]["in_align"][0]
#             H, W = layer.get_insert()["split"]["feat_i"][0]
#         else:
#             C = layer.get_insert()["in_align"][0]
#             H, W = layer.get_insert()["feat_i"][0]

#         Layer = ""
#         for content in [i_type, o_type, i_fmt, o_fmt, H, W, C]:
#             Layer = _write(Layer, content)

#         LayerPost = "{"
#         quant = self.LayerQuant[""]
#         qi_type = self.NPU_DataType[layer.get_output_type()[0]]
#         quant_u = self.LayerPrePost[quant]
#         for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
#             LayerPost = _write(LayerPost, content)
#         LayerPost += "},"
#         LayerPost += "}"

#         contents = bytearray()

#         return contents


@NETWORK_V3.register_module(name="lstm")
class LAYER_LSTM(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_LSTM, self).__init__(**kwargs)

    def save(self, layer):
        # in_len, out_len = 1, 1 #self.get_io_len(layer)

        # qparams = []
        # for param in layer.get_scales()[0]:
        #     qparam = []
        #     # for k, v in param.items():
        #     #     if k in ['out_shift', 'out_scale', 'int_scale']:
        #     #         qparam.append(v)
        #     qparams.append(qparam)

        # LayerType = self.layer_map_inv[layer.get_layer_type()]
        # LayerInfo = self.LayerInstance[LayerType]

        # lstm_q = "LSTM_QUANT_I_H_DIFF"
        # LayerPre = "{" + lstm_q + ","
        # LayerPre += "{"
        # quant = self.LayerQuant[""]
        # qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        # quant_u = self.LayerPrePost[quant]
        # qparams = 0
        # if qi_type in ["NPU_FP32", "NPU_FP64"]:
        #     # is_perchannel = isinstance(layer.get_scales()[0]["out_shift"], np.ndarray)
        #     # is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        #     # quant = ""
        #     # if is_perchannel:
        #     #     quant += "perchannel_"
        #     # quant += "quant"
        #     # if is_asymquan:
        #     #     quant += "_asy"
        #     # quant = self.LayerQuant[quant]
        #     # quant_u = self.LayerPrePost[quant]
        #     # qparams = [0, layer.get_in_quantize()[0].get_scale()[0]]
        #     # qparams = self.list2Cstyle(qparams)

        #     quant = 'QUANT_FSCALE'
        #     quant_u = self.LayerPrePost[quant]
        #     qparams = [0, 1.0/layer.get_in_quantize()[0].get_scale()[0]]
        #     qparams = self.list2Cstyle(qparams)
        # for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, qparams)]:
        #     LayerPre = _write(LayerPre, content)
        # LayerPre += "}"

        # q_i = [v for k, v in layer.get_in_scale()[0].items()]
        # q_h = [v for k, v in layer.get_in_scale()[1].items()]
        # q_w = [v for k, v in layer.get_w_scale()[0].items()]
        # q_r = [v for k, v in layer.get_w_scale()[1].items()]
        # q_ib = [1.0, 0.0]
        # q_hb = [1.0, 0.0]
        # q_wb = [1.0, 0.0]
        # q_rb = [1.0, 0.0]
        # i_type = self.NPU_DataType[
        #     layer.get_output_type()[3]
        # ]  # layer.get_input_type()[0]
        # o_type = self.NPU_DataType[layer.get_output_type()[0]]
        # i_fmt = self.MatFmt[self.fmt]
        # o_fmt = self.MatFmt[self.fmt]

        # seq_len = 1  # layer.get_layer_ops()['attrs']['sequence_lens']
        # seq_len = 1 #layer.get_layer_ops()['attrs']['sequence_lens']
        # i_size = layer.get_insert()['split']['in_align'][0]
        # o_size = layer.get_insert()['split']['out_align'][0]
        # hidden_size = layer.get_ops_setting()['attrs'][0]['hidden_size']
        # fc_o_size = layer.get_insert()['split']['out_align'][-1]
        # input_forget = -1
        # has_bias = 1
        # direction = "LSTM_FORWARD"
        # act_list = ["ACT_SIGMOID", "ACT_TANH", "ACT_TANH"]
        # for i in range(6):
        #     if i >= 3:
        #         act_list.append("ACT_NONE")
        # act_list_u = [[0] for i in range(6)]
        # lut = ["LUT_NONE" for i in range(6)]
        # tmp_offset = layer.get_w_offset()["tmp_offset"]
        # lut_off = [tmp_offset[i] for i in range(6)]
        # w_off = tmp_offset[6]
        # r_off = tmp_offset[7]
        # wb_off = tmp_offset[8]
        # rb_off = tmp_offset[9]
        # init_h_off = tmp_offset[10]
        # init_c_off = tmp_offset[11]
        # p_off = -1
        # pb_off = -1

        # Layer = ""
        # for content in [
        #     self.list2Cstyle(q_i),
        #     self.list2Cstyle(q_h),
        #     self.list2Cstyle(q_w),
        #     self.list2Cstyle(q_r),
        #     self.list2Cstyle(q_ib),
        #     self.list2Cstyle(q_hb),
        #     self.list2Cstyle(q_wb),
        #     self.list2Cstyle(q_rb),
        #     i_type,
        #     o_type,
        #     i_fmt,
        #     o_fmt,
        #     seq_len,
        #     i_size,
        #     hidden_size,
        #     fc_o_size,
        #     o_size,
        #     input_forget,
        #     has_bias,
        #     direction,
        #     self.list2Cstyle(act_list),
        #     self.list2Cstyle(act_list_u),
        #     self.list2Cstyle(lut),
        #     self.list2Cstyle(lut_off),
        #     w_off,
        #     r_off,
        #     wb_off,
        #     rb_off,
        #     init_h_off,
        #     init_c_off,
        #     p_off,
        #     pb_off,
        # ]:
        #     Layer = _write(Layer, content)

        # LayerPost = "{"
        # quant = self.LayerQuant[""]
        # qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        # quant_u = self.LayerPrePost[quant]
        # for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
        #     LayerPost = _write(LayerPost, content)
        # LayerPost += "},"
        # LayerPost += "}"

        contents = bytearray()

        return contents


@NETWORK_V3.register_module(name="gru")
class LAYER_GRU(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_GRU, self).__init__(**kwargs)

    def save(self, layer):
        # in_len, out_len = 1, 1 #self.get_io_len(layer)

        # qparams = []
        # for param in layer.get_scales()[0]:
        #     qparam = []
        #     # for k, v in param.items():
        #     #     if k in ['out_shift', 'out_scale', 'int_scale']:
        #     #         qparam.append(v)
        #     qparams.append(qparam)

        # LayerType = self.layer_map_inv[layer.get_layer_type()]
        # LayerInfo = self.LayerInstance[LayerType]

        # if layer.get_hx_combine() and layer.get_wr_combine():
        #     lstm_q = "LSTM_QUANT_I_H_SAME"
        # else:
        #     lstm_q = "LSTM_QUANT_I_H_DIFF"
        # LayerPre = "{" + lstm_q + ","
        # LayerPre += "{"
        # quant = self.LayerQuant[""]
        # qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        # quant_u = self.LayerPrePost[quant]
        # qparams = 0
        # if qi_type in ["NPU_FP32", "NPU_FP64"]:
        #     # is_perchannel = isinstance(layer.get_scales()[0]["out_shift"], np.ndarray)
        #     # is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        #     # quant = ""
        #     # if is_perchannel:
        #     #     quant += "perchannel_"
        #     # quant += "quant"
        #     # if is_asymquan:
        #     #     quant += "_asy"
        #     # quant = self.LayerQuant[quant]
        #     # quant_u = self.LayerPrePost[quant]
        #     # qparams = [0, layer.get_in_quantize()[0].get_scale()[0]]
        #     # qparams = self.list2Cstyle(qparams)

        #     quant = 'QUANT_FSCALE'
        #     quant_u = self.LayerPrePost[quant]
        #     qparams = [0, 1.0/layer.get_in_quantize()[0].get_scale()[0]]
        #     qparams = self.list2Cstyle(qparams)
        # for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, qparams)]:
        #     LayerPre = _write(LayerPre, content)
        # LayerPre += "}"

        # q_i = [v for k, v in layer.get_in_scale()[0].items()]
        # q_h = [v for k, v in layer.get_in_scale()[1].items()]
        # q_w = [v for k, v in layer.get_w_scale()[0].items()]
        # q_r = [v for k, v in layer.get_w_scale()[1].items()]
        # q_ib = [1.0, 0.0]
        # q_hb = [1.0, 0.0]
        # q_wb = [1.0, 0.0]
        # q_rb = [1.0, 0.0]
        # i_type = self.NPU_DataType[
        #     layer.get_output_type()[3]
        # ]  # layer.get_input_type()[0]
        # o_type = self.NPU_DataType[layer.get_output_type()[0]]
        # i_fmt = self.MatFmt[self.fmt]
        # o_fmt = self.MatFmt[self.fmt]

        # seq_len = 1  # layer.get_layer_ops()['attrs']['sequence_lens']
        # seq_len = 1 #layer.get_layer_ops()['attrs']['sequence_lens']
        # i_size = layer.get_insert()['split']['in_align'][0]
        # o_size = layer.get_insert()['split']['out_align'][0]
        # hidden_size = layer.get_ops_setting()['attrs'][0]['hidden_size']
        # fc_o_size = layer.get_insert()['split']['out_align'][-1]
        # input_forget = -1
        # has_bias = int(layer.get_ops_setting()["attrs"][0]["bias"])
        # direction = "LSTM_FORWARD"
        # act_list = ["ACT_SIGMOID", "ACT_TANH", "ACT_TANH"]
        # for i in range(6):
        #     if i >= 3:
        #         act_list.append("ACT_NONE")
        # act_list_u = [[0] for i in range(6)]
        # lut = ["LUT_NONE" for i in range(6)]
        # tmp_offset = layer.get_w_offset()["tmp_offset"]
        # lut_off = [tmp_offset[i] for i in range(6)]
        # w_off = tmp_offset[6]
        # r_off = tmp_offset[7]
        # wb_off = tmp_offset[8]
        # rb_off = tmp_offset[9]
        # init_h_off = tmp_offset[10]
        # p_off = -1
        # pb_off = -1

        # Layer = ""
        # for content in [
        #     self.list2Cstyle(q_i),
        #     self.list2Cstyle(q_h),
        #     self.list2Cstyle(q_w),
        #     self.list2Cstyle(q_r),
        #     self.list2Cstyle(q_ib),
        #     self.list2Cstyle(q_hb),
        #     self.list2Cstyle(q_wb),
        #     self.list2Cstyle(q_rb),
        #     i_type,
        #     o_type,
        #     i_fmt,
        #     o_fmt,
        #     seq_len,
        #     i_size,
        #     hidden_size,
        #     fc_o_size,
        #     o_size,
        #     input_forget,
        #     has_bias,
        #     direction,
        #     self.list2Cstyle(act_list),
        #     self.list2Cstyle(act_list_u),
        #     self.list2Cstyle(lut),
        #     self.list2Cstyle(lut_off),
        #     w_off,
        #     r_off,
        #     wb_off,
        #     rb_off,
        #     init_h_off,
        #     p_off,
        #     pb_off,
        # ]:
        #     Layer = _write(Layer, content)

        # LayerPost = "{"
        # quant = self.LayerQuant[""]
        # qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        # quant_u = self.LayerPrePost[quant]
        # for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
        #     LayerPost = _write(LayerPost, content)
        # LayerPost += "},"
        # LayerPost += "}"

        contents = bytearray()

        return contents


@NETWORK_V3.register_module(name="batchnormalization")
@NETWORK_V3.register_module(name="layernormalization")
@NETWORK_V3.register_module(name="instancenormalization")
@NETWORK_V3.register_module(name="groupnormalization")
class LAYER_NORM(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_NORM, self).__init__(**kwargs)

    def get_process(self, layer, i_type, o_type, w_type):
        layer_type = layer.get_layer_type()
        LayerType = self.layer_map_inv[layer_type]
    
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]

        N = 1
        if "split" in layer.get_insert().keys():
            C = layer.get_insert()["split"]["in_align"][0]
            H, W = layer.get_insert()["split"]["feat_i"][0]
        else:
            C = layer.get_insert()["in_align"][0]
            H, W = layer.get_insert()["feat_i"][0]

        eps = layer.get_ops_setting()['attrs'][0]['epsilon']
        affine_offset = layer.get_w_offset()["w_offset"]
        affine = 1 if affine_offset != -1 else 0
        groups = 0    
        if layer_type == "batchnormalization":
            operation = NormType_t["BN"].value
        elif layer_type == "layernormalization":
            operation = NormType_t["LN"].value
        elif layer_type == "instancenormalization":
            operation = NormType_t["IN"].value
        elif layer_type == "groupnormalization":
            operation = NormType_t["GN"].value
            groups = 0
        type = LayerType_t[LayerType].value
        i_type = NpuType_t[i_type].value
        o_type = NpuType_t[o_type].value
        i_fmt = LayerFmt_t[i_fmt].value
        o_fmt = LayerFmt_t[o_fmt].value
        LayerProcess = bytearray()
        for value, dtype in zip(
            [
                type, i_type, o_type,
                i_fmt, o_fmt, 
                N, H, W, C, 
                operation, affine, 
                groups,
                eps,
                affine_offset,
            ],
            [
                np.uint32, np.uint8, np.uint8,
                np.uint8, np.uint8,
                np.uint16, np.uint16, np.uint16, np.uint16,
                np.uint8, np.uint8,
                np.uint16,
                np.float32,
                np.uint32,
            ],
        ):
            LayerProcess += to_bytes(value, dtype=dtype) # type: ignore
                
        return LayerProcess
          
            
    def get_types(self, layer, idx=0):
        assert layer.get_scale_type() in ["float"]
        
        qi_type = self.NPU_DataType[layer.get_input_type()[idx]]
        i_type = o_type = "NPU_FP32"
    
        return qi_type, i_type, o_type, None
    
    # def save(self, layer):
    #     # in_len, out_len = self.get_io_len(layer)

    #     # qparams = []

    #     # LayerType = self.layer_map_inv[layer.get_layer_type()]
    #     # LayerInfo = self.LayerInstance[LayerType]

    #     # LayerPre = "{"
    #     # LayerPre += "{"
    #     # quant = self.LayerQuant[""]
    #     # qi_type = self.NPU_DataType[layer.get_input_type()[0]]
    #     # quant_u = self.LayerPrePost[quant]
    #     # for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
    #     #     LayerPre = _write(LayerPre, content)
    #     # LayerPre += "}"

    #     # i_type = self.NPU_DataType[layer.get_input_type()[0]]
    #     # o_type = self.NPU_DataType[layer.get_output_type()[0]]
    #     # i_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]
    #     # o_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]

    #     # # _, C, H, W = layer.get_in_data()[0]['output'].shape
    #     # if "split" in layer.get_insert().keys():
    #     #     C = layer.get_insert()["split"]["in_align"][0]
    #     #     H, W = layer.get_insert()["split"]["feat_i"][0]
    #     # else:
    #     #     C = layer.get_insert()["in_align"][0]
    #     #     H, W = layer.get_insert()["feat_i"][0]

    #     # eps = layer.get_ops_setting()['attrs'][0]['epsilon']
    #     # offset = layer.get_w_offset()["w_offset"]
    #     # affine = 1 if offset != -1 else 0

    #     # Layer = ""
    #     # for content in [i_type, o_type, i_fmt, o_fmt, H, W, C, eps, affine, offset]:
    #     #     Layer = _write(Layer, content)

    #     # LayerPost = "{"
    #     # quant = self.LayerQuant[""]
    #     # qi_type = self.NPU_DataType[layer.get_output_type()[0]]
    #     # quant_u = self.LayerPrePost[quant]
    #     # for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
    #     #     LayerPost = _write(LayerPost, content)
    #     # LayerPost += "},"
    #     # LayerPost += "}"

    #     contents = bytearray()

    #     return contents
    
    
@NETWORK_V3.register_module(name="log")
class LAYER_LOG(NetworkV3):
    def __init__(self, **kwargs):
        super(LAYER_LOG, self).__init__(**kwargs)
