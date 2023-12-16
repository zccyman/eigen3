# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/20 11:32
# @File     : network.py
import copy

import numpy as np

try:
    from export import NetworkBase, _write
    from utils import (Registry, get_last_layer_quant, get_scale_param,
                       invert_dict)
except Exception:
    from onnx_converter.export import NetworkBase, _write # type: ignore
    from onnx_converter.utils import (Registry, get_last_layer_quant, # type: ignore
                                      get_scale_param, invert_dict)

NETWORK_V2: Registry = Registry("network_v2", scope="")


# export special layer in model.c
# written network structure in binary file
class NetworkV2(NetworkBase):
    def __init__(self, **kwargs):
        super(NetworkV2, self).__init__()

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
        self.Csize = kwargs["bits"]["Csize"]
        self.Ksize = kwargs["bits"]["Ksize"]
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

    def get_quant(self, layer, scale, mode="quant"):
        # is_perchannel = isinstance(layer.get_scales()[0]['out_shift'], np.ndarray)
        is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        quant = ""
        # if is_perchannel:
        # quant += 'perchannel_'
        quant += mode
        if is_asymquan:
            quant += "_asy"
        quant = self.LayerQuant[quant]
        # if is_perchannel:
        #     qparams = [layer.get_w_offset()['tmp_offset'][0]] #offset
        # else:

        if isinstance(scale, list):
            scale = scale[0]

        if is_asymquan:
            qparams = [
                scale["scale"],
                scale["zero_point"],
            ]  # scale, zero
        else:
            qparams = [scale["scale"]]  # scale
            
        # if quant in ["QUANT_QUANT", "QUANT_QUANT_ASY"]:
        #     quant = "QUANT_FSCALE"
        #     qparams[0] = 1.0 / qparams[0]
        #     qparams.insert(0, 0)
        # elif quant in ["QUANT_DEQUANT", "QUANT_DEQUANT_ASY"]:   
        #     quant = "QUANT_FSCALE" 
        #     qparams.insert(0, 0)
            
        qparams = self.list2Cstyle(qparams)

        return quant, qparams

    def get_contents_v2(self, layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost):
        contents = "{"
        contents = _write(contents, LayerType)
        if not self.is_debug:
            contents = _write(contents, ".layer.{}=".format(LayerInfo), tail="")
        else:
            contents = _write(
                contents,
                ".layer_{}_{}.{}=".format(
                    layer.get_idx(), layer.get_layer_name(), LayerInfo
                ),
                tail="",
            )
        contents = _write(contents, LayerPre)
        contents = _write(contents, Layer, tail="")
        contents = _write(contents, LayerPost)
        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


# @NETWORK_V2.register_module(name="splice")
# class LAYER_PLACEHOLDER(NetworkV2):
#     def __init__(self, **kwargs):
#         super(LAYER_PLACEHOLDER, self).__init__(**kwargs)

#     def save(self, layer):
#         layer_type = layer.get_layer_type()
#         LayerType = self.layer_map_inv[layer_type]
#         LayerInfo = self.LayerInstance[LayerType]
#         quant = self.LayerQuant[""]  # layer.get_scale_type()

#         # dtypes = invert_dict(layer.get_ops_setting()['setting']['bits_dict'])
#         # dtype = layer.get_in_data().dtype.type
#         # if dtype in dtypes.keys():
#         #     datatype = dtypes[dtype]
#         # else:
#         #     raise Exception('Not Implemented datatype !!!')

#         qi_type = self.NPU_DataType[layer.get_output_type()[0]]
#         quant_u = self.LayerPrePost[quant]

#         # qparam = [layer.get_scales()[-1]['out_shift'], layer.get_scales()[-1]['out_scale']]
#         # qparam = self.list2Cstyle(qparam)

#         LayerPre = "{"
#         LayerPre += "{"
#         LayerPre = _write(LayerPre, quant)
#         LayerPre = _write(LayerPre, qi_type)
#         LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, 0))
#         LayerPre += "}"

#         # i_type = layer.get_ops_setting()['setting']['bits_dict'][
#         #     layer.get_input_type(
#         #     )].__name__  # self.NPU_DataType[layer.get_input_type()]
#         # o_type = layer.get_ops_setting()['setting']['bits_dict'][
#         #     layer.get_output_type(
#         #     )].__name__  # self.NPU_DataType[layer.get_output_type()]
#         # c = 'NPU_' + qi_type.replace('u', 'U').replace('int', 'INT')
#         i_type = (
#             o_type
#         ) = qi_type  #'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')

#         if len(layer.get_insert()["feat_o"][0]) == 2:
#             H, W = layer.get_insert()["feat_o"][0]
#             if [H, W] == [1, 1]:
#                 i_fmt = self.MatFmt[self.fmt]
#                 o_fmt = self.MatFmt[self.fmt]
#             else:
#                 i_fmt = self.CubeFmt[self.fmt]
#                 o_fmt = self.CubeFmt[self.fmt]
#         else:
#             H, W = 1, 1
#             i_fmt = self.MatFmt[self.fmt]
#             o_fmt = self.MatFmt[self.fmt]

# B = 74
@NETWORK_V2.register_module(name="splice")
@NETWORK_V2.register_module(name="data")
class LAYER_PLACEHOLDER(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_PLACEHOLDER, self).__init__(**kwargs)

    def save(self, layer):
        layer_type = layer.get_layer_type()
        LayerType = self.layer_map_inv[layer_type]
        LayerInfo = self.LayerInstance[LayerType]
        quant = self.LayerQuant[""]  # layer.get_scale_type()

        # dtypes = invert_dict(layer.get_ops_setting()['setting']['bits_dict'])
        # dtype = layer.get_in_data().dtype.type
        # if dtype in dtypes.keys():
        #     datatype = dtypes[dtype]
        # else:
        #     raise Exception('Not Implemented datatype !!!')

        qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]

        # qparam = [layer.get_scales()[-1]['out_shift'], layer.get_scales()[-1]['out_scale']]
        # qparam = self.list2Cstyle(qparam)

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, 0))
        LayerPre += "}"

        # i_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_input_type(
        #     )].__name__  # self.NPU_DataType[layer.get_input_type()]
        # o_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_output_type(
        #     )].__name__  # self.NPU_DataType[layer.get_output_type()]
        # c = 'NPU_' + qi_type.replace('u', 'U').replace('int', 'INT')
        i_type = (
            o_type
        ) = qi_type  #'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')

        if len(layer.get_insert()["feat_o"][0]) == 2:
            H, W = layer.get_insert()["feat_o"][0]
            if [H, W] == [1, 1]:
                i_fmt = self.MatFmt[self.fmt]
                o_fmt = self.MatFmt[self.fmt]
            else:
                i_fmt = self.CubeFmt[self.fmt]
                o_fmt = self.CubeFmt[self.fmt]
        else:
            H, W = 1, 1
            i_fmt = self.MatFmt[self.fmt]
            o_fmt = self.MatFmt[self.fmt]

        _, C = layer.get_insert()["out_pad"][0]
        OC = layer.get_insert()["out_align"][0]
        OH, OW = H, W

        Layer = ""
        for content in [i_type, o_type, i_fmt, o_fmt, H, W, C, OH, OW, OC]:
            Layer = _write(Layer, content)

        quant = self.LayerQuant[""]  # self.LayerQuant[layer.get_scale_type()]
        qo_type = o_type  # self.NPU_DataType[layer.get_output_type()]
        quant_u = self.LayerPrePost[quant]

        # qparam = [layer.get_scales()[-1]['out_shift'], layer.get_scales()[-1]['out_scale']]
        # qparam = self.list2Cstyle(qparam)

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="conv")
@NETWORK_V2.register_module(name="depthwiseconv")
class LAYER_CONV2D(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_CONV2D, self).__init__(**kwargs)

    def save(self, layer):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        # if qi_type in ["NPU_FP32", "NPU_FP64"]:
        #     # is_perchannel = isinstance(layer.get_scales()[0]['out_shift'], np.ndarray)
        #     is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        #     quant = ""
        #     # if is_perchannel:
        #     # quant += 'perchannel_'
        #     quant += "quant"
        #     if is_asymquan:
        #         quant += "_asy"
        #     quant = self.LayerQuant[quant]
        #     # if is_perchannel:
        #     #     qparams = [layer.get_w_offset()['tmp_offset'][0]] #offset
        #     # else:
        #     if is_asymquan:
        #         qparams = [
        #             layer.get_in_scale()["scale"],
        #             layer.get_in_scale()["zero_point"],
        #         ]  # scale, zero
        #     else:
        #         qparams = [layer.get_in_scale()["scale"]]  # scale
        #     qparams = self.list2Cstyle(qparams)
        # else:
        #     quant = self.LayerQuant[""]
        #     qparams = 0

        # quant_u = self.LayerPrePost[quant]

        op = layer.get_ops_instance()
        op = op[0] if isinstance(op, list) else op
        i_type = self.NPU_DataType[op.bit_select]
        mode = self.get_quant_mode(qi_type, i_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_in_scale(), mode)
        else:
            quant = self.LayerQuant[""]  # layer.get_scale_type()
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, qparams))
        LayerPre += "}"

        # op = layer.get_ops_instance()
        # op = op[0] if isinstance(op, list) else op
        # i_type = self.NPU_DataType[op.bit_select]
        w_type = str(layer.get_qweight().dtype)
        w_type = "NPU_" + w_type.replace("u", "U").replace("int", "INT")
        # o_type = self.NPU_DataType[op.high_bits_calc(op.bit_select)]
        if not op.get_precision():  # layer.get_ops_setting()['setting']['precision']:
            o_type = i_type  # int8 | int16
        else:
            o_type = (
                "NPU_INT32" if i_type == "NPU_INT8" else "NPU_INT64"
            )  # int32 | int64

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
        has_bias = 1 #int(layer.get_ops_setting()["attrs"][0]["bias"])
        act = self.ActivationType[layer.get_ops_setting()["ops_string"][-1]]
        act_u = ".act_u.none=0"
        # w_off = layer.get_w_offset()['w_offset']
        w_off = layer.get_w_offset()["tmp_offset"][1]

        Layer = ""
        for content in [
            i_type,
            w_type,
            o_type,
            i_fmt,
            w_fmt,
            o_fmt,
            H,
            W,
            C,
            FH,
            FW,
            K,
            SH,
            SW,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            OH,
            OW,
            has_bias,
            act,
            act_u,
            w_off,
        ]:
            Layer = _write(Layer, content)

        scales = copy.deepcopy(layer.get_scales())
        if isinstance(scales, list):
            scales = scales[0]
        is_perchannel = isinstance(scales["out_shift"], np.ndarray)
        is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        # if layer.get_is_result_layer():
        #     quant = get_last_layer_quant(layer)
        #     if is_perchannel:
        #         qparams = [layer.get_w_offset()["tmp_offset"][2]]  # offset
        #         qparams = self.list2Cstyle(qparams)
        #     else:
        #         qparams = get_scale_param(layer, quant)
        #         qparams = self.list2Cstyle(qparams)
        # else:
        #     if is_perchannel:
        #         if is_asymquan:
        #             quant = self.LayerQuant[
        #                 "perchannel_" + layer.get_scale_type() + "_asy"
        #             ]
        #         else:
        #             quant = self.LayerQuant["perchannel_" + layer.get_scale_type()]
        #         qparams = [layer.get_w_offset()["tmp_offset"][2]]  # offset
        #         qparams = self.list2Cstyle(qparams)
        #     else:
        #         if is_asymquan:
        #             quant = self.LayerQuant[layer.get_scale_type() + "_asy"]
        #         else:
        #             quant = self.LayerQuant[layer.get_scale_type()]
        #         qparams = get_scale_param(layer, quant)
        #         qparams = self.list2Cstyle(qparams)

        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        mode = self.get_quant_mode(o_type, qo_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_scale(), mode)
        elif layer.get_is_result_layer():
            quant = get_last_layer_quant(layer)
            if is_perchannel:
                qparams = [layer.get_w_offset()["tmp_offset"][2]]
                qparams = self.list2Cstyle(qparams)
            else:
                qparams = get_scale_param(layer, quant)
                qparams = self.list2Cstyle(qparams)
        else:
            if is_perchannel:
                if is_asymquan:
                    quant = self.LayerQuant[
                        "perchannel_" + layer.get_scale_type() + "_asy"
                    ]
                else:
                    quant = self.LayerQuant["perchannel_" + layer.get_scale_type()]
                qparams = [layer.get_w_offset()["tmp_offset"][2]]
                qparams = self.list2Cstyle(qparams)
            else:
                if is_asymquan:
                    pass
                else:
                    quant = self.LayerQuant[layer.get_scale_type()]
                qparams = get_scale_param(layer, quant)
                qparams = self.list2Cstyle(qparams)
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="convtranspose")
class LAYER_CONVTRANSPOSE2D(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_CONVTRANSPOSE2D, self).__init__(**kwargs)

    def save(self, layer):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        # if qi_type in ["NPU_FP32", "NPU_FP64"]:
        #     # is_perchannel = isinstance(layer.get_scales()[0]['out_shift'], np.ndarray)
        #     is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        #     quant = ""
        #     # if is_perchannel:
        #     # quant += 'perchannel_'
        #     quant += "quant"
        #     if is_asymquan:
        #         quant += "_asy"
        #     quant = self.LayerQuant[quant]
        #     # if is_perchannel:
        #     #     qparams = [layer.get_w_offset()['tmp_offset'][0]] #offset
        #     # else:
        #     if is_asymquan:
        #         qparams = [
        #             layer.get_in_scale()["scale"],
        #             layer.get_in_scale()["zero_point"],
        #         ]  # scale, zero
        #     else:
        #         qparams = [layer.get_in_scale()["scale"]]  # scale
        #     qparams = self.list2Cstyle(qparams)
        # else:
        #     quant = self.LayerQuant[""]
        #     qparams = 0

        # quant_u = self.LayerPrePost[quant]

        op = layer.get_ops_instance()
        op = op[0] if isinstance(op, list) else op
        i_type = self.NPU_DataType[op.bit_select]
        mode = self.get_quant_mode(qi_type, i_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_in_scale(), mode)
        else:
            quant = self.LayerQuant[""]  # layer.get_scale_type()
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, qparams))
        LayerPre += "}"

        # op = layer.get_ops_instance()
        # op = op[0] if isinstance(op, list) else op
        # i_type = self.NPU_DataType[op.bit_select]
        w_type = str(layer.get_qweight().dtype)
        w_type = "NPU_" + w_type.replace("u", "U").replace("int", "INT")
        # o_type = self.NPU_DataType[op.high_bits_calc(op.bit_select)]
        if not op.get_precision():  # layer.get_ops_setting()['setting']['precision']:
            o_type = i_type  # int8 | int16
        else:
            o_type = (
                "NPU_INT32" if i_type == "NPU_INT8" else "NPU_INT64"
            )  # int32 | int64

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
        pad_t, pad_l, pad_b, pad_r = layer.get_ops_setting()["attrs"][0]["pads"]
        OH, OW = layer.get_insert()["feat_o"][0]
        has_bias = int(layer.get_ops_setting()["attrs"][0]["bias"])
        act = self.ActivationType[layer.get_ops_setting()["ops_string"][-1]]
        act_u = ".act_u.none=0"
        # w_off = layer.get_w_offset()['w_offset']
        w_off = layer.get_w_offset()["tmp_offset"][1]

        Layer = ""
        for content in [
            i_type,
            w_type,
            o_type,
            i_fmt,
            w_fmt,
            o_fmt,
            H,
            W,
            C,
            FH,
            FW,
            K,
            SH,
            SW,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            OH,
            OW,
            has_bias,
            act,
            act_u,
            w_off,
        ]:
            Layer = _write(Layer, content)

        scales = copy.deepcopy(layer.get_scales())
        if isinstance(scales, list):
            scales = scales[0]
        is_perchannel = isinstance(scales["out_shift"], np.ndarray)
        is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        # if layer.get_is_result_layer():
        #     quant = get_last_layer_quant(layer)
        #     if is_perchannel:
        #         qparams = [layer.get_w_offset()["tmp_offset"][2]]  # offset
        #         qparams = self.list2Cstyle(qparams)
        #     else:
        #         qparams = get_scale_param(layer, quant)
        #         qparams = self.list2Cstyle(qparams)
        # else:
        #     if is_perchannel:
        #         if is_asymquan:
        #             quant = self.LayerQuant[
        #                 "perchannel_" + layer.get_scale_type() + "_asy"
        #             ]
        #         else:
        #             quant = self.LayerQuant["perchannel_" + layer.get_scale_type()]
        #         qparams = [layer.get_w_offset()["tmp_offset"][2]]  # offset
        #         qparams = self.list2Cstyle(qparams)
        #     else:
        #         if is_asymquan:
        #             quant = self.LayerQuant[layer.get_scale_type() + "_asy"]
        #         else:
        #             quant = self.LayerQuant[layer.get_scale_type()]
        #         qparams = get_scale_param(layer, quant)
        #         qparams = self.list2Cstyle(qparams)

        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        mode = self.get_quant_mode(o_type, qo_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_scale(), mode)
        elif layer.get_is_result_layer():
            quant = get_last_layer_quant(layer)
            if is_perchannel:
                qparams = [layer.get_w_offset()["tmp_offset"][2]]
                qparams = self.list2Cstyle(qparams)
            else:
                qparams = get_scale_param(layer, quant)
                qparams = self.list2Cstyle(qparams)
        else:
            if is_perchannel:
                if is_asymquan:
                    quant = self.LayerQuant[
                        "perchannel_" + layer.get_scale_type() + "_asy"
                    ]
                else:
                    quant = self.LayerQuant["perchannel_" + layer.get_scale_type()]
                qparams = [layer.get_w_offset()["tmp_offset"][2]]
                qparams = self.list2Cstyle(qparams)
            else:
                if is_asymquan:
                    pass
                else:
                    quant = self.LayerQuant[layer.get_scale_type()]
                qparams = get_scale_param(layer, quant)
                qparams = self.list2Cstyle(qparams)
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="fc")
class LAYER_FC(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_FC, self).__init__(**kwargs)

    def save(self, layer):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        # quant_u = self.LayerPrePost[quant]
        # qparams = 0
        # if qi_type in ['NPU_FP32']:
        #     quant = 'QUANT_FSCALE'
        #     quant_u = self.LayerPrePost[quant]
        #     qparams = [0, 1.0/layer.get_in_quantize()[0].get_scale()[0]]
        #     qparams = self.list2Cstyle(qparams)

        # if qi_type in ["NPU_FP32", "NPU_FP64"]:
        #     # is_perchannel = isinstance(layer.get_scales()[0]['out_shift'], np.ndarray)
        #     is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        #     quant = ""
        #     # if is_perchannel:
        #     # quant += 'perchannel_'
        #     quant += "quant"
        #     if is_asymquan:
        #         quant += "_asy"
        #     quant = self.LayerQuant[quant]
        #     # if is_perchannel:
        #     #     qparams = [layer.get_w_offset()['tmp_offset'][0]] #offset
        #     # else:
        #     if is_asymquan:
        #         qparams = [
        #             layer.get_in_scale()["scale"],
        #             layer.get_in_scale()["zero_point"],
        #         ]  # scale, zero
        #     else:
        #         qparams = [layer.get_in_scale()["scale"]]  # scale
        #     qparams = self.list2Cstyle(qparams)
        # else:
        #     quant = self.LayerQuant[""]
        #     qparams = 0

        # quant_u = self.LayerPrePost[quant]

        op = layer.get_ops_instance()
        op = op[0] if isinstance(op, list) else op
        i_type = self.NPU_DataType[op.bit_select]
        mode = self.get_quant_mode(qi_type, i_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_in_scale(), mode)
        else:
            quant = self.LayerQuant[""]  # layer.get_scale_type()
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, qparams))
        LayerPre += "}"

        # op = layer.get_ops_instance()
        # op = op[0] if isinstance(op, list) else op
        # i_type = self.NPU_DataType[op.bit_select]
        w_type = str(layer.get_qweight().dtype)
        w_type = "NPU_" + w_type.replace("u", "U").replace("int", "INT")
        # o_type = self.NPU_DataType[op.high_bits_calc(op.bit_select)]
        if not op.get_precision():  # layer.get_ops_setting()['setting']['precision']:
            o_type = i_type  # int8 | int16
        else:
            o_type = (
                "NPU_INT32" if i_type == "NPU_INT8" else "NPU_INT64"
            )  # int32 | int64

        process_scale = layer.get_scale_type()
        use_table = process_scale in ["shiftfloatscaletable", "shiftfloatscaletable2float"]
        if use_table:
            if o_type == "NPU_INT8":
                act_type = "ACT_LUT8"
            elif o_type == "NPU_INT16":
                act_type = "ACT_LUT16"      
            else:
                act_type = "ACT_NONE"
                
            if process_scale in ["shiftfloatscaletable2float"]:    
                act_type += "_FP"    
                o_type = 'NPU_FP32' 
        else:
            act = self.ActivationType[layer.get_ops_setting()["ops_string"][-1]]
            if act in ["ACT_RELU", "ACT_BRELU", "ACT_RELU6"]:
                act_type = act
            else:
                act_type = "ACT_NONE"

        i_fmt = self.MatFmt[self.fmt]
        w_fmt = self.FcWFmt[self.fmt]
        o_fmt = self.MatFmt[self.fmt]
        M, K, N = (
            1,
            layer.get_insert()["in_align"][0],
            layer.get_insert()["out_align"][0],
        )
        has_bias = 1 #int(layer.get_ops_setting()["attrs"][0]["bias"])
        # act = self.ActivationType[layer.get_ops_setting()["ops_string"][-1]]
        # act_u = ".act_u.none=0"
        w_off = layer.get_w_offset()["tmp_offset"][1]
        offset = layer.get_w_offset()["tmp_offset"][2]
        if use_table:
            act_u = ".act_u={" + str(offset) + "}"
        else:
            act_u = ".act_u={0}"
            
        Layer = ""
        for content in [
            i_type,
            w_type,
            o_type,
            i_fmt,
            w_fmt,
            o_fmt,
            M,
            K,
            N,
            has_bias,
            act_type,
            act_u,
            w_off,
        ]:
            Layer = _write(Layer, content)

        scales = copy.deepcopy(layer.get_scales())
        if isinstance(scales, list):
            scales = scales[0]
        is_perchannel = isinstance(scales["out_shift"], np.ndarray)
        is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        # if layer.get_scale_type() in ["ffloatscale"]:
        #     qparams = [
        #         0,
        #         layer.get_in_scale()[0]["scale"] * layer.get_w_scale()["scale"],
        #     ]
        #     qparams = self.list2Cstyle(qparams)

        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        if process_scale in ["shiftfloatscaletable2float"] and qo_type in ["NPU_FP32"] or \
            process_scale in ["shiftfloatscaletable"] and qo_type in ["NPU_INT8", "NPU_INT16"]:
            quant = "QUANT_SHIFT"
            qparams = self.list2Cstyle(get_scale_param(layer, quant))
        else:
            mode = self.get_quant_mode(o_type, qo_type)
            if mode != "":
                quant, qparams = self.get_quant(layer, layer.get_scale(), mode)
            elif layer.get_is_result_layer():
                quant = get_last_layer_quant(layer)
                if is_perchannel:
                    qparams = [layer.get_w_offset()["tmp_offset"][1]]
                    qparams = self.list2Cstyle(qparams)
                else:
                    qparams = get_scale_param(layer, quant)
                    qparams = self.list2Cstyle(qparams)
            else:
                if is_perchannel:
                    if is_asymquan:
                        quant = self.LayerQuant[
                            "perchannel_" + layer.get_scale_type() + "_asy"
                        ]
                    else:
                        quant = self.LayerQuant["perchannel_" + layer.get_scale_type()]
                    qparams = [layer.get_w_offset()["tmp_offset"][1]]
                    qparams = self.list2Cstyle(qparams)
                else:
                    if is_asymquan:
                        pass
                    else:
                        quant = self.LayerQuant[layer.get_scale_type()]
                    qparams = get_scale_param(layer, quant)
                    qparams = self.list2Cstyle(qparams)
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="relu")
@NETWORK_V2.register_module(name="relu6")
@NETWORK_V2.register_module(name="relux")
@NETWORK_V2.register_module(name="leakyrelu")
@NETWORK_V2.register_module(name="prelu")
@NETWORK_V2.register_module(name="sigmoid")
@NETWORK_V2.register_module(name="swish")
@NETWORK_V2.register_module(name="gelu")
@NETWORK_V2.register_module(name="tanh")
@NETWORK_V2.register_module(name="hardsigmoid")
@NETWORK_V2.register_module(name="hardtanh")
@NETWORK_V2.register_module(name="hardswish")
@NETWORK_V2.register_module(name="hardshrink")
class LAYER_ACTIVATION(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_ACTIVATION, self).__init__(**kwargs)

    def save(self, layer):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]

        op = layer.get_ops_instance()
        op = op[0] if isinstance(op, list) else op
        if layer.get_scale_type() == "float":
            i_type = o_type = "NPU_FP32"
        else:
            i_type = qi_type
            o_type = self.NPU_DataType[
                layer.get_output_type()[-1]
            ]  #'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')
        mode = self.get_quant_mode(qi_type, i_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_in_scale(), mode)
        else:
            quant = self.LayerQuant[""]  # layer.get_scale_type()
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, qparams))
        LayerPre += "}"

        # if layer.get_scale_type() == "float":
        #     i_type = o_type = "NPU_FP32"
        # else:
        #     i_type = qi_type
        #     o_type = self.NPU_DataType[
        #         layer.get_output_type()[-1]
        #     ]  #'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')
        i_fmt = self.MatFmt[self.fmt]
        o_fmt = self.MatFmt[self.fmt]
        H, W = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        act = self.ActivationType[layer.get_layer_type()]
        if layer.get_layer_type() in ["relux", "relu6"]:
            max_value = layer.get_ops_setting()['attrs'][0]['max']
            act_u = ".act_u.b_relu={}".format(
                self.list2Cstyle([max_value])
            )
        elif layer.get_layer_type() == "leakyrelu":
            act_u = ".act_u.leaky_relu={}".format(
                self.list2Cstyle([layer.get_ops_setting()["attrs"][0]["alpha"]])
            )
        elif layer.get_layer_type() == "hardsigmoid":
            act_u = ".act_u.hard_sigmoid={}".format(
                self.list2Cstyle([
                    layer.get_ops_setting()["attrs"][0]["alpha"],
                    layer.get_ops_setting()["attrs"][0]["beta"],
                    ])
            )
        else:
            act_u = ".act_u.none=0"
        is_perchannel = isinstance(layer.get_scales()[0]["out_shift"], np.ndarray)
        if layer.get_scale_type() == "table":
            if is_perchannel:
                lut = "LUT_PER_CHN"
            else:
                lut = "LUT_NORMAL"
        else:
            lut = "LUT_NONE"
        # if layer.get_layer_type() == 'relu':
        #     print('test')
        lut_off = layer.get_w_offset()["w_offset"]

        Layer = ""
        for content in [
            i_type,
            o_type,
            i_fmt,
            o_fmt,
            H,
            W,
            C,
            act,
            act_u,
            lut,
            lut_off,
        ]:
            Layer = _write(Layer, content)

        # if layer.get_scale_type() in ["table", "float"]:
        #     quant = self.LayerQuant[""]
        #     qparams = 0
        # else:
        #     quant = self.LayerQuant[layer.get_scale_type()]
        #     qparams = get_scale_param(layer, quant)
        #     qparams = self.list2Cstyle(qparams)

        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        mode = self.get_quant_mode(o_type, qo_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_scale(), mode)
        else:
            quant = self.LayerQuant[""]  # layer.get_scale_type()
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="reducemax")
@NETWORK_V2.register_module(name="reducemin")
@NETWORK_V2.register_module(name="reducemean")
@NETWORK_V2.register_module(name="reducesum")
@NETWORK_V2.register_module(name="reduceprod")
class LAYER_REDUCE(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_REDUCE, self).__init__(**kwargs)

    def save(self, layer):
        contents = ""
        return contents


@NETWORK_V2.register_module(name="transpose")
class LAYER_TRANSPOSE(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_TRANSPOSE, self).__init__(**kwargs)

    def save(self, layer):
        contents = ""
        return contents


@NETWORK_V2.register_module(name="matmul")
class LAYER_MATMUL(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_MATMUL, self).__init__(**kwargs)

    def save(self, layer):
        contents = ""
        return contents


@NETWORK_V2.register_module(name="maxpool")
@NETWORK_V2.register_module(name="averagepool")
@NETWORK_V2.register_module(name="globalaveragepool")
class LAYER_POOL(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_POOL, self).__init__(**kwargs)

    def save(self, layer):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        quant = self.LayerQuant[layer.get_scale_type()]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]

        qparams = [0, layer.get_in_scale()[0]["scale"]]
        qparams = self.list2Cstyle(qparams)

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, qparams))
        LayerPre += "}"

        # i_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_input_type(
        #     )].__name__  # self.NPU_DataType[layer.get_input_type()]
        # o_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_output_type(
        #     )].__name__  # self.NPU_DataType[layer.get_output_type()]
        # i_type = 'NPU_' + i_type.replace('u', 'U').replace('int', 'INT')
        # o_type = 'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')
        i_type = o_type = "NPU_FP32"
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        pool = self.PoolType[layer.get_layer_type()]
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

        Layer = ""
        for content in [
            i_type,
            o_type,
            i_fmt,
            o_fmt,
            pool,
            H,
            W,
            C,
            FH,
            FW,
            SH,
            SW,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            OH,
            OW,
        ]:
            Layer = _write(Layer, content)

        quant = self.LayerQuant[layer.get_scale_type()]
        qo_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]

        if layer.get_scale_type() == "smooth":
            qparams = [0, layer.get_in_scale()[0]["scale"]]
        else:
            qparams = [0, layer.get_scale()[0]["scale"]]
        qparams = self.list2Cstyle(qparams)
        # qparams = get_scale_param(layer, quant)
        # qparams = self.list2Cstyle(qparams)

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="mul")
@NETWORK_V2.register_module(name="pmul")
@NETWORK_V2.register_module(name="add")
@NETWORK_V2.register_module(name="sub")
class LAYER_EWS(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_EWS, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = 2, 1  # self.get_io_len(layer)

        op = layer.get_ops_instance()
        op = op[0] if isinstance(op, list) else op
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        if "pmul" in layer.get_layer_type():
            mode = "" if layer.get_scale_type() != "float" else self.get_quant_mode(qi_type, "NPU_FP32")
            if mode != "":
                quants, qparams = [], []
                for i in range(in_len):
                    quant, qparam = self.get_quant(layer, layer.get_in_scale()[i], mode)
                    qparams.append(qparam)
                    quants.append(quant)
            else:
                qparams = [0 for _ in range(in_len)]
                quants = [self.LayerQuant[""] for _ in range(in_len)]
        else:
            qparams = []
            scales = copy.deepcopy(layer.get_scales())
            for param in scales:
                qparam = []
                for k, v in param.items():
                    if k in ["out_shift", "out_scale", "int_scale"]:
                        if k in ["out_shift", "int_scale"] and v > 0: v = -v
                        qparam.append(v)
                qparams.append(qparam)

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        LayerPre = "{" + str(in_len) + ","
        LayerPre += "{"
        for i in range(self.ELEMENT_WISE_MAX_IN):
            if i < in_len:
                if "pmul" in layer.get_layer_type():
                    quant = quants[i] # type: ignore
                    qi_type = self.NPU_DataType[layer.get_input_type()[i]]
                    quant_u = self.LayerPrePost[quant]
                    qparam = qparams[i]
                    # i_type = qi_type
                    # if layer.get_scale_type() == "float":
                        # i_type = o_type = "NPU_FP32"
                    # if i_type in ['NPU_INT8']:
                    #     o_type = 'NPU_INT32'
                    # elif i_type in ['NPU_INT16']:
                    #     o_type = 'NPU_INT64'
                    # else:
                    #     assert Exception('Not implemented o_type in mul operator!!!')
                else:
                    quant = self.LayerQuant[layer.get_scale_type()]
                    qi_type = self.NPU_DataType[layer.get_input_type()[i]]
                    quant_u = self.LayerPrePost[quant]
                    c_param = (
                        qparams[i] if isinstance(qparams[i], list) else [qparams[i]]
                    )
                    qparam = self.list2Cstyle(c_param)
                    # i_type = qi_type
                    # if i_type in ['NPU_INT8']:
                    #     o_type = 'NPU_INT8'
                    # elif i_type in ['NPU_INT16']:
                    #     o_type = 'NPU_INT16'
                    # else:
                    #     assert Exception('Not implemented o_type in add operator!!!')

                LayerPre += "{"
                for content in [
                    quant,
                    qi_type,
                    ".quant_u.{}={}".format(quant_u, qparam),
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
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        if layer.get_scale_type() in ["float"]:
            i_type = o_type = "NPU_FP32"
        else:
            i_type = o_type = qo_type
        operation = self.ElementWiseType[layer.get_layer_type()]
        H, W = layer.get_insert()["feat_o"][0]
        C = layer.get_insert()["out_align"][0]
        Layer = "{},{},{},{},".format(i_type, o_type, operation, C * H * W)

        # if "pmul" in layer.get_layer_type():
        #     quant = self.LayerQuant[layer.get_scale_type()]
        #     qo_type = o_type
        #     quant_u = self.LayerPrePost[quant]
        #     qparams = get_scale_param(layer, quant)
        #     qparams = self.list2Cstyle(qparams)
        # else:
        #     quant = self.LayerQuant[""]
        #     qo_type = o_type
        #     quant_u = self.LayerPrePost[quant]
        #     qparams = 0

        # if layer.get_scale_type() in ["float"]:  ### FSCALE
        #     if layer.get_is_result_layer():
        #         qparams = [0, layer.get_scale()[0]["scale"]]
        #         qparams = self.list2Cstyle(qparams)
        #     else:
        #         quant = self.LayerQuant[""]
        #         quant_u = self.LayerPrePost[quant]
        #         qparams = 0

        mode = self.get_quant_mode(o_type, qo_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_scale(), mode)
        elif "pmul" in layer.get_layer_type() and \
            not (layer.get_scale_type() == "float" and o_type == qo_type):
            quant = self.LayerQuant[layer.get_scale_type()]
            qparams = get_scale_param(layer, quant)
            qparams = qparams if isinstance(qparams, list) else [qparams]
            qparams = self.list2Cstyle(qparams)
        else:
            quant = self.LayerQuant[""]
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="cmul")
@NETWORK_V2.register_module(name="cadd")
@NETWORK_V2.register_module(name="csub")
class LAYER_CWS(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_CWS, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = 2, 1  # self.get_io_len(layer)

        op = layer.get_ops_instance()
        op = op[0] if isinstance(op, list) else op
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        if "cmul" in layer.get_layer_type():
            mode = "" if layer.get_scale_type() != "float" else self.get_quant_mode(qi_type, "NPU_FP32")
            if mode != "":
                quants, qparams = [], []
                for i in range(in_len):
                    quant, qparam = self.get_quant(layer, layer.get_in_scale()[i], mode)
                    qparams.append(qparam)
                    quants.append(quant)
            else:
                qparams = [0 for _ in range(in_len)]
                quants = [self.LayerQuant[""] for _ in range(in_len)]
        else:
            qparams = []
            scales = copy.deepcopy(layer.get_scales())
            # if isinstance(scales, list):
            #     scales = scales[0]
            for param in scales:
                qparam = []
                for k, v in param.items():
                    if k in ["out_shift", "out_scale", "int_scale"]:
                        if k in ["out_shift", "int_scale"] and v > 0: v = -v
                        qparam.append(v)
                qparams.append(qparam)

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        LayerPre = "{" + str(in_len) + ","
        LayerPre += "{"
        for i in range(self.ELEMENT_WISE_MAX_IN):
            if i < in_len:
                if "cmul" in layer.get_layer_type():
                    quant = quants[i] # type: ignore
                    qi_type = self.NPU_DataType[layer.get_input_type()[i]]
                    quant_u = self.LayerPrePost[quant]
                    qparam = qparams[i]
                    # i_type = qi_type
                    # if layer.get_scale_type() == "float":
                        # i_type = o_type = "NPU_FP32"
                    # if i_type in ['NPU_INT8']:
                    #     o_type = 'NPU_INT32'
                    # elif i_type in ['NPU_INT16']:
                    #     o_type = 'NPU_INT64'
                    # else:
                    #     assert Exception('Not implemented o_type in mul operator!!!')
                else:
                    quant = self.LayerQuant[layer.get_scale_type()]
                    qi_type = self.NPU_DataType[layer.get_input_type()[i]]
                    quant_u = self.LayerPrePost[quant]
                    qparam = self.list2Cstyle(qparams[i])
                    # i_type = qi_type
                    # if i_type in ['NPU_INT8']:
                    #     o_type = 'NPU_INT8'
                    # elif i_type in ['NPU_INT16']:
                    #     o_type = 'NPU_INT16'
                    # else:
                    #     assert Exception('Not implemented o_type in add operator!!!')

                LayerPre += "{"
                for content in [
                    quant,
                    qi_type,
                    ".quant_u.{}={}".format(quant_u, qparam),
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
        qo_type = self.NPU_DataType[layer.get_output_type()[-1]]
        if layer.get_scale_type() in ["float"]:
            i_type = o_type = "NPU_FP32"
        else:
            i_type = o_type = qo_type
        operation = self.ChnWiseType[layer.get_layer_type()]
        H, W = layer.get_insert()["feat_o"][0]
        C = layer.get_insert()["out_align"][0]
        Layer = "{},{},{},{},".format(i_type, o_type, operation, C * H * W)

        # if "cmul" in layer.get_layer_type():
        #     quant = self.LayerQuant[layer.get_scale_type()]
        #     qo_type = o_type
        #     quant_u = self.LayerPrePost[quant]
        #     qparams = get_scale_param(layer, quant)
        #     qparams = self.list2Cstyle(qparams)
        # else:
        #     quant = self.LayerQuant[""]
        #     qo_type = o_type
        #     quant_u = self.LayerPrePost[quant]
        #     qparams = 0

        # if layer.get_scale_type() in ["float"]:  ### FSCALE
        #     if layer.get_is_result_layer():
        #         qparams = [0, layer.get_scale()[0]["scale"]]
        #         qparams = self.list2Cstyle(qparams)
        #     else:
        #         quant = self.LayerQuant[""]
        #         quant_u = self.LayerPrePost[quant]
        #         qparams = 0

        mode = self.get_quant_mode(o_type, qo_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_scale(), mode)
        elif "cmul" in layer.get_layer_type() and \
            not (layer.get_scale_type() == "float" and o_type == qo_type):
            quant = self.LayerQuant[layer.get_scale_type()]
            qparams = get_scale_param(layer, quant)
            qparams = self.list2Cstyle(qparams)
        else:
            quant = self.LayerQuant[""]
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="concat")
class LAYER_CONCAT(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_CONCAT, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = self.get_io_len(layer)

        qparams = []
        for param in layer.get_scales():
            qparam = []
            for k, v in param.items():
                if k in ["out_shift", "out_scale", "int_scale"]:
                    qparam.append(v)
            qparams.append(qparam)

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        LayerPre = "{" + str(in_len) + ","
        LayerPre += "{"
        for i in range(self.CONCAT_MAX_IN):
            if i < in_len:
                quant = self.LayerQuant[layer.get_scale_type()]
                qi_type = self.NPU_DataType[layer.get_input_type()[i]]
                quant_u = self.LayerPrePost[quant]
                qparam = self.list2Cstyle(qparams[i])

                i_type = o_type = qi_type

                LayerPre += "{"
                for content in [
                    quant,
                    qi_type,
                    ".quant_u.{}={}".format(quant_u, qparam),
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

        Layer = ""
        for content in [
            i_type, # type: ignore
            o_type, # type: ignore
            i_fmt,
            o_fmt,
            H,
            W,
            self.list2Cstyle(C),
            self.list2Cstyle(real_c),
            OC,
            real_oc,
        ]:
            Layer = _write(Layer, content)

        quant = self.LayerQuant[""]
        qo_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="shuffle_only")
class LAYER_SHUFFLE(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_SHUFFLE, self).__init__(**kwargs)

    def save(self, layer):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, 0))
        LayerPre += "}"

        # i_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_input_type(
        #     )].__name__  # self.NPU_DataType[layer.get_input_type()]
        # o_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_output_type(
        #     )].__name__  # self.NPU_DataType[layer.get_output_type()]
        # i_type = 'NPU_' + i_type.replace('u', 'U').replace('int', 'INT')
        # o_type = 'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')
        i_type = self.NPU_DataType[layer.get_input_type()[0]]
        o_type = self.NPU_DataType[layer.get_output_type()[0]]
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        H, W = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        sec_num = len(layer.get_insert()["in_pad"][0])

        c_start = [a for a, b in layer.get_insert()["in_pad"]]
        while len(c_start) < self.SHUFFLE_MAX_IN_SECTION:
            c_start.append(0)

        c_end = [b for a, b in layer.get_insert()["in_pad"]]
        while len(c_end) < self.SHUFFLE_MAX_IN_SECTION:
            c_end.append(0)
        OC = layer.get_insert()["out_align"][0]

        Layer = ""
        for content in [
            i_type,
            o_type,
            i_fmt,
            o_fmt,
            H,
            W,
            C,
            self.list2Cstyle(c_start),
            self.list2Cstyle(c_end),
            sec_num,
            OC,
        ]:
            Layer = _write(Layer, content)

        quant = self.LayerQuant[""]
        qo_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="split")
class LAYER_SPLIT(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_SPLIT, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = self.get_io_len(layer)

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]

        LayerPre = "{" + str(out_len) + ","
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, 0))
        LayerPre += "}"

        # i_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_input_type(
        #     )].__name__  # self.NPU_DataType[layer.get_input_type()]
        # o_type = layer.get_ops_setting()['setting']['bits_dict'][
        #     layer.get_output_type(
        #     )].__name__  # self.NPU_DataType[layer.get_output_type()]
        i_type = self.NPU_DataType[
            layer.get_input_type()[0]
        ]  #'NPU_' + i_type.replace('u', 'U').replace('int', 'INT')
        o_type = self.NPU_DataType[
            layer.get_input_type()[0]
        ]  #'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')
        i_fmt = self.CubeFmt[self.fmt]
        o_fmt = self.CubeFmt[self.fmt]
        H, W = layer.get_insert()["split"]["feat_i"][0]

        C = layer.get_insert()["split"]["in_align"][0]
        real_c = np.sum([b - a for a, b in layer.get_insert()["split"]["in_pad"]])

        # split_ids = [v for _, v in layer.get_insert()['split_ids'].items()]

        OC = layer.get_insert()["split"]["out_align"]
        # OC = [OC[id] for id in split_ids]
        while len(OC) < self.SPLIT_MAX_OUT:
            OC.append(0)

        real_oc = [b - a for a, b in layer.get_insert()["split"]["out_pad"]]
        # real_oc = [real_oc[id] for id in split_ids]
        while len(real_oc) < self.SPLIT_MAX_OUT:
            real_oc.append(0)

        Layer = ""
        for content in [
            i_type,
            o_type,
            i_fmt,
            o_fmt,
            H,
            W,
            C,
            real_c,
            self.list2Cstyle(OC),
            self.list2Cstyle(real_oc),
        ]:
            Layer = _write(Layer, content)

        # qparams = []
        # for param in layer.get_scales()[0]:
        #     qparams.append([v for k, v in param.items()])

        qparams = []
        scales = copy.deepcopy(layer.get_scales())
        # if isinstance(scales, list):
        #     scales = scales[0]
        for param in scales:
            qparam = []
            for k, v in param.items():
                if k in ["out_shift", "out_scale", "int_scale"]:
                    qparam.append(v)
            qparams.append(qparam)

        split_ids = layer.get_insert()["split_ids"]

        LayerPost = "{"
        for i in range(self.SPLIT_MAX_OUT):
            if i < out_len:
                param_id = split_ids[layer.get_output_idx()[i]]
                quant = self.LayerQuant[layer.get_scale_type()]
                qo_type = self.NPU_DataType[layer.get_output_type()[param_id]]
                quant_u = self.LayerPrePost[quant]
                qparam = self.list2Cstyle(qparams[param_id])

                LayerPost += "{"
                for content in [
                    quant,
                    qo_type,
                    ".quant_u.{}={}".format(quant_u, qparam),
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

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="shuffle")
class LAYER_CONCAT_SHUFFLE_SPLIT(NetworkV2):
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

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="resize")
class LAYER_RESIZE(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_RESIZE, self).__init__(**kwargs)

    def save(self, layer):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]

        # if qi_type in ["NPU_INT8", "NPU_INT16"] and layer.get_scale_type() == "float":
        #     # is_perchannel = isinstance(layer.get_scales()[0]['out_shift'], np.ndarray)
        #     is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        #     quant = ""
        #     # if is_perchannel:
        #     # quant += 'perchannel_'
        #     quant += "dequant"
        #     if is_asymquan:
        #         quant += "_asy"
        #     quant = self.LayerQuant[quant]
        #     # if is_perchannel:
        #     #     qparams = [layer.get_w_offset()['tmp_offset'][0]] #offset
        #     # else:
        #     if is_asymquan:
        #         qparams = [
        #             layer.get_in_scale()["scale"],
        #             layer.get_in_scale()["zero_point"],
        #         ]  # scale, zero
        #     else:
        #         qparams = [layer.get_in_scale()["scale"]]  # scale
        #     qparams = self.list2Cstyle(qparams)
        # else:
        #     quant = self.LayerQuant[""]
        #     qparams = 0
        op = layer.get_ops_instance()
        op = op[0] if isinstance(op, list) else op
        i_type = o_type = "NPU_FP32"
        mode = self.get_quant_mode(qi_type, i_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_in_scale(), mode)
        else:
            quant = self.LayerQuant[""]
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, qparams))
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

        IH, IW = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        OH, OW = layer.get_insert()["feat_o"][0]
        mode = layer.get_ops_setting()["attrs"][0]["mode"]
        method = self.ResizeMethod[mode]  # 'RESIZE_BILINEAR'
        if "INT" in i_type:
            method += "_FIXED_POINT"
        if mode in ["linear", "cubic"]:
            mode = "bi" + mode
        param = ".param.{}=".format(mode) + "{0,0}"

        Layer = ""
        for content in [i_type, o_type, i_fmt, o_fmt, IH, IW, C, OH, OW, method, param]:
            Layer = _write(Layer, content)

        qo_type = self.NPU_DataType[layer.get_output_type()[0]]
        if layer.get_is_result_layer():
            quant = get_last_layer_quant(layer)
        else:
            if (
                qo_type in ["NPU_INT8", "NPU_INT16"]
                and layer.get_scale_type() == "float"
            ):
                quant = self.LayerQuant["quant"]
                qparams = [layer.get_scale()[0]["scale"]]
                qparams = self.list2Cstyle(qparams)
            else:
                quant = self.LayerQuant[""]
                qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="softmax")
class LAYER_SOFTMAX(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_SOFTMAX, self).__init__(**kwargs)

    def save(self, layer):
        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]

        # if qi_type in ["NPU_INT8", "NPU_INT16"] and layer.get_scale_type() == "float":
        #     # is_perchannel = isinstance(layer.get_scales()[0]['out_shift'], np.ndarray)
        #     is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
        #     quant = ""
        #     # if is_perchannel:
        #     # quant += 'perchannel_'
        #     quant += "dequant"
        #     if is_asymquan:
        #         quant += "_asy"
        #     quant = self.LayerQuant[quant]
        #     # if is_perchannel:
        #     #     qparams = [layer.get_w_offset()['tmp_offset'][0]] #offset
        #     # else:
        #     if is_asymquan:
        #         qparams = [
        #             layer.get_in_scale()["scale"],
        #             layer.get_in_scale()["zero_point"],
        #         ]  # scale, zero
        #     else:
        #         qparams = [layer.get_in_scale()["scale"]]  # scale
        #     qparams = self.list2Cstyle(qparams)
        # else:
        #     quant = self.LayerQuant[""]
        #     qparams = 0
        op = layer.get_ops_instance()
        op = op[0] if isinstance(op, list) else op
        i_type = o_type = "NPU_FP32"
        mode = self.get_quant_mode(qi_type, i_type)
        if mode != "":
            quant, qparams = self.get_quant(layer, layer.get_in_scale(), mode)
        else:
            quant = self.LayerQuant[""]
            qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPre = "{"
        LayerPre += "{"
        LayerPre = _write(LayerPre, quant)
        LayerPre = _write(LayerPre, qi_type)
        LayerPre = _write(LayerPre, ".quant_u.{}={}".format(quant_u, qparams))
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

        IH, IW = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        axis = layer.get_ops_setting()["attrs"][0]["axis"]

        Layer = ""
        for content in [i_type, o_type, i_fmt, o_fmt, IH, IW, C, axis]:
            Layer = _write(Layer, content)

        qo_type = self.NPU_DataType[layer.get_output_type()[0]]
        if layer.get_is_result_layer():
            quant = get_last_layer_quant(layer)
        else:
            if (
                qo_type in ["NPU_INT8", "NPU_INT16"]
                and layer.get_scale_type() == "float"
            ):
                quant = self.LayerQuant["quant"]
                qparams = [layer.get_scale()[0]["scale"]]
                qparams = self.list2Cstyle(qparams)
            else:
                quant = self.LayerQuant[""]
                qparams = 0
        quant_u = self.LayerPrePost[quant]

        LayerPost = "{"
        for content in [quant, qo_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="reshape")
class LAYER_RESHAPE(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_RESHAPE, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = self.get_io_len(layer)

        qparams = []

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]

        LayerPre = "{"
        LayerPre += "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPre = _write(LayerPre, content)
        LayerPre += "}"

        i_type = self.NPU_DataType[layer.get_input_type()[0]]
        o_type = self.NPU_DataType[layer.get_output_type()[0]]
        i_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]
        o_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]

        # _, C, H, W = layer.get_in_data()[0]['output'].shape
        if "split" in layer.get_insert().keys():
            C = layer.get_insert()["split"]["in_align"][0]
            H, W = layer.get_insert()["split"]["feat_i"][0]
        else:
            C = layer.get_insert()["in_align"][0]
            H, W = layer.get_insert()["feat_i"][0]

        Layer = ""
        for content in [i_type, o_type, i_fmt, o_fmt, H, W, C]:
            Layer = _write(Layer, content)

        LayerPost = "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="pad")
class LAYER_PAD(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_PAD, self).__init__(**kwargs)

    def save(self, layer):
        contents = ""
        return contents
        
        
@NETWORK_V2.register_module(name="lstm")
class LAYER_LSTM(NetworkV2):
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

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]

        lstm_q = "LSTM_QUANT_I_H_DIFF"
        LayerPre = "{" + lstm_q + ","
        LayerPre += "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]
        qparams = 0
        is_quantize_from_in_data = False
        if hasattr(layer, "get_quantize_from_in_data"):
            is_quantize_from_in_data = layer.get_quantize_from_in_data()
            
        if qi_type in ["NPU_FP32", "NPU_FP64"]:
            is_perchannel = isinstance(layer.get_scales()[0]["out_shift"], np.ndarray)
            is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
            quant = ""
            if is_perchannel:
                quant += "perchannel_"
            quant += "quant"
            if is_asymquan:
                quant += "_asy"
            quant = self.LayerQuant[quant]
            quant_u = self.LayerPrePost[quant]
            if "ASY" in quant:
                qparams = [layer.get_in_quantize()[0].get_scale()[0], layer.get_in_quantize()[0].get_scale()[1]]
            else:
                qparams = [layer.get_in_quantize()[0].get_scale()[0]]
            qparams = self.list2Cstyle(qparams)

            # quant = 'QUANT_FSCALE'
            # quant_u = self.LayerPrePost[quant]
            # qparams = [0, 1.0/layer.get_in_quantize()[0].get_scale()[0]]
            # qparams = self.list2Cstyle(qparams)
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPre = _write(LayerPre, content)
        LayerPre += "}"
        
        dynamic_q_i = np.int32(1) if is_quantize_from_in_data else np.int32(0)
        
        LayerPre += _write(",", dynamic_q_i, tail="")

        q_i = [v for k, v in layer.get_in_scale()[0].items()]
        q_h = [v for k, v in layer.get_in_scale()[1].items()]
        q_w = [v for k, v in layer.get_w_scale()[0].items()]
        q_r = [v for k, v in layer.get_w_scale()[1].items()]
        q_ib = [1.0, 0.0]
        q_hb = [1.0, 0.0]
        q_wb = [1.0, 0.0]
        q_rb = [1.0, 0.0]
        i_type = self.NPU_DataType[
            layer.get_output_type()[3]
        ]  # layer.get_input_type()[0]
        o_type = self.NPU_DataType[layer.get_output_type()[0]]
        i_fmt = self.MatFmt[self.fmt]
        o_fmt = self.MatFmt[self.fmt]

        seq_len = 1  # layer.get_layer_ops()['attrs']['sequence_lens']
        seq_len = 1 #layer.get_layer_ops()['attrs']['sequence_lens']
        i_size = layer.get_insert()['split']['in_align'][0]
        o_size = layer.get_insert()['split']['out_align'][0]
        hidden_size = layer.get_ops_setting()['attrs'][0]['hidden_size']
        fc_o_size = layer.get_insert()['split']['out_align'][-1]
        input_forget = -1
        has_bias = 1
        direction = "LSTM_FORWARD"
        act_list = ["ACT_SIGMOID", "ACT_TANH", "ACT_TANH"]
        for i in range(6):
            if i >= 3:
                act_list.append("ACT_NONE")
        act_list_u = [[0] for i in range(6)]
        lut = ["LUT_NONE" for i in range(6)]
        tmp_offset = layer.get_w_offset()["tmp_offset"]
        lut_off = [tmp_offset[i] for i in range(6)]
        w_off = tmp_offset[6]
        r_off = tmp_offset[7]
        wb_off = tmp_offset[8]
        rb_off = tmp_offset[9]
        init_h_off = tmp_offset[10]
        init_c_off = tmp_offset[11]
        p_off = -1
        pb_off = -1

        Layer = ""
        for content in [
            self.list2Cstyle(q_i),
            self.list2Cstyle(q_h),
            self.list2Cstyle(q_w),
            self.list2Cstyle(q_r),
            self.list2Cstyle(q_ib),
            self.list2Cstyle(q_hb),
            self.list2Cstyle(q_wb),
            self.list2Cstyle(q_rb),
            i_type,
            o_type,
            i_fmt,
            o_fmt,
            seq_len,
            i_size,
            hidden_size,
            fc_o_size,
            o_size,
            input_forget,
            has_bias,
            direction,
            self.list2Cstyle(act_list),
            self.list2Cstyle(act_list_u),
            self.list2Cstyle(lut),
            self.list2Cstyle(lut_off),
            w_off,
            r_off,
            wb_off,
            rb_off,
            init_h_off,
            init_c_off,
            p_off,
            pb_off,
        ]:
            Layer = _write(Layer, content)

        LayerPost = "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="gru")
class LAYER_GRU(NetworkV2):
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

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]

        if layer.get_hx_combine() and layer.get_wr_combine():
            lstm_q = "LSTM_QUANT_I_H_SAME"
        else:
            lstm_q = "LSTM_QUANT_I_H_DIFF"
        LayerPre = "{" + lstm_q + ","
        LayerPre += "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]
        qparams = 0
        if qi_type in ["NPU_FP32", "NPU_FP64"]:
            # is_perchannel = isinstance(layer.get_scales()[0]["out_shift"], np.ndarray)
            # is_asymquan = "asymquan" in layer.get_ops_setting()["setting"]["method"]
            # quant = ""
            # if is_perchannel:
            #     quant += "perchannel_"
            # quant += "quant"
            # if is_asymquan:
            #     quant += "_asy"
            # quant = self.LayerQuant[quant]
            # quant_u = self.LayerPrePost[quant]
            # qparams = [0, layer.get_in_quantize()[0].get_scale()[0]]
            # qparams = self.list2Cstyle(qparams)

            quant = 'QUANT_FSCALE'
            quant_u = self.LayerPrePost[quant]
            qparams = [0, 1.0/layer.get_in_quantize()[0].get_scale()[0]]
            qparams = self.list2Cstyle(qparams)
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, qparams)]:
            LayerPre = _write(LayerPre, content)
        LayerPre += "}"

        q_i = [v for k, v in layer.get_in_scale()[0].items()]
        q_h = [v for k, v in layer.get_in_scale()[1].items()]
        q_w = [v for k, v in layer.get_w_scale()[0].items()]
        q_r = [v for k, v in layer.get_w_scale()[1].items()]
        q_ib = [1.0, 0.0]
        q_hb = [1.0, 0.0]
        q_wb = [1.0, 0.0]
        q_rb = [1.0, 0.0]
        i_type = self.NPU_DataType[
            layer.get_output_type()[3]
        ]  # layer.get_input_type()[0]
        o_type = self.NPU_DataType[layer.get_output_type()[0]]
        i_fmt = self.MatFmt[self.fmt]
        o_fmt = self.MatFmt[self.fmt]

        seq_len = 1  # layer.get_layer_ops()['attrs']['sequence_lens']
        seq_len = 1 #layer.get_layer_ops()['attrs']['sequence_lens']
        i_size = layer.get_insert()['split']['in_align'][0]
        o_size = layer.get_insert()['split']['out_align'][0]
        hidden_size = layer.get_ops_setting()['attrs'][0]['hidden_size']
        fc_o_size = layer.get_insert()['split']['out_align'][-1]
        input_forget = -1
        has_bias = int(layer.get_ops_setting()["attrs"][0]["bias"])
        direction = "LSTM_FORWARD"
        act_list = ["ACT_SIGMOID", "ACT_TANH", "ACT_TANH"]
        for i in range(6):
            if i >= 3:
                act_list.append("ACT_NONE")
        act_list_u = [[0] for i in range(6)]
        lut = ["LUT_NONE" for i in range(6)]
        tmp_offset = layer.get_w_offset()["tmp_offset"]
        lut_off = [tmp_offset[i] for i in range(6)]
        w_off = tmp_offset[6]
        r_off = tmp_offset[7]
        wb_off = tmp_offset[8]
        rb_off = tmp_offset[9]
        init_h_off = tmp_offset[10]
        p_off = -1
        pb_off = -1

        Layer = ""
        for content in [
            self.list2Cstyle(q_i),
            self.list2Cstyle(q_h),
            self.list2Cstyle(q_w),
            self.list2Cstyle(q_r),
            self.list2Cstyle(q_ib),
            self.list2Cstyle(q_hb),
            self.list2Cstyle(q_wb),
            self.list2Cstyle(q_rb),
            i_type,
            o_type,
            i_fmt,
            o_fmt,
            seq_len,
            i_size,
            hidden_size,
            fc_o_size,
            o_size,
            input_forget,
            has_bias,
            direction,
            self.list2Cstyle(act_list),
            self.list2Cstyle(act_list_u),
            self.list2Cstyle(lut),
            self.list2Cstyle(lut_off),
            w_off,
            r_off,
            wb_off,
            rb_off,
            init_h_off,
            p_off,
            pb_off,
        ]:
            Layer = _write(Layer, content)

        LayerPost = "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="layernormalization")
class LAYER_LN(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_LN, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = self.get_io_len(layer)

        qparams = []

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]

        LayerPre = "{"
        LayerPre += "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPre = _write(LayerPre, content)
        LayerPre += "}"

        i_type = self.NPU_DataType[layer.get_input_type()[0]]
        o_type = self.NPU_DataType[layer.get_output_type()[0]]
        i_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]
        o_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]

        # _, C, H, W = layer.get_in_data()[0]['output'].shape
        # B = 74
        if "split" in layer.get_insert().keys():
            C = layer.get_insert()["split"]["in_align"][0]
            H, W = layer.get_insert()["split"]["feat_i"][0]
        else:
            C = layer.get_insert()["in_align"][0]
            H, W = layer.get_insert()["feat_i"][0]

        eps = layer.get_ops_setting()['attrs'][0]['epsilon']
        offset = layer.get_w_offset()["w_offset"]
        affine = 1 if offset != -1 else 0

        Layer = ""
        for content in [i_type, o_type, i_fmt, o_fmt, H, W, C, eps, affine, offset]:
            Layer = _write(Layer, content)

        LayerPost = "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="instancenormalization")
class LAYER_IN(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_IN, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = self.get_io_len(layer)

        qparams = []

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]

        LayerPre = "{"
        LayerPre += "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPre = _write(LayerPre, content)
        LayerPre += "}"

        i_type = self.NPU_DataType[layer.get_input_type()[0]]
        o_type = self.NPU_DataType[layer.get_output_type()[0]]
        i_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]
        o_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]

        # _, C, H, W = layer.get_in_data()[0]['output'].shape
        if "split" in layer.get_insert().keys():
            C = layer.get_insert()["split"]["in_align"][0]
            H, W = layer.get_insert()["split"]["feat_i"][0]
        else:
            C = layer.get_insert()["in_align"][0]
            H, W = layer.get_insert()["feat_i"][0]

        eps = layer.get_ops_setting()['attrs'][0]['epsilon']
        offset = layer.get_w_offset()["w_offset"]
        affine = 1 if offset != -1 else 0

        Layer = ""
        for content in [i_type, o_type, i_fmt, o_fmt, H, W, C, eps, affine, offset]:
            Layer = _write(Layer, content)

        LayerPost = "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents


@NETWORK_V2.register_module(name="batchnormalization")
class LAYER_BN(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_BN, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = self.get_io_len(layer)

        qparams = []

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]

        LayerPre = "{"
        LayerPre += "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPre = _write(LayerPre, content)
        LayerPre += "}"

        i_type = self.NPU_DataType[layer.get_input_type()[0]]
        o_type = self.NPU_DataType[layer.get_output_type()[0]]
        i_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]
        o_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]

        # _, C, H, W = layer.get_in_data()[0]['output'].shape
        if "split" in layer.get_insert().keys():
            C = layer.get_insert()["split"]["in_align"][0]
            H, W = layer.get_insert()["split"]["feat_i"][0]
        else:
            C = layer.get_insert()["in_align"][0]
            H, W = layer.get_insert()["feat_i"][0]

        eps = layer.get_ops_setting()['attrs'][0]['epsilon']
        offset = layer.get_w_offset()["w_offset"]
        affine = 1 if offset != -1 else 0

        Layer = ""
        for content in [i_type, o_type, i_fmt, o_fmt, H, W, C, eps, affine, offset]:
            Layer = _write(Layer, content)

        LayerPost = "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents
    
    
@NETWORK_V2.register_module(name="log")
class LAYER_LOG(NetworkV2):
    def __init__(self, **kwargs):
        super(LAYER_LOG, self).__init__(**kwargs)

    def save(self, layer):
        in_len, out_len = self.get_io_len(layer)

        qparams = []

        LayerType = self.layer_map_inv[layer.get_layer_type()]
        LayerInfo = self.LayerInstance[LayerType]

        LayerPre = "{"
        LayerPre += "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_input_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPre = _write(LayerPre, content)
        LayerPre += "}"

        i_type = self.NPU_DataType[layer.get_input_type()[0]]
        o_type = self.NPU_DataType[layer.get_output_type()[0]]
        # i_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]
        # o_fmt = getattr(self, layer.get_insert()["fmt"])[self.fmt]

        # _, C, H, W = layer.get_in_data()[0]['output'].shape
        if "split" in layer.get_insert().keys():
            C = layer.get_insert()["split"]["in_align"][0]
            H, W = layer.get_insert()["split"]["feat_i"][0]
        else:
            C = layer.get_insert()["in_align"][0]
            H, W = layer.get_insert()["feat_i"][0]

        Layer = ""
        # node_name = layer.get_nodes()[0].get_name()
        base = 2.0 #if node_name[-1] == "2" else 10
        # for content in [i_type, o_type, i_fmt, o_fmt, H, W, C]:
        for content in [i_type, o_type, H*W*C, self.list2Cstyle(base)]:
            Layer = _write(Layer, content)

        LayerPost = "{"
        quant = self.LayerQuant[""]
        qi_type = self.NPU_DataType[layer.get_output_type()[0]]
        quant_u = self.LayerPrePost[quant]
        for content in [quant, qi_type, ".quant_u.{}={}".format(quant_u, 0)]:
            LayerPost = _write(LayerPost, content)
        LayerPost += "},"
        LayerPost += "}"

        contents = self.get_contents_v2(
            layer, LayerType, LayerInfo, LayerPre, Layer, LayerPost
        )

        return contents