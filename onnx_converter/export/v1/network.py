# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/20 11:32
# @File     : network.py
import copy
import os
import struct

try:
    from utils import Object, Registry, invert_dict
except Exception:
    from onnx_converter.utils import Object, Registry, invert_dict # type: ignore

NETWORK_V1: Registry = Registry("network_v1", scope="")


def _write(content, val, tail=","):
    return content + str(val) + tail


def float_to_hex(f):
    """convert from the float value to the string of its hex value
    for example, 0.1234 is converted to "0x3dfcb924"
    """
    return hex(struct.unpack("<i", struct.pack("<f", f))[0])


class NetworkBase(Object): # type: ignore
    def __init__(self, **kwargs):
        super(NetworkBase, self).__init__()
        self.MAX_IN_OUT_LEN = None
        self.kwargs = None

    @staticmethod
    def list2Cstyle(x: list):
        return str(x).replace("[", "{").replace("]", "}")

    @staticmethod
    def get_align_channel(ch, align_size):
        return ((ch + align_size - 1) // align_size) * align_size

    def invert_type(self, invert_types):
        for type in invert_types:
            tmp = {}
            for key, value in self.kwargs[type].items(): # type: ignore
                for v in value:
                    tmp[v] = key
            setattr(self, type, tmp)

    def invert_dict_of_list(self, dict_a):
        dict_b = {}
        for key, value in dict_a.items():
            for v in value:
                dict_b[v] = key

        return dict_b

    def get_io_len(self, layer):
        in_len = len(layer.get_input_idx())
        out_len = len(layer.get_output_idx())

        return in_len, out_len

    def get_contents(self, layer, contents):
        input_ids = copy.deepcopy(layer.get_input_idx())
        output_ids = copy.deepcopy(layer.get_output_idx())
        if layer.get_layer_type() in ["lstm"]:
            in_len = len(input_ids) #len([id for id in input_ids if id >= 0])
            out_len = len(output_ids) #len([id for id in output_ids if id >= 0])
        else:
            in_len = len(input_ids)
            out_len = len(output_ids)
        while len(input_ids) < self.MAX_IN_OUT_LEN: # type: ignore
            input_ids.append(0)
        while len(output_ids) < self.MAX_IN_OUT_LEN: # type: ignore
            output_ids.append(0)

        contents = _write(contents, in_len, tail=",")
        contents = _write(contents, out_len, tail=",")

        contents += "{"
        for id, data in enumerate(input_ids):
            if id == len(input_ids) - 1:
                contents = _write(contents, data, tail="")
            else:
                contents = _write(contents, data, tail=",")
        contents += "},"

        contents += "{"
        for id, data in enumerate(output_ids):
            if id == len(output_ids) - 1:
                contents = _write(contents, data, tail="")
            else:
                contents = _write(contents, data, tail=",")
        contents += "}"

        contents = contents.replace(",}", "}")
        contents = contents.replace("'", "")

        return contents

    def save(self, layer):
        pass

    def __call__(self, layer):
        self.save(layer)


# export special layer in model.c
# written network structure in binary file
class NetworkV1(NetworkBase):
    def __init__(self, **kwargs):
        super(NetworkV1, self).__init__(**kwargs)

        self.kwargs = kwargs
        self.quan_method = kwargs["QUAN_METHOD"]
        self.MAX_IN_OUT_LEN = kwargs["MAX_IN_OUT_LEN"]
        self.QUAN_FLOAT = kwargs["QUAN_FLOAT"]
        self.RESULT_INT = kwargs["RESULT_INT"]
        self.QUAN_SHIFT = kwargs["QUAN_SHIFT"]
        self.QUAN_POST_SHIFT = kwargs["QUAN_POST_SHIFT"]
        self.ops_type = kwargs["ops_type"]
        self.AT5050 = kwargs["AT5050_C_EXTEND"]
        self.AT5050_C_EXTEND = kwargs["AT5050_C_EXTEND"]
        self.MAC_256 = kwargs["MAC_256"]
        self.AC = kwargs["Csize"]
        self.AK = kwargs["Ksize"]
        self.I_Align = kwargs["I_Align"]
        self.O_Align = kwargs["O_Align"]
        self.ADD_FLOAT = kwargs["ADD_FLOAT"]
        self.QUAN_PRE_CONCAT = kwargs["QUAN_PRE_CONCAT"]
        self.OPS_INT8 = kwargs["OPS_INT8"]
        self.OPS_INT8_TO_FP = kwargs["OPS_INT8_TO_FP"]
        self.OPS_INT16 = kwargs["OPS_INT16"]
        self.OPS_INT32_TO_FP = kwargs["OPS_INT32_TO_FP"]
        self.OPS_INT32_TO_INT8 = kwargs["OPS_INT32_TO_INT8"]
        self.OPS_INT8_TO_INT8_INT_SCALE = kwargs["OPS_INT8_TO_INT8_INT_SCALE"]
        self.OPS_FP32 = kwargs["OPS_FP32"]
        self.FC_W_LEN = kwargs["FC_W_LEN"]
        self.CONV_W_LEN = kwargs["CONV_W_LEN"]
        self.DPCONV_W_LEN = kwargs["DPCONV_W_LEN"]
        self.QUAN_FLOAT_SYM = kwargs["QUAN_FLOAT_SYM"]
        self.QUAN_NONE = kwargs["QUAN_NONE"]
        self.CONCAT_FLOAT = kwargs["CONCAT_FLOAT"]
        self.INTERP_NEAREST = kwargs["INTERP_NEAREST"]
        self.INTERP_BILINEAR = kwargs["INTERP_BILINEAR"]

        self.layer_map = kwargs["layer_map"]
        self.layer_map_inv = self.invert_dict_of_list(self.layer_map)
        self.LayerInstance = kwargs["LayerInstance"]
        self.LayerQuant = self.invert_dict_of_list(kwargs["LayerQuant"])
        self.NPU_DataType = kwargs["NPU_DataType"]
        self.CubeFmt = kwargs["CubeFmt"]
        self.ConvWFmt = kwargs["ConvWFmt"]
        self.Csize = kwargs["bits"]["Csize"]
        self.Ksize = kwargs["bits"]["Ksize"]
        self.LayerPrePost = self.invert_dict_of_list(kwargs["LayerPrePost"])
        self.ActivationType = self.invert_dict_of_list(kwargs["ActivationType"])
        self.invert_type(["ReduceType", "PoolType", "ElementWiseType"])
        self.ResizeMethod = invert_dict(kwargs["ResizeMethod"])
        self.CONCAT_SHUFFLE_SPLIT_MAX_IN = kwargs["model_c"][
            "CONCAT_SHUFFLE_SPLIT_MAX_IN"
        ]
        self.CONCAT_SHUFFLE_SPLIT_MAX_OUT = kwargs["model_c"][
            "CONCAT_SHUFFLE_SPLIT_MAX_OUT"
        ]
        self.SHUFFLE_MAX_IN_SECTION = kwargs["model_c"]["SHUFFLE_MAX_IN_SECTION"]
        self.CONCAT_MAX_IN = kwargs["model_c"]["CONCAT_MAX_IN"]
        self.SPLIT_MAX_OUT = kwargs["model_c"]["SPLIT_MAX_OUT"]


@NETWORK_V1.register_module(name="data")
class LAYER_PLACEHOLDER(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_PLACEHOLDER, self).__init__(**kwargs)

    def save(self, layer):
        type = "PLACEHOLDER"
        scale = layer.get_scale()[0]["scale"]
        if len(layer.get_insert()["feat_o"][0]) == 2:
            H, W = layer.get_insert()["feat_o"][0]
        else:
            H, W = 1, 1
        # C = layer.get_insert()['out_pad'][0][1]
        # C = 4 if self.AT5050_C_EXTEND else self.get_align_channel(C, self.AC)
        C = layer.get_insert()["out_align"][0]

        contents = "{"
        contents = _write(contents, type)
        contents = _write(contents, ".u.{}=".format(type.lower()), tail="")
        contents = _write(contents, "{", tail="")
        contents = _write(contents, H, tail=",")
        contents = _write(contents, W, tail=",")
        contents = _write(contents, C, tail=",")
        if self.quan_method < self.QUAN_FLOAT:
            # contents = _write(contents, shift, tail='},')
            print("NotImplementedError")
            os._exit(-1)
        else:
            contents = _write(contents, float_to_hex(scale), tail="},")
        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


@NETWORK_V1.register_module(name="conv")
@NETWORK_V1.register_module(name="depthwiseconv")
class LAYER_CONV2D(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_CONV2D, self).__init__(**kwargs)

    def save(self, layer):
        type = self.layer_map_inv[layer.get_layer_type()]
        # if self.MAC_256:
        #     if type == 'CONV2D':
        #         self.AK = 16
        #     elif type == 'DEPTH_WISE_CONV2D':
        #         self.AK = 8
        #     else:
        #         assert 'NotImplementedError!!!'
        #       os._exit(-1)

        pad_t, pad_b, pad_l, pad_r = layer.get_layer_ops()["attrs"][0]["pads"]
        SH, SW = layer.get_layer_ops()["attrs"][0]["strides"]
        H, W = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]

        # C = layer.get_ops_setting()['attrs'][0]['in_c']
        # if C == 3 and self.AT5050_C_EXTEND:
        #     C = 4
        # else:
        #     C = self.get_align_channel(C, self.AC)

        FH, FW = layer.get_layer_ops()["attrs"][0]["kernel_shape"]
        if type == "DEPTH_WISE_CONV2D":
            K = 1
        else:
            K = layer.get_insert()["out_align"][0]
        # K = layer.get_ops_setting()['attrs'][0]['out_c']
        # K = self.get_align_channel(K, self.AK)

        w_offset = layer.get_w_offset()["w_offset"]
        w_act = self.ActivationType[layer.get_ops_setting()["ops_string"][-1]]
        hasBias = int(layer.get_ops_setting()["attrs"][0]["bias"])
        PH, PW, PSH, PSW, pPad = 0, 0, 0, 0, 0
        is_result_layer = layer.get_is_result_layer()

        in_type = self.OPS_INT8
        out_type = self.OPS_INT8_TO_INT8_INT_SCALE
        if is_result_layer and self.RESULT_INT:
            out_type = self.OPS_INT8
        if (type == "CONV2D" and self.CONV_W_LEN == 16) or (
            type == "DEPTH_WISE_CONV2D" and self.DPCONV_W_LEN == 16
        ):
            w_type = self.OPS_INT16
        else:
            w_type = self.OPS_INT8

        # not used currently
        symmetric = 1
        pad_value = 0

        if self.quan_method < self.QUAN_FLOAT:  # shift quan method
            print("NotImplementedError")
            os._exit(-1)
        else:  # float Quan method
            in_shift = layer.get_scale()[0]["scale"]
            conv_shift = layer.get_scales()[-1]["out_shift"]  # out_shift
            final_scale_shift = layer.get_scales()[-1]["out_scale"]  # out_scale

        contents = "{"
        contents = _write(contents, type)  # type
        contents = _write(contents, ".u.conv={" + str(pad_t))
        contents = _write(contents, pad_b)
        contents = _write(contents, pad_l)
        contents = _write(contents, pad_r)
        contents = _write(contents, SH)
        contents = _write(contents, SW)
        contents = _write(contents, H)
        contents = _write(contents, W)
        contents = _write(contents, C)
        if self.quan_method < self.QUAN_FLOAT:
            # contents = _write(contents, in_shift)
            print("NotImplementedError")
            os._exit(-1)
        else:
            # write the scale to i32 to work with the interpreter
            if is_result_layer and self.RESULT_INT:
                contents = _write(contents, 0)
            else:
                contents = _write(contents, float_to_hex(in_shift))
        contents = _write(contents, FH)
        contents = _write(contents, FW)
        contents = _write(contents, K)
        contents = _write(contents, w_offset)
        if is_result_layer:
            contents = _write(contents, -conv_shift)
        else:
            contents = _write(contents, conv_shift)

        contents = _write(contents, w_act)
        contents = _write(contents, "N_POOL")

        contents = _write(contents, PH)
        contents = _write(contents, PW)
        contents = _write(contents, PSH)
        contents = _write(contents, PSW)
        contents = _write(contents, pPad)
        if self.quan_method < self.QUAN_FLOAT:
            # contents = _write(contents, out_shift)
            print("NotImplementedError")
            os._exit(-1)
        else:
            # write the scale to i32 to work with the interpreter
            if is_result_layer and self.RESULT_INT:
                contents = _write(contents, 0)  # change divide to multiply
            else:
                contents = _write(
                    contents, float_to_hex(final_scale_shift)
                )  # change divide to multiply

        contents = _write(contents, hasBias)

        # For SDKV2
        contents = _write(contents, symmetric)
        contents = _write(contents, pad_value)  # the pad value
        contents = _write(contents, in_type)
        contents = _write(contents, w_type)
        contents = _write(contents, out_type)

        if self.AT5050:
            if self.RESULT_INT and is_result_layer:
                contents = _write(contents, self.QUAN_SHIFT)
                contents = _write(contents, "0")
            else:
                contents = _write(contents, self.QUAN_POST_SHIFT)
                contents = _write(contents, ".u.post_shift.scale=", tail="")
                if self.quan_method < self.QUAN_FLOAT:
                    # contents = _write(contents, out_shift)
                    print("NotImplementedError")
                    os._exit(-1)
                else:
                    # write the scale to i32 to work with the interpreter
                    if isinstance(final_scale_shift, float):
                        contents = _write(contents, float_to_hex(final_scale_shift))
                    else:
                        contents = _write(contents, final_scale_shift)

        contents = _write(contents, "}")
        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


@NETWORK_V1.register_module(name="fc")
class LAYER_FC(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_FC, self).__init__(**kwargs)

    def save(self, layer):
        type = self.layer_map_inv[layer.get_layer_type()]
        hasBias = int(layer.get_ops_setting()["attrs"][0]["bias"])
        act = self.ActivationType[layer.get_ops_setting()["ops_string"][-1]]
        w_offset = layer.get_w_offset()["w_offset"]
        is_result_layer = layer.get_is_result_layer()
        iter_size = 1

        # in_size = layer.get_ops_setting()['attrs'][0]['in_c']
        # in_size = self.get_align_channel(in_size, self.I_Align)
        # out_size = layer.get_ops_setting()['attrs'][0]['out_c']
        # out_size = self.get_align_channel(out_size, self.O_Align)
        in_size = layer.get_insert()["in_align"][0]
        out_size = layer.get_insert()["out_align"][0]

        in_shift = layer.get_scale()[0]["scale"]
        w_shift = layer.get_scales()[-1]["out_shift"]  # the shift of the weight
        final_scale_shift = layer.get_scales()[-1][
            "out_scale"
        ]  # the scale from int32/int8 to int8/FP

        in_type = self.OPS_INT8
        if self.FC_W_LEN == 16:
            w_type = self.OPS_INT16
        else:
            w_type = self.OPS_INT8

        if self.ops_type == self.OPS_INT32_TO_INT8:
            if is_result_layer:
                out_type = self.OPS_INT32_TO_FP
                # output_scale_shift = so
                # w_scale_shift = sk
                # final_scale_shift = si * sk
                # fd_out = d_out * final_scale_shift
            else:
                out_type = self.OPS_INT32_TO_INT8
                # output_scale_shift = so
                # w_scale_shift = sk
                # final_scale_shift = si * sk / so
        else:
            if is_result_layer:
                out_type = self.OPS_INT8_TO_FP
            else:
                out_type = self.OPS_INT8_TO_INT8_INT_SCALE

        if is_result_layer and self.RESULT_INT:
            out_type = self.OPS_INT8

        contents = "{"
        contents = _write(contents, type)  # type
        contents = _write(contents, ".u.fc={", tail="")
        contents = _write(contents, in_size)
        contents = _write(contents, out_size)
        contents = _write(contents, iter_size)
        contents = _write(contents, in_shift)

        if is_result_layer:
            contents = _write(contents, -w_shift)
        else:
            contents = _write(contents, w_shift)
        contents = _write(contents, w_offset)

        contents = _write(contents, hasBias)
        contents = _write(contents, act)

        if is_result_layer and self.RESULT_INT:
            contents = _write(contents, "0")
        else:
            if self.quan_method < self.QUAN_FLOAT:
                # contents = _write(contents, out_shift)
                print("NotImplementedError")
                os._exit(-1)
            else:
                # the scale from int32/int8 to int8/FP depending on the output type
                contents = _write(contents, float_to_hex(final_scale_shift))

        # for SDKV2
        # contents = _write(contents, symmetric)
        # contents = _write(contents, pad_value)  # the pad value
        contents = _write(contents, in_type)
        contents = _write(contents, w_type)
        contents = _write(contents, out_type)
        if self.AT5050:
            if self.RESULT_INT and is_result_layer:
                contents = _write(contents, self.QUAN_SHIFT)
                contents = _write(contents, "0")
            else:
                contents = _write(contents, self.QUAN_POST_SHIFT)
                contents = _write(contents, ".u.post_shift.scale=", tail="")
                if self.quan_method < self.QUAN_FLOAT:
                    # contents = _write(contents, out_shift)
                    print("NotImplementedError")
                    os._exit(-1)
                else:
                    # write the scale to i32 to work with the interpreter
                    if isinstance(final_scale_shift, float):
                        contents = _write(contents, float_to_hex(final_scale_shift))
                    else:
                        contents = _write(contents, final_scale_shift)

        contents = _write(contents, "}")
        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


@NETWORK_V1.register_module(name="maxpool")
@NETWORK_V1.register_module(name="averagepool")
@NETWORK_V1.register_module(name="globalaveragepool")
class LAYER_POOL(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_POOL, self).__init__(**kwargs)

    def save(self, layer):
        type = (
            layer.get_layer_type().upper()
        )  # self.layer_map_inv[layer.get_layer_type()]
        pad_t, pad_b, pad_l, pad_r = layer.get_layer_ops()["attrs"][0]["pads"]
        PH, PW = layer.get_layer_ops()["attrs"][0]["kernel_shape"]
        PSH, PSW = layer.get_layer_ops()["attrs"][0]["strides"]

        in_scale = layer.get_in_scale()[0]["scale"]
        scale = layer.get_scale()[0]["scale"]

        contents = "{"
        contents = _write(contents, type)
        contents = _write(contents, ".u.{}=".format(type.lower()), tail="")
        contents = _write(contents, "{" + str(pad_t))
        contents = _write(contents, pad_b)
        contents = _write(contents, pad_l)
        contents = _write(contents, pad_r)
        contents = _write(contents, PH)
        contents = _write(contents, PW)
        contents = _write(contents, PSH)
        contents = _write(contents, PSW)
        if self.quan_method < self.QUAN_FLOAT:
            # contents = _write(contents, in_shift)
            # contents = _write(contents, shift)
            print("NotImplementedError")
            os._exit(-1)
        else:
            # write the scale to i32 to work with the interpreter
            contents = _write(contents, float_to_hex(in_scale))
            contents = _write(contents, float_to_hex(scale))

        contents = _write(contents, "}")
        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


@NETWORK_V1.register_module(name="mul")
@NETWORK_V1.register_module(name="cmul")
@NETWORK_V1.register_module(name="pmul")
@NETWORK_V1.register_module(name="add")
@NETWORK_V1.register_module(name="sub")
class LAYER_EWS(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_EWS, self).__init__(**kwargs)

    def save(self, layer):
        type = (
            layer.get_layer_type().upper()
        )  # self.layer_map_inv[layer.get_layer_type()]
        H, W = layer.get_insert()["feat_i"][0]
        C = layer.get_insert()["in_align"][0]
        scales = copy.deepcopy(layer.get_scales())
        if isinstance(scales, list):
            scales = scales[0]
        in_scale1 = scales[0]["out_shift"]
        in_scale2 = scales[1]["out_shift"]
        out_scale1 = [scales[0]["out_scale"], scales[0]["int_scale"]]
        out_scale2 = [scales[1]["out_scale"], scales[1]["int_scale"]]
        scale = copy.deepcopy(layer.get_scales())
        if isinstance(scales, list):
            scale = scales[0]

        contents = "{"
        contents = _write(contents, type)
        contents = _write(contents, ".u.{}=".format(type.lower()), tail="")
        contents = _write(contents, "{", tail="")
        contents = _write(contents, H)
        contents = _write(contents, W)
        contents = _write(contents, C)

        if self.ADD_FLOAT:
            contents = _write(contents, self.QUAN_FLOAT)
            contents = _write(contents, ".pre.fp_sym={", tail="")
        else:
            contents = _write(contents, self.QUAN_PRE_CONCAT)
            contents = _write(contents, ".pre.pre_concat={", tail="")
        contents = _write(contents, "{", tail="")
        contents = _write(contents, self.OPS_INT8)

        if self.ADD_FLOAT:
            contents = _write(contents, self.OPS_FP32)
            contents = _write(contents, in_scale1, tail="")
        else:
            contents = _write(contents, self.OPS_INT8)
            contents = _write(contents, out_scale1[0])
            contents = _write(contents, out_scale1[1], tail="")
        contents = _write(contents, "}")
        contents = _write(contents, "{", tail="")
        contents = _write(contents, self.OPS_INT8)

        if self.ADD_FLOAT:
            contents = _write(contents, self.OPS_FP32)
            contents = _write(contents, in_scale2, tail="")
        else:
            contents = _write(contents, self.OPS_INT8)
            contents = _write(contents, out_scale2[0])
            contents = _write(contents, out_scale2[1], tail="")
        contents = _write(contents, "}")
        contents = _write(contents, "}")

        if self.ADD_FLOAT:
            contents = _write(contents, self.QUAN_FLOAT_SYM)
            contents = _write(contents, ".post.fp_sym=", tail="")
            contents = _write(contents, "{", tail="")
            contents = _write(contents, self.OPS_FP32)
            contents = _write(contents, self.OPS_INT8)
            contents = _write(contents, scale, tail="")
        else:
            contents = _write(contents, self.QUAN_NONE)
            contents = _write(contents, ".post.none=", tail="")
            contents = _write(contents, "{", tail="")
            contents = _write(contents, self.OPS_INT8)
            contents = _write(contents, self.OPS_INT8, tail="")
        contents = _write(contents, "}")

        contents = _write(contents, "}")
        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


@NETWORK_V1.register_module(name="concat")
class LAYER_CONCAT(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_CONCAT, self).__init__(**kwargs)

    def save(self, layer):
        type = self.layer_map_inv[layer.get_layer_type()]
        in_C = [b - a for a, b in layer.get_insert()["out_pad"]]
        out_shape = layer.get_insert()["feat_i"][0]

        in_scales = layer.get_in_scale()
        scale = layer.get_scale()[0]
        out_scales, out_shifts = [], []
        scales = copy.deepcopy(layer.get_scales())
        if isinstance(scales, list):
            scales = scales[0]
        for s in scales:
            out_scale, out_shift = s["out_scale"], s["int_scale"]
            out_scales.append(out_scale)
            out_shifts.append(out_shift)

        contents = "{"
        contents = _write(contents, type)
        contents = _write(contents, ".u.concat=", tail="")
        contents = _write(contents, "{", tail="")
        contents = _write(contents, len(out_scales))
        contents = _write(contents, out_shape[0])
        contents = _write(contents, out_shape[1])
        contents = _write(contents, "{", tail="")

        for idx in range(4):
            if idx < len(in_C):
                contents = _write(contents, in_C[idx])
            else:
                contents = _write(contents, 0)
        contents = _write(contents, "}")
        contents = _write(contents, "{", tail="")

        for idx in range(4):
            if idx < len(in_C):
                contents = _write(contents, in_C[idx])
            else:
                contents = _write(contents, 0)
        contents = _write(contents, "}")

        if self.CONCAT_FLOAT:
            contents = _write(contents, self.QUAN_FLOAT_SYM)
            contents = _write(contents, ".pre.fp_sym=", tail="")
        else:
            contents = _write(contents, self.QUAN_PRE_CONCAT)
            contents = _write(contents, ".pre.pre_concat={", tail="")

        for idx in range(4):
            contents = _write(contents, "{", tail="")
            if idx < len(in_C):
                contents = _write(contents, self.OPS_INT8)
                if self.CONCAT_FLOAT:
                    contents = _write(contents, self.OPS_FP32)
                    contents = _write(contents, in_scales[idx])
                else:
                    contents = _write(contents, self.OPS_INT8)
                    contents = _write(contents, out_scales[idx])
                    contents = _write(contents, out_shifts[idx])
            else:
                contents = _write(contents, 0)
            contents = _write(contents, "}")
        contents = _write(contents, "}")

        if self.CONCAT_FLOAT:
            contents = _write(contents, self.QUAN_FLOAT)
            contents = _write(contents, ".post.fp_sym=", tail="")
            contents = _write(contents, "{", tail="")
            contents = _write(contents, self.OPS_FP32)
            contents = _write(contents, self.OPS_INT8)
            contents = _write(contents, scale, tail="")
        else:
            contents = _write(contents, self.QUAN_NONE)
            contents = _write(contents, ".post.none=", tail="")
            contents = _write(contents, "{", tail="")
            contents = _write(contents, self.OPS_INT8)
            contents = _write(contents, self.OPS_INT8, tail="")
        contents = _write(contents, "}")

        contents = _write(contents, "}")
        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


@NETWORK_V1.register_module(name="shuffle_only")
class LAYER_SHUFFLE(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_SHUFFLE, self).__init__(**kwargs)

    def save(self, layer):
        type = layer.get_layer_type().upper()
        real_ic = [b - a for a, b in layer.get_insert()["in_pad"]]
        in_C = real_ic[0] + real_ic[1]
        in_scale = layer.get_in_scale()[0]
        out_scale = layer.get_scale()[0]

        contents = "{"
        contents = _write(contents, type)
        contents = _write(contents, "{", tail="")
        contents = _write(contents, self.quan_method)
        contents = _write(contents, in_C)
        if self.quan_method < self.QUAN_FLOAT:
            # contents = _write(contents, in_shift[0])
            # contents = _write(contents, in_shift[1])
            # contents = _write(contents, out_shift[0])
            # contents = _write(contents, out_shift[1])
            print("NotImplementedError")
            os._exit(-1)
        else:
            contents = _write(contents, float_to_hex(1 / in_scale))
            contents = _write(contents, float_to_hex(1 / out_scale), "},")

        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


@NETWORK_V1.register_module(name="split")
class LAYER_SPLIT(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_SPLIT, self).__init__(**kwargs)

    def save(self, layer):
        # in_len, out_len = self.get_io_len(layer)

        # LayerType = self.layer_map_inv[layer.get_layer_type()]
        # LayerInfo = self.LayerInstance[LayerType]
        # quant = self.LayerQuant[layer.get_scale_type()]
        # qi_type = self.NPU_DataType[layer.get_input_type()]
        # quant_u = self.LayerPrePost['']
        # LayerPre = '{' + str(out_len) + ","
        # LayerPre += '{}{},{},.quant_u.{}={}{}'.format('{', quant, qi_type,
        #                                              quant_u, 0, '}')

        # i_type = layer.get_ops_setting()['setting']['bits_dict'][layer.get_input_type()].__name__ #self.NPU_DataType[layer.get_input_type()]
        # o_type = layer.get_ops_setting()['setting']['bits_dict'][layer.get_output_type()].__name__ #self.NPU_DataType[layer.get_output_type()]
        # i_type = 'NPU_' + i_type.replace('u', 'U').replace('int', 'INT')
        # o_type = 'NPU_' + o_type.replace('u', 'U').replace('int', 'INT')
        # i_fmt = self.CubeFmt[self.fmt]
        # o_fmt = self.CubeFmt[self.fmt]
        # H, W = layer.get_insert()['feat_i'][0]

        # C = layer.get_insert()['in_align'][0]
        # real_c = np.sum([b - a for a, b in layer.get_insert()['in_pad']])

        # OC = layer.get_insert()['out_align']
        # while len(OC) < self.SPLIT_MAX_OUT:
        #     OC.append(0)

        # real_oc = [b - a for a, b in layer.get_insert()['out_pad']]
        # while len(real_oc) < self.SPLIT_MAX_OUT:
        #     real_oc.append(0)

        # Layer = '{},{},{},{},{},{},{},{},{},{}'.format(
        #     i_type, o_type, i_fmt, o_fmt, H, W, C, real_c,
        #     self.list2Cstyle(OC), self.list2Cstyle(real_oc))

        # LayerPost = '{'
        # for i in range(self.SPLIT_MAX_OUT):
        #     if i < out_len:
        #         quant = self.LayerQuant[layer.get_scale_type()]
        #         qo_type = self.NPU_DataType[layer.get_input_type()]
        #         quant_u = self.LayerPrePost['']
        #         LayerPost += '{}{},{},.quant_u.{}={}{}'.format('{', quant, qo_type,
        #                                              quant_u, 0, '},')
        #     else:
        #         quant = self.LayerQuant['']
        #         qo_type = self.NPU_DataType['']
        #         quant_u = self.LayerPrePost['']
        #         LayerPost += '{}{},{},.quant_u.{}={}{}'.format('{', quant, qo_type,
        #                                              quant_u, 0, '},')
        # LayerPost += '},'
        # LayerPost += '},'

        # contents = self.get_contents(layer, contents)

        # contents = contents.replace(',}', '}')

        print("NotImplementedError")
        os._exit(-1)


@NETWORK_V1.register_module(name="shuffle")
class LAYER_CONCAT_SHUFFLE_SPLIT(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_CONCAT_SHUFFLE_SPLIT, self).__init__(**kwargs)

    def save(self, layer):
        type = layer.get_layer_type().upper()
        real_ic = [b - a for a, b in layer.get_insert()["concat"]["out_pad"]]
        in_C_0 = real_ic[0]
        in_C_1 = real_ic[1]
        in_scale = layer.get_in_scale()
        out_scale = layer.get_scale()[1:]

        contents = "{"
        contents = _write(contents, type)
        contents = _write(contents, "{", tail="")
        contents = _write(contents, self.quan_method)
        contents = _write(contents, in_C_0)
        contents = _write(contents, in_C_1)
        if self.quan_method < self.QUAN_FLOAT:
            # contents = _write(contents, in_shift[0])
            # contents = _write(contents, in_shift[1])
            # contents = _write(contents, out_shift[0])
            # contents = _write(contents, out_shift[1], '}')
            print("NotImplementedError")
            os._exit(-1)
        else:
            contents = _write(contents, float_to_hex(1 / in_scale[0]))
            contents = _write(contents, float_to_hex(1 / in_scale[1]))
            contents = _write(contents, float_to_hex(1 / out_scale[0]))
            contents = _write(contents, float_to_hex(1 / out_scale[1]), "},")

        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents


@NETWORK_V1.register_module(name="resize")
class LAYER_RESIZE(NetworkV1):
    def __init__(self, **kwargs):
        super(LAYER_RESIZE, self).__init__(**kwargs)

    def save(self, layer):
        type = self.layer_map_inv[layer.get_layer_type()]
        channels = layer.get_insert()["in_align"][0]
        in_shape = layer.get_insert()["feat_i"][0]
        out_shape = layer.get_insert()["feat_o"][0]
        mode = layer.get_ops_setting()["attrs"][0]["mode"]
        half_pixel_centers = 0
        align_corners = 0

        contents = "{"
        contents = _write(contents, type)
        contents = _write(contents, ".u.{}=".format(type.lower()), tail="")
        contents = _write(contents, "{", tail="")
        contents = _write(contents, self.OPS_INT8)
        contents = _write(contents, in_shape[0])
        contents = _write(contents, in_shape[1])
        contents = _write(contents, channels)
        contents = _write(contents, out_shape[0])
        contents = _write(contents, out_shape[1])

        if mode == "nearest":
            contents = _write(contents, self.INTERP_NEAREST)
            contents = _write(contents, ".u.nearest_param={", tail="")
        else:
            contents = _write(contents, self.INTERP_BILINEAR)
            contents = _write(contents, ".u.bilinear_param={", tail="")
        contents = _write(contents, int(half_pixel_centers))
        contents = _write(contents, int(align_corners))
        contents = _write(contents, "}")

        contents = _write(contents, self.QUAN_NONE)
        contents = _write(contents, ".pre.none={", tail="")
        contents = _write(contents, self.OPS_INT8)
        contents = _write(contents, self.OPS_INT8)
        contents = _write(contents, "}")
        contents = _write(contents, self.QUAN_NONE)
        contents = _write(contents, ".post.none={", tail="")
        contents = _write(contents, self.OPS_INT8)
        contents = _write(contents, self.OPS_INT8)
        contents = _write(contents, "}")

        contents = _write(contents, "}")
        contents = self.get_contents(layer, contents)
        contents += "},"

        return contents
