# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/07/05 15:01:16
# @File    : export_layer.py
import sys

sys.path.append("./")  # NOQA: E402

import copy
import os

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnxruntime as rt

try:
    from config import Config
    from export import mExportV3, mExportV2, mExportV1, writeFile
    from graph.layer import LAYER as layer_factory
    from quantizer import QUANTIZE as quantize_factory
    from simulator import OPERATORS as operators_factory
    from simulator import error_factory
    from utils import Object, Registry, invert_dict, get_same_padding, to_bytes
    from extension import pyops # type: ignore
except Exception:
    from onnx_converter.config import Config # type: ignore
    from onnx_converter.export import mExportV3, mExportV2, mExportV1, writeFile # type: ignore
    from onnx_converter.graph.layer import LAYER as layer_factory # type: ignore
    from onnx_converter.quantizer import QUANTIZE as quantize_factory # type: ignore
    from onnx_converter.simulator import OPERATORS as operators_factory # type: ignore
    from onnx_converter.simulator import error_factory # type: ignore
    from onnx_converter.utils import Object, Registry, invert_dict, get_same_padding, to_bytes # type: ignore
    from onnx_converter.extension import pyops # type: ignore


EXPORT_LAYER: Registry = Registry("export_layer", scope="")

global w_offset
w_offset = dict(w_offset=0, tmp_offset=[])


class ExportLayer(Object): # type: ignore
    def __init__(self, **kwargs):
        super(ExportLayer, self).__init__(**kwargs)

        self.is_stdout = kwargs["is_stdout"]
        self.is_single_layer = kwargs.get("is_single_layer", True)
        self.log_name = kwargs["log_name"]
        self.log_level = kwargs.get('log_level', 20)
        self.log_dir = kwargs["log_dir"]
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)
        self.attr_case_id = kwargs['attr_case_id'] if "attr_case_id" in kwargs.keys() else 0
        self.case_attr = kwargs['case_attr'] if 'case_attr' in kwargs.keys() else dict()
        self.case_name = kwargs["case_name"]
        self.weights_dir = os.path.join(
            kwargs["weights_dir"], self.case_name, "weights"
        )
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir, mode=0o777)
        self.arguments = kwargs["arguments"]
        self.is_voice_model = self.arguments["layer_type"] in [
            "layernormalization",
            "lstm",
            "gru",
        ]
        self.is_acc_woffset = self.arguments["export_args"]["is_acc_woffset"]
        self.mode = self.arguments["export_args"]["mode"]
        self.chip_type = self.arguments["export_args"]["chip_type"]
        self.export_version = self.arguments["export_args"]["export_version"]
        self.metrics_list = ["L1", "L2", "Cosine"]
        self.quantization_args = self.arguments["quantization_args"]
        self.init_quant_cfg(
            quant_cfg=kwargs["quant_cfg"],
            voice_quant_cfg=kwargs["voice_quant_cfg"],
            vision_quant_cfg=kwargs["vision_quant_cfg"],
        )
        self.init_export(export_cfg=kwargs["export_cfg"])
        self.check_error = {m: error_factory.get(m)() for m in self.metrics_list} # type: ignore

    def init_quant_cfg(self, quant_cfg, voice_quant_cfg, vision_quant_cfg):
        if not isinstance(quant_cfg, dict):
            save_cfg = Config.fromfile(quant_cfg)
            cfg_dict, _ = save_cfg._file2dict(quant_cfg)
        else:
            cfg_dict = quant_cfg

        quant_cfg_ = voice_quant_cfg if self.is_voice_model else vision_quant_cfg

        if not isinstance(quant_cfg_, dict):
            save_cfg_ = Config.fromfile(quant_cfg_)
            cfg_dict_, _ = save_cfg_._file2dict(quant_cfg_)
        else:
            cfg_dict_ = quant_cfg_
        cfg_dict = Config._merge_a_into_b(cfg_dict, cfg_dict_)

        self.quan_dict: dict = copy.deepcopy(cfg_dict)  # type: ignore
        self.update_cfg(self.quantization_args)
        self.setting = dict(setting=self.quan_dict)

    def init_export(self, export_cfg):
        export_fun = "mExportV{}".format(self.export_version)
        if not isinstance(export_cfg, dict):
            cfg_dict = {}
            export_cfg_files = [
                os.path.join(os.path.split(export_cfg)[0], "export.py"), 
                export_cfg,
            ]
            for export_cfg in export_cfg_files:
                save_cfg = Config.fromfile(export_cfg)
                cfg_dict_, _ = save_cfg._file2dict(export_cfg)
                cfg_dict_.update(dict(is_stdout=self.is_stdout)) # type: ignore
                cfg_dict.update(cfg_dict_)
        else:
            cfg_dict = export_cfg

        cfg_dict["chip_type"] = self.chip_type
        if self.chip_type in ["AT1K"]:
            cfg_dict["bits"]["DATA_C_EXTEND"] = cfg_dict["AT5050_C_EXTEND"] = False
            cfg_dict["bits"]["Csize"] = cfg_dict["Csize"] = 8
            cfg_dict["bits"]["Ksize"] = cfg_dict["Ksize"] = 8
        elif self.chip_type in ["AT5050"]:
            cfg_dict["bits"]["DATA_C_EXTEND"] = cfg_dict["AT5050_C_EXTEND"] = True
            cfg_dict["bits"]["Csize"] = cfg_dict["Csize"] = 16
            cfg_dict["bits"]["Ksize"] = cfg_dict["Ksize"] = 32
        else:
            raise Exception("Not supported chip type: {}".format(self.chip_type))

        cfg_dict.update(
            dict(log_name=self.log_name, log_dir=self.log_dir, is_stdout=self.is_stdout)
        )
        self.model_export = eval(export_fun)(**cfg_dict)

    def update_cfg(self, quan_args):
        # Update the user defined quantization configs
        # Make sure input configs exist and valid
        out_type_dict = invert_dict(copy.deepcopy(self.quan_dict["bits_dict"])) # type: ignore
        out_type_dict = {k.split("np.")[1]: v for k, v in out_type_dict.items()}
        assert {"out_type", "method", "process_scale"} <= set(
            quan_args.keys()
        ), "out_type, method and process_scale should be specified, please check"
        assert (
            quan_args["out_type"].lower() in out_type_dict.keys()
        ), "You specified invalid out_type, which should be one of [int8, int16, int32, int64, float32, float64]"
        assert quan_args["process_scale"] in [
            "intscale",
            "floatscale",
            "shiftfloatscale",
            "shiftfloatscaletable",
            "shiftfloatscaletable2float",
            "ffloatscale",
            "preintscale",
            "preintscaleex",
            "smooth",
            "table",
            "float",
        ], "You specified invalid process_scale, please check the data type dict"

        out_type_str = quan_args["out_type"]
        out_type_id = out_type_dict[out_type_str.lower()]
        self.quan_dict["out_type"] = out_type_id # type: ignore

        # added by henson
        in_type_str = quan_args["in_type"]
        in_type_id = out_type_dict[in_type_str.lower()]
        self.quan_dict["in_type"] = in_type_id # type: ignore

        self.quan_dict["txme_saturation"] = 1  # default open # type: ignore
        if quan_args["process_scale"] == "floatscale": # type: ignore
            self.quan_dict["txme_saturation"] = 0 # type: ignore

        self.quan_dict["process_scale"] = quan_args["process_scale"] # type: ignore
        if self.quan_dict["process_scale"] == "floatscale": # type: ignore
            self.quan_dict["precision"] = 1 # type: ignore
        else:
            self.quan_dict["precision"] = 0 # type: ignore

        bitwidth = int(quan_args["type"].split("int")[-1])
        self.quan_dict["int_scale"] = bitwidth
        self.quan_dict["pre_int_scale"] = bitwidth - 1  # right??
        self.quan_dict["bit_select"] = out_type_dict[quan_args["type"].lower()]
        self.quan_dict["virtual_round"] = quan_args["virtual_round"]

        feat_method = quan_args["method"]["feature"]
        if feat_method == "sym":
            self.quan_dict["feat"]["method"] = "floatsymquan" # type: ignore
        else:
            self.quan_dict["feat"]["method"] = "floatquan" # type: ignore
            self.quan_dict["txme_saturation"] = 0 # type: ignore
        self.quan_dict["feat"]["bit_select"] = self.quan_dict["bit_select"] # type: ignore

        weight_method = quan_args["method"]["weight"]
        if "sym" == weight_method[0]:
            self.quan_dict["normal"]["method"] = "floatsymquan" # type: ignore
            self.quan_dict["normal"]["bit_select"] = self.quan_dict["bit_select"] # type: ignore
            self.quan_dict["perchannel"]["method"] = "perchannelfloatsymquan" # type: ignore
            self.quan_dict["perchannel"]["bit_select"] = self.quan_dict["bit_select"] # type: ignore
            self.quan_dict["int16"]["method"] = "floatsymquan" # type: ignore
        else:
            self.quan_dict["normal"]["method"] = "floatquan" # type: ignore
            self.quan_dict["normal"]["bit_select"] = self.quan_dict["bit_select"] # type: ignore
            self.quan_dict["perchannel"]["method"] = "perchannelfloatquan" # type: ignore
            self.quan_dict["perchannel"]["bit_select"] = self.quan_dict["bit_select"] # type: ignore
            self.quan_dict["int16"]["method"] = "floatquan" # type: ignore
            self.quan_dict["txme_saturation"] = 0 # type: ignore

        for key in self.quan_dict["default_setting"].keys(): # type: ignore
            layer_info = self.quan_dict["default_setting"][key] # type: ignore
            layer_info["precision"] = self.quan_dict["precision"] # type: ignore
            layer_info["virtual_round"] = self.quan_dict["virtual_round"] # type: ignore
            if key in [
                "shuffle",
                "shuffle_only",
                "add",
                "concat",
                "split",
                "slice",
            ]:
                layer_info["int_scale"] = self.quan_dict["pre_int_scale"]
            else:
                layer_info["int_scale"] = self.quan_dict["int_scale"]
            layer_info["out_type"] = self.quan_dict["out_type"]
            layer_info["in_type"] = self.quan_dict["in_type"]
            layer_info["feat"] = self.quan_dict["feat"]

            if key in ["conv", "depthwiseconv", "fc", "gemm", "matmul", "lstm", "gru", "splice"]:
                if quan_args["method"]["weight"][1] == "perchannel":
                    layer_info["weights"] = self.quan_dict["perchannel"]
                else:
                    layer_info["weights"] = self.quan_dict["normal"]
            layer_info["process_scale"] = quan_args["process_scale"]

        # contents = json.dumps(self.quan_dict, sort_keys=False, indent=4)
        # with open('quan_dict.json', 'w') as f:
        #     f.write(contents)
        # print('test')

    def parser_weight(self, model):
        weights_data = dict()
        for weight in model.graph.initializer:
            data = np.frombuffer(weight.raw_data, dtype=np.float32)
            weights_data[weight.name] = np.reshape(data, weight.dims)
        return weights_data

    @staticmethod
    def to_torch(data):
        if isinstance(data, list):
            data = np.array(data)
        return torch.from_numpy(data)

    @staticmethod
    def to_numpy(data):
        return data.detach().numpy()

    def export(self, attrs):
        self.model_export.set_bias_data(self.layer)
        self.model_export.set_layer_datatype(self.layer)
        self.model_export.set_voice_model_feats(
            self.layer, is_voice_model=self.is_voice_model
        )

        if self.layer_type == "concat":
            in_cs = []
            for in_data_ in self.layer.get_in_data():
                in_c = in_data_["output"].shape[1]
                in_cs.append(in_c)
            out_c = self.layer.get_out_data()["output"].shape[1]
            feat_i = feat_o = self.layer.get_out_data()["output"].shape[2:]
        elif self.layer_type == "split":
            out_cs = []
            for out_data_ in self.layer.get_out_data():
                out_c = out_data_["output"].shape[1]
                out_cs.append(out_c)
            in_c = self.layer.get_in_data()[0]["output"].shape[1]
            feat_i = feat_o = self.layer.get_out_data()[0]["output"].shape[2:]
        elif self.layer_type in ["shuffle", "concat_shuffle_split"]:
            in_cs = []
            for in_data_ in self.layer.get_in_data():
                in_c = in_data_["output"].shape[1]
                in_cs.append(in_c)
            feat_i = feat_o = self.layer.get_in_data()[0]["output"].shape[2:]
        elif self.layer_type in ["shuffle_only"]:
            in_cs = attrs["in_channels"]
            out_cs = attrs["out_channels"]
            feat_i = feat_o = self.layer.get_in_data()[0]["output"].shape[2:]
        else:
            in_data = self.layer.get_in_data()
            if isinstance(in_data, list):
                in_data = in_data[0]
            out_data = self.layer.get_out_data()
            if isinstance(out_data, list):
                out_data = out_data[0]                
            in_c = in_data["output"].shape[1]
            if len(in_data["output"].shape) == 2:
                feat_i = [1, 1]
            else:
                feat_i = in_data["output"].shape[2:]
            out_c = out_data["output"].shape[1]
            if len(out_data["output"].shape) == 2:
                feat_o = [1, 1]
            else:
                feat_o = out_data["output"].shape[2:]

        if self.layer_type in ["lstm"]:
            in_c = self.layer.get_ops_setting()["attrs"][0]["in_c"]
            hidden_size = self.layer.get_ops_setting()["attrs"][0]["hidden_size"]
            in_pad, in_align = [], []
            for ch in [in_c, in_c, hidden_size]:
                in_pad.append([0, ch])
                in_align.append(
                    self.model_export.get_align_channel(ch, self.model_export.I_Align)
                )

            out_c1, out_c2 = (
                self.layer.get_qweight()[0].shape[1],
                self.layer.get_qweight()[1].shape[1],
            )
            result = [hidden_size, out_c1, out_c2]
            out_align = [
                self.model_export.get_align_channel(ch, self.model_export.O_Align)
                for ch in result
            ]
            out_pad = [[0, ch] for ch in result]

            insert = {
                "in_pad": in_pad,
                "in_align": in_align,
                "out_pad": out_pad,
                "out_align": out_align,
            }
            insert = dict(feat_i=[feat_i], feat_o=[feat_o], split=insert)
        elif self.layer_type in ["gru"]:
            in_c = self.layer.get_ops_setting()["attrs"][0]["in_c"]
            hidden_size = self.layer.get_ops_setting()["attrs"][0]["hidden_size"]
            in_pad, in_align = [], []
            for ch in [in_c, hidden_size]:
                in_pad.append([0, ch])
                in_align.append(
                    self.model_export.get_align_channel(ch, self.model_export.I_Align)
                )

            out_c = self.layer.get_qweight()[0].shape[1]
            result = [hidden_size, out_c]
            out_align = [
                self.model_export.get_align_channel(ch, self.model_export.O_Align)
                for ch in result
            ]
            out_pad = [[0, ch] for ch in result]

            insert = {
                "in_pad": in_pad,
                "in_align": in_align,
                "out_pad": out_pad,
                "out_align": out_align,
            }
            insert = dict(feat_i=[feat_i], feat_o=[feat_o], split=insert)
        elif self.layer_type in ["mul", "cmul", "pmul", "add", "sub", "cadd", "csub", "matmul"]:
            if self.layer_type in ["matmul"]:
                feat_i = [in_data["output"].shape[2:] for in_data in self.layer.get_in_data()]  
            else:
                feat_i = [feat_i]              
            in_cs = [in_data["output"].shape[1] for in_data in self.layer.get_in_data()]
            out_cs = [self.layer.get_out_data()["output"].shape[1]]
            in_pad, in_align = [], []
            for ch in in_cs:
                in_pad.append([0, ch])
                in_align.append(
                    self.model_export.get_align_channel(ch, self.model_export.I_Align)
                )
            out_align = [
                self.model_export.get_align_channel(ch, self.model_export.O_Align)
                for ch in out_cs
            ]
            out_pad = [[0, ch] for ch in out_cs]

            insert = {
                "in_pad": in_pad,
                "in_align": in_align,
                "out_pad": out_pad,
                "out_align": out_align,
                "feat_i": feat_i,
                "feat_o": [feat_o],
            }
        elif self.layer_type == "concat":
            in_pad, in_align = [], []
            for ch in in_cs: # type: ignore
                in_pad.append([0, ch])
                in_align.append(
                    self.model_export.get_align_channel(ch, self.model_export.Csize)
                )
            out_align = [np.sum(in_align)]
            out_pad = []
            start_pad = 0
            for i in range(len(in_pad)):
                end_pad = start_pad + (in_pad[i][1] - in_pad[i][0])
                out_pad.append([start_pad, end_pad])
                if i < len(in_align):
                    start_pad += in_align[i]         
            insert = {
                "in_pad": in_pad,
                "in_align": in_align,
                "out_pad": out_pad,
                "out_align": out_align,
                "feat_i": [feat_i],
                "feat_o": [feat_o],
            }
        elif self.layer_type == "split":
            out_pad, out_align = [], []
            for ch in out_cs: # type: ignore
                out_pad.append([0, ch])
                out_align.append(
                    self.model_export.get_align_channel(ch, self.model_export.Csize)
                )
            in_align = [np.sum(out_align)]
            in_pad = [
                list(np.array(pad) + align)
                for pad, align in zip(out_pad[1:], out_align[:-1])
            ]
            in_pad.insert(0, out_pad[0])
            insert = {
                "in_pad": in_pad,
                "in_align": in_align,
                "out_pad": out_pad,
                "out_align": out_align,
                "feat_i": [feat_i],
                "feat_o": [feat_o],
            }
            insert = dict(
                # feat_i=[feat_i], feat_o=[feat_o],
                split=insert,
                split_ids={-1: 0, -2: 1},
            )
        elif self.layer_type in ["shuffle", "concat_shuffle_split"]:
            in_pad, in_align = [], []
            for ch in in_cs: # type: ignore
                in_pad.append([0, ch])
                in_align.append(
                    self.model_export.get_align_channel(ch, self.model_export.Csize)
                )
            out_pad, out_align = in_pad, in_align
            insert = {
                "in_pad": in_pad,
                "in_align": in_align,
                "out_pad": out_pad,
                "out_align": out_align,
            }
            insert = dict(
                feat_i=[feat_i],
                feat_o=[feat_o],
                split=insert,
                split_ids={-1: 0, -2: 1},
                is_align=False,
            )
        elif self.layer_type in ["shuffle_only"]: 
            if len(in_cs) > 1: # concat      
                in_pad, in_align = [], []
                for ch in in_cs: # type: ignore
                    in_pad.append([0, ch])
                    in_align.append(
                        self.model_export.get_align_channel(ch, self.model_export.Csize)
                    )
                out_align = in_align #[np.sum(in_align)]
                out_pad = []
                start_pad = 0
                for i in range(len(in_pad)):
                    end_pad = start_pad + (in_pad[i][1] - in_pad[i][0])
                    out_pad.append([start_pad, end_pad])
                    if i < len(in_align):
                        start_pad += in_align[i]         
                insert = {
                    "in_pad": out_pad,
                    "in_align": out_align,
                    "feat_i": [feat_i],
                    "feat_o": [feat_o],
                } 
            else:
                in_align = self.model_export.get_align_channel(in_cs[0], self.align_size[0]) # type: ignore
                insert = {
                    "in_pad": [[0, in_cs[0]]], # type: ignore
                    "in_align": [in_align],
                    "feat_i": [feat_i],
                    "feat_o": [feat_o],                    
                }
                
            if len(out_cs) > 1: ### split
                out_pad, out_align = [], []
                for ch in out_cs: # type: ignore
                    out_pad.append([0, ch])
                    out_align.append(
                        self.model_export.get_align_channel(ch, self.model_export.Csize)
                    )
                in_align = [np.sum(out_align)]
                in_pad = [
                    list(np.array(pad) + align)
                    for pad, align in zip(out_pad[1:], out_align[:-1])
                ]
                in_pad.insert(0, out_pad[0])
                out_pad = [(np.array(a) + b).tolist() for a, b in zip(out_pad, [0, out_align[:-1]])]
                insert.update({
                    "out_pad": out_pad,
                    "out_align": out_align,
                })
            else:
                out_align = self.model_export.get_align_channel(out_cs[0], self.align_size[1]) # type: ignore
                insert.update({
                    "out_pad": [[0, out_cs[0]]], # type: ignore
                    "out_align": [out_align],
                })
        else:
            in_align = self.model_export.get_align_channel(in_c, self.align_size[0]) # type: ignore
            if self.layer_type in ["depthwiseconv", "maxpool"]:
                out_align = in_align
            else:
                out_align = self.model_export.get_align_channel(out_c, self.align_size[1]) # type: ignore
            insert = {
                "in_pad": [[0, in_c]], # type: ignore
                "in_align": [in_align],
                "out_pad": [[0, out_c]], # type: ignore
                "out_align": [out_align],
                "feat_i": [feat_i],
                "feat_o": [feat_o],
            }

        if self.is_voice_model:
            insert.update(dict(fmt="MatFmt")) # type: ignore
        else:
            insert.update(dict(fmt="CubeFmt")) # type: ignore
        self.layer.set_insert(insert)

        self.logger.info(
            "-------------------------------------------------------------------------"
        )
        self.logger.info(
            "layer index is: {}/{}, input index is:{}, output index is: {}".format(
                self.layer.get_idx(),
                1,
                self.layer.get_input_idx(),
                self.layer.get_output_idx(),
            )
        )
        self.logger.info(
            "layer name is: {}, layer type is: {}, export parameter is: ".format(
                self.layer.get_layer_name(), self.layer.get_layer_type()
            )
        )
        for k, v in sorted(
            self.layer.get_insert().items(), key=lambda d: d[0], reverse=True
        ):
            if k in ["split", "concat"]:
                self.logger.info("{}: ".format(k))
                for k_, v_ in v.items():
                    self.logger.info("    {}: {}".format(k_, v_))
            else:
                self.logger.info("{}: {}".format(k, v))
        self.logger.info(
            "-------------------------------------------------------------------------"
        )

        binary_weight = bytearray()
        if self.export_w:
            if self.is_acc_woffset:
                self.model_export.w_offset["w_offset"] = w_offset["w_offset"]
            self.export_w(
                self.layer, self.model_export.save_weights, self.model_export.w_offset
            )
            if not (self.export_version >= 3):
                writeFile(
                    self.model_export.save_weights,
                    os.path.join(self.weights_dir, "weight.b"),
                    mode=self.mode,
                )
            else:
                if len(self.model_export.save_weights) > 0:
                    dtype = self.model_export.save_weights[0].dtype
                    binary_weight = to_bytes(self.model_export.save_weights, dtype=dtype)

        ### write output feature
        self.model_export.export_features(self.layer)
        for key, feats in self.model_export.save_feats.items():
            key = key.replace("/", "-").replace(":", "-")
            writeFile(feats, os.path.join(self.weights_dir, key), mode=self.mode)

        # ### write input feature
        func_data = getattr(self.model_export, "serialize_{}".format("data"))
        for idx, qdata in enumerate(self.layer.get_in_data()):
            layer_name = "indata_{}".format(idx)
            if self.layer.get_layer_type() in ["shuffle_only"]:
                insert = {
                    key: self.layer.get_insert()[key]
                    for key in ["in_pad", "in_align"]
                }
            else:
                if "split" in self.layer.get_insert().keys():
                    insert = {
                        key: [self.layer.get_insert()["split"][key][idx]]
                        for key in ["in_pad", "in_align"]
                    }
                else:
                    insert = {
                        key: [self.layer.get_insert()[key][idx]]
                        for key in ["in_pad", "in_align"]
                    }
            res = self.layer.feat_export(
                func_data,
                qdata["output"],
                insert=insert,
                layer=self.layer,
                layer_name=layer_name,
                is_out=False,
                name=os.path.join(self.weights_dir, layer_name),
            )
            writeFile(
                res, os.path.join(self.weights_dir, layer_name + ".b"), mode=self.mode
            )

        content = getattr(self.model_export, "network_{}".format(self.layer_type)).save(
            self.layer,
        )

        if self.is_acc_woffset:
            w_offset["w_offset"] = self.model_export.w_offset["w_offset"]

        # with open(os.path.join(self.weights_dir, "model.c"), "w") as file:
        #     file.write('#include "npu_layer.h"\n\n')
        #     file.write("LayerInfo_t layers[]={\n")
        #     file.write(content + '\n')
        #     file.write("};\n\n")
        #     file.write(
        #         "u32 weight_total_size={};\n".format(
        #             self.model_export.w_offset["w_offset"]
        #         )
        #     )
        #     content = 'char version[] = "{}";'.format(
        #         self.model_export.get_version())
        #     file.write(content + '\n')
        if not self.is_single_layer:
            self.remove()
            
        if not (self.export_version >= 3):
            return content
        else:
            return [content, binary_weight, self.layer]

    def get_float_result(self):
        pass

    def run_export_layer(self):
        pass

    def calc_error(self, fresults, qresults):
        self.logger.info(
            "-------------------------------------------------------------------------"
        )
        self.logger.info("processing: {}".format(self.case_name))
        if self.case_attr:
            self.logger.info("combination{}: {}".format(self.attr_case_id, self.case_attr))
        for idx, (qout, fout) in enumerate(zip(qresults, fresults)):
            for metric in self.check_error.keys():
                error = self.check_error[metric](qout, fout)
                self.logger.info("{} error: {}".format(metric, error))
                if metric == "Cosine":
                    flag = True if error < 5.0 else False
                    if not flag:
                        self.logger.info("maybe low precision, need to check!!!")
                    pytest.assume(flag) # type: ignore
        # si = self.layer.get_in_scale()
        # if isinstance(si, list):
        #     si = si[0]

        # sk = self.layer.get_w_scale()
        # if isinstance(sk, list):
        #     sk = sk[0]

        # so = self.layer.get_scale()
        # if isinstance(so, list):
        #     so = so[0]

        # self.logger.info('si scale/zero_point: {}/{}'.format(
        #     si['scale'], si['zero_point']))
        # self.logger.info('sk scale/zero_point: {}/{}'.format(
        #     sk['scale'], sk['zero_point']))
        # self.logger.info('so scale/zero_point: {}/{}'.format(
        #     so['scale'], so['zero_point']))

        self.logger.info(
            "-------------------------------------------------------------------------"
        )

    def __call__(self, attrs, xs, ws):
        self.run_export_layer(attrs, xs, ws) # type: ignore
        return self.export(attrs)


@EXPORT_LAYER.register_module(name="reducemax")
@EXPORT_LAYER.register_module(name="reducemin")
@EXPORT_LAYER.register_module(name="reducemean")
@EXPORT_LAYER.register_module(name="reducesum")
@EXPORT_LAYER.register_module(name="reduceprod")
class ExportReduceOps(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportReduceOps, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        dims = attrs["axes"]
        keepdims = True if attrs["keepdims"] else False
        
        ys = self.to_torch(X)
        for idx, dim in enumerate(dims):
            if self.layer_type in ["reducemax"]:
                ys = torch.max(ys, dim=dim, keepdim=True)[0]
            elif self.layer_type in ["reducemin"]:
                ys = torch.min(ys, dim=dim, keepdim=True)[0]                
            elif self.layer_type in ["reducesum"]:
                ys = torch.sum(ys, dim=dim, keepdim=True)
            elif self.layer_type in ["reducemean"]:
                ys = torch.mean(ys, dim=dim, keepdim=True)
            elif self.layer_type in ["reduceprod"]:
                ys = torch.prod(ys, dim=dim, keepdim=True)  
        if not keepdims:
            for idx, dim in enumerate(dims):
                dim -= idx
                ys = torch.squeeze(ys, dim)                                    
        ys = self.to_numpy(ys)
        ys = [ys]
        
        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            attrs_cpy = copy.deepcopy(attrs)
            attrs_cpy.update(dict())

            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs_cpy], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs_cpy]))

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0])) # type: ignore
                qxs = copy.deepcopy(dict(output=in_data_q))
            in_data = [qxs]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]
            self.calc_error(fresults, qresults)
            
            
@EXPORT_LAYER.register_module(name="transpose")
class ExportTranspose(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportTranspose, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        perm = attrs["perm"]
        
        ys = self.to_torch(X)
        ys = ys.permute(perm)
        ys = self.to_numpy(ys)
        ys = [ys]
        
        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            attrs_cpy = copy.deepcopy(attrs)
            attrs_cpy.update(dict())

            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs_cpy], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs_cpy]))

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0])) # type: ignore
                qxs = copy.deepcopy(dict(output=in_data_q))
            in_data = [qxs]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]
            self.calc_error(fresults, qresults)
            
            
@EXPORT_LAYER.register_module(name="reshape")
class ExportReshape(ExportTranspose):
    def __init__(self, **kwargs):
        super(ExportReshape, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        
        shape = np.array(attrs["shape"])
        for i in range(len(shape)):
            if shape[i] == 0:
                shape[i] = 1
                
        ys = np.reshape(X, shape)
        ys = [ys]
        
        return ys
    

@EXPORT_LAYER.register_module(name="pad")
class ExportPad(ExportTranspose):
    def __init__(self, **kwargs):
        super(ExportPad, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        
        pads = attrs["pads"]
        mode = attrs["mode"]
        if mode == "constant":
            ys = F.pad(self.to_torch(X), pad=pads, mode="constant", value=0)        
        # elif mode == "reflect":
        #     ys = F.pad(self.to_torch(X), pad=pads, mode="reflect")
        # elif mode == "edge":
        #     ys = F.pad(self.to_torch(X), pad=pads, mode="edge")
        # elif mode == "warp":
        #     ys = F.pad(self.to_torch(X), pad=pads, mode="warp")
        else:
            raise Exception("Not supported!!!")
        
        ys = self.to_numpy(ys)
        ys = [ys]
        
        return ys
    
    
@EXPORT_LAYER.register_module(name="log")
class ExportLog(ExportTranspose):
    def __init__(self, **kwargs):
        super(ExportLog, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
                
        ys = np.log2(X)
        ys = [ys]
        
        return ys
    
                            
@EXPORT_LAYER.register_module(name="batchnormalization")
class ExportBatchnormalization(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportBatchnormalization, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = self.model_export.wExport_batchnormalization

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        W = ws["weight"][0]
        B = ws["bias"][0]
        mean = ws["running_mean"][0]
        var = ws["running_var"][0]
        epsilon = np.array(ws["epsilon"])
        
        ys = F.batch_norm(
            self.to_torch(X),
            running_mean=self.to_torch(mean),
            running_var=self.to_torch(var),
            weight=self.to_torch(W),
            bias=self.to_torch(B),
            eps=self.to_torch(epsilon), # type: ignore
        )
        ys = self.to_numpy(ys)

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            attrs_cpy = copy.deepcopy(attrs)
            attrs_cpy.update(dict(
                                scale=ws["weight"][0], 
                                bias=ws["bias"][0],
                                mean=ws["running_mean"][0],
                                var=ws["running_var"][0],                                
                                epsilon=ws["epsilon"],
                            ))

            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs_cpy], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs_cpy]))

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0])) # type: ignore
                qxs = copy.deepcopy(dict(output=in_data_q))
            in_data = [qxs]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="layernormalization")
class ExportLayernormlization(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportLayernormlization, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = self.model_export.wExport_layernormalization

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        W = ws["weight"][0]
        B = ws["bias"][0]
        epsilon = np.array(ws["epsilon"])
        
        ys = F.layer_norm(
            self.to_torch(X),
            normalized_shape=W.shape,
            weight=self.to_torch(W),
            bias=self.to_torch(B),
            eps=self.to_torch(epsilon), # type: ignore
        )
        ys = self.to_numpy(ys)

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            attrs_cpy = copy.deepcopy(attrs)
            attrs_cpy.update(dict(
                                scale=ws["weight"][0], 
                                bias=ws["bias"][0],
                                epsilon=ws["epsilon"],
                            ))

            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs_cpy], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs_cpy]))

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0])) # type: ignore
                qxs = copy.deepcopy(dict(output=in_data_q))
            in_data = [qxs]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="instancenormalization")
class ExportInstancenormalization(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportInstancenormalization, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = self.model_export.wExport_instancenormalization

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        W = ws["weight"][0]
        B = ws["bias"][0]
        epsilon = np.array(ws["epsilon"])

        ys = F.instance_norm(
            self.to_torch(X),
            weight=self.to_torch(W),
            bias=self.to_torch(B),
            eps=self.to_torch(epsilon), # type: ignore
        )
        ys = self.to_numpy(ys)

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            attrs_cpy = copy.deepcopy(attrs)
            attrs_cpy.update(dict(
                                scale=ws["weight"][0], 
                                bias=ws["bias"][0],
                                epsilon=ws["epsilon"],
                            ))

            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs_cpy], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs_cpy]))

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0])) # type: ignore
                qxs = copy.deepcopy(dict(output=in_data_q))
            in_data = [qxs]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="softmax")
class ExportSoftmax(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportSoftmax, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = self.model_export.wExport_table

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]

        ys = F.softmax(self.to_torch(X), dim=attrs["axis"])
        ys = [self.to_numpy(ys)]

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0])) # type: ignore
                qxs = copy.deepcopy(dict(output=in_data_q))
            in_data = [qxs]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="conv")
@EXPORT_LAYER.register_module(name="depthwiseconv")
class ExportConv(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportConv, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Ksize]
        self.export_w = self.model_export.wExport_conv
        self.act_type = kwargs["attrs"]["fuse_op"]
        self.act_attrs = kwargs["attrs"]["act_attrs"]
        if self.act_type:
            self.act_type = self.act_type[0]
        self.ops_map = {
            "relu": F.relu,
            "relu6": lambda x: torch.clamp(
                x, 
                torch.Tensor([0.0]),
                torch.Tensor([self.act_attrs.get("value", 6.0)]),
            ),
            "relux": lambda x: torch.clamp(
                x, 
                torch.Tensor([0.0]),
                torch.Tensor([self.act_attrs.get("value", 12.0)]),
            ),
            "leakyrelu": lambda x: F.leaky_relu(x, negative_slope=self.act_attrs.get("alpha", 0.001)),
            "prelu": lambda x: F.prelu(x, weight=torch.from_numpy(self.act_attrs.get("slope", 0.001).astype(np.float32))),
            "sigmoid": torch.sigmoid,
            "swish": lambda x: x * torch.sigmoid(x), 
            "gelu": F.gelu,            
            "tanh": torch.tanh,
            "hardsigmoid": lambda x: torch.max(
                torch.Tensor([0.0]), 
                torch.min(torch.Tensor([1.0]), 
                    self.act_attrs.get("alpha", 0.2) * x \
                    + self.act_attrs.get("beta", 0.5)),
                ),
            "hardtanh": F.hardtanh,
            "hardswish": F.hardswish,
            "hardshrink": F.hardshrink,
        }
        
    def get_session(self, attrs, weight, bias, opset_version=14):
        def create_initializer(data, name): return onnx.helper.make_tensor( # type: ignore
            name=name, data_type=onnx.TensorProto.FLOAT, # type: ignore
            dims=data.shape, vals=data.tobytes(), raw=True)
        def create_in_out(name, shape): return onnx.helper.make_tensor_value_info( # type: ignore
            name, onnx.TensorProto.FLOAT, shape) # type: ignore

        in_c = attrs["in_c"]
        out_c = attrs["out_c"]
        kernel_shape = attrs["kernel_shape"]
        strides = attrs["strides"]
        dilations = attrs["dilations"]
        auto_pad = attrs.get("auto_pad")

        node = onnx.helper.make_node( # type: ignore
            "Conv",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            auto_pad=auto_pad,
        )

        if isinstance(weight, np.ndarray):
            W = weight.astype(np.float32)
        else:
            W = np.random.randn([in_c, out_c, kernel_shape[0], kernel_shape[1]]).astype(np.float32) # type: ignore
        if isinstance(bias, np.ndarray):
            B = bias.astype(np.float32)
        else:
            B = np.zeros([out_c]).astype(np.float32)

        initializers = [
            create_initializer(W, "W"),
            create_initializer(B, "B"),
        ]
        inputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["X"]]
        outputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["Y"]]

        graph = onnx.helper.make_graph( # type: ignore
            nodes=[node],
            name='test_conv',
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset_version)]) # type: ignore
        # onnx.save(model, "test_conv.onnx")

        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        return sess
    
    def get_float_result(self, attrs, xs, ws):
        stride = attrs["strides"]
        dilation = attrs["dilations"]
        groups = attrs["group"]
        kernel_shape = attrs["kernel_shape"]

        X = xs[0]
        W = ws["weight"][0]
        B = ws["bias"][0]

        input_shape = X.shape

        auto_pad = attrs.get("auto_pad")
        if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            pad_h = get_same_padding(input_shape[2], kernel_shape[0], stride[0], auto_pad=auto_pad)
            pad_w = get_same_padding(input_shape[3], kernel_shape[1], stride[1], auto_pad=auto_pad)
            padding = pad_h + pad_w
        elif auto_pad in ["VALID"]:
            padding = [0, 0, 0, 0]
        else:
            # torch: left, right, top, bottom <- onnx: pad_t, pad_l, pad_b, pad_r
            padding = [
                attrs["pads"][1],
                attrs["pads"][3],
                attrs["pads"][0],
                attrs["pads"][2],
            ]

        # sess = self.get_session(
        #     attrs=attrs,
        #     weight=W,
        #     bias=B,
        #     opset_version=15,
        # )
        # y_ort = sess.run(None, {"X": X})[0]
        
        # left, right, top, bottom
        padding_instance = nn.ZeroPad2d(tuple(padding)) 
        y = F.conv2d(
            input=padding_instance(self.to_torch(X)),
            weight=self.to_torch(W),
            bias=self.to_torch(B),
            stride=tuple(stride),
            padding=(0, 0),
            dilation=tuple(dilation),
            groups=groups,
        )
        ys = [self.to_numpy(y)]

        # b, _, h, w = y.shape
        # y += self.to_torch(B)[None, :, None, None].repeat(b, 1, h, w)
        # ys.append(self.to_numpy(y))

        if self.act_type:
            y = self.ops_map[self.act_type](y)
        ys.append(self.to_numpy(y))

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"] # type: ignore
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)

            ops_string = [self.layer_type, "bias", "act"]
            if len(attrs["fuse_op"]) > 0:
                ops_string[-1] = attrs["fuse_op"][0]

            in_quantizes = []
            in_quantize = quantize_factory.get(
                default_setting__[self.layer_type]["feat"]["method"]
            )(bit_select) # type: ignore
            in_quantize.get_quan_param(xs[0])
            in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)
            
            so = []
            quantizes = {}
            for i, out_data in enumerate(ys[:1]): # type: ignore
                quantize = quantize_factory.get(
                    # default_setting__[self.layer_type]["feat"]["method"]
                    "floatsymquan",
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                quantizes[f"sc{i}"] = quantize            
            for i, out_data in enumerate(ys[1:]): # type: ignore
                quantize = quantize_factory.get(
                    default_setting__[self.layer_type]["feat"]["method"]
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            w_quantize = quantize_factory.get(
                default_setting__[self.layer_type]["weights"]["method"]
            )(bit_select) # type: ignore
            w_quantize.get_quan_param(ws["weight"][0])
            qweight = w_quantize.get_quan_data(ws["weight"][0])
            self.layer.set_qweights(qweight)

            si = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
            sk = dict(zip(["scale", "zero_point"], w_quantize.get_scale()))
            so = dict(zip(["scale", "zero_point"], quantize.get_scale())) # type: ignore
            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            qbias = np.round(ws["bias"][0] / (si["scale"] * sk["scale"]))
            self.layer.set_qbias(qbias)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs, dict(), self.act_attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0]))
                qxs = copy.deepcopy(dict(output=in_data_q))
            in_data = [qxs]
            
            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()
            
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantizes['feat']['so0'].get_quan_data(ys[-1])], [
                    out_data[-1]["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data[-1]["output"]]

            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="convtranspose")
class ExportConvTranspose(ExportConv):
    def __init__(self, **kwargs):
        super(ExportConvTranspose, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Ksize]
        self.export_w = self.model_export.wExport_convtranspose

    def get_session(self, attrs, weight, bias, opset_version=14):
        def create_initializer(data, name): return onnx.helper.make_tensor( # type: ignore
            name=name, data_type=onnx.TensorProto.FLOAT, # type: ignore
            dims=data.shape, vals=data.tobytes(), raw=True)
        def create_in_out(name, shape): return onnx.helper.make_tensor_value_info( # type: ignore
            name, onnx.TensorProto.FLOAT, shape) # type: ignore

        in_c = attrs["in_c"]
        out_c = attrs["out_c"]
        kernel_shape = attrs["kernel_shape"]
        strides = attrs["strides"]
        pads = attrs["pads"]
        dilations = attrs["dilations"]
        if "output_padding" in attrs.keys():
            output_padding = attrs["output_padding"]
        else:
            output_padding = [0, 0]

        node = onnx.helper.make_node( # type: ignore
            "ConvTranspose",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
            dilations=dilations,
            output_padding=output_padding,
            # auto_pad="SAME_UPPER",
        )

        if isinstance(weight, np.ndarray):
            W = weight.astype(np.float32)
        else:
            W = np.random.randn([in_c, out_c, kernel_shape[0], kernel_shape[1]]).astype(np.float32) # type: ignore
        if isinstance(bias, np.ndarray):
            B = bias.astype(np.float32)
        else:
            B = np.zeros([out_c]).astype(np.float32)

        initializers = [
            create_initializer(W, "W"),
            create_initializer(B, "B"),
        ]
        inputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["X"]]
        outputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["Y"]]

        graph = onnx.helper.make_graph( # type: ignore
            nodes=[node],
            name='test_convtranspose',
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset_version)]) # type: ignore
        # onnx.save(model, "test_convtranspose.onnx")

        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        return sess

    def get_float_result(self, attrs, xs, ws):
        padding = [
            attrs["pads"][1],
            attrs["pads"][3],
            attrs["pads"][0],
            attrs["pads"][2],
        ]
        stride = attrs["strides"]
        dilation = attrs["dilations"]
        groups = attrs["group"]
        output_padding = attrs["output_padding"]

        X = xs[0]
        W = ws["weight"][0]
        B = ws["bias"][0]

        # padz = nn.ZeroPad2d(tuple([1, 2, 1, 2]))
        # y = F.conv_transpose2d(
        #     input=self.to_torch(X),
        #     weight=self.to_torch(W),
        #     bias=self.to_torch(B),
        #     stride=tuple(stride),
        #     # padding=tuple(padding[:2]),
        #     output_padding=tuple(output_padding),
        #     dilation=tuple(dilation),
        #     groups=groups,
        # )
        # ys = [self.to_numpy(y)]

        sess = self.get_session(
            attrs=attrs,
            weight=W,
            bias=B,
            opset_version=15,
        )

        y = sess.run(None, {"X": X})[0]
        ys = [y]

        # b, _, h, w = y.shape
        # y += self.to_torch(B)[None, :, None, None].repeat(b, 1, h, w)
        # ys.append(self.to_numpy(y))

        if isinstance(y, np.ndarray):
            y = self.to_torch(y)
        if self.act_type:
            y = self.ops_map[self.act_type](y)
        ys.append(self.to_numpy(y))

        return ys


@EXPORT_LAYER.register_module(name="fc")
class ExportFC(ExportConv):
    def __init__(self, **kwargs):
        super(ExportFC, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.I_Align, self.model_export.O_Align]
        self.export_w = self.model_export.wExport_fc

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        W = ws["weight"][0]
        B = ws["bias"][0]

        y = F.linear(
            input=self.to_torch(X), weight=self.to_torch(W), bias=self.to_torch(B)
        )
        ys = [self.to_numpy(y)]

        # b, _ = y.shape
        # y += self.to_torch(B)[None, :].repeat(b, 1)
        # ys.append(self.to_numpy(y))

        if self.act_type:
            y = self.ops_map[self.act_type](y)
        ys.append(self.to_numpy(y))

        return ys


@EXPORT_LAYER.register_module(name="globalaveragepool")
@EXPORT_LAYER.register_module(name="averagepool")
@EXPORT_LAYER.register_module(name="maxpool")
class ExportPool(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportPool, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"] # type: ignore
        self.align_size = [self.model_export.Csize, self.model_export.Ksize]
        self.export_w = None

    def get_session(self, attrs, opset_version=15):
        def create_initializer(data, name): return onnx.helper.make_tensor( # type: ignore
            name=name, data_type=onnx.TensorProto.FLOAT, # type: ignore
            dims=data.shape, vals=data.tobytes(), raw=True)
        def create_in_out(name, shape): return onnx.helper.make_tensor_value_info( # type: ignore
            name, onnx.TensorProto.FLOAT, shape) # type: ignore
                
        initializers = []

        if attrs == dict():
            op_type = "GlobalAveragePool"
        else:
            op_type = "AveragePool"

        node = onnx.helper.make_node(# type: ignore
            op_type,
            inputs=['X'],
            outputs=["Y"],
            **attrs
        )

        inputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["X"]]
        outputs = [create_in_out(name, ['n', 'c', 'h', 'w']) for name in ["Y"]]

        graph = onnx.helper.make_graph(# type: ignore
            nodes=[node],
            name='test_Avgpool',
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset_version)])# type: ignore
        # onnx.save(model, "test_convtranspose.onnx")

        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        return sess
    
    def do_averagepool(self, attrs, xs, ws):
        data = xs[0]

        pads = attrs["pads"]
        pads = [pads[1], pads[3], pads[0], pads[2]]
        kernel_size = attrs["kernel_shape"]
        stride = attrs["strides"]
        ceil_mode = attrs["ceil_mode"]

        in_data = torch.from_numpy(data)
        max_value = torch.max(in_data) + 1.0
        c_padding = nn.ConstantPad2d(tuple(pads), value=max_value) # type: ignore
        z_padding = nn.ZeroPad2d(tuple(pads))
        x_ = (c_padding(in_data) < max_value).float()
        # x_w = F.avg_pool2d(
        #     input=x_,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=(0, 0),
        #     ceil_mode=ceil_mode,
        #     divisor_override=1,
        # )
        y = F.avg_pool2d(
            input=z_padding(in_data),
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0),
            ceil_mode=ceil_mode,
            divisor_override=1,
        )
        y = y / (kernel_size[0] * kernel_size[1])  # x_w

        ys = [self.to_numpy(y)]

        return ys

    def do_maxpool(self, attrs, xs, ws):
        data = xs[0]

        pads = attrs["pads"]
        pads = [pads[1], pads[3], pads[0], pads[2]]
        kernel_size = attrs["kernel_shape"]
        stride = attrs["strides"]
        ceil_mode = attrs["ceil_mode"]

        in_data = data.astype(np.float32)
        min_v = np.min(in_data)

        in_data = torch.from_numpy(in_data)
        padding = nn.ConstantPad2d(tuple(pads), value=min_v - 1)
        y = F.max_pool2d(
            input=padding(in_data),
            kernel_size=kernel_size,
            stride=tuple(stride),
            padding=(0, 0),
            ceil_mode=ceil_mode,
        )
        ys = [self.to_numpy(y)]

        return ys

    def get_float_result(self, attrs, xs, ws):

        if self.layer_type == "globalaveragepool":
            layer_type = "averagepool"
        else:
            layer_type = self.layer_type

        ys = getattr(self, "do_{}".format(layer_type))(attrs, xs, ws)

        # sess = self.get_session(attrs)
        # output = sess.run(None, {"X": xs[0].astype(np.float32)})[0]
        
        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            ops_string = [self.layer_type]

            if self.layer_type == "maxpool":
                default_setting = "floatsymquan"
            else:
                default_setting = default_setting__[self.layer_type]["feat"]["method"]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            if process_scale in ["smooth"]:
                quantizes["so0"] = in_quantizes[0]
                so = si
            else:
                for i, out_data in enumerate(ys):
                    quantize = quantize_factory.get( # type: ignore
                        default_setting # type: ignore
                    )(bit_select) # type: ignore
                    quantize.get_quan_param(out_data)
                    so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                    so.append(so_)
                    quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)
                
            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                in_data = [copy.deepcopy(dict(output=xs[0]))]
            else:
                in_data_q = in_quantizes[0].get_quan_data(copy.deepcopy(xs[0]))
                in_data_q = dict(output=in_data_q)
                in_data = [copy.deepcopy(in_data_q)]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantizes["feat"]["so0"].get_quan_data(ys[-1])], [
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]
            self.calc_error(fresults, qresults)
            
            
@EXPORT_LAYER.register_module(name="relu")
@EXPORT_LAYER.register_module(name="relu6")
@EXPORT_LAYER.register_module(name="relux")
@EXPORT_LAYER.register_module(name="leakyrelu")
@EXPORT_LAYER.register_module(name="prelu")
@EXPORT_LAYER.register_module(name="sigmoid")
@EXPORT_LAYER.register_module(name="swish")
@EXPORT_LAYER.register_module(name="gelu")
@EXPORT_LAYER.register_module(name="tanh")
@EXPORT_LAYER.register_module(name="hardsigmoid")
@EXPORT_LAYER.register_module(name="hardtanh")
@EXPORT_LAYER.register_module(name="hardswish")
@EXPORT_LAYER.register_module(name="hardshrink")
class ExportAct(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportAct, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Ksize]
        self.export_w = self.model_export.wExport_table
        self.ops_map = {
            "relu": F.relu,
            "relu6": lambda x: torch.clamp(
                x, 
                torch.Tensor([0.0]),
                torch.Tensor([kwargs["attrs"].get("value", 6.0)]),
            ),
            "relux": lambda x: torch.clamp(
                x, 
                torch.Tensor([0.0]),
                torch.Tensor([kwargs["attrs"].get("value", 12.0)]),
            ),
            "leakyrelu": lambda x: F.leaky_relu(x, negative_slope=kwargs["attrs"].get("alpha", 0.001)),
            "prelu": lambda x: F.prelu(x, weight=torch.from_numpy(kwargs["attrs"].get("slope", 0.001).astype(np.float32))),
            "sigmoid": torch.sigmoid,
            "swish": lambda x: x * torch.sigmoid(x), 
            "gelu": F.gelu,            
            "tanh": torch.tanh,
            "hardsigmoid": lambda x: torch.max(
                torch.Tensor([0.0]), 
                torch.min(torch.Tensor([1.0]), 
                    kwargs["attrs"].get("alpha", 0.2) * x \
                    + kwargs["attrs"].get("beta", 0.5)),
                ),
            "hardtanh": F.hardtanh,
            "hardswish": F.hardswish,
            "hardshrink": F.hardshrink,
        }

    def get_float_result(self, attrs, xs, ws):
        data = xs[0]
        y = self.ops_map[self.layer_type](torch.from_numpy(data))
        ys = [self.to_numpy(y)]

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_table([])
            self.layer.set_scale_type(process_scale)
            
            attrs_cpy = copy.deepcopy(attrs)
            if self.layer_type in ["relu6", "relux"]:
                value = attrs_cpy.pop("value")
                attrs_cpy.update(dict(min=0.0, max=value))            

            ops_string = [self.layer_type]

            in_quantizes = []
            in_quantize = quantize_factory.get( # type: ignore
                default_setting__[self.layer_type]["feat"]["method"] # type: ignore
            )(bit_select) # type: ignore
            in_quantize.get_quan_param(xs[0])
            in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys[-1:]):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            si = [dict(zip(["scale", "zero_point"], in_quantize.get_scale()))]
            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))
            if process_scale in ["smooth"]:
                so = si
            else:
                so = [dict(zip(["scale", "zero_point"], quantize.get_scale()))] # type: ignore

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs_cpy], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs_cpy]))

            if in_type in [np.float32, np.float64]:
                in_data = [copy.deepcopy(dict(output=xs[0]))]
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0]))
                in_data = [copy.deepcopy(dict(output=in_data_q))]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]

            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="resize")
class ExportResize(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportResize, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None
    
    def get_session(self, attrs, opset_version=15):
        def create_in_out(name, shape, data_type): return onnx.helper.make_tensor_value_info( # type: ignore
            name, data_type, shape)
        def create_initializer(data, name, data_type): return onnx.helper.make_tensor( # type: ignore
            name=name, data_type=data_type,
            dims=data.shape, vals=data.tobytes(), raw=True)
                
        new_attrs = {}
        for key in attrs.keys():
            if key in ['scale', 'roi']:
                continue
            if key not in ['coordinate_transformation_mode', 'cubic_coeff_a', 'mode', 'nearest_mode']:
                continue
            new_attrs[key] = attrs[key]

        scales = dict()
        if 'scale' in attrs.keys():
            scale = attrs['scale']
            scale = np.array(scale) if isinstance(scale, list) else scale
            scales = dict(scale=scale)
        else:
            scales = dict(scale=None)
        if 'roi' in attrs.keys():
            roi = attrs['roi']
            roi = np.array(roi) if isinstance(roi, list) else roi            
            scales.update(roi=roi) # type: ignore
        else:
            scales.update(roi=None)
                        
        initializers = []

        inputs = ['X', '', '', '']
        outputs = [create_in_out(name, ['n', 'c', 'h', 'w'], onnx.TensorProto.FLOAT) for name in ["Y"]] # type: ignore
        if isinstance(scales['roi'], np.ndarray):
            inputs[1] = 'roi'
            roi = scales['roi'] if isinstance(scales['roi'], np.ndarray) else 0
            initializers.extend([
                create_initializer(roi, 'roi', onnx.TensorProto.FLOAT) # type: ignore
            ])
        if isinstance(scales['scale'], np.ndarray):
            inputs[2] = 'scales'
            inputs = inputs[:3]
            scale = scales['scale'] if isinstance(scales['scale'], np.ndarray) else 0
            initializers.extend([
                create_initializer(scale.astype(np.float32), 'scales', onnx.TensorProto.FLOAT), # type: ignore
            ])
        else:
            inputs[3] = 'sizes'
            sizes = (
                np.array(attrs["sizes"]).astype(np.int64) if "sizes" in attrs.keys() else None
            )
            size_o = [int(s) for s in sizes]  # type: ignore
            outputs = [create_in_out(name, size_o, onnx.TensorProto.FLOAT) for name in ["Y"]] # type: ignore
            initializers.extend([
                create_initializer(np.array(size_o), "sizes", onnx.TensorProto.INT64), # type: ignore
                ])

        node = onnx.helper.make_node( # type: ignore
            "Resize",
            inputs=inputs,
            outputs=["Y"],
            **new_attrs
        )

        inputs = [create_in_out(name, ['n', 'c', 'h', 'w'], onnx.TensorProto.FLOAT) for name in ["X"]] # type: ignore

        graph = onnx.helper.make_graph( # type: ignore
            nodes=[node],
            name='test_resize',
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
            )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset_version)]) # type: ignore
        # onnx.save(model, "test_resize.onnx")
        model.ir_version = 8
        sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

        return sess
        
    def get_float_result(self, attrs, xs, ws):
        data = xs[0]
        
        sess = self.get_session(attrs)

        x_name = sess.get_inputs()[0].name
        y_name = sess.get_outputs()[0].name

        y = sess.run(
            [y_name],
            {
                x_name: data,
            },
        )[0]

        ys = [y]

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']
            if attrs["mode"] in ['nearest']:
               process_scale = "smooth"
                 
            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                in_data = [copy.deepcopy(dict(output=xs[0]))]
            else:
                in_data_q = in_quantizes[0].get_quan_data(copy.deepcopy(xs[0]))
                in_data_q = dict(output=in_data_q)
                in_data = [copy.deepcopy(in_data_q)]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data["output"]]
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="lstm")
class ExportLstm(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportLstm, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = self.model_export.wExport_lstm

    def lstm_cell(self, x, h, c, w1, w2, b1, b2):
        x = torch.from_numpy(x).type(torch.float32)
        h = torch.from_numpy(h).type(torch.float32)
        c = torch.from_numpy(c).type(torch.float32)

        w1 = torch.from_numpy(w1).type(torch.float32)
        w2 = torch.from_numpy(w2).type(torch.float32)
        b1 = torch.from_numpy(b1).type(torch.float32)
        b2 = torch.from_numpy(b2).type(torch.float32)

        xw = F.linear(x, w1.squeeze(dim=0), b1.squeeze(dim=0))
        xr = F.linear(h, w2.squeeze(dim=0), b2.squeeze(dim=0))
        y = xw + xr
        # it, ft, ct, ot = torch.chunk(y, 4, 1)
        it, ot, ft, ct = torch.chunk(y, 4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        ct = torch.tanh(ct)
        ot = torch.sigmoid(ot)

        ct = ft * c + it * ct
        ht = ot * torch.tanh(ct)

        return ht, ht, ct

    def get_float_result(self, attrs, xs, ws):
        x, h, c = xs
        w1, w2 = ws["weight"]
        b1, b2 = ws["bias"]
        time_step = attrs["sequence_lens"]

        # lstm_cell = nn.LSTM(attrs['in_c'], attrs['hidden_size'], 1, bidirectional=False)
        # lstm_cell.all_weights[0][0].data = self.to_torch(w1.squeeze())
        # lstm_cell.all_weights[0][1].data = self.to_torch(w2.squeeze())
        # lstm_cell.all_weights[0][2].data = self.to_torch(b1.squeeze())
        # lstm_cell.all_weights[0][3].data = self.to_torch(b2.squeeze())

        y = []
        for time in range(time_step):
            xt, ht, ct = self.lstm_cell(x[time], h[time], c[time], w1, w2, b1, b2)
            y.append(xt)
        y = torch.stack(y, dim=0)
        ht = ht.unsqueeze(dim=0) # type: ignore
        ct = ct.unsqueeze(dim=0) # type: ignore

        # y, (ht, ct) = lstm_cell(self.to_torch(x), (self.to_torch(h), self.to_torch(c)))

        ys = [self.to_numpy(y), self.to_numpy(ht), self.to_numpy(ct)]

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        in_type = self.setting["setting"]["in_type"]
        in_type = eval(self.setting["setting"]["bits_dict"][in_type])
        out_type = self.setting["setting"]["out_type"]
        out_type = eval(self.setting["setting"]["bits_dict"][out_type])
        if self.layer_type in layer_factory.module_dict:
            default_setting = self.setting["setting"]["default_setting"]
            process_scale = default_setting[self.layer_type]["process_scale"]
            bit_select = self.setting["setting"]["bit_select"]
            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_table([])
            self.layer.set_layer_ops(
                dict(
                    attrs=[
                        dict(
                            weight=ws["weight"],
                            bias=ws["bias"],
                            hidden_size=attrs["hidden_size"],
                            in_c=attrs["in_c"],
                            sequence_lens=attrs["sequence_lens"],
                            initial_h=attrs["initial_h"],
                            initial_c=attrs["initial_c"],
                        )
                    ]
                )
            )
            self.layer.set_scale_type(process_scale)

            ops_names = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            wr_combine = attrs["wr_combine"]
            hx_combine = attrs["hx_combine"]

            sk = []
            qweights, qbiass = [], []
            w_quantizes = []
            for idx, (w_data, b_data) in enumerate(zip(ws["weight"], ws["bias"])):
                if wr_combine:
                    w_data_ = np.concatenate(
                        [
                            copy.deepcopy(ws["weight"][0]).reshape(1, -1),
                            copy.deepcopy(ws["weight"][1]).reshape(1, -1),
                        ],
                        axis=1,
                    )
                    # b_data_ = np.concatenate([
                    #     copy.deepcopy(ws['bias'][0]).reshape(1, -1),
                    #     copy.deepcopy(ws['bias'][1]).reshape(1, -1)], axis=1)
                else:
                    w_data_ = w_data

                w_quantize = quantize_factory.get( # type: ignore
                    default_setting[self.layer_type]["weights"]["method"] # type: ignore
                )(bit_select) # type: ignore
                w_quantize.get_quan_param(w_data_)
                qweight = w_quantize.get_quan_data(w_data)
                qweights.append(qweight)

                sk_ = dict(zip(["scale", "zero_point"], w_quantize.get_scale()))
                sk.append(sk_)

                if hx_combine:
                    qbias = np.round(b_data / (si[0]["scale"] * sk[0]["scale"]))
                else:
                    qbias = np.round(b_data / (si[idx]["scale"] * sk[idx]["scale"]))
                qbiass.append(qbias)

                w_quantizes.append(w_quantize)

            self.layer.set_qweights(qweights)
            self.layer.set_qbias(qbiass)

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)
            is_update_quantize_from_in_data=attrs["is_update_quantize_from_in_data"]
            self.layer.set_quantize_from_in_data(is_update_quantize_from_in_data)
            
            bits_dict = {
                k: eval(v) for k, v in self.setting["setting"]["bits_dict"].items()
            }
            op_setting = dict(
                w_bit_select=bit_select,
                bit_select=bit_select,
                bits_dict=bits_dict,
                mins=self.setting["setting"]["mins"],
                maxs=self.setting["setting"]["maxs"],
                txme_saturation=self.setting["setting"]["txme_saturation"],
				virtual_round=self.setting["setting"]["virtual_round"],
                si=si,
                sk=sk,
                so=so,
                in_quantize=in_quantizes,
                quantize=quantizes,
                setting=self.setting["setting"],
            )
            op_setting["setting"]["method"] = default_setting[self.layer_type]["feat"][ # type: ignore
                "method"
            ]
            op_setting.update(default_setting[self.layer_type])
            op_setting.update({"attrs": self.layer.get_layer_ops()["attrs"]})
            op_setting.update(self.layer.get_layer_ops()["attrs"][0])
            op_setting.update(dict(p_weights=qweights, bias=qbiass))
            self.layer.set_ops_setting(op_setting)

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0])) # type: ignore
                qxs = copy.deepcopy(dict(output=in_data_q))
            in_data = [qxs]
            self.layer.set_in_data(in_data)

            out_data = []
            qys = [copy.deepcopy(qxs)]
            for ops_name in ops_names:
                # op_setting['process_scale'] = default_setting[ops_name]['process_scale']
                op = operators_factory.get(ops_name)(**op_setting) # type: ignore
                self.layer.set_ops_instance(op)
                qys = op(qys, is_update_quantize_from_in_data=is_update_quantize_from_in_data)
                # out_data.append(copy.deepcopy(qys[0]))
                out_data.extend(qys)
                if process_scale in ["table"]:
                    table = op.get_table()
                    self.layer.set_table(table)

            self.layer.set_out_data(out_data)

            self.layer.set_layer_ops(dict(ops=ops_names))
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)

            # fresults, qresults = [quantizes['feat']['so0'].get_quan_data(
            #     ys[0])], [out_data[0]['output']]
            fresults, qresults = [ys[0]], [out_data[0]["output"]]
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="gru")
class ExportGru(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportGru, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = self.model_export.wExport_gru

    def gru_cell(self, x, h, W, R, Wb, Rb, linear_before_reset=0, is_ort=True):
        x = torch.from_numpy(x).type(torch.float32)
        h = torch.from_numpy(h).type(torch.float32)

        W = torch.from_numpy(W).type(torch.float32)
        R = torch.from_numpy(R).type(torch.float32)
        Wb = torch.from_numpy(Wb).type(torch.float32)
        Rb = torch.from_numpy(Rb).type(torch.float32)

        gate_x = F.linear(x, W.squeeze(dim=0), Wb.squeeze(dim=0))
        gate_h = F.linear(h, R.squeeze(dim=0), Rb.squeeze(dim=0))

        if is_ort:
            iz, ir, ih = gate_x.chunk(3, 1)
            hz, hr, hh = gate_h.chunk(3, 1)
        else:
            ir, iz, ih = gate_x.chunk(3, 1)
            hr, hz, hh = gate_h.chunk(3, 1)

        rt = F.sigmoid(ir + hr)
        zt = F.sigmoid(iz + hz)
        if linear_before_reset != 0: ### pytorch default is 1
            ht = F.tanh(ih + (rt * hh))
        else: ### onnx default is 0
            tmp = rt * h
            Rh = R.chunk(3, dim=1)[-1]
            Rbh = Rb.chunk(3, dim=1)[-1]
            tmp = F.linear(tmp, Rh.squeeze(dim=0), Rbh.squeeze(dim=0))
            ht = F.tanh(ih + tmp)

        Ht = (1 - zt) * ht + zt * h
        y = Ht

        return y, Ht

    def get_float_result(self, attrs, xs, ws):
        x, h = xs
        W, R = ws["weight"]
        Wb, Rb = ws["bias"]
        time_step = attrs["sequence_lens"]
        linear_before_reset = attrs["linear_before_reset"]

        # gru_cell = nn.GRU(attrs['in_c'], attrs['hidden_size'], 1, bidirectional=False)
        # gru_cell.all_weights[0][0].data = self.to_torch(W.squeeze())
        # gru_cell.all_weights[0][1].data = self.to_torch(R.squeeze())
        # gru_cell.all_weights[0][2].data = self.to_torch(Wb.squeeze())
        # gru_cell.all_weights[0][3].data = self.to_torch(Rb.squeeze())
        # y_torch, ht_torch = gru_cell(self.to_torch(x), (self.to_torch(h)))
        # y_torch, ht_torch = y_torch.detach().numpy(), ht_torch.detach().numpy()
        # torch.onnx.export(gru_cell,
        #                 (self.to_torch(x), self.to_torch(h)),
        #                 "test_torch_gru.onnx",
        #                 export_params=True,
        #                 opset_version=15,
        #                 do_constant_folding=True,
        #                 input_names=['input', "h0"],
        #                 output_names=['output', "ht"])
        # model = onnx.load_model("test_torch_gru.onnx")
        # sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        # y_rt, ht_rt = sess.run(None, {"input": x, "h0": h})

        is_ort = True
        if is_ort: # pytorch weight parameter -> onnxruntime  weight parameter
            # weights_data = self.parser_weight(model)
            # W = weights_data["onnx::GRU_87"]
            # R = weights_data["onnx::GRU_88"]
            # bias = weights_data["onnx::GRU_89"].reshape(1, 2, -1)
            # Wb = bias[:, 0, :]
            # Rb = bias[:, 1, :]

            w1, w2, w3 = np.split(W, 3, axis=1)
            W = np.concatenate([w2, w1, w3], axis=1)
            r1, r2, r3 = np.split(R, 3, axis=1)
            R = np.concatenate([r2, r1, r3], axis=1)

            wb1, wb2, wb3 = np.split(Wb, 3, axis=1)
            Wb = np.concatenate([wb2, wb1, wb3], axis=1)
            rb1, rb2, rb3 = np.split(Rb, 3, axis=1)
            Rb = np.concatenate([rb2, rb1, rb3], axis=1)
            ws["weight"] = [W, R]
            ws["bias"] = [Wb, Rb]

        y = []
        for time in range(time_step):
            xt, ht = self.gru_cell(x[time], h[time], W, R, Wb, Rb, linear_before_reset, is_ort=is_ort)
            y.append(xt)
        y = torch.stack(y, dim=0)
        ht = ht.unsqueeze(dim=0) # type: ignore

        ys = [self.to_numpy(y), self.to_numpy(ht)]

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx([-1, -2])
            self.layer.set_output_idx([-1, -2])
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_table([])

            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            wr_combine = attrs["wr_combine"]
            hx_combine = attrs["hx_combine"]

            sk = []
            qweights, qbiass = [], []
            w_quantizes = []
            for idx, (w_data, b_data) in enumerate(zip(ws["weight"], ws["bias"])):
                if wr_combine:
                    w_data_ = np.concatenate(
                        [
                            copy.deepcopy(ws["weight"][0]).reshape(1, -1),
                            copy.deepcopy(ws["weight"][1]).reshape(1, -1),
                        ],
                        axis=1,
                    )
                else:
                    w_data_ = copy.deepcopy(w_data)

                w_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["weights"]["method"] # type: ignore
                )(bit_select) # type: ignore
                w_quantize.get_quan_param(w_data_)
                qweight = w_quantize.get_quan_data(w_data)
                qweights.append(qweight)

                sk_ = dict(zip(["scale", "zero_point"], w_quantize.get_scale()))
                sk.append(sk_)

                if hx_combine:
                    qbias = np.round(b_data / (si[0]["scale"] * sk[0]["scale"]))
                else:
                    qbias = np.round(b_data / (si[idx]["scale"] * sk[idx]["scale"]))
                qbiass.append(qbias)

                w_quantizes.append(w_quantize)

            self.layer.set_qweights(qweights)
            self.layer.set_qbias(qbiass)

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            setting.update(dict(hx_combine=attrs["hx_combine"], wr_combine=attrs["wr_combine"]))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                qxs0 = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantizes[0].get_quan_data(copy.deepcopy(xs[0]))
                qxs0 = copy.deepcopy(dict(output=in_data_q))
            qxs1 = copy.deepcopy(dict(output=xs[1]))
            in_data = [qxs0, qxs1]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantizes['feat']['so0'].get_quan_data(
                    ys[0])], [out_data[0]['output']]
            else:
                fresults, qresults = [ys[0]], [out_data[0]["output"]]

            self.calc_error(fresults, qresults)

@EXPORT_LAYER.register_module(name="splice")
class ExportSplice(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportSplice, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.I_Align, self.model_export.O_Align]
        self.export_w = None #self.model_export.wExport_fc
        self.debug = False
        self.bais_correction = True

    def get_float_result(self, attrs, xs, ws):
        X = xs[0]
        context = attrs["context"]
        forward_indexes = attrs["forward_indexes"]

        y = pyops.py_cpu_splice(
            X.astype(np.float32),
            np.array(context).astype(np.int32),
            np.array(forward_indexes).astype(np.int32))

        ys = [y]

        if len(attrs["fuse_op"]) > 0 and attrs["fuse_op"][0] == "fc":
            W = ws["weight"][0]
            B = ws["bias"][0]
            y = F.linear(
                input=self.to_torch(y), weight=self.to_torch(W), bias=self.to_torch(B)
            )
            ys.append(self.to_numpy(y))

        return ys

    def loadtxt(self, filename, delimiter=','):
        datas = []
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n").strip(delimiter).split(delimiter)
                for data in line:
                    datas.append(float(data))
                # print("test")

        return np.array(datas)

    def run_export_layer(self, attrs, xs, ws):
        splice_in = self.loadtxt('splice_in.txt', delimiter=",")
        splice_in = splice_in.reshape(191, attrs["in_c"]).astype(np.float32)
        xs[0] = splice_in[38:112, :]
        weights = self.loadtxt('mat.txt', delimiter=" ")
        weights = weights.reshape(attrs["out_c"], attrs["out_c"] + 1)
        ws["weight"] = [weights[:, :-1].astype(np.float32)]
        ws["bias"] = [weights[:, -1].astype(np.float32)]
        feat_out = self.loadtxt('feat_out.txt', delimiter=" ")
        feat_out = feat_out.reshape(191, attrs["out_c"])
        feat_out = feat_out[38:112, :].astype(np.float32)
        ys = self.get_float_result(attrs, xs, ws)
        diff = feat_out[12:62] - ys[-1][12:62]
        assert np.max(np.abs(diff)) < 1.0e-5
        ys[-1] = feat_out

        in_type = self.setting["setting"]["in_type"]
        in_type = eval(self.setting["setting"]["bits_dict"][in_type])
        out_type = self.setting["setting"]["out_type"]
        out_type = eval(self.setting["setting"]["bits_dict"][out_type])
        if self.layer_type in layer_factory.module_dict:
            default_setting = self.setting["setting"]["default_setting"]
            process_scale = default_setting[self.layer_type]["process_scale"]
            bit_select = self.setting["setting"]["bit_select"]
            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_table([])
            attrs.update(bias=True)
            self.layer.set_layer_ops(dict(attrs=[attrs]))
            self.layer.set_scale_type(process_scale)

            ops_names = [self.layer_type]

            in_quantize = quantize_factory.get( # type: ignore
                default_setting[self.layer_type]["feat"]["method"] # type: ignore
            )(bit_select) # type: ignore

            in_quantize.get_quan_param(xs[0])
            self.layer.set_in_quantize([in_quantize])
            so = []
            quantizes = {}
            for i, out_data in enumerate(ys[-1:]):
                quantize = quantize_factory.get( # type: ignore
                    default_setting[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            si = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
            so = dict(zip(["scale", "zero_point"], quantize.get_scale())) # type: ignore

            if attrs["is_first"]:
                w_quantize = quantize_factory.get( # type: ignore
                    default_setting[self.layer_type]["weights"]["method"] # type: ignore
                )(bit_select) # type: ignore


                qweight = w_quantize.get_quan_data(ws["weight"][0])
                # qweight = ws["weight"][0]
                self.layer.set_qweights(qweight)
                sk = dict(zip(["scale", "zero_point"], w_quantize.get_scale()))
                qbias = np.round(ws["bias"][0] / (si["scale"] * sk["scale"]))
                self.layer.set_qbias(qbias)

            else:
                sk = dict(zip(["scale", "zero_point"], [1.0, 0]))
                qweight, qbias = None, None

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            bits_dict = {
                k: eval(v) for k, v in self.setting["setting"]["bits_dict"].items()
            }
            op_setting = dict(
                w_bit_select=bit_select,
                bit_select=bit_select,
                bits_dict=bits_dict,
                mins=self.setting["setting"]["mins"],
                maxs=self.setting["setting"]["maxs"],
                txme_saturation=self.setting["setting"]["txme_saturation"],
                virtual_round=self.setting["setting"]["virtual_round"],
                si=si,
                sk=sk,
                so=so,
                in_quantize=[in_quantize],
                quantize=quantizes,
                setting=self.setting["setting"],
            )
            op_setting.update(default_setting[self.layer_type])
            op_setting.update(attrs)
            op_setting.update(dict(p_weights=qweight, bias=qbias)) # type: ignore
            op_setting.update(dict(attrs=[attrs]))
            op_setting.update(dict(ops_string=ops_names))
            if attrs["is_first"]:
                op_setting["setting"].update( # type: ignore
                    dict(method=default_setting[self.layer_type]["weights"]["method"])
                )
            self.layer.set_ops_setting(op_setting)

            if in_type in [np.float32, np.float64]:
                qxs = copy.deepcopy(dict(output=xs[0]))
            else:
                in_data_q = in_quantize.get_quan_data(copy.deepcopy(xs[0]))
                qxs = copy.deepcopy(dict(output=in_data_q))

            in_data = [qxs]
            self.layer.set_in_data(in_data)
            qys = copy.deepcopy(qxs)
            out_data = []
            op_list = []
            for ops_name in ops_names:
                op = operators_factory.get(ops_name)(**op_setting) # type: ignore
                op_list.append(op)
                qys = op(qys)
                out_data.append(copy.deepcopy(qys))
                if process_scale in ["table"]:
                    table = op.get_table()
                    self.layer.set_table(table)

            if self.bais_correction:
                ### bias correction
                fresults, qresults = [ys[-1]], [out_data[-1]["output"]]
                delta_bias = qresults[0][12:62] - fresults[0][12:62]
                delta_bias = delta_bias.mean(axis=0)
                qbias = np.round((ws["bias"][0] - delta_bias) / (si["scale"] * sk["scale"]))
                op_setting.update(dict(p_weights=qweight, bias=qbias)) # type: ignore
                self.layer.set_ops_setting(op_setting)

                in_data = [qxs]
                self.layer.set_in_data(in_data)
                qys = copy.deepcopy(qxs)
                out_data = []
                op_list = []
                for ops_name in ops_names:
                    op = operators_factory.get(ops_name)(**op_setting) # type: ignore
                    op_list.append(op)
                    qys = op(qys)
                    out_data.append(copy.deepcopy(qys))
                    if process_scale in ["table"]:
                        table = op.get_table()
                        self.layer.set_table(table)
                ### bias correction

            self.layer.set_ops_instance(op_list)
            self.layer.set_out_data(out_data)
            self.layer.set_layer_ops(dict(ops=ops_names))
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)

            if not out_type in [np.float32, np.float64]:
                fresults, qresults = [quantize.get_quan_data(ys[-1])], [ # type: ignore
                    out_data[-1]["output"]
                ]
            else:
                fresults, qresults = [ys[-1]], [out_data[-1]["output"]]

            self.calc_error([fresults[0][12:62]], [qresults[0][12:62]])


@EXPORT_LAYER.register_module(name="add")
@EXPORT_LAYER.register_module(name="cadd")
@EXPORT_LAYER.register_module(name="sub")
@EXPORT_LAYER.register_module(name="csub")
@EXPORT_LAYER.register_module(name="mul")
@EXPORT_LAYER.register_module(name="pmul")
@EXPORT_LAYER.register_module(name="cmul")
class ExportElementwise(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportElementwise, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        a = xs[0]
        b = xs[1]

        if attrs["layer_type"] == "add":
            y = a + b
        elif attrs["layer_type"] == "sub":
            y = a - b
        elif attrs["layer_type"] == "pmul":
            y = a * b
        elif attrs["layer_type"] in ["cmul", "cadd", "csub"]:
            a = torch.from_numpy(a)
            b = torch.from_numpy(b)
            a = a.squeeze(dim=3).squeeze(dim=2)
            b = b.squeeze(dim=3).squeeze(dim=2)
            if len(a.shape) < len(b.shape):
                a = a[:, :, None, None]
                a = a.expand_as(b)
            else:
                b = b[:, :, None, None]
                b = b.expand_as(a)
            if "mul" in attrs["layer_type"]:
                y = a * b
            elif "add" in attrs["layer_type"]:
                y = a + b
            elif "sub" in attrs["layer_type"]:
                y = a - b

            y = self.to_numpy(y) # type: ignore

        ys = [y] # type: ignore

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx([-1 for _ in xs])
            self.layer.set_output_idx([-1])
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            if 'mul' in self.layer_type:
                sk = si[1]
            else:
                sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f'so{i}'] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                in_data = [copy.deepcopy(dict(output=xs[0])), dict(output=xs[1])]
            else:
                in_data_q0 = in_quantizes[0].get_quan_data(copy.deepcopy(xs[0]))
                in_data_q1 = in_quantizes[1].get_quan_data(copy.deepcopy(xs[1]))
                in_data_q = [dict(output=in_data_q0), dict(output=in_data_q1)]
                in_data = copy.deepcopy(in_data_q)

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults.append(quantizes["feat"]["so0"].get_quan_data(ys[0])) # type: ignore
                qresults.append(out_data["output"])
            else:
                fresults.append(ys[0])
                qresults.append(out_data["output"])                    
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="matmul")
class ExportMatMul(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportMatMul, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.I_Align, self.model_export.O_Align]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        X0, X1 = xs
        return [np.matmul(X0, X1)]
    
    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx([-1 for _ in xs])
            self.layer.set_output_idx([-1])
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)
            
            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f'so{i}'] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            sk = dict(zip(["scale", "zero_point"], (1.0, 0.0)))
            
            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                in_data = [copy.deepcopy(dict(output=xs[0])), dict(output=xs[1])]
            else:
                in_data_q0 = in_quantizes[0].get_quan_data(copy.deepcopy(xs[0]))
                in_data_q1 = in_quantizes[1].get_quan_data(copy.deepcopy(xs[1]))
                in_data_q = [dict(output=in_data_q0), dict(output=in_data_q1)]
                in_data = copy.deepcopy(in_data_q)

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults.append(quantizes["feat"]["so0"].get_quan_data(ys[0])) # type: ignore
                qresults.append(out_data["output"])
            else:
                fresults.append(ys[0])
                qresults.append(out_data["output"])                    
            self.calc_error(fresults, qresults)
            
                
@EXPORT_LAYER.register_module(name="concat")
class ExportConcat(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportConcat, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        y = np.concatenate(xs, axis=1)
        ys = [y]

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx([-1 for _ in xs])
            self.layer.set_output_idx([-1])
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f'so{i}'] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                in_data = []
                for xs_ in xs:
                    in_data.append(dict(output=xs_))
            else:
                in_data_q = []
                for i, xs_ in enumerate(xs):
                    in_data_ = in_quantizes[i].get_quan_data(copy.deepcopy(xs_))
                    in_data_q.append(dict(output=in_data_))
                in_data = copy.deepcopy(in_data_q)

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            if not out_type in [np.float32, np.float64]:
                fresults.append(quantizes["feat"]["so0"].get_quan_data(ys[0])) # type: ignore
                qresults.append(out_data["output"])
            else:
                fresults.append(ys[0])
                qresults.append(out_data["output"])                    
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="split")
class ExportSplit(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportSplit, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        a = xs[0]
        ch = attrs["split"]
        axis = attrs["axis"]

        ys = []
        for y in torch.split(torch.from_numpy(a), ch, dim=axis):
            ys.append(self.to_numpy(y))

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx([-1])
            self.layer.set_output_idx([-1 for _ in ys])
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            self.layer.set_output_type(setting__["out_type"])
            ops_string = [self.layer_type]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f'so{i}'] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [attrs], 
                        'setting': setting,                         
                        }
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                in_data = [copy.deepcopy(dict(output=xs[0]))]
            else:
                in_data_q0 = in_quantizes[0].get_quan_data(copy.deepcopy(xs[0]))
                in_data_q = [dict(output=in_data_q0)]
                in_data = copy.deepcopy(in_data_q)

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            for i in range(len(out_data)):
                if not out_type in [np.float32, np.float64]:
                    fresults.append(quantizes["feat"][f"so{i}"].get_quan_data(ys[i])) # type: ignore
                    qresults.append(out_data[i]["output"])
                else:
                    fresults.append(ys[i])
                    qresults.append(out_data[i]["output"])                    
            self.calc_error(fresults, qresults)           


@EXPORT_LAYER.register_module(name="shuffle_only")
class ExportShuffleOnly(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportShuffleOnly, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        y = xs[0]

        shape1 = attrs["shape1"]
        perm = attrs["perm"]
        shape2 = attrs["shape2"]

        y = y.reshape(shape1)
        # ys = [copy.deepcopy(y)]

        y = y.transpose(perm)
        # ys.append(copy.deepcopy(y))

        y = y.reshape(shape2)
        # ys.append(copy.deepcopy(y))
        ys = [y]

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx(-1)
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)

            ops_string = ["reshape", "transpose", "reshape"]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)
            self.layer.set_in_quantize(in_quantizes)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [dict(), dict(), dict()], 
                        'setting': setting,                         
                        }
            settings["attrs"][0].update(dict(
                shape=attrs["shape1"],
            ))
            settings["attrs"][1].update(dict(
                perm=attrs["perm"],
            ))
            settings["attrs"][2].update(dict(
                shape=attrs["shape2"],
            ))
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                in_data = [copy.deepcopy(dict(output=xs[0]))]
            else:
                in_data_q = in_quantizes[0].get_quan_data(copy.deepcopy(xs[0]))
                in_data_q = dict(output=in_data_q)
                in_data = [copy.deepcopy(in_data_q)]

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            for i, _ in enumerate(out_data):
                if not out_type in [np.float32, np.float64]:
                    fresults.append(quantizes["feat"][f"so{i}"].get_quan_data(ys[i]))
                    qresults.append(out_data[i]["output"])
                else:
                    fresults.append(ys[i])
                    qresults.append(out_data[i]["output"])                    
            self.calc_error(fresults, qresults)


@EXPORT_LAYER.register_module(name="concat_shuffle_split")
@EXPORT_LAYER.register_module(name="shuffle")
class ExportShuffle(ExportLayer):
    def __init__(self, **kwargs):
        super(ExportShuffle, self).__init__(**kwargs)
        self.layer_type = kwargs["layer_type"]
        self.align_size = [self.model_export.Csize, self.model_export.Csize]
        self.export_w = None

    def get_float_result(self, attrs, xs, ws):
        a = xs[0]
        b = xs[1]

        shape1 = attrs["shape1"]
        perm = attrs["perm"]
        shape2 = attrs["shape2"]
        axis = attrs["axis"]
        ch = attrs["split"]

        y = np.concatenate([a, b], axis=axis)

        y = y.reshape(shape1)
        # ys = [copy.deepcopy(y)]

        y = y.transpose(perm)
        # ys.append(copy.deepcopy(y))

        y = y.reshape(shape2)
        # ys.append(copy.deepcopy(y))

        ys = [copy.deepcopy(y)]
        for y in torch.split(torch.from_numpy(y), ch, dim=axis):
            ys.append(self.to_numpy(y))

        return ys

    def run_export_layer(self, attrs, xs, ws):
        ys = self.get_float_result(attrs, xs, ws)

        setting__ = self.setting["setting"]
        bit_select = setting__["bit_select"]
        w_bit_select = bit_select
        bits_dict = {
            k: eval(v) for k, v in setting__["bits_dict"].items()
        }

        default_setting__ = setting__["default_setting"]
        in_type = setting__["in_type"]
        in_type = eval(setting__["bits_dict"][in_type])
        out_type = eval(setting__["bits_dict"][setting__["out_type"]])
        if self.layer_type in layer_factory.module_dict:
            process_scale = default_setting__[self.layer_type]['process_scale']

            self.layer = layer_factory.get(self.layer_type)() # type: ignore
            self.layer.set_layer_type(self.layer_type)
            self.layer.set_layer_name(self.layer_type + "_0")
            self.layer.set_idx(0)
            self.layer.set_input_idx([-1, -1])
            self.layer.set_output_idx(-1)
            self.layer.set_result_layer(False)
            self.layer.set_first_conv(False)
            self.layer.set_scale_type(process_scale)
            
            ops_string = ["concat", "reshape", "transpose", "reshape", "split"]

            si = []
            in_quantizes = []
            for in_data in xs:
                in_quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                in_quantize.get_quan_param(in_data)
                si_ = dict(zip(["scale", "zero_point"], in_quantize.get_scale()))
                si.append(si_)
                in_quantizes.append(in_quantize)

            sk = dict(zip(["scale", "zero_point"], [1.0, 0]))

            so = []
            quantizes = {}
            for i, out_data in enumerate(ys):
                quantize = quantize_factory.get( # type: ignore
                    default_setting__[self.layer_type]["feat"]["method"] # type: ignore
                )(bit_select) # type: ignore
                quantize.get_quan_param(out_data)
                so_ = dict(zip(["scale", "zero_point"], quantize.get_scale()))
                so.append(so_)
                quantizes[f"so{i}"] = quantize
            quantizes = dict(feat=quantizes)
            self.layer.set_quantize(quantizes)

            self.layer.set_in_scale(si)
            self.layer.set_w_scale(sk)
            self.layer.set_scale(so)

            setting = { 'w_bit_select': w_bit_select,
                        'maxs': setting__["maxs"], 
                        'mins': setting__["mins"],
                        'precision': setting__["precision"], 
                        'int_scale': setting__["int_scale"]}
            setting.update(default_setting__[self.layer_type]['feat'])
            ops_setting = dict(
                process_scale=process_scale,
                precision=default_setting__[self.layer_type]['precision'],
                int_scale=default_setting__[self.layer_type]['int_scale'],
                virtual_round=setting__["virtual_round"],
                txme_saturation=setting__["txme_saturation"],
                in_quantize=in_quantizes, quantize=quantizes,
                out_type=setting__["out_type"],)
            setting.update(ops_setting)
            setting.update(dict(bits_dict=bits_dict))
            settings = {'in_scale': si, 'w_scale': sk, 'scale': so,
                        'ops_string': ops_string, 
                        'attrs': [dict(), dict(), dict(), dict(), dict()], 
                        'setting': setting,                         
                        }
            settings["attrs"][0].update(dict(
                input_len=attrs["input_len"],
                axis=attrs["axis"],
                process_scale=self.setting["setting"]["process_scale"],
            ))
            settings["attrs"][1].update(dict(
                shape=attrs["shape1"],
                process_scale="smooth",
            ))
            settings["attrs"][2].update(dict(
                perm=attrs["perm"],
                process_scale="smooth",
            ))
            settings["attrs"][3].update(dict(
                shape=attrs["shape2"],
                process_scale="smooth",
            ))
            settings["attrs"][4].update(dict(
                axis=attrs["axis"],
                split=attrs["split"],
                process_scale=self.setting["setting"]["process_scale"],
            ))
            self.layer.setting_ops(settings)
            self.layer.set_ops_setting(settings)
            self.layer.set_layer_ops(dict(ops=ops_string, attrs=[attrs]))

            if in_type in [np.float32, np.float64]:
                in_data = copy.deepcopy(xs)
            else:
                in_data_q0 = in_quantizes[0].get_quan_data(copy.deepcopy(xs[0]))
                in_data_q1 = in_quantizes[1].get_quan_data(copy.deepcopy(xs[1]))
                in_data_q = [dict(output=in_data_q0), dict(output=in_data_q1)]
                in_data = copy.deepcopy(in_data_q)

            self.layer.forward(in_data)

            out_data = self.layer.get_out_data()

            fresults, qresults = [], []
            for i, _ in enumerate(out_data):
                if i == 0: continue
                if not out_type in [np.float32, np.float64]:
                    fresults.append(quantizes["feat"][f"so{i}"].get_quan_data(ys[i]))
                    qresults.append(out_data[i]["output"])
                else:
                    fresults.append(ys[i])
                    qresults.append(out_data[i]["output"])                    
            self.calc_error(fresults, qresults)
