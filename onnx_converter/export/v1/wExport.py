# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/07/08 18:09:23
# @File    : wExport.py
import copy
import os
import shutil
from typing import List

# import encrypt
import numpy as np

try:
    from utils import Object, Registry, to_bytes, export_perchannel, invert_dict
except Exception:
    from onnx_converter.utils import Object, Registry, to_bytes, invert_dict, export_perchannel # type: ignore

from .serialize import SERIALIZE as serialize_factory

wExport: Registry = Registry("weight_export", scope="")


class WeightExport(Object): # type: ignore
    def __init__(self, **kwargs):
        super(WeightExport, self).__init__(**kwargs)

        self.export_version = kwargs["export_version"]
        self.bits = kwargs["bits"]
        self.Csize = self.bits["Csize"]
        self.Ksize = self.bits["Ksize"]
        self.I_Align = self.bits["I_Align"]
        self.O_Align = self.bits["O_Align"]
        self.ABGR = self.bits["ABGR"]
        self.data_channel_extension = self.bits["DATA_C_EXTEND"]
        self.bgr_format = self.bits["bgr_format"]
        self.chip_type = self.bits["chip_type"]
        algin_bias_dict = {"fc_bias_align": self.O_Align, "conv_bias_align": self.Ksize}
        self.bits.update(algin_bias_dict)
        for m in kwargs["serialize_wlist"]:
            setattr(
                self, "serialize_{}".format(m), serialize_factory.get(m)(**self.bits) # type: ignore
            )
        self.is_stdout = kwargs["is_stdout"]
        self.log_name = (
            kwargs["log_name"] if "log_name" in kwargs.keys() else "export.log"
        )
        self.log_level = kwargs.get('log_level', 20)
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)
        self.weights_dir = os.path.join(self.log_dir, "weights")
        # if os.path.exists(self.weights_dir):
        # self.logger.info("delete directory: {}".format(self.weights_dir))
        # shutil.rmtree(self.weights_dir)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir, mode=0o777, exist_ok=True)

    @staticmethod
    def calc_w_offset(weight):
        """
        The data type of weight is float32 or int8.
        """
        b_size = 4 if isinstance(weight[0], np.float32) else 1 # type: ignore
        w_offset = len(weight) * b_size
        return w_offset

    def __call__(self, layer, save_weights, w_offset):
        pass


@wExport.register_module(name="conv")
@wExport.register_module(name="depthwiseconv")
@wExport.register_module(name="convtranspose")
class WeightExportConv(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportConv, self).__init__(**kwargs)

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))

        if len(w_offset["tmp_offset"]) > 0:
            w_offset["tmp_offset"] = []

        w_offset["tmp_offset"].append(-1)

        # export conv weights and bias into weight.b
        layer_type = layer.get_layer_type()
        first_conv = layer.get_first_conv()
        func_weight = getattr(self, "serialize_{}".format(layer_type))
        func_bias = getattr(self, "serialize_{}".format("bias"))
        layer.set_data_channel_extension(self.data_channel_extension)
        qweight = layer.get_qweight()
        if not self.bgr_format and first_conv:
            qweight = qweight[:, [2, 1, 0], :, :]
        res = layer.weight_export(
            func_weight,
            qweight,
            chip_type=self.chip_type,
            layer=layer,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "weight.b"),
            first_conv=first_conv,
            data_channel_extension=layer.get_data_channel_extension(),
        )
        save_weights.extend(res)
        w_offset["tmp_offset"].append(w_offset["w_offset"])
        w_offset["w_offset"] += self.calc_w_offset(res)

        res = layer.bias_export(
            func_bias,
            layer.get_qbias(),
            layer=layer,
            is_fc_bias=False,
            name=os.path.join(self.weights_dir, "weight.b"),
        )
        save_weights.extend(res)
        w_offset["w_offset"] += self.calc_w_offset(res)

        w_offset, save_weights = export_perchannel(
            layer,
            self.weights_dir,
            func_bias,
            save_weights,
            w_offset,
            self.calc_w_offset,
            is_fc_bias=False,
        )
        
        if layer.get_scale_type() in ["shiftfloatscaletable", "shiftfloatscaletable2float"]:
            res_list = []
            table = layer.get_ops_instance()[-1].table[:, 0]
            for tb in np.array_split(table, 2)[::-1]:
                res = to_bytes(tb, dtype=table[0].dtype)
                res_list.extend(res)
            save_weights.extend(res_list)
            w_offset["tmp_offset"][-1] = w_offset["w_offset"]
            w_offset["w_offset"] += self.calc_w_offset(res_list)
            layer.set_w_offset(copy.deepcopy(w_offset))
                        
        # print('test')


@wExport.register_module(name="fc")
class WeightExportFc(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportFc, self).__init__(**kwargs)
        self.version = kwargs.get("version", None)
        
    @staticmethod
    def export_lut(table, is_byte=False):
        if is_byte:
            res = to_bytes(table, dtype=table[0].dtype)
        else:
            res = table.flatten().view(np.int8)
        
        return res

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))

        if len(w_offset["tmp_offset"]) > 0:
            w_offset["tmp_offset"] = []

        w_offset["tmp_offset"].append(-1)

        # export fc weights and bias into weight.b
        layer_type = layer.get_layer_type()
        func_weight = getattr(self, "serialize_{}".format(layer_type))
        func_bias = getattr(self, "serialize_{}".format("bias"))
        res = layer.weight_export(
            func_weight,
            layer.get_qweight(),
            layer=layer,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "weight.b"),
        )
        save_weights.extend(res)
        w_offset["tmp_offset"].append(w_offset["w_offset"])
        w_offset["w_offset"] += self.calc_w_offset(res)

        res = layer.bias_export(
            func_bias,
            layer.get_qbias(),
            layer=layer,
            is_fc_bias=True,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "weight.b"),
        )
        save_weights.extend(res)
        w_offset["w_offset"] += self.calc_w_offset(res)

        w_offset, save_weights = export_perchannel(
            layer,
            self.weights_dir,
            func_bias,
            save_weights,
            w_offset,
            self.calc_w_offset,
            is_fc_bias=True,
        )
        
        if layer.get_scale_type() in ["shiftfloatscaletable", "shiftfloatscaletable2float"]:
            res_list = []
            if len(layer.get_ops_instance()) > 3:
                table = layer.get_ops_instance()[-2].table[:, 0]
            else:
                table = layer.get_ops_instance()[-1].table[:, 0]
            is_byte = False if self.version in ["v1", "v2"] else True
            if layer.get_scale_type() in ["shiftfloatscaletable", "shiftfloatscaletable2float"]:
                is_byte = False
            for tb in np.array_split(table, 2)[::-1]:
                res = self.export_lut(tb, is_byte=is_byte)
                res_list.extend(res)
            save_weights.extend(res_list)
            w_offset["tmp_offset"][-1] = w_offset["w_offset"]
            w_offset["w_offset"] += self.calc_w_offset(res_list)
            layer.set_w_offset(copy.deepcopy(w_offset))
                        
        # print('test')


@wExport.register_module(name="lstm")
class WeightExportLstm(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportLstm, self).__init__(**kwargs)

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))
        tmp_offset = []

        layer_type = layer.get_layer_type()
        func_table = getattr(self, "serialize_{}".format("table"))
        func_weight = getattr(self, "serialize_{}".format("fc"))
        func_bias = getattr(self, "serialize_{}".format("bias"))
        for i in range(6):            
            tables = layer.get_table()
            if i < len(tables) and tables != []:
                table = tables[i]
                res = layer.weight_export(
                    func_table,
                    table,
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, "weight.b"),
                )
                save_weights.extend(res)
                tmp_offset.append(w_offset["w_offset"])
                w_offset["w_offset"] += self.calc_w_offset(res)
            else:
                tmp_offset.append(-1)

        same_weights, same_bias = [], []
        same_inserts = dict()

        for idx, (qweight, qbias) in enumerate(
            zip(layer.get_qweight(), layer.get_qbias())
        ):
            insert = {
                key: [layer.get_insert()["split"][key][idx + 1]]
                for key in ["in_pad", "in_align", "out_pad", "out_align"]
            }
            if not (layer.get_hx_combine() and layer.get_wr_combine()):
                res = layer.weight_export(
                    func_weight,
                    np.squeeze(qweight),
                    insert=insert,
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, "weight.b"),
                )
                save_weights.extend(res)
                tmp_offset.append(w_offset["w_offset"])
                w_offset["w_offset"] += self.calc_w_offset(res)

                res = layer.bias_export(
                    func_bias,
                    np.squeeze(qbias),
                    is_fc_bias=True,
                    insert=insert,
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, "weight.b"),
                )
                save_weights.extend(res)
                w_offset["w_offset"] += self.calc_w_offset(res)
            else:
                same_weights.append(qweight)
                same_bias.append(qbias)
                if same_inserts == dict():
                    same_inserts = copy.deepcopy(insert)
                else:
                    same_inserts['in_pad'].extend(insert['in_pad'])
                    same_inserts['in_align'].extend(insert['in_align'])

        if layer.get_hx_combine() and layer.get_wr_combine():
            weight_combine = []#np.concatenate(same_weights, axis=-1)
            bias_combine = []#same_bias[0] + same_bias[1]
            for shape, weight, bias in zip(same_inserts["in_align"], same_weights, same_bias):
                align_out_c = same_inserts["out_align"][-1]
                out_c, in_c = weight.shape[1:]
                new_w = np.zeros((align_out_c, shape), dtype=weight.dtype)
                new_b = np.zeros(align_out_c, dtype=bias.dtype)
                new_w[:out_c, :in_c] = np.squeeze(weight)
                new_b[:out_c] = np.squeeze(bias)
                weight_combine.append(new_w)
                bias_combine.append(new_b)

            weight_combine = np.concatenate(weight_combine, axis=-1)
            bias_combine = bias_combine[0] + bias_combine[1]
            same_inserts["in_pad"] = [[0, weight_combine.shape[-1]]]
            same_inserts["in_align"] = [weight_combine.shape[-1]]
            res = layer.weight_export(
                    func_weight,
                    np.squeeze(weight_combine),
                    insert=same_inserts,
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, "weight.b"),
                )
            save_weights.extend(res)
            tmp_offset.append(w_offset["w_offset"])
            w_offset["w_offset"] += self.calc_w_offset(res)
            res = layer.bias_export(
                    func_bias,
                    np.squeeze(bias_combine),
                    is_fc_bias=True,
                    insert=same_inserts,
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, "weight.b"),
                )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)
            tmp_offset.append(w_offset["w_offset"])
            # tmp_offset.append(-1)


        ### write wb_off, rb_off
        tmp_offset.append(-1)
        tmp_offset.append(-1)

        ### write init_h
        init_h = layer.get_init_h()
        if np.sum(np.abs(init_h)) > 0:
            insert = {
                key1: [layer.get_insert()["split"][key][2]]
                for key1, key in zip(["out_pad", "out_align"], ["in_pad", "in_align"])
            }
            res = layer.bias_export(
                func_bias,
                np.squeeze(init_h),
                is_fc_bias=True,
                insert=insert,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            tmp_offset.append(w_offset["w_offset"])
            w_offset["w_offset"] += self.calc_w_offset(res)
        else:
            tmp_offset.append(-1)

        ### write init_c
        init_c = layer.get_init_c()
        if np.sum(np.abs(init_c)) > 0:
            insert = {
                key1: [layer.get_insert()["split"][key][2]]
                for key1, key in zip(["out_pad", "out_align"], ["in_pad", "in_align"])
            }
            res = layer.bias_export(
                func_bias,
                np.squeeze(init_c),
                is_fc_bias=True,
                insert=insert,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            tmp_offset.append(w_offset["w_offset"])
            w_offset["w_offset"] += self.calc_w_offset(res)
        else:
            tmp_offset.append(-1)

        w_offset = layer.get_w_offset()
        w_offset["tmp_offset"] = tmp_offset
        layer.set_w_offset(copy.deepcopy(w_offset))


@wExport.register_module(name="gru")
class WeightExportGru(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportGru, self).__init__(**kwargs)

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))
        tmp_offset = []

        layer_type = layer.get_layer_type()
        func_table = getattr(self, "serialize_{}".format("table"))
        func_weight = getattr(self, "serialize_{}".format("fc"))
        func_bias = getattr(self, "serialize_{}".format("bias"))
        for i in range(6):
            tables = layer.get_table()
            if i < len(tables):
                table = tables[i]
                res = layer.weight_export(
                    func_table,
                    table,
                    layer_name=layer.get_layer_name(),
                    name=os.path.join(self.weights_dir, "weight.b"),
                )
                save_weights.extend(res)
                tmp_offset.append(w_offset["w_offset"])
                w_offset["w_offset"] += self.calc_w_offset(res)
            else:
                tmp_offset.append(-1)

        for idx, (qweight, qbias) in enumerate(
            zip(layer.get_qweight(), layer.get_qbias())
        ):
            insert = {
                key: [layer.get_insert()["split"][key][idx]]
                for key in ["in_pad", "in_align", "out_pad", "out_align"]
            }
            res = layer.weight_export(
                func_weight,
                np.squeeze(qweight),
                insert=insert,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            tmp_offset.append(w_offset["w_offset"])
            w_offset["w_offset"] += self.calc_w_offset(res)

            res = layer.bias_export(
                func_bias,
                np.squeeze(qbias),
                is_fc_bias=True,
                insert=insert,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)

        ### write wb_off, rb_off
        tmp_offset.append(-1)
        tmp_offset.append(-1)

        ### write init_h
        init_h = layer.get_init_h()
        if np.sum(np.abs(init_h)) > 0:
            insert = {
                key1: [layer.get_insert()["split"][key][1]]
                for key1, key in zip(["out_pad", "out_align"], ["in_pad", "in_align"])
            }
            res = layer.bias_export(
                func_bias,
                np.squeeze(init_h),
                is_fc_bias=True,
                insert=insert,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            tmp_offset.append(w_offset["w_offset"])
            w_offset["w_offset"] += self.calc_w_offset(res)
        else:
            tmp_offset.append(-1)

        w_offset = layer.get_w_offset()
        w_offset["tmp_offset"] = tmp_offset
        layer.set_w_offset(copy.deepcopy(w_offset))


@wExport.register_module(name="table")
class WeightExportTable(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportTable, self).__init__(**kwargs)

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))
        if layer.get_scale_type() == "table":
            func_table = getattr(self, "serialize_{}".format("table"))
            res = layer.table_export(
                func_table,
                layer.get_table(),
                is_fc_bias=False,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)


@wExport.register_module(name="leakyrelu")
@wExport.register_module(name="prelu")
@wExport.register_module(name="sigmoid")
@wExport.register_module(name="swish")
@wExport.register_module(name="tanh")
@wExport.register_module(name="hardsigmoid")
@wExport.register_module(name="hardtanh")
@wExport.register_module(name="hardswish")
@wExport.register_module(name="hardshrink")
class WeightExportAct(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportAct, self).__init__(**kwargs)

        self.export_table = wExport.get("table")(**kwargs) # type: ignore

    def __call__(self, layer, save_weights, w_offset):
        self.export_table(layer, save_weights, w_offset)


@wExport.register_module(name="batchnormalization")
class WeightExportBn(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportBn, self).__init__(**kwargs)

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))

        layer_type = layer.get_layer_type()
        weights = layer.get_layer_ops()["attrs"][0]
        func_bn = getattr(self, "serialize_{}".format("bias"))
        res = layer.bias_export(
            func_bn,
            weights["scale"],
            is_fc_bias=False,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "weight.b"),
        )
        save_weights.extend(res)
        w_offset["w_offset"] += self.calc_w_offset(res)
        res = layer.bias_export(
            func_bn,
            weights["bias"],
            is_fc_bias=False,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "weight.b"),
        )
        save_weights.extend(res)
        w_offset["w_offset"] += self.calc_w_offset(res)
        res = layer.bias_export(
            func_bn,
            weights["mean"],
            is_fc_bias=False,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "weight.b"),
        )
        save_weights.extend(res)
        w_offset["w_offset"] += self.calc_w_offset(res)
        res = layer.bias_export(
            func_bn,
            weights["var"],
            is_fc_bias=False,
            layer_name=layer.get_layer_name(),
            name=os.path.join(self.weights_dir, "weight.b"),
        )
        save_weights.extend(res)
        w_offset["w_offset"] += self.calc_w_offset(res)


@wExport.register_module(name="layernormalization")
class WeightExportLn(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportLn, self).__init__(**kwargs)
        self.is_voice_model = True

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))

        layer_type = layer.get_layer_type()
        weights = layer.get_layer_ops()["attrs"][0]
        if self.is_voice_model:
            func_bn = getattr(self, "serialize_{}".format("bias"))
            qscale = np.squeeze(weights["scale"])
            qbias = np.squeeze(weights["bias"])
            res = layer.bias_export(
                func_bn,
                qscale,
                is_fc_bias=False,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)
            res = layer.bias_export(
                func_bn,
                qbias,
                is_fc_bias=False,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)
        else:
            func_data = getattr(self, "serialize_{}".format("data"))
            qscale = np.expand_dims(weights["scale"], axis=0)
            qbias = np.expand_dims(weights["bias"], axis=0)
            res = layer.feat_export(
                func_data,
                qscale,
                is_out=True,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)
            res = layer.feat_export(
                func_data,
                qbias,
                is_out=True,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)


@wExport.register_module(name="instancenormalization")
class WeightExportIn(WeightExport):
    def __init__(self, **kwargs):
        super(WeightExportIn, self).__init__(**kwargs)
        self.is_voice_model = True

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))

        layer_type = layer.get_layer_type()
        weights = layer.get_layer_ops()["attrs"][0]
        if self.is_voice_model:
            func_bn = getattr(self, "serialize_{}".format("bias"))
            qscale = np.squeeze(weights["scale"])
            qbias = np.squeeze(weights["bias"])
            res = layer.bias_export(
                func_bn,
                qscale,
                is_fc_bias=False,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)
            res = layer.bias_export(
                func_bn,
                qbias,
                is_fc_bias=False,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)
        else:
            func_data = getattr(self, "serialize_{}".format("data"))
            qscale = np.expand_dims(weights["scale"], axis=0)
            qbias = np.expand_dims(weights["bias"], axis=0)
            res = layer.feat_export(
                func_data,
                qscale,
                is_out=True,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)
            res = layer.feat_export(
                func_data,
                qbias,
                is_out=True,
                layer_name=layer.get_layer_name(),
                name=os.path.join(self.weights_dir, "weight.b"),
            )
            save_weights.extend(res)
            w_offset["w_offset"] += self.calc_w_offset(res)


@wExport.register_module(name="relu")
@wExport.register_module(name="relu6")
@wExport.register_module(name="relux")
class NoWeightExport(WeightExport):
    def __init__(self, **kwargs):
        super(NoWeightExport, self).__init__(**kwargs)

    def __call__(self, layer, save_weights, w_offset):
        layer.set_w_offset(copy.deepcopy(w_offset))
        tmp_offset = [-1, -1, -1]

        w_offset = layer.get_w_offset()
        w_offset["tmp_offset"] = tmp_offset
        layer.set_w_offset(copy.deepcopy(w_offset))
