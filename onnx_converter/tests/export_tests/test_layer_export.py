# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/07/05 15:01:46
# @File    : test_layer_export.py
import sys  # NOQA: E402

sys.path.append("./")  # NOQA: E402


import json
import copy

import pytest

try:
    from utest import LayerBase, parse_config, generate_all_combination_on_layer_attrs
except:
    from onnx_converter.utest import LayerBase, parse_config, generate_all_combination_on_layer_attrs # type: ignore


def save_json(arguments, json_file):
    contents = json.dumps(arguments, sort_keys=False, indent=4)
    with open(json_file, "w") as f:
        f.write(contents)


def find_element(a, b_list):
    flag = any(b == a for b in b_list)
    return flag


class GenerateAttrsCombination(object):
    def __init__(self, **kwargs):
        self.input_settings_combination = copy.deepcopy(
            kwargs["input_settings_combination"])
        self.input_settings_combination = parse_config(
            self.input_settings_combination)
        self.use_input_settings_combination = kwargs[
            "use_input_settings_combination"]
        self.valid_layer_types = kwargs["valid_layer_types"]

    def run(self):
        gac_cases = []  # default: [dict(conv_0=dict()), dict(conv_1=dict())]
        for layer_type in self.valid_layer_types:
            if self.use_input_settings_combination:
                kwargs = dict(
                    layer_type=layer_type,
                    layer_attrs_combination=self.
                    input_settings_combination[layer_type],
                )
                gac_case = generate_all_combination_on_layer_attrs(kwargs)
            else:
                gac_case = {f"{layer_type}_0": {}}
            gac_cases.extend(gac_case)

        return gac_cases


class TestLayer(LayerBase): # type: ignore
    def setup_class(self):
        self.input_settings = parse_config("tests/export_tests/input_settings.py")
        self.use_input_settings = True
        self.chips_only_support_model_c = ["AT1K"]
        
    def test_export_conv(self):
        layer_type = "conv"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"  # int8 | int16
        in_type = "int8"  # int8 | int16 |
        out_type = "int8"  # int8 | int16 |
        process_scale = "intscale"  # intscale | floatscale | shiftfloatscale | shiftfloatscaletable
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_depthwiseconv(self):
        layer_type = "depthwiseconv"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"  # int8 | int16
        in_type = "int8"  # int8 | int16 |
        out_type = "int8"  # int8 | int16 |
        process_scale = "intscale"  # intscale | floatscale | shiftfloatscale
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_convtranspose(self):
        layer_type = "convtranspose"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"  # int8 | int16
        in_type = "int8"  # int8 | int16 |
        out_type = "int8"  # int8 | int16 |
        process_scale = "intscale"  # intscale | floatscale | shiftfloatscale
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )
                
    def test_export_fc(self):
        layer_type = "fc"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"  # int8 | int16
        in_type = "float32"  # int8 | int16 |
        out_type = "float32"  # int8 | int16 |
        process_scale = "shiftfloatscaletable2float"  # intscale | floatscale | shiftfloatscale
        chip_type = "AT1K"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_matmul(self):
        layer_type = "matmul"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"  # int8 | int16
        in_type = "int8"  # int8 | int16 |
        out_type = "int8"  # int8 | int16 |
        process_scale = "intscale"  # intscale | floatscale | shiftfloatscale
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_act(self):
        # [ "leakyrelu", "prelu", "gelu", "sigmoid", "swish", "tanh", "hardsigmoid", "hardswish", "hardtanh", "hardshrink" ]-> table | float
        # ["relu", "relux", "relu6"] -> preintscale
        layer_type = "swish"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "table"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_averagepool(self):
        layer_type = "averagepool"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "smooth"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_maxpool(self):
        layer_type = "maxpool"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_shuffle_only(self):
        layer_type = "shuffle_only"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "smooth"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_concat(self):
        layer_type = "concat"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "preintscale"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_split(self):
        layer_type = "split"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "preintscale"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_shuffle(self):
        layer_type = "shuffle"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int16"
        in_type = "int16"
        out_type = "int16"
        process_scale = "preintscale"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_add(self):
        layer_type = "add"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "float32"
        out_type = "float32"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_cadd(self):
        layer_type = "cadd"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "float32"
        out_type = "float32"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_sub(self):
        layer_type = "sub"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "preintscale"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_csub(self):
        layer_type = "csub"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "preintscale"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_pmul(self):
        layer_type = "pmul"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "intscale"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_cmul(self):
        layer_type = "cmul"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "float"  # intscale
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_layernormalization(self):
        layer_type = "layernormalization"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_batchnormalization(self):
        layer_type = "batchnormalization"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_instancenormalization(self):
        layer_type = "instancenormalization"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_softmax(self):
        layer_type = "softmax"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_resize(self):
        layer_type = "resize"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_lstm(self):
        layer_type = "lstm"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "float32"
        out_type = "float32"
        process_scale = "ffloatscale"
        chip_type = "AT1K"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )

    def test_export_gru(self):
        layer_type = "gru"
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "float32"
        out_type = "float32"
        process_scale = "ffloatscale"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )


    def test_export_reduce(self):
        layer_type = "reducemax" # ["reducemax", "reducemin", "reducemean", "reducesum", "reduceprod"]
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "float"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )
        
        
    def test_export_transpose(self):
        layer_type = "transpose" 
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "smooth"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )
        
           
    def test_export_reshape(self):
        layer_type = "reshape" 
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "smooth"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )
        
        
    def test_export_log(self):
        layer_type = "log" 
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "float32"
        out_type = "float32"
        process_scale = "float"
        chip_type = "AT1K"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )
        
                
    def test_export_pad(self):
        layer_type = "pad" 
        feature = "sym"
        weight = ["sym", "pertensor"]
        quantize_dtype = "int8"
        in_type = "int8"
        out_type = "int8"
        process_scale = "smooth"
        chip_type = "AT5050"
        export_version = 2 if chip_type in self.chips_only_support_model_c else 3
        is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        virtual_round = 3
        weights_dir = "work_dir/test_layer"
        
        params = self.input_settings[layer_type] if self.use_input_settings else dict()

        self.run(
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
        )
           
                                            
if __name__ == "__main__":
    pytest.main(["tests/export_tests/test_layer_export.py::TestLayer::test_export_pad"])
    # pytest.main(["tests/export_tests/test_layer_export.py::TestLayer::test_export_fc"])
    # pytest.main(["tests/export_tests/test_layer_export.py::TestLayer::test_export_convtranspose"])    
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_depthwiseconv'])
    # pytest.main(["tests/export_tests/test_layer_export.py::TestLayer::test_export_fc"])
    # pytest.main(
    # ['tests/export_tests/test_layer_export.py::TestLayer::test_export_lstm'])
    # pytest.main(
    # ['tests/export_tests/test_layer_export.py::TestLayer::test_export_gru'])    
    # pytest.main(
    # ['tests/export_tests/test_layer_export.py::TestLayer::test_export_batchnormalization'])
    # pytest.main(
    # ['tests/export_tests/test_layer_export.py::TestLayer::test_export_layernormalization'])
    # pytest.main(
    #         ['tests/export_tests/test_layer_export.py::TestLayer::test_export_instancenormalization'])
    # pytest.main(
    #             ['tests/export_tests/test_layer_export.py::TestLayer::test_export_softmax'])    
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_resize'])
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_shuffle_only'])
    # pytest.main(["tests/export_tests/test_layer_export.py::TestLayer::test_export_shuffle"])
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_add'])
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_cadd'])
    # pytest.main(
    # ['tests/export_tests/test_layer_export.py::TestLayer::test_export_sub'])
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_csub'])
    # pytest.main(["tests/export_tests/test_layer_export.py::TestLayer::test_export_pmul"])
    # pytest.main(["tests/export_tests/test_layer_export.py::TestLayer::test_export_act"])
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_cmul'])
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_maxpool'])
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_averagepool'])
    # pytest.main(
    #     ['tests/export_tests/test_layer_export.py::TestLayer::test_export_split'])
