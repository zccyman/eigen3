# Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/07/05 15:01:59
# @File    : test_single_layer_export.py
import sys

sys.path.append("./")  # NOQA: E402

import os
import shutil

import pytest
from test_layer_export import find_element, GenerateAttrsCombination

try:
    from utest import SingleLayerBase
    from utest import parse_config
except:
    from onnx_converter.utest import SingleLayerBase # type: ignore
    from onnx_converter.utest import parse_config # type: ignore


cfg_file = "tests/export_tests/arguments_single_layer.json"
args = parse_config(cfg_file)

gac = GenerateAttrsCombination(
    input_settings_combination="tests/export_tests/input_settings_combination.py",
    use_input_settings_combination=args.use_input_settings_combination,
    valid_layer_types=args.valid_layer_types,
)

quantize_method_process_scale_layer_type_in_type_out_type_list = []
for process_scales, layer_types in args.process_scale.items(): # type: ignore
    for process_scale in process_scales.split("/"):
        for layer_type in layer_types:
            for in_type, in_type_process_scale in args.in_types.items(): # type: ignore
                if find_element(process_scale, in_type_process_scale):
                    for out_type, out_type_process_scale in args.out_types.items(): # type: ignore
                        if find_element(process_scale, out_type_process_scale):
                            for (
                                quantize_method,
                                quantize_method_layer_types,
                            ) in args.quantize_methods.items(): # type: ignore
                                if find_element(
                                    layer_type, quantize_method_layer_types
                                ):
                                    s = quantize_method + "/"
                                    s += process_scale + "/"
                                    s += layer_type + "/"
                                    s += in_type + "/"
                                    s += out_type
                                    quantize_method_process_scale_layer_type_in_type_out_type_list.append(
                                        s
                                    )


@pytest.fixture(scope="class", params=[str(data) for data in args.quantize_dtypes]) # type: ignore
def quantize_dtype(request):
    return request.param


@pytest.fixture(
    scope="class",
    params=quantize_method_process_scale_layer_type_in_type_out_type_list,
)
def quantize_method_process_scale_layer_type_in_type_out_type(request):
    return request.param


@pytest.fixture(scope="class", params=args.chip_types)
def chip_type(request):
    return request.param


@pytest.fixture(scope="class", params=gac.run())
def input_settings_combination(request):
    return request.param


@pytest.mark.usefixtures(
    "quantize_dtype",
    "quantize_method_process_scale_layer_type_in_type_out_type",
    "input_settings_combination",
    "chip_type",
)
class TestLayers(SingleLayerBase): # type: ignore
    is_export_model_c = False
    def setup_class(self):
        self.input_settings = parse_config("tests/export_tests/input_settings.py")
        self.use_input_settings = args.use_input_settings
        self.use_input_settings_combination = args.use_input_settings_combination
        self.weights_dir = args.weights_dir
        self.chips_only_support_model_c = args.chips_only_support_model_c
        self.virtual_round = args.virtual_round
        self.valid_layer_types = args.valid_layer_types
        self.layer_type = args.valid_layer_types[0] # type: ignore
        self.contents = []
        self.is_acc_woffset = True
        self.mode = "ab"
        if os.path.exists(self.weights_dir): # type: ignore
            shutil.rmtree(self.weights_dir) # type: ignore

    def test_layer(
        self,
        quantize_dtype,
        quantize_method_process_scale_layer_type_in_type_out_type,
        input_settings_combination,
        chip_type,
    ):
        self.__class__.is_export_model_c = True if chip_type in self.chips_only_support_model_c else False
        self.export_version = 2 if chip_type in self.chips_only_support_model_c else 3    
        self.run(
            quantize_dtype,
            quantize_method_process_scale_layer_type_in_type_out_type,
            input_settings_combination,
            chip_type,
        )


if __name__ == "__main__":
    pytest.main(["tests/export_tests/test_single_layer_export.py::TestLayers::test_layer"])
