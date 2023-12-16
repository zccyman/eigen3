# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/2/16 10:28
# @File     : test_process.py
import onnx

from checkpoint import OnnxProcess
from checkpoint.preprocess import ir_version, opset_version, regular_operation_pipeline


def demo():
    regular_tasks = [
        dict(
            model_path='/home/ts300026/workspace/trained_models/voice/model_p1_sim.onnx',
            save_path='/home/ts300026/workspace/trained_models/voice/model_p1_adj_for_test.onnx',
            node_operations=regular_operation_pipeline
        ),
        dict(
            model_path='/home/ts300026/workspace/trained_models/voice/model_p2_sim.onnx',
            save_path='/home/ts300026/workspace/trained_models/voice/model_p2_adj_for_test.onnx',
            node_operations=regular_operation_pipeline
        ),
        dict(
            model_path='/home/ts300026/workspace/trained_models/imagenet/ResNeXt101_32x4d_ImageNet_classification-sim.onnx',
            save_path='/home/ts300026/workspace/trained_models/imagenet/ResNeXt101_32x4d_ImageNet_classification-adj_for_test.onnx',
            node_operations=regular_operation_pipeline
        ),
        dict(
            model_path='/home/ts300026/workspace/trained_models/imagenet/MobileNetv3_ImageNet_classification_raw_sim.onnx',
            save_path='/home/ts300026/workspace/trained_models/imagenet/MobileNetv3_ImageNet_classification_adj_for_test.onnx',
            node_operations=regular_operation_pipeline
        ),
    ]

    manual_tasks = [
        dict(
            model_path='/home/ts300026/workspace/trained_models/voice/model_p1_adj_for_test.onnx',
            save_path='/home/ts300026/workspace/trained_models/voice/model_p1_simplify_for_test.onnx',
            node_operations=[
                {"method": "DeleteOpsByName", 'delete_node_names': ['Reshape_6_2', 'Reshape_11', 'Reshape_13']},
            ]
        ),

        dict(model_path='/home/ts300026/workspace/trained_models/voice/model_p2_adj_for_test.onnx',
             save_path='/home/ts300026/workspace/trained_models/voice/model_p2_simplify_for_test.onnx',
             node_operations=[
                 {"method": "DeleteOpsByName",
                  'delete_node_names': ['Reshape_0_0', 'Reshape_9_2', 'Reshape_23', 'Reshape_25']},
             ]
             )
    ]

    # Regular modification
    for task in regular_tasks:
        model_path = task['model_path']
        save_path = task['save_path']
        operations = task['node_operations']
        model = onnx.load(model_path)
        mengine = OnnxProcess(model=model, node_operation_pipeline=operations, ir_version=ir_version,
                              opset_version=opset_version)
        model_simplified = mengine.process()
        mengine.check()
        mengine.save_model(save_path)  # optional

    # Manual modification
    for task in manual_tasks:
        model_path = task['model_path']
        save_path = task['save_path']
        operations = task['node_operations']
        model = onnx.load(model_path)
        mengine = OnnxProcess(model=model, node_operation_pipeline=operations, ir_version=ir_version,
                              opset_version=opset_version)
        model_simplified = mengine.process()
        mengine.check()
        mengine.save_model(save_path)  # optional

if __name__=='__main__':
    demo()
