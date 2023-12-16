# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/03/29 19:40:44
# @File    : test_face_recognition.py
import sys

sys.path.append('./')  # NOQA: E402

import json
import os
import unittest

import numpy as np
import pytest
from benchmark import (collect_accuracy, parse_config, save_config,
                       save_export, save_tables)
from benchmark.tests.test_base import TestBasse
from eval.face_recognition import RecEval
try:    
    from utils import Registry
except:
    from onnx_converter.utils import Registry

BENCHMARK: Registry = Registry('benchmark', scope='')

cfg_file = 'benchmark/benchmark_config/face_recognition.py'
args = parse_config(cfg_file)

first_model_name = ''
model_name_list = []
for model_type, model_paths in args.model_paths.items():
    model_rootdir_list = []
    for model_id, model_path in enumerate(model_paths):
        model_rootdir, model_name = os.path.split(model_path)
        model_name_list.append(model_type + '/' + str(model_id) + '/' + model_name)
        if first_model_name == '':
            first_model_name = model_name

quantize_method_list = []
for quantize_method_f, quantize_method_weights in args.quantize_methods.items():
    for quantize_method_w in quantize_method_weights:
        quantize_method_list.append(quantize_method_f + '/' + quantize_method_w)

process_scale_w_list = args.process_scale['weight']

params = model_name_list


@pytest.fixture(scope='class', params=params)
def model_paths(request):
    return request.param


params = [str(data) for data in args.quantize_dtypes]


@pytest.fixture(scope='class', params=params)
def quantize_dtype(request):
    return request.param


### feat/weight quant method
params = quantize_method_list


@pytest.fixture(scope='class', params=params)
def quantize_method(request):
    return request.param


params = process_scale_w_list


@pytest.fixture(scope='class', params=params)
def process_scale_w(request):
    return request.param


@pytest.mark.usefixtures('model_paths', 'quantize_dtype', 'process_scale_w', 'quantize_method')
class TestFaceRecognition(TestBasse):

    def compose_evaluator(self, **kwargs):
        args = kwargs['args']
        model_dir = kwargs['model_dir']
        model_name = kwargs['model_name']
        dataset_dir = kwargs['dataset_dir']
        selected_mode = kwargs['selected_mode']
        quantize_dtype = kwargs['quantize_dtype']
        process_scale_w = kwargs['process_scale_w']
        
        log_level = args.log_level
        eval_mode = args.eval_mode
        acc_error = args.acc_error

        model_path = os.path.join(model_dir, args.task_name, model_name)  # args.model_paths[model_type][model_id]
        dataset_name, imgsize, net_name, preprocess_name, postprocess_name = model_type.split('_')
        image_path = os.path.join(dataset_dir, args.dataset_dir[dataset_name])

        model_name = os.path.basename(model_path).split('.onnx')[0]
        export_version = '' if args.export_version > 1 else '_v{}'.format(
            args.export_version)
        log_name = '{}.{}.{}.{}.{}.log'.format(
            model_name, quantize_method_f, quantize_method_w, quantize_dtype, process_scale_w)

        process_args = {
            'log_name': 'process.log',
            'log_level': log_level, 
            'model_path': model_path,
            'parse_cfg': 'benchmark/benchmark_config/base/parse.py',
            'graph_cfg': 'config/graph.py',
            'quan_cfg': 'benchmark/benchmark_config/base/quantize.py',
            # 'analysis_cfg': 'config/analysis.py',
            'export_cfg':
                'config/export{}.py'.format(export_version),
            'offline_quan_mode': args.offline_quan_mode,
            'offline_quan_tool': args.offline_quan_tool,
            'quan_table_path': args.quan_table_path,
            'device': args.device,
            'simulation_level': 1,
            'fp_result': args.fp_result,
            'is_ema': True,
            'ema_value': 0.99,
            'is_array': False,
            'is_simplify': True,
            'is_stdout': args.is_stdout,
            'error_metric': args.error_metric,  ## Cosine | L2 | L1
        }

        kwargs_receval = {
            'log_dir': args.log_dir,
            'log_name': log_name,
            'log_level': log_level, 
            # 'is_stdout': args.is_stdout,
            'eval_first_frame': args.eval_first_frame,
            "dataset_dir": image_path,
            "dataset_name": dataset_name,
            "test_sample_num": args.test_sample_num[selected_mode],
            "feat_dim": args.feat_dim,
            "fp_result": args.fp_result,
            "metric": args.metric,  # "cosine" or "euclidean"
            "max_threshold": args.max_threshold,  # 1.0 | 4.0
            "batch_size": 1,
            "nrof_folds": 5,
            "is_loadtxt": False,
            "result_path": args.results_path,
            "process_args": process_args,
            'is_calc_error': args.is_calc_error,
            "eval_mode": eval_mode,  # [single | dataset] quatize
            'acc_error': acc_error,
        }

        myRecEval = RecEval(**kwargs_receval)

        return myRecEval

    def test_face_recognition(self, model_paths, quantize_dtype, process_scale_w, quantize_method, model_dir,
                              dataset_dir, selected_mode, password):
        ms = model_paths
        qd = quantize_dtype
        ps = process_scale_w
        qm = quantize_method
        self.entrance(args, ms, qm, qd, ps, selected_mode, model_dir, dataset_dir, password)


if __name__ == '__main__':
    pytest.main()
