# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/03/29 19:40:30
# @File    : test_imagenet.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import glob
import json
import os
import unittest

import cv2
import numpy as np
import pytest
from benchmark import (collect_accuracy, parse_config, save_config,
                       save_export, save_tables)
from benchmark.tests.test_base import TestBasse
from eval.imagenet_eval import ClsEval

try:    
    from utils import Registry
except:
    from onnx_converter.utils import Registry

BENCHMARK: Registry = Registry('benchmark', scope='')

cfg_file = 'benchmark/benchmark_config/imagenet.py'
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


# @pytest.mark.usefixtures('model_paths', 'quantize_dtype', 'process_scale_w', 'quantize_method')
@BENCHMARK.register_module(name="pre1")
class PreProcess(object):
    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = [103.53, 116.28, 123.675]
        self.img_std = [57.375, 57.12, 58.395]
        self.input_size = kwargs['input_size']
        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img):
        img_resize = cv2.resize(img, self.input_size)
        img_input = img_resize.astype(np.float32) / 255
        img_mean = np.array(self.img_mean, dtype=np.float32) / 255
        img_std = np.array(self.img_std, dtype=np.float32) / 255
        img_input = (img_input - img_mean) / img_std
        # expand dims
        img_input = np.transpose(img_input, [2, 0, 1])
        img_input = np.expand_dims(img_input, axis=0)

        return img_input


@pytest.mark.usefixtures('model_paths', 'quantize_dtype', 'process_scale_w', 'quantize_method')
class TestImageNet(TestBasse):

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

        kwargs_preprocess = {'input_size': args.input_size[imgsize]}
        preprocess = BENCHMARK.get(preprocess_name)(**kwargs_preprocess)

        image_files = []
        for idx in args.image_subdir:
            tmpfiles = sorted(
                glob.glob('{}/{}/*.{}'.format(
                    image_path,
                    idx, args.img_prefix)
                ))[:args.selected_sample_num_per_class[selected_mode]]
            image_files.extend(tmpfiles)

        export_version = '' if args.export_version > 1 else '_v{}'.format(
            args.export_version)
        model_name = os.path.basename(model_path).split('.onnx')[0]
        log_name = '{}.{}.{}.{}.{}.log'.format(
            model_name, quantize_method_f, quantize_method_w, quantize_dtype, process_scale_w)

        process_args = {
            'log_name': 'process.log',
            'log_level': log_level, 
            'model_path': model_path,
            'parse_cfg': 'benchmark/benchmark_config/base/parse.py',
            'graph_cfg': 'config/graph.py',
            'quan_cfg': 'benchmark/benchmark_config/base/quantize.py',
            # 'analysis_cfg': '{}/config/analysis.py',
            'export_cfg':
                'config/export{}.py'.format(export_version),
            'offline_quan_mode': args.offline_quan_mode,
            'offline_quan_tool': args.offline_quan_tool,
            'quan_table_path': args.quan_table_path,
            'device': args.device,
            'transform': preprocess,
            'simulation_level': 1,
            'fp_result': args.fp_result,
            'is_ema': True,
            'ema_value': 0.99,
            'is_array': False,
            'is_stdout': args.is_stdout,
            'error_metric': args.error_metric,  ## Cosine | L2 | L1
        }

        kwargs_clseval = {
            'log_dir': args.log_dir,
            'log_name': log_name,
            'log_level': log_level, 
            # 'is_stdout': args.is_stdout,
            'eval_first_frame': args.eval_first_frame,
            'image_files': image_files,
            'image_path': image_path,
            'image_subdir': args.image_subdir,  # [idx for idx in range(0, 1000, 1)] | []
            'selected_sample_num_per_class': args.selected_sample_num_per_class[selected_mode],
            'img_prefix': args.img_prefix,
            'gt_json': 'work_dir/tmpfiles/classification/gt.json',
            'pred_json':
                'work_dir/tmpfiles/classification/pred.json',
            'fpred_json':
                'work_dir/tmpfiles/classification/fpred.json',
            'fake_data': False,
            'class_num': args.class_num,
            'transform': preprocess,
            'process_args': process_args,
            'is_calc_error': args.is_calc_error,
            'acc_error': acc_error,
            'fp_result': args.fp_result,
            'eval_mode': eval_mode,  # single | dataset
            'model_path': model_path,
        }

        myClsEval = ClsEval(**kwargs_clseval)

        return myClsEval

    def test_imagenet(self, model_paths, quantize_dtype, process_scale_w, quantize_method, model_dir, dataset_dir,
                      selected_mode, password):
        ms = model_paths
        qd = quantize_dtype
        ps = process_scale_w
        qm = quantize_method
        self.entrance(args, ms, qm, qd, ps, selected_mode, model_dir, dataset_dir, password)


if __name__ == '__main__':
    pytest.main()
