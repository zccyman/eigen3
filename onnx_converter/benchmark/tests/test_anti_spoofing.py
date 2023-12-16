# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/03/29 19:41:02
# @File    : test_anti_spoofing.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import json
import os
import unittest

import cv2
import numpy as np
import pytest
from benchmark import (collect_accuracy, parse_config, save_config,
                       save_export, save_tables)
from benchmark.tests.test_base import TestBasse
from eval.face_antispoof import FaceSpoofEvaluator

try:
    from utils import Registry
except:
    from onnx_converter.utils import Registry

BENCHMARK: Registry = Registry('benchmark', scope='')

cfg_file = 'benchmark/benchmark_config/face_antispoof.py'
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


@pytest.fixture(scope='class', params=params, autouse=True)
def quantize_dtype(request):
    return request.param


### feat/weight quant method
params = quantize_method_list


@pytest.fixture(scope='class', params=params, autouse=True)
def quantize_method(request):
    return request.param


params = process_scale_w_list


@pytest.fixture(scope='class', params=params, autouse=True)
def process_scale_w(request):
    return request.param


# @pytest.mark.usefixtures('model_paths', 'quantize_dtype', 'process_scale_w', 'quantize_method')
@BENCHMARK.register_module(name="pre1")
class PreProcess(object):
    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = kwargs['img_mean']
        self.img_std = kwargs['img_std']
        self.face_size = kwargs['face_size']
        self.swapRB = kwargs['swapRB']
        self.frame_size = kwargs['frame_size']  # bbox's coordinate and shape values are based on a frame of (224,224)
        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img):
        face_resize = cv2.resize(img, self.face_size, interpolation=cv2.INTER_CUBIC)
        if self.swapRB:
            face_resize = cv2.cvtColor(face_resize, cv2.COLOR_BGR2RGB)
        face_resize = face_resize.astype(np.float32) / 255
        face_resize = (face_resize - self.img_mean) / self.img_std
        face_resize = np.transpose(face_resize, [2, 0, 1])
        face_resize = np.expand_dims(face_resize, axis=0)
        return face_resize.astype(np.float32)


@pytest.mark.usefixtures('model_paths', 'quantize_dtype', 'process_scale_w', 'quantize_method')
class TestFaceAntiSpoof(TestBasse):

    def compose_evaluator(self, **kwargs):
        args = kwargs['args']
        model_dir = kwargs['model_dir']
        model_name = kwargs['model_name']
        dataset_dir = kwargs['dataset_dir']
        quantize_dtype = kwargs['quantize_dtype']
        process_scale_w = kwargs['process_scale_w']

        log_level = args.log_level
        eval_mode = args.eval_mode
        acc_error = args.acc_error

        model_path = os.path.join(model_dir, args.task_name, model_name)  # args.model_paths[model_type][model_id]
        dataset_name, imgsize, net_name, preprocess_name, postprocess_name = model_type.split('_')
        image_path = os.path.join(dataset_dir, args.dataset_dir[dataset_name])

        # build proprcessor
        kwargs_preprocess = {
            "img_mean": args.normalizations[0],
            "img_std": args.normalizations[1],
            'face_size': args.face_size,
            'swapRB': args.swapRB,
            "frame_size": args.frame_size,
        }
        preprocess = PreProcess(**kwargs_preprocess)

        export_version = '' if args.export_version > 1 else '_v{}'.format(
            args.export_version)
        model_name = os.path.basename(model_path).split('.onnx')[0]
        log_name = '{}.{}.{}.{}.{}.log'.format(
            model_name, quantize_method_f, quantize_method_w, quantize_dtype, process_scale_w)

        # build main processor
        process_args = {
            'log_name': 'process.log',
            'log_level': log_level, 
            'model_path': model_path,
            'parse_cfg': 'benchmark/benchmark_config/base/parse.py',
            'graph_cfg': 'config/graph.py',
            'quan_cfg': 'benchmark/benchmark_config/base/quantize.py',
            # 'analysis_cfg': 'config/analysis.py',
            'export_cfg': 'config/export{}.py'.format(export_version),
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
            'error_metric': args.error_metric,
        }

        kwargs_antispoof = {
            'log_dir': args.log_dir,
            'log_name': log_name,
            'log_level': log_level, 
            # 'is_stdout': args.is_stdout,
            'eval_first_frame': args.eval_first_frame,
            'image_root': image_path,
            'image_subdir': args.image_subdir[dataset_name],
            'img_prefix': args.image_prefix,
            'gt_json': 'work_dir/tmpfiles/gt.json',
            'pred_json': 'work_dir/tmpfiles/pred.json',
            'fpred_json': 'work_dir/tmpfiles/fpred.json',
            'class_num': args.class_num,
            'transform': preprocess,
            'process_args': process_args,
            'is_calc_error': args.is_calc_error,
            'model_path': model_path,
            'frame_size': args.frame_size,
            'conf_threshold': args.spoof_prob_threshold,
            'roc_save_path': args.roc_save_path,
            'acc_error': acc_error,
            'fp_result': args.fp_result,
            'eval_mode': eval_mode,  # single | dataset
        }

        evaluator = FaceSpoofEvaluator(**kwargs_antispoof)

        return evaluator

    # model_paths, quantize_dtype, process_scale_w, quantize_method, model_dir, dataset_dir, selected_mode, password
    def test_face_antispoof(self, model_paths, quantize_dtype, process_scale_w, quantize_method, model_dir, dataset_dir,
                            selected_mode, password):
        ms = model_paths
        qd = quantize_dtype
        ps = process_scale_w
        qm = quantize_method
        # args, model_paths, quantize_method, quantize_dtype, process_scale_w, selected_mode, model_dir,
        # dataset_dir, password
        self.entrance(args, ms, qm, qd, ps, selected_mode, model_dir, dataset_dir, password)


if __name__ == '__main__':
    pytest.main(["benchmark/tests/test_anti_spoofing.py::TestFaceAntiSpoof::test_face_antispoof"])
