# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/03/29 19:39:33
# @File    : test_retina_face_detection.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import copy
import json
import os
import unittest
from itertools import product
from math import ceil

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from benchmark import (collect_accuracy, parse_config, save_config,
                       save_export, save_tables)
from benchmark.tests.test_base import TestBasse
from torchvision.ops import nms
from eval.alfw import AlfwEval

try:
    
    from utils import Registry
except:
    from onnx_converter.utils import Registry

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BENCHMARK: Registry = Registry('benchmark', scope='')

cfg_file = 'benchmark/benchmark_config/retina_face_detection.py'
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


def generate_prior(steps, image_sizes, min_sizes):
    """generate priors"""

    feature_maps = [[ceil(image_sizes[0] / step),
                     ceil(image_sizes[1] / step)] for step in steps]

    anchor_lst = []
    for k, f in enumerate(feature_maps):
        min_size = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size_value in min_size:
                s_kx = min_size_value / image_sizes[1]
                s_ky = min_size_value / image_sizes[0]
                dense_cx = [x * steps[k] / image_sizes[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_sizes[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchor_lst += [cx, cy, s_kx, s_ky]

    return np.array(anchor_lst).reshape(-1, 4)


@BENCHMARK.register_module(name="pre1")
class PreProcess(object):

    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = [104, 117, 123]
        self.img_std = [1.0, 1.0, 1.0]
        if 'img_mean' in kwargs.keys():
            self.img_mean = kwargs['img_mean']
        if 'img_std' in kwargs.keys():
            self.img_std = kwargs['img_std']
        self.trans = 0
        self.input_size = kwargs['input_size']
        # self.target_w, self.target_h = 0, 0
        # self.ratio, self.w_pad, self.h_pad = 1, 0, 0
        # self.src_shape = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img, color=(114, 114, 114)):
        target_w, target_h = self.input_size

        img_h, img_w = img.shape[:2]
        # print(f"img_h: {img_h}, img_w: {img_w}") # 331, 500
        if img_w > img_h:
            r = target_w / img_w
            new_shape_w, new_shape_h = target_w, int(round(img_h * r))
            if new_shape_h > 256:
                r = target_h / img_h
                new_shape_w, new_shape_h = int(round(img_w * r)), target_h
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h
            else:
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h
            # print(w_pad, h_pad, r)
            # print('-----------------------------------------------------')
        else:
            r = target_h / img_h
            new_shape_w, new_shape_h = int(round(img_w * r)), target_h
            if new_shape_w > 320:
                r = target_w / img_w
                new_shape_w, new_shape_h = target_w, int(round(img_h * r))
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h
            else:
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h

        w_pad /= 2
        h_pad /= 2

        resize_img = cv2.resize(img, (new_shape_w, new_shape_h),
                                interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
        left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))

        # 固定值边框，统一都填充color
        img_final = cv2.copyMakeBorder(resize_img,
                                       top,
                                       bottom,
                                       left,
                                       right,
                                       cv2.BORDER_CONSTANT,
                                       value=color)

        img_final = img_final - (104, 117, 123)

        img_final = np.transpose(img_final, (2, 0, 1)).astype(np.float32)

        img_final = np.expand_dims(img_final, axis=0)

        # self.ratio, self.w_pad, self.h_pad = r, w_pad, h_pad
        self.trans = dict(target_w=target_w,
                          target_h=target_h,
                          w_pad=w_pad,
                          h_pad=h_pad,
                          src_shape=img.shape[:2],
                          ratio=r)

        return img_final


@BENCHMARK.register_module(name="post1")
class PostProcess(object):

    def __init__(self, **kwargs):
        super(PostProcess, self).__init__()
        self.steps = kwargs["steps"]
        self.min_sizes = kwargs['min_sizes']
        self.nms_threshold = kwargs["nms_threshold"]
        self.variances = kwargs['variances']
        self.prob_threshold = kwargs["prob_threshold"]
        self.top_k = kwargs["top_k"]

    def reshpe_out(self, out):
        if out is None:
            return None
        n, c, h, w = out.shape
        out = np.transpose(out, (0, 2, 3, 1))
        return np.reshape(out, (n, -1, c))

    def __call__(self, outputs, trans):
        target_w = trans['target_w']
        target_h = trans['target_h']
        w_pad = trans['w_pad']
        h_pad = trans['h_pad']
        src_shape = trans['src_shape']
        resize = trans['ratio']
        image_w, image_h = src_shape
        # compute_out = [self.reshpe_out(outputs[key]) for key in outputs.keys()]
        compute_out = {}
        for key in outputs.keys():
            compute_out[key] = self.reshpe_out(outputs[key])
        loc = np.row_stack([compute_out['output1'].reshape(-1, 4),
                            compute_out['output2'].reshape(-1, 4),
                            compute_out['output3'].reshape(-1, 4),
                            compute_out['output4'].reshape(-1, 4)])

        conf = np.row_stack([compute_out['output5'].reshape(-1, 2),
                             compute_out['output6'].reshape(-1, 2),
                             compute_out['output7'].reshape(-1, 2),
                             compute_out['output8'].reshape(-1, 2)])

        landmark = np.row_stack([compute_out['output9'].reshape(-1, 10),
                                 compute_out['output10'].reshape(-1, 10),
                                 compute_out['output11'].reshape(-1, 10),
                                 compute_out['output12'].reshape(-1, 10)])

        conf = F.softmax(torch.from_numpy(conf), dim=-1).numpy()

        priors = generate_prior(steps=self.steps,
                                image_sizes=(target_h, target_w),
                                min_sizes=self.min_sizes)
        # decode bounding box predictions
        boxes = np.concatenate(
            (priors[:, :2] + loc[:, :2] * self.variances[0] * priors[:, 2:],
             priors[:, 2:] * np.exp(loc[:, 2:] * self.variances[1])),
            axis=1)

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        scale_loc = np.array([target_w, target_h, target_w, target_h])
        boxes = boxes * scale_loc / resize
        boxes[:, [0, 2]] -= (w_pad / resize)
        boxes[:, [1, 3]] -= (h_pad / resize)
        # print(w_pad, h_pad, resize)
        # decode landmark
        landmarks = np.concatenate((
            priors[:, :2] +
            landmark[:, :2] * self.variances[0] * priors[:, 2:],
            priors[:, :2] +
            landmark[:, 2:4] * self.variances[0] * priors[:, 2:],
            priors[:, :2] +
            landmark[:, 4:6] * self.variances[0] * priors[:, 2:],
            priors[:, :2] +
            landmark[:, 6:8] * self.variances[0] * priors[:, 2:],
            priors[:, :2] +
            landmark[:, 8:10] * self.variances[0] * priors[:, 2:],
        ),
            axis=1)
        scale_landmark = np.array([
            image_w, target_h, target_w, target_h, target_w, target_h,
            target_w, target_h, target_w, target_h
        ])

        landmarks = landmarks * scale_landmark / resize
        landmarks[:, [0, 2, 4, 6, 8]] -= (w_pad / resize)
        landmarks[:, [1, 3, 5, 7, 9]] -= (h_pad / resize)

        scores = conf[:, 1]
        indexes = scores > self.prob_threshold
        scores = scores[indexes]
        boxes = boxes[indexes, :]
        landmarks = landmarks[indexes, :]

        indexes = np.argsort(scores)
        scores = scores[indexes]
        boxes = boxes[indexes, :]
        landmarks = landmarks[indexes, :]

        # nms
        select_index = nms(torch.from_numpy(boxes.astype(np.float32)),
                           torch.from_numpy(scores),
                           iou_threshold=self.nms_threshold)
        select_index = select_index.numpy().tolist()
        if len(select_index) < self.top_k:
            select_index = select_index[:self.top_k]
        boxes = boxes[select_index, :]
        scores = scores[select_index]
        landmarks = landmarks[select_index, :]
        # print(landmarks)
        # print('-----------------------------------------------------')

        detects = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32,
                                                                   copy=False)
        detects = np.concatenate((detects, landmarks), axis=1)

        #### filter abnormal bbox, added by henson
        for idx in range(detects.shape[0]):
            x0, y0, x1, y1 = detects[idx][:4]
            x0, y0 = np.max((x0, 0)), np.max((y0, 0))
            x1, y1 = np.min((x1, src_shape[1] - 1)), np.min(
                (y1, src_shape[0] - 1))
            detects[idx][:4] = x0, y0, x1, y1

        return detects


@pytest.mark.usefixtures('model_paths', 'quantize_dtype', 'process_scale_w', 'quantize_method')
class TestRetinaFaceDetection(TestBasse):

    def compose_evaluator(self, **kwargs):
        args = kwargs['args']
        model_dir = kwargs['model_dir']
        model_name = kwargs['model_name']
        dataset_dir = kwargs['dataset_dir']
        # selected_mode = kwargs['selected_mode']
        quantize_dtype = kwargs['quantize_dtype']
        process_scale_w = kwargs['process_scale_w']
        
        log_level = args.log_level
        eval_mode = args.eval_mode
        acc_error = args.acc_error

        model_path = os.path.join(model_dir, args.task_name, model_name)  # args.model_paths[model_type][model_id]
        dataset_name, imgsize, net_name, preprocess_name, postprocess_name = model_type.split('_')
        image_path = os.path.join(dataset_dir, args.dataset_dir[dataset_name])

        kwargs_preprocess = {
            "img_mean": args.normalizations[net_name][0],
            "img_std": args.normalizations[net_name][1],
            "input_size": args.input_size[imgsize]
        }
        kwargs_postprocess = {
            "top_k": args.topk,
            "prob_threshold": args.prob_threshold,
            "nms_threshold": args.nms_threshold,
            "num_candidate": args.num_candidate,
            "steps": args.steps[net_name],
            'num_class': args.num_classes,
            "min_sizes": args.min_sizes[net_name],
            "variances": args.variances[net_name],
        }

        preprocess = BENCHMARK.get(preprocess_name)(**kwargs_preprocess)
        postprocess = BENCHMARK.get(postprocess_name)(**kwargs_postprocess)

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
            'transform': preprocess,
            'postprocess': postprocess,
            "fp_result": args.fp_result,
            'is_ema': True,
            'ema_value': 0.99,
            'is_array': False,
            'is_stdout': args.is_stdout,
            'error_metric': args.error_metric
        }

        results_path = args.results_path
        save_imgs = os.path.join(results_path, 'images')
        if not os.path.exists(save_imgs):
            os.makedirs(save_imgs)

        kwargs_bboxeval = {
            'log_dir': args.log_dir,
            'log_name': log_name,
            'log_level': log_level, 
            # 'is_stdout': args.is_stdout,
            'eval_first_frame': args.eval_first_frame,
            'img_prefix': args.img_prefix,
            "iou_threshold": args.nms_threshold,
            "prob_threshold": args.prob_threshold,
            "process_args": process_args,
            'is_calc_error': args.is_calc_error,
            "draw_result": args.draw_result,
            "fp_result": args.fp_result,
            "eval_mode": eval_mode,  # single quantize, dataset quatize
            'acc_error': acc_error,
        }

        alfweval = AlfwEval(**kwargs_bboxeval)
        parameter = dict(quan_dataset_path=image_path,
                         dataset_path=image_path,
                         ann_path=image_path,
                         event_lst=args.event_lst,
                         save_dir=args.results_path)

        return dict(evaluator=alfweval, parameters=parameter)

    def test_retina_face_detection(self, model_paths, quantize_dtype, process_scale_w, quantize_method, model_dir,
                                   dataset_dir, selected_mode, password):
        ms = model_paths
        qd = quantize_dtype
        ps = process_scale_w
        qm = quantize_method
        self.entrance(args, ms, qm, qd, ps, selected_mode, model_dir, dataset_dir, password)


if __name__ == '__main__':
    pytest.main()
