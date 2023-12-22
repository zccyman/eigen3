# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/2/15 9:58
# @File     : test_imagenet.py
import copy
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image
from typing import Tuple

from eval.imagenet_eval import ClsEval

try:
    from tools import WeightOptimization
except:
    from onnx_converter.tools import WeightOptimization
    
    
class ClsEvalWeightOpt(ClsEval):
    def __init__(self, **kwargs):
        super(ClsEvalWeightOpt, self).__init__(**kwargs)
        self.weight_optimization = kwargs["weight_optimization"]
        self.process_args_wo = copy.deepcopy(kwargs['process_args'])
        self.process_wo = WeightOptimization(**self.process_args_wo)
    
    def __call__(self, config_file=None):
        if self.weight_optimization in ["cross_layer_equalization"]:
            eval("self.process_wo." + self.weight_optimization)()
        else:
            eval("self.process_wo." + self.weight_optimization)(config_file=config_file)

class PreProcess(object):

    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = kwargs['img_mean']
        self.img_std = kwargs['img_std']
        self.input_size = kwargs['input_size']
        self.resize_scale = kwargs['resize_scale']
        self.to_rgb = kwargs['to_rgb']        
        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def bbox_clip(self, bboxes, img_shape:Tuple[int, int, int]):  # img_shape h, w, c
        assert bboxes.shape[-1] % 4 == 0
        cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
        cmin[0::2] = img_shape[1] - 1
        cmin[1::2] = img_shape[0] - 1
        clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
        return clipped_bboxes
    
    # def __call__(self, img):
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_resize = cv2.resize(img, [256, 256], cv2.INTER_CUBIC)
    #     x0, y0 = int((256 - self.input_size[0]) / 2), int((256 - self.input_size[1]) / 2)
    #     img_resize = img_resize[y0:y0 + self.input_size[1], x0:x0 + self.input_size[0], :]
    #     # img_resize = cv2.resize(img, self.input_size)
    #     img_input = img_resize.astype(np.float32) / 255
    #     img_mean = np.array(self.img_mean, dtype=np.float32) / 255
    #     img_std = np.array(self.img_std, dtype=np.float32) / 255
    #     img_input = (img_input - img_mean) / img_std
    #     # expand dims
    #     img_input = np.transpose(img_input, [2, 0, 1])
    #     img_input = np.expand_dims(img_input, axis=0)

    #     return img_input
    
    def __call__(self, img):
        # to RGB
        if self.to_rgb:
            img = img[:, :, ::-1]
        # ReszieEdge
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type.'
        h, w = img.shape[:2]
        if w < h:
            width = self.resize_scale
            height = int(self.resize_scale * h / w)
        else:
            height = self.resize_scale
            width = int(self.resize_scale * w / h)
        size = (width, height)
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, Image.BICUBIC)
        resized_img = np.array(pil_image)
        w_scale = size[0] / w
        h_scale = size[1] / h
        # CenterCrop
        crop_width, crop_height = self.input_size[0], self.input_size[1]
        img_height, img_width = resized_img.shape[:2]
        if crop_height > img_height or crop_width > img_width:
            crop_height = min(crop_height, img_height)
            crop_width = min(crop_width, img_width)
        y1 = max(0, int(round((img_height - crop_height) / 2.)))
        x1 = max(0, int(round((img_width - crop_width) / 2.)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1
        bboxes = np.array([x1, y1, x2, y2])
        chn = resized_img.shape[2]
        bboxes = bboxes[None, ...].astype(np.int32)
        clipped_bbox = self.bbox_clip(bboxes, resized_img.shape)[0]
        x1, y1, x2, y2 = tuple(clipped_bbox)
        patch = resized_img[y1:y2 + 1, x1:x2 + 1, ...]
        
        img = (patch - self.img_mean) / self.img_std
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        
        return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, 
                        default='/data/vision/buffer/ImageNet/val',
                        )
    parser.add_argument('--quan_dataset_path', type=str, 
                        default='/buffer/calibrate_dataset/ImageNet_calibrate',
                        )
    parser.add_argument('--model_path', type=str,
                        default='RTM_CSPNeXt_tiny_onnx/rtmdet_tiny_cls_224x224_sim.onnx',
                        )
    # torchvision_MobileNetV3_small_ImageNet_classification_simplify.onnx
    # MobileNetv3_classification-sim-remove-expandOp_replace-reduce.onnx

    parser.add_argument('--input_size', type=list, default=[224, 224])
    parser.add_argument('--output_name', type=str, default='375')
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)
    parser.add_argument('--log_level', type=int, default=10)    
    parser.add_argument('--error_analyzer', type=bool, default=False) 
    parser.add_argument('--is_calc_error', type=bool, default=False)  # whether to calculate each layer error
    # bias_correction cross_layer_equalization
    parser.add_argument('--weight_optimization', type=str, default="qat")   
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        eval_mode = 'single'
        acc_error = False
    else:
        eval_mode = 'dataset'
        acc_error = True

    kwargs_preprocess = {
        # "img_mean": [103.53, 116.28, 123.675],
        # "img_std": [57.375, 57.12, 58.395],
        "img_mean": [123.675, 116.28, 103.53],
        "img_std": [58.395, 57.12, 57.375],
        'input_size': args.input_size,
        'resize_scale': 236,
        'to_rgb': True,
    }
    preprocess = PreProcess(**kwargs_preprocess)

    process_args = {
        'log_name': 'process.log',
        'log_level': args.log_level,
        'model_path': args.model_path,
        'parse_cfg': 'config/parse.py',
        'graph_cfg': 'config/graph.py',
        'quan_cfg': 'config/quantize.py',
        'analysis_cfg': 'config/analysis.py',
        'export_cfg': 'config/export_v{}.py'.format(args.export_version),
        'offline_quan_mode': None,
        'offline_quan_tool': None,
        'quan_table_path': None,
        'device': args.device,
        'fp_result': args.fp_result,
        'transform': preprocess,
        'simulation_level': 1,
        'is_ema': True,
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,
        'error_analyzer': args.error_analyzer,
        'error_metric': ['L1', 'L2', 'Cosine'],  ## Cosine | L2 | L1
    }

    kwargs_clseval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_imagenet.log',
        'log_level': args.log_level,
        'is_stdout': args.is_stdout,
        'dataset_path': args.dataset_path,
        'quan_dataset_path': args.quan_dataset_path,
        'img_prefix': 'JPEG',
        'output_name': args.output_name,
        'class_num': 1000,
        'transform': preprocess,
        'process_args': process_args,
        'is_calc_error': args.is_calc_error,
        'acc_error': acc_error,
        "weight_optimization": args.weight_optimization,
        'fp_result': args.fp_result,
        'eval_mode': eval_mode,  # single | dataset
        'model_path': args.model_path,
    }

    cls_eval = ClsEvalWeightOpt(**kwargs_clseval)
    cls_eval(config_file="tests/qat/test_imagenet.json") 
