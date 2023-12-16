# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/4/18 9:58
# @File     : test_visualization.py

# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/24 9:47
# @File     : test_object_detection.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import copy
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torchvision.ops import nms

try:
    from tools import ModelProcess, OnnxruntimeInfer
    from utils import Object
except:
    from onnx_converter.tools import ModelProcess, OnnxruntimeInfer
    from onnx_converter.utils import Object
    

class Visualization(Object):
    def __init__(self, **kwargs):
        super(Visualization, self).__init__(**kwargs)

        self.class_names = kwargs['class_names']
        self.process_args = kwargs['process_args']
        self.is_calc_error = kwargs['is_calc_error']
        self.draw_result = kwargs['draw_result']
        self.save_key = kwargs['save_key']
        self.eval_mode = kwargs['eval_mode']
        self.fp_result = kwargs['fp_result']
        self.acc_error = kwargs['acc_error']
        self.num_classes = len(self.class_names)
        self.qout_color = np.array([np.array([1, 0, 0]) for _ in range(self.num_classes)])
        self.fout_color = np.array([np.array([0, 0, 1]) for _ in range(self.num_classes)])
        self.img_prefix = kwargs['img_prefix']
        self.is_stdout = kwargs['is_stdout']

        if 'postprocess' in self.process_args.keys():
            setattr(self, 'postprocess', self.process_args['postprocess'])

        self.process = ModelProcess(**self.process_args)
                
    def __call__(self, quan_dataset_path, dataset_path, save_eval_path):
        output_names, input_names = self.process.get_output_names(), self.process.get_input_names()
        onnxinferargs = copy.deepcopy(self.process_args)
        onnxinferargs.update(out_names=output_names, input_names=input_names)
        onnxinfer = OnnxruntimeInfer(**onnxinferargs)

        is_dataset = False
        if os.path.isfile(quan_dataset_path):
            is_dataset = False
        elif os.path.isdir(quan_dataset_path):
            is_dataset = True
        else:
            print('invaild input quan file!')
            os._exit(-1)

        if self.eval_mode == "dataset":
            self.process.quantize(fd_path=quan_dataset_path, is_dataset=is_dataset)
        
        save_imgs = os.path.join(save_eval_path, 'images')
        if not os.path.exists(save_imgs):
            os.makedirs(save_imgs)
        
        images = glob.glob(os.path.join(dataset_path, "*." + self.img_prefix))

        images_ = tqdm.tqdm(images, postfix='image files') if self.is_stdout else images
        for image_id, item in enumerate(images_):
            image_name = os.path.basename(item)
            img = cv2.imread(os.path.join(dataset_path, image_name))

            if self.eval_mode == "single":
                self.process.quantize(fd_path=os.path.join(dataset_path, image_name), is_dataset=False)

            if self.fp_result:
                true_outputs = onnxinfer(in_data=img)
            else:
                true_outputs = None
                
            if self.is_calc_error:
                self.process.checkerror(img, acc_error=self.acc_error)
            else:
                self.process.dataflow(img, acc_error=True, onnx_outputs=true_outputs)
                                     
            outputs = self.process.get_outputs()
            q_out, t_out = outputs['qout'], outputs['true_out']

            if self.draw_result and outputs is not None:
                if hasattr(self, 'postprocess') and hasattr(self.postprocess, 'draw_image'):
                    draw_image = self.postprocess.draw_image
                else:
                    draw_image = self.draw_image
                qres = draw_image(copy.deepcopy(img), q_out, class_names=self.class_names, colors=self.qout_color) #np.array([[1, 0, 0]]) self._COLORS
                if self.fp_result:
                    qres = draw_image(qres, t_out, class_names=self.class_names, colors=self.fout_color) #np.array([[0, 0, 1]]) self._COLORS
                cv2.imwrite(os.path.join(save_imgs, os.path.basename(image_name)), qres)  ### results for draw bbox

            # if 0 == image_id: break

### object_detection
def object_detection_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', 
                        type=str, 
                        default='/home/henson/dataset/trained_models/object-detection/nanodet_1.5x_320_simplify.onnx', 
                        help='checkpoint path')
    parser.add_argument('--quan_dataset_path', 
                        type=str, 
                        default='/buffer/coco/val2017', 
                        help='quantize image path')
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='/buffer/coco/val2017', 
                        help='eval image path')
    parser.add_argument('--input_size', type=list, default=[320, 320])
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--results_path', type=str, default='./work_dir/tmpfiles/vis')
    parser.add_argument('--topk', type=int, default=-1)
    parser.add_argument('--reg_max', type=int, default=7)
    parser.add_argument('--prob_threshold', type=float, default=0.3)
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    parser.add_argument('--num_candidate', type=int, default=1000)
    parser.add_argument('--strides', type=list, default=[8, 16, 32])
    parser.add_argument('--draw_result', type=bool, default=True)
    parser.add_argument('--agnostic_nms', type=bool, default=False)
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export_version', type=int, default=1)
    parser.add_argument('--export', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)  
    parser.add_argument('--log_level', type=int, default=30)  
    parser.add_argument('--is_calc_error', type=bool, default=True) #whether to calculate each layer error
    args = parser.parse_args()
    return args


### pedestrian_detection
def pedestrian_detection_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/home/henson/dataset/trained_models/pedestrian-detection/yolov5n_320_v1_simplify.onnx',
                        # default='./trained_models/pedestrian-detection/nanodet_plus_m_1.0x_320_v1_simplify.onnx',
                        help='checkpoint path')
    parser.add_argument('--quan_dataset_path', type=str, 
                        default='/buffer/crowdhuman/val',
                        help='quantize image path')
    parser.add_argument('--dataset_path', type=str, 
                        default='/buffer/crowdhuman/val',
                        help='eval image path')
    parser.add_argument('--model_name', type=str, default="yolov5") # nano | yolov5
    parser.add_argument('--input_size', type=list, default=[320, 320])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--results_path', type=str, default='./work_dir/tmpfiles/vis')
    parser.add_argument('--topk', type=int, default=-1)
    parser.add_argument('--reg_max', type=int, default=7)
    parser.add_argument('--prob_threshold', type=float, default=0.3)
    parser.add_argument('--iou_threshold', type=float, default=0.6)
    parser.add_argument('--num_candidate', type=int, default=1000)
    parser.add_argument('--draw_result', type=bool, default=True)
    parser.add_argument('--agnostic_nms', type=bool, default=False)
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export_version', type=int, default=1)
    parser.add_argument('--export', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True) 
    parser.add_argument('--log_level', type=int, default=30)   
    parser.add_argument('--is_calc_error', type=bool, default=True) #whether to calculate each layer error
    args = parser.parse_args()
    return args


### retina_face_detection
def retina_face_detection_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', 
                        type=str, 
                        default='/home/henson/dataset/trained_models/face-detection/slim_special_Final_simplify.onnx', 
                        help='checkpoint path')
    parser.add_argument('--quan_dataset_path', 
                        type=str, 
                        default="/buffer/AFLW/sub_test_data/flickr_3/image", 
                        help='eval dataset path')
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='/buffer/AFLW/sub_test_data/flickr_3/image',
                        help='eval image path')
    parser.add_argument('--input_size', type=list, default=[320, 256])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--results_path', type=str, default='./work_dir/tmpfiles/vis')
    parser.add_argument('--topk', type=int, default=1000)
    parser.add_argument('--prob_threshold', type=float, default=0.7)
    parser.add_argument('--nms_threshold', type=float, default=0.4)
    parser.add_argument('--num_candidate', type=int, default=1000)
    parser.add_argument('--steps', type=list, default=[8, 16, 32, 64]) # slim
    # parser.add_argument('--steps', type=list, default=[8, 16, 32]) # mobilenet
    parser.add_argument('--variances', type=list, default=[0.1, 0.2])
    parser.add_argument('--min_sizes', type=list, default=[[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]) # slim
    # parser.add_argument('--min_sizes', type=list, default=[[10, 20], [32, 64], [128, 256]]) # mobilenet
    parser.add_argument('--draw_result', type=bool, default=True)
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export_version', type=int, default=1)
    parser.add_argument('--export', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)
    parser.add_argument('--is_calc_error', type=bool, default=True) #whether to calculate each layer error
    args = parser.parse_args()
    return args


def process_object_detection():
    from eval.test_object_detection import PostProcess, PreProcess, class_names
    args = object_detection_parse_args()

    if args.debug:
        args.eval_mode = 'single'
        args.acc_error = False
    else:
        args.eval_mode = 'dataset'
        args.acc_error = True
        
    normalization = [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
    kwargs_preprocess = {
        "img_mean": normalization[0],
        "img_std": normalization[1],
        "input_size": args.input_size,
    }
    kwargs_postprocess = {
        "reg_max": args.reg_max,
        "top_k": args.topk,
        "prob_threshold": args.prob_threshold,
        "iou_threshold": args.iou_threshold,
        "num_candidate": args.num_candidate,
        "strides": args.strides,
        'num_classes': args.num_classes,
        'agnostic_nms': args.agnostic_nms,
    }

    preprocess = PreProcess(**kwargs_preprocess)
    postprocess = PostProcess(**kwargs_postprocess)

    return args, class_names, preprocess, postprocess


def process_pedestrian_detection():
    from eval.test_pedestrian_detection import (NanodetPostProcess,
                                                NanodetPreProcess,
                                                Yolov5PostProcess,
                                                Yolov5PreProcess, class_names,
                                                normalizations,
                                                pre_post_instances, strides)
    args = pedestrian_detection_parse_args()

    if args.debug:
        args.eval_mode = 'single'
        args.acc_error = False
    else:
        args.eval_mode = 'dataset'
        args.acc_error = True
        
    normalization = normalizations[args.model_name]
    kwargs_preprocess = {
        "img_mean": normalization[0],
        "img_std": normalization[1],
        "input_size": args.input_size,
        "stride": 64
    }
    kwargs_postprocess = {
        "reg_max": args.reg_max,
        "top_k": args.topk,
        "prob_threshold": args.prob_threshold,
        "iou_threshold": args.iou_threshold,
        "num_candidate": args.num_candidate,
        "strides": strides[args.model_name],
        "agnostic_nms": args.agnostic_nms,
        "num_classes": args.num_classes,
    }

    preprocess = eval(pre_post_instances[args.model_name][0])(**kwargs_preprocess)
    postprocess = eval(pre_post_instances[args.model_name][1])(**kwargs_postprocess)

    return args, class_names, preprocess, postprocess


def process_retina_face_detection():
    from eval.test_retina_face_detection import (PostProcess, PreProcess,
                                                 class_names)
    args = retina_face_detection_parse_args()

    if args.debug:
        args.eval_mode = 'single'
        args.acc_error = False
    else:
        args.eval_mode = 'dataset'
        args.acc_error = True
        
    normalization = [[104, 117, 123], [1.0, 1.0, 1.0]]
    kwargs_preprocess = {
        "img_mean": normalization[0],
        "img_std": normalization[1],
    }
    kwargs_postprocess = {
        "top_k": args.topk,
        "prob_threshold": args.prob_threshold,
        "nms_threshold": args.nms_threshold,
        "num_candidate": args.num_candidate,
        "steps": args.steps,
        'num_class': args.num_classes,
        "min_sizes": args.min_sizes,
        "variances": args.variances,
    }

    preprocess = PreProcess(input_size=args.input_size)
    postprocess = PostProcess(**kwargs_postprocess)

    return args, class_names, preprocess, postprocess


if __name__ == "__main__":
    # args, class_names, preprocess, postprocess = process_object_detection()
    # args, class_names, preprocess, postprocess = process_pedestrian_detection()
    args, class_names, preprocess, postprocess = process_retina_face_detection()

    export_version = '' if args.export_version > 1 else '_v{}'.format(args.export_version)
    process_args = {'log_name': 'process.log',
                    'log_level': args.log_level,
                    'model_path': args.model_path,
                    'parse_cfg': 'config/parse.py',
                    'graph_cfg': 'config/graph.py',
                    'quan_cfg': 'config/quantize.py',
                    'analysis_cfg': 'config/analysis.py',
                    'export_cfg': 'config/export{}.py'.format(export_version),
                    'offline_quan_mode': None,
                    'offline_quan_tool': None,
                    'quan_table_path': None,
                    'simulation_level': 1, 
                    'transform': preprocess, 
                    'postprocess': postprocess,
                    'is_ema': True, 
                    'ema_value': 0.99, 
                    'is_array': False, 
                    'is_stdout': args.is_stdout, 
                    'error_metric': ['L1', 'L2', 'Cosine']
                    }

    kwargs_bboxeval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_object_detection.log',   
        'log_level': args.log_level,
        'is_stdout': args.is_stdout,
        'img_prefix': 'jpg',              
        "save_key": "mAP",
        "draw_result": args.draw_result,
        "class_names": class_names,
        "process_args": process_args,
        'is_calc_error': args.is_calc_error,
        "eval_mode": args.eval_mode,  # single quantize, dataset quantize
        "fp_result": args.fp_result,
        'acc_error': args.acc_error,
    }

    vis = Visualization(**kwargs_bboxeval)
    vis(quan_dataset_path=args.quan_dataset_path, 
        dataset_path=args.dataset_path, 
        save_eval_path=args.results_path)
