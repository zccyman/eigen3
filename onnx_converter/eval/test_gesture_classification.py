# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/2/15 9:58
# @File     : test_imagenet.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import os
import cv2
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from eval import Eval

try:    
    from tools import ModelProcess
except:
    from onnx_converter.tools import ModelProcess


class Metric(object):
    def __init__(self, sample_num):
        self.sample_num = sample_num
        self.correct_num = 0
        self.correct_prob_lst = []
    
    def update(self, predict_label, gt_label, probs):
        if predict_label == gt_label:
            self.correct_num += 1
            self.correct_prob_lst.append(probs[0, predict_label])
    
    def get_acc(self):
        return 100 * self.correct_num / self.sample_num          
        
        
class ClsEval(Eval):
    def __init__(self, **kwargs):
        super(ClsEval, self).__init__(**kwargs)
        
        self.dataset_path = kwargs['dataset_path']
        self.quan_dataset_path = kwargs['quan_dataset_path']
        self.img_prefix = kwargs['img_prefix']
        self.output_name = kwargs['output_name']
        self.class_num = kwargs['class_num']
        # self.transform = kwargs['transform']
        self.process_args = kwargs['process_args']
        self.is_calc_error = kwargs['is_calc_error']
        # self.is_stdout = self.process_args['is_stdout']
        self.acc_error = kwargs['acc_error']
        self.eval_mode = kwargs['eval_mode']
        self.fp_result = kwargs['fp_result']
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)

        self.process = ModelProcess(**self.process_args)
        if self.is_calc_error:
            self.process.set_onnx_graph(False)
            # self.process.onnx_graph = False

                          
    def __call__(self):
        features = np.load(os.path.join(self.dataset_path, "test_features.npy")).astype(np.float32) 
        features = np.array_split(features, features.shape[0])    
        labels = np.load(os.path.join(self.dataset_path, "test_labels.npy")).astype(np.float32) 
        calibration_data = np.load(self.quan_dataset_path).astype(np.float32)     
        calibration_data = np.array_split(calibration_data, calibration_data.shape[0])     
        
        fp_m = Metric(sample_num=len(features))
        quant_m = Metric(sample_num=len(features))
                      
        # self.eval_mode = "single"
        if self.eval_mode == 'dataset':
            self.process.quantize(fd_path=calibration_data, is_dataset=True)
                    
        features_ = tqdm(features, postfix='image files') if self.is_stdout else features
        for image_id, in_data in enumerate(features_):
            if self.fp_result:
                true_outputs = self.process.post_quan.onnx_infer(in_data)
            else:
                true_outputs = None
                
            # self.acc_error = False
            # self.is_calc_error = True                
            if self.eval_mode == 'single':
                self.process.quantize(in_data, is_dataset=False)
            
            if self.is_calc_error:
                self.process.checkerror(in_data, acc_error=self.acc_error)
            else:
                self.process.dataflow(in_data, acc_error=True, onnx_outputs=true_outputs)
            
            outputs = self.process.get_outputs()
            
            qout = outputs['qout'][self.output_name]
            qprobs = softmax(qout)
            qpredict_label = np.argmax(qprobs, 1)[0]   
            quant_m.update(qpredict_label, labels[image_id], qprobs)         
            if self.fp_result:
                fout = outputs['true_out'][self.output_name]
                fprobs = softmax(fout)
                fpredict_label = np.argmax(fprobs, 1)[0]   
                fp_m.update(fpredict_label, labels[image_id], fprobs)
            
            if 0 == image_id and self.eval_first_frame: break
            # if 0 == image_id:
            #     break
            
        if not self.is_calc_error and self.process.onnx_graph:
            img = cv2.imread(features_[image_id])
            true_outputs = self.process.post_quan.onnx_infer(img)
            self.process.numpygraph(img, acc_error=True, onnx_outputs=true_outputs)
        
        accuracy = dict()
        if self.fp_result:
            accuracy['faccuracy'] = {
                "acc": fp_m.get_acc(),
            }
        accuracy['qaccuracy'] = {
                "acc": quant_m.get_acc(),
        }
        tb = self.get_results(accuracy)

        return accuracy, tb
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, 
                        default='Gesture_classification/data',
                        )
    parser.add_argument('--quan_dataset_path', type=str, 
                        default='Gesture_classification/data/calibrate_features.npy',
                        )
    parser.add_argument('--model_path', type=str,
                        default='Gesture_classification/onnx_weights/rules_model_simplify.onnx',
                        )
    parser.add_argument('--output_name', type=str, default='output')
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)
    parser.add_argument('--log_level', type=int, default=30)    
    parser.add_argument('--error_analyzer', type=bool, default=False) 
    parser.add_argument('--is_calc_error', type=bool, default=False)  # whether to calculate each layer error

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
        # 'input_size': args.input_size,
        'resize_scale': 236,
        'to_rgb': True,
    }
    preprocess = None

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
        # 'transform': preprocess,
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
        # 'transform': preprocess,
        'process_args': process_args,
        'is_calc_error': args.is_calc_error,
        'acc_error': acc_error,
        'fp_result': args.fp_result,
        'eval_mode': eval_mode,  # single | dataset
        'model_path': args.model_path,
    }

    cls_eval = ClsEval(**kwargs_clseval)
    accuracy, tb = cls_eval()
    if args.is_calc_error:
        cls_eval.collect_error_info()     
    if args.error_analyzer:
        cls_eval.error_analysis()     
    if args.export:
        cls_eval.export()
    print(tb)
