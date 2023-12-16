# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/24 9:58
# @File     : test_face_recognition.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import os
import copy
import numpy as np

try:
    from eval.face_recognition import RecEval, get_validation_pair
    from tools import WeightOptimization
except:
    from onnx_converter.eval.face_recognition import RecEval, get_validation_pair
    from onnx_converter.tools import WeightOptimization

class RecEvalWeightOpt(RecEval):
    def __init__(self, **kwargs):
        super(RecEvalWeightOpt, self).__init__(**kwargs)
        self.dataset_dir = kwargs["dataset_dir"]
        self.dataset_name = kwargs["dataset_name"]
        self.test_sample_num = kwargs["test_sample_num"]        
        self.carray, self.issame = get_validation_pair(self.dataset_name, self.dataset_dir)
        if hasattr(self, "test_sample_num"):
            self.carray = self.carray[:self.test_sample_num, ...]
            self.issame = self.issame[:self.test_sample_num // 2]

        self.weight_optimization = kwargs["weight_optimization"]
        self.process_args_wo = copy.deepcopy(kwargs['process_args'])
        self.process_args_wo["quan_cfg"] = "config/quantize_fp.py"
        self.process_wo = WeightOptimization(**self.process_args_wo)

    def __call__(self, quan_dataset_path=None):
        smin, smax = 0, len(self.carray) - 2
        select_carray = np.arange(smin, smax)
        select_issame = np.arange(smin // 2, smax // 2)
        carray = self.carray[select_carray, :, :, :]
        issame = self.issame[select_issame]

        if self.weight_optimization in ["cross_layer_equalization"]:
            eval("self.process_wo." + self.weight_optimization)()
        else:
            eval("self.process_wo." + self.weight_optimization)(self.carray[:, ...])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/buffer/faces_emore')
    parser.add_argument('--dataset_name', type=str, default='lfw')
    parser.add_argument('--model_path', type=str, 
                        default='trained_models/face-recognition/mobilefacenet_method_3_simplify.onnx'
                        )
    parser.add_argument('--export_version', type=int, default=1)
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')     
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True) 
    parser.add_argument('--log_level', type=int, default=30)    
    parser.add_argument('--is_calc_error', type=bool, default=False) #whether to calculate each layer error
    # bias_correction cross_layer_equalization
    parser.add_argument('--weight_optimization', type=str, default="bias_correction")
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
        
    export_version = '' if args.export_version > 1 else '_v{}'.format(args.export_version)
    process_args = {
        'log_name': 'process_{}.log'.format(args.weight_optimization),
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
        'is_ema': True,
        'device': args.device,
        'fp_result': args.fp_result,
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,          
        'error_metric': ['L1', 'L2', 'Cosine'], ## Cosine | L2 | L1
        "is_fused_act": False,
    }

    kwargs_receval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_face_recognition.log',     
        'log_level': args.log_level,
        # 'is_stdout': args.is_stdout,               
        "dataset_dir": args.dataset_dir,
        "dataset_name": args.dataset_name,
        "test_sample_num": 1200,
        "feat_dim": 512,
        'fp_result': args.fp_result,
        "metric": "euclidean",  # "cosine" or "euclidean"
        "max_threshold": 4.0,  # 1.0 | 4.0
        "batch_size": 1,
        "nrof_folds": 5,
        "is_loadtxt": False,
        "result_path": "work_dir/tmpfiles/recognition",
        "process_args": process_args,
        'is_calc_error': args.is_calc_error,
        "weight_optimization": args.weight_optimization,
        "eval_mode": eval_mode,  # [single | dataset] quatize
        'acc_error': acc_error,
    }

    myRecEval = RecEvalWeightOpt(**kwargs_receval)
    myRecEval()