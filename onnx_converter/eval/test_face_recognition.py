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

from eval.face_recognition import RecEval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/buffer/faces_emore')
    parser.add_argument('--dataset_name', type=str, default='lfw')
    parser.add_argument('--model_path', type=str, 
                        default='/buffer/trained_models/face-recognition/mobilefacenet_pad_qat_new_2.onnx',
                        # default='work_dir/mobilefacenet_method_3_simplify_bias_correction.onnx',
                        # default='work_dir/mobilefacenet_method_3_simplify_cross_layer_equalization.onnx',
                        )
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--fp_result', type=bool, default=False)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=False) 
    parser.add_argument('--log_level', type=int, default=30)    
    parser.add_argument('--error_analyzer', type=bool, default=False)       
    parser.add_argument('--is_calc_error', type=bool, default=False) #whether to calculate each layer error
    
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
        'simulation_level': 1,
        'is_ema': True,
        'device': args.device,
        'fp_result': args.fp_result,
        'is_simplify': True,
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,          
        'error_analyzer': args.error_analyzer,
        'error_metric': ['L1', 'L2', 'Cosine'], ## Cosine | L2 | L1
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
        "eval_mode": eval_mode,  # [single | dataset] quatize
        'acc_error': acc_error,
    }

    myRecEval = RecEval(**kwargs_receval)
    accuracy, tb = myRecEval()
    if args.error_analyzer:
        myRecEval.error_analysis()     
    if args.export:
        myRecEval.export()
    print(tb)
