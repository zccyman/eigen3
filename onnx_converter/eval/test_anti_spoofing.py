# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/24 9:59
# @File     : test_anti_spoofing.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import os

import cv2
import numpy as np

from eval.face_antispoof import FaceSpoofEvaluator


class PreProcess(object):
    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = kwargs['img_mean']
        self.img_std = kwargs['img_std']
        self.face_size = kwargs['face_size']
        self.swapRB = kwargs['swapRB']
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

def parse_args():
    # model update
    test_models = [
        '/buffer2/zhangcc/trained_models/anti-spoofing/MobileLiteNetB_simplify.onnx',\
        '/buffer2/zhangcc/trained_models/anti-spoofing/MobileNet3_simplify.onnx',\
        '/buffer2/zhangcc/trained_models/anti-spoofing/MobileLiteNetB_brightness_simplify.onnx']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=test_models[1], help='onnx model path')
    parser.add_argument('--calib_dataset', type=str, 
                        default="//buffer/anti_spoofing/CelebA_Spoof/Data/test/", 
                        help='dataset subdir path for quantization calibration')
    parser.add_argument('--val_dataset', type=str, 
                        default="//buffer/anti_spoofing/CelebA_Spoof/Data/test/", 
                        help='dataset subdir path for quantization calibration')
    # parser.add_argument('--image_subdir', type=list,
    #                     default=['5010', '5013', '5015', '5023', '5028', '5033', '5030', '5035', '5051', '5052', '5061',
    #                              '5072'])
    parser.add_argument('--image_subdir', type=list, default=['5010', '5013', '5015', '5072'])
    parser.add_argument('--frame_size', type=list, default=[224, 224])
    parser.add_argument('--face_size', type=list, default=[128, 128])
    parser.add_argument('--mean', type=list, default=[0.5931, 0.4690, 0.4229])
    parser.add_argument('--std', type=list, default=[0.2471, 0.2214, 0.2157])   
    parser.add_argument('--swapRB', type=bool, default=True) 
    parser.add_argument('--results_path', type=str, default='./work_dir/tmpfiles/face_antispoof/')
    parser.add_argument('--spoof_prob_threshold', type=float, default=0.6)
    parser.add_argument('--image_prefix', type=str, default='png')
    parser.add_argument('--roc_save_path', type=str, default='work_dir')
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')
    parser.add_argument('--fp_result', type=bool, default=False)
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=False) 
    parser.add_argument('--log_level', type=int, default=30)      
    parser.add_argument('--error_analyzer', type=bool, default=False)   
    parser.add_argument('--is_calc_error', type=bool, default=False) #whether to calculate each layer error
    ## new features
    parser.add_argument('--offline_quan_mode', type=bool, default=False, help='if true, load offline quantize table')
    parser.add_argument('--offline_quan_tool', type=str, default='NCNN', help='ThirdParty quantize tool name')
    parser.add_argument('--quan_table_path', type=str, default='work_dir/quan_table/NCNN/quantize.table')
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
            
    # build proprcessor
    kwargs_preprocess = {
        "img_mean": args.mean,
        "img_std": args.std,
        'face_size': args.face_size,
        'swapRB': args.swapRB,
    }
    preprocess = PreProcess(**kwargs_preprocess)
    
    # build main processor
    process_args = {
        'log_name': 'process.log',
        'log_level': args.log_level,
        'model_path': args.model_path,
        'parse_cfg': 'config/parse.py',
        'graph_cfg': 'config/graph.py',
        'quan_cfg': 'config/quantize.py',
        'analysis_cfg': 'config/analysis.py',
        'export_cfg': 'config/export_v{}.py'.format(args.export_version),      
        'offline_quan_mode': args.offline_quan_mode,
        'offline_quan_tool': args.offline_quan_tool,
        'quan_table_path': args.quan_table_path,          
        'transform': preprocess,
        'simulation_level': 1, 
        'is_ema': True,
        'device': args.device,
        'fp_result': args.fp_result,
        'ema_value': 0.99, 
        'is_array': False,
        'is_stdout': args.is_stdout,          
        'error_metric': ['L1', 'L2', 'Cosine'], 
        'error_analyzer': args.error_analyzer,
        }

    kwargs_antispoof = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_anti_spoofing.log', 
        'log_level': args.log_level,
        # 'is_stdout': args.is_stdout,                 
        'image_root': args.val_dataset,
        'image_subdir': args.image_subdir,
        'img_prefix': args.image_prefix,
        'gt_json': 'work_dir/tmpfiles/antispoof/gt.json',
        'pred_json': 'work_dir/tmpfiles/antispoof/pred.json',
        'fpred_json': 'work_dir/tmpfiles/antispoof/fpred.json',
        'class_num': 2,
        'transform': preprocess,
        'process_args': process_args,
        'is_calc_error': args.is_calc_error,
        'model_path': args.model_path,
        'frame_size':args.frame_size,
        'conf_threshold':args.spoof_prob_threshold,
        'roc_save_path':args.roc_save_path,
        'acc_error': acc_error,        
        'fp_result': args.fp_result,
        'eval_mode': eval_mode,  # single | dataset        
    }

    evaluator = FaceSpoofEvaluator(**kwargs_antispoof)
    accuracy, tb = evaluator()
    if args.is_calc_error:
        evaluator.collect_error_info()    
    if args.error_analyzer:
        evaluator.error_analysis()         
    if args.export:
        evaluator.export()    
    print(tb)