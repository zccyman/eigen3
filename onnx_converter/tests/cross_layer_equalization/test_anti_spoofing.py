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
import copy
import glob

import cv2
import numpy as np

try:
    from eval.face_antispoof import FaceSpoofEvaluator
    from utils import Object
    from tools import WeightOptimization    
except:
    from onnx_converter.eval.face_antispoof import FaceSpoofEvaluator
    from onnx_converter.utils import Object
    from onnx_converter.tools import WeightOptimization

class FaceSpoofEvaluatorWeightOpt(Object):
    def __init__(self, **kwargs):
        super(FaceSpoofEvaluatorWeightOpt, self).__init__(**kwargs)
        self.image_root = kwargs['image_root']
        self.image_subdir = kwargs['image_subdir']
        self.img_prefix = kwargs['img_prefix']
        self.gt_json = kwargs['gt_json']
        self.pred_json = kwargs['pred_json']
        self.fpred_json = kwargs['fpred_json']
        self.class_num = kwargs['class_num']
        self.preprocess = kwargs['transform']
        self.process_args = kwargs['process_args']
        self.is_calc_error = kwargs['is_calc_error']
        self.is_stdout = self.process_args['is_stdout']
        self.acc_error = kwargs['acc_error']
        self.eval_mode = kwargs['eval_mode']
        self.fp_result = kwargs['fp_result']        
        self.model_path = kwargs['model_path']
        self.frame_size = kwargs['frame_size']
        self.spoof_conf_threshold = kwargs['conf_threshold']
        self.roc_save_path = kwargs['roc_save_path']
        self.target_accumulate = np.array([])
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)

        os.system('rm -rf {}'.format(self.gt_json))
        os.system('rm -rf {}'.format(self.pred_json))

        json_path, _ = os.path.split(self.gt_json)
        os.makedirs(json_path, exist_ok=True)

        if self.image_root:
            self.parse_image_and_annot()

        self.weight_optimization = kwargs["weight_optimization"]
        self.process_args_wo = copy.deepcopy(kwargs['process_args'])
        self.process_args_wo["quan_cfg"] = "config/quantize_fp.py"
        self.process_wo = WeightOptimization(**self.process_args_wo)
    
    def parse_image_and_annot(self):
        assert os.path.exists(self.image_root), "Evaluation image path do not exist\n"
        self.face_data = list()
        self.image_annot_list = list()
        for subdir in self.image_subdir:
            image_paths = glob.glob(os.path.join(
                self.image_root, subdir, '*', '*.{}'.format(self.img_prefix)))
            for image_path in image_paths:
                image_name = os.path.basename(image_path).split('.')[0]
                image_info_file = image_path.replace('.' + self.img_prefix, '_BB.txt')
                with open(image_info_file, "r") as f:
                    for line in f.readlines():
                        line = line.rstrip().split(' ')
                        bbox = [int(line[i]) for i in range(len(line) - 1)]
                        break
                label = 'live' if 'live' in image_path else 'spoof'
                frame = cv2.imread(image_path)
                # crop face region
                h,w=frame.shape[:2]
                [h_in, w_in] = self.frame_size
                x, y = max(0,int(bbox[0]*w/w_in)), max(0,int(bbox[1]*h/h_in))
                dw, dh = max(0,int(bbox[2]*w/w_in)), max(0,int(bbox[3]*h/h_in))
                crop_face = frame[y:min(h, y+dh),x : min(w, x+dw), :]
                
                info = dict()
                info.update({"image_path" : image_path, 
                            "label" : label,
                            'face_data': crop_face})
                self.image_annot_list.append(info)
                self.face_data.append(crop_face)

    def __call__(self, quan_dataset_path=None):
        if self.weight_optimization in ["cross_layer_equalization"]:
            skip_layer_names=[
            ]
            eval("self.process_wo." + self.weight_optimization)(skip_layer_names=skip_layer_names)
        else:
            eval("self.process_wo." + self.weight_optimization)(self.face_data)


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
        'trained_models/anti-spoofing/MobileLiteNetB_simplify.onnx',\
        'trained_models/anti-spoofing/MobileNet3_simplify.onnx',\
        'trained_models/anti-spoofing/MobileLiteNetB_brightness_simplify.onnx']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        # default=test_models[1], 
                        default="work_dir/MobileNet3_simplify_bias_correction.onnx", 
                        help='onnx model path')
    parser.add_argument('--calib_dataset', type=str, default="/buffer/anti_spoofing/CelebA_Spoof/Data/test/", help='dataset subdir path for quantization calibration')
    parser.add_argument('--val_dataset', type=str, default="/buffer/anti_spoofing/CelebA_Spoof/Data/test/", help='dataset subdir path for quantization calibration')
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
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True) 
    parser.add_argument('--log_level', type=int, default=30)    
    parser.add_argument('--is_calc_error', type=bool, default=False) #whether to calculate each layer error
    ## new features
    parser.add_argument('--offline_quan_mode', type=bool, default=False, help='if true, load offline quantize table')
    parser.add_argument('--offline_quan_tool', type=str, default='NCNN', help='ThirdParty quantize tool name')
    parser.add_argument('--quan_table_path', type=str, default='work_dir/quan_table/NCNN/quantize.table')
    # bias_correction cross_layer_equalization
    parser.add_argument('--weight_optimization', type=str, default="cross_layer_equalization")
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
    
    export_version = '' if args.export_version > 1 else '_v{}'.format(args.export_version)
    # build main processor
    process_args = {
        'log_name': 'process_{}.log'.format(args.weight_optimization),
        'log_level': args.log_level,
        'model_path': args.model_path,
        'parse_cfg': 'config/parse.py',
        'graph_cfg': 'config/graph.py',
        'quan_cfg': 'config/quantize.py',
        'analysis_cfg': 'config/analysis.py',
        'export_cfg': 'config/export{}.py'.format(export_version),      
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
        "is_fused_act": True,
        }

    kwargs_antispoof = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_anti_spoofing.log', 
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
        "weight_optimization": args.weight_optimization,
        'model_path': args.model_path,
        'frame_size':args.frame_size,
        'conf_threshold':args.spoof_prob_threshold,
        'roc_save_path':args.roc_save_path,
        'acc_error': acc_error,        
        'fp_result': args.fp_result,
        'eval_mode': eval_mode,  # single | dataset        
    }

    evaluator = FaceSpoofEvaluatorWeightOpt(**kwargs_antispoof)
    evaluator()
    if args.export:
        evaluator.export()    