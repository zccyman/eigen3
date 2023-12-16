# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/2/15 9:58
# @File     : test_export.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import copy
import random
import onnx

import cv2
import numpy as np

try:
    from eval import Eval
    from tools import ModelProcess
    from utils import generate_random
except:
    from onnx_converter.eval import Eval # type: ignore
    from onnx_converter.tools import ModelProcess # type: ignore
    from onnx_converter.utils import generate_random # type: ignore


def generate_random_str(randomlength=16, postfix='.jpg'):
  random_str = ''
  base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
  length =len(base_str) -1
  for i in range(randomlength):
    random_str +=base_str[random.randint(0, length)]
  return random_str + postfix


class NetEval(Eval): # type: ignore
    def __init__(self, **kwargs):
        super(NetEval, self).__init__(**kwargs)

        self.input_size = kwargs['input_size']
        self.process_args = kwargs['process_args']
        
        self.process = ModelProcess(**self.process_args)
        self.process.set_onnx_graph(onnx_graph=False)
        
    def __call__(self):
        for _ in range(1):
            image_name = generate_random_str()
            if isinstance(self.input_size, dict):
                in_datas = {}
                for input_name, size in self.input_size.items():
                    in_datas[input_name] = generate_random(size, seed=0, range=[0, 255], method='randn').astype(np.uint8)
            else:
                h, w = self.input_size
                in_datas = generate_random([h, w, 3], seed=0, range=[0, 255], method='randn').astype(np.uint8)
                # in_datas = cv2.imread('danrenbudaikouzhaozhengchang.mp4_048_snapshot_003.jpg')
            self.process.quantize(in_datas, is_dataset=False)
            # self.process.set_onnx_graph(onnx_graph=False)
            # self.process.dataflow(
            #     in_datas, acc_error=True, 
            #     onnx_outputs=self.process.post_quan.onnx_infer(in_datas))
            self.process.checkerror(in_datas, acc_error=True)
            self.collect_error_info()


class PreProcess(object):

    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = kwargs['img_mean']
        self.img_std = kwargs['img_std']
        self.input_size = kwargs['input_size']
        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans
        
    def __call__(self, img):
        if not isinstance(self.input_size, dict):
            img_resize = cv2.resize(img, self.input_size)
            img_input = img_resize.astype(np.float32) / 255
            img_mean = np.array(self.img_mean, dtype=np.float32) / 255
            img_std = np.array(self.img_std, dtype=np.float32) / 255
            img_input = (img_input - img_mean) / img_std
            # expand dims
            img_input = np.transpose(img_input, [2, 0, 1])
            img_input = np.expand_dims(img_input, axis=0)
        else:
            # img_input = img_input[:, 0, :, :]
            # img_input = img_input.reshape(img_input.shape[0], -1)
            # img_input = img_input.reshape(1, 1, 20)
            
            if isinstance(img, dict):
                img_input = dict()
                for input_name, im in img.items():
                    im_ = im[np.newaxis, :].astype(np.float32)
                    img_input[input_name] = im_
            else:
                img_input = img[np.newaxis, :].astype(np.float32)
                img_input = img_input.transpose(1, 0, 2)

        return img_input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        # default='tmp_sim.onnx',
                        default="landmark_clip_sim.onnx",
                        # default='MobileNetv3_classification-sim-remove-expandOp_replace-reduce.onnx',
                        # default='work_dir/MobileNetv3_classification-sim-remove-expandOp_replace-reduce.onnx',
                        # default='work_dir/lstm.onnx',
                        # default='work_dir/lstm_timestep.onnx',
                        # default='model_p2_sim_adj_sim.onnx',
                        # default='/home/henson/dataset/trained_models/net.onnx',
                        # default='/home/henson/dataset/trained_models/face-detection/slim_special_Final_simplify_removed_pad.onnx',
                        # default='/home/henson/dataset/trained_models/face-detection/slimv2_Final_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/face-detection/shufflenetv2_Final_simplify.onnx',

                        # [will export]
                        # default='/home/henson/dataset/trained_models/pedestrian-detection/yolov5n_320_v1_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/pedestrian-detection/yolov5s_320_v1_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/pedestrian-detection/yolov5s_320_v2_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/pedestrian-detection/nanodet_plus_m_1.0x_320_v1_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/pedestrian-detection/nanodet_plus_m_1.0x_320_v2_simplify.onnx',
                        # [will export]
                        # default='/home/henson/dataset/trained_models/pedestrian-detection/nanodet_plus_m_1.5x_320_v1_simplify.onnx',
                        
                        # default='/home/henson/dataset/trained_models/object-detection/nanodet_1.5x_320_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/object-detection/nanodet_1.0x_320_simplify.onnx',

                        # default='test_gru_sim.onnx',
                        # default='test_gru_sim-after.onnx',
                        # default='model_simplify_opsion_14_offline.onnx',
                        # default='test_gru_sim_offline.onnx',
                        # default='test_torch_gru.onnx',
                        # default='test_torch_lstm.onnx',
                        # default='mobilenetv2_unet_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/anti-spoofing/MobileNet3_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/anti-spoofing/MobileLiteNetB_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/anti-spoofing/MobileLiteNetB_brightness_simplify.onnx',
                        
                        # default='/home/henson/dataset/trained_models/face-recognition/mobilefacenet_pad_qat_new_2.onnx',
                        # default='/home/henson/dataset/trained_models/face-recognition/mobilefacenet_pad_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/face-recognition/mobilefacenet_method_1_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/face-recognition/mobilefacenet_method_2_simplify.onnx',
                        # default='/home/henson/dataset/trained_models/face-recognition/mobilefacenet_method_3_simplify.onnx',
                        
                        # default='/home/henson/dataset/trained_models/classification/resnet34_cls1000.onnx',
                        )
    # [28, 28], [320, 256], [320, 320], [128, 128], [112, 112], [192, 192], [192, 256]
    parser.add_argument('--input_size', type=list, default=[256, 256])
    # parser.add_argument('--input_size', type=list, default=[20, 20])
    # parser.add_argument('--input_size', type=list, default=[640, 640])
    # parser.add_argument('--input_size', type=list, default=[320, 320])
    # parser.add_argument('--input_size', type=list, default=[128, 128])
    # parser.add_argument('--input_size', type=list, default=[256, 144]) 
    # parser.add_argument('--input_size', type=dict, 
    #                     default={
    #                         'input:0':[1, 38], 
    #                         'h0:0':[24],
    #                     },
    #                     )    
    # parser.add_argument('--input_size', type=dict, 
    #                     default={
    #                         'input:0':[1, 38], 
    #                         'const_fold_opt__50':[1, 24],
    #                     },
    #                     )  
    # parser.add_argument('--input_size', type=dict, # type: ignore
    #                     default={
    #                         'main_input:0':[1, 38], 
    #                         'const_fold_opt__114':[1, 24],
    #                         'const_fold_opt__113':[1, 96],
    #                         'const_fold_opt__112':[1, 48],
    #                         },
    #                     )                
    # parser.add_argument('--input_size', type=dict, 
    #                     default={
    #                         'input':[1, 257], 
    #                         'h0':[1, 128],
    #                         # 'c0':[1, 128],
    #                     },
    #                     )                           
    # parser.add_argument('--input_size', type=dict, 
    #                     default={
    #                         'input1':[5, 10], 
    #                         'input2':[1, 20], 
    #                         'onnx::LSTM_2':[1, 20]},
    #                     )
    # parser.add_argument('--input_size', type=dict, 
    #                     default={
    #                         'input1':[2, 20], 
    #                         'input2':[2, 20], 
    #                         'onnx::Gather_2':[2, 20]},
    #                     )   
    # parser.add_argument('--input_size', type=dict, 
    #                     default={
    #                             'h1_in':[1, 128], 
    #                             'c1_in':[1, 128],
    #                             'h2_in':[1, 128],
    #                             'c2_in':[1, 128],
    #                             # 'mag':[1, 257], 
    #                             'y1':[1, 512], 
    #                         },
    #                     )                                              
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--export', type=bool, default=True)
    parser.add_argument('--is_stdout', type=bool, default=True)
    parser.add_argument('--log_level', type=int, default=10)
    parser.add_argument('--error_analyzer', type=bool, default=False) 
    
    args = parser.parse_args()
    return args

def offline_process(model_path='model_simplify_opsion_14.onnx', save_path='model_simplify_opsion_14_offline.onnx'):
    from checkpoint.preprocess import OnnxProcessOffLine
    model=onnx.load(model_path)
    offline_operations = ['p2o.Resize.8']
    engine = OnnxProcessOffLine(model=model, delete_node_names=offline_operations, save_path=save_path)
    model = engine.process()
    del engine

def offline_process_gru_model(model_path='test_gru_sim.onnx', save_path='test_gru_sim_offline.onnx'):
    from checkpoint.preprocess import OnnxProcessOffLine
    model=onnx.load(model_path)
    offline_operations = []
    engine = OnnxProcessOffLine(model=model, delete_node_names=offline_operations, save_path=save_path)
    model = engine.process()
    del engine

if __name__ == '__main__':
    # offline_process_gru_model()

    args = parse_args()

    kwargs_preprocess = {
        "img_mean": [103.53, 116.28, 123.675],
        "img_std": [57.375, 57.12, 58.395],
        'input_size': args.input_size
    }
    preprocess = PreProcess(**kwargs_preprocess)

    process_args = {
        'log_name': 'process.log',
        'log_level': args.log_level,
        'model_path': args.model_path,
        'parse_cfg': 'config/parse.py',
        'graph_cfg': 'config/graph.py',
        'base_quan_cfg': 'config/quantize.py',
        'quan_cfg': 'config/voice_quantize.py',
        # 'quan_cfg': 'config/vision_quantize.py',
        'analysis_cfg': 'config/analysis.py',
        'export_cfg': 'config/export_v{}.py'.format(args.export_version),
        'offline_quan_mode': None,
        'offline_quan_tool': None,
        'quan_table_path': None,
        'device': 'cpu',
        'transform': preprocess,
        'simulation_level': 1,
        'fp_result': True,
        'is_ema': True,
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,
        'error_analyzer': args.error_analyzer,
        'error_metric': ['L1', 'L2', 'Cosine'],  # Cosine | L2 | L1
    }

    kwargs_clseval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_net.log',
        'log_level': args.log_level,
        'is_stdout': args.is_stdout,
        'input_size': args.input_size,
        'process_args': process_args,
    }

    myClsEval = NetEval(**kwargs_clseval)
    myClsEval()
    if args.error_analyzer:
        myClsEval.error_analysis()     
    if args.export:
        myClsEval.export()