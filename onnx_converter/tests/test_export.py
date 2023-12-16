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

import cv2
import numpy as np

try:
    from eval import Eval
    from tools import ModelProcess, OnnxruntimeInfer
except Exception:
    from onnx_converter.eval import Eval # type: ignore
    from onnx_converter.tools import ModelProcess, OnnxruntimeInfer # type: ignore


def generate_random_str(randomlength=16, postfix='.jpg'):
  random_str = ''
  base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
  length =len(base_str) -1
  for i in range(randomlength):
    random_str +=base_str[random.randint(0, length)]
  return random_str + postfix


class NetEval(Eval):
    def __init__(self, **kwargs):
        super(NetEval, self).__init__(**kwargs)

        self.input_size = kwargs['input_size']
        self.process_args = kwargs['process_args']

        self.process = ModelProcess(**self.process_args)

    def __call__(self):
        output_names, input_names = self.process.get_output_names(), self.process.get_input_names()
        onnxinferargs = copy.deepcopy(self.process_args)
        onnxinferargs.update(out_names=output_names, input_names=input_names)
        onnxinfer = OnnxruntimeInfer(**onnxinferargs)

        for _ in range(1):
            image_name = generate_random_str()
            if isinstance(self.input_size, dict):
                in_datas = {}
                for input_name, (h, w) in self.input_size.items():
                    in_datas[input_name] = np.random.random((h, w))
            else:
                h, w = self.input_size
                in_datas = np.random.random((h, w, 3))
            self.process.quantize(in_datas, is_dataset=False)
            self.process.set_onnx_graph(onnx_graph=False)
            self.process.dataflow(in_datas, acc_error=True, onnx_outputs=onnxinfer(in_data=in_datas))
            # self.process.checkerror(in_datas, acc_error=False)


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
                img_input = {}
                for input_name, im in img.items():
                    im_ = im[np.newaxis, :].astype(np.float32)
                    if input_name == 'y1':
                        img_input[input_name] = im_.transpose(1, 2, 0)
                    else:
                        img_input[input_name] = im_.transpose(1, 0, 2)
            else:
                img_input = img[np.newaxis, :].astype(np.float32)
                img_input = img_input.transpose(1, 0, 2)

        return img_input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        # default='tmp_sim.onnx',
                        # default='MobileNetv3_classification-sim-remove-expandOp_replace-reduce.onnx',
                        # default='work_dir/MobileNetv3_classification-sim-remove-expandOp_replace-reduce.onnx',
                        # default='work_dir/lstm.onnx',
                        # default='work_dir/lstm_timestep.onnx',
                        # default='model_p2_sim_adj_sim.onnx',
                        # default='/home/henson/dataset/trained_models/net.onnx',
                        default='/home/henson/dataset/trained_models/face-detection/slim_special_Final_simplify_removed_pad.onnx',
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
    # parser.add_argument('--input_size', type=list, default=[320, 320])
    parser.add_argument('--input_size', type=list, default=[320, 256])
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
    parser.add_argument('--log_level', type=int, default=30)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    kwargs_preprocess = {
        "img_mean": [103.53, 116.28, 123.675],
        "img_std": [57.375, 57.12, 58.395],
        'input_size': args.input_size
    }
    preprocess = PreProcess(**kwargs_preprocess)

    export_version = '' if args.export_version > 1 else '_v{}'.format(
        args.export_version)
    process_args = {
        'log_name': 'process.log',
        'log_level': args.log_level,
        'model_path': args.model_path,
        'parse_cfg': 'config/parse.py',
        'graph_cfg': 'config/graph.py',
        # 'quan_cfg': 'config/quantize.py',
        'base_quan_cfg': 'config/quantize.py',
        'quan_cfg': 'config/vision_quantize.py',
        'analysis_cfg': 'config/analysis.py',
        'export_cfg': 'config/export{}.py'.format(export_version),
        'offline_quan_mode': None,
        'offline_quan_tool': None,
        'quan_table_path': None,
        'transform': preprocess,
        'simulation_level': 1,
        'fp_result': True,
        'is_ema': True,
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,
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
    if args.export:
        myClsEval.export()