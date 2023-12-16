# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : nan.qin
# @Company  : SHIQING TECH
# @Time     : 2022/09/01 14:28
# @File     : test_perf.py

# Usage example of imagenet2012 classification
import argparse
import glob
import onnx
import onnxruntime as rt
import os, sys
sys.path.append(os.getcwd())
from eval.imagenet_eval import ClsEval
import cv2
import numpy as np
from simulator.perf_analysis import PerfAnalyzer, encryption_perf_data

class PreProcess(object):
    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = kwargs['img_mean']
        self.img_std = kwargs['img_std']
        self.input_shape = kwargs['input_size']
        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img):
        h, w =self.input_shape
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)[16:240, 16:240, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.img_mean) / self.img_std
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

class OnnxInfer():
    def __init__(self, **kwargs):
        self.model_path = kwargs['model_path']
        self.input_names = kwargs['input_names']
        self.output_names = kwargs['output_names']
        assert os.path.exists(self.model_path), "Error, model path not found!"
        self.model = onnx.load(self.model_path)
        self.sess = self.create_session()

    def create_session(self):
        for layer in self.model.graph.node:
            for output_name in layer.output:
                self.model.graph.output.extend([onnx.ValueInfoProto(name=output_name)])
        return rt.InferenceSession(self.model.SerializeToString(), None)
    
    def __call__(self, in_datas):
        in_data = dict()
        results = dict()
        if isinstance(in_datas, np.ndarray):
            in_datas = [in_datas]
        for i, input_name in enumerate(self.input_names):
            in_data.update({input_name:in_datas[i]})
        output_names = [x.name for x in self.model.graph.output]
        outs = self.sess.run(output_names, in_data)
        for i, out in enumerate(outs):
            results.update({output_names[i]: out})
        for key in in_data:
            results.update({key:in_data[key]})
        return results

class ImageNetDataset(object):
    def __init__(self, **kwargs):
        self.with_preprocess = False
        if "preprocess_args" in kwargs.keys():
            self.preprocessor = PreProcess(**kwargs["preprocess_args"])
            self.with_preprocess = True
        self.prefix = kwargs['image_type']
        self.dataset_dir = kwargs['dataset_dir']
        self.selected_idx_list = kwargs['selected_idx_list'] if 'selected_idx_lsit' in kwargs else [-1]
        self.image_array_list, self.annotation_list = self.parse_dataset()
        self.num_samples = len(self.annotation_list)
        self.start_num = -1
        if "num_samples" in kwargs.keys() and kwargs["num_samples"] <= len(self.annotation_list):
            self.num_samples = kwargs["num_samples"]

    def get_image_array_list(self):
        return self.image_array_list[:self.num_samples]
    
    def get_annotation_list(self):
        return self.annotation_list[:self.num_samples]
    
    def __iter__(self):
        return self

    def __next__(self): 
        self.start_num += 1
        if self.start_num >= self.num_samples:
            self.start_num = -1
            raise StopIteration() 
        return (self.image_array_list[self.start_num], self.annotation_list[self.start_num])
                    
    def parse_dataset(self):
        files = os.listdir(self.dataset_dir)

        image_array_list = list()
        annotation_list = list()        
        for name in sorted(files):# sub-folder
            subfolder = os.path.join(self.dataset_dir, name)
            if not os.path.isdir(subfolder):
                continue
            label = int(name)
            files = sorted(os.listdir(subfolder))
            imgs = [x for x in files if x.endswith(self.prefix)]

            for selected_idx in self.selected_idx_list:
                image = cv2.imread(os.path.join(subfolder, imgs[selected_idx]))
                image = cv2.imread(os.path.join(subfolder, imgs[selected_idx]))
                if self.with_preprocess:
                    image = self.preprocessor(image) 
                image_array_list.append(image)
                annotation_list.append(label)
        return image_array_list, annotation_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/home/ts300026/workspace/dataset/imagenet2012/')
    parser.add_argument('--quan_dataset_path', type=str, default='/home/ts300026/workspace/dataset/imagenet2012/')
    parser.add_argument('--model_path', type=str, default ='/home/ts300026/workspace/model_zoo/onnx/imagenet/ResNet34_ImageNet_classification_sim.onnx')
    parser.add_argument('--input_size', type=list, default=[256, 256])
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--fp_result', type=bool, default=False)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)
    parser.add_argument('--log_level', type=int, default=30)
    parser.add_argument('--is_calc_error', type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    encrypt_flag = False
    if encrypt_flag:
        encryption_perf_data('perf_data')
    args = parse_args()
    if args.debug:
        eval_mode = 'single'
        acc_error = False
    else:
        eval_mode = 'dataset'
        acc_error = True

    kwargs_preprocess = {
        "img_mean": [123.675, 116.28, 103.53],
        "img_std": [58.395, 57.12, 57.375],
        'input_size': args.input_size
    }
    preprocess = PreProcess(**kwargs_preprocess)

    img_prefix = 'JPEG'
    image_files = []
    selected_sample_num_per_class = 1
    image_subdir = [idx for idx in range(0, 1000, 1)]
    for idx in image_subdir:
        tmpfiles = sorted(
            glob.glob(
                '{}/{}/*.{}'.format(args.quan_dataset_path, idx, img_prefix)))[:selected_sample_num_per_class]
        image_files.extend(tmpfiles)
    export_version = '' if args.export_version > 1 else '_v{}'.format(args.export_version)
    process_args = {
        'log_name': 'process.log',
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
        'fp_result': args.fp_result,
        'transform': preprocess,
        'simulation_level': 1,
        'is_ema': True,
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,
        'error_metric': ['L1', 'L2', 'Cosine'],
    }

    kwargs_clseval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_imagenet.log',
        'log_level': args.log_level,
        'is_stdout': args.is_stdout,
        'image_files': image_files,
        'image_path': args.dataset_path,
        'image_subdir': image_subdir,
        'selected_sample_num_per_class': selected_sample_num_per_class,
        'img_prefix': img_prefix,
        'gt_json': 'work_dir/tmpfiles/classification/gt.json',
        'pred_json': 'work_dir/tmpfiles/classification/pred.json',
        'fpred_json': 'work_dir/tmpfiles/classification/fpred.json',
        'fake_data': False,
        'class_num': 1000,
        'transform': preprocess,
        'process_args': process_args,
        'is_calc_error': args.is_calc_error,
        'acc_error': acc_error,
        'fp_result': args.fp_result,
        'eval_mode': eval_mode,
        'model_path': args.model_path,
    }

    myClsEval = ClsEval(**kwargs_clseval)
    accuracy, tb = myClsEval()
    myClsEval.export()
    perf_estimator = PerfAnalyzer(model_exporter = myClsEval.process.model_export, \
        chip_model='5050', ref_data_dir='perf_data/', mem_addr = 'l2',
        encrypt_flag = encrypt_flag)
    perf_estimator()