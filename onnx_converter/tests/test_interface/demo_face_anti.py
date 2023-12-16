# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : SHIQING TECH
# @Company  : SHIQING TECH
# @Time     : 2023/7/16
# @File     : test_OnnxConverter.py

# from __future__ import annotations

import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

from abc import abstractmethod
import argparse
from OnnxConverter import OnnxConverter

import os
import json
import sys
import cv2
import numpy as np

class BaseDataset(object):
    def __init__(self, **kwargs):          
        super(BaseDataset, self).__init__()
        self.image_array_list, self.annotation_list = self.parse_dataset(**kwargs)
        self.num_samples = len(self.annotation_list)
        self.start_num = -1
        if "num_samples" in kwargs.keys() and kwargs["num_samples"] <= len(self.annotation_list):
            self.num_samples = kwargs["num_samples"]
            
    @staticmethod
    def preprocess(img):
        face_resize = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        face_resize = cv2.cvtColor(face_resize, cv2.COLOR_BGR2RGB)
        face_resize = face_resize.astype(np.float32) / 255
        face_resize = (face_resize - [0.5931, 0.4690, 0.4229]) / [0.2471, 0.2214, 0.2157]
        face_resize = np.transpose(face_resize, [2, 0, 1])
        face_resize = np.expand_dims(face_resize, axis=0)
        return face_resize.astype(np.float32)
    
    ### parse_dataset is used to generate a image_array_list and an annotation list
    ### You should overload this function which is a virtual function defined in BaseDataset 
    def parse_dataset(self, **kwargs):
        dataset_dir = kwargs['dataset_dir']
        prefix = kwargs['image_type']
        files = os.listdir(dataset_dir)

        image_array_list = list()
        annotation_list = list()        
        for name in sorted(files):
            if not name.endswith(prefix):
                continue
            name_info = name.split(".")[0].split('_')
            label = name_info[-1]
            image = cv2.imread(os.path.join(dataset_dir, name))
            #image = self.preprocess(image) 
            image_array_list.append(image)
            annotation_list.append(label)
        return image_array_list, annotation_list 
    
    def get_image_array_list(self):
        return self.image_array_list[:self.num_samples]
    
    def get_annotation_list(self):
        return self.annotation_list[:self.num_samples]
    
    def __iter__(self):
        return self

    def __next__(self): 
        self.start_num += 1
        if self.start_num >= self.num_samples:
            raise StopIteration() 

        return (self.image_array_list[self.start_num], self.annotation_list[self.start_num])
    
class AntispoofEvaluator(object):
    def __init__(self, **kwargs):
        super(AntispoofEvaluator, self).__init__()
        self.dataset = kwargs['dataset_eval']
        self.output_names = ["output"]
 
        self.conf_threshold = 0.6
        self.results_cvt = {"num_tp":0, "num_fp":0, "num_tn":0, "num_fn":0}

    def postprocess(self, result):
        # get output tensor
        result = result[self.output_names[0]]
        # do softmax
        result = result.flatten()
        result = np.exp(result - np.max(result)) / np.sum(np.exp(result - np.max(result)))
        # find label
        target_idx = np.argmax(result)
        if target_idx == 1 and result[target_idx] >= self.conf_threshold:
            return "spoof"
        else:
            return "live"
    
    def update_result(self, label, result_cvt):
        if label == "spoof":
            if result_cvt == "spoof":
                self.results_cvt['num_tp']+=1
            else:
                self.results_cvt["num_fn"]+=1   
        else:
            if result_cvt == "live":
                self.results_cvt['num_tn']+=1
            else:
                self.results_cvt["num_fp"]+=1    

    def calclate_final_result(self):
        self.recall_cvt = self.results_cvt["num_tp"] / (self.results_cvt["num_tp"] + self.results_cvt["num_fn"])\
            if self.results_cvt["num_tp"] + self.results_cvt["num_fn"] > 0 else 0
        self.precision_cvt = self.results_cvt["num_tp"] / (self.results_cvt["num_tp"] + self.results_cvt["num_fp"])\
            if self.results_cvt["num_tp"] + self.results_cvt["num_fp"] > 0 else 0

    def print_result(self):
        print(" result  |    recall    |    precision   |\n")
        print("converter|   %.6f   |    %.6f    |\n"%(self.recall_cvt, self.precision_cvt))
        
    def __call__(self, converter):
        for (img, label) in iter(self.dataset):
            results = converter.model_simulation(img)
            results_cvt = results['result_converter']
            pred_cvt = self.postprocess(results_cvt)    
            self.update_result(label, pred_cvt)
        self.calclate_final_result()
        self.print_result()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_json', default='arguments_face_antispoof.json', type=str)
    parser.add_argument('--model_export', type=bool, default=True)
    parser.add_argument('--perf_analyse', type=bool, default=True)
    parser.add_argument('--vis_qparams', type=bool, default=True)
    parser.add_argument('--mem_addr', type=str, default='psram')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("******* Start model convert *******")
    # parse user defined arguments
    argsparse = parse_args()
    args_json_file = argsparse.args_json
    flag_model_export = argsparse.model_export
    flag_perf_analyse = argsparse.perf_analyse
    flag_vis_qparams = argsparse.vis_qparams
    assert os.path.exists(args_json_file), "Please check argument json exists"
    args = json.load(open(args_json_file, 'r'))
    args_cvt = args['converter_args']
    error_analyzer = args_cvt['error_analyzer']
    
    calibration_dataset_dir = args_cvt["calibration_dataset_dir"]
    evaluation_dataset_dir = args_cvt["evaluation_dataset_dir"]
    cali_data_dir = "/home/shiqing/Downloads/test_package/converter-package/customer_release/images/face_antispoof/calib"
    eval_dataset_eval = "/home/shiqing/Downloads/test_package/converter-package/customer_release/images/face_antispoof/eval/"
    dataset = BaseDataset(dataset_dir=cali_data_dir, image_type="png")
    args_cvt["transform"] = dataset.preprocess
    # Build onnx converter
    converter = OnnxConverter(**args_cvt)     
    
    converter.load_model("/home/shiqing/Downloads/test_package/converter-package/customer_release/models/MobileNet3_simplify.onnx")
    
    # Calibration
    converter.calibration(cali_data_dir)
    print("calibration done!")
    
    # Build evaluator
    eavl_dataset = BaseDataset(dataset_dir=eval_dataset_eval, image_type="png")
    evaluator = AntispoofEvaluator(dataset_eval=eavl_dataset)

    # Evaluate quantized model accuracy
    evaluator(converter=converter)
    print("******* Finish model evaluate *******")
    
    if flag_vis_qparams:
        converter.visualize_qparams()
        print("******* Finish qparams visualization *******") 

    if flag_model_export:
        # Run model convert and export(optional)
        converter.model_export()
        print("******* Finish model convert *******")
    
    # performance analysis
    if flag_perf_analyse:
        time = converter.perf_analyze(mem_addr = argsparse.mem_addr)
        print("******* The estimated time cost is %f ms *******"%(time/1000))
        print("******* Finish performance analysis *******")