# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : nan.qin
# @Company  : SHIQING TECH
# @Time     : 2022/09/01 14:28
# @File     : error_analysis.py

from abc import abstractmethod
import numpy as np
from tqdm import tqdm
import os
import numpy as np
import cv2
import sys
print(sys.path)
# try:
#     from utils import Registry
# except:
#     from onnx_converter import Registry
print(sys.path)
print(os.getcwd())
sys.path.append(os.getcwd())
from OnnxConverter import OnnxConverter

from simulator.error_analysis import ErrorAnalyzer
#from simulator.perf_analysis import PerfAnalyzer



class BaseEvaluator(object):
    def __init__(self, **kwargs):
        super(BaseEvaluator, self).__init__()
        self.converter = kwargs['onnx_converter']
        self.dataset = kwargs['dataset_eval']
        self.output_names = kwargs['output_name_list']
    
    @abstractmethod
    def postprocess(self, result):
        pass

    @abstractmethod
    def update_result(self, label, pred_onnx, pred_cvt):
        pass

    @abstractmethod
    def calclate_final_result(self):
        pass

    @abstractmethod
    def print_result(self):
        pass

    def __call__(self):
        for (img, label) in iter(self.dataset):
            results = self.converter.model_simulation(img)
            results_onnx, results_cvt = results['result_onnx'],results['result_converter']
            pred_onnx = self.postprocess(results_onnx)
            pred_cvt = self.postprocess(results_cvt)    
            self.update_result(label, pred_onnx, pred_cvt)
        self.calclate_final_result()
        self.print_result()


class ImagenetEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        super(ImagenetEvaluator, self).__init__(**kwargs)
        self.top1_onnx, self.top5_onnx = 0, 0
        self.top1_int8, self.top5_int8 = 0, 0

    def postprocess(self, result):
        # get output tensor
        result = result[self.output_names[0]]
        # do softmax
        result = result.flatten()
        result = np.exp(result - np.max(result)) / np.sum(np.exp(result - np.max(result)))
        # find label
        pred_top_5 = np.argsort(result)[::-1][:5]
        return pred_top_5
    
    def update_result(self, label, result_onnx, result_cvt):
        if label == result_onnx[0]:
            self.top1_onnx += 1
        if label in result_onnx:
            self.top5_onnx += 1
        if label == result_cvt[0]:
            self.top1_int8 += 1
        if label in result_cvt:
            self.top5_int8 += 1

    def calclate_final_result(self):
        self.top1_int8 /= self.dataset.num_samples
        self.top1_onnx /= self.dataset.num_samples
        self.top5_int8 /= self.dataset.num_samples
        self.top5_onnx /= self.dataset.num_samples

    def print_result(self):
        print("Evaluate_model: %s\n"%(self.converter.model_path))
        print(" result  |    top1    |     top5    |\n")
        print(" onnx    |   %.6f   |    %.6f    |\n"%(self.top1_onnx, self.top5_onnx))
        print("converter|   %.6f   |    %.6f    |\n"%(self.top1_int8, self.top5_int8))
        result_file_name = "evaluate_result_%s-%s.txt"%(os.path.split(self.converter.model_path)[-1].split(".onnx")[0],\
            self.converter.calib_method)
        if not os.path.exists('work_dir/imagenet/'):
            os.makedirs('work_dir/imagenet/')
        f=open(os.path.join("work_dir/imagenet/",result_file_name), 'w')
        f.write("Evaluate_model: %s\n"%(self.converter.model_path))
        f.write(" result  |    top1    |     top5    |\n")
        f.write(" onnx    |   %.6f   |    %.6f    |\n"%(self.top1_onnx, self.top5_onnx))
        f.write("converter|   %.6f   |    %.6f    |\n"%(self.top1_int8, self.top5_int8))
        f.close()



class PreProcess(object):
    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = kwargs['image_mean']
        self.img_std = kwargs['image_std']
        self.input_shape = kwargs['input_shape']
        #self.swapRB = kwargs['swapRB']

    def __call__(self, img):
        c, w, h=self.input_shape
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)[16:240, 16:240, :]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.img_mean) / self.img_std
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

class Dataset(object):
    def __init__(self, **kwargs):
        self.with_preprocess = False
        if "preprocess_args" in kwargs.keys():
            self.preprocessor = PreProcess(**kwargs["preprocess_args"])
            self.with_preprocess = True
        self.image_array_list, self.annotation_list = self.parse_dataset(**kwargs)
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
                    
    def parse_dataset(self, **kwargs):
        dataset_dir = kwargs['dataset_dir']
        prefix = kwargs['image_type']
        files = os.listdir(dataset_dir)

        image_array_list = list()
        annotation_list = list()        
        for name in sorted(files):
            if not name.endswith(prefix):
                continue
            name_info = name.split(".")[0].split('---')[0]
            label = int(name_info)
            image = cv2.imread(os.path.join(dataset_dir, name))
            if self.with_preprocess:
                image = self.preprocessor(image) 
            image_array_list.append(image)
            annotation_list.append(label)
        return image_array_list, annotation_list   

if __name__ == '__main__':
    # args of dataset and preprocess
    args_preprocess = {
        "input_shape":[3, 256, 256],
        "image_mean":[123.675, 116.28, 103.53],
        "image_std":[58.395, 57.12, 57.375],
    }
    args_data_calib = {
        "dataset_dir" : "/home/ts300026/workspace/dataset/imagenet_calib", 
        "image_type" : "JPEG",
        "preprocess_args":args_preprocess
    }
    args_dataset_eval = {
        "dataset_dir": "/home/ts300026/workspace/dataset/imagenet_val", 
        "image_type" : "JPEG",
        "preprocess_args":args_preprocess
    }
    args_data_analyse={
        "dataset_dir":"/home/ts300026/workspace/dataset/imagenet_val",
        "image_type" : "JPEG",
        "preprocess_args":args_preprocess
    }
    # args of converter
    # test model list
    test_model_list = [        
        '/home/ts300026/workspace/model_zoo/onnx/imagenet/MobileNetv3_ImageNet_classification_raw_simplify.onnx',   #11M
        #'/home/ts300026/workspace/model_zoo/onnx/imagenet/Ghostnet_ImageNet_classification-sim.onnx',               #20M
        '/home/ts300026/workspace/model_zoo/onnx/imagenet/ResNet34_ImageNet_classification_sim.onnx',               #85M
        '/home/ts300026/workspace/model_zoo/onnx/imagenet/ResNeXt101_32x4d_ImageNet_classification-simplify.onnx',  #170M   
        '/home/ts300026/workspace/model_zoo/onnx/imagenet/HRNet_ImageNet_classification-sim.onnx',                  #83M
        '/home/ts300026/workspace/model_zoo/onnx/imagenet/VGG13_ImageNet_classification.onnx',                      #519M
    ]
    for i, test_model in enumerate(test_model_list):                  
        args_cvt = {
            "chip_model": "AT5050_C_EXTEND",
            "model_path": test_model,
            "quantization_args":
            {  
                "out_type" : "int8",
                "bit_width": 8,
                "method":
                {
                    "feature": "symm", 
                    "weight": ["symm", "per_tensor"]
                },
                "process_scale":"intscale",
            },
            "calib_method" : "kld", # "kld", "smooth"
            "log_args":
            {
                "log_name":"process.log",
                "log_level": 50
            },
            "layer_error_args":
            {
                "check_error": True,
                "acc_error": True,
                "metrics_list" : ["Cosine", "L1", "L2"]
            },
            "is_simplify":False
        }
        
        # model convert
        dataset_calib = Dataset(**args_data_calib)
        args_cvt.update({'dataset_calib': dataset_calib})
        converter = OnnxConverter(**args_cvt)
        converter.run_model_convert()

        # quan model evaluate
        evaluator_args={
            "evaluator_name" : "ImagenetEvaluator",
            "output_name_list": ["output"],
        }

        dataset_eval = Dataset(**args_dataset_eval)
        evaluator_args.update({'onnx_converter': converter, 'dataset_eval': dataset_eval})
        evaluator = ImagenetEvaluator(**evaluator_args)
        evaluator()     

        # error analyzer 
        # args_analyse = {
        #     'quan_graph':converter.quan_graph,
        #     'onnx_infer':converter.post_quan.onnx_infer,
        #     'output_dir':"work_dir/imagenet/" + os.path.split(args_cvt['model_path'])[-1].split(".onnx")[0],
        #     'simulation_level': None
        # }

        # dataset_analyse = Dataset(**args_data_analyse)
        # args_analyse.update({'dataset': dataset_analyse})
        # analyzer = ErrorAnalyzer(**args_analyse)
        # analyzer()
