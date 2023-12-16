#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/29 20:05:10
#@File    : face_antispoof.py

import copy
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from eval import Eval

try:
    from tools import ModelProcess, OnnxruntimeInfer
    from utils import Object
except:
    from onnx_converter.tools import ModelProcess, OnnxruntimeInfer
    from onnx_converter.utils import Object


class FaceSpoofEvaluator(Eval):
    def __init__(self, **kwargs):
        super(FaceSpoofEvaluator, self).__init__(**kwargs)
                
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
        # self.is_stdout = self.process_args['is_stdout']
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

        self.process = ModelProcess(**self.process_args)
        if self.is_calc_error:
            self.process.set_onnx_graph(False)
        if self.image_root:
            self.parse_image_and_annot()
            self.create_gt_json()

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

    def create_gt_json(self):
        # output_names, input_names = self.process.get_output_names(), self.process.get_input_names()
        # onnxinferargs = copy.deepcopy(self.process_args)
        # onnxinferargs.update(out_names=output_names, input_names=input_names)
        # onnxinfer = OnnxruntimeInfer(**onnxinferargs)
        
        if self.eval_mode == 'dataset':
            self.process.quantize(fd_path=self.face_data,
                                    is_dataset=True,
                                    prefix=self.img_prefix)
        
        gt_dict, fpred_dict, pred_dict = {}, {}, {}  
        image_annot_list_ = tqdm(self.image_annot_list) if self.is_stdout else self.image_annot_list
        for image_id, image_annot_info in enumerate(image_annot_list_):
            gt_cls = image_annot_info['label']
            if gt_cls == "live":
                self.target_accumulate = np.concatenate((self.target_accumulate, np.array([0])))
            else:
                self.target_accumulate = np.concatenate((self.target_accumulate, np.array([1]))) 

            image_path = image_annot_info['image_path']
            in_data = image_annot_info['face_data']
            gt_dict[image_path] = gt_cls

            if self.fp_result:
                true_outputs = self.process.post_quan.onnx_infer(in_data) #onnxinfer(in_data=in_data)
                fpred_cls = true_outputs["output"]
                fpred_cls = fpred_cls.reshape(fpred_cls.shape[0], -1)
                fpred_cls = softmax(fpred_cls, axis=1)
                fpred_dict[image_path] = np.array(fpred_cls[0, :]).tolist()
            else:
                true_outputs = None
                
            if self.eval_mode == 'single':
                self.process.quantize(in_data, is_dataset=False)
            
            if self.is_calc_error:
                self.process.checkerror(in_data, acc_error=self.acc_error)
            else:
                self.process.dataflow(in_data, acc_error=True, onnx_outputs=true_outputs)
            # if 'analysis_cfg' in self.process_args.keys() and self.fp_result:
            #     self.process.checkerror_weight(onnx_outputs=None)
            #     self.process.checkerror_feature(onnx_outputs=true_outputs)                
            outputs = self.process.get_outputs()['qout']
            outputs = outputs["output"]

            # ['output']
            outputs = outputs.reshape(outputs.shape[0], -1)

            pred_cls = softmax(outputs, axis=1)
            pred_dict[image_path] = np.array(pred_cls[0, :]).tolist()

            if 0 == image_id and self.eval_first_frame: break
            
        with open(self.gt_json, 'w') as file:
            json.dump(gt_dict, file)
        with open(self.fpred_json, 'w') as file:
            json.dump(fpred_dict, file)
        with open(self.pred_json, 'w') as file:
            json.dump(pred_dict, file)

        if not self.is_calc_error and self.process.onnx_graph:
            img = self.image_annot_list[0]['face_data']
            true_outputs = self.process.post_quan.onnx_infer(img)
            self.process.numpygraph(img, acc_error=True, onnx_outputs=true_outputs)

    def accuracy(self, mode='float'):
        res = {'recall': 0, 'precision': 0}
        
        with open(self.gt_json) as file:
            gt_json = json.load(file)

        if mode == 'float':
            with open(self.fpred_json) as file:
                pred_json = json.load(file)
        else:
            with open(self.pred_json) as file:
                pred_json = json.load(file)
        
        eps = 1.0e-6
        prob_accumulate = np.array([])
        num_tp = 0 # true spoof and pred spoof
        num_fp = 0 # true live but pred spoof
        num_tn = 0 # true live and pred live
        num_fn = 0 # true spoof but pred live
                
        for image_path, gt_cls in gt_json.items(): 
            pred_cls = pred_json[image_path]       
            spoof_conf = pred_cls[1]
            prob_accumulate = np.concatenate((prob_accumulate, np.array([spoof_conf])))
            if spoof_conf > self.spoof_conf_threshold:
                pred = "spoof"
            else:
                pred = "live"
            
            if gt_cls == "spoof" and pred == "spoof":
                num_tp += 1
            if gt_cls == "live" and pred == "spoof":
                num_fp += 1
            if gt_cls == "spoof" and pred == "live":
                num_fn += 1
            if gt_cls == "live" and pred == "live":
                num_tn += 1
        
        fpr, tpr, _ = roc_curve(y_true=self.target_accumulate, y_score=prob_accumulate, pos_label=1)
        self.plot_roc_curve(fpr, tpr, path=os.path.join(self.roc_save_path, 'roc_curve_{}.png'.format(mode)))

        res['recall'] = num_tp / (num_tp + num_fn + eps)
        res['precision'] = num_tp / (num_tp + num_fp + eps)

        return res

    def plot_roc_curve(self, fpr, tpr, path):
        plt.figure()
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.00])
        plt.plot(fpr, tpr, lw=3, label="ROC curve (area= {:0.2f})".format(auc(fpr, tpr)))
        plt.xlabel('FPR', fontsize=16)
        plt.ylabel('TPR', fontsize=16)
        plt.title('ROC curve', fontsize=16)
        plt.legend(loc='lower right', fontsize=13)
        plt.plot([0, 1], [0, 1], lw=3, linestyle='--', color='navy')
        plt.savefig(path)

    def __call__(self):
        accuracy = dict()
        if self.fp_result:
            accuracy['faccuracy'] = self.accuracy(mode='float')
        accuracy['qaccuracy'] = self.accuracy(mode='quant')
        tb = self.get_results(accuracy)
        
        return accuracy, tb
