# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/24 9:58
# @File     : test_retina_face_detection.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import copy
import os
from itertools import product
from math import ceil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

from eval.alfw import AlfwEval

class_names = [
    "face"
]


def generate_prior(steps, image_sizes, min_sizes):
    """generate priors"""

    feature_maps = [[ceil(image_sizes[0] / step), ceil(image_sizes[1] / step)] for step in steps]

    anchor_lst = []
    for k, f in enumerate(feature_maps):
        min_size = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size_value in min_size:
                s_kx = min_size_value / image_sizes[1]
                s_ky = min_size_value / image_sizes[0]
                dense_cx = [x * steps[k] / image_sizes[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_sizes[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchor_lst += [cx, cy, s_kx, s_ky]

    return np.array(anchor_lst).reshape(-1, 4)


class PreProcess(object):
    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = [104, 117, 123]
        self.img_std = [1.0, 1.0, 1.0]
        if 'img_mean' in kwargs.keys():
            self.img_mean = kwargs['img_mean']
        if 'img_std' in kwargs.keys():
            self.img_std = kwargs['img_std']
        self.trans = 0
        self.input_size = kwargs['input_size']
        # self.target_w, self.target_h = 0, 0
        # self.ratio, self.w_pad, self.h_pad = 1, 0, 0
        # self.src_shape = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img, color=(114, 114, 114)):
        target_w, target_h = self.input_size

        img_h, img_w = img.shape[:2]
        # print(f"img_h: {img_h}, img_w: {img_w}") # 331, 500
        if img_w > img_h:
            r = target_w / img_w
            new_shape_w, new_shape_h = target_w, int(round(img_h * r))
            if new_shape_h > 256:
                r = target_h / img_h
                new_shape_w, new_shape_h = int(round(img_w * r)), target_h
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h
            else:
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h
            # print(w_pad, h_pad, r)
            # print('-----------------------------------------------------')
        else:
            r = target_h / img_h
            new_shape_w, new_shape_h = int(round(img_w * r)), target_h
            if new_shape_w > 320:
                r = target_w / img_w
                new_shape_w, new_shape_h = target_w, int(round(img_h * r))
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h
            else:
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h

        w_pad /= 2
        h_pad /= 2

        resize_img = cv2.resize(img, (new_shape_w, new_shape_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
        left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))

        # 固定值边框，统一都填充color
        img_final = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        img_final = img_final - (104, 117, 123)

        img_final = np.transpose(img_final, (2,0,1)).astype(np.float32)

        img_final = np.expand_dims(img_final, axis=0)
        
        # self.ratio, self.w_pad, self.h_pad = r, w_pad, h_pad
        self.trans = dict(target_w=target_w, target_h=target_h, w_pad=w_pad, h_pad=h_pad,
                          src_shape=img.shape[:2], ratio=r)

        return img_final


class PostProcess(object):
    def __init__(self, **kwargs):
        super(PostProcess, self).__init__()
        self.steps = kwargs["steps"]
        self.min_sizes = kwargs['min_sizes']
        self.nms_threshold = kwargs["nms_threshold"]
        self.variances = kwargs['variances']
        self.prob_threshold = kwargs["prob_threshold"]
        self.top_k = kwargs["top_k"]

    def reshpe_out(self, out):
        if out is None:
            return None
        n, c, h, w = out.shape
        out = np.transpose(out, (0, 2, 3, 1))
        return np.reshape(out, (n, -1, c))


    def draw_image(self, img, detects, class_names, colors):
        for det in detects:
            if det[4] < self.prob_threshold:
                continue

            ### filter abnormal bbox, added by henson
            x0, y0, x1, y1 = det[:4]
            if abs(x1 - x0) < 8 or abs(x1 - x0) > int(0.8 * img.shape[1]) or abs(y1 - y0) < 8 or abs(
                    y1 - y0) > int(0.8 * img.shape[0]):
                continue

            text = "{:.4f}".format(det[4])
            det = list(map(int, det))
            color = (int(colors[0][0] * 255), int(colors[0][1] * 255), int(colors[0][2] * 255))
            cv2.rectangle(img=img, pt1=(det[0], det[1]), pt2=(det[2], det[3]), color=color, thickness=2)
            cx, cy = det[0], det[1] + 12
            cv2.putText(img=img, text=text, org=(cx, cy), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5,
                        color=(255, 255, 255))
            # landmarks
            cv2.circle(img=img, center=(det[5], det[6]), radius=1, color=(0, 0, 255), thickness=4)
            cv2.circle(img=img, center=(det[7], det[8]), radius=1, color=(0, 255, 255), thickness=4)
            cv2.circle(img=img, center=(det[9], det[10]), radius=1, color=(255, 0, 255), thickness=4)
            cv2.circle(img=img, center=(det[11], det[12]), radius=1, color=(0, 255, 0), thickness=4)
            cv2.circle(img=img, center=(det[13], det[14]), radius=1, color=(255, 0, 0), thickness=4)

        return img
    
    def __call__(self, outputs, trans):
        target_w = trans['target_w']
        target_h = trans['target_h']
        w_pad = trans['w_pad']
        h_pad = trans['h_pad']
        src_shape = trans['src_shape']
        resize = trans['ratio']
        image_w, image_h = src_shape
        # compute_out = [self.reshpe_out(outputs[key]) for key in outputs.keys()]
        compute_out = {}
        for key in outputs.keys():
            compute_out[key] = self.reshpe_out(outputs[key])
        loc = np.row_stack([compute_out['output1'].reshape(-1,4),
                            compute_out['output2'].reshape(-1, 4),
                            compute_out['output3'].reshape(-1, 4),
                            compute_out['output4'].reshape(-1, 4)])

        conf = np.row_stack([compute_out['output5'].reshape(-1, 2),
                             compute_out['output6'].reshape(-1, 2),
                             compute_out['output7'].reshape(-1, 2),
                             compute_out['output8'].reshape(-1, 2)])

        landmark = np.row_stack([compute_out['output9'].reshape(-1, 10),
                                 compute_out['output10'].reshape(-1, 10),
                                 compute_out['output11'].reshape(-1, 10),
                                 compute_out['output12'].reshape(-1, 10)])

        conf = F.softmax(torch.from_numpy(conf), dim=-1).numpy()

        priors = generate_prior(
            steps=self.steps, image_sizes=(target_h, target_w), min_sizes=self.min_sizes)
        # decode bounding box predictions
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * self.variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * self.variances[1])
            ),
            axis=1
        )

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        scale_loc = np.array([target_w, target_h, target_w, target_h])
        boxes = boxes * scale_loc / resize
        boxes[:, [0, 2]] -= (w_pad / resize)
        boxes[:, [1, 3]] -= (h_pad / resize)
        # print(w_pad, h_pad, resize)
        # decode landmark
        landmarks = np.concatenate(
            (
                priors[:, :2] + landmark[:, :2] * self.variances[0] * priors[:, 2:],
                priors[:, :2] + landmark[:, 2:4] * self.variances[0] * priors[:, 2:],
                priors[:, :2] + landmark[:, 4:6] * self.variances[0] * priors[:, 2:],
                priors[:, :2] + landmark[:, 6:8] * self.variances[0] * priors[:, 2:],
                priors[:, :2] + landmark[:, 8:10] * self.variances[0] * priors[:, 2:],
            ),
            axis=1
        )
        scale_landmark = np.array([
            target_w, target_h, target_w, target_h, target_w, target_h, target_w, target_h, target_w, target_h])

        landmarks = landmarks * scale_landmark / resize
        landmarks[:, [0, 2, 4, 6, 8]] -= (w_pad / resize)
        landmarks[:, [1, 3, 5, 7, 9]] -= (h_pad / resize)

        scores = conf[:, 1]
        indexes = scores > self.prob_threshold
        scores = scores[indexes]
        boxes = boxes[indexes, :]
        landmarks = landmarks[indexes, :]

        indexes = np.argsort(scores)
        scores = scores[indexes]
        boxes = boxes[indexes, :]
        landmarks = landmarks[indexes, :]

        # nms
        select_index = nms(torch.from_numpy(boxes.astype(np.float32)), torch.from_numpy(scores), iou_threshold=self.nms_threshold)
        select_index = select_index.numpy().tolist()
        if len(select_index) < self.top_k:
            select_index = select_index[:self.top_k]
        boxes = boxes[select_index, :]
        scores = scores[select_index]
        landmarks = landmarks[select_index, :]
        # print(landmarks)
        # print('-----------------------------------------------------')

        detects = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        detects = np.concatenate((detects, landmarks), axis=1)

        #### filter abnormal bbox, added by henson
        for idx in range(detects.shape[0]):
            x0, y0, x1, y1 = detects[idx][:4]
            x0, y0 = np.max((x0, 0)), np.max((y0, 0))
            x1, y1 = np.min((x1, src_shape[1] - 1)), np.min((y1, src_shape[0] - 1))
            detects[idx][:4] = x0, y0, x1, y1

        return detects


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', 
                        type=str, 
                        default='/buffer/trained_models/face-detection/slimv2_Final_simplify.onnx',
                        help='checkpoint path')
    parser.add_argument('--quan_dataset_path', 
                        type=str, 
                        default="/buffer/AFLW/sub_test_data",
                        help='eval dataset path')
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='/buffer/AFLW/sub_test_data',
                        help='eval image path')
    parser.add_argument('--ann_path', type=str, 
                        default='/buffer/AFLW/sub_test_data',
                        help='eval anno path')
    parser.add_argument('--event_lst', 
                        type=list, 
                        default=["flickr_3"], 
                        help='eval anno path') # ["flickr_0", "flickr_2", "flickr_3"]
    parser.add_argument('--input_size', type=list, default=[320, 256])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--results_path', type=str, default='./work_dir/tmpfiles/retina_face_det_resluts')
    parser.add_argument('--topk', type=int, default=1000)
    parser.add_argument('--prob_threshold', type=float, default=0.7)
    parser.add_argument('--nms_threshold', type=float, default=0.3)
    parser.add_argument('--num_candidate', type=int, default=1000)
    parser.add_argument('--steps', type=list, default=[8, 16, 32, 64]) # slim
    # parser.add_argument('--steps', type=list, default=[8, 16, 32]) # mobilenet
    parser.add_argument('--variances', type=list, default=[0.1, 0.2])
    parser.add_argument('--min_sizes', type=list, default=[[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]) # slim
    # parser.add_argument('--min_sizes', type=list, default=[[10, 20], [32, 64], [128, 256]]) # mobilenet
    parser.add_argument('--draw_result', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')
    parser.add_argument('--fp_result', type=bool, default=False)
    parser.add_argument('--export_version', type=int, default=3)
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
    # root_dir = '/home/shiqing/Downloads/onnx_converter'
    if args.debug:
        eval_mode = 'single'
        acc_error = False
    else:
        eval_mode = 'dataset'
        acc_error = True
        
    normalization = [[104, 117, 123], [1.0, 1.0, 1.0]]
    kwargs_preprocess = {
        "img_mean": normalization[0],
        "img_std": normalization[1],
    }
    kwargs_postprocess = {
        "top_k": args.topk,
        "prob_threshold": args.prob_threshold,
        "nms_threshold": args.nms_threshold,
        "num_candidate": args.num_candidate,
        "steps": args.steps,
        'num_class': args.num_classes,
        "min_sizes": args.min_sizes,
        "variances": args.variances,
    }

    preprocess = PreProcess(input_size=args.input_size)
    postprocess = PostProcess(**kwargs_postprocess)

    process_args = {'log_name': 'process.log',
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
                    'transform': preprocess, 
                    'postprocess': postprocess,
                    'device': args.device,
                    'is_ema': True,
                    'fp_result': args.fp_result,
                    'ema_value': 0.99, 
                    'is_array': False, 
                    'is_stdout': args.is_stdout,                     
                    'error_analyzer': args.error_analyzer,                    
                    'error_metric': ['L1', 'L2', 'Cosine']
                    }

    kwargs_bboxeval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_retina_face_detection.log',     
        'log_level': args.log_level,
        # 'is_stdout': args.is_stdout,   
        'img_prefix': 'jpg',              
        "iou_threshold": args.nms_threshold,
        "prob_threshold": args.prob_threshold,
        "process_args": process_args,
        'is_calc_error': args.is_calc_error,
        "draw_result": args.draw_result,
        "fp_result": args.fp_result,
        "eval_mode": eval_mode,  # single quantize, dataset quatize
        'acc_error': acc_error,
    }

    alfweval = AlfwEval(**kwargs_bboxeval)

    accurracy, tb = alfweval(
        quan_dataset_path=args.quan_dataset_path,
        dataset_path=args.dataset_path,
        ann_path=args.ann_path, 
        event_lst=args.event_lst, 
        save_dir=args.results_path)
    if args.error_analyzer:
        alfweval.error_analysis()         
    if args.export:
        alfweval.export()
    print(tb)