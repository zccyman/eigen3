# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2023/10/16 16:49
# @File     : eval_yolox_nano_clipped.py
import random
import sys
from typing import Any

import matplotlib  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import copy
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops import nms

from eval.coco_eval import CocoEval, overlay_bbox_cv

from model_zoo.coco_names import class_names, _COLORS

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.array(scores.argsort())[::-1]#scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

class PreProcess:
    def __init__(self, **kwargs):
        self.input_size = kwargs.get("input_size", [416, 416])
        self.swap = kwargs.get("swap", (2, 0, 1))
        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img, *args: Any, **kwargs: Any) -> Any:
        if len(img.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114

        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(self.swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        import torch
        new_img = torch.from_numpy(padded_img)
        img = torch.concat([new_img[:,0::2,0::2],new_img[:,1::2,0::2],
                            new_img[:,0::2,1::2],new_img[:,1::2,1::2]],
                            axis=0).numpy()
        
        self.trans = r

        return img[np.newaxis,:,:,:]

class PostProcess:
    def __init__(self, **kwargs):
        self.img_size = kwargs.get("img_size", [416, 416]) 
        # self.ratio = kwargs.get("ratio", 1.0) 
        self.p6 = kwargs.get("p6", False)

    @staticmethod
    def draw_image(img, results, class_names, colors):
        raw_img = copy.deepcopy(img)
        if results is None:
            return raw_img
        bbox, score, label = results['bbox'], results['score'], results['label']

        img = raw_img.copy()
        all_box = [
            [
                x,
            ]
            + y
            + [
                z,
            ]
            for x, y, z in zip(label, bbox.tolist(), score)
        ]
        img_draw = overlay_bbox_cv(img, all_box, class_names, _COLORS=colors)
        return img_draw

    def __call__(self, output_list, ratio, **kwargs: Any) -> Any:
        outputs = []
        if output_list is None:
            return None
        for i in range(0,9,3):
            keys = ['output{}'.format(i+1), 'output{}'.format(i+2), 'output{}'.format(i+3)]
            output = torch.cat([torch.from_numpy(output_list[keys[0]]), 
                                torch.from_numpy(output_list[keys[1]]).sigmoid(), 
                                torch.from_numpy(output_list[keys[2]]).sigmoid()], 1)
            outputs.append(output)
        hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not self.p6 else [8, 16, 32, 64]

        hsizes = [self.img_size[0] // stride for stride in strides]
        wsizes = [self.img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        outputs = outputs[0]

        boxes = outputs[:, :4]
        scores = outputs[:, 4:5] * outputs[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        if dets is None:
            return None

        return {'bbox': dets[:, :4].astype(np.int32), 'label': dets[:, 5].astype(np.int32).reshape(-1), "score": dets[:, 4].reshape(-1)}#dets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', 
                        type=str, 
                        default='/buffer/trained_models/model-zoo/yolox-nano-clipped-.onnx', 
                        help='checkpoint path')
    parser.add_argument('--quan_dataset_path', 
                        type=str, 
                        default='/buffer/coco/val2017', 
                        help='quantize image path')
    parser.add_argument('--ann_path', 
                        type=str, 
                        default='/buffer/coco/annotations/instances_val2017.json', 
                        help='eval anno path')
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='/buffer/coco/val2017', 
                        help='eval image path')
    parser.add_argument('--input_size', type=list, default=[416, 416])
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--results_path', type=str, default='./work_dir/tmpfiles/object_det_resluts')
    parser.add_argument('--topk', type=int, default=-1)
    parser.add_argument('--reg_max', type=int, default=7)
    parser.add_argument('--prob_threshold', type=float, default=0.3)
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    parser.add_argument('--num_candidate', type=int, default=1000)
    parser.add_argument('--strides', type=list, default=[8, 16, 32])
    parser.add_argument('--draw_result', type=bool, default=True)
    parser.add_argument('--agnostic_nms', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')    
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)    
    parser.add_argument('--log_level', type=int, default=10)    
    parser.add_argument('--error_analyzer', type=bool, default=False)
    parser.add_argument('--is_calc_error', type=bool, default=False) #whether to calculate each layer error
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        eval_mode = 'single'
        acc_error = False
    else:
        eval_mode = 'dataset'
        acc_error = True
        
    normalization = [[0, 0, 0], [1, 1, 1]]
    kwargs_preprocess = {
        "input_size": args.input_size,
    }
    kwargs_postprocess = {
        "reg_max": args.reg_max,
        "top_k": args.topk,
        "prob_threshold": args.prob_threshold,
        "iou_threshold": args.iou_threshold,
        "num_candidate": args.num_candidate,
        "strides": args.strides,
        'num_classes': args.num_classes,
        'agnostic_nms': args.agnostic_nms,
    }

    preprocess = PreProcess(**kwargs_preprocess)
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
        'log_dir': 'work_dir/eval_yolox_nano_clipped',
        'log_name': 'eval_yolox_nano_clipped.log',   
        'log_level': args.log_level,
        # 'is_stdout': args.is_stdout,   
        'img_prefix': 'jpg',                
        "save_key": "mAP",
        "draw_result": args.draw_result,
        "_COLORS": _COLORS,
        "class_names": class_names,
        "process_args": process_args,
        'is_calc_error': args.is_calc_error,
        "eval_mode": eval_mode,  # single quantize, dataset quantize
        "fp_result": args.fp_result,
        'acc_error': acc_error,
    }

    cocoeval = CocoEval(**kwargs_bboxeval)
    cocoeval.set_colors(colors=_COLORS)
    cocoeval.set_class_names(names=class_names)
    cocoeval.set_draw_result(is_draw=args.draw_result)
    cocoeval.set_iou_threshold(iou_threshold=args.iou_threshold)
    # quan_dataset_path, images_path, ann_path, input_size, normalization, save_eval_path
    evaluation, tb = cocoeval(
        quan_dataset_path=args.quan_dataset_path, dataset_path=args.dataset_path, ann_path=args.ann_path,
        input_size=args.input_size, normalization=normalization, save_eval_path=args.results_path)
    if args.error_analyzer:
        cocoeval.error_analysis()      
    if args.export:
        cocoeval.export()