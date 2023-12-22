# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/24 9:47
# @File     : test_object_detection.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import copy
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms

from eval.coco_eval import CocoEval, overlay_bbox_cv
try:
    from tools import WeightOptimization
except:
    from onnx_converter.tools import WeightOptimization

class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
]

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
        .astype(np.float32)
        .reshape(-1, 3)
)

class CocoEvalWeightOpt(CocoEval):
    def __init__(self, **kwargs):
        super(CocoEvalWeightOpt, self).__init__(**kwargs)
        self.weight_optimization = kwargs["weight_optimization"]
        self.process_args_wo = copy.deepcopy(kwargs['process_args'])
        self.process_wo = WeightOptimization(**self.process_args_wo)
    
    def __call__(self, config_file=None):
        if self.weight_optimization in ["cross_layer_equalization"]:
            eval("self.process_wo." + self.weight_optimization)()
        else:
            eval("self.process_wo." + self.weight_optimization)(config_file=config_file)
            
class PriorGen:
    def __init__(self, strides):
        self.strides = [self._pair(stride) for stride in strides]
        self.num_levels = len(self.strides)
    def _pair(self, stride):
        if not isinstance(stride, (tuple, list)):
            return tuple([stride]*2)
        else:
            return stride

    def grid_priors(self, featmap_sizes, dtype=np.float32):
        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i, dtype=dtype)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self, featmap_size, level_idx, dtype=np.float32):
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = torch.arange(0, feat_h) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        all_points = shifts
        return all_points

    def _meshgrid(self, x, y):
        yy, xx = torch.meshgrid(y, x)
        return xx.reshape(-1), yy.reshape(-1)
    

def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list    
    
def filter_scores_and_topk(scores, score_thr, topk, results=None):
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results

def cat_boxes(data_list, dim=0):
    return torch.cat(data_list, dim=dim)
    
def distance2bbox(points, distance, max_shape):
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes
            
def decode(
        points,
        pred_bboxes,
        max_shape 
    ):
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        bboxes = distance2bbox(points, pred_bboxes, max_shape)

        return bboxes
        
def result_select(results, indexs):
    new_results = {}
    for k, v in results.items():
        new_results[k] = v[indexs]
    return new_results
        
def scale_boxes(boxes, scale_factor):
    repeat_num = int(boxes.size(-1) / 2)
    scale_factor = boxes.new_tensor(scale_factor).repeat((1, repeat_num))
    return boxes * scale_factor

def get_box_wh(boxes):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return w, h
        
def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop('type', 'nms')
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        dets = torch.cat((boxes_for_nms[keep], scores[keep].reshape(-1, 1)), dim=1)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            dets = torch.cat((boxes_for_nms[mask][keep], scores[mask][keep].reshape(-1, 1)), dim=1)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep
        
def bbox_post_process(results, cfg, rescale, with_nms, img_meta):
    if rescale:
        assert img_meta.get('scale_factor') is not None
        scale_factor = [1 / s for s in img_meta['scale_factor']]
        results['bbox'] = scale_boxes(results['bbox'], scale_factor)

    if cfg.get('min_bbox_size', -1) >= 0:
        w, h = get_box_wh(results['bbox'])
        valid_mask = (w > cfg['min_bbox_size']) & (h > cfg['min_bbox_size'])
        if not valid_mask.all():
            results = result_select(results, valid_mask)

    if with_nms and results['bbox'].numel() > 0:
        bboxes = results['bbox']
        det_bboxes, keep_idxs = batched_nms(bboxes, results['score'],
                                            results['label'], cfg['nms'])
        results = result_select(results, keep_idxs)
        # some nms would reweight the score, such as softnms
        results['score'] = det_bboxes[:, -1]
        results = result_select(results, slice(cfg['max_per_img']))

    return results
        
def predict_by_feat_single(cfg, cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, img_meta, rescale):
    img_shape = img_meta['input_shape']
    nms_pre = cfg.get('nms_pre', -1)

    mlvl_bbox_preds = []
    mlvl_valid_priors = []
    mlvl_scores = []
    mlvl_labels = []
    for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
            enumerate(zip(cls_score_list, bbox_pred_list,
                            score_factor_list, mlvl_priors)):

        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        dim = 4
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
        cls_score = cls_score.permute(1, 2,
                                        0).reshape(-1, cls_score_list[0].shape[0])
        scores = cls_score.sigmoid()
        score_thr = cfg.get('score_thr', 0)

        results = filter_scores_and_topk(
            scores, score_thr, nms_pre,
            dict(bbox_pred=bbox_pred, priors=priors))
        scores, labels, keep_idxs, filtered_results = results

        bbox_pred = filtered_results['bbox_pred']
        priors = filtered_results['priors']

        mlvl_bbox_preds.append(bbox_pred)
        mlvl_valid_priors.append(priors)
        mlvl_scores.append(scores)
        mlvl_labels.append(labels)

    bbox_pred = torch.cat(mlvl_bbox_preds)
    priors = cat_boxes(mlvl_valid_priors)
    bboxes = decode(priors, bbox_pred, max_shape=img_shape)

    results = {}
    results['bbox'] = bboxes
    results['score'] = torch.cat(mlvl_scores)
    results['label'] = torch.cat(mlvl_labels)

    return bbox_post_process(
        results=results,
        cfg=cfg,
        rescale=rescale,
        with_nms=True,
        img_meta=img_meta)
        

class PreProcess(object):
    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = [103.53, 116.28, 123.675]
        self.img_std = [57.375, 57.12, 58.395]
        if 'img_mean' in kwargs.keys():
            self.img_mean = kwargs['img_mean']
        if 'img_std' in kwargs.keys():
            self.img_std = kwargs['img_std']
        self.trans = 0
        self.input_size = kwargs['input_size']
        self.color = (114, 114, 114)
        
    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img):
        res = dict()
        res["raw_shape"] = img.shape

        self.target_h, self.target_w = self.input_size
        img_h, img_w, _ = img.shape
        if img_w > img_h:
            r = self.target_w / img_w
            new_shape_w, new_shape_h = self.target_w, int(round(img_h * r))
            if new_shape_h > self.target_h:
                r = self.target_h / img_h
                new_shape_w, new_shape_h = int(round(img_w * r)), self.target_h
                w_pad, h_pad = self.target_w - new_shape_w, self.target_h - new_shape_h
            else:
                w_pad, h_pad = self.target_w - new_shape_w, self.target_h - new_shape_h
        else:
            r = self.target_h / img_h
            new_shape_w, new_shape_h = int(round(img_w * r)), self.target_h
            if new_shape_w > self.target_w:
                r = self.target_w / img_w
                new_shape_w, new_shape_h = self.target_w, int(round(img_h * r))
                w_pad, h_pad = self.target_w - new_shape_w, self.target_h - new_shape_h
            else:
                w_pad, h_pad = self.target_w - new_shape_w, self.target_h - new_shape_h

        resize_img = cv2.resize(img, (new_shape_w, new_shape_h), interpolation=cv2.INTER_LINEAR)
        left, top = 0, 0
        bottom, right = int(h_pad), int(w_pad)

        # 固定值边框，统一都填充color
        img_resize = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        
        # normalize image
        img_input = img_resize.astype(np.float32) / 255
        img_mean = np.array(self.img_mean, dtype=np.float32) / 255
        img_std = np.array(self.img_std, dtype=np.float32) / 255
        img_input = (img_input - img_mean) / img_std
        # expand dims
        img_input = np.transpose(img_input, [2, 0, 1])
        img_input = np.expand_dims(img_input, axis=0)
                
        res['scale_factor'] = (r, r)
        res['input_shape'] = self.input_size
        self.set_trans(res)

        return img_input


class PostProcess(object):
    def __init__(self, **kwargs):
        super(PostProcess, self).__init__()
        self.strides = kwargs["strides"]
        self.reg_max = kwargs["reg_max"]
        self.num_candidate = kwargs["num_candidate"]
        self.prob_threshold = kwargs["prob_threshold"]
        self.iou_threshold = kwargs["iou_threshold"]
        self.top_k = kwargs["top_k"]
        self.num_classes = kwargs["num_classes"]
        self.agnostic_nms = kwargs["agnostic_nms"]
        self.priorgen = PriorGen(strides=self.strides)
        
    def draw_image(self, img, results, class_names, colors):
        raw_img = copy.deepcopy(img)
        if not isinstance(results, dict):
            return raw_img
        bbox, label, score = results["bbox"], results["label"], results["score"]

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
        img_draw = overlay_bbox_cv(img, all_box, class_names, colors=colors)
        return img_draw

    def __call__(self, outputs, trans):
        # scale_factor, raw_shape, input_shape = trans['scale_factor'], trans['raw_shape'], trans['input_shape']
        img_meta = trans
        
        cls_scores, bbox_preds = [], []
        for i, key in enumerate(outputs.keys()):
            output = torch.from_numpy(outputs[key])
            if output.shape[1] == self.num_classes:
                cls_scores.append(output)
            else:
                bbox_preds.append(output)
        
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.priorgen.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype)

        cls_score_list = select_single_mlvl(cls_scores, 0, detach=True)
        bbox_pred_list = select_single_mlvl(bbox_preds, 0, detach=True)
        score_factor_list = [None for _ in range(num_levels)]
        cfg = {
            'max_per_img': self.num_candidate, 'min_bbox_size':0, 
            'nms':{'iou_threshold': self.iou_threshold, 'type':'nms'}, 
            'nms_pre': 30000,
            'score_thr': self.prob_threshold,
        }        
        results = predict_by_feat_single(
            cfg=cfg,
            cls_score_list=cls_score_list,
            bbox_pred_list=bbox_pred_list,
            score_factor_list=score_factor_list,
            mlvl_priors=mlvl_priors,
            img_meta=img_meta,
            rescale=True)
        
        results['bbox'] = np.round(results['bbox'].numpy()).astype(np.int32)
        results['score'] = results['score'].numpy()
        results['label'] = results['label'].numpy()
        
        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', 
                        type=str, 
                        default='RTM_CSPNeXt_tiny_onnx/rtmdet_tiny_det_320x320_sim.onnx', 
                        help='checkpoint path')
    parser.add_argument('--quan_dataset_path', 
                        type=str, 
                        default='/buffer/calibrate_dataset/coco_80_cls', 
                        help='quantize image path')
    parser.add_argument('--ann_path', 
                        type=str, 
                        default='/buffer/coco/annotations/instances_val2017.json', 
                        help='eval anno path')
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='/buffer/coco/val2017', 
                        help='eval image path')
    parser.add_argument('--input_size', type=list, default=[320, 320])
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--results_path', type=str, default='./work_dir/tmpfiles/object_det_resluts')
    parser.add_argument('--topk', type=int, default=-1)
    parser.add_argument('--reg_max', type=int, default=7)
    parser.add_argument('--prob_threshold', type=float, default=0.3)
    parser.add_argument('--iou_threshold', type=float, default=0.65)
    parser.add_argument('--num_candidate', type=int, default=300)
    parser.add_argument('--strides', type=list, default=[8, 16, 32])
    parser.add_argument('--draw_result', type=bool, default=True)
    parser.add_argument('--agnostic_nms', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')    
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--export', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)    
    parser.add_argument('--log_level', type=int, default=10)    
    parser.add_argument('--error_analyzer', type=bool, default=False)
    parser.add_argument('--is_calc_error', type=bool, default=False) #whether to calculate each layer error
    # bias_correction cross_layer_equalization
    parser.add_argument('--weight_optimization', type=str, default="qat")      
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
        
    normalization = [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
    kwargs_preprocess = {
        "img_mean": normalization[0],
        "img_std": normalization[1],
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
        'log_dir': 'work_dir/eval',
        'log_name': 'test_object_detection.log',   
        'log_level': args.log_level,
        # 'is_stdout': args.is_stdout,   
        'img_prefix': 'jpg',                
        "save_key": "mAP",
        "draw_result": args.draw_result,
        "_COLORS": _COLORS,
        "class_names": class_names,
        "process_args": process_args,
        'is_calc_error': args.is_calc_error,
        "weight_optimization": args.weight_optimization,
        "eval_mode": eval_mode,  # single quantize, dataset quantize
        "fp_result": args.fp_result,
        'acc_error': acc_error,
    }

    cocoeval = CocoEvalWeightOpt(**kwargs_bboxeval)
    cocoeval(config_file="tests/qat/test_object_detection_rtmdet.json")