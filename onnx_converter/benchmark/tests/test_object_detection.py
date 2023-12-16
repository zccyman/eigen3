# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- encoding: utf-8 -*-
# @Author  : henson.zhang
# @Company : SHIQING TECH
# @Time    : 2022/03/29 19:40:09
# @File    : test_object_detection.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import json
import os
import unittest

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from benchmark import (collect_accuracy, parse_config, save_config,
                       save_export, save_tables)
from benchmark.tests.test_base import TestBasse
from torch import nn
from torchvision.ops import nms
from eval.coco_eval import CocoEval

try:    
    from utils import Registry
except:
    from onnx_converter.utils import Registry

BENCHMARK: Registry = Registry('benchmark', scope='')

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

_COLORS = (np.array([
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
]).astype(np.float32).reshape(-1, 3))

cfg_file = 'benchmark/benchmark_config/object_detection.py'
args = parse_config(cfg_file)

first_model_name = ''
model_name_list = []
for model_type, model_paths in args.model_paths.items():
    model_rootdir_list = []
    for model_id, model_path in enumerate(model_paths):
        model_rootdir, model_name = os.path.split(model_path)
        model_name_list.append(model_type + '/' + str(model_id) + '/' + model_name)
        if first_model_name == '':
            first_model_name = model_name

quantize_method_list = []
for quantize_method_f, quantize_method_weights in args.quantize_methods.items():
    for quantize_method_w in quantize_method_weights:
        quantize_method_list.append(quantize_method_f + '/' + quantize_method_w)

process_scale_w_list = args.process_scale['weight']

params = model_name_list


@pytest.fixture(scope='class', params=params)
def model_paths(request):
    return request.param


params = [str(data) for data in args.quantize_dtypes]


@pytest.fixture(scope='class', params=params)
def quantize_dtype(request):
    return request.param


### feat/weight quant method
params = quantize_method_list


@pytest.fixture(scope='class', params=params)
def quantize_method(request):
    return request.param


params = process_scale_w_list


@pytest.fixture(scope='class', params=params)
def process_scale_w(request):
    return request.param


# @pytest.mark.usefixtures('model_paths', 'quantize_dtype', 'process_scale_w', 'quantize_method')
class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   reg_max,
                   iou_threshold,
                   class_agnostic,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0),
                                              num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels
    # boxes, scores, idxs, reg_max=1000, iou_threshold=0.3, class_agnostic=False
    dets, keep = batched_nms(bboxes, scores, labels, reg_max, iou_threshold,
                             class_agnostic)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def batched_nms(boxes,
                scores,
                idxs,
                reg_max=1000,
                iou_threshold=0.3,
                class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    if len(boxes_for_nms) < reg_max:
        # dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        keep = nms(boxes_for_nms, scores, iou_threshold)
        boxes = boxes[keep]
        # scores = dets[:, -1]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            # dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            keep = nms(boxes_for_nms[mask], scores[mask], iou_threshold)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def get_single_level_center_point(featmap_size,
                                  stride,
                                  dtype,
                                  device,
                                  flatten=True):
    """
    Generate pixel centers of a single stage feature map.
    :param featmap_size: height and width of the feature map
    :param stride: down sample stride of the feature map
    :param dtype: data type of the tensors
    :param device: device of the tensors
    :param flatten: flatten the x and y tensors
    :return: y and x of the center points
    """
    h, w = featmap_size
    x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
    y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
    y, x = torch.meshgrid(y_range, x_range)
    if flatten:
        y = y.flatten()
        x = x.flatten()
    return y, x


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def get_bboxes_single(cls_scores,
                      bbox_preds,
                      strides,
                      img_shape,
                      scale_factor,
                      device,
                      num_classes,
                      score_thr,
                      iou_threshold,
                      reg_max,
                      class_agnostic,
                      rescale=False):
    """
    Decode output tensors to bboxes on one image.
    :param cls_scores: classification prediction tensors of all stages
    :param bbox_preds: regression prediction tensors of all stages
    :param img_shape: shape of input image
    :param scale_factor: scale factor of boxes
    :param device: device of the tensor
    :return: predict boxes and labels
    """
    assert len(cls_scores) == len(bbox_preds)
    mlvl_bboxes = []
    mlvl_scores = []
    # strides = [8, 16, 32]
    distribution_project = Integral(7)
    for stride, cls_score, bbox_pred in zip(strides, cls_scores, bbox_preds):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        featmap_size = cls_score.size()[-2:]
        y, x = get_single_level_center_point(featmap_size,
                                             stride,
                                             cls_score.dtype,
                                             device,
                                             flatten=True)
        center_points = torch.stack([x, y], dim=-1)
        scores = cls_score.permute(1, 2, 0).reshape(-1, num_classes).sigmoid()
        bbox_pred = bbox_pred.permute(1, 2, 0)
        bbox_pred = distribution_project(bbox_pred) * stride

        nms_pre = 1000
        if scores.shape[0] > nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            center_points = center_points[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]

        bboxes = distance2bbox(center_points, bbox_pred, max_shape=img_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)

    mlvl_bboxes = torch.cat(mlvl_bboxes)
    if rescale:
        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

    mlvl_scores = torch.cat(mlvl_scores)
    # add a dummy background class at the end of all labels, same with mmdetection2.0
    padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
    mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
    # multi_bboxes, multi_scores, score_thr, reg_max, iou_threshold, class_agnostic, max_num=-1, score_factors=None
    det_bboxes, det_labels = multiclass_nms(mlvl_bboxes,
                                            mlvl_scores,
                                            score_thr=score_thr,
                                            iou_threshold=iou_threshold,
                                            reg_max=reg_max,
                                            class_agnostic=class_agnostic,
                                            max_num=100)
    return det_bboxes, det_labels


def get_bboxes(cls_scores,
               bbox_preds,
               input_shape,
               num_classes,
               strides,
               score_thr,
               iou_threshold,
               reg_max,
               class_agnostic,
               rescale=False):
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)
    device = cls_scores[0].device

    # input_shape = [input_height, input_width]

    result_list = []
    for img_id in range(cls_scores[0].shape[0]):
        cls_score_list = [
            cls_scores[i][img_id].detach() for i in range(num_levels)
        ]
        bbox_pred_list = [
            bbox_preds[i][img_id].detach() for i in range(num_levels)
        ]
        scale_factor = 1
        # cls_scores, bbox_preds, strides, img_shape, scale_factor, device, num_classes,
        #         score_thr,
        #         iou_threshold,
        #         reg_max,
        #         class_agnostic,
        #         rescale=False
        dets = get_bboxes_single(cls_score_list, bbox_pred_list, strides,
                                 input_shape, scale_factor, device,
                                 num_classes, score_thr, iou_threshold,
                                 reg_max, class_agnostic, rescale)

        result_list.append(dets)
    return result_list


def post_process(preds, warp_matrix, resizeed_shape, raw_shape, num_classes,
                 strides, score_thr, iou_threshold, reg_max, class_agnostic):
    cls_scores, bbox_preds = preds
    # cls_scores, bbox_preds, input_shape,
    # num_classes, score_thr, iou_threshold, reg_max, class_agnostic,
    result_list = get_bboxes(cls_scores, bbox_preds, resizeed_shape,
                             num_classes, strides, score_thr, iou_threshold,
                             reg_max, class_agnostic)
    img_height, img_width = raw_shape[:2]
    bboxes, scores, labels = list(), list(), list()
    for result in result_list:
        det_bboxes, det_labels = result
        det_bboxes = det_bboxes.cpu().numpy()
        det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4],
                                       np.linalg.inv(warp_matrix), img_width,
                                       img_height)
        classes = det_labels.cpu().numpy()
        bboxes.append(det_bboxes[:, :4])
        scores.append(det_bboxes[:, -1].reshape(-1))
        labels.append(classes)
    return np.column_stack(bboxes), np.column_stack(scores).reshape(
        -1), np.column_stack(labels).reshape(-1)


@BENCHMARK.register_module(name="pre1")
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

    def get_resize_matrix(self, raw_shape, dst_shape, keep_ratio):
        """
        Get resize matrix for resizing raw img to input size
        :param raw_shape: (width, height) of raw image
        :param dst_shape: (width, height) of input image
        :param keep_ratio: whether keep original ratio
        :return: 3x3 Matrix
        """
        r_w, r_h = raw_shape
        d_w, d_h = dst_shape
        Rs = np.eye(3)
        if keep_ratio:
            C = np.eye(3)
            C[0, 2] = -r_w / 2
            C[1, 2] = -r_h / 2

            if r_w / r_h < d_w / d_h:
                ratio = d_h / r_h
            else:
                ratio = d_w / r_w
            Rs[0, 0] *= ratio
            Rs[1, 1] *= ratio

            T = np.eye(3)
            T[0, 2] = 0.5 * d_w
            T[1, 2] = 0.5 * d_h
            return T @ Rs @ C
        else:
            Rs[0, 0] *= d_w / r_w
            Rs[1, 1] *= d_h / r_h
            return Rs

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img):
        res = dict()
        res["raw_shape"] = img.shape
        # resize image
        ResizeM = self.get_resize_matrix((img.shape[1], img.shape[0]),
                                         self.input_size, True)
        img_resize = cv2.warpPerspective(img,
                                         ResizeM,
                                         dsize=tuple(self.input_size))
        # normalize image
        img_input = img_resize.astype(np.float32) / 255
        img_mean = np.array(self.img_mean, dtype=np.float32) / 255
        img_std = np.array(self.img_std, dtype=np.float32) / 255
        img_input = (img_input - img_mean) / img_std
        # expand dims
        img_input = np.transpose(img_input, [2, 0, 1])
        img_input = np.expand_dims(img_input, axis=0)
        res['ResizeM'] = ResizeM
        res['input_shape'] = self.input_size
        self.set_trans(res)

        return img_input


@BENCHMARK.register_module(name="post1")
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

    def __call__(self, outputs, trans):
        ResizeM, raw_shape, input_shape = trans['ResizeM'], trans[
            'raw_shape'], trans['input_shape']
        scores, raw_boxes, res = [], [], []
        cls_res, reg_res = [], []
        for key in outputs.keys():
            if outputs[key] is None:
                continue
            output = torch.from_numpy(outputs[key])
            cls_res.append(
                torch.split(output, [self.num_classes, 32], dim=1)[0])
            reg_res.append(
                torch.split(output, [self.num_classes, 32], dim=1)[1])
        if cls_res == list() or reg_res == list():
            return None
        preds = tuple([cls_res, reg_res])
        out = post_process(preds, ResizeM, input_shape, raw_shape,
                           self.num_classes, self.strides, self.prob_threshold,
                           self.iou_threshold, self.num_candidate,
                           self.agnostic_nms)
        bbox = out[0].astype(np.int32)
        score = out[1].reshape(-1)
        label = out[2].astype(np.int32)
        if len(bbox) == 0 or len(score) == 0 or len(label) == 0:
            return None

        return dict(bbox=bbox, label=label, score=score)


@pytest.mark.usefixtures('model_paths', 'quantize_dtype', 'process_scale_w', 'quantize_method')
class TestObjectDetection(TestBasse):

    def compose_evaluator(self, **kwargs):
        args = kwargs['args']
        model_dir = kwargs['model_dir']
        model_name = kwargs['model_name']
        dataset_dir = kwargs['dataset_dir']
        selected_mode = kwargs['selected_mode']
        quantize_dtype = kwargs['quantize_dtype']
        process_scale_w = kwargs['process_scale_w']
        
        log_level = args.log_level
        eval_mode = args.eval_mode
        acc_error = args.acc_error

        model_path = os.path.join(model_dir, args.task_name, model_name)  # args.model_paths[model_type][model_id]
        dataset_name, imgsize, net_name, preprocess_name, postprocess_name = model_type.split('_')
        image_path = os.path.join(dataset_dir, args.dataset_dir[dataset_name])
        anno_file = os.path.join(dataset_dir, args.anno_file[dataset_name])

        kwargs_preprocess = {
            "img_mean": args.normalizations[0],
            "img_std": args.normalizations[1],
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

        preprocess = BENCHMARK.get(preprocess_name)(**kwargs_preprocess)
        postprocess = BENCHMARK.get(postprocess_name)(**kwargs_postprocess)

        model_name = os.path.basename(model_path).split('.onnx')[0]
        export_version = '' if args.export_version > 1 else '_v{}'.format(
            args.export_version)
        log_name = '{}.{}.{}.{}.{}.log'.format(
            model_name, quantize_method_f, quantize_method_w, quantize_dtype, process_scale_w)

        process_args = {
            'log_name': 'process.log',
            'log_level': log_level, 
            'model_path': model_path,
            'parse_cfg': 'benchmark/benchmark_config/base/parse.py',
            'graph_cfg': 'config/graph.py',
            'quan_cfg': 'benchmark/benchmark_config/base/quantize.py',
            # 'analysis_cfg': '{}/config/analysis.py',
            'export_cfg':
                'config/export{}.py'.format(export_version),
            'offline_quan_mode': args.offline_quan_mode,
            'offline_quan_tool': args.offline_quan_tool,
            'quan_table_path': args.quan_table_path,
            'device': args.device,
            'simulation_level': 1,
            'transform': preprocess,
            'postprocess': postprocess,
            'fp_result': args.fp_result,
            'is_ema': True,
            'ema_value': 0.99,
            'is_array': False,
            'is_stdout': args.is_stdout,
            'error_metric': args.error_metric
        }

        kwargs_bboxeval = {
            'log_dir': args.log_dir,
            'log_name': log_name,
            'log_level': log_level, 
            # 'is_stdout': args.is_stdout,
            'eval_first_frame': args.eval_first_frame,
            'img_prefix': args.img_prefix,
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

        parameter = dict(quan_dataset_path=image_path,
                         dataset_path=image_path,
                         ann_path=image_path,
                         event_lst=args.event_lst,
                         save_dir=args.results_path,
                         calibration_params_json_path=args.calibration_params_json_path,
                    )
                                 
        return dict(evaluator=cocoeval, parameters=parameter)

    def test_object_detection(self, model_paths, quantize_dtype, process_scale_w, quantize_method, model_dir,
                              dataset_dir, selected_mode, password):
        ms = model_paths
        qd = quantize_dtype
        ps = process_scale_w
        qm = quantize_method
        self.entrance(args, ms, qm, qd, ps, selected_mode, model_dir, dataset_dir, password)


if __name__ == '__main__':
    pytest.main()
