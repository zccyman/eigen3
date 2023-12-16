# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/24 9:49
# @File     : test_pedestrian_detection.py
import random
import sys

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


class_names = [
    "pedestrain"
]

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
        ]
    ).astype(np.float32)
        .reshape(-1, 3)
)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def _make_grid(nx=20, ny=20, i=0):
    anchors = [[[ 2.37500,  3.37500],
         [ 5.50000,  5.00000],
         [ 4.75000, 11.75000]],

        [[ 6.00000,  4.25000],
         [ 5.37500,  9.50000],
         [11.25000,  8.56250]],

        [[ 4.37500,  9.40625],
         [ 9.46875,  8.25000],
         [ 7.43750, 16.93750]],

        [[ 6.81250,  9.60938],
         [11.54688,  5.93750],
         [14.45312, 12.37500]]]
    anchors = torch.tensor(anchors)
    stride = [8, 16, 32, 64]
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).expand((1, 1, ny, nx, 2)).float()
    # anchor_grid = [torch.zeros(1)] * 4
    # for i in range(len(stride)):
    anchor_grid = (
        (anchors[i].clone() * stride[i])
        .view((1, 3, 1, 1, 2))
        .float()
    )
    return grid, anchor_grid


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), kpt_label=False, nc=None, nkpt=None):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if nc is None:
        nc = prediction.shape[2] - 5  if not kpt_label else prediction.shape[2] - 56 # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0,6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            if not kpt_label:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            else:
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]


        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def clip_coords(boxes, img_shape, step=2):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0::step].clamp_(0, img_shape[1])  # x1
    boxes[:, 1::step].clamp_(0, img_shape[0])  # y1


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, kpt_label=False, step=2):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    if isinstance(gain, (list, tuple)):
        gain = gain[0]
    if not kpt_label:
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, [0, 2]] /= gain
        coords[:, [1, 3]] /= gain
        clip_coords(coords[0:4], img0_shape)
        #coords[:, 0:4] = coords[:, 0:4].round()
    else:
        coords[:, 0::step] -= pad[0]  # x padding
        coords[:, 1::step] -= pad[1]  # y padding
        coords[:, 0::step] /= gain
        coords[:, 1::step] /= gain
        clip_coords(coords, img0_shape, step=step)
        #coords = coords.round()
    return coords


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


def get_single_level_center_priors(
        batch_size, featmap_size, stride, dtype, device
):
    """Generate centers of a single stage feature map.
    Args:
        batch_size (int): Number of images in one batch.
        featmap_size (tuple[int]): height and width of the feature map
        stride (int): down sample stride of the feature map
        dtype (obj:`torch.dtype`): data type of the tensors
        device (obj:`torch.device`): device of the tensors
    Return:
        priors (Tensor): center priors of a single level feature map.
    """
    h, w = featmap_size
    x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
    y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
    y, x = torch.meshgrid(y_range, x_range)
    y = y.flatten()
    x = x.flatten()
    strides = x.new_full((x.shape[0],), stride)
    proiors = torch.stack([x, y, strides, strides], dim=-1)
    return proiors.unsqueeze(0).repeat(batch_size, 1, 1)


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


def batched_nms(boxes, scores, idxs, iou_threshold, num_candidate, class_agnostic=False):
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    if len(boxes_for_nms) < num_candidate:
        # dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        keep = nms(boxes_for_nms, scores, iou_threshold=iou_threshold)
        boxes = boxes[keep]
        # scores = dets[:, -1]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            # dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            keep = nms(boxes_for_nms[mask], scores[mask], iou_threshold=iou_threshold)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_threshold,
                   class_agnostic,
                   max_num=-1,
                   score_factors=None):
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
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
    # boxes, scores, idxs, iou_threshold, num_candidate, class_agnostic=False
    dets, keep = batched_nms(bboxes, scores, labels, iou_threshold, 2000, class_agnostic)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


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
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def get_bboxes(cls_preds, reg_preds, img_shape, num_candidate, strides,
               score_threshold=0.05, iou_threshold=0.6, class_agnostic=False):
    device = cls_preds.device
    b = cls_preds.shape[0]
    input_height, input_width = img_shape[:2]
    input_shape = (input_height, input_width)
    # strides = [8, 16, 32, 64]
    distribution_project = Integral(7)
    featmap_sizes = [
        (math.ceil(input_height / stride), math.ceil(input_width) / stride)
        for stride in strides
    ]
    # get grid cells of one image
    mlvl_center_priors = [
        get_single_level_center_priors(
            b,
            featmap_sizes[i],
            stride,
            dtype=torch.float32,
            device=device,
        )
        for i, stride in enumerate(strides)
    ]
    center_priors = torch.cat(mlvl_center_priors, dim=1)
    dis_preds = distribution_project(reg_preds) * center_priors[..., 2, None]
    bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
    scores = cls_preds.sigmoid()
    result_list = []
    for i in range(b):
        # add a dummy background class at the end of all labels
        # same with mmdetection2.0
        score, bbox = scores[i], bboxes[i]
        padding = score.new_zeros(score.shape[0], 1)
        score = torch.cat([score, padding], dim=1)
        results = multiclass_nms(
            bbox,
            score,
            score_thr=score_threshold,
            iou_threshold=iou_threshold,
            max_num=num_candidate,
            class_agnostic=class_agnostic
        )
        result_list.append(results)
    return result_list


def post_process(preds, warp_matrix, resizeed_shape, raw_shape, strides,
                 num_candidate, score_threshold, iou_threshold, class_agnostic):
    cls_scores, bbox_preds = preds.split(
        [1, 32], dim=-1
    )
    # img_shape, num_candidate, score_threshold=0.05,
    # iou_threshold=0.6, class_agnostic=False
    result_list = get_bboxes(cls_scores, bbox_preds, resizeed_shape, num_candidate, strides,
                             score_threshold, iou_threshold, class_agnostic)
    det_results = {}
    img_height, img_width = raw_shape[:2]

    for result in result_list:
        det_result = {}
        det_bboxes, det_labels = result
        det_bboxes = det_bboxes.detach().cpu().numpy()
        det_bboxes[:, :4] = warp_boxes(
            det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
        )
        classes = det_labels.detach().cpu().numpy()
        for i in range(1):
            inds = classes == i
            det_result[i] = np.concatenate(
                [
                    det_bboxes[inds, :4].astype(np.float32),
                    det_bboxes[inds, 4:5].astype(np.float32),
                ],
                axis=1,
            ).tolist()
        det_results = det_result
    return det_results


def plot_one_box(x, im, color=None, label=None, line_thickness=3, kpt_label=False, kpts=None, steps=2, orig_shape=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, (255,0,0), thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(' ')) > 1:
            label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)
    if kpt_label:
        plot_skeleton_kpts(im, kpts, steps, orig_shape=orig_shape)


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        
    return im

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    
colors = Colors()

class Yolov5PreProcess(object):
    def __init__(self, **kwargs):
        super(Yolov5PreProcess, self).__init__()
        self.img_mean = [0, 0, 0]
        self.img_std = [1, 1, 1]
        if 'img_mean' in kwargs.keys():
            self.img_mean = kwargs['img_mean']
        if 'img_std' in kwargs.keys():
            self.img_std = kwargs['img_std']
        self.trans = 0
        self.input_size = kwargs['input_size']
        self.stride = kwargs['stride']

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
        res['raw_img'] = copy.deepcopy(img)
        res["raw_shape"] = img.shape[:2]
        res['input_shape'] = self.input_size
        self.set_trans(res)

        img_resize = letterbox(img, tuple(self.input_size), stride=self.stride)[0]
        # normalize image
        img_input = img_resize.astype(np.float32)
        img_mean = np.array(self.img_mean, dtype=np.float32)
        img_std = np.array(self.img_std, dtype=np.float32)
        img_input = (img_input - img_mean) / img_std
        # expand dims
        img_input = img_input.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_input = np.ascontiguousarray(img_input)
        img_input = np.expand_dims(img_input, axis=0)

        return img_input


class Yolov5PostProcess(object):
    def __init__(self, **kwargs):
        super(Yolov5PostProcess, self).__init__()
        self.strides = kwargs["strides"]
        self.reg_max = kwargs["reg_max"]
        self.num_candidate = kwargs["num_candidate"]
        self.prob_threshold = kwargs["prob_threshold"]
        self.iou_threshold = kwargs["iou_threshold"]
        self.top_k = kwargs["top_k"]
        # self.num_classes = kwargs["num_classes"]
        self.agnostic_nms = False
        # self.class_names = kwargs["class_names"]

    def draw_image(self, img, results, class_names, colors):
        raw_img = copy.deepcopy(img)
        if not isinstance(results, dict):
            return raw_img
        bbox, label, score, keypoints = results["bbox"], results["label"], results["score"], results["keypoints"]

        img = raw_img.copy()
        all_box = [
            [
                x,
            ]
            + y
            + [
                z,
            ]
            for x, y, z in zip(label, bbox, score)
        ]
        img_draw = overlay_bbox_cv(img, all_box, class_names, _COLORS=colors)
        for kpts in keypoints:
            img_draw = plot_skeleton_kpts(img_draw, kpts, 3, orig_shape=raw_img.shape[:2])
        return img_draw

    def __call__(self, outputs, trans):
        device = "cpu"
        input_size, raw_size, im0 = trans['input_shape'], trans['raw_shape'], trans['raw_img']
        z = list()
        grid, anchor_grid = [torch.zeros(1)] * 3, [torch.zeros(1)] * 3
        output = [v for _, v in outputs.items()]
        x, z = [], []
        for i in range(0,len(output)-1, 2):
            t = np.concatenate([output[i], output[i+1]], axis=1)
            t = torch.from_numpy(t).to(device)
            x.append(t)
        grid = [torch.zeros(1)] * 4 
        anchor_grid = [torch.zeros(1)] * 4
        stride = [8, 16, 32, 64]
        for i in range(4):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, 3, 57, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]
            grid[i], anchor_grid[i]= _make_grid(nx, ny, i)[0].to(x[i].device), _make_grid(nx, ny, i)[1].to(x[i].device)
            kpt_grid_x = grid[i][..., 0:1]
            kpt_grid_y = grid[i][..., 1:2]

            y = x_det.sigmoid()

            xy = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * anchor_grid[i].view(1, 3, 1, 1, 2)  
            x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * stride[i]  # xy
            x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * stride[i]  # xy
            x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

            y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

            
            z.append(y.view(bs, -1, 57))
        pred = (torch.cat(z, 1), x)[0] 

        # NMS
        pred = non_max_suppression(pred, conf_thres=self.prob_threshold, iou_thres=self.iou_threshold, classes=None, \
                                    agnostic=False, kpt_label=True)
        bbox, score, label, keypoints = list(), list(), list(), list()
        kpt_label = True
        for i, det in enumerate(pred):
            # Rescale boxes from img_size to im0 size
            scale_coords(input_size, det[:, :4], im0.shape, kpt_label=False)
            scale_coords(input_size, det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

            # Write results
            for det_index, (*xyxy, conf, cls) in enumerate(det[:,:6]):
                c = int(cls)  # integer class
                kpts = det[det_index, 6:]
                # plot_one_box(xyxy, im0, label=f'person {conf:.2f}', color=colors(c, True), 
                    # line_thickness=3, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                bbox.append([int(xy+0.5) for xy in xyxy])
                score.append(float(conf))
                label.append(int(cls))
                keypoints.append([float(kpt) for kpt in kpts])

        # cv2.imwrite("test.jpg", im0)
        # print("test")
        
        return dict(
            bbox=bbox,
            label=label,
            score=score,
            keypoints=keypoints)

pre_post_instances = {"yolov5": ["Yolov5PreProcess", "Yolov5PostProcess"]}

normalizations = {"yolov5": [[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]]}

strides = {'yolov5': [8, 16, 32]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='yolov5n_640_v2_simplify.onnx',
                        # default='work_dir/yolov5n_640_v2_simplify_qat_last.onnx',
                        # default="work_dir/yolov5n_640_v2_simplify_cross_layer_equalization.onnx",
                        # default='trained_models/pose-estimation/yolov5n_640_v2_simplify.onnx',
                        # default="work_dir/yolov5n_640_v2_simplify_cross_layer_equalization_bias_correction.onnx",
                        # default="work_dir/yolov5n_640_v2_simplify_bias_correction.onnx",
                        # default="work_dir/yolov5n_640_v2_simplify_adaround.onnx",
                        # default="work_dir/yolov5n_640_v2_simplify_adaround_cross_layer_equalization.onnx",
                        # default="work_dir/yolov5n_640_v2_simplify_cross_layer_equalization_adaround.onnx",
                        # yolov5n_320_v1_simplify, yolov5s_320_v1_simplify, nanodet_plus_m_1.0x_320_v2_simplify,
                        # default='./best.onnx',
                        # default='./best_1.onnx',
                        help='checkpoint path')
    parser.add_argument('--quan_dataset_path', type=str, 
                        # default='/buffer/crowdhuman/val',
                        default='/buffer/coco/cocoperson/images',
                        help='quantize image path')
    parser.add_argument('--ann_path', type=str, 
                        # default='/buffer/coco/annotations/person_keypoints_val2017.json',
                        default='/buffer/coco/coco_kps_100/keypoint.json',
                        help='eval anno path')
    parser.add_argument('--dataset_path', type=str, 
                        # default='/buffer/coco/person_yolo/val/images',
                        default='/buffer/coco/coco_kps_100/images',
                        help='eval image path')
    parser.add_argument('--model_name', type=str, default="yolov5") # nano | yolov5
    parser.add_argument('--input_size', type=list, default=[640, 640])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--results_path', type=str, 
                        # default='./work_dir/tmpfiles/crowdhuman_det_resluts',
                        default='./work_dir/tmpfiles/coco_person_results',
                        )
    parser.add_argument('--topk', type=int, default=-1)
    parser.add_argument('--reg_max', type=int, default=7)
    parser.add_argument('--prob_threshold', type=float, default=0.3) #origin: 0.3, new: 0.3
    parser.add_argument('--iou_threshold', type=float, default=0.6)
    parser.add_argument('--num_candidate', type=int, default=1000)  #origin: 1000, new: 300
    parser.add_argument('--draw_result', type=bool, default=True)
    parser.add_argument('--agnostic_nms', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')       
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)  
    parser.add_argument('--log_level', type=int, default=30) 
    parser.add_argument('--error_analyzer', type=bool, default=False) 
    parser.add_argument('--is_calc_error', type=bool, default=False) #whether to calculate each layer error
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # root_dir = '/home/shiqing/Downloads/onnx_converter'
    if args.debug:
        eval_mode = 'single'
        acc_error = False
    else:
        eval_mode = 'dataset'
        acc_error = True
        
    normalization = normalizations[args.model_name]
    kwargs_preprocess = {
        "img_mean": normalization[0],
        "img_std": normalization[1],
        "input_size": args.input_size,
        "stride": 64
    }
    kwargs_postprocess = {
        "reg_max": args.reg_max,
        "top_k": args.topk,
        "prob_threshold": args.prob_threshold,
        "iou_threshold": args.iou_threshold,
        "num_candidate": args.num_candidate,
        "strides": strides[args.model_name],
        "agnostic_nms": args.agnostic_nms,
        "num_classes": args.num_classes,
    }

    preprocess = eval(pre_post_instances[args.model_name][0])(**kwargs_preprocess)
    postprocess = eval(pre_post_instances[args.model_name][1])(**kwargs_postprocess)

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
                    'error_metric': ['L1', 'L2', 'Cosine'],
                    }

    kwargs_bboxeval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_pose_estimation.log',  
        'log_level': args.log_level,     
        # 'is_stdout': args.is_stdout,   
        'img_prefix': 'jpg',            
        "save_key": "mAP",
        "draw_result": args.draw_result,
        "_COLORS": _COLORS,
        "class_names": class_names,
        "process_args": process_args,
        'is_calc_error': args.is_calc_error,
        "fp_result": args.fp_result,
        "eval_mode": eval_mode,  # single quantize, dataset quatize
        'acc_error': acc_error,
    }

    cocoeval = CocoEval(**kwargs_bboxeval)
    # cocoeval.set_colors(colors=_COLORS)
    cocoeval.set_class_names(names=class_names)
    cocoeval.set_draw_result(is_draw=args.draw_result)
    cocoeval.set_iou_threshold(iou_threshold=args.iou_threshold)
    # quan_dataset_path, images_path, ann_path, input_size, normalization, save_eval_path
    evaluation, tb = cocoeval(
        quan_dataset_path=args.quan_dataset_path, dataset_path=args.dataset_path,
        ann_path=args.ann_path, input_size=args.input_size, normalization=normalization,
        save_eval_path=args.results_path)
    if args.is_calc_error:
        cocoeval.collect_error_info()      
    if args.error_analyzer:
        cocoeval.error_analysis() 
    if args.export:
        cocoeval.export()
    print(tb)