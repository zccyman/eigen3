# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/24 9:58
# @File     : test_retina_face_detection.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import glob
import json
import linecache
import re
from OnnxConverter import OnnxConverter
import prettytable as pt
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
import tqdm


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

def get_line_num_lst(txt_path):
    """得到首字母为#的行号"""

    txt_length = len(linecache.getlines(txt_path))
    txt_name_line_num_lst = []
    for j in range(txt_length):
        line_content = linecache.getline(txt_path, j)
        search_obj = re.search(r"# .*", line_content, re.M | re.I)
        if search_obj:
            txt_name_line_num_lst.append(j)

    return txt_name_line_num_lst, txt_length


def get_current_line_num(txt_path, image_name):
    """根据image_name定位行号"""

    txt_length = len(linecache.getlines(txt_path))
    for j in range(txt_length):
        line_content = linecache.getline(txt_path, j)
        search_obj = re.search(f"# .*{os.path.basename(image_name)}", line_content, re.M | re.I)
        if search_obj:
            return j


def get_gt_array(box_num):
    """根据box number 生成gt array"""

    gt_lst = []
    for i in range(1, box_num + 1):
        gt_lst.append([i])

    return np.array(gt_lst)


def get_box_by_line_index(txt_path, start_index, end_index):
    """根据line index get box"""

    box_lst = []
    for i in range(start_index + 1, end_index):
        line_content = linecache.getline(txt_path, i)
        line_content_lst = line_content.split(" ")
        box_lst.append(list(map(int, line_content_lst[:4])))

    return np.array(box_lst)


def get_ground_truth(gt_dir, event_lst=["flickr_0", "flickr_2", "flickr_3"]):
    file_name_dct, facebox_dct, gt_dct = {}, {}, {}
    for event in event_lst:
        event_name_dct, event_facebox_dct, event_gt_dct = {}, {}, {}
        # label
        txt_file_path = f"{gt_dir}/{event}/label.txt"
        txt_name_line_num_lst, txt_length = get_line_num_lst(txt_path=txt_file_path)
        # image
        img_dir = f"{gt_dir}/{event}/image"
        img_name_lst = os.listdir(img_dir)
        img_name_lst = img_name_lst.remove('label.txt') if 'label.txt' in img_name_lst else img_name_lst
        # print(img_name_lst)
        for i, img_name in enumerate(img_name_lst):
            current_line_num = get_current_line_num(txt_path=txt_file_path, image_name=img_name)
            # print(img_name, current_line_num)
            current_index = txt_name_line_num_lst.index(current_line_num)
            if current_index == len(txt_name_line_num_lst) - 1:
                next_img_line_num = txt_length
                box_array = get_box_by_line_index(
                    txt_path=txt_file_path, start_index=current_line_num, end_index=next_img_line_num + 1)
            else:
                next_img_line_num = txt_name_line_num_lst[current_index + 1]
                box_array = get_box_by_line_index(
                    txt_path=txt_file_path, start_index=current_line_num, end_index=next_img_line_num)
            event_name_dct[i] = img_name
            event_facebox_dct[i] = box_array
            event_gt_dct[i] = get_gt_array(box_num=box_array.shape[0])
        file_name_dct[event] = event_name_dct
        facebox_dct[event] = event_facebox_dct
        gt_dct[event] = event_gt_dct

    return facebox_dct, event_lst, file_name_dct, gt_dct


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    # b = lines[0].rstrip('\r\n').split(' ')[:-1]
    # c = float(b)
    # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] == '':
            continue
        # a = float(line[4])
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    # print(events)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            # print(os.path.join(event_dir, imgtxt))
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    # _pred = _pred.astype('double')
    # overlaps = bbox_overlaps(_pred[:, :4], _gt)
    overlaps = box_iou(torch.from_numpy(_pred[:, :4]), torch.from_numpy(_gt))
    if isinstance(overlaps, torch.Tensor):
        overlaps = overlaps.cpu().numpy()

    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        # print(max_overlap, max_idx)
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, gt, ignore, iou_thresh):
    pred_recall, proposal_list = image_eval(pred_info, gt, ignore, iou_thresh)
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        # print(thresh, r_index)
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # print(i)

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    # print(ap)
    return ap


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    # print(pred)
    if len(pred) != 0:
        _min = np.min(pred[:, -1])
        _max = np.max(pred[:, -1])
        max_score = max(_max, max_score)
        min_score = min(_min, min_score)

    diff = max_score - min_score
    if len(pred) != 0:
        if diff == 0:
            diff = 0.00001
        pred[:, -1] = (pred[:, -1] - min_score) / diff


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        if pr_curve[i, 0] == 0:
            continue
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def compute_ap(pred_info, pr_curve, gt_boxes, keep_index, thresh_num, iou_thresh=0.5):
    if len(gt_boxes) != 0 and len(pred_info) != 0:
        ignore = np.zeros(gt_boxes.shape[0])
        if len(keep_index) != 0:
            ignore[keep_index - 1] = 1
            _img_pr_info = img_pr_info(thresh_num, pred_info, gt_boxes, ignore, iou_thresh)
            pr_curve += _img_pr_info
    return pr_curve


def compute_tp_fp(dt_per_classes_dict, cate, gt_data, MIN_OVERLAP=0.5):
    dt_list = dt_per_classes_dict[cate]
    # {cate_1: [], cate2: [], ...}
    nd = len(dt_list)
    tp = [0] * nd  # creates an array of zeros of size nd
    fp = [0] * nd
    # 遍历所有候选预测框，判断TP/FP/FN
    for idx, dt_data in enumerate(dt_list):
        # {"confidence": "0.999", "file_id": "cucumber_61", "bbox": [16, 42, 225, 163]}
        # 读取保存的信息
        file_id = dt_data['file_id']
        dt_bbox = dt_data['bbox']
        confidence = dt_data['confidence']

        # 逐个计算预测边界框和对应类别的真值标注框的IoU，得到其对应最大IoU的真值标注框
        ovmax = -1
        gt_match = -1
        # load detected object bounding-box
        for g_idx, obj in enumerate(gt_data[file_id]):
            # {"cate": "cucumber", "bbox": [23, 42, 206, 199], "used": true}
            # 读取保存的信息
            obj_cate = obj['cate']
            obj_bbox = obj['bbox']

            # look for a class_name match
            if obj_cate == cate:
                bi = [
                    max(dt_bbox[0], obj_bbox[0]),
                    max(dt_bbox[1], obj_bbox[1]),
                    min(dt_bbox[2], obj_bbox[2]),
                    min(dt_bbox[3], obj_bbox[3])
                ]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (dt_bbox[2] - dt_bbox[0] + 1) * (dt_bbox[3] - dt_bbox[1] + 1) + \
                         (obj_bbox[2] - obj_bbox[0] + 1) * (obj_bbox[3] - obj_bbox[1] + 1) \
                         - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj
        # 如果大于最小IoU阈值，还需要进一步判断是否为TP
        if ovmax >= MIN_OVERLAP:
            if not bool(gt_match["used"]):
                # true positive
                tp[idx] = 1
                gt_match["used"] = True
            else:
                # false positive (multiple detection)
                fp[idx] = 1
        else:
            # false positive
            fp[idx] = 1

    return tp, fp


def compute_precision_recall(tp, fp, gt_per_classes_num):
    """
    计算不同阈值下的precision/recall
    """
    # compute precision/recall
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(val) / gt_per_classes_num
    # print(rec)
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    return prec, rec


def voc_ap_henson(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


class PreProcess(object):
    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = [104, 117, 123]
        self.img_std = [1.0, 1.0, 1.0]
        self.trans = 0
        self.input_size = [320, 256]

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

class ISPPreProcess(object):
    def __init__(self, **kwargs):
        super(ISPPreProcess, self).__init__()
        self.img_mean = [104, 117, 123]
        self.img_std = [1.0, 1.0, 1.0]
        self.trans = 0
        self.input_size = [320, 256]

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
        img_final = cv2.copyMakeBorder(resize_img, top, bottom+1, left, right, cv2.BORDER_CONSTANT, value=color)

        img_final = img_final - (104, 117, 123)

        img_final = np.transpose(img_final, (2,0,1)).astype(np.int32)
        
        img_final = ((img_final * 885) >> 10).astype(np.int8)

        img_final = np.expand_dims(img_final, axis=0)
        
        # self.ratio, self.w_pad, self.h_pad = r, w_pad, h_pad
        self.trans = dict(target_w=target_w, target_h=target_h, w_pad=w_pad, h_pad=h_pad,
                          src_shape=img.shape[:2], ratio=r)

        return img_final

class PostProcess(object):
    def __init__(self, **kwargs):
        super(PostProcess, self).__init__()
        self.steps = [8, 16, 32, 64]
        self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.nms_threshold = 0.3
        self.variances = [0.1, 0.2]
        self.prob_threshold = 0.7
        self.top_k = 1000

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

class RetinaFaceDataset(object):
    def __init__(self, **kwargs):
        assert {'evaluation_dataset_dir'} <= set(kwargs.keys()), \
            'Arguments coco_json_path and dataset_dir are required, please make sure them defined in your argmuments'
        self.dataset_dir = kwargs["evaluation_dataset_dir"]
        assert os.path.exists(self.dataset_dir), 'dataset root path is not found, please check the folder exists'
        self.start_num = -1
        self.file_name_dct, self.facebox_dct, self.gt_dct  = self.parse_dataset(**kwargs)
        self.num_samples = len(self.gt_dct.keys())
        assert self.num_samples > 1, 'num_sample must be great than 1!'
    
    @staticmethod
    def preprocess(img):
        return PreProcess()(img)
    
    @staticmethod
    def isppreprocess(img):
        return ISPPreProcess()(img)

    def get_ann_path(self):
        return self.ann_path

    def get_image_dir(self):
        return self.image_dir

    def id_to_image_path(self, id):
        assert id in self.image_path_dict.keys(), 'error: image_id not found'
        return self.image_path_dict[id]

    def parse_dataset(self, **kwargs):
        # img_dct = dict()
        # imgs = [] 
        
        file_name_dct, facebox_dct, gt_dct = {}, {}, {}
        # label
        txt_file_path = f"{self.dataset_dir}/label.txt"
        txt_name_line_num_lst, txt_length = get_line_num_lst(txt_path=txt_file_path)
        # image
        img_dir = f"{self.dataset_dir}/image"
        img_name_lst = glob.glob(img_dir + '/*.jpg')
        img_name_lst.sort()
        
        # print(img_name_lst)
        for i, img_name in enumerate(img_name_lst):
            current_line_num = get_current_line_num(txt_path=txt_file_path, image_name=img_name)
            # print(img_name, current_line_num)
            current_index = txt_name_line_num_lst.index(current_line_num)
            if current_index == len(txt_name_line_num_lst) - 1:
                next_img_line_num = txt_length
                box_array = get_box_by_line_index(
                    txt_path=txt_file_path, start_index=current_line_num, end_index=next_img_line_num + 1)
            else:
                next_img_line_num = txt_name_line_num_lst[current_index + 1]
                box_array = get_box_by_line_index(
                    txt_path=txt_file_path, start_index=current_line_num, end_index=next_img_line_num)
            file_name_dct[i] = img_name
            # imgs.append(os.path.join(img_dir, "image", img_name))
            facebox_dct[i] = box_array
            gt_dct[i] = get_gt_array(box_num=box_array.shape[0])
        
        return file_name_dct, facebox_dct, gt_dct   
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.start_num += 1
        if self.start_num >= self.num_samples:
            raise StopIteration()

        return self.file_name_dct[self.start_num], self.facebox_dct[self.start_num], self.gt_dct[self.start_num]

class AlfwEval(object):
    def __init__(self, **kwargs):
        super(AlfwEval, self).__init__()

        self.iou_threshold = 0.3
        self.prob_threshold = 0.7
        
        self.draw_result = False
        self.dataset = kwargs["dataset_eval"]   
        self.postprocess = PostProcess()
        
        self._preprocess = PreProcess()  
        self._isp_preprocess = ISPPreProcess()   
    
    @staticmethod
    def draw_image(img, detects, class_names, colors, prob_threshold=0.6):
        for det in detects:
            if det[4] < prob_threshold:
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
            
    def get_map(self, det_boxes, label, gt_boxes, count_face):
        gt_boxes_ = copy.deepcopy(gt_boxes)
        det_boxes_ = copy.deepcopy(det_boxes)
        tp, fp = compute_tp_fp(det_boxes_, label, gt_boxes_, self.iou_threshold)
        prec, rec = compute_precision_recall(tp, fp, count_face)

        # ap = voc_ap2(rec[:], prec[:])
        ap, mrec, mprec = voc_ap_henson(rec[:], prec[:])
        recall = np.mean(mrec)
        precision = np.mean(mprec)

        return ap, recall, precision
    
    def draw_table(self, res):
        align_method = 'c'
        table_head = ['pedestrain'] # type: ignore
        table_head_align = [align_method]
        
        quant_result = ["quantion"]
        for k, v in res['qaccuracy'].items():
            quant_result.append("{:.5f}".format(v))
            table_head.append(k)
            table_head_align.append(align_method)

        tb = pt.PrettyTable()
        tb.field_names = table_head

        tb.add_row(quant_result)

        for head, align in zip(table_head, table_head_align):
            tb.align[head] = align
            
        return tb  
    
    def get_results(self, accuracy):
        # errors = self.process.get_layer_average_error()
        # max_errors = self.print_layer_average_error(errors)
        if accuracy is not None:
            tb = self.draw_table(accuracy)
        else:
            tb = ''
            
        print(tb)    
    
    def __call__(self, converter):

        thresh_num = 1000

        count_face, qdet_face, tdet_face = 0, 0, 0

        qdet_boxes, gt_boxes = dict(face=list()), dict()
        
        for data in tqdm.tqdm(iter(self.dataset)):
            image_path, facebox_dct, gt_dct = data
            
            img = cv2.imread(image_path)
            
            results = converter.model_simulation(img)            
            results_cvt = results['result_converter']
            self.preprocess(img)
            trans = self.preprocess.get_trans()
            pred_cvt = self.postprocess(results_cvt, trans=trans)
            
            for output in pred_cvt:
                qdet_face += 1
                qdet_boxes['face'].append(
                    dict(confidence=output[4],
                        file_id=os.path.basename(image_path),
                        bbox=output[:4].astype(np.int32).tolist()))
                    
            file_id, gts = os.path.basename(image_path).split('.')[0], list()
            for gt in facebox_dct:
                xyxy = copy.deepcopy(gt)
                
                count_face += 1
                xyxy[2:] = gt[2:] + gt[:2]
                gts.append(dict(
                    cate='face',
                    used=False,
                    file_id=file_id,
                    bbox=xyxy.tolist()))
            gt_boxes[os.path.basename(image_path)] = gts
                            
        qap, qrecall, qprecision = self.get_map(qdet_boxes, "face", gt_boxes, count_face)
        accuracy = dict(qaccuracy=dict(AP=qap, Recall=qrecall, Precision=qprecision))

        tb = self.get_results(accuracy)
        
        return accuracy, tb  
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_json', default='/home/shiqing/Downloads/test_package/saturation/onnx-converter/tests/test_interface/arguments_retinaface.json', type=str)
    parser.add_argument('--model_export', type=bool, default=True)
    parser.add_argument('--perf_analyse', type=bool, default=True)
    parser.add_argument('--vis_qparams', type=bool, default=False)
    parser.add_argument('--vis_result', type=bool, default=False)
    parser.add_argument('--isp_inference', type=bool, default=True)
    parser.add_argument('--mem_addr', type=str, default='psram')
    args = parser.parse_args()
    return args

def ispdata_export(converter: OnnxConverter, abgr_file: str):
    # Run model convert and export(optional)
    abgr_img = np.fromfile(abgr_file,dtype=np.uint8)
    dummy_input = abgr_img.reshape(1080,1920,4)[:,:,1:4]
    preprocess = ISPPreProcess()
    postprocess = PostProcess()
    converter.reset_preprocess(preprocess)
    results = converter.model_simulation(dummy_input, isp_data=True)            
    results_cvt = results['result_converter']
    trans = preprocess.get_trans()
    pred_cvt = postprocess(results_cvt, trans=trans)
    qres = AlfwEval.draw_image(copy.deepcopy(dummy_input), pred_cvt, class_names=None, colors=[(1, 0, 0)])
    cv2.imwrite("/home/shiqing/Downloads/test_package/saturation/onnx-converter/test_retina_Face_abgr.jpg", qres)  ### results for draw bbox
    converter.model_export(dummy_input=dummy_input)
    print("******* Finish model convert *******")
    
def single_img(converter: OnnxConverter, image_path: str):
    # image_path="/buffer/ssy/export_inner_model/retinaface/retinaface-abgr-1920x1080.png"
    preprocess = PreProcess() #ISPPreProcess()
    postprocess = PostProcess()
    img = cv2.imread(image_path)    
    preprocess(img)    
    results = converter.model_simulation(img, isp_data=False)            
    results_cvt = results['result_converter']
    trans = preprocess.get_trans()
    pred_cvt = postprocess(results_cvt, trans=trans)

    qres = AlfwEval.draw_image(copy.deepcopy(img), pred_cvt, class_names=None, colors=[(1, 0, 0)])
    cv2.imwrite("/home/shiqing/Downloads/test_package/saturation/onnx-converter/test_retina.jpg", qres)  ### results for draw bbox

if __name__ == '__main__':
    print("******* Start model convert *******")
    # parse user defined arguments
    argsparse = parse_args()
    args_json_file = argsparse.args_json
    flag_model_export = argsparse.model_export
    flag_perf_analyse = argsparse.perf_analyse
    flag_vis_qparams = argsparse.vis_qparams 
    flag_vis_result = argsparse.vis_result 
    flag_isp_inference = argsparse.isp_inference 
    assert os.path.exists(args_json_file), "Please check argument json exists"
    args = json.load(open(args_json_file, 'r'))
    args_cvt = args['converter_args']
    
    calibration_dataset_dir = args_cvt["calibration_dataset_dir"]
    evaluation_dataset_dir = args_cvt["evaluation_dataset_dir"]
    eval_dataset = RetinaFaceDataset(evaluation_dataset_dir=evaluation_dataset_dir)
    args_cvt["transform"] = eval_dataset.preprocess
    # Build onnx converter
    converter = OnnxConverter(**args_cvt)     
    
    converter.load_model("/buffer/trained_models/face-detection/slim_special_Final_simplify_removed_pad.onnx")
    
    # Calibration
    converter.calibration(calibration_dataset_dir)
    print("calibration done!")
    
    # Build evaluator
    evaluator = AlfwEval(dataset_eval=eval_dataset)

    # Evaluate quantized model accuracy
    evaluator(converter=converter)
    print("******* Finish model evaluate *******")
    
    if args_cvt['layer_error_args']['do_check_error'] == 2:
        converter.error_analysis()
        print("******* Finish error analysis *******") 
    
    if flag_vis_qparams:
        converter.visualize_qparams()
        print("******* Finish qparams visualization *******") 

    if flag_vis_result:
        image_path = "/buffer/ssy/export_inner_model/retinaface/retinaface-abgr-1920x1080.png"
        single_img(converter, image_path)
    
    if flag_model_export:
        #Run model convert and export(optional)
        image_file = "/buffer/calibrate_dataset/face_recognition/9_1_70.jpg"
        dummy_input = cv2.imread(image_file)
        converter.model_export(dummy_input=dummy_input)
        os.system(f"cp {image_file} work_dir/")
        print("******* Finish model convert *******")
        
    if flag_isp_inference:
        print("******* Start isp data model convert *******")
        # Run model convert and export(optional)
        abgr_file = "/buffer/ssy/export_inner_model/retinaface/retinaface-abgr-1920x1080.bin"
        ispdata_export(converter, abgr_file)
        print("******* Finish isp data model convert *******")
    
    # performance analysis
    # if flag_perf_analyse:
    #     time = converter.perf_analyze(mem_addr = argsparse.mem_addr)
    #     print("******* The estimated time cost is %f ms *******"%(time/1000))
    #     print("******* Finish performance analysis *******")
