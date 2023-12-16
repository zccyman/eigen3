# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/2/7 16:33
# @File     : alfw.py
import copy
import glob
import linecache
import os
import re

import cv2
import numpy as np
import torch
import tqdm
from scipy.io import loadmat
from torchvision.ops import box_iou
from eval import Eval

try:
    
    from tools import ModelProcess, OnnxruntimeInfer
    from utils import Object
except:
    from onnx_converter.tools import ModelProcess, OnnxruntimeInfer
    from onnx_converter.utils import Object


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
        search_obj = re.search(f"# .*{image_name}", line_content, re.M | re.I)
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


class AlfwEval(Eval):
    def __init__(self, **kwargs):
        super(AlfwEval, self).__init__(**kwargs)

        self.iou_threshold = kwargs['iou_threshold']
        self.prob_threshold = kwargs['prob_threshold']
        self.eval_mode = kwargs['eval_mode']
        self.process_args = kwargs['process_args']
        self.is_calc_error = kwargs['is_calc_error']
        # self.ModelProcess = kwargs['ModelProcess']
        # self.process = ModelProcess(**self.process_args)
        self.draw_result = kwargs['draw_result']
        self.fp_result = kwargs['fp_result']
        self.acc_error = kwargs['acc_error']
        self.img_prefix = kwargs['img_prefix']
        # self.is_stdout = self.process_args['is_stdout']
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)

        if 'postprocess' in self.process_args.keys():
            setattr(self, 'postprocess', self.process_args['postprocess'])
        
        self.process = ModelProcess(**self.process_args)
        if self.is_calc_error:
            self.process.set_onnx_graph(False)
            # self.process.onnx_graph = False
        
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
        
    def __call__(self, quan_dataset_path, dataset_path, ann_path, event_lst, save_dir, color=(0,0,255)):
        global img_dct
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        get_ims = lambda path, prefix: glob.glob(os.path.join(path, prefix))
        imgs = get_ims(quan_dataset_path, '*.' + self.img_prefix)
        for sub_dir in os.listdir(quan_dataset_path):
            if sub_dir in event_lst:
               imgs.extend(get_ims(os.path.join(quan_dataset_path, sub_dir, 'image'), '*.' + self.img_prefix)) 
        imgs.sort()    
        output_names, input_names = self.process.get_output_names(), self.process.get_input_names()
        onnxinferargs = copy.deepcopy(self.process_args)
        onnxinferargs.update(out_names=output_names, input_names=input_names)
        onnxinfer = OnnxruntimeInfer(**onnxinferargs)
        if self.eval_mode == 'dataset':
            self.process.quantize(fd_path=imgs, is_dataset=True)
            
        imgs = list()
        file_name_dct, facebox_dct, gt_dct = {}, {}, {}
        for event in event_lst:
            event_name_dct, event_facebox_dct, event_gt_dct = {}, {}, {}
            # label
            txt_file_path = f"{ann_path}/{event}/label.txt"
            txt_name_line_num_lst, txt_length = get_line_num_lst(txt_path=txt_file_path)
            # image
            img_dir = f"{dataset_path}/{event}/image"
            img_name_lst = os.listdir(img_dir)
            img_name_lst.sort()
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
                imgs.append(os.path.join(img_dir, img_name))
                event_facebox_dct[i] = box_array
                event_gt_dct[i] = get_gt_array(box_num=box_array.shape[0])
            file_name_dct[event] = event_name_dct
            facebox_dct[event] = event_facebox_dct
            gt_dct[event] = event_gt_dct

        event_num = len(event_lst)  # event_num:61
        thresh_num = 1000

        count_face, qdet_face, tdet_face = 0, 0, 0
        quan_pr_curve = np.zeros((thresh_num, 2)).astype('float')

        qdet_boxes, tdet_boxes, gt_boxes = dict(face=list()), dict(face=list()), dict()
        pbar = tqdm.tqdm(range(event_num)) if self.is_stdout else range(event_num)
        for i in pbar:
            if self.is_stdout:
                pbar.set_description('Processing {}'.format('easy'))
            event_name = str(event_lst[i])  # 0--Parade, 1--Handshaking, 10--People_Marching
            img_dct = file_name_dct[event_name]
            gt_bbx_dct = facebox_dct[event_name]
            sub_gt_dct = gt_dct[event_name]
            
            img_dct_ = tqdm.tqdm(range(len(img_dct))) if self.is_stdout else range(len(img_dct))
            for k in img_dct_:
                image_path = os.path.join(dataset_path, event_name, 'image', str(img_dct[k]))
                # image_path = '/home/ubuntu/zhangcc/code/2022/onnx-converter/pic1_1.jpg'
                
                img = cv2.imread(image_path)
                if self.eval_mode == "single":
                    self.process.quantize(img, is_dataset=False)
                    
                if self.fp_result:
                    true_outputs = onnxinfer(in_data=img)
                else:
                    true_outputs = None
                
                if self.is_calc_error:
                    self.process.checkerror(img, acc_error=self.acc_error)
                else:        
                    self.process.dataflow(img, acc_error=self.acc_error, onnx_outputs=true_outputs)
                # if 'analysis_cfg' in self.process_args.keys() and self.fp_result:
                #     self.process.checkerror_weight(onnx_outputs=None)
                #     self.process.checkerror_feature(onnx_outputs=true_outputs)   
                                    
                outputs = self.process.get_outputs()
                q_out, t_out = outputs['qout'], outputs['true_out']

                # quan_boxes = copy.deepcopy(outputs[:, :5])
                for output in q_out:
                    qdet_face += 1
                    qdet_boxes['face'].append(
                        dict(confidence=output[4],
                            file_id=img_dct[k],
                            bbox=output[:4].astype(np.int32).tolist()))
                if self.fp_result:
                    for output in t_out:
                        tdet_face += 1
                        tdet_boxes['face'].append(
                            dict(confidence=output[4],
                                file_id=img_dct[k],
                                bbox=output[:4].astype(np.int32).tolist()))
                file_id, gts = os.path.basename(img_dct[k]).split('.')[0], list()
                for gt in gt_bbx_dct[k]:
                    xyxy = copy.deepcopy(gt)
                    # if gt[-1] < 32 and gt[-2] < 32:
                    #     print('invaild face!')
                    #     continue
                    count_face += 1
                    xyxy[2:] = gt[2:] + gt[:2]
                    gts.append(dict(
                        cate='face',
                        used=False,
                        file_id=file_id,
                        bbox=xyxy.tolist()))
                gt_boxes[img_dct[k]] = gts

                if self.draw_result and outputs is not None:
                    if hasattr(self, 'postprocess') and hasattr(self.postprocess, 'draw_image'):
                        draw_image = self.postprocess.draw_image
                    else:
                        draw_image = self.draw_image
                    qres = draw_image(copy.deepcopy(img), q_out, class_names=None, colors=[(1, 0, 0)])
                    if self.fp_result:
                        qres = draw_image(qres, t_out, class_names=None, colors=[(0, 0, 1)])
                    cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), qres)  ### results for draw bbox
                    
                if k == 0 and self.eval_first_frame: break

        if not self.is_calc_error and self.process.onnx_graph:
            img = cv2.imread(os.path.join(dataset_path, event_lst[0], 'image', str(img_dct[0])))
            true_outputs = self.process.post_quan.onnx_infer(img)
            self.process.numpygraph(img, acc_error=True, onnx_outputs=true_outputs)
                            
        qap, qrecall, qprecision = self.get_map(qdet_boxes, "face", gt_boxes, count_face)
        if self.fp_result:
            tap, trecall, tprecision = self.get_map(tdet_boxes, "face", gt_boxes, count_face)
            accuracy = dict(qaccuracy=dict(AP=qap, Recall=qrecall, Precision=qprecision), 
                    faccuracy=dict(AP=tap, Recall=trecall, Precision=tprecision))
        else:
            accuracy = dict(qaccuracy=dict(AP=qap, Recall=qrecall, Precision=qprecision))

        tb = self.get_results(accuracy)
        # self.process.report()
        
        return accuracy, tb