#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/02/27 11:21:40
#@File    : test_hisense.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import copy
import os

import cv2
import numpy as np
import tensorflow as tf
from torchvision.ops import nms

from eval.alfw import AlfwEval

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FLAG():
    # GLOBAL PARAMETERS FOR HISENSE
    input_width = 192
    input_height = 192
    feature_list = [[6, 6], [12, 12]]
    anchor_6 = np.array([[37.14, 77.23], [29.48, 57.30], [
                        22.91, 44.78]], dtype=np.float32)
    anchor_12 = np.array([[17.59, 34.40], [13.09, 25.37],
                         [9.33, 17.88]], dtype=np.float32)
    anchor_list = [anchor_6, anchor_12]
    num_anchor = anchor_list[0].shape[0]
    mean = 120.0
    std = 80.0

    # INFERENCE PARAMETERS
    camera_height = 360
    camera_width = 640
    num_target = 1000
    nms_threshold = 0.3
    conf_threshold = 0.9


class PreProcess(object):

    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.img_mean = kwargs['img_mean']
        self.img_std = kwargs['img_std']
        self.input_size = kwargs['input_size']

        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img, color=(114, 114, 114)):
        target_w, target_h = self.input_size

        img_h, img_w = img.shape[:2]
        if img_w > img_h:
            r = target_w / img_w
            new_shape_w, new_shape_h = target_w, int(round(img_h * r))
            if new_shape_h > target_h:
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
            if new_shape_w > target_w:
                r = target_w / img_w
                new_shape_w, new_shape_h = target_w, int(round(img_h * r))
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h
            else:
                w_pad, h_pad = target_w - new_shape_w, target_h - new_shape_h

        w_pad /= 2
        h_pad /= 2

        resize_img = cv2.resize(img, (new_shape_w, new_shape_h),
                                interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
        left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))

        # 固定值边框，统一都填充color
        img_final = cv2.copyMakeBorder(resize_img,
                                       top,
                                       bottom,
                                       left,
                                       right,
                                       cv2.BORDER_CONSTANT,
                                       value=color)
        img_final = np.array(img_final, dtype=np.float32)
        img_final = (img_final - self.img_mean) / self.img_std
        img_final = np.transpose(img_final, [2, 0, 1])
        img_final = np.expand_dims(img_final, axis=0)
        img_final = img_final.astype(np.float32)

        # self.ratio, self.w_pad, self.h_pad = r, w_pad, h_pad
        self.trans = dict(target_w=target_w,
                          target_h=target_h,
                          w_pad=w_pad,
                          h_pad=h_pad,
                          src_shape=img.shape[:2],
                          ratio=r)

        return img_final


class PostProcess(object):

    def __init__(self, **kwargs):
        super(PostProcess, self).__init__()
        self.prob_threshold = kwargs['prob_threshold']
        self.nms_threshold = kwargs['nms_threshold']

    def iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                     np.maximum(inter_rect_y2 - inter_rect_y1, 0)

        area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
        return iou

    def sigmoid(self, x):
        x_ravel = x.ravel()  # 将numpy数组展平
        length = len(x_ravel)
        y = []
        for index in range(length):
            if x_ravel[index] >= 0:
                y.append(1.0 / (1 + np.exp(-x_ravel[index])))
            else:
                y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
        return np.array(y).reshape(x.shape)

    def pred_bbox_reorg(self, pred_bbox, feature_index, input_width):
        '''
        这个函数主要是将模型输出的offset尺度结果计算到real尺度
        :return:
        xy_gird: [feature_height,feature_width,1,2]
        pred_bbox: [feature_height,feature_width,num_anchor,4]
        pred_conf: [feature_height,feature_width,num_anchor,1]
        '''
        pred_bbox_temp = np.reshape(pred_bbox, [
            -1, FLAG.feature_list[feature_index][1],
            FLAG.feature_list[feature_index][0], FLAG.num_anchor, 5
        ])
        pred_xy, pred_wh, pred_conf = np.split(pred_bbox_temp, [2, 4], axis=-1)
        # print('pred_xy:', pred_xy.shape)
        # print('pred_wh:', pred_wh.shape)
        # print('pred_conf:', pred_conf.shape)

        # 设定ratio和anchor的参数
        ratio = input_width / FLAG.feature_list[feature_index][0]
        anchor = np.array(FLAG.anchor_list[feature_index])  # [num_anchor,2]
        # print('ratio:', ratio)

        # 生成用于xy计算的xy_offset
        x_grid_temp1 = np.arange(FLAG.feature_list[feature_index][0],
                                 dtype=np.int32)
        y_grid_temp1 = np.arange(FLAG.feature_list[feature_index][1],
                                 dtype=np.int32)
        x_grid_temp2, y_grid_temp2 = np.meshgrid(x_grid_temp1, y_grid_temp1)
        x_grid_temp3 = np.reshape(x_grid_temp2, (-1, 1))
        y_grid_temp3 = np.reshape(y_grid_temp2, (-1, 1))
        xy_grid = np.concatenate((x_grid_temp3, y_grid_temp3), axis=-1)
        xy_grid = np.reshape(xy_grid, [
            FLAG.feature_list[feature_index][1],
            FLAG.feature_list[feature_index][0], 1, 2
        ])
        xy_grid = xy_grid.astype(np.float32)  # [height,width,1,2]
        # print(xy_grid.reshape([-1,2]));

        # 计算xywh的real尺度
        xy = (self.sigmoid(pred_xy) + xy_grid) * ratio
        # print(xy.reshape([-1, 2]));
        wh = np.exp(
            pred_wh) * anchor  # [height,width,num_anchor,2] * [num_anchor,2]
        # print(wh.reshape([-1, 2]));
        pred_bbox = np.concatenate((xy, wh), axis=-1)
        return pred_bbox, pred_conf

    def bbox_processing(self, pred_bbox, feature_index, input_width):
        bbox_temp1, conf_temp1 = self.pred_bbox_reorg(pred_bbox, feature_index,
                                                      input_width)
        num_bbox = FLAG.feature_list[feature_index][0] * FLAG.feature_list[
            feature_index][1] * FLAG.num_anchor
        bbox_temp2 = np.reshape(bbox_temp1, (num_bbox, 4))
        conf_temp2 = self.sigmoid(conf_temp1)
        conf = np.reshape(conf_temp2, [num_bbox])

        x_center = bbox_temp2[:, 0:1]
        y_center = bbox_temp2[:, 1:2]
        width = bbox_temp2[:, 2:3]
        height = bbox_temp2[:, 3:4]

        x1 = x_center - 0.5 * width
        y1 = y_center - 0.5 * height
        x2 = x_center + 0.5 * width
        y2 = y_center + 0.5 * height
        bbox = np.concatenate((x1, y1, x2, y2), axis=-1)
        # print(bbox);
        return bbox, conf

    def res_reshape(self, real_output):
        real_output_names = [
            'ZSinp_Mnet_HEAD21/Conv2D_5:0',
            'ZSinp_Mnet_HEAD21/Conv2D_6:0',
            'ZSinp_Mnet_HEAD26/Conv2D_5:0',
            'ZSinp_Mnet_HEAD26/Conv2D_6:0',
        ]
                
        bbox6_data = real_output[real_output_names[0]].transpose(0, 2, 3, 1) #real_output['output.0']  #real_output[0]
        bbox6_data = bbox6_data.reshape([-1, 12])
        conf6_data = real_output[real_output_names[1]].transpose(0, 2, 3, 1) #real_output['output.1']  #real_output[2]
        conf6_data = conf6_data.reshape([-1, 3])
        # print(bbox6_data.shape)
        # print(conf6_data.shape)

        bbox12_data = real_output[real_output_names[2]].transpose(0, 2, 3, 1) #real_output['output.2']  #real_output[1]
        bbox12_data = bbox12_data.reshape([-1, 12])
        conf12_data = real_output[real_output_names[3]].transpose(0, 2, 3, 1) #real_output['output.3']  #real_output[3]
        conf12_data = conf12_data.reshape([-1, 3])
        # print(bbox12_data.shape)
        # print(conf12_data.shape)

        pred_bbox_6 = np.concatenate((bbox6_data, conf6_data), axis=-1)
        pred_bbox_12 = np.concatenate((bbox12_data, conf12_data), axis=-1)
        return pred_bbox_6, pred_bbox_12

    def draw_image(self, origin_image, post_result, class_names, colors):
        img_show = copy.deepcopy(origin_image)
        for i in range(len(post_result)):
            bbox_x1_input = int(post_result[i][0])
            bbox_y1_input = int(post_result[i][1])
            bbox_x2_input = int(post_result[i][2])
            bbox_y2_input = int(post_result[i][3])
            cv2.rectangle(img_show, (bbox_x1_input, bbox_y1_input),
                          (bbox_x2_input, bbox_y2_input), colors[0], 5)

        return img_show

    def __call__(self, outputs, trans):
        input_width = trans['target_w']
        w_pad = trans['w_pad']
        h_pad = trans['h_pad']
        resize = trans['ratio']
        origin_img_h, origin_img_w = trans['src_shape']
        """model post process"""
        # compute_out = [(com).astype(np.float32) for com in compute_out]
        pred_bbox_6, pred_bbox_12 = self.res_reshape(outputs)
        # pred_bbox_6, pred_bbox_12 = pred_bbox_6[0], pred_bbox_12[0]

        single_bbox_6, single_conf_6 = self.bbox_processing(
            pred_bbox_6, 0, input_width)
        single_bbox_12, single_conf_12 = self.bbox_processing(
            pred_bbox_12, 1, input_width)
        bbox = np.concatenate((single_bbox_6, single_bbox_12), axis=0)
        conf = np.concatenate((single_conf_6, single_conf_12), axis=0)
        # # print(np.concatenate((bbox, conf.reshape([-1,1])), axis=-1))

        # print('bbox', bbox.shape)
        # print('conf', conf.shape)

        indexes = conf > self.prob_threshold
        scores = conf[indexes]
        boxes = bbox[indexes, :]

        # detects = np.concatenate((boxes, scores), axis=-1)
        detects = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32,
                                                                   copy=False)

        # sort
        indexes = np.argsort(scores)[::-1]
        detects = detects[indexes, :]
        # print('detects', detects.shape)

        detects_num = detects.shape[0]  #### bbox return to original image
        for i in range(detects_num):
            detects = np.array(detects)
            detects[i, :4] = detects[i, :4] / resize
            detects[i, [0, 2]] -= (w_pad / resize)
            detects[i, [1, 3]] -= (h_pad / resize)
            if detects[i, 0] < 0:
                detects[i, 0] = 0
            if detects[i, 1] < 0:
                detects[i, 1] = 0
            if detects[i, 2] >= origin_img_w - 1:
                detects[i, 2] = origin_img_w - 1
            if detects[i, 3] >= origin_img_h - 1:
                detects[i, 3] = origin_img_h - 1
                
        # nms
        best_box = []
        while np.shape(detects)[0] > 0:
            # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
            best_box.append(detects[0])
            if len(detects) == 1:
                break
            ious = self.iou(best_box[-1], detects[1:])
            detects = detects[1:][ious < self.nms_threshold]

        # print('------------non_max_suppression----------------')
        # print(best_box)

        return best_box


class HisensePreProcessV2(object):

    def __init__(self, **kwargs):
        super(HisensePreProcessV2, self).__init__()
        self.img_mean = kwargs['img_mean']
        self.img_std = kwargs['img_std']
        self.input_size = kwargs['input_size']
        self.swapRB = 0  # kwargs['swapRB']

        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img, color=(114, 114, 114)):
        h, w = self.input_size
        bbox_ratio_w = img.shape[1] / h
        bbox_ratio_h = img.shape[0] / w

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.swapRB == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.img_mean) / self.img_std
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        self.trans = dict(bbox_ratio_w=bbox_ratio_w, bbox_ratio_h=bbox_ratio_h)

        return img


class HisensePostProcessV2(object):

    def __init__(self, **kwargs):
        super(HisensePostProcessV2, self).__init__()
        self.prob_threshold = kwargs['prob_threshold']
        self.nms_threshold = kwargs['nms_threshold']

    def pred_bbox_reorg(self, pred_bbox, feature_index):
        '''
        :return:
        xy_gird: [feature_height,feature_width,1,2]
        pred_bbox: [feature_height,feature_width,num_anchor,4]
        pred_conf: [feature_height,feature_width,num_anchor,1]
        '''
        pred_bbox_temp = tf.reshape(
            pred_bbox, [-1, FLAG.feature_list[feature_index][1], FLAG.feature_list[feature_index][0], FLAG.num_anchor, 5])
        pred_xy, pred_wh, pred_conf = tf.split(pred_bbox_temp, [2, 2, 1], axis=-1)

        # 设定ratio和anchor的参数
        ratio = FLAG.input_width / FLAG.feature_list[feature_index][0]
        anchor = tf.constant(FLAG.anchor_list[feature_index])  # [num_anchor,2]

        # 生成用于xy计算的xy_offset
        x_grid_temp1 = tf.range(
            FLAG.feature_list[feature_index][0], dtype=tf.int32)
        y_grid_temp1 = tf.range(
            FLAG.feature_list[feature_index][1], dtype=tf.int32)

        x_grid_temp2, y_grid_temp2 = tf.meshgrid(x_grid_temp1, y_grid_temp1)
        x_grid_temp3 = tf.reshape(x_grid_temp2, (-1, 1))
        y_grid_temp3 = tf.reshape(y_grid_temp2, (-1, 1))
        xy_gird = tf.concat([x_grid_temp3, y_grid_temp3], axis=-1)
        xy_gird = tf.cast(tf.reshape(xy_gird, [FLAG.feature_list[feature_index][1],
                        FLAG.feature_list[feature_index][0], 1, 2]), tf.float32)  # [height,width,1,2]

        # 计算xywh的real尺度
        xy = (tf.nn.sigmoid(pred_xy) + xy_gird) * ratio
        # [height,width,num_anchor,2] * [num_anchor,2]
        wh = tf.exp(pred_wh) * anchor
        pred_bbox = tf.concat([xy, wh], axis=-1)
        return xy_gird, pred_bbox, pred_conf


    def bbox_processing(self, pred_bbox, feature_index):
        with tf.name_scope('BBOX_PROCESSING' + str(feature_index)):
            _, bbox_temp1, conf_temp1 = self.pred_bbox_reorg(pred_bbox, feature_index)
            num_bbox = FLAG.feature_list[feature_index][0] * \
                FLAG.feature_list[feature_index][1]*FLAG.num_anchor
            bbox_temp2 = tf.reshape(bbox_temp1, (num_bbox, 4))
            conf_temp2 = tf.nn.sigmoid(conf_temp1)
            conf = tf.reshape(conf_temp2, [num_bbox])

            x_center = tf.slice(bbox_temp2, [0, 0], [num_bbox, 1])
            y_center = tf.slice(bbox_temp2, [0, 1], [num_bbox, 1])
            width = tf.slice(bbox_temp2, [0, 2], [num_bbox, 1])
            height = tf.slice(bbox_temp2, [0, 3], [num_bbox, 1])

            # x_center = bbox_temp2[:,0:1]
            # y_center = bbox_temp2[:,1:2]
            # width = bbox_temp2[:,2:3]
            # height = bbox_temp2[:,3:4]
            x1 = x_center - 0.5 * width
            y1 = y_center - 0.5 * height
            x2 = x_center + 0.5 * width
            y2 = y_center + 0.5 * height
            bbox = tf.concat([y1, x1, y2, x2], axis=-1)
        return bbox, conf

    def bbox_select(self, pred_bbox_6, pred_bbox_12, nms_version):
        with tf.name_scope('BBOX_SELECT'):
            single_bbox_6, single_conf_6 = self.bbox_processing(pred_bbox_6, 0)
            single_bbox_12, single_conf_12 = self.bbox_processing(pred_bbox_12, 1)
            bbox = tf.concat([single_bbox_6, single_bbox_12], axis=0)
            conf = tf.concat([single_conf_6, single_conf_12], axis=0)

            if nms_version == 'Max' and FLAG.num_target == 1:
                select_index = tf.expand_dims(
                    tf.argmax(conf, output_type=tf.int32), axis=0)
            elif nms_version == 'V3':
                select_index = tf.image.non_max_suppression(
                    bbox, conf, FLAG.num_target, self.nms_threshold, score_threshold=self.prob_threshold)
            elif nms_version == 'V5':
                select_index, _ = tf.image.non_max_suppression_with_scores(
                    bbox, conf, FLAG.num_target, self.nms_threshold)
            else:
                raise ValueError(
                    'Select Correct Nms Version or Set Correct Number of Target')

            bbox_nms_temp = tf.gather(bbox, select_index)
            conf_nms_temp = tf.expand_dims(
                tf.gather(conf, select_index), axis=-1)

            bbox_nms = tf.concat([bbox_nms_temp, conf_nms_temp], axis=-1)
        return bbox_nms

    def res_reshape(self, real_output):
        real_output_names = [
            'ZSinp_Mnet_HEAD21/Conv2D_5:0',
            'ZSinp_Mnet_HEAD21/Conv2D_6:0',
            'ZSinp_Mnet_HEAD26/Conv2D_5:0',
            'ZSinp_Mnet_HEAD26/Conv2D_6:0',
        ]

        pred_bbox_6_box = real_output[real_output_names[0]].transpose(
            0, 2, 3, 1)  # real_output['output.0']  #real_output[0]
        pred_bbox_6_conf = real_output[real_output_names[1]].transpose(
            0, 2, 3, 1)  # real_output['output.1']  #real_output[2]

        pred_bbox_12_box = real_output[real_output_names[2]].transpose(
            0, 2, 3, 1)  # real_output['output.2']  #real_output[1]
        pred_bbox_12_conf = real_output[real_output_names[3]].transpose(
            0, 2, 3, 1)  # real_output['output.3']  #real_output[3]

        pred_bbox_6 = tf.concat([pred_bbox_6_box, pred_bbox_6_conf], axis=-1)
        pred_bbox_12 = tf.concat(
            [pred_bbox_12_box, pred_bbox_12_conf], axis=-1)

        return pred_bbox_6, pred_bbox_12

    def draw_image(self, origin_image, post_result, class_names, colors):
        img_show = copy.deepcopy(origin_image)
        for i in range(len(post_result)):
            bbox_x1_input = int(post_result[i][0])
            bbox_y1_input = int(post_result[i][1])
            bbox_x2_input = int(post_result[i][2])
            bbox_y2_input = int(post_result[i][3])
            cv2.rectangle(img_show, (bbox_x1_input, bbox_y1_input),
                          (bbox_x2_input, bbox_y2_input), colors[0], 5)

        return img_show

    def __call__(self, outputs, trans):
        bbox_ratio_w = trans['bbox_ratio_w']
        bbox_ratio_h = trans['bbox_ratio_h']
        """model post process"""
        pred_bbox_6, pred_bbox_12 = self.res_reshape(outputs)
        bbox_nms = self.bbox_select(pred_bbox_6, pred_bbox_12, 'V3')
        with tf.Session() as sess:
            bbox_nms_res = sess.run(bbox_nms)
            bbox_nms_y1x1y2x2 = bbox_nms_res[:, :4]
            bbox_nms_conf_temp1 = bbox_nms_res[:, 4:5]
            bbox_nms_conf = np.reshape(bbox_nms_conf_temp1, (-1))

            results = []
            for i in range(len(bbox_nms_res)):
                if bbox_nms_conf[i] >= self.prob_threshold:
                    bbox_y1_input = bbox_nms_y1x1y2x2[i][0]
                    bbox_x1_input = bbox_nms_y1x1y2x2[i][1]
                    bbox_y2_input = bbox_nms_y1x1y2x2[i][2]
                    bbox_x2_input = bbox_nms_y1x1y2x2[i][3]
                    bbox_x1_camera = int(bbox_x1_input * bbox_ratio_w)
                    bbox_y1_camera = int(bbox_y1_input * bbox_ratio_h)
                    bbox_x2_camera = int(bbox_x2_input * bbox_ratio_w)
                    bbox_y2_camera = int(bbox_y2_input * bbox_ratio_h)
                    conf = bbox_nms_conf[i]
                    res = np.array([bbox_x1_camera, bbox_y1_camera,
                            bbox_x2_camera, bbox_y2_camera, conf])
                    results.append(res)

        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default='/buffer2/zhangcc/trained_models/face-detection/hisense.onnx',
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
                        help='eval anno path')  # ["flickr_0", "flickr_2", "flickr_3"]
    parser.add_argument('--input_size', type=list, default=[192, 192])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--results_path',
                        type=str,
                        default='./work_dir/tmpfiles/hisense_resluts')
    parser.add_argument('--prob_threshold', type=float, default=0.9) ### 0.7
    parser.add_argument('--nms_threshold', type=float, default=0.3)  ### 0.4
    parser.add_argument('--draw_result', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cpu", help='There can be two options: cpu or cuda:3')
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export_version', type=int, default=1)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)  
    parser.add_argument('--log_level', type=int, default=30)    
    parser.add_argument('--error_analyzer', type=bool, default=True)         
    parser.add_argument('--is_calc_error', type=bool, default=True) #whether to calculate each layer error
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        eval_mode = 'dataset'
        acc_error = False
    else:
        eval_mode = 'dataset'
        acc_error = True
        
    normalization = [[120, 120, 120], [80.0, 80.0, 80.0]]
    kwargs_preprocess = {
        "img_mean": normalization[0],
        "img_std": normalization[1],
        'input_size': args.input_size,
    }
    kwargs_postprocess = {
        'prob_threshold': args.prob_threshold,
        'nms_threshold': args.nms_threshold,
    }

    preprocess = HisensePreProcessV2(**kwargs_preprocess)
    postprocess = HisensePostProcessV2(**kwargs_postprocess)

    process_args = {
        'log_name': 'process.log',
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
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,          
        'error_analyzer': args.error_analyzer,
        'error_metric': ['L1', 'L2', 'Cosine']
    }

    results_path = args.results_path
    save_imgs = os.path.join(results_path, 'images')
    if not os.path.exists(save_imgs):
        os.makedirs(save_imgs)

    kwargs_bboxeval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_hisense.log',  
        'log_level': args.log_level,
        # 'is_stdout': args.is_stdout,   
        'img_prefix': 'jpg',                 
        "iou_threshold": args.nms_threshold,
        'prob_threshold': args.prob_threshold,
        "process_args": process_args,
        'is_calc_error': args.is_calc_error,
        "draw_result": args.draw_result,
        "fp_result": args.fp_result,
        "eval_mode": eval_mode,  # single quantize, dataset quatize
        'acc_error': acc_error,
    }

    alfweval = AlfwEval(**kwargs_bboxeval)

    errors, max_errors, accuracy, tb = alfweval(
        quan_dataset_path=args.quan_dataset_path, 
        dataset_path=args.dataset_path,
        ann_path=args.ann_path, 
        event_lst=args.event_lst, 
        save_dir=args.results_path)
    if args.error_analyzer:
        alfweval.error_analysis() 
    if args.export:
        alfweval.export()