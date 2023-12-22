# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/1/24 9:49
# @File     : test_gesture_recognition.py
import sys  # NOQA: E402

sys.path.append("./")  # NOQA: E402

import argparse
from pybaseutils import image_utils
from pybaseutils.pose import bones_utils

import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn

from eval.hand_keypoint import CocoEval


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(a=src[0, :], b=src[1, :])
    dst[2:, :] = get_3rd_point(a=dst[0, :], b=dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """

    assert isinstance(
        batch_heatmaps, np.ndarray
    ), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask

    return preds, maxvals


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_predicts(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)

    return target_coords


def get_final_predicts(batch_heatmaps, down_scale):
    preds, maxvals = get_max_preds(batch_heatmaps=batch_heatmaps)
    return preds * down_scale, maxvals


class HandKeyPointPreProcess(object):
    def __init__(self, **kwargs):
        super(HandKeyPointPreProcess, self).__init__()
        self.img_mean = [0.5, 0.5, 0.5]
        self.img_std = [0.5, 0.5, 0.5]
        if "img_mean" in kwargs.keys():
            self.img_mean = kwargs["img_mean"]
        if "img_std" in kwargs.keys():
            self.img_std = kwargs["img_std"]
        self.trans = 0
        self.input_size = kwargs["input_size"]

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img):
        img_input = img.astype(np.float32) / 255
        img_mean = np.array(self.img_mean, dtype=np.float32)
        img_std = np.array(self.img_std, dtype=np.float32)
        img_input = (img_input - img_mean) / img_std
        img_input = np.transpose(img_input, [2, 0, 1])
        img_input = np.expand_dims(img_input, axis=0)

        self.set_trans(dict(input_size=self.input_size))

        return img_input


class HandKeyPointPostProcess(object):
    def __init__(self, **kwargs):
        super(HandKeyPointPostProcess, self).__init__()
        self.prob_threshold = kwargs["prob_threshold"]
        self.target_bones = bones_utils.get_target_bones(target="hand")
        self.skeleton = self.target_bones["skeleton"]
        self.colors = self.target_bones["colors"]

    def draw_image(self, image, points):
        """
        :param image:
        :param boxes: 检测框
        :param points: 关键点
        """

        image = image_utils.draw_key_point_in_image(
            image,
            points,
            pointline=self.skeleton,
            boxes=[],
            thickness=1,
            colors=self.colors,
        )

        return image

    def __call__(self, outputs, trans):
        input_size = trans["input_size"]
        predicts = outputs["output"]
        down_scale = input_size[0] // predicts.shape[2]
        
        kp_point, kp_score = get_final_predicts(batch_heatmaps=predicts, down_scale=down_scale)
        kp_point, kp_score = kp_point[0, :], kp_score[0, :]

        index = kp_score < self.prob_threshold
        index = index.reshape(-1)
        kp_point[index, :] = (0, 0)
        kp_point = np.abs(kp_point)

        results = np.concatenate([kp_point, kp_score], axis=-1)

        return dict(heatmap=predicts, results=results, down_scale=down_scale)


pre_post_instances = {
    "handpose": ["HandKeyPointPreProcess", "HandKeyPointPostProcess"],
}

normalizations = {
    "handpose": [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        # default="Keypoint-MobileNetV2/onnx_weights/mobilenetv2_192_192_simplify.onnx",
        # default="Keypoint-MobileNetV2/onnx_weights/mobilenetv2_upsample_192_192_simplify.onnx",
        # default="Keypoint-MobileNetV2/onnx_weights/mobilenetv2_upsample_no_bn_192_192_simplify.onnx",
        default="Keypoint-LiteHRNet18/onnx_weights/litehrnet18_192_192_simplify.onnx",
        # default="work_dir/mobilenetv2_upsample_no_bn_192_192_simplify_cross_layer_equalization.onnx",
        # default="work_dir/mobilenetv2_upsample_no_bn_192_192_simplify_bias_correction.onnx",
        # default="work_dir/qat/mobilenetv2_upsample_no_bn_192_192_simplify_qat_60.onnx",
        help="checkpoint path",
    )
    parser.add_argument(
        "--quan_dataset_path",
        type=str,
        # default="/buffer/hand_keypoint/crop_img/test/HandPose-v1",
        default="/buffer/hand_keypoint/crop_img/calibrate_img",
        help="quantize image path",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        default="/buffer/hand_keypoint/crop_img/test/HandPose-v1_test.json",
        help="eval anno path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/buffer/hand_keypoint/crop_img/test/HandPose-v1",
        help="eval image path",
    )
    parser.add_argument("--model_name", type=str, default="handpose")
    parser.add_argument("--input_size", type=list, default=[192, 192])
    parser.add_argument(
        "--results_path",
        type=str,
        default="./work_dir/tmpfiles/hand_keypoint",
    )
    parser.add_argument("--prob_threshold", type=float, default=0.3)  #
    parser.add_argument("--draw_result", type=bool, default=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="There can be two options: cpu or cuda:3",
    )
    parser.add_argument("--fp_result", type=bool, default=True)
    parser.add_argument("--export_version", type=int, default=3)
    parser.add_argument("--export", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--is_stdout", type=bool, default=True)
    parser.add_argument("--log_level", type=int, default=30)
    parser.add_argument("--error_analyzer", type=bool, default=False)
    parser.add_argument(
        "--is_calc_error", type=bool, default=False
    )  # whether to calculate each layer error
    parser.add_argument(
        "--calibration_params_json_path", 
        type=str, 
        # default="work_dir/qat/calibration_99.json",
        default=None,
    )    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        eval_mode = "single"
        acc_error = False
    else:
        eval_mode = "dataset"
        acc_error = True

    normalization = normalizations[args.model_name]
    kwargs_preprocess = {
        "img_mean": normalization[0],
        "img_std": normalization[1],
        "input_size": args.input_size,
    }
    kwargs_postprocess = {
        "prob_threshold": args.prob_threshold,
    }

    preprocess = eval(pre_post_instances[args.model_name][0])(**kwargs_preprocess)
    postprocess = eval(pre_post_instances[args.model_name][1])(**kwargs_postprocess)

    process_args = {
        "log_name": "process.log",
        "log_level": args.log_level,
        "model_path": args.model_path,
        "parse_cfg": "config/parse.py",
        "graph_cfg": "config/graph.py",
        "quan_cfg": "config/quantize.py",
        "analysis_cfg": "config/analysis.py",
        "export_cfg": "config/export_v{}.py".format(args.export_version),
        "offline_quan_mode": None,
        "offline_quan_tool": None,
        "quan_table_path": None,
        "simulation_level": 1,
        "transform": preprocess,
        "postprocess": postprocess,
        "device": args.device,
        "is_ema": True,
        "fp_result": args.fp_result,
        "ema_value": 0.99,
        "is_array": False,
        "is_stdout": args.is_stdout,
        "error_analyzer": args.error_analyzer,
        "error_metric": ["L1", "L2", "Cosine"],
    }

    kwargs_bboxeval = {
        "log_dir": "work_dir/eval",
        "log_name": "test_hand_keypoint_detection.log",
        "log_level": args.log_level,
        # 'is_stdout': args.is_stdout,
        "img_prefix": "jpg",
        "save_key": "mAP",
        "draw_result": args.draw_result,
        "process_args": process_args,
        "is_calc_error": args.is_calc_error,
        "fp_result": args.fp_result,
        "eval_mode": eval_mode,  # single quantize, dataset quatize
        "acc_error": acc_error,
    }

    cocoeval = CocoEval(**kwargs_bboxeval)
    cocoeval.set_draw_result(is_draw=args.draw_result)
    # quan_dataset_path, images_path, ann_path, input_size, normalization, save_eval_path
    evaluation, tb = cocoeval(
        quan_dataset_path=args.quan_dataset_path,
        dataset_path=args.dataset_path,
        ann_path=args.ann_path,
        input_size=args.input_size,
        normalization=normalization,
        save_eval_path=args.results_path,
        calibration_params_json_path=args.calibration_params_json_path,
    )
    if args.is_calc_error:
        cocoeval.collect_error_info()
    if args.error_analyzer:
        cocoeval.error_analysis()
    if args.export:
        cocoeval.export()
    print(tb)
