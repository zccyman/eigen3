# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/2/8 17:27
# @File     : cocoeval.py
import copy
import os
import cv2
import numpy as np
import tqdm
import json

from eval import Eval

try:
    from utils import Object
    from tools import ModelProcess
except:
    from onnx_converter.utils import Object  # type: ignore
    from onnx_converter.tools import ModelProcess  # type: ignore


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class GenerateTarget:
    def __init__(self, image_size, sigma, num_joints):
        self.image_size = np.array(image_size)
        self.sigma = sigma
        self.num_joints = num_joints

    def __call__(self, joints, joints_vis, down_scale):
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        self.heatmap_size = self.image_size // down_scale

        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32,
        )
        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (
                ul[0] >= self.heatmap_size[0]
                or ul[1] >= self.heatmap_size[1]
                or br[0] < 0
                or br[1] < 0
            ):
                # or br[0] < 0 or br[1] < 0 or ul[0] < 0 or ul[1] < 0:
                # fix a bug: ul<0
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[
                    g_y[0] : g_y[1], g_x[0] : g_x[1]
                ]

        return target, target_weight


def calc_dists(predicts, target, normalize):
    target = target.astype(np.float32)
    predicts = predicts.astype(np.float32)
    dists = np.zeros((predicts.shape[1], predicts.shape[0]))

    for n in range(predicts.shape[0]):
        for c in range(predicts.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_predicts = predicts[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(x=normed_predicts - normed_targets)
                # print(c, n, dists[c, n])
                # print("test")
            else:
                dists[c, n] = -1

    return dists


def dist_acc(dists, thr=0.5):
    """Return percentage below threshold while ignoring values with a -1"""

    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


class CocoEval(Eval):  # type: ignore
    def __init__(self, **kwargs):
        super(CocoEval, self).__init__(**kwargs)

        self.process_args = kwargs["process_args"]
        self.is_calc_error = kwargs["is_calc_error"]
        self.draw_result = False
        self.save_key = kwargs["save_key"]
        self.eval_mode = kwargs["eval_mode"]
        self.fp_result = kwargs["fp_result"]
        self.acc_error = kwargs["acc_error"]
        # self.is_stdout = self.process_args['is_stdout']
        self.logger = self.get_log(
            log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout
        )

        self.process = ModelProcess(**self.process_args)
        if self.is_calc_error:
            self.process.set_onnx_graph(False)

        self.num_joints = 21
        self.generate_target = GenerateTarget(
            image_size=np.array(self.process_args["transform"].input_size),
            sigma=2,
            num_joints=self.num_joints,
        )

    def set_draw_result(self, is_draw):
        self.draw_result = is_draw

    def get_all_predicts_boxes(self, acc, predicts, target, input_size):
        # calculate accuracy
        predicts = predicts[None, :]
        target = target[None, :]
        _, avg_acc, cnt, pred = self.accuracy(
            predict=predicts, target=target, input_size=input_size
        )
        acc.update(avg_acc, cnt)

        return acc

    def accuracy(self, predict, target, input_size):
        """
        Calculate accuracy according to PCK0.5,
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        :param output:
        :param target:
        :param hm_type:
        :param thr:
        :return:
        """

        idx = list(range(predict.shape[1]))
        h, w = input_size
        norm = np.ones((predict.shape[0], 2)) * np.array([h, w]) / 10

        dists = calc_dists(predicts=predict, target=target, normalize=norm)

        acc = np.zeros((len(idx) + 1))
        avg_acc = 0
        cnt = 0

        for i in range(len(idx)):
            acc[i + 1] = dist_acc(dists[idx[i]])  # th=0.5
            if acc[i + 1] >= 0:
                avg_acc = avg_acc + acc[i + 1]
                cnt += 1

        avg_acc = avg_acc / cnt if cnt != 0 else 0
        if cnt != 0:
            acc[0] = avg_acc

        return acc, avg_acc, cnt, predict

    def __call__(
        self,
        quan_dataset_path,
        dataset_path,
        ann_path,
        input_size,
        normalization,
        save_eval_path,
        calibration_params_json_path,
    ):
        is_dataset = False
        if os.path.isfile(quan_dataset_path):
            is_dataset = False
        elif os.path.isdir(quan_dataset_path):
            is_dataset = True
        else:
            print("invaild input quan file!")
            os._exit(-1)

        save_imgs = os.path.join(save_eval_path, "images")
        if not os.path.exists(save_imgs):
            os.makedirs(save_imgs)

        # self.eval_mode = "single"
        if self.eval_mode == "dataset":
            self.process.quantize(fd_path=quan_dataset_path, is_dataset=is_dataset, calibration_params_json_path=calibration_params_json_path)

        json_fr = open(ann_path, "r")
        content = json_fr.read()
        json_data = json.loads(content)

        acc_quant = AverageMeter()
        acc_float = AverageMeter()

        for i, image_name in enumerate(
            tqdm.tqdm(json_data.keys(), postfix="image files")
        ):
            # if i < 2500:
            #     continue
            img = cv2.imread(os.path.join(dataset_path, image_name))
            img_clone = copy.deepcopy(img)

            if self.eval_mode == "single":
                self.process.quantize(fd_path=img, is_dataset=False)

            if self.fp_result:
                true_outputs = self.process.post_quan.onnx_infer(img)
            else:
                true_outputs = None

            # self.acc_error = False
            # self.is_calc_error = True
            if self.is_calc_error:
                self.process.checkerror(img, acc_error=self.acc_error)
            else:
                self.process.dataflow(
                    img, acc_error=self.acc_error, onnx_outputs=true_outputs
                )

            outputs = self.process.get_outputs()
            predicts_quant, predicts_float = (
                outputs["qout"]["heatmap"],
                outputs["true_out"]["heatmap"],
            )
            results_quant, results_float = (
                outputs["qout"]["results"],
                outputs["true_out"]["results"],
            )
            results_float_points = [results_float[:, :-1]]
            results_quant_points = [results_quant[:, :-1]]

            # generate target
            img_info_dct = json_data[image_name]
            joints_3d, joints_3d_vis = (
                img_info_dct["joints"],
                img_info_dct["joints_vis"],
            )
            joints_3d_array = np.array(joints_3d, dtype=np.float32).reshape(
                self.num_joints, 3
            )
            joints_3d_vis_array = np.array(joints_3d_vis, dtype=np.float32).reshape(
                self.num_joints, 3
            )
            target, target_weight = self.generate_target(
                joints=joints_3d_array,
                joints_vis=joints_3d_vis_array,
                down_scale=outputs["qout"]["down_scale"],
            )
            target = np.expand_dims(target, 0)
            trans = self.process.transform.get_trans()
            input_size = trans["input_size"]
            outputs_gt = self.process.postprocess(
                dict(output=target),
                trans=trans,
            )
            predicts_gt = outputs_gt["heatmap"]
            results_gt = outputs_gt["results"][:, :-1]

            acc_float = self.get_all_predicts_boxes(
                acc_float,
                results_float_points[0],
                results_gt,
                input_size=input_size,
            )
            acc_quant = self.get_all_predicts_boxes(
                acc_quant,
                results_quant_points[0],
                results_gt,
                input_size=input_size,
            )

            if self.draw_result and outputs is not None:
                if "postprocess" in self.process_args.keys() and hasattr(
                    self.process_args["postprocess"], "draw_image"
                ):
                    draw_image = self.process_args["postprocess"].draw_image
                else:
                    draw_image = self.draw_image
                qres = copy.deepcopy(img_clone)
                # qres = draw_image(qres, results_quant_points)
                if self.fp_result:
                    qres = draw_image(qres, results_float_points)
                cv2.imwrite(
                    os.path.join(save_imgs, os.path.basename(image_name)), qres
                )  ### results for draw bbox

            # if 0 == i:
            #     break

        accuracy = dict(
            faccuracy={"acc.avg": acc_float.avg},
            qaccuracy={"acc.avg": acc_quant.avg},
        )
        tb = self.get_results(accuracy)

        return accuracy, tb
