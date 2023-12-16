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
from .det_metric import build_evaluator

try:
    from utils import Object
    from tools import ModelProcess
except:
    from onnx_converter.utils import Object  # type: ignore
    from onnx_converter.tools import ModelProcess  # type: ignore


class CocoEval(Eval):  # type: ignore
    def __init__(self, **kwargs):
        super(CocoEval, self).__init__(**kwargs)

        self.process_args = kwargs["process_args"]
        self.is_calc_error = kwargs["is_calc_error"]
        self.draw_result = False
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

        self.db_evaluator_quant = build_evaluator(dict(name="DetMetric", main_indicator="hmean"))
        if self.fp_result:
            self.db_evaluator_float = build_evaluator(dict(name="DetMetric", main_indicator="hmean"))

    def set_draw_result(self, is_draw):
        self.draw_result = is_draw

    def get_metric(self, result, img_name, image, target_h, target_w, r):
        if result.shape[0]:
            dt_boxes = result[0][5:].reshape((4, 2))
            predict_lst = [{"points": np.expand_dims(dt_boxes, 0)}]

            # get GT info
            str_list = img_name[:-4].split('-')
            keypoint_str_list = str_list[3].split('_')
            new_keypoint_str_list = [keypoint_str_list[2], keypoint_str_list[3], keypoint_str_list[0], keypoint_str_list[1]]
            keypoint_list = []
            for new_keypoint_str in new_keypoint_str_list:
                keypoint_str = new_keypoint_str.split("&")
                keypoint_list.append([int(keypoint_str[0]), int(keypoint_str[1])])
            keypoint_arr = np.asarray([[keypoint_list]])
            str_list = img_name.split('-')

            batch = [
                image,
                np.array([[target_h, target_w, r, r]]),
                keypoint_arr,
                np.array([[False]])
            ]
        else:
            predict_lst, batch = [], []
            
        return predict_lst, batch


    def __call__(
        self,
        quan_dataset_path,
        dataset_path,
        ann_path,
        input_size,
        normalization,
        save_eval_path,
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
            self.process.quantize(fd_path=quan_dataset_path, is_dataset=is_dataset)

        json_fr = open(ann_path, "r")
        content = json_fr.read()
        json_data = json.loads(content)

        for i, image_info in enumerate(
            tqdm.tqdm(json_data["images"], postfix="image files")
        ):
            image_name = image_info["file_name"]
            img = cv2.imread(os.path.join(dataset_path, image_name))
            img_clone = copy.deepcopy(img)
            # os.system("cp {}/{} work_dir/".format(dataset_path, image_name))
            
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
            trans = self.process.transform.get_trans()
            target_h, target_w = trans["input_size"]    
            ratio = trans["ratio"]      
            for key in outputs.keys():
                result = outputs[key]
                predict_lst, batch = self.get_metric(result, image_name, img_clone, target_h, target_w, ratio)
                if len(predict_lst) > 0:
                    if key == "true_out":
                        self.db_evaluator_float(preds=predict_lst, batch=batch)
                    else:
                        self.db_evaluator_quant(preds=predict_lst, batch=batch)
          
            if self.draw_result:
                if "postprocess" in self.process_args.keys() and hasattr(
                    self.process_args["postprocess"], "draw_image"
                ):
                    draw_image = self.process_args["postprocess"].draw_image
                else:
                    draw_image = self.draw_image
                qres = copy.deepcopy(img_clone)
                # qres = draw_image(outputs["quant_out"], qres, image_name)
                if self.fp_result:
                    qres = draw_image(outputs["true_out"], qres, image_name)
                cv2.imwrite(
                    os.path.join(save_imgs, os.path.basename(image_name)), qres
                )  ### results for draw bbox

            # if 100-1 == i:
            #     break

        if self.fp_result:
            metric_float = self.db_evaluator_float.get_metric()
            metric_quant = self.db_evaluator_quant.get_metric()
            accuracy = dict(
                faccuracy=metric_float,
                qaccuracy=metric_quant,
            )            
        else:
            metric_quant = self.db_evaluator_quant.get_metric()
            accuracy = dict(
                qaccuracy=metric_quant,
            )
        tb = self.get_results(accuracy)

        return accuracy, tb
