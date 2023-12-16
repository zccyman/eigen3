# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/2/8 17:27
# @File     : cocoeval.py
import copy
import os
import tempfile
import cv2
import glob
import numpy as np
import tqdm
import json

from .evaluator import build_evaluator
from .data.dataset import build_dataset
from eval import Eval
try:
    from utils import Object
    from tools import ModelProcess    
except:
    from onnx_converter.utils import Object # type: ignore
    from onnx_converter.tools import ModelProcess # type: ignore
    

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


def overlay_bbox_cv(img, all_box, class_names, colors):
    """Draw result boxes
    Copy from nanodet/util/visualization.py
    """
    # all_box array of [label, x0, y0, x1, y1, score]
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        if isinstance(x0, np.float):
            x0 = np.round(x0).astype(np.int32)
        if isinstance(x1, np.float):    
            x1 = np.round(x1).astype(np.int32)
        if isinstance(y0, np.float):
            y0 = np.round(y0).astype(np.int32)
        if isinstance(y1, np.float):
            y1 = np.round(y1).astype(np.int32)
        # color = self.cmap(i)[:3]
        color = (colors[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(colors[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img


def compute(results, dataset, save_dir, name):

    eval_cfg = dict(name="CocoDetectionEvaluator", save_key=save_dir)

    evaluator = build_evaluator(eval_cfg, dataset)
    tmp_dir = tempfile.TemporaryDirectory()
    coco_eval, eval_results = evaluator.evaluate(
        results=results, save_dir=tmp_dir.name, rank=-1
    )
    if not evaluator.exist_kps:
        evaluator.pr_curve(coco_eval, curve_name=name)

    return eval_results


class CocoEval(Eval): # type: ignore
    def __init__(self, **kwargs):
        super(CocoEval, self).__init__(**kwargs)

        # self.iou_threshold = kwargs['iou_threshold'] #0.3
        self.process_args = kwargs['process_args']
        self.is_calc_error = kwargs['is_calc_error']
        self.class_names = class_names
        self._COLORS = None
        self.draw_result = False
        self.save_key = kwargs['save_key']
        self.eval_mode = kwargs['eval_mode']
        self.fp_result = kwargs['fp_result']
        self.acc_error = kwargs['acc_error']
        # self.is_stdout = self.process_args['is_stdout']
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)
        
        # self.process_args_11 = copy.deepcopy(self.process_args)
        # self.process_args_11['quan_cfg'] = './config/voice_quantize_fp.py'
        # self.process_args_11.update(dict(log_name="p11.log"))
        # self.process_11 = MyModelProcessBiasCorrection(**self.process_args_11) ###MyModelProcessEasyQuant
        self.process = ModelProcess(**self.process_args)
        if self.is_calc_error:
            self.process.set_onnx_graph(False)
            # self.process.onnx_graph = False
        
    def set_iou_threshold(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def set_class_names(self, names):
        self.class_names = names

    def set_colors(self, colors):
        self._COLORS = colors

    def set_draw_result(self, is_draw):
        self.draw_result = is_draw
                
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

    def __call__(self, quan_dataset_path, dataset_path, ann_path, input_size, normalization, save_eval_path, calibration_params_json_path=None):
        is_dataset = False
        if os.path.isfile(quan_dataset_path):
            is_dataset = False
        elif os.path.isdir(quan_dataset_path):
            is_dataset = True
        else:
            print('invaild input quan file!')
            os._exit(-1)

        # self.eval_mode = "single"
        if self.eval_mode == "dataset":
            fd_path = "model_zoo/eval_yolox_nano_clipped/{}".format("resume")
            self.process.quantize(fd_path=quan_dataset_path, is_dataset=is_dataset, 
                                  saved_calib_name=fd_path, 
                                  calibration_params_json_path=calibration_params_json_path,
                                )
            # if not self.reload_calibration(fd_path):
            #     self.process.quantize(fd_path=quan_dataset_path, is_dataset=is_dataset, saved_calib_name=fd_path)
            #     self.save_calibration(fd_path)
        
        save_imgs = os.path.join(save_eval_path, 'images')
        if not os.path.exists(save_imgs):
            os.makedirs(save_imgs)

        qresults, tresults = dict(), dict()
        
        cfg = dict(
            name="CocoDataset",
            img_path=dataset_path,
            ann_path=ann_path,
            input_size=input_size,  # [w,h]
            # multi_scale=[1.5, 1.5],
            keep_ratio=True,
            # use_instance_mask=False,
            pipeline=dict(normalize=normalization),
        )
        dataset = build_dataset(cfg, "val")
        images = dataset.get_data_info(ann_path)

        float_results, quant_results = [], []
        images_ = tqdm.tqdm(images, postfix='image files') if self.is_stdout else images
        for image_id, item in enumerate(images_):
            idx, image_name = item['id'], item['file_name']
            # if image_id < 2000:
            #         continue
            
            img = cv2.imread(os.path.join(dataset_path, image_name))
            if self.eval_mode == "single":
                self.process.quantize(fd_path=os.path.join(dataset_path, image_name), is_dataset=False)

            if self.fp_result:
                true_outputs = self.process.post_quan.onnx_infer(img)
            else:
                true_outputs = None
            
            # self.acc_error = False 
            # self.is_calc_error = True   
            if self.is_calc_error:
                self.process.checkerror(img, acc_error=self.acc_error)
            else:
                self.process.dataflow(img, acc_error=self.acc_error, onnx_outputs=true_outputs)
            # if 'analysis_cfg' in self.process_args.keys() and self.fp_result:
            #     self.process.checkerror_weight(onnx_outputs=None)
            #     self.process.checkerror_feature(onnx_outputs=true_outputs)   
                                     
            outputs = self.process.get_outputs()
            q_out, t_out = outputs['qout'], outputs['true_out']

            qmydict, tmydict = dict(), dict()
            if isinstance(q_out, dict):
                for bbox, label, score in zip(q_out["bbox"], q_out["label"], q_out["score"]):
                    if label in qmydict.keys():
                        value = list(np.append(bbox, score))
                        qmydict[label].append(value)
                    else:
                        dets = []
                        dets.append(list(np.append(bbox, score)))
                        qmydict[label] = dets

                    quant_dict = dict()
                    quant_dict['image_id'] = idx
                    quant_dict['category_id'] = 1
                    bbox_ = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    quant_dict['bbox'] = [np.float64(p) for p in bbox_]
                    quant_dict['score'] = np.float64(score)
                    quant_results.append(quant_dict)

                if "keypoints" in q_out.keys():
                    for object_id, (label, keypoints) in enumerate(zip(q_out["label"], q_out["keypoints"])):
                        qmydict[label][object_id].extend(keypoints)
                        quant_results[object_id]['keypoints'] = [np.float64(p) for p in keypoints]
                                                
            if isinstance(t_out, dict) and self.fp_result:
                for bbox, label, score in zip(t_out["bbox"], t_out["label"], t_out["score"]):
                    if label in tmydict.keys():
                        value = list(np.append(bbox, score))
                        tmydict[label].append(value)
                    else:
                        dets = []
                        dets.append(list(np.append(bbox, score)))
                        tmydict[label] = dets

                    float_dict = dict()
                    float_dict['image_id'] = idx
                    float_dict['category_id'] = 1
                    bbox_ = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    float_dict['bbox'] = [np.float64(p) for p in bbox_]
                    float_dict['score'] = np.float64(score)
                    float_results.append(float_dict)

                if "keypoints" in t_out.keys():
                    for object_id, (label, keypoints) in enumerate(zip(t_out["label"], t_out["keypoints"])):
                        tmydict[label][object_id].extend(keypoints)
                        float_results[object_id]['keypoints'] = [np.float64(p) for p in keypoints]

            qresults.update({idx: qmydict})
            tresults.update({idx: tmydict})

            if self.draw_result and outputs is not None:
                if 'postprocess' in self.process_args.keys() and hasattr(self.process_args["postprocess"], 'draw_image'):
                    draw_image = self.process_args["postprocess"].draw_image
                else:
                    draw_image = self.draw_image    
                if self._COLORS is not None:               
                    qres = draw_image(copy.deepcopy(img), q_out, class_names=self.class_names, colors=self._COLORS) #np.array([[1, 0, 0]])
                else:
                    qres = draw_image(copy.deepcopy(img), q_out, class_names=self.class_names, colors=np.array([[1, 0, 0] for _ in self.class_names]))
                if self.fp_result:
                    if self._COLORS is not None:
                        qres = draw_image(qres, t_out, class_names=self.class_names, colors=self._COLORS) #np.array([[0, 0, 1]])
                    else:
                        qres = draw_image(qres, t_out, class_names=self.class_names, colors=np.array([[0, 0, 1] for _ in self.class_names]))
                cv2.imwrite(os.path.join(save_imgs, os.path.basename(image_name)), qres)  ### results for draw bbox

            # if 0 == image_id and self.eval_first_frame: break
            # if 5-1 == image_id: break
        # self.error_analysis() 
        # results, img_path, ann_path, input_size, normalize, save_dir
        model_name = os.path.basename(self.process_args['model_path']).split('.')[0]
        qevaluation = compute(qresults, dataset, save_eval_path, '{}_{}'.format(model_name, 'quant'))
        if self.fp_result:
            tevaluation = compute(tresults, dataset, save_eval_path, '{}_{}'.format(model_name, 'fp32'))
            accuracy = dict(qaccuracy=qevaluation, faccuracy=tevaluation)
            # with open("float_results.json", "w") as f:
            #     f.write(json.dumps(float_results))
            # with open("quant_results.json", "w") as f:
            #     f.write(json.dumps(quant_results))
        else:
            accuracy = dict(qaccuracy=qevaluation)
            # with open("quant_results.json", "w") as f:
            #     f.write(json.dumps(quant_results))             

        tb = self.get_results(accuracy)
        if not self.is_calc_error and self.process.onnx_graph:
            img = cv2.imread(os.path.join(dataset_path, images[image_id]['file_name']))
            os.system("cp {}/{} work_dir/".format(dataset_path, images[image_id]['file_name']))
            true_outputs = self.process.post_quan.onnx_infer(img)
            self.process.numpygraph(img, acc_error=True, onnx_outputs=true_outputs)
        else:
            os.system("cp {}/{} work_dir/".format(dataset_path, images[image_id]['file_name']))
                            
        return accuracy, tb
