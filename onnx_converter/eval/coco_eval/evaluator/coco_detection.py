# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import warnings

from pycocotools.cocoeval import COCOeval


def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


class CocoDetectionEvaluator:
    def __init__(self, dataset):
        assert hasattr(dataset, "coco_api")
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ["mAP", "AP_50", "AP_75", "AP_small", "AP_m", "AP_l"]
        self.exist_kps = False

    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, dets_ in dets.items():
                category_id = self.cat_ids[label]
                for det in dets_:
                    bbox = det[:4]
                    score = float(det[4])

                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score,
                    )
                    # Check if keypoints exist and add them to detection
                    if len(det) > 5:
                        self.exist_kps = True
                        keypoint = det[5:]
                        detection["keypoints"] = keypoint

                    json_results.append(detection)
        return json_results

    def pr_curve(self, coco_eval, curve_name):
        precisions = coco_eval.eval['precision']
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = precisions
        import numpy as np
        x = np.arange(0.0, 1.01, 0.01)
        import matplotlib.pyplot as plt
        plt.plot(x, p1[:, 0, 0, 2], label="iou=0.50")
        plt.plot(x, p2[:, 0, 0, 2], label="iou=0.55")
        plt.plot(x, p3[:, 0, 0, 2], label="iou=0.60")
        plt.plot(x, p4[:, 0, 0, 2], label="iou=0.65")
        plt.plot(x, p5[:, 0, 0, 2], label="iou=0.70")
        plt.plot(x, p6[:, 0, 0, 2], label="iou=0.75")
        plt.plot(x, p7[:, 0, 0, 2], label="iou=0.80")
        plt.plot(x, p8[:, 0, 0, 2], label="iou=0.85")
        plt.plot(x, p9[:, 0, 0, 2], label="iou=0.90")
        plt.plot(x, p10[:, 0, 0, 2], label="iou=0.95")
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.savefig('./work_dir/{}'.format(curve_name))

    def evaluate(self, results, save_dir, rank=-1):
        results_json = self.results2json(results)
        if len(results_json) == 0:
            warnings.warn(
                "Detection result is empty! Please check whether "
                "training set is too small (need to increase val_interval "
                "in config and train more epochs). Or check annotation "
                "correctness."
            )
            empty_eval_results = {}
            for key in self.metric_names:
                empty_eval_results[key] = 0
            return empty_eval_results
        json_path = os.path.join(save_dir, "results{}.json".format(rank))
        json.dump(results_json, open(json_path, "w"))
        coco_dets = self.coco_api.loadRes(json_path)
        iouType = "keypoints" if self.exist_kps else "bbox"
        coco_eval = COCOeval(
            cocoGt=copy.deepcopy(self.coco_api), cocoDt=copy.deepcopy(coco_dets), iouType=iouType
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        aps = coco_eval.stats[:6]
        eval_results = {}
        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
        return coco_eval, eval_results
