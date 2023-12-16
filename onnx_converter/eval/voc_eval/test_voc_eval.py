import sys

import cv2
import tqdm

import os
import numpy as np
import glob

import xml.etree.ElementTree as ET

from eval.voc_eval.voc_map import compute_precision_recall
from eval.voc_eval.voc_map import compute_tp_fp
from eval.voc_eval.voc_map import voc_ap2, voc_ap
from eval.voc_eval.misc import pretreat
from eval.voc_eval.misc import parse_ground_truth
from eval.voc_eval.misc import parse_detection_results
try:
    from tools import ModelProcess
except:
    from onnx_converter.tools import ModelProcess # type: ignore


class XML2TXT(object):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

        self.xml_path = kwargs["xml_path"]
        self.groundtruth = kwargs["groundtruth"]
        self.detection_results = kwargs["detection_results"]
        self.is_fakedata = kwargs["is_fakedata"]

        self.xml_files = glob.glob(os.path.join(self.xml_path, "*.xml"))

    def load_annotations(self):
        img_infos = []
        for xml_file in self.xml_files:
            img_id = os.path.basename(xml_file).split(".xml")[0]
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text) # type: ignore
            height = int(size.find('height').text) # type: ignore
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return img_infos

    def get_ann_info(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text # type: ignore
            label = self.cat2label[name] # type: ignore
            difficult = int(obj.find('difficult').text) # type: ignore
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text), # type: ignore
                int(bnd_box.find('ymin').text), # type: ignore
                int(bnd_box.find('xmax').text), # type: ignore
                int(bnd_box.find('ymax').text)  # type: ignore
            ]
            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(bboxes=bboxes.astype(np.float32),
                   labels=labels.astype(np.int64),
                   bboxes_ignore=bboxes_ignore.astype(np.float32),
                   labels_ignore=labels_ignore.astype(np.int64))

        return ann

    def anno2txt(self, anno, txt_file, has_score=False):
        with open(txt_file, "a") as file:
            for bbox, label in zip(anno["bboxes"], anno["labels"]):
                file.write(self.CLASSES[label - 1] + " ")
                if has_score:
                    file.write("1.0" + " ")

                for idx, box in enumerate(bbox):
                    box = str(int(box))
                    if idx == len(bbox) - 1:
                        file.write(box + "\n")
                    else:
                        file.write(box + " ")

    def __call__(self):
        for xml_file in self.xml_files:
            anno = self.get_ann_info(xml_file)
            txt_file = os.path.join(
                self.groundtruth,
                os.path.basename(xml_file).replace(".xml", ".txt"))
            self.anno2txt(anno=anno, txt_file=txt_file)

            ###fake data
            if self.is_fakedata:
                txt_file = os.path.join(
                    self.detection_results,
                    os.path.basename(xml_file).replace(".xml", ".txt"))
                self.anno2txt(anno=anno, txt_file=txt_file, has_score=True)


class VocEval(object):
    def __init__(self, **kwargs):
        self.groundtruth = kwargs["xml2txt"]["groundtruth"]
        self.detection_results = kwargs["detection_results"]
        self.json_results = kwargs["json_results"]
        for dir in [
                self.groundtruth, self.detection_results, self.json_results
        ]:
            os.system("rm -rf {}".format(dir))
            os.makedirs(dir, exist_ok=True)

        if kwargs["xml2txt"]["xml_path"]:
            self.xml2txt = XML2TXT(**kwargs["xml2txt"])()

    def voc_eval(self):
        pretreat(self.groundtruth, self.detection_results, self.json_results)

        # 将.txt文件解析成json格式
        gt_per_classes_dict = parse_ground_truth(self.groundtruth,
                                                 self.json_results)
        gt_classes = list(gt_per_classes_dict.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)
        # print(gt_classes)
        # print(gt_per_classes_dict)

        dt_per_classes_dict = parse_detection_results(self.detection_results,
                                                      self.json_results)

        MIN_OVERLAP = 0.5

        # 计算每个类别的tp/fp
        sum_AP = 0.0
        sum_recall = 0.0
        sum_precision = 0.0

        metrics = dict()
        for cate in gt_classes:
            tp, fp = compute_tp_fp(dt_per_classes_dict,
                                   self.json_results,
                                   cate,
                                   MIN_OVERLAP=MIN_OVERLAP)

            prec, rec = compute_precision_recall(tp, fp,
                                                 gt_per_classes_dict[cate])

            # ap = voc_ap2(rec[:], prec[:])
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            recall = np.mean(mrec)
            precision = np.mean(mprec)

            sum_AP += ap
            sum_recall += recall
            sum_precision += precision

            metrics[cate + "(ap/recall/precision)"] = [ap, recall, precision]

        mAP = sum_AP / n_classes
        metrics['mAP'] = mAP
        metrics['mRecall'] = sum_recall / n_classes
        metrics['mPrecision'] = sum_precision / n_classes

        for key, item in metrics.items():
            if isinstance(item, list):
                item = [round(x, 4) for x in item]
            elif isinstance(item, float):
                item = round(item, 4)
            metrics[key] = item

        return metrics

    def __call__(self):
        return self.voc_eval()


class CocoEval(object):
    def __init__(self, **kwargs):
        super(CocoEval, self).__init__()
        self.iou_threshold = 0.3
        self.process_args = kwargs['process_args']
        self.class_names = []
        self._COLORS = []
        self.draw_result = False
        self.save_key = kwargs['save_key']
        self.eval_mode = kwargs['eval_mode']

    def set_iou_threshold(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def set_class_names(self, names):
        self.class_names = names

    def set_colors(self, colors):
        self._COLORS = colors

    def set_draw_result(self, is_draw):
        self.draw_result = is_draw

    def __call__(self, quan_dataset_path, images_path, ann_path, input_size, normalization, save_eval_path, prefix=".jpg"):

        process = ModelProcess(**self.process_args)
        is_dataset = False
        if os.path.isfile(quan_dataset_path):
            is_dataset = False
        elif os.path.isdir(quan_dataset_path):
            is_dataset = True
        else:
            print('invaild input quan file!')
            os._exit(-1)
        if self.eval_mode == "dataset":
            process.quantize(fd_path=quan_dataset_path, is_dataset=is_dataset)
        save_imgs = os.path.join(save_eval_path, 'images')
        if not os.path.exists(save_imgs):
            os.makedirs(save_imgs)

        root_dir = os.getcwd()
        xml_path = "/home/ubuntu/zhangcc/dataset/voc/test/VOCdevkit/VOC2007/Annotations"
        groundtruth = "./work_dir/xml2txt/groundtruth"
        detection_results = "./work_dir/xml2txt/detection_results"
        json_results = "./work_dir/xml2txt/json_results"

        kwargs_xml2txt = {
            "xml_path": ann_path,
            "groundtruth": groundtruth,
            "detection_results": detection_results,
            "is_fakedata": False
        }

        kwargs_voceval = {
            "xml2txt": kwargs_xml2txt,
            "detection_results": detection_results,
            "json_results": json_results
        }

        myVocEval = VocEval(**kwargs_voceval)
        metrics = myVocEval()

        images = glob.glob(images_path+'/*'+prefix)

        for item in tqdm.tqdm(images):
            idx, image_name = item['id'], item['file_name']
            img = cv2.imread(os.path.join(images_path, image_name))
            if self.eval_mode == "single":
                del process
                process = ModelProcess(**self.process_args)
                process.quantize(fd_path=quan_dataset_path, is_dataset=is_dataset)
            process.dataflow(img, acc_error=True)
            outputs = process.get_outputs()['qout']

            # if self.draw_result:
            #     qres = draw_box(img, outputs, class_names=self.class_names, colors=self._COLORS)
            #     cv2.imwrite(os.path.join(save_imgs, os.path.basename(image_name)), qres)  ### results for draw bbox

        return metrics


if __name__ == '__main__':
    root_dir = os.getcwd()
    xml_path = "/home/ubuntu/zhangcc/dataset/voc/test/VOCdevkit/VOC2007/Annotations"
    groundtruth = "{}/work_dir/xml2txt/groundtruth".format(root_dir)
    detection_results = "{}/work_dir/xml2txt/detection_results".format(
        root_dir)
    json_results = "{}/work_dir/xml2txt/json_results".format(root_dir)

    kwargs_xml2txt = {
        "xml_path": xml_path,
        "groundtruth": groundtruth,
        "detection_results": detection_results,
        "is_fakedata": True
    }

    kwargs_voceval = {
        "xml2txt": kwargs_xml2txt,
        "detection_results": detection_results,
        "json_results": json_results
    }

    myVocEval = VocEval(**kwargs_voceval)
    metrics = myVocEval()
    for key, item in metrics.items():
        print("{}: {}".format(key, item))
