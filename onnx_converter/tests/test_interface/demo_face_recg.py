# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 22023/7/16
# @File     : demo_face_recg.py


import sys

import cv2  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import json
import os
import copy
import os
import time

import bcolz
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import tqdm
from OnnxConverter import OnnxConverter
import prettytable as pt


def get_validation_pair(name, data_dir):
    carray_data = bcolz.carray(rootdir=data_dir + "/{}".format(name), mode="r")
    issame = np.load(data_dir + "/{}_list.npy".format(name))
    return carray_data, issame


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_dist(embeddings1, embeddings2, metric):
    if metric == "cosine":
        dist = np.dot(embeddings1, embeddings2.T)
        dist = 1.0 - np.mean(dist, axis=1)
    elif metric == "euclidean":
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), axis=1)    
        
    return dist


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0, metric="cosine"):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        # diff = np.subtract(embeddings1, embeddings2)
        # dist = np.sum(np.square(diff), 1)
        dist = calculate_dist(embeddings1, embeddings2, metric=metric)
        
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            # diff = np.subtract(embed1, embed2)
            # dist = np.sum(np.square(diff), 1)
            dist = calculate_dist(embed1, embed2, metric=metric)
            
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        #         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0, metric="cosine", max_threshold=1.0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, max_threshold, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca, metric=metric)

    return tpr, fpr, accuracy, best_thresholds


def preprocess(img):
    in_data = copy.deepcopy(img)
    if img.shape[-1] == 3:
        in_data = img.transpose(2,0,1)
        in_data = in_data.astype(np.float32) / 255
        in_data = (in_data - 0.5) / 0.5
    expand_im = np.expand_dims(in_data, axis=0)
    return expand_im.astype(np.float32)

def preprocess_1(img):
    in_data = copy.deepcopy(img)
    if img.shape[-1] == 3:
        in_data = img.transpose(2,0,1)
        in_data = in_data.astype(np.float32) / 255
        in_data = (in_data - 0.5) / 0.5
    expand_im = np.expand_dims(in_data, axis=0)
    return expand_im.astype(np.float32)

def preprocess_isp(img):
    in_data = copy.deepcopy(img)
    if img.shape[-1] == 3:
        in_data = img.transpose(2,0,1)
        in_data = (in_data.astype(np.int32) * 2 - 255) * 257
        in_data = (in_data >> 11).astype(np.int8)
    expand_im = np.expand_dims(in_data, axis=0)
    return expand_im.astype(np.float32)

def postprocess(output):
    # return np.linalg.norm(output, ord=2, axis=0)
    return F.normalize(torch.from_numpy(output), p=2, dim=1).numpy()


class RecEval(object):
    def __init__(self, **kwargs):
        super(RecEval, self).__init__()
        
        self.feat_dim = kwargs.get("feat_dim", 512)
        self.dataset_name = kwargs.get("dataset_name", 512)
        self.nrof_folds = kwargs.get("nrof_folds", 5)
        self.metric = kwargs.get("metric", "euclidean")
        self.max_threshold = kwargs.get("max_threshold", 4)
        self.dataset_dir = kwargs.get("dataset_dir", "/buffer/faces_emore/")
        self.test_sample_num = kwargs.get("test_sample_num", 1200)

        self.carray, self.issame = get_validation_pair(self.dataset_name, self.dataset_dir)
        if self.test_sample_num > 0:
            self.carray = self.carray[:self.test_sample_num, ...]
            self.issame = self.issame[:self.test_sample_num // 2]
            
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
        smin, smax = 0, len(self.carray) - 2
        select_carray = np.arange(smin, smax)
        select_issame = np.arange(smin // 2, smax // 2)
        carray = self.carray[select_carray, :, :, :]
        issame = self.issame[select_issame]

        # idx = 0
        qembeddings = np.zeros([len(carray), self.feat_dim])
        for idx in tqdm.tqdm(range(len(carray))):    
        # while idx <= len(carray):

            results = converter.model_simulation(carray[idx])
            results_cvt = results['result_converter']
            pred_cvt = postprocess(results_cvt['output'])
            qembeddings[idx] = pred_cvt

            # idx += 1
            
        _, _, qaccuracy, qbest_threshold = evaluate(qembeddings, issame, self.nrof_folds, metric=self.metric,
                                                  max_threshold=self.max_threshold)
        qaccuracy = dict(accuracy=qaccuracy.mean(), best_threshold=qbest_threshold.mean())
        accuracy = dict(qaccuracy=qaccuracy)

        tb = self.get_results(accuracy)
            
        return accuracy, tb

def ispdata_export(converter: OnnxConverter, abgr_file: str):
    # Run model convert and export(optional)
    abgr_img = np.fromfile(abgr_file,dtype=np.uint8)
    dummy_input = abgr_img.reshape(1080,1920,4)[:,:,1:4]
    preprocess = preprocess_isp()
    postprocess = postprocess()
    converter.reset_preprocess(preprocess)
    results = converter.model_simulation(dummy_input, isp_data=True)            
    results_cvt = results['result_converter']
    trans = preprocess.get_trans()
    pred_cvt = postprocess(results_cvt, trans=trans)
    converter.model_export(dummy_input=dummy_input)
    print("******* Finish model convert *******")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_json', default='/home/shiqing/Downloads/test_package/saturation/onnx-converter/tests/test_interface/arguments_face_recg.json', type=str)
    parser.add_argument('--model_export', type=bool, default=True)
    parser.add_argument('--perf_analyse', type=bool, default=True)
    parser.add_argument('--vis_qparams', type=bool, default=True)
    parser.add_argument('--isp_inference', type=bool, default=False)
    parser.add_argument('--mem_addr', type=str, default='psram')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("******* Start model convert *******")
    # parse user defined arguments
    argsparse = parse_args()
    args_json_file = argsparse.args_json
    flag_model_export = argsparse.model_export
    flag_perf_analyse = argsparse.perf_analyse
    flag_vis_qparams = argsparse.vis_qparams
    flag_isp_inference = argsparse.isp_inference 
    assert os.path.exists(args_json_file), "Please check argument json exists"
    args = json.load(open(args_json_file, 'r'))
    args_cvt = args['converter_args']
    error_analyzer = args_cvt['error_analyzer']
    
    calibration_dataset_dir = args_cvt["calibration_dataset_dir"]
    evaluation_dataset_dir = args_cvt["evaluation_dataset_dir"]
    args_cvt["transform"] = preprocess
    args_cvt["postprocess"] = postprocess
    # Build onnx converter
    converter = OnnxConverter(**args_cvt)     
    
    converter.load_model("/buffer/trained_models/face-recognition/mobilefacenet_pad_qat_simplify.onnx")
    converter.reset_preprocess(preprocess_1)
    
    # Calibration
    carray, _ = get_validation_pair(args_cvt["dataset_name"], calibration_dataset_dir)
    # converter.calibration(carray[:, ...])
    converter.calibration("/buffer/calibrate_dataset/face_recognition")
    print("calibration done!")
    
    # Build evaluator
    evaluator = RecEval(**args_cvt)

    # Evaluate quantized model accuracy
    evaluator(converter=converter)
    print("******* Finish model evaluate *******")
    
    if flag_vis_qparams:
        converter.visualize_qparams()
        print("******* Finish qparams visualization *******") 

    if flag_model_export:
        # Run model convert and export(optional)
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
    if flag_perf_analyse:
        time = converter.perf_analyze(mem_addr = argsparse.mem_addr)
        print("******* The estimated time cost is %f ms *******"%(time/1000))
        print("******* Finish performance analysis *******")