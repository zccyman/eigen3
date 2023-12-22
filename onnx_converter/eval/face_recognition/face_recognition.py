"""
@Project    : cosmo-face
@Module     : validation_tf_test.py
@Author     : HuangJiWen[jiwen.huang@timesintelli.com]
@Created    : 2021/6/8 17:30
@Desc       : 
"""
import copy
import os
import time

# import bcolz
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from eval import Eval

try:
    
    from tools import ModelProcess, OnnxruntimeInfer
    from utils.BaseObject import Object
except:
    from onnx_converter.tools import ModelProcess, OnnxruntimeInfer
    from onnx_converter.utils import Object


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


class RecEval(Eval):
    def __init__(self, **kwargs):
        super(RecEval, self).__init__(**kwargs)
        
        self.batch_size = kwargs["batch_size"]
        self.is_loadtxt = kwargs["is_loadtxt"]
        self.feat_dim = kwargs["feat_dim"]
        self.fp_result = kwargs["fp_result"]
        self.nrof_folds = kwargs["nrof_folds"]
        self.metric = kwargs["metric"]
        self.max_threshold = kwargs["max_threshold"]
        self.dataset_dir = kwargs["dataset_dir"]
        self.dataset_name = kwargs["dataset_name"]
        self.test_sample_num = kwargs["test_sample_num"]
        self.result_path = kwargs["result_path"]
        self.process_args = kwargs["process_args"]
        self.is_calc_error = kwargs['is_calc_error']
        # self.is_stdout = self.process_args['is_stdout']
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)
        self.acc_error = kwargs['acc_error']
        self.eval_mode = kwargs["eval_mode"]
        os.makedirs(self.result_path, exist_ok=True)

        self.carray, self.issame = get_validation_pair(self.dataset_name, self.dataset_dir)
        if hasattr(self, "test_sample_num"):
            self.carray = self.carray[:self.test_sample_num, ...]
            self.issame = self.issame[:self.test_sample_num // 2]

        self.process = ModelProcess(**self.process_args)
  
    def __call__(self):
        smin, smax = 0, len(self.carray) - 2
        select_carray = np.arange(smin, smax)
        select_issame = np.arange(smin // 2, smax // 2)
        carray = self.carray[select_carray, :, :, :]
        issame = self.issame[select_issame]

        idx = 0
        qembeddings = np.zeros([len(carray), self.feat_dim])
        tembeddings = np.zeros([len(carray), self.feat_dim])
        
        output_names, input_names = self.process.get_output_names(), self.process.get_input_names()
        onnxinferargs = copy.deepcopy(self.process_args)
        onnxinferargs.update(out_names=output_names, input_names=input_names)
        onnxinfer = OnnxruntimeInfer(**onnxinferargs)
        
        if self.eval_mode == "dataset":
            self.process.quantize(fd_path=self.carray[:, ...], is_dataset=True)
            
        while idx + self.batch_size <= len(carray):

            # start_time = time.time()

            sname = str(smin + idx).zfill(6)
            batch = carray[idx:idx + self.batch_size]

            if self.fp_result:
                true_outputs = onnxinfer(in_data=batch)
            else:
                true_outputs = None
                    
            if self.is_loadtxt:
                qoutputs = np.loadtxt("{}/{}.txt".format(self.result_path, sname), delimiter=',')
                qoutputs = qoutputs.reshape(-1, self.feat_dim).reshape(-1, self.feat_dim)
            else:
                if self.eval_mode == "single":
                    self.process.quantize(batch, is_dataset=False)
                    
                if self.is_calc_error:
                    self.process.checkerror(batch, acc_error=self.acc_error)
                else:
                    self.process.dataflow(batch, acc_error=True, onnx_outputs=true_outputs)
                # if 'analysis_cfg' in self.process_args.keys() and self.fp_result:
                #     self.process.checkerror_weight(onnx_outputs=None)
                #     self.process.checkerror_feature(onnx_outputs=true_outputs)
                                        
                qoutputs = self.process.get_outputs()['qout']['output']
                qoutputs = F.normalize(torch.from_numpy(qoutputs), p=2, dim=1).numpy()
                np.savetxt("{}/{}.txt".format(self.result_path, sname), qoutputs.reshape(-1), fmt='%f',
                           delimiter=',')

            qembeddings[idx:idx + self.batch_size] = qoutputs
            if self.fp_result:
                foutputs = F.normalize(torch.from_numpy(true_outputs["output"]), p=2, dim=1).numpy()
                tembeddings[idx:idx + self.batch_size] = foutputs
            # print("=> idx: ", smin + idx, issame[idx // 2], "elapsed_time:", time.time() - start_time, "s")

            idx += self.batch_size

            if self.batch_size == idx and self.eval_first_frame: break
            
        _, _, qaccuracy, qbest_threshold = evaluate(qembeddings, issame, self.nrof_folds, metric=self.metric,
                                                  max_threshold=self.max_threshold)
        qaccuracy = dict(accuracy=qaccuracy.mean(), best_threshold=qbest_threshold.mean())
        if self.fp_result:
            _, _, taccuracy, tbest_threshold = evaluate(tembeddings, issame, self.nrof_folds, metric=self.metric,
                                                      max_threshold=self.max_threshold)
            taccuracy = dict(accuracy=taccuracy.mean(), best_threshold=tbest_threshold.mean())
            accuracy = dict(qaccuracy=qaccuracy, faccuracy=taccuracy)
        else:
            accuracy = dict(qaccuracy=qaccuracy)

        tb = self.get_results(accuracy)

        if not self.is_calc_error and self.process.onnx_graph:
            img = carray[0:1]
            true_outputs = self.process.post_quan.onnx_infer(img)
            self.process.numpygraph(img, acc_error=True, onnx_outputs=true_outputs)
            
        return accuracy, tb
