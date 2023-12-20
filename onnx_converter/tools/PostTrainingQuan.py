# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/6 15:02
# @File     : PostTrainingQuan.py

import copy
import os
from glob import glob
import pickle
import onnx
import cv2
import copy
import numpy as np
import onnxruntime as rt
import torch
from tqdm import tqdm
import json
from absl import logging
from collections import Counter
import numpy as np
from scipy.stats import entropy

try:
    from utils import flatten_list, Object, process_im, save_txt
    from config import Config
    from extension.libs.kld import calculate_threshold, calculate_kld_vector, calculate_kld_kernel # type: ignore
except:
    from onnx_converter.utils import flatten_list, Object, process_im, props_with_, save_txt # type: ignore
    from onnx_converter.config import Config # type: ignore
    from onnx_converter.extension.libs.kld import calculate_threshold, calculate_kld_vector, calculate_kld_kernel # type: ignore


def _compute_amax_entropy(calib_hist, calib_bin_edges, num_bits=8, unsigned=True, stride=1, start_bin=128):
    """Returns amax that minimizes KL-Divergence of the collected histogram"""

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    def _normalize_distr(distr):
        summ = np.sum(distr)
        if summ != 0:
            distr = distr / summ

    bins = calib_hist[:]
    bins[0] = bins[1]

    total_data = np.sum(bins)

    divergences = []
    arguments = []

    # we are quantizing to 128 values + sign if num_bits=8, nbins=256
    nbins = 1 << (num_bits - 1 + int(unsigned))

    starting = start_bin
    stop = len(bins)

    new_density_counts = np.zeros(nbins, dtype=np.float64)

    for i in range(starting, stop + 1, stride):
        new_density_counts.fill(0)
        space = np.linspace(0, i, num=nbins + 1)
        digitized_space = np.digitize(range(i), space) - 1

        digitized_space[bins[:i] == 0] = -1

        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density_counts[digitized] += bins[idx]

        counter = Counter(digitized_space)
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        total_counts_new = np.sum(new_density) + np.sum(bins[i:])
        _normalize_distr(new_density)

        reference_density = np.array(bins[:len(digitized_space)])
        reference_density[-1] += np.sum(bins[i:])

        total_counts_old = np.sum(reference_density)
        if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
            raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
                total_counts_new, total_counts_old, total_data))

        _normalize_distr(reference_density)

        ent = entropy(reference_density, new_density)
        divergences.append(ent)
        arguments.append(i)

    divergences = np.array(divergences)
    logging.debug("divergences={}".format(divergences))
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    calib_amax = calib_bin_edges[last_argmin * stride + starting]
    calib_amax = torch.tensor(calib_amax.item()) #pylint: disable=not-callable

    return calib_amax

def _compute_amax_mse(calib_hist, calib_bin_edges, num_bits=8, unsigned=True, stride=1, start_bin=128):
    """Returns amax that minimizes MSE of the collected histogram"""

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    counts = torch.from_numpy(calib_hist[:]).float().cuda()
    edges = torch.from_numpy(calib_bin_edges[:]).float().cuda()
    centers = (edges[1:] + edges[:-1]) / 2
    max_value = 2.0 ** (num_bits - 1) - 1
    min_value = -2.0 ** (num_bits - 1)
    
    mses = []
    arguments = []

    for i in range(start_bin, len(centers), stride):

        amax = centers[i]
        scale = amax / max_value
        quant_centers = torch.round(centers / scale).clamp(min_value, max_value) * scale
        # quant_centers = fake_tensor_quant(centers, amax, num_bits, unsigned)

        mse = ((quant_centers - centers)**2 * counts).mean()

        mses.append(mse.cpu())
        arguments.append(i)

    logging.debug("mses={}".format(mses))
    argmin = np.argmin(mses)
    calib_amax = centers[arguments[argmin]]

    return calib_amax.cpu()

def _compute_amax_percentile(calib_hist, calib_bin_edges, percentile):
    """Returns amax that clips the percentile fraction of collected data"""

    if percentile < 0 or percentile > 100:
        raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges is None and calib_hist is None:
        return None

    total = calib_hist.sum()
    cdf = np.cumsum(calib_hist / total)
    idx = np.searchsorted(cdf, percentile / 100)
    calib_amax = calib_bin_edges[idx]
    calib_amax = torch.tensor(calib_amax.item()) #pylint: disable=not-callable

    return calib_amax


class HistogramCalibrator(object):
    """Unified histogram calibrator

    Histogram will be only collected once. compute_amax() performs entropy, percentile, or mse
        calibration based on arguments

    Args:
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.
        num_bins: An integer. Number of histograms bins. Default 2048.
        grow_method: A string. DEPRECATED. default None.
        skip_zeros: A boolean. If True, skips zeros when collecting data for histogram. Default False.
        torch_hist: A boolean. If True, collect histogram by torch.histc instead of np.histogram. If input tensor
            is on GPU, histc will also be running on GPU. Default True.
    """
    def __init__(self, num_bits=8, axis=None, unsigned=True, num_bins=2048, grow_method=None, skip_zeros=False, torch_hist=False):
        super(HistogramCalibrator, self).__init__()
        self._num_bins = num_bins
        self._skip_zeros = skip_zeros

        self._calib_bin_edges = None
        self._calib_hist = None

        self._torch_hist = torch_hist

        if axis is not None:
            raise NotImplementedError("Calibrator histogram collection only supports per tensor scaling")

        if grow_method is not None:
            logging.warning("grow_method is deprecated. Got %s, ingored!", grow_method)

    def collect(self, x):
        """Collect histogram"""
        if np.min(x) < 0.:
            logging.log_first_n(
                logging.INFO,
                ("Calibrator encountered negative values. It shouldn't happen after ReLU. "
                 "Make sure this is the right tensor to calibrate."),
                1)
            x = np.abs(x)

        x = x.astype(np.float32)

        if not self._torch_hist:
            x_np = x #.cpu().detach().numpy()

            if self._skip_zeros:
                x_np = x_np[np.where(x_np != 0)]

            if self._calib_bin_edges is None and self._calib_hist is None:
                # first time it uses num_bins to compute histogram.
                self._calib_hist, self._calib_bin_edges = np.histogram(x_np, bins=self._num_bins)
            else:
                temp_amax = np.max(x_np)
                if temp_amax > self._calib_bin_edges[-1]:
                    # increase the number of bins
                    width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                    # NOTE: np.arange may create an extra bin after the one containing temp_amax
                    new_bin_edges = np.arange(self._calib_bin_edges[-1] + width, temp_amax + width, width)
                    self._calib_bin_edges = np.hstack((self._calib_bin_edges, new_bin_edges))
                hist, self._calib_bin_edges = np.histogram(x_np, bins=self._calib_bin_edges)
                hist[:len(self._calib_hist)] += self._calib_hist
                self._calib_hist = hist
        else:
            # This branch of code is designed to match numpy version as close as possible
            with torch.no_grad():
                if self._skip_zeros:
                    x = x[torch.where(x != 0)]

                # Because we collect histogram on absolute value, setting min=0 simplifying the rare case where
                # minimum value is not exactly 0 and first batch collected has larger min value than later batches
                x_max = x.max()
                if self._calib_bin_edges is None and self._calib_hist is None:
                    self._calib_hist = torch.histc(x, bins=self._num_bins, min=0, max=x_max)
                    self._calib_bin_edges = torch.linspace(0, x_max, self._num_bins + 1)
                else:
                    if x_max > self._calib_bin_edges[-1]:
                        width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                        self._num_bins = int((x_max / width).ceil().item())
                        self._calib_bin_edges = torch.arange(0, x_max + width, width, device=x.device)

                    hist = torch.histc(x, bins=self._num_bins, min=0, max=self._calib_bin_edges[-1])
                    hist[:self._calib_hist.numel()] += self._calib_hist
                    self._calib_hist = hist

    def reset(self):
        """Reset the collected histogram"""
        self._calib_bin_edges = None
        self._calib_hist = None

    def compute_amax(
            self, method: str, *, stride: int = 1, start_bin: int = 128, percentile: float = 99.99):
        """Compute the amax from the collected histogram

        Args:
            method: A string. One of ['entropy', 'mse', 'percentile']

        Keyword Arguments:
            stride: An integer. Default 1
            start_bin: An integer. Default 128
            percentils: A float number between [0, 100]. Default 99.99.

        Returns:
            amax: a tensor
        """
        if isinstance(self._calib_hist, torch.Tensor):
            calib_hist = self._calib_hist.int().cpu().numpy()
            calib_bin_edges = self._calib_bin_edges.cpu().numpy()
        else:
            calib_hist = self._calib_hist
            calib_bin_edges = self._calib_bin_edges

        if method == 'entropy':
            calib_amax = _compute_amax_entropy(
                calib_hist, calib_bin_edges, self._num_bits, self._unsigned, stride, start_bin)
        elif method == 'mse':
            calib_amax = _compute_amax_mse(
                calib_hist, calib_bin_edges, self._num_bits, self._unsigned, stride, start_bin)
        elif method == 'percentile':
            calib_amax = _compute_amax_percentile(calib_hist, calib_bin_edges, percentile)
        else:
            raise TypeError("Unknown calibration method {}".format(method))

        return calib_amax
    
    
def calibration_histogram(feat_arrays: dict):
    pass


def calibration_ema(feat_arrays: dict):
    pass


def calibration_kld(fea_arrays: dict):
    pass


def calibration_saturation_kld(fea_arrays: dict):
    pass


class OnnxruntimeInfer(Object): # type: ignore
    def __init__(self, **kwargs):
        # output with layer in quantize graph
        # get data scales using onnx runtime inference
        super(OnnxruntimeInfer, self).__init__(**kwargs)
        self.model_path = kwargs['model_path']
        if 'log_name' in kwargs.keys():
            self.logger = self.get_log(log_name=kwargs['log_name'], log_level=kwargs.get('log_level', 20))
        else:
            self.logger = self.get_log('postquan.log', log_level=kwargs.get('log_level', 20))
        if 'transform' in kwargs.keys():
            setattr(self, 'transform', kwargs['transform'])
            # self.logger.info('user define transform will using data tranform!')
        self.out_names, self.input_names = kwargs['out_names'], kwargs['input_names']
        # try:
        #     parsed_cfg = Config.fromfile(kwargs['parse_cfg'])
        #     parse_dict, _ = parsed_cfg._file2dict(kwargs['parse_cfg'])
        # except:
        #     from onnx_converter.config import parse
        #     parse_dict = props_with_(parse)

        self.logger.info('onnx runtime log level is: {}'.format(3))
        rt.set_default_logger_severity(3)
        self.model = self.add_layer_output_to_graph()
        self.device = kwargs['device'] if 'device' in kwargs.keys() else 'cpu'
        self.sess_options = kwargs['sess_options'] if 'sess_options' in kwargs.keys() else None

        providers = ['CPUExecutionProvider']
        device = None
        if 'cuda' in self.device:
            if 'CUDAExecutionProvider' in rt.get_available_providers():
                providers = ['CUDAExecutionProvider']
                device = [{'device_id': int(self.device.split(":")[-1])}]
        self.logger.info(rt.get_available_providers())
        self.logger.info(device)
        if isinstance(self.model, str):
            # self.sess = rt.InferenceSession(self.model, providers=providers, provider_options=device)
            self.__sess = rt.InferenceSession(self.model, providers=providers, provider_options=device)
            # self.sess = rt.InferenceSession(self.model, providers=['CUDAExecutionProvider'])
            # elf.sess = rt.InferenceSession(self.model, providers=providers)
        else:
            # self.sess = rt.InferenceSession(self.model.SerializeToString(), providers=providers, provider_options=device)
            self.__sess = rt.InferenceSession(self.model.SerializeToString(), providers=providers, provider_options=device)
            # self.sess = rt.InferenceSession(self.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            # elf.sess = rt.InferenceSession(self.model.SerializeToString(), providers=providers)

    def set_transform(self, transform):
        setattr(self, 'transform', transform)

    def add_layer_output_to_graph(self):
        if isinstance(self.model_path, str):
            model_ = onnx.load_model(self.model_path)
        else:
            model_ = copy.deepcopy(self.model_path)
        # if self.is_remove_transpose:
        #     model_ = remove_transpose(model_)

        self.output_names = []
        output_names_ = flatten_list(self.out_names)
        input_names_ = flatten_list(self.input_names)
        for output in output_names_:
            if output in input_names_: continue
            model_.graph.output.extend([onnx.ValueInfoProto(name=output)])
            self.output_names.append(output)

        return model_

    def get_session(self):
        return self.__sess

    def get_layer_output_names(self):
        return [output.name for output in self.__sess.get_outputs()]

    def get_layer_input_names(self):
        return [input.name for input in self.__sess.get_inputs()]

    def forward(self, in_datas):
        inputs = self.__sess.get_inputs()
        # outputs = self.sess.get_outputs()
        # if len(inputs) != len(self.input_names):
        #     self.logger.info('model input length must be the same with self.input_names!')

        # output_names = [out.name for out in outputs]
        input_feed, results = dict(), dict()
        if isinstance(in_datas, dict):
            if hasattr(self, 'transform'):
                input_feed = results = self.transform(in_datas)
            else:
                os._exit(-1)
        else:
            for idx, item in enumerate(inputs):
                name = item.name
                if isinstance(in_datas, dict):
                    data = in_datas[name]
                else:
                    data = in_datas
                if hasattr(self, 'transform') and self.transform:
                    data = self.transform(data)
                else:
                    data = copy.deepcopy(data)
                    data = process_im(data, item.shape[2:])

                input_feed[name] = data
                results[name] = data

        preds = self.__sess.run(output_names=self.output_names, input_feed=input_feed)

        for idx, _ in enumerate(self.output_names):
            results[self.output_names[idx]] = preds[idx]

        return results

    def __call__(self, in_data):
        # if hasattr(self, 'transform'):
        #     data = self.transform(in_data)
        # else:
        #     data = copy.deepcopy(in_data)

        infers = self.forward(in_data)
        return infers


ema_f = lambda x, y, ema_value: x * ema_value + (1 - ema_value) * y

def save_calibration(cali_scales, key, mean, new_mean, scales, min_v, max_v):
    root_fd = "/home/shiqing/Downloads/test_package/saturation/onnx-converter/work_dir/reset_lstm_hidden_state_calibration"
    if not os.path.exists(root_fd):
        os.makedirs(root_fd)
    save_txt(os.path.join(root_fd, "{}_calibration_dataset_mean.txt".format(key)), "a+", mean)
    save_txt(os.path.join(root_fd, "{}_calibration_mean_abs.txt".format(key)), "a+", new_mean)
    save_txt(os.path.join(root_fd, "{}_calibration_mean.txt".format(key)), "a+", np.mean(scales[key]))
    save_txt(os.path.join(root_fd, "{}_calibration_std.txt".format(key)), "a+", np.std(scales[key]))
    save_txt(os.path.join(root_fd, "{}_calibration_min.txt".format(key)), "a+", min_v)
    save_txt(os.path.join(root_fd, "{}_calibration_max.txt".format(key)), "a+", max_v)
    save_txt(os.path.join(root_fd, "{}_calibration_ema_min.txt".format(key)), "a+", cali_scales[key]['min'])
    save_txt(os.path.join(root_fd, "{}_calibration_ema_max.txt".format(key)), "a+", cali_scales[key]['max'])
    save_txt(os.path.join(root_fd, "{}_calibration_minmax_min.txt".format(key)), "a+", cali_scales[key]['cali_min_v'])
    save_txt(os.path.join(root_fd, "{}_calibration_minmax_max.txt".format(key)), "a+", cali_scales[key]['cali_max_v'])

# get each layer so: from single image, from datasets
class PostTrainingQuan(Object): # type: ignore
    def __init__(self, **kwargs):
        # kwargs must be included image transform
        # checkpoint->quan graph
        super(PostTrainingQuan, self).__init__(**kwargs)
        self.__model = OnnxruntimeInfer(**kwargs)
        self.__ema_value = kwargs['ema_value'] if 'ema_value' in kwargs.keys() else 0.99
        self.__quan_graph = kwargs['graph']
        self.__out_names = kwargs['out_names']
        self.__scales = dict()
        self.__scales_ = dict()
        self.__is_ema = kwargs.get('is_ema', False)
        self.__is_array = kwargs.get('is_array', False)
        log_name=kwargs.get("log_name", "postquan.log")
        log_level=kwargs.get("log_level")
        self.is_stdout = kwargs['is_stdout']
        self.logger = self.get_log(log_name=log_name, log_level=log_level)

    def get_layer_output_names(self):
        return self.__model.get_layer_output_names()

    def get_onnx_session(self):
        return self.__model.get_session()

    def get_scales(self):
        return self.__scales

    def set_graph(self, graph):
        self.__quan_graph = graph

    def get_graph(self):
        return self.__quan_graph

    def get_layers(self):
        return self.get_graph().get_layers()

    def set_transform(self, transform):
        self.__model.set_transform(transform)

    def onnx_infer(self, infile):
        global in_data
        if isinstance(infile, str) and os.path.isfile(infile):
            # self.logger.info('input data is file!')
            if infile.endswith(".npy"):
                in_data = np.load(infile)
            else:
                in_data = cv2.imread(infile)
            if not isinstance(in_data, np.ndarray):
                self.logger.fatal('invalid input file!')
                os._exit(-1)
        elif isinstance(infile, dict):
            in_data = copy.deepcopy(infile)
        elif isinstance(infile, np.ndarray):
            in_data = copy.deepcopy(infile)
        else:
            self.logger.fatal('not support input data format!')
            os._exit(-1)
        scales = self.__model.forward(in_data)
        return scales

    def quan_file(self, infile):
        # in_data = self.onnx_infer(infile)
        # self.__scales = self.__model.forward(in_data)
        # self.logger.info('input data is file!')
        self.__scales = self.onnx_infer(infile)
        # del in_data
        # self.analysis_feat(self.scales)
        self.__quan_graph.quantize(self.__scales)
        # scales is numpy array dictionary
        return self.get_graph()

    def __init__scale(self, scales):        
        if not self.__is_array:
            for key in scales.keys():
                min_v, max_v = np.min(scales[key]), np.max(scales[key])
                mean = np.mean(np.abs(scales[key]))
                self.__scales[key] = dict(max=max_v, 
                                          min=min_v, 
                                          mean_v=mean, 
                                          cali_min_v=min_v,
                                          cali_max_v=max_v,
                                          zeros_point=0)
                # self.__scales_[key] = [scales[key]]
        else:
            for key in scales.keys():
                self.__scales_[key] = [scales[key]]
                # self.__scales[key] = [scales[key]]
    
    @staticmethod
    def __get_minmax_scale(scales, cali_scales, key, idx, ema_value, is_array, is_skip_shrink=False):
        if not is_array:
            min_v, max_v = np.min(scales[key]), np.max(scales[key])
            mean = cali_scales[key]['mean_v']
            new_mean = np.mean(np.abs(scales[key]))
            if is_skip_shrink and new_mean < 1e-4 \
               and np.abs(max_v) < 1e-3 and np.abs(min_v) < 1e-3:
                pass                    
            else:
                mean = (mean * (idx - 1) + new_mean) / idx
            
            cali_scales[key]['max'] = max(cali_scales[key]['max'], max_v)
            cali_scales[key]['min'] = min(cali_scales[key]['min'], min_v)
            cali_scales[key]['mean_v'] = mean
            cali_scales[key]['cali_max_v'] = ema_f(cali_scales[key]['max'], max_v, ema_value)
            cali_scales[key]['cali_min_v'] = ema_f(cali_scales[key]['min'], min_v, ema_value)
            # save_calibration(cali_scales, key, mean, new_mean, scales, min_v, max_v)
            # cali_scales[key].append(scales[key])
        else:
            cali_scales[key].append(scales[key])
            # cali_scales[key].append(scales[key])
            # cali_scales[key] = np.row_stack([cali_scales[key], scales[key]])
    
    @staticmethod
    def __get_ema_scales(scales, cali_scales, key, idx, ema_value, is_array, is_skip_shrink=False):
        if not is_array:
            min_v, max_v, zeros_point = np.min(scales[key]), np.max(scales[key]), 0
            mean = (cali_scales[key]['mean_v'] * (idx - 1) + np.mean(scales[key])) / idx
            cali_scales[key]['max'] = ema_f(cali_scales[key]['max'], max_v, ema_value)
            cali_scales[key]['min'] = ema_f(cali_scales[key]['min'], min_v, ema_value)
            cali_scales[key]['mean_v'] = mean
        else:
            cali_scales[key].append(scales[key])
    
    def MinMaxScales(self, scales, idx=-1):
        if not bool(self.__scales):
            self.__init__scale(scales)
        else:
            # f = lambda x,y:x*self.ema_value + (1-self.ema_value)*y
            # f = lambda x, y: x * self.__ema_value + (1 - self.__ema_value) * y
            for key in self.__scales.keys():
                self.__get_minmax_scale(scales, self.__scales, key, idx, self.__ema_value, self.__is_array, True)

    def ema_scales(self, scales, idx=0):
        if not bool(self.__scales):
            self.__init__scale(scales)
        else:            
            for key in self.__scales.keys():
                self.__get_ema_scales(scales, self.__scales, key, idx, self.__ema_value, self.__is_array)

    def calbiration_scales(self, scales):
        if not bool(self.__scales):
            self.__init__scale(scales)
        else:
            pass
    
    def quan_dataset_with_alread_scales(self, fd_path, prefix=['png', 'jpeg', 'jpg', '.npy'], saved_calib_name="calibration_scales"):
        if os.path.exists("{}.scales".format(saved_calib_name)):
            with open("{}.scales".format(saved_calib_name), "rb+") as f:
                self.__scales = pickle.load(f)
                self.__quan_graph.quantize(self.__scales)
                return self.get_graph()
        else:
            return self.quan_dataset(fd_path=fd_path, prefix=prefix, saved_calib_name=save_calibration)
    
    def quan_dataset(self, fd_path, prefix=['png', 'jpeg', 'jpg', '.npy'], saved_calib_name="calibration_scales", calibration_params_json_path=None):
        
        if isinstance(fd_path, np.ndarray):
            datasets = fd_path
        elif isinstance(fd_path, list) or isinstance(fd_path, tuple):
            datasets = fd_path
        else:
            if not os.path.isdir(fd_path):
                self.logger.fatal('input path is invaild!')
                os._exit(-1)
            datasets = list()
            if isinstance(prefix, list):
                datasets = flatten_list([glob(os.path.join(fd_path, '*.{}'.format(item))) for item in prefix])
            elif isinstance(prefix, str):
                datasets = glob(os.path.join(fd_path, '*.{}'.format(prefix)))
            else:
                self.logger.fatal('prefix is invalid, prefix is: {}'.format(prefix))
                os._exit(-1)
            datasets.sort()

        if len(datasets) < 1:
            self.logger.fatal('file path has not data file!')
            os._exit(-1)
        self.logger.info('dataset len is: {}'.format(len(datasets)))

        self.__is_ema = len(datasets) > 100
        if 0:
            self.logger.info('start quantize datasets!')
            idx = 0
            for name in tqdm(datasets, postfix="quant datasets"):
                if isinstance(name, str) and not os.path.exists(name):
                    self.logger.info('file is not exists!')
                    continue
                # in_data = self.onnx_infer(name)
                # scales = self.__model.forward(in_data)
                scales = self.onnx_infer(name)
                self.ema_scales(scales)
                # self.MinMaxScales(scales, idx=idx+1)
                idx += 1
            self.logger.info('end quantize datasets!')
        else:
            self.logger.info('start quantize datasets!')
            HistogramCalibrators = {}
            for name in tqdm(datasets, postfix='quantize datasets') if self.is_stdout else datasets:
                if isinstance(name, str) and not os.path.exists(name):
                    self.logger.info('file is not exists!')
                    continue
                scales = self.onnx_infer(name)
                for key in scales.keys():
                    if key not in HistogramCalibrators.keys():
                        HistogramCalibrators[key] = HistogramCalibrator()
                    HistogramCalibrators[key].collect(scales[key])
            
            for key in HistogramCalibrators.keys():
                max_v = HistogramCalibrators[key].compute_amax(method='entropy') # ['entropy', 'mse', 'percentile']
                min_v = -max_v
                scales_ = dict(min=min_v.numpy(), max=max_v.numpy(), zeros_point=0)  
                self.__scales[key] = scales_
                
            self.logger.info('end quantize datasets!')
        
        if self.__is_ema:
            for key in self.__scales.keys():
                if isinstance(self.__scales[key], list):
                    self.__scales[key] = np.row_stack(self.__scales[key])

                # self.analysis_feat(self.scales)
        
        if calibration_params_json_path and os.path.exists(calibration_params_json_path):
            dmax_dmin = json.load(open(calibration_params_json_path, "r"))
            print("reload calibration params from: {}".format(calibration_params_json_path))
            scales_tmp = copy.deepcopy(self.__scales)
            for key in tqdm(scales_tmp.keys(), postfix='reload calibration scales'):
                print(key, scales_tmp[key], dmax_dmin[key])
                scales_tmp[key]["max"] = dmax_dmin[key]["max"]
                scales_tmp[key]["min"] = dmax_dmin[key]["min"]
            self.__scales = scales_tmp
        
        # with open("{}.scales".format(saved_calib_name), "wb+") as f:
        #     pickle.dump(self.__scales, f)
                
        self.__quan_graph.quantize(self.__scales)

        return self.get_graph()

    def map_quant_table(self, weight_scale_dict, top_scale_dict, fd_path, prefix='jpg', saved_calib_name="calibration_scales"):
        if top_scale_dict == None:
            if isinstance(fd_path, np.ndarray):
                datasets = fd_path
            elif isinstance(fd_path, list) or isinstance(fd_path, tuple):
                datasets = fd_path
            else:
                if not os.path.isdir(fd_path):
                    self.logger.fatal('input path is invaild!')
                    os._exit(-1)
                datasets = list()
                if isinstance(prefix, list):
                    datasets = flatten_list([glob(os.path.join(fd_path, '*.{}'.format(item))) for item in prefix])
                elif isinstance(prefix, str):
                    datasets = glob(os.path.join(fd_path, '*.{}'.format(prefix)))
                else:
                    self.logger.fatal('prefix is invalid, prefix is: {}'.format(prefix))
                    os._exit(-1)

            if len(datasets) < 1:
                self.logger.fatal('file path has not data file!')
                os._exit(-1)
            self.logger.info('dataset len is: {}'.format(len(datasets)))

            self.__is_ema = len(datasets) > 100
            self.logger.info('start quantize datasets!')
            for name in tqdm(datasets, postfix='quantize datasets') if self.is_stdout else datasets:
                if isinstance(name, str) and not os.path.exists(name):
                    self.logger.info('file is not exists!')
                    continue
                # in_data = self.onnx_infer(name)
                # scales = self.__model.forward(in_data)
                scales = self.onnx_infer(name)
                self.ema_scales(scales)
            self.logger.info('end quantize datasets!')

            if self.__is_ema:
                for key in self.__scales.keys():
                    if isinstance(self.__scales[key], list):
                        self.__scales[key] = np.row_stack(self.__scales[key])

                    # self.analysis_feat(self.scales)
            self.__quan_graph.quantize(self.__scales, weight_scale_dict)
        else:
            self.__quan_graph.quantize(top_scale_dict, weight_scale_dict)
        return self.get_graph()

    def quan_dataset_kld(self, fd_path, prefix='jpg', saved_calib_name="calibration_scales"):
        fast_mode = False
        if isinstance(fd_path, np.ndarray):
            datasets = fd_path
        elif isinstance(fd_path, list) or isinstance(fd_path, tuple):
            datasets = fd_path
        else:
            if not os.path.isdir(fd_path):
                self.logger.fatal('input path is invaild!')
                os._exit(-1)
            datasets = list()
            if isinstance(prefix, list):
                datasets = flatten_list([glob(os.path.join(fd_path, '*.{}'.format(item))) for item in prefix])
            elif isinstance(prefix, str):
                datasets = glob(os.path.join(fd_path, '*.{}'.format(prefix)))
            else:
                self.logger.fatal('prefix is invalid, prefix is: {}'.format(prefix))
                os._exit(-1)

        if len(datasets) < 1:
            self.logger.fatal('file path has not data file!')
            os._exit(-1)
        self.logger.info('dataset len is: {}'.format(len(datasets)))

        num_bins = 2048
        eps = 1e-7
        # find the featuremap's abs-max value based on calibration dataset
        featuremap_scale_dict = dict()

        # setp 1 find abs-max
        for data in tqdm(datasets, postfix='quantize datasets') if self.is_stdout else datasets:
            if isinstance(data, str) and not os.path.exists(data):
                self.logger.info('file is not exists!')
                continue
            results = self.onnx_infer(data)
            for key in results:
                if not key in featuremap_scale_dict.keys():
                    featuremap_scale_dict.update({key:{"absmax":np.max(np.abs(results[key])), "histogram" : np.zeros(num_bins).astype(np.float32)}})
                else:
                    featuremap_scale_dict[key]['absmax'] = np.maximum(featuremap_scale_dict[key]['absmax'], np.max(np.abs(results[key])))

        # step 2 build histogram
        for key in featuremap_scale_dict.keys():
            featuremap_scale_dict[key].update({"interval":featuremap_scale_dict[key]['absmax']/num_bins})

        for data in tqdm(datasets, postfix='quantize datasets') if self.is_stdout else datasets:
            if isinstance(data, str) and not os.path.exists(data):
                self.logger.info('file is not exists!')
                continue
            results = self.onnx_infer(data)
            for key in results:
                output = np.abs(results[key]).flatten()
                output = output[output > eps]
                hist, _ = np.histogram(output, bins = num_bins, range=(0, featuremap_scale_dict[key]['absmax']))
                featuremap_scale_dict[key]['histogram'] += hist
        # normalize histogram
        if not fast_mode:
            for key in featuremap_scale_dict.keys():
                featuremap_scale_dict[key]['histogram'] /= np.sum(featuremap_scale_dict[key]['histogram'])
        # # step 3 thresholding
        for name in featuremap_scale_dict.keys():
            #featuremap_scale_dict[name]['histogram'] /= sum(featuremap_scale_dict[name]['histogram'])
            if fast_mode:
                scale = calculate_threshold(featuremap_scale_dict[name]['histogram'], featuremap_scale_dict[name]['interval'], -1, -1)
                if scale < 1e-5:
                    scale = 1e-5
                    best_bin = None
                else:
                    best_bin = np.round(scale * 127 / featuremap_scale_dict[name]['interval'])
                featuremap_scale_dict[name].update({"scale": scale, "best_bin":best_bin})
            else:
                kl_dist_list = calculate_kld_vector(featuremap_scale_dict[name]['histogram'], featuremap_scale_dict[name]['interval'], -1, -1)
                featuremap_scale_dict[name].update({'kld': kl_dist_list})
                target_thr = np.argmin(kl_dist_list) + 1
                scale = target_thr * featuremap_scale_dict[name]['interval']/127
                if scale < 1e-5:
                    scale = 1e-5
                    target_thr = None
                featuremap_scale_dict[name].update({"scale": scale, "best_bin":target_thr})

        # # step 4 set scale to layers
        layers = self.__quan_graph.get_layers()
        layer_names = np.array([x.get_layer_name() for x in layers])
        for layer in layers:
            print(layer.get_layer_name())
            # layer_input_names = layer.get_onnx_input_name()
            # layer_output_names = layer.get_onnx_output_name()
            layer_type = layer.get_layer_type()
            if layer_type == "data":
                continue

            if not layer_type in ["conv", "depthwiseconv", "gemm", "fc", "matmul"]:
                continue
            input_name = layer.get_onnx_input_name()[0]
            si = featuremap_scale_dict[input_name]['scale']
            weight_data = layer.get_nodes()[0].get_weights()[0]['weight']
            sw = np.max(np.abs(weight_data)) / 127
            output_name = layer.get_onnx_output_name()[0]
            if featuremap_scale_dict[output_name]['best_bin'] is None:
                continue
            # scale of real clip-position
            if fast_mode:
                scale = calculate_threshold(featuremap_scale_dict[output_name]['histogram'], featuremap_scale_dict[output_name]['interval'], si, sw)
            else:
                kl_dist_list = calculate_kld_vector(featuremap_scale_dict[output_name]['histogram'], featuremap_scale_dict[output_name]['interval'], si, sw)
                target_thr = np.argmin(kl_dist_list) + 1
                scale = target_thr * featuremap_scale_dict[output_name]['interval']/127

            next_layer_names = layer.get_output_name()
            # next layer info
            if len(next_layer_names) == 0 or not next_layer_names[0] in layer_names:
                featuremap_scale_dict[output_name]['scale'] = scale
            else:
                layer_idx = np.argwhere(layer_names==next_layer_names[0])[0][0]
                layer_type = layers[layer_idx].get_layer_type()
                if len(next_layer_names) == 1 and layer_type in ["conv", "depthwiseconv", "gemm", "fc", "MatMul"]:
                    next_conv_layer = layers[layer_idx]
                    next_layer_output_name = next_conv_layer.get_onnx_output_name()[0]
                    so_next = featuremap_scale_dict[next_layer_output_name]['scale']
                    weight_data_next = next_conv_layer.get_nodes()[0].get_weights()[0]['weight']
                    sw_next = np.max(np.abs(weight_data_next)) / 127
                    n = np.floor(np.log2(so_next/sw_next/scale))
                    # weighted by two layers
                    scale_ref = so_next/sw_next * 2**(-n)
                    if fast_mode:
                        scale = 0.5*(scale + scale_ref)
                    else:
                        dx = (scale_ref - scale) / 20                #featuremap_scale_dict[output_name]['interval']
                        scale_candidates = [scale + i * dx for i in range(21)]
                        interval = featuremap_scale_dict[output_name]['interval']
                        bin_candidates = [int(x * 127 / interval + 0.5) for x in scale_candidates]
                        # scale_candidates = [i * interval / 127 for i in range(128, num_bins) if i * interval / 127 >= scale and i * interval / 127 <=scale_ref]
                        # bin_candidates = [i  for i in range(128, num_bins) if i * interval / 127 >= scale and i * interval / 127 <=scale_ref]
                        interval_next = featuremap_scale_dict[next_layer_output_name]['interval']
                        kld_next_list = list()

                        for scale_cand in scale_candidates:
                            scale_next_tests = [scale_cand * sw_next * 2**(n), scale_cand * sw_next * 2**(n+1)]# obtain two real clip thr
                            bin_next_0 = np.maximum(int(scale_next_tests[0] * 127 / interval_next + 0.5), 128) # lower limit
                            bin_next_1 = np.minimum(int(scale_next_tests[1] * 127 / interval_next + 0.5), num_bins-1) # upper limit
                            kld0 = featuremap_scale_dict[next_layer_output_name]['kld'][bin_next_0]
                            kld1 = featuremap_scale_dict[next_layer_output_name]['kld'][bin_next_1]
                            kld_next = min([kld0, kld1]) # better one as the final one
                            kld_next_list.append(kld_next)

                        kld_current_list = list()
                        thr = target_thr # best real clip thr for sw and si of current layer # type: ignore
                        for i, bin_idx in enumerate(bin_candidates):
                            target_bin = int((2 * thr - bin_idx) / thr * 64 + 64) # = out_scale / 2 # type: ignore
                            kld_current = calculate_kld_kernel(featuremap_scale_dict[output_name]['histogram'], thr, target_bin)
                            kld_current_list.append(kld_current)

                        kld_sum = np.array(kld_current_list) + np.array(kld_next_list)
                        if False:
                            import matplotlib.pyplot as plt
                            if not os.path.exists('work_dir/test'):
                                os.makedirs('work_dir/test')
                            plt.figure("current layers loss")
                            plt.subplot(311)
                            plt.plot(np.arange(0,21,1), kld_current_list, 'bo')
                            plt.subplot(312)
                            plt.plot(np.arange(0,21,1), kld_next_list, 'go')
                            plt.subplot(313)
                            plt.plot(np.arange(0,21,1), kld_sum, 'ro')
                            plt.savefig('work_dir/test/%s.png'%layer.get_layer_name())
                            plt.clf()
                        best_idx = np.argmin(kld_sum)
                        scale = scale_candidates[best_idx]
                    # finetune scale_out as real best threshold
                print(scale)
                featuremap_scale_dict[output_name]['scale'] = scale
        for key in featuremap_scale_dict.keys():
            clip_threshold = featuremap_scale_dict[key]['scale'] * 127
            self.__scales.update({key:{"max":clip_threshold, "min": -clip_threshold, "zeros_point":0}})

        self.__quan_graph.quantize(self.__scales)
        return self.get_graph()

    def export(self):
        pass

# if __name__ == '__main__':
#     def nms(dets, thresh):
#         # x1、y1、x2、y2、以及score赋值
#         # (x1、y1)(x2、y2)为box的左上和右下角标
#         x1 = dets[:, 0]
#         y1 = dets[:, 1]
#         x2 = dets[:, 2]
#         y2 = dets[:, 3]
#         scores = dets[:, 4]
#
#         # 每一个候选框的面积
#         areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#         # order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
#         order = scores.argsort()[::-1]
#         # ::-1表示逆序
#
#         temp = []
#         while order.size > 0:
#             i = order[0]
#             temp.append(i)
#             # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
#             # 由于numpy的broadcast机制，得到的是向量
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.minimum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.maximum(y2[i], y2[order[1:]])
#
#             # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
#             w = np.maximum(0.0, xx2 - xx1 + 1)
#             h = np.maximum(0.0, yy2 - yy1 + 1)
#             inter = w * h
#             # 计算重叠度IoU
#             ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#             # 找到重叠度不高于阈值的矩形框索引
#             inds = np.where(ovr <= thresh)[0]
#             # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
#             order = order[inds + 1]
#
#         return dets[temp]
#
#
#     def transform(images):
#         divisible, pad_value = 32, 0.0
#         img_heights, img_widths = [], []
#         for idx, img in enumerate(images):
#             img = torch.from_numpy(img.transpose(2, 0, 1))
#             images[idx] = img
#             img_heights.append(img.shape[-2])
#             img_widths.append(img.shape[-1])
#
#         max_h, max_w = max(img_heights), max(img_widths)
#         if divisible > 0:
#             max_h = (max_h + divisible - 1) // divisible * divisible
#             max_w = (max_w + divisible - 1) // divisible * divisible
#
#         batch_imgs = []
#         for img in images:
#             padding_size = [0, max_w - img.shape[-1], 0, max_h - img.shape[-2]]
#             batch_imgs.append(F.pad(img, padding_size, value=pad_value))
#
#         ### mean and std
#         mean = 116.28
#         std = 1.0 / 0.017429
#         tensor_image = torch.stack(batch_imgs, dim=0)
#         tensor_image = (tensor_image + mean) / std
#
#         return tensor_image.contiguous().numpy()
#
#
#     class A(object):
#         def __init__(self, **kwargs):
#             super(A, self).__init__()
#
#             for k, v in kwargs.items():
#                 setattr(self, k, v)
#
#         def __call__(self, x, fun="test"):
#             return getattr(self, fun)(x)
#
#
#     kwargs = dict(transform=transform)
#     a = A(**kwargs)
#
#     image = cv2.imread("lena.jpg")
#     images = [cv2.resize(image, (256, 256), image)]
#     print(a(images, "transform"))
