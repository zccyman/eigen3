# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/2/15 9:58
# @File     : test_asr.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import copy
import glob
import os

import numpy as np
import pandas as pd
from tqdm.contrib import tzip

from eval import Eval
from simulator import error_factory
from tools import ModelProcess
from simulator.perf_analysis import PerfAnalyzer, encryption_perf_data, analysis_perf_data

# def read_txt(file_name="feats_1.txt", delimiter=','):
#     data = []
#     with open(file_name) as in_f:
#         lines = in_f.readlines()

#         for line in lines:
#             for num in line.split(delimiter):
#                 if num == '\n' or num == "":
#                     continue
#                 data.append(np.float32(num))
#     return np.array(data)


def read_txt(filename, delimiter=' '):
    datas = []
    with open(filename) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            line = line.rstrip("\n").strip(delimiter).split(delimiter)
            if "[" in line:
                break
            for data in line:
                if data == "]":
                    continue
                datas.append(float(data))
            # print("test")

    return np.array(datas)


class VoiceEval(Eval):
    def __init__(self, **kwargs):
        super(VoiceEval, self).__init__(**kwargs)

        self.quan_dataset_path = kwargs['quan_dataset_path']
        self.eval_dataset_path = kwargs['eval_dataset_path']
        self.results_path = kwargs['results_path']
        self.model_path = kwargs['model_path']
        self.model_name = os.path.basename(self.model_path).split(".onnx")[0]
        self.process_args = kwargs['process_args']
        self.is_calc_error = kwargs['is_calc_error']
        self.acc_error = kwargs['acc_error']
        self.eval_mode = kwargs['eval_mode']
        self.fp_result = kwargs['fp_result']
        self.check_error = {}
        for m in self.process_args['error_metric']:
            self.check_error[m] = error_factory.get(m)()

        self.process = ModelProcess(**self.process_args)

        self.is_save_results = True
        if self.is_save_results:
            self.df_error = []

    def ASRORTInference(self, data, chunk_size, feat_size, in_size,
                        left_chunk_size, right_chunk_size):
        in_data = np.array(data, dtype=np.float32).reshape(-1, 43)
        n_block = (chunk_size - left_chunk_size)
        frame_size = in_data.shape[0] // n_block + 1

        input_datas = []
        for idx in range(frame_size):
            # start_c, end_c = chunk_size*idx, chunk_size*idx + chunk_size
            start_c, end_c = n_block * idx, n_block * idx + chunk_size
            end_c = np.clip(end_c, 0, in_data.shape[0])
            data = in_data[start_c:end_c]
            input_data = np.zeros((in_size, feat_size), dtype=np.float32)
            start = start_c - left_chunk_size
            end = end_c + right_chunk_size
            if start < 0 and end < in_data.shape[0] - 1:
                input_data[:-start] = in_data[0]
                input_data[-start:] = in_data[0:end]
            elif end <= in_data.shape[0] - 1:
                input_data = in_data[start:end]
            else:
                input_data[:in_data.shape[0] - start] = in_data[start:]
                input_data[in_data.shape[0] -
                           start:] = in_data[in_data.shape[0] - 1]

            input_datas.append(input_data)

        return input_datas

    def ASRInference(self,
                     fd_path,
                     feat_file,
                     chunk_size=50,
                     feat_size=43,
                     in_size=74,
                     output_size=2120,
                     left_chunk_size=12,
                     right_chunk_size=12,
                     delimiter=" "):
        data = read_txt(os.path.join(fd_path, feat_file),
                        delimiter=delimiter).reshape(-1, feat_size)
        input_datas = self.ASRORTInference(data, chunk_size, feat_size, in_size,
                                           left_chunk_size, right_chunk_size)
        return input_datas

    def get_single_quant(self,
                         fd_path,
                         feat_file,
                         output_file=None,
                         right_chunk_size=12,
                         delimiter=" "):
        chunk_size = 50
        feat_size = 43
        in_size = 74
        output_size = 2120
        left_chunk_size = right_chunk_size
        ref_out = None
        if output_file:
            ref_out = np.array(
                read_txt(os.path.join(fd_path, output_file),
                         delimiter=delimiter)).reshape(-1, output_size)
        input_datas = self.ASRInference(fd_path,
                                        feat_file,
                                        chunk_size,
                                        feat_size,
                                        in_size,
                                        output_size,
                                        left_chunk_size,
                                        right_chunk_size,
                                        delimiter=delimiter)

        return input_datas, ref_out

    def __call__(self):
        right_chunk_size = 12

        fd_path = self.quan_dataset_path
        voice_files = glob.glob(os.path.join(fd_path, "*_feats.txt"))

        input_datas = []
        for voice_file in voice_files[:1]:
            feat_file = os.path.basename(voice_file)
            input_data, ref_out = self.get_single_quant(
                fd_path,
                feat_file,
                output_file=None,
                right_chunk_size=right_chunk_size)
            input_datas.extend(input_data)

        if self.eval_mode == 'dataset':
            self.process.quantize(fd_path=input_datas, is_dataset=True)

        fd_path = self.eval_dataset_path
        voice_files = glob.glob(os.path.join(fd_path, "*_feat_cmvn.txt"))
        input_data_list = []
        ref_out_list = []
        voice_file_names = []
        for voice_file in voice_files[:1]:
            feat_file = os.path.basename(voice_file)
            output_file = feat_file.replace("_feat_cmvn.txt", "_nnet_out.txt")
            delimiter = " "
            input_data, ref_out = self.get_single_quant(
                fd_path,
                feat_file,
                output_file=output_file,
                right_chunk_size=right_chunk_size,
                delimiter=delimiter)
            input_data_list.append(input_data)
            ref_out_list.append(ref_out)
            voice_file_names.append(feat_file)

        # if self.eval_mode == 'dataset':
            # self.process.quantize(fd_path=input_data_list[0], is_dataset=True)

        error_dict = {}
        for idx, (input_datas, ref_out, voice_file_name) in enumerate(
                tzip(input_data_list,
                     ref_out_list,
                     voice_file_names,
                     postfix="simulation")):
            fouts, qouts = [], []
            for in_data in input_datas:
                onnx_outputs = self.process.post_quan.onnx_infer(
                    copy.deepcopy(in_data))
                if self.is_calc_error:
                    self.process.checkerror(in_data=in_data, acc_error=True)
                else:
                    self.process.dataflow(in_data=in_data,
                                          acc_error=True,
                                          onnx_outputs=onnx_outputs)
                outputs = self.process.get_outputs()
                qout = outputs['qout']["output.affine"][right_chunk_size:]
                fout = onnx_outputs["output.affine"][right_chunk_size:]
                qouts.append(qout)
                fouts.append(fout)
            qouts = np.array(qouts).reshape(-1, qouts[-1].shape[-1])
            qouts = qouts[:ref_out.shape[0]]
            fouts = np.array(fouts).reshape(-1, fouts[-1].shape[-1])
            fouts = fouts[:ref_out.shape[0]]

            if self.is_save_results:
                if not os.path.exists(f"{self.results_path}/qresults"):
                    os.makedirs(f"{self.results_path}/qresults")
                qresult_file = open(
                    os.path.join(f"{self.results_path}/qresults",
                                 voice_file_name), "w")                
                for qres in copy.deepcopy(qouts.reshape(-1)):
                    qresult_file.write(str(qres) + " ")
                qresult_file.close()

            for error_metric, error_func in self.check_error.items():
                if error_metric not in error_dict.keys():
                    error_dict[error_metric] = [error_func(qouts, ref_out)]
                    # error_dict[error_metric +
                    #            "_onnx-ref"] = [error_func(fouts, ref_out)]
                else:
                    error_dict[error_metric].extend(
                        [error_func(qouts, ref_out)])
                    # error_dict[error_metric + "_onnx-ref"].extend(
                    #     [error_func(fouts, ref_out)])

            if self.is_save_results:
                tmps = [voice_file_name]
                for id, (error_metric, error) in enumerate(error_dict.items()):
                    tmps.append(error[-1])
                self.df_error.append(tmps)

        if self.is_calc_error:
            self.collect_error_info()

        tmps = ["total"]
        for id, (error_metric, error) in enumerate(error_dict.items()):
            print(error_metric + ": ", np.array(error).mean())
            tmps.append(np.array(error).mean())
        self.df_error.append(tmps)

        if self.is_save_results:
            df_columns = ["file_name"]
            df_columns.extend([key for key in self.check_error.keys()])
            df = pd.DataFrame(self.df_error, columns=df_columns)
            if not os.path.exists(self.results_path):
                os.makedirs(self.results_path)            
            df.to_csv(f'{self.results_path}/{self.model_name}_error.csv',
                      index=False)


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

    def __call__(self, img):

        return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quan_dataset_path',
                        type=str,
                        default="/buffer/ssy/kaldi/workspace/ASR/date_set")
    parser.add_argument(
        '--eval_dataset_path',
        type=str,
        default="/buffer/ssy/kaldi/workspace/ASR/decode_set_complex") #decode_set_complex decode_set_without_softmax
    parser.add_argument(
        '--results_path',
        type=str,
        default="work_dir/asr_results")    
    parser.add_argument(
        '--model_path',
        type=str,
        default='trained_models/voice/mdl_0621_n_lda_n_softmax_adj.onnx')
    parser.add_argument('--input_size', type=list, default=[224, 224])
    parser.add_argument('--export_version', type=int, default=3)
    parser.add_argument('--fp_result', type=bool, default=True)
    parser.add_argument('--export', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)
    parser.add_argument('--log_level', type=int, default=30)
    parser.add_argument('--is_calc_error', type=bool,
                        default=False)  # whether to calculate each layer error
    parser.add_argument('--chip_model', type=str, default='5050')
    parser.add_argument('--mem_addr', type=str, default='psram')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    encrypt_flag = True
    if encrypt_flag:
        encryption_perf_data('perf_data')

    args = parse_args()
    if args.debug:
        eval_mode = 'single'
        acc_error = False
    else:
        eval_mode = 'dataset'
        acc_error = True

    kwargs_preprocess = {
        "img_mean": [123.675, 116.28, 103.53],
        "img_std": [58.395, 57.12, 57.375],
        'input_size': args.input_size
    }
    preprocess = PreProcess(**kwargs_preprocess)

    export_version = '' if args.export_version > 1 else '_v{}'.format(
        args.export_version)
    process_args = {
        'log_name': 'process.log',
        'log_level': args.log_level,
        'model_path': args.model_path,
        'parse_cfg': 'config/parse.py',
        'graph_cfg': 'config/graph.py',
        'base_quan_cfg': 'config/quantize.py',
        'quan_cfg': 'config/voice_quantize.py',
        'analysis_cfg': 'config/analysis.py',
        'export_cfg': 'config/export{}.py'.format(export_version),
        'offline_quan_mode': None,
        'offline_quan_tool': None,
        'quan_table_path': None,
        'fp_result': args.fp_result,
        'transform': preprocess,
        'simulation_level': 1,
        'is_ema': True,
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,
        'error_metric': ['L1', 'L2', 'Cosine'],  ## Cosine | L2 | L1
    }

    kwargs_voiceeval = {
        'log_dir': 'work_dir/eval',
        'log_name': 'test_voice.log',
        'log_level': args.log_level,
        'is_stdout': args.is_stdout,
        'quan_dataset_path': args.quan_dataset_path,
        'eval_dataset_path': args.eval_dataset_path,
        'results_path': args.results_path,
        # 'transform': preprocess,
        'process_args': process_args,
        'is_calc_error': args.is_calc_error,
        'acc_error': acc_error,
        'fp_result': args.fp_result,
        'eval_mode': eval_mode,  # single | dataset
        'model_path': args.model_path,
    }

    voice_eval = VoiceEval(**kwargs_voiceeval)
    voice_eval()
    voice_eval.export()

    perf_estimator = PerfAnalyzer(model_exporter = voice_eval.process.model_export, chip_model=args.chip_model, \
       ref_data_dir='perf_data/', mem_addr = args.mem_addr, encrypt_flag = encrypt_flag)
    perf_estimator(debug=True) 

    analysis_perf_data(pred_file="work_dir/perf_analysis_result.txt",
                       gt_file="ASR_model.txt",
                       res_file="result.txt")

    # python -m cProfile -o work_dir/log.profile tests/test_asr.py
    # snakeviz work_dir/log.profile
