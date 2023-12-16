# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/6/1 13:58
# @File     : test_voice.py
import pickle
import sys  # NOQA: E402
import os
sys.path.insert(0, os.getcwd())  # NOQA: E402
import torch
import argparse
import copy
import random
import pandas as pd
from tkinter import _flatten

import onnx
import onnxruntime as rt
import tqdm
from tqdm.contrib import tzip
from scipy.signal import windows

try:
    from export import serializeDataToInt8, writeFile
    from simulator import CosineSimiarity, L2Simiarity
    from tools import ModelProcess, OnnxruntimeInfer
    from utils import Object, add_layer_output_to_graph
except Exception:
    from onnx_converter.tools import ModelProcess, OnnxruntimeInfer
    from onnx_converter.utils import Object, add_layer_output_to_graph
    from onnx_converter.export import serializeDataToInt8, writeFile

import io

import librosa
import numpy as np
import soundfile


def wav_read(filename, tgt_fs=None):
    y, fs = soundfile.read(filename, dtype='float32')
    if tgt_fs is not None:
        if fs != tgt_fs:
            if fs != 16000:
                y = librosa.resample(y, tgt_fs, 16000)
                fs = tgt_fs
    return y, fs


def wav_write(data, fs, filename):
    max_value_int16 = (1 << 15) - 1
    data *= max_value_int16
    soundfile.write(filename, data.astype(np.int16), fs, subtype='PCM_16',
                    format='WAV')


def generate_random_str(randomlength=16, postfix='.jpg'):
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return random_str + postfix


class BaseNetEval(Object):
    def __init__(self):
        super(BaseNetEval, self).__init__()

    def get_onnx_infer(self, model_process, process_args):
        output_names, input_names = model_process.get_output_names(), model_process.get_input_names()
        onnxinferargs = copy.deepcopy(process_args)
        onnxinferargs.update(out_names=output_names, input_names=input_names)
        onnxinfer = OnnxruntimeInfer(**onnxinferargs)
        return onnxinfer

    @staticmethod
    def setting_graph_output(model_path, modelprocess):
        model = onnx.load(model_path)
        out_names = list(_flatten(modelprocess.get_output_names()))
        in_names = list(_flatten(modelprocess.get_input_names()))
        output_names_ = []
        for output in out_names:
            if output in in_names:
                continue
            output_names_.append(output)
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        return rt.InferenceSession(model.SerializeToString()), output_names_

    @staticmethod
    def list2dict(outputs, names):
        results = {}
        for output, name in zip(outputs, names):
            results[name] = output

        return results

    @staticmethod
    def transfer_voice(audio, block_len, block_shift, inputs):
        out = np.zeros((len(audio)))
        out_buffer = np.zeros((block_len))
        for idx, in_data in enumerate(inputs):
            n, c = in_data.shape[:2]
            out_block = in_data.reshape(n, c)
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = np.zeros((block_shift))
            out_buffer += np.squeeze(out_block)
            out[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]
        return out


class NetEval(BaseNetEval):
    def __init__(self, **kwargs):
        super(NetEval, self).__init__()
        self.base_quan_cfg = kwargs['process_args']['base_quan_cfg']
        self.quan_cfg = kwargs['process_args']['quan_cfg']
        self.is_stdout = kwargs['process_args']['is_stdout']
        self.log_dir = kwargs['log_dir']
        self.log_name = kwargs["log_name"]
        # self.logger = self.get_log(log_name=self.log_name, stdout=self.is_stdout)
        self.model_path = kwargs['process_args']['model_path']
        self.input_names = kwargs['input_names']
        self.export = kwargs['export']

        self.input_size = kwargs['input_size']
        self.dataset_path = kwargs['dataset_path']
        self.eval_fisrt_frame = kwargs['eval_fisrt_frame']
        self.process_args_1 = copy.deepcopy(kwargs['process_args'])
        self.process_args_2 = copy.deepcopy(kwargs['process_args'])
        self.process_args_1.update(dict(log_name= self.log_name[0],
                                        model_path=self.model_path[0],
                                        input_names=self.input_names[0]))
        self.process_args_2.update(dict(log_name=self.log_name[1],
                                        model_path=self.model_path[1],
                                        input_names=self.input_names[1]))

        self.process_1 = ModelProcess(**self.process_args_1)
        self.process_2 = ModelProcess(**self.process_args_2)
        # self.onnxinfer_1 = rt.InferenceSession(self.model_path[0])
        # self.onnxinfer_2 = rt.InferenceSession(self.model_path[1])

    def get_dataset(self, audio, num_blocks, block_len=512, block_shift=128):
        out = np.zeros((len(audio)))
        in_buffer = np.zeros((block_len))
        out_buffer = np.zeros((block_len))
        lstm_map_1 = {'h1_in': '38', 'c1_in': '39',
                      'h2_in': '63', 'c2_in': '64'}
        result_name_1 = 'y1'
        lstm_map_2 = {'h1_in': '54', 'c1_in': '55',
                      'h2_in': '79', 'c2_in': '80'}
        result_name_2 = 'y'

        onnxinfer_1 = rt.InferenceSession(self.model_path[0])
        onnxinfer_2 = rt.InferenceSession(self.model_path[1])

        model_input_names_1 = [inp.name for inp in onnxinfer_1.get_inputs()]
        model_inputs_1 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in onnxinfer_1.get_inputs()}
        model_input_names_2 = [inp.name for inp in onnxinfer_2.get_inputs()]

        # preallocate input
        model_inputs_2 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in onnxinfer_2.get_inputs()}
        quan_datasets_1, quan_datasets_2 = [], []
        fft_phases, fout_1, fout_2 = [], [], []
        for idx in tqdm.tqdm(range(num_blocks)):
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]
            in_block = np.expand_dims(in_buffer, axis=0).astype('float32')

            in_block_fft = np.fft.rfft(in_buffer)
            in_mag = np.abs(in_block_fft)
            in_phase = np.angle(in_block_fft)

            # reshape magnitude to input dimensions
            in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
            in_mag = (in_mag + 1.0e-7).astype(np.float32)
            
            # set block to input
            model_inputs_1[model_input_names_1[0]] = in_mag
            quan_datasets_1.append(copy.deepcopy(model_inputs_1))
            model_outputs_1 = onnxinfer_1.run(None, model_inputs_1)
            estimated_mag = model_outputs_1[0]

            # set out states back to input
            model_inputs_1["h1_in"][0] = model_outputs_1[1]
            model_inputs_1["c1_in"][0] = model_outputs_1[2]
            model_inputs_1["h2_in"][0] = model_outputs_1[3]
            model_inputs_1["c2_in"][0] = model_outputs_1[4]

            # calculate the ifft
            fft_phases.append(np.exp(1j * in_phase))
            estimated_complex = estimated_mag * np.exp(1j * in_phase)
            estimated_block = np.fft.irfft(estimated_complex)

            # reshape the time domain block
            estimated_block = np.reshape(estimated_block, (1, -1, 1)).astype('float32')

            # set tensors to the second block
            # interpreter_2.set_tensor(input_details_1[1]['index'], states_2)
            n, c = estimated_block.shape[:2]
            model_inputs_2[model_input_names_2[0]] = estimated_block#.reshape(n, c)
            quan_datasets_2.append(copy.deepcopy(model_inputs_2))

            # run calculation
            model_outputs_2 = onnxinfer_2.run(None, model_inputs_2)

            # get output
            out_block = model_outputs_2[0]

            # set out states back to input
            model_inputs_2["h1_in"][0] = model_outputs_2[1]
            model_inputs_2["c1_in"][0] = model_outputs_2[2]
            model_inputs_2["h2_in"][0] = model_outputs_2[3]
            model_inputs_2["c2_in"][0] = model_outputs_2[4]

            # shift values and write to buffer
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = np.zeros((block_shift))
            out_buffer += np.squeeze(out_block)

            # print(idx, np.abs(out_buffer).sum())
            # write block to output file
            out[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]
            fout_1.append(copy.deepcopy(model_outputs_1[0]))
            fout_2.append(copy.deepcopy(model_outputs_2[0]))

        quan_datasets_1.append(model_inputs_1)
        quan_datasets_2.append(model_inputs_2)

        return quan_datasets_1, quan_datasets_2, fft_phases, (out, np.array(fout_1), np.array(fout_2))

    def quant_vioce(self, audio, num_blocks, quan_datasets_1, quan_datasets_2, fft_phases, block_len=512,
                    block_shift=128):

        out = np.zeros((len(audio)))
        out_buffer = np.zeros((block_len))

        # quantize two submodules
        self.process_1.quantize(quan_datasets_1, is_dataset=True)
        self.process_2.quantize(quan_datasets_2, is_dataset=True)
        outputs_1, outputs_2 = None, None
        '''
        lstm_map_1 = {'h1_in': '38', 'c1_in': '39',
                      'h2_in': '63', 'c2_in': '64'}
        result_name_1 = 'y1'
        lstm_map_2 = {'h1_in': '54', 'c1_in': '55',
                      'h2_in': '79', 'c2_in': '80'}
        '''
        idx = 0
        qout_1, qout_2 = [], []
        for data_1, data_2, fft_phase in tzip(quan_datasets_1, quan_datasets_2, fft_phases):
            # if outputs_1:
            #     data_1["h1_in"][0] = outputs_1['38']
            #     data_1["c1_in"][0] = outputs_1['39']
            #     data_1["h2_in"][0] = outputs_1['63']
            #     data_1["c2_in"][0] = outputs_1['64']
            # if outputs_2:process_1.dataflow
            #     data_2["h1_in"][0] = outputs_2['54']
            #     data_2["c1_in"][0] = outputs_2['55']
            #     data_2["h2_in"][0] = outputs_2['79']
            #     data_2["c2_in"][0] = outputs_2['80']
            # onnxinfer_1, out_names_1 = self.setting_graph_output(self.model_path[0], self.process_1)
            # model_outputs_1 = onnxinfer_1.run(output_names=out_names_1, input_feed=copy.deepcopy(data_1))
            # true_outputs_1 = self.list2dict(model_outputs_1, out_names_1)
            self.process_1.checkerror(data_1, acc_error=True)
            # self.process_1.dataflow(data_1, acc_error=True, onnx_outputs=None)
            outputs_1 = self.process_1.get_outputs()['qout']

            # calculate the ifft
            estimated_mag = outputs_1['76']
            estimated_complex = estimated_mag * fft_phase
            
            estimated_complex_in = np.concatenate([
                estimated_complex.real.squeeze()[:, None], 
                estimated_complex.imag.squeeze()[:, None]], axis=1)
            estimated_complex_in = estimated_complex_in.reshape(-1)
            irfft_in = serializeDataToInt8(copy.deepcopy(estimated_complex_in.astype(np.float32)))
            writeFile(irfft_in, 'work_dir/irfft_in.b', mode="wb")            
            
            estimated_block = np.fft.irfft(estimated_complex)
             
            estimated_block_out = estimated_block.squeeze()       
            irfft_out = serializeDataToInt8(copy.deepcopy(estimated_block_out.astype(np.float32)))
            writeFile(irfft_out, 'work_dir/irfft_out.b', mode="wb")
        
            # reshape the time domain block
            estimated_block = np.reshape(estimated_block, (1, -1, 1)).astype('float32')

            # set tensors to the second block
            # interpreter_2.set_tensor(input_details_1[1]['index'], states_2)
            n, c = estimated_block.shape[:2]
            data_2['y1'] = estimated_block.reshape(n, c)
            # onnxinfer_2, out_names_2 = self.setting_graph_output(self.model_path[1], self.process_2)
            # model_outputs_2 = onnxinfer_2.run(output_names=out_names_2, input_feed=data_2)
            # true_outputs_2 = self.list2dict(model_outputs_2, out_names_2)
            self.process_2.dataflow(data_2, acc_error=True, onnx_outputs=None)
            outputs_2 = self.process_2.get_outputs()['qout']

            # shift values and write to buffer
            n, c = outputs_2['y'].shape[:2]
            out_block = outputs_2['y'].reshape(n, c)
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = np.zeros((block_shift))
            out_buffer += np.squeeze(out_block)
            qout_1.append(copy.deepcopy(outputs_1['76']))
            # qout_1.append(copy.deepcopy(model_outputs_1[-1]))
            qout_2.append(copy.deepcopy(outputs_2['y']))

            # print(idx, np.abs(out_buffer).sum())
            # write block to output file
            out[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]
            idx += 1

            if self.eval_fisrt_frame:
                break
        
            # break

        return (out, np.array(qout_1), np.array(qout_2))

    def __call__(self):

        wav_in = self.dataset_path
        print('==> read wav from: ', wav_in)
        audio, fs = wav_read(wav_in, tgt_fs=16000)
        audio_in = serializeDataToInt8(copy.deepcopy(audio))
        writeFile(audio_in, 'work_dir/audio_in.b', mode="wb")
        print('==> audio len: {} secs'.format(len(audio) / fs))
        block_len = 512
        block_shift = 128
        # calculate number of blocks
        num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift

        quan_datasets_1, quan_datasets_2, fft_phases, fout = self.get_dataset(audio, num_blocks, block_len, block_shift)

        # quantize two submodules
        self.process_1.quantize(quan_datasets_1, is_dataset=True)
        self.process_2.quantize(quan_datasets_2, is_dataset=True)

        # create buffer
        quant_out, qout_1, qout_2 = self.quant_vioce(audio, num_blocks, quan_datasets_1, quan_datasets_2, fft_phases, block_len, block_shift)
        fout, fout_1, fout_2 = fout
        l2, cosine = L2Simiarity(), CosineSimiarity()
        if not self.eval_fisrt_frame:
            print('out buffer l2 error is: {}, cosine error is: {}'.format(l2(fout, quant_out), cosine(fout, quant_out)))
            print('qout_1 l2 error is: {}, cosine error is: {}'.format(l2(fout_1, qout_1), cosine(fout_1, qout_1)))
            print('qout_2 l2 error is: {}, cosine error is: {}'.format(l2(fout_2, qout_2), cosine(fout_2, qout_2)))
            wav_out = wav_in.replace('.wav', '_quant.wav')
            print('==> save wav to: ', wav_out)
            wav_write(quant_out, 16000, wav_out)
            wav_write(fout, 16000, wav_in.replace('.wav', '_fp.wav'))

            audio_out = serializeDataToInt8(copy.deepcopy(quant_out.astype(np.float32)))
            writeFile(audio_out, 'work_dir/audio_out.b', mode="wb")

        if self.export:
            self.process_1.export()
            os.system('cp -rf work_dir/export.log work_dir/export_1.log')
            os.system('cp -rf work_dir/model.c work_dir/model_1.c')
            os.system('cp -rf work_dir/weights work_dir/weights_1')
            os.system('rm -rf work_dir/weights/*')
            os.system('cp -rf work_dir/test_vis.mmd.simplify.pdf work_dir/test_vis.mmd.simplify_1.pdf')
            self.process_2.export()
            os.system('cp -rf work_dir/export.log work_dir/export_2.log')
            os.system('cp -rf work_dir/model.c work_dir/model_2.c')
            os.system('cp -rf work_dir/weights work_dir/weights_2')
            os.system('rm -rf work_dir/weights/*')
            os.system('cp -rf work_dir/test_vis.mmd.simplify.pdf work_dir/test_vis.mmd.simplify_2.pdf')
        print('out buffer l2 error is: {}, cosine error is: {}'.format(l2(fout, quant_out), cosine(fout, quant_out)))
        print('qout_1 l2 error is: {}, cosine error is: {}'.format(l2(fout_1, qout_1), cosine(fout_1, qout_1)))
        print('qout_2 l2 error is: {}, cosine error is: {}'.format(l2(fout_2, qout_2), cosine(fout_2, qout_2)))
        wav_out = wav_in.replace('.wav', '_quant.wav')
        print('==> save wav to: ', wav_out)
        wav_write(quant_out, 16000, wav_out)
        wav_write(fout, 16000, wav_in.replace('.wav', '_fp.wav'))

        audio_out = serializeDataToInt8(copy.deepcopy(quant_out.astype(np.float32)))
        writeFile(audio_out, 'work_dir/audio_out.b', mode="wb")
        # import soundfile as sf
        # from pypesq import pesq
        # ref_gt, sr_gt = sf.read(wav_in.replace('_in.wav', '_gt.wav'))
        # ref_fp, sr_fp = sf.read(wav_in.replace('.wav', '_fp.wav'))
        # ref_quant, sr_quant = sf.read(wav_in.replace('.wav', '_quant.wav'))
        #
        # score_fp = pesq(ref_fp, ref_gt, sr_gt)
        # score_quant = pesq(ref_quant, ref_gt, sr_gt)
        # print(score_fp, score_quant)

        # self.process_1.export()
        # os.system('cp -rf work_dir/export.log work_dir/export_1.log')
        # os.system('cp -rf work_dir/model.c work_dir/model_1.c')
        # os.system('cp -rf work_dir/weights work_dir/weights_1')
        # os.system('rm -rf work_dir/weights/*')
        # os.system('cp -rf work_dir/test_vis.mmd.simplify.pdf work_dir/test_vis.mmd.simplify_1.pdf')
        # self.process_2.export()
        # os.system('cp -rf work_dir/export.log work_dir/export_2.log')
        # os.system('cp -rf work_dir/model.c work_dir/model_2.c')
        # os.system('cp -rf work_dir/weights work_dir/weights_2')
        # os.system('rm -rf work_dir/weights/*')
        # os.system('cp -rf work_dir/test_vis.mmd.simplify.pdf work_dir/test_vis.mmd.simplify_2.pdf')


class NetEvalOneModel(BaseNetEval):
    def __init__(self, **kwargs):
        super(NetEvalOneModel, self).__init__()
        self.base_quan_cfg = kwargs['process_args']['base_quan_cfg']
        self.quan_cfg = kwargs['process_args']['quan_cfg']
        self.is_stdout = kwargs['process_args']['is_stdout']
        self.log_dir = kwargs['log_dir']
        self.log_name = kwargs["log_name"]
        self.print_frame = kwargs['print_frame']
        # self.logger = self.get_log(log_name=self.log_name, stdout=self.is_stdout)
        self.model_path = kwargs['process_args']['model_path']
        self.input_names = kwargs['input_names']
        check_error = kwargs.get("check_error", 0)
        error_analyzer = True if check_error == 2 else False

        self.input_size = kwargs['input_size']
        self.dataset_path = kwargs['dataset_path']
        self.eval_path = kwargs['eval_path']
        self.process_args = copy.deepcopy(kwargs['process_args'])
        self.process_args.update(dict(log_name=self.log_name[0],
                                      model_path=self.model_path[0],
                                      input_names=self.input_names[0],
                                      error_analyzer=error_analyzer,
                                      is_simplify=False))

        self.process = ModelProcess(**self.process_args)
        self.result_name = kwargs['output_names'][0][0]
        self.block_len = kwargs['block_len'] # 512
        self.block_shift = kwargs['block_shift'] # 160   
        self.is_use_hn = kwargs["is_use_hn"]      
        self.in_buffer = np.zeros((self.block_len))
        self.out_buffer = np.zeros((self.block_len))
        self.input_feed, self.output_names = dict(), list()
        self.onnxinfer = None
        self.resume_path = kwargs.get("resume_path", "work_dir/resume_int8")
        # self.eval_fisrt_frame = kwargs.get("eval_fisrt_frame", False)
        self.input_feed = {}
        
    def rename_file_name(self, wav_file):
        model_name = os.path.basename(self.model_path[0]).split('.onnx')[0]
        file_path, file_name = os.path.split(wav_file)
        file_name = file_name.split('.wav')[0]
        if "fp_voice" in wav_file:
            file_name = file_name.replace("fp_", "")
            file_name += "_fp"
            file_path = file_path.replace("/fp_voice", "")
        elif "quant_voice" in wav_file:
            file_name = file_name.replace("quant_", "")
            file_name += "_quant"
            file_path = file_path.replace("/quant_voice", "")
            
        if self.is_use_hn:
            file_name += "_hn"
            
        file_name += "_{}.wav".format(self.block_shift // 16)
        
        out_path = os.path.join(file_path, model_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_wav_file = os.path.join(out_path, file_name)
        
        return out_wav_file
    
    def get_dataset(self, wav_in, block_len=512, block_shift=128, reuse_lstm_state=False, is_reverse_state=False):
        if os.path.isdir(wav_in):
            import glob
            wavs = glob.glob(wav_in + '/*.wav')#[:1]#[:6]
        else:
            wavs = [wav_in]
        quan_datasets = []
        if not self.onnxinfer:
            self.onnxinfer = rt.InferenceSession(self.model_path[0], providers=["CPUExecutionProvider"])

        model_input_names = [inp.name for inp in self.onnxinfer.get_inputs()]
        if self.input_feed == {}:
            self.input_feed = {
                inp.name: np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                    dtype=np.float32)
                for inp in self.onnxinfer.get_inputs()}
        if not reuse_lstm_state:
            self.input_feed = {
                inp.name: np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                    dtype=np.float32)
                for inp in self.onnxinfer.get_inputs()}
        # reuse_lstm_state = False
        self.output_names = [item.name for item in self.onnxinfer.get_outputs()]
        for wav in wavs:
            audio, fs = wav_read(wav, tgt_fs=16000)
            num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
            
            len_orig = len(audio)
            ##### 原始音频后方补零 ######
            pad_size = block_len - block_shift
            zero_pad = np.zeros(pad_size)
            audio = np.concatenate((audio, zero_pad, zero_pad), axis=0)
                
            out = np.zeros((len(audio)))
            in_buffer = np.zeros((block_len))
            out_buffer = np.zeros((block_len))
            if not reuse_lstm_state:
                self.input_feed = {
                    inp.name: np.zeros(
                        [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                        dtype=np.float32)
                    for inp in self.onnxinfer.get_inputs()}
            # preallocate input

            fout = []
            win = windows.hann(block_len)
            for idx in tqdm.tqdm(range(num_blocks)):
                in_buffer[:-block_shift] = in_buffer[block_shift:]
                in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]
                in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
                # in_buffer = torch.sigmoid(torch.from_numpy(in_buffer)).numpy()
                if self.is_use_hn:
                    in_block_fft = np.fft.rfft(in_buffer*np.sqrt(win))
                else:
                    in_block_fft = np.fft.rfft(in_buffer)
                
                in_mag = np.abs(in_block_fft)
                in_phase = np.angle(in_block_fft)

                # reshape magnitude to input dimensions
                in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
                in_mag = (in_mag + 1.0e-7).astype(np.float32)
                
                # set block to input
                self.input_feed[model_input_names[0]] = in_mag
                quan_datasets.append(copy.deepcopy(self.input_feed))
                
                model_outputs = self.onnxinfer.run(self.output_names, self.input_feed)
                estimated_mag = model_outputs[0]
                if is_reverse_state:
                    # set out states back to input
                    self.input_feed["h1_in"] = model_outputs[7]
                    self.input_feed["c1_in"] = model_outputs[8]
                    self.input_feed["h2_in"] = model_outputs[5]
                    self.input_feed["c2_in"] = model_outputs[6]
                    self.input_feed["h3_in"] = model_outputs[3]
                    self.input_feed["c3_in"] = model_outputs[4]
                    self.input_feed["h4_in"] = model_outputs[1]
                    self.input_feed["c4_in"] = model_outputs[2]
                else:                
                    self.input_feed["h1_in"] = model_outputs[1]
                    self.input_feed["c1_in"] = model_outputs[2]
                    self.input_feed["h2_in"] = model_outputs[3]
                    self.input_feed["c2_in"] = model_outputs[4]
                    self.input_feed["h3_in"] = model_outputs[5]
                    self.input_feed["c3_in"] = model_outputs[6]
                    self.input_feed["h4_in"] = model_outputs[7]
                    self.input_feed["c4_in"] = model_outputs[8]

                # calculate the ifft
                # fft_phases.append(np.exp(1j * in_phase))
                estimated_complex = estimated_mag * np.exp(1j * in_phase)
                estimated_block = np.fft.irfft(estimated_complex)

                # shift values and write to buffer
                out_buffer[:-block_shift] = out_buffer[block_shift:]
                out_buffer[-block_shift:] = np.zeros((block_shift))
                out_buffer += np.squeeze(estimated_block)

                # write block to output file
                # if self.is_use_hn:
                #     out[idx * block_shift:(idx * block_shift) + block_shift] = np.float32(out_buffer[:block_shift]/(0.67))
                # else:
                #     out[idx * block_shift:(idx * block_shift) + block_len] = np.float32(out_buffer[:block_len])
                out[idx * block_shift:(idx * block_shift) + block_shift] = np.float32(out_buffer[:block_shift])

                fout.append(copy.deepcopy(model_outputs[0]))
            ###### 截取出降噪后目标音频长度 ####
            out = out[pad_size: pad_size + len_orig]
        # quan_datasets.append(model_inputs)

        return quan_datasets

    @staticmethod
    def get_pyseq(quant_file, fp_file, gt_file):
        import soundfile as sf
        from pypesq import pesq
        ref_gt, sr_gt = sf.read(gt_file)
        ref_fp, sr_fp = sf.read(fp_file)
        ref_quant, sr_quant = sf.read(quant_file)

        score_fp = pesq(ref_fp, ref_gt, sr_gt)
        score_quant = pesq(ref_quant, ref_gt, sr_gt)
        return score_fp, score_quant

    def fp_infer(self, audio, num_blocks, block_len=512, block_shift=128, reuse_lstm_state=False, is_reverse_state=False):
        len_orig = len(audio)
        ##### 原始音频后方补零 ######
        pad_size = block_len - block_shift
        zero_pad = np.zeros(pad_size)
        audio = np.concatenate((audio, zero_pad, zero_pad), axis=0)
            
        out = np.zeros((len(audio)))
        in_buffer = np.zeros((block_len))
        out_buffer = np.zeros((block_len))

        onnxinfer = rt.InferenceSession(self.model_path[0], providers=rt.get_available_providers())

        model_input_names_1 = [inp.name for inp in onnxinfer.get_inputs()]
        if not reuse_lstm_state:
            self.input_feed = {
                inp.name: np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                    dtype=np.float32)
                for inp in onnxinfer.get_inputs()}

        # preallocate input
        # quan_datasets = []
        # fout = []
        win = windows.hann(block_len)
        for idx in tqdm.tqdm(range(num_blocks), postfix="fp32"):
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]
            in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
            # in_buffer = torch.sigmoid(torch.from_numpy(in_buffer)).numpy()
            if self.is_use_hn:
                in_block_fft = np.fft.rfft(in_buffer*np.sqrt(win))
            else:
                in_block_fft = np.fft.rfft(in_buffer)
            in_mag = np.abs(in_block_fft)
            in_phase = np.angle(in_block_fft)

            # reshape magnitude to input dimensions
            in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
            in_mag = (in_mag + 1.0e-7).astype(np.float32)
            
            # set block to input
            self.input_feed[model_input_names_1[0]] = in_mag
            # quan_datasets.append(copy.deepcopy(self.input_feed))
            output_names = [item.name for item in onnxinfer.get_outputs()]
            model_outputs = onnxinfer.run(output_names, self.input_feed)
            estimated_mag = model_outputs[0]
            
            if is_reverse_state:
                # set out states back to input
                self.input_feed["h1_in"] = model_outputs[7]
                self.input_feed["c1_in"] = model_outputs[8]
                self.input_feed["h2_in"] = model_outputs[5]
                self.input_feed["c2_in"] = model_outputs[6]
                self.input_feed["h3_in"] = model_outputs[3]
                self.input_feed["c3_in"] = model_outputs[4]
                self.input_feed["h4_in"] = model_outputs[1]
                self.input_feed["c4_in"] = model_outputs[2]
            else:
                self.input_feed["h1_in"] = model_outputs[1]
                self.input_feed["c1_in"] = model_outputs[2]
                self.input_feed["h2_in"] = model_outputs[3]
                self.input_feed["c2_in"] = model_outputs[4]
                self.input_feed["h3_in"] = model_outputs[5]
                self.input_feed["c3_in"] = model_outputs[6]
                self.input_feed["h4_in"] = model_outputs[7]
                self.input_feed["c4_in"] = model_outputs[8]
                
            # calculate the ifft
            # fft_phases.append(np.exp(1j * in_phase))
            estimated_complex = estimated_mag * np.exp(1j * in_phase)
            estimated_block = np.fft.irfft(estimated_complex)

            # shift values and write to buffer
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = np.zeros((block_shift))
            out_buffer += np.squeeze(estimated_block)

            # write block to output file
            # if self.is_use_hn:
            #     out[idx * block_shift:(idx * block_shift) + block_shift] = np.float32(out_buffer[:block_shift]/(0.67))
            # else:
            #     out[idx * block_shift:(idx * block_shift) + block_shift] = np.float32(out_buffer[:block_shift])
            out[idx * block_shift:(idx * block_shift) + block_len] = np.float32(out_buffer[:block_len])
        
        ###### 截取出降噪后目标音频长度 ####
        out = out[pad_size: pad_size + len_orig]
    
        return out

    def inference_with_error(self, audio, num_blocks, block_len=512, block_shift=128):
        len_orig = len(audio)
        ##### 原始音频后方补零 ######
        pad_size = block_len - block_shift
        zero_pad = np.zeros(pad_size)
        audio = np.concatenate((audio, zero_pad, zero_pad), axis=0)
                
        out = np.zeros((len(audio)))
        in_buffer = np.zeros((block_len))
        out_buffer = np.zeros((block_len))
        win = windows.hann(block_len)
        # for idx in tqdm.tqdm(range(num_blocks)):
        for idx in range(num_blocks):
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]
            # in_buffer = torch.sigmoid(torch.from_numpy(in_buffer)).numpy()
            if self.is_use_hn:
                in_block_fft = np.fft.rfft(in_buffer*np.sqrt(win))
            else:
                in_block_fft = np.fft.rfft(in_buffer)

            in_mag = np.abs(in_block_fft)
            in_phase = np.angle(in_block_fft)

            # reshape magnitude to input dimensions
            in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
            in_mag = (in_mag + 1.0e-7).astype(np.float32)
            # if idx == 3084:
            #     print()
            self.process_1.checkerror(in_mag, acc_error=True, onnx_outputs=None)
            # self.process.dataflow(in_mag, acc_error=True, onnx_outputs=None)
            outputs_1 = self.process.get_outputs()['qout']

            # calculate the ifft
            estimated_mag = outputs_1[self.result_name]
            estimated_complex = estimated_mag * np.exp(1j * in_phase)
            estimated_block = np.fft.irfft(estimated_complex)

            # shift values and write to buffer
            # n, c = estimated_block.shape[:2]
            out_block = estimated_block.astype(np.float32)#.reshape(n, c)
            out_buffer[:-block_shift] = out_buffer[block_shift:]
            out_buffer[-block_shift:] = np.zeros((block_shift))
            out_buffer += np.squeeze(out_block)
            # qout_1.append(copy.deepcopy(outputs_1[self.result_name]))

            # print(idx, np.abs(out_buffer).sum())
            # write block to output file
            if self.is_use_hn:
                out[idx * block_shift:(idx * block_shift) + block_shift] = np.float32(out_buffer[:block_shift]/(0.67))
            else:
                out[idx * block_shift:(idx * block_shift) + block_shift] = np.float32(out_buffer[:block_shift])
                # out[idx * block_shift:(idx * block_shift) + block_len] = np.float32(out_buffer[:block_len])
            # out_buffer_int8 = serializeDataToInt8(copy.deepcopy(out_buffer[:block_shift]).astype(np.float32))
            # fft_dir = '/home/shiqing/Downloads/onnx_converter/trained_models/test_voice/fft'
            # writeFile(out_buffer_int8, '{}/{}.b'.format(fft_dir, str(idx)), mode="wb")
            # if idx == 3084:
            #     break
            #if self.print_frame:
            #    self.process.export_frame(str(idx))
            
        ###### 截取出降噪后目标音频长度 ####
        out = out[pad_size: pad_size + len_orig]
                    
        return out
    
    def quant_vioce(self, audio, num_blocks, block_len=512, block_shift=128, check_error: int=0):
        len_orig = len(audio)
        ##### 原始音频后方补零 ######
        pad_size = block_len - block_shift
        zero_pad = np.zeros(pad_size)
        audio = np.concatenate((audio, zero_pad, zero_pad), axis=0)
                
        out = np.zeros((len(audio)))
        in_buffer = np.zeros((block_len))
        out_buffer = np.zeros((block_len))
        win = windows.hann(block_len)
        # for idx in tqdm.tqdm(range(num_blocks)):
        for idx in range(num_blocks):
            in_buffer[:-block_shift] = in_buffer[block_shift:]
            in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]
            # in_buffer = torch.sigmoid(torch.from_numpy(in_buffer)).numpy()
            if self.is_use_hn:
                in_block_fft = np.fft.rfft(in_buffer*np.sqrt(win))
            else:
                in_block_fft = np.fft.rfft(in_buffer)

            in_mag = np.abs(in_block_fft)
            in_phase = np.angle(in_block_fft)

            # reshape magnitude to input dimensions
            in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
            in_mag = (in_mag + 1.0e-7).astype(np.float32)
            # if idx == 3084:
            #     print()
            out[idx * block_shift:(idx * block_shift) + block_shift] = self.inference(in_mag, check_error)
            if self.print_frame and idx < 50:
                self.process.export_frame(str(idx))
            
        ###### 截取出降噪后目标音频长度 ####
        out = out[pad_size: pad_size + len_orig]
                    
        return out

    def quant_vioce_api(self, audio, num_blocks, block_len=512, block_shift=128, check_error:int=0, reuse_lstm_state:bool=False):
        len_orig = len(audio)
        ##### 原始音频后方补零 ######
        pad_size = block_len - block_shift
        zero_pad = np.zeros(pad_size)
        audio = np.concatenate((audio, zero_pad, zero_pad), axis=0)
                
        out = np.zeros((len(audio)))
        # self.reset_session()
        for idx in tqdm.tqdm(range(num_blocks), postfix="quant"):
        # for idx in range(num_blocks):
            # if not reuse_lstm_state and idx % 80000 == 0 :
            #     self.process.reset("lstm")
            in_data = audio[idx * block_shift:(idx * block_shift) + block_shift]
            out[idx * block_shift:(idx * block_shift) + block_shift] = self.inference(in_data, check_error)
            if self.print_frame and idx < 50:
                self.process.export_frame(str(idx))
                
        ###### 截取出降噪后目标音频长度 ####
        out = out[pad_size: pad_size + len_orig]
                        
        return out

    def quant_dataset(self, is_dir, block_len=512, block_shift=128, check_error: int=0, reuse_lstm_state: bool=False, is_reverse_state: bool=False):
        if is_dir:
            import glob
            wavs = glob.glob(self.eval_path + '/*.wav')[:1]#[:3]
        else:
            wavs = copy.deepcopy(self.eval_path)
        df_columns = ["file_name", "noise_name", "amp", "sn", "score_fp", "score_quant"]
        
        results_path = copy.deepcopy(self.eval_path) #"./trained_models/test_voice"
        if not os.path.exists(results_path):
            os.makedirs(results_path)            
        if not os.path.exists(results_path.replace("in_voice", "quant_voice")):
            os.makedirs(results_path.replace("in_voice", "quant_voice"))
        if not os.path.exists(results_path.replace("in_voice", "fp_voice")):
            os.makedirs(results_path.replace("in_voice", "fp_voice"))

        infos = []

        for wav in wavs:
            if not reuse_lstm_state:
                self.process.reset("lstm")
            # if "s5" not in wav and "s2" not in wav:
            # # if "s5" not in wav:
            #     continue
            audio, fs = wav_read(wav, tgt_fs=16000)
            num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
            quant_out = self.quant_vioce_api(audio, num_blocks, block_len, block_shift, check_error, reuse_lstm_state)#[80000:]
            eval_out = self.fp_infer(audio, num_blocks, block_len, block_shift, reuse_lstm_state, is_reverse_state)#[80000:]
            # print('#############################################################')
            # print('#############################################################')
            # print('#############################################################')
            # print('#############################################################')
            # for layer in self.process.simulation.get_layers():
            #     if layer.get_layer_type() == "lstm":
            #         op_instance = layer.get_ops_instance()
            #         log_info = "layer of {}, c of hidden state max is: {}, min is: {}, mean is: {}".\
            #             format(layer.get_layer_name(), op_instance.c.max(), op_instance.c.min(), op_instance.c.mean())
            #         print(log_info)
            # print('#############################################################')
            # print('#############################################################')
            # print('#############################################################')
            # print('#############################################################')
            # fout, fout_1 = fout
            # self.in_buffer = np.zeros_like(self.in_buffer)
            # self.out_buffer = np.zeros_like(self.out_buffer)
            l2, cosine = L2Simiarity()(eval_out, quant_out), CosineSimiarity()(eval_out, quant_out)
            print('{} out buffer l2 error is: {}, cosine error is: {}'.format(wav, l2, cosine))
            qwav_out = wav.replace('in_voice', 'quant_voice')
            fwav_out = wav.replace('in_voice', 'fp_voice')
            qwav_out = os.path.join(os.path.dirname(qwav_out), "quant_"+os.path.basename(qwav_out))
            fwav_out = os.path.join(os.path.dirname(fwav_out), "fp_"+os.path.basename(fwav_out))
            qwav_out = self.rename_file_name(qwav_out)
            fwav_out = self.rename_file_name(fwav_out)
            wav_write(quant_out, 16000, qwav_out)
            wav_write(eval_out, 16000, fwav_out)
            gt_path = os.path.join(os.path.dirname(wav), os.path.basename(wav).split(".")[0]+".wav").replace('noisy', 'clean')
            if os.path.exists(gt_path):
                score_fp, score_quant = \
                    self.get_pyseq(qwav_out, fwav_out, gt_path)
                # noise_name, amp, sn = wav.split(".")[1:-1]
                # noise_name, amp, sn = wav.split(".")[1:-1]
                noise_name, amp, sn = "noise", 0, 0
                infos.append([os.path.basename(wav), noise_name, amp, sn, score_fp, score_quant])
        audio_out = serializeDataToInt8(copy.deepcopy(quant_out.astype(np.float32)))
        writeFile(audio_out, 'work_dir/audio_out.b', mode="wb")
        self.process.export()
        df = pd.DataFrame(infos, columns=df_columns)
        df.to_csv(f'{results_path}/pseq.csv', index=False)
            
    def reset_rnn_state(self):
        self.in_buffer = np.zeros((self.block_len))
        self.out_buffer = np.zeros((self.block_len))      
        self.process.reset("lstm")

    def run_calibration(self, resume_path, block_len, block_shift, is_reverse_state):
        if self.process.reload_calibration(saved_calib_name=resume_path):
            self.process.save(self.resume_path)
        else:
            if not self.process.load(self.resume_path):
                # calculate number of blocks
                quan_datasets = self.get_dataset(self.dataset_path, block_len, block_shift, reuse_lstm_state=False, is_reverse_state=is_reverse_state)
                # quantize two submodules
                # self.process.quan_graph.quan_ops()
                self.process.quantize(quan_datasets, is_dataset=True, saved_calib_name=self.resume_path)
        # quan_datasets = self.get_dataset(self.dataset_path, self.block_len, self.block_shift)
        # self.process.quantize(quan_datasets, is_dataset=True)
                
    # inputs: file/path
    def reset_calibration(self, inputs):
        self.reset_rnn_state()
        self.run_calibration(inputs)
    
    def reset_session(self):
        session = self.process.post_quan.get_onnx_session()
        self.input_feed = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in session.get_inputs()}
    
    def session_infer(self, in_mag, is_reverse_state=False):
        # model = onnx.load_model(self.model_path[0])
        # output_names = copy.deepcopy(self.output_names)
        # for name in self.process.post_quan.get_layer_output_names():
        #     if name not in self.output_names:
        #         self.output_names.append(name)    
        # for output in self.output_names:
        #     model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        session = self.process.post_quan.get_onnx_session()
        
        model_input_names = [inp.name for inp in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        self.input_feed[model_input_names[0]] = in_mag
        model_outputs = session.run(output_names, self.input_feed)
        if is_reverse_state:
            # set out states back to input
            self.input_feed["h1_in"] = model_outputs[7]
            self.input_feed["c1_in"] = model_outputs[8]
            self.input_feed["h2_in"] = model_outputs[5]
            self.input_feed["c2_in"] = model_outputs[6]
            self.input_feed["h3_in"] = model_outputs[3]
            self.input_feed["c3_in"] = model_outputs[4]
            self.input_feed["h4_in"] = model_outputs[1]
            self.input_feed["c4_in"] = model_outputs[2] 
        else:        
            self.input_feed["h1_in"] = model_outputs[1]
            self.input_feed["c1_in"] = model_outputs[2]
            self.input_feed["h2_in"] = model_outputs[3]
            self.input_feed["c2_in"] = model_outputs[4]
            self.input_feed["h3_in"] = model_outputs[5]
            self.input_feed["c3_in"] = model_outputs[6]
            self.input_feed["h4_in"] = model_outputs[7]
            self.input_feed["c4_in"] = model_outputs[8]
        onnx_outputs = dict()
        for i in range(len(output_names)):
            onnx_outputs[output_names[i]] = model_outputs[i]
        onnx_outputs.update(self.input_feed)
        
       
        return onnx_outputs
                
    def inference(self, in_data, is_check_error: int):
        self.in_buffer[:-self.block_shift] = self.in_buffer[self.block_shift:]
        self.in_buffer[-self.block_shift:] = in_data
        win = windows.hann(self.block_len)
        if self.is_use_hn:
            in_block_fft = np.fft.rfft(self.in_buffer*np.sqrt(win))
        else:
            in_block_fft = np.fft.rfft(self.in_buffer)

        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)

        # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
        in_mag = (in_mag + 1.0e-7).astype(np.float32)
        if is_check_error:
            onnx_outputs = self.session_infer(in_mag)
            if is_check_error == 1:
                self.process.checkerror(in_mag, acc_error=True, onnx_outputs=onnx_outputs)
            else:
                self.process.dataflow(in_mag, acc_error=True, onnx_outputs=onnx_outputs)
        else:
            self.process.dataflow(in_mag, acc_error=True, onnx_outputs=None)
        outputs_1 = self.process.get_outputs()['qout']

        # calculate the ifft
        estimated_mag = outputs_1[self.result_name]
        estimated_complex = estimated_mag * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)

        # shift values and write to buffer
        # n, c = estimated_block.shape[:2]
        out_block = estimated_block.astype(np.float32)#.reshape(n, c)
        self.out_buffer[:-self.block_shift] = self.out_buffer[self.block_shift:]
        self.out_buffer[-self.block_shift:] = np.zeros((self.block_shift))
        self.out_buffer += np.squeeze(out_block)
        # if self.is_use_hn:
        #     return np.float32(self.out_buffer[:self.block_shift]/(0.67)) #self.out_buffer[:self.block_shift]
        # else:
        #     return np.float32(self.out_buffer[:self.block_shift]) #self.out_buffer[:self.block_shift]
        return np.float32(self.out_buffer[:self.block_shift]) #self.out_buffer[:self.block_shift]

    def quant_file(self, block_len=512, block_shift=128, check_error: int=0):
        eval_audio, eval_fs = wav_read(self.eval_path, tgt_fs=16000)
        num_blocks = (eval_audio.shape[0] - (block_len - block_shift)) // block_shift
        audio_in = serializeDataToInt8(copy.deepcopy(eval_audio))
        writeFile(audio_in, 'work_dir/audio_in.b', mode="wb")
        print('==> audio len: {} secs'.format(len(eval_audio) / eval_fs))
        import time
        start_t = time.time()
        quant_out = self.quant_vioce(eval_audio, num_blocks, block_len, block_shift, check_error=check_error)
        print('time is: {} ms!'.format((time.time() - start_t)*1000/num_blocks))
        eval_out = self.fp_infer(eval_audio, num_blocks, block_len, block_shift)
        # fout, fout_1 = fout
        l2, cosine = L2Simiarity()(eval_out, quant_out), CosineSimiarity()(eval_out, quant_out)
        print('out buffer l2 error is: {}, cosine error is: {}'.format(l2, cosine))
        wav_out = self.eval_path.replace('.wav', '_quant.wav')
        print('==> save wav to: ', wav_out)
        wav_write(quant_out, 16000, wav_out)
        wav_write(eval_out, 16000, self.eval_path.replace('.wav', '_fp.wav'))
        self.process.export()
        audio_out = serializeDataToInt8(copy.deepcopy(quant_out.astype(np.float32)))
        writeFile(audio_out, 'work_dir/audio_out.b', mode="wb")

    def test_api(self, *args, **kwargs):
        self.run_calibration(self.dataset_path)
        check_error = kwargs.get("check_error", 0)
        print("############################### reset_calibration: start")
        self.reset_calibration(self.dataset_path)
        print("############################### reset_calibration: finish")
        
        if isinstance(self.eval_path, list) or isinstance(self.eval_path, tuple):
            self.quant_dataset(False, self.block_len, self.block_shift, check_error=check_error)
        elif os.path.isdir(self.eval_path):
            self.quant_dataset(True, self.block_len, self.block_shift, check_error=check_error)
        else:            
            self.quant_file(self.block_len, self.block_shift)
            
    def __call__(self, *args, **kwargs):
        
        if not self.onnxinfer:
            self.onnxinfer = rt.InferenceSession(self.model_path[0], providers=["CPUExecutionProvider"])
            self.input_feed = {
                inp.name: np.zeros(
                    [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                    dtype=np.float32)
                for inp in self.onnxinfer.get_inputs()}

        block_len = self.block_len #512 # 512
        block_shift = self.block_shift #160 # 128
        check_error = kwargs.get("check_error", 0)
        reuse_lstm_state = kwargs.get("reuse_lstm_state", False)
        is_reverse_state = kwargs.get("is_reverse_state", False)
        if not os.path.exists(self.resume_path):
            os.makedirs(self.resume_path)
        saved_calib_name=self.resume_path+"/../"+self.resume_path.split('/')[-1]
        if self.process.reload_calibration(saved_calib_name=saved_calib_name):
            self.process.save(self.resume_path)
        else:
            if not self.process.load(self.resume_path):
                # calculate number of blocks
                quan_datasets = self.get_dataset(self.dataset_path, block_len, block_shift, reuse_lstm_state, is_reverse_state)
                # quantize two submodules
                # self.process.quan_graph.quan_ops()
                self.process.quantize(quan_datasets, is_dataset=True, saved_calib_name=self.resume_path)                

        if isinstance(self.eval_path, list) or isinstance(self.eval_path, tuple):
            self.quant_dataset(False, block_len, block_shift, check_error, reuse_lstm_state, is_reverse_state)
        elif os.path.isdir(self.eval_path):
            self.quant_dataset(True, block_len, block_shift, check_error, reuse_lstm_state, is_reverse_state)
        else:            
            self.quant_file(block_len, block_shift, check_error)
        if check_error == 2:
            self.process.error_analysis()

        # self.process.export()


class PreProcess(object):

    def __init__(self, **kwargs):
        super(PreProcess, self).__init__()
        self.input_size = kwargs['input_size']
        self.trans = 0

    def set_trans(self, trans):
        self.trans = trans

    def get_trans(self):
        return self.trans

    def __call__(self, img):
        if isinstance(img, dict):
            img_input = {}
            for input_name, im in img.items():
                if input_name in ['mag', 'y1']:
                    im_ = im.astype(np.float32)
                    img_input[input_name] = im_  # .transpose(1, 2, 0)
                else:
                    im_ = im.astype(np.float32)  # im[np.newaxis, :]
                    img_input[input_name] = im_.transpose(1, 0, 2)
        else:
            img_input = img.astype(np.float32)

        return img_input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=list,
                        default=[
                            # denoise_0725_simplify 20220809_model_simplify B1_model_480_20221013_simplify
                            # 'trained_models/test_voice/B1_model_simplify-sim.onnx',
                            'B1Net-to-Candy-2023.1.13/models_B1_model_512_160_norm/B1_model_512_160_norm-sim-offline2.onnx',
                            # 'B1Net-to-Candy-2023.1.13/models_B1_model_norm/B1_model_norm-sim-offline2.onnx',
                            # 'trained_models/test_voice/B1_model_512_160_norm_ln_del_log_sim.onnx',
                            # 'trained_models/test_voice/B1_model__norm_ln_del_log_sim.onnx',
                            # "B1Net-to-Candy-2023.1.13/models_B1_model_512_160_norm/B1_model_512_160_norm-sim.onnx",
                            ]
                            #/home/shiqing/Downloads/onnx_converter/trained_models/test_voice/model_p2_simplify_update.onnx'],
                        )
    parser.add_argument('--dataset_path', type=str,
                        # default='/buffer/trained_models/test_voice/example.wav',
                        # default='trained_models/test_voice/in_voice/Ts-mic_voice_ch2.wav',
                        # default='/home/shiqing/Downloads/onnx_converter/trained_models/test_voice/record_20220812.wav',
                        # default='/buffer/trained_models/test_voice/inner_model/calibration/noisy',
                        default='/buffer/trained_models/test_voice/inner_model/calibration/0919_noisy',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/0831_v1/calibration',
                        )
    parser.add_argument('--eval_path', type=str,
                        # default='/buffer/trained_models/test_voice/calibration/example.wav',
                        # default='/home/shiqing/Downloads/onnx_converter/trained_models/test_voice/record_20220812.wav',
                        # default=['/home/shiqing/Downloads/onnx_converter/trained_models/test_voice/record_20220812.wav',
                                #  '/home/shiqing/Downloads/onnx_converter/trained_models/test_voice/example.wav'],
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/real/noisy',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/for_sy_0822',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/scale_out',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/scale_out_new',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/out_indoor_40_15db1',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/real/long_voice',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/real',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/real',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/0831_v1/test_set',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/0831_v1/20230912',
                        # default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/0831_v1/wenjin',
                        default='/home/shiqing/Downloads/onnx-converter/trained_models/test_voice/inner_model/real/0919_eval',
                        )
    parser.add_argument('--check_error', type=int, default=0)
    parser.add_argument('--resume_path', type=str, default="work_dir/resume_int16")
    parser.add_argument('--input_size', type=dict,
                        default=dict(
                            p1=dict(h1_in=[1, 166],
                                    c1_in=[1, 166],
                                    h2_in=[1, 166],
                                    c2_in=[1, 166],
                                    h3_in=[1, 166],
                                    c3_in=[1, 166],
                                    h4_in=[1, 166],
                                    c4_in=[1, 166],
                                    mag=[1, 257]),
                            p2=dict(h1_in=[1, 128],
                                    c1_in=[1, 128],
                                    h2_in=[1, 128],
                                    c2_in=[1, 128],
                                    y1=[1, 512]), )
                        )
    parser.add_argument('--input_names', type=list,
                        default=[['mag'], ['y1']])
    parser.add_argument('--output_names', type=list,
                        # default=[['282'], ['y1']])
                        # default=[['onnx::Reshape_282'], ['y1']])
                        # default=[['284'], ['y1']])
                        default=[['y'], ['y1']])
    parser.add_argument('--block_len', type=int, default=512)
    parser.add_argument('--block_shift', type=int, default=128)  
    parser.add_argument('--is_use_hn', type=str, default=True)  
    parser.add_argument('--export_version', type=int, default=2)
    parser.add_argument('--export', type=bool, default=False)
    parser.add_argument('--is_stdout', type=bool, default=True)
    parser.add_argument('--print_frame', type=bool, default=True)
    parser.add_argument('--is_reverse_state', type=bool, default=False) # zx is true, ours is false

    args = parser.parse_args()
    return args

def extract_wavs():
    from glob import glob
    import os
    import shutil

    root_dirs = [
        "trained_models/test_voice/B1_model_512_160_simplify-sim",
        "trained_models/test_voice/B1_model_simplify-sim",
    ]
    for root_dir in root_dirs:
        wav_files = glob(os.path.join(root_dir, "*", "*", "*.wav"))
        for wav_file in wav_files:
            wave_name = os.path.basename(wav_file).split(".wav")[0]
            if "fp_voice" in wav_file:
                wave_name += "_fp"
            elif "quant_voice" in wav_file:
                wave_name += "_quant"
                
            if "_hn" in wav_file:
                wave_name += "_hn"
                        
            if "160" in wav_file:
                wave_name += "_10"
            elif "128" in wav_file:
                wave_name += "_8"
                
            wave_name += ".wav"
            new_file = os.path.join(root_dir, wave_name)
            shutil.copy(wav_file, new_file)
            
def test_combinations(model_path, 
                      block_shift=None, 
                      is_use_hn=None, 
                      resume_path=None,
                      reuse_lstm_state=False):
    args = parse_args()
    
    args.model_path = model_path
    if block_shift is not None:
        args.block_shift = block_shift
    if is_use_hn is not None:
        args.is_use_hn = is_use_hn
    
    print_log = True

    kwargs_preprocess = {
        'input_size': args.input_size
    }
    preprocess = PreProcess(**kwargs_preprocess)

    export_version = '' if args.export_version > 1 else '_v{}'.format(
        args.export_version)
    
    process_args = {
        'log_name': 'process.log',
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
        'transform': preprocess,
        'simulation_level': 1,
        'fp_result': True,
        'is_ema': True,
        'ema_value': 0.99,
        'is_array': False,
        'is_stdout': args.is_stdout,
        'print_log': print_log,
        'error_metric': ['L1', 'L2', 'Cosine'],  # Cosine | L2 | L1
    }

    kwargs_clseval = {
        'log_dir': 'work_dir/eval',
        'log_name': ['p1.log', 'p2.log'],
        'is_stdout': args.is_stdout,
        'print_frame': args.print_frame,
        'resume_path': resume_path,
        'input_size': args.input_size,
        'dataset_path': args.dataset_path,
        'eval_path': args.eval_path,
        'eval_fisrt_frame': True,
        'process_args': process_args,
        'input_names': args.input_names,
        'output_names': args.output_names,
        "check_error": args.check_error,
        'export': args.export,
        'block_len': args.block_len,
        'block_shift': args.block_shift, 
        "is_use_hn": args.is_use_hn,    
        "is_reverse_state": args.is_reverse_state,   
    }
    kwargs = dict(check_error=args.check_error, reuse_lstm_state=reuse_lstm_state, is_reverse_state=args.is_reverse_state)
    myClsEval = NetEvalOneModel(**kwargs_clseval)
    myClsEval(**kwargs)
        
if __name__ == '__main__':
    model_path = [
        # "/home/shiqing/Downloads/test_package/saturation/onnx-converter/trained_models1/test_voice/DTLN_model_0316_v2_B1_166_reduce.onnx",
        # "/home/shiqing/Downloads/test_package/saturation/onnx-converter/trained_models1/test_voice/DTLN_snr_dpesq_0831_v1_B1_166_norm_16ms_reduce.onnx",
        # "/home/shiqing/Downloads/test_package/saturation/onnx-converter/trained_models1/test_voice/DTLN_snr_dpesq_0831_v1_B1_166_normfalse_16ms_new_reduce.onnx",
        # "/home/shiqing/Downloads/test_package/saturation/onnx-converter/trained_models1/test_voice/DTLN_snr_dpesq_0831_v1_B1_166_normfalse_16ms_new_ep10_sim_reduce.onnx",
        # "/home/shiqing/Downloads/test_package/saturation/onnx-converter/trained_models1/test_voice/DTLN_snr_dpesq_0831_v1_B1_166_normfalse_16ms_new_window_sim_reduce.onnx",
        "/home/shiqing/Downloads/test_package/saturation/onnx-converter/trained_models1/test_voice/DTLN_snr_dpesq_0919_v1_B1_166_normfalse_16ms_new_window_ep23_sim_reduce.onnx",
        # "/home/shiqing/Downloads/test_package/saturation/onnx-converter/trained_models1/test_voice/20220809_model_simplify.onnx",
        # "B1Net-to-Candy-2023.1.13/models_B1_model_norm/B1_model_norm-sim-offline2.onnx",
    ]
    
    reuse_lstm_state = True
    fd_path = "calibration_resuse_lstm_hidden_state" if reuse_lstm_state \
        else "calibration_reset_lstm_hidden_state"
    reuse_lstm_state = False
    
    # resume_path = "resume/{}/{}/resume_int8".\
    #     format(os.path.basename(model_path[0]).replace(".onnx", ""), fd_path)
    
    resume_path = "resume/hn_win/{}/{}/resume_int8_fc_sigmoid".\
        format(os.path.basename(model_path[0]).replace(".onnx", ""), fd_path)
        
    # resume_path = "resume/{}/{}/resume_int8_fc_sigmoid_mul_int8".\
    #     format(os.path.basename(model_path[0]).replace(".onnx", ""), fd_path)
        
    # resume_path =  "resume/{}/{}/resume_first_lstm_feat_int16_fc_sigmoid".\
    #     format(os.path.basename(model_path[0]).replace(".onnx", ""), fd_path) 
        
    # resume_path = "resume/{}/{}/resume_int16".\
    #     format(os.path.basename(model_path[0]).replace(".onnx", ""), fd_path)
        
    test_combinations(model_path=model_path, 
                      block_shift=256, 
                      is_use_hn=True, 
                      resume_path=resume_path,
                      reuse_lstm_state=reuse_lstm_state)
    # if "models_B1_model_512_160_norm" in model_path[0]:
    #     block_shift_list = [160]
    # else:
    #     block_shift_list = [160, 128]
    # for block_shift in block_shift_list:        
    #     for is_use_hn in [False, True]:
    #         test_combinations(model_path=model_path, block_shift=block_shift, is_use_hn=is_use_hn)
