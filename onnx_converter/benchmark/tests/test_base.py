# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : TIMESINETLLI TECH
# @Time     : 2022/7/15 17:46
# @File     : test_base.py

import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import json
import os
import unittest
from abc import abstractmethod

import cv2
import numpy as np
import pytest
from benchmark import (collect_accuracy, parse_config, save_config,
                       save_export, save_tables)


class TestBasse(object):

    # @staticmethod
    # def eval(self, args, model_type, model_id, quantize_dtype, process_scale_w, quantize_method_f, quantize_method_w,
    #          model_name, model_dir, dataset_dir, selected_mode, password):
    #
    #     evaluator = self.compose_evaluator(
    #         **dict(
    #             args=args,
    #             model_dir=model_dir,
    #             model_name=model_name,
    #             dataset_dir=dataset_dir,
    #             selected_mode=selected_mode
    #         )
    #     )
    #
    #     # quan_dataset_path, images_path, ann_path, input_size, normalization, save_eval_path
    #     self.function(model_type, model_id, quantize_dtype, process_scale_w, quantize_method_f, quantize_method_w,
    #                   model_name, model_dir, password, evaluator)

    @abstractmethod
    def compose_evaluator(self, **kwargs):
        NotImplemented

    def function(self, model_type, model_id, quantize_dtype, process_scale_w,
                 quantize_method_f, quantize_method_w, model_name, model_dir,
                 password, args, evaluator):

        model_path = os.path.join(model_dir, args.task_name, model_name)  # args.model_paths[model_type][model_id]
        model_name = os.path.basename(model_path).split('.onnx')[0]
        log_name = '{}.{}.{}.{}.{}.log'.format(
            model_name, quantize_method_f, quantize_method_w, quantize_dtype, process_scale_w)

        if isinstance(evaluator, dict):
            eval_func = evaluator['evaluator']
            parameter = evaluator['parameters']
            accuracy, tb = eval_func(**parameter)
        else:
            accuracy, tb = evaluator()
        if args.generate_experience_value:
            for key in ['recall', 'precision']:
                args.accuracy[model_type][key][model_id] = np.round(accuracy['qaccuracy'][key], 4)
            json_content = [args.layer_error, args.accuracy]
            with open('benchmark/benchmark_config/experience/{}.json'.format(args.task_name), 'w') as f:
                json.dump(json_content, f, indent=4, ensure_ascii=False)
        else:
            with open('benchmark/benchmark_config/experience/{}.json'.format(args.task_name), 'r') as f:
                json_data = json.load(f)
                args.layer_error = json_data[0]
                args.accuracy = json_data[1]

        if args.is_assert:
            # errors, max_errors = evaluator.collect_error_info()
            # for _, error in errors.items():
            #     for metric, err in error.items():
            #         pytest.assume(err <= args.layer_error[model_type][metric][model_id] * (1.0 + args.variable_rate))
            for key in args.accuracy[model_type].keys():
                if key not in accuracy['qaccuracy'].keys():
                    with pytest.raises(ValueError) as exc_info:
                        raise ValueError("{} must be in args.accuracy {}!".format(key, 'qaccuracy'))
                pytest.assume(
                    accuracy['qaccuracy'][key] >= args.accuracy[model_type][key][model_id] * (
                            1.0 - args.variable_rate)
                )
        if args.export:
            if isinstance(evaluator, dict):
                evaluator['evaluator'].export()
            else:
                evaluator.export()
            # evaluator.export()
            case_name = log_name.split('.log')[0]
            save_export(args, case_name, password)

        args.tables[model_type][model_id] = collect_accuracy(args.tables[model_type][model_id], tb)
        save_tables(args)

        return accuracy, tb

    def is_skip(self, setting, args, condition):
        model_name = setting['model_name']
        quantize_method_w = setting['quantize_method_w']
        quantize_dtype = setting['quantize_dtype']
        process_scale_w = setting['process_scale_w']

        flag = model_name not in args.MR_model[condition]
        flag = flag or (quantize_method_w not in args.MR_quantize[condition])
        flag = flag or (quantize_dtype not in args.MR_quantize[condition])
        flag = flag or (process_scale_w not in args.MR_quantize[condition])
        return flag

    # read setting from testing config files, make evaluation calc or not
    # select mode in ['MR_MASTER', 'MR_RELEASE', 'MR_DEV', 'MR_OTHER']
    def select_mode(self, args, model_paths, quantize_dtype, process_scale_w, quantize_method, selected_mode):
        model_type, model_id, model_name = model_paths.split('/')
        quantize_method_f, quantize_method_w = quantize_method.split('/')
        # model_id = int(model_id)
        quantize_dtype = int(quantize_dtype)
        setting = dict(
            model_name=model_name,
            quantize_method_w=quantize_method_w,
            quantize_dtype=quantize_dtype,
            process_scale_w=process_scale_w
        )
        skip_info = lambda condition: \
            pytest.skip("test mode is {}\n {},{},{},{} is skiped!".format(condition,
                                                                          model_name,
                                                                          quantize_method_w,
                                                                          quantize_dtype,
                                                                          process_scale_w))
        if selected_mode == 'MR_RELEASE' and self.is_skip(setting, args, selected_mode):
            skip_info(selected_mode)
        elif selected_mode == 'MR_MASTER' and self.is_skip(setting, args, selected_mode):
            skip_info(selected_mode)
        elif selected_mode == 'MR_DEV' and self.is_skip(setting, args, selected_mode):
            skip_info(selected_mode)
        elif selected_mode == 'MR_OTHER' and self.is_skip(setting, args, selected_mode):
            skip_info(selected_mode)
        # else:
        #     pass
        else:
            if selected_mode not in ["MR_RELEASE", "MR_MASTER", "MR_DEV", "MR_OTHER"]:
                with pytest.raises(ValueError) as exc_info:
                    raise ValueError("select_mode must in range of [MR_RELEASE, MR_MASTER, MR_DEV, MR_OTHER]")

    def entrance(self, args, model_paths, quantize_method, quantize_dtype, process_scale_w, selected_mode, model_dir,
                 dataset_dir, password):
        model_type, model_id, model_name = model_paths.split('/')
        quantize_method_f, quantize_method_w = quantize_method.split('/')
        model_id = int(model_id)
        quantize_dtype = int(quantize_dtype)
        self.select_mode(args, model_paths, quantize_dtype, process_scale_w, quantize_method, selected_mode)

        save_config(
            args=args,
            model_type=model_type,
            quantize_dtype=quantize_dtype,
            process_scale_w=process_scale_w,
            quantize_method_f=quantize_method_f,
            quantize_method_w=quantize_method_w)

        evaluator = self.compose_evaluator(
            args=args,
            model_dir=model_dir,
            model_name=model_name,
            dataset_dir=dataset_dir,
            selected_mode=selected_mode,
            model_type=model_type,
            quantize_dtype=quantize_dtype,
            process_scale_w=process_scale_w,
        )

        # quan_dataset_path, images_path, ann_path, input_size, normalization, save_eval_path
        self.function(model_type, model_id, quantize_dtype, process_scale_w, quantize_method_f, quantize_method_w,
                      model_name, model_dir, password, args, evaluator)
