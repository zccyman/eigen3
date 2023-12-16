#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/30 09:53:07
#@File    : eval.py
import os

import numpy as np
import prettytable as pt

try:
    from config import Config
    from utils import Object
except:
    from onnx_converter.config import Config # type: ignore
    from onnx_converter.utils import Object # type: ignore

class Eval(Object): # type: ignore
    def __init__(self, **kwargs):
        super(Eval, self).__init__()
        
        self.eval_first_frame = kwargs['eval_first_frame'] if 'eval_first_frame' in kwargs.keys() else False
        self.quan_cfg = kwargs['process_args']['quan_cfg']
        self.is_stdout = kwargs['process_args']['is_stdout']
        self.log_dir = kwargs['log_dir']
        self.log_name = kwargs["log_name"]
        self.log_level = kwargs.get('log_level', 20)
        self.logger = self.get_log(log_name=self.log_name, log_level=self.log_level, stdout=self.is_stdout)
        self.fp_result, self.process_args = None, None
        self.process = None

    def reload_calibration(self, resume_path='work_dir/resume'):
        reload_flag = False
        reload_flag = self.process.load(resume_path)
        if not reload_flag:
            reload_flag = self.process.reload_calibration(saved_calib_name=resume_path)
            self.process.save(resume_path)
            reload_flag = self.process.load(resume_path)
        return reload_flag
    
    def save_calibration(self, resume_path):
        self.process.save(resume_path)

    def get_quant_method(self):
        save_cfg = Config.fromfile(self.quan_cfg)
        cfg_dict, _ = save_cfg._file2dict(self.quan_cfg)
        if 'qs' in cfg_dict.keys(): # type: ignore
            qs = cfg_dict['qs'] # type: ignore

            if 'symquan' in qs.feat_method:
                quant_method_f = 'symquan'
            else:
                quant_method_f = 'asymquan'
                            
            if 'symquan' in qs.weight_method:
                if 'perchannel' in qs.weight_method:
                    quant_method_w = 'symquan/perchannel'
                else:
                    quant_method_w = 'symquan/pertensor'
            else:
                if 'perchannel' in qs.weight_method:
                    quant_method_w = 'asymquan/perchannel'
                else:
                    quant_method_w = 'asymquan/pertensor'

            bits_dict = cfg_dict['bits_dict'] # type: ignore
            data_type = bits_dict[qs.bit_select].split('np.')[1]
            process_scale_w = qs.process_scale_w
            txme_saturation = str(qs.txme_saturation)
            precision = str(qs.precision)
            quant = quant_method_f + '/' + quant_method_w + '/' + data_type + '/' + \
                    process_scale_w + '/' + txme_saturation + '/' + precision
        else:
            quant = 'quant'

        return quant

    def draw_table(self, res):
        align_method = 'c'
        table_head = [os.path.basename(self.process_args['model_path'])] # type: ignore
        table_head_align = [align_method]
        
        quant = self.get_quant_method()
        quant_result = [quant]
        for k, v in res['qaccuracy'].items():
            quant_result.append("{:.5f}".format(v))
            table_head.append(k)
            table_head_align.append(align_method)

        tb = pt.PrettyTable()
        tb.field_names = table_head

        if self.fp_result:
            fp_result = ['float']
            for k, v in res['faccuracy'].items():
                fp_result.append("{:.5f}".format(v))
            tb.add_row(fp_result)
        tb.add_row(quant_result)

        for head, align in zip(table_head, table_head_align):
            tb.align[head] = align
            
        return tb
    
    def print_layer_average_error(self, errors_dict):
        fp_cosine = []
        metrics = {}
        for layer_idx, (layer_name, errors) in enumerate(errors_dict.items()):
            metrics = errors.keys() if metrics is None else metrics
            self.logger.info('-------------------------------------------------------------------------')
            for metric, error in errors.items():
                self.logger.info('layer_idx: {}, layer_name: {}, average {} error is: {:.5f}'.format(
                    layer_idx, layer_name, metric, error))  
            self.logger.info('-------------------------------------------------------------------------')

            if "_fp" in layer_name:
                fp_cosine.append(errors["Cosine"])

        if len(fp_cosine) > 0:
            import matplotlib.pyplot as plt
            plt.scatter(range(len(fp_cosine)), fp_cosine, alpha=0.6)
            plt.savefig('fp_cosine.jpg')

        max_errors = {}
        for metric in metrics:
            error_list = list()
            for layer_name in errors_dict.keys():
                error_list.append(errors_dict[layer_name][metric])
            max_errors[metric] = np.max(error_list)

            self.logger.info('-------------------------------------------------------------------------')
            self.logger.info('{} min error is: {:.5f}'.format(metric, np.min(error_list)))
            self.logger.info('{} max error is: {:.5f}'.format(metric, np.max(error_list))) 
            self.logger.info('{} mean error is: {:.5f}'.format(metric, np.mean(error_list)))  
            self.logger.info('{} median error is: {:.5f}'.format(metric, np.median(error_list)))
            self.logger.info('-------------------------------------------------------------------------')     

        return max_errors

    def export(self):
        self.process.export() # type: ignore
        self.logger.info('export model done !!!')

    def error_analysis(self):
        self.process.error_analysis() # type: ignore
        self.logger.info('error analysis done !!!')
        
    def collect_error_info(self):
        errors = self.process.get_layer_average_error() # type: ignore
        max_errors = self.print_layer_average_error(errors)
        return errors, max_errors
        
    def get_results(self, accuracy):
        # errors = self.process.get_layer_average_error()
        # max_errors = self.print_layer_average_error(errors)
        if accuracy is not None:
            tb = self.draw_table(accuracy)
            self.logger.info(tb)
        else:
            tb = ''
            
        return tb