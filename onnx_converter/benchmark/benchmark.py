#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/29 18:04:31
#@File    : test_benchmark.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

try:
    from config import Config
    from utils import Dict2Object
except:
    from onnx_converter.config import Config
    from onnx_converter.utils import Dict2Object

import copy
import json
import os


def parse_config(cfg_file):
    save_cfg = Config.fromfile(cfg_file)
    cfg_dict, _ = save_cfg._file2dict(cfg_file)
    args = Dict2Object(cfg_dict)
    
    return args


def save_config(args, model_type, quantize_dtype, process_scale_w, quantize_method_f, quantize_method_w):
    with open('benchmark/benchmark_config/base/quantize_selected.py', 
                'w', encoding='utf-8') as f:
        # bits_dict = {0: 'np.uint8', 1: 'np.int8', 2: 'np.uint16', 3: 'np.int16', 
        # 4: 'np.uint32', 5: 'np.int32', 6: 'np.uint64', 7: 'np.int64', 
        # 8: 'np.float32', 9: 'np.float64'}
        if quantize_dtype in {0, 1}: 
            datatype = 8
        elif quantize_dtype in {2, 3}: 
            datatype = 16
        elif quantize_dtype in {4, 5}: 
            datatype = 32
        elif quantize_dtype in {6, 7}: 
            datatype = 64            
        else:
            raise Exception('Not Implemented quantize_dtype')
        
        dataset_name, imgsize, net_name, preprocess_name, postprocess_name = model_type.split('_')
        # is_remove_transpose = args.is_remove_transpose[net_name]
        
        # txme_saturation = 0
        if 'symquan' in quantize_method_f and 'symquan' in quantize_method_w:
            txme_saturation = 1
        else:
            txme_saturation = 0

        if process_scale_w == 'floatscale':
            precision = 1
        else:
            precision = 0

        if 'perchannel' in quantize_method_w:
            output_quant_w = 'perchannelfloatsymquan'
        else:
            output_quant_w = 'floatsymquan'
        output_quant_f = 'floatsymquan'

        content = 'bit_select = {}\n'.format(quantize_dtype)
        content += 'datatype = {}\n'.format(datatype)
        content += 'precision = {}\n'.format(precision)
        content += 'txme_saturation = {}\n'.format(txme_saturation)
        content += 'process_scale_w = \"{}\"\n'.format(process_scale_w)
        content += 'feat_method =  \"{}\"\n'.format(quantize_method_f)
        content += 'weight_method = \"{}\"\n'.format(quantize_method_w)
        content += 'output_quant_w =  \"{}\"\n'.format(output_quant_w)
        content += 'output_quant_f =  \"{}\"\n'.format(output_quant_f)

        f.write(content)


def save_export(args, case_name, password):
    export_dir = os.path.join(args.export_dir, case_name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    json_content = dict(maxTextSize=99999999)
    with open(os.path.join(export_dir, "mermaidRenderConfig.json"), "w") as f:
        json.dump(json_content, f, indent=4, ensure_ascii=False)
    json_content = dict(args=['--no-sandbox'])
    with open(os.path.join(export_dir, "puppeteer-config.json"), "w") as f:
        json.dump(json_content, f, indent=4, ensure_ascii=False)  

    if password is not None:
        os.system('bash benchmark/scripts/save_export.sh {} {} {}'.format(
            './', export_dir, password))
    else:
        file_list = ['weights', 'process.log', 'export.log', 'model.c', 'test_vis.mmd']
        for file_name in file_list:
            os.system('mv work_dir/{} {}'.format(file_name, export_dir))

    os.system('python benchmark/scripts/mmd2pdf.py --export_dir {}'.format(export_dir))


def collect_accuracy(tables, tb):
    if tables is not None:
        for data in tb.rows[1:]:
            tables.add_row(data)
    else:
        tables = copy.deepcopy(tb)

    return tables


def save_tables(args):
    with open(os.path.join(args.report_path, args.tables_head.split(': ')[-1] + '_accuracy.html'), 'w') as f:
        f.write(args.tables_head + '\n')
        for _, tables in args.tables.items():
            for table in tables:
                if table is not None:
                    table.format = True
                    f.write(str(table.get_html_string()))
                    f.write('\n')