#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/05/18 12:11:31
#@File    : mmd2pdf.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, default='work_dir/benchmark/export')
    # parser.add_argument('--export_dir', type=str, default='work_dir')
    args = parser.parse_args()
    return args


def get_recursive_file_list(path):
    if not os.path.exists(path):
        return []
    current_files = os.listdir(path)
    all_files = []
    for file_name in current_files:
        full_file_name = os.path.join(path, file_name)
        all_files.append(full_file_name)
 
        if os.path.isdir(full_file_name):
            next_level_files = get_recursive_file_list(full_file_name)
            all_files.extend(next_level_files)
 
    return all_files


if __name__ == '__main__':
    args = parse_args()
    for file in get_recursive_file_list(args.export_dir):
        if 'test_vis.mmd' in file and '.pdf' not in file:
            path, _ = os.path.split(file)
            os.system('bash benchmark/scripts/export_visualization.sh {}'.format(path))