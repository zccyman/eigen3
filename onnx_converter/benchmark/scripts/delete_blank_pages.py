#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/05/22 12:11:31
#@File    : delete_blank_pages.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import argparse
import os

try:
    from PyPDF2 import PdfFileReader
except:
    from PyPDF2 import PdfReader as PdfFileReader

try:
    from PyPDF2 import PdfFileWriter
except:
    from PyPDF2 import PdfWriter as PdfFileReader


def get_recursive_file_list(path):
    current_files = os.listdir(path)
    all_files = []
    for file_name in current_files:
        full_file_name = os.path.join(path, file_name)
        all_files.append(full_file_name)
 
        if os.path.isdir(full_file_name):
            next_level_files = get_recursive_file_list(full_file_name)
            all_files.extend(next_level_files)
 
    return all_files

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--export_dir', type=str, default='work_dir/benchmark/export')
    parser.add_argument('--export_dir', type=str, default='work_dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    for file in get_recursive_file_list(args.export_dir):
        if 'test_vis.mmd.pdf' in file:
            reader = PdfFileReader(open(file, 'rb'))
            writer = PdfFileWriter()

            pages = reader.getNumPages()
            for i in range(pages):
                page = reader.getPage(i)
                if "/XObject" in page["/Resources"].keys() or "/Font" in page["/Resources"].keys():
                    writer.addPage(page)
            output_pdf = open(file.replace('.pdf', '.simplify.pdf'), 'wb')
            writer.write(output_pdf)
            os.remove(file)