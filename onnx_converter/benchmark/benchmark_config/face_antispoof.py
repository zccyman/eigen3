#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/29 19:42:54
#@File    : face_antispoof.py

_base_ = ['./base/base.py']

MR_model = dict(
    MR_DEV=['MobileNet3_simplify.onnx'],
    MR_MASTER=['MobileNet3_simplify.onnx'],
    MR_RELEASE=[
        'MobileNet3_simplify.onnx', 
        'MobileLiteNetB_simplify.onnx', 
        'MobileLiteNetB_brightness_simplify.onnx'
        ],
    MR_OTHER=[
        'MobileNet3_simplify.onnx', 
        'MobileLiteNetB_simplify.onnx', 
        'MobileLiteNetB_brightness_simplify.onnx'
        ],
)

### dataset, imgsize, network, preprocess, postprocess 
model_paths = dict(
    antispoof_imgsize1_mobilelite_pre1_post1=[
    'MobileNet3_simplify.onnx',
    'MobileLiteNetB_simplify.onnx',
    'MobileLiteNetB_brightness_simplify.onnx'
    ]
)
task_name = 'anti-spoofing'

frame_size = [224, 224]
face_size = [128, 128]
normalizations = [[0.5931, 0.4690, 0.4229], [0.2471, 0.2214, 0.2157]]
swapRB = True
spoof_prob_threshold = 0.6
class_num = 2

offline_quan_mode = False
offline_quan_tool = 'NCNN'
quan_table_path = 'work_dir/quan_table/NCNN/quantize.table'
dataset_dir = dict(antispoof='anti_spoofing/CelebA_Spoof/Data/test')
image_subdir = dict(antispoof=['5010','5013', '5015', '5023', '5028','5033', '5030', '5035', '5051','5052', '5061', '5072'])
image_prefix = 'png'
roc_save_path = 'work_dir/tmpfiles/'
results_path = 'work_dir/tmpfiles/{}/'.format(task_name)
log_dir = 'work_dir/benchmark/log/{}'.format(task_name)
export_dir = 'work_dir/benchmark/export/{}'.format(task_name)

# is_remove_transpose = dict(mobilelite=False)

### reference of accuracy/error
tables_head = 'TASK1: test_anti_spoofing'
tables = dict(
    antispoof_imgsize1_mobilelite_pre1_post1=[None, None, None],
)
layer_error = dict(
    antispoof_imgsize1_mobilelite_pre1_post1=dict(
        L1=[0, 0, 0], 
        L2=[0, 0, 0], 
        Cosine=[0, 0, 0]
    ),
)
accuracy = dict(
    antispoof_imgsize1_mobilelite_pre1_post1=dict(
        recall=[0, 0, 0], 
        precision=[0, 0, 0]),
)