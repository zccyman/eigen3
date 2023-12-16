#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/29 19:43:10
#@File    : face_recognition.py

_base_ = ['./base/base.py']

MR_model = dict(
    MR_DEV=['mobilefacenet_pad_qat_new_2.onnx'],
    MR_MASTER=['mobilefacenet_pad_qat_new_2.onnx'],
    MR_RELEASE=[
        'mobilefacenet_pad_qat_new_2.onnx', 
        'mobilefacenet_method_3_simplify.onnx'
        ],
    MR_OTHER=[
        'mobilefacenet_pad_qat_new_2.onnx', 
        'mobilefacenet_method_3_simplify.onnx'
        ],        
)

### dataset, imgsize, network, preprocess, postprocess 
model_paths = dict(lfw_imgsize1_mobilenet_pre1_post1=[
    'mobilefacenet_pad_qat_new_2.onnx',
    # 'mobilefacenet_pad_simplify.onnx',
    # 'mobilefacenet_method_1_simplify.onnx',
    # 'mobilefacenet_method_2_simplify.onnx',
    'mobilefacenet_method_3_simplify.onnx'
    ]
)
task_name = 'face-recognition'

dataset_dir = dict(lfw="faces_emore")
test_sample_num = dict(MR_RELEASE=1200, MR_MASTER=1200, MR_DEV=1200, MR_OTHER=1200) ####12000
feat_dim = 512
metric = "euclidean"  # "cosine" or "euclidean"
max_threshold = 4.0  # 1.0 | 4.0

offline_quan_mode = False
offline_quan_tool = 'NCNN'
quan_table_path = 'work_dir/quan_table/NCNN/quantize.table'
results_path = 'work_dir/tmpfiles/{}_resluts'.format(task_name)
log_dir = 'work_dir/benchmark/log/{}'.format(task_name)
export_dir = 'work_dir/benchmark/export/{}'.format(task_name)

# is_remove_transpose = dict(mobilenet=False)

### reference of accuracy/error
tables_head = 'TASK2: test_face_recognition'
tables = dict(
    lfw_imgsize1_mobilenet_pre1_post1=[None, None],
)
layer_error = dict(
    lfw_imgsize1_mobilenet_pre1_post1=dict(
        L1=[0, 0],
        L2=[0, 0],
        Cosine=[0, 0],     
    ),
)
accuracy = dict(lfw_imgsize1_mobilenet_pre1_post1=dict(
    accuracy=[0, 0], best_threshold=[0, 0])
)