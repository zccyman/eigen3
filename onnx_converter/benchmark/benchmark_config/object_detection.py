#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/29 19:43:40
#@File    : object_detection.py

_base_ = ['./base/base.py']

MR_model = dict(
    MR_DEV=['nanodet_1.0x_320_simplify.onnx'],
    MR_MASTER=['nanodet_1.0x_320_simplify.onnx'],
    MR_RELEASE=[
        'nanodet_1.0x_320_simplify.onnx',
        'nanodet_1.5x_320_simplify.onnx'
        ],
    MR_OTHER=[
        'nanodet_1.0x_320_simplify.onnx',
        'nanodet_1.5x_320_simplify.onnx'
        ]        
)

### dataset, imgsize, network, preprocess, postprocess 
model_paths = dict(coco_imgsize1_nano_pre1_post1=[
    'nanodet_1.0x_320_simplify.onnx',
    'nanodet_1.5x_320_simplify.onnx',
])
task_name = 'object-detection'
dataset_dir = dict(coco='coco/val2017')
anno_file = dict(coco='coco/annotations/instances_val2017.json')
img_prefix='jpg'
input_size = [320, 320]
num_classes = 80
normalizations = [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
strides = [8, 16, 32]
topk = -1
reg_max = 7
prob_threshold = 0.3
iou_threshold = 0.3
num_candidate = 1000
agnostic_nms = False

offline_quan_mode = False
offline_quan_tool = 'NCNN'
quan_table_path = 'work_dir/quan_table/NCNN/quantize.table'
calibration_params_json_path = None
results_path = 'work_dir/tmpfiles/{}_resluts'.format(task_name)
log_dir = 'work_dir/benchmark/log/{}'.format(task_name)
export_dir = 'work_dir/benchmark/export/{}'.format(task_name)

# is_remove_transpose = dict(nano=False)

### reference of accuracy/error
tables_head = 'TASK4: test_object_detection'
tables = dict(
    coco_imgsize1_nano_pre1_post1=[None, None],
)
layer_error = dict(
    coco_imgsize1_nano_pre1_post1=dict(
        L1=[0, 0], 
        L2=[0, 0], 
        Cosine=[0, 0]
    ),
)
accuracy = dict(
    coco_imgsize1_nano_pre1_post1=dict(mAP=[0, 0])
) # reference AP of each model