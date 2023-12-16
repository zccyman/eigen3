#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/29 19:43:57
#@File    : pedestrian_detection.py

_base_ = ['./base/base.py']

MR_model = dict(
    MR_DEV=['yolov5s_320_v1_simplify.onnx'],
    MR_MASTER=['yolov5s_320_v1_simplify.onnx'],
    MR_RELEASE=[
        'yolov5s_320_v1_simplify.onnx', 
        'yolov5s_320_v2_simplify.onnx', 
        'yolov5n_320_v1_simplify.onnx',
        'nanodet_plus_m_1.0x_320_v1_simplify.onnx', 
        'nanodet_plus_m_1.0x_320_v2_simplify.onnx', 
        'nanodet_plus_m_1.5x_320_v1_simplify.onnx'
        ],
    MR_OTHER=[
        'yolov5s_320_v1_simplify.onnx', 
        'yolov5s_320_v2_simplify.onnx', 
        'yolov5n_320_v1_simplify.onnx',
        'nanodet_plus_m_1.0x_320_v1_simplify.onnx', 
        'nanodet_plus_m_1.0x_320_v2_simplify.onnx', 
        'nanodet_plus_m_1.5x_320_v1_simplify.onnx'
        ],        
)

### dataset, imgsize, network, preprocess, postprocess 
model_paths = dict(
    crowdhuman_imgsize1_yolov5_pre1_post1=[
    'yolov5s_320_v1_simplify.onnx',
    'yolov5s_320_v2_simplify.onnx',
    'yolov5n_320_v1_simplify.onnx'
    ],
    crowdhuman_imgsize1_nano_pre2_post2=[               
    'nanodet_plus_m_1.0x_320_v1_simplify.onnx',
    'nanodet_plus_m_1.0x_320_v2_simplify.onnx',
    'nanodet_plus_m_1.5x_320_v1_simplify.onnx'
    ]
)
task_name = 'pedestrian-detection'
# dataset_dir = dict(crowdhuman='crowdhuman/val')
# anno_file = dict(crowdhuman='crowdhuman/annotation_val.json')
dataset_dir = dict(crowdhuman='coco/cocoperson/images')
anno_file = dict(crowdhuman='coco/cocoperson/cocoperson.json')
img_prefix='jpg'
input_size = dict(imgsize1=[320, 320])
normalizations = dict(
    yolov5=[[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]],
    nano=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
)
strides = dict(yolov5=[8, 16, 32], nano=[8, 16, 32, 64])
num_classes = 1
topk = -1
reg_max = 7
prob_threshold =  0.3
iou_threshold = 0.6
num_candidate = 1000
agnostic_nms = False

offline_quan_mode = False
offline_quan_tool = 'NCNN'
quan_table_path = 'work_dir/quan_table/NCNN/quantize.table'
calibration_params_json_path = None
results_path = 'work_dir/tmpfiles/{}_resluts'.format(task_name)
log_dir = 'work_dir/benchmark/log/{}'.format(task_name)
export_dir = 'work_dir/benchmark/export/{}'.format(task_name)

# is_remove_transpose = dict(yolov5=False, nano=False)

### reference of accuracy/error
tables_head = 'TASK5: test_pedestrian_detection'
tables = dict(
    crowdhuman_imgsize1_yolov5_pre1_post1=[None, None, None],
    crowdhuman_imgsize1_nano_pre2_post2=[None, None, None],
)
layer_error = dict(
    crowdhuman_imgsize1_yolov5_pre1_post1=dict(
        L1=[0, 0, 0],
        L2=[0, 0, 0], 
        Cosine=[0, 0, 0]
    ),
    crowdhuman_imgsize1_nano_pre2_post2=dict(
        L1=[0, 0, 0], 
        L2=[0, 0, 0], 
        Cosine=[0, 0, 0]
    ),    
)
accuracy = dict(
    crowdhuman_imgsize1_yolov5_pre1_post1=dict(mAP=[0, 0, 0]), 
    crowdhuman_imgsize1_nano_pre2_post2=dict(mAP=[0, 0, 0])
) # reference AP of each model