#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/29 19:44:09
#@File    : retina_face_detection.py

_base_ = ['./base/base.py']

MR_model = dict(
    MR_DEV=['slimv2_Final_simplify.onnx'],
    MR_MASTER=['slimv2_Final_simplify.onnx'],
    MR_RELEASE=[
        'shufflenetv2_Final_simplify.onnx', 
        'slimv2_Final_simplify.onnx', 
        'slim_special_Final_simplify_removed_pad.onnx'
        ],
    MR_OTHER=[
        'shufflenetv2_Final_simplify.onnx', 
        'slimv2_Final_simplify.onnx', 
        'slim_special_Final_simplify_removed_pad.onnx'
        ],        
)

### dataset, imgsize, network, preprocess, postprocess 
model_paths = dict(
    flickr_imgsize1_slim_pre1_post1=[
    'slim_special_Final_simplify_removed_pad.onnx',
    'slimv2_Final_simplify.onnx',
    'shufflenetv2_Final_simplify.onnx'
    ],
    # flickr_imgsize2_hisense_pre2_post2=[
    # 'hisense.onnx',
    # 'hisense_qat.onnx',
    # ],
    # flickr_imgsize2_hisense_pre3_post3=[
    # 'hisense.onnx',
    # 'hisense_qat.onnx',
    # ],    
)
task_name = 'face-detection'
dataset_dir = dict(flickr='AFLW/sub_test_data')
event_lst = ["flickr_3"]  # ["flickr_0", "flickr_2", "flickr_3"]
img_prefix='jpg'
input_size = dict(imgsize1=[320, 256], imgsize2=[192, 192])
num_classes = 1

topk = 1000
prob_threshold = 0.7
nms_threshold = 0.4
num_candidate = 1000
# mobilenet：[8, 16, 32]
steps = dict(slim=[8, 16, 32, 64], hisense=[],)
variances = dict(slim=[0.1, 0.2], hisense=[],)
# mobilenet: [[10, 20], [32, 64], [128, 256]]
min_sizes = dict(
    slim=[[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    hisense=[],
)
normalizations=dict(
    slim=[[104, 117, 123], [1.0, 1.0, 1.0]],
    hisense=[[120, 120, 120], [80.0, 80.0, 80.0]]
)

offline_quan_mode = False
offline_quan_tool = 'NCNN'
quan_table_path = 'work_dir/quan_table/NCNN/quantize.table'
results_path = 'work_dir/tmpfiles/{}_resluts'.format(task_name)
log_dir = 'work_dir/benchmark/log/{}'.format(task_name)
export_dir = 'work_dir/benchmark/export/{}'.format(task_name)

# is_remove_transpose = dict(slim=False, hisense=True)

### reference of accuracy/error
tables_head = 'TASK6: test_retina_face_detection'
tables = dict(
    flickr_imgsize1_slim_pre1_post1=[None, None, None],
)
layer_error = dict(
    flickr_imgsize1_slim_pre1_post1=dict(
        L1=[0, 0, 0], 
        L2=[0, 0, 0], 
        Cosine=[0, 0, 0]
    ),
    # flickr_imgsize2_hisense_pre2_post2=dict(
    #     L1=[0, 0, 0], 
    #     L2=[0, 0, 0], 
    #     Cosine=[0, 0, 0]
    # ), 
    # flickr_imgsize2_hisense_pre3_post3=dict(
    #     L1=[0, 0], 
    #     L2=[0, 0], 
    #     Cosine=[0, 0]
    # ),        
)
accuracy = dict(
    flickr_imgsize1_slim_pre1_post1=dict(AP=[0, 0, 0]),
    # flickr_imgsize2_hisense_pre2_post2=dict(AP=[0, 0]),
    # flickr_imgsize2_hisense_pre3_post3=dict(AP=[0, 0]),
) # reference AP of each model