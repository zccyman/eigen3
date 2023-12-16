#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/03/29 19:43:27
#@File    : imagenet.py

_base_ = ['./base/base.py']

MR_model = dict(
    MR_DEV=['resnet18_cls1000.onnx'],
    MR_MASTER=['resnet18_cls1000.onnx'],
    MR_RELEASE=[
        'resnet18_cls1000.onnx', 
        'resnet34_cls1000.onnx'
        ],
    MR_OTHER=[
        'resnet18_cls1000.onnx', 
        'resnet34_cls1000.onnx'
        ],       
)

### dataset, imgsize, network, preprocess, postprocess 
model_paths = dict(
    imagenet_imgsize1_resnet_pre1_post1=[
    'resnet18_cls1000.onnx',
    'resnet34_cls1000.onnx'
    ],
    # imagenet_imgsize1_vgg_pre1_post1=[
    # 'vgg11_cls1000.onnx',
    # 'vgg13_cls1000.onnx',
    # ]
)
task_name = 'classification'  

input_size = dict(imgsize1=[224, 224])
dataset_dir = dict(imagenet='imagenet2012')
img_prefix = 'JPEG'
selected_sample_num_per_class = dict(MR_RELEASE=1, MR_MASTER=1, MR_DEV=1, MR_OTHER=1) ###50
image_subdir = [idx for idx in range(0, 1000, 1)]
class_num = 1000

offline_quan_mode = False
offline_quan_tool = 'NCNN'
quan_table_path = 'work_dir/quan_table/NCNN/quantize.table'
log_dir = 'work_dir/benchmark/log/{}'.format(task_name)
export_dir = 'work_dir/benchmark/export/{}'.format(task_name)

# is_remove_transpose = dict(resnet=False, vgg=False)

### reference of accuracy/error
tables_head = 'TASK3: test_imagenet'
tables = dict(
    imagenet_imgsize1_resnet_pre1_post1=[None, None],
)
layer_error = dict(
    imagenet_imgsize1_resnet_pre1_post1=dict(
        L1=[0, 0],
        L2=[0, 0],
        Cosine=[0, 0],      
    ),
    # imagenet_imgsize1_vgg_pre1_post1=dict(
    #     L1=[0, 0], 
    #     L2=[0, 0], 
    #     Cosine=[0, 0]
    # ),    
)
accuracy = dict(
    imagenet_imgsize1_resnet_pre1_post1=dict(
        top1=[0, 0],
        top5=[0, 0],            
    ),
    # imagenet_imgsize1_vgg_pre1_post1=dict(top1=[0, 0], top5=[0, 0])
)
