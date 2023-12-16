#Copyright (c) shiqing. All rights reserved.
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
#@Author  : henson.zhang
#@Company : SHIQING TECH
#@Time    : 2022/05/13 19:43:27
#@File    : base.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

#floatsymquan, floatquan, perchannelfloatsymquan, perchannelfloatquan
quantize_methods = {
    'floatsymquan': ['floatsymquan', 'perchannelfloatsymquan'], #, 'perchannelfloatsymquan'
    'floatquan': ['floatquan', 'perchannelfloatquan'],
}
# bits_dict = {0: 'np.uint8', 1: 'np.int8', 2: 'np.uint16', 3: 'np.int16', 4: 'np.uint32', 5: 'np.int32', 6: 'np.uint64', 7: 'np.int64', 8: 'np.float32', 9: 'np.float64'}
quantize_dtypes = [1, 3]
process_scale = {'weight': ['intscale', 'floatscale', 'shiftfloatscale']}

MR_quantize = dict(
    MR_OTHER=['floatsymquan', 1, 'intscale'],
    MR_DEV=['floatsymquan', 'perchannelfloatsymquan', 'floatquan', 
              1, 3, 'intscale', 'floatscale'],
    MR_MASTER=['floatsymquan', 'perchannelfloatsymquan', 'floatquan', 'perchannelfloatquan', 
                1, 3, 'intscale', 'floatscale'],
    MR_RELEASE=['floatsymquan', 'perchannelfloatsymquan', 'floatquan', 'perchannelfloatquan', 
                1, 3, 'intscale', 'floatscale', 'shiftfloatscale']
)

draw_result = False
fp_result = True
error_metric = ['L1', 'L2', 'Cosine']
device = "cuda:3"

export_version = 2
export = False
is_calc_error = False
log_level = 30
acc_error = True

generate_experience_value = False
variable_rate = 0.1
is_stdout = False
eval_first_frame = False
eval_mode = 'single' if eval_first_frame else 'dataset'
is_assert = not eval_first_frame
if generate_experience_value:
    report_path = 'benchmark/benchmark_config/experience'
else:
    report_path = 'work_dir/benchmark/report'