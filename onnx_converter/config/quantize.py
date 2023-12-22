# quantize for ops
# import numpy as np

bits_dict = {0: 'np.uint8', 1: 'np.int8', 2: 'np.uint16', 3: 'np.int16', 4: 'np.uint32', 5: 'np.int32', 6: 'np.uint64', 7: 'np.int64', 8: 'np.float32', 9: 'np.float64'}
# bits_dict = {0: 'uint4', 1: 'int4', 1: 'np.int8', 2: 'np.int8', 3: 'np.int16', 4: 'np.int16', 5: 'np.int32', 6: 'np.int32', 7: 'np.int64', 8: 'np.int64', 9: 'np.float32', 10: 'np.float64'}
# question ?
# precision: 0--hardward calc biasadd/activation, 1--software calc biasadd/activation
# bit_select, int_scale, pre_int_scale, out_type = 3, 16, 16-1, 3
bit_select, int_scale, pre_int_scale, out_type = 1, 8, 8-1, 1
txme_saturation = 1
virtual_round = 3
# if is_last_layer_fuse_act=True, nonlinear act will be fused into conv/fc and process_scale is shiftfloattable;
# if is_last_layer_fuse_act=True, linear act will be fused into conv/fc and process_scale is intscale/.../...;
is_last_layer_fuse_act = True
is_fuse_linear_act = True # if is_fuse_linear_act=True, use shiftfloattable, otherwise use intscale;
is_fuse_nonlinear_act = True # if is_fuse_nonlinear_act=True, use shiftfloattable, otherwise nonlinear act will not be fused into conv/fc;
act = []
if is_fuse_nonlinear_act:
    fuse_act = ["swish", "leakyrelu", "hardswish", "hardsigmoid", "tanh", "sigmoid"]
else:
    fuse_act = []
if is_fuse_linear_act:
    fuse_act.extend(["relu", "relu6", "relux", "act"])
else:
    act.extend(["relu", "relu6", "relux", "act"])
search_smaller_sk = False
reload_sk_params = True # False | True
# sk_params_json_path = "sk_params_99.json"
# sk_params_json_path = "work_dir/qat/test_imagenet/sk_params_56.json"
sk_params_json_path = "work_dir/qat/test_hand_detection/sk_params_340.json"
maxs = {0: 255, 1: 127, 2: 65535, 3: 32767, 4: 4294967295, 5: 2147483647, 6: 1844674407370955161, 7: 9223372036854775807}
mins = {0: 0, 1: -128, 2: 0, 3: -32768, 4: 0, 5: -2147483648, 6: 0, 7: -9223372036854775808}
# maxs = {0: 127, 1: 127, 2: 32767, 3: 32767, 4: 2147483647, 5: 2147483647}
# mins = {0: -128, 1: -128, 2: -32768, 3: -32768, 4: -2147483648, 5: -2147483648}
bit_lower = 0

normal = dict(method='floatsymquan', # floatsymquan, floatquan
              bit_select=bit_select,
              maxs=maxs,
              mins=mins,
              bits_dict=bits_dict)

perchannel = dict(method='perchannelfloatsymquan',
                        bit_select=bit_select,
                        maxs=maxs,
                        mins=mins,
                        bits_dict=bits_dict)

int16 = dict(method='floatsymquan',
                  bit_select=3,
                  maxs=maxs,
                  mins=mins,
                  bits_dict=bits_dict)

feat = dict(method='floatsymquan',
            bit_select=bit_select,
            maxs=maxs,
            mins=mins,
            bits_dict=bits_dict)

asym_feat = dict(method='floatsymquan',
            bit_select=bit_select,
            maxs=maxs,
            mins=mins,
            bits_dict=bits_dict)

# now not support this method
# # todo maybe will implement soon
# shuffle=dict(weights=None,
#              feat=[feat, feat],
#              process_scale=['preintscaleexex', 'preintscaleexex'],
#              int_scale=[int_scale, int_scale],
#              out_type=[out_type, out_type])

# add=dict(weights=None,
#          feat=[feat, feat],
#          process_scale=['preintscaleexex', 'preintscaleexex'],
#          int_scale=[int_scale, int_scale],
#          out_type=[out_type, out_type])

# concat=dict(weights=None,
#             feat=[feat, feat],
#             process_scale=['preintscaleexex', 'preintscaleexex'],
#              int_scale=[int_scale, int_scale],
#              out_type=[out_type, out_type])

# split=dict(weights=None,
#            feat=[feat, feat],
#            process_scale=['preintscaleexex', 'preintscaleexex'],
#            int_scale=[int_scale, int_scale],
#            out_type=[out_type, out_type])

# mul=dict(weights=None,
#          feat=[feat, feat],
#          process_scale=['preintscaleexex', 'preintscaleexex'],
#          int_scale=[int_scale, int_scale],
#          out_type=[out_type, out_type])

output = dict(layer_type=['conv', 'depthwiseconv', 'gemm', 'matmul', 'fc'],
            #   weights=dict(method='floatquan', bit_select=bit_select),
            #   feat=dict(method='floatsymquan', bit_select=bit_select),
              process_scale='shiftfloatscaletable2float', out_type=out_type)

# output = dict(layer_type=['conv', 'depthwiseconv', 'resize', 'matmul', 'fc'], weights='floatsymquan', feat='floatsymquan', process_scale='rshiftscale')
# output = dict(layer_type=['conv', 'depthwiseconv', 'resize', 'matmul', 'fc'], weights='floatsymquan', feat='floatsymquan', process_scale='rrshiftscale')
# output = dict(layer_type=['conv', 'depthwiseconv', 'resize', 'fc', 'mul'], weights='floatsymquan', feat='floatsymquan', process_scale='intscale')

# precision: 0-software implement bias-add and activation, convolution
#            output maybe int32/int64, out shift as software input parameter
# precision: 1-hardware implement bias-add and activation, convolution output maybe int8/int16,
#            out shift is post-shift
default_setting = dict(data=dict(weights=None, feat=feat, process_scale='floatscale', int_scale=int_scale, out_type=out_type),
                       conv=dict(weights=normal, feat=feat, process_scale='intscale', int_scale=int_scale, out_type=out_type),
                       depthwiseconv=dict(weights=normal, feat=feat, process_scale='intscale', int_scale=int_scale, out_type=out_type),
                       convtranspose=dict(weights=normal, feat=feat, process_scale='intscale', int_scale=int_scale, out_type=out_type),
                       fc=dict(weights=normal, feat=feat, process_scale='floatscale', int_scale=int_scale, out_type=out_type),
                       gemm=dict(weights=normal, feat=feat, process_scale='floatscale', int_scale=int_scale, out_type=out_type),
                       matmul=dict(weights=normal, feat=feat, process_scale='floatscale', int_scale=int_scale, out_type=out_type),
                       lstm=dict(weights=normal, feat=feat, process_scale='ffloatscale', int_scale=int_scale, out_type=out_type, hx_combine=False, wr_combine=False),
                       gru=dict(weights=normal, feat=feat, process_scale='ffloatscale', int_scale=int_scale, out_type=out_type, hx_combine=False, wr_combine=False),
                       mul=dict(weights=normal, feat=feat, process_scale='intscale', int_scale=int_scale, out_type=out_type),
                       cmul=dict(weights=normal, feat=feat, process_scale='intscale', int_scale=int_scale, out_type=out_type),
                       pmul=dict(weights=normal, feat=feat, process_scale='intscale', int_scale=int_scale, out_type=out_type),
                       shuffle=dict(weights=None, feat=feat, process_scale='preintscaleex', int_scale=pre_int_scale, out_type=out_type),
                       concat_shuffle_only=dict(weights=None, feat=feat, process_scale='preintscaleex', int_scale=pre_int_scale, out_type=out_type),
                       shuffle_only=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type),
                       reducemin=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       reducemax=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       reducemean=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       reducesum=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       reduceprod=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       sub=dict(weights=normal, feat=feat, process_scale='preintscaleex', int_scale=int_scale, out_type=out_type),
                       csub=dict(weights=normal, feat=feat, process_scale='preintscaleex', int_scale=int_scale, out_type=out_type),
                       add=dict(weights=normal, feat=feat, process_scale='preintscaleex', int_scale=pre_int_scale, out_type=out_type),
                       cadd=dict(weights=normal, feat=feat, process_scale='preintscaleex', int_scale=pre_int_scale, out_type=out_type),
                       concat=dict(weights=None, feat=feat, process_scale='preintscaleex', int_scale=pre_int_scale, out_type=out_type),
                       maxpool=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type),
                       averagepool=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type),
                       globalaveragepool=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type),
                       resize=dict(weights=None, feat=feat, process_scale='preintscaleex', int_scale=int_scale, out_type=out_type),
                       split=dict(weights=None, feat=feat, process_scale='preintscaleex', int_scale=pre_int_scale, out_type=out_type),
                       batchnormalization=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       layernormalization=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       instancenormalization=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       slice=dict(weights=None, feat=feat, process_scale='preintscaleex', int_scale=pre_int_scale, out_type=out_type),
                       reshape=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type),
                       pad=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type),
                       flatten=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type),
                       transpose=dict(weights=None, feat=feat, process_scale='smooth',int_scale=int_scale, out_type=out_type),
                       relu=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       relu6=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       relux=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       sigmoid=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       gelu=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       leakyrelu=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       prelu=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       tanh=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       hardsigmoid=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       hardswish=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       swish=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       lrn=dict(weights=None, feat=feat, process_scale='intscale', int_scale=int_scale, out_type=out_type),
                       dropout=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type),
                       softmax=dict(weights=None, feat=feat, process_scale='table', int_scale=int_scale, out_type=out_type),
                       log=dict(weights=None, feat=feat, process_scale='float', int_scale=int_scale, out_type=out_type),
                       default=dict(weights=None, feat=feat, process_scale='smooth', int_scale=int_scale, out_type=out_type))