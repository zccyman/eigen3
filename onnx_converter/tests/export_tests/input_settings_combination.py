import numpy as np

try:
    from utils import generate_random
except:
    from onnx_converter.utils import generate_random # type: ignore

fuse_op_list = [
    [], ["relu"], ["relu6"], ["relux"], 
    ["swish"], ["leakyrelu"], ["hardswish"], ["hardsigmoid"],
    ["tanh"], ["sigmoid"],
]
act_attrs_list = [
    dict(), dict(), dict(value=6.0), dict(value=12.0), 
    dict(), dict(alpha=0.01), dict(), dict(alpha=0.2, beta=0.5),
    dict(), dict(),
]
# conv input settings combination
conv = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "feat_i": ([[320, 256]], True),
    "batch_size": ([1], True),
    "in_c": ([3], True),
    "out_c": ([32], True),
    "kernel_shape": ([[3, 3]], True),
    "strides": ([[2, 2]], True),
    "pads": ([[0, 0, 1, 1]], True),
    "auto_pad": (["SAME_UPPER", "SAME_LOWER", "VALID"], True),
    "dilations": ([[1, 1]], True),
    "group": ([1], True),
    "fuse_op": (fuse_op_list, False),
    "act_attrs": (act_attrs_list, False),
    "isolated": ([False], True),
    "has_bias": ([True, False], True),
}
# conv input settings combination

# depthwiseconv input settings combination
depthwiseconv = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "feat_i": ([[320, 256]], True),
    "batch_size": ([1], True),
    "in_c": ([12], True),
    "out_c": ([12], True),
    "kernel_shape": ([[3, 3]], True),
    "strides": ([[2, 2]], True),
    "pads": ([[0, 0, 1, 1]], True),
    "dilations": ([[1, 1]], True),
    "group": ([12], True),
    "fuse_op": (fuse_op_list, False),
    "act_attrs": (act_attrs_list, False),
    "isolated": ([False], True),
    "has_bias": ([True, False], True),
}
# depthwiseconv input settings combination

# convtranspose input settings combination
convtranspose = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "feat_i": ([[20, 12], [20, 48]], True),
    "batch_size": ([1], True),
    "in_c": ([3], True),
    "out_c": ([32], True),
    "kernel_shape": ([[5, 5]], True),
    "strides": ([[2, 2]], True),
    "pads": ([[2, 2, 2, 2]], True),
    "output_padding": ([[1, 1]], True),
    "dilations": ([[1, 1]], True),
    "group": ([1], True),
    "fuse_op": (fuse_op_list, False),
    "act_attrs": (act_attrs_list, False),
    "isolated": ([False], True),
    "has_bias": ([True, False], True),
}
# convtranspose input settings combination


# fc input settings combination
fc = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "in_c": ([512], True),
    "out_c": ([512], True),
    "fuse_op": (fuse_op_list, False),
    "act_attrs": (act_attrs_list, False),
    "isolated": ([False], True),
    "has_bias": ([True, False], True),
}
# fc input settings combination


### matmul settings combination
matmul = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "in_c": ([3], True),
    "feat_i0": ([[32, 16], [22, 12]], False),
    "feat_i1": ([[16, 32], [12, 22]], False),
}
### matmul settings combination


# elementwise input settings combination
add = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "feat_i": ([[112, 112]], True),
    "channel": ([64], True),
}
sub = add
pmul = add
# elementwise input settings combination


# channelwise input settings combination
cadd = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "feat_i0": ([[112, 112]], True),
    "feat_i1": ([[1, 1]], True),
    "channel": ([64], True),
}
csub = cadd
cmul = cadd
# channelwise input settings combination

# lstm input settings combination
lstm = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "has_bias": ([True], True),
    "hidden_size": ([128], True),
    "sequence_lens": ([1], True),
    "in_c": ([257], True),
    "wr_combine": ([True, False], True),
    "hx_combine": ([True, False], True),
    "is_update_quantize_from_in_data": ([True, False], True),
    "has_init_h": ([True], True),
    "has_init_c": ([True], True),
}
# lstm input settings combination

# gru input settings combination
gru = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "has_bias": ([True], True),
    "hidden_size": ([128], True),
    "sequence_lens": ([1], True),
    "linear_before_reset": ([0, 1], True),
    "in_c": ([257], True),
    "wr_combine": ([True, False], True),
    "hx_combine": ([True, False], True),
    "has_init_h": ([True], True),
}
# gru input settings combination


# layernormalization input settings combination
layernormalization = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "feat_i": ([[1, 1, 128]], True),
    "axis": ([1], True),
    "epsilon": ([0.001], True),
}
# layernormalization input settings combination


# batchnormalization input settings combination
batchnormalization = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([3], True),
    "feat_i": ([[112, 112]], True),
    "epsilon": ([0.001], True),
}
# batchnormalization input settings combination


# instancenormalization input settings combination
instancenormalization = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([3], True),
    "feat_i": ([[112, 112]], True),
    "epsilon": ([0.001], True),
}
# instancenormalization input settings combination


# softmax input settings combination
softmax = {
    "method": (["randn"], True),
    "range": ([[-25, 25]], True),
    "batch_size": ([1], True),
    "channel": ([1000], True),
    "feat_i": ([[1, 1]], True),
    "axis": ([1], True),
}
# softmax input settings combination

# resize input settings combination
resize = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([3], True),
    "feat_i": ([[112, 112]], True),
    "sizes": ([[224, 224]], True),
    "scale": ([[1, 1, 2, 2]], True),
    "mode": (["linear", "nearest"], True), # "cubic" , "linear", "nearest"
    "coordinate_transformation_mode": (["align_corners", "asymmetric", "half_pixel"], True),
    "cubic_coeff_a": ([-0.5, -0.75], True),
    "nearest_mode": (["floor", "ceil"], True),
}
# resize input settings combination


# reducemax input settings combination
reducemax = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([64], True),
    "feat_i": ([[112, 112]], True),
    "axes": ([[1]], True),
    "keepdims": ([1], True),
}
reducemin = reducemax
reducemean = reducemax
reducesum = reducemax
reduceprod = reducemax
# reducemax input settings combination


# transpose input settings combination
reducemax = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([64], True),
    "feat_i": ([[24, 12]], True),
    "perm": ([[0, 1, 3, 2]], True),
}
# transpose input settings combination


# reshape input settings combination
reshape = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([128], True),
    "feat_i": ([[2, 2]], True),
    "shape": ([[0, 8, 16, -1]], True),
}
# reshape input settings combination


# pad input settings combination
pad = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([24], True),
    "feat_i": ([[64, 64]], True),
    "mode": (["constant"], True), # "reflect", "edge", "warp", "constant", 
    "pads": ([[0, 0, 0, 0, 0, 24, 0, 0]], True),
}
# pad input settings combination


# averagepool input settings combination
averagepool = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([3], True),
    "ceil_mode": ([True], True),
    "kernel_shape": ([[2, 2]], True),
    "pads": ([[1, 1, 1, 1]], True),
    "strides": ([[2, 2]], True),
}
# averagepool input settings combination


# globalaveragepool input settings combination
globalaveragepool = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([3], True),
    "ceil_mode": ([True], True),
    "kernel_shape": ([[7, 7]], True),
    "pads": ([[0, 0, 0, 0]], True),
    "strides": ([[1, 1]], True),
}
# globalaveragepool input settings combination


# maxpool input settings combination
maxpool = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([3], True),
    "feat_i": ([[112, 112]], True),
    "ceil_mode": ([False], True),
    "kernel_shape": ([[3, 3]], True),
    "pads": ([[1, 1, 1, 1]], True),
    "strides": ([[2, 2]], True),
}
# maxpool input settings combination


# shuffle_only input settings combination
shuffle_only = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([116], True),
    "in_channels": ([[58, 58], [56, 60], [58*2]], True),
    "out_channels": ([[58, 58], [56, 60], [58*2]], True),
    "feat_i": ([[40, 40]], True),
    "perm": ([[0, 2, 1, 3, 4]], True),
}
# shuffle_only input settings combination


# shuffle input settings combination
shuffle = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([58], True),
    "feat_i": ([[40, 40]], True),
    "axis": ([1], True),
    "input_len": ([2], True),
    "perm": ([[0, 2, 1, 3, 4]], True),
}
# shuffle input settings combination


# concat input settings combination
concat = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channels": ([[12, 14]], True),
    "feat_i": ([[112, 112]], True),
    "axis": ([1], True),
    "input_len": ([2], True),
}
# concat input settings combination


# split input settings combination
split = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "feat_i": ([[112, 112]], True),
    "axis": ([1], True),
    "split": ([[12, 24]], True),
}
# split input settings combination


# activation input settings combination
leakyrelu = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([64], True),
    "feat_i": ([[112, 112]], True),
    "isolated": ([True], True),
    "alpha": ([0.001], True),
}

slope = generate_random(64, seed=1).astype(np.float32)
prelu = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([64], True),
    "feat_i": ([[112, 112]], True),
    "isolated": ([True], True),
    "slope": ([slope, ], True),
}

relux = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([64], True),
    "feat_i": ([[112, 112]], True),
    "isolated": ([True], True),
    "value": ([12.0], True),
}

relu6 = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([64], True),
    "feat_i": ([[112, 112]], True),
    "isolated": ([True], True),
    "value": ([6.0], True),
}

sigmoid = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([64], True),
    "feat_i": ([[112, 112]], True),
    "isolated": ([True], True),
}

hardsigmoid = {
    "method": (["randn"], True),
    "range": ([[-1, 1]], True),
    "batch_size": ([1], True),
    "channel": ([64], True),
    "feat_i": ([[112, 112]], True),
    "isolated": ([True], True),
    "alpha": ([0.2], True),
    "beta": ([0.5], True),
}

swish = sigmoid
relu = sigmoid
gelu = sigmoid
tanh = sigmoid
hardtanh = sigmoid
hardswish = sigmoid
hardshrink = sigmoid
# activation input settings combination
