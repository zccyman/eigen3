import numpy as np

try:
    from utils import generate_random
except:
    from onnx_converter.utils import generate_random # type: ignore

### conv input settings
has_bias = True
bias_method = "randn" if has_bias else "zeros"
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
selected_idx = 1
fuse_op = fuse_op_list[selected_idx]
act_attrs = act_attrs_list[selected_idx]
feat_i = [256, 320]
attrs = {
    "in_c": 3,
    "out_c": 32,
    "kernel_shape": [3, 3],
    "strides": [2, 2],
    "pads": [0, 0, 1, 1],
    "auto_pad": "SAME_UPPER",
    "dilations": [1, 1],
    "group": 1,
    "fuse_op": fuse_op,
    "act_attrs": act_attrs, 
    "isolated": False,
    "bias": has_bias,
}
xs = [
    generate_random(
        [
            1,
            attrs["in_c"],
            feat_i[0],
            feat_i[1],
        ], seed=0
    ).astype(np.float32)
]
ws = {
    "weight": [
        generate_random(
            [
                attrs["out_c"],
                attrs["in_c"],
                attrs["kernel_shape"][0],
                attrs["kernel_shape"][1],
            ], seed=1, is_weight=True
        ).astype(np.float32)
    ],
    "bias": [generate_random(attrs["out_c"], method=bias_method, seed=2, is_weight=True).astype(np.float32)],
}
conv = dict(attrs=attrs, xs=xs, ws=ws)  ### conv input settings


### depthwiseconv input settings
has_bias = True
bias_method = "randn" if has_bias else "zeros"
selected_idx = -3
fuse_op = fuse_op_list[selected_idx]
act_attrs = act_attrs_list[selected_idx]
feat_i = [256, 320]
attrs = {
    "in_c": 3,
    "out_c": 3,
    "kernel_shape": [3, 3],
    "strides": [2, 2],
    "pads": [0, 0, 1, 1],
    "dilations": [1, 1],
    "group": 3,
    "fuse_op": fuse_op,
    "act_attrs": act_attrs, 
    "isolated": False,
    "bias": has_bias,
}
xs = [
    generate_random(
        [
            1,
            attrs["in_c"],
            feat_i[0],
            feat_i[1],
        ], seed=0
    ).astype(np.float32)
]
ws = {
    "weight": [
        generate_random(
            [
                attrs["out_c"],
                1,
                attrs["kernel_shape"][0],
                attrs["kernel_shape"][1],
            ], seed=1, is_weight=True
        ).astype(np.float32)
    ],
    "bias": [generate_random(attrs["out_c"], method=bias_method, seed=2, is_weight=True).astype(np.float32)],
}
depthwiseconv = dict(attrs=attrs, xs=xs, ws=ws)  ### depthwiseconv input settings

### convtranspose input settings
has_bias = True
bias_method = "randn" if has_bias else "zeros"
selected_idx = -3
fuse_op = fuse_op_list[selected_idx]
act_attrs = act_attrs_list[selected_idx]
feat_i = [20, 20]
attrs = {
    "in_c": 3,
    "out_c": 32,
    "kernel_shape": [5, 5],
    "strides": [2, 2],
    "pads": [2, 2, 2, 2],
    "output_padding": [1, 1],
    "dilations": [1, 1],
    "group": 1,
    "fuse_op": fuse_op,
    "act_attrs": act_attrs, 
    "isolated": False,
    "bias": has_bias,
}
xs = [
    generate_random(
        [
            1,
            attrs["in_c"],
            feat_i[0],
            feat_i[1],
        ], seed=0
    ).astype(np.float32)
]
ws = {
    "weight": [
        generate_random(
            [
                attrs["in_c"],
                attrs["out_c"],
                attrs["kernel_shape"][0],
                attrs["kernel_shape"][1],
            ], seed=1, is_weight=True
        ).astype(np.float32)
    ],
    "bias": [generate_random(attrs["out_c"], method=bias_method, seed=2, is_weight=True).astype(np.float32)],
}
convtranspose = dict(attrs=attrs, xs=xs, ws=ws)  ### convtranspose input settings

### fc input settings
has_bias = True
bias_method = "randn" if has_bias else "zeros"
selected_idx = -1
fuse_op = fuse_op_list[selected_idx]
act_attrs = act_attrs_list[selected_idx]
attrs = {
    "in_c": 166, "out_c": 257, "fuse_op": fuse_op,
    "act_attrs": act_attrs, "isolated": False, "bias": has_bias,
}
xs = [generate_random([1, attrs["in_c"]], seed=0).astype(np.float32)]
ws = {
    "weight": [generate_random([attrs["out_c"], attrs["in_c"]], seed=1, is_weight=True).astype(np.float32)],
    "bias": [generate_random(attrs["out_c"], method=bias_method, seed=2, is_weight=True).astype(np.float32)],
}
fc = dict(attrs=attrs, xs=xs, ws=ws)  ### fc input settings


### matmul input settings
in_c = 3
feat_i0 = [36, 16]
feat_i1 = [16, 36]
attrs = {
    "in_c": in_c,
    "feat_i0": feat_i0,
    "feat_i1": feat_i1,
}
xs = [
    generate_random([1, in_c, feat_i0[0], feat_i0[1]], seed=0).astype(np.float32),
    generate_random([1, in_c, feat_i1[0], feat_i1[1]], seed=1).astype(np.float32),
]
ws = {}
matmul = dict(attrs=attrs, xs=xs, ws=ws)  
### matmul input settings


### elementwise input settings
feat_i = [112, 112]
channel = 64
xs = [
    1.0 * generate_random([1, channel, feat_i[0], feat_i[1]], method="randn", seed=0).astype(np.float32),
    0.5 * generate_random([1, channel, feat_i[0], feat_i[1]], method="rand", seed=1).astype(np.float32),
]
add = dict(xs=xs)
sub = add
pmul = add  ### elementwise input settings

### channelwise input settings
feat_i = [[112, 112], [1, 1]]
channel = 64
xs = [
    1.0 * generate_random([1, channel, feat_i[0][0], feat_i[0][1]], seed=0).astype(np.float32),
    1.5 * generate_random([1, channel, feat_i[1][0], feat_i[1][1]], seed=1).astype(np.float32),
]
cadd = dict(xs=xs)
csub = cadd
cmul = cadd  ### channelwise input settings

### lstm input settings
hidden_size = 128
has_init_h, has_init_c, has_bias = True, True, True
init_h_method = "randn" if has_init_h else "zeros"
init_c_method = "randn" if has_init_h else "zeros"
bias_method = "randn" if has_bias else "zeros"
initial_h = generate_random([1, 1, hidden_size], method=init_h_method, seed=0).astype(np.float32)
initial_c = generate_random([1, 1, hidden_size], method=init_c_method, seed=1).astype(np.float32)
is_update_quantize_from_in_data = False
attrs = {
    "bias": has_bias,
    "hidden_size": hidden_size,
    "sequence_lens": 1,
    "in_c": 257,
    "wr_combine": True,
    "hx_combine": False,
    "is_update_quantize_from_in_data": is_update_quantize_from_in_data,
    "initial_h": initial_h,
    "initial_c": initial_c,
}
xs = [
    generate_random([1, 1, attrs["in_c"]], seed=2).astype(np.float32),
    initial_h,
    initial_c,
]
ws = {
    "weight": [
        generate_random([1, hidden_size * 4, attrs["in_c"]], seed=3, is_weight=True).astype(np.float32),
        generate_random([1, hidden_size * 4, hidden_size], seed=4, is_weight=True).astype(np.float32),
    ],
    "bias": [
        generate_random([1, hidden_size * 4], method=bias_method, seed=5, is_weight=True).astype(np.float32),
        generate_random([1, hidden_size * 4], method=bias_method, seed=6, is_weight=True).astype(np.float32),
    ],
}
lstm = dict(attrs=attrs, xs=xs, ws=ws)  ### lstm input settings

### gru input settings
hidden_size = 128
has_init_h, has_bias = True, True
init_h_method = "randn" if has_init_h else "zeros"
bias_method = "randn" if has_bias else "zeros"
initial_h = generate_random([1, 1, hidden_size], method=init_h_method, seed=0).astype(np.float32)
attrs = {
    "bias": has_bias,
    "hidden_size": hidden_size,
    "sequence_lens": 1,
    "linear_before_reset": 1,
    "in_c": 257,
    "wr_combine": True,
    "hx_combine": False,
}
xs = [
    generate_random([1, 1, attrs["in_c"]], seed=1).astype(np.float32),
    initial_h,
]
ws = {
    "weight": [
        generate_random([1, hidden_size * 3, attrs["in_c"]], seed=2, is_weight=True).astype(np.float32),
        generate_random([1, hidden_size * 3, hidden_size], seed=3, is_weight=True).astype(np.float32),
    ],
    "bias": [
        1.0 * generate_random([1, hidden_size * 3], method=bias_method, seed=4, is_weight=True).astype(np.float32),
        1.0 * generate_random([1, hidden_size * 3], method=bias_method, seed=5, is_weight=True).astype(np.float32),
    ],
}
gru = dict(attrs=attrs, xs=xs, ws=ws)  ### gru input settings

### layernormalization input settings
attrs = {"axis": 1}
x_shape = [1, 1, 128]
w_shape = x_shape[attrs["axis"] :]
xs = [generate_random(x_shape, seed=0).astype(np.float32)]
ws = {
    "weight": [generate_random(w_shape, seed=1, is_weight=True).astype(np.float32)],
    "bias": [generate_random(w_shape, seed=2, is_weight=True).astype(np.float32)],
    "epsilon": 0.001,
}
layernormalization = dict(
    attrs=attrs, xs=xs, ws=ws
)  ### layernormalization input settings

### batchnormalization input settings
feat_i = [112, 112]
channel = 3
x_shape = [1, channel, feat_i[0], feat_i[1]]
w_shape = x_shape[1]
xs = [generate_random(x_shape, seed=0).astype(np.float32)]
ws = {
    "weight": [generate_random(w_shape, seed=1, is_weight=True).astype(np.float32)],
    "bias": [generate_random(w_shape, seed=2, is_weight=True).astype(np.float32)],
    "running_mean": [generate_random(w_shape, seed=3, is_weight=True).astype(np.float32)],
    "running_var": [generate_random(w_shape, seed=4, is_weight=True, range=[0, 1]).astype(np.float32)],
    "epsilon": 0.001,
}
batchnormalization = dict(xs=xs, ws=ws)  ### batchnormalization input settings

### instancenormalization input settings
feat_i = [112, 112]
channel = 3
x_shape = [1, channel, feat_i[0], feat_i[1]]
w_shape = x_shape[1]
xs = [generate_random(x_shape, seed=0).astype(np.float32)]
ws = {
    "weight": [generate_random(w_shape, seed=1, is_weight=True).astype(np.float32)],
    "bias": [generate_random(w_shape, seed=2, is_weight=True).astype(np.float32)],
    "epsilon": 0.001,
}
instancenormalization = dict(xs=xs, ws=ws)  ### instancenormalization input settings

### softmax input settings
x_shape = [1, 1000, 1, 1]
xs = [generate_random(x_shape, seed=0, range=[-25, 25]).astype(np.float32)]
attrs = {
    "axis": 1,
}
softmax = dict(attrs=attrs, xs=xs)  ### softmax input settings

### resize input settings
feat_i = [112, 112]
channel = 64
attrs = {
    "scale": [1, 1, 2, 2],
    "mode": "linear",
    "coordinate_transformation_mode": "align_corners",
}
# feat_i = [1, 1]
# channel = 128
# attrs = {
#     "sizes": [1, 128, 9, 16],
#     "mode": "linear",
#     "coordinate_transformation_mode": "half_pixel",
# }
# feat_i = [1, 1]
# channel = 128
# attrs = {
#     "scale": [1, 1, 2, 2],
#     "mode": "nearest",
#     "coordinate_transformation_mode": "asymmetric",
#     "nearest_mode": "floor",
#     "cubic_coeff_a": -0.75,
# }
xs = [generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32)]
resize = dict(attrs=attrs, xs=xs)  ### resize input settings


### globalaveragepool input settings
channel = 512
attrs = {
    "ceil_mode": True,
    "kernel_shape": [7, 7],
    "pads": [0, 0, 0, 0],
    "strides": [1, 1],
}
xs = [
    generate_random(
        [1, channel, attrs["kernel_shape"][0], attrs["kernel_shape"][1]], seed=0
    ).astype(np.float32)
]
globalaveragepool = dict(attrs=attrs, xs=xs)  ### globalaveragepool input settings


### averagepool input settings
channel = 512
attrs = {
    "ceil_mode": True,
    "kernel_shape": [2, 2],
    "pads": [1, 1, 1, 1],
    "strides": [2, 2],
}
xs = [
    generate_random(
        [1, channel, 12, 12], seed=0
    ).astype(np.float32)
]
averagepool = dict(attrs=attrs, xs=xs)  ### averagepool input settings


### reducemax input settings
feat_i = [112, 112]
channel = 64
attrs = {
    "axes": [1],
    "keepdims": 1,
}
xs = [generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32)]
reducemax = dict(attrs=attrs, xs=xs)
reducemin = dict(attrs=attrs, xs=xs)
reducemean = dict(attrs=attrs, xs=xs)
reducesum = dict(attrs=attrs, xs=xs)
reduceprod = dict(attrs=attrs, xs=xs)
### reducemax input settings


### transpose input settings
feat_i = [24, 12]
channel = 64
attrs = {
    "perm": [0, 1, 3, 2],
}
xs = [generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32)]
transpose = dict(attrs=attrs, xs=xs)
### transpose input settings


### reshape input settings
feat_i = [2, 2]
channel = 128
attrs = {
    "shape": [0, 8, 16, -1],
}
xs = [generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32)]
reshape = dict(attrs=attrs, xs=xs)
### reshape input settings

### log input settings
xs = [generate_random([128, 32, 32], seed=0).astype(np.float32)]
ys = [np.abs(xs) for item in xs]
log = dict({}, xs=ys)
### log input settings

### pad input settings
feat_i = [64, 64]
channel = 24
attrs = {
    "mode": "constant",
    "pads": [0, 0, 0, 0, 0, 24, 0, 0],
}
xs = [generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32)]
pad = dict(attrs=attrs, xs=xs)
### pad input settings


### maxpool input settings
feat_i = [112, 112]
channel = 64
attrs = {
    "ceil_mode": False,
    "kernel_shape": [3, 3],
    "pads": [1, 1, 1, 1],
    "strides": [2, 2],
}
xs = [generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32)]
maxpool = dict(attrs=attrs, xs=xs)
### maxpool input settings

### shuffle_only input settings
attrs = {
    "in_channels": [58, 58],
    "out_channels": [58, 58],
    "shape1": [1, 2, 58, 40, 40],
    "perm": [0, 2, 1, 3, 4],
    "shape2": [1, -1, 40, 40],
}
channel = attrs["shape1"][1] * attrs["shape1"][2]
feat_i = attrs["shape1"][-2:]
xs = [
    generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32),
]
shuffle_only = dict(attrs=attrs, xs=xs)  ### shuffle_only input settings


### shuffle input settings
channel = 58
attrs = {
    "axis": 1,
    "input_len": 2,
    "shape1": [1, 2, channel, 40, 40],
    "perm": [0, 2, 1, 3, 4],
    "shape2": [1, -1, 40, 40],
    "split": [channel, channel],
}
feat_i = attrs["shape1"][-2:]
xs = [
    generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32),
    generate_random([1, channel, feat_i[0], feat_i[1]], seed=1).astype(np.float32),
]
shuffle = dict(attrs=attrs, xs=xs)  ### shuffle input settings

### concat input settings
feat_i = [112, 112]
channels = [12, 14]
attrs = {"axis": 1, "input_len": 2}
xs = [
    1.0 * generate_random([1, channels[0], feat_i[0], feat_i[1]], seed=0).astype(np.float32),
    1.5 * generate_random([1, channels[1], feat_i[0], feat_i[1]], seed=1).astype(np.float32),
]
concat = dict(attrs=attrs, xs=xs)  ### concat input settings

### split input settings
feat_i = [112, 112]
attrs = {"axis": 1, "split": [12, 24]}
xs = [
    generate_random([1, np.array(attrs["split"]).sum(), feat_i[0], feat_i[1]], seed=0).astype(
        np.float32
    ),
]
split = dict(attrs=attrs, xs=xs)  ### split input settings

### activation input settings
feat_i = [112, 112]
channel = 64
xs = [generate_random([1, channel, feat_i[0], feat_i[1]], seed=0).astype(np.float32)]
leakyrelu = dict(attrs={"isolated": True, "alpha": 0.001}, xs=xs)
slope = generate_random(channel, seed=1).astype(np.float32)
prelu = dict(attrs={"isolated": True, "slope": slope}, xs=xs)
relux = dict(attrs={"isolated": True, "value": 12.0}, xs=xs)
relu6 = dict(attrs={"isolated": True, "value": 6.0}, xs=xs)
sigmoid = dict(attrs={"isolated": True}, xs=xs)
swish = sigmoid
relu = sigmoid
gelu = sigmoid
tanh = sigmoid
hardsigmoid = dict(attrs={"isolated": True, "alpha": 0.2, "beta": 0.5}, xs=xs)
hardtanh = sigmoid
hardswish = sigmoid
hardshrink = sigmoid  ### activation input settings
