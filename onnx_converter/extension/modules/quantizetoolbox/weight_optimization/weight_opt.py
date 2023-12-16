import numpy as np
import math
import torch
import copy


def SQuant(W):
    M, N, K = W.shape
    E = np.zeros(W.shape)
    deltaE = np.zeros(W.shape)
    K_ = np.zeros([M, N, K])
    deltaK_ = np.zeros([M, N, K])
    C = np.zeros([M, N, K])
    for m in range(M):
        for n in range(N):
            for i in range(K):
                E[m, n, i] = np.round(W[m, n, i])
                deltaE[m, n, i] = E[m, n, i] - W[m, n, i]
            K_[m, n, :] = SQuantFlip(E[m, n, :], deltaE[m, n, :])
            deltaK_[m, n, :] = UpdatePerturbation(deltaE[m, n, :])
        C[m, :] = SQuantFlip(K_[m, :], deltaK_[m, :])

    return K_


def SQuantFlip(w, p):
    shape = w.shape
    if len(shape) == 2:
        w = w.reshape(shape[0] * shape[1])
        p = p.reshape(shape[0] * shape[1])

    e = np.sum(p)
    p[e * p < 0] = 0
    k = int(np.round(np.abs(e)))
    f = np.argsort(np.abs(p))[::-1][:k]

    delta = []
    for d1 in p[f]:
        if d1 < 0:
            delta.append(1)
        else:
            delta.append(-1)

    w[f] = w[f] + delta  # flip

    if len(shape) == 2:
        w = w.reshape(shape[0], shape[1])

    return w


def UpdatePerturbation(p):
    e = np.sum(p)
    p[e * p < 0] = 0
    k = int(np.round(np.abs(e)))
    f = np.argsort(np.abs(p))[::-1][:k]
    if k > np.abs(e):
        i = f[-1]
    else:
        i = (np.argsort(np.abs(p))[::-1][:k + 1])[-1]
    v = p[i]

    p[:] = 0
    p[i] = v

    return p


def SQuantW(weight, asymmetric=False):
    C = weight
    datatype = C.dtype
    if len(weight.shape) == 4:
        out_c, in_c, k_h, k_w = weight.shape
        if asymmetric:
            min_value, max_value = np.min(weight.reshape(-1),
                                          axis=0), np.max(weight.reshape(-1),
                                                          axis=0)
            scale = (max_value - min_value) / 255.0
            zero_point = np.clip(-128 - np.round(min_value / scale), -128, 127)
        else:
            scale = np.max(np.abs(weight)) / 127.0
            zero_point = 0
        weight = weight / scale + zero_point
        weight = weight.reshape(out_c, in_c, -1)
        C = SQuant(weight)
        C = C.reshape(out_c, in_c, k_h, k_w)
        C = np.clip(np.round(C), -128, 127)
        C = (C - zero_point) * scale
    elif len(weight.shape) == 2:
        out_c, in_c = weight.shape
        if asymmetric:
            min_value, max_value = np.min(weight.reshape(-1),
                                          axis=0), np.max(weight.reshape(-1),
                                                          axis=0)
            scale = (max_value - min_value) / 255.0
            zero_point = np.clip(-128 - np.round(min_value / scale), -128, 127)
        else:
            scale = np.max(np.abs(weight)) / 127.0
            zero_point = 0
        weight = weight / scale + zero_point
        weight = weight.reshape(out_c, in_c, 1)
        C = SQuant(weight)
        C = C.reshape(out_c, in_c)
        C = np.clip(np.round(C), -128, 127)
        C = (C - zero_point) * scale

    return C.astype(datatype)


def admm(weight, quan_method, iters=1000):
    quan_method.get_quan_param(weight)
    qweight = quan_method.get_quan_data(weight)

    i = 0
    while i < iters:
        qweight = qweight.astype(np.float32)
        s0 = np.sum(weight * qweight) / np.sum(qweight * qweight)
        quan_method.set_scale(dict(scale=s0, zero_point=0))
        qweight = quan_method.get_quan_data(weight)
        i += 1

    weight = quan_method.get_dequan_data(qweight)

    return weight, quan_method


def Cosine_distance(simulation_data, true_data, eps=1.0e-5):
    s_data = copy.deepcopy(simulation_data).astype(np.float32)
    t_data = copy.deepcopy(true_data).astype(np.float32)
    s_data = torch.from_numpy(s_data.reshape(-1))
    t_data = torch.from_numpy(t_data.reshape(-1))
    normal = torch.sqrt(torch.sum(s_data * s_data) * torch.sum(t_data * t_data))
    dist = torch.sum(s_data * t_data) / (normal + eps)
    dist = (1 - np.abs(dist.item())) * 100

    return np.float32(dist)


def percentile(data, quan_method, selected_idx=0):
    max_vlist = []

    bins = 2048 * 8
    # data_ = data[data != 0]
    calib_hist, calib_bin_edges = np.histogram(np.abs(data.reshape(-1)),
                                               bins=bins)
    calib_hist = calib_hist / calib_hist.sum()
    cdf = np.cumsum(calib_hist)
    percentiles = [
        96, 96.5, 97, 97.5, 98, 98.5, 99, 99.5, 99.9, 99.99, 99.999, 99.9999, 99.99999
    ] #99.5, 99.6, 99.7, 99.8,
    for percentile in percentiles:
        idx = np.searchsorted(cdf, percentile / 100)
        calib_amax = calib_bin_edges[idx]
        max_vlist.append([-calib_amax, calib_amax])

    error_list = []
    scale_list = []
    for min_v, max_v in max_vlist:
        quan_method.set_scale(dict(min=min_v, max=max_v, zeros_point=0))
        qdata = quan_method.get_quan_data(copy.deepcopy(data))
        qdata = quan_method.get_dequan_data(qdata)
        error = Cosine_distance(qdata, data)
        # qhist, _ = np.histogram(qdata.reshape(-1), bins=bins)
        # hist, _ = np.histogram(data.reshape(-1), bins=bins)
        # qhist = qhist / qhist.sum()
        # hist = hist / hist.sum()
        # error = 1.0 - np.sum(np.sqrt(qhist*hist))
        error_list.append(error)
        scale_list.append(dict(min=min_v, max=max_v, zeros_point=0))

    selected_idx = np.argmin(error_list)
    scale = scale_list[selected_idx]
    quan_method.set_scale(scale)
    print("=>>> selected_idx: ", selected_idx, scale)

    return quan_method
    # return scale


def fit_distribution(data, quan_method, method="cauchy"):
    import scipy.special as sp
    from sympy import symbols, solve, log, exp
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    try:
        from extension import Fitter
    except:
        from onnx_converter.extension import Fitter

    if "perchannel" in quan_method.get_class_name().lower():
        data_ = copy.deepcopy(data)
        params = np.ones(data_.shape[0], dtype=np.float32)
        for i in range(data_.shape[0]):
            f = Fitter(data_[i, :], distributions=[method], bins=2048, timeout=30)
            f.fit()
            f.summary(method="cosine_error")
            if method == "cauchy":
                params[i] = f.fitted_param[method][1] #6.894022024947274e-06
            elif method == "dweibull":
                params[i] = f.fitted_param[method][1]
            elif method == "expon":
                params[i] = 1.0 / f.fitted_param[method][0]
            elif method == "laplace":
                params[i] = f.fitted_param[method][1]
    else:
        f = Fitter(copy.deepcopy(data).reshape(-1), distributions=[method], bins=2048, timeout=30)
        f.fit()
        f.summary(method="cosine_error") #返回排序好的分布拟合质量（拟合效果从好到坏）,并绘制数据分布和Nbest分布
        # f.get_best(method='kl_div')
        if method == "cauchy":
            params = f.fitted_param[method][1] #6.894022024947274e-06
        elif method == "dweibull":
            params = f.fitted_param[method][1]
        elif method == "expon":
            params = 1.0 / f.fitted_param[method][0]
        elif method == "laplace":
            params = f.fitted_param[method][1]

    # alpha = symbols('alpha')
    M = 8
    k = 3 * 2 ** (2 * M)

    if method == "cauchy":
        def get_threshold(c):
            def func(x):
                # return [2 * x[0] / k - 2.0 * c / (3.14159265358979323846 * (x[0] ** 2 + c ** 2))]
                return [2 * x[0] / k - 2.0 * c * c / (3.14159265358979323846 * (x[0] ** 2 + c ** 2))]
            res_x = fsolve(func, [np.max(np.abs(data))])
            return res_x[0]
        # res_x = solve(2 * alpha / k - 2.0 * c / (3.14159265358979323846 * (alpha ** 2 + c ** 2)), alpha)
    elif method == "dweibull":
        def get_threshold(c):
            def func(x):
                return [2 * x[0] / k - 1.0 * c * x[0] ** (c - 1) * exp(-x[0] ** c)]
            res_x = fsolve(func, [np.max(np.abs(data))])
            return res_x[0]
        # res_x = solve(2 * alpha / k - 2.0 * c * alpha ** (c - 1) * exp(-alpha ** c), alpha)
    elif method == "expon":
        def get_threshold(c):
            def func(x):
                return [2 * x[0] / k - 2 * c * exp(-c * x[0])]
            res_x = fsolve(func, [np.max(np.abs(data))])
            return res_x[0]
    elif method == "laplace":
        def get_threshold(c):
            def func(x):
                return [2 * x[0] / k - 2 * c * exp(-x[0] / c)]
            res_x = fsolve(func, [np.max(np.abs(data))])
            return res_x[0]
    else:
        pass

    if isinstance(params, np.ndarray):
        res_x = np.ones(params.shape[0], dtype=np.float32)
        for i in range(params.shape[0]):
            res_x[i] = get_threshold(params[i])
        max_v = res_x
    else:
        max_v = float(get_threshold(params))

    return max_v

def ACIQW(data, quan_method, is_laplacian=True, layer=None, method=""):
    if is_laplacian:
        max_v = fit_distribution(data, quan_method, method=method)
    else:
        absmaxval = np.max(np.abs(data))
        batch_size = data.shape[0]
        N = data.reshape(-1).shape[0] / batch_size
        gaussian_const = (0.5 *
                          0.35) * (1 +
                                   (3.14159265358979323846 * math.log(4))**0.5)
        b = 2 * absmaxval * gaussian_const / (
            (2 * math.log(N * batch_size))**0.5)  ### gaussian
        max_v = 3.92403714 * b

    min_v = -max_v

    quan_method.set_scale(dict(min=min_v, max=max_v, zeros_point=0))

    return quan_method
