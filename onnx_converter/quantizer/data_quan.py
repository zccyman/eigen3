# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/9/30 15:47
# @File     : data_quan.py
# import cv2
import os
import numpy as np
import sys
import contextlib
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import pylab
import scipy.stats

import joblib
from joblib.parallel import Parallel, delayed
from tqdm import tqdm
from scipy.stats import entropy as kl_div, kstest

import torch
import copy
import scipy.special as sp
from sympy import symbols, solve, log, exp
from scipy.optimize import fsolve

try:
    from utils import Registry, clip
except:
    from onnx_converter.utils import Registry, clip # type: ignore


def Cosine_distance(simulation_data, true_data, eps=1.0e-5):
    s_data = copy.deepcopy(simulation_data).astype(np.float32)
    t_data = copy.deepcopy(true_data).astype(np.float32)
    s_data = torch.from_numpy(s_data.reshape(-1))
    t_data = torch.from_numpy(t_data.reshape(-1))
    normal = torch.sqrt(torch.sum(s_data * s_data) * torch.sum(t_data * t_data))
    dist = torch.sum(s_data * t_data) / (normal + eps)
    dist = (1 - np.abs(dist.item())) * 100

    return np.float32(dist)

# A solution to wrap joblib parallel call in tqdm from 
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
# and https://github.com/louisabraham/tqdm_joblib
@contextlib.contextmanager
def tqdm_joblib(*args, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar
    given as argument"""

    tqdm_object = tqdm(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def get_distributions():
    distributions = []
    for this in dir(scipy.stats):
        if "fit" in eval("dir(scipy.stats." + this + ")"):
            distributions.append(this)
    return distributions


def get_common_distributions():
    distributions = get_distributions()
    # to avoid error due to changes in scipy
    common = [
        "cauchy",
        "chi2",
        "expon",
        "exponpow",
        "gamma",
        "lognorm",
        "norm",
        "powerlaw",
        "rayleigh",
        "uniform",
    ]
    common = [x for x in common if x in distributions]
    return common


class Fitter(object):
    """Fit a data sample to known distributions

    A naive approach often performed to figure out the undelying distribution that
    could have generated a data set, is to compare the histogram of the data with
    a PDF (probability distribution function) of a known distribution (e.g., normal).

    Yet, the parameters of the distribution are not known and there are lots of
    distributions. Therefore, an automatic way to fit many distributions to the data
    would be useful, which is what is implemented here.

    Given a data sample, we use the `fit` method of SciPy to extract the parameters
    of that distribution that best fit the data. We repeat this for all available distributions.
    Finally, we provide a summary so that one can see the quality of the fit for those distributions

    Here is an example where we generate a sample from a gamma distribution.

    ::

        >>> # First, we create a data sample following a Gamma distribution
        >>> from scipy import stats
        >>> data = stats.gamma.rvs(2, loc=1.5, scale=2, size=20000)

        >>> # We then create the Fitter object
        >>> import fitter
        >>> f = fitter.Fitter(data)

        >>> # just a trick to use only 10 distributions instead of 80 to speed up the fitting
        >>> f.distributions = f.distributions[0:10] + ['gamma']

        >>> # fit and plot
        >>> f.fit()
        >>> f.summary()
                sumsquare_error
        gamma          0.000095
        beta           0.000179
        chi            0.012247
        cauchy         0.044443
        anglit         0.051672
        [5 rows x 1 columns]

    Once the data has been fitted, the :meth:`summary` metod returns a sorted dataframe where the

    Looping over the 80 distributions in SciPy could takes some times so you can overwrite the
    :attr:`distributions` with a subset if you want. In order to reload all distributions,
    call :meth:`load_all_distributions`.

    Some distributions do not converge when fitting. There is a timeout of 30 seconds after which
    the fitting procedure is cancelled. You can change this :attr:`timeout` attribute if needed.

    If the histogram of the data has outlier of very long tails, you may want to increase the
    :attr:`bins` binning or to ignore data below or above a certain range. This can be achieved
    by setting the :attr:`xmin` and :attr:`xmax` attributes. If you set xmin, you can come back to
    the original data by setting xmin to None (same for xmax) or just recreate an instance.
    """

    def __init__(
        self,
        data,
        xmin=None,
        xmax=None,
        bins=100,
        distributions=None,
        timeout=30,
        density=True,
    ):
        """.. rubric:: Constructor

        :param list data: a numpy array or a list
        :param float xmin: if None, use the data minimum value, otherwise histogram and
            fits will be cut
        :param float xmax: if None, use the data maximum value, otherwise histogram and
            fits will be cut
        :param int bins: numbers of bins to be used for the cumulative histogram. This has
            an impact on the quality of the fit.
        :param list distributions: give a list of distributions to look at. If none, use
            all scipy distributions that have a fit method. If you want to use
            only one distribution and know its name, you may provide a string (e.g.
            'gamma'). Finally, you may set to 'common' to  include only common
            distributions, which are: cauchy, chi2, expon, exponpow, gamma,
                 lognorm, norm, powerlaw, irayleigh, uniform.
        :param timeout: max time for a given distribution. If timeout is
            reached, the distribution is skipped.

        .. versionchanged:: 1.2.1 remove verbose argument, replacedb by logging module.
        .. versionchanged:: 1.0.8 increase timeout from 10 to 30 seconds.
        """
        self.timeout = timeout
        # USER input
        self._data = None

        # Issue https://github.com/cokelaer/fitter/issues/22 asked for setting
        # the density to False in the fitting and plotting. I first tought it
        # would be possible, but the fitting is performed using the PDF of scipy
        # so one would still need to normalise the data so that it is
        # comparable. Therefore I do not see anyway to do it without using
        # density set to True for now.
        self._density = True

        #: list of distributions to test
        self.distributions = distributions
        if self.distributions == None:
            self._load_all_distributions()
        elif self.distributions == "common":
            self.distributions = get_common_distributions()
        elif isinstance(distributions, str):
            self.distributions = [distributions]

        self.bins = bins

        self._alldata = np.array(data)
        if xmin == None:
            self._xmin = self._alldata.min()
        else:
            self._xmin = xmin
        if xmax == None:
            self._xmax = self._alldata.max()
        else:
            self._xmax = xmax

        self._trim_data()
        self._update_data_pdf()

        # Other attributes
        self._init()

    def _init(self):
        self.fitted_param = {}
        self.fitted_pdf = {}
        self._fitted_errors = {}
        self._fitted_errors_cosine = {}
        self._aic = {}
        self._bic = {}
        self._kldiv = {}
        self._ks_stat = {}
        self._ks_pval = {}
        self._fit_i = 0  # fit progress
        #self.pb = None

    def _update_data_pdf(self):
        # histogram retuns X with N+1 values. So, we rearrange the X output into only N
        self.y, self.x = np.histogram(self._data, bins=self.bins, density=self._density)
        self.x = [(this + self.x[i + 1]) / 2.0 for i, this in enumerate(self.x[0:-1])]

    def _trim_data(self):
        self._data = self._alldata[np.logical_and(self._alldata >= self._xmin, self._alldata <= self._xmax)]

    def _get_xmin(self):
        return self._xmin

    def _set_xmin(self, value):
        if value == None:
            value = self._alldata.min()
        elif value < self._alldata.min():
            value = self._alldata.min()
        self._xmin = value
        self._trim_data()
        self._update_data_pdf()

    xmin = property(_get_xmin, _set_xmin, doc="consider only data above xmin. reset if None")

    def _get_xmax(self):
        return self._xmax

    def _set_xmax(self, value):
        if value == None:
            value = self._alldata.max()
        elif value > self._alldata.max():
            value = self._alldata.max()
        self._xmax = value
        self._trim_data()
        self._update_data_pdf()

    xmax = property(_get_xmax, _set_xmax, doc="consider only data below xmax. reset if None ")

    def _load_all_distributions(self):
        """Replace the :attr:`distributions` attribute with all scipy distributions"""
        self.distributions = get_distributions()

    def hist(self):
        """Draw normed histogram of the data using :attr:`bins`


        .. plot::

            >>> from scipy import stats
            >>> data = stats.gamma.rvs(2, loc=1.5, scale=2, size=20000)
            >>> # We then create the Fitter object
            >>> import fitter
            >>> fitter.Fitter(data).hist()

        """
        _ = pylab.hist(self._data, bins=self.bins, density=self._density)
        pylab.grid(True)

    def _fit_single_distribution(self, distribution):
        try:
            # need a subprocess to check time it takes. If too long, skip it
            dist = eval("scipy.stats." + distribution)

            # TODO here, dist.fit may take a while or just hang forever
            # with some distributions. So, I thought to use signal module
            # to catch the error when signal takes too long. It did not work
            # presumably because another try/exception is inside the
            # fit function, so I used threading with a recipe from stackoverflow
            # See timed_run function above
            param = self._timed_run(dist.fit, distribution, args=self._data)

            # with signal, does not work. maybe because another expection is caught
            # hoping the order returned by fit is the same as in pdf
            pdf_fitted = dist.pdf(self.x, *param)

            self.fitted_param[distribution] = param[:] # type: ignore
            self.fitted_pdf[distribution] = pdf_fitted

            # calculate error
            sq_error = pylab.sum((self.fitted_pdf[distribution] - self.y) ** 2)
            cosine_error = Cosine_distance(self.fitted_pdf[distribution], self.y)

            # calculate information criteria
            logLik = np.sum(dist.logpdf(self.x, *param))
            k = len(param[:]) # type: ignore
            n = len(self._data) # type: ignore
            aic = 2 * k - 2 * logLik
            bic = n * np.log(sq_error / n) + k * np.log(n)

            # calculate kullback leibler divergence
            kullback_leibler = kl_div(self.fitted_pdf[distribution], self.y)

            # calculate goodness-of-fit statistic
            dist_fitted = dist(*param)
            ks_stat, ks_pval = kstest(self._data, dist_fitted.cdf)

            # logging.info("Fitted {} distribution with L2 error={})".format(distribution, sq_error))
            # logging.info("Fitted {} distribution with Cosine error={})".format(distribution, cosine_error))

            # compute some errors now
            self._fitted_errors[distribution] = sq_error
            self._fitted_errors_cosine[distribution] = cosine_error
            self._aic[distribution] = aic
            self._bic[distribution] = bic
            self._kldiv[distribution] = kullback_leibler
            self._ks_stat[distribution] = ks_stat
            self._ks_pval[distribution] = ks_pval
        except Exception:  # pragma: no cover
            # print("SKIPPED {} distribution (taking more than {} seconds)".format(distribution, self.timeout))
            # if we cannot compute the error, set it to large values
            self._fitted_errors[distribution] = np.inf
            self._aic[distribution] = np.inf
            self._bic[distribution] = np.inf
            self._kldiv[distribution] = np.inf

    def fit(self, progress=False, n_jobs=-1, max_workers=-1):
        r"""Loop over distributions and find best parameter to fit the data for each

        When a distribution is fitted onto the data, we populate a set of
        dataframes:

            - :attr:`df_errors`  :sum of the square errors between the data and the fitted
              distribution i.e., :math:`\sum_i \left( Y_i - pdf(X_i) \right)^2`
            - :attr:`fitted_param` : the parameters that best fit the data
            - :attr:`fitted_pdf` : the PDF generated with the parameters that best fit the data

        Indices of the dataframes contains the name of the distribution.

        """
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        N = len(self.distributions) # type: ignore
        with tqdm_joblib(desc=f"Fitting {N} distributions", total=N) as progress_bar:
            Parallel(n_jobs=max_workers, backend='threading')(delayed(self._fit_single_distribution)(dist) for dist in self.distributions) # type: ignore


        self.df_errors = pd.DataFrame(
            {
                "sumsquare_error": self._fitted_errors,
                "cosine_error": self._fitted_errors_cosine,
                "aic": self._aic,
                "bic": self._bic,
                "kl_div": self._kldiv,
                "ks_statistic": self._ks_stat,
                "ks_pvalue": self._ks_pval,
            }
        )

    def plot_pdf(self, names=None, Nbest=5, lw=2, method="sumsquare_error"):
        """Plots Probability density functions of the distributions

        :param str,list names: names can be a single distribution name, or a list
            of distribution names, or kept as None, in which case, the first Nbest
            distribution will be taken (default to best 5)


        """
        assert Nbest > 0
        if Nbest > len(self.distributions): # type: ignore
            Nbest = len(self.distributions) # type: ignore

        if isinstance(names, list):
            for name in names:
                pylab.plot(self.x, self.fitted_pdf[name], lw=lw, label=name)
        elif names:
            pylab.plot(self.x, self.fitted_pdf[names], lw=lw, label=names)
        else:
            try:
                names = self.df_errors.sort_values(by=method).index[0:Nbest]
            except Exception:
                names = self.df_errors.sort(method).index[0:Nbest]

            for name in names:
                if name in self.fitted_pdf.keys():
                    pylab.plot(self.x, self.fitted_pdf[name], lw=lw, label=name)
                else:  # pragma: no cover
                    import logging
                    logging.warning("%s was not fitted. no parameters available" % name)
        pylab.grid(True)
        pylab.legend()

    def get_best(self, method="sumsquare_error"):
        """Return best fitted distribution and its parameters

        a dictionary with one key (the distribution name) and its parameters

        """
        # self.df should be sorted, so then us take the first one as the best
        name = self.df_errors.sort_values(method).iloc[0].name
        params = self.fitted_param[name]
        distribution = getattr(scipy.stats, name) # type: ignore
        param_names = (distribution.shapes + ", loc, scale").split(", ") if distribution.shapes else ["loc", "scale"]

        param_dict = {}
        for d_key, d_val in zip(param_names, params):
            param_dict[d_key] = d_val
        return {name: param_dict}

    def summary(self, Nbest=5, lw=2, plot=True, method="sumsquare_error", clf=True):
        """Plots the distribution of the data and Nbest distribution"""
        if plot:
            if clf:
                pylab.clf()
            self.hist()
            self.plot_pdf(Nbest=Nbest, lw=lw, method=method)
            pylab.grid(True)

        Nbest = min(Nbest, len(self.distributions)) # type: ignore
        try:
            names = self.df_errors.sort_values(by=method).index[0:Nbest]
        except:  # pragma: no cover
            names = self.df_errors.sort(method).index[0:Nbest]
        return self.df_errors.loc[names]

    def _timed_run(self, func, distribution, args=(), kwargs={}, default=None):
        """This function will spawn a thread and run the given function
        using the args, kwargs and return the given default value if the
        timeout is exceeded.

        http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
        """

        class InterruptableThread(threading.Thread):
            def __init__(self):
                threading.Thread.__init__(self)
                self.result = default
                self.exc_info = (None, None, None)

            def run(self):
                try:
                    self.result = func(args, **kwargs)
                except Exception as err:  # pragma: no cover
                    self.exc_info = sys.exc_info()

            def suicide(self):  # pragma: no cover
                raise RuntimeError("Stop has been called")

        it = InterruptableThread()
        it.start()
        started_at = datetime.now()
        it.join(self.timeout)
        ended_at = datetime.now()
        diff = ended_at - started_at

        if it.exc_info[0] is not None:  # pragma: no cover ;  if there were any exceptions
            a, b, c = it.exc_info
            raise Exception(a, b, c)  # communicate that to caller

        if it.is_alive():  # pragma: no cover
            it.suicide()
            raise RuntimeError
        else:
            return it.result


def fit_distribution(data, bit_num=8, is_perchannel=False, method="cauchy", bins=2048, timeout=120):

    if is_perchannel:
        data_ = copy.deepcopy(data)
        params = np.ones(data_.shape[0], dtype=np.float32)
        for i in range(data_.shape[0]):
            f = Fitter(data_[i, :], distributions=[method], bins=bins, timeout=timeout)
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
        f = Fitter(copy.deepcopy(data).reshape(-1), distributions=[method], bins=bins, timeout=timeout)
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
    M = int(np.round(np.log2(bit_num + 1) / 2) * 2)
    k = 3 * 2 ** (2 * M)
    
    if method not in ["cauchy", "dweibull", "expon", "laplace"]:
        method = "laplace"

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
        def get_threshold(c):
            def func(x):
                return [2 * x[0] / k - 1.0 * c * x[0] ** (c - 1) * exp(-x[0] ** c)]
            res_x = fsolve(func, [np.max(np.abs(data))])
            return res_x[0]

    if isinstance(params, np.ndarray): # type: ignore
        res_x = np.ones(params.shape[0], dtype=np.float32)
        for i in range(params.shape[0]):
            res_x[i] = get_threshold(params[i])
        max_v = res_x
    else:
        max_v = float(get_threshold(params)) # type: ignore

    return max_v


QUANTIZE: Registry = Registry('quantize', scope='')


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


@QUANTIZE.register_module(name='base')
class QuanMethod(object):
    # default: 0: uint8 1:int8 2:uint16 3:int16 4:uint32 5:int32 6:int4 7:uint4
    def __init__(self, bit_select=1, **kwargs):
        self.bits_dict = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32}
        self.maxs = {0: 255, 1: 127, 2: 65535, 3: 32767, 4: 4294967295, 5: 2147483647}
        self.mins = {0: 0, 1: -128, 2: 0, 3: -32768, 4: 0, 5: -2147483648}
        self.eps = 1e-5
        # self.maxs = {0: 127, 1: 127, 2: 32767, 3: 32767, 4: 2147483647, 5: 2147483647}
        # self.mins = {0: -128, 1: -128, 2: -32768, 3: -32768, 4: -2147483648, 5: -2147483648}
        if 'bits_dict' in kwargs:
            self.bits_dict.update(kwargs['bits_dict'])
            self.maxs.update(kwargs['maxs'])
            self.mins.update(kwargs['mins'])

        # config change data type to string in cfg_dict
        # then coding eval make str to data type
        for key in self.bits_dict.keys():
            if isinstance(self.bits_dict[key], str):
                self.bits_dict[key] = eval(self.bits_dict[key])
        self.bits_select = bit_select
        self.quan_bit = self.bits_dict[bit_select]
        self.shift = np.int32(0)
        self.scale = np.float32(1.0)
        self.zero_point = np.int32(0)
        self.d_high = self.maxs[bit_select]
        self.d_low = self.mins[bit_select]
        self.is_percentile = False

    def reset_bit_select(self, bit_select):
        self.bits_select = bit_select
        self.d_high = self.maxs[bit_select]
        self.d_low = self.mins[bit_select]

    def clip_scale(self):
        min_scale = self.eps / self.maxs[self.bits_select]
        self.scale = np.clip(self.scale, min_scale, 1/self.eps)

    def get_max_value(self, data):
        aa = np.asarray(data)
        xmax = max(aa.flat)
        xmin = min(aa.flat)
        return xmax, xmin

    # @abc.abstractmethod
    def get_quan_param(self, data):
        pass

    # @abc.abstractmethod
    def get_quan_data(self, data):
        return data

    # @abc.abstractmethod
    def get_dequan_data(self, data):
        return data

    def get_class_name(self):
        return self.__class__.__name__

    def get_upper(self):
        return self.d_high

    def get_lower(self):
        return self.d_low

    def align_bit(self, data):
        min_v, max_v = self.mins[self.bits_select], self.maxs[self.bits_select]
        data = clip(data, min_v, max_v)
        if self.bits_select > 5:
            return np.int8(data)
        else:
            return data.astype(self.bits_dict[self.bits_select])

    def set_shift(self, data):
        pass

    def get_shift(self):
        pass

    def set_scale(self, data):
        pass

    def get_scale(self):
        pass

    # synchronization zero-point between quantize graph and simulation
    # make more parameter synchronization using dictionary
    def get_quant_param(self) -> dict:
        return dict(scale=self.scale, zero_point=self.zero_point)
    
    # synchronization zero-point between quantize graph and simulation
    # make more parameter synchronization using dictionary
    def set_quant_param(self, param: dict):
        self.scale = param.get('scale', self.scale)
        self.zero_point = param.get('zero_point', self.zero_point)

    @staticmethod
    def percentile(self, bit_num, data, selected_idx=0): # type: ignore
        max_vlist = []

        bins = 2048 * int(np.round(np.log2(bit_num + 1) / 2) * 2)
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
            self.set_scale(dict(min=min_v, max=max_v, zeros_point=0))
            qdata = self.get_quan_data(copy.deepcopy(data))
            qdata = self.get_dequan_data(qdata)
            error = Cosine_distance(qdata, data)
            error_list.append(error)
            scale_list.append(dict(min=min_v, max=max_v, zeros_point=0))

        selected_idx = np.argmin(error_list)
        return max_vlist[selected_idx]


@QUANTIZE.register_module(name='baseshiftquan')
class BaseShiftQuan(QuanMethod):
    def __init__(self, bit_select=1, **kwargs):
        super(BaseShiftQuan, self).__init__(bit_select, **kwargs)

    def get_quan_param(self, data):
        max_, min_ = self.get_max_value(data)
        val = np.max(np.abs(max_), np.abs(min_))
        if val == 0:
            return 0
        bit = int(np.log2(self.maxs[self.bits_select] + 1))
        if bit - 1 - np.log2(val) > 0:
            self.shift = int(bit - 1 - np.log2(val))
        else:
            self.shift = int(bit - 1 - np.log2(val) - 1)

    def get_quan_data(self, data):
        if self.shift > 0:
            out = data * (1 << self.shift)
        else:
            out = np.right_shift(data.astype(np.int32), -self.shift)  # (data/(1<<(-offset)))

        out[out > self.d_high] = self.d_high
        out[out < self.d_low] = self.d_low
        out = self.align_bit(data=out)
        return out

    def get_dequan_data(self, data):
        return data.astype(np.float32) / float(2 ** self.shift)

    def set_shift(self, shift):
        self.shift = shift

    def get_shift(self):
        return self.shift


@QUANTIZE.register_module(name='bestshiftquan')
class BestShiftQuan(BaseShiftQuan):
    def __init__(self, bit_select, margin=5, **kwargs):
        super(BaseShiftQuan, self).__init__(bit_select, **kwargs)
        self.margin = margin

    def get_quan_param(self, data):
        sum = []
        super.get_quan_param(data) # type: ignore
        init_shift = self.get_shift()
        for i in range(init_shift - self.margin, init_shift + self.margin + 1):
            err = abs(data - (self._getQData(data, i).astype(np.float32) / (2 ** i))).sum() # type: ignore
            sum.append(err)
        self.shift = sum.index(min(sum)) + init_shift - self.margin


@QUANTIZE.register_module(name='floatsymquan')  # symmetric
class FloatSymQuan(QuanMethod):
    def __init__(self, bit_select=1, **kwargs):
        super(FloatSymQuan, self).__init__(bit_select, **kwargs)
        self.zero_point = np.int32(0)

    def get_quan_param(self, data, is_aciq=False, method="laplace"):
        if isinstance(data, np.ndarray):
            if is_aciq:
                dmax = fit_distribution(data, self.maxs[self.bits_select], method=method)
                dmin = -dmax
            else:
                if self.is_percentile:
                    dmax = self.percentile(self, self.maxs[self.bits_select], data)
                    dmin = -dmax
                else:
                    dmax, dmin = self.get_max_value(data.reshape(-1))
            val = max(np.abs(dmax), np.abs(dmin))
            self.scale = np.float32(val / self.d_high)
        self.clip_scale()

    # quantize for input ndarry
    def get_quan_data(self, data, is_squant=False):
        if isinstance(data, np.ndarray):
            if is_squant:
                out_c, in_c = data.shape[:2]
                weight = data / self.scale + self.zero_point
                shaped_w = weight.reshape(out_c, in_c, -1)
                transformed_val = SQuant(shaped_w)
                transformed_val = transformed_val.reshape(data.shape)
            else:
                transformed_val = data.reshape(-1) / self.scale + self.zero_point
            # transformed_val = np.round(transformed_val)
            # clamped_val = np.clip(transformed_val, self.d_low, self.d_high)
            quantized = self.align_bit(data=np.round(transformed_val))

            return np.reshape(quantized, data.shape)
        else:
            return data

    def get_dequan_data(self, data):
        dequantize = (data.reshape(-1).astype(np.float32) - self.zero_point) * self.scale
        return np.reshape(dequantize, data.shape)

    # process extract max/min from origin ndarry
    def set_scale(self, data: dict) -> None:
        if 'scale' in data.keys():  # offline quantize mode
            self.scale = data['scale']
            self.zeros_point = data['zero_point']
        else:  # online quantize mode
            dmax, dmin, self.zeros_point = data['max'], data['min'], data['zeros_point']
            val = max(np.abs(dmax), np.abs(dmin))
            self.scale = val / self.d_high
        self.clip_scale()

    def get_scale(self):
        return self.scale, self.zero_point


@QUANTIZE.register_module(name='floatquan')
class FloatQuan(FloatSymQuan):
    def __init__(self, bit_select=1, **kwargs):
        super(FloatQuan, self).__init__(bit_select, **kwargs)

    def get_quan_param(self, data, is_aciq=False, method="laplace"):
        dmax, dmin = self.get_max_value(data.reshape(-1))
        self.max, self.min = copy.deepcopy(dmax), copy.deepcopy(dmin)
        self.scale = np.float32((dmax - dmin) / (self.d_high - self.d_low))
        self.clip_scale()
        self.zero_point = np.round(self.d_low - dmin / self.scale)
        self.zero_point = np.clip(self.zero_point, a_min=self.d_low, a_max=self.d_high)

        self.zero_point = self.bits_dict[self.bits_select](self.zero_point)

    def set_scale(self, data: dict) -> None:
        dmax, dmin, self.zero_point = data['max'], data['min'], data['zeros_point']
        self.scale = (dmax - dmin) / (self.d_high - self.d_low)
        self.clip_scale()
        self.zero_point = np.round(self.d_low - dmin / self.scale)
        self.zero_point = np.clip(self.zero_point, a_min=self.d_low, a_max=self.d_high)
        # if self.zero_point < self.d_low:
        #     self.zero_point = self.d_low
        # if self.zero_point > self.d_high:
        #     self.zero_point = self.d_high
        self.zero_point = self.bits_dict[self.bits_select](self.zero_point)


@QUANTIZE.register_module(name='perchannelbaseshiftquan')
class BaseShiftQuanPerchannel(QuanMethod):
    def __init__(self, bit_select=1, **kwargs):
        super(BaseShiftQuanPerchannel, self).__init__(bit_select, **kwargs)
        self.shift = 0

    def get_quan_param(self, data):
        out_c, in_c, k_h, k_w = data.shape
        data_ = data.reshape(out_c, -1)
        dmax, dmin = np.max(data_, axis=1), np.min(data_, axis=1)
        select = np.abs(dmax) - np.abs(dmin)
        val = np.zeros_like(dmax)
        val[select > 0] = np.abs(dmax[select > 0])
        val[select <= 0] = np.abs(dmin[select <= 0])
        bit = int(np.log2(self.maxs[self.bits_select] + 1))
        self.shift = np.zeros_like(dmax)
        select = bit - 1 - np.log2(val)
        self.shift[select > 0] = int(bit - 1 - np.log2(val[select > 0]))
        self.shift[select <= 0] = int(bit - 1 - np.log2(val[select <= 0]))

    def get_quan_data(self, data):
        out_c, in_c, k_h, k_w = data.shape
        data_ = data.reshape(out_c, -1)
        for index in data_.shape[0]:
            if self.shift[index] > 0: # type: ignore
                data_[index] = data_[index] * \
                               (1 << self.shift[index]) # type: ignore
            else:
                data_[index] = np.right_shift(
                    data_[index].astype(np.int32),
                    -self.shift[index])  # (data/(1<<(-offset))) # type: ignore

        data_[data_ > self.d_high] = self.d_high
        data_[data_ < self.d_low] = self.d_low
        out = self.align_bit(data=data_)
        return out.reshape(data)

    def get_dequan_data(self, data):
        out_c, in_c, k_h, k_w = data.shape
        data_ = data.reshape(out_c, -1)
        data_ = data_.astype(np.float32) / (2 ** self.shift).reshape(-1, 1)
        return data_.reshape(data)

    def set_shift(self, shift):
        self.shift = shift

    def get_shift(self):
        return self.shift


@QUANTIZE.register_module(name='perchannelbestshiftquan')
class BestShiftQuanPerchannel(BaseShiftQuanPerchannel):
    def __init__(self, bit_select=1, margin=5, **kwargs):
        super(BestShiftQuanPerchannel, self).__init__(bit_select, **kwargs)
        self.margin = margin

    def get_quan_param(self, data):
        super().get_quan_param(data)
        data_ = data.reshape(data.shape[0], -1)
        for idx in self.shift.shape[0]: # type: ignore
            init_shift = self.shift[idx] # type: ignore
            for i in range(init_shift - self.margin, init_shift + self.margin + 1):
                err = abs(data - (self._getQData(data_[idx], i).astype(np.float32) / (2 ** i))).sum() # type: ignore
                sum.append(err)

            self.shift[idx] = sum.index(min(sum)) + init_shift - self.margin # type: ignore


@QUANTIZE.register_module('perchannelfloatsymquan')
class FloatSymQuanPerchannel(QuanMethod):
    def __init__(self, bit_select=1, **kwargs):
        super(FloatSymQuanPerchannel, self).__init__(bit_select, **kwargs)
        self.scale = np.array([1.0], dtype=np.float32)
        self.zero_point = np.array([0], dtype=self.bits_dict[self.bits_select])

    def get_quan_param(self, data: np.ndarray, is_aciq=False, method="laplace"):
        if is_aciq:
            dmax = fit_distribution(data, self.maxs[self.bits_select], method=method)
            dmin = -dmax
        else:
            # data[np.abs(data) < self.eps] = 0
            out_c, in_c = data.shape[:2]
            data_ = data.reshape(out_c, -1)
            dmax, dmin = np.max(data_, axis=1), np.min(data_, axis=1)
            # select = np.abs(dmax) - np.abs(dmin)
        val = np.max(np.column_stack([np.abs(dmax), np.abs(dmin)]), axis=1)
        self.max, self.min = copy.deepcopy(dmax), copy.deepcopy(dmin)
        # val = np.zeros_like(dmax)
        # val[select > 0] = dmax[select > 0]
        # val[select <= 0] = dmin[select <= 0]
        self.scale = (np.abs(val) / self.d_high).astype(np.float32)
        self.clip_scale()
        self.zero_point = np.zeros_like(self.scale)
        self.zero_point = self.align_bit(self.zero_point)

    def get_scale(self):
        return self.scale, self.zero_point

    # process extract max/min from origin ndarry
    def set_scale(self, data: dict):
        if 'scale' in data.keys():  # offline quantize mode
            self.scale = np.array(data['scale'])
            self.zero_point = np.array(data['zero_point'])
        else:  # online quantize mode
            dmax, dmin, self.zero_point = data['max'], data['min'], data['zeros_point']
            val = np.max(np.column_stack([np.abs(dmax), np.abs(dmin)]), axis=1)
            # val = np.zeros_like(dmax)
            # val[select > 0] = dmax[select > 0]
            # val[select <= 0] = dmin[select <= 0]
            self.scale = np.abs(val) / self.d_high
            self.clip_scale()
            self.zero_point = np.zeros_like(self.scale)
            self.zero_point = self.align_bit(self.zero_point)

        # dmax, dmin, self.zeros_point = data['max'], data['min'], data['zeros_point']
        # select = np.abs(dmax) - np.abs(dmin)
        # val = np.zeros_like(dmax)
        # val[select > 0] = dmax[select > 0]
        # val[select <= 0] = dmin[select <= 0]
        # self.scale = np.abs(val) / self.d_high
        # self.zero_point = np.zeros_like(self.scale)
        # self.zero_point = self.align_bit(self.zero_point)
        #
        # val = max(np.abs(dmax), np.abs(dmin))
        # self.scale = val / self.d_high

    def get_quan_data(self, data, is_squant=False):
        out_c, in_c = data.shape[:2]

        transformed_val = data.reshape(out_c, -1) / self.scale.reshape(-1, 1) + \
                          self.zero_point.reshape(-1, 1)

        quantized = self.align_bit(data=np.round(transformed_val))

        return np.reshape(quantized, data.shape)

    def get_dequan_data(self, data):
        out_c, int_c = data.shape[:2]
        dequantize = (data.reshape(out_c, -1).astype(np.float32) -
                      self.zero_point.reshape(-1, 1)) * self.scale.reshape(-1, 1)
        return np.reshape(dequantize, data.shape)


@QUANTIZE.register_module('perchannelfloatquan')
class FloatQuanPerchannel(FloatSymQuanPerchannel):
    def __init__(self, bit_select=1, **kwargs):
        super(FloatQuanPerchannel, self).__init__(bit_select, **kwargs)

    def get_quan_param(self, data: np.ndarray, is_aciq=False, method="laplace"):
        out_c, int_c = data.shape[:2]
        data_ = data.reshape(out_c, -1)
        dmax, dmin = np.max(data_, axis=1), np.min(data_, axis=1)
        self.max, self.min = copy.deepcopy(dmax), copy.deepcopy(dmin)
        # filter = np.max(np.abs(data_), axis=1) < self.eps
        self.scale = ((dmax - dmin) / (self.d_high - self.d_low)).astype(np.float32)
        self.clip_scale()
        self.zero_point = np.round(self.d_low - dmin / self.scale)
        # self.zero_point = np.clip(self.zero_point, a_min=self.d_low, a_max=self.d_high)
        self.zero_point = self.align_bit(self.zero_point)

    def set_scale(self, data: dict) -> None:
        dmax, dmin, self.zero_point = data['max'], data['min'], data['zeros_point']
        self.scale = ((dmax - dmin) / (self.d_high - self.d_low)).astype(np.float32)
        self.clip_scale()
        self.zero_point = np.round(self.d_low - dmin / self.scale)
        # self.zero_point = np.clip(self.zero_point, a_min=self.d_low, a_max=self.d_high)
        self.zero_point = self.align_bit(self.zero_point)


DefaultQuant = QuanMethod().get_quant_param()

# @QUANTIZE.register_module(name='')
# class


# if __name__ == '__main__':
#     # base_shift = QUANTIZE.get('BaseShiftQuan')
#     # object_base_shift = base_shift(quan_size=16)
#     # sym = QUANTIZE.get('FloatSymQuan')
#     # quan_sym = sym()
#     # import cv2
#     # img = cv2.imread(r'D:\code\converter\tensorflow\converter-bin\4_Dancing_Dancing_4_162.jpg')
#     # img1 = img - (104, 117, 123)
#     # quan_sym.get_quan_param(img1)
#     # quantized = quan_sym.get_quan_data(img1)
#     # dequantized = np.array(quan_sym.get_dequan_data(quantized)+(104, 117, 123), np.uint8)
#     # cv2.imwrite(r'D:\code\converter\tensorflow\converter-bin\111.jpg', dequantized)
#     max, min, zeros_point = np.abs(np.random.randn((32))), -np.abs(np.random.randn((32))), np.zeros((32))
#     sym = QUANTIZE.get('perchannelfloatquan')(bit_select=1)
#     sym.set_scale(data={'max': max, 'min': min, 'zeros_point': zeros_point})
#     print('done!')
