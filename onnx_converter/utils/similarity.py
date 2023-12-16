from array import array


import numpy as np

class Similarity(object):
    def __init__(self):
        pass
    @classmethod
    def cosine_similarity(cls, x, y):
        res = isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        res = res or (isinstance(x, list) and isinstance(y, list))
        assert res == True, "input type should be nparray or list"
        x = np.array(x, dtype = np.float64).flatten()
        y = np.array(y, dtype = np.float64).flatten()
        assert x.size == y.size, "Sizes of inputs should be equal"
        return np.sum(x*y) / np.sqrt(np.sum(x*x)*sum(y*y))

    @classmethod
    def mean_abs_error(cls, x, y):
        res = isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        res = res or (isinstance(x, list) and isinstance(y, list))
        assert res == True, "Types of inputs should be nparray or list"
        x = np.array(x, dtype = np.float64).flatten()
        y = np.array(y, dtype = np.float64).flatten()
        assert x.size == y.size, "Sizes of inputs should be equal"
        return np.sum(abs(x-y)) / x.size # type: ignore