import sys
import os
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("./extension/modules/operationstoolbox/"))
import numpy as np
import pytest

from libs.pyops import pyops


def test_float():
    a = np.random.randn(100, 200).astype(np.float32)
    b = np.random.randn(200, 300).astype(np.float32)
    c = np.random.randn(1, 300).astype(np.float32)
    out = np.zeros([100, 300]).astype(np.float32)
    res1 = np.matmul(a, b) + c
    pyops.py_eigen_fc(a, b, c, 100, 200, 300, out)
    diff = np.mean(np.abs(res1 - out))
    assert diff < 1e-5
    print("test_float pass")

def test_double():
    a = np.random.randn(100, 200).astype(np.float64)
    b = np.random.randn(200, 300).astype(np.float64)
    c = np.random.randn(1, 300).astype(np.float64)
    out = np.zeros([100, 300]).astype(np.float64)
    res1 = np.matmul(a, b) + c
    pyops.py_eigen_fc(a, b, c, 100, 200, 300, out)
    diff = np.mean(np.abs(res1 - out))
    assert diff < 1e-5
    print("test_double pass")

if __name__ == "__main__":
    test_float()
    test_double()
