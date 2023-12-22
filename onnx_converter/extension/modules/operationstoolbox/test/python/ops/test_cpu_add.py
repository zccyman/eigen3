import sys

sys.path.append("./")

import numpy as np
import pytest
from libs.pyops import pyops


def test_add():
    shape = np.array([32, 32, 32, 32], dtype=np.int32)
    a = (np.random.random(shape) * 1000).astype(np.float32)
    b = (np.random.random(shape) * 1000).astype(np.float32)
    c = pyops.add(a, b, shape)

    assert c.all() == (a + b).all()


if __name__ == "__main__":
    test_add()
