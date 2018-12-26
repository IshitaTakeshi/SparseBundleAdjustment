from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix

from sfm.lm import LevenbergMarquardt


def f(p):
    return np.array([10 * (p[1] - pow(p[0], 2)), (1 - p[0])])


def J(p):
    return csr_matrix([
        [-20 * p[0], 10],
        [-1, 0]
    ])


x = np.zeros(2)  # exact minimum is x = f(p) = [0, 0] at p = [1, 1]


class TestLevenbergMarquadt(unittest.TestCase):
    def test_optimize(self):
        lm = LevenbergMarquardt(f, J, x, n_input_dims=2, nu=1.2,
                                p0=np.array([2.0, 2.0]))
        # find p such that f(p) = x
        p = lm.optimize(max_iter=int(1e3))
        assert_array_almost_equal(p, np.array([1.0, 1.0]))


unittest.main()
