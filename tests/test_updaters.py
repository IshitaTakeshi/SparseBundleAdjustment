from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from scipy import sparse

from sfm.updaters import LMUpdater

from sfm import config

config.n_pose_parameters = 1
config.n_point_parameters = 1

from sfm.sba import ParameterManager


class TestLMUpdater(unittest.TestCase):
    def setUp(self):
        self.manager = ParameterManager(4, 2)

        self.lambda_ = 1

        A = np.array([
            [2, 0],
            [0, -1],
            [1, 0]
        ])
        B = np.array([
            [-1, 0, 4, 5],
            [1, 0, 2, -3],
            [0, 3, -3, 1]
        ])

        self.A = sparse.csr_matrix(A)
        self.B = sparse.csr_matrix(B)

        self.residual = np.array([3, 4, -1])

    def ordinary_lm(self):
        J = sparse.hstack((self.A, self.B))
        I = sparse.identity(J.shape[1])

        return sparse.linalg.spsolve(
            J.T.dot(J) + self.lambda_ * I,
            J.T.dot(self.residual)
        )

    def test_updater(self):
        updater = LMUpdater(self.manager, None, None)
        updater.precomputation(self.A, self.B, self.residual)
        dp = updater.calc_update(self.lambda_)

        # the result should be same in both ways of calculation
        assert_array_almost_equal(self.ordinary_lm(), dp)


unittest.main()
