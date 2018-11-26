from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sfm.lm import LevenbergMarquardt



class TestLevenbergMarquadt(unittest.TestCase):
    def setUp(self):
        def f(p):
            return np.array([10 * (p[1] - pow(p[0], 2)), (1 - p[0])])

        def J(p):
            return np.array([
                [-10 * p[0], 10],
                [-1, 0]
            ])

        x = np.zeros(2)  # target

        self.lm = LevenbergMarquardt(f, J, x, n_input_dims=2, tau=0.1,
                                     threshold_relative_change=0.0,
                                     initial_p=np.array([2.0, 2.0]))

    def test_update_positive(self):
        p, mu, nu = self.lm.update_positive(
            p=np.zeros(2),
            delta_p=np.ones(2),
            mu=3.0,
            nu=1.0,
            rho=1.0
        )

        assert_array_equal(p, np.ones(2))
        self.assertEqual(mu, 1.0)
        self.assertEqual(nu, 2.0)

        p, mu, nu = self.lm.update_positive(
            p=np.zeros(2),
            delta_p=np.ones(2),
            mu=3.0,
            nu=1.0,
            rho=0.5
        )
        self.assertEqual(mu, 3.0)

    def test_update_negative(self):
        mu, nu = self.lm.update_negative(
            mu=1.0,
            nu=2.0
        )

        self.assertEqual(mu, 2.0)
        self.assertEqual(nu, 4.0)

    def test_optimize(self):
        # find p such that f(p) = x
        p = self.lm.optimize(max_iter=int(1e6))
        assert_array_almost_equal(p, np.array([1.0, 1.0]))


unittest.main()
