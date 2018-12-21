import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


import numpy as np
from numpy.testing import assert_array_equal

from sfm.sba import SBA
from sfm.camera import CameraParameters
from sfm import config


class TestSBA(unittest.TestCase):
    def setUp(self):
        config.n_pose_parameters = 6
        config.n_point_parameters = 3
        self.sba = SBA(CameraParameters(1, 0), n_viewpoints=2, n_3dpoints=3)

        self.points3d = np.array([
            [9, 0, 2],
            [0, 9, 2],
            [3, 8, 4]
        ])

        self.poses = np.array([
            [1, 0, -2, 4, -3, 5],
            [1, -1, 1, -3, -9, 8]
        ])

        self.dpoints3d = np.array([
            [1e-2, 5e-2, 3e-2],
            [2e-2, 8e-2, 9e-2],
            [6e-2, 3e-2, 2e-2]
        ])

        self.dposes = np.array([
            [1e-2, 2e-2, -1e-2, -2e-2, 1e-2, 5e-2],
            [7e-2, 1e-2, 3e-2, -2e-2, -3e-2, 7e-2]
        ])

        self.p = np.array([
            1, 0, -2, 4, -3, 5,
            1, -1, 1, -3, -9, 8,
            9, 0, 2,
            0, 9, 2,
            3, 8, 4
        ])

    def test_compose(self):
        assert_array_equal(
            self.sba.compose(self.points3d, self.poses),
            self.p
        )

    def test_decompose(self):
        points3d, poses = self.sba.decompose(self.p)
        assert_array_equal(points3d, self.points3d)
        assert_array_equal(poses, self.poses)

    def observe_change(self, dp):
        x0 = self.sba.projection(self.p)
        x1 = self.sba.projection(self.p + dp)

        J = self.sba.jacobian(self.p)

        np.set_printoptions(precision=4, linewidth=1e8, suppress=True)

        print("J.dot(dp) / abs(x0)")
        print(J.dot(dp) / np.abs(x0))
        print("(x1 - x0) / abs(x0)")
        print((x1 - x0) / np.abs(x0))
        print("")

    def test_jacobian(self):
        print("Change all")
        dp = self.sba.compose(self.dpoints3d, self.dposes)
        self.observe_change(dp)

    def test_jacobian_3dpoints(self):
        print("Change only 3D points")
        dp = self.sba.compose(self.dpoints3d, np.zeros(self.dposes.shape))
        self.observe_change(dp)

    def test_jacobian_poses(self):
        print("Change only poses")
        dp = self.sba.compose(np.zeros(self.dpoints3d.shape), self.dposes)
        self.observe_change(dp)


unittest.main()
