from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from scipy import sparse

from sfm.camera import CameraParameters
from sfm.projection import (projection, jacobian_wrt_exp_coordinates,
                            jacobian_pi)


def test_jacobian_wrt_exp_coordinates():
    """
    calculate d(R(v) * u) / dv
    """

    v = [np.pi / 2, 0, -np.pi / 2]
    R = rodrigues(v)

    u = [0, 2, 1]
    I = np.eye(3)
    A = np.array([
        [0, -1, 2],
        [1, 0, 0],
        [-2, 0, 0]
    ])
    B = np.array([
        [1, 0, -1],
        [0, 0, 0],
        [-1, 0, 1]
    ])
    C = np.array([
        [0, 1, 0],
        [-1, 0, -1],
        [0, 1, 0]
    ])
    D = 1/2 * B + 1/np.pi * np.dot(R.T-I, C)
    GT = -R.dot(A).dot(D)

    R = rodrigues(v)
    assert_array_almost_equal(jacobian_wrt_exp_coordinates(R, v, u), GT)


def test_jacobian_wrt_exp_coordinates():
    """
    calculate [[d(R(v) * u) / dv for v in vs] for u in U]
    See https://arxiv.org/pdf/1312.0788.pdf
    """

    # rotation parameters
    V = np.array([
        [0, 0, np.pi/2],
        [-np.pi/2, 0, 0]
    ])

    # corresponding rotaton matrices
    RS = np.array([
        [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 1]],
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]]
    ])

    # corresponding cross product matrices
    XVS = np.array([
        [[0, -np.pi / 2, 0],
         [np.pi / 2, 0, 0],
         [0, 0, 0]],
        [[0, 0,  0],
         [0, 0, np.pi / 2],
         [0, -np.pi / 2, 0]]
    ])

    B = np.array([
        [-1, 2, 1],
        [0, 3, 1]
    ])

    XBS = np.array([
        [[0, -1, 2],
         [1, 0, 1],
         [-2, -1, 0]],
        [[0, -1, 3],
         [1, 0, 0],
         [-3, 0, 0]]
    ])

    I = np.eye(3)

    JV = jacobian_wrt_exp_coordinates(V, B)
    for i, XB in enumerate(XBS):
        for j, (v, R, XV) in enumerate(zip(V, RS, XVS)):
            # Eq. (8).  d(R(v) * v) / dv
            GT = -R.dot(XB).dot(np.outer(v, v) + np.dot(R.T-I, XV)) / np.dot(v, v)
            assert_array_almost_equal(JV[i, j], GT)


def test_jacobian_pi():
    P = np.array([
        [3, 2, 4],
        [8, 7, 5]
    ])

    GT = np.array([
        [[1 / 4, 0, -3 / 16],
         [0, 1 / 4, -2 / 16]],
        [[1 / 5, 0, -8 / 25],
         [0, 1 / 5, -7 / 25]]
    ])

    assert_array_equal(jacobian_pi(P), GT)


def test_projection():
    camera_parameters = CameraParameters(10, 0)

    poses = np.array([
        [np.pi / 2, 0, 0, 1, -3, 8],
        [0, -np.pi / 2, 0, -2, 4, 5],
        [0, 0, np.pi, 8, 1, 7]
    ])

    points3d = np.array([
        [-2, 2, 1],
        [1, -2, 3],
        [-3, 1, 4],
        [1, 8, 2]
    ])

    GT = np.array([
        [[-1, -4],  # [-2, -1, 2] + [1, -3, 8] = [-1, -4, 10]
         [-10, 20],  # [-1, 2 -2] + [-2, 4, 5] = [-3, 6, 3]
        [100/8, -10/8]],  # [2, -2, 1] + [8, 1, 7] = [10, -1, 8]
        [[20/6, -10],  # [1, -3, -2] + [1, -3, 8] = [2, -6, 6]
         [-50/6, 20/6],  # [-3, -2, 1] + [-2, 4, 5] = [-5, 2, 6]
         [7, 3]],  # [-1, 2, 3] + [8, 1, 7] = [7, 3, 10]
        [[-20/9, -70/9],  # [-3, -4, 1] + [1, -3, 8] = [-2, -7, 9]
         [-30, 25],  # [-4, 1, -3] + [-2, 4, 5] = [-6, 5, 2]
         [10, 0]],  # [3, -1, 4] + [8, 1, 7] = [11, 0, 11]
        [[20/16, -50/16],  # [1, -2, 8] + [1, -3, 8] = [2, -5, 16]
         [-40/6, 20],  # [-2, 8, 1] + [-2, 4, 5] = [-4, 12, 6]
         [70/9, -70/9]]  # [-1, -8, 2] + [8, 1, 7] = [7, -7, 9]
    ]).astype(np.float64)

    X = projection(camera_parameters, poses, points3d)
    assert_array_almost_equal(X, GT)


test_jacobian_wrt_exp_coordinates()
test_projection()
test_jacobian_pi()
