from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sfm.rigid import cross_product_matrix, rodrigues, transform3d


def test_cross_product_matrix():
    V = np.array([
        [3, 1, 2],
        [-2, 4, 3],
    ])

    GT = np.array([
        [[0, -2, 1],
         [2, 0, -3],
         [-1, 3, 0]],
        [[0, -3, 4],
         [3, 0, 2],
         [-4, -2, 0]]
    ])

    assert_array_equal(cross_product_matrix(V), GT)


def test_rodrigues():
    V = np.array([
        [np.pi / 2, 0, 0],
        [0, -np.pi / 2, 0],
        [0, 0, np.pi],
        [-np.pi, 0, 0]
    ])

    R = rodrigues(V)

    GT = np.empty((4, 3, 3))

    GT[0] = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    GT[1] = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    GT[2] = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

    GT[3] = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    assert_array_almost_equal(R, GT)


def test_transform3d():
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
        [[-1, -4, 10],  # [-2, -1, 2] + [1, -3, 8] = [-1, -4, 10]
         [-3, 6, 3],  # [-1, 2 -2] + [-2, 4, 5] = [-3, 6, 3]
         [10, -1, 8]],  # [2, -2, 1] + [8, 1, 7] = [10, -1, 8]
        [[2, -6, 6],  # [1, -3, -2] + [1, -3, 8] = [2, -6, 6]
         [-5, 2, 6],  # [-3, -2, 1] + [-2, 4, 5] = [-5, 2, 6]
         [7, 3, 10]],  # [-1, 2, 3] + [8, 1, 7] = [7, 3, 10]
        [[-2, -7, 9],  # [-3, -4, 1] + [1, -3, 8] = [-2, -7, 9]
         [-6, 5, 2],  # [-4, 1, -3] + [-2, 4, 5] = [-6, 5, 2]
         [11, 0, 11]],  # [3, -1, 4] + [8, 1, 7] = [11, 0, 11]
        [[2, -5, 16],  # [1, -2, 8] + [1, -3, 8] = [2, -5, 16]
         [-4, 12, 6],  # [-2, 8, 1] + [-2, 4, 5] = [-4, 12, 6]
         [7, -7, 9]]  # [-1, -8, 2] + [8, 1, 7] = [7, -7, 9]
    ]).astype(np.float64)

    X = transform3d(poses, points3d)
    np.set_printoptions(suppress=True)
    assert_array_almost_equal(X, GT)


test_cross_product_matrix()
test_rodrigues()
test_transform3d()
