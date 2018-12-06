from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sfm.camera import CameraParameters
from sfm.projection import drotation_tensor, cross_product_matrix, rodrigues


def test_cross_product_matrix():
    GT = np.array([
        [0, -3, 2],
        [3, 0, -1],
        [-2, 1, 0]
    ])
    assert_array_equal(cross_product_matrix([1, 2, 3]), GT)


def test_drotation_tensor():
    t = np.random.randint(0, 10, 3)
    tx, ty, tz = t

    x, y, z = np.random.randint(0, 10, 3)

    D = drotation_tensor(t)
    R = np.array([
        [0, z, -y, y * tz - z * ty],
        [-z, 0, x, z * tx - x * tz],
        [y, -x, 0, x * ty - y * tx]
    ])

    # assuming that elements in D are linear
    assert_array_equal(D[0] * x + D[1] * y + D[2] * z, R)


def test_rodrigues():
    theta = np.pi / np.sqrt(2)
    vv = np.array([
        [1 / 2, 0, -1 / 2],
        [0, 0, 0],
        [-1 / 2, 0, 1 / 2]
    ])
    K = np.array([
        [0, 1 / np.sqrt(2), 0],
        [-1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
        [0, 1 / np.sqrt(2), 0]
    ])
    GT = np.cos(theta) * np.eye(3) + (1-np.cos(theta)) * vv + np.sin(theta) * K

    assert_array_almost_equal(rodrigues([np.pi / 2, 0, - np.pi / 2]), GT)


def test_jacobian_rotation():
    pass


def test_jacobian_translation():
    pass



test_cross_product_matrix()
test_drotation_tensor()
test_rodrigues()
