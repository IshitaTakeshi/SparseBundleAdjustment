from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sfm.camera import CameraParameters
from sfm.projection import (cross_product_matrix, rodrigues, projection_,
                            jacobian_wrt_exp_coordinates, jacobian_pi,
                            jacobian_pose_and_3dpoint)


def test_cross_product_matrix():
    GT = np.array([
        [0, -3, 2],
        [3, 0, -1],
        [-2, 1, 0]
    ])
    assert_array_equal(cross_product_matrix([1, 2, 3]), GT)


def test_jacobian_wrt_exp_coordinates():
    R = rodrigues([np.pi / 2, 0, -np.pi / 2])
    v = [np.pi / 2, 0, -np.pi / 2]
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


def test_jacobian_pi():
    GT = np.array([
        [1 / 4, 0, -3 / 16],
        [0, 1 / 4, -2 / 16]
    ])
    assert_array_equal(jacobian_pi([3, 2, 4]), GT)


def test_jaocobian_pose_and_3dpoint():
    K = CameraParameters(1, 0)

    def test_pose_approximation():
        b = np.array([1.0, 1.0, 1.0])

        a0 = np.array([1.00, 1.00, 1.00, 0.0, 0.0, 0.0])
        a1 = np.array([0.95, 1.05, 1.00, 0.0, 0.0, 0.0])

        p1 = projection_(K, rodrigues(a1[:3]), a1[3:], b)
        p0 = projection_(K, rodrigues(a0[:3]), a0[3:], b)
        JA, JB = jacobian_pose_and_3dpoint(K, a0, b)
        print("diff         : ", p1 - p0)
        print("linearization: ", np.dot(JA, a1-a0))

        a0 = np.array([1.0, 0.0, 0.0, 1.00, 1.00, 1.00])
        a1 = np.array([1.0, 0.0, 0.0, 1.01, 1.01, 1.01])

        p1 = projection_(K, rodrigues(a1[:3]), a1[3:], b)
        p0 = projection_(K, rodrigues(a0[:3]), a0[3:], b)
        JA, JB = jacobian_pose_and_3dpoint(K, a0, b)
        print("diff         : ", p1 - p0)
        print("linearization: ", np.dot(JA, a1-a0))

    def test_3dpoint_approximation():
        b0 = np.array([1.00, 1.00, 1.00])
        b1 = np.array([1.01, 1.01, 1.00])
        a = np.array([1.0, 0.0, 0.0, 1.00, 1.00, 1.00])

        p1 = projection_(K, rodrigues(a[:3]), a[3:], b1)
        p0 = projection_(K, rodrigues(a[:3]), a[3:], b0)
        JA, JB = jacobian_pose_and_3dpoint(K, a, b0)
        print("diff         : ", p1 - p0)
        print("linearization: ", np.dot(JB, b1-b0))

    test_pose_approximation()
    test_3dpoint_approximation()


def test_projection_():
    R = np.array([
        [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [0, 1, 0],
        [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4)]
    ])
    t = np.array([0, 1, 0])
    b = np.array([0, 0, 1])
    K = CameraParameters(np.sqrt(2), 2)

    GT = np.array([2 + np.sqrt(2), 4])

    assert_array_almost_equal(projection_(K, R, t, b), GT)


def test_jacobian_translation():
    pass



test_cross_product_matrix()
test_rodrigues()
test_jacobian_wrt_exp_coordinates()
test_projection_()
test_jaocobian_pose_and_3dpoint()
