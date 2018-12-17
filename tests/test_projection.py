from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from scipy import sparse

from sfm.camera import CameraParameters
from sfm.projection import (projection_, projection,
                            jacobian_wrt_exp_coordinates, jacobian_pi,
                            jacobian_pose_and_3dpoint)


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


def test_jacobian_pi():
    GT = np.array([
        [1 / 4, 0, -3 / 16],
        [0, 1 / 4, -2 / 16]
    ])
    assert_array_equal(jacobian_pi([3, 2, 4]), GT)


def test_jaocobian_pose_and_3dpoint():
    K = CameraParameters(1, 0).matrix

    def test_rotation_approximation():
        b = np.array([1.0, 1.0, 1.0])
        t = np.array([1.0, 2.0, 2.0])

        v0 = np.array([1.00, 0.502, 0.050])
        v1 = np.array([1.01, 0.500, 0.051])

        a0 = np.hstack([v0, t])
        a1 = np.hstack([v1, t])

        x0 = projection_(K, rodrigues(v0), t, b)
        x1 = projection_(K, rodrigues(v1), t, b)
        JA, JB = jacobian_pose_and_3dpoint(K, rodrigues(v0), v0, t, b)

        print("JA.dot([dv 0])")
        print("diff         : ", x1 - x0)
        print("linearization: ", JA.dot(a1-a0))

    def test_translation_approximation():
        b = np.array([1.0, 1.0, 1.0])
        v = np.array([1.0, 0.0, 0.0])
        R = rodrigues(v)
        t0 = np.array([1.00, 1.00, 1.00])
        t1 = np.array([1.01, 1.01, 1.01])

        a0 = np.hstack([v, t0])
        a1 = np.hstack([v, t1])

        x0 = projection_(K, R, t0, b)
        x1 = projection_(K, R, t1, b)
        JA, JB = jacobian_pose_and_3dpoint(K, R, v, t0, b)

        print("JA.dot([0 dt])")
        print("diff         : ", x1 - x0)
        print("linearization: ", JA.dot(a1-a0))

    def test_3dpoint_approximation():
        b0 = np.array([1.00, 1.00, 1.00])
        b1 = np.array([1.01, 1.01, 1.01])
        v = np.array([1.0, 0.0, 0.0])
        R = rodrigues(v)
        t = np.array([1.00, 1.00, 1.00])

        x0 = projection_(K, R, t, b0)
        x1 = projection_(K, R, t, b1)
        JA, JB = jacobian_pose_and_3dpoint(K, R, v, t, b0)

        print("JB.dot(db)")
        print("diff         : ", x1 - x0)
        print("linearization: ", JB.dot(b1-b0))

    def test_pose_and_3dpoint_approximation():
        b0 = np.array([1.00, 1.00, 1.00])
        b1 = np.array([1.03, 0.98, 1.02])

        v0 = np.array([1.00, 0.502, 0.500])
        v1 = np.array([1.01, 0.500, 0.501])

        t0 = np.array([1.00, 1.00, 1.00])
        t1 = np.array([0.97, 1.02, 0.99])

        a0 = np.hstack([v0, t0])
        a1 = np.hstack([v1, t1])

        x0 = projection_(K, rodrigues(v0), t0, b0)
        x1 = projection_(K, rodrigues(v1), t1, b1)
        JA, JB = jacobian_pose_and_3dpoint(K, rodrigues(v0), v0, t0, b0)

        print("JA.dot([dv dt]) + JB.dot(db)")
        print("diff         : ", x1 - x0)
        print("linearization: ", JA.dot(a1-a0) + JB.dot(b1-b0))

    test_rotation_approximation()
    test_translation_approximation()
    test_3dpoint_approximation()
    test_pose_and_3dpoint_approximation()


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


def test_projection_():
    R = np.array([
        [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [0, 1, 0],
        [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4)]
    ])
    t = np.array([0, 1, 0])
    b = np.array([0, 0, 1])
    K = CameraParameters(np.sqrt(2), 2).matrix

    GT = np.array([2 + np.sqrt(2), 4])

    assert_array_almost_equal(projection_(K, R, t, b), GT)


def print_rotation_approximation():
    v0 = np.array([1.00, 1.00, 1.00])
    v1 = np.array([1.01, 1.01, 1.01])
    u = np.array([-2.5, 1, 1.5])
    J = jacobian_wrt_exp_coordinates(rodrigues(v0), v0, u)
    x1 = np.dot(rodrigues(v1), u)
    x0 = np.dot(rodrigues(v0), u)
    print("exp")
    print("diff         : ", x1 - x0)
    print("linearization: ", np.dot(J, v1-v0))


test_cross_product_matrix()
test_rodrigues()
test_transform3d()
# test_jacobian_wrt_exp_coordinates()
test_projection()
# test_projection_()
# test_jaocobian_pose_and_3dpoint()
# test_jacobian_pi()
# print_rotation_approximation()
