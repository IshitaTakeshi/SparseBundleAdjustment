from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal
from sfm.jacobian import camera_pose as CJ
from sfm.jacobian import structure as SJ


def test_camera_pose_jacobian_indices():
    # if n_pose_parameters = 4, n_viewpoints = 3, n_3dpoints = 2
    # the row indices should be
    # [0 0 0 0]
    # [1 1 1 1]
    #          [2 2 2 2]
    #          [3 3 3 3]
    #                   [4 4 4 4]
    #                   [5 5 5 5]
    # [6 6 6 6]
    # [7 7 7 7]
    #          [8 8 8 8]
    #          [9 9 9 9]
    #                   [10 10 10 10]
    #                   [11 11 11 11]

    row = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9, 9, 9,
        10, 10, 10, 10,
        11, 11, 11, 11,
    ])

    # and the column indices should be
    # [0 1 2 3]
    # [0 1 2 3]
    #          [4 5 6 7]
    #          [4 5 6 7]
    #                   [8 9 10 11]
    #                   [8 9 10 11]
    # [0 1 2 3]
    # [0 1 2 3]
    #          [4 5 6 7]
    #          [4 5 6 7]
    #                   [8 9 10 11]
    #                   [8 9 10 11]

    col = np.array([
        0, 1, 2, 3,
        0, 1, 2, 3,
        4, 5, 6, 7,
        4, 5, 6, 7,
        8, 9, 10, 11,
        8, 9, 10, 11,
        0, 1, 2, 3,
        0, 1, 2, 3,
        4, 5, 6, 7,
        4, 5, 6, 7,
        8, 9, 10, 11,
        8, 9, 10, 11
    ])

    n_viewpoints = 3
    n_3dpoints = 2
    n_pose_parameters = 4

    assert_array_equal(
        CJ.row_indices(n_viewpoints, n_3dpoints, n_pose_parameters),
        row
    )

    assert_array_equal(
        CJ.col_indices(n_viewpoints, n_3dpoints, n_pose_parameters),
        col
    )


def test_structure_jacobian_indices():
    # if n_point_parameters = 4, n_viewpoints = 3, n_3dpoints = 2
    # the row indices should be

    # [0 0 0 0]
    # [1 1 1 1]
    # [2 2 2 2]
    # [3 3 3 3]
    # [4 4 4 4]
    # [5 5 5 5]
    #          [ 6  6  6  6]
    #          [ 7  7  7  7]
    #          [ 8  8  8  8]
    #          [ 9  9  9  9]
    #          [10 10 10 10]
    #          [11 11 11 11]

    row = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9, 9, 9,
        10, 10, 10, 10,
        11, 11, 11, 11
    ])

    # and the column indices should be
    # [0 1 2 3]
    # [0 1 2 3]
    # [0 1 2 3]
    # [0 1 2 3]
    # [0 1 2 3]
    # [0 1 2 3]
    #          [4 5 6 7]
    #          [4 5 6 7]
    #          [4 5 6 7]
    #          [4 5 6 7]
    #          [4 5 6 7]
    #          [4 5 6 7]

    col = np.array([
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        0, 1, 2, 3,
        4, 5, 6, 7,
        4, 5, 6, 7,
        4, 5, 6, 7,
        4, 5, 6, 7,
        4, 5, 6, 7,
        4, 5, 6, 7
    ])

    n_viewpoints = 3
    n_3dpoints = 2
    n_point_parameters = 4

    assert_array_equal(
        SJ.row_indices(n_viewpoints, n_3dpoints, n_point_parameters),
        row
    )

    assert_array_equal(
        SJ.col_indices(n_viewpoints, n_3dpoints, n_point_parameters),
        col
    )


def test_camera_pose_jacobian_construction():
    # n_3dpoints=2, n_viewpoints=3, n_pose_parameters=4
    jacobians = np.array([
        [[[6, 5, 8, 6],
          [4, 0, 3, 0]],
         [[2, 8, 0, 4],
          [7, 6, 3, 9]],
         [[5, 3, 0, 9],
          [1, 4, 9, 3]]],
        [[[8, 1, 7, 9],
          [0, 3, 5, 8]],
         [[2, 1, 2, 3],
          [9, 2, 4, 4]],
         [[8, 7, 7, 7],
          [3, 7, 5, 1]]]
    ])

    GT = np.array([
        [6, 5, 8, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 8, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 6, 3, 9, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0, 9],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 9, 3],
        [8, 1, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 5, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 1, 2, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 9, 2, 4, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 7, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 5, 1]
    ])

    assert_array_equal(CJ.camera_pose_jacobian(jacobians).todense(), GT)


def test_structure_jacobian_construction():
    jacobians = np.array([
        [[[2, 9, 6, 8],
          [7, 3, 4, 5]],
         [[3, 5, 2, 7],
          [5, 0, 7, 7]],
         [[7, 3, 2, 5],
          [1, 5, 1, 5]]],
        [[[7, 6, 3, 6],
          [1, 9, 4, 1]],
         [[7, 6, 2, 2],
          [6, 1, 9, 6]],
         [[4, 8, 6, 4],
          [5, 6, 8, 0]]]
    ])

    GT = np.array([
        [2, 9, 6, 8, 0, 0, 0, 0],
        [7, 3, 4, 5, 0, 0, 0, 0],
        [3, 5, 2, 7, 0, 0, 0, 0],
        [5, 0, 7, 7, 0, 0, 0, 0],
        [7, 3, 2, 5, 0, 0, 0, 0],
        [1, 5, 1, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 6, 3, 6],
        [0, 0, 0, 0, 1, 9, 4, 1],
        [0, 0, 0, 0, 7, 6, 2, 2],
        [0, 0, 0, 0, 6, 1, 9, 6],
        [0, 0, 0, 0, 4, 8, 6, 4],
        [0, 0, 0, 0, 5, 6, 8, 0]
    ])
    assert_array_equal(SJ.structure_jacobian(jacobians).todense(), GT)


test_camera_pose_jacobian_indices()
test_structure_jacobian_indices()
test_camera_pose_jacobian_construction()
test_structure_jacobian_construction()
