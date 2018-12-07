from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.testing import assert_array_equal
from sfm.jacobian import camera_pose as CJ
from sfm.jacobian import structure as SJ


def test_camera_pose_jacobian():
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
    #

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


def test_structure_jacobian():
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


test_camera_pose_jacobian()
test_structure_jacobian()
