import numpy as np
from scipy.sparse import bsr_matrix


def row_indices(n_viewpoints, n_3dpoints, n_pose_parameters):
    U = np.arange(2 * n_viewpoints * n_3dpoints)
    return np.repeat(U, n_pose_parameters)


def col_indices(n_viewpoints, n_3dpoints, n_pose_parameters):
    N = n_pose_parameters * n_viewpoints
    U = np.arange(N).reshape(n_viewpoints, n_pose_parameters)
    U = np.repeat(U, 2, axis=0)  # projection is 2D
    U = np.tile(U, (n_3dpoints, 1))
    return U.flatten()


def camera_pose_jacobian(jacobians, n_3dpoints, n_viewpoints,
                         n_pose_parameters):
    # Jacobian of projected points w.r.t camera parameters
    # the camera parameter side of J in the paper

    # FIXME generation of indices is ugly
    # use np.repeat(np.arange(n).reshape(H, W), m)

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

    row = row_indices()
    col = col_indices()
    data = jacobians.flatten()
    return bsr_matrix((data, (row, col)), blocksize=((2, n_pose_parameters)))
