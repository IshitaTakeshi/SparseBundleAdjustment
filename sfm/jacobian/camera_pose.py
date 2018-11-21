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

    row = row_indices(n_viewpoints, n_3dpoints, n_pose_parameters)
    col = col_indices(n_viewpoints, n_3dpoints, n_pose_parameters)
    data = jacobians.flatten()
    return bsr_matrix((data, (row, col)), blocksize=((2, n_pose_parameters)))
