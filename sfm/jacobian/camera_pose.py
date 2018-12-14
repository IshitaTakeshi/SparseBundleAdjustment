import numpy as np
from scipy.sparse import bsr_matrix


def row_indices(n_3dpoints, n_viewpoints, n_pose_parameters):
    U = np.arange(2 * n_3dpoints * n_viewpoints)
    return np.repeat(U, n_pose_parameters)


def col_indices(n_3dpoints, n_viewpoints, n_pose_parameters):
    N = n_pose_parameters * n_viewpoints
    U = np.arange(N).reshape(n_viewpoints, n_pose_parameters)
    U = np.repeat(U, 2, axis=0)  # projection is 2D
    U = np.tile(U, (n_3dpoints, 1))
    return U.flatten()


def camera_pose_jacobian(jacobians):
    """
    Jacobian of projected points w.r.t camera parameters
    :math:`A_{ij}` in the original paper

    Args:
        jacobians: Array of Jacobians which its shape is
            (n_3dpoints, n_viewpoints, 2, n_pose_parameters)
    """

    n_3dpoints, n_viewpoints, _, n_pose_parameters = jacobians.shape

    row = row_indices(n_3dpoints, n_viewpoints, n_pose_parameters)
    col = col_indices(n_3dpoints, n_viewpoints, n_pose_parameters)
    data = jacobians.flatten()
    return bsr_matrix((data, (row, col)), blocksize=((2, n_pose_parameters)))
