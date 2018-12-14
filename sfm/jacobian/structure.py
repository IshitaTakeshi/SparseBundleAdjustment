import numpy as np
from scipy.sparse import bsr_matrix


def row_indices(n_3dpoints, n_viewpoints, n_point_parameters):
    U = np.arange(2 * n_viewpoints * n_3dpoints)
    return np.repeat(U, n_point_parameters)


def col_indices(n_3dpoints, n_viewpoints, n_point_parameters):
    U = np.arange(n_3dpoints * n_point_parameters)
    U = U.reshape(n_3dpoints, n_point_parameters)
    return np.repeat(U, 2 * n_viewpoints, axis=0).flatten()


def structure_jacobian(jacobians):
    """
    Jacobian of projected points w.r.t structure parameters.
    :math:`B_{ij}` in the paper

    Args:
        jacobians: Array of Jacobians which its shape is
            (n_3dpoints, n_viewpoints, 2, n_point_parameters)
    """

    n_3dpoints, n_viewpoints, _, n_point_parameters = jacobians.shape

    row = row_indices(n_3dpoints, n_viewpoints, n_point_parameters)
    col = col_indices(n_3dpoints, n_viewpoints, n_point_parameters)
    data = jacobians.flatten()
    return csr_matrix((data, (row, col)))
