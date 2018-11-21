import numpy as np
from scipy.sparse import bsr_matrix


def row_indices(n_viewpoints, n_3dpoints, n_point_parameters):
    U = np.arange(2 * n_viewpoints * n_3dpoints)
    return np.repeat(U, n_point_parameters)


def col_indices(n_viewpoints, n_3dpoints, n_point_parameters):
    U = np.arange(n_3dpoints * n_point_parameters)
    U = U.reshape(n_3dpoints, n_point_parameters)
    return np.repeat(U, 2 * n_viewpoints, axis=0).flatten()


def structure_jacobian(jacobians, n_3dpoints, n_viewpoints,
                       n_point_parameters):
    # Jacobian of projected points w.r.t structure parameters
    # the structure parameter side of J in the paper

    row = row_indices(n_viewpoints, n_3dpoints, n_point_parameters)
    col = col_indices(n_viewpoints, n_3dpoints, n_point_parameters)
    data = jacobians.flatten()
    JB = csr_matrix((data, (row, col)), blocksize=(2, n_point_parameters))

    return JB
