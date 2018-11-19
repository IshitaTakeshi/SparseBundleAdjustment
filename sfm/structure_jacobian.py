import numpy as np
from scipy.sparse import bsr_matrix


def structure_jacobian(jacobians, n_3dpoints, n_viewpoints,
                       n_point_parameters):
    # Jacobian of projected points w.r.t structure parameters
    # the structure parameter side of J in the paper

    H = 2 * n_3dpoints * n_viewpoints
    W = n_point_parameters * n_3dpoints

    row = row_indices()
    col = column_indices()
    data = jacobians.flatten()
    JB = csr_matrix((data, (row, col)), blocksize=(2, n_point_parameters))

    return JB
