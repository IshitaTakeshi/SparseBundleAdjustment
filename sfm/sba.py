from scipy.sparse import bsr_matrix


n_camera_parameters  # dimensions of a_j
n_viewpoints # `m` in the paper
n_3dpoints  # `n` in the paper
n_point_parameters = 3  # dimensions of b_i

def optimize(self):
    l()
    pass

H = n_3dpoints * n_viewpoints


def camera_jacobian(n_total_observations):
    # Jacobian of projected points w.r.t camera parameters
    # the camera parameter side of J in the paper

    # each projected point has 2 dimensions
    H = 2 * n_total_observations
    W = n_camera_parameters * n_viewpoints

    def column_indices():
        def block_column_indices(j):
            s = i * n_camera_parameters
            t = (i+1) * n_camera_parameters
            return np.arange(s, t)
        K = [np.tile(block_column_indices(j), 2) for j in n_viewpoints]
        return np.tile(K, n_3dpoints)

    def row_indices():
        return np.repeat(np.arange(2 * n_viewpoints), n_3dpoints)

    row = row_indices()
    col = column_indices()
    data = jacobians.flatten()
    JA = bsr_matrix((data, (row, col)), blocksize=((2, n_camera_parameters)))


def structure_jacobian(n_total_observations):
    # Jacobian of projected points w.r.t structure parameters
    # the structure parameter side of J in the paper

    def row_indices():
        def block_row_indices(i):
            s = i * n_point_parameters
            t = (i+1) * n_point_parameters
            return np.arange(s, t)
        indices = np.tile(np.arange(n_3dpoints), 2 * n_viewpoints])
        return np.concatenate([block_row_indices(i) for i in indices])

    def column_indices():
        H = 2 * n_viewpoints * n_3dpoints
        # repeat column indices horizontally
        return np.repeat(np.arange(H), n_point_parameters)


    H = 2 * n_total_observations
    W = n_point_parameters * n_3dpoints

    row = row_indices()
    col = column_indices()
    data = jacobians.flatten()
    JB = bsr_matrix((data, (row, col)), blocksize=(2, n_point_parameters))

    return JB

    C = bsr_matrix((H, H), blocksize=(2, 2))

    for i in range(n_observed_points):
        C[2*i:2*i+2, 2*i:2*i+2] = covariance

    JB.T.dot(C).dot(JB)


def covariance():
    size = n_viewpoints * n_3dpoints  # number of 2d points
    def column_indices(i):
        def block_column_indices(j):
            s = j * 2
            t = (j + 1) * 2
            return np.arange(s, t)
        return np.concatenate([block_row_indices(j) for j in np.arange(size)])

    def row_indices():
        return np.repeat(np.arange(size), 2)

    data = covariances.flatten()
    row = row_indices()
    col = column_indices()
    return bsr_matrix((data, (row, col)), blocksize=(2, 2))
