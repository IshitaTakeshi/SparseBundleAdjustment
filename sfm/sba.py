from scipy.sparse import bsr_matrix


n_pose_parameters  # dimensions of a_j
n_viewpoints # `m` in the paper
n_3dpoints  # `n` in the paper
n_point_parameters = 3  # dimensions of b_i


H = n_3dpoints * n_viewpoints


def calc_jacobian(initial_rotation, camera_pose):
    P = np.empty((n_viewpoints, 2, 6))
    S = np.empty((n_3dpoints, 2, 6))
    for i in range(n_3dpoints):
        for j in range(n_viewpoints):
            P[j], S[i] = calc_image_jacobian(
                intrinsic,
                initial_rotation,
                camera_pose[j],
                points3d[i]
            )

    A = camera_pose_jacobian(P)
    B = structure_jacobian(S)
    return A, B


def inverse_u(U):
    # inv(U) = diag(inv(U_1), ..., inv(U_i), ..., inv(U_n))
    for i in range(n_3dpoints):
        s = i * n_point_parameters
        t = (i+1) * n_point_parameters
        U[s:t, s:t] = inv(U[s:t, s:t].todense())
    return U


def inverse_v(V):
    # inv(V) = diag(inv(V_1), ..., inv(V_i), ..., inv(V_m))
    for j in range(n_viewpoints):
        s = j * n_viewpoints
        t = (j+1) * n_viewpoints
        V[s:t, s:t] = inv(V[s:t, s:t].todense())
    return V


def calc_update(A, B, C_inv, epsilon):
    # eq. 12
    U = A.dot(C_inv).dot(A.T)
    W = A.dot(C_inv).dot(B.T)
    V = B.dot(C_inv).dot(B.T)

    CE = np.dot(C_inv, epsilon)
    epsilon_a = np.dot(A.T, CE)
    epsilon_b = np.dot(B.T, CE)

    # add the damping term to the diagonals
    U = U + mu * np.eye(n_pose_parameters * n_viewpoints)
    V = V + mu * np.eye(n_point_parameters * n_3dpoints)

    V_inv = inverse_v(V)
    U_inv = inverse_u(U)
    Y = W.dot(V_inv)

    S = U_inv - Y.dot(W.T)

    # eq. 21: calculate update of the pose parameters
    delta_a = np.linalg.solve(S, epsilon_a - np.dot(Y, epsilon_b))
    # eq. 22: calculate update of the structure parameters
    delta_b = np.linalg.solve(V, epsilon_b - np.dot(W, delta_a))

    return delta_a, delta_b


def optimize():
    initial_quaternion = np.hstack(
        np.ones(n_viewpoints, 1),
        np.zeros(n_viewpoints, 3)
    )

    while condition():
        x_pred = projection(
            intrinsic,
            initial_rotation,
            camera_pose,
            points3d
        )

        A, B = calc_jacobian(initial_rotation, camera_pose, points3d)

        delta_a, delta_b = calc_update(A, B, C_inv, x_pred - x_observation)

        pose_parameters -= delta_a.reshape(pose_parameters.shape)
        structure_parameters -= delta_b.reshape(structure_parameters)

    return pose_parameters, structure_parameters



def camera_pose_jacobian(jacobians):
    # Jacobian of projected points w.r.t camera parameters
    # the camera parameter side of J in the paper

    # each projected point has 2 dimensions
    H = 2 * n_3dpoints * n_viewpoints
    W = n_pose_parameters * n_viewpoints

    def column_indices():
        def block_column_indices(j):
            s = i * n_pose_parameters
            t = (i+1) * n_pose_parameters
            return np.arange(s, t)
        K = [np.tile(block_column_indices(j), 2) for j in n_viewpoints]
        return np.tile(K, n_3dpoints)

    def row_indices():
        return np.repeat(np.arange(2 * n_viewpoints), n_3dpoints)

    row = row_indices()
    col = column_indices()
    data = jacobians.flatten()
    JA = csr_matrix((data, (row, col)), blocksize=((2, n_pose_parameters)))
    return JA


def structure_jacobian(jacobians):
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


    H = 2 * n_3dpoints * n_viewpoints
    W = n_point_parameters * n_3dpoints

    row = row_indices()
    col = column_indices()
    data = jacobians.flatten()
    JB = csr_matrix((data, (row, col)), blocksize=(2, n_point_parameters))

    return JB


    C_inv = csr_matrix((H, H), blocksize=(2, 2))

    for i in range(n_observed_points):
        C_inv[2*i:2*i+2, 2*i:2*i+2] = covariance

    JB.T.dot(C).dot(JB)


def sparse_eye_matrix(size):
    row = np.arange(size)
    col = np.arange(size)
    data = np.ones(size)
    return bsr_matrix((data, (row, col)))


def inv_covariance(inv_covariances=None):
    if inv_covariances is None:
        return sparse_eye_matrix(2 * len(inv_covariances))

    size = n_viewpoints * n_3dpoints  # number of 2d points

    def column_indices(i):
        def block_column_indices(j):
            s = j * 2
            t = (j + 1) * 2
            return np.arange(s, t)
        return np.concatenate([block_row_indices(j) for j in np.arange(size)])

    def row_indices():
        return np.repeat(np.arange(size), 2)

    data = inv_covariances.flatten()
    row = row_indices()
    col = column_indices()
    return bsr_matrix((data, (row, col)), blocksize=(2, 2))

