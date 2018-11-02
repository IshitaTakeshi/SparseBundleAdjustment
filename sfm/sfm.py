# SBA: A Software Package for Generic Sparse Bundle Adjustment
# http://users.ics.forth.gr/~lourakis/sba/sba-toms.pdf

def image_jacobian_3dpoints():
    """Jacobian of image projections w.r.t 3D points"""
    return J


def image_jacobian_camera_params():
    """Jacobian of image projections w.r.t camera parameters"""
    return J


def covariance_matrix():
    return


def compute_U():

de jacobian():
    C = inv(Sigma)
    U = A.T @ C @ A
    V = B.T @ C @ B
    W = A.T @ C @ B


def update_camera_parameters():
    Y = np.dot(W, np.linalg.inv(V))
    S = U - np.dot(Y, W.T)

    e = epsilon_a - np.dot(Y, epsilon_b)

    delta_a = np.linalg.solve(S, e)

    # TODO inv(V) can be decomposed
    delta_b = inv(V) * (epsilon_b - np.dot(W.T, delta_a))

    return delta_a, delta_b
