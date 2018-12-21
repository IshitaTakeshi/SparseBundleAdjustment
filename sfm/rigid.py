import numpy as np


def cross_product_matrix(V):
    K = np.empty((V.shape[0], 3, 3))
    K[:, [0, 1, 2], [0, 1, 2]] = 0  # diag(K) = 0
    K[:, 2, 1] = V[:, 0]  # x
    K[:, 1, 2] = -V[:, 0]  # -x
    K[:, 0, 2] = V[:, 1]  # y
    K[:, 2, 0] = -V[:, 1]  # -y
    K[:, 1, 0] = V[:, 2]  # z
    K[:, 0, 1] = -V[:, 2]  # -z
    return K


def rodrigues(V):
    """
    # see
    # https://docs.opencv.org/2.4/modules/calib3d/doc/
    # camera_calibration_and_3d_reconstruction.html#rodrigues

    .. codeblock:
        theta = np.linalg.norm(r)
        r = r / theta
        K = cross_product_matrix_(r)
        I = np.eye(3, 3)
        return I + np.sin(theta) * K + (1-np.cos(theta)) * np.dot(K, K)

    # I + sin(theta) * K + (1-cos(theta)) * dot(K, K) is equivalent to
    # cos(theta) * I + (1-cos(theta)) * outer(r, r) + sin(theta) * K
    """

    assert(V.shape[1] == 3)

    N = V.shape[0]

    # HACK this can be accelerated by calculating (V * V).sum(axis=1)
    theta = np.linalg.norm(V, axis=1)
    V = V / theta[:, np.newaxis]
    K = cross_product_matrix(V)

    A = np.zeros((N, 3, 3))
    A[:, [0, 1, 2], [0, 1, 2]] = 1  # [np.eye(3) for i in range(N)]

    B = np.einsum('i,ijk->ijk', np.sin(theta), K)
    C = np.einsum('ijk,ikl->ijl', K, K)  # [dot(L, L) for L in K]
    C = np.einsum('i,ijk->ijk', 1-np.cos(theta), C)
    return A + B + C


def transform3d(poses, points3d):
    """
        Rigid body transformation

        Returns:
            Transformed points of shape (n_3dpoints, n_viewpoints, 3)
    """

    V = poses[:, :3]  # V.shape = (n_viewpoints, 3)
    T = poses[:, 3:]  # T.shape = (n_viewpoints, 3)
    R = rodrigues(V)  # R.shape = (n_viewpoints, 3, 3)

    # The following computation is equivalent to
    # [[projection_(K, R, t, b) for R_, t in zip(R, T)] for b in points3d]
    # where projection_(K, R, t, b) = pi(np.dot(K, transform3d(R, t, b)))

    # X.shape = (n_3dpoints, n_viewpoints, 3)
    X = np.einsum('ijk,lk->lij', R, points3d)
    X = X + T[np.newaxis]
    return X
