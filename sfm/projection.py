import numpy as np
from sfm.jacobian import camera_pose_jacobian, structure_jacobian
from sfm.config import n_pose_parameters, n_point_parameters


def cross_product_matrix_(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


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


# @profile
# TODO this can be accelerated by calculating jacobians for all points at once
def jacobian_pi(p):
    """
    Jacobian of the projection function defined below w.r.t point \\mathbf{p}

    .. math:
        pi([x, y, z]) = \\frac{1}{z} \\begin{bmatrix}
            x \\\\
            y
        \\end{bmatrix}
    """

    x, y, z = p
    return np.array([
        [1 / z, 0, -x / pow(z, 2)],
        [0, 1 / z, -y / pow(z, 2)],
    ])


# @profile
def jacobian_wrt_exp_coordinates(R, v, b):
    """
    Calculate
    .. math::
    \\begin{align}
        \\frac{\partial (R(\\mathbf{v})\\mathbf{b} + \\mathbf{t})}{\\partial \\mathbf{v}}
        &= ...
    \\end{align}
    """

    # See https://arxiv.org/abs/1312.0788

    U = cross_product_matrix(b)
    V = cross_product_matrix(v)
    I = np.eye(3)
    S = np.outer(v, v) + np.dot(R.T - I, V)
    return -R.dot(U).dot(S) / np.dot(v, v)


# @profile
def jacobian_pose_and_3dpoint(K, R, v, t, b):
    p = np.dot(K, transform3d(R, t, b))
    JP = jacobian_pi(p)
    JV = jacobian_wrt_exp_coordinates(R, v, b)
    JR = JP.dot(JV)
    JT = JP.dot(K)
    JA = np.hstack([JR, JT])
    JB = JT.dot(R)
    return JA, JB


# @profile
def jacobian_projection(camera_parameters, poses, points3d):
    """
    Args:
        poses (np.ndarray): Camera poses of shape
            (n_viewpoints, n_pose_parameters)
        points3d (np.ndarray): 3D point coordinates of shape
            (n_3dpoints, n_point_parameters)

    Returns:
        A: Left side of the Jacobian.
           :math:`\\frac{\\partial X}{\\partial \\a_j}, j=1,\dots,m`
        B: Right side of the Jacobian.
           :math:`\\frac{\\partial X}{\\partial \\b_i}, i=1,\dots,n`
    """

    n_viewpoints = poses.shape[0]
    n_3dpoints = points3d.shape[0]
    K = camera_parameters.matrix

    P = np.empty((n_3dpoints, n_viewpoints, 2, n_pose_parameters))
    S = np.empty((n_3dpoints, n_viewpoints, 2, n_point_parameters))

    for j, a in enumerate(poses):
        v, t = a[:3], a[3:]
        R = rodrigues(v)
        for i, b in enumerate(points3d):
            P[i, j], S[i, j] = jacobian_pose_and_3dpoint(K, R, v, t, b)

    JA = camera_pose_jacobian(P)
    JB = structure_jacobian(S)
    return JA, JB


def rodrigues_(r):
    # see
    # https://docs.opencv.org/2.4/modules/calib3d/doc/
    # camera_calibration_and_3d_reconstruction.html#rodrigues

    theta = np.linalg.norm(r)
    r = r / theta
    K = cross_product_matrix_(r)
    I = np.eye(3, 3)

    # I + sin(theta) * K + (1-cos(theta)) * dot(K, K) is equivalent to
    # cos(theta) * I + (1-cos(theta)) * outer(r, r) + sin(theta) * K
    return I + np.sin(theta) * K + (1-np.cos(theta)) * np.dot(K, K)


def rodrigues(V):
    assert(V.shape[1] == 3)

    N = V.shape[0]

    theta = np.linalg.norm(V, axis=1)
    V = V / theta[:, np.newaxis]
    K = cross_product_matrix(V)

    A = np.zeros((N, 3, 3))
    A[:, [0, 1, 2], [0, 1, 2]] = 1  # [np.eye(3) for i in range(N)]

    B = np.einsum('i,ijk->ijk', np.sin(theta), K)
    C = np.einsum('ijk,ikl->ijl', K, K)  # [dot(L, L) for L in K]
    C = np.einsum('i,ijk->ijk', 1-np.cos(theta), C)
    return A + B + C


def pi(p):
    return p[0:2] / p[2]


def transform3d(poses, points3d):
    """
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


def projection(camera_parameters, poses, points3d):

    """
    Project 3D points to multiple image planes

    Args:
        camera_parameters (CameraParameters): Camera intrinsic parameters
        poses (np.ndarray): Camera pose array of shape
            (n_viewpoints, n_pose_parameters)
        points3d (np.ndarray): 3D points of shape
            (n_3dpoints, n_point_parameters) projected on each image plane

    Returns:
        Projected images of shape (n_viewpoints * n_image_points, 2)
        If n_viewpoints = 2 and n_3dpoints = 3, the result array is

        ..
            [[x_11, y_11],
             [x_12, y_12],
             [x_21, y_21],
             [x_22, y_22],
             [x_31, y_31],
             [x_32, y_32]]

        where [x_ij, y_ij] is a predicted projection of point `i` on image`j`
    """

    # Assume that camera parameters are same in all viewpoints
    K = camera_parameters.matrix

    # X.shape = (n_3dpoints, n_viewpoints, 3)
    X = transform3d(poses, points3d)

    # X.shape = (n_3dpoints, n_viewpoints, 3)
    X = np.einsum('lk,ijk->ijl', K, X)

    # projet onto the 2D image planes
    Z = X[:, :, 2]
    X = X[:, :, 0:2] / Z[:, :, np.newaxis]
    return X


def projection_(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                b: np.ndarray) -> np.ndarray:
    """
    Args:
        K: Camera intrinsic matrix
        b: 3D point :math:`b = [x, y, z]` to be projected onto the image plane

    Returns:
        Projected image of :math:`b`
    """

    return pi(np.dot(K, transform3d(R, t, b)))
