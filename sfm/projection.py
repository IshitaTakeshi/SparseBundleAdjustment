import numpy as np
from sfm.jacobian import camera_pose_jacobian, structure_jacobian
from sfm.config import n_pose_parameters, n_point_parameters

def cross_product_matrix(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


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


def jacobian_pose_and_3dpoint(K, R, v, t, b):
    p = np.dot(K, transform3d(R, t, b))
    JP = jacobian_pi(p)
    JV = jacobian_wrt_exp_coordinates(R, v, b)
    JR = JP.dot(JV)
    JT = JP.dot(K)
    JA = np.hstack([JR, JT])
    JB = JT.dot(R)
    return JA, JB


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
    P = np.empty((n_3dpoints, n_viewpoints, 2, n_pose_parameters))
    S = np.empty((n_3dpoints, n_viewpoints, 2, n_point_parameters))
    K = camera_parameters.matrix

    for j, a in enumerate(poses):
        v, t = a[:3], a[3:]
        R = rodrigues(v)
        for i, b in enumerate(points3d):
            P[i, j], S[i, j] = jacobian_pose_and_3dpoint(K, R, v, t, b)

    JA = camera_pose_jacobian(P)
    JB = structure_jacobian(S)
    return JA, JB


def rodrigues(r):
    # see
    # https://docs.opencv.org/2.4/modules/calib3d/doc/
    # camera_calibration_and_3d_reconstruction.html#rodrigues

    theta = np.linalg.norm(r)
    r = r / theta
    K = cross_product_matrix(r)
    I = np.eye(3, 3)
    # I + sin(theta) * K + (1-cos(theta)) * dot(K, K) is equivalent to
    # cos(theta) * I + (1-cos(theta)) * outer(r, r) + sin(theta) * K
    return I + np.sin(theta) * K + (1-np.cos(theta)) * np.dot(K, K)


def pi(p):
    return p[0:2] / p[2]


def projection(camera_parameters, points3d, poses):
    K = camera_parameters.matrix

    n_viewpoints = poses.shape[0]
    n_3dpoints = points3d.shape[0]

    X = np.empty((n_3dpoints, n_viewpoints, 2))
    for j, a in enumerate(poses):
        v, t = a[:3], a[3:]
        R = rodrigues(v)
        for i, b in enumerate(points3d):
            X[i, j] = projection_(K, R, t, b)
    X = np.array(X)
    return X


def transform3d(R, t, b):
    return R.dot(b) + t


def projection_(K, R, t, b):
    """
    Args:
        K
        b: 3D point :math:`b = [x, y, z]` to be projected onto the image plane

    Returns:
        Image of :math:`b`
    """

    return pi(np.dot(K, transform3d(R, t, b)))
