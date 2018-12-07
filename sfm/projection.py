import numpy as np


def cross_product_matrix(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


def jacobian_projection(p):
    """
    Jacobian of the projection function defined below w.r.t point \\mathbf{p}

    .. math:
        pi(p) = \\frac{1}{z} \\begin{bmatrix}
            a \\\\ b
        \\end{bmatrix}
    """

    x, y, z = p
    return np.array([
        [1 / z, 0, -1 / pow(z, 2)],
        [0, 1 / z, -1 / pow(z, 2)],
    ])


def jacobian_wrt_exp_coordinates(R, v, b):
    # See https://arxiv.org/abs/1312.0788
    """
    Calculate
    :math:`\\frac{R(\\mathbf{v})\\mathbf{b} + \\mathbf{t}}{\\mathbf{v}}`
    """
    U = cross_product_matrix(b)
    V = cross_product_matrix(v)
    I = np.eye(3)
    S = np.outer(v, v) + np.dot(R.T - I, V)
    return -R.dot(U).dot(S) / np.dot(v, v)


def jacobian_pose_and_3dpoint(camera_parameters, a, b):
    K = camera_parameters.matrix
    v, t = a[:3], a[3:]
    R = rodrigues(v)
    p = np.dot(K, transform3d(R, b, t))
    JP = jacobian_projection(p)
    JV = jacobian_wrt_exp_coordinates(R, v, b)
    JR = JP.dot(JV)
    JT = JP.dot(K)
    JA = np.hstack([JR, JT])
    JB = JT.dot(R)
    return JA, JB



def rodrigues(r):
    # see
    # https://docs.opencv.org/2.4/modules/calib3d/doc/
    # camera_calibration_and_3d_reconstruction.html#rodrigues

    theta = np.linalg.norm(r)
    r = r / theta
    K = cross_product_matrix(r)
    I = np.eye(3, 3)
    return I + np.sin(theta) * K + (1-np.cos(theta)) * np.dot(K, K)


def pi(p):
    return p[0:2] / p[2]


def projection(camera_parameters, points3d, poses):
    P = []
    for a in poses:
        omega, t = a[:3], a[3:]
        R = rodrigues(omega)
        for b in points3d:
            p = projection_(camera_parameters, R, t, b)
            P.append(p)
    P = np.array(P)
    return P.flatten()


def projection_(K, R, t, b):
    """
    Args:
        K
        b: 3D point :math:`b = [x, y, z]` to be projected onto the image plane

    Returns:
        Image of :math:`b`
    """

    return K.projection(np.dot(R.T, b - t))
