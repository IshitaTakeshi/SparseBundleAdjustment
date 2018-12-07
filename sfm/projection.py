import numpy as np


def cross_product_matrix(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


def drotation_tensor(t):
    # calculate D = d [-cross_product_matrix(omega), cross(omega, t)] / d omega
    # which its shape should be D.shape == (3, 3, 6)
    # x, y, z = omega
    # dR = [
    #   [ 0   z  -y  y * tz - z * ty]
    #   [-z   0   x  z * tx - x * tz]
    #   [ y  -x   0  x * ty - y * tx],
    # ]
    # dR / d omega

    tx, ty, tz = t
    # TODO rewire using cross_product_matrix and np.cross
    return np.array([
        [[0, 0, 0, 0],
         [0, 0, 1, -tz],
         [0, -1, 0, ty]],
        [[0, 0, -1, tz],
         [0, 0, 0, 0],
         [1, 0, 0, -tx]],
        [[0, 1, 0, -ty],
         [-1, 0, 0, tx],
         [0, 0, 0, 0]]
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


def jacobian_rotation(K, R, t, b):
    X = np.hstack((b, 1))
    DR = drotation_tensor(t)
    D = np.empty((2, 3))
    for i in range(3):
        drx = R.T.dot(DR[i]).dot(X)
        p = np.dot(K.matrix, drx)
        JP = jacobian_projection(p)
        D[:, i] = np.dot(JP, p)
    return D


def jacobian_translation(K, R, t, b):
    KRT = np.dot(K.matrix, R.T)
    p = np.dot(KRT, b-t)
    PJ = jacobian_projection(p)  # calc jacobian at point p = K * R^T * (b-t)
    return PJ.dot(KRT)


def rodrigues(r):
    # see
    # https://docs.opencv.org/2.4/modules/calib3d/doc/
    # camera_calibration_and_3d_reconstruction.html#rodrigues

    theta = np.linalg.norm(r)
    r = r / theta
    K = cross_product_matrix(r)
    I = np.eye(3, 3)
    return I + np.sin(theta) * K + (1-np.cos(theta)) * np.dot(K, K)


def jacobian_pose_and_3dpoint(K, a, b):
    omega, t = a[:3], a[3:]
    R = rodrigues(omega)
    JR = jacobian_rotation(K, R, t, b)
    JT = jacobian_translation(K, R, t, b)
    jacobian_pose = np.hstack([JR, JT])
    jacobian_3dpoint = -JT
    return jacobian_pose, jacobian_3dpoint


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
