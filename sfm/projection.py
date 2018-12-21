import numpy as np
from sfm.jacobian import camera_pose_jacobian, structure_jacobian
from sfm.config import n_pose_parameters, n_point_parameters
from sfm.rigid import cross_product_matrix, rodrigues, transform3d


def cross_product_matrix_(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


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


def jacobian_pi(P):
    # TODO check if z > 0

    # jacobian_pi(P)[i] = [
    #     [1 / z, 0, -x / pow(z, 2)],
    #     [0, 1 / z, -y / pow(z, 2)]
    # ]
    # where x, y, z = P[i]

    z = P[:, 2]
    z_squared = np.power(z, 2)

    J = np.zeros((P.shape[0], 2, 3))
    J[:, 0, 0] = J[:, 1, 1] = 1 / z
    J[:, :, 2] = -P[:, 0:2] / z_squared.reshape(-1, 1)
    return J


def elementwise_outer(A, B):
    assert(A.shape[0] == B.shape[0])
    # Equivalent to np.vstack([np.outer(a, b) for a, b in zip(A, B)])
    return np.einsum('ij,ik->ijk', A, B)


def elementwise_dot(A, B):
    # Equivalent to np.array([np.dot(a, b) for a, b in zip(A, B)])
    return np.einsum('ij,ij->i', A, B)


def jacobian_wrt_exp_coordinates(V, B):
    """
    Calculate
    JR[i, j] = d(R(v) * b[i] + t[j]) / dv
             = d(R(v) * b[i]) / dv  at v = v[j]

    JR[i, j] = -R[j] * XB[i] * W[j] / dot(vs[j], vs[j])

    where

    W[j] = outer(vs[j], vs[j]) + (R[j].T - I) * XV[j]

    XB[i] = cross_product_matrix(B[i])
    XV[j] = cross_product_matrix(V[j])
    """

    # See Eq. (8) in https://arxiv.org/abs/1312.0788

    # XV[i] is the cross product matrix of V[i]
    # XV.shape == (n_viewpoints, 3, 3)
    XV = cross_product_matrix(V)
    # Q.shape == (n_viewpoints, 3, 3)
    Q = elementwise_outer(V, V)

    # R.shape == (n_viewpoints, 3, 3)
    R = rodrigues(V)
    # Equivalent to [R_.T for R_ in R]
    RT = np.swapaxes(R, 1, 2)
    I = np.eye(3).reshape(1, 3, 3)  # align the shape

    # W.shape == (n_viewpoints, 3, 3)
    W = np.einsum('ijk,ikl->ijl', RT - I, XV)

    Y = Q + W

    # XB.shape == (n_3dpoints, 3, 3)
    XB = cross_product_matrix(B)
    # S.shape == (n_3dpoints, n_viewpoints, 3, 3)
    S = np.einsum('ijk,lkm->lijm', R, XB)
    # S.shape == (n_3dpoints, n_viewpoints, 3, 3)
    S = np.einsum('ijkl,jlm->ijkm', S, Y)

    # Equivalent to np.array([np.dot(v, v) for v in V])
    D = (V * V).sum(axis=1)
    # D.shape == (n_viewpoints, 1, 1)
    D = D.reshape(D.shape[0], 1, 1)

    # (n_3dpoints, n_viewpoints, 3, 3)
    return -S / D


# @profile
def jacobian_wrt_exp_coordinates_(R, v, b):
    """
    Calculate
    .. math::
    \\begin{align}
        \\frac{\partial (R(\\mathbf{v})\\mathbf{b} + \\mathbf{t})}{\\partial \\mathbf{v}}
        &= ...
    \\end{align}
    """

    # See Eq. (8) in https://arxiv.org/abs/1312.0788

    B = cross_product_matrix(b)
    V = cross_product_matrix(v)
    I = np.eye(3)
    S = np.outer(v, v) + np.dot(R.T - I, V)
    return -R.dot(B).dot(S) / np.dot(v, v)


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

    """
    JP.shape == (n_3dpoints, n_viewpoints, 2, n_pose_parameters)
    JS.shape == (n_3dpoints, n_viewpoints, 2, n_point_parameters)

    # derivative w.r.t rotation parameters
    JR[i, j] = d(Q(points3d[i], poses[j])) / d(poses[j])
             = JP[i, j] * K * JV[i, j]
             = JPK[i, j] * JV[i, j]

    # derivative w.r.t translation
    JT[i, j] = d(Q(points3d[i], poses[j])) / d(poses[j])
             = JP[i, j] * K
             = JPK[i, j]

    JA[i, j] = hstack((JR[i, j], JT[i, j]))

    # derivative w.r.t translation
    JB[i, j] = d(Q(points3d[i], poses[j])) / d(points3d[i])
             = JP[i, j] * K * R[j]
             = JPK[i, j] * R[j]

    where
        JP[i, j] = d(pi(p)) / dp                at p = P[i, j]
        JV[i, j] = d(R[j] * b[i] + t[j]) / dv   at v = v[j]

        P[i, j] = K * (R[j] * b[i] + t[j])
        b[i] = 3dpoints[i]
        R[j] = R(v[j])
        v[j] = poses[j, :3]
        t[j] = poses[j, 3:]
    """

    K = camera_parameters.matrix

    n_3dpoints = points3d.shape[0]
    n_viewpoints = poses.shape[0]
    def calculate_p():
        # P.shape == (n_3dpoints, n_viewpoints, 3)
        P = transform3d(poses, points3d)
        P = np.einsum('lk,ijk->ijl', K, P)
        return P

    # JP.shape == (n_3dpoints, n_viewpoints, 3)
    P = calculate_p()

    # because jacobian_pi accepts array of shape (-, 3)
    P = P.reshape(-1, 3)
    # JP.shape == (n_3dpoints * n_viewpoints, 2, 3)
    JP = jacobian_pi(P)
    JP = JP.reshape(n_3dpoints, n_viewpoints, 2, 3)  # make the shape back

    # JPK[i, j] = JP[i, j] * K
    # JPK.shape == (n_3dpoints, n_viewpoints, 2, 3)
    JPK = np.einsum('ijkl,lm->ijkm', JP, K)

    V = poses[:, :3]

    # JV.sahpe == (n_3dpoints, n_viewpoints, 3, 3)
    JV = jacobian_wrt_exp_coordinates(V, points3d)

    # JR.shape == (n_3dpoints, n_viewpoints, 2, 3)
    JR = np.einsum('ijkl,ijlm->ijkm', JPK, JV)


    P = np.concatenate((JR, JPK), axis=3)

    # R.shape == (n_viewpoints, 3, 3)
    R = rodrigues(V)
    S = np.einsum('ijkl,jlm->ijkm', JPK, R)

    JA = camera_pose_jacobian(P)
    JB = structure_jacobian(S)

    return JA, JB


def jacobian_pose_and_3dpoint_(K, R, v, t, b):
    # TODO add derivation of the equation
    """
    Calculate jacobians w.r.t pose parameters `a = [v, t]` and
    3D points `b` respectively

    Returns: (JA, JB) where JA = dx / da and JB = dx / db
    """

    p = np.dot(K, transform3d(R, t, b))
    JP = jacobian_pi(p)  # d(pi) / dp at p = K(R * b + t)
    JV = jacobian_wrt_exp_coordinates(R, v, b)  # d(R(v) * b + t) / dv
    JT = JP.dot(K)
    JR = JT.dot(JV)  # JP.dot(K).dot(JV)
    JA = np.hstack([JR, JT])
    JB = JT.dot(R)  # JP.dot(K).dot(R)
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


def pi(p):
    return p[0:2] / p[2]


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
