import numpy as np


def projection_matrix(a):
    fx = a[0]
    fy = a[0] * a[3]
    s = a[4]
    cx = a[1]
    cy = a[2]

    A = np.array([
        [fx, s, cx],
        [0, fy, cy]
    ])
    return A


def camera_projection(a, X):
    """
    X: 3D point :math:`X = [x, y, z]` to be projected
    """

    # TODO this calculation is redundant
    # use projection_matrix to make this simple

    return a.projection(X)


def orthogonal_jacobian(q):
    a, b, c, d = q
    R = np.array([
        [b*b + a*a - d*d - c*c, -2.0 * (a*d + b*c), 2.0 * (a*c - b*d)],
        [2.0 * (a*d - b*c), c*c + a*a - b*b - d*d, 2.0 * (a*b + c*d)],
        [-2.0 * (a*c + d*b), 2.0 * (d*c - a*b), d*d + a*a - c*c - b*b]
    ])
    return R


def left_matrix(q):
    Q = np.array([
        [+q[0], -q[1], -q[2], -q[3]],
        [+q[1], +q[0], -q[3], +q[2]],
        [+q[2], +q[3], +q[0], -q[1]],
        [+q[3], -q[2], +q[1], +q[0]]
    ])
    return Q


def right_matrix(q):
    Q = np.array([
        [+q[0], -q[1], -q[2], -q[3]],
        [+q[1], +q[0], +q[3], -q[2]],
        [+q[2], -q[3], +q[0], +q[1]],
        [+q[3], +q[2], -q[1], +q[0]]
    ])
    return Q


def projection(camera_parameters, initial_quaternions, points3d, poses):
    P = []
    for q, pose in zip(initial_quaternions, poses):
        for point3d in points3d:
            p = projection_(camera_parameters, q, point3d, pose)
            P.append(p)
    P = np.array(P)
    return P.flatten()


# TODO remove this interface by modifying projection_one_point_
def projection_(camera_parameters, initial_quaternion, point3d, pose):
    return projection_one_point_(
        camera_parameters,
        initial_quaternion,
        pose[:3],
        pose[3:],
        point3d
    )


def projection_one_point_(a, qr0, v, t, M):  # q -> qr0, m -> M
    """
    v : vector part of a unit quaternion that represents a camera rotation
    t : 3D vector which stores a camera position
    M : Point coordinate in the 3D space
    """

    w = np.sqrt(1.0 - np.dot(v, v))
    v = np.hstack(([w], v))

    r = np.array([
        qr0[0]*v[0] - qr0[1]*v[1] - qr0[2]*v[2] - qr0[3]*v[3],
        qr0[1]*v[0] + qr0[0]*v[1] + qr0[3]*v[2] - qr0[2]*v[3],
        qr0[2]*v[0] - qr0[3]*v[1] + qr0[0]*v[2] + qr0[1]*v[3],
        qr0[3]*v[0] + qr0[2]*v[1] - qr0[1]*v[2] + qr0[0]*v[3]
    ])

    p = np.array([
        - r[1]*M[0] - r[2]*M[1] - r[3]*M[2],
        + r[0]*M[0] - r[3]*M[1] + r[2]*M[2],
        + r[3]*M[0] + r[0]*M[1] - r[1]*M[2],
        - r[2]*M[0] + r[1]*M[1] + r[0]*M[2]
    ])

    u = np.array([
        + p[1]*r[0] - p[0]*r[1] - p[3]*r[2] - p[2]*r[3] + t[0],
        + p[2]*r[0] - p[3]*r[1] - p[0]*r[2] + p[1]*r[3] + t[1],
        + p[3]*r[0] + p[2]*r[1] - p[1]*r[2] - p[0]*r[3] + t[2]
    ])

    return camera_projection(a, u)


def image_projection_full_rotation(a, q, t, m):
    qr = np.array([
        [-q[1], -q[2], -q[3]],
        [+q[0], -q[3], +q[2]],
        [+q[3], +q[0], -q[1]],
        [-q[2], +q[1], +q[0]]
    ])

    ql = np.array([
        [-q[1], +q[0], -q[3], +q[2]],
        [-q[2], +q[3], +q[0], -q[1]],
        [-q[3], -q[2], +q[1], +q[0]]
    ])

    u = np.array(ql, np.dot(qr, m)) + t
    return camera_projection(a, u)


def pose_and_structure_jacobian(camera_parameters, initial_quaternion,
                                point3d, pose):
    return pose_and_structure_jacobian_(
        camera_parameters,
        initial_quaternion,
        pose[:3],
        pose[3:],
        point3d
    )


def pose_and_structure_jacobian_(a, qr0, v, t, m):
    """
    Args:
        v (np.ndarray) : Vector part of a unit quaternion that
            represents a camera rotation
        t (np.ndarray) : 3D vector which stores a camera position
        M (np.ndarray) : Point coordinate in the 3D space
    Returns:
        JRT (np.ndarray) : Jacobian w.r.t a camera pose
        JS (np.ndarray) : Jacobian w.r.t a 3D point
    """

    w = np.sqrt(1.0-np.dot(v, v))
    v = np.array([w, v[0], v[1], v[2]])

    Q = np.array([
        [+qr0[0], -qr0[1], -qr0[2], -qr0[3]],
        [-qr0[1], -qr0[0], -qr0[3], +qr0[2]],
        [+qr0[2], -qr0[3], +qr0[0], +qr0[1]],
        [+qr0[3], +qr0[2], -qr0[1], +qr0[0]]
    ])

    p = np.dot(Q, v)

    P1 = np.array([
        [+p[1], -p[2], -p[3]],
        [+p[0], -p[3], +p[2]],
        [+p[3], +p[0], +p[1]],
        [-p[2], -p[1], +p[0]]
    ])

    u = np.dot(P1, m)

    P2 = np.array([
        [+p[1], +p[0], -p[3], +p[2]],
        [-p[2], +p[3], +p[0], +p[1]],
        [-p[3], -p[2], -p[1], +p[0]]
    ])

    h = np.dot(P2, u) + t

    R = np.array([
        [-qr0[1], -qr0[2], -qr0[3]],
        [+qr0[0], +qr0[3], -qr0[2]],
        [-qr0[3], +qr0[0], +qr0[1]],
        [+qr0[2], -qr0[1], +qr0[0]]
    ])

    C = - (1/w) * np.outer(qr0, v[1:]) + R

    M = np.array([
        [0, -m[0], -m[1], -m[2]],
        [+m[0], 0, +m[2], -m[1]],
        [+m[1], -m[2], 0, +m[0]],
        [+m[2], +m[1], -m[0], 0]
    ])

    U = np.array([
        [+u[1], -u[0], +u[3], -u[2]],
        [+u[2], -u[3], -u[0], +u[1]],
        [+u[3], +u[2], -u[1], -u[0]]
    ])

    W = np.dot(U, C) + np.dot(P2, np.dot(M, C))

    A = a.matrix[:2]

    n = np.dot(A, h) / (h[2] * h[2])

    JRT = np.empty((2, 6))
    JRT[:, 0:3] = np.dot(A, W) / h[2] - np.outer(n, W[2])
    JRT[:, 3:6] = A / h[2]
    JRT[:, 5] -= n  # subtract from the last column

    J = orthogonal_jacobian(p)
    JS = np.dot(A, J) / h[2] - np.outer(n, J[2])

    return JRT, JS
