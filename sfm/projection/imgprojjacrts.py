# here we call 3D point coordinates structure parameters

def orthogonal_jacobian(q):
    a, b, c, d = q
    R = np.array([
        [b*b + a*a - d*d - c*c, -2.0 * (a*d + b*c), 2.0 * (a*c - b*d)],
        [2.0 * (a*d - b*c), c*c + a*a - b*b - d*d, 2.0 * (a*b + c*d)],
        [-2.0 * (a*c + d*b), 2.0 * (d*c - a*b), d*d + a*a - c*c - b*b]
    ])
    return R


def calc_image_jacobian(camera_parameters, initial_quaternion, pose, point3d):
    return calc_image_jacobian_rts(
        camera_parameters,
        initial_quaternion,
        pose[:3],
        pose[3:],
        point3d
    )


def calc_pose_and_structure_jacobian(a[5], qr0[4], v[3], t[3], m[3]):
    """
    Returns:
        JRT (np.ndarray) : Jacobian w.r.t a camera pose
        JS (np.ndarray) : Jacobian w.r.t a 3D point
    """

    w = sqrt(1.0-np.dot(v, v))
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

    C = - (1/w) * np.outer(qr0, v) + R

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

    A = projection_matrix(a)

    n = np.dot(A, h) / (h[2] * h[2])

    JRT[:, 0:3] = np.dot(A, W) / h[2] - np.outer(n, W[2])
    JRT[:, 3:6] = A / h[2]
    JRT[:, 5] -= n  # subtract from the last column

    J = orthogonal_jacobian(p)
    JS = np.dot(A, J) / h[2] - np.outer(n, J[2])

    return JRT, JS
