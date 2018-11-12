def orthogonal_jacobian(q):
    a, b, c, d = q
    R = np.array([
        [b*b + a*a - d*d - c*c, -2.0 * (a*d + b*c), 2.0 * (a*c - b*d)],
        [2.0 * (a*d - b*c), c*c + a*a - b*b - d*d, 2.0 * (a*b + c*d)],
        [-2.0 * (a*c + d*b), 2.0 * (d*c - a*b), d*d + a*a - c*c - b*b]
    ])
    return R


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


def calcImgProjJacRTS(a[5], qr0[4], v[3], t[3], m[3]):
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

    jacmRT[:, 0:3] = np.dot(A, W) / h[2] - np.outer(n, W[2])
    jacmRT[:, 3:6] = A / h[2]
    jacmRT[:, 5] -= n

    J = orthogonal_jacobian(p)
    jacmS = np.dot(A, J) / h[2] - np.outer(n, J[2])
    return jacmRT, jacmS
