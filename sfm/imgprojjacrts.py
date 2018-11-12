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


def multiply(u, p, m, C):
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

    P = np.array([
        [+p[1], +p[0], -p[3], +p[2]],
        [-p[2], +p[3], +p[0], +p[1]],
        [-p[3], -p[2], -p[1], +p[0]]
    ])

    return np.dot(U, C) + np.dot(P, np.dot(M, C))


def calcImgProjJacRTS(a[5], qr0[4], v[3], t[3], M[3]):

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
    u = np.dot(P1, M)

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

    V = - (1/w) * np.outer(qr0, v) + R

    U = multiply(u, p, M, V)

    A = projection_matrix(a)

    n = np.dot(A, h) / (h[2] * h[2])

    jacmRT[:, 0:3] = np.dot(A, U) / h[2] - np.outer(n, U[2])
    jacmRT[:, 3:6] = A / h[2]
    jacmRT[:, 5] -= n

    J = orthogonal_jacobian(p)
    jacmS = np.dot(A, J) / h[2] - np.outer(n, J[2])
    return jacmRT, jacmS



def fu(M, p):
    u[0] = + M[0]*p[1] - M[1]*p[2] - M[2]*p[3]
    u[1] = + M[0]*p[0] - M[1]*p[3] + M[2]*p[2]
    u[2] = + M[0]*p[3] + M[1]*p[0] + M[2]*p[1]
    u[3] = - M[0]*p[2] - M[1]*p[1] + M[2]*p[0]
    return u

def fp(L, v, qr0):
    p[0] = + L*qr0[0] - v[0]*qr0[1] - v[1]*qr0[2] - v[2]*qr0[3]
    p[1] = - L*qr0[1] - v[0]*qr0[0] - v[1]*qr0[3] + v[2]*qr0[2]
    p[2] = + L*qr0[2] - v[0]*qr0[3] + v[1]*qr0[0] + v[2]*qr0[1]
    p[3] = + L*qr0[3] + v[0]*qr0[2] - v[1]*qr0[1] + v[2]*qr0[0]
    return p


def fs(M, r):
    s[0] = - M[0]*r[1] - M[1]*r[2] - M[2]*r[3]
    s[1] = + M[0]*r[0] - M[1]*r[3] + M[2]*r[2]
    s[2] = + M[0]*r[3] + M[1]*r[0] - M[2]*r[1]
    s[3] = - M[0]*r[2] + M[1]*r[1] + M[2]*r[0]
    return s


def fw(p, u, s, r):
    ps[0] = + p[0]*s[2] + p[1]*s[3] - p[2]*s[0] + p[3]*s[1]
    ps[1] = + p[0]*s[3] - p[1]*s[2] - p[2]*s[1] - p[3]*s[0]
    ps[2] = + p[0]*s[1] + p[1]*s[0] + p[2]*s[3] - p[3]*s[2]
    ur[0] = - u[0]*r[2] + u[1]*r[3] + u[2]*r[0] - u[3]*r[1]
    ur[1] = - u[0]*r[3] - u[1]*r[2] + u[2]*r[1] + u[3]*r[0]
    ur[2] = - u[0]*r[1] + u[1]*r[0] - u[2]*r[3] + u[3]*r[2]
    return ps + ur

def fn(a, h):
    n[0] = (a[0]*h[0] + a[4]*h[1] + a[1]*h[2]) / (h[2]*h[2])
    n[1] = (a[0]*a[3]*h[1] + a[2]*h[2]) / (h[2]*h[2])
    return n

def calc_image_projection_jacobian_rts(a[5], qr0[4], v[3], t[3], M[3]):
    L = np.sqrt(1-np.dot(v, v))
    q = qr0 / L

    u = fu(M, p)
    p = fp(L, v, qr0)

    R = np.array([
        [-qr0[1], -qr0[2], -qr0[3]],
        [+qr0[0], +qr0[3], -qr0[2]],
        [-qr0[3], +qr0[0], +qr0[1]],
        [+qr0[2], -qr0[1], +qr0[0]]
    ])

    R = - np.outer(q, v) + R

    for i in range(3):
        s = fs(M, r)
        w = fw(p, u, s, R[:, i])


    return jacmRT, jacmS

