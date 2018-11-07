def camera_projection(a, X):
    fx = a[0]
    fy = a[0] * a[3]
    s = a[4]
    c = a[1:3]  # offset

    A = np.array([
        [fx, s, 0],
        [0, fy, 0],
        [0,  0, 1]
    ])

    Z = np.dot(A, X)
    n = Z[0:2] / Z[2] + c
    return n


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


def calcImgProj(a, q, v, t, m):  # q -> qr0, m -> M
    L = sqrt(1.0 - np.dot(v, v));

    Q = np.array([
        [-q[1], -q[2], -q[3]],
        [+q[0], +q[3], -q[2]],
        [-q[3], +q[0], +q[1]],
        [+q[2], -q[1], +q[0]]
    ])

    r = L * q + np.dot(Q, v)

    R = np.array([
        [-r[1], -r[2], -r[3]],
        [+r[0], -r[3], +r[2]],
        [+r[3], +r[0], -r[1]],
        [-r[2], +r[1], +r[0]]
    ])

    p = np.dot(R, m)

    P = np.array([
        [-p[1], +p[0], -p[3], +p[2]],
        [-p[2], +p[3], +p[0], -p[1]],
        [-p[3], -p[2], +p[1], +p[0]]
    ])

    u = np.dot(-P, r) + t

    return camera_projection(a, u)


def calc_image_projection(a, qr0, v, t, m):
    A = qr0[0]
    q = qr0[1:]

    L = np.sqrt(1.0-np.dot(v, v))
    r = L * A + v * A + np.cross(v, q)
    D = L * A - np.dot(q, v)
    p = D * m + np.cross(r, m)
    u = np.cross(r, p) + D * p + np.dot(r, m) * r + t

    return camera_projection(a, u)


def calcImgProjFullR(a, q, t, m):
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
