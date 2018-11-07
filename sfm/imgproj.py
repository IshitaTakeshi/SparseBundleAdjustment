def calcImgProj(a[5], q[4], v[3], t[3], m[3]):  # q -> qr0, m -> M
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

    fx, fy = a[0], a[0] * a[3]
    cx, cy = a[1], a[2]
    s = a[4]

    n[0] = (fx * u[0] + s * u[1]) / u[2] + cx
    n[1] = (fy * u[1]) / u[2] + cy

    return n

def calc_image_projection(a, qr0, v, t, m):
    A = qr0[0]
    q = qr0[1:]

    L = np.sqrt(1.0-np.dot(v, v))
    r = L * A + v * A + np.cross(v, q)
    D = L * A - np.dot(v, q)
    p = D * m + np.cross(r, m)
    u = np.cross(r, p) + D * p + np.dot(r, m) * r + t

    n[0] = (a[0] * u[0] + a[4] * u[1]) / u[2] + a[1]
    n[1] = (a[0] * a[3] * u[1]) / u[2]        + a[2]



def calcImgProjFullR(a[5], qr0[4], t[3], M[3], n[2]):
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

    X = np.array(ql, np.dot(qr, M)) + t

    n[0] = (a[0]*X[0] + a[4]*X[1] + a[1]*X[2]) / X[2]
    n[1] = (a[0]*a[3]*X[1] + a[2]*X[2]) / X[2]
    return n
