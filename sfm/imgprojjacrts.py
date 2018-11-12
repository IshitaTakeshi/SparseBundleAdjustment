def multiply(u, c, p, d):
    j = np.empty(3)
    j[0] = - u[0]*c[2] + u[1]*c[3] + u[2]*c[0] - u[3]*c[1] + p[0]*d[2] + p[1]*d[3] - p[2]*d[0] + p[3]*d[1]
    j[1] = - u[0]*c[3] - u[1]*c[2] + u[2]*c[1] + u[3]*c[0] + p[0]*d[3] - p[1]*d[2] - p[2]*d[1] - p[3]*d[0]
    j[2] = - u[0]*c[1] + u[1]*c[0] - u[2]*c[3] + u[3]*c[2] + p[0]*d[1] + p[1]*d[0] + p[2]*d[3] - p[3]*d[2]
    return j


def calcImgProjJacRTS(a[5], qr0[4], v[3], t[3], M[3],
                      jacmRT[2][6], jacmS[2][3]):

    L = sqrt(1.0-np.dot(v, v))

    q = qr0 / L

    p[0] = + L*qr0[0] - v[0]*qr0[1] - v[1]*qr0[2] - v[2]*qr0[3]
    p[1] = - L*qr0[1] - v[0]*qr0[0] - v[1]*qr0[3] + v[2]*qr0[2]
    p[2] = + L*qr0[2] - v[0]*qr0[3] + v[1]*qr0[0] + v[2]*qr0[1]
    p[3] = + L*qr0[3] + v[0]*qr0[2] - v[1]*qr0[1] + v[2]*qr0[0]

    u[0] = + M[0]*p[1] - M[1]*p[2] - M[2]*p[3]
    u[1] = + M[0]*p[0] - M[1]*p[3] + M[2]*p[2]
    u[2] = + M[0]*p[3] + M[1]*p[0] + M[2]*p[1]
    u[3] = - M[0]*p[2] - M[1]*p[1] + M[2]*p[0]

    h[0] = + p[1]*u[0] + p[0]*u[1] - p[3]*u[2] + p[2]*u[3] + t[0]
    h[1] = - p[2]*u[0] + p[0]*u[2] + p[1]*u[3] + p[3]*u[1] + t[1]
    h[2] = - p[3]*u[0] + p[0]*u[3] - p[2]*u[1] - p[1]*u[2] + t[2]

    r[0] = - q[0]*v[0] - qr0[1]
    r[1] = - q[1]*v[0] + qr0[0]
    r[2] = - q[2]*v[0] - qr0[3]
    r[3] = - q[3]*v[0] + qr0[2]

    s[0] = - M[0]*r[1] - M[1]*r[2] - M[2]*r[3]
    s[1] = + M[0]*r[0] - M[1]*r[3] + M[2]*r[2]
    s[2] = + M[0]*r[3] + M[1]*r[0] - M[2]*r[1]
    s[3] = - M[0]*r[2] + M[1]*r[1] + M[2]*r[0]

    w = multiply(u, r, p, s)

    g[0] = - q[0]*v[1] - qr0[2]
    g[1] = - q[1]*v[1] + qr0[3]
    g[2] = - q[2]*v[1] + qr0[0]
    g[3] = - q[3]*v[1] - qr0[1]

    b[0] = - M[0]*g[1] - M[1]*g[2] - M[2]*g[3]
    b[1] = + M[0]*g[0] - M[1]*g[3] + M[2]*g[2]
    b[2] = + M[0]*g[3] + M[1]*g[0] - M[2]*g[1]
    b[3] = - M[0]*g[2] + M[1]*g[1] + M[2]*g[0]

    k = multiply(u, g, p, b)

    c[0] = - q[0]*v[2] - qr0[3]
    c[1] = - q[1]*v[2] - qr0[2]
    c[2] = - q[2]*v[2] + qr0[1]
    c[3] = - q[3]*v[2] + qr0[0]

    d[0] = - M[0]*c[1] - M[1]*c[2] - M[2]*c[3]
    d[1] = + M[0]*c[0] - M[1]*c[3] + M[2]*c[2]
    d[2] = + M[0]*c[3] + M[1]*c[0] - M[2]*c[1]
    d[3] = - M[0]*c[2] + M[1]*c[1] + M[2]*c[0]

    j = multiply(u, c, p, d)

    n[0] = (a[0]*h[0] + a[4]*h[1] + a[1]*h[2]) / (h[2]*h[2])
    n[1] = (a[0]*a[3]*h[1] + a[2]*h[2]) / (h[2]*h[2])

    jacmRT[0][0] = (a[4]*w[0] + a[1]*w[1] + a[0]*w[2]) / h[2] - n[0]*w[1]
    jacmRT[0][1] = (a[4]*k[0] + a[1]*k[1] + a[0]*k[2]) / h[2] - n[0]*k[1]
    jacmRT[0][2] = (a[4]*j[0] + a[1]*j[1] + a[0]*j[2]) / h[2] - n[0]*j[1]
    jacmRT[0][3] = a[0] / h[2]
    jacmRT[0][4] = a[4] / h[2]
    jacmRT[0][5] = a[1] / h[2] - n[0]

    jacmRT[1][0] = (a[0]*a[3]*w[0] + a[2]*w[1]) / h[2] - n[1]*w[1]
    jacmRT[1][1] = (a[0]*a[3]*k[0] + a[2]*k[1]) / h[2] - n[1]*k[1]
    jacmRT[1][2] = (a[0]*a[3]*j[0] + a[2]*j[1]) / h[2] - n[1]*j[1]
    jacmRT[1][3] = 0.0
    jacmRT[1][4] = a[0] * a[3] / h[2]
    jacmRT[1][5] = a[2] / h[2] - n[1]

    t303 =  2.0 * (p[0]*p[3] - p[1]*p[2])
    t309 = -2.0 * (p[0]*p[2] + p[3]*p[1])
    t331 =  2.0 * (p[3]*p[2] - p[0]*p[1])
    t342 = -2.0 * (p[0]*p[3] + p[1]*p[2])
    t347 =  2.0 * (p[0]*p[1] + p[2]*p[3])
    t353 =  2.0 * (p[0]*p[2] - p[1]*p[3])

    t350 = p[3]*p[3] + p[0]*p[0] - p[2]*p[2] - p[1]*p[1]
    t325 = p[2]*p[2] + p[0]*p[0] - p[1]*p[1] - p[3]*p[3]
    t301 = p[1]*p[1] + p[0]*p[0] - p[3]*p[3] - p[2]*p[2]

    jacmS[0][0] = (a[0]*t301 + a[4]*t303 + a[1]*t309) / h[2] - n[0]*t309
    jacmS[0][1] = (a[0]*t342 + a[4]*t325 + a[1]*t331) / h[2] - n[0]*t331
    jacmS[0][2] = (a[0]*t353 + a[4]*t347 + a[1]*t350) / h[2] - n[0]*t350

    jacmS[1][0] = (a[0]*a[3]*t303 + a[2]*t309) / h[2] - n[1]*t309
    jacmS[1][1] = (a[0]*a[3]*t325 + a[2]*t331) / h[2] - n[1]*t331
    jacmS[1][2] = (a[0]*a[3]*t347 + a[2]*t350) / h[2] - n[1]*t350



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

