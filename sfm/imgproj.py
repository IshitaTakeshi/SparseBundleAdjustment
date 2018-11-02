def calcImgProj(a[5], qr0[4], v[3], t[3], M[3]):

    L = sqrt(1.0-v[0]*v[0]-v[1]*v[1]-v[2]*v[2]);

    R[0] = L*qr0[1] + v[0]*qr0[0] + v[1]*qr0[3] - v[2]*qr0[2];
    R[1] = L*qr0[2] + v[1]*qr0[0] + v[2]*qr0[1] - v[0]*qr0[3];
    R[2] = L*qr0[3] + v[2]*qr0[0] + v[0]*qr0[2] - v[1]*qr0[1];

    D    = L*qr0[0] - v[0]*qr0[1] - v[1]*qr0[2] - v[2]*qr0[3];

    P[0] =   D*M[0] + R[1]*M[2] - R[2]*M[1];
    P[1] =   D*M[1] + R[2]*M[0] - R[0]*M[2];
    P[2] =   D*M[2] + R[0]*M[1] - R[1]*M[0];
    S =   R[0]*M[0] + R[1]*M[1] + R[2]*M[2];

    t88 =  R[1]*P[2] - R[2]*P[1] + P[0]*D + S*R[0] + t[0];
    t69 =  R[2]*P[0] - R[0]*P[2] + P[1]*D + S*R[1] + t[1];
    t77 =  R[0]*P[1] - R[1]*P[0] + P[2]*D + S*R[2] + t[2];

    n[0] = (a[0] * t88 + a[4] * t69) / t77 + a[1];
    n[1] = (a[0] * a[3] * t69) / t77 + a[2];
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
