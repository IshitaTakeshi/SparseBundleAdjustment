import numpy as np


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




def projection(camera_parameters, initial_rotation, pose, points3d):
    return projection_(
        camera_parameters,
        initial_quaternion,
        pose[:3],
        pose[3:],
        point3d
    )


def projection_(a, q, v, t, m):
    w = sqrt(1.0-np.dot(v, v))
    v = np.concatenate(([w], v))
    m = np.concatenate(([0], m))

    Q = np.array([
        [q[0], - q[1], - q[2], - q[3]],
        [q[1], + q[0], + q[3], - q[2]],
        [q[2], - q[3], + q[0], + q[1]],
        [q[3], + q[2], - q[1], + q[0]]
    ])

    r = np.dot(Q, v)

    M = np.array([
        [+m[0], -m[1], -m[2], -m[3]],
        [+m[1], +m[0], +m[3], -m[2]],
        [+m[2], -m[3], +m[0], +m[1]],
        [+m[3], +m[2], -m[1], +m[0]]
    ])

    p = np.dot(M, r)

    p[0] =  - r[1]*m[0] - r[2]*m[1] - r[3]*m[2]
    p[1] =  + r[0]*m[0] - r[3]*m[1] + r[2]*m[2]
    p[2] =  + r[3]*m[0] + r[0]*m[1] - r[1]*m[2]
    p[3] =  - r[2]*m[0] + r[1]*m[1] + r[0]*m[2]

    u[0] = + p[1]*r[0] - p[0]*r[1] + p[3]*r[2] - p[2]*r[3] + t[0]
    u[1] = + p[2]*r[0] - p[3]*r[1] - p[0]*r[2] + p[1]*r[3] + t[1]
    u[2] = + p[3]*r[0] + p[2]*r[1] - p[1]*r[2] - p[0]*r[3] + t[2]

    u[0] = + p[1]*r[0] - p[0]*r[1] + p[3]*r[2] - p[2]*r[3] + t[0]
    u[1] = + p[2]*r[0] - p[3]*r[1] - p[0]*r[2] + p[1]*r[3] + t[1]
    u[2] = + p[3]*r[0] + p[2]*r[1] - p[1]*r[2] - p[0]*r[3] + t[2]

    u[0] = - p[0]*r[1] + p[1]*r[0] - p[2]*r[3] + p[3]*r[2] + t[0]
    u[1] = - p[0]*r[2] + p[1]*r[3] + p[2]*r[0] - p[3]*r[1] + t[1]
    u[2] = - p[0]*r[3] - p[1]*r[2] + p[2]*r[1] + p[3]*r[0] + t[2]

    return camera_projection(a, u)


def calcImgProj(a, q, v, t, m):  # q -> qr0, m -> M
    """
    v : vector part of a unit quaternion that represents a camera rotation
    t : 3D vector which stores a camera position
    """

    w = sqrt(1.0 - np.dot(v, v));  # extract the real part of quaternion

    v = np.array([w, v[0], v[1], v[2]])  # full quaternion

    Q = np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1], +q[0], +q[3], -q[2]],
        [q[2], -q[3], +q[0], +q[1]],
        [q[3], +q[2], -q[1], +q[0]]
    ])

    r = np.dot(Q, v)

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

    u = -np.dot(P, r) + t

    return camera_projection(a, u)


def calc_image_projection_(a, q, t, m):
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
