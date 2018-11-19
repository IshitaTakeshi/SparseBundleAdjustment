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


def projection_(a, q, v, t, m):  # q -> qr0, m -> M
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


def calc_image_projection_full_rotation(a, q, t, m):
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
