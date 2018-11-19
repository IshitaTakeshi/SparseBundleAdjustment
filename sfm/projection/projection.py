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
    n = Z[0:2] / Z[2] + c  # TODO make sure Z[2] != 0
    return n
