import numpy as np


def to_tuple_if_scalar(value):
    if isinstance(value, int) or isinstance(value, float):
        return (value, value)
    return value


class CameraParameters(object):
    def __init__(self, focal_length, offset, skew=0):
        self.focal_length = to_tuple_if_scalar(focal_length)
        self.offset = to_tuple_if_scalar(offset)
        self.skew = skew

    @property
    def matrix(self):
        ox, oy = self.offset
        fx, fy = self.focal_length
        s = self.skew

        return np.array([
            [fx, s, ox],
            [0, fy, oy],
            [0, 0, 1]
        ])

    def projection(self, x):
        Z = np.dot(self.matrix, x)
        return Z[0:2] / Z[2]  # TODO make sure Z[2] != 0

    def jacobian(self, point):
        x, y, z = point
        fx, fy = self.focal_length
        return np.array([
            [fx / z, 0, -fx / pow(z, 2)],
            [0, fy / z, -fy / pow(z, 2)],
        ])


