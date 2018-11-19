import numpy as np


class CameraParameters(object):
    def __init__(self, fx, fy, s, ox, oy):
        self.fx = fx
        self.fx = fx
        self.fy = fy
        self.s = s
        self.ox = ox
        self.oy = oy

    @property
    def matrix(self):
        return np.array([
            [self.fx, self.s, self.ox],
            [0, self.fy, self.oy],
            [0, 0, 1]
        ])

    def projection(self, x):
        Z = np.dot(self.matrix, x)
        return Z[0:2] / Z[2]  # TODO make sure Z[2] != 0
