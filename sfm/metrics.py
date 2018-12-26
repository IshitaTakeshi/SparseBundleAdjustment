import numpy as np

# TODO Add an interface and summarize


class SquaredNormResidual(object):
    def __init__(self, function, target):
        self.f = function
        self.x = target

    def calculate(self, p):
        d = self.x - self.f(p)
        return np.dot(d, d)


class MahalanobisResidual(object):
    def __init__(self, function, target, weights):
        self.f = function
        self.x = target
        self.W = weights

    def calculate(self, p):
        d = self.x - self.f(p)
        return np.dot(d, np.dot(self.W, d))
