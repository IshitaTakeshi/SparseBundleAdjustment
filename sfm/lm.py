import numpy as np
from scipy.sparse import linalg, identity, diags


class ErrorNotReducedException(Exception):
    pass


class Residual(object):
    def __init__(self, function, target, weights):
        self.f = function
        self.x = target
        self.W = weights

    def calculate(self, p):
        d = self.x - self.f(p)
        if self.W is None:
            return np.dot(d, d)
        return np.dot(d, np.dot(self.W, d))


class Updater(object):
    def __init__(self, function, jacobian, target, p):
        self.f = function
        self.p = p

        J = jacobian(p)

        mask = np.logical_not(np.isnan(target))
        indices = np.arange(J.shape[0])[mask]

        x = target[indices]
        f = self.f(p)[indices]
        J = J[indices, :]  # d x[mask] / dp

        self.A = J.T.dot(J)
        self.D = diags(self.A.diagonal())
        self.g = J.T.dot(x - f)

    def calculate(self, lambda_):
        dp = linalg.spsolve(self.A + lambda_ * self.D, self.g)
        return self.p + dp


class LevenbergMarquardt(object):
    def __init__(self, function, jacobian, target, weights=None, n_input_dims=None,
                 p0=None, lambda0=1e-3, nu=1.01):
        """
        Args:
            n_input_dims (int): Number of dimensions of :math:`\\mathbf{p}`
        """

        self.f = function
        self.J = jacobian
        self.x = target

        self.residual = Residual(function, target, weights)
        self.lambda0 = lambda0

        if p0 is None:
            self.p0 = np.random.normal(size=n_input_dims)
        else:
            self.p0 = p0

        if nu <= 1.0:
            raise ValueError("nu must be >= 1")

        self.nu = nu
        print("Initialized")

    def evaluate(self, updater, r0, lambda_):
        p1 = updater.calculate(lambda_)
        r1 = self.residual.calculate(p1)
        return r1 < r0, r1, p1, lambda_

    def update(self, r0, p0, lambda0, n_attempts=10):
        # See the link below:
        # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm#
        # Choice_of_damping_parameter

        updater = Updater(self.f, self.J, self.x, p0)
        nu = self.nu

        reduced, r1, p1, lambda1 = self.evaluate(updater, r0, lambda0 / nu)
        if reduced:
            return r1, p1, lambda1
        print("Cost not reduced at lambda0 / nu")

        reduced, r1, p1, lambda1 = self.evaluate(updater, r0, lambda0)
        if reduced:
            return r1, p1, lambda1

        print("Cost not reduced at lambda0")

        # update lambda0 until error reduces
        for k in range(1, n_attempts+1):
            reduced, r1, p1, lambda0 = self.evaluate(updater, r0, lambda0 * nu)
            if reduced:
                return r1, p1, lambda0

        raise ErrorNotReducedException

    def print_status(self, r0, r1, lambda_):
        print(" {:8.5e}  {:8.5e}  {:8.5e}".format(r0, r1-r0, lambda_))

    def optimize(self, max_iter=200):
        print("Error     Error reduction   lambda")

        p, lambda_ = self.p0, self.lambda0

        r0 = self.residual.calculate(p)

        for i in range(max_iter):
            try:
                r1, p, lambda_ = self.update(r0, p, lambda_)
            except ErrorNotReducedException:
                return p

            self.print_status(r0, r1, lambda_)
            r0 = r1
        return p
