import numpy as np


class ErrorNotReducedException(Exception):
    pass


class LevenbergMarquardt(object):
    def __init__(self, updater, residual, initializer,
                 initial_lambda=1e-3, nu=1.01, tolerance=1e-4):

        """
        Args:
            n_input_dims (int): Number of dimensions of :math:`\\mathbf{p}`
        """

        self.updater = updater
        self.residual = residual
        self.initializer = initializer

        self.initial_lambda = initial_lambda

        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        self.tolerance = tolerance

        if nu <= 1.0:
            raise ValueError("nu must be >= 1")

        self.nu = nu
        print("Initialized")

    def one_step_forward(self, updater, r0, lambda_):
        p1 = updater.calculate(lambda_)
        r1 = self.residual.calculate(p1)
        return r1 < r0, r1, p1, lambda_

    def update(self, r0, p0, lambda0, n_attempts=100):
        # See the link below:
        # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm#
        # Choice_of_damping_parameter

        self.updater.evaluate_at(p0)

        nu = self.nu

        # one step forward with damping factor = lambda0 / nu
        reduced, r1, p1, lambda1 = self.one_step_forward(self.updater, r0, lambda0 / nu)
        print("r0 - r1 = {}".format(r0 - r1))
        print("lambda = {}".format(lambda1))
        if reduced:
            return r1, p1, lambda1

        # one step forward with damping factor = lambda0
        reduced, r1, p1, lambda1 = self.one_step_forward(self.updater, r0, lambda0)
        print("r0 - r1 = {}".format(r0 - r1))
        print("lambda = {}".format(lambda1))
        if reduced:
            return r1, p1, lambda1

        # update lambda0 until error reduces if above two attepmts failed
        for k in range(1, n_attempts+1):
            reduced, r1, p1, lambda0 = self.one_step_forward(self.updater, r0, lambda0 * nu)
            print("r0 - r1 = {}".format(r0 - r1))
            print("lambda = {}".format(lambda0))
            if reduced:
                return r1, p1, lambda0

        # TODO there should be a better idea than raising an exception
        raise ErrorNotReducedException

    def print_status(self, r0, r1, lambda_):
        print(" {:8.5e}  {:8.5e}  {:8.5e}".format(r0, r1-r0, lambda_))

    def optimize(self, max_iter=200):
        print("Error     Error reduction   lambda")

        p = self.initializer.initial_value()
        lambda_ = self.initial_lambda

        r0 = self.residual.calculate(p)

        for i in range(max_iter):
            print("i = {}".format(i))
            try:
                r1, p, lambda_ = self.update(r0, p, lambda_)
            except ErrorNotReducedException:
                # p is the local minima
                return p

            if r1 < self.tolerance:
                return p

            self.print_status(r0, r1, lambda_)
            r0 = r1
        return p
