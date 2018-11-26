import numpy as np


def squared_norm(v):
    return np.sum(np.power(v, 2))


def squared_mahalanobis_distance(x, y, C):
    return np.dot(np.dot(x.T, C), y)


def weighted_squared_norm(x, W):
    return squared_mahalanobis_distance(x, x, W)


class LevenbergMarquardt(object):
    """
    Find :math:`\\mathbf{p}` which satisfies
    :math:`\\mathbf{f}(\\mathbf{p}) = \\mathbf{x}`
    """
    """Fig. 2"""

    def __init__(self, function, jacobian, target, n_input_dims, tau,
                 initial_p=None, weights=None,
                 threshold_singular=0.01, threshold_relative_change=1e-4,
                 threshold_sba=0.01):

        self.f = function
        self.J = jacobian
        self.x = target
        self.n_input_dims = n_input_dims
        self.I = np.eye(n_input_dims)
        self.tau = tau
        self.threshold_singular = threshold_singular
        self.threshold_relative_change = threshold_relative_change
        self.threshold_sba = threshold_sba

        if initial_p is None:
            self.initial_p = self.initialize_p()
        else:
            self.initial_p = initial_p

        if weights is None:
            self.W = np.eye(len(self.x))
        else:
            self.W = weights

    def initialize_p(self):
        return np.random.normal(size=self.n_input_dims)

    def condition_almost_singular(self, delta_p, p):
        t = self.threshold_singular
        a = squared_norm(delta_p)
        b = (squared_norm(p) + t) / self.threshold_sba
        return a >= b

    def condition_relative_change(self, delta_p, p):
        """
        ||\\delta_{\\mathbf{p}}||^2 \leq \\epsilon^2 ||\\mathbf{p}||^2
        """
        t = self.threshold_relative_change
        a = squared_norm(delta_p)
        b = pow(t, 2) * squared_norm(p)
        return a <= b

    def calculate_rho(self, p, J, g, delta_p, epsilon_p, mu):
        dc = weighted_squared_norm(epsilon_p, self.W)
        dn = weighted_squared_norm(self.x - self.f(p + delta_p), self.W)

        dl = np.dot(delta_p, (mu * delta_p + g))

        if dl == 0:  # avoid ZeroDivisionError
            return 0
        return (dc-dn) / dl

    def update(self, p, delta_p, mu, nu, rho):
        if rho > 0:
            p, mu, nu = self.update_positive(p, delta_p, mu, nu, rho)
        else:
            mu, nu = self.update_negative(mu, nu)
        return p, mu, nu

    def update_positive(self, p, delta_p, mu, nu, rho):
        p = p + delta_p
        mu = mu * max(1/3, 1 - pow(2*rho - 1, 3))
        nu = 2.0
        return p, mu, nu

    def update_negative(self, mu, nu):
        mu = mu * nu
        nu = 2 * nu
        return mu, nu

    def initialize(self):
        p = self.initialize_p()
        J = self.J(p)
        g = np.dot(J.T, self.x - self.f(p))

        v = np.sum(J * J, axis=0)  # equivalent to np.diag(np.dot(J.T, J))

        mu = self.tau * v.max()
        nu = 2
        return p, J, g, mu, nu

    def calculate_update(self, J, g, mu):
        A = np.dot(J.T, J)
        delta_p = np.linalg.solve(A + mu * self.I, g)
        return delta_p

    def optimize(self, max_iter=200):
        p, J, g, mu, nu = self.initialize()

        for k in range(max_iter):
            epsilon_p = self.x - self.f(p)

            delta_p = self.calculate_update(J, g, mu)

            if self.condition_relative_change(delta_p, p):
                print("Condition relative change")
                return p

            # if self.condition_almost_singular(delta_p, p):
            #     raise RuntimeError("#TODO add comment")

            rho = self.calculate_rho(p, J, g, delta_p, epsilon_p, mu)
            p, mu, nu = self.update(p, delta_p, mu, nu, rho)

            J = self.J(p)
            g = np.dot(J.T, epsilon_p)
        return p
