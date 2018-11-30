import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def squared_norm(v):
    return np.sum(np.power(v, 2))


def squared_mahalanobis_distance(x, y, C):
    """
    Calculate mahalanobis distance between :math:`\\mathbf{x}` and
    :math:`\\mathbf{y}` i.e. :math:`\\mathbf{x}^{\\top}C\\mathbf{y}`
    """
    return x.T.dot(C.dot(y))


def weighted_squared_norm(x, W):
    return squared_mahalanobis_distance(x, x, W)


class LevenbergMarquardt(object):
    """
    Find :math:`\\mathbf{p}` which satisfies
    :math:`\\mathbf{f}(\\mathbf{p}) = \\mathbf{x}`
    """
    """Fig. 2"""

    # TODO set the hyperparameter
    def __init__(self, function, jacobian, target, n_input_dims=None,
                 initial_p=None, weights=None,
                 tau=0.01, threshold_singular=0.01,
                 threshold_relative_change=1e-4,
                 threshold_sba=0.01):
        """
        Args:
            n_input_dims (int): Number of dimensions of :math:`\\mathbf{p}`
        """

        self.f = function
        self.J = jacobian
        self.x = target
        self.I = sparse.eye(n_input_dims)
        self.tau = tau

        # TODO rename the variables below. they are actually not thresholds
        self.threshold_singular = threshold_singular
        self.threshold_relative_change = threshold_relative_change
        self.threshold_sba = threshold_sba  # TODO remove this if possible

        if initial_p is None:
            self.initial_p = self.initialize_p(n_input_dims)
        else:
            self.initial_p = initial_p

        if weights is None:
            self.W = sparse.eye(len(self.x))
        else:
            self.W = weights

    def initialize_p(self, n_input_dims):
        return np.random.normal(size=n_input_dims)

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

    def calculate_rho(self, p, g, delta_p, epsilon_p, mu):
        # mahalanobis distance from the current estimation to the target
        current = weighted_squared_norm(epsilon_p, self.W)
        # and the distance from the next candidate estimation to the target
        next_ = weighted_squared_norm(self.x - self.f(p + delta_p), self.W)

        # dl = mu * ||delta_p||^2 + dot(delta_p, g)
        #    = mu * ||delta_p||^2 + dot(delta_p, dot(A + mu * I, delta_p))
        #    = mu * ||delta_p||^2 + mahalanobis(delta_p, delta_p, A + mu * I)^2
        # Since A = dot(J.T, J) is symmetric, A is a positive definite matrix.
        # Therefore dl > 0

        dl = delta_p.dot(mu * delta_p + g)

        if dl == 0:  # avoid ZeroDivisionError
            return 0
        return (current-next_) / dl

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
        p = self.initial_p
        J = self.J(p)
        epsilon = self.x - self.f(p)
        g = J.T.dot(epsilon)

        v = J.multiply(J).sum(axis=0)  # equivalent to diag(dot(J.T, J))

        mu = self.tau * v.max()
        nu = 2
        return p, J, g, mu, nu

    def calculate_update(self, J, g, mu):
        A = J.T.dot(J)
        delta_p = spsolve(A + mu * self.I, g)
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
            g = J.T.dot(epsilon_p)
        return p
