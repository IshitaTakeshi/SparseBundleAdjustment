
def squared_norm(v):
    np.sum(np.power(v, 2))


def squared_mahalanobis_distance(x, y, C):
    return np.dot(np.dot(x.T, C), y)


def weighted_squared_norm(x, W):
    return squared_mahalanobis_distance(x, x, W)


class LevenbergMarquardt(object):
    """Fig. 2"""

    def __init__(self, f):
        self.f = f

    def condition_almost_singular(self, delta_p, p):
        a = squared_norm(delta_p)
        b = (squared_norm(p) + self.threshold2) / self.threshold_sba
        return a >= b

    def condition_relative_change(self, delta_p, p):
        a = squared_norm(delta_p)
        b = pow(self.threshold2, 2) * squared_norm(p)
        return a <= b

    def calculate_rho(self, p, delta_p, epsilon_p, W, mu):
        dc = weighted_squared_norm(epsilon_p, self.covariance)
        dn = weighted_squared_norm(x - self.f(p + delta_p), self.covariance)

        g = np.dot(self.J.T, epsilon_p)
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
        nu = 2
        return p, mu, nu

    def update_negative(self, mu, nu):
        mu = mu * nu
        nu = 2 * nu
        return mu, nu

    def optimize(self):
        for k in range(max_iter):
            epsilon_p = x - self.f(p)

            delta_p = self.calculate_update(mu, epsilon_p)

            if self.condition_relative_change(delta_p, p):
                return
            if self.condition_almost_singular(delta_p, p):
                raise RuntimeError("#TODO add comment")

            rho = self.calculate_rho(p, delta_p, epsilon_p, mu)
            p, mu, nu = self.update(p, delta_p, mu, nu, rho)
        return p
