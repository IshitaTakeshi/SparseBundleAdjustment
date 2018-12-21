import numpy as np
from scipy.optimize import least_squares


def optimize_scipy(sba, observations):
    x = observations.flatten()  # target value

    mask = np.logical_not(np.isnan(x))

    def masked_diff(x1, x2):
        return x1[mask] - x2[mask]

    def error(p):
        d = masked_diff(sba.projection(p), x)
        return np.dot(d, d)

    def error_jacobian(p):
        d = masked_diff(sba.projection(p), x)
        J = sba.jacobian(p)

        indices = np.arange(len(x))[mask]
        J = J[indices, :]  # d x[mask] / dp
        return 2 * J.T.dot(d)


    p0 = np.random.normal(size=sba.total_parameter_size)
    result = least_squares(error, p0, error_jacobian, verbose=2)
    p = result.x

    print("cost: ", result.cost)
    print("grad: ", result.grad)
    print("status: ", result.status)

    return sba.decompose(p)


def optimize_lm(sba, observation):
    lm = LevenbergMarquardt(sba.projection, sba.jacobian, observation,
                            n_input_dims=sba.total_parameter_size,
                            threshold_relative_change=1e-9,
                            initial_p=sba.initial_p, tau=0.1)
    p = lm.optimize(max_iter=100)
    return sba.decompose(p)
