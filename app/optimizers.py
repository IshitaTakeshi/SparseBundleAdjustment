import numpy as np
from scipy.optimize import least_squares

from sfm.lm import LevenbergMarquardt
from sfm.metrics import SquaredNormResidual
from sfm.updaters import PCGUpdater
from sfm.initializers import Initializer


def optimize_scipy(sba, observations):
    x = observations.flatten()  # target value

    mask = np.logical_not(np.isnan(x))
    indices = np.arange(x.shape[0])[mask]

    def masked_diff(x1, x2):
        return x1[indices] - x2[indices]

    def error(p):
        d = masked_diff(sba.projection(p), x)
        return np.dot(d, d)

    def error_jacobian(p):
        d = masked_diff(sba.projection(p), x)
        J = sba.jacobian(p)

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
    function = sba.projection
    jacobian = sba.jacobian
    target = observation.flatten()

    residual = SquaredNormResidual(function, target)
    updater = PCGUpdater(function, jacobian, target, sba.length_all_poses)
    initializer = Initializer(sba.total_parameter_size)
    lm = LevenbergMarquardt(updater, residual, initializer,
                            initial_lambda=1e-6, nu=1.2)
    p = lm.optimize(max_iter=100)
    return sba.decompose(p)
