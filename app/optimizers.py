import numpy as np
from scipy.optimize import least_squares

from sfm.sba import ParameterManager, SBA
from sfm.lm import LMIterator
from sfm.metrics import SquaredNormResidual
from sfm.updaters import LMUpdater
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


def optimize_lm(camera_parameters, observations):
    target = observations.flatten()

    n_3dpoints, n_viewpoints = observations.shape[:2]

    manager = ParameterManager(n_3dpoints, n_viewpoints)

    sba = SBA(manager, camera_parameters)

    residual = SquaredNormResidual(sba.projection, target)
    updater = LMUpdater(manager, sba, target)
    initializer = Initializer(manager)

    lm = LMIterator(updater, residual, initializer,
                    initial_lambda=1e+6, nu=1.2)
    p = lm.optimize(max_iter=2000)

    return manager.decompose(p)
