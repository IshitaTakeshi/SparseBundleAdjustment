import numpy as np

from sfm.config import n_pose_parameters, n_point_parameters


np.random.seed(1234)


def initial_rotations(n_viewpoints):
    q = np.random.random((n_viewpoints, 4))
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / n

    return np.hstack((
        np.ones((n_viewpoints, 1)),
        np.zeros((n_viewpoints, 3))
    ))


def initial_translations(n_viewpoints):
    return np.random.randn(n_viewpoints, 3)


def initial_poses(n_viewpoints):
    rotation = initial_rotations(n_viewpoints)
    translation = initial_translations(n_viewpoints)
    return np.hstack((rotation[:, 1:], translation))


def initial_3dpoints(n_3dpoints):
    return np.random.randn(n_3dpoints, n_point_parameters)


class Initializer(object):
    def __init__(self, parameter_manager):
        self.manager = parameter_manager

    def initial_value(self):
        points3d = initial_3dpoints(self.manager.n_3dpoints)
        poses = initial_poses(self.manager.n_viewpoints)
        return self.manager.compose(points3d, poses)
