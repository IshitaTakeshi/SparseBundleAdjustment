
import numpy as np
from numpy.linalg import inv
from scipy.sparse import bsr_matrix

from sfm.utils import bsr_eye_matrix
from sfm.projection import projection, calc_pose_and_structure_jacobian
from sfm.jacobian import camera_pose_jacobian, structure_jacobian


# here we call 3D point coordinates structure parameters
n_pose_parameters = 6  # dimensions of a_j
n_point_parameters = 3  # dimensions of b_i


def calc_jacobian(camera_intrinsic, initial_rotations, points3d, poses):
    n_viewpoints = poses.shape[0]
    n_3dpoints = points3d.shape[0]
    P = np.empty((n_viewpoints, 2, n_pose_parameters))
    S = np.empty((n_3dpoints, 2, n_point_parameters))
    for i, point3d in enumerate(points3d):
        for j, (pose, rotation) in enumerate(zip(poses, initial_rotations)):
            P[j], S[i] = calc_pose_and_structure_jacobian(
                camera_intrinsic,
                rotation,
                pose[:3],
                pose[3:],
                point3d
            )

    A = camera_pose_jacobian(P, n_3dpoints, n_viewpoints, n_pose_parameters)
    B = structure_jacobian(S, n_3dpoints, n_viewpoints, n_point_parameters)
    return A, B



def inv_v(V):
    # inv(V) = diag(inv(V_1), ..., inv(V_i), ..., inv(V_n))
    for i in range(n_3dpoints):
        s = i * n_point_parameters
        t = (i+1) * n_point_parameters
        V[s:t, s:t] = inv(V[s:t, s:t].todense())
    return V


def inv_v(U):
    # inv(U) = diag(inv(U_1), ..., inv(U_i), ..., inv(U_m))
    for j in range(n_viewpoints):
        s = j * n_pose_parameters
        t = (j+1) * n_pose_parameters
        U[s:t, s:t] = inv(U[s:t, s:t].todense())
    return U


def calc_update(A, B, C_inv, epsilon, n_3dpoints, n_viewpoints):
    # eq. 12
    U = A.dot(C_inv).dot(A.T)
    W = A.dot(C_inv).dot(B.T)
    V = B.dot(C_inv).dot(B.T)

    CE = np.dot(C_inv, epsilon)
    epsilon_a = np.dot(A.T, CE)
    epsilon_b = np.dot(B.T, CE)

    # add the damping term to the diagonals
    U = U + mu * np.eye(n_pose_parameters * n_viewpoints)
    V = V + mu * np.eye(n_point_parameters * n_3dpoints)

    U_inv = inv_u(U)
    V_inv = inv_v(V)

    Y = W.dot(V_inv)

    S = U_inv - Y.dot(W.T)

    # eq. 21: calculate update of the pose parameters
    delta_a = np.linalg.solve(S, epsilon_a - np.dot(Y, epsilon_b))
    # eq. 22: calculate update of the structure parameters
    delta_b = np.linalg.solve(V, epsilon_b - np.dot(W, delta_a))

    return delta_a, delta_b


# TODO make compatibility with n_pose_parameters
def initial_rotations(n_viewpoints):
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


def initial_structure(n_3dpoints):
    return np.random.randn(n_3dpoints, n_point_parameters)


class SBA(object):
    # TODO add mask to indicate invisible points
    def __init__(self, camera_intrinsic, observation,
                 covariances=None, initial_rotations_=None):
        self.camera_intrinsic = camera_intrinsic
        self.x_observation = observation
        # n_viewpoints : `m` in the paper
        # n_3dpoints   : `n` in the paper
        self.n_3dpoints, self.n_viewpoints = self.x_observation.shape[0:2]
        self.inv_covariance = inv_covariance(
            self.n_3dpoints,
            self.n_viewpoints,
            covariances
        )

        if initial_rotations_ is None:
            initial_rotations_ = initial_rotations(self.n_viewpoints)
        self.initial_rotations = initial_rotations_

    def update(self, structure_parameters, pose_parameters):
        x_pred = projection(
            self.camera_intrinsic,
            self.initial_rotations,
            structure_parameters,
            pose_parameters
        )

        print(structure_parameters.shape)
        print(pose_parameters.shape)
        print(self.n_viewpoints, self.n_3dpoints)

        A, B = calc_jacobian(
            self.camera_intrinsic,
            self.initial_rotations,
            structure_parameters,
            pose_parameters
        )
        epsilon = x_pred - self.x_observation
        return calc_update(A, B, self.inv_covariance, epsilon,
                           self.n_3dpoints, self.n_viewpoints)

    def optimize(self, max_iter):
        structure_parameters = initial_structure(self.n_3dpoints)
        pose_parameters = initial_poses(self.n_viewpoints)
        for i in range(max_iter):
            delta_a, delta_b = self.update(structure_parameters,
                                           pose_parameters)

            pose_parameters -= delta_a.reshape(pose_parameters.shape)
            structure_parameters -= delta_b.reshape(structure_parameters)

            # if condition():
            #     break

        return pose_parameters, structure_parameters


# FIXME the interface looks redundant
def inv_covariance(n_viewpoints, n_3dpoints, covariances=None):
    size = n_viewpoints * n_3dpoints  # number of 2d points

    if covariances is None:
        return bsr_eye_matrix(2 * size)

    def column_indices(i):
        def block_column_indices(j):
            s = j * 2
            t = (j + 1) * 2
            return np.arange(s, t)
        return np.concatenate([block_row_indices(j) for j in np.arange(size)])

    def row_indices():
        return np.repeat(np.arange(size), 2)

    inv_covariances = np.array([inv(c) for c in covariances])
    data = inv_covariances.flatten()
    row = row_indices()
    col = column_indices()
    return bsr_matrix((data, (row, col)), blocksize=(2, 2))
