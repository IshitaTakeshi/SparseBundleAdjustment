
import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from scipy import sparse

from sfm.projection import projection, jacobian_projection
from sfm.config import n_pose_parameters, n_point_parameters

# here we call 3D point coordinates structure parameters


def initialize(A, B):
    # np.sum(X * X, axis=0) is equivalent to np.diag(np.dot(X.T, X))
    # J = np.hstack([A, B])

    A, B = J()
    g = np.hstack([
        np.dot(A.T, epsilon_a),
        np.dot(B.T, epsilon_b)
    ])

    a = np.sum(A * A, axis=0).max()
    b = np.sum(B * B, axis=0).max()
    mu = tau * np.max([a, b])

    J = sparse.hstack(A, B)
    epsilon = x - projection_one_point(x)
    g = J.dot(epsilon)

    return J


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


class SBA(object):
    def __init__(self, camera_parameters, n_3dpoints, n_viewpoints):
        """
        Args:
            camera_parameters (CameraParameters): Camera intrinsic parameters
            n_3dpoints (int): Number of 3D points to be reconstructed. `n` in the paper
            n_viewpoints (int) : Number of viewpoints. `m` in the paper
        """

        self.camera_parameters = camera_parameters
        self.n_3dpoints = n_3dpoints
        self.n_viewpoints = n_viewpoints

    @property
    def initial_p(self):
        points3d = initial_3dpoints(self.n_3dpoints)
        poses = initial_poses(self.n_viewpoints)
        return self.compose(points3d, poses)

    @property
    def length_all_3dpoints(self):
        return self.n_3dpoints * n_point_parameters

    @property
    def length_all_poses(self):
        return self.n_viewpoints * n_pose_parameters

    @property
    def total_parameter_size(self):
        return self.length_all_3dpoints + self.length_all_poses

    def compose(self, points3d, poses):
        # This part is confusing. The left side of the vector p is
        # `poses` and the right side is `points3d`
        return np.concatenate((poses.flatten(), points3d.flatten()))

    def decompose(self, p):
        N = self.length_all_poses
        M = self.length_all_3dpoints

        assert(len(p) == self.total_parameter_size)
        # This part is confusing. The left side of the vector p is
        # `poses` and the right side is `points3d`

        poses = p[:N].reshape(self.n_viewpoints, n_pose_parameters)
        points3d = p[N:N+M].reshape(self.n_3dpoints, n_point_parameters)
        return points3d, poses

    # @profile
    def projection(self, p):
        """
        If n_viewpoints = 2 and n_3dpoints = 3, the result array is

        ..
            [x_11, y_11
             x_12, y_12
             x_21, y_21
             x_22, y_22
             x_31, y_31
             x_32, y_32]

        where [x_ij, y_ij] is a predicted projection of point `i` on image`j`
        """

        # Honestly the definition of the observation is confusing.
        # P.shape == (n_viewpoints, n_3dpoints, 2)
        # is more clear and intuitive than
        # P.shape == (n_3dpoints, n_viewpoints, 2)
        # because `P[i]` contains observation from the view point `i`.
        # Although the observation sequence have to be correctly
        # associated with the rows of J (= sba.jacobian(p))

        points3d, poses = self.decompose(p)
        # P.shape == (n_3dpoints, n_viewpoints, 2)
        P = projection(self.camera_parameters, points3d, poses)
        return P.flatten()

    # @profile
    def jacobian(self, p):
        """
        Calculate J = dx / dp where x = self.projection(p)

        Returns:
            Jacobian of shape
            (len(x), len(p)) =
            (n_3dpoints * n_viewpoints * 2,
             n_viewpoints * n_pose_parameters +
             n_3dpoints * n_point_parameters)
        """

        points3d, poses = self.decompose(p)
        A, B = jacobian_projection(
            self.camera_parameters,
            points3d, poses
        )

        J = sparse.hstack((A, B))
        return J.tocsr()


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


def calc_update(A, B, C_inv, epsilon, n_3dpoints, n_viewpoints, mu):
    # eq. 12
    U = A.T.dot(C_inv).dot(A)
    W = A.T.dot(C_inv).dot(B)
    V = B.T.dot(C_inv).dot(B)

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


# FIXME the interface looks redundant
def inv_covariance(n_3dpoints, n_viewpoints, covariances=None):
    size = n_3dpoints * n_viewpoints   # number of 2d points

    if covariances is None:
        return sparse.eye(2 * size)

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
    return csr_matrix((data, (row, col)))
