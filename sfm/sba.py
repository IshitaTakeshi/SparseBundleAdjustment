
import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from scipy import sparse

from sfm.projection import projection, jacobian_projection
from sfm.config import n_pose_parameters, n_point_parameters


# here we call 3D point coordinates 'structure parameters'


class ParameterManager(object):
    def __init__(self, n_3dpoints, n_viewpoints):
        self.n_3dpoints = n_3dpoints
        self.n_viewpoints = n_viewpoints

    @property
    def length_all_3dpoints(self):
        return self.n_3dpoints * n_point_parameters

    @property
    def length_all_poses(self):
        return self.n_viewpoints * n_pose_parameters

    @property
    def total_parameter_size(self):
        return self.length_all_3dpoints + self.length_all_poses

    def decompose(self, p):
        N = self.length_all_poses
        M = self.length_all_3dpoints

        assert(len(p) == N + M)
        # This part is confusing. The left side of the vector p is
        # `poses` and the right side is `points3d`
        poses = p[:N].reshape(self.n_viewpoints, n_pose_parameters)
        points3d = p[N:N+M].reshape(self.n_3dpoints, n_point_parameters)
        return points3d, poses

    def compose(self, points3d, poses):
        # This part is confusing. The left side of the vector p is
        # `poses` and the right side is `points3d`
        return np.concatenate((poses.flatten(), points3d.flatten()))


class SBA(object):
    def __init__(self, manager, camera_parameters):
        """
        Args:
            camera_parameters (CameraParameters): Camera intrinsic parameters
        """

        self.manager = manager
        self.camera_parameters = camera_parameters

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

        # P.shape == (n_3dpoints, n_viewpoints, 2)

        points3d, poses = self.manager.decompose(p)
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
