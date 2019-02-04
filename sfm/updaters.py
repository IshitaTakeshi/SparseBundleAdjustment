# References:

# [1] Manolis I. A. Lourakis and Antonis A. Argyros.
# "SBA: A software package for generic sparse bundle adjustment."
# ACM Transactions on Mathematical Software (TOMS) 36.1 (2009): 2.
#
# [2] Sameer, Noah et al. "Bundle adjustment in the large." ECCV 2010


import numpy as np
from scipy.sparse import lil_matrix, diags, linalg, identity


# Preconditioned Conjugate Gradients
# See [1]


class PCGUpdater(object):
    def __init__(self, function, jacobian, target, length_all_poses):
        self.f = function
        self.J = jacobian
        self.x = target

        self.length_all_poses = length_all_poses

    def compute_preconditioner(self, K):
        N = self.length_all_poses

        M = lil_matrix(K.shape)
        M[:N, :N] = K[:N, :N]  # U* in [1] or C in [2]
        M[N:, N:] = K[N:, N:]  # V* in [1] or C in [2]
        return linalg.inv(M.tocsc())

    def evaluate_at(self, p):
        self.p = p

        J = self.J(p)

        self.K = J.T.dot(J)
        self.D = diags(self.K.diagonal())

        self.M_inv = self.compute_preconditioner(self.K)

        g = J.T.dot(self.x - self.f(p))
        self.b = self.M_inv.dot(g)

    def calculate(self, lambda_):
        H = self.K + lambda_ * self.D
        A = self.M_inv.dot(H)
        dp, info = linalg.cg(A, self.b)

        # TODO force this calculation by implementing in the abstract class
        # to avoid bugs
        return self.p + dp


class LMUpdater(object):
    def __init__(self, parameter_manager, sba, target):
        self.manager = parameter_manager
        self.sba = sba
        self.target = target

    def evaluate_at(self, p):
        A, B = self.sba.jacobian(p)
        residual = self.target - self.sba.projection(p)
        self.precomputation(A, B, residual)

    def precomputation(self, A, B, residual):
        # it's still ok to write the process below directly to 'evaluate_at'
        # but separating functions to make tests easier

        self.A = A
        self.B = B
        self.U = A.T.dot(A)
        self.V = B.T.dot(B)
        self.W = A.T.dot(B)

        self.epsilon_a = A.T.dot(residual)
        self.epsilon_b = B.T.dot(residual)

    def calc_update(self, lambda_):
        A, B = self.A, self.B
        W = self.W
        epsilon_a, epsilon_b = self.epsilon_a, self.epsilon_b

        # Actually U and V below are U_star and V_star respectively,
        # although beautiful is better than ugly
        U = self.U + lambda_ * identity(self.manager.length_all_poses)
        V = self.V + lambda_ * identity(self.manager.length_all_3dpoints)

        V_inv = linalg.inv(V)

        S = U - W.dot(V_inv).dot(W.T)
        b = epsilon_a - W.dot(V_inv).dot(epsilon_b)
        delta_a = linalg.spsolve(S, b)

        b = epsilon_b - W.T.dot(delta_a)
        delta_b = linalg.spsolve(V, b)

        # in this case it seems spsolve returns np.ndarray
        # so we can just concatenate directly
        dp = np.concatenate((delta_a, delta_b))
        return dp
