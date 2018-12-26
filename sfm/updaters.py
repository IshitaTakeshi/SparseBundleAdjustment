# References:

# [1] Manolis I. A. Lourakis and Antonis A. Argyros.
# "SBA: A software package for generic sparse bundle adjustment."
# ACM Transactions on Mathematical Software (TOMS) 36.1 (2009): 2.
#
# [2] Sameer, Noah et al. "Bundle adjustment in the large." ECCV 2010


import numpy as np
from scipy.sparse import lil_matrix, diags, linalg


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
    def __init__(self, function, jacobian, target):
        self.f = function
        self.J = jacobian

        self.mask = self.compute_mask(target)
        self.masked_x = target[self.mask]

    def compute_mask(self):
        boolean_mask = np.logical_not(np.isnan(x))
        # represent the mask as indices
        return np.arange(J.shape[0])[boolean_mask]

    def evaluate_at(self, p):
        self.p = p

        J = self.J(p)
        f = self.f(p)

        f, J = f[self.mask], J[self.mask, :]

        self.K = J.T.dot(J)
        self.D = diags(self.K.diagonal())
        self.g = J.T.dot(self.masked_x - f)

    def calculate(self, lambda_):
        dp = linalg.spsolve(self.K + lambda_ * self.D, self.g)
        return self.p + dp
