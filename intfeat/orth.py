import numpy as np
from scipy.linalg import solve_triangular
from .gram import powerlaw_gram_matrix


class OrthPowerlawBasis:
    def __init__(self, atoms_fn, alpha=0.01, beta=2, orth_n_max=100000):
        self.atoms_fn = atoms_fn
        self.alpha = alpha
        self.beta = beta
        self.orth_n_max = orth_n_max
        self.gram_cholesky_factor = None
        self.normalization = None

    def vander(self, xs):
        self._compute_orthogonalization_params()
        atoms = self.atoms_fn(xs)
        return self._orthogonalize(atoms) / self.normalization

    def _orthogonalize(self, atoms):
        return solve_triangular(self.gram_cholesky_factor, atoms.T, lower=True).T

    def _compute_orthogonalization_params(self):
        if self.gram_cholesky_factor is not None:
            return

        atoms = self.atoms_fn(np.arange(self.orth_n_max))
        gram_matrix = powerlaw_gram_matrix(atoms, self.alpha, self.beta)
        self.gram_cholesky_factor = np.linalg.cholesky(gram_matrix)
        self.normalization = np.arange(1, 1 + atoms.shape[1]).astype(gram_matrix.dtype)
        self.normalization = 1.0
