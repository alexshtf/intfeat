import numpy as np
from scipy.sparse import diags_array
from scipy.linalg import eigh_tridiagonal
from scipy.special import logit


def _build_cs_matrix(cs):
    head = cs[:-1]
    tail = cs[1:]
    main_diagonal = np.r_[head[0], head + tail, tail[-1]]
    off_diagonal = -cs
    return diags_array(
        (off_diagonal, main_diagonal, off_diagonal),
        offsets=[-1, 0, 1],
    )


def _compute_eigenfunctions(cs, ws, k):
    cs_mat = _build_cs_matrix(cs)
    ds_mat = diags_array((1 / np.sqrt(ws)))
    eig_mat = ds_mat @ cs_mat @ ds_mat

    main_diag = eig_mat.diagonal(0)
    off_diag = eig_mat.diagonal(-1)
    vals, vecs = eigh_tridiagonal(
        main_diag, off_diag, select="i", select_range=(0, k - 1)
    )
    ortho_vecs = (vecs.T / np.sqrt(ws)).T
    return vals, ortho_vecs


class StrumLiouvilleBasis:
    def __init__(
        self, *, max_val=2000, num_funcs=20, weight_config=(1, 1), curvature_config=0.5
    ):
        """Initializes a Strum-Liouville basis with given parameters.

        Args:
            max_val (int): The maximum value for the basis functions.
            num_funcs (int): The number of basis functions to compute. Must be less than max_val.
            weight_config (tuple or np.ndarray): A tuple containing the weight parameters (alpha, beta).
            curvature_config (float or np.ndarray): The curvature parameter(s). A float between 0 and 1 controls the
                concentration of curvature, where values close to 0 concentrate curvature near small integers,
                values close to 1 concentrate curvature near large integers, and 0.5 means uniform curvature.
        """
        if num_funcs >= max_val:
            raise ValueError(
                f"Number of functions {num_funcs} must be less than the maximum value {max_val}"
            )

        self.max_val = max_val
        match weight_config:
            case (alpha, beta) if alpha > 0 and beta > 0:
                self.weights = 1 / (1 + np.arange(max_val)) ** beta
            case np.ndarray() as ws if len(ws) == max_val and np.all(ws > 0):
                self.weights = ws
            case _:
                raise ValueError(
                    "Invalid weight_config provided. Must be either a tuple of two positive floats, or a 1D numpy array."
                )
        self.weights /= np.sum(self.weights)

        match curvature_config:
            case float() as gamma if gamma >= 0:
                xs = (1.0 + np.arange(max_val - 1).astype(float)) / (max_val + 1.0)
                cs = xs ** logit(gamma)
                self.cs = cs / np.max(cs)
            case np.ndarray() as cs if len(cs) == max_val - 1 and np.all(cs > 0):
                self.cs = cs / np.max(cs)
            case _:
                raise ValueError(
                    "Invalid curvature_config provided. Must be either a positive float, or a 1D numpy array."
                )

        self.num_funcs = num_funcs
        self.eigenfunctions_ = None
        self.eigenvalues_ = None

    def __call__(self, xs):
        if self.eigenfunctions_ is None:
            self.eigenvalues_, self.eigenfunctions_ = _compute_eigenfunctions(
                self.cs, self.weights, self.num_funcs
            )

        orig_shape = xs.shape
        xs = np.minimum(xs.flatten(), self.max_val)
        vander = self.eigenfunctions_[xs, :]
        return vander.reshape((*orig_shape, self.num_funcs))
