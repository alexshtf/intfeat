import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
import scipy.sparse as sp
from sklearn.utils.validation import column_or_1d
from .strum_liouville import _compute_eigenfunctions
from .orth_base import HistogramFitter, CurvatureSpec
from .hist_fit import KTHistogramFitter


class StrumLiouvilleColumnTransformer:
    def __init__(
        self,
        num_funcs=8,
        *,
        hist_fitter: Optional[HistogramFitter] = None,
        curvature_spec: Optional[CurvatureSpec] = None,
    ):
        """Initialize an sklearn transformer-like class for transforming one column of data.

        Args:
            num_funcs (int): The number of basis functions to use.
            hist_fitter (HistogramFitter, optional): An object used to fit the column data distribution. If `None`,
                then `LaplacianHistogramFitter` is used.
            curvature_spec (CurvatureSpec, optional): Curvature specification object. If `None`, the default curvature
                spec if used, which imposes _uniform_ curvature everywhere.
        """
        self.num_funcs = num_funcs
        self.hist_fitter = KTHistogramFitter() if hist_fitter is None else hist_fitter
        self.curvature_spec = (
            CurvatureSpec() if curvature_spec is None else curvature_spec
        )

    def fit(self, X, y=None):
        X = X[np.isfinite(X)]
        self.hist_fitter.fit(X)
        self.cs_ = self.curvature_spec.compute_weights(
            np.arange(len(self.hist_fitter.apx_hist_) - 1)
        )
        num_eigenvectors = min(self.num_funcs, self.hist_fitter.tail_cutoff())
        self.eigenvalues_, self.eigenvectors_ = _compute_eigenfunctions(
            self.cs_, self.hist_fitter.pmf(), num_eigenvectors
        )

    def transform(self, X):
        X = column_or_1d(X)
        num_rows = len(X)
        num_cols = self.eigenvectors_.shape[1]

        finite_mask = np.isfinite(X)
        X = np.clip(X[finite_mask], 0, self.hist_fitter.tail_cutoff())
        if not np.isdtype(X.dtype, "integral"):
            X = np.floor(X).astype(np.int64)

        result = np.zeros((num_rows, num_cols), dtype=self.eigenvectors_.dtype)
        result[finite_mask, :] = self.eigenvectors_[X, :]
        return result


class StrumLiouvilleTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_funcs=8,
        *,
        hist_fitter: Optional[HistogramFitter] = None,
        curvature_spec: Optional[CurvatureSpec] = None,
    ):
        self.num_funcs = num_funcs
        self.hist_fitter = hist_fitter
        self.curvature_spec = curvature_spec

    def fit(self, X, y=None):
        X = check_array(
            X, ensure_2d=True, accept_sparse=True, ensure_all_finite="allow-nan"
        )
        if sp.issparse(X):
            X = X.toarray()

        _, num_cols = X.shape

        self.col_transformers_ = [
            StrumLiouvilleColumnTransformer(
                num_funcs=self.num_funcs,
                hist_fitter=self.hist_fitter,
                curvature_spec=self.curvature_spec,
            )
            for _ in range(num_cols)
        ]
        for i in range(num_cols):
            self.col_transformers_[i].fit(X[:, i], y)
        return self

    def transform(self, X, y=None):
        X = check_array(
            X, ensure_2d=True, accept_sparse=True, ensure_all_finite="allow-nan"
        )
        if sp.issparse(X):
            X = X.toarray()

        _, num_cols = X.shape
        transformed = [None] * num_cols
        for i in range(num_cols):
            transformed[i] = self.col_transformers_[i].transform(X[:, i])
        return np.hstack(transformed)
