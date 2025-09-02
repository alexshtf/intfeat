from .strum_liouville import StrumLiouvilleBasis
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
import scipy.sparse as sp


def kt_smoothed_histogram(data, max_val, prior=0.5):
    data = np.clip(data, 0, max_val - 1)  # chop-off the tail
    raw_counts = np.bincount(data, minlength=max_val)  # compute raw counts
    return (raw_counts + prior) / (np.size(data) + prior * max_val)


class StrumLiouvilleColumnTransformer:
    def __init__(
        self,
        num_funcs=20,
        *,
        max_val=None,
        weight_config=None,
        curvature_gamma=0.0,
        include_bias=False,
        max_infer_quantile=0.95,
        max_infer_factor=1.1,
    ):
        """Initialize an sklearn transformer-like class for transforming one column of data.

        Args:
            num_funcs (int): The number of basis functions to use.
            max_val (int, optional): The maximum value for the input data. If not specified, determined
                from the data during fitting as the maximum value + 10% margin.
            weight_config ((float, float) | optional): Configuration for determining the data
                distribution. If two floats (a, b) are specified the distribution follows the power-law
                proportional to w(x) = (x+a)^{-b} defined for x = 0, 1, ..., max_val. Otherwise,
                determined from the data using KT smoothing.
            curvature_gamma (float): The curvature parameter for the basis functions.
                Curvature will be determined by conductance weights of the form
                c(x) = (1 + x)^{-curvature_gamma} for x = 0, 1, ..., max_val - 1.
            include_bias (bool): Whether to include a bias term in the basis functions.
            max_infer_quantile (float): The quantile to use when infering the maximum range of the feature during fit.
            max_infer_factor (float): The factor multiplying the quantile for infering maximum during a fit.
        """
        self.num_funcs = num_funcs
        self.max_val = max_val
        self.weight_config = weight_config
        self.curvature_gamma = curvature_gamma
        self.include_bias = include_bias
        self.max_infer_quantile = max_infer_quantile
        self.max_infer_factor = max_infer_factor

    def fit(self, X, y=None):
        X = np.ravel(X)

        # determine max_val_
        if self.max_val:
            self.max_val_ = self.max_val
        else:
            self.max_val_ = (
                int(np.quantile(X, self.max_infer_quantile) * self.max_infer_factor) + 1
            )
        self.max_val_ = max(self.max_val_, self.num_funcs + 1)

        if self.weight_config:
            weight_config = self.weight_config
        else:
            weight_config = kt_smoothed_histogram(X, self.max_val_)

        # build basis functions
        self.basis_ = StrumLiouvilleBasis(
            num_funcs=self.num_funcs,
            max_val=self.max_val_,
            weight_config=weight_config,
            curvature_config=self.curvature_gamma,
        )

    def transform(self, X):
        clipped_x = np.clip(X, 0, self.max_val_ - 1)
        basis = self.basis_(clipped_x)
        if not self.include_bias:
            basis = basis[:, 1:]
        return basis


class StrumLiouvilleTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_funcs=20,
        *,
        max_val=None,
        weight_config=None,
        curvature_gamma=0.0,
        include_bias=False,
        max_infer_quantile=0.95,
        max_infer_factor=1.1,
    ):
        self.num_funcs = num_funcs
        self.max_val = max_val
        self.weight_config = weight_config
        self.curvature_gamma = curvature_gamma
        self.include_bias = include_bias
        self.max_infer_quantile = max_infer_quantile
        self.max_infer_factor = max_infer_factor

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=True, accept_sparse=True)
        if sp.issparse(X):
            X = X.toarray()

        _, num_cols = X.shape

        self.col_transformers = [
            StrumLiouvilleColumnTransformer(
                num_funcs=self.num_funcs,
                max_val=self.max_val,
                weight_config=self.weight_config,
                curvature_gamma=self.curvature_gamma,
                include_bias=self.include_bias,
                max_infer_quantile=self.max_infer_quantile,
                max_infer_factor=self.max_infer_factor,
            )
            for _ in range(num_cols)
        ]
        for i in range(num_cols):
            self.col_transformers[i].fit(X[:, i], y)
        return self

    def transform(self, X, y=None):
        X = check_array(X, ensure_2d=True, accept_sparse=True)
        if sp.issparse(X):
            X = X.toarray()

        _, num_cols = X.shape
        transformed = [None] * num_cols
        for i in range(num_cols):
            transformed[i] = self.col_transformers[i].transform(X[:, i])
        return np.hstack(transformed)
