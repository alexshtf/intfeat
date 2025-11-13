import numpy as np
import scipy
from sklearn.utils.validation import column_or_1d
from sklearn.utils import as_float_array
from typing import Tuple
from .orth_base import HistogramFitter


def fit_laplacian_hist(hist, num_coefs=10, shaping_strength=1):
    """
    Fit a histogram to a set of coefficients.
    """
    # Build the tridiagonal matrix
    max_val = len(hist)
    base = 2 * np.ones(max_val)
    base[0] = base[-1] = 1.0
    diag = base - shaping_strength * hist
    off_diag = -np.ones(max_val - 1)

    # project onto eigenspace
    eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(
        diag, off_diag, select="i", select_range=(0, num_coefs - 1)
    )
    coefficients = eigvecs.T @ hist
    apx_hist = eigvecs @ coefficients

    return apx_hist, eigvals, eigvecs


class CutoffStrategy:
    def __init__(
        self, max_val: int = 65536, quantile: float = 0.99, quantile_factor: float = 1.1
    ):
        self.max_val = max_val
        self.quantile = quantile
        self.quantile_factor = quantile_factor

    def get_cutoff(self, col) -> Tuple[float, np.typing.NDArray]:
        max_col = np.max(col).item()
        if max_col < self.max_val:
            return max_col, col

        high_quantile = np.quantile(col, self.quantile)
        cutoff = int(min(high_quantile * self.quantile_factor, self.max_val))
        clipped = np.clip(col, 0, cutoff)
        return cutoff, clipped


class KTHistogramFitter(HistogramFitter):
    def __init__(
        self, cutoff_strategy: CutoffStrategy | None = None, prior_count: float = 0.5
    ):
        self.cutoff_strategy = (
            cutoff_strategy if cutoff_strategy is not None else CutoffStrategy()
        )
        self.prior_count = prior_count

    def fit(self, data):
        data = column_or_1d(data, dtype=[np.int64, np.int32, np.int16])
        self.cutoff_, data = self.cutoff_strategy.get_cutoff(data)

        counts = np.bincount(data, minlength=self.cutoff_)
        self.apx_hist_ = (counts + self.prior_count) / (
            np.sum(counts) + self.prior_count * len(counts)
        )
        self.cdf_ = np.cumsum(self.apx_hist_)

    def pmf(self, x=None):
        if x is None:
            return self.apx_hist_
        return self.apx_hist_[x]

    def cdf(self, x=None):
        if x is None:
            return self.cdf_
        return self.cdf_[x]

    def tail_cutoff(self):
        return self.cutoff_


class LaplacianHistogramFitter(HistogramFitter):
    def __init__(
        self,
        cutoff_strategy: CutoffStrategy | None = None,
        min_prob_factor: float = 1e-3,
    ):
        """
        Fit a histogram to data.
        """
        self.cutoff_strategy = (
            cutoff_strategy if cutoff_strategy is not None else CutoffStrategy()
        )
        self.min_prob_factor = min_prob_factor

    def fit(self, data):
        data = column_or_1d(data, dtype=[np.int64, np.int32, np.int16])
        self.cutoff_, data = self.cutoff_strategy.get_cutoff(data)

        # compute histogram
        hist = as_float_array(np.bincount(data, minlength=self.cutoff_))
        hist /= np.sum(hist)
        self.hist_ = hist

        # fit histogram using eigenspace projection
        apx_hist, _, _ = fit_laplacian_hist(
            hist,
            self.num_coefs,
        )

        self.apx_hist_ = self._normalize(apx_hist)
        self.cdf_ = np.cumsum(self.apx_hist_)

    def pmf(self, x=None):
        if x is None:
            return self.apx_hist_
        return self.apx_hist[x]

    def cdf(self, x=None):
        if x is None:
            return self.cdf_
        return self.cdf_[x]

    def tail_cutoff(self):
        return self.cutoff_

    def _normalize(self, apx_hist):
        min_prob = self.min_prob_factor / self.cutoff_
        apx_hist = (np.hypot(2 * min_prob, apx_hist) + apx_hist) / 2
        return apx_hist / np.sum(apx_hist)
