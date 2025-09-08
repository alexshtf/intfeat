import numpy as np
import scipy
from sklearn.utils.validation import column_or_1d
from sklearn.utils import as_float_array


def fit_hist(hist, num_coefs=10, shaping_strength=1):
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


class HistogramFitter:
    def __init__(
        self,
        num_coefs: int = 10,
        max_val: int = 65536,
        quantile: float = 0.99,
        quantile_factor: float = 1.1,
        min_prob: float = 1e-8,
    ):
        """
        Fit a histogram to data.
        """
        self.num_coefs = num_coefs
        self.max_val = max_val
        self.quantile = quantile
        self.quantile_factor = quantile_factor
        self.min_prob = min_prob

    def fit(self, data):
        # convert to 1D numpy array of correct type
        data = column_or_1d(data, dtype=[np.int64, np.int32, np.int16])

        # cut-off data
        self.cutoff_ = self._get_cutoff(data)
        data = np.clip(data, 0, self.cutoff_)

        # compute histogram
        hist = as_float_array(np.bincount(data, minlength=self.cutoff_))
        hist /= np.sum(hist)

        # fit histogram using eigenspace projection
        apx_hist, _, _ = fit_hist(
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

    def _normalize(self, apx_hist):
        apx_hist = np.maximum(apx_hist, self.min_prob)
        return apx_hist / np.sum(apx_hist)

    def _get_cutoff(self, col):
        max_col = np.max(col).item()
        if max_col < self.max_val:
            return max_col

        high_quantile = np.quantile(col, self.quantile)
        return int(min(high_quantile * self.quantile_factor, self.max_val))
