import numpy as np
import scipy


def fit_hist(hist, num_coefs=10, shaping_strength=1):
    """
    Fit a histogram to a set of coefficients.
    """
    # normalize the histogram
    hist = hist / np.sum(hist)

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

    # ensure valid PMF
    apx_hist = np.maximum(apx_hist, 0)
    s = apx_hist.sum()
    if s > 0:
        apx_hist = apx_hist / s

    return apx_hist, eigvals, eigvecs


class HistogramFitter:
    def __init__(self, num_coefs=10, shaping_strength=1):
        """
        Fit a histogram to data
        """
        self.num_coefs = num_coefs
        self.shaping_strength = shaping_strength

    def fit(self, data, max_val=None):
        # compute empirical histogram
        if max_val is None:
            max_val = np.max(data)
        data = np.clip(data, 0, max_val)
        hist = np.bincount(data, minlength=max_val)

        # fit histogram using eigenspace projection
        self.apx_hist_, self.eigvals_, self.eigvecs_ = fit_hist(
            hist, self.num_coefs, self.shaping_strength
        )
        self.cdf_ = np.cumsum(self.apx_hist_)

    def pmf(self, x=None):
        if x is None:
            return self.apx_hist_
        return self.apx_hist[x]

    def cdf(self, x=None):
        if x is None:
            return self.cdf_
        return self.cdf_[x]
