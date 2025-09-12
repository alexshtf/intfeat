from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import minmax_scale


class HistogramFitter:
    def fit(self, data):
        pass

    def pmf(self, x=None):
        pass

    def tail_cutoff(self):
        pass


@dataclass
class CurvatureSpec:
    alpha: float = 1.0
    """A positive value specifying the concentration of curvature. Higher is more uniform"""

    beta: float = 0.0
    """A non-negative value specifying the 'tailedness' of the curvature spec according to a power-law.

    A completely uniform curvature has zero tailedness.
    """

    center: float = 0.0
    """A value between 0 and 1 specifying the location of curvature concentration.

    0 means most of the curvature is concentrated at smaller values
    1 means most curvature is concentrated at higher values.
    """

    def compute_weights(self, xs: np.ndarray):
        xs = minmax_scale(xs)
        return 1 - self.alpha / (self.alpha + np.abs(xs - self.center) ** self.beta)
