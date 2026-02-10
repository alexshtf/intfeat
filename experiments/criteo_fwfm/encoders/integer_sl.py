from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from intfeat.orth_base import CurvatureSpec
from intfeat.strum_liouville import _compute_eigenfunctions

from .common import to_nullable_int


@dataclass
class SLIntegerEncoderConfig:
    cap_max: int
    num_basis: int
    prior_count: float
    cutoff_quantile: float
    cutoff_factor: float
    curvature_alpha: float
    curvature_beta: float
    curvature_center: float
    conductance_eps: float
    positive_overflow: str


class SLIntegerEncoder:
    def __init__(self, config: SLIntegerEncoderConfig) -> None:
        self.config = config
        self.missing_id = 0
        self.non_positive_to_id: dict[int, int] = {}

        self.cap_value: int = 1
        self.num_basis: int = max(1, int(config.num_basis))
        self.basis_matrix: np.ndarray = np.zeros((2, self.num_basis), dtype=np.float32)
        self._fitted = False

    @property
    def discrete_cardinality(self) -> int:
        return 1 + len(self.non_positive_to_id)

    def fit(self, series: pd.Series) -> "SLIntegerEncoder":
        ints = to_nullable_int(series)

        non_positive = ints[(ints.notna()) & (ints <= 0)].astype(int)
        unique_non_positive = sorted(non_positive.unique().tolist())
        self.non_positive_to_id = {
            value: index + 1 for index, value in enumerate(unique_non_positive)
        }

        positive = ints[(ints.notna()) & (ints > 0)].astype(int).to_numpy(dtype=np.int64)
        self.cap_value = self._fit_cap_value(positive)
        support_size = max(self.cap_value, 2)

        counts = np.zeros(support_size, dtype=np.float64)
        if positive.size > 0:
            clipped = np.clip(positive, 1, self.cap_value)
            indices = clipped - 1
            bincounts = np.bincount(indices, minlength=support_size)
            counts[: len(bincounts)] = bincounts

        ws = (counts + self.config.prior_count) / (
            np.sum(counts) + self.config.prior_count * support_size
        )

        curvature = CurvatureSpec(
            alpha=self.config.curvature_alpha,
            beta=self.config.curvature_beta,
            center=self.config.curvature_center,
        )
        cs = curvature.compute_weights(np.arange(support_size - 1, dtype=np.float64))
        cs = np.maximum(cs, self.config.conductance_eps)

        num_basis_eff = min(self.num_basis, support_size)
        _, eigenvectors = _compute_eigenfunctions(cs, ws, num_basis_eff)

        self.basis_matrix = np.zeros((support_size, self.num_basis), dtype=np.float32)
        self.basis_matrix[:, :num_basis_eff] = eigenvectors.astype(np.float32)

        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_fitted()

        ints = to_nullable_int(series)
        length = len(series)

        token_ids = np.full(length, self.missing_id, dtype=np.int64)
        positive_mask = np.zeros(length, dtype=bool)
        basis = np.zeros((length, self.num_basis), dtype=np.float32)

        valid_mask = ints.notna().to_numpy()
        if not valid_mask.any():
            return token_ids, positive_mask, basis

        # Non-positive values use discrete lookups.
        non_positive_mask = valid_mask & (ints.to_numpy(dtype=np.float64, na_value=np.nan) <= 0)
        if np.any(non_positive_mask):
            non_positive_idx = np.flatnonzero(non_positive_mask)
            non_positive_values = ints.iloc[non_positive_idx].astype(int).to_numpy()
            mapped = np.array(
                [self.non_positive_to_id.get(value, self.missing_id) for value in non_positive_values],
                dtype=np.int64,
            )
            token_ids[non_positive_idx] = mapped

        # Positive values use SL basis expansion.
        positive_candidate_mask = valid_mask & ~non_positive_mask
        if np.any(positive_candidate_mask):
            positive_idx = np.flatnonzero(positive_candidate_mask)
            positive_values = ints.iloc[positive_idx].astype(int).to_numpy()

            if self.config.positive_overflow == "missing":
                valid_positive = positive_values <= self.cap_value
                positive_idx = positive_idx[valid_positive]
                positive_values = positive_values[valid_positive]
                if positive_values.size == 0:
                    return token_ids, positive_mask, basis

            clipped = np.clip(positive_values, 1, self.cap_value)
            basis_indices = clipped - 1
            basis_rows = self.basis_matrix[basis_indices]

            basis[positive_idx] = basis_rows
            positive_mask[positive_idx] = True

        return token_ids, positive_mask, basis

    def fit_transform(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.fit(series).transform(series)

    def _fit_cap_value(self, positive_values: np.ndarray) -> int:
        if positive_values.size == 0:
            return 1

        observed_max = int(np.max(positive_values))
        if observed_max <= self.config.cap_max:
            return max(observed_max, 1)

        quantile_val = float(np.quantile(positive_values, self.config.cutoff_quantile))
        cutoff = int(min(quantile_val * self.config.cutoff_factor, self.config.cap_max))
        return max(cutoff, 1)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SLIntegerEncoder must be fitted before transform")
