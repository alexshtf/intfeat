from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import to_nullable_int


@dataclass
class BSplineIntegerEncoderConfig:
    cap_max: int
    cap_mode: str
    cap_quantile: float
    cap_quantile_factor: float
    input_map: str
    out_min: float
    out_max: float
    positive_overflow: str


class BSplineIntegerEncoder:
    def __init__(self, config: BSplineIntegerEncoderConfig) -> None:
        self.config = config
        self.missing_id = 0
        self.non_positive_to_id: dict[int, int] = {}
        self.large_id: int | None = None
        self.cap_value: int = 1
        self._fitted = False

    @property
    def discrete_cardinality(self) -> int:
        cardinality = 1 + len(self.non_positive_to_id)
        if self.config.positive_overflow == "large_token":
            cardinality += 1
        return cardinality

    def fit(self, series: pd.Series) -> "BSplineIntegerEncoder":
        ints = to_nullable_int(series)

        non_positive = ints[(ints.notna()) & (ints <= 0)].astype(int)
        unique_non_positive = sorted(non_positive.unique().tolist())
        self.non_positive_to_id = {
            value: index + 1 for index, value in enumerate(unique_non_positive)
        }
        self.large_id = (
            1 + len(self.non_positive_to_id)
            if self.config.positive_overflow == "large_token"
            else None
        )

        positive = ints[(ints.notna()) & (ints > 0)].astype(int).to_numpy(dtype=np.int64)
        self.cap_value = self._fit_cap_value(positive)

        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_fitted()

        ints = to_nullable_int(series)
        length = len(series)

        token_ids = np.full(length, self.missing_id, dtype=np.int64)
        positive_mask = np.zeros(length, dtype=bool)
        scalar = np.zeros(length, dtype=np.float32)

        valid_mask = ints.notna().to_numpy()
        if not valid_mask.any():
            return token_ids, positive_mask, scalar

        non_positive_mask = valid_mask & (ints.to_numpy(dtype=np.float64, na_value=np.nan) <= 0)
        if np.any(non_positive_mask):
            non_positive_idx = np.flatnonzero(non_positive_mask)
            non_positive_values = ints.iloc[non_positive_idx].astype(int).to_numpy()
            mapped = np.array(
                [self.non_positive_to_id.get(value, self.missing_id) for value in non_positive_values],
                dtype=np.int64,
            )
            token_ids[non_positive_idx] = mapped

        positive_candidate_mask = valid_mask & ~non_positive_mask
        if np.any(positive_candidate_mask):
            positive_idx = np.flatnonzero(positive_candidate_mask)
            positive_values = ints.iloc[positive_idx].astype(int).to_numpy()

            overflow_mode = str(self.config.positive_overflow or "clip_to_cap")
            if overflow_mode == "missing":
                valid_positive = positive_values <= self.cap_value
                positive_idx = positive_idx[valid_positive]
                positive_values = positive_values[valid_positive]
                if positive_values.size == 0:
                    return token_ids, positive_mask, scalar
            elif overflow_mode == "large_token":
                if self.large_id is None:
                    raise RuntimeError("large_id is not set; did you call fit()?")
                overflow = positive_values > self.cap_value
                if np.any(overflow):
                    overflow_idx = positive_idx[overflow]
                    token_ids[overflow_idx] = int(self.large_id)
                in_range = ~overflow
                positive_idx = positive_idx[in_range]
                positive_values = positive_values[in_range]
                if positive_values.size == 0:
                    return token_ids, positive_mask, scalar
            elif overflow_mode != "clip_to_cap":
                raise ValueError(
                    "Unsupported bspline positive_overflow: "
                    f"{overflow_mode!r} (expected 'clip_to_cap', 'missing', or 'large_token')"
                )

            clipped = np.clip(positive_values, 1, self.cap_value)
            mapped_values = self._map_positive(clipped)
            scalar[positive_idx] = mapped_values.astype(np.float32)
            positive_mask[positive_idx] = True

        return token_ids, positive_mask, scalar

    def fit_transform(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.fit(series).transform(series)

    def _fit_cap_value(self, positive_values: np.ndarray) -> int:
        if positive_values.size == 0:
            return 1

        observed_max = int(np.max(positive_values))
        if self.config.cap_mode == "max":
            return max(min(observed_max, self.config.cap_max), 1)

        if self.config.cap_mode != "quantile":
            raise ValueError(f"Unsupported BSpline cap_mode: {self.config.cap_mode!r}")

        quantile_val = float(np.quantile(positive_values, self.config.cap_quantile))
        cutoff = int(
            min(quantile_val * self.config.cap_quantile_factor, self.config.cap_max)
        )
        return max(cutoff, 1)

    def _map_positive(self, positive_values: np.ndarray) -> np.ndarray:
        cap = float(max(self.cap_value, 1))

        if self.config.input_map == "log1p_cap_to_unit":
            mapped_unit = np.log1p(positive_values) / np.log1p(cap)
        elif self.config.input_map == "linear_cap_to_unit":
            mapped_unit = positive_values / cap
        else:
            raise ValueError(
                f"Unsupported bspline input_map: {self.config.input_map!r}"
            )

        mapped_unit = np.clip(mapped_unit, 0.0, 1.0)
        return self.config.out_min + mapped_unit * (
            self.config.out_max - self.config.out_min
        )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("BSplineIntegerEncoder must be fitted before transform")
