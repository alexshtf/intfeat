from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import missing_mask


@dataclass
class CategoricalOrdinalEncoder:
    min_count: int = 10
    missing_token: str = "__MISSING__"
    infrequent_token: str = "__INFREQUENT__"

    def __post_init__(self) -> None:
        self.frequent_to_id: dict[str, int] = {}
        self.infrequent_values: set[str] = set()
        self.missing_id: int = 0
        self.infrequent_id: int = 1
        self._fitted: bool = False

    @property
    def cardinality(self) -> int:
        self._require_fitted()
        return 2 + len(self.frequent_to_id)

    def fit(self, series: pd.Series) -> "CategoricalOrdinalEncoder":
        mask_missing = missing_mask(series)
        values = series.astype("string")
        non_missing = values[~mask_missing].dropna()

        counts = non_missing.value_counts(dropna=False)
        frequent_values = sorted(counts[counts >= self.min_count].index.tolist())
        infrequent_values = counts[counts < self.min_count].index.tolist()

        self.frequent_to_id = {
            value: index + 2 for index, value in enumerate(frequent_values)
        }
        self.infrequent_values = set(str(value) for value in infrequent_values)
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        self._require_fitted()

        mask_missing = missing_mask(series)
        values = series.astype("string")

        encoded = np.full(len(series), self.missing_id, dtype=np.int64)
        non_missing_idx = np.flatnonzero(~mask_missing.to_numpy())
        if non_missing_idx.size == 0:
            return encoded

        non_missing_values = values.iloc[non_missing_idx]
        mapped = non_missing_values.map(self.frequent_to_id)

        is_frequent = ~mapped.isna().to_numpy()
        if np.any(is_frequent):
            frequent_indices = non_missing_idx[is_frequent]
            encoded[frequent_indices] = mapped[is_frequent].astype(np.int64).to_numpy()

        not_frequent_idx = non_missing_idx[~is_frequent]
        if not_frequent_idx.size == 0:
            return encoded

        if self.infrequent_values:
            not_frequent_values = non_missing_values.iloc[~is_frequent]
            is_infrequent = not_frequent_values.isin(self.infrequent_values).to_numpy()
            encoded[not_frequent_idx[is_infrequent]] = self.infrequent_id

        return encoded

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        return self.fit(series).transform(series)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Encoder must be fitted before transform")
