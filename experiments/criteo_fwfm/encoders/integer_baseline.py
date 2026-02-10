from __future__ import annotations

import math

import pandas as pd

from .categorical import CategoricalOrdinalEncoder
from .common import to_nullable_int


def winner_integer_token(value: int, *, negative_prefix: str, logsq_prefix: str) -> str:
    if value < 1:
        return f"{negative_prefix}{value}"
    bucket = int(math.floor(math.log(value) ** 2))
    return f"{logsq_prefix}{bucket}"


class WinnerIntegerEncoder:
    def __init__(
        self,
        *,
        min_count: int,
        missing_token: str,
        infrequent_token: str,
        negative_prefix: str,
        logsq_prefix: str,
    ) -> None:
        self.negative_prefix = negative_prefix
        self.logsq_prefix = logsq_prefix
        self.cat_encoder = CategoricalOrdinalEncoder(
            min_count=min_count,
            missing_token=missing_token,
            infrequent_token=infrequent_token,
        )

    @property
    def cardinality(self) -> int:
        return self.cat_encoder.cardinality

    def fit(self, series: pd.Series) -> "WinnerIntegerEncoder":
        token_series = self._to_token_series(series)
        self.cat_encoder.fit(token_series)
        return self

    def transform(self, series: pd.Series):
        token_series = self._to_token_series(series)
        return self.cat_encoder.transform(token_series)

    def fit_transform(self, series: pd.Series):
        return self.fit(series).transform(series)

    def _to_token_series(self, series: pd.Series) -> pd.Series:
        ints = to_nullable_int(series)
        out = pd.Series([None] * len(series), index=series.index, dtype="object")

        valid_mask = ints.notna().to_numpy()
        if valid_mask.any():
            values = ints[valid_mask].astype(int).to_list()
            tokens = [
                winner_integer_token(
                    value,
                    negative_prefix=self.negative_prefix,
                    logsq_prefix=self.logsq_prefix,
                )
                for value in values
            ]
            out.iloc[valid_mask] = tokens

        return out
