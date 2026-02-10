from __future__ import annotations

import pandas as pd


def missing_mask(series: pd.Series) -> pd.Series:
    as_obj = series.astype("object")
    return as_obj.isna() | (as_obj.astype("string").str.len().fillna(0) == 0)


def to_nullable_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64")
