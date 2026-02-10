from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

FieldKind = Literal["discrete", "sl_integer", "bspline_integer"]


@dataclass
class FieldArrayData:
    token_ids: np.ndarray
    positive_mask: np.ndarray
    basis: np.ndarray | None = None
    scalar: np.ndarray | None = None


@dataclass
class EncodedSplit:
    labels: np.ndarray
    fields: dict[str, FieldArrayData]


@dataclass
class FieldSpec:
    name: str
    kind: FieldKind
    discrete_cardinality: int
    num_basis: int = 0

    bspline_degree: int = 0
    bspline_knots: int = 0
    bspline_normalize_fn: str | None = None
    bspline_normalization_scale: float = 1.0


@dataclass
class FieldStat:
    name: str
    raw_type: Literal["integer", "categorical"]
    routed_type: Literal["integer", "categorical"]
    unique_non_missing_train: int
    extra: dict[str, object] = field(default_factory=dict)
