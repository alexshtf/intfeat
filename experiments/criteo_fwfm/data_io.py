from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get_config_value
from .schema import ALL_COLUMNS, CATEGORICAL_COLUMNS, INTEGER_COLUMNS, LABEL_COLUMN

LOGGER = logging.getLogger(__name__)


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _normalize_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for column in INTEGER_COLUMNS + CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("object")
    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").astype(
            np.float32
        )
    return df


def _read_head_rows(path: Path, *, n_rows: int | None, skip_rows: int) -> pd.DataFrame:
    if n_rows is not None and n_rows <= 0:
        return pd.DataFrame(columns=ALL_COLUMNS)

    dtype_map = {column: "string" for column in INTEGER_COLUMNS + CATEGORICAL_COLUMNS}
    dtype_map[LABEL_COLUMN] = "float32"

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=ALL_COLUMNS,
        nrows=n_rows,
        skiprows=skip_rows,
        na_values=[""],
        keep_default_na=True,
        dtype=dtype_map,
    )
    return _normalize_feature_dtypes(df)


def _read_tail_rows(path: Path, *, n_rows: int) -> pd.DataFrame:
    if n_rows <= 0:
        return pd.DataFrame(columns=ALL_COLUMNS)

    tail = deque(maxlen=n_rows)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tail.append(line.rstrip("\n"))

    rows = []
    for line in tail:
        parts = line.split("\t")
        if len(parts) < len(ALL_COLUMNS):
            parts.extend([""] * (len(ALL_COLUMNS) - len(parts)))
        rows.append(parts[: len(ALL_COLUMNS)])

    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    df = df.replace("", np.nan)
    return _normalize_feature_dtypes(df)


def _validate_rows(train_rows: int, val_rows: int, test_rows: int) -> None:
    if train_rows < 0 or val_rows < 0 or test_rows < 0:
        raise ValueError("Split row counts must be non-negative")


def _load_tail_holdout(path: Path, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    train_rows = int(get_config_value(config, "data.split.train_rows", required=True))
    val_rows = int(get_config_value(config, "data.split.val_rows", default=0))
    test_rows = int(get_config_value(config, "data.split.test_rows", default=0))
    _validate_rows(train_rows, val_rows, test_rows)

    total_rows = count_rows(path)
    LOGGER.info("Detected %s rows in %s", total_rows, path)

    head_rows = train_rows + val_rows
    if head_rows > total_rows:
        raise ValueError(
            f"train_rows + val_rows ({head_rows}) exceeds total rows ({total_rows})"
        )

    train_val_df = _read_head_rows(path, n_rows=head_rows, skip_rows=0)
    train_df = train_val_df.iloc[:train_rows].reset_index(drop=True)
    val_df = train_val_df.iloc[train_rows : train_rows + val_rows].reset_index(drop=True)

    test_df = _read_tail_rows(path, n_rows=test_rows)

    if test_rows > 0 and total_rows < (head_rows + test_rows):
        LOGGER.warning(
            "Tail test split likely overlaps train/val ranges (total_rows=%s, head_rows=%s, test_rows=%s)",
            total_rows,
            head_rows,
            test_rows,
        )

    return {"train": train_df, "val": val_df, "test": test_df.reset_index(drop=True)}


def _load_random_split(path: Path, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    train_rows = int(get_config_value(config, "data.split.train_rows", required=True))
    val_rows = int(get_config_value(config, "data.split.val_rows", default=0))
    test_rows = int(get_config_value(config, "data.split.test_rows", default=0))
    _validate_rows(train_rows, val_rows, test_rows)

    max_rows = get_config_value(config, "data.max_rows", default=None)
    requested_rows = train_rows + val_rows + test_rows
    if max_rows is None:
        max_rows = requested_rows if requested_rows > 0 else None
    elif max_rows is not None:
        max_rows = int(max_rows)

    df = _read_head_rows(path, n_rows=max_rows, skip_rows=0)
    if requested_rows > len(df):
        raise ValueError(
            f"Requested split rows ({requested_rows}) exceed loaded rows ({len(df)})"
        )

    seed = int(get_config_value(config, "data.split.random_seed", default=0))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    train_idx = indices[:train_rows]
    val_idx = indices[train_rows : train_rows + val_rows]
    test_idx = indices[train_rows + val_rows : train_rows + val_rows + test_rows]

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def load_criteo_splits(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    path = Path(get_config_value(config, "data.path", required=True)).expanduser()
    split_mode = str(get_config_value(config, "data.split.mode", default="tail_holdout"))

    if split_mode == "tail_holdout":
        splits = _load_tail_holdout(path, config)
    elif split_mode == "random":
        splits = _load_random_split(path, config)
    else:
        raise ValueError(f"Unsupported split mode: {split_mode!r}")

    LOGGER.info(
        "Split sizes -> train=%s val=%s test=%s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits
