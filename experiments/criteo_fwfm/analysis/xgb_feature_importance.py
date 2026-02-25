from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.criteo_fwfm.schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    INTEGER_COLUMNS,
    LABEL_COLUMN,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit a small XGBoost model on a contiguous slice of the Criteo dataset and "
            "rank columns by importance."
        )
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/alex/datasets/criteo_kaggle_challenge/train.txt"),
    )
    p.add_argument("--start-row", type=int, default=0)
    p.add_argument(
        "--n-rows",
        type=int,
        default=400_000,
        help="Number of rows to read and fit on.",
    )
    p.add_argument("--seed", type=int, default=0)

    # Keep this modest; we just want a stable-ish importance ranking.
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)

    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON report (params + importance tables).",
    )
    return p.parse_args()


def _dtype_map() -> dict[str, Any]:
    dtype: dict[str, Any] = {LABEL_COLUMN: "float32"}
    for column in INTEGER_COLUMNS:
        # Float is fine for trees and avoids nullable Int64 conversions.
        dtype[column] = "float32"
    for column in CATEGORICAL_COLUMNS:
        dtype[column] = "category"
    return dtype


def read_rows(path: Path, start_row: int, n_rows: int) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=ALL_COLUMNS,
        skiprows=int(start_row),
        nrows=int(n_rows),
        na_values=[""],
        keep_default_na=True,
        dtype=_dtype_map(),
    )
    if len(df) != n_rows:
        raise RuntimeError(
            f"Expected {n_rows} rows from start {start_row}, got {len(df)} rows"
        )
    dt = time.perf_counter() - t0
    print(f"Loaded rows [{start_row}, {start_row + n_rows}) in {dt:.1f}s", flush=True)
    return df


def _normalize_importance(raw: dict[str, float], *, feature_names: list[str]) -> dict[str, float]:
    out = {name: float(raw.get(name, 0.0)) for name in feature_names}
    total = float(sum(out.values()))
    if total <= 0:
        return {k: 0.0 for k in out}
    return {k: float(v) / total for k, v in out.items()}


def main() -> int:
    args = parse_args()

    # Imported lazily so that `uv run --with xgboost ...` is enough.
    import xgboost as xgb
    from sklearn.metrics import log_loss

    df = read_rows(args.data_path, args.start_row, args.n_rows)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.float32, copy=False)

    feature_names = [*INTEGER_COLUMNS, *CATEGORICAL_COLUMNS]
    X = df[feature_names]

    model = xgb.XGBClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        min_child_weight=float(args.min_child_weight),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_lambda=float(args.reg_lambda),
        reg_alpha=float(args.reg_alpha),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        enable_categorical=True,
        random_state=int(args.seed),
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    model.fit(X, y)
    fit_s = time.perf_counter() - t0

    pred = model.predict_proba(X)[:, 1]
    train_logloss = float(log_loss(y, pred, labels=[0.0, 1.0]))

    booster = model.get_booster()
    booster_feature_names = booster.feature_names or feature_names

    gain_raw = booster.get_score(importance_type="gain")
    weight_raw = booster.get_score(importance_type="weight")
    cover_raw = booster.get_score(importance_type="cover")

    gain_norm = _normalize_importance(gain_raw, feature_names=booster_feature_names)
    weight_norm = _normalize_importance(weight_raw, feature_names=booster_feature_names)
    cover_norm = _normalize_importance(cover_raw, feature_names=booster_feature_names)

    rows = []
    for name in booster_feature_names:
        rows.append(
            {
                "feature": str(name),
                "gain": float(gain_raw.get(name, 0.0)),
                "gain_norm": float(gain_norm.get(name, 0.0)),
                "weight": float(weight_raw.get(name, 0.0)),
                "weight_norm": float(weight_norm.get(name, 0.0)),
                "cover": float(cover_raw.get(name, 0.0)),
                "cover_norm": float(cover_norm.get(name, 0.0)),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: (r["gain"], r["weight"]), reverse=True)

    print()
    print(
        "XGBoost feature importance on Criteo",
        f"rows=[{args.start_row}, {args.start_row + args.n_rows})",
        f"fit_seconds={fit_s:.1f}",
        f"train_logloss={train_logloss:.6f}",
        flush=True,
    )
    print("Sorted by: total gain (descending)")
    print()
    for idx, r in enumerate(rows_sorted, start=1):
        print(
            f"{idx:>2}. {r['feature']:<3}  gain_norm={r['gain_norm']:.4f}  "
            f"weight_norm={r['weight_norm']:.4f}  cover_norm={r['cover_norm']:.4f}"
        )

    if args.output_json is not None:
        payload = {
            "data_path": str(args.data_path),
            "start_row": int(args.start_row),
            "n_rows": int(args.n_rows),
            "train_logloss": train_logloss,
            "fit_seconds": float(fit_s),
            "xgb_params": {
                "n_estimators": int(args.n_estimators),
                "learning_rate": float(args.learning_rate),
                "max_depth": int(args.max_depth),
                "min_child_weight": float(args.min_child_weight),
                "subsample": float(args.subsample),
                "colsample_bytree": float(args.colsample_bytree),
                "reg_lambda": float(args.reg_lambda),
                "reg_alpha": float(args.reg_alpha),
                "tree_method": "hist",
                "enable_categorical": True,
                "seed": int(args.seed),
            },
            "importance": rows_sorted,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Wrote: {args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

