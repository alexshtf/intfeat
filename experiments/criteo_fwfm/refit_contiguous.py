from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import resolve_config
from .interactions import save_interaction_matrix_json, save_interaction_matrix_npz
from .model.fwfm import build_fwfm_model
from .preprocess import CriteoFeaturePreprocessor
from .schema import ALL_COLUMNS, CATEGORICAL_COLUMNS, INTEGER_COLUMNS, LABEL_COLUMN
from .train import encoded_split_to_torch, evaluate_split, set_global_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit a single Criteo FwFM variant on a contiguous train/val/test split and "
            "report val/test metrics for a fixed (lr, weight_decay) configuration."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/alex/datasets/criteo_kaggle_challenge/train.txt"),
    )
    parser.add_argument(
        "--split-rows",
        type=int,
        required=True,
        help="Size of each contiguous block (train/val/test).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["baseline_winner", "sl_integer_basis", "bspline_integer_basis"],
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--sl-num-basis", type=int, default=10)
    parser.add_argument("--bspline-knots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--interaction-heatmap-out",
        type=Path,
        default=None,
        help=(
            "If set, write a terminal-rendered heatmap of abs(field interaction matrix) to this file "
            "(and print it to stdout)."
        ),
    )
    parser.add_argument(
        "--interaction-matrix-out",
        type=Path,
        default=None,
        help=(
            "If set, save the raw effective interaction matrix R (symmetric, zero diagonal) "
            "to this path. Use .npz (recommended) or .json."
        ),
    )
    return parser.parse_args()


def _dtype_map() -> dict[str, Any]:
    dtype: dict[str, Any] = {LABEL_COLUMN: "float32"}
    for column in INTEGER_COLUMNS:
        dtype[column] = "Int64"
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
        skiprows=start_row,
        nrows=n_rows,
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


def make_config(variant_config: Path, args: argparse.Namespace) -> dict[str, Any]:
    default_cfg = Path("experiments/criteo_fwfm/config/default.yaml")
    config = resolve_config(default_cfg, [variant_config], [])

    config["experiment"]["seed"] = int(args.seed)
    config["data"]["path"] = str(args.data_path)

    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_epochs"] = int(args.num_epochs)
    config["train"]["device"] = "cpu"
    config["train"]["early_stopping"]["patience"] = 1
    config["train"]["lr"] = float(args.lr)
    config["train"]["weight_decay"] = float(args.weight_decay)

    config["model"]["embedding_dim"] = int(args.embedding_dim)

    variant = str(config["experiment"]["variant"])
    if variant == "sl_integer_basis":
        config["model"]["integer"]["sl"]["num_basis"] = int(args.sl_num_basis)
    if variant == "bspline_integer_basis":
        config["model"]["integer"]["bspline"]["knots_config"] = int(args.bspline_knots)

    return config


def _render_interaction_heatmap(
    *,
    abs_interactions: np.ndarray,
    field_order: list[str],
) -> str:
    if abs_interactions.ndim != 2 or abs_interactions.shape[0] != abs_interactions.shape[1]:
        raise ValueError(
            f"Expected square interaction matrix, got shape={abs_interactions.shape}"
        )

    name_to_idx = {name: idx for idx, name in enumerate(field_order)}
    integer_fields = [name for name in field_order if name.startswith("I")]
    categorical_fields = [name for name in field_order if name.startswith("C")]
    other_fields = [
        name
        for name in field_order
        if name not in set(integer_fields) and name not in set(categorical_fields)
    ]

    perm_names = integer_fields + categorical_fields + other_fields
    perm = np.array([name_to_idx[name] for name in perm_names], dtype=np.int64)
    m = abs_interactions[np.ix_(perm, perm)]

    i_count = len(integer_fields)
    c_count = len(categorical_fields)

    mask = ~np.eye(m.shape[0], dtype=bool)
    values = m[mask]
    if values.size == 0:
        t50 = t90 = t99 = 0.0
    else:
        t50 = float(np.quantile(values, 0.50))
        t90 = float(np.quantile(values, 0.90))
        t99 = float(np.quantile(values, 0.99))

    def shade(v: float) -> str:
        # Unicode "block elements": none/light/medium/full shade.
        if v < t50:
            return " "
        if v < t90:
            return "░"
        if v < t99:
            return "▒"
        return "█"

    cols_left = " ".join(integer_fields) if integer_fields else "(none)"
    cols_right = " ".join(categorical_fields) if categorical_fields else "(none)"

    lines = [
        "abs(R) heatmap (FwFM field interaction matrix)",
        f"fields: I*={i_count} | C*={c_count} | other={len(other_fields)}",
        (
            "legend (|R| quantiles): "
            f"' '<p50={t50:.4g}  '░'<p90={t90:.4g}  '▒'<p99={t99:.4g}  '█'>=p99"
        ),
        f"cols: {cols_left} | {cols_right}",
    ]

    # Render aligned column labels above the matrix using vertical text.
    labels = list(integer_fields + categorical_fields + other_fields)
    if i_count > 0 and c_count > 0:
        labels.insert(i_count, "|")

    max_label_len = max((len(label) for label in labels), default=0)
    prefix = " " * 4  # aligns with f"{row_name:>3} " in matrix rows
    for pos in range(max_label_len):
        header_cells = []
        for label in labels:
            padded = label.ljust(max_label_len)
            header_cells.append(padded[pos])
        lines.append(prefix + "".join(header_cells))

    for row_idx, row_name in enumerate(perm_names):
        row_cells = "".join(shade(float(v)) for v in m[row_idx])
        if i_count > 0 and c_count > 0:
            row_cells = row_cells[:i_count] + "|" + row_cells[i_count : i_count + c_count]
        label = f"{row_name:>3}"
        lines.append(f"{label} {row_cells}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    variant_configs = {
        "baseline_winner": Path("experiments/criteo_fwfm/config/model_baseline.yaml"),
        "sl_integer_basis": Path("experiments/criteo_fwfm/config/model_sl.yaml"),
        "bspline_integer_basis": Path("experiments/criteo_fwfm/config/model_bspline.yaml"),
    }
    variant = str(args.variant)
    variant_config = variant_configs[variant]

    config = make_config(variant_config, args)
    split_rows = int(args.split_rows)

    train_start = 0
    val_start = split_rows
    test_start = split_rows * 2

    train_df = read_rows(args.data_path, train_start, split_rows)
    val_df = read_rows(args.data_path, val_start, split_rows)
    test_df = read_rows(args.data_path, test_start, split_rows)

    prep_t0 = time.perf_counter()
    preprocessor = CriteoFeaturePreprocessor(config)
    train_encoded = preprocessor.fit_transform(train_df)
    val_encoded = preprocessor.transform(val_df)
    test_encoded = preprocessor.transform(test_df)
    preprocess_seconds = time.perf_counter() - prep_t0

    train_torch = encoded_split_to_torch(train_encoded, preprocessor.field_specs)
    val_torch = encoded_split_to_torch(val_encoded, preprocessor.field_specs)
    test_torch = encoded_split_to_torch(test_encoded, preprocessor.field_specs)

    device = torch.device("cpu")

    set_global_seed(int(config["experiment"]["seed"]))
    model = build_fwfm_model(preprocessor.field_specs, config).to(device)

    train_t0 = time.perf_counter()
    train_result = train_model(model, train_torch, val_torch, config=config, device=device)
    train_seconds = time.perf_counter() - train_t0

    val_eval = evaluate_split(
        model,
        val_torch,
        device=device,
        batch_size=int(config["train"]["batch_size"]),
    )
    test_eval = evaluate_split(
        model,
        test_torch,
        device=device,
        batch_size=int(config["train"]["batch_size"]),
    )

    heatmap_out = args.interaction_heatmap_out
    matrix_out = args.interaction_matrix_out
    raw_r = None

    if heatmap_out is not None or matrix_out is not None:
        raw_r = model._interaction_matrix().detach().cpu().numpy()

    if matrix_out is not None:
        if matrix_out.suffix.lower() == ".json":
            save_interaction_matrix_json(model=model, out_path=matrix_out)
        else:
            # Default to compact binary for easy downstream analysis.
            save_interaction_matrix_npz(model=model, out_path=matrix_out)
        print(f"Wrote interaction matrix: {matrix_out}", flush=True)

    if heatmap_out is not None:
        if raw_r is None:
            raw_r = model._interaction_matrix().detach().cpu().numpy()
        abs_r = np.abs(raw_r)
        heatmap_text = _render_interaction_heatmap(
            abs_interactions=abs_r,
            field_order=list(model.field_order),
        )
        heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        heatmap_out.write_text(heatmap_text, encoding="utf-8")
        print(heatmap_text, flush=True)
        print(f"Wrote heatmap: {heatmap_out}", flush=True)

    payload: dict[str, Any] = {
        "variant": variant,
        "data_path": str(args.data_path),
        "split_rows": split_rows,
        "train_range": [train_start, train_start + split_rows],
        "val_range": [val_start, val_start + split_rows],
        "test_range": [test_start, test_start + split_rows],
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "embedding_dim": int(args.embedding_dim),
        "sl_num_basis": int(args.sl_num_basis),
        "bspline_knots": int(args.bspline_knots),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "best_epoch": int(train_result["best_epoch"]),
        "preprocess_seconds": float(preprocess_seconds),
        "train_seconds": float(train_seconds),
        "val": val_eval["metrics"],
        "test": test_eval["metrics"],
    }
    if matrix_out is not None:
        payload["interaction_matrix_path"] = str(matrix_out)
    if heatmap_out is not None:
        payload["interaction_heatmap_path"] = str(heatmap_out)

    print(
        "RESULT",
        variant,
        "val_logloss=",
        f"{payload['val']['logloss']:.6f}",
        "test_logloss=",
        f"{payload['test']['logloss']:.6f}",
        flush=True,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote summary: {args.output_json}", flush=True)

    del test_eval
    del val_eval
    del train_result
    del model
    del train_torch
    del val_torch
    del test_torch
    del train_encoded
    del val_encoded
    del test_encoded
    del preprocessor
    gc.collect()


if __name__ == "__main__":
    main()
