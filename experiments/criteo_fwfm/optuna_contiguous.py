from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import torch

from .config import resolve_config
from .model.fwfm import build_fwfm_model
from .preprocess import CriteoFeaturePreprocessor
from .schema import ALL_COLUMNS, CATEGORICAL_COLUMNS, INTEGER_COLUMNS, LABEL_COLUMN
from .train import (
    TorchEncodedSplit,
    encoded_split_to_torch,
    evaluate_split,
    set_global_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for Criteo FwFM variants")
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
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--sl-num-basis", type=int, default=10)
    parser.add_argument(
        "--bspline-knots",
        type=int,
        default=10,
        help="torchcurves: number of control points.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-8)
    parser.add_argument("--wd-max", type=float, default=1e-2)
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline_winner,sl_integer_basis,bspline_integer_basis",
        help="Comma-separated variants to run.",
    )
    parser.add_argument(
        "--tune-train-rows",
        type=int,
        default=0,
        help="If >0, use only first N training rows in each Optuna trial.",
    )
    parser.add_argument(
        "--tune-val-rows",
        type=int,
        default=0,
        help="If >0, use only first N validation rows in each Optuna trial.",
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
    print(f"Loaded rows [{start_row}, {start_row + n_rows}) in {dt:.1f}s")
    return df


def make_base_config(variant_config: Path, args: argparse.Namespace) -> dict[str, Any]:
    default_cfg = Path("experiments/criteo_fwfm/config/default.yaml")
    config = resolve_config(default_cfg, [variant_config], [])

    config["experiment"]["seed"] = int(args.seed)
    config["data"]["path"] = str(args.data_path)

    config["model"]["embedding_dim"] = int(args.embedding_dim)
    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_epochs"] = int(args.num_epochs)
    config["train"]["device"] = "cpu"
    config["train"]["early_stopping"]["patience"] = 1

    variant = str(config["experiment"]["variant"])
    if variant == "sl_integer_basis":
        config["model"]["integer"]["sl"]["num_basis"] = int(args.sl_num_basis)
    if variant == "bspline_integer_basis":
        config["model"]["integer"]["bspline"]["knots_config"] = int(args.bspline_knots)

    return config


def subset_torch_split(split: TorchEncodedSplit, n_rows: int) -> TorchEncodedSplit:
    if n_rows <= 0 or n_rows >= split.size:
        return split
    fields = {
        name: {key: tensor[:n_rows] for key, tensor in bundle.items()}
        for name, bundle in split.fields.items()
    }
    return TorchEncodedSplit(labels=split.labels[:n_rows], fields=fields)


def run_variant(
    *,
    variant_name: str,
    variant_config: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"\n===== Variant: {variant_name} =====")
    config = make_base_config(variant_config, args)

    split_rows = int(args.split_rows)
    train_start = 0
    val_start = split_rows
    test_start = split_rows * 2

    train_df = read_rows(args.data_path, train_start, split_rows)
    preprocessor = CriteoFeaturePreprocessor(config)
    train_encoded = preprocessor.fit_transform(train_df)
    del train_df
    gc.collect()

    val_df = read_rows(args.data_path, val_start, split_rows)
    val_encoded = preprocessor.transform(val_df)
    del val_df
    gc.collect()

    train_torch = encoded_split_to_torch(train_encoded, preprocessor.field_specs)
    val_torch = encoded_split_to_torch(val_encoded, preprocessor.field_specs)
    del train_encoded
    del val_encoded
    gc.collect()

    tune_train_rows = int(args.tune_train_rows)
    tune_val_rows = int(args.tune_val_rows)
    tune_train_torch = subset_torch_split(train_torch, tune_train_rows)
    tune_val_torch = subset_torch_split(val_torch, tune_val_rows)
    print(
        f"Tuning rows: train={tune_train_torch.size} val={tune_val_torch.size} "
        f"(full train={train_torch.size}, full val={val_torch.size})"
    )

    device = torch.device("cpu")

    def objective(trial: optuna.trial.Trial) -> float:
        trial_config = copy.deepcopy(config)
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True)
        trial_config["train"]["lr"] = float(lr)
        trial_config["train"]["weight_decay"] = float(weight_decay)

        seed = int(trial_config["experiment"]["seed"])
        set_global_seed(seed)
        model = build_fwfm_model(preprocessor.field_specs, trial_config).to(device)

        train_result = train_model(
            model,
            tune_train_torch,
            tune_val_torch,
            config=trial_config,
            device=device,
        )
        val_eval = evaluate_split(
            model,
            tune_val_torch,
            device=device,
            batch_size=int(trial_config["train"]["batch_size"]),
        )
        val_logloss = float(val_eval["metrics"]["logloss"])
        val_auc = float(val_eval["metrics"]["auc"])

        trial.set_user_attr("best_epoch", int(train_result["best_epoch"]))
        trial.set_user_attr("val_logloss", val_logloss)
        trial.set_user_attr("val_auc", val_auc)

        del model
        del train_result
        del val_eval
        gc.collect()
        return val_logloss

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(args.trials), gc_after_trial=True)

    best_lr = float(study.best_trial.params["lr"])
    best_wd = float(study.best_trial.params["weight_decay"])

    final_config = copy.deepcopy(config)
    final_config["train"]["lr"] = best_lr
    final_config["train"]["weight_decay"] = best_wd

    set_global_seed(int(final_config["experiment"]["seed"]))
    final_model = build_fwfm_model(preprocessor.field_specs, final_config).to(device)
    final_train_result = train_model(
        final_model,
        train_torch,
        val_torch,
        config=final_config,
        device=device,
    )

    final_val_eval = evaluate_split(
        final_model,
        val_torch,
        device=device,
        batch_size=int(final_config["train"]["batch_size"]),
    )

    test_df = read_rows(args.data_path, test_start, split_rows)
    test_encoded = preprocessor.transform(test_df)
    del test_df
    gc.collect()

    test_torch = encoded_split_to_torch(test_encoded, preprocessor.field_specs)
    del test_encoded
    gc.collect()

    final_test_eval = evaluate_split(
        final_model,
        test_torch,
        device=device,
        batch_size=int(final_config["train"]["batch_size"]),
    )

    result = {
        "variant": variant_name,
        "best_trial": int(study.best_trial.number),
        "best_params": {"lr": best_lr, "weight_decay": best_wd},
        "best_trial_val_logloss": float(study.best_trial.value),
        "final_retrain_best_epoch": int(final_train_result["best_epoch"]),
        "final_val_logloss": float(final_val_eval["metrics"]["logloss"]),
        "final_test_logloss": float(final_test_eval["metrics"]["logloss"]),
        "final_val_auc": float(final_val_eval["metrics"]["auc"]),
        "final_test_auc": float(final_test_eval["metrics"]["auc"]),
    }

    print(
        "RESULT",
        variant_name,
        "best_trial_val_logloss=",
        f"{result['best_trial_val_logloss']:.6f}",
        "final_val_logloss=",
        f"{result['final_val_logloss']:.6f}",
        "final_test_logloss=",
        f"{result['final_test_logloss']:.6f}",
    )

    del final_model
    del final_train_result
    del final_val_eval
    del final_test_eval
    del test_torch
    del train_torch
    del val_torch
    del preprocessor
    gc.collect()

    return result


def main() -> None:
    args = parse_args()

    variant_configs = {
        "baseline_winner": Path("experiments/criteo_fwfm/config/model_baseline.yaml"),
        "sl_integer_basis": Path("experiments/criteo_fwfm/config/model_sl.yaml"),
        "bspline_integer_basis": Path("experiments/criteo_fwfm/config/model_bspline.yaml"),
    }

    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not requested:
        raise ValueError("No variants specified")
    for variant in requested:
        if variant not in variant_configs:
            raise ValueError(f"Unknown variant: {variant!r}")

    all_results: list[dict[str, Any]] = []
    for variant_name in requested:
        result = run_variant(
            variant_name=variant_name,
            variant_config=variant_configs[variant_name],
            args=args,
        )
        all_results.append(result)

    payload = {
        "data_path": str(args.data_path),
        "split_rows": int(args.split_rows),
        "train_range": [0, int(args.split_rows)],
        "val_range": [int(args.split_rows), int(args.split_rows) * 2],
        "test_range": [int(args.split_rows) * 2, int(args.split_rows) * 3],
        "trials": int(args.trials),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "embedding_dim": int(args.embedding_dim),
        "sl_num_basis": int(args.sl_num_basis),
        "bspline_knots": int(args.bspline_knots),
        "variants": requested,
        "tune_train_rows": int(args.tune_train_rows),
        "tune_val_rows": int(args.tune_val_rows),
        "results": all_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {args.output_json}")


if __name__ == "__main__":
    main()

