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
from optuna.trial import TrialState

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
    parser = argparse.ArgumentParser(
        description=(
            "Resumable Optuna tuning for Criteo FwFM variants on contiguous head splits "
            "(train block, then val block, then test block). Tunes lr and weight_decay. "
            "For SL, can optionally fix the u_exp_valley conductance parameters."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/alex/datasets/criteo_kaggle_challenge/train.txt"),
    )
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--val-rows", type=int, required=True)
    parser.add_argument("--test-rows", type=int, required=True)

    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Target number of completed trials per variant.",
    )
    parser.add_argument(
        "--max-new-trials",
        type=int,
        default=0,
        help="If >0, run at most this many additional trials per variant in this invocation.",
    )
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/criteo_fwfm/optuna_head_contiguous.checkpoints"),
    )
    parser.add_argument(
        "--study-prefix",
        type=str,
        default="criteo_optuna_head",
        help="Per-variant study name is '{prefix}_{variant}'.",
    )
    parser.add_argument(
        "--storage-url",
        type=str,
        default="sqlite:///artifacts/criteo_fwfm/optuna_head_contiguous.sqlite3",
    )

    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-8)
    parser.add_argument("--wd-max", type=float, default=1e-2)

    parser.add_argument(
        "--variants",
        type=str,
        default="baseline_winner,bspline_integer_basis,sl_integer_basis",
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

    # SL-only: fixed conductance for transfer experiments.
    parser.add_argument(
        "--sl-fixed-u-exp-valley",
        action="store_true",
        help="If set, force SL conductance family to 'u_exp_valley' with fixed params.",
    )
    parser.add_argument("--sl-u0", type=float, default=0.0)
    parser.add_argument("--sl-left-slope", type=float, default=1.0)
    parser.add_argument("--sl-right-slope", type=float, default=1.0)

    return parser.parse_args()


def _dtype_map() -> dict[str, Any]:
    dtype: dict[str, Any] = {LABEL_COLUMN: "float32"}
    for column in INTEGER_COLUMNS:
        dtype[column] = "Int64"
    for column in CATEGORICAL_COLUMNS:
        dtype[column] = "category"
    return dtype


def read_rows(path: Path, start_row: int, n_rows: int) -> pd.DataFrame:
    if n_rows <= 0:
        return pd.DataFrame(columns=ALL_COLUMNS)
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


def make_base_config(
    *,
    variant_config: Path,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    default_cfg = Path("experiments/criteo_fwfm/config/default.yaml")
    config = resolve_config(default_cfg, [variant_config], [])

    config["experiment"]["seed"] = int(args.seed)
    config["data"]["path"] = str(args.data_path)

    config["model"]["embedding_dim"] = int(args.embedding_dim)
    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_epochs"] = int(args.num_epochs)
    config["train"]["device"] = "cpu"
    config["train"]["early_stopping"]["patience"] = 1

    if variant == "sl_integer_basis":
        config["model"]["integer"]["sl"]["num_basis"] = int(args.sl_num_basis)
        if bool(args.sl_fixed_u_exp_valley):
            config["model"]["integer"]["sl"]["conductance"]["family"] = "u_exp_valley"
            config["model"]["integer"]["sl"]["conductance"]["u_exp_valley"]["u0"] = float(
                args.sl_u0
            )
            config["model"]["integer"]["sl"]["conductance"]["u_exp_valley"][
                "left_slope"
            ] = float(args.sl_left_slope)
            config["model"]["integer"]["sl"]["conductance"]["u_exp_valley"][
                "right_slope"
            ] = float(args.sl_right_slope)

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


def _completed_count(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)


def _checkpoint_payload(
    *,
    variant: str,
    study: optuna.Study,
    args: argparse.Namespace,
    hpo_seconds: float | None,
) -> dict[str, Any]:
    completed = _completed_count(study)
    failed = sum(1 for t in study.trials if t.state == TrialState.FAIL)
    running = sum(1 for t in study.trials if t.state == TrialState.RUNNING)

    payload: dict[str, Any] = {
        "variant": variant,
        "study_name": study.study_name,
        "storage_url": args.storage_url,
        "target_trials": int(args.trials),
        "completed_trials": int(completed),
        "failed_trials": int(failed),
        "running_trials": int(running),
        "total_trials_seen": int(len(study.trials)),
        "hpo_seconds": None if hpo_seconds is None else float(hpo_seconds),
        "timestamp": time.time(),
    }
    if completed > 0:
        payload["best_trial"] = int(study.best_trial.number)
        payload["best_trial_val_logloss"] = float(study.best_trial.value)
        payload["best_params"] = {k: float(v) for k, v in study.best_trial.params.items()}
    return payload


def _ensure_sqlite_parent(storage_url: str) -> None:
    abs_prefix = "sqlite:////"
    rel_prefix = "sqlite:///"

    if storage_url.startswith(abs_prefix):
        path = Path("/" + storage_url[len(abs_prefix) :])
    elif storage_url.startswith(rel_prefix) and not storage_url.startswith(abs_prefix):
        path_part = storage_url[len(rel_prefix) :]
        if not path_part or path_part.startswith(":memory:"):
            return
        path = Path(path_part)
    else:
        return

    path.parent.mkdir(parents=True, exist_ok=True)


def run_variant(
    *,
    variant_name: str,
    variant_config: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"\n===== Variant: {variant_name} =====")
    base_config = make_base_config(
        variant_config=variant_config, variant=variant_name, args=args
    )

    train_rows = int(args.train_rows)
    val_rows = int(args.val_rows)
    test_rows = int(args.test_rows)

    train_start = 0
    val_start = train_rows
    test_start = train_rows + val_rows

    train_df = read_rows(args.data_path, train_start, train_rows)
    preprocessor = CriteoFeaturePreprocessor(base_config)
    train_encoded = preprocessor.fit_transform(train_df)
    del train_df
    gc.collect()

    val_df = read_rows(args.data_path, val_start, val_rows)
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

    _ensure_sqlite_parent(str(args.storage_url))
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study_name = f"{args.study_prefix}_{variant_name}"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=args.storage_url,
        load_if_exists=True,
    )

    completed_before = _completed_count(study)
    remaining = max(int(args.trials) - completed_before, 0)
    if int(args.max_new_trials) > 0:
        remaining = min(remaining, int(args.max_new_trials))

    print(
        f"Study '{study_name}': completed_before={completed_before}, "
        f"target={args.trials}, running_now={remaining}"
    )

    device = torch.device("cpu")

    def objective(trial: optuna.trial.Trial) -> float:
        trial_config = copy.deepcopy(base_config)
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay", args.wd_min, args.wd_max, log=True
        )
        trial_config["train"]["lr"] = float(lr)
        trial_config["train"]["weight_decay"] = float(weight_decay)

        set_global_seed(int(trial_config["experiment"]["seed"]))
        model = build_fwfm_model(preprocessor.field_specs, trial_config).to(device)

        train_t0 = time.perf_counter()
        train_result = train_model(
            model, tune_train_torch, tune_val_torch, config=trial_config, device=device
        )
        train_s = time.perf_counter() - train_t0

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
        trial.set_user_attr("train_seconds", round(train_s, 6))

        del model
        del train_result
        del val_eval
        gc.collect()
        return val_logloss

    checkpoint_path = Path(args.checkpoint_dir) / f"{study_name}.checkpoint.json"

    def checkpoint_callback(
        study_obj: optuna.Study,
        _trial: optuna.trial.FrozenTrial,
    ) -> None:
        payload = _checkpoint_payload(
            variant=variant_name,
            study=study_obj,
            args=args,
            hpo_seconds=(time.perf_counter() - t_hpo0),
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    t_hpo0 = time.perf_counter()
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            callbacks=[checkpoint_callback],
        )
    hpo_seconds = time.perf_counter() - t_hpo0

    if _completed_count(study) == 0:
        raise RuntimeError("No completed trials in study; cannot finalize")

    best_lr = float(study.best_trial.params["lr"])
    best_wd = float(study.best_trial.params["weight_decay"])

    final_config = copy.deepcopy(base_config)
    final_config["train"]["lr"] = best_lr
    final_config["train"]["weight_decay"] = best_wd

    set_global_seed(int(final_config["experiment"]["seed"]))
    final_model = build_fwfm_model(preprocessor.field_specs, final_config).to(device)
    final_train_result = train_model(
        final_model, train_torch, val_torch, config=final_config, device=device
    )

    final_val_eval = evaluate_split(
        final_model,
        val_torch,
        device=device,
        batch_size=int(final_config["train"]["batch_size"]),
    )

    test_df = read_rows(args.data_path, test_start, test_rows)
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

    result: dict[str, Any] = {
        "variant": variant_name,
        "study_name": study_name,
        "storage_url": str(args.storage_url),
        "target_trials": int(args.trials),
        "completed_trials": int(_completed_count(study)),
        "best_trial": int(study.best_trial.number),
        "best_params": {"lr": best_lr, "weight_decay": best_wd},
        "best_trial_val_logloss": float(study.best_trial.value),
        "hpo_seconds_this_invocation": float(hpo_seconds),
        "final_retrain_best_epoch": int(final_train_result["best_epoch"]),
        "final_val_logloss": float(final_val_eval["metrics"]["logloss"]),
        "final_test_logloss": float(final_test_eval["metrics"]["logloss"]),
        "final_val_auc": float(final_val_eval["metrics"]["auc"]),
        "final_test_auc": float(final_test_eval["metrics"]["auc"]),
        "checkpoint_json": str(checkpoint_path),
    }

    if variant_name == "sl_integer_basis" and bool(args.sl_fixed_u_exp_valley):
        result["sl_conductance_family"] = "u_exp_valley"
        result["sl_u0"] = float(args.sl_u0)
        result["sl_left_slope"] = float(args.sl_left_slope)
        result["sl_right_slope"] = float(args.sl_right_slope)
        result["sl_num_basis"] = int(args.sl_num_basis)

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

    train_rows = int(args.train_rows)
    val_rows = int(args.val_rows)
    test_rows = int(args.test_rows)
    if train_rows <= 0 or val_rows <= 0 or test_rows <= 0:
        raise ValueError("train_rows, val_rows, and test_rows must all be > 0")

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

    payload: dict[str, Any] = {
        "data_path": str(args.data_path),
        "train_range": [0, train_rows],
        "val_range": [train_rows, train_rows + val_rows],
        "test_range": [train_rows + val_rows, train_rows + val_rows + test_rows],
        "trials": int(args.trials),
        "max_new_trials": int(args.max_new_trials),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "embedding_dim": int(args.embedding_dim),
        "sl_num_basis": int(args.sl_num_basis),
        "bspline_knots": int(args.bspline_knots),
        "variants": requested,
        "tune_train_rows": int(args.tune_train_rows),
        "tune_val_rows": int(args.tune_val_rows),
        "study_prefix": str(args.study_prefix),
        "storage_url": str(args.storage_url),
        "sl_fixed_u_exp_valley": bool(args.sl_fixed_u_exp_valley),
        "sl_u0": float(args.sl_u0),
        "sl_left_slope": float(args.sl_left_slope),
        "sl_right_slope": float(args.sl_right_slope),
        "results": all_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {args.output_json}")


if __name__ == "__main__":
    main()

