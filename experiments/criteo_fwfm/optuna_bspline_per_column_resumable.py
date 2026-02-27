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
from .interactions import save_interaction_matrix_npz
from .model.fwfm import build_fwfm_model
from .preprocess import CriteoFeaturePreprocessor
from .schema import ALL_COLUMNS, CATEGORICAL_COLUMNS, INTEGER_COLUMNS, LABEL_COLUMN
from .train import encoded_split_to_torch, evaluate_split, set_global_seed, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Resumable Optuna tuning for a B-spline-only Criteo FwFM model, with optional "
            "per-column B-spline encoder overrides. This is meant for fair comparisons "
            "against SL-per-column experiments."
        )
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/alex/datasets/criteo_kaggle_challenge/train.txt"),
    )
    p.add_argument("--split-rows", type=int, default=400_000)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--max-new-trials", type=int, default=0)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--checkpoint-json", type=Path, required=True)
    p.add_argument("--study-name", type=str, required=True)
    p.add_argument("--storage-url", type=str, required=True)
    p.add_argument(
        "--bspline-config",
        type=Path,
        default=Path("experiments/criteo_fwfm/config/model_bspline_quantile_large_token.yaml"),
        help="Base YAML config for bspline_integer_basis variant.",
    )

    # Fixed bspline curve params (model side; not tuned here).
    p.add_argument("--bspline-knots", type=int, default=10)

    # Column to tune.
    p.add_argument("--column", type=str, default="I6")

    # Optimizer search space.
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--lr-max", type=float, default=1e-2)
    p.add_argument("--wd-min", type=float, default=1e-8)
    p.add_argument("--wd-max", type=float, default=1e-2)

    # Per-column bspline encoder search space.
    p.add_argument(
        "--cap-quantile-min",
        type=float,
        default=0.95,
        help="Lower bound for per-column cap_quantile (quantile cap).",
    )
    p.add_argument(
        "--cap-quantile-max",
        type=float,
        default=0.9995,
        help="Upper bound for per-column cap_quantile (quantile cap).",
    )
    p.add_argument(
        "--overflow-modes",
        type=str,
        default="large_token,clip_to_cap",
        help="Comma-separated overflow modes to consider for the tuned column.",
    )
    return p.parse_args()


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


def _completed_count(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)


def _checkpoint_payload(
    *,
    study: optuna.Study,
    args: argparse.Namespace,
    hpo_seconds: float | None,
) -> dict[str, Any]:
    completed = _completed_count(study)
    failed = sum(1 for t in study.trials if t.state == TrialState.FAIL)
    running = sum(1 for t in study.trials if t.state == TrialState.RUNNING)

    payload: dict[str, Any] = {
        "study_name": args.study_name,
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
        payload["best_params"] = {k: study.best_trial.params[k] for k in study.best_trial.params}
    return payload


def make_base_config(args: argparse.Namespace) -> dict[str, Any]:
    default_cfg = Path("experiments/criteo_fwfm/config/default.yaml")
    config = resolve_config(default_cfg, [Path(args.bspline_config)], [])

    if str(config["experiment"]["variant"]) != "bspline_integer_basis":
        raise ValueError(
            f"Expected bspline_integer_basis variant, got {config['experiment']['variant']!r}"
        )

    config["experiment"]["seed"] = int(args.seed)
    config["data"]["path"] = str(args.data_path)

    config["model"]["embedding_dim"] = int(args.embedding_dim)
    config["model"]["integer"]["bspline"]["knots_config"] = int(args.bspline_knots)

    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_epochs"] = int(args.num_epochs)
    config["train"]["device"] = "cpu"
    config["train"]["early_stopping"]["patience"] = 1

    return config


def apply_trial_params(
    config: dict[str, Any],
    *,
    column: str,
    params: dict[str, Any],
) -> None:
    config["train"]["lr"] = float(params["lr"])
    config["train"]["weight_decay"] = float(params["weight_decay"])

    bs_cfg = config.setdefault("model", {}).setdefault("integer", {}).setdefault("bspline", {})
    per_column = bs_cfg.get("per_column")
    if per_column is None or not isinstance(per_column, dict):
        per_column = {}
        bs_cfg["per_column"] = per_column

    col_cfg = per_column.get(column)
    if col_cfg is None or not isinstance(col_cfg, dict):
        col_cfg = {}
        per_column[column] = col_cfg

    # Ensure we're using quantile cap for the tuned column.
    col_cfg["cap_mode"] = "quantile"
    col_cfg["cap_quantile"] = float(params["bspline_cap_quantile"])
    col_cfg["positive_overflow"] = str(params["bspline_positive_overflow"])


def _trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trial in study.trials:
        out.append(
            {
                "number": int(trial.number),
                "value": None if trial.value is None else float(trial.value),
                "state": str(trial.state),
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
            }
        )
    return out


def main() -> None:
    args = parse_args()

    split_rows = int(args.split_rows)
    train_start = 0
    val_start = split_rows
    test_start = split_rows * 2

    train_df = read_rows(args.data_path, train_start, split_rows)
    val_df = read_rows(args.data_path, val_start, split_rows)
    test_df = read_rows(args.data_path, test_start, split_rows)

    base_config = make_base_config(args)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=str(args.study_name),
        storage=str(args.storage_url),
        load_if_exists=True,
    )

    completed_before = _completed_count(study)
    target = int(args.trials)
    remaining = max(0, target - completed_before)
    if int(args.max_new_trials) > 0:
        remaining = min(remaining, int(args.max_new_trials))

    overflow_modes = [m.strip() for m in str(args.overflow_modes).split(",") if m.strip()]
    if not overflow_modes:
        raise ValueError("--overflow-modes must contain at least one mode")

    device = torch.device("cpu")
    print(
        "B-spline per-column Optuna:",
        f"variant={base_config['experiment']['variant']}",
        f"bspline_config={args.bspline_config}",
        f"tuned_column={args.column}",
        f"completed_before={completed_before} target={target} remaining={remaining}",
        flush=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params: dict[str, Any] = {
            "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
            "weight_decay": trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True),
            "bspline_cap_quantile": trial.suggest_float(
                "bspline_cap_quantile",
                float(args.cap_quantile_min),
                float(args.cap_quantile_max),
            ),
            "bspline_positive_overflow": trial.suggest_categorical(
                "bspline_positive_overflow",
                overflow_modes,
            ),
        }

        trial_config = copy.deepcopy(base_config)
        apply_trial_params(trial_config, column=str(args.column), params=trial_params)

        prep_t0 = time.perf_counter()
        preprocessor = CriteoFeaturePreprocessor(trial_config)
        train_encoded = preprocessor.fit_transform(train_df)
        val_encoded = preprocessor.transform(val_df)
        prep_s = time.perf_counter() - prep_t0

        train_torch = encoded_split_to_torch(train_encoded, preprocessor.field_specs)
        val_torch = encoded_split_to_torch(val_encoded, preprocessor.field_specs)

        set_global_seed(int(trial_config["experiment"]["seed"]))
        model = build_fwfm_model(preprocessor.field_specs, trial_config).to(device)

        train_t0 = time.perf_counter()
        train_result = train_model(
            model, train_torch, val_torch, config=trial_config, device=device
        )
        train_s = time.perf_counter() - train_t0

        val_eval = evaluate_split(
            model,
            val_torch,
            device=device,
            batch_size=int(trial_config["train"]["batch_size"]),
        )

        val_logloss = float(val_eval["metrics"]["logloss"])
        val_auc = float(val_eval["metrics"]["auc"])

        trial.set_user_attr("best_epoch", int(train_result["best_epoch"]))
        trial.set_user_attr("val_logloss", val_logloss)
        trial.set_user_attr("val_auc", val_auc)
        trial.set_user_attr("preprocess_seconds", round(prep_s, 6))
        trial.set_user_attr("train_seconds", round(train_s, 6))

        del val_eval
        del train_result
        del model
        del train_torch
        del val_torch
        del train_encoded
        del val_encoded
        del preprocessor
        gc.collect()

        return val_logloss

    def checkpoint_callback(study_obj: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        payload = _checkpoint_payload(
            study=study_obj,
            args=args,
            hpo_seconds=(time.perf_counter() - t_hpo0),
        )
        args.checkpoint_json.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    t_hpo0 = time.perf_counter()
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            callbacks=[checkpoint_callback],
        )
    hpo_seconds = time.perf_counter() - t_hpo0

    completed_after = _completed_count(study)
    if completed_after == 0:
        raise RuntimeError("No completed trials in study; cannot finalize")

    best_params = dict(study.best_trial.params)
    final_config = copy.deepcopy(base_config)
    apply_trial_params(final_config, column=str(args.column), params=best_params)

    final_preprocessor = CriteoFeaturePreprocessor(final_config)
    train_encoded = final_preprocessor.fit_transform(train_df)
    val_encoded = final_preprocessor.transform(val_df)
    test_encoded = final_preprocessor.transform(test_df)

    train_torch = encoded_split_to_torch(train_encoded, final_preprocessor.field_specs)
    val_torch = encoded_split_to_torch(val_encoded, final_preprocessor.field_specs)
    test_torch = encoded_split_to_torch(test_encoded, final_preprocessor.field_specs)

    set_global_seed(int(final_config["experiment"]["seed"]))
    final_model = build_fwfm_model(final_preprocessor.field_specs, final_config).to(device)
    final_train_result = train_model(
        final_model, train_torch, val_torch, config=final_config, device=device
    )

    final_val_eval = evaluate_split(
        final_model,
        val_torch,
        device=device,
        batch_size=int(final_config["train"]["batch_size"]),
    )
    final_test_eval = evaluate_split(
        final_model,
        test_torch,
        device=device,
        batch_size=int(final_config["train"]["batch_size"]),
    )

    interaction_matrix_path = args.output_json.parent / "interaction_matrix.npz"
    save_interaction_matrix_npz(model=final_model, out_path=interaction_matrix_path)

    print(
        "RESULT bspline_integer_basis",
        f"tuned_column={args.column}",
        "best_trial_val_logloss=",
        f"{study.best_trial.value:.6f}",
        "final_val_logloss=",
        f"{final_val_eval['metrics']['logloss']:.6f}",
        "final_test_logloss=",
        f"{final_test_eval['metrics']['logloss']:.6f}",
        flush=True,
    )

    payload = {
        "variant": "bspline_integer_basis",
        "bspline_config": str(args.bspline_config),
        "tuned_column": str(args.column),
        "data_path": str(args.data_path),
        "split_rows": int(split_rows),
        "train_range": [train_start, train_start + split_rows],
        "val_range": [val_start, val_start + split_rows],
        "test_range": [test_start, test_start + split_rows],
        "target_trials": int(args.trials),
        "completed_trials": int(completed_after),
        "max_new_trials_this_invocation": int(args.max_new_trials),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "embedding_dim": int(args.embedding_dim),
        "bspline_knots": int(args.bspline_knots),
        "search_space": {
            "lr": [float(args.lr_min), float(args.lr_max)],
            "weight_decay": [float(args.wd_min), float(args.wd_max)],
            "bspline_cap_quantile": [float(args.cap_quantile_min), float(args.cap_quantile_max)],
            "bspline_positive_overflow": overflow_modes,
        },
        "study_name": str(args.study_name),
        "storage_url": str(args.storage_url),
        "best_trial": int(study.best_trial.number),
        "best_params": best_params,
        "best_trial_val_logloss": float(study.best_trial.value),
        "hpo_seconds_this_invocation": float(hpo_seconds),
        "final_retrain_best_epoch": int(final_train_result["best_epoch"]),
        "final_val_logloss": float(final_val_eval["metrics"]["logloss"]),
        "final_test_logloss": float(final_test_eval["metrics"]["logloss"]),
        "final_val_auc": float(final_val_eval["metrics"]["auc"]),
        "final_test_auc": float(final_test_eval["metrics"]["auc"]),
        "interaction_matrix_path": str(interaction_matrix_path),
        "trials_detail": _trial_rows(study),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cp = _checkpoint_payload(study=study, args=args, hpo_seconds=hpo_seconds)
    cp["final_val_logloss"] = float(final_val_eval["metrics"]["logloss"])
    cp["final_test_logloss"] = float(final_test_eval["metrics"]["logloss"])
    cp["interaction_matrix_path"] = str(interaction_matrix_path)
    args.checkpoint_json.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_json.write_text(json.dumps(cp, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

