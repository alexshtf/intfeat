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
from .train import encoded_split_to_torch, evaluate_split, set_global_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable Optuna tuning for SL-FwFM with per-trial basis recomputation. "
            "Tunes lr/weight_decay and jointly tunes SL conductance (u_exp_valley) + "
            "histogram smoothing (prior_count, cutoff_quantile, cutoff_factor)."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/alex/datasets/criteo_kaggle_challenge/train.txt"),
    )
    parser.add_argument("--split-rows", type=int, default=200_000)
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Target number of completed trials in the study.",
    )
    parser.add_argument(
        "--max-new-trials",
        type=int,
        default=0,
        help="If >0, run at most this many additional trials in this invocation.",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--sl-num-basis", type=int, default=10)
    parser.add_argument("--sl-cap-max", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-json",
        type=Path,
        default=Path("artifacts/criteo_fwfm/sl_optuna_u_exp_valley_joint.checkpoint.json"),
    )
    parser.add_argument("--study-name", type=str, default="sl_optuna_u_exp_valley_joint")
    parser.add_argument(
        "--storage-url",
        type=str,
        default="sqlite:///artifacts/criteo_fwfm/sl_optuna_u_exp_valley_joint.sqlite3",
    )
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--wd-min", type=float, default=1e-8)
    parser.add_argument("--wd-max", type=float, default=1e-2)

    # Conductance search space.
    parser.add_argument("--u0-min", type=float, default=0.1)
    parser.add_argument("--u0-max", type=float, default=0.5)
    parser.add_argument("--left-slope-min", type=float, default=0.2)
    parser.add_argument("--left-slope-max", type=float, default=5.0)
    parser.add_argument("--right-slope-min", type=float, default=0.2)
    parser.add_argument("--right-slope-max", type=float, default=5.0)

    # Histogram / pmf smoothing search space.
    parser.add_argument("--prior-count-min", type=float, default=1e-3)
    parser.add_argument("--prior-count-max", type=float, default=5.0)
    parser.add_argument("--cutoff-q-min", type=float, default=0.95)
    parser.add_argument("--cutoff-q-max", type=float, default=0.999)
    parser.add_argument("--cutoff-factor-min", type=float, default=1.0)
    parser.add_argument("--cutoff-factor-max", type=float, default=2.0)

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


def make_base_config(args: argparse.Namespace) -> dict[str, Any]:
    default_cfg = Path("experiments/criteo_fwfm/config/default.yaml")
    sl_cfg = Path("experiments/criteo_fwfm/config/model_sl.yaml")
    config = resolve_config(default_cfg, [sl_cfg], [])

    config["experiment"]["seed"] = int(args.seed)
    config["data"]["path"] = str(args.data_path)

    config["model"]["embedding_dim"] = int(args.embedding_dim)
    config["model"]["integer"]["sl"]["num_basis"] = int(args.sl_num_basis)
    config["model"]["integer"]["sl"]["cap_max"] = int(args.sl_cap_max)

    config["train"]["batch_size"] = int(args.batch_size)
    config["train"]["num_epochs"] = int(args.num_epochs)
    config["train"]["device"] = "cpu"
    config["train"]["early_stopping"]["patience"] = 1

    return config


def apply_trial_params(config: dict[str, Any], params: dict[str, float]) -> None:
    config["train"]["lr"] = float(params["lr"])
    config["train"]["weight_decay"] = float(params["weight_decay"])

    config["model"]["integer"]["sl"]["conductance"]["family"] = "u_exp_valley"
    config["model"]["integer"]["sl"]["conductance"]["u_exp_valley"]["u0"] = float(
        params["sl_u0"]
    )
    config["model"]["integer"]["sl"]["conductance"]["u_exp_valley"]["left_slope"] = float(
        params["sl_left_slope"]
    )
    config["model"]["integer"]["sl"]["conductance"]["u_exp_valley"]["right_slope"] = float(
        params["sl_right_slope"]
    )

    config["model"]["integer"]["sl"]["hist"]["prior_count"] = float(params["sl_prior_count"])
    config["model"]["integer"]["sl"]["hist"]["cutoff_quantile"] = float(
        params["sl_cutoff_quantile"]
    )
    config["model"]["integer"]["sl"]["hist"]["cutoff_factor"] = float(
        params["sl_cutoff_factor"]
    )


def _trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    out = []
    for trial in study.trials:
        out.append(
            {
                "number": int(trial.number),
                "value": None if trial.value is None else float(trial.value),
                "state": str(trial.state),
                "params": {k: float(v) for k, v in trial.params.items()},
                "user_attrs": {
                    k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in trial.user_attrs.items()
                },
            }
        )
    return out


def _completed_count(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)


def _checkpoint_payload(
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
    device = torch.device("cpu")

    _ensure_sqlite_parent(str(args.storage_url))

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage_url,
        load_if_exists=True,
    )

    completed_before = _completed_count(study)
    remaining = max(int(args.trials) - completed_before, 0)
    if int(args.max_new_trials) > 0:
        remaining = min(remaining, int(args.max_new_trials))

    print(
        f"Study '{args.study_name}': completed_before={completed_before}, "
        f"target={args.trials}, running_now={remaining}",
        flush=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = {
            "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", args.wd_min, args.wd_max, log=True
            ),
            "sl_u0": trial.suggest_float("sl_u0", args.u0_min, args.u0_max),
            "sl_left_slope": trial.suggest_float(
                "sl_left_slope", args.left_slope_min, args.left_slope_max, log=True
            ),
            "sl_right_slope": trial.suggest_float(
                "sl_right_slope", args.right_slope_min, args.right_slope_max, log=True
            ),
            "sl_prior_count": trial.suggest_float(
                "sl_prior_count", args.prior_count_min, args.prior_count_max, log=True
            ),
            "sl_cutoff_quantile": trial.suggest_float(
                "sl_cutoff_quantile", args.cutoff_q_min, args.cutoff_q_max
            ),
            "sl_cutoff_factor": trial.suggest_float(
                "sl_cutoff_factor", args.cutoff_factor_min, args.cutoff_factor_max
            ),
        }

        trial_config = copy.deepcopy(base_config)
        apply_trial_params(trial_config, trial_params)

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

    def checkpoint_callback(
        study_obj: optuna.Study,
        _trial: optuna.trial.FrozenTrial,
    ) -> None:
        payload = _checkpoint_payload(
            study_obj,
            args,
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

    best_params = {
        "lr": float(study.best_trial.params["lr"]),
        "weight_decay": float(study.best_trial.params["weight_decay"]),
        "sl_u0": float(study.best_trial.params["sl_u0"]),
        "sl_left_slope": float(study.best_trial.params["sl_left_slope"]),
        "sl_right_slope": float(study.best_trial.params["sl_right_slope"]),
        "sl_prior_count": float(study.best_trial.params["sl_prior_count"]),
        "sl_cutoff_quantile": float(study.best_trial.params["sl_cutoff_quantile"]),
        "sl_cutoff_factor": float(study.best_trial.params["sl_cutoff_factor"]),
    }

    final_config = copy.deepcopy(base_config)
    apply_trial_params(final_config, best_params)

    final_prep_t0 = time.perf_counter()
    final_preprocessor = CriteoFeaturePreprocessor(final_config)
    train_encoded = final_preprocessor.fit_transform(train_df)
    val_encoded = final_preprocessor.transform(val_df)
    test_encoded = final_preprocessor.transform(test_df)
    final_prep_seconds = time.perf_counter() - final_prep_t0

    train_torch = encoded_split_to_torch(train_encoded, final_preprocessor.field_specs)
    val_torch = encoded_split_to_torch(val_encoded, final_preprocessor.field_specs)
    test_torch = encoded_split_to_torch(test_encoded, final_preprocessor.field_specs)

    set_global_seed(int(final_config["experiment"]["seed"]))
    final_model = build_fwfm_model(final_preprocessor.field_specs, final_config).to(device)

    final_train_t0 = time.perf_counter()
    final_train_result = train_model(
        final_model, train_torch, val_torch, config=final_config, device=device
    )
    final_train_seconds = time.perf_counter() - final_train_t0

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

    print(
        "RESULT sl_integer_basis",
        "best_trial_val_logloss=",
        f"{study.best_trial.value:.6f}",
        "final_val_logloss=",
        f"{final_val_eval['metrics']['logloss']:.6f}",
        "final_test_logloss=",
        f"{final_test_eval['metrics']['logloss']:.6f}",
        flush=True,
    )

    payload = {
        "variant": "sl_integer_basis",
        "sl_conductance_family": "u_exp_valley",
        "data_path": str(args.data_path),
        "split_rows": split_rows,
        "train_range": [train_start, train_start + split_rows],
        "val_range": [val_start, val_start + split_rows],
        "test_range": [test_start, test_start + split_rows],
        "target_trials": int(args.trials),
        "completed_trials": int(completed_after),
        "max_new_trials_this_invocation": int(args.max_new_trials),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "embedding_dim": int(args.embedding_dim),
        "sl_num_basis": int(args.sl_num_basis),
        "sl_cap_max": int(args.sl_cap_max),
        "search_space": {
            "lr": [float(args.lr_min), float(args.lr_max)],
            "weight_decay": [float(args.wd_min), float(args.wd_max)],
            "sl_u0": [float(args.u0_min), float(args.u0_max)],
            "sl_left_slope": [float(args.left_slope_min), float(args.left_slope_max)],
            "sl_right_slope": [float(args.right_slope_min), float(args.right_slope_max)],
            "sl_prior_count": [float(args.prior_count_min), float(args.prior_count_max)],
            "sl_cutoff_quantile": [float(args.cutoff_q_min), float(args.cutoff_q_max)],
            "sl_cutoff_factor": [float(args.cutoff_factor_min), float(args.cutoff_factor_max)],
        },
        "study_name": args.study_name,
        "storage_url": args.storage_url,
        "best_trial": int(study.best_trial.number),
        "best_params": best_params,
        "best_trial_val_logloss": float(study.best_trial.value),
        "hpo_seconds_this_invocation": float(hpo_seconds),
        "final_retrain_best_epoch": int(final_train_result["best_epoch"]),
        "final_val_logloss": float(final_val_eval["metrics"]["logloss"]),
        "final_test_logloss": float(final_test_eval["metrics"]["logloss"]),
        "final_val_auc": float(final_val_eval["metrics"]["auc"]),
        "final_test_auc": float(final_test_eval["metrics"]["auc"]),
        "final_preprocess_seconds": float(final_prep_seconds),
        "final_train_seconds": float(final_train_seconds),
        "trials_detail": _trial_rows(study),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote summary: {args.output_json}", flush=True)

    cp = _checkpoint_payload(study, args, hpo_seconds=hpo_seconds)
    cp["final_val_logloss"] = float(final_val_eval["metrics"]["logloss"])
    cp["final_test_logloss"] = float(final_test_eval["metrics"]["logloss"])
    args.checkpoint_json.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_json.write_text(json.dumps(cp, indent=2), encoding="utf-8")
    print(f"Wrote checkpoint: {args.checkpoint_json}", flush=True)


if __name__ == "__main__":
    main()

