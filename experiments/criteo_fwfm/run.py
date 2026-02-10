from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    build_run_dir,
    save_encoder_state,
    save_field_stats,
    save_history,
    save_metrics,
    save_model,
    save_predictions,
    save_resolved_config,
)
from .config import resolve_config
from .data_io import load_criteo_splits
from .model.fwfm import build_fwfm_model
from .preprocess import CriteoFeaturePreprocessor
from .train import encoded_split_to_torch, evaluate_split, train_model

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Criteo FwFM experiments with baseline, SL, or B-spline integer encoders."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to YAML config. Can be provided multiple times; applied in order.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and run preprocessing only, without training.",
    )
    return parser.parse_args()


def _configure_logging(run_dir: Path) -> None:
    run_log_path = run_dir / "run.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def _resolve_device(config: dict[str, Any]) -> torch.device:
    raw_device = str(config["train"].get("device", "auto")).lower()
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(raw_device)


def main() -> None:
    args = _parse_args()

    root = Path(__file__).resolve().parent
    default_config_path = root / "config" / "default.yaml"
    user_config_paths = [Path(path).expanduser() for path in args.config]

    config = resolve_config(default_config_path, user_config_paths, args.overrides)
    run_dir = build_run_dir(config)
    _configure_logging(run_dir)

    LOGGER.info("Run directory: %s", run_dir)
    save_resolved_config(config, run_dir)

    splits = load_criteo_splits(config)
    preprocessor = CriteoFeaturePreprocessor(config)
    train_encoded = preprocessor.fit_transform(splits["train"])
    val_encoded = preprocessor.transform(splits["val"])
    test_encoded = preprocessor.transform(splits["test"])

    save_encoder_state(preprocessor, run_dir)

    field_stats_payload = {
        "routing": preprocessor.routing_report(),
        "field_stats": preprocessor.field_stats_report(),
    }
    save_field_stats(field_stats_payload, run_dir=run_dir)

    if args.dry_run:
        LOGGER.info("Dry run completed before model training.")
        return

    if len(splits["val"]) == 0:
        raise ValueError("Validation split is empty; set data.split.val_rows > 0")

    train_torch = encoded_split_to_torch(train_encoded, preprocessor.field_specs)
    val_torch = encoded_split_to_torch(val_encoded, preprocessor.field_specs)
    test_torch = encoded_split_to_torch(test_encoded, preprocessor.field_specs)

    model = build_fwfm_model(preprocessor.field_specs, config)
    device = _resolve_device(config)
    model = model.to(device)

    LOGGER.info("Training on device: %s", device)
    training_result = train_model(
        model,
        train_torch,
        val_torch,
        config=config,
        device=device,
    )
    save_history(training_result["history"], run_dir)

    batch_size = int(config["train"]["batch_size"])
    train_eval = evaluate_split(model, train_torch, device=device, batch_size=batch_size)
    val_eval = evaluate_split(model, val_torch, device=device, batch_size=batch_size)
    test_eval = evaluate_split(model, test_torch, device=device, batch_size=batch_size)

    metrics_payload = {
        "variant": config["experiment"]["variant"],
        "best_epoch": int(training_result["best_epoch"]),
        "best_value": float(training_result["best_value"]),
        "train": train_eval["metrics"],
        "val": val_eval["metrics"],
        "test": test_eval["metrics"],
        "routing": preprocessor.routing_report(),
        "field_stats": preprocessor.field_stats_report(),
    }
    save_metrics(metrics_payload, run_dir)

    if bool(config["artifacts"].get("save_model", True)):
        save_model(model, run_dir)

    if bool(config["artifacts"].get("save_predictions", True)):
        save_predictions(
            labels=val_eval["labels"],
            logits=val_eval["logits"],
            probs=val_eval["probs"],
            split_name="val",
            run_dir=run_dir,
        )
        save_predictions(
            labels=test_eval["labels"],
            logits=test_eval["logits"],
            probs=test_eval["probs"],
            split_name="test",
            run_dir=run_dir,
        )

    LOGGER.info("Run complete. Metrics: %s", metrics_payload)


if __name__ == "__main__":
    main()
