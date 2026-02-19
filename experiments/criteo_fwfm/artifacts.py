from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import save_yaml
from .interactions import save_interaction_matrix_npz


def build_run_dir(config: dict[str, Any]) -> Path:
    base_dir = Path(config["artifacts"]["dir"]).expanduser()
    run_name = str(config["experiment"]["name"])
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_metrics(metrics: dict[str, Any], run_dir: Path) -> None:
    save_json(metrics, run_dir / "metrics.json")


def save_history(history: list[dict[str, float]], run_dir: Path) -> None:
    save_json({"history": history}, run_dir / "history.json")


def save_field_stats(payload: dict[str, Any], run_dir: Path) -> None:
    save_json(payload, run_dir / "field_stats.json")


def save_predictions(
    *,
    labels,
    logits,
    probs,
    split_name: str,
    run_dir: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "label": labels,
            "logit": logits,
            "prob": probs,
        }
    )
    parquet_path = run_dir / f"predictions_{split_name}.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
    except ImportError:
        # Keep runs usable in minimal environments without optional parquet extras.
        frame.to_csv(run_dir / f"predictions_{split_name}.csv", index=False)


def save_model(model: torch.nn.Module, run_dir: Path) -> None:
    torch.save(model.state_dict(), run_dir / "model.pt")


def save_interaction_matrix(model: torch.nn.Module, run_dir: Path) -> None:
    """Save the effective FwFM field interaction matrix R (symmetric, zero diagonal)."""
    save_interaction_matrix_npz(model=model, out_path=run_dir / "interaction_matrix.npz")


def save_resolved_config(config: dict[str, Any], run_dir: Path) -> None:
    save_yaml(config, run_dir / "config.resolved.yaml")


def save_encoder_state(preprocessor: object, run_dir: Path) -> None:
    torch.save(preprocessor, run_dir / "encoder_state.pkl")
