from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Criteo FwFM run metrics")
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory containing timestamped run subdirectories",
    )
    return parser.parse_args()


def _load_metrics(run_dir: Path) -> dict[str, object] | None:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.resolved.yaml"
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    row = {
        "run_dir": run_dir.name,
        "variant": payload.get("variant"),
        "best_epoch": payload.get("best_epoch"),
        "val_logloss": payload.get("val", {}).get("logloss"),
        "val_auc": payload.get("val", {}).get("auc"),
        "test_logloss": payload.get("test", {}).get("logloss"),
        "test_auc": payload.get("test", {}).get("auc"),
    }

    # Variant may not be duplicated in metrics, so try from config fallback.
    if row["variant"] is None and config_path.exists():
        import yaml

        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        row["variant"] = cfg.get("experiment", {}).get("variant")

    return row


def main() -> None:
    args = _parse_args()
    runs_dir = Path(args.runs_dir).expanduser()

    rows = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        row = _load_metrics(entry)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No run metrics found.")
        return

    table = pd.DataFrame(rows).sort_values(["variant", "run_dir"])
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
