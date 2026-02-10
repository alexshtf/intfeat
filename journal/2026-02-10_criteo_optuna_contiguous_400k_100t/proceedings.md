# Proceedings

## Setup

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Split mode: contiguous blocks
  - train: rows `[0, 400000)`
  - val: rows `[400000, 800000)`
  - test: rows `[800000, 1200000)`
- Model: `experiments/criteo_fwfm/model/fwfm.py` (FwFM)
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience set to `1`

## Run Command (Recommended, In-Repo)

This experiment was originally launched from `/tmp` before we standardized journaling.
Reproduce it using the in-repo script (added after this run started):

```bash
UV_CACHE_DIR=/tmp/uv_cache uv run python -m experiments.criteo_fwfm.optuna_contiguous \
  --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
  --split-rows 400000 \
  --trials 100 \
  --num-epochs 1 \
  --batch-size 256 \
  --embedding-dim 8 \
  --sl-num-basis 10 \
  --bspline-knots 10 \
  --variants baseline_winner,bspline_integer_basis \
  --output-json journal/2026-02-10_criteo_optuna_contiguous_400k_100t/results.json
```

## Live Artifacts (Current Run)

- PID file: `/tmp/criteo_optuna_400k_100t.pid`
- Log: `/tmp/criteo_optuna_400k_100t.log`
- Intended summary JSON (written at end): `/tmp/criteo_optuna_400k_100t_results.json`

## Status Snapshot

As of 2026-02-10 21:39 IST (parsed from `/tmp/criteo_optuna_400k_100t.log`):

- `baseline_winner`: Optuna complete (100/100), best trial val logloss `0.473367065759`
- `bspline_integer_basis`: Optuna in progress (62/100), best-so-far trial val logloss `0.471806406758`

Note: final "retrain with best params" val/test metrics are only produced after each variant finishes.
