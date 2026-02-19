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

## Artifacts (Original Run)

- PID file (stale after termination): `/tmp/criteo_optuna_400k_100t.pid`
- Log: `/tmp/criteo_optuna_400k_100t.log`
- Intended summary JSON (written at end): `/tmp/criteo_optuna_400k_100t_results.json`

## Status Snapshot

As of 2026-02-10 22:48 IST (parsed from `/tmp/criteo_optuna_400k_100t.log`):

- `baseline_winner`: Optuna complete (100/100)
  - best trial: `31`
  - best trial val logloss: `0.4733670657590525`
  - best params: `lr=0.0019861309562720034`, `weight_decay=1.0724458502932633e-06`
- `bspline_integer_basis`: Optuna stopped early (72/100)
  - best trial: `37`
  - best trial val logloss: `0.47180640675774904`
  - best params: `lr=0.001961117089253444`, `weight_decay=6.748165404097594e-07`

Note:

- The process was terminated manually at 72 trials because we already had enough evidence that B-splines are ahead.
- Termination: killed PID `576515` on 2026-02-10 22:48 IST.
- The script only writes its summary JSON at the end, so `/tmp/criteo_optuna_400k_100t_results.json` was not produced.

## Refit Best B-spline (Compute Test Logloss)

To compute test logloss for the best B-spline configuration found above (trial 37 hyperparameters):

```bash
UV_CACHE_DIR=/tmp/uv_cache uv run python -m experiments.criteo_fwfm.refit_contiguous \
  --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
  --split-rows 400000 \
  --variant bspline_integer_basis \
  --lr 0.001961117089253444 \
  --weight-decay 6.748165404097594e-07 \
  --num-epochs 1 \
  --batch-size 256 \
  --embedding-dim 8 \
  --bspline-knots 10 \
  --output-json journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37.json
```

Output:

- `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37.json`
