# Proceedings

## Setup

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Split mode: contiguous blocks
  - train: rows `[0, 400000)`
  - val: rows `[400000, 800000)`
  - test: rows `[800000, 1200000)`
- Variant: `sl_integer_basis`
- SL conductance family: `u_exp_valley`
- Histogram/pmf settings: fixed (default config)
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience set to `1`
- Optuna storage: SQLite (resumable)

## Run Command (setsid, detached)

```bash
mkdir -p journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t

setsid -f bash -lc '
  cd /home/alex/git/intfeat
  echo $$ > journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/pid.txt
  exec env PYTHONUNBUFFERED=1 UV_CACHE_DIR=/tmp/uv_cache \
    uv run python -m experiments.criteo_fwfm.optuna_sl_u_exp_valley_resumable \
      --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
      --split-rows 400000 \
      --trials 30 \
      --num-epochs 1 \
      --batch-size 256 \
      --embedding-dim 8 \
      --sl-num-basis 10 \
      --sl-cap-max 10000000 \
      --study-name sl_optuna_u_exp_valley_narrow_400k_30t \
      --storage-url sqlite:///journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/study.sqlite3 \
      --output-json journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/results.json \
      --checkpoint-json journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/checkpoint.json \
      --u0-min 0.18 --u0-max 0.42 \
      --left-slope-min 0.2 --left-slope-max 3.0 \
      --right-slope-min 0.2 --right-slope-max 3.0
' > journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/run.log 2>&1
```

## Artifacts

- PID: `journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/pid.txt`
- Log: `journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/run.log`
- Study DB: `journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/study.sqlite3`
- Checkpoint: `journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/checkpoint.json` (updated after each trial)
- Summary: `journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/results.json` (written at end)

## Status Snapshot

Started: `2026-02-11T00:06:11+02:00`

- PID:
- PID: `685990`
- Completed trials: `9` (stopped early; see `checkpoint.json`)
- Best-trial val logloss: `0.4725625080548662` (trial `6`)

## Early Stop + Finalization

We stopped the long-running job early after 9 completed trials (trial 9 was running) and then finalized the study by re-running the same script with `--trials 9` so it skips HPO and writes a `results.json`.

Finalization command:

```bash
env PYTHONUNBUFFERED=1 UV_CACHE_DIR=/tmp/uv_cache uv run python -m experiments.criteo_fwfm.optuna_sl_u_exp_valley_resumable \
  --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
  --split-rows 400000 \
  --trials 9 \
  --max-new-trials 0 \
  --num-epochs 1 \
  --batch-size 256 \
  --embedding-dim 8 \
  --sl-num-basis 10 \
  --sl-cap-max 10000000 \
  --study-name sl_optuna_u_exp_valley_narrow_400k_30t \
  --storage-url sqlite:///journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/study.sqlite3 \
  --output-json journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/results.json \
  --checkpoint-json journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/checkpoint.json
```
