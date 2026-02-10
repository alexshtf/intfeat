# Proceedings

## Setup

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Split mode: contiguous blocks
  - train: rows `[0, 400000)`
  - val: rows `[400000, 800000)`
  - test: rows `[800000, 1200000)`
- Variant: `sl_integer_basis`
- SL conductance family: `u_exp_valley`
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience set to `1`
- Optuna storage: SQLite (resumable)

## Run Command (setsid, detached)

This continues the existing study DB from `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/`.

```bash
mkdir -p journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25

setsid -f bash -lc '
  cd /home/alex/git/intfeat
  echo $$ > journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/pid.txt
  exec env PYTHONUNBUFFERED=1 UV_CACHE_DIR=/tmp/uv_cache \
    uv run python -m experiments.criteo_fwfm.optuna_sl_u_exp_valley_resumable \
      --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
      --split-rows 400000 \
      --trials 25 \
      --num-epochs 1 \
      --batch-size 256 \
      --embedding-dim 8 \
      --sl-num-basis 10 \
      --sl-cap-max 10000000 \
      --study-name sl_optuna_u_exp_valley_400k_60t \
      --storage-url sqlite:///journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/study.sqlite3 \
      --output-json journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/results.json \
      --checkpoint-json journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/checkpoint.json
' > journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/run.log 2>&1
```

## Artifacts

- PID: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/pid.txt`
- Log: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/run.log`
- Checkpoint: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/checkpoint.json` (updated after each trial)
- Summary: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/results.json` (written at end)

## Status Snapshot

Started: `2026-02-11T00:28:29+02:00`

- PID:
- PID: `691271`
- Completed trials: `25` (completed)
- Best-trial val logloss: `0.4720136818700594` (trial `4`)
