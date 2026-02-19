# Proceedings

## Setup

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Split mode: contiguous blocks
  - train: rows `[0, 400000)`
  - val: rows `[400000, 800000)`
  - test: rows `[800000, 1200000)`
- Model: `experiments/criteo_fwfm/model/fwfm.py` (FwFM)
- Integer encodings:
  - `I11`: SL basis (conductance + potential tuned)
  - all other integer columns: B-spline scalar input (fixed `knots=10`)
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience `1`

## Run Command (Detached)

This is a long-running job. It is launched detached so it will survive shell/Codex shutdown.

```bash
setsid -f /bin/bash -lc '
  cd /home/alex/git/intfeat
  EXP_DIR=journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t
  mkdir -p ${EXP_DIR}
  echo $$ > ${EXP_DIR}/pid.txt
  exec env UV_CACHE_DIR=/tmp/uv_cache PYTHONUNBUFFERED=1 \
    uv run python -m experiments.criteo_fwfm.optuna_hybrid_bspline_sl_i11_potential_resumable \
      --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
      --split-rows 400000 \
      --trials 100 \
      --num-epochs 1 \
      --batch-size 256 \
      --embedding-dim 8 \
      --bspline-knots 10 \
      --sl-num-basis 10 \
      --potential-family u_right_inverse_square \
      --study-name hybrid_bspline_sl_i11_potential_barrier_400k \
      --storage-url sqlite:///${EXP_DIR}/study.sqlite3 \
      --checkpoint-json ${EXP_DIR}/checkpoint.json \
      --output-json ${EXP_DIR}/results.json \
    >> ${EXP_DIR}/run.log 2>&1
'
```

Artifacts:

- PID: `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/pid.txt`
- Log: `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/run.log`
- Study DB: `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/study.sqlite3`
- Checkpoint: `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/checkpoint.json`
- Final summary: `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/results.json`

## Monitoring

```bash
tail -n 50 journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/run.log
cat journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/checkpoint.json
```

## Live Status

- Launched: 2026-02-17
- PID: `1714399` (exited after completion)
- Completed: 2026-02-18 (100/100 trials)
  - Best trial: `78`
  - Best/Final val logloss (best trial retrain): `0.47177167979007906`
  - Final test logloss (best trial retrain): `0.4668762496723887`
  - Summary JSON: `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/results.json`
