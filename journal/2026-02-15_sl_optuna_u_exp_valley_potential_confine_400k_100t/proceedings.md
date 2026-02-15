# Proceedings

## Setup

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Split mode: contiguous blocks
  - train: rows `[0, 400000)`
  - val: rows `[400000, 800000)`
  - test: rows `[800000, 1200000)`
- Model: `experiments/criteo_fwfm/model/fwfm.py` (FwFM)
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience `1`

## Run Command (Detached)

This is a long-running job. It is launched detached so it will survive shell/Codex shutdown.

```bash
setsid -f /bin/bash -lc '
  cd /home/alex/git/intfeat
  EXP_DIR=journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t
  echo $$ > ${EXP_DIR}/pid.txt
  exec env UV_CACHE_DIR=/tmp/uv_cache PYTHONUNBUFFERED=1 \
    uv run python -m experiments.criteo_fwfm.optuna_sl_u_exp_valley_potential_resumable \
      --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
      --split-rows 400000 \
      --trials 100 \
      --num-epochs 1 \
      --batch-size 256 \
      --embedding-dim 8 \
      --sl-num-basis 10 \
      --potential-family u_power \
      --study-name sl_u_exp_valley_potential_confine_400k \
      --storage-url sqlite:///${EXP_DIR}/study.sqlite3 \
      --checkpoint-json ${EXP_DIR}/checkpoint.json \
      --output-json ${EXP_DIR}/results.json \
    >> ${EXP_DIR}/run.log 2>&1
'
```

Artifacts:

- PID: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/pid.txt`
- Log: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/run.log`
- Study DB: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/study.sqlite3`
- Checkpoint: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/checkpoint.json`
- Final summary: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/results.json`

## Monitoring

```bash
tail -n 50 journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/run.log
cat journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/checkpoint.json
```

## Live Status

- Launched: 2026-02-15
- PID: 1150783
- Completed: 2026-02-15 (100/100 trials)
  - Best trial: `54` (val logloss `0.4725398507508993`)
  - Final test logloss (best trial retrain): `0.4676545385640717`
  - Summary JSON: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/results.json`
  - PID exited after writing results.

## Resume

If you want to re-run this experiment from scratch, delete the experiment directory and re-launch the detached command.
