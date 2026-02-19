# Proceedings

Code version:

- git rev: `e6b974098a93a5da39821f72eab53d5cd90fbf8d`

## Commands

All long-running commands are launched detached via `setsid -f`, with:

- stdout/stderr redirected to `run.log`
- PID recorded in `pid.txt`
- Optuna SQLite DB stored under this directory

### 1) B-spline (quantile cap + LARGE token)

Paths:

- Log: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_run.log`
- PID: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_pid.txt` (current: `1889130`)
- Optuna DB: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_study.sqlite3`
- Summary JSON: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`
- Interaction matrix: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/interaction_matrix_bspline_integer_basis.npz`

Detached command:

```bash
setsid -f bash -lc '
  cd /home/alex/git/intfeat
  export PYTHONUNBUFFERED=1
  echo $$ > journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_pid.txt
  exec .venv/bin/python -m experiments.criteo_fwfm.optuna_head_contiguous_resumable \
    --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
    --train-rows 400000 --val-rows 400000 --test-rows 400000 \
    --variants bspline_integer_basis \
    --bspline-config experiments/criteo_fwfm/config/model_bspline_quantile_large_token.yaml \
    --bspline-knots 10 \
    --trials 100 \
    --num-epochs 1 \
    --batch-size 256 \
    --embedding-dim 8 \
    --study-prefix qcap_large_token_400k \
    --storage-url sqlite:///journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_study.sqlite3 \
    --checkpoint-dir journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_checkpoints \
    --output-json journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json
' </dev/null > journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_run.log 2>&1
```

### 2) SL (quantile cap + LARGE token, tune conductance + potential)

Paths:

- Log: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_run.log`
- PID: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_pid.txt` (current: `1889183`)
- Optuna DB: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_study.sqlite3`
- Checkpoint JSON: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_checkpoint.json`
- Summary JSON: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_results.json`
- Interaction matrix: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/interaction_matrix.npz`

Detached command:

```bash
setsid -f bash -lc '
  cd /home/alex/git/intfeat
  export PYTHONUNBUFFERED=1
  echo $$ > journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_pid.txt
  exec .venv/bin/python -m experiments.criteo_fwfm.optuna_sl_u_exp_valley_potential_resumable \
    --sl-config experiments/criteo_fwfm/config/model_sl_quantile_large_token.yaml \
    --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
    --split-rows 400000 \
    --trials 100 \
    --num-epochs 1 \
    --batch-size 256 \
    --embedding-dim 8 \
    --sl-num-basis 10 \
    --sl-cap-max 10000000 \
    --sl-cap-mode quantile \
    --sl-cutoff-quantile 0.99 \
    --sl-cutoff-factor 1.1 \
    --sl-positive-overflow large_token \
    --potential-family u_power \
    --study-name sl_qcap_large_token_u_exp_valley_u_power_400k_100t \
    --storage-url sqlite:///journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_study.sqlite3 \
    --checkpoint-json journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_checkpoint.json \
    --output-json journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_results.json
' </dev/null > journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_run.log 2>&1
```

## Final Status

- SL finished: see `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_results.json`
- B-spline finished: see `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`
