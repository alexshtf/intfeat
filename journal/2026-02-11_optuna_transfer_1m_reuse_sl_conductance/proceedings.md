# Proceedings

## Run Command (Detached)

Notes:

- We use `setsid -f` for detaching (more reliable than plain `nohup` in this environment).
- `UV_CACHE_DIR=/tmp/uv_cache` avoids permission issues under `~/.cache/uv`.

```bash
mkdir -p journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance

setsid -f bash -lc '
  cd /home/alex/git/intfeat
  echo $$ > journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/pid.txt
  exec env PYTHONUNBUFFERED=1 UV_CACHE_DIR=/tmp/uv_cache \
    uv run python -m experiments.criteo_fwfm.optuna_head_contiguous_resumable \
      --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
      --train-rows 1000000 \
      --val-rows 200000 \
      --test-rows 200000 \
      --trials 100 \
      --num-epochs 1 \
      --batch-size 256 \
      --embedding-dim 8 \
      --sl-num-basis 10 \
      --bspline-knots 10 \
      --variants baseline_winner,bspline_integer_basis,sl_integer_basis \
      --study-prefix criteo_1m200k_transfer_uvalley_fixed \
      --storage-url sqlite:///journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/study.sqlite3 \
      --checkpoint-dir journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/checkpoints \
      --sl-fixed-u-exp-valley \
      --sl-u0 0.12714125920232416 \
      --sl-left-slope 0.10599526233947935 \
      --sl-right-slope 3.508157381375444 \
      --output-json journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/results.json
' > journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/run.log 2>&1
```

## Artifacts

- PID: `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/pid.txt`
- Log: `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/run.log`
- Optuna DB (shared across variants via study name): `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/study.sqlite3`
- Checkpoints (per variant): `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/checkpoints/`
- Summary JSON (written at end): `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/results.json`

## Monitoring

```bash
tail -f journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/run.log
ls -la journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/checkpoints
```

## Status Snapshot

Started: `2026-02-11T09:52:22+02:00`

- PID: `784955`
- Current log head: `===== Variant: baseline_winner =====`
