# Proceedings

## Run Command (Detached)

This is a long-running job. It is launched detached so it will survive shell/Codex shutdown.

```bash
setsid -f /bin/bash -lc '
  cd /home/alex/git/intfeat
  mkdir -p journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t
  echo $$ > journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/pid.txt
  exec env UV_CACHE_DIR=/tmp/uv_cache PYTHONUNBUFFERED=1 \
    uv run python -m experiments.criteo_fwfm.optuna_hybrid_bspline_sl_i11_potential_resumable \
      --hybrid-config experiments/criteo_fwfm/config/model_hybrid_bspline_sl_i5_i11.yaml \
      --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
      --split-rows 400000 \
      --trials 100 \
      --num-epochs 1 \
      --batch-size 256 \
      --embedding-dim 8 \
      --bspline-knots 10 \
      --sl-num-basis 10 \
      --potential-family u_right_inverse_square \
      --study-name hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k \
      --storage-url sqlite:///journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/study.sqlite3 \
      --checkpoint-json journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/checkpoint.json \
      --output-json journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/results.json \
    >> journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/run.log 2>&1
'
```

Artifacts:

- PID: `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/pid.txt`
- Log: `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/run.log`
- Study DB: `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/study.sqlite3`
- Checkpoint: `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/checkpoint.json`
- Final summary: `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/results.json`
- Interaction matrix (final best-model retrain): `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/interaction_matrix.npz`

## Monitoring

```bash
tail -n 50 journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/run.log
cat journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/checkpoint.json
```

## Live Status

- Launched: 2026-02-18
- PID: `1761630` (exited after completion)
- Completed: 2026-02-18 (100/100 trials)
  - Best trial: `90`
  - Best/Final val logloss (best trial retrain): `0.4718875464682844`
  - Final test logloss (best trial retrain): `0.4668398528148151`
  - Summary JSON: `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/results.json`
