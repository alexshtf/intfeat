# Proceedings

Code version:

- git rev: `8059851`

## Command

Paths:

- Log: `journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/run.log`
- PID: `journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/pid.txt`
- Optuna DB: `journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/study.sqlite3`
- Checkpoint JSON: `journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/checkpoint.json`
- Results JSON: `journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/results.json`
- Interaction matrix: `journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/interaction_matrix.npz`

Detached command:

```bash
setsid -f bash -lc '
  cd /home/alex/git/intfeat
  export PYTHONUNBUFFERED=1
  echo $$ > journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/pid.txt
  exec .venv/bin/python -m experiments.criteo_fwfm.optuna_bspline_per_column_resumable \
    --bspline-config experiments/criteo_fwfm/config/model_bspline_quantile_large_token.yaml \
    --column I6 \
    --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
    --split-rows 400000 \
    --trials 100 \
    --num-epochs 1 \
    --batch-size 256 \
    --embedding-dim 8 \
    --bspline-knots 10 \
    --cap-quantile-min 0.95 \
    --cap-quantile-max 0.9995 \
    --overflow-modes large_token,clip_to_cap \
    --study-name bspline_i6_tune_capq_overflow_400k_100t \
    --storage-url sqlite:///journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/study.sqlite3 \
    --checkpoint-json journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/checkpoint.json \
    --output-json journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/results.json
' </dev/null > journal/2026-02-27_bspline_i6_tune_cap_quantile_overflow_400k_100t/run.log 2>&1
```

Started:

- `2026-02-27T10:16:40+02:00`
- PID: `2708658` (see `.../pid.txt`)
