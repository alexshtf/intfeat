# Proceedings

Code version:

- git rev: `23bedc1a5fe107a4816d9e5240387c94a04121a6`

## Command

Paths:

- Log: `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/run.log`
- PID: `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/pid.txt`
- Optuna DB: `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/study.sqlite3`
- Checkpoint JSON: `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/checkpoint.json`
- Results JSON: `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/results.json`
- Interaction matrix: `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/interaction_matrix.npz`

Detached command:

```bash
setsid -f bash -lc '
  cd /home/alex/git/intfeat
  export PYTHONUNBUFFERED=1
  echo $$ > journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/pid.txt
  exec .venv/bin/python -m experiments.criteo_fwfm.optuna_hybrid_bspline_sl_i11_potential_resumable \
    --hybrid-config experiments/criteo_fwfm/config/model_hybrid_bspline_sl_i5_quantile_large_token.yaml \
    --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
    --split-rows 400000 \
    --trials 100 \
    --num-epochs 1 \
    --batch-size 256 \
    --embedding-dim 8 \
    --bspline-knots 10 \
    --sl-num-basis 10 \
    --sl-cap-max 10000000 \
    --potential-family inverse_square \
    --kappa-min 0.05 \
    --kappa-max 50.0 \
    --x0-min 1e-3 \
    --x0-max 1e3 \
    --study-name hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t \
    --storage-url sqlite:///journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/study.sqlite3 \
    --checkpoint-json journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/checkpoint.json \
    --output-json journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/results.json
' </dev/null > journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/run.log 2>&1
```

Started:

- `2026-02-20 11:03:40 IST`
- PID: `2012250`
