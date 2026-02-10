# Proceedings

## Setup

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Split mode: contiguous blocks
  - train: rows `[0, 400000)`
  - val: rows `[400000, 800000)`
  - test: rows `[800000, 1200000)`
- Variant: `sl_integer_basis`
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience set to `1`
- Optuna storage: SQLite (resumable)

## Run Command (Recommended, In-Repo)

This run was launched before journaling was standardized; reproduce it using the in-repo script:

```bash
UV_CACHE_DIR=/tmp/uv_cache uv run python -m experiments.criteo_fwfm.optuna_sl_conductance_resumable \
  --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
  --split-rows 400000 \
  --trials 100 \
  --num-epochs 1 \
  --batch-size 256 \
  --embedding-dim 8 \
  --sl-num-basis 10 \
  --sl-cap-max 10000000 \
  --study-name sl_optuna_conductance_400k_100t \
  --storage-url sqlite:///journal/2026-02-10_sl_optuna_conductance_400k_100t/study.sqlite3 \
  --output-json journal/2026-02-10_sl_optuna_conductance_400k_100t/results.json \
  --checkpoint-json journal/2026-02-10_sl_optuna_conductance_400k_100t/checkpoint.json
```

## Artifacts (Original Run)

- Log: `/tmp/sl_optuna_conductance_400k_100t.log`
- Study DB: `/tmp/sl_optuna_conductance_400k_100t.sqlite3`
- Summary: `/tmp/sl_optuna_conductance_400k_100t.json`
- Checkpoint: `/tmp/sl_optuna_conductance_400k_100t.checkpoint.json`

## Results Snapshot

This run completed (100/100 trials) on 2026-02-10.

- Best trial val logloss: `0.47254733012305855`
- Final retrain val logloss: `0.47254733012305855`
- Final test logloss: `0.46713404881787735`

Best parameters:

- `lr = 0.0012940077533808388`
- `weight_decay = 1.6965892356085282e-08`
- `sl_alpha = 0.3581980808609457`
- `sl_beta = 0.575839137288588`
- `sl_center = 0.028450174282538376`
