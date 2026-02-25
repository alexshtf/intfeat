# Proceedings

Code version:

- git rev: `a3b1686`

## Command

This uses `uv run --with xgboost` so we don't have to pin XGBoost in `pyproject.toml` yet.

```bash
cd /home/alex/git/intfeat
UV_CACHE_DIR=/tmp/uv_cache uv run --with xgboost \
  -m experiments.criteo_fwfm.analysis.xgb_feature_importance \
  --data-path /home/alex/datasets/criteo_kaggle_challenge/train.txt \
  --start-row 0 \
  --n-rows 400000 \
  --seed 0 \
  --output-json journal/2026-02-25_xgb_feature_importance_400k/importance.json
```

Output:

- `journal/2026-02-25_xgb_feature_importance_400k/importance.json`

