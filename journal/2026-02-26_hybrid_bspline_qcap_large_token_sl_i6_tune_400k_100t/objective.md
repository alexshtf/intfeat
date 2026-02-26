# Objective

Run the "one-column tuning" hybrid experiment again, but tune SL **for `I6`** (instead of `I5`),
since `I6` looks like the most important integer column by XGBoost gain importance on the 400k head.

This experiment:

- Variant: `hybrid_bspline_sl`
- Integer encodings:
  - `I6`: SL basis (tuned)
  - all other integer columns: B-splines (fixed)
- Capping/overflow scheme:
  - values `<= 0`: treated as categorical tokens (per-column)
  - positive values:
    - `1..cap_value`: use basis
    - `> cap_value`: per-column `LARGE` discrete token, no basis
  - `cap_value`: per-column quantile cap from train positives:
    - `cap_value = min(cap_max, floor(q-quantile * factor))`
    - `q=0.99`, `factor=1.1`, `cap_max=10_000_000`

Setup (400k/400k/400k contiguous split):

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Train rows: `[0, 400000)`
- Val rows: `[400000, 800000)`
- Test rows: `[800000, 1200000)`
- Model: FwFM (`experiments/criteo_fwfm/model/fwfm.py`)
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience `1`

What we tune (Optuna, 100 trials):

- `lr`
- `weight_decay`
- `I6` SL conductance (u-exp-valley): `u0`, `left_slope`, `right_slope`
- `I6` SL potential (u-right inverse-square): `V(u)=kappa/(1-u+eps)^2`, tuning `kappa`, `eps`

Reference:

- XGBoost importance on 400k head: `journal/2026-02-25_xgb_feature_importance_400k/importance.json`

