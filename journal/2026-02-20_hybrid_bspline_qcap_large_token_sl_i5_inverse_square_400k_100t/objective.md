# Objective

Test whether switching the tuned SL potential family for **only `I5`** from the right-end barrier
(`u_right_inverse_square`) to an **x-space inverse-square** potential improves validation loss over
the current quantile-cap baselines.

This experiment:

- Variant: `hybrid_bspline_sl`
- Integer encodings:
  - `I5`: SL basis (tuned)
  - all other integer columns: B-splines (fixed)
- Capping/overflow scheme (same as the quantile-cap baseline):
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
- `I5` SL conductance (u-exp-valley): `u0`, `left_slope`, `right_slope`
- `I5` SL potential (x inverse-square): `V(x)=kappa/(x+x0)^2`, tuning `kappa`, `x0`

Baseline references (completed):

- All integer columns B-splines, quantile-cap + LARGE token:
  - Val logloss: `0.4731016777`
  - Test logloss: `0.4678396553`
  - Stored: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`
- Hybrid `I5=SL` with **u-right-barrier** potential (`u_right_inverse_square`):
  - Val logloss: `0.4725881407`
  - Test logloss: `0.4676060331`
  - Stored: `journal/2026-02-19_hybrid_bspline_qcap_large_token_sl_i5_tune_400k_100t/results.json`

