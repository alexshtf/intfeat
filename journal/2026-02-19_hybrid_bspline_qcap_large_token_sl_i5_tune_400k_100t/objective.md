# Objective

Test whether making **only `I5`** use an **SL** basis (with tuned conductance + potential) helps over the current best **all-B-spline** baseline under the **quantile-cap + LARGE overflow token** scheme.

Baseline reference (completed):

- All integer columns: B-splines, quantile-cap + LARGE token.
- Results: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`

This experiment:

- Variant: `hybrid_bspline_sl`
- Integer encodings:
  - `I5`: SL basis (tuned)
  - all other integer columns: B-splines (fixed)
- Shared capping/overflow scheme for both B-spline and SL integer encoders:
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
- `I5` SL potential: right-end barrier `V(u)=kappa/(1-u+eps)^2` with `kappa`, `eps`

What we fix:

- B-spline settings for all other integer columns: degree=3, knots=10, `input_map=log1p_cap_to_unit`
- SL basis size: `num_basis=10`
- SL capping: quantile cap, LARGE overflow (as above)

