# Objective

Re-run the **B-spline vs Sturm-Liouville (SL)** comparison on Criteo with a new, more deliberate integer feature capping scheme:

- Values `<= 0` remain **categorical** (one token per distinct non-positive value per column, plus missing).
- Positive values are split into:
  - **in-range**: `1 <= x <= cap_value` uses a smooth basis (B-spline scalar curve or SL eigenbasis).
  - **overflow**: `x > cap_value` maps to a per-column **LARGE** discrete token and **does not** use the basis.
- `cap_value` is chosen per column from the training positives using a **quantile cap**:
  - `cap_value = min(cap_max, floor(quantile(q) * factor))`, with `q=0.99`, `factor=1.1`, `cap_max=10_000_000`.

This isolates “how much tail we model smoothly” (cap/overflow) from the basis construction itself, and gives both B-splines and SL the same overflow signal.

Setup (400k/400k/400k contiguous split):

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Train rows: `[0, 400000)`
- Val rows: `[400000, 800000)`
- Test rows: `[800000, 1200000)`
- Model: FwFM (`experiments/criteo_fwfm/model/fwfm.py`)
- Integer routing:
  - Integer columns with low unique count are routed to categorical (as in `experiments/criteo_fwfm/preprocess.py`).
  - Remaining integer columns use either B-splines or SL, depending on the variant.

Experiments:

1. **B-spline variant** (`bspline_integer_basis`)
   - Tune (Optuna, 100 trials): `lr`, `weight_decay`
   - Fixed: degree=3, knots=10, scalar input map `log1p(x)/log1p(cap)`
   - Config: `experiments/criteo_fwfm/config/model_bspline_quantile_large_token.yaml`

2. **SL variant** (`sl_integer_basis`)
   - Tune (Optuna, 100 trials):
     - `lr`, `weight_decay`
     - shared conductance (u-exp-valley): `u0`, `left_slope`, `right_slope`
     - shared potential (confining family): `V(u)=kappa*u^p` with `kappa`, `p`
   - Fixed: `num_basis=10`, `u=log1p(x)/log1p(cap)`
   - Config: `experiments/criteo_fwfm/config/model_sl_quantile_large_token.yaml`

