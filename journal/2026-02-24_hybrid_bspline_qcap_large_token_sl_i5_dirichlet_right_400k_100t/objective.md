# Objective

Test whether changing the discrete SL eigenproblem boundary condition for **only `I5`**
to **Dirichlet-right meshpoint** improves validation loss, in the hybrid setup where:

- Variant: `hybrid_bspline_sl`
- Integer encodings:
  - `I5`: SL basis (tuned)
  - all other integer columns: B-splines (fixed)
- Capping/overflow scheme (quantile-cap + per-column LARGE token):
  - values `<= 0`: treated as categorical tokens (per-column)
  - positive values:
    - `1..cap_value`: use basis
    - `> cap_value`: per-column `LARGE` discrete token, no basis
  - `cap_value`: per-column quantile cap from train positives:
    - `cap_value = min(cap_max, floor(q-quantile * factor))`
    - `q=0.99`, `factor=1.1`, `cap_max=10_000_000`

Motivation:

- With the default Neumann-like right boundary, we often see eigenfunctions flatten to a
  **nonzero constant** near the right end of the support, even with tail-oriented potentials.
- A Dirichlet-right meshpoint condition modifies only the last row of the stiffness/Laplacian,
  and should encourage eigenfunctions to bend down towards the boundary, improving tail decay
  behavior at the cap.

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
- `I5` SL potential (u-right inverse-square): `V(u)=kappa/(1-u+eps)^2`, tuning `kappa`, `eps`

What is fixed:

- Right boundary: `right_boundary=dirichlet_meshpoint` (only affects SL; only `I5` uses SL here).

Baseline references (completed):

- All integer columns B-splines, quantile-cap + LARGE token:
  - Val logloss: `0.4731016777`
  - Test logloss: `0.4678396553`
  - Stored: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`
- Hybrid `I5=SL` with **Neumann-right** (default) + u-right-barrier potential (`u_right_inverse_square`):
  - Val logloss: `0.4725881407`
  - Test logloss: `0.4676060331`
  - Stored: `journal/2026-02-19_hybrid_bspline_qcap_large_token_sl_i5_tune_400k_100t/results.json`
- Hybrid `I5=SL` with **Neumann-right** (default) + x inverse-square potential (`inverse_square`):
  - Val logloss: `0.4726930034`
  - Test logloss: `0.4677482637`
  - Stored: `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/results.json`

