# Objective

Test whether switching a single heavy-tailed integer column (`I5`) from B-splines to a tuned SL basis helps, while holding `I11` fixed to a previously-tuned SL basis and keeping all other integer columns as B-splines.

Setup (400k/400k/400k contiguous split):

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Train rows: `[0, 400000)`
- Val rows: `[400000, 800000)`
- Test rows: `[800000, 1200000)`
- Model: FwFM (`experiments/criteo_fwfm/model/fwfm.py`)
- Integer encodings:
  - `I5`: SL basis (tuned)
  - `I11`: SL basis (fixed to prior best)
  - all other integer columns: B-spline scalar input (`knots=10`)
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience `1`

What we tune (Optuna, 100 trials):

- `lr`
- `weight_decay`
- `I5` SL conductance (u-exp-valley): `u0`, `left_slope`, `right_slope`
- `I5` SL potential (right barrier): `V(u)=kappa/(1-u+eps)^2` with `kappa`, `eps`

What we fix:

- `I11` SL conductance/potential: fixed to the best params from
  `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/results.json` (trial 78)

