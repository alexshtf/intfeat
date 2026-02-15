# Objective

Beat the best B-spline validation logloss on the smallest contiguous split we have spline results for:

- Split: contiguous blocks of size `400_000`
  - train: rows `[0, 400000)`
  - val: rows `[400000, 800000)`
  - test: rows `[800000, 1200000)`
- Reference (from `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/proceedings.md`):
  - `bspline_integer_basis` best-trial val logloss: `0.47180640675774904`

Hypothesis:

- A diagonal potential term (effective `V := q/w` in the symmetric SL eigenproblem) that increases to the right
  will suppress eigenfunction amplitude in the far tail, producing basis functions that spend their oscillations
  where the model needs them (closer to the head/ROI) while keeping the `L2(w)` notion of error intact.

Experiment:

- Variant: `sl_integer_basis` (FwFM)
- Conductance: `u_exp_valley` (tune `u0`, `left_slope`, `right_slope`)
- Potential (confine, in normalized log-space): `V(u) = kappa * u^p`, where `u = log1p(x) / log1p(cap)`
  - Tune: `kappa`, `p`
- Also tune: `lr`, `weight_decay`
- Keep histogram/pmf smoothing fixed (use `model_sl.yaml` defaults); do not tune `w` in this experiment.

Success criterion:

- Beat `0.471806` validation logloss by a meaningful margin on this split, then proceed to the next potential family
  (right-barrier) per `journal/roadmap.md`.

