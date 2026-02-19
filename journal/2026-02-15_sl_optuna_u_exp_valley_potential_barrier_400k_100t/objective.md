# Objective

Test whether a **right-end barrier** potential can improve SL basis performance enough to beat B-splines on the
smallest contiguous split we have spline results for.

## Setup

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Split: contiguous blocks of size `400_000`
  - train: rows `[0, 400000)`
  - val: rows `[400000, 800000)`
  - test: rows `[800000, 1200000)`
- Reference (from `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/proceedings.md`):
  - `bspline_integer_basis` best-trial val logloss: `0.47180640675774904`

## Hypothesis

The confining potential `V(u)=kappa*u^p` did not beat splines. A **barrier** potential that becomes large near
the right boundary should more directly discourage eigenfunction amplitude and oscillations near `u≈1` (very large
integers), potentially yielding a basis that spends its sign changes in the head/ROI while keeping the `L2(w)`
notion of error intact.

## Experiment

- Variant: `sl_integer_basis` (FwFM)
- Conductance: `u_exp_valley` (tune `u0`, `left_slope`, `right_slope`)
- Potential (right barrier, in normalized log-space `u=log1p(x)/log1p(cap)`):
  - Family: `u_right_inverse_square`
  - Effective potential in symmetric eigenproblem: `V(u) = kappa / (1 - u + eps)^2`
  - Tune: `kappa`, `eps`
- Also tune: `lr`, `weight_decay`
- Keep histogram/pmf smoothing fixed (use `model_sl.yaml` defaults); do not tune `w` in this experiment.

## Success Criterion

Beat `0.471806` validation logloss by a meaningful margin on this split.

