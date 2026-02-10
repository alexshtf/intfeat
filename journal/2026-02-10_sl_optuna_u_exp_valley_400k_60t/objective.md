# Objective

Tune a heavy-tail-aware SL conductance family for Criteo FwFM that can place a local "valley" (region of interest) in normalized log-space:

- conductance family: `u_exp_valley`
  - edges use `u = log1p(edge) / log1p(max_edge)` (so `u in [0, 1]`)
  - conductance is minimized at `u0` and grows away from it with asymmetric slopes:
    - left slope: `left_slope`
    - right slope: `right_slope`

We tune:

- optimization hyperparameters: `lr`, `weight_decay`
- conductance hyperparameters: `sl_u0`, `sl_left_slope`, `sl_right_slope`

Goal: beat the current contiguous-split B-spline benchmark validation logloss:

- B-spline best-trial val logloss (400k contiguous blocks): `0.47180640675774904`

