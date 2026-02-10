# Objective

Jointly tune both SL knobs on Criteo (contiguous split):

- `c` via the heavy-tail-aware `u_exp_valley` conductance family (region-of-interest control)
- `w` via histogram/pmf smoothing parameters (distribution adaptation)

We tune:

- optimization: `lr`, `weight_decay`
- conductance: `sl_u0`, `sl_left_slope`, `sl_right_slope`
- histogram smoothing: `sl_prior_count`, `sl_cutoff_quantile`, `sl_cutoff_factor`

Goal: beat the B-spline best-trial validation logloss on the same split:

- B-spline benchmark (400k contiguous blocks): `0.47180640675774904`

