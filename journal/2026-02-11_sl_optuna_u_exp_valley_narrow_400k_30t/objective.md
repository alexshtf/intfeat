# Objective

Follow up on the `u_exp_valley` pilot by narrowing the conductance search space around the observed best region (`sl_u0 ~ 0.29`) while keeping histogram/pmf settings fixed.

We tune:

- optimization: `lr`, `weight_decay`
- conductance: `sl_u0`, `sl_left_slope`, `sl_right_slope`

Goal: beat the B-spline benchmark validation logloss on the same 400k contiguous split:

- B-spline benchmark (best-trial val logloss): `0.47180640675774904`

