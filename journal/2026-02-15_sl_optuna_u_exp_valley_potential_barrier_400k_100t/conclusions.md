# Conclusions

This experiment **did not** beat B-splines on validation for the 400k/400k/400k contiguous split, but it did
improve over the confine potential on validation.

## Results

Reference (from `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/proceedings.md`):

- `bspline_integer_basis` best-trial val logloss: `0.47180640675774904` (Optuna stopped at 72/100 trials)
  - Refit best B-spline config (trial 37 hyperparameters; seed=42): test logloss `0.46680837257297464`
    (see `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37.json`)

This run (right-barrier potential `V(u)=kappa/(1-u+eps)^2`, tuning conductance + potential + optimizer):

- Best/final val logloss: `0.4723620800910768` (trial `80`)
- Final test logloss: `0.46680955940279606`
- Gap vs B-spline on val: `+0.0005556733333277331` (SL worse)
- Improvement vs confine-on-400k val: `-0.00017777065982255147`

Best params (trial 80):

- Optimizer: `lr=0.0014866479703934165`, `weight_decay=4.2328895342024825e-06`
- Conductance (`u_exp_valley`): `u0=0.09062994881433425`, `left_slope=13.469870646441864`, `right_slope=3.233536270863566`
- Potential (right barrier, `u_right_inverse_square`): `kappa=0.07363333964878228`, `eps=0.011951423525094157`

## Artifacts

- Summary JSON: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_barrier_400k_100t/results.json`
- Study DB: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_barrier_400k_100t/study.sqlite3`
- Full log: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_barrier_400k_100t/run.log`

## Decision

- Move to the next roadmap section: introduce a more flexible conductance family (heavy-tail + local ROI) and repeat
  the potential+conductance joint tuning on the 400k split.
