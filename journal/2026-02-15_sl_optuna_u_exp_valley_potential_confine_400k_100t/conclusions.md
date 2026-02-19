# Conclusions

This experiment **did not** beat B-splines on validation for the 400k/400k/400k contiguous split.

## Results

Reference (from `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/proceedings.md`):

- `bspline_integer_basis` best-trial val logloss: `0.47180640675774904` (Optuna stopped at 72/100 trials)
  - Refit best B-spline config (trial 37 hyperparameters; seed=42): test logloss `0.46680837257297464`
    (see `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37.json`)

This run (confine potential `V(u)=kappa*u^p`, tuning conductance + potential + optimizer):

- Best/final val logloss: `0.4725398507508993` (trial `54`)
- Final test logloss: `0.4676545385640717`
- Gap vs B-spline on val: `+0.0007334439931502845` (SL worse)

Best params (trial 54):

- Optimizer: `lr=0.0016331572636421516`, `weight_decay=1.6640139559090427e-08`
- Conductance (`u_exp_valley`): `u0=0.8919522745212638`, `left_slope=0.46717497427765636`, `right_slope=9.537520074218435`
- Potential (confine, `u_power`): `kappa=8.085778071975742`, `power=3.2638685416591717`

## Artifacts

- Summary JSON: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/results.json`
- Study DB: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/study.sqlite3`
- Full log: `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/run.log`
- Best-trial conductance/potential/eigenfunction terminal plots:
  - `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/trial54_conductance_potential_eigenfunctions_terminal.txt`

## Decision

- Proceed to the next roadmap item: tune a **right-barrier** potential (`u_right_inverse_square`) jointly with conductance.
