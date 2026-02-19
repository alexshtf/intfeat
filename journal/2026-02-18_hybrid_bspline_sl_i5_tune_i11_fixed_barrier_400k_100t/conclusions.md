# Conclusions

## Result Summary

Hybrid model (400k/400k/400k contiguous):

- Integer columns:
  - B-splines for all integer columns except `I5` and `I11` (fixed `knots=10`)
  - `I11`: SL basis with fixed conductance + fixed potential (copied from the prior best I11-only hybrid run)
  - `I5`: SL basis with tuned conductance + tuned potential
- Tuned (Optuna, 100 trials): `lr`, `weight_decay`, and the *I5-only* SL parameters (global SL params apply only to I5 due to per-column overrides for I11)

Best trial: `90`.

- Validation logloss: `0.4718875464682844`
- Test logloss: `0.4668398528148151`

Best hyperparameters (trial 90):

- `lr=0.0018982018326131338`
- `weight_decay=5.915858749259879e-08`
- I5 conductance (u-exp-valley): `u0=0.5990455762768341`, `left_slope=1.615133259988079`, `right_slope=0.929030675802002`
- I5 potential (right barrier): `V(u)=kappa/(1-u+eps)^2` with `kappa=0.4496834270130303`, `eps=0.02770244434064551`

Details: `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/results.json`

## Comparison To All-B-Spline (400k)

All-B-spline best refit (trial 37): `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37.json`

- B-spline val logloss: `0.47180640675774904`
- B-spline test logloss: `0.46680837257297464`

Deltas (this hybrid - b-spline):

- Val: `+8.113971053536062e-05` (worse)
- Test: `+3.148024184046472e-05` (worse)

## Comparison To I11-Only Hybrid (400k)

I11-only hybrid (SL only on `I11`): `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/results.json`

- I11-only hybrid val logloss: `0.47177167979007906`
- I11-only hybrid test logloss: `0.4668762496723887`

Deltas (this hybrid - I11-only hybrid):

- Val: `+0.00011586667820534524` (worse)
- Test: `-3.6396857573617325e-05` (slightly better)

## Takeaway

Switching `I5` from B-splines to tuned SL (with `I11` fixed to its tuned SL parameters) did not improve validation/test logloss on this 400k contiguous split.

## Diagnostics Plots

- Terminal plots (I5): `journal/2026-02-18_hybrid_bspline_sl_i5_tune_i11_fixed_barrier_400k_100t/i5_conductance_potential_basis_terminal.txt`
