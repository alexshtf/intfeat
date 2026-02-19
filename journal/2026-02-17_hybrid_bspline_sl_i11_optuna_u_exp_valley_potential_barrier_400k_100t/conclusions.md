# Conclusions

## Result Summary

Hybrid model (400k/400k/400k contiguous):

- Integer columns: B-splines for all integer columns except `I11` (fixed `knots=10`)
- `I11`: SL basis with tuned conductance + tuned potential
- Tuned (Optuna): `lr`, `weight_decay`, `I11` SL conductance params, `I11` SL potential params

Best trial: `78`.

- Validation logloss: `0.47177167979007906`
- Test logloss: `0.4668762496723887`

Best hyperparameters (trial 78):

- `lr=0.0021449981209467326`
- `weight_decay=1.0897802505961267e-07`
- I11 conductance (u-exp-valley): `u0=0.5641659424306465`, `left_slope=0.6518595975615621`, `right_slope=0.5755890874474235`
- I11 potential (right barrier): `V(u)=kappa/(1-u+eps)^2` with `kappa=0.47349290941720057`, `eps=0.10380685981957297`

## Comparison To All-B-Spline (400k)

All-B-spline best refit (trial 37): `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37.json`

- B-spline val logloss: `0.47180640675774904`
- B-spline test logloss: `0.46680837257297464`

Deltas (hybrid - b-spline):

- Val: `-3.472696766998462e-05` (slightly better)
- Test: `+6.787709941408204e-05` (slightly worse)

## Takeaway

This hybrid (only `I11` switched to tuned SL) does not produce a convincing improvement over the best all-b-spline model on this split; differences are very small and the test result is slightly worse.

## Diagnostics Plots

- Terminal plots (I11): `journal/2026-02-17_hybrid_bspline_sl_i11_optuna_u_exp_valley_potential_barrier_400k_100t/trial78_i11_basis_conductance_potential_terminal.txt`
