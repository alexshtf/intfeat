# Conclusions

## Results (400k contiguous blocks)

This study was launched with a target of 60 trials, but we stopped early after 9 completed trials and then finalized the study (re-ran the script with `--trials 9`, so `remaining=0` and it writes `results.json`).

- Best trial: `4`
- Best-trial val logloss: `0.4720136818700594`
- Final retrain val logloss: `0.4720136818700594`
- Final test logloss: `0.46676543009340093`

Best parameters:

- `lr = 0.0016738085788752138`
- `weight_decay = 6.870101665590006e-08`
- `sl_u0 = 0.29214464853521815`
- `sl_left_slope = 0.6966418981413739`
- `sl_right_slope = 1.120548642504815`

Artifacts:

- Summary: `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/results.json`
- Study DB: `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/study.sqlite3`
- Log: `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/run.log`
- Plot: `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/plots/conductance_best_trial4.png`

## Comparison (Validation Logloss)

- B-spline benchmark (best-trial): `0.47180640675774904`
- SL `CurvatureSpec` (best-trial): `0.47254733012305855`
- SL `u_exp_valley` (this study, best-trial): `0.4720136818700594`

Deltas:

- `u_exp_valley` vs B-spline: `+0.00020727511231036` (still worse)
- `u_exp_valley` vs `CurvatureSpec`: `-0.00053364825299915` (better)

## Conclusion

The `u_exp_valley` family (normalized log-space valley) meaningfully improves SL performance and gets very close to the B-spline baseline within just 9 trials, but does not beat it yet.

Next experiment: rerun `u_exp_valley` with a narrower, ROI-focused search space (restrict `sl_u0` around the current best `~0.29` and moderate slope ranges) with a larger trial budget, to test whether we can surpass B-splines on validation by a meaningful margin.
