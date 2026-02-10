# Conclusions

## Results (400k contiguous blocks)

This study was launched with a target of 20 trials, but we stopped early after 7 completed trials and then finalized the study (re-ran the script with `--trials 7`, so `remaining=0` and it writes `results.json`).

- Best trial: `2`
- Best-trial val logloss: `0.47389764797396333`
- Final retrain val logloss: `0.47389764797396333`
- Final test logloss: `0.46838266806114087`

Best parameters:

- `lr = 0.0004059611610484307`
- `weight_decay = 1.4077923139972383e-05`
- `sl_u0 = 0.2727780074568463`
- `sl_left_slope = 0.5106823667399514`
- `sl_right_slope = 1.433387578475623`
- `sl_prior_count = 0.003280829084730048`
- `sl_cutoff_quantile = 0.9643150877782256`
- `sl_cutoff_factor = 1.3663618432936917`

Artifacts:

- Summary: `journal/2026-02-10_sl_optuna_u_exp_valley_joint_400k_20t/results.json`
- Study DB: `journal/2026-02-10_sl_optuna_u_exp_valley_joint_400k_20t/study.sqlite3`
- Log: `journal/2026-02-10_sl_optuna_u_exp_valley_joint_400k_20t/run.log`

## Conclusion

Under this small trial budget, the joint search over (`c` via `u_exp_valley`) and (`w` via histogram smoothing) did not match the earlier `u_exp_valley` performance with fixed histogram settings, and it is far from beating splines.

Interpretation: joint tuning adds significant search dimension; with limited trials it mostly explores poorly-conditioned regions. A better next step is to keep histogram settings fixed while we finish getting `c` to decisively beat splines (or alternately, sweep histogram settings with `c` fixed near a good setting to isolate the effect of `w`).
