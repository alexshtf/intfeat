# Conclusions

## Results (400k contiguous blocks)

This study was launched with a target of 30 trials, but we stopped early after 9 completed trials and then finalized the study (re-ran the script with `--trials 9`, so `remaining=0` and it writes `results.json`).

- Best trial: `6`
- Best-trial val logloss: `0.4725625080548662`
- Final retrain val logloss: `0.4725625080548662`
- Final test logloss: `0.46743331439775554`

Best parameters:

- `lr = 0.0016409286730647919`
- `weight_decay = 1.0547383621352015e-07`
- `sl_u0 = 0.19561238231646708`
- `sl_left_slope = 2.612197333954733`
- `sl_right_slope = 2.733388665640456`

Artifacts:

- Summary: `journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/results.json`
- Study DB: `journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/study.sqlite3`
- Log: `journal/2026-02-11_sl_optuna_u_exp_valley_narrow_400k_30t/run.log`

## Conclusion

This narrower-search run did not beat the earlier `u_exp_valley` pilot result, and remains behind the B-spline benchmark.

Next experiment: resume the original `u_exp_valley` study that already found `0.472013...` and extend it with more trials, since TPE can now exploit around that good region.
