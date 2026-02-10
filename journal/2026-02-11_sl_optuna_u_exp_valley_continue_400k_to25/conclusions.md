# Conclusions

## Results (400k contiguous blocks)

This run continued the existing `u_exp_valley` study from 9 completed trials up to 25 completed trials.

- Best trial: `4`
- Best-trial val logloss: `0.4720136818700594`
- Final retrain val logloss: `0.4720136818700594`
- Final test logloss: `0.46676543009340093`

Best parameters (unchanged from the earlier pilot):

- `lr = 0.0016738085788752138`
- `weight_decay = 6.870101665590006e-08`
- `sl_u0 = 0.29214464853521815`
- `sl_left_slope = 0.6966418981413739`
- `sl_right_slope = 1.120548642504815`

Artifacts:

- Summary: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/results.json`
- Log: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/run.log`
- Checkpoint: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to25/checkpoint.json`
- Study DB (shared): `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/study.sqlite3`

## Conclusion

Extending the `u_exp_valley` study to 25 trials did not find a setting that beats the B-spline benchmark (`0.47180640675774904` val logloss). The best validation logloss remains `0.472013...`.

Next experiment: change the SL model capacity and/or basis shape (e.g., tune `sl_num_basis` and potentially `embedding_dim`, or introduce a richer conductance family such as multi-valley schedules or a coordinate defined by the CDF of `w`) and re-run tuning.
