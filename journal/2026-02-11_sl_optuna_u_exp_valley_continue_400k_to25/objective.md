# Objective

Continue the original `u_exp_valley` Optuna study (which already found a good region near `0.472013...`) with additional trials so TPE can exploit around the best settings.

Goal: beat the B-spline benchmark validation logloss on the same 400k contiguous split:

- B-spline benchmark (best-trial val logloss): `0.47180640675774904`

This experiment continues the existing study storage:

- Study DB: `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/study.sqlite3`
- Study name: `sl_optuna_u_exp_valley_400k_60t`

