# Objective

Continue the existing `u_exp_valley` Optuna study (400k contiguous blocks) until we reach 100 completed trials, to fairly compare against the earlier 100-trial SL `CurvatureSpec` study.

Goal: beat the B-spline benchmark validation logloss on the same split:

- B-spline benchmark (best-trial val logloss): `0.47180640675774904`

This experiment continues the existing study storage:

- Study DB: `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/study.sqlite3`
- Study name: `sl_optuna_u_exp_valley_400k_60t`

