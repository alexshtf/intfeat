# Conclusions

This transfer experiment **did not** beat B-splines on the larger split.

## Setup

- Split (contiguous head): train `[0, 1_000_000)`, val `[1_000_000, 1_200_000)`, test `[1_200_000, 1_400_000)`
- Optuna: `100` trials per variant, tuning only `lr` + `weight_decay`
- Training: CPU, `num_epochs=1`, `batch_size=256`

## Results (Best Trial == Final Retrain)

All numbers below are from `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/results.json`.

- `baseline_winner`
  - Best/final val logloss: `0.4587562087`
  - Final test logloss: `0.4589380908`
- `bspline_integer_basis` (`degree=3`, `knots=10`)
  - Best/final val logloss: `0.4575268364`
  - Final test logloss: `0.4577045677`
- `sl_integer_basis` (fixed `u_exp_valley`, `num_basis=10`)
  - Fixed conductance params:
    - `u0=0.12714125920232416`
    - `left_slope=0.10599526233947935`
    - `right_slope=3.508157381375444`
  - Best/final val logloss: `0.4581337128`
  - Final test logloss: `0.4584020767`

## Takeaways

- B-spline wins on both val and test on this split.
  - On val: SL is `+6.07e-04` worse than B-spline.
  - Both SL and B-spline beat the baseline winner encoding.
- The transferred conductance (`u_exp_valley`) appears to be at least *plausible* (SL beats baseline),
  but is not strong enough to beat splines here.

## Artifacts

- Summary JSON: `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/results.json`
- Study DB: `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/study.sqlite3`
- Full log: `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/run.log`
