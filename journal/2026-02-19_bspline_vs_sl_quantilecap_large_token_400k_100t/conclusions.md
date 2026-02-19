# Conclusions

Both runs completed.

## Metrics (400k/400k/400k contiguous split)

Reference (prior all-B-spline run, *without* quantile-cap + LARGE overflow):

- Val logloss: `0.47180640675774904`
- Test logloss: `0.46680837257297464`
- Source: `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37_with_r.json`

This experiment (quantile-cap + per-column LARGE overflow token):

- B-spline (`bspline_integer_basis`)
  - Best/final val logloss: `0.47310167772615197`
  - Final test logloss: `0.4678396552799285`
  - Best params: `lr=0.001417324920318447`, `weight_decay=5.297945691114149e-07`
  - Results: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`
- SL (`sl_integer_basis`, shared params across integer columns)
  - Best/final val logloss: `0.47236197516394707`
  - Final test logloss: `0.4668877449921019`
  - Best params:
    - `lr=0.0014577309788313892`, `weight_decay=4.0241732267213483e-07`
    - conductance (u-exp-valley): `u0=0.15690082766987387`, `left_slope=3.271669161177461`, `right_slope=0.7487886974656032`
    - potential (u_power): `kappa=2.9335554095906717`, `power=5.8467512273292295`
  - Results: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/sl_results.json`

## Runtime (per-trial, mean)

- B-spline: ~`300s`/trial training time
- SL: ~`22s` preprocessing + ~`99s` training per trial

## Takeaway

On this split, the quantile-cap + LARGE overflow bucket feature engineering **degraded** the B-spline baseline and did not make SL beat the prior best B-spline configuration.

Once both Optuna studies finish, we will summarize:

- best validation logloss (trial-best and final-retrain)
- test logloss for the best config
- comparison vs the prior all-B-spline baseline (~0.4718 val on the 400k split)
