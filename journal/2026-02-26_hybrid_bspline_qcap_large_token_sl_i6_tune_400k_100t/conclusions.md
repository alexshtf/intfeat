# Conclusions

This run completed all `100` Optuna trials and wrote `results.json` + `interaction_matrix.npz`.

## Headline

Hybrid `I6=SL` improves over the **quantile-cap + LARGE-token** all-spline baseline, and improves
over the prior `I5=SL` hybrid run. This supports the idea that we should spend SL tuning budget
on the most predictive integer columns (for this dataset, `I6` looks dominant by tree importance).

It is still worse than the older (non-quantile-cap) best spline baseline.

## Best Metrics

Hybrid (`I6=SL`, others B-spline; trial `74`):

- Val logloss: `0.4721500809`
- Test logloss: `0.4673242588`
- Stored: `journal/2026-02-26_hybrid_bspline_qcap_large_token_sl_i6_tune_400k_100t/results.json`

Baseline (all B-spline; quantile-cap + LARGE-token):

- Val logloss: `0.4731016777`
- Test logloss: `0.4678396553`
- Stored: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`

Hybrid reference (`I5=SL`, others B-spline; quantile-cap + LARGE-token):

- Val logloss: `0.4725881407`
- Test logloss: `0.4676060331`
- Stored: `journal/2026-02-19_hybrid_bspline_qcap_large_token_sl_i5_tune_400k_100t/results.json`

Reference (older best all-B-spline, without quantile-cap + LARGE-token):

- Val logloss: `0.4718064068`
- Test logloss: `0.4668083726`
- Stored: `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37_with_r.json`

Delta (this run - baseline):

- Val: `-0.0009515968`
- Test: `-0.0005153965`

Delta (this run - `I5=SL` hybrid):

- Val: `-0.0004380598`
- Test: `-0.0002817744`

## Best Hyperparameters (Trial 74)

- `lr=0.0014504996402307344`
- `weight_decay=4.0508276526550636e-08`
- Conductance (`u_exp_valley`): `u0=0.6857513173450294`, `left_slope=0.2101849309154991`, `right_slope=0.35149274000847375`
- Potential (`u_right_inverse_square`): `kappa=1.2917810814954327`, `eps=0.10950439575168318`

## Takeaways

- Picking a column with higher predictive power matters: `I6=SL` moved the needle more than `I5=SL`.
- Under the quantile-cap + LARGE-token engineering, the best one-column hybrid we have so far is `I6=SL`,
  but this capping scheme still seems to underperform the older best spline setup.

## Next Experiment

Two obvious "next" one-column hybrids to test (same setup, tune only that column's SL params):

- `I11=SL` (since `I11` is the second-most important integer column by XGBoost gain on 400k head).
- `I6=SL` fixed at this run's best params, and tune `I11=SL` on top (increases search space but may have higher upside).
