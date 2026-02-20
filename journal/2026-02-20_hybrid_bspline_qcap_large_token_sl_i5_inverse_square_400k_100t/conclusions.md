# Conclusions

This run completed all `100` Optuna trials and wrote `results.json` + `interaction_matrix.npz`.

## Headline

Switching `I5` to an SL basis with an **x-space inverse-square** potential slightly improves over both:

- the **quantile-cap + LARGE-token** all-spline baseline, and
- the prior hybrid `I5=SL` run that used the `u_right_inverse_square` (right-edge barrier in `u`).

The gains are small (on the order of `1e-4` to `1e-3` in logloss), so we should treat this as directional evidence rather than a decisive win.

## Best Metrics

Hybrid `I5=SL` with `inverse_square` potential (trial `61`):

- Val logloss: `0.4724508504`
- Test logloss: `0.4675406160`
- Stored: `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/results.json`

Prior hybrid `I5=SL` with `u_right_inverse_square` potential:

- Val logloss: `0.4725881407`
- Test logloss: `0.4676060331`
- Stored: `journal/2026-02-19_hybrid_bspline_qcap_large_token_sl_i5_tune_400k_100t/results.json`

Baseline (all B-spline; quantile-cap + LARGE-token):

- Val logloss: `0.4731016777`
- Test logloss: `0.4678396553`
- Stored: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`

Delta (inverse-square hybrid - u-right-barrier hybrid):

- Val: `-0.0001372904`
- Test: `-0.0000654171`

Delta (inverse-square hybrid - all-spline baseline):

- Val: `-0.0006508273`
- Test: `-0.0002990393`

## Best Hyperparameters (Trial 61)

- `lr=0.0018783311607265317`
- `weight_decay=1.0638596069456267e-08`
- Conductance (`u_exp_valley`): `u0=0.5812346533528041`, `left_slope=0.39896576053893634`, `right_slope=9.912300076171999`
- Potential (`inverse_square`): `kappa=29.028166370848655`, `x0=355.1417847878527`

## Interpretation

`inverse_square` is a much gentler tail penalty than the right-end barrier in `u`, so it is less likely to completely suppress variation for large `x`. Empirically it does slightly better here.

## Next Experiment

Repeat the same potential comparison for another important feature (likely `I11` or `I7`), and/or allow **per-column** potentials while keeping B-splines on other integer fields.
