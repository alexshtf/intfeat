# Conclusions

This run completed all `100` Optuna trials and wrote `results.json` + `interaction_matrix.npz`.

## Headline

Hybrid `I5=SL` beats the **quantile-cap + LARGE-token** all-spline baseline by a small but consistent margin on both validation and test, but it is still worse than the older (non-quantile-cap) best spline baseline.

## Best Metrics

Hybrid (`I5=SL`, others B-spline; trial `95`):

- Val logloss: `0.4725881407`
- Test logloss: `0.4676060331`
- Stored: `journal/2026-02-19_hybrid_bspline_qcap_large_token_sl_i5_tune_400k_100t/results.json`

Baseline (all B-spline; quantile-cap + LARGE-token):

- Val logloss: `0.4731016777`
- Test logloss: `0.4678396553`
- Stored: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`

Delta (hybrid - baseline):

- Val: `-0.0005135370`
- Test: `-0.0002336222`

Reference (older best all-B-spline, without quantile-cap + LARGE-token):

- Val logloss: `0.4718064068`
- Test logloss: `0.4668083726`
- Stored: `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37_with_r.json`

## Best Hyperparameters (Trial 95)

These are the tuned hyperparameters for this hybrid run:

- `lr=0.001711146366251425`
- `weight_decay=6.918032107891065e-08`
- Conductance (`u_exp_valley`): `u0=0.5679351275204167`, `left_slope=0.15004981827198377`, `right_slope=0.509481171284692`
- Potential (`u_right_inverse_square`): `kappa=0.7869681052568523`, `eps=0.01992236650781661`

Interpretation: the optimum prefers a fairly mild right-end barrier (small `kappa`, moderate `eps`) rather than a very sharp wall.

## Takeaways

- Per-column tuning can matter: allowing `I5` to learn its own SL basis gives a measurable improvement over “everything spline” under the same quantile-cap + LARGE-token scheme.
- The quantile-cap + LARGE-token engineering still appears to be a net negative vs the older “best spline” setup (it likely throws away information in the rare-but-very-large tail).

## Next Experiment

Run the same hybrid idea for **another high-impact integer feature** (e.g. `I7` or `I11`), starting from the same quantile-cap + LARGE-token baseline:

- Variant A: `I7=SL`, all others B-spline.
- Variant B: keep `I5=SL` fixed at its best params from this run, and tune `I7=SL` on top (more hyperparameters, but likely higher upside).
