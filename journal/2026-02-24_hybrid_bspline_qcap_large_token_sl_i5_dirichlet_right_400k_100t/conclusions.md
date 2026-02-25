# Conclusions

This run completed all `100` Optuna trials and wrote `results.json` + `interaction_matrix.npz`.

## Headline

Switching `I5`'s SL eigenproblem to **Dirichlet-right meshpoint** (instead of the default Neumann-like
right boundary) did **not** improve performance: the best hyperparameters and the final val/test
logloss match the prior Neumann-right hybrid run exactly.

## Best Metrics

Hybrid (`I5=SL`, others B-spline; `right_boundary=dirichlet_meshpoint`; trial `95`):

- Val logloss: `0.4725881407`
- Test logloss: `0.4676060331`
- Stored: `journal/2026-02-24_hybrid_bspline_qcap_large_token_sl_i5_dirichlet_right_400k_100t/results.json`

Reference (same hybrid setup, Neumann-right; trial `95`):

- Val logloss: `0.4725881407`
- Test logloss: `0.4676060331`
- Stored: `journal/2026-02-19_hybrid_bspline_qcap_large_token_sl_i5_tune_400k_100t/results.json`

Baseline (all B-spline; quantile-cap + LARGE-token):

- Val logloss: `0.4731016777`
- Test logloss: `0.4678396553`
- Stored: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`

Delta (this run - baseline):

- Val: `-0.0005135370`
- Test: `-0.0002336222`

## Best Hyperparameters (Trial 95)

These are the tuned hyperparameters (same as the Neumann-right hybrid run):

- `lr=0.001711146366251425`
- `weight_decay=6.918032107891065e-08`
- Conductance (`u_exp_valley`): `u0=0.5679351275204167`, `left_slope=0.15004981827198377`, `right_slope=0.509481171284692`
- Potential (`u_right_inverse_square`): `kappa=0.7869681052568523`, `eps=0.01992236650781661`

## Interpretation

The optimum uses a fairly sharp right-end barrier (small `eps`), which already heavily penalizes
mass at the far right end of the support. In that regime, changing only the boundary row of the
stiffness/Laplacian appears to be redundant for the first `K=10` eigenfunctions used by the model,
so it doesn't move the validation/test logloss.

