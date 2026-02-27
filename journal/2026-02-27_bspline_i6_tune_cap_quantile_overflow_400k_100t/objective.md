# Objective

Make the baseline "all-spline" comparison fairer against `I6=SL` by allowing **extra tuning**
for the B-spline encoding of **`I6`**.

We tune, for **`I6` only**:

- `cap_quantile` (quantile cap for positives)
- `positive_overflow` (`large_token` vs `clip_to_cap`)

and we also tune the optimizer:

- `lr`
- `weight_decay`

All other integer columns remain B-splines with the existing quantile-cap + LARGE-token settings.

Setup (400k/400k/400k contiguous split):

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Train rows: `[0, 400000)`
- Val rows: `[400000, 800000)`
- Test rows: `[800000, 1200000)`
- Variant: `bspline_integer_basis`

Baseline references:

- All-spline (quantile-cap + LARGE-token, global params): val `0.4731016777`, test `0.4678396553`
  - Stored: `journal/2026-02-19_bspline_vs_sl_quantilecap_large_token_400k_100t/bspline_results.json`
- Hybrid `I6=SL`: val `0.4721500809`, test `0.4673242588`
  - Stored: `journal/2026-02-26_hybrid_bspline_qcap_large_token_sl_i6_tune_400k_100t/results.json`

