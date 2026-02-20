# Objective

Run a larger contiguous-head benchmark on Criteo using the quantile-capped integer preprocessing with a per-column `LARGE` overflow token, and compare three model families on the same split:

- **All-spline**: `bspline_integer_basis` (all integer columns use B-splines).
- **Hybrid (fixed u-space right barrier)**: `hybrid_bspline_sl` where only `I5` uses SL, with conductance + potential frozen to the best 400k run from `journal/2026-02-19_hybrid_bspline_qcap_large_token_sl_i5_tune_400k_100t/` (trial 95).
- **Hybrid (fixed x-space inverse-square)**: `hybrid_bspline_sl` where only `I5` uses SL, with conductance + potential frozen to the best 400k run from `journal/2026-02-20_hybrid_bspline_qcap_large_token_sl_i5_inverse_square_400k_100t/` (trial 61).

## Split

Contiguous head blocks (chronological):

- Train: first **2,000,000** rows
- Validation: next **400,000** rows
- Test: next **400,000** rows

Ranges: `train=[0, 2,000,000)`, `val=[2,000,000, 2,400,000)`, `test=[2,400,000, 2,800,000)`.

## Tuning Policy

- **Optuna trials**: 100 per model.
- **No subsetting**: `tune_train_rows=0`, `tune_val_rows=0` (use the full split within each trial).
- **Tune only**: learning rate and weight decay.
- **Freeze** SL parameters (conductance + potential) for the two hybrid models (transfer sanity check).

## Fixed Training Setup

- `embedding_dim=8`
- `batch_size=256`
- `num_epochs=1`
- `bspline_knots=10`
- `sl_num_basis=10`

## Hypothesis

If the tuned SL "shape" for `I5` (conductance + potential) captures stable structure, it should transfer to a larger train prefix and remain competitive after re-tuning only optimizer hyperparameters.

