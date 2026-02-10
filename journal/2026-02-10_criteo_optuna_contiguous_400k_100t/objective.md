# Objective

Compare integer feature strategies under a shared FwFM model on Criteo (Kaggle-format) using contiguous splits.

Variants:

- `baseline_winner`: "winner" log-squared bucketing of positive integers + categorical-style embeddings.
- `bspline_integer_basis`: map positive integers to a scalar (default `log1p_cap_to_unit`) and use a learned B-spline curve to produce embeddings/linear terms.

We tune `lr` and `weight_decay` via Optuna per variant, using validation logloss as the objective.

