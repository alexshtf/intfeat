# Objective

Run the Criteo FwFM baselines on a larger contiguous split and test **transferability** of the best SL conductance found on the 400k split.

## Setup

- Dataset: `/home/alex/datasets/criteo_kaggle_challenge/train.txt`
- Split mode: contiguous head blocks (chronological)
  - train: rows `[0, 1000000)`
  - val: rows `[1000000, 1200000)`
  - test: rows `[1200000, 1400000)`
- Training: CPU, `num_epochs=1`, `batch_size=256`, early stopping patience set to `1`
- Optuna: tune `lr` and `weight_decay`

## Variants

1. `baseline_winner`: integer baseline encoder + FwFM.
2. `bspline_integer_basis`: B-spline integer basis (torchcurves), fixed knots:
   - `bspline_knots=10`
3. `sl_integer_basis`: Sturm–Liouville integer basis, **fixed** conductance transferred from the 400k run:
   - `sl_num_basis=10` (reused)
   - conductance family: `u_exp_valley`
   - fixed params (from 400k best trial 35):
     - `u0=0.12714125920232416`
     - `left_slope=0.10599526233947935`
     - `right_slope=3.508157381375444`

The hypothesis is that `c(u)` is (mostly) transferable across train sizes, so we fix it and only retune optimizer hyperparameters on the larger split.

