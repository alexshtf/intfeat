# Objective

We previously picked integer columns to focus SL tuning on using heuristics from the FwFM runs.
To sanity-check that choice, fit a simple **XGBoost** model on the same **400k contiguous head**
training slice and rank the original Criteo columns (`I*` and `C*`) by importance.

This is not expected to match the FwFM + embedding setting exactly (different model class), but
it provides a quick proxy for which raw columns are predictive.

