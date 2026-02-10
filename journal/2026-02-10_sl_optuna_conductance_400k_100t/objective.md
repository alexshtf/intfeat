# Objective

Tune the SL integer basis configuration for Criteo FwFM using Optuna, with a persistent trial database.

This experiment tunes both:

- optimization hyperparameters (`lr`, `weight_decay`), and
- SL conductance/curvature hyperparameters (`sl_alpha`, `sl_beta`, `sl_center`)

The goal is to test whether the intended SL knobs can yield a better validation logloss (and good test logloss) than baseline or splines under similar training settings.

