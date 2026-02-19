# Conclusions (Stopped Early)

This experiment was stopped early (we terminated the process) and we treat the best-so-far Optuna results as final for decision-making.

Key observations (2026-02-10 22:48 IST snapshot):

- Best validation logloss (Optuna best trial value):
  - `baseline_winner`: `0.4733670657590525`
  - `bspline_integer_basis`: `0.47180640675774904` (72/100 trials complete when stopped)
- Refit best B-spline config (trial 37 hyperparameters; seed=42) for final eval:
  - final val logloss: `0.47180640675774904`
  - final test logloss: `0.46680837257297464`
  - details: `journal/2026-02-10_criteo_optuna_contiguous_400k_100t/bspline_refit_best_trial37.json`
- Comparison to SL (separate run):
  - Best SL(+potential right-barrier) val logloss (2026-02-15): `0.4723620800910768`
  - Gap to B-spline on val: `+0.0005556733333277331`

Working hypotheses for "why splines win" in this codebase:

- The B-spline path uses an explicit coordinate warp (`log1p_cap_to_unit`) and a local, non-oscillatory basis, which tends to be very sample-efficient for heavy-tailed count effects.
- The SL eigenbasis is global/oscillatory; with small `K` it can be less efficient for monotone-ish or sharply-local behavior unless `c` is chosen extremely carefully.
- The SL pmf estimate uses a per-bin `prior_count`; when the per-column cap/support is large, this can flatten `w` and weaken distribution adaptation.
- We currently only do truncation (first `K` eigenvectors). We do not apply eigenvalue-weighted shrinkage/regularization, which is where the “energy” interpretation of `c` often pays off.

Decision:

- Stop baseline/spline HPO for now.
- Next experiments should only run new SL variants (conductance families, `w` estimation variants, optional warps) until we beat the B-spline benchmark (`~0.471806` val logloss) by a meaningful margin.
