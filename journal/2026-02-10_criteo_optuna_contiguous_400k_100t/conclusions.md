# Conclusions (Interim)

This experiment is still running (B-spline tuning in progress), so conclusions are provisional.

Key observations so far (2026-02-10 21:39 IST snapshot):

- Best validation logloss so far:
  - `baseline_winner`: `0.473367065759`
  - `bspline_integer_basis`: `0.471806406758`
- Current progress:
  - `bspline_integer_basis`: 62/100 trials complete
- B-splines are currently ahead on validation logloss despite SL's intended two-knob story (`w` vs `c`).

Working hypotheses for "why splines win" in this codebase:

- The B-spline path uses an explicit coordinate warp (`log1p_cap_to_unit`) and a local, non-oscillatory basis, which tends to be very sample-efficient for heavy-tailed count effects.
- The SL eigenbasis is global/oscillatory; with small `K` it can be less efficient for monotone-ish or sharply-local behavior unless `c` is chosen extremely carefully.
- The SL pmf estimate uses a per-bin `prior_count`; when the per-column cap/support is large, this can flatten `w` and weaken distribution adaptation.
- We currently only do truncation (first `K` eigenvectors). We do not apply eigenvalue-weighted shrinkage/regularization, which is where the “energy” interpretation of `c` often pays off.

Next:

- Wait for the run to finish to get final retrain val/test metrics per variant.
- Ensure a comparable SL run is evaluated under the same contiguous-split + retrain/test protocol.
