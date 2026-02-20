# Roadmap

## Goal (Criteo Conductance Tuning)

- Verify that the two SL knobs are useful in practice:
  - `w` (pmf / node weights) controls what errors matter (distribution adaptation).
  - `c` (conductance / edge weights) controls where variation is cheap vs expensive (region-of-interest).
- On Criteo, demonstrate that tuning `c` changes *where the low-frequency eigenmodes spend their sign changes* and can improve logloss beyond "uniform c" while keeping `w` fixed.
- Separately, demonstrate that tuning `w` (histogram estimation/smoothing) changes the basis in a different way than tuning `c`.
- Current benchmarks (Optuna best-trial val logloss):
  - 400k contiguous split: B-spline `0.47180640675774904` (historical benchmark).
  - 1M/200k/200k contiguous head split: B-spline `0.45752683640816766` (see `journal/2026-02-11_optuna_transfer_1m_reuse_sl_conductance/results.json`).

## Research Questions

- What conductance families can express heavy-tail structure (multi-scale behavior) while still letting us place a local "valley" (ROI) near small integers?
- How should we parameterize `c` (in x, log1p(x), or CDF_w coordinates) so that "ROI at x=10" means the same thing across different caps/support sizes?
- When (and how) does an explicit coordinate warp (e.g. log1p) become necessary, vs being representable by a good `c` schedule alone?

## Evaluation Protocol (Keep Fixed)

- Dataset: Criteo Kaggle `train.txt` (same path as existing journal entries).
- Goal: everything here is tuned to find the appropriate hyperparameter configuration (not a single fixed-config comparison).
- Splits: chronological/contiguous blocks (train/val/test) and also the default tail-holdout config; keep this explicit per experiment.
- Tuning schedule (chronological split):
  - Small: 200k train, 200k val, 200k test.
  - Larger (verification only): 1M train, 200k val, 200k test.
  - The larger schedule is only used once SL is beating splines on the small schedule (to verify the finding).
  - Exception: we may run the larger schedule for explicitly-labeled *transfer sanity checks* (e.g. reusing a conductance shape and re-tuning only optimizer hyperparameters).
  - Extra-large (transfer sanity checks only): 2M train, 400k val, 400k test.
  - We do NOT reuse hyperparameters from the small schedule on the larger schedule; we re-tune from scratch.
- Model: FwFM with the same `embedding_dim` and training loop across variants.
- Metrics: always report best-trial val logloss and final retrain val/test logloss.
- Seeds: at least 3 seeds for "final" claims (the 1-epoch 1-seed sweeps are for direction-finding only).

## Experiment Roadmap (Conductance)

1. Establish baselines (already started)
   - Baseline winner bucketing vs B-spline Optuna on contiguous split (stopped early; enough trials collected).
   - SL conductance Optuna (alpha/beta/center) on contiguous split (done).
   - Until SL variants beat the B-spline benchmark by a meaningful margin, do not spend more cycles on baseline/B-spline tuning.
2. Add a diagonal potential term `q(x)` (missing knob vs the continuous SL form)
   - Motivation: `q(x)` penalizes amplitude (an `L2(q)` term) rather than differences. This may help control tail amplitude of the first `K` modes without changing `w`-orthogonality.
   - Important: in the generalized eigenproblem `(L_c + diag(q)) phi = lambda diag(w) phi`, the symmetric tridiagonal that we actually eigensolve sees a diagonal term `q/w`.
     In other words, we should think in terms of an "effective potential" `V := q/w` that we design, then set `q = w * V`.
   - We want eigenfunctions whose amplitude is suppressed toward the right/tail (large x), not near zero.
     That means `V` should increase as we move right, or create a barrier near the right boundary.
   - Next two experiments (400k contiguous split first; only verify on 1M/200k/200k if we beat B-splines by a meaningful margin):
     - Confine potential (do this first): work in normalized log-coordinates `u = log1p(x)/log1p(cap) in [0, 1]` and use a monotone confining family
       `V(u) = kappa * u^p` with `kappa > 0`, `p >= 1`.
       Tune: `kappa`, `p`, and conductance hyperparameters jointly (plus training hyperparameters like `lr/wd`).
       Status: **done** (2026-02-15) on 400k split; best val `0.4725398508`, test `0.4676545386` (did not beat B-spline). See `journal/2026-02-15_sl_optuna_u_exp_valley_potential_confine_400k_100t/results.json`.
     - Right-barrier potential (do this second): also in `u`, use a right-end barrier
       `V(u) = kappa / (1 - u + eps)^2` with `kappa > 0`, `eps > 0`.
       Tune: `kappa`, `eps`, and conductance hyperparameters jointly (plus `lr/wd`).
       Status: **done** (2026-02-15) on 400k split; best val `0.4723620801`, test `0.4668095594` (did not beat B-spline). See `journal/2026-02-15_sl_optuna_u_exp_valley_potential_barrier_400k_100t/results.json`.
   - Success criterion: SL beats B-spline on the 400k contiguous split by a meaningful margin, then confirm on 1M/200k/200k.
3. Add conductance families that can represent heavy-tail + local ROI
   - Monotone heavy-tail: `c(x) = eps + ((x + x_shift) / x_scale)^p` (p>0).
   - U-shaped valley in log-space (ROI near x0): let `u=log1p(x)`, `u0=log1p(x0)`, set
     `c(x) = eps + exp(p*(u-u0)) + exp(q*(u0-u))`
     which is equivalent to power laws:
     `c(x) = eps + ((x+1)/(x0+1))^p + ((x0+1)/(x+1))^q` (p,q>0).
   - Same U-shape but define it in normalized coordinates `u = log1p(x)/log1p(cap)` so "x0=10" is stable when cap changes.
4. Sweep `c` while holding `w` fixed
   - Freeze `w` estimation settings (prior_count/cutoff) and run Optuna over just the conductance parameters
     (e.g. x0, p, q; or u0, p, q in normalized log-space).
   - Goal: show conductance alone can move performance and changes are consistent with "valley concentrates sign changes".
5. Sweep `w` while holding `c` fixed
   - Hold a chosen `c` family fixed and sweep histogram smoothing choices:
     `prior_count`, cutoff quantile/factor, maybe alternative histogram smoothing (if added).
   - Goal: show `w` changes the basis differently (distribution adaptation) and can be tuned independently.
6. Joint sweep (small, then confirm)
   - Joint Optuna over (lr, wd) + a compact conductance parameterization + `w` smoothing.
   - Confirm best settings with full train/val/test evaluation and multiple seeds.
7. Diagnostics to make "two knobs" falsifiable
   - For a few representative columns, log or plot where eigenmodes change:
     summary stats like `sum_x |phi_k(x+1)-phi_k(x)|` over bins, and where sign changes occur as k increases.
   - Record these diagnostics alongside the final logloss so we can tell if improved performance corresponds to the intended ROI behavior.
