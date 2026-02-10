# Roadmap

## Immediate

- Establish a reproducible workflow: experiment scripts in-repo, journal entries with commands, results, and conclusions.
- Criteo FwFM: understand why `bspline_integer_basis` currently outperforms SL variants (and whether SL can match it).

## Research Questions

- How to best realize the two intended SL knobs in practice:
  - `w`: distribution adaptation (what errors matter).
  - `c`: region-of-interest focus (where variation is cheap/expensive).
- When (and how) should we introduce an explicit coordinate warp (e.g. `log1p`-style) in the SL construction vs relying on `w`/`c` alone?

## Next Experiments

- Finish and summarize the running `criteo_optuna_contiguous_400k_100t` run (baseline vs B-spline).
- Run an in-repo Optuna sweep for `sl_integer_basis` under the same evaluation setup as the spline/baseline run.
- Explore SL constructions that behave more "local" in x-space (or that introduce an explicit warp) while retaining interpretable knobs.

