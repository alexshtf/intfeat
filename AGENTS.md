# intfeat: Agent Notes

This repo is a small research codebase for turning **heavy-tailed integer/count features** into smooth, low-dimensional **basis expansions** (feature maps) using discrete 1‑D spectral/Sturm–Liouville ideas.

## Purpose (What We Are Trying To Achieve)

We want an integer feature map that is simultaneously:

- **Distribution-adapted**: basis orthogonality / approximation error should respect where the data mass is (via node weights `w`).
- **Region-of-interest focused**: we should be able to concentrate "approximation power" in a user-chosen region of the integer axis (via edge conductances `c`).

The working hypothesis is that a discrete Sturm–Liouville eigenbasis provides two (partially) separable knobs:

- `w` controls *what errors matter* (an `L2(w)` notion of size/fit).
- `c` controls *where variation is cheap vs expensive* (a curvature/energy penalty over edges).

This is in contrast to spline bases where the knot placement and input warping typically couples "where you have resolution" with "what distribution you are adapted to".

## Quick Orientation

- Primary library code: `intfeat/`
- Research writeups (math + intuition): `docs/`
- Demos/experiments: `notebooks/`, `experiments/`
- Data preprocessing helpers (Polars): `data_utils/`
- Full repository summary: `docs/project.md`

The “public” imports are re-exported from `intfeat/__init__.py`.

## Core Mental Model

For integer values `x in {0, 1, 2, ...}`, build an orthonormal basis `φ_k(x)` by solving a weighted 1‑D eigenproblem on a truncated grid `x=0..N`.

- **Node weights** `w_x > 0`: a pmf/weighting over values (typically estimated from data).
- **Edge conductances** `c_x > 0`: a curvature schedule controlling where oscillations are cheap/expensive.

The implementation uses a symmetric tridiagonal eigensolve via `scipy.linalg.eigh_tridiagonal`.

Background (recommended first read): `docs/strum_liouville_heavy_tailed_features.md`.

## Research Journal (Process)

We keep a lightweight research journal in `journal/`:

- `journal/roadmap.md`: high-level roadmap (questions, hypotheses, near-term tasks).
- Each experiment has its own directory under `journal/` with:
  - `objective.md`: what we are testing and why
  - `proceedings.md`: commands, notes, intermediate results
  - `conclusions.md`: final readout and next steps

The journal should always include the CLI command(s) needed to reproduce a run, and those commands should reference scripts/configs that live in this repo (not `/tmp`).

## Math Notes (Process)

We maintain a small `math/` directory with summaries of mathematical results that we repeatedly use for reasoning (e.g., variational characterizations, oscillation theorems for tridiagonals, eigenvalue perturbation bounds, graph Laplacian facts).

When we learn something new (and it is reasonably standard / non-esoteric), we add it to `math/` with a short explanation and references.

## Main Data Flow (Column → Features)

1. Fit a histogram/pmf `w` for a single integer column (and choose a cutoff `N`) via a `HistogramFitter`.
   - Default in the transformer: `KTHistogramFitter` in `intfeat/hist_fit.py`.
2. Build conductances `c` from `CurvatureSpec` over edges `0..N-1`.
3. Compute the first `K` eigenfunctions via `_compute_eigenfunctions(c, w, K)`.
4. Transform each value `x` into the vector `[φ_0(x), ..., φ_{K-1}(x)]` (after clipping/flooring).
5. For a matrix `X`, do this per column and `hstack` the results.

## Key Modules (and what to edit)

- `intfeat/strum_liouville.py`
  - `_compute_eigenfunctions(cs, ws, k)`: builds the tridiagonal operator and computes eigenpairs.
  - `StrumLiouvilleBasis`: “function basis” object for `x -> φ(x)`.
- `intfeat/strum_liouville_transformer.py`
  - `StrumLiouvilleColumnTransformer`: fits per-column `w` and `c`, then maps values to eigenfunction rows.
  - `StrumLiouvilleTransformer`: sklearn-style multi-column wrapper (fit each column, then concatenate).
- `intfeat/hist_fit.py`
  - `KTHistogramFitter`: simple histogram w/ cutoff + prior smoothing; exposes `pmf()` + `tail_cutoff()`.
  - `fit_laplacian_hist(...)`: spectral projection smoother for bounded histograms (see also `docs/laplacian_histogram_smoothing.md`).
  - `LaplacianHistogramFitter`: experimental alternative histogram smoother.
- `intfeat/orth_base.py`
  - `HistogramFitter` interface and `CurvatureSpec` (defines conductance weights over edges).
- `intfeat/missing_wrapper.py`
  - `MissingAwareColumnWrapper`: adds NaN indicators + applies a base transformer safely with missing data.
- `data_utils/preprocess.py`
  - Polars binners for compressing/clipping huge integer ranges before fitting.
- `intfeat/orth.py` and `intfeat/gram.py`
  - Alternative “orthogonalize custom atoms under power-law weights” path (not used by the SL transformer).

## Running / Reproducing

- Python version: `3.12` (see `.python-version`).
- Dependency management: `uv` (see `uv.lock`).

Common commands:

- `uv sync`
- `uv run python3 -c "from intfeat import StrumLiouvilleTransformer; print(StrumLiouvilleTransformer)"`
- `uv run jupyter lab` (notebooks are the main demos)

If `uv` fails with a cache permission error (for example: `failed to open file ~/.cache/uv/...: Permission denied`),
run with a writable cache directory, e.g. `UV_CACHE_DIR=/tmp/uv_cache uv ...`.

There are currently no unit tests; the notebooks and `experiments/` scripts are the main validation harnesses.

## Notebooks vs Current API (important)

Some notebooks may reference older parameter names like `curvature_gamma`. The current transformer API uses:

- `curvature_spec=CurvatureSpec(...)`
- `hist_fitter=...`

If you change constructor args or behavior, expect to update notebooks in `notebooks/`.

## Sharp Edges / Known Issues

- Naming: `StrumLiouville*` is a misspelling of “Sturm–Liouville”; kept for continuity.
- Indexing: `StrumLiouvilleBasis.__call__` clips with `np.minimum(xs, self.max_val)`, but valid indices are `0..self.max_val-1`.
  - If you touch this code, consider tightening the clip to `self.max_val - 1` to avoid out-of-bounds.
- `StrumLiouvilleColumnTransformer.fit` passes finite values straight into the histogram fitter (it does **not** floor/cast first).
  - Keep training data integer-typed (or swap in a histogram fitter that can safely coerce floats).
- `CurvatureSpec.compute_weights` can return exact zeros when `beta > 0` and `center` coincides with a scaled edge index.
  - Many spectral constructions assume strictly-positive conductances; if you see degenerate spectra, consider flooring `c` to a small epsilon.
- `LaplacianHistogramFitter` in `intfeat/hist_fit.py` appears incomplete/experimental (it references `self.num_coefs` but doesn’t set it).

## If You Add New Functionality

- New histogram estimator: implement `HistogramFitter` (`fit`, `pmf`, `tail_cutoff`) in `intfeat/hist_fit.py` (or a new module) and thread it through `StrumLiouville*Transformer`.
- New curvature schedule: extend `CurvatureSpec` (or add a new spec object) in `intfeat/orth_base.py`.
- If you add new public API, re-export it from `intfeat/__init__.py` and update `README.md`.
