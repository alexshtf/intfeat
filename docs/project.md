# intfeat Project Summary

## 1. What this repository is

`intfeat` is a research codebase for building feature maps for non-negative integer data, especially heavy-tailed count-like columns (for example, frequencies, event counts, and sparse ID-derived counts).

The main idea is to replace a raw integer value with a short vector of smooth basis coordinates that are adapted to:

- the observed data distribution, and
- a user-controlled curvature schedule.

In practical terms, this gives a compact embedding per integer column that can be fed into downstream ML models.

## 2. High-level architecture

At the center of the repository is a discrete 1-D spectral construction that looks like a weighted Sturm-Liouville problem on a finite grid.

- Grid: integer support `0..N`
- Node weights `w`: estimated from column data (histogram/pmf)
- Edge conductances `c`: produced by `CurvatureSpec`, controlling where basis functions are allowed to oscillate
- Solver: symmetric tridiagonal eigensolve (`scipy.linalg.eigh_tridiagonal`)

The first `K` eigenfunctions become the learned basis. Transforming a value is then a fast row lookup in the eigenvector matrix.

## 3. Repository layout

- `intfeat/`: main library code
- `docs/`: technical writeups and math intuition
- `notebooks/`: interactive demos and exploratory usage
- `experiments/`: script-style experiments for smoothing ideas
- `data_utils/`: Polars-based preprocessing utilities for integer columns
- `README.md`: user-facing quickstart
- `AGENTS.md`: maintenance and implementation notes for contributors/agents

## 4. Public API surface (`intfeat/__init__.py`)

The package re-exports the symbols below:

- `StrumLiouvilleTransformer`: sklearn-style multi-column transformer
- `StrumLiouvilleColumnTransformer`: single-column transformer
- `StrumLiouvilleBasis`: standalone basis object
- `HistogramFitter`, `CurvatureSpec`: base interface and curvature schedule definition
- `MissingAwareColumnWrapper`: missing-data wrapper with optional indicators
- `fit_laplacian_hist`, `LaplacianHistogramFitter`: spectral histogram smoothing utilities (experimental)
- `OrthPowerlawBasis`: alternative orthogonalization path for custom atoms
- `powerlaw_weight`, `gram_matrix`, `powerlaw_gram_matrix`: weighted Gram helpers
- `plot_sl_basis`: basis plotting helper

## 5. Core pipeline (single column)

Implemented mainly in `intfeat/strum_liouville_transformer.py`.

### Fit path

1. Keep only finite values.
2. Fit a histogram model (`HistogramFitter`) to get:
   - `pmf` (weights on integer support)
   - `tail_cutoff` (effective max index)
3. Build edge conductances from `CurvatureSpec.compute_weights`.
4. Compute the first `K` eigenpairs with `_compute_eigenfunctions(cs, ws, k)`.

### Transform path

1. Build a finite-value mask.
2. Clip finite values to `[0, tail_cutoff]`.
3. Floor/cast to integer indices (if input dtype is not integral).
4. Fill output rows:
   - non-missing rows get eigenvector rows
   - missing rows remain zeros

## 6. Multi-column pipeline

`StrumLiouvilleTransformer` wraps the column transformer:

- `fit(X)` loops over columns, fitting one column transformer per column.
- `transform(X)` transforms each column and horizontally stacks all blocks.

This behaves like a standard sklearn feature transformer, with missing values allowed in inputs.

## 7. Spectral construction details

Implemented in `intfeat/strum_liouville.py`.

- `_build_cs_matrix(cs)` builds a tridiagonal Laplacian-like matrix from conductances.
- `_compute_eigenfunctions(cs, ws, k)` forms:
  - `M` from `c`,
  - diagonal scaling from `w`,
  - symmetric operator `S = W^{-1/2} M W^{-1/2}`,
  then solves the first `k` eigenpairs via tridiagonal eigensolver.
- Eigenvectors are mapped back to weighted-space basis functions.

`StrumLiouvilleBasis` is a standalone callable basis object with manually supplied weight and curvature configs. It is useful for direct basis inspection and plotting.

## 8. Histogram fitting subsystem

Implemented in `intfeat/hist_fit.py`.

### `CutoffStrategy`

Chooses a cutoff to cap large values:

- if observed max is below `max_val`, keep full range;
- otherwise clip at a quantile-based cutoff with a margin.

### `KTHistogramFitter` (default)

- Integer histogram with optional value clipping.
- Applies additive smoothing (`prior_count`, default `0.5`).
- Exposes `pmf()`, `cdf()`, and `tail_cutoff()`.

This is the default fitter used by column transformers when no fitter is provided.

### `fit_laplacian_hist`

Projects a histogram into a low-frequency eigenspace of a tridiagonal operator for smoothing.

### `LaplacianHistogramFitter` (experimental)

Provides a wrapper around spectral smoothing, but currently has incomplete/fragile implementation details (see sharp edges section).

## 9. Missing-data wrapper

Implemented in `intfeat/missing_wrapper.py`.

`MissingAwareColumnWrapper` wraps any sklearn-like transformer and adds explicit missing indicators.

Two modes:

- `pass_through_missing=True`: fit/transform one global clone on full matrix (only if base transformer can handle NaNs).
- `pass_through_missing=False`: fit one clone per column on non-missing rows; missing rows are zero-filled in each transformed block.

Indicator options:

- `add_indicator='all'`: add one NaN flag per input column
- `add_indicator='if_missing'`: add only for columns that were missing during fit
- `add_indicator='none'`: no indicators

## 10. Alternative basis path

Implemented in `intfeat/orth.py` and `intfeat/gram.py`.

This path orthogonalizes user-defined atoms under power-law weights using a weighted Gram matrix and Cholesky factorization (`OrthPowerlawBasis`). It is separate from the Sturm-Liouville transformer path.

## 11. Data preprocessing utilities

Implemented in `data_utils/preprocess.py` (Polars).

- `LogSquaredBinner`: compresses large values using `2 + floor(log(x)^2)` for `x > 1`, preserving small values.
- `ClippingBinner`: clips each column at a learned quantile/max-based cutoff.

These utilities help make extremely long-tailed integer features easier to model.

## 12. Docs, notebooks, and experiments

### Docs

- `docs/strum_liouville_heavy_tailed_features.md`: full mathematical walkthrough for the discrete weighted spectral basis.
- `docs/laplacian_histogram_smoothing.md`: tutorial on spectral smoothing for bounded histograms.
- `docs/literature.md`: lightweight notes/pointers.

### Notebooks

`notebooks/` contains exploratory and demo notebooks for:

- basis behavior,
- histogram fitting,
- toy datasets,
- practical usage experiments.

### Experiments

`experiments/` has script-style experiments for:

- spectral histogram fit comparisons (`plot_spectral_histogram_fits.py`),
- convex smoothing variants (`plot_cvx_smoothers.py`).

## 13. Runtime and dependencies

- Python: `>=3.12` (`.python-version` is `3.12`)
- Package/dependency manager: `uv`
- Core numerical stack: NumPy, SciPy, scikit-learn
- Optional analysis stack includes Jupyter, Matplotlib/Seaborn, Polars, cvxpy, torch, and others listed in `pyproject.toml`

Typical commands:

- `uv sync`
- `uv run python3 -c "from intfeat import StrumLiouvilleTransformer; print(StrumLiouvilleTransformer)"`
- `uv run jupyter lab`

## 14. Current sharp edges and caveats

These are important for contributors and users:

- Naming typo is intentional for compatibility: `StrumLiouville*` (not `SturmLiouville*`).
- `StrumLiouvilleBasis.__call__` clips with `np.minimum(xs, self.max_val)` while valid indices are `0..self.max_val-1`; this can cause out-of-bounds indexing for exact `xs == max_val`.
- `StrumLiouvilleColumnTransformer.fit` passes finite values directly to the histogram fitter without explicit floor/cast; training data should already be integer-typed.
- `CurvatureSpec.compute_weights` can produce exact zeros for some parameter choices (`beta > 0` near `center`), which can create degenerate conductances.
- `StrumLiouvilleColumnTransformer.fit` does not return `self`, so it is not chainable in sklearn style.
- In `StrumLiouvilleTransformer`, passing a custom `hist_fitter` instance reuses that same object across all columns (shared mutable state).
- `LaplacianHistogramFitter` is incomplete/experimental:
  - references `self.num_coefs` without setting it in `__init__`,
  - `pmf` uses `self.apx_hist` (missing underscore), likely a bug.
- No dedicated unit test suite is present yet; notebooks and experiments are the current validation path.

## 15. How to extend the repository

### Add a new histogram estimator

1. Implement `HistogramFitter` methods:
   - `fit(data)`
   - `pmf(x=None)`
   - `tail_cutoff()`
2. Plug it into `StrumLiouvilleColumnTransformer(hist_fitter=...)`.
3. Consider re-exporting in `intfeat/__init__.py` if intended for public use.

### Add a new curvature schedule

1. Extend `CurvatureSpec` or add an equivalent spec object with a `compute_weights(xs)` method.
2. Ensure conductance weights are strictly positive (or floored) for stable eigensolves.

### Add new public functionality

1. Implement in `intfeat/`.
2. Re-export from `intfeat/__init__.py`.
3. Update `README.md`, `AGENTS.md`, and relevant notebooks.

## 16. Suggested reading order for new contributors

1. `README.md`
2. `AGENTS.md`
3. `docs/strum_liouville_heavy_tailed_features.md`
4. `intfeat/strum_liouville_transformer.py`
5. `intfeat/strum_liouville.py`
6. `intfeat/hist_fit.py`
7. `notebooks/strum_liouville_transformer.ipynb`
