# Criteo FwFM Experiments

This folder contains script-based experiments for comparing three integer feature strategies under the same FwFM model:

1. `baseline_winner`
2. `sl_integer_basis`
3. `bspline_integer_basis`

## What Each Variant Means

All variants share:

- Categorical columns: ordinal encoding with `min_count` filtering, plus per-column `__MISSING__` and `__INFREQUENT__` tokens.
- Same downstream model: an FwFM with per-field embeddings + per-field linear terms, a global bias, and pairwise field interactions.

### `baseline_winner`

This is a close variant of a common "Criteo winners" feature-engineering approach:

- Integers:
  - Missing: mapped to a per-column missing token.
  - Non-positive values (`<= 0`): mapped to per-value tokens `T{n}` (so each distinct non-positive value gets its own token).
  - Positive values (`>= 1`): bucketed via `q = floor(log(n)^2)` and mapped to tokens `S{q}`.
- Each column has its own token vocabulary (no sharing across fields).
- The model consumes token ids through per-field `nn.Embedding` tables for the embedding vector and for the linear term.

### `sl_integer_basis`

Integers are represented as a mixture of a discrete path and an SL basis expansion:

- Missing: mapped to a per-column missing token id (discrete).
- Non-positive values (`<= 0`): mapped to per-value discrete token ids (like the baseline).
- Positive values (`>= 1`): mapped to a basis row `B(n) in R^K` from a discrete Sturm–Liouville eigenbasis, then used as
  - embedding: `sum_i B_i(n) * v_i`
  - linear term: `sum_i B_i(n) * a_i`
  where `{v_i}` / `{a_i}` are learned per-field parameters.
- The basis is recomputed per column from:
  - a pmf estimate `w` (distribution adaptation),
  - a conductance schedule `c` from `CurvatureSpec` (region-of-interest / smoothness via edge penalties),
  - a cap/cutoff (to keep the eigenproblem finite).

### `bspline_integer_basis`

Integers are represented as a mixture of a discrete path and a learned B-spline curve:

- Missing + non-positive values: same discrete handling as `sl_integer_basis`.
- Positive values: mapped to a scalar in `[out_min, out_max]` (default: a `log1p` warp to `[0,1]` followed by affine scaling),
  then fed into `torchcurves.BSplineCurve` to produce:
  - embedding curve output (vector)
  - linear curve output (scalar)

## What Is Criteo-Specific vs General

General (reusable across datasets):

- The FwFM model family (field embeddings + field linear terms + pairwise field interactions).
- The “discrete + continuous” integer routing:
  - missing / non-positive handled as discrete ids,
  - positive values mapped to either `B(n) in R^K` (SL basis) or to a scalar fed into a spline curve.
- The basis-parameterization of embeddings and linear terms:
  - `e(n) = B(n) @ V`
  - `l(n) = B(n) @ a`

Criteo-specific (quirks / conventions for this experiment suite):

- Input schema and file format: tab-separated `label, I1..I13, C1..C26` without a header.
- The `baseline_winner` integer bucketing (`floor(log(n)^2)` buckets and `T{n}` / `S{q}` token strings) is motivated by common Criteo CTR baselines.
- The default split logic in configs is designed around the Criteo file’s row order (contiguous/tail holdout).

Other datasets may have:

- real-valued numeric (float) features that should not be treated as integers,
- different missingness conventions,
- different “special value” semantics (e.g. `0` as a valid measurement vs “not present”),
- different evaluation split requirements (random vs time-based vs group-based).

## Dataset structure

For the Criteo Display Advertising Challenge format, each row is:

1. `label`
2. `I1..I13` integer-like fields
3. `C1..C26` categorical/hash-like fields

Important: public sources document schema but do **not** disclose per-column semantic meaning for individual `I*`/`C*` fields.

- Criteo 1TB dataset docs: <https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/>
- Kaggle challenge page: <https://www.kaggle.com/c/criteo-display-ad-challenge>
- HF mirror notes: <https://huggingface.co/datasets/criteo/CriteoClickLogs>

## Integer low-cardinality routing

Some integer columns may have very few distinct values. If `encoding.integer.low_cardinality_as_categorical=true`, integer fields with train distinct count `< encoding.integer.low_cardinality_threshold` are routed to categorical encoding.

Override lists are available:

1. `encoding.integer.force_integer_path`
2. `encoding.integer.force_categorical_path`

## Configuration

Use YAML files and optional CLI overrides.

### Example: baseline medium profile

```bash
uv run python -m experiments.criteo_fwfm.run \
  --config experiments/criteo_fwfm/config/model_baseline.yaml \
  --config experiments/criteo_fwfm/config/profile_medium.yaml \
  --set data.path=/path/to/criteo/train.txt
```

### Example: SL with custom cap and curvature

```bash
uv run python -m experiments.criteo_fwfm.run \
  --config experiments/criteo_fwfm/config/model_sl.yaml \
  --config experiments/criteo_fwfm/config/profile_medium.yaml \
  --set data.path=/path/to/criteo/train.txt \
  --set model.integer.sl.cap_max=10000000 \
  --set model.integer.sl.curvature.alpha=1.0 \
  --set model.integer.sl.curvature.beta=0.6 \
  --set model.integer.sl.curvature.center=0.05
```

### Example: B-spline normalization and clamp controls

```bash
uv run python -m experiments.criteo_fwfm.run \
  --config experiments/criteo_fwfm/config/model_bspline.yaml \
  --config experiments/criteo_fwfm/config/profile_medium.yaml \
  --set data.path=/path/to/criteo/train.txt \
  --set model.integer.bspline.normalize_fn=clamp \
  --set model.integer.bspline.normalization_scale=1.0 \
  --set model.integer.bspline.input_map=log1p_cap_to_unit \
  --set model.integer.bspline.cap_max=10000000
```

## Artifacts

Each run creates a timestamped subdirectory under `artifacts/criteo_fwfm/` with:

1. `config.resolved.yaml`
2. `run.log`
3. `metrics.json`
4. `history.json`
5. `field_stats.json` (routing + field stats)
6. `model.pt` (if enabled)
7. `encoder_state.pkl`
8. `predictions_val.parquet` and `predictions_test.parquet` when parquet extras are installed; otherwise CSV fallbacks are written.
