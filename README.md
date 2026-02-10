## intfeat

Research code for representing heavy‑tailed integer/count features with discrete spectral (Sturm–Liouville‑style) bases.

For agent-oriented orientation and maintenance notes, see `AGENTS.md`.

### What’s in here

- `intfeat/`: library code (feature maps, histogram fitters, sklearn transformers)
- `docs/`: math + intuition writeups
- `notebooks/`: demos (primary “how to use”)
- `experiments/`: scripts exploring histogram smoothing / baselines
- `experiments/criteo_fwfm/`: config-driven Criteo FwFM comparison scripts (winner baseline vs SL vs B-spline)
- `data_utils/`: preprocessing helpers (Polars)

Start with `docs/strum_liouville_heavy_tailed_features.md` for the theory.

### Quickstart (uv)

This repo is set up to use `uv` and `uv.lock`:

```bash
uv sync
uv run python3 -c "from intfeat import StrumLiouvilleTransformer; print(StrumLiouvilleTransformer)"
uv run jupyter lab
```

If `uv` fails with a cache permission error, set a writable cache directory, e.g. `UV_CACHE_DIR=/tmp/uv_cache`.

### General Integer Encoding Pattern (Not Criteo-Specific)

Many ML models want each feature field to produce:

- an embedding vector `e(x) in R^d`, and
- a linear term `l(x) in R`.

For heavy-tailed integer/count features, a useful general pattern is:

- Use a **discrete path** for missing and “special” values (for example: missing, non-positive).
- Use a **basis expansion** for positive values:
  - Choose basis functions `B_1, ..., B_K` on integers (for example: a discrete Sturm–Liouville eigenbasis, or a spline basis).
  - For a value `n >= 1`, compute `B(n) = [B_1(n), ..., B_K(n)]`.
  - Learn per-field parameters `V in R^{K x d}` and `a in R^K` and set:
    - `e(n) = B(n) @ V`
    - `l(n) = B(n) @ a`

The experiments under `experiments/criteo_fwfm/` apply this pattern with an FwFM model, but the pattern itself is generic.

### Minimal usage (single integer column)

```python
import numpy as np
from intfeat import StrumLiouvilleColumnTransformer, CurvatureSpec

x_train = np.array([0, 1, 1, 2, 10, 10, 50], dtype=np.int64)
x_test = np.array([0, 1, 2, 10, 50, np.nan], dtype=float)
tr = StrumLiouvilleColumnTransformer(
    num_funcs=8,
    curvature_spec=CurvatureSpec(),  # uniform by default
)
tr.fit(x_train)
X_feat = tr.transform(x_test)  # shape: (len(x_test), <= num_funcs)
```

### Notes

- Notebooks may reference older API names (e.g. `curvature_gamma`); current code uses `curvature_spec=CurvatureSpec(...)`.
- Scripted experiments include targeted tests under `experiments/criteo_fwfm/tests/`.
