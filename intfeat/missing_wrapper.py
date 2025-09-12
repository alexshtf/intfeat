from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._mask import _get_mask  # internal, but stable for NaN masks


def _ensure_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


class MissingAwareColumnWrapper(BaseEstimator, TransformerMixin):
    """
    Add per-column missingness indicators and apply a base transformer either:
      * to the whole matrix (pass_through_missing=True), or
      * per column, only on non-missing rows (pass_through_missing=False).

    Parameters
    ----------
    base_transformer : TransformerMixin
        The sklearn transformer to wrap (e.g., SplineTransformer, KBinsDiscretizer).

    pass_through_missing : bool, default=False
        True: fit/transform a single cloned transformer on the entire input, passing NaNs through
              (only if the base transformer can handle NaNs).
        False: clone and fit one transformer per column on non-missing rows; at transform,
               missing rows in that column's block are filled with zeros.

    add_indicator : {'all','if_missing','none'}, default='all'
        Which missingness indicators to append (always appended at the end).

    indicator_dtype : data-type, default=np.int8
        dtype of indicator columns.

    handle_all_missing : {'skip','error'}, default='skip'
        Behavior when a column has all values missing during fit and pass_through_missing=False.
    """

    def __init__(
        self,
        base_transformer: TransformerMixin,
        *,
        pass_through_missing: bool = False,
        add_indicator: str = "all",
        indicator_dtype=np.int8,
        handle_all_missing: str = "skip",
    ):
        self.base_transformer = base_transformer
        self.pass_through_missing = pass_through_missing
        self.add_indicator = add_indicator
        self.indicator_dtype = indicator_dtype
        self.handle_all_missing = handle_all_missing

    # --------------------- fit ---------------------

    def fit(self, X, y=None):
        # Let scikit-learn set n_features_in_ and (if present) feature_names_in_
        Xv = validate_data(
            self,
            X,
            reset=True,
            ensure_2d=True,
            dtype=None,
            ensure_min_features=1,
            ensure_all_finite=False,  # allow NaNs
        )

        # Build internal input names used only for naming outputs
        if hasattr(self, "feature_names_in_"):
            input_names = list(map(str, self.feature_names_in_))
        else:
            input_names = [f"x{i}" for i in range(self.n_features_in_)]
        self._input_names_ = np.asarray(input_names, dtype=object)

        nan_mask = _get_mask(Xv, np.nan)
        self.had_missing_in_fit_ = nan_mask.any(axis=0)

        if self.pass_through_missing:
            self._global_transformer_ = clone(self.base_transformer).fit(Xv, y=None)

            # Try to obtain core names from the base transformer
            core_names = None
            if hasattr(self._global_transformer_, "get_feature_names_out"):
                try:
                    # Many sklearn transformers will fallback appropriately
                    core_names = list(
                        self._global_transformer_.get_feature_names_out(None)
                    )
                except Exception:
                    core_names = None
            if core_names is None:
                probe = Xv[:1, :]
                width = _ensure_dense(self._global_transformer_.transform(probe)).shape[
                    1
                ]
                core_names = [f"base__f{i}" for i in range(width)]

            self._feature_names_core_ = core_names

        else:
            self._per_col_transformers_: List[Optional[TransformerMixin]] = []
            self._per_col_width_: List[int] = []
            self._per_col_names_: List[List[str]] = []

            for j, col_name in enumerate(self._input_names_):
                col = Xv[:, [j]]
                nonmiss = ~nan_mask[:, j]

                if not np.any(nonmiss):
                    if self.handle_all_missing == "skip":
                        self._per_col_transformers_.append(None)
                        self._per_col_width_.append(0)
                        self._per_col_names_.append([])
                        continue
                    raise ValueError(
                        f"Column {col_name!r} is all missing during fit; "
                        "set handle_all_missing='skip' to ignore its transformed part."
                    )

                tr = clone(self.base_transformer).fit(col[nonmiss, :], y=None)

                names = None
                if hasattr(tr, "get_feature_names_out"):
                    try:
                        names = list(tr.get_feature_names_out([str(col_name)]))
                    except Exception:
                        names = None
                if names is None:
                    probe = col[nonmiss, :][:1, :]
                    width = _ensure_dense(tr.transform(probe)).shape[1]
                    names = [f"{col_name}__f{i}" for i in range(width)]

                self._per_col_transformers_.append(tr)
                self._per_col_width_.append(len(names))
                self._per_col_names_.append(names)

            self._feature_names_core_ = [
                nm for block in self._per_col_names_ for nm in block
            ]

        # Decide which columns get indicators and cache their integer indices
        if self.add_indicator == "none":
            indicator_idxs = np.array([], dtype=int)
        elif self.add_indicator == "if_missing":
            indicator_idxs = np.flatnonzero(self.had_missing_in_fit_)
        elif self.add_indicator == "all":
            indicator_idxs = np.arange(self.n_features_in_, dtype=int)
        else:
            raise ValueError("add_indicator must be one of {'all','if_missing','none'}")

        self._indicator_indices_ = indicator_idxs
        self._indicator_names_ = [
            f"{self._input_names_[k]}__nan_indicator" for k in indicator_idxs
        ]

        self._feature_names_out_ = self._feature_names_core_ + self._indicator_names_
        return self

    # --------------------- transform ---------------------

    def transform(self, X):
        # When fit saw named data, validate_data(reset=False) will check names and
        # emit the standard warning on mismatch — same as built-ins.
        fitted_attrs = ["_feature_names_out_"]
        if hasattr(self, "feature_names_in_"):
            fitted_attrs.append("feature_names_in_")
        check_is_fitted(self, fitted_attrs)

        Xv = validate_data(
            self,
            X,
            reset=False,
            ensure_2d=True,
            dtype=None,
            ensure_min_features=self.n_features_in_,
            ensure_all_finite=False,
        )

        nan_mask = _get_mask(Xv, np.nan)

        if self.pass_through_missing:
            core = _ensure_dense(self._global_transformer_.transform(Xv))
        else:
            parts: List[np.ndarray] = []
            for j in range(self.n_features_in_):
                tr = getattr(self, "_per_col_transformers_", [None])[j]
                if tr is None:
                    continue
                width = self._per_col_width_[j]
                col = Xv[:, [j]]
                nonmiss = ~nan_mask[:, j]

                out = np.zeros((Xv.shape[0], width), dtype=float)
                if np.any(nonmiss):
                    out[nonmiss, :] = _ensure_dense(tr.transform(col[nonmiss, :]))
                parts.append(out)

            core = (
                np.hstack(parts) if parts else np.empty((Xv.shape[0], 0), dtype=float)
            )

        if self._indicator_indices_.size:
            indicators = nan_mask[:, self._indicator_indices_].astype(
                self.indicator_dtype
            )
            X_out = np.hstack([core, indicators])
        else:
            X_out = core

        return X_out

    # --------------------- names / tags ---------------------

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        check_is_fitted(self, ["_feature_names_out_"])
        # Follow sklearn docs: if input_features is provided and we were fitted with names,
        # require equality (simple check to avoid private imports).
        if input_features is not None and hasattr(self, "feature_names_in_"):
            inp = np.asarray(input_features, dtype=object)
            if len(inp) != len(self.feature_names_in_) or np.any(
                inp != self.feature_names_in_
            ):
                raise ValueError(
                    "input_features must match feature_names_in_ used during fit."
                )
        return np.asarray(self._feature_names_out_, dtype=object)

    def _more_tags(self):
        # Output may contain NaNs if the underlying transformer emits them (when pass_through_missing=True).
        return {"allow_nan": True}
