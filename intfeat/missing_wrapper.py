from typing import List, Optional, Sequence
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
    _get_feature_names,
    _check_feature_names_in,
)
from sklearn.utils._mask import _get_mask  # internal helper for NaN masks


def _ensure_dense(X):
    # Handle sparse outputs from some transformers (e.g., onehot encodings).
    # Avoid importing scipy explicitly; rely on duck-typing.
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


class MissingAwareColumnWrapper(BaseEstimator, TransformerMixin):
    """
    Add per-column missingness indicators and apply a transformer either:
      * to the whole matrix as a single transformer (pass_through_missing=True), or
      * per column, operating only on non-missing rows and filling missing rows with zeros.

    Parameters
    ----------
    base_transformer : TransformerMixin
        The sklearn transformer to wrap (e.g., SplineTransformer, KBinsDiscretizer).

    pass_through_missing : bool, default=False
        - True: fit/transform a single cloned transformer on the entire input, passing NaNs through.
                 Use only if the base transformer can handle NaNs.
        - False: clone and fit one transformer per column on non-missing rows; at transform,
                 missing rows in that column's block are filled with zeros.

    add_indicator : {'all','if_missing','none'}, default='all'
        Which missingness indicators to append (they are always appended at the end).

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
        Xv = validate_data(
            self,
            X,
            reset=True,
            ensure_2d=True,
            dtype=None,
            ensure_min_features=1,
            ensure_all_finite=False,  # allow NaNs
        )
        n_samples, n_features = Xv.shape

        in_names = _get_feature_names(X)
        if in_names is None:
            in_names = list(getattr(X, "columns", [])) or [
                f"x{i}" for i in range(n_features)
            ]
        self.feature_names_in_ = np.asarray(in_names, dtype=object)

        nan_mask = _get_mask(Xv, np.nan)
        self.had_missing_in_fit_ = nan_mask.any(axis=0)
        self._name_to_idx_ = {name: j for j, name in enumerate(self.feature_names_in_)}

        if self.pass_through_missing:
            # One transformer for the whole matrix
            self._global_transformer_ = clone(self.base_transformer).fit(Xv, y=None)

            core_names = None
            if hasattr(self._global_transformer_, "get_feature_names_out"):
                try:
                    core_names = list(
                        self._global_transformer_.get_feature_names_out(
                            list(map(str, self.feature_names_in_))
                        )
                    )
                except Exception:
                    core_names = None
            if core_names is None:
                # Fallback: infer width from one-row transform (NaNs allowed here)
                probe = Xv[:1, :]
                width = _ensure_dense(self._global_transformer_.transform(probe)).shape[
                    1
                ]
                core_names = [f"base__f{i}" for i in range(width)]

            self._feature_names_core_ = core_names
        else:
            # Per-column transformers on non-missing rows
            self._per_col_transformers_: List[Optional[TransformerMixin]] = []
            self._per_col_width_: List[int] = []
            self._per_col_names_: List[List[str]] = []

            for j, col_name in enumerate(self.feature_names_in_):
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
                    # Infer width via a single non-missing sample
                    probe = col[nonmiss, :][:1, :]
                    width = _ensure_dense(tr.transform(probe)).shape[1]
                    names = [f"{col_name}__f{i}" for i in range(width)]

                self._per_col_transformers_.append(tr)
                self._per_col_width_.append(len(names))
                self._per_col_names_.append(names)

            self._feature_names_core_ = [
                nm for block in self._per_col_names_ for nm in block
            ]

        # Indicators: always appended at the end
        if self.add_indicator == "none":
            indicator_cols = []
        elif self.add_indicator == "if_missing":
            indicator_cols = [
                n
                for n, had in zip(self.feature_names_in_, self.had_missing_in_fit_)
                if had
            ]
        elif self.add_indicator == "all":
            indicator_cols = list(self.feature_names_in_)
        else:
            raise ValueError("add_indicator must be one of {'all','if_missing','none'}")

        self._indicator_cols_ = list(map(str, indicator_cols))
        self._indicator_names_ = [f"{c}__nan_indicator" for c in self._indicator_cols_]

        self._feature_names_out_ = self._feature_names_core_ + self._indicator_names_

        return self

    # --------------------- transform ---------------------

    def transform(self, X):
        fitted_attrs = ["_feature_names_out_", "feature_names_in_"]
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

        in_names = _get_feature_names(X)
        if in_names is not None:
            _check_feature_names_in(self, in_names)

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

                # Allocate zeros for all rows, then fill the non-missing rows
                out = np.zeros((Xv.shape[0], width), dtype=float)
                if np.any(nonmiss):
                    out[nonmiss, :] = _ensure_dense(tr.transform(col[nonmiss, :]))
                parts.append(out)

            core = (
                np.hstack(parts) if parts else np.empty((Xv.shape[0], 0), dtype=float)
            )

        # Indicators: always appended after the transformed features
        if self._indicator_cols_:
            idxs = [self._name_to_idx_[name] for name in self._indicator_cols_]
            indicators = np.column_stack(
                [nan_mask[:, k].astype(self.indicator_dtype) for k in idxs]
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
        return np.asarray(self._feature_names_out_, dtype=object)

    def _more_tags(self):
        # Output may contain NaNs if the underlying transformer emits them (when pass_through_missing=True).
        return {"allow_nan": True}
