from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import get_config_value
from .encoders.categorical import CategoricalOrdinalEncoder
from .encoders.integer_baseline import WinnerIntegerEncoder
from .encoders.integer_bspline import BSplineIntegerEncoder, BSplineIntegerEncoderConfig
from .encoders.integer_sl import SLIntegerEncoder, SLIntegerEncoderConfig
from .encoders.common import to_nullable_int
from .schema import CATEGORICAL_COLUMNS, INTEGER_COLUMNS, LABEL_COLUMN
from .types import EncodedSplit, FieldArrayData, FieldSpec, FieldStat


class CriteoFeaturePreprocessor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.variant = str(get_config_value(config, "experiment.variant", required=True))

        self.label_column = str(
            get_config_value(config, "data.columns.label", default=LABEL_COLUMN)
        )
        self.integer_columns = list(
            get_config_value(config, "data.columns.integer", default=INTEGER_COLUMNS)
        )
        self.categorical_columns = list(
            get_config_value(
                config, "data.columns.categorical", default=CATEGORICAL_COLUMNS
            )
        )
        self.field_order = [*self.integer_columns, *self.categorical_columns]

        self.min_count = int(get_config_value(config, "encoding.categorical.min_count", default=10))
        self.missing_token = str(
            get_config_value(
                config, "encoding.categorical.missing_token", default="__MISSING__"
            )
        )
        self.infrequent_token = str(
            get_config_value(
                config,
                "encoding.categorical.infrequent_token",
                default="__INFREQUENT__",
            )
        )

        self.low_card_as_categorical = bool(
            get_config_value(
                config,
                "encoding.integer.low_cardinality_as_categorical",
                default=True,
            )
        )
        self.low_card_threshold = int(
            get_config_value(
                config, "encoding.integer.low_cardinality_threshold", default=10
            )
        )
        self.low_card_count_mode = str(
            get_config_value(
                config,
                "encoding.integer.low_cardinality_count_mode",
                default="non_missing_train_only",
            )
        )
        self.force_integer_path = set(
            get_config_value(config, "encoding.integer.force_integer_path", default=[])
        )
        self.force_categorical_path = set(
            get_config_value(config, "encoding.integer.force_categorical_path", default=[])
        )

        self.categorical_encoders: dict[str, CategoricalOrdinalEncoder] = {}
        self.baseline_integer_encoders: dict[str, WinnerIntegerEncoder] = {}
        self.sl_integer_encoders: dict[str, SLIntegerEncoder] = {}
        self.bspline_integer_encoders: dict[str, BSplineIntegerEncoder] = {}

        self.integer_routing: dict[str, str] = {}
        self.field_specs: list[FieldSpec] = []
        self.field_stats: list[FieldStat] = []
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "CriteoFeaturePreprocessor":
        self._validate_force_paths()
        self.integer_routing = self._compute_integer_routing(train_df)

        categorical_final = list(self.categorical_columns)
        categorical_final.extend(
            [column for column, route in self.integer_routing.items() if route == "categorical"]
        )
        integer_final = [
            column for column, route in self.integer_routing.items() if route == "integer"
        ]

        self.categorical_encoders = {}
        for column in categorical_final:
            encoder = CategoricalOrdinalEncoder(
                min_count=self.min_count,
                missing_token=self.missing_token,
                infrequent_token=self.infrequent_token,
            )
            encoder.fit(train_df[column])
            self.categorical_encoders[column] = encoder

        if self.variant == "baseline_winner":
            self.baseline_integer_encoders = self._fit_baseline_integer_encoders(
                train_df, integer_final
            )
        elif self.variant == "sl_integer_basis":
            self.sl_integer_encoders = self._fit_sl_integer_encoders(train_df, integer_final)
        elif self.variant == "bspline_integer_basis":
            self.bspline_integer_encoders = self._fit_bspline_integer_encoders(
                train_df, integer_final
            )
        else:
            raise ValueError(f"Unsupported variant: {self.variant!r}")

        self.field_specs = self._build_field_specs()
        self.field_stats = self._build_field_stats(train_df)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> EncodedSplit:
        self._require_fitted()

        labels = df[self.label_column].to_numpy(dtype=np.float32)
        fields: dict[str, FieldArrayData] = {}

        for spec in self.field_specs:
            column = spec.name

            if column in self.categorical_encoders:
                token_ids = self.categorical_encoders[column].transform(df[column])
                fields[column] = FieldArrayData(
                    token_ids=token_ids,
                    positive_mask=np.zeros(len(df), dtype=bool),
                )
                continue

            if self.variant == "baseline_winner":
                token_ids = self.baseline_integer_encoders[column].transform(df[column])
                fields[column] = FieldArrayData(
                    token_ids=token_ids,
                    positive_mask=np.zeros(len(df), dtype=bool),
                )
                continue

            if self.variant == "sl_integer_basis":
                token_ids, positive_mask, basis = self.sl_integer_encoders[column].transform(
                    df[column]
                )
                fields[column] = FieldArrayData(
                    token_ids=token_ids,
                    positive_mask=positive_mask,
                    basis=basis,
                )
                continue

            token_ids, positive_mask, scalar = self.bspline_integer_encoders[column].transform(
                df[column]
            )
            fields[column] = FieldArrayData(
                token_ids=token_ids,
                positive_mask=positive_mask,
                scalar=scalar,
            )

        return EncodedSplit(labels=labels, fields=fields)

    def fit_transform(self, train_df: pd.DataFrame) -> EncodedSplit:
        return self.fit(train_df).transform(train_df)

    def routing_report(self) -> dict[str, str]:
        self._require_fitted()
        return dict(self.integer_routing)

    def field_stats_report(self) -> list[dict[str, Any]]:
        self._require_fitted()
        return [asdict(item) for item in self.field_stats]

    def _fit_baseline_integer_encoders(
        self, train_df: pd.DataFrame, integer_columns: list[str]
    ) -> dict[str, WinnerIntegerEncoder]:
        negative_prefix = str(
            get_config_value(
                self.config, "encoding.integer_baseline.negative_prefix", default="T"
            )
        )
        logsq_prefix = str(
            get_config_value(
                self.config, "encoding.integer_baseline.logsq_prefix", default="S"
            )
        )

        encoders: dict[str, WinnerIntegerEncoder] = {}
        for column in integer_columns:
            encoder = WinnerIntegerEncoder(
                min_count=self.min_count,
                missing_token=self.missing_token,
                infrequent_token=self.infrequent_token,
                negative_prefix=negative_prefix,
                logsq_prefix=logsq_prefix,
            )
            encoder.fit(train_df[column])
            encoders[column] = encoder
        return encoders

    def _fit_sl_integer_encoders(
        self, train_df: pd.DataFrame, integer_columns: list[str]
    ) -> dict[str, SLIntegerEncoder]:
        conductance_family = str(
            get_config_value(
                self.config,
                "model.integer.sl.conductance.family",
                default="curvature_spec",
            )
        )
        sl_config = SLIntegerEncoderConfig(
            cap_max=int(get_config_value(self.config, "model.integer.sl.cap_max", default=10_000_000)),
            num_basis=int(get_config_value(self.config, "model.integer.sl.num_basis", default=16)),
            prior_count=float(
                get_config_value(self.config, "model.integer.sl.hist.prior_count", default=0.5)
            ),
            cutoff_quantile=float(
                get_config_value(
                    self.config, "model.integer.sl.hist.cutoff_quantile", default=0.99
                )
            ),
            cutoff_factor=float(
                get_config_value(
                    self.config, "model.integer.sl.hist.cutoff_factor", default=1.1
                )
            ),
            curvature_alpha=float(
                get_config_value(self.config, "model.integer.sl.curvature.alpha", default=1.0)
            ),
            curvature_beta=float(
                get_config_value(self.config, "model.integer.sl.curvature.beta", default=0.0)
            ),
            curvature_center=float(
                get_config_value(self.config, "model.integer.sl.curvature.center", default=0.0)
            ),
            conductance_eps=float(
                get_config_value(
                    self.config, "model.integer.sl.conductance_eps", default=1e-8
                )
            ),
            positive_overflow=str(
                get_config_value(
                    self.config,
                    "model.integer.sl.positive_overflow",
                    default="clip_to_cap",
                )
            ),
            conductance_family=conductance_family,
            uvalley_u0=float(
                get_config_value(
                    self.config,
                    "model.integer.sl.conductance.u_exp_valley.u0",
                    default=0.0,
                )
            ),
            uvalley_left_slope=float(
                get_config_value(
                    self.config,
                    "model.integer.sl.conductance.u_exp_valley.left_slope",
                    default=1.0,
                )
            ),
            uvalley_right_slope=float(
                get_config_value(
                    self.config,
                    "model.integer.sl.conductance.u_exp_valley.right_slope",
                    default=1.0,
                )
            ),
        )

        encoders: dict[str, SLIntegerEncoder] = {}
        for column in integer_columns:
            encoder = SLIntegerEncoder(sl_config)
            encoder.fit(train_df[column])
            encoders[column] = encoder
        return encoders

    def _fit_bspline_integer_encoders(
        self, train_df: pd.DataFrame, integer_columns: list[str]
    ) -> dict[str, BSplineIntegerEncoder]:
        bspline_config = BSplineIntegerEncoderConfig(
            cap_max=int(
                get_config_value(self.config, "model.integer.bspline.cap_max", default=10_000_000)
            ),
            cap_mode=str(
                get_config_value(self.config, "model.integer.bspline.cap_mode", default="max")
            ),
            cap_quantile=float(
                get_config_value(
                    self.config, "model.integer.bspline.cap_quantile", default=0.99
                )
            ),
            cap_quantile_factor=float(
                get_config_value(
                    self.config,
                    "model.integer.bspline.cap_quantile_factor",
                    default=1.1,
                )
            ),
            input_map=str(
                get_config_value(
                    self.config,
                    "model.integer.bspline.input_map",
                    default="log1p_cap_to_unit",
                )
            ),
            out_min=float(
                get_config_value(self.config, "model.integer.bspline.out_min", default=-1.0)
            ),
            out_max=float(
                get_config_value(self.config, "model.integer.bspline.out_max", default=1.0)
            ),
            positive_overflow=str(
                get_config_value(
                    self.config,
                    "model.integer.bspline.positive_overflow",
                    default="clip_to_cap",
                )
            ),
        )

        encoders: dict[str, BSplineIntegerEncoder] = {}
        for column in integer_columns:
            encoder = BSplineIntegerEncoder(bspline_config)
            encoder.fit(train_df[column])
            encoders[column] = encoder
        return encoders

    def _compute_integer_routing(self, train_df: pd.DataFrame) -> dict[str, str]:
        routing: dict[str, str] = {}

        for column in self.integer_columns:
            unique_count = self._count_unique_non_missing_integers(train_df[column])
            route = "integer"
            if self.low_card_as_categorical and unique_count < self.low_card_threshold:
                route = "categorical"

            if column in self.force_integer_path:
                route = "integer"
            if column in self.force_categorical_path:
                route = "categorical"

            routing[column] = route

        return routing

    def _build_field_specs(self) -> list[FieldSpec]:
        specs: list[FieldSpec] = []

        bspline_degree = int(
            get_config_value(self.config, "model.integer.bspline.degree", default=3)
        )
        bspline_knots = int(
            get_config_value(self.config, "model.integer.bspline.knots_config", default=16)
        )
        bspline_normalize_fn = str(
            get_config_value(
                self.config,
                "model.integer.bspline.normalize_fn",
                default="clamp",
            )
        )
        bspline_normalization_scale = float(
            get_config_value(
                self.config,
                "model.integer.bspline.normalization_scale",
                default=1.0,
            )
        )

        for column in self.field_order:
            if column in self.categorical_encoders:
                specs.append(
                    FieldSpec(
                        name=column,
                        kind="discrete",
                        discrete_cardinality=self.categorical_encoders[column].cardinality,
                    )
                )
                continue

            if self.variant == "baseline_winner":
                specs.append(
                    FieldSpec(
                        name=column,
                        kind="discrete",
                        discrete_cardinality=self.baseline_integer_encoders[
                            column
                        ].cardinality,
                    )
                )
                continue

            if self.variant == "sl_integer_basis":
                encoder = self.sl_integer_encoders[column]
                specs.append(
                    FieldSpec(
                        name=column,
                        kind="sl_integer",
                        discrete_cardinality=encoder.discrete_cardinality,
                        num_basis=encoder.num_basis,
                    )
                )
                continue

            encoder = self.bspline_integer_encoders[column]
            specs.append(
                FieldSpec(
                    name=column,
                    kind="bspline_integer",
                    discrete_cardinality=encoder.discrete_cardinality,
                    bspline_degree=bspline_degree,
                    bspline_knots=bspline_knots,
                    bspline_normalize_fn=bspline_normalize_fn,
                    bspline_normalization_scale=bspline_normalization_scale,
                )
            )

        return specs

    def _build_field_stats(self, train_df: pd.DataFrame) -> list[FieldStat]:
        stats: list[FieldStat] = []

        for column in self.integer_columns:
            unique_count = self._count_unique_non_missing_integers(train_df[column])
            stats.append(
                FieldStat(
                    name=column,
                    raw_type="integer",
                    routed_type=self.integer_routing[column],
                    unique_non_missing_train=unique_count,
                )
            )

        for column in self.categorical_columns:
            non_missing = train_df[column].dropna().astype("string")
            unique_count = int(non_missing[non_missing.str.len() > 0].nunique())
            stats.append(
                FieldStat(
                    name=column,
                    raw_type="categorical",
                    routed_type="categorical",
                    unique_non_missing_train=unique_count,
                )
            )

        return stats

    def _count_unique_non_missing_integers(self, series: pd.Series) -> int:
        ints = to_nullable_int(series)
        if self.low_card_count_mode != "non_missing_train_only":
            raise ValueError(
                "Unsupported integer low-cardinality count mode: "
                f"{self.low_card_count_mode!r}"
            )
        return int(ints.dropna().nunique())

    def _validate_force_paths(self) -> None:
        overlap = self.force_integer_path & self.force_categorical_path
        if overlap:
            overlap_list = sorted(overlap)
            raise ValueError(
                "Fields cannot be forced to both integer and categorical paths: "
                f"{overlap_list}"
            )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CriteoFeaturePreprocessor must be fitted before transform")
