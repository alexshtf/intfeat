from __future__ import annotations

import pandas as pd
import pytest

from experiments.criteo_fwfm.preprocess import CriteoFeaturePreprocessor


def _base_config() -> dict:
    return {
        "experiment": {"name": "test", "seed": 0, "variant": "baseline_winner"},
        "data": {
            "columns": {
                "label": "label",
                "integer": ["I1", "I2"],
                "categorical": ["C1"],
            }
        },
        "encoding": {
            "categorical": {
                "min_count": 2,
                "missing_token": "__MISSING__",
                "infrequent_token": "__INFREQUENT__",
            },
            "integer": {
                "low_cardinality_as_categorical": True,
                "low_cardinality_threshold": 10,
                "low_cardinality_count_mode": "non_missing_train_only",
                "force_integer_path": [],
                "force_categorical_path": [],
            },
            "integer_baseline": {
                "negative_prefix": "T",
                "logsq_prefix": "S",
            },
        },
        "model": {
            "integer": {
                "sl": {
                    "cap_max": 100,
                    "num_basis": 4,
                    "hist": {
                        "prior_count": 0.5,
                        "cutoff_quantile": 0.99,
                        "cutoff_factor": 1.1,
                    },
                    "curvature": {"alpha": 1.0, "beta": 0.0, "center": 0.0},
                    "conductance_eps": 1e-8,
                    "positive_overflow": "clip_to_cap",
                },
                "bspline": {
                    "cap_max": 100,
                    "cap_mode": "max",
                    "cap_quantile": 0.99,
                    "cap_quantile_factor": 1.1,
                    "input_map": "log1p_cap_to_unit",
                    "out_min": -1.0,
                    "out_max": 1.0,
                    "positive_overflow": "clip_to_cap",
                    "degree": 3,
                    "knots_config": 8,
                    "normalize_fn": "clamp",
                    "normalization_scale": 1.0,
                },
            }
        },
    }


def test_low_cardinality_integer_routes_to_categorical() -> None:
    config = _base_config()

    # I1 has 5 unique values (<10), I2 has 20 unique values (>=10).
    rows = 60
    train_df = pd.DataFrame(
        {
            "label": [0, 1] * (rows // 2),
            "I1": [i % 5 for i in range(rows)],
            "I2": [i % 20 for i in range(rows)],
            "C1": ["a", "b", "c"] * 20,
        }
    )

    preprocessor = CriteoFeaturePreprocessor(config)
    preprocessor.fit(train_df)

    routing = preprocessor.routing_report()
    assert routing["I1"] == "categorical"
    assert routing["I2"] == "integer"


def test_force_path_conflict_raises_error() -> None:
    config = _base_config()
    config["encoding"]["integer"]["force_integer_path"] = ["I1"]
    config["encoding"]["integer"]["force_categorical_path"] = ["I1"]

    train_df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "I1": [1, 2, 3, 4],
            "I2": [10, 11, 12, 13],
            "C1": ["x", "y", "x", "y"],
        }
    )

    preprocessor = CriteoFeaturePreprocessor(config)
    with pytest.raises(ValueError, match="both integer and categorical"):
        preprocessor.fit(train_df)
