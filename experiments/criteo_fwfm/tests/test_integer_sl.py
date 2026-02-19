from __future__ import annotations

import pandas as pd

from experiments.criteo_fwfm.encoders.integer_sl import SLIntegerEncoder, SLIntegerEncoderConfig


def test_sl_encoder_overflow_missing_mode() -> None:
    config = SLIntegerEncoderConfig(
        cap_max=5,
        num_basis=3,
        prior_count=0.5,
        cutoff_quantile=1.0,
        cutoff_factor=1.0,
        curvature_alpha=1.0,
        curvature_beta=0.0,
        curvature_center=0.0,
        conductance_eps=1e-8,
        positive_overflow="missing",
    )
    encoder = SLIntegerEncoder(config)

    train = pd.Series([1, 2, 3, 4, 5, -1])
    encoder.fit(train)

    token_ids, positive_mask, basis = encoder.transform(pd.Series([1, 5, 6, -1, None]))

    assert positive_mask.tolist() == [True, True, False, False, False]
    assert token_ids[2] == 0  # overflow in missing mode
    assert token_ids[3] == 1  # seen non-positive discrete id
    assert basis.shape == (5, 3)


def test_sl_encoder_overflow_large_token_mode() -> None:
    config = SLIntegerEncoderConfig(
        cap_max=5,
        num_basis=3,
        prior_count=0.5,
        cutoff_quantile=1.0,
        cutoff_factor=1.0,
        curvature_alpha=1.0,
        curvature_beta=0.0,
        curvature_center=0.0,
        conductance_eps=1e-8,
        positive_overflow="large_token",
        cap_mode="max",
    )
    encoder = SLIntegerEncoder(config)

    train = pd.Series([1, 2, 3, 4, 5, -1])
    encoder.fit(train)

    token_ids, positive_mask, basis = encoder.transform(pd.Series([1, 5, 6, -1, None]))

    assert positive_mask.tolist() == [True, True, False, False, False]
    assert token_ids[2] == 2  # overflow in large_token mode (id after non-positive ids)
    assert token_ids[3] == 1  # seen non-positive discrete id
    assert encoder.discrete_cardinality == 3
    assert basis.shape == (5, 3)
