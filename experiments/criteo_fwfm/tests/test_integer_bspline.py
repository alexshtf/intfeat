from __future__ import annotations

import pandas as pd

from experiments.criteo_fwfm.encoders.integer_bspline import (
    BSplineIntegerEncoder,
    BSplineIntegerEncoderConfig,
)


def test_bspline_encoder_overflow_large_token_mode() -> None:
    config = BSplineIntegerEncoderConfig(
        cap_max=5,
        cap_mode="max",
        cap_quantile=1.0,
        cap_quantile_factor=1.0,
        input_map="linear_cap_to_unit",
        out_min=-1.0,
        out_max=1.0,
        positive_overflow="large_token",
    )
    encoder = BSplineIntegerEncoder(config).fit(pd.Series([1, 2, 3, 4, 5, -1]))

    token_ids, positive_mask, scalar = encoder.transform(pd.Series([1, 5, 6, -1, None]))

    assert positive_mask.tolist() == [True, True, False, False, False]
    assert token_ids[2] == 2  # overflow bucket (id after non-positive ids)
    assert token_ids[3] == 1  # seen non-positive discrete id
    assert encoder.discrete_cardinality == 3
    assert scalar.shape == (5,)
