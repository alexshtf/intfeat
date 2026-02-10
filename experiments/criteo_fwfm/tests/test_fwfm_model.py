from __future__ import annotations

import torch

from experiments.criteo_fwfm.model.fwfm import FwFMConfig, FwFMModel
from experiments.criteo_fwfm.types import FieldSpec


def test_fwfm_discrete_forward_shape() -> None:
    specs = [
        FieldSpec(name="f1", kind="discrete", discrete_cardinality=5),
        FieldSpec(name="f2", kind="discrete", discrete_cardinality=7),
    ]
    config = FwFMConfig(
        embedding_dim=8,
        init_scale=0.01,
        dropout=0.0,
        use_bias=True,
        enforce_symmetric=True,
        zero_diag=True,
    )

    model = FwFMModel(specs, config)
    batch = {
        "f1": {
            "token_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "positive_mask": torch.tensor([False, False, False]),
        },
        "f2": {
            "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "positive_mask": torch.tensor([False, False, False]),
        },
    }

    logits = model(batch)
    assert logits.shape == (3,)
