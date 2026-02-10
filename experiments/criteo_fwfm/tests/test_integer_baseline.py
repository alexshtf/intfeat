from __future__ import annotations

from experiments.criteo_fwfm.encoders.integer_baseline import winner_integer_token


def test_winner_integer_tokenization() -> None:
    assert winner_integer_token(-3, negative_prefix="T", logsq_prefix="S") == "T-3"
    assert winner_integer_token(0, negative_prefix="T", logsq_prefix="S") == "T0"
    assert winner_integer_token(1, negative_prefix="T", logsq_prefix="S") == "S0"
    assert winner_integer_token(2, negative_prefix="T", logsq_prefix="S") == "S0"
    assert winner_integer_token(10, negative_prefix="T", logsq_prefix="S") == "S5"
