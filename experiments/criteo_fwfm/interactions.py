from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def interaction_matrix_numpy(model: torch.nn.Module) -> np.ndarray:
    """Return the *effective* FwFM interaction matrix R used in forward().

    This corresponds to the symmetrized / zero-diagonal matrix produced by
    `FwFMModel._interaction_matrix()`, not the raw parameter `r_raw`.
    """
    if not hasattr(model, "_interaction_matrix"):
        raise TypeError("Model does not expose _interaction_matrix(); expected FwFMModel")
    r = getattr(model, "_interaction_matrix")()
    if not isinstance(r, torch.Tensor):
        raise TypeError("model._interaction_matrix() did not return a torch.Tensor")
    return r.detach().cpu().numpy()


def _resolve_field_order(model: torch.nn.Module) -> list[str]:
    order = getattr(model, "field_order", None)
    if order is None:
        raise TypeError("Model does not expose field_order; expected FwFMModel")
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise TypeError(f"Expected model.field_order to be list[str], got {type(order)}")
    return order


def save_interaction_matrix_npz(*, model: torch.nn.Module, out_path: Path) -> None:
    """Save R and field_order to a compressed npz for downstream analysis."""
    r = interaction_matrix_numpy(model).astype(np.float32, copy=False)
    field_order = _resolve_field_order(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, r=r, field_order=np.array(field_order, dtype=object))


def save_interaction_matrix_json(*, model: torch.nn.Module, out_path: Path) -> None:
    """Save R and field_order to JSON (human-readable, but less compact)."""
    r = interaction_matrix_numpy(model)
    field_order = _resolve_field_order(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "field_order": field_order,
        "r": r.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

