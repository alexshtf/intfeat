from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from .types import EncodedSplit, FieldSpec

LOGGER = logging.getLogger(__name__)


@dataclass
class TorchEncodedSplit:
    labels: torch.Tensor
    fields: dict[str, dict[str, torch.Tensor]]

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encoded_split_to_torch(
    split: EncodedSplit,
    field_specs: list[FieldSpec],
) -> TorchEncodedSplit:
    fields: dict[str, dict[str, torch.Tensor]] = {}

    for spec in field_specs:
        field_data = split.fields[spec.name]
        tensor_bundle: dict[str, torch.Tensor] = {
            "token_ids": torch.as_tensor(field_data.token_ids, dtype=torch.long),
            "positive_mask": torch.as_tensor(
                field_data.positive_mask, dtype=torch.bool
            ),
        }
        if field_data.basis is not None:
            tensor_bundle["basis"] = torch.as_tensor(field_data.basis, dtype=torch.float32)
        if field_data.scalar is not None:
            tensor_bundle["scalar"] = torch.as_tensor(field_data.scalar, dtype=torch.float32)

        fields[spec.name] = tensor_bundle

    labels = torch.as_tensor(split.labels, dtype=torch.float32)
    return TorchEncodedSplit(labels=labels, fields=fields)


def _iter_minibatches(
    num_examples: int,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
):
    indices = np.arange(num_examples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, num_examples, batch_size):
        yield indices[start : start + batch_size]


def _slice_batch(
    split: TorchEncodedSplit,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
    idx = torch.as_tensor(indices, dtype=torch.long)
    labels = split.labels[idx].to(device=device)

    batch: dict[str, dict[str, torch.Tensor]] = {}
    for name, bundle in split.fields.items():
        batch[name] = {
            key: tensor[idx].to(device=device)
            for key, tensor in bundle.items()
        }

    return batch, labels


def _compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)

    metrics: dict[str, float] = {
        "logloss": float(log_loss(labels, probs)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auc"] = math.nan

    return metrics


def predict_logits(
    model: torch.nn.Module,
    split: TorchEncodedSplit,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for indices in _iter_minibatches(
            split.size,
            batch_size,
            shuffle=False,
            seed=0,
        ):
            batch, _ = _slice_batch(split, indices, device)
            logits = model(batch)
            outputs.append(logits.detach().cpu().numpy())

    if not outputs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def evaluate_split(
    model: torch.nn.Module,
    split: TorchEncodedSplit,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    logits = predict_logits(model, split, device=device, batch_size=batch_size)
    labels = split.labels.detach().cpu().numpy()
    metrics = _compute_metrics(labels, logits)
    probs = 1.0 / (1.0 + np.exp(-logits))

    return {
        "metrics": metrics,
        "labels": labels,
        "logits": logits,
        "probs": probs,
    }


def train_model(
    model: torch.nn.Module,
    train_split: TorchEncodedSplit,
    val_split: TorchEncodedSplit,
    *,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    train_cfg = config["train"]

    seed = int(config["experiment"]["seed"])
    set_global_seed(seed)

    batch_size = int(train_cfg["batch_size"])
    num_epochs = int(train_cfg["num_epochs"])
    learning_rate = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))

    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")

    criterion = torch.nn.BCEWithLogitsLoss()
    early_metric = str(train_cfg["early_stopping"]["metric"])
    patience = int(train_cfg["early_stopping"]["patience"])

    if early_metric not in {"val_logloss", "val_auc"}:
        raise ValueError(
            f"Unsupported early stopping metric: {early_metric!r}. "
            "Expected one of {'val_logloss', 'val_auc'}."
        )

    best_value = math.inf if early_metric == "val_logloss" else -math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0

    history: list[dict[str, float]] = []

    for epoch in range(num_epochs):
        model.train()
        train_losses: list[float] = []

        for step, indices in enumerate(
            _iter_minibatches(
                train_split.size,
                batch_size,
                shuffle=True,
                seed=seed + epoch,
            )
        ):
            batch, labels = _slice_batch(train_split, indices, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            train_losses.append(float(loss.item()))

            if (step + 1) % 100 == 0:
                LOGGER.info(
                    "epoch=%s step=%s train_loss=%.6f",
                    epoch + 1,
                    step + 1,
                    np.mean(train_losses),
                )

        val_eval = evaluate_split(model, val_split, device=device, batch_size=batch_size)
        val_metrics = val_eval["metrics"]

        epoch_record = {
            "epoch": float(epoch + 1),
            "train_loss": float(np.mean(train_losses) if train_losses else math.nan),
            "val_logloss": float(val_metrics["logloss"]),
            "val_auc": float(val_metrics["auc"]),
        }
        history.append(epoch_record)

        current_value = (
            epoch_record["val_logloss"]
            if early_metric == "val_logloss"
            else epoch_record["val_auc"]
        )

        improved = (
            current_value < best_value
            if early_metric == "val_logloss"
            else current_value > best_value
        )

        LOGGER.info(
            "epoch=%s train_loss=%.6f val_logloss=%.6f val_auc=%.6f",
            epoch + 1,
            epoch_record["train_loss"],
            epoch_record["val_logloss"],
            epoch_record["val_auc"],
        )

        if improved:
            best_value = current_value
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            LOGGER.info(
                "Early stopping triggered at epoch=%s (patience=%s)",
                epoch + 1,
                patience,
            )
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing a best model state")

    model.load_state_dict(best_state)

    return {
        "best_epoch": best_epoch,
        "best_value": best_value,
        "history": history,
    }
