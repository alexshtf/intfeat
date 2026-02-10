from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..types import FieldSpec

try:
    import torchcurves as tc
except ImportError:  # pragma: no cover - optional path for non-bspline variants.
    tc = None


class DiscreteFieldModule(nn.Module):
    def __init__(self, cardinality: int, embedding_dim: int, init_scale: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cardinality, embedding_dim)
        self.linear = nn.Embedding(cardinality, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.linear.weight)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(token_ids)
        lin = self.linear(token_ids).squeeze(-1)
        return emb, lin


class SLIntegerFieldModule(nn.Module):
    def __init__(
        self,
        *,
        discrete_cardinality: int,
        embedding_dim: int,
        num_basis: int,
        init_scale: float,
    ) -> None:
        super().__init__()
        self.discrete = DiscreteFieldModule(discrete_cardinality, embedding_dim, init_scale)
        self.basis_embedding = nn.Parameter(torch.empty(num_basis, embedding_dim))
        self.basis_linear = nn.Parameter(torch.zeros(num_basis))
        nn.init.normal_(self.basis_embedding, mean=0.0, std=init_scale)

    def forward(
        self,
        token_ids: torch.Tensor,
        positive_mask: torch.Tensor,
        basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disc_emb, disc_lin = self.discrete(token_ids)
        cont_emb = basis @ self.basis_embedding
        cont_lin = basis @ self.basis_linear

        mask = positive_mask.unsqueeze(-1)
        emb = torch.where(mask, cont_emb, disc_emb)
        lin = torch.where(positive_mask, cont_lin, disc_lin)
        return emb, lin


class BSplineIntegerFieldModule(nn.Module):
    def __init__(
        self,
        *,
        discrete_cardinality: int,
        embedding_dim: int,
        degree: int,
        knots_config: int,
        normalize_fn: str,
        normalization_scale: float,
        init_scale: float,
    ) -> None:
        super().__init__()
        if tc is None:
            raise ImportError(
                "torchcurves is required for bspline_integer_basis variant. "
                "Install it (for example: uv add torchcurves)."
            )

        self.discrete = DiscreteFieldModule(discrete_cardinality, embedding_dim, init_scale)

        self.curve_embedding = tc.BSplineCurve(
            num_curves=1,
            dim=embedding_dim,
            degree=degree,
            knots_config=knots_config,
            normalize_fn=normalize_fn,
            normalization_scale=normalization_scale,
        )
        self.curve_linear = tc.BSplineCurve(
            num_curves=1,
            dim=1,
            degree=degree,
            knots_config=knots_config,
            normalize_fn=normalize_fn,
            normalization_scale=normalization_scale,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        positive_mask: torch.Tensor,
        scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disc_emb, disc_lin = self.discrete(token_ids)

        curve_input = scalar.unsqueeze(-1)
        cont_emb = self.curve_embedding(curve_input).squeeze(1)
        cont_lin = self.curve_linear(curve_input).squeeze(-1).squeeze(-1)

        mask = positive_mask.unsqueeze(-1)
        emb = torch.where(mask, cont_emb, disc_emb)
        lin = torch.where(positive_mask, cont_lin, disc_lin)
        return emb, lin


@dataclass
class FwFMConfig:
    embedding_dim: int
    init_scale: float
    dropout: float
    use_bias: bool
    enforce_symmetric: bool
    zero_diag: bool


class FwFMModel(nn.Module):
    def __init__(
        self,
        field_specs: list[FieldSpec],
        config: FwFMConfig,
    ) -> None:
        super().__init__()

        self.field_specs = list(field_specs)
        self.field_order = [item.name for item in field_specs]
        self.embedding_dim = config.embedding_dim
        self.use_bias = config.use_bias
        self.enforce_symmetric = config.enforce_symmetric
        self.zero_diag = config.zero_diag

        self.field_modules = nn.ModuleDict()
        for spec in field_specs:
            if spec.kind == "discrete":
                self.field_modules[spec.name] = DiscreteFieldModule(
                    spec.discrete_cardinality,
                    config.embedding_dim,
                    config.init_scale,
                )
            elif spec.kind == "sl_integer":
                self.field_modules[spec.name] = SLIntegerFieldModule(
                    discrete_cardinality=spec.discrete_cardinality,
                    embedding_dim=config.embedding_dim,
                    num_basis=spec.num_basis,
                    init_scale=config.init_scale,
                )
            elif spec.kind == "bspline_integer":
                self.field_modules[spec.name] = BSplineIntegerFieldModule(
                    discrete_cardinality=spec.discrete_cardinality,
                    embedding_dim=config.embedding_dim,
                    degree=spec.bspline_degree,
                    knots_config=spec.bspline_knots,
                    normalize_fn=spec.bspline_normalize_fn or "clamp",
                    normalization_scale=spec.bspline_normalization_scale,
                    init_scale=config.init_scale,
                )
            else:  # pragma: no cover - guarded by preprocessor.
                raise ValueError(f"Unsupported field kind: {spec.kind!r}")

        num_fields = len(field_specs)
        self.r_raw = nn.Parameter(torch.empty(num_fields, num_fields))
        nn.init.normal_(self.r_raw, mean=0.0, std=config.init_scale)

        self.bias = nn.Parameter(torch.zeros(1)) if self.use_bias else None
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else None

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        linear_terms: list[torch.Tensor] = []

        for spec in self.field_specs:
            field_input = batch[spec.name]
            module = self.field_modules[spec.name]

            if spec.kind == "discrete":
                emb, lin = module(field_input["token_ids"])
            elif spec.kind == "sl_integer":
                emb, lin = module(
                    field_input["token_ids"],
                    field_input["positive_mask"],
                    field_input["basis"],
                )
            else:
                emb, lin = module(
                    field_input["token_ids"],
                    field_input["positive_mask"],
                    field_input["scalar"],
                )

            embeddings.append(emb)
            linear_terms.append(lin)

        field_embeddings = torch.stack(embeddings, dim=1)
        if self.dropout is not None:
            field_embeddings = self.dropout(field_embeddings)

        r = self._interaction_matrix()
        interaction = torch.einsum("bfd,fg,bgd->b", field_embeddings, r, field_embeddings)

        linear = torch.stack(linear_terms, dim=1).sum(dim=1)
        logits = linear + interaction
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    def _interaction_matrix(self) -> torch.Tensor:
        r = self.r_raw
        if self.enforce_symmetric:
            r = 0.5 * (r + r.t())
        if self.zero_diag:
            r = r - torch.diag_embed(torch.diagonal(r))
        return r


def build_fwfm_model(field_specs: list[FieldSpec], config: dict[str, Any]) -> FwFMModel:
    fwfm_config = FwFMConfig(
        embedding_dim=int(config["model"]["embedding_dim"]),
        init_scale=float(config["model"]["fwfm"]["init_scale"]),
        dropout=float(config["model"].get("dropout", 0.0)),
        use_bias=bool(config["model"].get("use_bias", True)),
        enforce_symmetric=bool(config["model"]["fwfm"].get("enforce_symmetric", True)),
        zero_diag=bool(config["model"]["fwfm"].get("zero_diag", True)),
    )
    return FwFMModel(field_specs, fwfm_config)
