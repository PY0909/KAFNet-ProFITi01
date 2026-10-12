from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class RoutingOutput:
    fused: Tensor
    weights: Tensor
    fallback: Tensor
    utility_prediction: Tensor
    route_entropy: Tensor


def counterfactual_utility(full_loss: Tensor, dropped_loss: Tensor) -> Tensor:
    """Return detached loss increase after removing a source from a training input."""
    if full_loss.shape != dropped_loss.shape:
        raise ValueError("full_loss and dropped_loss must have the same shape")
    if not torch.isfinite(full_loss).all() or not torch.isfinite(dropped_loss).all():
        raise ValueError("counterfactual losses must be finite")
    return (dropped_loss - full_loss).detach()


class CUDTRouting(nn.Module):
    """Target-conditioned sparse routing with a bounded baseline fallback."""

    def __init__(
        self,
        hidden_dim: int,
        query_dim: int,
        top_k: int,
        temperature: float = 1.0,
        source_feature_dim: int = 0,
        fallback_min: float = 0.0,
        fallback_max: float = 1.0,
        fallback_init: float = 1.0,
        hard_top_k: bool = True,
        utility_threshold=None,
        utility_clip: Optional[float] = None,
    ):
        super().__init__()
        if hidden_dim < 1 or query_dim < 1 or top_k < 1:
            raise ValueError("hidden_dim, query_dim and top_k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if source_feature_dim < 0:
            raise ValueError("source_feature_dim cannot be negative")
        if not 0 <= fallback_min < fallback_max <= 1:
            raise ValueError("fallback bounds must satisfy 0 <= min < max <= 1")
        if not fallback_min <= fallback_init <= fallback_max:
            raise ValueError("fallback_init must lie within fallback bounds")

        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.top_k = top_k
        self.temperature = float(temperature)
        self.source_feature_dim = source_feature_dim
        self.fallback_min = float(fallback_min)
        self.fallback_max = float(fallback_max)
        self.hard_top_k = bool(hard_top_k)
        self.utility_threshold = utility_threshold
        if utility_clip is not None and utility_clip <= 0:
            raise ValueError("utility_clip must be positive")
        self.utility_clip = utility_clip

        input_dim = hidden_dim * 2 + query_dim + source_feature_dim
        self.utility_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.fallback_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            final = self.fallback_gate[-1]
            final.weight.zero_()
            proportion = (fallback_init - fallback_min) / (fallback_max - fallback_min)
            proportion = min(max(proportion, 1e-6), 1.0 - 1e-6)
            final.bias.fill_(torch.logit(torch.tensor(proportion)).item())

    def set_hard_top_k(self, enabled: bool):
        self.hard_top_k = bool(enabled)

    def forward(
        self,
        base: Tensor,
        sources: Tensor,
        target_query: Tensor,
        source_observed: Tensor,
        source_features: Optional[Tensor] = None,
    ) -> RoutingOutput:
        if base.ndim != 2 or base.shape[-1] != self.hidden_dim:
            raise ValueError("base must have shape [batch, hidden_dim]")
        if sources.ndim != 3 or sources.shape[0] != base.shape[0] or sources.shape[-1] != self.hidden_dim:
            raise ValueError("sources must have shape [batch, source_count, hidden_dim]")
        batch, source_count, _ = sources.shape
        if target_query.shape != (batch, self.query_dim):
            raise ValueError("target_query must have shape [batch, query_dim]")
        if source_observed.shape != (batch, source_count):
            raise ValueError("source_observed must have shape [batch, source_count]")
        if not torch.isfinite(base).all() or not torch.isfinite(sources).all():
            raise ValueError("routing representations must be finite")
        if not torch.isfinite(target_query).all():
            raise ValueError("target_query must be finite")
        if not torch.isfinite(source_observed).all() or ((source_observed < 0) | (source_observed > 1)).any():
            raise ValueError("source_observed must contain finite values in [0, 1]")

        if self.source_feature_dim:
            if source_features is None or source_features.shape != (batch, source_count, self.source_feature_dim):
                raise ValueError("source_features must match [batch, source_count, source_feature_dim]")
            if not torch.isfinite(source_features).all():
                raise ValueError("source_features must be finite")
        elif source_features is not None:
            raise ValueError("source_features supplied but source_feature_dim is zero")

        features = [
            base[:, None, :].expand(-1, source_count, -1),
            sources,
            target_query[:, None, :].expand(-1, source_count, -1),
        ]
        if source_features is not None:
            features.append(source_features)
        utility_prediction = self.utility_predictor(torch.cat(features, dim=-1)).squeeze(-1)
        if self.utility_clip is not None:
            utility_prediction = self.utility_clip * torch.tanh(utility_prediction / self.utility_clip)

        available = source_observed > 0
        if self.utility_threshold is not None:
            available = available & (utility_prediction >= self.utility_threshold)
        masked_scores = (utility_prediction / self.temperature).masked_fill(~available, -torch.inf)
        k = min(self.top_k, source_count)
        if source_count:
            any_available = available.any(dim=1, keepdim=True)
            if self.hard_top_k:
                top_values, top_indices = masked_scores.topk(k, dim=1)
                selected = torch.zeros_like(available).scatter_(1, top_indices, torch.isfinite(top_values))
                route_mask = available & selected
                route_scores = masked_scores.masked_fill(~route_mask, -torch.inf)
            else:
                route_mask = available
                route_scores = masked_scores
            safe_scores = torch.where(any_available, route_scores, torch.zeros_like(route_scores))
            weights = torch.softmax(safe_scores, dim=1) * route_mask.to(sources.dtype)
        else:
            weights = sources.new_zeros((batch, 0))

        projected = self.value_projection(sources)
        message = torch.sum(projected * weights.unsqueeze(-1), dim=1)
        message = self.message_norm(message)
        entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
        gate_input = torch.cat((base, message, entropy.unsqueeze(-1)), dim=-1)
        learned_fallback = torch.sigmoid(self.fallback_gate(gate_input).squeeze(-1))
        fallback = self.fallback_min + (self.fallback_max - self.fallback_min) * learned_fallback
        has_source = available.any(dim=1)
        fallback = torch.where(has_source, fallback, torch.ones_like(fallback))
        fused = (1.0 - fallback.unsqueeze(-1)) * message + fallback.unsqueeze(-1) * base
        fused = torch.where(has_source.unsqueeze(-1), fused, base)

        return RoutingOutput(
            fused=fused,
            weights=weights,
            fallback=fallback,
            utility_prediction=utility_prediction,
            route_entropy=entropy,
        )
