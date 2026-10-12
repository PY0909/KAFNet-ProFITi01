"""Aggregated, label-free diagnostics for CUDTR prediction artifacts."""

from __future__ import annotations

from typing import Any

import torch


def _finite_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().float().cpu()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def summarize_cudtr_diagnostics(
    route_weights,
    source_available,
    fallback_gate,
    route_disagreement,
    *,
    saturation_threshold: float = 0.95,
    inactive_disagreement_threshold: float = 1e-8,
) -> dict[str, Any]:
    """Summarize route use without reading targets or labels.

    ``route_weights`` and ``source_available`` have shape ``[B, R, S]``;
    ``R`` is the number of target/task routes and ``S`` is the source pool.
    Selection frequency is measured over all ``B * R`` route slots.  A source
    selected while unavailable is reported separately and never hidden by the
    aggregate rate.
    """
    weights = _finite_tensor(route_weights, "route_weights")
    available = torch.as_tensor(source_available).detach().cpu().bool()
    fallback = _finite_tensor(fallback_gate, "fallback_gate").reshape(-1)
    disagreement = _finite_tensor(route_disagreement, "route_disagreement").reshape(-1)
    if weights.ndim != 3:
        raise ValueError("route_weights must have shape [batch, route, source]")
    if available.shape != weights.shape:
        raise ValueError("source_available must have the same shape as route_weights")
    if fallback.numel() != weights.shape[0] * weights.shape[1]:
        raise ValueError("fallback_gate must have shape [batch, route]")
    if disagreement.numel() != fallback.numel():
        raise ValueError("route_disagreement must have shape [batch, route]")
    if (weights < 0).any():
        raise ValueError("route_weights must be non-negative")

    selected = weights > 0
    slots = selected.shape[0] * selected.shape[1]
    selected_count = selected.sum(dim=(0, 1)).tolist()
    source_weight_sum = weights.sum(dim=(0, 1))
    source_mean_weight = (source_weight_sum / max(1, slots)).tolist()
    unavailable_selected = (selected & ~available).sum().item()
    fallback_saturation = fallback.ge(float(saturation_threshold)).float().mean().item()
    route_inactive = bool(
        selected.sum().item() == 0
        or disagreement.abs().mean().item() <= float(inactive_disagreement_threshold)
    )
    return {
        "source_count": int(weights.shape[-1]),
        "route_count": int(weights.shape[0] * weights.shape[1]),
        "source_selected_count": [int(value) for value in selected_count],
        "source_selection_rate": [float(value) / max(1, slots) for value in selected_count],
        "source_mean_weight": [float(value) for value in source_mean_weight],
        "unavailable_source_selected_count": int(unavailable_selected),
        "fallback_mean": float(fallback.mean().item()),
        "fallback_p95": float(torch.quantile(fallback, 0.95).item()),
        "fallback_saturation_rate": float(fallback_saturation),
        "route_disagreement_mean": float(disagreement.mean().item()),
        "route_inactive": route_inactive,
    }
