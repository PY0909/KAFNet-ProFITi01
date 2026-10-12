"""Training-only auxiliary objectives for CUDTR.

The functions in this module consume counterfactual utility labels produced by
``utility_teacher``.  They never construct labels themselves, which keeps the
formal prediction path independent from supervised targets.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CUDTRLossConfig:
    """Weights for the optional CUDTR student objectives."""

    lambda_distill: float = 0.0
    lambda_rank: float = 0.0
    lambda_sparse: float = 0.0
    lambda_fallback: float = 0.0
    rank_margin: float = 0.1
    utility_clip: Optional[float] = None
    fallback_margin: float = 0.1

    def __post_init__(self):
        for name in ("lambda_distill", "lambda_rank", "lambda_sparse", "lambda_fallback"):
            value = getattr(self, name)
            if value < 0 or not torch.isfinite(torch.tensor(value)):
                raise ValueError("{} must be non-negative and finite".format(name))
        if self.rank_margin < 0 or not torch.isfinite(torch.tensor(self.rank_margin)):
            raise ValueError("rank_margin must be non-negative and finite")
        if self.utility_clip is not None and (
            self.utility_clip <= 0 or not torch.isfinite(torch.tensor(self.utility_clip))
        ):
            raise ValueError("utility_clip must be positive and finite")
        if self.fallback_margin < 0 or not torch.isfinite(torch.tensor(self.fallback_margin)):
            raise ValueError("fallback_margin must be non-negative and finite")


def _validate_pair(predicted: Tensor, target: Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target utility tensors must have the same shape")
    if predicted.ndim < 2:
        raise ValueError("utility tensors must have a source dimension")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("utility tensors must be finite")


def utility_distillation_loss(predicted: Tensor, target: Tensor, clip: Optional[float] = None) -> Tensor:
    """Huber regression from predicted utility to detached counterfactual utility."""
    _validate_pair(predicted, target)
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive")
        target = target.clamp(-clip, clip)
    return F.smooth_l1_loss(predicted, target)


def utility_ranking_loss(predicted: Tensor, target: Tensor, margin: float = 0.1) -> Tensor:
    """Pairwise ranking loss for sources with unequal counterfactual utility.

    A larger utility means that dropping the source hurts more, so it should
    receive a larger predicted utility.  Ties are omitted from the objective.
    """
    _validate_pair(predicted, target)
    if margin < 0 or not torch.isfinite(torch.tensor(margin)):
        raise ValueError("margin must be non-negative and finite")
    target_delta = target.unsqueeze(-1) - target.unsqueeze(-2)
    predicted_delta = predicted.unsqueeze(-1) - predicted.unsqueeze(-2)
    ordered = target_delta > 0
    if not ordered.any():
        return predicted.sum() * 0.0
    violations = F.relu(margin - predicted_delta)
    return violations.masked_select(ordered).mean()


def route_entropy_loss(weights: Optional[Tensor]) -> Tensor:
    """Return mean route entropy, with zero for an absent route tensor."""
    if weights is None:
        return torch.tensor(0.0)
    if weights.ndim < 2 or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative with a source dimension")
    normalizer = weights.sum(dim=-1)
    if (normalizer > 1.0 + 1e-5).any():
        raise ValueError("route weights must sum to at most one")
    # Missing-source rows are intentionally zero entropy.
    safe = weights.clamp_min(1e-12)
    entropy = -(safe.log() * weights).sum(dim=-1)
    return entropy.mean()


def fallback_consistency_loss(
    corrupted_fused: Optional[Tensor],
    base: Optional[Tensor],
    *,
    fallback: Optional[Tensor] = None,
    clean_fallback: Optional[Tensor] = None,
    corrupted: bool = True,
    margin: float = 0.1,
) -> Tensor:
    """Penalize corrupted-route representations that depart from the baseline."""
    if corrupted_fused is None and base is None:
        return torch.tensor(0.0)
    if corrupted_fused is None or base is None or corrupted_fused.shape != base.shape:
        raise ValueError("corrupted_fused and base must have matching shapes")
    if margin < 0 or not torch.isfinite(torch.tensor(margin)):
        raise ValueError("margin must be non-negative and finite")
    if not torch.isfinite(corrupted_fused).all() or not torch.isfinite(base).all():
        raise ValueError("fallback representations must be finite")
    representation_loss = F.mse_loss(corrupted_fused, base)
    result = representation_loss
    if fallback is not None:
        if not torch.isfinite(fallback).all() or (fallback < 0).any() or (fallback > 1).any():
            raise ValueError("fallback must be finite and lie in [0, 1]")
        target = torch.ones_like(fallback) if corrupted else torch.zeros_like(fallback)
        result = result + F.mse_loss(fallback, target)
    if clean_fallback is not None:
        if not torch.isfinite(clean_fallback).all() or (clean_fallback < 0).any() or (clean_fallback > 1).any():
            raise ValueError("clean_fallback must be finite and lie in [0, 1]")
        result = result + F.mse_loss(clean_fallback, torch.zeros_like(clean_fallback))
        if fallback is not None:
            result = result + F.relu(margin + clean_fallback - fallback).mean()
    return result


def cudtr_auxiliary_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    weights: Optional[Tensor] = None,
    corrupted_fused: Optional[Tensor] = None,
    base: Optional[Tensor] = None,
    fallback: Optional[Tensor] = None,
    clean_fallback: Optional[Tensor] = None,
    config: CUDTRLossConfig = CUDTRLossConfig(),
) -> Dict[str, Tensor]:
    """Compute CUDTR auxiliary terms without the task loss.

    The returned ``total`` can be added to the ordinary task loss.  When all
    weights are zero, every term is a differentiable zero on the predictor's
    device, preserving a no-op default for legacy V3C training.
    """
    _validate_pair(predicted, target)
    zero = predicted.sum() * 0.0
    distill = (
        utility_distillation_loss(predicted, target, config.utility_clip)
        if config.lambda_distill
        else zero
    )
    rank = (
        utility_ranking_loss(predicted, target, config.rank_margin)
        if config.lambda_rank
        else zero
    )
    sparse = route_entropy_loss(weights) if config.lambda_sparse else zero
    fallback_term = fallback_consistency_loss(
        corrupted_fused, base, fallback=fallback, clean_fallback=clean_fallback,
        corrupted=True, margin=config.fallback_margin,
    ) if config.lambda_fallback else zero
    total = (
        config.lambda_distill * distill
        + config.lambda_rank * rank
        + config.lambda_sparse * sparse
        + config.lambda_fallback * fallback_term
    )
    return {
        "total": total,
        "distill": distill,
        "rank": rank,
        "sparse": sparse,
        "fallback": fallback_term,
    }
