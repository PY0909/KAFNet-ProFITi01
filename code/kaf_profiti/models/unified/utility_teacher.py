from typing import Callable, Iterable

import torch
from torch import Tensor

from .model import HistoryBatch
from .routing import counterfactual_utility
from .encoder import EncoderOutput


def estimate_route_level_utilities(
    predictor: Callable[[Tensor, int], Tensor],
    source_tokens: Tensor,
    targets: Tensor,
    loss_fn: Callable[[Tensor, Tensor, int], Tensor],
    source_indices: Iterable[int],
    target_count: int,
) -> Tensor:
    """Estimate target/source utilities by deleting fusion-layer messages.

    ``source_tokens`` are the encoder representations immediately before the
    cross-sensor fusion block. For each target ``i`` and source ``j``, only
    token ``j`` is zeroed and the target-specific loss increase is measured.
    The returned tensor has shape ``[batch, target, source]`` and is detached;
    diagonal entries are zero because a target's own representation is not an
    external dependency. This function is intended for train-only teacher
    cache construction.
    """
    if source_tokens.ndim != 3:
        raise ValueError("source_tokens must have shape [batch, source, hidden]")
    batch, source_count, _ = source_tokens.shape
    if target_count != source_count or target_count < 1:
        raise ValueError("target_count must match the source token count")
    if not isinstance(targets, Tensor) or targets.shape[0] != batch or targets.shape[1] != target_count:
        raise ValueError("targets must have shape [batch, target_count]")
    indices = [int(index) for index in source_indices]
    if indices != list(range(source_count)):
        raise ValueError("source_indices must enumerate every source token in order")
    if not torch.isfinite(source_tokens).all() or not torch.isfinite(targets).all():
        raise ValueError("source_tokens and targets must be finite")

    with torch.no_grad():
        utilities = source_tokens.new_zeros((batch, target_count, source_count))
        for target_index in range(target_count):
            full_prediction = predictor(source_tokens, target_index)
            full_loss = loss_fn(full_prediction, targets, target_index)
            if full_loss.ndim != 1 or full_loss.shape[0] != batch:
                raise ValueError("loss_fn must return per-item losses with shape [batch]")
            if not torch.isfinite(full_loss).all():
                raise ValueError("loss_fn must return finite values")
            for source_index in range(source_count):
                if source_index == target_index:
                    continue
                dropped = source_tokens.clone()
                dropped[:, source_index, :] = 0
                dropped_prediction = predictor(dropped, target_index)
                dropped_loss = loss_fn(dropped_prediction, targets, target_index)
                if dropped_loss.shape != full_loss.shape or not torch.isfinite(dropped_loss).all():
                    raise ValueError("loss_fn must preserve finite per-item shape after source deletion")
                utilities[:, target_index, source_index] = counterfactual_utility(
                    full_loss, dropped_loss
                )
    return utilities.detach()


@torch.no_grad()
def estimate_state_route_utilities(model, history: HistoryBatch, targets) -> Tensor:
    """Build MetroPT target/source utilities at the fusion boundary.

    The frozen teacher encodes the raw history once. Each counterfactual then
    deletes one ``pre_cross_tokens`` source and reuses the encoder's actual
    cross-sensor fusion, context, normalization, and pooling path.
    """
    if getattr(model.config, "task", None) != "state":
        raise ValueError("route-level state utility requires a state teacher")
    encoded = model.encoder(history)
    source_tokens = encoded.pre_cross_tokens
    if source_tokens is None:
        raise ValueError("state teacher encoder must expose pre_cross_tokens")
    batch, channels, _ = source_tokens.shape
    truth = targets.values
    if truth is None or truth.shape[0] != batch or truth.shape[2] != channels:
        raise ValueError("state targets must have shape [batch, horizon, channels]")

    def item_loss(prediction):
        if prediction.quantiles is None:
            return (prediction.point - truth).abs().mean(dim=1)
        levels = torch.as_tensor(model.head.quantiles, device=truth.device, dtype=truth.dtype)
        error = truth.unsqueeze(-1) - prediction.quantiles
        return torch.maximum(levels * error, (levels - 1.0) * error).mean(dim=(1, 3))

    full_history, full_global = model.encoder.fuse_pre_cross_tokens(source_tokens, history)
    full_encoded = EncoderOutput(z_history=full_history, z_global=full_global)
    full_prediction = model.head(full_encoded, mode=model.config.head_mode)
    full_loss = item_loss(full_prediction)
    utilities = source_tokens.new_zeros((batch, channels, channels))
    for source_index in range(channels):
        dropped = source_tokens.clone()
        dropped[:, source_index, :] = 0
        fused, global_repr = model.encoder.fuse_pre_cross_tokens(dropped, history)
        dropped_encoded = EncoderOutput(z_history=fused, z_global=global_repr)
        dropped_prediction = model.head(dropped_encoded, mode=model.config.head_mode)
        dropped_loss = item_loss(dropped_prediction)
        utilities[:, :, source_index] = (dropped_loss - full_loss).detach()
    diagonal = torch.arange(channels, device=utilities.device)
    utilities[:, diagonal, diagonal] = 0.0
    return utilities.detach()


def build_source_drop_history(history: HistoryBatch, source_index: int) -> HistoryBatch:
    """Return a copy with one historical source channel masked out."""
    if not 0 <= source_index < history.X_obs.shape[-1]:
        raise ValueError("source_index is outside the history channel dimension")
    values = history.X_obs.clone()
    values[:, :, source_index] = 0
    mask = None if history.M_obs is None else history.M_obs.clone()
    if mask is None:
        mask = torch.ones_like(history.X_obs)
    mask[:, :, source_index] = 0
    return HistoryBatch(
        X_obs=values,
        T_obs=history.T_obs.clone(),
        context=None if history.context is None else history.context.clone(),
        M_obs=mask,
    )


def estimate_counterfactual_utilities(
    predictor: Callable[[HistoryBatch], Tensor],
    history: HistoryBatch,
    targets,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    source_indices: Iterable[int],
) -> Tensor:
    """Estimate train-only source utility as the loss increase after source removal.

    ``loss_fn`` must return finite per-item losses with shape ``[batch, ...]``.
    A preserved target axis yields ``[batch, target, source]`` utilities. The
    predictor may be a detached checkpoint or a training model; labels are
    always detached.
    """
    indices = [int(index) for index in source_indices]
    if not indices:
        raise ValueError("source_indices must not be empty")
    target_batch = targets.shape[0] if isinstance(targets, Tensor) else next(
        (value.shape[0] for value in (
            getattr(targets, "values", None), getattr(targets, "class_label", None),
        ) if value is not None),
        None,
    )
    if target_batch != history.X_obs.shape[0]:
        raise ValueError("targets batch dimension must match history")
    with torch.no_grad():
        full_loss = loss_fn(predictor(history), targets)
        if full_loss.ndim < 1 or full_loss.shape[0] != history.X_obs.shape[0]:
            raise ValueError("loss_fn must return per-item losses with batch as the first dimension")
        if not torch.isfinite(full_loss).all():
            raise ValueError("loss_fn must return finite values")
        utilities = []
        for source_index in indices:
            dropped_history = build_source_drop_history(history, source_index)
            dropped_loss = loss_fn(predictor(dropped_history), targets)
            if dropped_loss.shape != full_loss.shape:
                raise ValueError("loss_fn must preserve the same per-item loss shape for every source")
            if not torch.isfinite(dropped_loss).all():
                raise ValueError("loss_fn must return finite values")
            utilities.append(counterfactual_utility(full_loss, dropped_loss))
    return torch.stack(utilities, dim=-1)
