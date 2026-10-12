from typing import Iterable, Mapping

import torch
from torch.nn import functional as F

from kaf_profiti.models.unified.utility_teacher import (
    estimate_counterfactual_utilities,
    estimate_state_route_utilities,
)
from kaf_profiti.models.unified.model import HistoryBatch


def build_source_corruption(history, drop_rate=0.5, generator=None, protected_channels=None, source_count=None):
    """Drop random source channels consistently over each training history."""
    if not 0 < drop_rate <= 1:
        raise ValueError("drop_rate must be in (0, 1]")
    values = history.X_obs
    observed = torch.ones_like(values) if history.M_obs is None else history.M_obs.to(values)
    batch, _, channels = values.shape
    protected = set() if protected_channels is None else {int(index) for index in protected_channels}
    if any(index < 0 or index >= channels for index in protected):
        raise ValueError("protected_channels contains an invalid channel")
    candidates = [index for index in range(channels) if index not in protected]
    if not candidates:
        raise ValueError("protected_channels cannot contain every channel")
    if source_count is not None:
        if not 1 <= source_count <= len(candidates):
            raise ValueError("source_count must be between one and the number of candidate channels")
        dropped = torch.zeros((batch, channels), dtype=torch.bool, device=values.device)
        scores = torch.rand((batch, len(candidates)), device=values.device, generator=generator)
        chosen = scores.topk(source_count, dim=1).indices
        candidate_tensor = torch.tensor(candidates, device=values.device)
        dropped.scatter_(1, candidate_tensor[chosen], True)
    else:
        dropped = torch.rand((batch, channels), device=values.device, generator=generator) < drop_rate
    if protected:
        dropped[:, list(protected)] = False
    empty = ~dropped.any(dim=1)
    if empty.any():
        replacement_positions = torch.randint(len(candidates), (int(empty.sum()),), device=values.device, generator=generator)
        replacements = torch.tensor(candidates, device=values.device)[replacement_positions]
        dropped[empty, replacements] = True
    keep = (~dropped).to(values.dtype).unsqueeze(1)
    mask = observed * keep
    return HistoryBatch(
        X_obs=values * mask,
        T_obs=history.T_obs,
        context=history.context,
        M_obs=mask,
    )


def estimate_task_counterfactual_utilities(model, history, targets, teacher_model=None):
    """Build per-source train targets using a frozen baseline teacher."""
    if not getattr(getattr(model, "config", None), "cudtr_enabled", False):
        raise ValueError("counterfactual utility teacher requires CUDTR")
    teacher_model = model if teacher_model is None else teacher_model
    if teacher_model.config.task == "state" and teacher_model.config.dataset == "metropt3":
        return estimate_state_route_utilities(teacher_model, history, targets)

    def per_item_loss(prediction, task_targets):
        task = teacher_model.config.task
        if task == "state":
            truth = task_targets.values
            if prediction.quantiles is None:
                return (prediction.point - truth).abs().mean(dim=1)
            levels = torch.as_tensor(teacher_model.head.quantiles, device=truth.device, dtype=truth.dtype)
            error = truth.unsqueeze(-1) - prediction.quantiles
            pinball = torch.maximum(levels * error, (levels - 1.0) * error)
            return pinball.mean(dim=(1, 3))
        if task == "rul":
            truth = teacher_model.head.prepare_target(task_targets).reshape(-1)
            error = prediction.point.reshape(-1) - truth
            magnitude = error.abs()
            if teacher_model.head.over_penalty != 1.0:
                magnitude = torch.where(error > 0, teacher_model.head.over_penalty * magnitude, magnitude)
            return magnitude
        if task == "fault_detection_diagnosis":
            labels = task_targets.class_label.long()
            detection_logits = torch.logit(prediction.detection_probability.clamp(1e-6, 1 - 1e-6))
            detection = F.binary_cross_entropy_with_logits(
                detection_logits, (labels > 0).to(detection_logits.dtype), reduction="none"
            )
            diagnosis = F.nll_loss(
                prediction.diagnosis_probability.clamp_min(1e-7).log(),
                labels.clamp(0, teacher_model.head.num_fault_classes - 1), reduction="none",
            )
            return detection + diagnosis
        raise ValueError("unsupported task for CUDTR utility teacher: {}".format(task))

    utilities = estimate_counterfactual_utilities(
        lambda batch: teacher_model.predict(batch),
        history,
        targets,
        per_item_loss,
        range(teacher_model.config.input_dim),
    )
    if model.config.task == "state":
        diagonal = torch.arange(model.config.input_dim, device=utilities.device)
        utilities[:, diagonal, diagonal] = 0.0
    return utilities.detach()


@torch.no_grad()
def precompute_counterfactual_utilities(model, batches, teacher_model=None):
    """Compute train-only utility targets once and keep them on CPU."""
    outputs = []
    for history, targets in batches:
        outputs.append(estimate_task_counterfactual_utilities(
            model, history, targets, teacher_model=teacher_model,
        ).cpu())
    if not outputs:
        raise ValueError("utility precomputation requires at least one batch")
    return torch.cat(outputs, dim=0)


def train_step(model, optimizer, history, targets, device=None, scaler=None, amp=False,
               utility_targets=None, utility_teacher_model=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device is not None:
        history = history.to(device)
        targets = targets.to(device)
    if utility_teacher_model is not None:
        utility_targets = estimate_task_counterfactual_utilities(
            model, history, targets, teacher_model=utility_teacher_model,
        )
    device_type = history.X_obs.device.type
    with torch.autocast(device_type=device_type, enabled=amp and device_type in {"cuda", "cpu"}):
        prediction = model.predict(history)
        loss = model.loss(prediction, targets)
        if isinstance(loss, dict):
            loss = loss["total"]
        auxiliary = getattr(model, "cudtr_auxiliary_loss", None)
        loss_config = getattr(model, "cudtr_loss_config", None)
        auxiliary_active = loss_config is not None and any((
            loss_config.lambda_distill, loss_config.lambda_rank,
            loss_config.lambda_sparse, loss_config.lambda_fallback,
        ))
        if utility_targets is not None or auxiliary_active:
            if auxiliary is None:
                raise ValueError("CUDTR auxiliary losses require cudtr_auxiliary_loss")
            fallback_active = loss_config is not None and loss_config.lambda_fallback > 0
            corrupted = build_source_corruption(
                history,
                drop_rate=model.config.cudtr_fallback_drop_rate,
            ) if fallback_active else None
            terms = auxiliary(
                utility_targets=None if utility_targets is None else utility_targets.to(history.X_obs.device),
                corrupted_history=corrupted,
            )
            loss = loss + terms["total"]
    if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("training loss must be a finite scalar")
    if scaler is not None and amp and device_type == "cuda":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def evaluate(model, batches: Iterable, split="validation", evaluate_test=False):
    if split == "test" and not evaluate_test:
        raise ValueError("test evaluation requires explicit evaluate_test=True")
    model.eval()
    weighted_loss = 0.0
    example_count = 0
    batch_count = 0
    for history, targets in batches:
        prediction = model.predict(history)
        loss = model.loss(prediction, targets)
        if isinstance(loss, dict):
            loss = loss["total"]
        if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("evaluation loss must be a finite scalar")
        batch_size = int(history.X_obs.shape[0])
        weighted_loss += float(loss.detach().cpu()) * batch_size
        example_count += batch_size
        batch_count += 1
    if not example_count:
        raise ValueError("evaluation requires at least one batch")
    return {
        "split": split,
        "loss": weighted_loss / example_count,
        "batches": batch_count,
        "examples": example_count,
    }


def select_best_validation(history: Iterable[Mapping]):
    records = list(history)
    if not records or any(record.get("split", "validation") != "validation" for record in records):
        raise ValueError("checkpoint selection accepts validation records only")
    return min(records, key=lambda record: float(record["metric"]))
