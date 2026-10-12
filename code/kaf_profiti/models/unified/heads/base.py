from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..output import PredictionOutput


@dataclass(frozen=True)
class TaskTargets:
    """Training/evaluation labels, intentionally separate from HistoryBatch."""

    values: Optional[Tensor]
    schema: str
    class_label: Optional[Tensor] = None
    fault_number: Optional[Tensor] = None
    simulation_run: Optional[Tensor] = None
    raw_rul: Optional[Tensor] = None
    capped_rul: Optional[Tensor] = None

    def to(self, device):
        fields = {}
        for name in ("values", "class_label", "fault_number", "simulation_run", "raw_rul", "capped_rul"):
            value = getattr(self, name)
            fields[name] = value.to(device) if value is not None else None
        return TaskTargets(schema=self.schema, **fields)


class TaskHead(nn.Module):
    def forward(self, encoded, horizon=None):
        raise NotImplementedError

    def loss(self, prediction, targets: TaskTargets):
        raise NotImplementedError


def pinball_loss(prediction: Tensor, target: Tensor, quantiles) -> Tensor:
    target = target.unsqueeze(-1) if prediction.ndim == target.ndim + 1 else target
    q = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device)
    error = target - prediction
    loss = torch.maximum(q * error, (q - 1.0) * error)
    return loss.mean()
