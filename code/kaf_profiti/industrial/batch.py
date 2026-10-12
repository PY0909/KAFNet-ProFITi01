from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from kaf_profiti.models.unified.heads.base import TaskTargets


@dataclass(frozen=True)
class HistorySample:
    X_obs: Tensor
    T_obs: Tensor
    context: Optional[Tensor] = None
    target: Optional[Tensor] = None
    target_schema: str = ""
    class_label: Optional[Tensor] = None
    fault_number: Optional[int] = None
    simulation_run: Optional[int] = None
    metadata: Dict[str, object] = None
    M_obs: Optional[Tensor] = None


@dataclass
class HistoryBatch:
    X_obs: Tensor
    T_obs: Tensor
    context: Optional[Tensor]
    targets: TaskTargets
    M_obs: Optional[Tensor] = None

    def to(self, device):
        return HistoryBatch(
            X_obs=self.X_obs.to(device), T_obs=self.T_obs.to(device),
            context=None if self.context is None else self.context.to(device),
            targets=self.targets.to(device),
            M_obs=None if self.M_obs is None else self.M_obs.to(device),
        )


def collate_history(samples: List[HistorySample]) -> HistoryBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    first = samples[0]
    if any(sample.X_obs.shape != first.X_obs.shape for sample in samples):
        raise ValueError("all history tensors must have equal shape")
    context = None
    if first.context is not None:
        context = torch.stack([sample.context for sample in samples])
    target = None
    if first.target is not None:
        target = torch.stack([sample.target for sample in samples])
    class_label = None
    if first.class_label is not None:
        class_label = torch.stack([sample.class_label for sample in samples]).long()
    targets = TaskTargets(
        values=target,
        schema=first.target_schema,
        class_label=class_label,
        fault_number=None if first.fault_number is None else torch.tensor([sample.fault_number for sample in samples]),
        simulation_run=None if first.simulation_run is None else torch.tensor([sample.simulation_run for sample in samples]),
        raw_rul=None if not first.metadata or "raw_rul" not in first.metadata else torch.tensor([sample.metadata["raw_rul"] for sample in samples], dtype=torch.float32),
        capped_rul=target,
    )
    return HistoryBatch(
        X_obs=torch.stack([sample.X_obs for sample in samples]),
        T_obs=torch.stack([sample.T_obs for sample in samples]),
        context=context,
        targets=targets,
        M_obs=torch.stack([
            torch.ones_like(sample.X_obs) if sample.M_obs is None else sample.M_obs
            for sample in samples
        ]),
    )
