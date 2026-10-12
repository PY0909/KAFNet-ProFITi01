import os
from dataclasses import dataclass
from typing import Iterable

import torch


def split_identity(unit_ids: Iterable[str], algorithm="stable_unit_split_v1") -> str:
    units = sorted(str(unit) for unit in unit_ids)
    if not units:
        raise ValueError("split identity requires at least one unit")
    return "{}-units{}".format(algorithm, len(units))


@dataclass
class TrainNormalizer:
    source_split: str = ""
    mean: torch.Tensor = None
    std: torch.Tensor = None
    schema: str = ""
    identity: str = ""

    def fit(self, values: torch.Tensor, split="train"):
        if split != "train":
            raise ValueError("normalization statistics must come from train split")
        finite = values.float()
        if not torch.isfinite(finite).all():
            raise ValueError("normalizer cannot fit non-finite values")
        self.mean = finite.mean(dim=0)
        self.std = finite.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.source_split = split
        self.schema = "features-{}".format(self.mean.numel())
        self.identity = "train-zscore-features{}".format(self.mean.numel())
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None or self.source_split != "train":
            raise RuntimeError("normalizer must be fitted on train split")
        if not torch.isfinite(values.float()).all():
            raise ValueError("normalizer cannot transform non-finite values")
        return (values.float() - self.mean) / self.std


def build_manifest(**fields):
    checkpoint = fields.get("checkpoint")
    if checkpoint and os.path.isabs(checkpoint):
        raise ValueError("manifest paths must be relative to the result root")
    required = {
        "run_id", "dataset", "task", "seed", "split_identity", "target_schema",
        "encoder_version", "checkpoint", "dataset_schema", "normalizer_identity",
        "target_convention", "evaluator_version", "model_version",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError("manifest missing fields: {}".format(", ".join(missing)))
    return dict(fields)
