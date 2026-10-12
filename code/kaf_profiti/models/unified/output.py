from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor


@dataclass
class PredictionOutput:
    dataset: str
    task: str
    encoder_version: str
    seed: int
    split_identity: str
    target_schema: str
    point: Optional[Tensor] = None
    quantiles: Optional[Tensor] = None
    detection_probability: Optional[Tensor] = None
    diagnosis_probability: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
