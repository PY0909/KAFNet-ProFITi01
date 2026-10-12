from dataclasses import asdict, dataclass
from typing import Any, Dict


ENCODER_VERSION = "unified_temporal_channelwise_v2"
ENCODER_VERSIONS = {
    "v2": ENCODER_VERSION,
    "v2t": "unified_temporal_channelwise_v2t",
    "v2tc": "unified_temporal_channelwise_v2tc",
    "v3a": "unified_temporal_channelwise_v3a",
    "v3b": "unified_temporal_channelwise_v3b",
    "v3c": "unified_temporal_channelwise_v3c",
    "v4": "unified_temporal_channelwise_v4",
    "v5a": "unified_temporal_patchwise_v5a",
    "v5b": "unified_temporal_patchwise_v5b",
    "v5c": "unified_temporal_patchwise_v5c",
}
DATASET_TASKS = {
    "metropt3": "state",
    "cmapss_fd004": "rul",
    "tep": "fault_detection_diagnosis",
}


@dataclass(frozen=True)
class ModelConfig:
    dataset: str
    task: str
    input_dim: int
    seed: int
    split_identity: str
    target_schema: str
    context_dim: int = 0
    hidden_dim: int = 64
    patch_lens: tuple = (12, 24, 48)
    pred_len: int = 1
    num_fault_classes: int = 1
    head_mode: str = "point"
    training_cap: Any = 125.0
    rul_over_penalty: float = 1.0
    rul_head_input: str = "global"
    sensor_pooling: str = "attention"
    cross_channel_mode: str = "attention"
    state_decoder: str = "global"
    tep_class_weights: tuple = ()
    encoder_version: str = ENCODER_VERSION
    variant: str = "v2"
    parameter_scope: str = "dataset_independent"
    cudtr_enabled: bool = False
    cudtr_top_k: int = 2
    cudtr_query_dim: int = 8
    cudtr_temperature: float = 1.0
    # Neutral bounded initialization avoids permanently suppressing routing.
    cudtr_fallback_init: float = 0.5
    cudtr_lambda_distill: float = 0.0
    cudtr_lambda_rank: float = 0.0
    cudtr_lambda_sparse: float = 0.0
    cudtr_lambda_fallback: float = 0.0
    cudtr_rank_margin: float = 0.1
    cudtr_utility_clip: Any = None
    cudtr_fallback_drop_rate: float = 0.5
    cudtr_min_observation_rate: float = 0.0
    cudtr_utility_threshold: Any = None

    def __post_init__(self):
        expected_task = DATASET_TASKS.get(self.dataset)
        if expected_task is None:
            raise ValueError("unsupported dataset: {}".format(self.dataset))
        if self.task != expected_task:
            raise ValueError(
                "dataset {!r} requires task {!r}".format(self.dataset, expected_task)
            )
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.context_dim < 0 or self.hidden_dim < 1:
            raise ValueError("context_dim must be non-negative and hidden_dim positive")
        if self.pred_len < 1 or self.num_fault_classes < 1:
            raise ValueError("pred_len and num_fault_classes must be positive")
        if self.head_mode not in {"point", "quantile", "probability"}:
            raise ValueError("unsupported head_mode: {}".format(self.head_mode))
        if self.task == "fault_detection_diagnosis" and self.head_mode == "point":
            object.__setattr__(self, "head_mode", "probability")
        if not isinstance(self.rul_over_penalty, (int, float)) or self.rul_over_penalty <= 0:
            raise ValueError("rul_over_penalty must be a positive number")
        if self.rul_head_input not in {"global", "global_trend"}:
            raise ValueError("unsupported rul_head_input: {}".format(self.rul_head_input))
        if self.sensor_pooling not in {"attention", "mean", "mean_attention"}:
            raise ValueError("unsupported sensor_pooling: {}".format(self.sensor_pooling))
        if self.cross_channel_mode not in {"attention", "cudtr_only"}:
            raise ValueError("cross_channel_mode must be attention or cudtr_only")
        if self.state_decoder not in {"global", "scale_horizon"}:
            raise ValueError("state_decoder must be global or scale_horizon")
        if self.state_decoder == "scale_horizon" and self.task != "state":
            raise ValueError("scale_horizon decoder is only valid for state task")
        if self.task == "fault_detection_diagnosis":
            if self.tep_class_weights and len(self.tep_class_weights) != self.num_fault_classes:
                raise ValueError("tep_class_weights must match num_fault_classes")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.split_identity.strip() or not self.target_schema.strip():
            raise ValueError("split_identity and target_schema must be explicit")
        if self.variant not in ENCODER_VERSIONS:
            raise ValueError("unsupported encoder variant: {}".format(self.variant))
        expected_version = ENCODER_VERSIONS[self.variant]
        if self.encoder_version == ENCODER_VERSION and self.variant != "v2":
            object.__setattr__(self, "encoder_version", expected_version)
        elif self.encoder_version != expected_version:
            raise ValueError("unsupported encoder version: {}".format(self.encoder_version))
        if self.parameter_scope != "dataset_independent":
            raise ValueError("parameters must be trained independently per dataset")
        if not isinstance(self.cudtr_enabled, bool):
            raise ValueError("cudtr_enabled must be boolean")
        if self.cudtr_top_k < 1 or self.cudtr_query_dim < 1:
            raise ValueError("cudtr_top_k and cudtr_query_dim must be positive")
        if self.cudtr_temperature <= 0:
            raise ValueError("cudtr_temperature must be positive")
        if not 0 <= self.cudtr_fallback_init <= 1:
            raise ValueError("cudtr_fallback_init must be in [0, 1]")
        for name in ("cudtr_lambda_distill", "cudtr_lambda_rank", "cudtr_lambda_sparse", "cudtr_lambda_fallback"):
            if getattr(self, name) < 0:
                raise ValueError("{} must be non-negative".format(name))
        if self.cudtr_rank_margin < 0:
            raise ValueError("cudtr_rank_margin must be non-negative")
        if self.cudtr_utility_clip is not None and self.cudtr_utility_clip <= 0:
            raise ValueError("cudtr_utility_clip must be positive")
        if not 0 < self.cudtr_fallback_drop_rate <= 1:
            raise ValueError("cudtr_fallback_drop_rate must be in (0, 1]")
        if not 0 <= self.cudtr_min_observation_rate <= 1:
            raise ValueError("cudtr_min_observation_rate must be in [0, 1]")
        if self.cudtr_utility_threshold is not None and not isinstance(self.cudtr_utility_threshold, (int, float)):
            raise ValueError("cudtr_utility_threshold must be numeric or None")
        if not self.patch_lens or any(length < 1 for length in self.patch_lens):
            raise ValueError("patch_lens must contain positive lengths")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["patch_lens"] = list(self.patch_lens)
        result["pooling"] = "ordered_patch" if self.variant in {"v5a", "v5b", "v5c"} else ("attention" if self.variant in {"v3a", "v3b", "v3c", "v4"} else "mean")
        result["trend_features"] = self.variant in {"v2t", "v2tc", "v3c", "v4"}
        result["sensor_pooling"] = self.sensor_pooling if self.variant in {"v3c", "v5a", "v5b", "v5c"} else (
            "attention" if self.variant == "v2tc" else "mean"
        )
        result["patch_token_count"] = "ceil(history_len/min_patch_len)" if self.variant in {"v5a", "v5b", "v5c"} else (len(self.patch_lens) + 1 if self.variant in {"v3b", "v3c", "v4"} else 0)
        result["ordered_patch_tokens"] = self.variant in {"v5a", "v5b", "v5c"}
        result["temporal_mixer"] = "depthwise_patch_mixer" if self.variant in {"v5b", "v5c"} else ("depthwise_dilated_tcn" if self.variant == "v4" else "pointwise_mlp")
        return result
