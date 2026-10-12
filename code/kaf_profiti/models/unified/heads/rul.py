import torch
from torch import nn

from ..output import PredictionOutput
from .base import TaskHead, TaskTargets, pinball_loss


class RULHead(TaskHead):
    def __init__(self, hidden_dim: int, quantiles=(0.1, 0.5, 0.9), training_cap=None, over_penalty=1.0, input_mode="global"):
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        if any(q <= 0 or q >= 1 for q in self.quantiles) or tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("quantiles must be strictly ordered in (0, 1)")
        if over_penalty <= 0:
            raise ValueError("over_penalty must be positive")
        if input_mode not in {"global", "global_trend"}:
            raise ValueError("unsupported RUL input_mode: {}".format(input_mode))
        self.training_cap = training_cap
        self.over_penalty = float(over_penalty)
        self.input_mode = input_mode
        input_dim = hidden_dim * 2 if input_mode == "global_trend" else hidden_dim
        self.base = nn.Linear(input_dim, 1)
        self.increments = nn.Linear(input_dim, len(self.quantiles) - 1)

    def prepare_target(self, targets: TaskTargets):
        source = targets.raw_rul if targets.raw_rul is not None else targets.values
        if source is None:
            raise ValueError("RUL target is required")
        return self._apply_cap(source)

    def _apply_cap(self, target):
        return target.clamp_max(self.training_cap) if self.training_cap is not None else target

    def forward(self, encoded, mode="point") -> PredictionOutput:
        global_state = encoded.z_global
        if self.input_mode == "global_trend":
            if encoded.trend_features is None:
                raise ValueError("global_trend RUL head requires encoder trend_features")
            trend_state = encoded.trend_features.mean(dim=1)
            global_state = torch.cat((global_state, trend_state), dim=-1)
        convention = "raw_output_training_cap_{}".format("{:g}".format(self.training_cap)) if self.training_cap is not None else "raw"
        if mode not in {"point", "quantile", "probability"}:
            raise ValueError("RUL head mode must be point or quantile")
        base = torch.nn.functional.softplus(self.base(global_state))
        increments = torch.nn.functional.softplus(self.increments(global_state))
        quantiles = torch.cat((base, base + torch.cumsum(increments, dim=-1)), dim=-1)
        center_index = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
        point = quantiles[..., center_index]
        return PredictionOutput(dataset="cmapss_fd004", task="rul", encoder_version="unified_temporal_channelwise_v2",
            seed=-1, split_identity="unbound", target_schema="cmapss_fd004_rul_raw_v1", point=point,
            quantiles=None if mode == "point" else quantiles, metadata={"target_convention": convention, "rul_head_input": self.input_mode})

    def loss(self, prediction, targets: TaskTargets):
        target = self.prepare_target(targets)
        if target.ndim > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if prediction.point is not None:
            # NASA score 对高估呈指数惩罚，低估近似线性；over_penalty>1 使
            # L1 目标从条件中位数向低分位偏移，压低大额高估。
            error = prediction.point - target
            magnitude = error.abs()
            if self.over_penalty != 1.0:
                magnitude = torch.where(error > 0, self.over_penalty * magnitude, magnitude)
            return magnitude.mean()
        if prediction.quantiles is None:
            raise ValueError("RUL loss requires a point or quantile prediction")
        return pinball_loss(prediction.quantiles, target, self.quantiles)
