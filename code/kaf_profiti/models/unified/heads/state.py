from typing import Sequence

import torch
from torch import Tensor, nn

from ..output import PredictionOutput
from .base import TaskHead, TaskTargets, pinball_loss


class StateHead(TaskHead):
    def __init__(self, hidden_dim: int, num_sensors: int, pred_len: int, quantiles=(0.1, 0.5, 0.9), decoder_mode="global"):
        super().__init__()
        if num_sensors < 1 or pred_len < 1:
            raise ValueError("num_sensors and pred_len must be positive")
        self.num_sensors = num_sensors
        self.pred_len = pred_len
        if decoder_mode not in {"global", "scale_horizon"}:
            raise ValueError("decoder_mode must be global or scale_horizon")
        self.decoder_mode = decoder_mode
        self.quantiles = tuple(float(q) for q in quantiles)
        if any(q <= 0 or q >= 1 for q in self.quantiles) or tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("quantiles must be strictly ordered in (0, 1)")
        if decoder_mode == "scale_horizon":
            self.scale_query = nn.Parameter(torch.randn(pred_len, hidden_dim) * 0.02)
            self.scale_fusion = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.base = nn.Linear(hidden_dim, pred_len)
        self.increments = nn.Linear(hidden_dim, pred_len * (len(self.quantiles) - 1))

    def forward(self, encoded, mode="point") -> PredictionOutput:
        if encoded.z_history.shape[1] != self.num_sensors:
            raise ValueError("encoded sensor count does not match state head")
        sensor = encoded.z_history
        if self.decoder_mode == "scale_horizon":
            if encoded.patch_tokens is None or encoded.patch_tokens.ndim != 4:
                raise ValueError("scale_horizon decoder requires encoded patch/scale tokens")
            scale_tokens = encoded.patch_tokens.clone()
            scale_tokens[:, :, 0, :] = sensor
            scale_summary = scale_tokens.mean(dim=2)
            horizon_context = self.scale_query.unsqueeze(0).unsqueeze(0) + scale_summary.unsqueeze(2)
            sensor = self.scale_fusion(horizon_context).permute(0, 2, 1, 3)
            base_all = self.base(sensor)
            horizon_index = torch.arange(self.pred_len, device=sensor.device)
            base = base_all[:, horizon_index, :, horizon_index].permute(1, 0, 2).unsqueeze(-1)
            increment_all = torch.nn.functional.softplus(self.increments(sensor)).view(
                sensor.shape[0], self.pred_len, self.num_sensors, self.pred_len, len(self.quantiles) - 1
            )
            increments = increment_all[:, horizon_index, :, horizon_index, :].permute(1, 0, 2, 3)
            quantiles = torch.cat((base, base + torch.cumsum(increments, dim=-1)), dim=-1)
            center_index = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
            point = quantiles[..., center_index]
            return PredictionOutput(dataset="metropt3", task="state", encoder_version="unified_temporal_channelwise_v2",
                seed=-1, split_identity="unbound", target_schema="metropt3_state_v1", point=point,
                quantiles=None if mode == "point" else quantiles)
        if mode not in {"point", "quantile", "probability"}:
            raise ValueError("state head mode must be point or quantile")
        base = self.base(sensor).transpose(1, 2).unsqueeze(-1)
        increments = torch.nn.functional.softplus(self.increments(sensor)).view(
            sensor.shape[0], self.num_sensors, self.pred_len, len(self.quantiles) - 1
        ).permute(0, 2, 1, 3)
        quantiles = torch.cat((base, base + torch.cumsum(increments, dim=-1)), dim=-1)
        center_index = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
        point = quantiles[..., center_index]
        return PredictionOutput(dataset="metropt3", task="state", encoder_version="unified_temporal_channelwise_v2",
            seed=-1, split_identity="unbound", target_schema="metropt3_state_v1", point=point,
            quantiles=None if mode == "point" else quantiles)

    def loss(self, prediction, targets: TaskTargets):
        if prediction.quantiles is not None:
            return pinball_loss(prediction.quantiles, targets.values, self.quantiles)
        if prediction.point is None:
            return (prediction.point - targets.values).abs().mean()
        error = prediction.point - targets.values
        return error.abs().mean()
