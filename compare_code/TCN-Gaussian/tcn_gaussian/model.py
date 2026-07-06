import math
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: Tensor) -> Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.out = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.out(y + residual)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for level in range(int(levels)):
            dilation = 2**level
            in_channels = input_dim if level == 0 else hidden_dim
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class TCNGaussian(nn.Module):
    """Causal TCN baseline with independent Gaussian prediction head."""

    def __init__(
        self,
        num_sensors: int,
        context_dim: int,
        pred_len: int,
        hidden_dim: int = 64,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        min_scale: float = 0.05,
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.context_dim = int(context_dim)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.min_scale = float(min_scale)

        input_dim = self.num_sensors * 2 + self.context_dim
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            levels=levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.pred_len * self.num_sensors * 2),
        )

    def _features(self, x_obs: Tensor, mask: Tensor, context: Tensor = None) -> Tensor:
        if x_obs.shape != mask.shape:
            raise ValueError("x_obs and mask must have the same shape")
        batch_size, history_len, num_sensors = x_obs.shape
        if num_sensors != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensors, got {num_sensors}")
        if context is None:
            context = torch.zeros(batch_size, 0, device=x_obs.device, dtype=x_obs.dtype)
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"Expected context_dim {self.context_dim}, got {context.shape[-1]}")
        context_seq = context[:, None, :].expand(batch_size, history_len, self.context_dim)
        features = torch.cat([x_obs * mask, mask, context_seq], dim=-1)
        return features.transpose(1, 2)

    def forward(self, x_obs: Tensor, mask: Tensor, context: Tensor = None) -> Tuple[Tensor, Tensor]:
        features = self._features(x_obs, mask, context)
        encoded = self.tcn(features)
        last_state = encoded[:, :, -1]
        raw = self.head(last_state)
        raw = raw.view(x_obs.shape[0], self.pred_len, self.num_sensors, 2)
        mean = raw[..., 0]
        scale = F.softplus(raw[..., 1]) + self.min_scale
        return mean, scale

    def nll(self, y: Tensor, mean: Tensor, scale: Tensor, mask: Tensor = None) -> Tensor:
        if mask is None:
            mask = torch.ones_like(y)
        log_prob = -0.5 * ((y - mean) / scale).pow(2) - torch.log(scale) - 0.5 * math.log(
            2.0 * math.pi
        )
        nll = -(log_prob * mask).sum() / mask.sum().clamp_min(1.0)
        return nll

    def sample(self, x_obs: Tensor, mask: Tensor, context: Tensor = None, nsamples: int = 100) -> Tensor:
        mean, scale = self.forward(x_obs, mask, context)
        eps = torch.randn(
            mean.shape[0],
            int(nsamples),
            mean.shape[1],
            mean.shape[2],
            device=mean.device,
            dtype=mean.dtype,
        )
        return mean.unsqueeze(1) + scale.unsqueeze(1) * eps

    @staticmethod
    def mse(y: Tensor, mean: Tensor, mask: Tensor = None) -> Tensor:
        if mask is None:
            mask = torch.ones_like(y)
        return (((mean - y) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
