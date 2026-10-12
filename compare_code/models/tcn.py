import torch.nn.functional as F
from torch import nn

from .base import ForecastBaseline


class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(channels, 3, dilation), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, 3, dilation), nn.GELU(), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        normalized = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x + self.layers(normalized)


class TCN(ForecastBaseline):
    def __init__(self, config):
        super().__init__(config)
        self.input_projection = nn.Conv1d(config.input_dim, config.d_model, kernel_size=1)
        self.blocks = nn.Sequential(*[
            ResidualTemporalBlock(config.d_model, 2 ** index, config.dropout)
            for index in range(config.layers)
        ])
        self.output_projection = nn.Linear(config.d_model, config.pred_len * config.input_dim)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        nn.init.zeros_(self.output_projection.bias)

    def _forecast(self, x):
        encoded = self.blocks(self.input_projection(x.transpose(1, 2)))[:, :, -1]
        return self.output_projection(encoded).view(-1, self.config.pred_len, self.config.input_dim)
