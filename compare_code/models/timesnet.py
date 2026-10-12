import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import ForecastBaseline


class TimesBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.inception = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, top_k):
        length = x.shape[1]
        spectrum = torch.fft.rfft(x, dim=1).abs().mean(dim=(0, 2))
        available = max(0, spectrum.numel() - 1)
        if available == 0:
            return self.norm(x + self.inception(x.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2))
        spectrum = spectrum.clone()
        spectrum[0] = 0
        count = min(top_k, available)
        amplitudes, bins = torch.topk(spectrum[1:], count)
        outputs = []
        frequency_bins = (bins + 1).detach().cpu().tolist()
        for frequency_bin in frequency_bins:
            period = max(1, length // frequency_bin)
            segments = math.ceil(length / period)
            padded_length = segments * period
            padded = F.pad(x, (0, 0, 0, padded_length - length)) if padded_length > length else x
            grid = padded.reshape(padded.shape[0], segments, period, x.shape[-1]).permute(0, 3, 1, 2)
            transformed = self.inception(grid)
            restored = transformed.permute(0, 2, 3, 1).reshape(x.shape[0], padded_length, x.shape[-1])
            outputs.append(restored[:, :length])
        weights = torch.softmax(amplitudes, dim=0)
        mixed = sum(weight * output for weight, output in zip(weights, outputs))
        return self.norm(x + mixed)


class TimesNet(ForecastBaseline):
    """FFT period selection and 2D inception blocks for regular sequences."""

    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            TimesBlock(config.input_dim, config.d_model, config.dropout)
            for _ in range(config.layers)
        ])
        self.temporal_projection = nn.Linear(config.history_len, config.pred_len)

    def _forecast(self, x):
        for block in self.blocks:
            x = block(x, self.config.top_k)
        return self.temporal_projection(x.transpose(1, 2)).transpose(1, 2)
