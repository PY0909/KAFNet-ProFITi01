import math

import torch.nn.functional as F
import torch
from torch import nn

from .base import ForecastBaseline


class PatchTST(ForecastBaseline):
    """Channel-independent patch encoding with shared temporal weights."""

    def __init__(self, config):
        super().__init__(config)
        self.patch_count = math.ceil(config.history_len / config.patch_len)
        self.patch_embedding = nn.Conv1d(1, config.d_model, config.patch_len, stride=config.patch_len)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.channel_identity = nn.Parameter(torch.zeros(config.input_dim, config.d_model))
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, config.d_model))
        self.output_projection = nn.Linear(self.patch_count * config.d_model, config.pred_len)
        nn.init.normal_(self.channel_identity, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def _forecast(self, x):
        batch, history, channels = x.shape
        values = x.transpose(1, 2).reshape(batch * channels, 1, history)
        padded_length = self.patch_count * self.config.patch_len
        if padded_length > history:
            values = F.pad(values, (0, padded_length - history), mode="replicate")
        tokens = self.patch_embedding(values).transpose(1, 2)
        tokens = tokens.reshape(batch, channels, self.patch_count, self.config.d_model)
        tokens = tokens + self.channel_identity[None, :, None, :] + self.position[:, None, :, :]
        tokens = self.transformer(tokens.reshape(batch * channels, self.patch_count, self.config.d_model))
        forecast = self.output_projection(tokens.flatten(start_dim=1))
        return forecast.view(batch, channels, self.config.pred_len).transpose(1, 2)
