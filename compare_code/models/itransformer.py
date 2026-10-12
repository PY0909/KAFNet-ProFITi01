import torch
from torch import nn

from .base import ForecastBaseline


class ITransformer(ForecastBaseline):
    """Inverted attention: each sensor history is encoded as one token."""

    def __init__(self, config):
        super().__init__(config)
        self.temporal_projection = nn.Linear(config.history_len, config.d_model)
        self.channel_identity = nn.Parameter(torch.zeros(config.input_dim, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.output_projection = nn.Linear(config.d_model, config.pred_len)
        nn.init.normal_(self.channel_identity, std=0.02)

    def _forecast(self, x):
        tokens = self.temporal_projection(x.transpose(1, 2))
        tokens = tokens + self.channel_identity.unsqueeze(0)
        encoded = self.transformer(tokens)
        return self.output_projection(encoded).transpose(1, 2)
