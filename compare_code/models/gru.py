from torch import nn

from .base import ForecastBaseline


class GRU(ForecastBaseline):
    def __init__(self, config):
        super().__init__(config)
        self.recurrent = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.d_model,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(config.d_model, config.pred_len * config.input_dim)

    def _forecast(self, x):
        _, hidden = self.recurrent(x)
        return self.output_projection(hidden[-1]).view(-1, self.config.pred_len, self.config.input_dim)
