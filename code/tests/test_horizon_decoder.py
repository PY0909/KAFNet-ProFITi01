import torch

from kaf_profiti.models.unified import ModelConfig, UnifiedModel
from kaf_profiti.models.unified.model import HistoryBatch


def test_metropt_horizon_decoder_consumes_scale_tokens():
    config = ModelConfig(
        dataset="metropt3", task="state", input_dim=3, seed=1,
        split_identity="s", target_schema="state", variant="v3c",
        pred_len=4, head_mode="quantile", state_decoder="scale_horizon",
    )
    model = UnifiedModel(config)
    output = model.predict(HistoryBatch(
        torch.randn(2, 8, 3), torch.arange(8., dtype=torch.float32).expand(2, -1),
    ))
    assert output.quantiles.shape == (2, 4, 3, 3)
    assert torch.all(output.quantiles[..., 1:] >= output.quantiles[..., :-1])
    assert model.head.decoder_mode == "scale_horizon"


def test_non_metro_tasks_keep_global_decoder_contract():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=1,
        split_identity="s", target_schema="rul", variant="v3c",
        state_decoder="global",
    )
    assert config.state_decoder == "global"
