import torch

from kaf_profiti.models.unified import ModelConfig, UnifiedModel
from kaf_profiti.models.unified.model import HistoryBatch


def test_cudtr_only_disables_dense_encoder_cross_channel_attention():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=4, seed=1,
        split_identity="s", target_schema="r", variant="v3c",
        cudtr_enabled=True, cross_channel_mode="cudtr_only",
    )
    model = UnifiedModel(config)
    assert model.encoder.cross_channel_enabled is False
    output = model.predict(HistoryBatch(
        torch.randn(2, 5, 4), torch.arange(5., dtype=torch.float32).expand(2, -1),
    ))
    assert output.metadata["cudtr"]["cross_channel_mode"] == "cudtr_only"


def test_attention_mode_keeps_dense_encoder_cross_channel_attention():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=4, seed=1,
        split_identity="s", target_schema="r", variant="v3c",
        cudtr_enabled=True, cross_channel_mode="attention",
    )
    assert UnifiedModel(config).encoder.cross_channel_enabled is True
