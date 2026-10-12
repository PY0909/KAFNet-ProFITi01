import importlib

import pytest
import torch


def test_cudtr_diagnostic_summary_reports_route_and_fallback_usage():
    module = importlib.import_module("kaf_profiti.experiments.cudtr_diagnostics")
    summary = module.summarize_cudtr_diagnostics(
        route_weights=torch.tensor([
            [[0.0, 0.7, 0.0], [0.4, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.2, 0.8]],
        ]),
        source_available=torch.tensor([
            [[False, True, False], [True, False, False]],
            [[False, False, False], [False, True, True]],
        ]),
        fallback_gate=torch.tensor([[0.2, 0.8], [1.0, 0.4]]),
        route_disagreement=torch.tensor([[0.1, 0.2], [0.0, 0.3]]),
    )

    assert summary["source_selected_count"] == [1, 2, 1]
    assert summary["source_selection_rate"] == pytest.approx([0.25, 0.5, 0.25])
    assert summary["source_mean_weight"] == pytest.approx([0.1, 0.225, 0.2])
    assert summary["unavailable_source_selected_count"] == 0
    assert summary["fallback_mean"] == pytest.approx(0.6)
    assert summary["fallback_p95"] == pytest.approx(0.97)
    assert summary["fallback_saturation_rate"] == pytest.approx(0.25)
    assert summary["route_disagreement_mean"] == pytest.approx(0.15)
    assert summary["route_inactive"] is False


def test_cudtr_metadata_excludes_fully_missing_sources():
    from kaf_profiti.models.unified import ModelConfig
    from kaf_profiti.models.unified.model import HistoryBatch, UnifiedModel

    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=1,
        split_identity="s", target_schema="r", variant="v3c",
        cudtr_enabled=True, cudtr_top_k=3,
    )
    history = HistoryBatch(
        torch.randn(1, 4, 3), torch.arange(4, dtype=torch.float32).unsqueeze(0),
        M_obs=torch.tensor([[[1., 0., 0.]] * 4]),
    )
    output = UnifiedModel(config).predict(history)
    diagnostics = output.metadata["cudtr"]

    assert diagnostics["source_available"][0, 0, 1:].sum().item() == 0
    assert diagnostics["route_weights"][0, 0, 1:].sum().item() == 0


def test_cudtr_uses_one_shared_pre_head_fusion_boundary():
    from kaf_profiti.models.unified import ModelConfig
    from kaf_profiti.models.unified.model import HistoryBatch, UnifiedModel

    config = ModelConfig(
        dataset="metropt3", task="state", input_dim=3, seed=1,
        split_identity="s", target_schema="state", variant="v3c",
        cudtr_enabled=True, cudtr_top_k=1,
    )
    model = UnifiedModel(config)
    output = model.predict(HistoryBatch(
        torch.randn(1, 5, 3), torch.arange(5, dtype=torch.float32).unsqueeze(0),
    ))

    assert output.metadata["cudtr"]["fusion_boundary"] == "pre_cross_shared_norm"


def test_cudtr_metadata_reports_fallback_saturation_and_inactive_route():
    from kaf_profiti.models.unified import ModelConfig
    from kaf_profiti.models.unified.model import HistoryBatch, UnifiedModel

    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=1,
        split_identity="s", target_schema="r", variant="v3c",
        cudtr_enabled=True,
    )
    output = UnifiedModel(config).predict(HistoryBatch(
        torch.randn(1, 5, 3), torch.arange(5, dtype=torch.float32).unsqueeze(0),
    ))

    assert "fallback_saturation_rate" in output.metadata["cudtr"]
    assert "route_inactive" in output.metadata["cudtr"]


def test_cudtr_training_uses_differentiable_dense_routing():
    from kaf_profiti.models.unified import ModelConfig
    from kaf_profiti.models.unified.model import HistoryBatch, UnifiedModel

    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=4, seed=1,
        split_identity="s", target_schema="r", variant="v3c",
        cudtr_enabled=True, cudtr_top_k=1,
    )
    model = UnifiedModel(config).train()
    output = model.predict(HistoryBatch(
        torch.randn(1, 5, 4), torch.arange(5, dtype=torch.float32).unsqueeze(0),
    ))

    assert (output.metadata["cudtr"]["route_weights"] > 0).sum(dim=-1).min().item() == 4


def test_metropt_manifest_semantics_do_not_claim_unused_global_pooling():
    from kaf_profiti.models.unified import ModelConfig
    from kaf_profiti.models.unified.model import HistoryBatch, UnifiedModel

    config = ModelConfig(
        dataset="metropt3", task="state", input_dim=3, seed=1,
        split_identity="s", target_schema="state", variant="v3c",
        cudtr_enabled=True,
    )
    output = UnifiedModel(config).predict(HistoryBatch(
        torch.randn(1, 5, 3), torch.arange(5, dtype=torch.float32).unsqueeze(0),
    ))

    assert output.metadata["representation_contract"] == "per_sensor_state_head"
