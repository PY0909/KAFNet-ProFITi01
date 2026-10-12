import torch
import pytest

from kaf_profiti.models.unified import ModelConfig
from kaf_profiti.models.unified.model import HistoryBatch, UnifiedModel, _safe_quantile


def _history(channels):
    return HistoryBatch(
        X_obs=torch.randn(2, 5, channels),
        T_obs=torch.arange(5, dtype=torch.float32).expand(2, -1),
    )


@pytest.mark.parametrize(
    "dataset,task,channels",
    [("metropt3", "state", 3), ("cmapss_fd004", "rul", 3), ("tep", "fault_detection_diagnosis", 3)],
)
def test_unified_model_routes_dataset_specific_head(dataset, task, channels):
    config = ModelConfig(
        dataset=dataset,
        task=task,
        input_dim=channels,
        seed=2026,
        split_identity="fixture-split-v1",
        target_schema="fixture-target-v1",
        pred_len=4,
        num_fault_classes=3,
    )
    model = UnifiedModel(config)
    output = model.predict(_history(channels))
    assert output.dataset == dataset
    assert output.task == task
    assert output.encoder_version == config.encoder_version
    assert output.seed == 2026
    assert output.split_identity == "fixture-split-v1"
    assert output.target_schema == "fixture-target-v1"


def test_parameter_count_reports_encoder_head_and_total():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=2, seed=1,
        split_identity="s", target_schema="r", pred_len=1,
    )
    counts = UnifiedModel(config).parameter_count()
    assert set(counts) == {"encoder", "head", "routing", "queries", "total"}
    assert counts["total"] == sum(counts[key] for key in ("encoder", "head", "routing", "queries"))


def test_cudtr_default_initialization_keeps_finite_routed_prediction():
    torch.manual_seed(17)
    baseline_config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=2026,
        split_identity="fixture-baseline", target_schema="rul", variant="v3c",
    )
    baseline = UnifiedModel(baseline_config)
    cudtr_config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=2026,
        split_identity="fixture-baseline", target_schema="rul", variant="v3c",
        cudtr_enabled=True,
    )
    cudtr = UnifiedModel(cudtr_config)
    cudtr.encoder.load_state_dict(baseline.encoder.state_dict())
    cudtr.head.load_state_dict(baseline.head.state_dict())
    history = _history(3)
    actual = cudtr.predict(history).point

    assert torch.isfinite(actual).all()
    assert torch.all(cudtr.predict(history).metadata["cudtr"]["fallback_gate"] < 0.99)


def test_cudtr_fallback_diagnostics_accept_mixed_precision():
    config = ModelConfig(
        dataset="metropt3", task="state", input_dim=3, seed=1,
        split_identity="s", target_schema="state-v1", pred_len=2,
        variant="v3c", cudtr_enabled=True,
    )
    model = UnifiedModel(config)
    history = _history(3)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model.predict(history)
    assert torch.isfinite(output.metadata["cudtr"]["fallback_p95"]).all()


def test_safe_quantile_promotes_non_float_fallback_tensor():
    fallback = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int64)
    result = _safe_quantile(fallback, 0.95, dim=1)
    assert result.dtype.is_floating_point
    assert torch.isfinite(result).all()


def test_parameter_count_includes_cudtr_router_and_target_queries():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=1,
        split_identity="s", target_schema="r", variant="v3c", cudtr_enabled=True,
    )
    model = UnifiedModel(config)
    counts = model.parameter_count()

    assert counts["routing"] == sum(parameter.numel() for parameter in model.cudtr.parameters())
    assert counts["queries"] == model.cudtr_queries.numel()
    assert counts["total"] == sum(counts[key] for key in ("encoder", "head", "routing", "queries"))


def test_cudtr_routes_with_observation_variability_and_context_features():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, context_dim=2, seed=1,
        split_identity="s", target_schema="r", variant="v3c", cudtr_enabled=True,
    )
    model = UnifiedModel(config)
    assert model.cudtr.source_feature_dim == 4
    history = HistoryBatch(
        X_obs=torch.tensor([[[1.0, 2.0, 3.0], [2.0, 2.0, 7.0], [3.0, 2.0, 1.0]]]),
        T_obs=torch.arange(3, dtype=torch.float32).unsqueeze(0),
        M_obs=torch.tensor([[[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]),
        context=torch.tensor([[0.25, -0.5]]),
    )
    output = model.predict(history)
    assert output.metadata["cudtr"]["source_feature_dim"] == 4
    assert torch.isfinite(output.point).all()


def test_cudtr_source_features_ignore_values_at_missing_positions():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=1,
        split_identity="s", target_schema="r", variant="v3c", cudtr_enabled=True,
    )
    model = UnifiedModel(config)
    history = _history(3)
    mask = torch.tensor([[[1.0, 0.0, 1.0]]]).expand_as(history.X_obs)
    first = HistoryBatch(history.X_obs, history.T_obs, M_obs=mask)
    changed = history.X_obs.clone()
    changed[mask == 0] = 1e6
    second = HistoryBatch(changed, history.T_obs, M_obs=mask)
    first_output = model.predict(first).point
    second_output = model.predict(second).point
    assert torch.allclose(first_output, second_output, atol=1e-6, rtol=1e-6)


def test_cudtr_minimum_observation_rate_marks_sparse_source_unavailable():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=1,
        split_identity="s", target_schema="r", variant="v3c", cudtr_enabled=True,
        cudtr_min_observation_rate=0.5,
    )
    model = UnifiedModel(config)
    history = HistoryBatch(
        torch.randn(1, 4, 3), torch.arange(4, dtype=torch.float32).unsqueeze(0),
        M_obs=torch.tensor([[[1., 1., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]]),
    )
    output = model.predict(history)
    # source 1 is observed in only one of four positions and cannot be routed
    assert output.metadata["cudtr"]["route_weights"][0, 0, 1].item() == 0.0


@pytest.mark.parametrize(
    "dataset,task,channels",
    [("metropt3", "state", 3), ("cmapss_fd004", "rul", 3), ("tep", "fault_detection_diagnosis", 3)],
)
def test_v3a_routes_all_task_heads_and_records_version(dataset, task, channels):
    config = ModelConfig(
        dataset=dataset, task=task, input_dim=channels, seed=2026,
        split_identity="fixture-v3a", target_schema="fixture-target-v1",
        pred_len=4, num_fault_classes=3, variant="v3a",
    )
    model = UnifiedModel(config)
    output = model.predict(_history(channels))

    assert output.encoder_version == "unified_temporal_channelwise_v3a"
    assert output.metadata["config"]["variant"] == "v3a"


@pytest.mark.parametrize(
    "dataset,task,channels",
    [("metropt3", "state", 3), ("cmapss_fd004", "rul", 3), ("tep", "fault_detection_diagnosis", 3)],
)
def test_v3b_routes_all_task_heads_and_records_patch_schema(dataset, task, channels):
    config = ModelConfig(
        dataset=dataset, task=task, input_dim=channels, seed=2026,
        split_identity="fixture-v3b", target_schema="fixture-target-v1",
        pred_len=4, num_fault_classes=3, patch_lens=(2, 4), variant="v3b",
    )
    model = UnifiedModel(config)
    output = model.predict(_history(channels))

    assert output.encoder_version == "unified_temporal_channelwise_v3b"
    assert output.metadata["config"]["variant"] == "v3b"
    assert output.metadata["patch_token_count"] == 3
    assert output.metadata["patch_lens"] == [2, 4]


@pytest.mark.parametrize(
    "dataset,task,channels",
    [("metropt3", "state", 3), ("cmapss_fd004", "rul", 3), ("tep", "fault_detection_diagnosis", 3)],
)
def test_v3c_routes_all_task_heads_and_records_sensor_pooling(dataset, task, channels):
    config = ModelConfig(
        dataset=dataset, task=task, input_dim=channels, seed=2026,
        split_identity="fixture-v3c", target_schema="fixture-target-v1",
        pred_len=4, num_fault_classes=3, patch_lens=(2, 4), variant="v3c",
    )
    model = UnifiedModel(config)
    output = model.predict(_history(channels))

    assert output.encoder_version == "unified_temporal_channelwise_v3c"
    assert output.metadata["config"]["variant"] == "v3c"
    assert output.metadata["config"]["sensor_pooling"] == "attention"
    assert output.metadata["patch_token_count"] == 3
    assert output.metadata["sensor_pooling"] == "attention"


def test_v3c_rul_global_trend_head_exposes_trend_input_contract():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=2026,
        split_identity="fixture-v3c-trend", target_schema="fixture-target-v1",
        variant="v3c", rul_head_input="global_trend",
    )
    model = UnifiedModel(config)
    history = _history(3)
    output = model.predict(history)
    encoded = model.encoder(history)

    assert encoded.trend_features is not None
    assert encoded.trend_features.shape == (2, 3, config.hidden_dim)
    assert output.metadata["config"]["rul_head_input"] == "global_trend"
    assert model.head.base.in_features == config.hidden_dim * 2


def test_v3c_sensor_pooling_modes_are_explicit():
    for mode in ("attention", "mean", "mean_attention"):
        config = ModelConfig(
            dataset="cmapss_fd004", task="rul", input_dim=3, seed=2026,
            split_identity="fixture-v3c-pooling", target_schema="fixture-target-v1",
            variant="v3c", sensor_pooling=mode,
        )
        model = UnifiedModel(config)
        output = model.predict(_history(3))
        assert output.metadata["config"]["sensor_pooling"] == mode


@pytest.mark.parametrize(
    "dataset,task,channels",
    [("metropt3", "state", 3), ("cmapss_fd004", "rul", 3), ("tep", "fault_detection_diagnosis", 3)],
)
def test_cudtr_is_optional_and_records_route_outputs(dataset, task, channels):
    config = ModelConfig(
        dataset=dataset, task=task, input_dim=channels, seed=2026,
        split_identity="fixture-cudtr", target_schema="fixture-target-v1",
        pred_len=4, num_fault_classes=3, variant="v3c",
        cudtr_enabled=True, cudtr_top_k=2, cudtr_query_dim=4,
    )
    model = UnifiedModel(config)
    output = model.predict(_history(channels))

    assert output.metadata["cudtr"]["enabled"] is True
    assert output.metadata["cudtr"]["route_weights"].shape[0] == 2
    assert output.metadata["cudtr"]["fallback_gate"].shape[0] == 2
    assert output.metadata["cudtr"]["route_disagreement"].shape[0] == 2
    if task == "fault_detection_diagnosis":
        assert torch.allclose(output.diagnosis_probability.sum(dim=-1), torch.ones(2), atol=1e-6)
    elif task == "rul":
        assert output.metadata["probability_contract"]["kind"] == "rul_scalar"
    else:
        assert output.metadata["probability_contract"]["kind"] == "state_point"


def test_cudtr_disabled_preserves_default_prediction_contract():
    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=2026,
        split_identity="fixture-no-cudtr", target_schema="fixture-target-v1",
    )
    output = UnifiedModel(config).predict(_history(3))

    assert output.metadata["cudtr"]["enabled"] is False
    assert "route_weights" not in output.metadata["cudtr"]


def test_cudtr_preserves_state_quantile_order_and_probability_contract():
    config = ModelConfig(
        dataset="metropt3", task="state", input_dim=3, seed=2026,
        split_identity="fixture-cudtr-quantile", target_schema="state-v1",
        pred_len=4, head_mode="quantile", variant="v3c", cudtr_enabled=True,
    )
    output = UnifiedModel(config).predict(_history(3))

    assert output.quantiles.shape == (2, 4, 3, 3)
    assert torch.all(output.quantiles[..., 1:] >= output.quantiles[..., :-1])
    assert output.metadata["probability_contract"]["kind"] == "state_quantile"
@pytest.mark.parametrize(
    "dataset,task,channels",
    [("metropt3", "state", 3), ("cmapss_fd004", "rul", 3), ("tep", "fault_detection_diagnosis", 3)],
)
def test_v4_routes_all_task_heads_and_exposes_shared_temporal_features(dataset, task, channels):
    config = ModelConfig(
        dataset=dataset, task=task, input_dim=channels, seed=2026,
        split_identity="fixture-v4", target_schema="fixture-target-v1",
        pred_len=4, num_fault_classes=3, patch_lens=(2, 4), variant="v4",
    )
    model = UnifiedModel(config)
    history = _history(channels)
    output = model.predict(history)
    encoded = model.encoder(history)

    assert output.encoder_version == "unified_temporal_channelwise_v4"
    assert output.metadata["config"]["variant"] == "v4"
    assert encoded.recent_features.shape == (2, channels, config.hidden_dim)
    assert encoded.trend_features.shape == (2, channels, config.hidden_dim)
