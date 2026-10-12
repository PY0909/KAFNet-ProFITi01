import copy

import pytest
import torch

from kaf_profiti.models.unified.encoder import UnifiedTemporalEncoder
from kaf_profiti.models.unified.model import HistoryBatch


def _history(x, times=None, context=None):
    batch, length, _ = x.shape
    if times is None:
        times = torch.arange(length, dtype=x.dtype).expand(batch, -1)
    return HistoryBatch(X_obs=x, T_obs=times, context=context)


def test_channelwise_encoder_responds_to_channel_changes():
    torch.manual_seed(10)
    model = UnifiedTemporalEncoder(input_dim=3, hidden_dim=16, patch_lens=(2, 4))
    values = torch.randn(2, 7, 3)
    changed = values.clone()
    changed[:, 2, 1] += 1.0

    first = model(_history(values))
    second = model(_history(changed))

    assert not torch.allclose(first.z_history[:, 1], second.z_history[:, 1])


def test_nonfinite_history_is_rejected():
    model = UnifiedTemporalEncoder(input_dim=3, hidden_dim=12, patch_lens=(2, 5))
    values = torch.full((2, 1, 3), float("nan"))
    try:
        model(_history(values))
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite history must fail")


def test_context_and_irregular_timestamps_are_encoded():
    torch.manual_seed(11)
    model = UnifiedTemporalEncoder(
        input_dim=2, context_dim=1, hidden_dim=12, patch_lens=(2, 4)
    )
    values = torch.ones(1, 5, 2)
    regular = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    irregular = torch.tensor([[0.0, 0.5, 2.0, 4.5, 8.0]])
    context = torch.ones(1, 1)

    regular_output = model(_history(values, regular, context))
    irregular_output = model(_history(values, irregular, context))
    changed_context = model(_history(values, irregular, -context))

    assert not torch.allclose(regular_output.z_global, irregular_output.z_global)
    assert not torch.allclose(irregular_output.z_global, changed_context.z_global)


def test_encoder_state_dict_round_trip_preserves_output():
    torch.manual_seed(12)
    model = UnifiedTemporalEncoder(input_dim=2, hidden_dim=12, patch_lens=(2, 4))
    history = _history(torch.randn(2, 6, 2))
    expected = model(history)
    restored = UnifiedTemporalEncoder(input_dim=2, hidden_dim=12, patch_lens=(2, 4))
    restored.load_state_dict(model.state_dict())
    actual = restored(history)

    torch.testing.assert_close(expected.z_global, actual.z_global)


def test_v3a_ignores_values_where_observation_mask_is_zero():
    torch.manual_seed(13)
    model = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v3a"
    )
    values = torch.randn(2, 6, 2)
    mask = torch.ones_like(values)
    mask[:, 1:4, 0] = 0
    changed = values.clone()
    changed[mask == 0] = 1e6
    times = torch.tensor([[0.0, 0.5, 1.0, 3.0, 4.0, 8.0]]).expand(2, -1)

    first = model(HistoryBatch(X_obs=values, T_obs=times, M_obs=mask))
    second = model(HistoryBatch(X_obs=changed, T_obs=times, M_obs=mask))

    torch.testing.assert_close(first.z_history, second.z_history)
    torch.testing.assert_close(first.z_global, second.z_global)


def test_v3a_exposes_normalized_attention_weights_for_each_scale():
    torch.manual_seed(14)
    model = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v3a"
    )
    output = model(_history(torch.randn(2, 6, 2)))

    assert output.pooling_weights.shape == (2, 2, 3, 6)
    torch.testing.assert_close(
        output.pooling_weights.sum(dim=-1), torch.ones(2, 2, 3)
    )


def test_v3b_returns_scale_ordered_patch_tokens():
    torch.manual_seed(15)
    model = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v3b"
    )
    output = model(_history(torch.randn(2, 6, 2)))

    assert output.patch_tokens.shape == (2, 2, 3, 12)
    assert model.scale_names == ("global", "2", "4")
    assert torch.isfinite(output.patch_tokens).all()
    assert not torch.allclose(output.patch_tokens[:, :, 0], output.patch_tokens[:, :, 1])


def test_v3b_is_invariant_to_masked_values():
    torch.manual_seed(16)
    model = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v3b"
    )
    values = torch.randn(2, 6, 2)
    mask = torch.ones_like(values)
    mask[:, 1:4, 0] = 0
    changed = values.clone()
    changed[mask == 0] = -1e6
    times = torch.tensor([[0.0, 0.5, 1.0, 3.0, 4.0, 8.0]]).expand(2, -1)

    first = model(HistoryBatch(X_obs=values, T_obs=times, M_obs=mask))
    second = model(HistoryBatch(X_obs=changed, T_obs=times, M_obs=mask))

    torch.testing.assert_close(first.patch_tokens, second.patch_tokens)
    torch.testing.assert_close(first.z_history, second.z_history)
    torch.testing.assert_close(first.z_global, second.z_global)


def test_v3b_state_dict_round_trip_preserves_patch_tokens():
    torch.manual_seed(17)
    model = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v3b"
    )
    history = _history(torch.randn(2, 6, 2))
    expected = model(history)
    restored = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v3b"
    )
    restored.load_state_dict(model.state_dict())
    actual = restored(history)

    torch.testing.assert_close(expected.patch_tokens, actual.patch_tokens)


@pytest.mark.parametrize("variant", ("v5a", "v5b", "v5c"))
def test_v5_variants_return_ordered_patch_tokens_and_mask_invariance(variant):
    model = UnifiedTemporalEncoder(input_dim=2, hidden_dim=8, patch_lens=(2, 4), variant=variant)
    x = torch.randn(2, 8, 2)
    mask = torch.ones_like(x)
    mask[:, 1, 0] = 0
    history = HistoryBatch(x, torch.arange(8).repeat(2, 1).float(), M_obs=mask)
    changed = HistoryBatch(x.masked_fill(mask == 0, 9999.0), history.T_obs, M_obs=mask)
    expected = model(history)
    actual = model(changed)
    assert expected.patch_tokens.shape == (2, 2, 4, 8)
    torch.testing.assert_close(expected.z_global, actual.z_global)


def test_v5a_ordered_patch_tokens_react_to_temporal_reordering():
    model = UnifiedTemporalEncoder(input_dim=2, hidden_dim=8, patch_lens=(2, 4), variant="v5a")
    x = torch.randn(1, 8, 2)
    history = _history(x)
    reversed_history = _history(torch.flip(x, dims=(1,)))
    assert not torch.allclose(model(history).patch_tokens, model(reversed_history).patch_tokens)


@pytest.mark.parametrize("variant", ("v5a", "v5b", "v5c"))
def test_v5_variants_support_short_and_non_divisible_histories(variant):
    model = UnifiedTemporalEncoder(input_dim=2, hidden_dim=8, patch_lens=(4, 8), variant=variant)
    for length, patch_count in ((1, 1), (6, 2), (9, 3)):
        output = model(_history(torch.randn(1, length, 2)))
        assert output.patch_tokens.shape == (1, 2, patch_count, 8)
        assert torch.isfinite(output.z_global).all()


def test_v5c_patch_query_is_trainable():
    model = UnifiedTemporalEncoder(input_dim=2, hidden_dim=8, patch_lens=(2, 4), variant="v5c")
    output = model(_history(torch.randn(2, 8, 2)))
    output.z_global.square().mean().backward()
    assert model.patch_query.weight.grad is not None
    assert model.patch_query.bias.grad is not None


def test_v3c_learned_sensor_weights_are_normalized_and_affect_global_summary():
    torch.manual_seed(18)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v3c"
    )
    values = torch.randn(2, 6, 3)
    first = model(_history(values))
    changed = values.clone()
    changed[:, :, 1] += 3.0
    second = model(_history(changed))

    assert first.sensor_weights.shape == (2, 3)
    torch.testing.assert_close(first.sensor_weights.sum(dim=1), torch.ones(2))
    assert not torch.allclose(first.z_global, second.z_global)


def test_v3c_is_invariant_to_masked_values():
    torch.manual_seed(19)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v3c"
    )
    values = torch.randn(2, 6, 3)
    mask = torch.ones_like(values)
    mask[:, 1:4, 0] = 0
    changed = values.clone()
    changed[mask == 0] = 1e6
    times = torch.tensor([[0.0, 0.5, 1.0, 3.0, 4.0, 8.0]]).expand(2, -1)

    first = model(HistoryBatch(X_obs=values, T_obs=times, M_obs=mask))
    second = model(HistoryBatch(X_obs=changed, T_obs=times, M_obs=mask))

    torch.testing.assert_close(first.z_global, second.z_global)
    torch.testing.assert_close(first.sensor_weights, second.sensor_weights)


def test_v3c_state_dict_round_trip_preserves_global_pooling():
    torch.manual_seed(20)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v3c"
    )
    history = _history(torch.randn(2, 6, 3))
    expected = model(history)
    restored = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v3c"
    )
    restored.load_state_dict(model.state_dict())
    actual = restored(history)

    torch.testing.assert_close(expected.sensor_weights, actual.sensor_weights)
    torch.testing.assert_close(expected.z_global, actual.z_global)


def test_v4_exposes_recent_and_trend_features_for_each_sensor():
    torch.manual_seed(21)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v4"
    )
    output = model(_history(torch.randn(2, 8, 3)))

    assert output.recent_features.shape == (2, 3, 12)
    assert output.trend_features.shape == (2, 3, 12)
    assert torch.isfinite(output.recent_features).all()
    assert torch.isfinite(output.trend_features).all()
    assert not torch.allclose(output.trend_features, torch.zeros_like(output.trend_features))


def test_v4_is_invariant_to_masked_values_and_handles_all_missing_channels():
    torch.manual_seed(22)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v4"
    )
    values = torch.randn(2, 8, 3)
    mask = torch.ones_like(values)
    mask[:, 1:6, 0] = 0
    mask[:, :, 2] = 0
    changed = values.clone()
    changed[mask == 0] = 1e6
    times = torch.tensor([[0.0, 0.5, 1.0, 2.0, 4.0, 7.0, 8.0, 12.0]]).expand(2, -1)

    first = model(HistoryBatch(X_obs=values, T_obs=times, M_obs=mask))
    second = model(HistoryBatch(X_obs=changed, T_obs=times, M_obs=mask))

    torch.testing.assert_close(first.z_history, second.z_history)
    torch.testing.assert_close(first.z_global, second.z_global)
    torch.testing.assert_close(first.recent_features, second.recent_features)
    torch.testing.assert_close(first.trend_features, second.trend_features)
    assert torch.isfinite(first.z_history).all()
    assert torch.isfinite(first.trend_features).all()


def test_v4_temporal_block_uses_dilated_depthwise_convolutions():
    model = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v4"
    )
    convolutions = [
        module for module in model.temporal_blocks.modules()
        if isinstance(module, torch.nn.Conv1d)
    ]

    assert convolutions
    assert any(module.dilation[0] > 1 for module in convolutions)
    assert all(module.groups == module.in_channels for module in convolutions if module.groups > 1)


def test_v4_recent_and_trend_features_contribute_to_prediction_representation():
    torch.manual_seed(23)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v4"
    )
    encoded = model(_history(torch.randn(2, 8, 3)))

    recent_grad, trend_grad = torch.autograd.grad(
        encoded.z_history.square().mean(),
        (encoded.recent_features, encoded.trend_features),
        allow_unused=True,
    )

    assert recent_grad is not None
    assert trend_grad is not None
    assert torch.isfinite(recent_grad).all() and recent_grad.abs().sum() > 0
    assert torch.isfinite(trend_grad).all() and trend_grad.abs().sum() > 0


def test_v4_cross_channel_attention_preserves_input_with_residual_path():
    class ZeroAttention(torch.nn.Module):
        def forward(self, query, key, value, need_weights=False):
            return torch.zeros_like(query), None

    torch.manual_seed(24)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v4"
    )
    model.cross_channel = ZeroAttention()
    encoded = model(_history(torch.randn(2, 8, 3)))

    assert encoded.z_history.abs().sum() > 0


def test_v2t_trend_features_recover_slope_and_ignore_masked_positions():
    model = UnifiedTemporalEncoder(input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v2t")
    history_len = 12
    ramp = torch.arange(history_len, dtype=torch.float32) * 2.0 + 1.0
    values = ramp.view(1, history_len, 1).expand(1, history_len, 2).clone()
    values[:, :, 1] = torch.arange(history_len, dtype=torch.float32) * -0.5
    observed = torch.ones_like(values)

    trend = model._channel_trend_features(values * observed, observed)

    assert trend.shape == (1, 2, 2)
    torch.testing.assert_close(trend[0, 0, 0], torch.tensor(2.0))
    torch.testing.assert_close(trend[0, 1, 0], torch.tensor(-0.5))

    masked = observed.clone()
    masked[:, -4:, 0] = 0.0
    safe_values = values * masked
    trend_masked = model._channel_trend_features(safe_values, masked)

    torch.testing.assert_close(trend_masked[0, 0, 0], torch.tensor(2.0))
    assert torch.isfinite(trend_masked).all()


def test_v2t_trend_features_handle_all_missing_channel():
    model = UnifiedTemporalEncoder(input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v2t")
    values = torch.randn(1, 8, 2)
    observed = torch.zeros_like(values)
    observed[:, :, 1] = 1.0

    trend = model._channel_trend_features(values * observed, observed)

    assert torch.isfinite(trend).all()
    torch.testing.assert_close(trend[0, 0], torch.zeros(2))


def test_v2t_is_invariant_to_masked_values():
    torch.manual_seed(25)
    model = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v2t"
    )
    values = torch.randn(2, 6, 2)
    mask = torch.ones_like(values)
    mask[:, 1:4, 0] = 0
    changed = values.clone()
    changed[mask == 0] = 1e6
    times = torch.tensor([[0.0, 0.5, 1.0, 3.0, 4.0, 8.0]]).expand(2, -1)

    first = model(HistoryBatch(X_obs=values, T_obs=times, M_obs=mask))
    second = model(HistoryBatch(X_obs=changed, T_obs=times, M_obs=mask))

    torch.testing.assert_close(first.z_history, second.z_history)
    torch.testing.assert_close(first.z_global, second.z_global)


def test_v2t_forward_shapes_and_state_dict_round_trip():
    torch.manual_seed(26)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v2t"
    )
    history = _history(torch.randn(2, 8, 3))
    output = model(history)

    assert output.z_history.shape == (2, 3, 12)
    assert output.z_global.shape == (2, 12)
    assert output.pooling_weights is None
    assert torch.isfinite(output.z_history).all()

    restored = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v2t"
    )
    restored.load_state_dict(model.state_dict())
    actual = restored(history)
    torch.testing.assert_close(output.z_history, actual.z_history)
    torch.testing.assert_close(output.z_global, actual.z_global)


def test_v2t_encoder_version_is_distinct_from_v2():
    model = UnifiedTemporalEncoder(input_dim=2, hidden_dim=8, patch_lens=(2,), variant="v2t")

    assert model.version == "unified_temporal_channelwise_v2t"


def test_v2tc_combines_trend_features_with_learned_sensor_pooling():
    torch.manual_seed(27)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v2tc"
    )
    output = model(_history(torch.randn(2, 8, 3)))

    assert output.z_history.shape == (2, 3, 12)
    assert output.pooling_weights is None
    assert output.sensor_weights is not None
    assert output.sensor_weights.shape == (2, 3)
    torch.testing.assert_close(output.sensor_weights.sum(dim=1), torch.ones(2))
    assert torch.isfinite(output.z_history).all()

    # v2tc 的逐时间点输入与 v2 相同（3 维），与 v3 系列的 4 维不同
    v2t_model = UnifiedTemporalEncoder(input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v2t")
    assert model.temporal[0].in_features == v2t_model.temporal[0].in_features


def test_v2tc_is_invariant_to_masked_values():
    torch.manual_seed(28)
    model = UnifiedTemporalEncoder(
        input_dim=2, hidden_dim=12, patch_lens=(2, 4), variant="v2tc"
    )
    values = torch.randn(2, 6, 2)
    mask = torch.ones_like(values)
    mask[:, 1:4, 0] = 0
    changed = values.clone()
    changed[mask == 0] = 1e6
    times = torch.tensor([[0.0, 0.5, 1.0, 3.0, 4.0, 8.0]]).expand(2, -1)

    first = model(HistoryBatch(X_obs=values, T_obs=times, M_obs=mask))
    second = model(HistoryBatch(X_obs=changed, T_obs=times, M_obs=mask))

    torch.testing.assert_close(first.z_history, second.z_history)
    torch.testing.assert_close(first.sensor_weights, second.sensor_weights)
    torch.testing.assert_close(first.z_global, second.z_global)


def test_v2tc_state_dict_round_trip_preserves_outputs():
    torch.manual_seed(29)
    model = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v2tc"
    )
    history = _history(torch.randn(2, 8, 3))
    expected = model(history)
    restored = UnifiedTemporalEncoder(
        input_dim=3, hidden_dim=12, patch_lens=(2, 4), variant="v2tc"
    )
    restored.load_state_dict(model.state_dict())
    actual = restored(history)

    torch.testing.assert_close(expected.z_history, actual.z_history)
    torch.testing.assert_close(expected.sensor_weights, actual.sensor_weights)
    torch.testing.assert_close(expected.z_global, actual.z_global)
