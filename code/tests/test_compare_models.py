from types import SimpleNamespace

import pytest
import torch

from compare_code.models.factory import MODEL_NAMES, create_baseline


@pytest.mark.parametrize("model_name", ("tcn", "gru"))
def test_tcn_and_gru_return_finite_forecasts_with_expected_shape(model_name):
    model = create_baseline(
        model_name,
        input_dim=3,
        history_len=12,
        pred_len=7,
        seed=11,
        split_identity="split-a",
        target_schema="metropt3_state_v1",
        d_model=16,
        layers=2,
        heads=4,
    )
    x = torch.randn(2, 12, 3)
    forecast = model(SimpleNamespace(X_obs=x))

    assert forecast.shape == (2, 7, 3)
    assert torch.isfinite(forecast).all()
    forecast.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tcn_direct_head_is_finite_at_initialization():
    model = create_baseline(
        "tcn", input_dim=3, history_len=12, pred_len=7, seed=11,
        split_identity="split-a", target_schema="metropt3_state_v1",
        d_model=16, layers=2, heads=4,
    )
    x = torch.randn(2, 12, 3)
    forecast = model(SimpleNamespace(X_obs=x))
    assert torch.isfinite(forecast).all()


def test_tcn_prediction_head_is_direct_multi_step_projection():
    model = create_baseline(
        "tcn", input_dim=2, history_len=8, pred_len=3, seed=11,
        split_identity="split-a", target_schema="metropt3_state_v1",
        d_model=8, layers=1, heads=2,
    )
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.fill_(1.0)
    x = torch.full((1, 8, 2), 5.0)
    forecast = model(SimpleNamespace(X_obs=x))
    assert torch.allclose(forecast, torch.ones_like(forecast))


def test_baseline_prediction_has_standard_identity_and_mae_loss():
    model = create_baseline(
        "gru", input_dim=2, history_len=8, pred_len=3, seed=5,
        split_identity="metro-split", target_schema="metropt3_state_v1", d_model=8, heads=2,
    )
    history = SimpleNamespace(X_obs=torch.randn(4, 8, 2))
    prediction = model.predict(history)
    targets = SimpleNamespace(values=torch.zeros(4, 3, 2))

    assert prediction.point.shape == targets.values.shape
    assert prediction.dataset == "metropt3"
    assert prediction.task == "state"
    assert prediction.seed == 5
    assert prediction.split_identity == "metro-split"
    assert prediction.target_schema == "metropt3_state_v1"
    assert model.loss(prediction, targets).ndim == 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_comparison_models_emit_scalar_raw_rul_predictions(model_name):
    model = create_baseline(
        model_name, input_dim=3, history_len=12, pred_len=10, seed=11,
        split_identity="fd004-split", target_schema="cmapss_fd004_rul_raw_v1",
        d_model=16, layers=1, heads=4, patch_len=4,
    )
    history = SimpleNamespace(X_obs=torch.randn(2, 12, 3))
    prediction = model.predict(history)
    targets = SimpleNamespace(values=torch.tensor([[40.0], [80.0]]))

    assert prediction.point.shape in {(2,), (2, 1)}
    assert prediction.dataset == "cmapss_fd004"
    assert prediction.task == "rul"
    assert prediction.metadata["target_convention"] == "raw_output_training_cap_125"
    assert torch.isfinite(prediction.point).all()
    assert model.loss(prediction, targets).ndim == 0
    model.loss(prediction, targets).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_baseline_config_rejects_attention_dimensions_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        create_baseline(
            "gru", input_dim=2, history_len=8, pred_len=3, seed=5,
            split_identity="split-a", target_schema="state", d_model=10, heads=4,
        )


def test_registered_model_names_are_explicit_and_stable():
    assert MODEL_NAMES == ("tcn", "gru", "patchtst", "timesnet", "itransformer")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_comparison_models_emit_tep_diagnosis_probabilities(model_name):
    model = create_baseline(
        model_name, input_dim=3, history_len=12, pred_len=1, seed=19,
        split_identity="tep-split", target_schema="tep_fault_v1", num_classes=4,
        d_model=8, layers=1, heads=2, patch_len=4,
    )
    history = SimpleNamespace(X_obs=torch.randn(2, 12, 3))
    prediction = model.predict(history)
    labels = SimpleNamespace(class_label=torch.tensor([1, 3]))

    assert prediction.task == "fault_detection_diagnosis"
    assert prediction.dataset == "tep"
    assert prediction.diagnosis_probability.shape == (2, 4)
    assert torch.allclose(prediction.diagnosis_probability.sum(-1), torch.ones(2))
    assert model.loss(prediction, labels).ndim == 0


@pytest.mark.parametrize("model_name", ("patchtst", "timesnet", "itransformer"))
def test_transformer_and_timesnet_family_support_short_histories(model_name):
    model = create_baseline(
        model_name,
        input_dim=3,
        history_len=10,
        pred_len=4,
        seed=13,
        split_identity="split-short",
        target_schema="metropt3_state_v1",
        d_model=8,
        layers=1,
        heads=2,
        patch_len=16,
        top_k=8,
    )
    x = torch.randn(2, 10, 3)
    forecast = model(SimpleNamespace(X_obs=x))

    assert forecast.shape == (2, 4, 3)
    assert torch.isfinite(forecast).all()
    forecast.abs().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_tep_baselines_emit_detection_probability_from_normal_class(model_name):
    model = create_baseline(
        model_name, input_dim=2, history_len=12, pred_len=2, seed=7,
        split_identity="split-tep", target_schema="tep_fault_v1",
        num_classes=4, d_model=8, heads=2,
    )
    prediction = model.predict(torch.randn(3, 12, 2))
    assert prediction.detection_probability is not None
    assert torch.allclose(prediction.detection_probability, 1.0 - prediction.diagnosis_probability[:, 0])
    assert bool(((prediction.detection_probability >= 0.0) & (prediction.detection_probability <= 1.0)).all())


def _tep_gru():
    return create_baseline(
        "gru", input_dim=2, history_len=12, pred_len=2, seed=7,
        split_identity="split-tep", target_schema="tep_fault_v1",
        num_classes=3, d_model=8, heads=2,
    )


def test_tep_baseline_loss_applies_configured_class_weights():
    from kaf_profiti.models.unified.heads.base import TaskTargets

    model = _tep_gru()
    weights = (0.5, 1.0, 2.0)
    model.set_tep_class_weights(weights)
    assert model.config.tep_class_weights == weights
    prediction = model.predict(torch.randn(2, 12, 2))
    targets = TaskTargets(values=None, schema="tep_fault_v1", class_label=torch.tensor([0, 2]))
    manual = torch.nn.functional.nll_loss(
        prediction.diagnosis_probability.clamp_min(1e-7).log(),
        torch.tensor([0, 2]),
        weight=torch.tensor(weights),
    )
    assert torch.allclose(model.loss(prediction, targets), manual)


def test_tep_baseline_loss_without_weights_stays_unweighted():
    from kaf_profiti.models.unified.heads.base import TaskTargets

    model = _tep_gru()
    assert model.config.tep_class_weights == ()
    prediction = model.predict(torch.randn(2, 12, 2))
    targets = TaskTargets(values=None, schema="tep_fault_v1", class_label=torch.tensor([1, 2]))
    manual = torch.nn.functional.nll_loss(
        prediction.diagnosis_probability.clamp_min(1e-7).log(), torch.tensor([1, 2]),
    )
    assert torch.allclose(model.loss(prediction, targets), manual)


def test_baseline_config_rejects_mismatched_tep_class_weights():
    with pytest.raises(ValueError):
        create_baseline(
            "gru", input_dim=2, history_len=12, pred_len=2, seed=7,
            split_identity="split-tep", target_schema="tep_fault_v1",
            num_classes=3, d_model=8, heads=2, tep_class_weights=(1.0, 1.0),
        )
