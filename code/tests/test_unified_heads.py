import torch

from kaf_profiti.models.unified.encoder import EncoderOutput
from kaf_profiti.models.unified.heads.rul import RULHead
from kaf_profiti.models.unified.heads.state import StateHead
from kaf_profiti.models.unified.heads.tep import TEPHead, detection_delay, evaluate_tep_events
from kaf_profiti.models.unified.heads.base import TaskTargets


def _encoded(batch=3, sensors=4, hidden=8):
    torch.manual_seed(20)
    z = torch.randn(batch, sensors, hidden)
    return EncoderOutput(z_history=z, z_global=z.mean(dim=1))


def test_state_head_point_and_monotonic_quantiles():
    encoded = _encoded()
    head = StateHead(hidden_dim=8, num_sensors=4, pred_len=5, quantiles=(0.1, 0.5, 0.9))
    point = head(encoded, mode="point")
    distribution = head(encoded, mode="quantile")
    assert point.point.shape == (3, 5, 4)
    assert distribution.quantiles.shape == (3, 5, 4, 3)
    assert torch.all(distribution.quantiles[..., 1:] >= distribution.quantiles[..., :-1])


def test_state_quantile_loss_uses_pinball_when_point_center_is_also_present():
    from kaf_profiti.models.unified.output import PredictionOutput

    head = StateHead(hidden_dim=2, num_sensors=1, pred_len=1, quantiles=(0.1, 0.5, 0.9))
    prediction = PredictionOutput(
        dataset="metropt3", task="state", encoder_version="test", seed=1,
        split_identity="state-split", target_schema="state-v1",
        point=torch.tensor([[[100.0]]]),
        quantiles=torch.tensor([[[[0.0, 0.0, 10.0]]]]),
    )
    targets = TaskTargets(values=torch.tensor([[[1.0]]]), schema="state-v1")

    # Pinball = (0.1*1 + 0.5*1 + 0.1*9) / 3, independent of the center point.
    assert torch.isclose(head.loss(prediction, targets), torch.tensor(0.5))


def test_rul_head_records_raw_and_capped_target_conventions():
    head = RULHead(hidden_dim=8, quantiles=(0.1, 0.5, 0.9), training_cap=125.0)
    output = head(_encoded(), mode="quantile")
    assert output.quantiles.shape == (3, 3)
    targets = TaskTargets(
        values=torch.tensor([40.0, 200.0, 80.0]),
        schema="cmapss_fd004_rul_raw_v1",
        raw_rul=torch.tensor([40.0, 200.0, 80.0]),
    )
    capped = head.prepare_target(targets)
    assert capped.tolist() == [40.0, 125.0, 80.0]
    assert output.metadata["target_convention"] == "raw_output_training_cap_125"


def test_rul_point_objective_matches_availability_maped_mae_protocol():
    from kaf_profiti.models.unified.output import PredictionOutput

    head = RULHead(hidden_dim=8, training_cap=125.0)
    prediction = PredictionOutput(
        dataset="cmapss_fd004", task="rul", encoder_version="test", seed=1,
        split_identity="fd004-split", target_schema="cmapss_fd004_rul_raw_v1",
        point=torch.tensor([8.0, 15.0]),
    )
    targets = TaskTargets(values=torch.tensor([[10.0], [10.0]]), schema="cmapss_fd004_rul_raw_v1")

    assert head.loss(prediction, targets).item() == 3.5


def test_rul_asymmetric_loss_penalizes_over_prediction():
    from kaf_profiti.models.unified.output import PredictionOutput

    targets = TaskTargets(values=torch.tensor([[10.0], [10.0]]), schema="cmapss_fd004_rul_raw_v1")
    prediction = PredictionOutput(
        dataset="cmapss_fd004", task="rul", encoder_version="test", seed=1,
        split_identity="fd004-split", target_schema="cmapss_fd004_rul_raw_v1",
        point=torch.tensor([12.0, 8.0]),
    )
    symmetric = RULHead(hidden_dim=8, training_cap=125.0, over_penalty=1.0)
    asymmetric = RULHead(hidden_dim=8, training_cap=125.0, over_penalty=2.0)

    assert symmetric.loss(prediction, targets).item() == 2.0
    # 高估 +2 记为 2*2，低估 -2 记为 2 → 均值 3
    assert asymmetric.loss(prediction, targets).item() == 3.0


def test_rul_quantile_prediction_without_point_uses_pinball_loss():
    from kaf_profiti.models.unified.output import PredictionOutput

    head = RULHead(hidden_dim=8, quantiles=(0.1, 0.5, 0.9), training_cap=125.0)
    prediction = PredictionOutput(
        dataset="cmapss_fd004", task="rul", encoder_version="test", seed=1,
        split_identity="fd004-split", target_schema="cmapss_fd004_rul_raw_v1",
        point=None,
        quantiles=torch.tensor([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]),
    )
    targets = TaskTargets(values=torch.tensor([[10.0], [10.0]]), schema="cmapss_fd004_rul_raw_v1")

    assert head.loss(prediction, targets).item() == 0.0


def test_rul_over_penalty_must_be_positive():
    try:
        RULHead(hidden_dim=8, over_penalty=0.0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive over_penalty must fail")


def test_tep_head_returns_detection_and_diagnosis_probabilities():
    head = TEPHead(hidden_dim=8, num_fault_classes=5)
    output = head(_encoded(), mode="probability")
    assert output.detection_probability.shape == (3,)
    assert output.diagnosis_probability.shape == (3, 5)
    assert torch.all((output.detection_probability >= 0) & (output.detection_probability <= 1))
    assert torch.allclose(output.diagnosis_probability.sum(dim=-1), torch.ones(3))
    targets = TaskTargets(
        values=None,
        schema="tep_fault_v1",
        class_label=torch.tensor([0, 2, 4]),
    )
    losses = head.loss(output, targets)
    assert set(losses) == {"detection", "diagnosis", "total"}
    assert torch.isfinite(losses["total"])


def test_tep_head_loss_is_autocast_safe():
    head = TEPHead(hidden_dim=8, num_fault_classes=5)
    output = head(_encoded(), mode="probability")
    targets = TaskTargets(values=None, schema="tep_fault_v1", class_label=torch.tensor([0, 2, 4]))
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        losses = head.loss(output, targets)
    assert torch.isfinite(losses["total"])


def test_detection_delay_uses_first_alarm_per_run():
    probability = torch.tensor([0.1, 0.8, 0.9, 0.2, 0.7, 0.8])
    labels = torch.tensor([0, 0, 1, 0, 0, 1])
    run_ids = torch.tensor([1, 1, 1, 2, 2, 2])
    samples = torch.tensor([0, 1, 2, 0, 1, 2])
    assert detection_delay(probability, labels, run_ids, samples, threshold=0.5) == 0.0


def test_tep_event_evaluation_reports_detection_miss_false_alarm_and_delay_distribution():
    probability = torch.tensor([0.1, 0.8, 0.9, 0.7, 0.2, 0.1])
    labels = torch.tensor([0, 0, 1, 0, 0, 0])
    run_ids = torch.tensor([1, 1, 1, 2, 2, 2])
    samples = torch.tensor([0, 1, 2, 0, 1, 2])
    result = evaluate_tep_events(probability, labels, run_ids, samples, threshold=0.5)
    assert result["fault_run_count"] == 1
    assert result["detected_run_count"] == 1
    assert result["missed_run_count"] == 0
    assert result["false_alarm_run_count"] == 1
    assert result["detection_rate"] == 1.0
    assert result["miss_rate"] == 0.0
    assert result["false_alarm_rate"] == 1.0
    assert result["delay_distribution"] == [0.0]
