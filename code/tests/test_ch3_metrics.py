import torch
import pytest

from kaf_profiti.experiments.metrics import (
    calibrate_state_quantiles, calibrate_state_quantiles_cqr, classification_metrics, rul_metrics,
    state_metrics, state_quantile_diagnostics, state_quantile_metrics,
)


def test_state_metrics_reports_overall_and_per_sensor_errors():
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.tensor([[[0.0, 2.0], [1.0, 8.0]]])
    result = state_metrics(prediction, target, ["a", "b"])

    assert result["overall"]["mae"] == 1.75
    assert result["overall"]["rmse"] == torch.tensor(21.0 / 4.0).sqrt().item()
    assert result["per_sensor"]["a"]["mae"] == 1.5
    assert result["per_sensor"]["a"]["rmse"] == torch.tensor(2.5).sqrt().item()
    assert result["per_sensor"]["b"]["mae"] == 2.0
    assert result["per_sensor"]["b"]["rmse"] == torch.tensor(8.0).sqrt().item()


def test_state_quantile_metrics_reports_coverage_and_width():
    target = torch.tensor([[[0.0, 1.0], [1.0, 2.0]]])
    quantiles = torch.tensor([[[[-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
                               [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]]])
    result = state_quantile_metrics(quantiles, target, ["a", "b"])
    assert result["coverage"] == 1.0
    assert result["interval_width"] == 2.0
    assert result["wis"] >= 0.0
    assert result["point_rmse_from_median"] == 0.0
    assert set(result["per_sensor"]) == {"a", "b"}
    assert all("wis" in values for values in result["per_sensor"].values())


def test_state_quantile_diagnostics_preserves_ordered_blocks_and_sensor_metrics():
    target = torch.zeros(8, 2)
    quantiles = torch.stack((target - 1, target, target + 1), dim=-1)
    result = state_quantile_diagnostics(quantiles, target, ["a", "b"], num_blocks=4)
    assert result["examples"] == 8 and result["num_blocks"] == 4
    assert [(item["start"], item["end"]) for item in result["per_block"]] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert all(item["examples"] == 2 and "per_sensor" in item for item in result["per_block"])


def test_quantile_calibration_uses_validation_targets_only():
    validation_target = torch.tensor([[[0.0], [2.0], [4.0], [6.0]]])
    validation_quantiles = torch.tensor([[[[-1.0, 0.0, 1.0]],
                                          [[1.0, 2.0, 3.0]],
                                          [[3.0, 4.0, 5.0]],
                                          [[5.0, 6.0, 7.0]]]])
    calibrated, radius = calibrate_state_quantiles(
        validation_quantiles, validation_target, validation_quantiles, coverage=0.8
    )
    assert radius.shape == (1,)
    assert torch.all(calibrated[..., 0] <= calibrated[..., 1])
    assert torch.all(calibrated[..., 1] <= calibrated[..., 2])


def test_cqr_calibration_uses_interval_nonconformity_and_is_channelwise():
    target = torch.tensor([[[0.0, 10.0], [4.0, 10.0], [8.0, 10.0]]])
    quantiles = torch.tensor([[[[-1.0, 0.0, 1.0], [9.0, 10.0, 11.0]],
                               [[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]],
                               [[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]]]])
    calibrated, radius = calibrate_state_quantiles_cqr(quantiles, target, quantiles, coverage=0.8)
    assert radius.shape == (2,)
    assert radius[0].item() > 0.0
    assert radius[1].item() == 0.0
    assert torch.all(calibrated[..., 0] <= calibrated[..., 1])
    assert torch.all(calibrated[..., 1] <= calibrated[..., 2])


def test_rul_metrics_reports_raw_cycle_errors_by_engine():
    result = rul_metrics(torch.tensor([12.0, 25.0]), torch.tensor([10.0, 20.0]), [7, 9])

    assert result["units"] == "cycles"
    assert result["per_engine"]["7"] == {"target": 10.0, "prediction": 12.0, "absolute_error": 2.0}
    assert result["per_engine"]["9"]["absolute_error"] == 5.0


def test_classification_metrics_report_per_class_confusion_and_run_summary():
    result = classification_metrics(
        labels=torch.tensor([0, 1, 1, 2, 0]),
        predictions=torch.tensor([0, 1, 2, 2, 1]),
        num_classes=3,
        run_ids=["train:0:1", "test:1:2", "test:1:2", "test:2:3", "train:0:1"],
    )

    assert result["accuracy"] == 0.6
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 2 / 3) / 3)
    assert result["per_class"]["2"]["support"] == 1
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert result["per_run"]["test:1:2"]["support"] == 2
    assert result["run_summary"]["run_count"] == 3


def test_run_macro_f1_ignores_classes_absent_from_truth_and_predictions():
    result = classification_metrics(
        labels=torch.tensor([0, 1, 1]),
        predictions=torch.tensor([0, 1, 1]),
        num_classes=5,
        run_ids=["fault1:run1"] * 3,
    )

    assert result["macro_f1"] == 1.0
    assert result["per_class"]["4"]["support"] == 0
    assert result["per_run"]["fault1:run1"]["macro_f1"] == 1.0
