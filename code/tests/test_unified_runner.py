import torch
import pytest

from kaf_profiti.experiments.task_runner import evaluate, select_best_validation, train_step
from kaf_profiti.models.unified import ModelConfig, UnifiedModel
from kaf_profiti.models.unified.model import HistoryBatch
from kaf_profiti.models.unified.heads.base import TaskTargets


def test_train_step_and_validation_selection_do_not_need_test_targets():
    model = UnifiedModel(
        ModelConfig(
            dataset="cmapss_fd004", task="rul", input_dim=2, seed=1,
            split_identity="s", target_schema="r", pred_len=1,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = HistoryBatch(
        X_obs=torch.randn(2, 4, 2),
        T_obs=torch.arange(4, dtype=torch.float32).expand(2, -1),
    )
    targets = TaskTargets(values=torch.tensor([10.0, 20.0]), schema="r")
    loss = train_step(model, optimizer, history, targets)
    assert loss >= 0
    metric = evaluate(model, [(history, targets)], split="validation")
    assert metric["split"] == "validation"
    assert select_best_validation([{"epoch": 1, "metric": 2.0}, {"epoch": 2, "metric": 1.0}])["epoch"] == 2


def test_evaluate_aggregates_validation_loss_by_example_not_by_batch():
    class ExampleLossModel:
        def eval(self):
            return self

        def predict(self, history):
            return history

        def loss(self, prediction, targets):
            return prediction.X_obs.float().mean()

    batches = [
        (HistoryBatch(torch.tensor([1.0, 1.0]).reshape(2, 1, 1), torch.zeros(2, 1)), None),
        (HistoryBatch(torch.tensor([9.0]).reshape(1, 1, 1), torch.zeros(1, 1)), None),
    ]
    result = evaluate(ExampleLossModel(), batches, split="validation")
    assert result["loss"] == pytest.approx(11.0 / 3.0)


def test_runner_rejects_nonfinite_training_loss():
    class BrokenModel:
        def train(self): pass
        def predict(self, history): return None
        def loss(self, prediction, targets): return torch.tensor(float("nan"), requires_grad=True)
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=1e-3)
    history = HistoryBatch(torch.randn(1, 4, 2), torch.arange(4.).unsqueeze(0))
    try:
        train_step(BrokenModel(), optimizer, history, TaskTargets(values=torch.ones(1), schema="r"))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite loss must fail")


def test_rul_metrics_use_raw_units_and_nasa_asymmetric_score():
    from kaf_profiti.experiments.metrics import rul_metrics

    metrics = rul_metrics(torch.tensor([8.0, 15.0]), torch.tensor([10.0, 10.0]))

    assert metrics["mae"] == 3.5
    assert abs(metrics["rmse"] - ((4.0 + 25.0) / 2.0) ** 0.5) < 1e-6
    expected = (torch.exp(torch.tensor(2.0 / 13.0)) - 1) + (torch.exp(torch.tensor(5.0 / 10.0)) - 1)
    assert abs(metrics["nasa_score"] - float(expected)) < 1e-6
    assert metrics["units"] == "cycles"
