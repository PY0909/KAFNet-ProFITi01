import torch
import pytest

from kaf_profiti.models.unified.model import HistoryBatch
from kaf_profiti.models.unified.utility_teacher import (
    build_source_drop_history,
    estimate_counterfactual_utilities,
    estimate_route_level_utilities,
)


def _history():
    return HistoryBatch(
        X_obs=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        T_obs=torch.arange(2, dtype=torch.float32).expand(2, -1),
        M_obs=torch.ones(2, 2, 3),
    )


def test_source_drop_history_does_not_mutate_original_batch():
    history = _history()
    dropped = build_source_drop_history(history, source_index=1)

    assert torch.equal(history.X_obs, _history().X_obs)
    assert torch.equal(history.M_obs, torch.ones(2, 2, 3))
    assert torch.equal(dropped.X_obs[:, :, 1], torch.zeros(2, 2))
    assert torch.equal(dropped.M_obs[:, :, 1], torch.zeros(2, 2))
    assert torch.equal(dropped.X_obs[:, :, 0], history.X_obs[:, :, 0])


def test_counterfactual_utilities_are_per_sample_loss_increases():
    history = _history()

    def predictor(batch):
        return batch.X_obs.sum(dim=(1, 2))

    def loss_fn(prediction, _targets):
        return prediction.square()

    targets = torch.zeros(2)
    utilities = estimate_counterfactual_utilities(
        predictor, history, targets, loss_fn, source_indices=[0, 1, 2]
    )

    assert utilities.shape == (2, 3)
    assert torch.isfinite(utilities).all()
    assert (utilities[:, 0] < 0).all()
    assert not utilities.requires_grad


def test_teacher_rejects_invalid_source_index():
    with pytest.raises(ValueError, match="source_index"):
        build_source_drop_history(_history(), source_index=3)


def test_teacher_never_backpropagates_into_predictor_graph():
    history = _history()
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    def predictor(batch):
        return batch.X_obs.sum(dim=(1, 2)) * parameter

    def loss_fn(prediction, _targets):
        return prediction.square()

    utilities = estimate_counterfactual_utilities(
        predictor, history, torch.zeros(2), loss_fn, source_indices=[0]
    )

    assert not utilities.requires_grad
    assert parameter.grad is None


def test_counterfactual_utilities_preserve_target_axis_for_state_forecasting():
    history = _history()

    def predictor(batch):
        return batch.X_obs.sum(dim=1)

    def loss_fn(prediction, _targets):
        return prediction.square()

    utilities = estimate_counterfactual_utilities(
        predictor, history, torch.zeros(2, 3), loss_fn, source_indices=[0, 1, 2]
    )

    assert utilities.shape == (2, 3, 3)


def test_route_level_teacher_deletes_only_selected_source_for_each_target():
    tokens = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    targets = torch.zeros(2, 3)

    def predictor(source_tokens, target_index):
        # The target-specific prediction depends only on source target_index.
        return source_tokens[:, target_index, :].sum(dim=-1)

    def loss_fn(prediction, target, target_index):
        return (prediction - target[:, target_index]).square()

    utilities = estimate_route_level_utilities(
        predictor, tokens, targets, loss_fn, source_indices=[0, 1, 2], target_count=3,
    )

    assert utilities.shape == (2, 3, 3)
    assert torch.equal(torch.diagonal(utilities, dim1=1, dim2=2), torch.zeros(2, 3))
    # Dropping source j must not change predictions for target i != j.
    off_diagonal = utilities.clone()
    diagonal = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    assert torch.allclose(off_diagonal.masked_fill(diagonal, 0), torch.zeros_like(off_diagonal))
    assert not utilities.requires_grad


def test_route_level_teacher_rejects_mismatched_source_count():
    tokens = torch.zeros(2, 2, 1)
    with pytest.raises(ValueError, match="target_count"):
        estimate_route_level_utilities(
            lambda values, target_index: values[:, target_index, 0],
            tokens, torch.zeros(2, 3),
            lambda prediction, target, target_index: prediction.square(),
            source_indices=[0, 1], target_count=3,
        )


@pytest.mark.parametrize("task", ["rul", "state", "tep"])
def test_model_teacher_builds_train_targets_matching_each_route_shape(task):
    from kaf_profiti.experiments.task_runner import estimate_task_counterfactual_utilities
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel
    from kaf_profiti.models.unified.heads.base import TaskTargets

    mapping = {
        "rul": ("cmapss_fd004", "rul", TaskTargets(values=torch.tensor([10.0, 20.0]), schema="r")),
        "state": ("metropt3", "state", TaskTargets(values=torch.randn(2, 4, 3), schema="state")),
        "tep": ("tep", "fault_detection_diagnosis", TaskTargets(values=None, schema="tep", class_label=torch.tensor([0, 2]))),
    }
    dataset, model_task, targets = mapping[task]
    model = UnifiedModel(ModelConfig(
        dataset=dataset, task=model_task, input_dim=3, seed=2,
        split_identity="s", target_schema=targets.schema, pred_len=4,
        num_fault_classes=3, variant="v3c", cudtr_enabled=True,
    ))
    history = HistoryBatch(
        torch.randn(2, 5, 3), torch.arange(5.).expand(2, -1), M_obs=torch.ones(2, 5, 3),
    )
    utilities = estimate_task_counterfactual_utilities(model, history, targets)

    expected = (2, 3, 3) if task == "state" else (2, 3)
    assert utilities.shape == expected
    assert torch.isfinite(utilities).all()
    assert not utilities.requires_grad
    if task == "state":
        assert torch.equal(torch.diagonal(utilities, dim1=1, dim2=2), torch.zeros(2, 3))


def test_metropt_state_teacher_uses_fusion_tokens_not_input_source_drop(monkeypatch):
    from kaf_profiti.experiments import task_runner
    from kaf_profiti.models.unified import utility_teacher
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel
    from kaf_profiti.models.unified.heads.base import TaskTargets

    model = UnifiedModel(ModelConfig(
        dataset="metropt3", task="state", input_dim=3, seed=2,
        split_identity="s", target_schema="state", pred_len=2,
        variant="v3c", cudtr_enabled=True, head_mode="point",
    ))
    history = HistoryBatch(
        torch.randn(2, 5, 3), torch.arange(5.).expand(2, -1), M_obs=torch.ones(2, 5, 3),
    )
    targets = TaskTargets(values=torch.randn(2, 2, 3), schema="state")

    def fail_old_source_drop(*args, **kwargs):
        raise AssertionError("route-level teacher must not delete raw input channels")

    monkeypatch.setattr(utility_teacher, "build_source_drop_history", fail_old_source_drop)
    utilities = task_runner.estimate_task_counterfactual_utilities(model, history, targets)
    assert utilities.shape == (2, 3, 3)


def test_train_step_can_compute_teacher_only_for_explicit_training_call():
    from kaf_profiti.experiments.task_runner import train_step
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel
    from kaf_profiti.models.unified.heads.base import TaskTargets

    model = UnifiedModel(ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=2, seed=2,
        split_identity="s", target_schema="r", variant="v3c", cudtr_enabled=True,
        cudtr_lambda_distill=0.1,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = HistoryBatch(torch.randn(2, 4, 2), torch.arange(4.).expand(2, -1))
    targets = TaskTargets(values=torch.tensor([20.0, 40.0]), schema="r")

    teacher = UnifiedModel(model.config)
    teacher.eval()
    loss = train_step(model, optimizer, history, targets, utility_teacher_model=teacher)

    assert torch.isfinite(torch.tensor(loss))


def test_source_corruption_is_channelwise_train_input_and_preserves_unmodified_values():
    from kaf_profiti.experiments.task_runner import build_source_corruption

    history = _history()
    corrupted = build_source_corruption(history, drop_rate=0.5, generator=torch.Generator().manual_seed(5))

    assert corrupted.X_obs.shape == history.X_obs.shape
    assert (corrupted.M_obs.sum(dim=1) < history.M_obs.sum(dim=1)).any()
    assert torch.equal(corrupted.X_obs * (corrupted.M_obs > 0), history.X_obs * (corrupted.M_obs > 0))
    assert (corrupted.X_obs[corrupted.M_obs == 0] == 0).all()


def test_source_corruption_can_protect_target_channels():
    from kaf_profiti.experiments.task_runner import build_source_corruption

    history = _history()
    corrupted = build_source_corruption(
        history, drop_rate=1.0, protected_channels=[1],
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal(corrupted.M_obs[:, :, 1], history.M_obs[:, :, 1])
    assert (corrupted.M_obs[:, :, [0, 2]] <= history.M_obs[:, :, [0, 2]]).all()


def test_fallback_consistency_loss_runs_without_utility_teacher():
    from kaf_profiti.experiments.task_runner import train_step
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel
    from kaf_profiti.models.unified.heads.base import TaskTargets

    model = UnifiedModel(ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=3,
        split_identity="s", target_schema="r", variant="v3c", cudtr_enabled=True,
        cudtr_lambda_fallback=0.5,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = HistoryBatch(torch.randn(2, 5, 3), torch.arange(5.).expand(2, -1))
    targets = TaskTargets(values=torch.tensor([10.0, 20.0]), schema="r")

    loss = train_step(model, optimizer, history, targets)

    assert torch.isfinite(torch.tensor(loss))


def test_source_corruption_can_drop_exactly_one_source():
    from kaf_profiti.experiments.task_runner import build_source_corruption

    history = HistoryBatch(
        torch.ones(2, 4, 3), torch.arange(4.).expand(2, -1),
        M_obs=torch.ones(2, 4, 3),
    )
    corrupted = build_source_corruption(
        history, drop_rate=1.0, source_count=1,
        generator=torch.Generator().manual_seed(3),
    )
    dropped = (corrupted.M_obs == 0).all(dim=1)
    assert dropped.sum(dim=1).tolist() == [1, 1]
