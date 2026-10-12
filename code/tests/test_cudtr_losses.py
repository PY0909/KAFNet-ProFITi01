import torch

from kaf_profiti.models.unified.cudtr_loss import (
    CUDTRLossConfig,
    cudtr_auxiliary_loss,
    fallback_consistency_loss,
    utility_distillation_loss,
    utility_ranking_loss,
)


def test_utility_distillation_and_ranking_are_finite_and_differentiable():
    predicted = torch.tensor([[0.8, -0.2, 0.1]], requires_grad=True)
    target = torch.tensor([[1.0, -1.0, 0.0]])
    distill = utility_distillation_loss(predicted, target)
    ranking = utility_ranking_loss(predicted, target, margin=0.1)

    assert distill.item() > 0
    assert ranking.item() >= 0
    (distill + ranking).backward()
    assert torch.isfinite(predicted.grad).all()


def test_auxiliary_loss_combines_only_enabled_terms():
    predicted = torch.tensor([[0.8, -0.2]], requires_grad=True)
    target = torch.tensor([[1.0, -1.0]])
    weights = torch.tensor([[0.8, 0.2]])
    base = torch.zeros(1, 2)
    corrupted = torch.zeros(1, 2)
    config = CUDTRLossConfig(
        lambda_distill=1.0,
        lambda_rank=1.0,
        lambda_sparse=0.5,
        lambda_fallback=0.25,
    )

    result = cudtr_auxiliary_loss(
        predicted,
        target,
        weights=weights,
        base=base,
        corrupted_fused=corrupted,
        config=config,
    )
    assert set(result) == {"total", "distill", "rank", "sparse", "fallback"}
    assert result["total"].ndim == 0
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert torch.isfinite(predicted.grad).all()


def test_auxiliary_loss_rejects_mismatched_teacher_shape_and_zeroes_disabled_terms():
    config = CUDTRLossConfig()
    predicted = torch.zeros(2, 3)
    target = torch.zeros(2, 2)
    try:
        cudtr_auxiliary_loss(predicted, target, config=config)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("mismatched utility shapes must fail")

    result = cudtr_auxiliary_loss(torch.zeros(1, 2), torch.zeros(1, 2), config=config)
    assert result["total"].item() == 0.0


def test_cudtr_loss_config_validates_nonnegative_weights():
    try:
        CUDTRLossConfig(lambda_rank=-1.0)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative loss weights must fail")


def test_fallback_consistency_can_supervise_gate_direction():
    fused = torch.zeros(2, 3, requires_grad=True)
    base = torch.ones(2, 3)
    fallback = torch.tensor([0.1, 0.9], requires_grad=True)
    result = fallback_consistency_loss(fused, base, fallback=fallback, corrupted=True)
    assert result.item() > 0
    result.backward()
    assert torch.isfinite(fallback.grad).all()


def test_fallback_consistency_keeps_clean_routes_open_and_corrupted_routes_closed():
    fused = torch.zeros(1, 2)
    base = torch.zeros(1, 2)
    clean = torch.tensor([0.8], requires_grad=True)
    corrupted = torch.tensor([0.2], requires_grad=True)
    result = fallback_consistency_loss(
        fused, base, fallback=corrupted, clean_fallback=clean, corrupted=True,
    )
    result.backward()
    assert clean.grad.item() > 0
    assert corrupted.grad.item() < 0


def test_fallback_consistency_requires_corrupted_gate_above_clean_gate():
    base = torch.zeros(2, 3, 4)
    fused = torch.zeros_like(base)
    clean = torch.full((2, 3), 0.9)
    corrupted = torch.full((2, 3), 0.9)

    loss = fallback_consistency_loss(
        fused, base, fallback=corrupted, clean_fallback=clean, margin=0.2,
    )
    separated = fallback_consistency_loss(
        fused, base, fallback=torch.ones(2, 3),
        clean_fallback=torch.zeros(2, 3), margin=0.2,
    )
    assert loss.item() > separated.item()


def test_model_exposes_differentiable_auxiliary_terms_only_after_cudtr_forward():
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel
    from kaf_profiti.models.unified.model import HistoryBatch
    from kaf_profiti.models.unified.heads.base import TaskTargets

    model = UnifiedModel(ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=1,
        split_identity="s", target_schema="r", cudtr_enabled=True,
        cudtr_lambda_distill=1.0,
    ))
    history = HistoryBatch(torch.randn(2, 5, 3), torch.arange(5.).expand(2, -1))
    prediction = model.predict(history)
    model_loss = model.loss(prediction, TaskTargets(values=torch.ones(2), schema="r"))
    terms = model.cudtr_auxiliary_loss(torch.zeros(2, 3))
    assert terms["total"].requires_grad
    (model_loss + terms["total"]).backward()
    assert any(parameter.grad is not None for parameter in model.cudtr.parameters())


def test_train_step_keeps_utility_teacher_explicit():
    from kaf_profiti.experiments.task_runner import train_step
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel
    from kaf_profiti.models.unified.model import HistoryBatch
    from kaf_profiti.models.unified.heads.base import TaskTargets

    model = UnifiedModel(ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=2, seed=1,
        split_identity="s", target_schema="r", cudtr_enabled=True,
        cudtr_lambda_distill=1.0,
    ))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = HistoryBatch(torch.randn(2, 4, 2), torch.arange(4.).expand(2, -1))
    targets = TaskTargets(values=torch.ones(2), schema="r")
    assert train_step(model, optimizer, history, targets, utility_targets=torch.zeros(2, 2)) >= 0
