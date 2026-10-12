import pytest
import torch

from kaf_profiti.models.unified.routing import CUDTRouting, counterfactual_utility


def test_top_k_route_is_normalized_and_limits_sources():
    router = CUDTRouting(hidden_dim=4, query_dim=2, top_k=2, temperature=1.0)
    base = torch.randn(3, 4)
    sources = torch.randn(3, 5, 4)
    query = torch.randn(3, 2)
    observed = torch.ones(3, 5)

    result = router(base, sources, query, observed)

    assert result.weights.shape == (3, 5)
    assert torch.allclose(result.weights.sum(dim=1), torch.ones(3), atol=1e-6)
    assert (result.weights > 0).sum(dim=1).max().item() <= 2
    assert torch.isfinite(result.fused).all()


def test_all_sources_missing_returns_base_representation():
    router = CUDTRouting(hidden_dim=4, query_dim=2, top_k=2, temperature=1.0)
    base = torch.randn(2, 4)
    sources = torch.randn(2, 3, 4)
    query = torch.randn(2, 2)
    observed = torch.zeros(2, 3)

    result = router(base, sources, query, observed)

    assert torch.allclose(result.fused, base, atol=1e-6)
    assert torch.allclose(result.weights, torch.zeros_like(result.weights))
    assert torch.allclose(result.fallback, torch.ones(2), atol=1e-6)


def test_counterfactual_utility_is_detached():
    full_loss = torch.tensor([1.0, 2.0], requires_grad=True)
    dropped_loss = torch.tensor([1.5, 1.0], requires_grad=True)

    utility = counterfactual_utility(full_loss, dropped_loss)

    assert torch.allclose(utility, torch.tensor([0.5, -1.0]))
    assert not utility.requires_grad


def test_prediction_route_does_not_accept_targets():
    router = CUDTRouting(hidden_dim=4, query_dim=2, top_k=1)
    with pytest.raises(TypeError):
        router(
            torch.zeros(1, 4),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            targets=torch.zeros(1),
        )


def test_router_has_finite_gradients_for_available_sources():
    router = CUDTRouting(
        hidden_dim=4,
        query_dim=2,
        top_k=2,
        source_feature_dim=3,
    )
    base = torch.randn(2, 4, requires_grad=True)
    sources = torch.randn(2, 3, 4, requires_grad=True)
    query = torch.randn(2, 2)
    observed = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    source_features = torch.randn(2, 3, 3)

    result = router(base, sources, query, observed, source_features)
    result.fused.square().mean().backward()

    assert torch.isfinite(base.grad).all()
    assert torch.isfinite(sources.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in router.parameters())


def test_router_rejects_invalid_observation_mask():
    router = CUDTRouting(hidden_dim=4, query_dim=2, top_k=1)
    with pytest.raises(ValueError, match="source_observed"):
        router(
            torch.zeros(1, 4),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2),
            torch.tensor([[1.0, float("nan")]]),
        )


def test_router_supports_dense_training_route_and_hard_inference_route():
    router = CUDTRouting(hidden_dim=4, query_dim=2, top_k=1, hard_top_k=False)
    base = torch.randn(2, 4)
    sources = torch.randn(2, 3, 4)
    query = torch.randn(2, 2)
    observed = torch.ones(2, 3)

    dense = router(base, sources, query, observed)
    assert (dense.weights > 0).sum(dim=1).min().item() == 3
    router.set_hard_top_k(True)
    hard = router(base, sources, query, observed)
    assert (hard.weights > 0).sum(dim=1).max().item() <= 1


def test_unavailable_sources_receive_zero_route_weight():
    router = CUDTRouting(hidden_dim=4, query_dim=2, top_k=2)
    result = router(
        torch.randn(1, 4), torch.randn(1, 3, 4), torch.randn(1, 2),
        torch.tensor([[1.0, 0.0, 1.0]]),
    )
    assert result.weights[0, 1].item() == 0.0


def test_router_can_disable_sources_below_utility_threshold():
    router = CUDTRouting(hidden_dim=2, query_dim=1, top_k=2, utility_threshold=0.0)
    with torch.no_grad():
        for layer in router.utility_predictor:
            if hasattr(layer, "weight"):
                layer.weight.zero_()
                layer.bias.zero_()
        router.utility_predictor[-1].bias.fill_(-1.0)
    result = router(torch.zeros(1, 2), torch.ones(1, 2, 2), torch.zeros(1, 1), torch.ones(1, 2))
    assert torch.equal(result.weights, torch.zeros_like(result.weights))
    assert torch.allclose(result.fused, torch.zeros_like(result.fused))
    assert torch.allclose(result.fallback, torch.ones(1))


def test_utility_prediction_is_bounded_when_utility_clip_is_set():
    router = CUDTRouting(hidden_dim=2, query_dim=1, top_k=1, utility_clip=1.0)
    with torch.no_grad():
        for layer in router.utility_predictor:
            if hasattr(layer, "weight"):
                layer.weight.fill_(100.0)
                layer.bias.fill_(100.0)
    result = router(
        torch.ones(2, 2), torch.ones(2, 3, 2), torch.ones(2, 1), torch.ones(2, 3),
    )
    assert result.utility_prediction.abs().max().item() <= 1.0 + 1e-6


def test_unified_model_uses_soft_route_during_training_and_hard_route_for_eval():
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel
    from kaf_profiti.models.unified.model import HistoryBatch

    model = UnifiedModel(ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=4, seed=1,
        split_identity="s", target_schema="r", cudtr_enabled=True, cudtr_top_k=1,
    ))
    model.train()
    output = model.predict(HistoryBatch(torch.randn(2, 5, 4), torch.arange(5.).expand(2, -1)))
    weights = output.metadata["cudtr"]["route_weights"]
    assert (weights > 0).sum(dim=-1).min().item() == 4
    model.eval()
    output = model.predict(HistoryBatch(torch.randn(2, 5, 4), torch.arange(5.).expand(2, -1)))
    weights = output.metadata["cudtr"]["route_weights"]
    assert (weights > 0).sum(dim=-1).max().item() <= 1
