from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .encoder import UnifiedTemporalEncoder
from .heads.rul import RULHead
from .heads.state import StateHead
from .heads.tep import TEPHead
from .output import PredictionOutput
from .routing import CUDTRouting
from .cudtr_loss import CUDTRLossConfig, cudtr_auxiliary_loss


def _safe_quantile(values, quantile, dim):
    """Compute diagnostics quantiles after promoting unsupported dtypes."""
    return torch.quantile(values.detach().float(), quantile, dim=dim)


@dataclass(frozen=True)
class HistoryBatch:
    """Prediction-time inputs only; supervised targets live in TaskTargets."""

    X_obs: Tensor
    T_obs: Tensor
    context: Optional[Tensor] = None
    M_obs: Optional[Tensor] = None

    def __post_init__(self):
        if self.X_obs.ndim != 3:
            raise ValueError("X_obs must have shape [batch, history, channels]")
        if self.T_obs.shape not in (
            self.X_obs.shape[:2],
            self.X_obs.shape[:2] + (1,),
        ):
            raise ValueError("T_obs must have shape [batch, history] or [batch, history, 1]")
        if self.context is not None and self.context.shape[0] != self.X_obs.shape[0]:
            raise ValueError("context batch dimension must match X_obs")
        if self.M_obs is not None:
            if self.M_obs.shape != self.X_obs.shape:
                raise ValueError("M_obs must have the same shape as X_obs")
            if not torch.isfinite(self.M_obs).all() or ((self.M_obs < 0) | (self.M_obs > 1)).any():
                raise ValueError("M_obs must contain finite values in [0, 1]")


class UnifiedModel(nn.Module):
    """Public prediction/loss contract; concrete stages provide computation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = UnifiedTemporalEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            context_dim=config.context_dim,
            patch_lens=config.patch_lens,
            variant=config.variant,
            sensor_pooling=config.sensor_pooling,
            cross_channel_mode=config.cross_channel_mode,
        )
        if config.task == "state":
            self.head = StateHead(config.hidden_dim, config.input_dim, config.pred_len, decoder_mode=config.state_decoder)
        elif config.task == "rul":
            self.head = RULHead(config.hidden_dim, training_cap=config.training_cap,
                                over_penalty=config.rul_over_penalty, input_mode=config.rul_head_input)
        elif config.task == "fault_detection_diagnosis":
            self.head = TEPHead(config.hidden_dim, config.num_fault_classes, class_weights=config.tep_class_weights)
        else:
            raise ValueError("unsupported task: {}".format(config.task))
        self.cudtr = (
            CUDTRouting(
                hidden_dim=config.hidden_dim,
                query_dim=config.cudtr_query_dim,
                top_k=min(config.cudtr_top_k, config.input_dim),
                temperature=config.cudtr_temperature,
                source_feature_dim=2 + config.context_dim,
                fallback_init=config.cudtr_fallback_init,
                utility_threshold=config.cudtr_utility_threshold,
                utility_clip=config.cudtr_utility_clip,
            )
            if config.cudtr_enabled else None
        )
        self.cudtr_queries = (
            nn.Parameter(torch.randn(max(1, config.input_dim), config.cudtr_query_dim) * 0.02)
            if config.cudtr_enabled else None
        )
        self.cudtr_loss_config = CUDTRLossConfig(
            lambda_distill=config.cudtr_lambda_distill,
            lambda_rank=config.cudtr_lambda_rank,
            lambda_sparse=config.cudtr_lambda_sparse,
            lambda_fallback=config.cudtr_lambda_fallback,
            rank_margin=config.cudtr_rank_margin,
            utility_clip=config.cudtr_utility_clip,
        )
        self._last_cudtr_routes = None
        self._last_cudtr_bases = None

    @staticmethod
    def _source_features(history, device):
        """Build source reliability features from history only.

        The two intrinsic features are observation rate and masked variability.
        Context is repeated for each source when the dataset provides it.
        Values at unobserved positions are zeroed before any statistic is used.
        """
        values = history.X_obs.to(device)
        observed = history.M_obs
        if observed is None:
            observed = torch.ones_like(values)
        else:
            observed = observed.to(device=device, dtype=values.dtype)
        observed = (observed > 0).to(values.dtype)
        safe_values = values * observed
        rate = observed.mean(dim=1)
        count = observed.sum(dim=1).clamp_min(1.0)
        mean = safe_values.sum(dim=1) / count
        variance = ((safe_values - mean[:, None, :]).square() * observed).sum(dim=1) / count
        variability = variance.clamp_min(0.0).sqrt()
        features = [rate.unsqueeze(-1), variability.unsqueeze(-1)]
        if history.context is not None:
            context = history.context.to(device=device, dtype=values.dtype)
            features.append(context[:, None, :].expand(-1, values.shape[-1], -1))
        return torch.cat(features, dim=-1)

    def _apply_cudtr(self, encoded, history):
        if self.cudtr is None:
            return encoded, {"enabled": False}
        source_tokens = encoded.pre_cross_tokens
        if source_tokens is None:
            raise RuntimeError("CUDTR requires encoder pre_cross_tokens")
        # Dense routing supplies gradients during training; hard top-k is an
        # inference-time sparsification rule.
        self.cudtr.set_hard_top_k(not self.training)
        observed = history.M_obs
        if observed is None:
            observed = torch.ones_like(history.X_obs)
        observed = observed.to(source_tokens.device)
        observation_rate = (observed > 0).to(source_tokens.dtype).mean(dim=1)
        source_observed = observation_rate.gt(
            max(float(self.config.cudtr_min_observation_rate), 0.0)
        ).to(source_tokens.dtype)
        source_features = self._source_features(history, source_tokens.device)
        batch, channels, hidden = source_tokens.shape
        routes = []
        base_representations = []
        available_masks = []
        if self.config.task == "state":
            # Source and target representations share the pre-cross fusion boundary.
            baseline_tokens = encoded.pre_cross_tokens
            for target_index in range(channels):
                # Keep the pre-CUDTR V3C representation as the fallback path.
                target_base = baseline_tokens[:, target_index]
                base_representations.append(target_base)
                available = source_observed.clone()
                available[:, target_index] = 0
                available_masks.append(available)
                route = self.cudtr(
                    target_base,
                    source_tokens,
                    self.cudtr_queries[target_index].unsqueeze(0).expand(batch, -1),
                    available,
                    source_features,
                )
                routes.append(route)
            encoded.z_history = torch.stack([route.fused for route in routes], dim=1)
            encoded.z_global = encoded.z_history.mean(dim=1)
        else:
            baseline_global = encoded.pre_cross_tokens.mean(dim=1)
            base_representations.append(baseline_global)
            route = self.cudtr(
                baseline_global,
                source_tokens,
                self.cudtr_queries[0].unsqueeze(0).expand(batch, -1),
                source_observed,
                source_features,
            )
            encoded.z_history = encoded.pre_cross_tokens + (route.fused - baseline_global).unsqueeze(1)
            encoded.z_global = route.fused
            routes = [route]
            available_masks.append(source_observed)
        route_weights = torch.stack([route.weights for route in routes], dim=1)
        fallback = torch.stack([route.fallback for route in routes], dim=1)
        utility = torch.stack([route.utility_prediction for route in routes], dim=1)
        entropy = torch.stack([route.route_entropy for route in routes], dim=1)
        disagreement = torch.stack([
            (route.fused - base).norm(dim=-1) for route, base in zip(routes, base_representations)
        ], dim=1)
        self._last_cudtr_routes = routes
        self._last_cudtr_bases = torch.stack(base_representations, dim=1)
        metadata = {
            "enabled": True,
            "route_weights": route_weights.detach().cpu(),
            "fallback_gate": fallback.detach().cpu(),
            "utility_prediction": utility.detach().cpu(),
            "route_entropy": entropy.detach().cpu(),
            "route_disagreement": disagreement.detach().cpu(),
            "source_available": torch.stack(available_masks, dim=1).detach().cpu(),
            "fusion_boundary": "pre_cross_shared_norm",
            "cross_channel_mode": self.config.cross_channel_mode,
            "top_k": self.config.cudtr_top_k,
            "temperature": self.config.cudtr_temperature,
            "source_feature_dim": self.cudtr.source_feature_dim,
            "utility_threshold": self.cudtr.utility_threshold,
        }
        metadata["fallback_mean"] = fallback.detach().mean(dim=1).cpu()
        metadata["fallback_p95"] = _safe_quantile(fallback, 0.95, dim=1).cpu()
        metadata["fallback_saturation_rate"] = fallback.detach().ge(0.95).float().mean(dim=1).cpu()
        metadata["route_inactive"] = (
            (disagreement.detach().abs().mean(dim=1) <= 1e-8)
            | (route_weights.detach().sum(dim=-1) == 0).all(dim=1)
        ).cpu()
        return encoded, metadata

    def predict(self, history: HistoryBatch) -> PredictionOutput:
        encoded = self.encoder(history)
        encoded, cudtr_metadata = self._apply_cudtr(encoded, history)
        output = self.head(encoded, mode=self.config.head_mode)
        output.dataset = self.config.dataset
        output.task = self.config.task
        output.encoder_version = self.config.encoder_version
        output.seed = self.config.seed
        output.split_identity = self.config.split_identity
        output.target_schema = self.config.target_schema
        output.metadata.update({"head_mode": self.config.head_mode, "config": self.config.to_dict()})
        output.metadata["probability_contract"] = {
            "kind": "state_quantile" if output.quantiles is not None else (
                "rul_scalar" if output.point is not None and self.config.task == "rul" else
                "tep_detection_diagnosis" if self.config.task == "fault_detection_diagnosis" else "state_point"
            ),
            "target_convention": output.metadata.get("target_convention", self.config.target_schema),
        }
        output.metadata["cudtr"] = cudtr_metadata
        output.metadata["representation_contract"] = (
            "per_sensor_state_head" if self.config.task == "state" else "global_task_head"
        )
        if encoded.patch_tokens is not None:
            output.metadata.update({
                "patch_token_count": encoded.patch_tokens.shape[2],
                "patch_lens": list(self.config.patch_lens),
            })
        if encoded.sensor_weights is not None:
            output.metadata["sensor_pooling"] = "attention"
        return output

    def loss(self, prediction: PredictionOutput, targets):
        return self.head.loss(prediction, targets)

    def cudtr_auxiliary_loss(self, utility_targets=None, corrupted_history=None):
        """Return train-only CUDTR terms after a differentiable ``predict`` call.

        ``utility_targets`` must be produced by the train-split counterfactual
        teacher. It is deliberately an explicit argument and is never read
        from ``HistoryBatch`` or ``PredictionOutput`` during inference.
        """
        zero = next(self.parameters()).sum() * 0.0
        if self.cudtr is None or self._last_cudtr_routes is None:
            return {"total": zero, "distill": zero, "rank": zero, "sparse": zero, "fallback": zero}
        predicted = torch.stack([route.utility_prediction for route in self._last_cudtr_routes], dim=1)
        weights = torch.stack([route.weights for route in self._last_cudtr_routes], dim=1)
        clean_fallback = torch.stack([route.fallback for route in self._last_cudtr_routes], dim=1)
        if utility_targets is None:
            utility_targets = torch.zeros_like(predicted)
            # No teacher means no utility supervision; keep optional structural
            # terms available without fabricating a target signal.
            config = CUDTRLossConfig(
                lambda_sparse=self.cudtr_loss_config.lambda_sparse,
                lambda_fallback=self.cudtr_loss_config.lambda_fallback,
            )
        else:
            utility_targets = utility_targets.to(predicted.device)
            if predicted.ndim == 3 and utility_targets.ndim == 2:
                utility_targets = utility_targets.unsqueeze(1).expand_as(predicted)
            config = self.cudtr_loss_config
        corrupted_fused = None
        corrupted_base = None
        if corrupted_history is not None and config.lambda_fallback:
            self.predict(corrupted_history)
            corrupted_fused = torch.stack([route.fused for route in self._last_cudtr_routes], dim=1)
            corrupted_base = self._last_cudtr_bases
            corrupted_fallback = torch.stack([route.fallback for route in self._last_cudtr_routes], dim=1)
        else:
            corrupted_fallback = None
        return cudtr_auxiliary_loss(
            predicted,
            utility_targets,
            weights=weights,
            corrupted_fused=corrupted_fused,
            base=corrupted_base,
            fallback=corrupted_fallback,
            clean_fallback=clean_fallback,
            config=config,
        )

    def parameter_count(self):
        encoder = sum(parameter.numel() for parameter in self.encoder.parameters())
        head = sum(parameter.numel() for parameter in self.head.parameters())
        routing = sum(parameter.numel() for parameter in self.cudtr.parameters()) if self.cudtr is not None else 0
        queries = self.cudtr_queries.numel() if self.cudtr_queries is not None else 0
        return {
            "encoder": encoder,
            "head": head,
            "routing": routing,
            "queries": queries,
            "total": encoder + head + routing + queries,
        }
