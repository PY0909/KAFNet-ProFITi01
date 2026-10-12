from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class EncoderOutput:
    z_history: Tensor
    z_global: Tensor
    pre_cross_tokens: Optional[Tensor] = None
    z_base: Optional[Tensor] = None
    recent_features: Optional[Tensor] = None
    trend_features: Optional[Tensor] = None
    patch_tokens: Optional[Tensor] = None
    pooling_weights: Optional[Tensor] = None
    sensor_weights: Optional[Tensor] = None


class MaskedDepthwiseTemporalBlock(nn.Module):
    """Residual temporal convolution that zeros invalid positions at every layer."""

    def __init__(self, hidden_dim: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=dilation,
            dilation=dilation, groups=hidden_dim,
        )
        self.pointwise_in = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.pointwise_out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: Tensor, observed: Tensor) -> Tensor:
        channels_first = values.transpose(1, 2)
        mixed = self.depthwise(channels_first).transpose(1, 2)
        gate, candidate = self.pointwise_in(mixed.transpose(1, 2)).chunk(2, dim=1)
        mixed = self.pointwise_out((torch.sigmoid(gate) * torch.nn.functional.gelu(candidate))).transpose(1, 2)
        result = self.norm(values + mixed)
        return result * observed


class SharedTemporalEncoder(nn.Module):
    """Interface for independently trained dataset-specific encoder instances."""

    version = "unified_temporal_channelwise_v2"

    def forward(self, history):
        raise NotImplementedError("temporal encoding is implemented by UnifiedTemporalEncoder")


class UnifiedTemporalEncoder(SharedTemporalEncoder):
    """Channelwise multi-scale temporal encoder with explicit cross-channel mixing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        context_dim: int = 0,
        patch_lens=(12, 24, 48),
        variant="v2",
        sensor_pooling="attention",
        cross_channel_mode="attention",
    ):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or context_dim < 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not patch_lens or any(int(length) < 1 for length in patch_lens):
            raise ValueError("patch_lens must contain positive lengths")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.patch_lens = tuple(int(length) for length in patch_lens)
        self.scale_names = ("global",) + tuple(str(length) for length in self.patch_lens)
        if variant not in {"v2", "v2t", "v2tc", "v3a", "v3b", "v3c", "v4", "v5a", "v5b", "v5c"}:
            raise ValueError("unsupported encoder variant: {}".format(variant))
        self.variant = variant
        self.sensor_pooling_mode = sensor_pooling
        if cross_channel_mode not in {"attention", "cudtr_only"}:
            raise ValueError("cross_channel_mode must be attention or cudtr_only")
        self.cross_channel_enabled = cross_channel_mode == "attention"
        self.version = {
            "v2": "unified_temporal_channelwise_v2",
            "v2t": "unified_temporal_channelwise_v2t",
            "v2tc": "unified_temporal_channelwise_v2tc",
            "v3a": "unified_temporal_channelwise_v3a",
            "v3b": "unified_temporal_channelwise_v3b",
            "v3c": "unified_temporal_channelwise_v3c",
            "v4": "unified_temporal_channelwise_v4",
            "v5a": "unified_temporal_patchwise_v5a",
            "v5b": "unified_temporal_patchwise_v5b",
            "v5c": "unified_temporal_patchwise_v5c",
        }[variant]
        temporal_input_dim = 3 if variant in {"v2", "v2t", "v2tc"} else 6 if variant == "v4" else 4
        self.channel_identity = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        self.temporal = nn.Sequential(
            nn.Linear(temporal_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_blocks = nn.ModuleList(
            MaskedDepthwiseTemporalBlock(hidden_dim, dilation)
            for dilation in (1, 2, 4)
        ) if variant == "v4" else nn.ModuleList()
        self.context = (
            nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.GELU())
            if context_dim
            else None
        )
        self.sensor_mixer = nn.Sequential(
            nn.Linear(hidden_dim * (len(self.patch_lens) + 1) + (2 if variant in {"v2t", "v2tc"} else 0), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_channel = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        if variant == "v4":
            self.recent_trend_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Sigmoid(),
            )
            self.recent_trend_update = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.norm = nn.LayerNorm(hidden_dim)
        if variant in {"v3a", "v3b", "v3c", "v4"}:
            self.pool_scorers = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(len(self.patch_lens) + 1))
        else:
            self.pool_scorers = None
        self.scale_scorer = (
            nn.Linear(hidden_dim, 1) if variant in {"v3b", "v3c", "v4"} else None
        )
        self.sensor_scorer = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            if variant == "v2tc" or (variant in {"v3c", "v5a", "v5b", "v5c"} and sensor_pooling in {"attention", "mean_attention"}) else None
        )
        self.patch_mixer = (
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                nn.GELU(), nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            ) if variant in {"v5b", "v5c"} else None
        )
        self.patch_query = nn.Linear(hidden_dim, hidden_dim) if variant == "v5c" else None
        self.trend_projection = (
            nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            if variant in {"v3c"} else None
        )

    @staticmethod
    def _channel_trend_features(values: Tensor, observed: Tensor) -> Tensor:
        """Masked per-channel least-squares slope and recent-mean shift.

        values: [batch, history, channels] already zeroed at unobserved positions.
        Returns [batch, channels, 2]: (slope, recent shift) over observed positions only.
        """
        history_len = values.shape[1]
        mask = observed
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        positions = torch.arange(history_len, dtype=values.dtype, device=values.device).view(1, history_len, 1)
        mean_position = (mask * positions).sum(dim=1, keepdim=True) / count
        mean_value = (mask * values).sum(dim=1, keepdim=True) / count
        centered_position = (positions - mean_position) * mask
        centered_value = (values - mean_value) * mask
        slope_denominator = centered_position.square().sum(dim=1, keepdim=True).clamp_min(1e-6)
        slope = (centered_position * centered_value).sum(dim=1, keepdim=True) / slope_denominator
        recent_width = min(10, history_len)
        recent_mask = mask[:, -recent_width:, :]
        recent_count = recent_mask.sum(dim=1, keepdim=True)
        recent_mean = (recent_mask * values[:, -recent_width:, :]).sum(dim=1, keepdim=True) / recent_count.clamp_min(1.0)
        has_recent = (recent_count > 0).to(values.dtype)
        recent_shift = (recent_mean - mean_value) * has_recent
        return torch.cat((slope, recent_shift), dim=1).permute(0, 2, 1)

    @staticmethod
    def _safe_time_features(timestamps: Tensor) -> Tensor:
        if timestamps.ndim == 3:
            timestamps = timestamps.squeeze(-1)
        timestamps = timestamps.to(dtype=torch.float32)
        origin = timestamps[:, :1]
        relative = timestamps - origin
        scale = relative.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        normalized = relative / scale
        delta = torch.diff(timestamps, dim=1, prepend=timestamps[:, :1])
        delta = torch.log1p(delta.abs()).unsqueeze(-1)
        return torch.cat((normalized.unsqueeze(-1), delta), dim=-1)

    def forward(self, history) -> EncoderOutput:
        values = history.X_obs
        if values.shape[-1] != self.input_dim:
            raise ValueError("history channel count does not match encoder input_dim")
        if not torch.isfinite(values).all():
            raise ValueError("history X_obs contains non-finite values")
        time_features = self._safe_time_features(history.T_obs).to(values.device)
        if not torch.isfinite(time_features).all():
            raise ValueError("history T_obs contains non-finite values")
        timestamps = history.T_obs.squeeze(-1) if history.T_obs.ndim == 3 else history.T_obs
        if timestamps.shape[1] > 1 and (torch.diff(timestamps, dim=1) < 0).any():
            raise ValueError("history T_obs must be non-decreasing")
        batch_size, history_len, channels = values.shape
        observed = getattr(history, "M_obs", None)
        if observed is None:
            observed = torch.ones_like(values)
        else:
            observed = observed.to(device=values.device, dtype=values.dtype)
            if observed.shape != values.shape:
                raise ValueError("history M_obs must match X_obs shape")
            if not torch.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
                raise ValueError("history M_obs must contain finite values in [0, 1]")
        observed = (observed > 0).to(values.dtype)
        safe_values = values * observed
        channel_values = safe_values.permute(0, 2, 1)
        channel_times = time_features.unsqueeze(1).expand(-1, channels, -1, -1)
        if self.variant in {"v2", "v2t", "v2tc"}:
            channel_features = torch.cat((channel_values.unsqueeze(-1), channel_times), dim=-1)
        elif self.variant == "v4":
            channel_mask = observed.permute(0, 2, 1)
            previous_values = torch.cat((channel_values[:, :, :1], channel_values[:, :, :-1]), dim=2)
            previous_mask = torch.cat((channel_mask[:, :, :1], channel_mask[:, :, :-1]), dim=2)
            delta_valid = channel_mask * previous_mask
            delta = (channel_values - previous_values) * delta_valid
            previous_delta = torch.cat((torch.zeros_like(delta[:, :, :1]), delta[:, :, :-1]), dim=2)
            if history_len > 1:
                previous_previous_mask = torch.cat((
                    torch.zeros_like(channel_mask[:, :, :2]), channel_mask[:, :, :-2],
                ), dim=2)
            else:
                previous_previous_mask = torch.zeros_like(channel_mask)
            second_delta_valid = channel_mask * previous_mask * previous_previous_mask
            second_delta = (delta - previous_delta) * second_delta_valid
            channel_features = torch.cat((
                channel_values.unsqueeze(-1), channel_times,
                channel_mask.unsqueeze(-1), delta.unsqueeze(-1), second_delta.unsqueeze(-1),
            ), dim=-1)
        elif self.variant in {"v3a", "v3b", "v3c", "v5a", "v5b", "v5c"}:
            channel_observed = observed.permute(0, 2, 1).unsqueeze(-1)
            channel_features = torch.cat((channel_values.unsqueeze(-1), channel_times, channel_observed), dim=-1)
        temporal = self.temporal(channel_features.reshape(batch_size * channels, history_len, channel_features.shape[-1]))
        temporal = temporal.reshape(batch_size, channels, history_len, self.hidden_dim)
        temporal = temporal + self.channel_identity.unsqueeze(0).unsqueeze(2)
        patch_tokens = None
        if self.variant in {"v5a", "v5b", "v5c"}:
            patch_len = min(self.patch_lens)
            patch_count = max(1, (history_len + patch_len - 1) // patch_len)
            padded = patch_count * patch_len - history_len
            if padded:
                temporal = torch.nn.functional.pad(temporal, (0, 0, padded, 0))
                observed_for_patches = torch.nn.functional.pad(observed, (0, 0, padded, 0))
            else:
                observed_for_patches = observed
            patch_values = temporal.reshape(batch_size, channels, patch_count, patch_len, self.hidden_dim)
            patch_mask = observed_for_patches.permute(0, 2, 1).reshape(
                batch_size, channels, patch_count, patch_len, 1
            )
            patch_tokens = (patch_values * patch_mask).sum(dim=3) / patch_mask.sum(dim=3).clamp_min(1.0)
            patch_valid = patch_mask.any(dim=3)
            if self.patch_mixer is not None:
                mixed = patch_tokens.reshape(batch_size * channels, patch_count, self.hidden_dim).transpose(1, 2)
                mixed = self.patch_mixer(mixed).transpose(1, 2)
                patch_tokens = patch_tokens + mixed.reshape(batch_size, channels, patch_count, self.hidden_dim)
                patch_tokens = patch_tokens * patch_valid
            if self.patch_query is not None:
                query = self.patch_query(patch_tokens[:, :, -1:, :])
                logits = (patch_tokens * query).sum(dim=-1) / (self.hidden_dim ** 0.5)
                logits = logits.masked_fill(~patch_valid.squeeze(-1), torch.finfo(logits.dtype).min)
                weights = torch.softmax(logits, dim=-1)
                weights = torch.where(patch_valid.squeeze(-1).any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))
                z_history = torch.sum(patch_tokens * weights.unsqueeze(-1), dim=2)
            else:
                valid_count = patch_valid.sum(dim=2).clamp_min(1)
                z_history = patch_tokens.sum(dim=2) / valid_count
        recent_features = None
        trend_features = None
        if self.variant == "v4":
            mask_by_channel = observed.permute(0, 2, 1).unsqueeze(-1)
            temporal = temporal * mask_by_channel
            temporal_flat = temporal.reshape(batch_size * channels, history_len, self.hidden_dim)
            mask_flat = mask_by_channel.reshape(batch_size * channels, history_len, 1)
            for block in self.temporal_blocks:
                temporal_flat = block(temporal_flat, mask_flat)
            temporal = temporal_flat.reshape(batch_size, channels, history_len, self.hidden_dim)
            recent_width = min(4, history_len)
            valid = observed.permute(0, 2, 1).unsqueeze(-1)
            recent_values = temporal[:, :, -recent_width:]
            recent_valid = valid[:, :, -recent_width:]
            recent_count = recent_valid.sum(dim=2).clamp_min(1.0)
            recent_features = (recent_values * recent_valid).sum(dim=2) / recent_count
            if history_len > recent_width:
                prior_values = temporal[:, :, :-recent_width]
                prior_valid = valid[:, :, :-recent_width]
                prior_count = prior_valid.sum(dim=2).clamp_min(1.0)
                prior_features = (prior_values * prior_valid).sum(dim=2) / prior_count
                has_prior = prior_valid.any(dim=2).to(temporal.dtype)
                trend_features = (recent_features - prior_features) * has_prior
            else:
                trend_features = torch.zeros_like(recent_features)
        trend_summary = None
        if self.variant in {"v2t", "v2tc", "v3c"}:
            trend_summary = self._channel_trend_features(safe_values, observed)
        if self.variant in {"v5a", "v5b", "v5c"}:
            summaries = []
            pooling_weights = None
        elif self.variant in {"v2", "v2t", "v2tc"}:
            temporal = temporal * observed.permute(0, 2, 1).unsqueeze(-1)
            summaries = [temporal.mean(dim=2)]
            for patch_len in self.patch_lens:
                window = temporal[:, :, -min(patch_len, history_len):, :]
                summaries.append(window.mean(dim=2))
            pooling_weights = None
        else:
            mask_by_channel = observed.permute(0, 2, 1).bool()
            temporal = temporal * mask_by_channel.unsqueeze(-1)
            summaries = []
            pooling_weights = torch.zeros(
                batch_size, channels, len(self.patch_lens) + 1, history_len,
                device=values.device, dtype=values.dtype,
            )
            for scale_index, patch_len in enumerate((history_len,) + self.patch_lens):
                length = min(patch_len, history_len)
                start = history_len - length
                window = temporal[:, :, start:, :]
                valid = mask_by_channel[:, :, start:]
                logits = self.pool_scorers[scale_index](window).squeeze(-1)
                logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
                has_observation = valid.any(dim=-1, keepdim=True)
                weights = torch.softmax(logits, dim=-1)
                weights = torch.where(has_observation, weights, torch.zeros_like(weights))
                summaries.append(torch.sum(window * weights.unsqueeze(-1), dim=2))
                pooling_weights[:, :, scale_index, start:] = weights
        if self.variant in {"v5a", "v5b", "v5c"}:
            pass
        elif self.variant in {"v3b", "v3c", "v4"}:
            patch_tokens = torch.stack(summaries, dim=2)
            scale_logits = self.scale_scorer(patch_tokens).squeeze(-1)
            scale_weights = torch.softmax(scale_logits, dim=2)
            z_history = torch.sum(patch_tokens * scale_weights.unsqueeze(-1), dim=2)
        else:
            if trend_summary is not None:
                summaries.append(trend_summary)
            z_history = self.sensor_mixer(torch.cat(summaries, dim=-1))
        if self.variant == "v4":
            recent_trend = torch.cat((z_history, recent_features, trend_features), dim=-1)
            update = self.recent_trend_update(recent_trend)
            gate = self.recent_trend_gate(recent_trend)
            z_history = z_history + gate * update
        if self.variant == "v3c" and trend_summary is not None:
            trend_features = self.trend_projection(trend_summary)
        pre_cross_tokens = z_history
        z_history, z_global = self.fuse_pre_cross_tokens(pre_cross_tokens, history)
        return EncoderOutput(
            z_history=z_history,
            z_global=z_global,
            pre_cross_tokens=pre_cross_tokens,
            z_base=z_global,
            recent_features=recent_features,
            trend_features=trend_features,
            patch_tokens=patch_tokens,
            pooling_weights=pooling_weights,
            sensor_weights=getattr(self, "_last_sensor_weights", None),
        )

    def fuse_pre_cross_tokens(self, tokens: Tensor, history) -> tuple[Tensor, Tensor]:
        """Apply the same cross-sensor path used by ``forward``.

        This is intentionally exposed for the train-only CUDTR teacher so a
        source token can be removed at the fusion boundary rather than by
        deleting a raw input channel.
        """
        z_history = tokens
        if self.cross_channel_enabled:
            attended, _ = self.cross_channel(z_history, z_history, z_history, need_weights=False)
            z_history = z_history + attended
        if self.context is not None:
            if history.context is None:
                context = torch.zeros(tokens.shape[0], self.context_dim, device=tokens.device, dtype=tokens.dtype)
            else:
                context = history.context.to(device=tokens.device, dtype=tokens.dtype)
                if context.ndim != 2 or context.shape[1] != self.context_dim:
                    raise ValueError("context must have shape [batch, context_dim]")
                if not torch.isfinite(context).all():
                    raise ValueError("history context contains non-finite values")
            z_history = z_history + self.context(context).unsqueeze(1)
        z_history = self.norm(z_history)
        sensor_weights = None
        if self.sensor_scorer is not None:
            sensor_logits = self.sensor_scorer(z_history).squeeze(-1)
            sensor_weights = torch.softmax(sensor_logits, dim=1)
            attended_global = torch.sum(z_history * sensor_weights.unsqueeze(-1), dim=1)
            if self.variant == "v3c" and self.sensor_pooling_mode == "mean_attention":
                z_global = z_history.mean(dim=1) + attended_global
            else:
                z_global = attended_global
        elif self.variant == "v3c" and self.sensor_pooling_mode == "mean":
            sensor_weights = None
            z_global = z_history.mean(dim=1)
        else:
            z_global = z_history.mean(dim=1)
        self._last_sensor_weights = sensor_weights
        return z_history, z_global
