from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class MissingMechanismSimulator:
    """Deterministic asynchronous-observation mask simulator."""

    mode: str = "mixed"
    random_keep_prob: float = 0.85
    min_period: int = 1
    max_period: int = 4
    block_prob: float = 0.25
    max_block_fraction: float = 0.25

    def __call__(self, shape: Tuple[int, int], seed: int) -> Tensor:
        history_len, num_sensors = shape
        if self.mode == "none":
            return torch.ones(history_len, num_sensors, dtype=torch.float32)

        generator = torch.Generator().manual_seed(int(seed))
        if self.mode == "random":
            mask = self._random_mask(history_len, num_sensors, generator)
        elif self.mode == "low_rate":
            mask = self._low_rate_mask(history_len, num_sensors, generator)
        elif self.mode == "block_offline":
            mask = torch.ones(history_len, num_sensors, dtype=torch.float32)
            mask = self._apply_block_offline(mask, generator)
        elif self.mode == "mixed":
            mask = self._low_rate_mask(history_len, num_sensors, generator)
            mask = mask * self._random_mask(history_len, num_sensors, generator)
            mask = self._apply_block_offline(mask, generator)
        else:
            raise ValueError(f"Unknown async mask mode: {self.mode}")

        return self._ensure_observed(mask, generator)

    def _random_mask(self, history_len: int, num_sensors: int, generator: torch.Generator) -> Tensor:
        return (
            torch.rand(history_len, num_sensors, generator=generator) < self.random_keep_prob
        ).float()

    def _low_rate_mask(self, history_len: int, num_sensors: int, generator: torch.Generator) -> Tensor:
        mask = torch.zeros(history_len, num_sensors, dtype=torch.float32)
        periods = torch.randint(
            self.min_period,
            self.max_period + 1,
            (num_sensors,),
            generator=generator,
        )
        for sensor_idx, period in enumerate(periods.tolist()):
            offset = int(torch.randint(0, period, (1,), generator=generator).item())
            mask[offset::period, sensor_idx] = 1.0
        return mask

    def _apply_block_offline(self, mask: Tensor, generator: torch.Generator) -> Tensor:
        history_len, num_sensors = mask.shape
        max_block = max(1, int(history_len * self.max_block_fraction))
        for sensor_idx in range(num_sensors):
            if torch.rand((), generator=generator).item() >= self.block_prob:
                continue
            block_len = int(torch.randint(1, max_block + 1, (1,), generator=generator).item())
            max_start = max(1, history_len - block_len + 1)
            start = int(torch.randint(0, max_start, (1,), generator=generator).item())
            mask[start : start + block_len, sensor_idx] = 0.0
        return mask

    @staticmethod
    def _ensure_observed(mask: Tensor, generator: torch.Generator) -> Tensor:
        history_len, num_sensors = mask.shape
        for sensor_idx in range(num_sensors):
            if mask[:, sensor_idx].sum() == 0:
                row = int(torch.randint(0, history_len, (1,), generator=generator).item())
                mask[row, sensor_idx] = 1.0
        if mask.sum() == 0:
            mask[0, 0] = 1.0
        return mask
