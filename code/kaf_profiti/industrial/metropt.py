from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .missing import MissingMechanismSimulator


METROPT_SENSOR_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]
METROPT_CONTEXT_COLUMNS = ["COMP", "DV_eletric", "MPG"]
METROPT_FAULT_WINDOWS = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]
CSV_NAME = "MetroPT3(AirCompressor).csv"
_METROPT_FRAME_CACHE = {}


@dataclass
class MetroPTWindowSample:
    X_obs: Tensor
    T_obs: Tensor
    M_obs: Tensor
    T_q: Tensor
    Y_q: Tensor
    M_q: Tensor
    context: Tensor
    rul: float
    unit_id: int


def load_metropt_frame(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    csv_path = data_dir / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    cache_key = str(csv_path.resolve())
    if cache_key in _METROPT_FRAME_CACHE:
        return _METROPT_FRAME_CACHE[cache_key].copy()

    frame = pd.read_csv(csv_path, index_col=0, parse_dates=["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    missing = [col for col in METROPT_SENSOR_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"MetroPT CSV missing sensor columns: {missing}")
    frame["fault_label"] = 0
    for start, end in METROPT_FAULT_WINDOWS:
        mask = (frame["timestamp"] >= pd.Timestamp(start)) & (
            frame["timestamp"] <= pd.Timestamp(end)
        )
        frame.loc[mask, "fault_label"] = 1
    frame["relative_time"] = (
        frame["timestamp"] - frame["timestamp"].iloc[0]
    ).dt.total_seconds()
    _METROPT_FRAME_CACHE[cache_key] = frame
    return frame.copy()


def _split_bounds(length: int, split: str, train_ratio: float, valid_ratio: float) -> Tuple[int, int]:
    split = split.lower()
    train_end = int(length * train_ratio)
    valid_end = int(length * (train_ratio + valid_ratio))
    if split == "train":
        return 0, train_end
    if split in {"valid", "val"}:
        return train_end, valid_end
    if split == "test":
        return valid_end, length
    if split == "all":
        return 0, length
    raise ValueError("split must be one of train, valid, test, all")


def _stats_from_frame(frame: pd.DataFrame, train_ratio: float) -> Dict[str, Tensor]:
    train_end = int(len(frame) * train_ratio)
    train_frame = frame.iloc[:train_end]
    sensors = torch.tensor(train_frame[METROPT_SENSOR_COLUMNS].to_numpy(), dtype=torch.float32)
    context = torch.tensor(train_frame[METROPT_CONTEXT_COLUMNS].to_numpy(), dtype=torch.float32)
    return {
        "sensor_mean": sensors.mean(dim=0),
        "sensor_std": sensors.std(dim=0).clamp_min(1e-6),
        "context_mean": context.mean(dim=0),
        "context_std": context.std(dim=0).clamp_min(1e-6),
    }


class MetroPTWindowDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split: str = "train",
        history_len: int = 60,
        pred_len: int = 12,
        stride: int = 12,
        async_mode: str = "mixed",
        seed: int = 42,
        normalize: bool = True,
        context_mode: str = "mean",
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.async_mode = async_mode
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.context_mode = context_mode
        self.train_ratio = float(train_ratio)
        self.valid_ratio = float(valid_ratio)

        if self.history_len <= 0 or self.pred_len <= 0:
            raise ValueError("history_len and pred_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if context_mode not in {"mean", "last"}:
            raise ValueError("context_mode must be 'mean' or 'last'")
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 <= valid_ratio < 1 or train_ratio + valid_ratio >= 1:
            raise ValueError("train_ratio + valid_ratio must be less than 1")

        full_frame = load_metropt_frame(self.data_dir)
        self.stats = _stats_from_frame(full_frame, self.train_ratio) if normalize else None
        start, end = _split_bounds(len(full_frame), split, self.train_ratio, self.valid_ratio)
        self.frame = full_frame.iloc[start:end].reset_index(drop=True)
        self.global_start = start
        self.masker = MissingMechanismSimulator(mode=async_mode)
        self.windows = self._build_windows()
        if not self.windows:
            raise ValueError(
                f"No MetroPT windows for split={split}; reduce history_len or pred_len"
            )

    def _build_windows(self) -> List[int]:
        total_len = self.history_len + self.pred_len
        limit = len(self.frame) - total_len + 1
        return list(range(0, max(0, limit), self.stride))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> MetroPTWindowSample:
        start = self.windows[index]
        hist = self.frame.iloc[start : start + self.history_len]
        fut = self.frame.iloc[start + self.history_len : start + self.history_len + self.pred_len]

        sensors_hist = torch.tensor(hist[METROPT_SENSOR_COLUMNS].to_numpy(), dtype=torch.float32)
        sensors_future = torch.tensor(fut[METROPT_SENSOR_COLUMNS].to_numpy(), dtype=torch.float32)
        context_hist = torch.tensor(hist[METROPT_CONTEXT_COLUMNS].to_numpy(), dtype=torch.float32)
        if self.normalize:
            sensors_hist = (
                sensors_hist - self.stats["sensor_mean"]
            ) / self.stats["sensor_std"]
            sensors_future = (
                sensors_future - self.stats["sensor_mean"]
            ) / self.stats["sensor_std"]
            context_hist = (
                context_hist - self.stats["context_mean"]
            ) / self.stats["context_std"]

        context = context_hist.mean(dim=0) if self.context_mode == "mean" else context_hist[-1]
        mask_seed = self.seed + (self.global_start + start) * 1009
        M_obs = self.masker((self.history_len, len(METROPT_SENSOR_COLUMNS)), mask_seed)
        X_obs = sensors_hist * M_obs
        future_fault = float(fut["fault_label"].max())

        return MetroPTWindowSample(
            X_obs=X_obs,
            T_obs=torch.arange(self.history_len, dtype=torch.float32),
            M_obs=M_obs,
            T_q=torch.arange(
                self.history_len,
                self.history_len + self.pred_len,
                dtype=torch.float32,
            ),
            Y_q=sensors_future,
            M_q=torch.ones_like(sensors_future),
            context=context,
            rul=future_fault,
            unit_id=0,
        )
