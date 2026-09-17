"""Unified dataset-profile pilot matrix runner.

Expands tracked pilot matrices into ordered scientific keys and executes them
under preregistered rules: roots are resolved through
:func:`resolve_runtime_paths`, manifests use result-root-relative artifact
paths, resume accepts only manifests whose identity and artifact hashes match,
single-key failures do not stop the batch, and the baseline-first gate is
recomputed from canonical matrix evidence. Run levels remain separate: smoke
reports never write pilot run manifests or test metrics.
"""

import hashlib
import importlib
import inspect
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import yaml

from kaf_profiti.experiments.evaluator import (
    artifact_sha256,
    metrics_from_prediction_payload,
    write_prediction_artifact,
)
from kaf_profiti.experiments.datasets import create_protocol_datasets
from kaf_profiti.experiments.masks import (
    TimelineMaskConfig,
    TimelineMaskedWindowDataset,
    generate_or_load_timeline_masks,
)
from kaf_profiti.industrial.batch import IndustrialCollator

PROFILE_DATASETS = {
    "fd004": "cmapss_fd004",
    "metropt3": "metropt3_chrono_502030_v2",
}

# Global gradient-clipping norm for every pilot training step. This restores
# the repository's established training recipe — ``run_experiment.py``,
# ``train_metropt_kaf_profiti.py``, and ``train_cmapss_kaf_profiti.py`` all
# clip at 1.0 — which the first pilot runner draft omitted. Applied uniformly
# to every model (baselines and ours); optimizer lr/budget are unchanged.
# Local 2026-09-17 diagnostics: without clipping the adapted Euler-expansion
# ODE-RNN diverges (grad-norm peaks ~6.6e7, valid MAE 9.30 vs persistence
# floor 0.9159); with clip=1.0 and the frozen lr=1e-3 it reaches valid MAE
# 0.747 within 6 epochs.
GRAD_CLIP_NORM = 1.0

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_CODE_FINGERPRINT_CACHE: Dict[str, tuple] = {}


def _validate_profile(profile_id: str) -> str:
    profile_id = str(profile_id)
    if profile_id not in PROFILE_DATASETS or not _PROFILE_NAME_RE.fullmatch(profile_id):
        raise ValueError(f"unknown or invalid pilot profile: {profile_id!r}")
    return profile_id


def _code_fingerprint(project_root: Path) -> str:
    """Tree hash over Python sources in code/ and compare_code/.

    Probability baselines import from compare_code, so both trees must be
    covered; a change in either invalidates resume for completed runs.
    """
    files = []
    for root_name in ("code", "compare_code"):
        root = Path(project_root) / root_name
        if root.exists():
            files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    files = sorted(files)
    signature = tuple(
        (path.relative_to(project_root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
    )
    cache_key = str(Path(project_root).resolve())
    cached = _CODE_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(project_root).as_posix()}:{_sha256_file(path)};".encode("utf-8"))
    fingerprint = digest.hexdigest()
    _CODE_FINGERPRINT_CACHE[cache_key] = (signature, fingerprint)
    return fingerprint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PilotMatrix:
    """One tracked pilot matrix YAML with its content hash."""

    path: Path
    track: str  # "point" | "probabilistic"
    matrix_id: str
    raw: dict
    sha256: str

    @property
    def name(self) -> str:
        return self.track + "_matrix"


def load_matrix(path) -> PilotMatrix:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    matrix_id = str(raw.get("matrix_id", ""))
    if matrix_id.endswith("_point"):
        track = "point"
    elif matrix_id.endswith("_probabilistic"):
        track = "probabilistic"
    else:
        raise ValueError(f"Cannot infer pilot track from matrix_id: {matrix_id!r}")
    return PilotMatrix(
        path=path, track=track, matrix_id=matrix_id, raw=raw, sha256=_sha256_file(path)
    )


@dataclass(frozen=True)
class PilotRunSpec:
    """One scientific pilot run: dataset x model x condition x seed."""

    key: str
    track: str
    matrix_name: str
    dataset: str
    model_id: str
    head_type: str
    family: str
    condition_id: str
    missing_mode: str
    target_missing_rate: float
    seed: int
    split_seed: int
    mask_seed: int
    history_len: int
    pred_len: int
    stride: int
    epochs: int
    batch_size: int
    hidden_dim: int
    interval_level: float = 0.95
    nsamples: int = 20

    @property
    def model_label(self) -> str:
        return f"{self.model_id}|{self.head_type}" if self.head_type else self.model_id


SMOKE_GROUPS = {
    "point_baselines": ("point", "baseline"),
    "probabilistic_baselines": ("probabilistic", "baseline"),
    "ours": (None, "ours"),
}


def build_model(spec: PilotRunSpec, num_sensors: int, context_dim: int, options: Dict, device: str = "cpu"):
    """Instantiate the registered model for ``spec`` at protocol dimensions."""

    hidden_dim = spec.hidden_dim
    if spec.track == "point":
        if spec.model_id == "kst_light":
            kst_light = importlib.import_module("kaf_profiti.models.lightweight_head")
            return kst_light.KSTLight(
                num_sensors=num_sensors,
                context_dim=context_dim,
                pred_len=spec.pred_len,
                hidden_dim=hidden_dim,
                head_type=spec.head_type,
                te_dim=int(options.get("te_dim", 5)),
                kernel_count=int(options.get("kernel_count", 4)),
                n_layers=int(options.get("n_layers", 2)),
                n_heads=int(options.get("n_heads", 2)),
                preconv_dim=int(options.get("preconv_dim", 8)),
                patch_lens=tuple(options.get("patch_lens", (12, 24, 48))),
            ).to(device)
        point = importlib.import_module("kaf_profiti.baselines.point")
        return point.create_point_baseline(
            spec.model_id,
            num_sensors=num_sensors,
            context_dim=context_dim,
            pred_len=spec.pred_len,
            hidden_dim=hidden_dim,
            head_type=spec.head_type,
        ).to(device)
    probabilistic = importlib.import_module("kaf_profiti.baselines.probabilistic")
    kwargs = dict(options) if spec.model_id == "kst_probflow" else {}
    model = probabilistic.create_probabilistic_baseline(
        spec.model_id,
        num_sensors=num_sensors,
        context_dim=context_dim,
        pred_len=spec.pred_len,
        hidden_dim=hidden_dim,
        **kwargs,
    )
    return model.to(device)


class RealProtocolProvider:
    """Shared FD004 protocol: split datasets + timeline masks + batches.

    The split, normalization, and per-split mask bundles are shared artifacts:
    every model on the same condition resolves the identical mask bundle path
    and content SHA, which is the fairness fingerprint recorded per run.
    """

    def __init__(
        self,
        data_root: Path,
        result_root: Path,
        dataset: str,
        history_len: int,
        pred_len: int,
        stride: int,
        mechanism: str,
        requested_rate: float,
        mask_seed: int,
        split_seed: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        pilot_root: str = "pilot/fd004",
    ):
        from kaf_profiti.experiments.runtime_paths import resolve_runtime_paths

        self.pilot_root = str(pilot_root)
        paths = resolve_runtime_paths(str(data_root), str(result_root), {})
        self.data_root = paths.data_root
        self.result_root = paths.output_root
        bundle = create_protocol_datasets(
            dataset,
            self.data_root,
            seed=mask_seed,
            history_len=history_len,
            pred_len=pred_len,
            stride=stride,
            async_mode="none",
            split_seed=split_seed,
        )
        self.bundle = bundle
        self.split_sha256 = bundle.split_info["split_sha256"]
        self.normalization_sha256 = bundle.split_info["normalization"]["sha256"]
        protocol_dir = self.result_root / self.pilot_root / "protocol" / "masks" / dataset
        self.mask_sha: Dict[str, str] = {}
        self.realized_rate: Dict[str, float] = {}
        self._split_datasets: Dict[str, object] = {}
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        for split in ("train", "valid", "test"):
            window_dataset = getattr(bundle, split)
            lengths = {int(unit): len(group) for unit, group in window_dataset._units.items()}
            config = TimelineMaskConfig(
                dataset=dataset,
                split=split,
                mechanism=mechanism,
                requested_rate=requested_rate,
                mask_seed=mask_seed,
                source_split_sha256=self.split_sha256,
            )
            mask_bundle = generate_or_load_timeline_masks(
                protocol_dir
                / f"v{config.schema_version}_{split}_{mechanism}_{requested_rate:.2f}_seed{mask_seed}.npz",
                config,
                lengths,
                bundle.num_sensors,
            )
            self.mask_sha[split] = mask_bundle.content_sha256
            self.realized_rate[split] = float(mask_bundle.realized_rate)
            self._split_datasets[split] = TimelineMaskedWindowDataset(window_dataset, mask_bundle)
        self.fingerprint = {
            "dataset": dataset,
            "mechanism": mechanism,
            "requested_rate": requested_rate,
            "mask_seed": mask_seed,
            "split_seed": split_seed,
            "history_len": history_len,
            "pred_len": pred_len,
            "stride": stride,
            "split_sha256": self.split_sha256,
            "normalization_sha256": self.normalization_sha256,
            "mask_sha": dict(self.mask_sha),
            "realized_rate": dict(self.realized_rate),
        }
        split_identity = bundle.split_info.get("split_identity", {})
        for extra_key in (
            "raw_data_sha256",
            "partition_sha256",
            "timeline_sha256",
            "window_catalog_sha256",
            "time_scale_sha256",
            "target_schema_sha256",
            "evaluator",
        ):
            if extra_key in split_identity:
                self.fingerprint[extra_key] = split_identity[extra_key]
            elif extra_key in bundle.split_info:
                self.fingerprint[extra_key] = bundle.split_info[extra_key]

    num_sensors = property(lambda self: self.bundle.num_sensors)
    context_dim = property(lambda self: self.bundle.context_dim)

    @property
    def model_options(self) -> Dict:
        return {"te_dim": 5, "kernel_count": 4, "n_layers": 2, "n_heads": 2,
                "preconv_dim": 8, "patch_lens": (12, 24, 48), "copula_rank": 32}

    def protocol_fingerprint(self) -> Dict:
        return dict(self.fingerprint)

    def datasets(self) -> Dict[str, object]:
        return dict(self._split_datasets)

    def batches(self, batch_size: int) -> Dict[str, object]:
        """One train/valid/test batch each (feature-only smoke usage)."""

        from torch.utils.data import DataLoader

        collator = IndustrialCollator()
        out = {}
        for split, dataset in self._split_datasets.items():
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collator
            )
            out[split] = next(iter(loader))
            del loader
        return out

    def loaders(self, batch_size: int, generator_seed: Optional[int] = None) -> Dict[str, object]:
        from torch.utils.data import DataLoader

        collator = IndustrialCollator()
        train_generator = None
        if generator_seed is not None:
            train_generator = torch.Generator().manual_seed(int(generator_seed))
        return {
            split: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                generator=train_generator if split == "train" else None,
                collate_fn=collator,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )
            for split, dataset in self._split_datasets.items()
        }


def _call_provider_factory(
    factory: Callable,
    spec: PilotRunSpec,
    data_root: Path,
    result_root: Path,
    pilot_root: str,
):
    """Call current or legacy provider factories during the API transition."""

    parameters = inspect.signature(factory).parameters.values()
    if any(
        parameter.name == "pilot_root" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        return factory(spec, data_root, result_root, pilot_root=pilot_root)
    return factory(spec, data_root, result_root)


def _loaders(provider, batch_size: int, seed: int):
    """Request seeded loaders while retaining compatibility with test doubles."""

    if "generator_seed" in inspect.signature(provider.loaders).parameters:
        return provider.loaders(batch_size, generator_seed=seed)
    return provider.loaders(batch_size)


def _build_provider(
    spec: PilotRunSpec, data_root: Path, result_root: Path, pilot_root: str = "pilot/fd004"
) -> RealProtocolProvider:
    """Full-run provider: the spec's own condition drives the mask protocol."""

    return RealProtocolProvider(
        data_root=data_root,
        result_root=result_root,
        dataset=spec.dataset,
        history_len=spec.history_len,
        pred_len=spec.pred_len,
        stride=spec.stride,
        mechanism=spec.missing_mode,
        requested_rate=spec.target_missing_rate,
        mask_seed=spec.mask_seed,
        split_seed=spec.split_seed,
        pilot_root=pilot_root,
    )


def _build_smoke_provider(
    spec: PilotRunSpec, data_root: Path, result_root: Path, pilot_root: str = "pilot/fd004"
) -> RealProtocolProvider:
    """Build smoke data from the selected matrix condition."""

    return _build_provider(spec, data_root, result_root, pilot_root=pilot_root)


# ---------------------------------------------------------------------------
# Default full-run trainer (exercised on the execution machine in P04)
# ---------------------------------------------------------------------------


def _batch_to_device(batch, device):
    """Move every tensor field of an ``IndustrialBatch`` to ``device``."""

    if str(device) == "cpu":
        return batch
    moved = {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.__dict__.items()
    }
    return type(batch)(**moved)


def _train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total += loss.detach().item()
        count += 1
    return total / max(count, 1)


def _interval_bounds(model, batch, samples, interval_level: float):
    """Return the requested central interval from model samples."""
    alpha = (1.0 - float(interval_level)) / 2.0
    return torch.quantile(samples, alpha, dim=1), torch.quantile(samples, 1.0 - alpha, dim=1)


def _synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _provider_evaluation_metadata(provider, num_sensors: int):
    split_info = getattr(getattr(provider, "bundle", None), "split_info", {})
    target_schema = split_info.get("target_schema", {})
    target_columns = list(
        target_schema.get(
            "continuous_columns", [f"sensor_{index}" for index in range(num_sensors)]
        )
    )
    normalization = split_info.get("normalization")
    canonical_normalization = None
    if isinstance(normalization, dict):
        mean = normalization.get("mean", normalization.get("sensor_mean"))
        std = normalization.get("std", normalization.get("sensor_std"))
        if mean is not None and std is not None:
            canonical_normalization = {
                "source_split": normalization.get("source_split", "train"),
                "mean": [float(value) for value in mean],
                "std": [float(value) for value in std],
                "sha256": normalization.get("sha256"),
            }
    return target_columns, canonical_normalization


def _crps_rows(target, samples, mask):
    """Sample CRPS sums per window, using the same sample tensor as intervals."""

    target_valid = target.isfinite()
    sample_valid = samples.isfinite()
    term_valid = sample_valid & target_valid.unsqueeze(1)
    term_diff = (samples - target.unsqueeze(1)).abs()
    term1 = torch.where(term_valid, term_diff, torch.zeros_like(term_diff)).sum(dim=1)
    sample_count = sample_valid.sum(dim=1)
    term1 = term1 / sample_count.clamp_min(1)

    # For sorted samples x_(i), sum_ij |x_i-x_j| equals
    # 2 * sum_i (2*i-n-1) * x_(i). This avoids materializing the
    # [B, S, S, Q] pairwise tensor for formal nsamples=100 runs.
    sortable = torch.where(
        sample_valid, samples, torch.full_like(samples, float("inf"))
    )
    ordered = sortable.sort(dim=1).values
    ranks = torch.arange(
        1, samples.shape[1] + 1, device=samples.device, dtype=samples.dtype
    ).view(1, -1, 1)
    valid_rank = ranks <= sample_count.unsqueeze(1)
    ordered = torch.where(valid_rank, ordered, torch.zeros_like(ordered))
    coefficients = 2.0 * ranks - sample_count.unsqueeze(1).to(samples.dtype) - 1.0
    pairwise = 2.0 * (ordered * coefficients).sum(dim=1)
    pairwise = pairwise / sample_count.clamp_min(1).to(samples.dtype).pow(2)
    valid = (mask > 0) & target_valid
    contributions = torch.where(
        valid, term1 - 0.5 * pairwise, torch.zeros_like(term1)
    )
    return contributions.sum(dim=-1)


def _nll_rows(model, batch):
    if hasattr(model, "batch_nll_rows"):
        return model.batch_nll_rows(batch)
    mean, scale = model.gaussian_params(batch)
    log_prob = (
        -0.5 * ((batch.Y_q - mean) / scale).pow(2)
        - torch.log(scale)
        - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=scale.device))
    )
    return -((log_prob * batch.M_q).sum(dim=(1, 2)))


def _inference_timing_repeats(model, loader, device, track, nsamples, repeats=3):
    """Time history-only inference without computing or reading test metrics."""

    timings = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(repeats)):
            _synchronize(device)
            start = time.perf_counter()
            for batch in loader:
                batch = _batch_to_device(batch, device)
                model.predict_point(batch)
                if track == "probabilistic":
                    model.sample_flat(batch, nsamples=nsamples)
            _synchronize(device)
            timings.append(time.perf_counter() - start)
    return timings


def _valid_score(model, loader, device, track, nsamples: int = 20):
    """Point -> masked MAE; probabilistic -> sample-based CRPS micro-mean."""

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            if track == "point":
                errors = (model.predict_point(batch) - batch.y_flat).abs() * batch.mq_flat
                total += float(errors.sum())
                count += float(batch.mq_flat.sum())
            else:
                samples = model.sample_flat(batch, nsamples=nsamples)
                crps_rows = _crps_rows(batch.y_flat, samples, batch.mq_flat)
                total += float(crps_rows.sum())
                count += float(batch.mq_flat.sum())
    return total / max(count, 1.0)


def _test_prediction_artifact(
    model,
    loader,
    device,
    spec: PilotRunSpec,
    provider,
) -> tuple:
    """Evaluate test exactly once and return metrics plus a replayable payload."""

    target_columns, normalization = _provider_evaluation_metadata(
        provider, provider.num_sensors
    )
    payload = {
        "schema_version": 1,
        "track": spec.track,
        "window_id": [],
        "target": [],
        "prediction": [],
        "mask": [],
        "query_channel_ids": None,
        "target_columns": target_columns,
        "normalization": normalization,
        "interval_level": spec.interval_level,
        "nsamples": spec.nsamples if spec.track == "probabilistic" else None,
    }
    if spec.track == "probabilistic":
        payload.update(
            {
                "lower": [],
                "upper": [],
                "nll_sum_per_window": [],
                "crps_sum_per_window": [],
                "score_count_per_window": [],
            }
        )
    model.eval()
    offset = 0
    _synchronize(device)
    evaluation_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            prediction = model.predict_point(batch)
            batch_size = int(batch.y_flat.shape[0])
            window_ids = getattr(batch, "window_id", None)
            if window_ids is None:
                window_ids = [
                    hashlib.sha256(
                        f"{spec.key}|test|{offset + index}".encode("utf-8")
                    ).hexdigest()
                    for index in range(batch_size)
                ]
            payload["window_id"].extend(str(value) for value in window_ids)
            payload["target"].extend(batch.y_flat.detach().cpu().tolist())
            payload["prediction"].extend(prediction.detach().cpu().tolist())
            payload["mask"].extend(batch.mq_flat.detach().cpu().tolist())
            if payload["query_channel_ids"] is None:
                payload["query_channel_ids"] = batch.query_channel_ids.detach().cpu().tolist()
            if spec.track == "probabilistic":
                samples = model.sample_flat(batch, nsamples=spec.nsamples)
                nll_rows = _nll_rows(model, batch)
                crps_rows = _crps_rows(batch.y_flat, samples, batch.mq_flat)
                lower, upper = _interval_bounds(
                    model, batch, samples, spec.interval_level
                )
                score_counts = (
                    (batch.mq_flat > 0) & batch.y_flat.isfinite() & prediction.isfinite()
                ).sum(dim=-1)
                payload["lower"].extend(lower.detach().cpu().tolist())
                payload["upper"].extend(upper.detach().cpu().tolist())
                payload["nll_sum_per_window"].extend(nll_rows.detach().cpu().tolist())
                payload["crps_sum_per_window"].extend(crps_rows.detach().cpu().tolist())
                payload["score_count_per_window"].extend(score_counts.detach().cpu().tolist())
            offset += batch_size
    _synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    payload["timing"] = {
        "evaluation_seconds": evaluation_seconds,
        "inference_seconds": _inference_timing_repeats(
            model,
            loader,
            device,
            spec.track,
            spec.nsamples,
            repeats=3,
        ),
    }
    return metrics_from_prediction_payload(payload), payload


def pilot_train_and_evaluate(model, loaders, spec: PilotRunSpec, provider, device: str = "cpu") -> Dict[str, object]:
    """Default full pilot trainer: validation-selected checkpoint, one test.

    All tensors and the model live on ``device``; the optimizer is created once
    so AdamW moments persist across epochs.
    """

    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_state, best_score = None, None
    train_start = time.perf_counter()
    for epoch in range(1, spec.epochs + 1):
        train_loss = _train_one_epoch(model, loaders["train"], optimizer, device)
        score = _valid_score(
            model, loaders["valid"], device, spec.track, nsamples=spec.nsamples
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_score": score})
        if best_score is None or score < best_score:
            best_score = score
            best_state = {
                name: value.detach().to("cpu").clone()
                for name, value in model.state_dict().items()
            }
    train_time = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    _, predictions = _test_prediction_artifact(
        model, loaders["test"], device, spec, provider
    )
    predictions["timing"]["train_seconds"] = [train_time]
    predictions["parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    metrics = metrics_from_prediction_payload(predictions)
    checkpoint_bytes = _serialize_state(model.state_dict())
    return {
        "history": history,
        "metrics": metrics,
        "checkpoint_bytes": checkpoint_bytes,
        "predictions": predictions,
        "train_time_sec": train_time,
        "checkpoint_selection": "best_valid",
        "valid_selection_score": best_score,
        "device": str(device),
    }


def _serialize_state(state_dict: Dict) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def pilot_sanity_train(
    model,
    loaders,
    spec: PilotRunSpec,
    provider,
    device: str = "cpu",
    epochs: int = 5,
) -> Dict[str, object]:
    """CH34-S03-T02: validation-only learnability sanity (never touches test).

    Trains a fixed number of epochs on the train split, scores every epoch on
    the validation split, and reports the three gate flags — ``finite``,
    ``updated``, ``validation_improved`` — plus an init-model baseline. The
    test loader is deliberately ignored and no prediction artifact is produced,
    so this outcome can never be mistaken for a complete pilot run.
    """

    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    before = [value.detach().clone() for value in model.parameters()]
    init_valid_mae = _valid_score(
        model, loaders["valid"], device, spec.track, nsamples=spec.nsamples
    )
    history = [{"epoch": 0, "valid_score": init_valid_mae}]
    best_valid_mae = init_valid_mae
    best_epoch = 0
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, loaders["train"], optimizer, device)
        valid_mae = _valid_score(
            model, loaders["valid"], device, spec.track, nsamples=spec.nsamples
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "valid_score": valid_mae}
        )
        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_epoch = epoch
    train_time = time.perf_counter() - train_start
    finite = all(
        math.isfinite(entry.get("valid_score", float("nan")))
        and ("train_loss" not in entry or math.isfinite(entry["train_loss"]))
        for entry in history
    )
    updated = any(
        not torch.equal(before_value, current_value.detach())
        for before_value, current_value in zip(before, model.parameters())
    )
    return {
        "history": history,
        "init_valid_mae": init_valid_mae,
        "best_valid_mae": best_valid_mae,
        "best_epoch": best_epoch,
        "finite": bool(finite),
        "updated": bool(updated),
        "validation_improved": bool(best_valid_mae < init_valid_mae),
        "train_time_sec": train_time,
        "epochs_run": epochs,
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Smoke execution (CH2.5-P03-T02/T03/T04)
# ---------------------------------------------------------------------------


def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    import resource
    import sys

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def _perturb_masked_targets(batch):
    """Copy of ``batch`` whose targets change only where ``mq_flat`` is zero."""

    generator = torch.Generator().manual_seed(777)
    garbage = torch.randn(batch.y_flat.shape, generator=generator)
    y_flat = torch.where(batch.mq_flat > 0, batch.y_flat, garbage)
    y_q = y_flat.reshape(batch.Y_q.shape)
    return type(batch)(**{**batch.__dict__, "Y_q": y_q, "y_flat": y_flat})


def _smoke_point_model(model, batches) -> Dict[str, object]:
    from kaf_profiti.experiments.model_api import assert_history_only

    train_batch, valid_batch, test_batch = batches["train"], batches["valid"], batches["test"]
    before = [value.detach().clone() for value in model.parameters()]
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = model.loss(train_batch)
    loss_finite = bool(torch.isfinite(loss))
    loss.backward()
    optimizer.step()
    batch_time = time.perf_counter() - start
    params_changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, list(model.parameters()))
    )
    model.eval()
    with torch.no_grad():
        valid_finite = bool(torch.isfinite(model.predict_point(valid_batch)).all())
        test_finite = bool(torch.isfinite(model.predict_point(test_batch)).all())
    assert_history_only(model, test_batch)
    return {
        "loss_finite": loss_finite,
        "params_changed": params_changed,
        "valid_finite": valid_finite,
        "feature_only_inference_finite": test_finite,
        "history_only": True,
        "test_metric_count": 0,
        "batch_time_sec": batch_time,
        "peak_memory_mb": _peak_memory_mb(),
    }


def _smoke_probabilistic_model(model, batches) -> Dict[str, object]:
    train_batch, valid_batch = batches["train"], batches["valid"]
    before = [value.detach().clone() for value in model.parameters()]
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = model.loss(train_batch)
    loss.backward()
    optimizer.step()
    batch_time = time.perf_counter() - start
    params_changed = any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, list(model.parameters()))
    )

    model.eval()
    with torch.no_grad():
        nll = model.batch_nll(valid_batch)
        nll_finite = bool(torch.isfinite(nll))
        invariant = torch.allclose(
            nll, model.batch_nll(_perturb_masked_targets(valid_batch)), atol=1e-6
        )
        generator = torch.Generator().manual_seed(11)
        samples = model.sample_flat(valid_batch, nsamples=16, generator=generator)
        query_count = valid_batch.y_flat.shape[1]
        flatten_ok = samples.shape == (valid_batch.y_flat.shape[0], 16, query_count)
        samples_finite = bool(torch.isfinite(samples).all())
        zero_mask = valid_batch.mq_flat == 0
        if bool(zero_mask.any()):
            expanded = zero_mask.unsqueeze(1).expand(-1, samples.shape[1], -1)
            masked_zero = bool((samples[expanded] == 0).all().item())
        else:
            masked_zero = True
        lower, upper = model.interval95_flat(valid_batch)
        interval_finite = bool(torch.isfinite(lower).all() and torch.isfinite(upper).all())
        upper_ok = bool((upper[valid_batch.mq_flat > 0] >= lower[valid_batch.mq_flat > 0]).all())

    distribution_grads = any(
        ("head" in name or "flow" in name) and parameter.grad is not None
        and bool(parameter.grad.abs().sum() > 0)
        for name, parameter in model.named_parameters()
    )

    checkpoint_ok = _checkpoint_round_trip(model, valid_batch)

    checks = {
        "loss_finite": bool(torch.isfinite(loss)),
        "params_changed": params_changed,
        "nll_finite": nll_finite,
        "denominator_mask_invariant": bool(invariant),
        "samples_finite": samples_finite,
        "mask_applied": masked_zero,
        "flatten_order_consistent": bool(flatten_ok),
        "interval_finite": interval_finite,
        "interval_ordered": upper_ok,
        "distribution_grads": distribution_grads,
        "checkpoint_round_trip": checkpoint_ok,
        "test_metric_count": 0,
        "batch_time_sec": batch_time,
        "peak_memory_mb": _peak_memory_mb(),
    }
    if getattr(model, "gaussian_kind", "diagonal") == "flow":
        checks["distribution_consistent"] = _flow_distribution_consistent(model, valid_batch)
        checks["picp_mpiw_source_deterministic"] = _interval_deterministic(model, valid_batch)
    return checks


def _checkpoint_round_trip(model, batch) -> bool:
    import copy

    saved = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        generator = torch.Generator().manual_seed(5)
        reference = model.sample_flat(batch, nsamples=4, generator=generator)
    fresh = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in fresh.parameters():
            parameter.data.add_(0.05)
        fresh.load_state_dict(saved)
        generator = torch.Generator().manual_seed(5)
        replay = fresh.sample_flat(batch, nsamples=4, generator=generator)
    return bool(torch.allclose(reference, replay, atol=1e-6))


def _flow_distribution_consistent(model, batch) -> bool:
    """NLL and samples must be driven by the identical conditioning state."""

    with torch.no_grad():
        first = model.flow_hidden(batch)
        second = model.flow_hidden(batch)
    return bool(torch.equal(first, second))


def _interval_deterministic(model, batch) -> bool:
    with torch.no_grad():
        lower_a, upper_a = model.interval95_flat(batch)
        lower_b, upper_b = model.interval95_flat(batch)
    return bool(
        torch.equal(lower_a, lower_b)
        and torch.equal(upper_a, upper_b)
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class PilotRunner:
    """Expand, gate, execute, and resume one or more tracked pilot matrices."""

    def __init__(
        self,
        matrices: List[PilotMatrix],
        result_root,
        data_root=None,
        device: str = "cpu",
        num_workers: int | str = 0,
        profile: str = "fd004",
    ):
        from kaf_profiti.experiments.runtime_paths import resolve_runtime_paths

        paths = resolve_runtime_paths(data_root, str(result_root), {})
        self.result_root = paths.output_root
        self.data_root = paths.data_root
        self.profile = _validate_profile(profile)
        self.pilot_root = str(Path("pilot") / self.profile)
        expected_dataset = PROFILE_DATASETS[self.profile]
        mismatched = [matrix for matrix in matrices if matrix.raw.get("dataset") != expected_dataset]
        if mismatched:
            raise ValueError(
                f"profile {self.profile!r} expects dataset {expected_dataset!r}, "
                f"got {[matrix.raw.get('dataset') for matrix in mismatched]!r}"
            )
        self.project_root = paths.project_root
        self.code_fingerprint = _code_fingerprint(paths.project_root)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if num_workers == "auto":
            num_workers = 4 if torch.cuda.is_available() else 0
        self.num_workers = int(num_workers)
        self.matrices = list(matrices)

    # -- expansion ----------------------------------------------------------

    def expand(self) -> List[PilotRunSpec]:
        specs: List[PilotRunSpec] = []
        for matrix in self.matrices:
            raw = matrix.raw
            models = sorted(
                enumerate(raw["models"]), key=lambda pair: (int(pair[1].get("priority", 0)), pair[0])
            )
            for _, model_entry in models:
                for condition in raw["conditions"]:
                    key = "|".join(
                        [
                            raw["dataset"],
                            matrix.track,
                            str(model_entry["model_id"]),
                            str(model_entry.get("head_type", "")),
                            str(condition["condition_id"]),
                            str(int(raw["seed"])),
                        ]
                    )
                    specs.append(
                        PilotRunSpec(
                            key=key,
                            track=matrix.track,
                            matrix_name=matrix.name,
                            dataset=str(raw["dataset"]),
                            model_id=str(model_entry["model_id"]),
                            head_type=str(model_entry.get("head_type", "")),
                            family=str(model_entry["family"]),
                            condition_id=str(condition["condition_id"]),
                            missing_mode=str(condition["missing_mode"]),
                            target_missing_rate=float(condition["target_missing_rate"]),
                            seed=int(raw["seed"]),
                            split_seed=int(raw["split_seed"]),
                            mask_seed=int(raw["mask_seed"]),
                            history_len=int(raw["history_len"]),
                            pred_len=int(raw["pred_len"]),
                            stride=int(raw["stride"]),
                            epochs=int(raw["epochs"]),
                            batch_size=int(raw["batch_size"]),
                            hidden_dim=int(raw["hidden_dim"]),
                            interval_level=float(raw.get("interval_level", 0.95)),
                            nsamples=int(raw.get("nsamples", 20)),
                        )
                    )
        return specs

    # -- shared artifacts / manifests ---------------------------------------

    def _shared_artifacts(self) -> Dict[str, str]:
        return {matrix.name: matrix.sha256 for matrix in self.matrices}

    def _runs_dir(self) -> Path:
        return self.result_root / self.pilot_root / "runs"

    def _manifest_path(self, key: str) -> Path:
        return self._runs_dir() / key / "manifest.json"

    @staticmethod
    def _manifest_identity(spec: PilotRunSpec) -> Dict[str, object]:
        """Canonical scientific identity required for resume and gating."""

        return {
            "key": spec.key,
            "run_level": "pilot",
            "matrix": spec.matrix_name,
            "track": spec.track,
            "model_id": spec.model_id,
            "head_type": spec.head_type,
            "family": spec.family,
            "condition_id": spec.condition_id,
            "missing_mode": spec.missing_mode,
            "target_missing_rate": spec.target_missing_rate,
            "seed": spec.seed,
            "split_seed": spec.split_seed,
            "mask_seed": spec.mask_seed,
            "dataset": spec.dataset,
            "history_len": spec.history_len,
            "pred_len": spec.pred_len,
            "stride": spec.stride,
            "epochs": spec.epochs,
            "batch_size": spec.batch_size,
            "hidden_dim": spec.hidden_dim,
            "interval_level": spec.interval_level,
            "nsamples": spec.nsamples,
        }

    def _provider_cache_key(self, spec: PilotRunSpec) -> tuple:
        return (
            str(self.data_root.resolve()),
            str(self.result_root.resolve()),
            self.pilot_root,
            spec.dataset,
            spec.condition_id,
            spec.history_len,
            spec.pred_len,
            spec.stride,
            spec.missing_mode,
            spec.target_missing_rate,
            spec.mask_seed,
            spec.split_seed,
        )

    def _provider(
        self,
        spec: PilotRunSpec,
        factory: Callable,
        cache: Dict[tuple, object],
    ):
        cache_key = self._provider_cache_key(spec)
        if cache_key not in cache:
            provider = _call_provider_factory(
                factory, spec, self.data_root, self.result_root, self.pilot_root
            )
            if hasattr(provider, "num_workers"):
                provider.num_workers = self.num_workers
                provider.pin_memory = str(self.device) == "cuda"
            cache[cache_key] = provider
        return cache[cache_key]

    @staticmethod
    def _protocol_fingerprint(provider) -> Optional[Dict]:
        if not hasattr(provider, "protocol_fingerprint"):
            return None
        fingerprint = provider.protocol_fingerprint()
        return dict(fingerprint) if fingerprint is not None else None

    @staticmethod
    def _validate_provider_condition(spec: PilotRunSpec, fingerprint: Optional[Dict]) -> None:
        if fingerprint is None:
            return
        mismatched = []
        if str(fingerprint.get("mechanism")) != spec.missing_mode:
            mismatched.append(
                f"mechanism {fingerprint.get('mechanism')!r} != spec {spec.missing_mode!r}"
            )
        observed_rate = fingerprint.get("requested_rate")
        if observed_rate is None or abs(float(observed_rate) - spec.target_missing_rate) > 1e-9:
            mismatched.append(
                f"requested_rate {observed_rate!r} != spec {spec.target_missing_rate!r}"
            )
        if mismatched:
            raise RuntimeError(
                f"provider condition mismatch for {spec.key}: " + "; ".join(mismatched)
            )

    def _artifact_path(self, spec: PilotRunSpec, relative: object) -> Optional[Path]:
        if not isinstance(relative, str) or not relative:
            return None
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return None
        resolved = (self.result_root / relative_path).resolve()
        run_dir = (self._runs_dir() / spec.key).resolve()
        try:
            resolved.relative_to(run_dir)
        except ValueError:
            return None
        return resolved

    def _verified_specs(
        self,
        specs: List[PilotRunSpec],
        provider_factory: Optional[Callable] = None,
        validate_protocol: bool = True,
        provider_cache: Optional[Dict[tuple, object]] = None,
    ) -> Dict[str, dict]:
        shared = self._shared_artifacts()
        verified = {}
        factory = provider_factory or _build_provider
        cache = provider_cache if provider_cache is not None else {}
        for spec in specs:
            path = self._manifest_path(spec.key)
            if not path.exists():
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(manifest, dict):
                continue
            if manifest.get("status") != "completed":
                continue
            identity = self._manifest_identity(spec)
            if any(manifest.get(name) != value for name, value in identity.items()):
                continue
            if manifest.get("run_id") != spec.key:
                continue
            if manifest.get("test_evaluation_count") != 1:
                continue
            if manifest.get("shared_artifacts") != shared:
                continue
            artifacts = manifest.get("artifacts", {})
            artifact_shas = manifest.get("artifact_sha256", {})
            if not isinstance(artifacts, dict) or not isinstance(artifact_shas, dict):
                continue
            if set(artifacts) != set(artifact_shas):
                continue
            if not {"history", "metrics", "checkpoint", "predictions"}.issubset(artifacts):
                continue
            resolved_artifacts = {
                name: self._artifact_path(spec, relative)
                for name, relative in artifacts.items()
            }
            if any(path is None or not path.is_file() for path in resolved_artifacts.values()):
                continue
            if any(
                _sha256_file(resolved_artifacts[name]) != expected_sha
                for name, expected_sha in artifact_shas.items()
            ):
                continue
            expected_checkpoint_sha = manifest.get("checkpoint_sha256")
            if not expected_checkpoint_sha:
                continue
            if artifact_shas.get("checkpoint") != expected_checkpoint_sha:
                continue
            if manifest.get("code_fingerprint") != self.code_fingerprint:
                continue
            if validate_protocol:
                provider = self._provider(spec, factory, cache)
                current_protocol = self._protocol_fingerprint(provider)
                if current_protocol is not None:
                    self._validate_provider_condition(spec, current_protocol)
                if manifest.get("protocol_sha") != current_protocol:
                    continue
            verified[spec.key] = manifest
        return verified

    # -- dry-run -------------------------------------------------------------

    def _filter_specs(
        self,
        specs: List[PilotRunSpec],
        family: Optional[str] = None,
        condition_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
    ) -> List[PilotRunSpec]:
        condition_set = set(condition_ids) if condition_ids is not None else None
        model_set = set(model_ids) if model_ids is not None else None
        unknown_conditions = (
            condition_set - {spec.condition_id for spec in specs}
            if condition_set is not None else set()
        )
        unknown_models = (
            model_set - {spec.model_id for spec in specs}
            if model_set is not None else set()
        )
        if unknown_conditions:
            raise ValueError(f"unknown condition-id(s): {sorted(unknown_conditions)}")
        if unknown_models:
            raise ValueError(f"unknown model-id(s): {sorted(unknown_models)}")
        if family is not None and family not in {"baseline", "ours"}:
            raise ValueError(f"unknown family: {family!r}")
        return [
            spec for spec in specs
            if (family is None or spec.family == family)
            and (condition_set is None or spec.condition_id in condition_set)
            and (model_set is None or spec.model_id in model_set)
        ]

    def dry_run(
        self,
        family: Optional[str] = None,
        condition_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        provider_factory: Optional[Callable] = None,
    ) -> Dict[str, object]:
        canonical_specs = self.expand()
        specs = self._filter_specs(canonical_specs, family, condition_ids, model_ids)
        verified = self._verified_specs(specs, provider_factory=provider_factory)
        return {
            "run_level": "dry-run",
            "profile": self.profile,
            "keys": [spec.key for spec in specs],
            "canonical_total": len(canonical_specs),
            "model_order": list(dict.fromkeys(spec.model_label for spec in specs)),
            "expected_total": len(specs),
            "verified_complete": len(verified),
            "expected_new": len(specs) - len(verified),
            "shared_artifacts": [
                {
                    "name": matrix.name,
                    "path": matrix.path.name,
                    "sha256": matrix.sha256,
                }
                for matrix in self.matrices
            ],
        }

    # -- gate ----------------------------------------------------------------

    def _gate_violations(
        self,
        canonical_specs: List[PilotRunSpec],
        scheduled_specs: List[PilotRunSpec],
        verified: Dict[str, dict],
    ) -> List[str]:
        """Require canonical baselines for every scheduled ours condition.

        Filtering changes what is scheduled, never what baseline evidence is
        required: the canonical matrix remains the source of gate truth.
        """
        required = []
        for ours in scheduled_specs:
            if ours.family != "ours":
                continue
            required.extend(
                spec.key
                for spec in canonical_specs
                if spec.family == "baseline"
                and spec.dataset == ours.dataset
                and spec.track == ours.track
                and spec.matrix_name == ours.matrix_name
                and spec.condition_id == ours.condition_id
                and spec.seed == ours.seed
            )
        return [key for key in dict.fromkeys(required) if key not in verified]

    # -- execution -------------------------------------------------------------

    def execute(
        self,
        trainer: Optional[Callable] = None,
        gate: bool = True,
        continue_on_error: bool = True,
        provider_factory: Optional[Callable] = None,
        family: Optional[str] = None,
        condition_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        force_rerun: bool = False,
    ) -> Dict[str, object]:
        trainer = trainer or pilot_train_and_evaluate
        factory = provider_factory or _build_provider
        canonical_specs = self.expand()
        specs = self._filter_specs(canonical_specs, family, condition_ids, model_ids)
        provider_cache: Dict[tuple, object] = {}
        verified = self._verified_specs(
            specs, provider_factory=factory, provider_cache=provider_cache
        )
        canonical_verified = self._verified_specs(
            canonical_specs, provider_factory=factory, provider_cache=provider_cache
        )
        initially_verified = len(verified)
        completed: List[str] = []
        failed: List[str] = []
        errors: Dict[str, str] = {}
        for spec in specs:
            protocol_sha = None
            if spec.key in verified and not force_rerun:
                continue
            if force_rerun:
                canonical_verified.pop(spec.key, None)
            gate_missing = self._gate_violations(
                canonical_specs, [spec], canonical_verified
            )
            if gate and spec.family == "ours" and gate_missing:
                message = (
                    "baseline-first gate blocked: incomplete baseline keys "
                    f"{gate_missing}"
                )
                failed.append(spec.key)
                errors[spec.key] = message
                self._write_manifest(spec, status="failed", error=message, outcome=None)
                if not continue_on_error:
                    raise RuntimeError(message)
                continue
            try:
                # Per-run seeding: model init and loader shuffling must honor the
                # scientific seed recorded in the manifest.
                torch.manual_seed(spec.seed)
                provider = self._provider(spec, factory, provider_cache)
                protocol_sha = self._protocol_fingerprint(provider)
                self._validate_provider_condition(spec, protocol_sha)
                run_start = time.perf_counter()
                model = build_model(
                    spec,
                    provider.num_sensors,
                    provider.context_dim,
                    provider.model_options,
                    device=self.device,
                )
                loaders = _loaders(provider, spec.batch_size, spec.seed)
                outcome = trainer(model, loaders, spec, provider, self.device)
                self._persist_outcome(spec, outcome)
                self._write_manifest(
                    spec, status="completed", error="", outcome=outcome,
                    protocol_sha=protocol_sha,
                )
                reverified = self._verified_specs(
                    [spec], provider_factory=factory, provider_cache=provider_cache
                )
                if spec.key not in reverified:
                    raise RuntimeError(
                        f"completed run failed manifest verification for {spec.key}"
                    )
                completed.append(spec.key)
                verified[spec.key] = reverified[spec.key]
                canonical_verified[spec.key] = reverified[spec.key]
                print(
                    json.dumps(
                        {
                            "event": "run",
                            "key": spec.key,
                            "status": "completed",
                            "device": str(outcome.get("device", self.device)),
                            "elapsed_sec": round(time.perf_counter() - run_start, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 — one key failing must not stop the batch
                failed.append(spec.key)
                errors[spec.key] = f"{type(error).__name__}: {error}"
                self._write_manifest(
                    spec, status="failed", error=errors[spec.key], outcome=None,
                    protocol_sha=protocol_sha,
                )
                print(
                    json.dumps(
                        {"event": "run", "key": spec.key, "status": "failed", "error": errors[spec.key]},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if not continue_on_error:
                    raise
        return {
            "completed": completed,
            "completed_count": len(completed),
            "failed": failed,
            "errors": errors,
            "skipped": 0 if force_rerun else initially_verified,
            "rerun_verified": initially_verified if force_rerun else 0,
            "profile": self.profile,
            "scheduled_count": len(specs),
        }

    def _persist_outcome(self, spec: PilotRunSpec, outcome: Dict[str, object]) -> None:
        run_dir = self._runs_dir() / spec.key
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, str] = {}
        payloads = {
            "history.json": json.dumps(outcome.get("history", []), ensure_ascii=False),
            "metrics.json": json.dumps(outcome.get("metrics", {}), ensure_ascii=False),
        }
        if outcome.get("checkpoint_bytes"):
            (run_dir / "checkpoint.pt").write_bytes(outcome["checkpoint_bytes"])
            artifacts["checkpoint"] = str((run_dir / "checkpoint.pt").relative_to(self.result_root))
        predictions = outcome.get("predictions")
        if not isinstance(predictions, dict) or not predictions:
            raise ValueError("completed pilot outcome requires a prediction artifact")
        prediction_path = run_dir / "predictions.json"
        write_prediction_artifact(prediction_path, predictions)
        artifacts["predictions"] = str(prediction_path.relative_to(self.result_root))
        for filename, payload in payloads.items():
            (run_dir / filename).write_text(payload, encoding="utf-8")
            artifacts[filename.split(".")[0]] = str((run_dir / filename).relative_to(self.result_root))

    def _write_manifest(
        self,
        spec: PilotRunSpec,
        status: str,
        error: str,
        outcome: Optional[Dict],
        protocol_sha: Optional[Dict] = None,
    ) -> None:
        run_dir = self._runs_dir() / spec.key
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, str] = {}
        if status == "completed":
            for name in ("history", "metrics", "checkpoint", "predictions"):
                candidate = run_dir / {
                    "history": "history.json",
                    "metrics": "metrics.json",
                    "checkpoint": "checkpoint.pt",
                    "predictions": "predictions.json",
                }[name]
                if candidate.exists():
                    artifacts[name] = str(candidate.relative_to(self.result_root))
        manifest = {
            **self._manifest_identity(spec),
            "run_id": spec.key,
            "status": status,
            "error": error,
            "device": str(outcome.get("device", self.device)) if outcome else self.device,
            "protocol_sha": protocol_sha,
            "shared_artifacts": self._shared_artifacts(),
            "artifacts": artifacts,
            "artifact_sha256": {
                name: artifact_sha256(self.result_root / relative)
                for name, relative in artifacts.items()
                if (self.result_root / relative).is_file()
            },
            "test_evaluation_count": 1 if status == "completed" else 0,
            "code_fingerprint": self.code_fingerprint,
            "checkpoint_sha256": (
                _sha256_file(run_dir / "checkpoint.pt")
                if status == "completed" and (run_dir / "checkpoint.pt").exists()
                else None
            ),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- smoke ---------------------------------------------------------------

    def _smoke_models(self, group: str):
        track, family = SMOKE_GROUPS[group]
        candidates: Dict[str, List[PilotRunSpec]] = {}
        for spec in self.expand():
            if family != spec.family:
                continue
            if track is not None and spec.track != track:
                continue
            candidates.setdefault(spec.model_label, []).append(spec)
        return [
            next(
                (
                    spec for spec in model_specs
                    if spec.missing_mode == "mixed"
                    and abs(spec.target_missing_rate - 0.30) <= 1e-9
                ),
                model_specs[0],
            )
            for model_specs in candidates.values()
        ]

    def smoke_expected(self, group: str) -> int:
        if group not in SMOKE_GROUPS:
            raise KeyError(f"Unknown smoke group: {group}")
        return len(self._smoke_models(group))

    def _baseline_smoke_ready(self) -> None:
        """T04 gate: both baseline smoke reports must be verified ready."""

        for group in ("point_baselines", "probabilistic_baselines"):
            expected = self.smoke_expected(group)
            if expected == 0:
                continue
            path = self.result_root / self.pilot_root / "smoke" / f"{group}_smoke.json"
            if not path.exists():
                raise RuntimeError(
                    f"baseline smoke gate failed: missing report {path.name} "
                    f"(run {group} smoke first)"
                )
            report = json.loads(path.read_text(encoding="utf-8"))
            counts = validate_smoke_report(report, expected_ready=expected)
            if counts["ready"] != expected or counts["failed"]:
                raise RuntimeError(
                    f"baseline smoke gate failed for {group}: {counts}"
                )

    def run_smoke(
        self,
        group: str,
        provider=None,
        hidden_dim: Optional[int] = None,
    ) -> Dict[str, object]:
        if group not in SMOKE_GROUPS:
            raise KeyError(f"Unknown smoke group: {group}")
        if group == "ours":
            self._baseline_smoke_ready()

        model_specs = self._smoke_models(group)
        if not model_specs:
            raise ValueError(f"smoke group {group!r} has no models in the loaded matrices")
        entries = []
        fingerprint = None
        for spec_template in model_specs:
            entry: Dict[str, object] = {
                "model_id": spec_template.model_id,
                "head_type": spec_template.head_type,
                "model_label": spec_template.model_label,
            }
            try:
                resolved = provider or _build_smoke_provider(
                    spec_template, self.data_root, self.result_root, pilot_root=self.pilot_root
                )
                spec = _with_smoke_dims(
                    spec_template,
                    hidden_dim,
                    getattr(resolved, "pred_len", spec_template.pred_len),
                )
                fingerprint = fingerprint or resolved.protocol_fingerprint()
                torch.manual_seed(spec.seed)
                model = build_model(
                    spec,
                    resolved.num_sensors,
                    resolved.context_dim,
                    resolved.model_options,
                )
                batches = (
                    resolved.batches(spec.batch_size)
                    if provider is None
                    else provider.batches()
                )
                if spec.track == "point":
                    checks = _smoke_point_model(model, batches)
                else:
                    checks = _smoke_probabilistic_model(model, batches)
                # Budget diagnostics live at entry level, not among pass/fail checks.
                for timing_key in ("batch_time_sec", "peak_memory_mb"):
                    if timing_key in checks:
                        entry[timing_key] = checks.pop(timing_key)
                entry["checks"] = checks
                entry["protocol_sha"] = resolved.protocol_fingerprint()
                entry["ok"] = all(
                    value for key, value in checks.items()
                    if isinstance(value, bool)
                )
            except Exception as error:  # noqa: BLE001 — record and continue
                entry["ok"] = False
                entry["error"] = f"{type(error).__name__}: {error}"
            entries.append(entry)

        report = {
            "group": group,
            "run_level": "smoke",
            "protocol_fingerprint": fingerprint,
            "test_metrics": None,
            "models": entries,
        }
        report_dir = self.result_root / self.pilot_root / "smoke"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{group}_smoke.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    # -- sanity train (CH34-S03-T02) -----------------------------------------

    def _sanity_dir(self) -> Path:
        """Isolation root for the learnability sanity; resume/gating never scan it."""

        return self.result_root / self.pilot_root / "sanity"

    def _naive_floor_reference(self) -> Dict[str, object]:
        """Best history-only validation floor from the S01 data gate."""

        path = self.result_root / self.pilot_root / "diagnostics" / "data_gate.json"
        if not path.is_file():
            return {"available": False}
        gate = json.loads(path.read_text(encoding="utf-8"))
        persistence = gate.get("floors", {}).get("valid", {}).get("persistence", {})
        return {
            "available": True,
            "persistence_mae_raw": persistence.get("mae"),
            "persistence_mae_std_micro": persistence.get("std_micro", {}).get("mae"),
            "learnability_gate": gate.get("learnability_gate", {}).get("result"),
            "note": (
                "data_gate floors aggregate every query position; sanity valid_mae "
                "aggregates masked query positions only (mixed@0.30, expected close)."
            ),
        }

    def run_sanity_train(
        self,
        epochs: int = 5,
        provider_factory: Optional[Callable] = None,
    ) -> Dict[str, object]:
        """CH34-S03-T02: one fixed-model validation-only learnability sanity.

        Selects the point-track ``li_tcn`` spec at ``point_mixed_030``, trains
        it for ``epochs`` epochs without ever loading the test split, and writes
        the outcome under ``sanity/`` (not ``runs/``) so resume and gating never
        see it as a complete pilot run.
        """

        factory = provider_factory or _build_provider
        specs = [
            spec
            for spec in self.expand()
            if spec.track == "point"
            and spec.model_id == "li_tcn"
            and spec.condition_id == "point_mixed_030"
        ]
        if len(specs) != 1:
            raise ValueError(
                f"sanity_train expects exactly one li_tcn@point_mixed_030 point spec, "
                f"got {len(specs)}: {[spec.key for spec in specs]}"
            )
        spec = specs[0]
        provider_cache: Dict[tuple, object] = {}
        torch.manual_seed(spec.seed)
        provider = self._provider(spec, factory, provider_cache)
        protocol_sha = self._protocol_fingerprint(provider)
        self._validate_provider_condition(spec, protocol_sha)
        model = build_model(
            spec,
            provider.num_sensors,
            provider.context_dim,
            provider.model_options,
            device=self.device,
        )
        loaders = _loaders(provider, spec.batch_size, spec.seed)
        outcome = pilot_sanity_train(
            model, loaders, spec, provider, device=self.device, epochs=epochs
        )
        naive_floor = self._naive_floor_reference()
        beat_naive = (
            outcome["best_valid_mae"] < naive_floor["persistence_mae_raw"]
            if naive_floor.get("available")
            else None
        )
        run_dir = self._sanity_dir() / spec.key
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "history.json").write_text(
            json.dumps(outcome["history"], ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            **self._manifest_identity(spec),
            "run_level": "sanity_train",
            "run_id": spec.key,
            "status": "completed",
            "error": "",
            "device": str(outcome["device"]),
            "protocol_sha": protocol_sha,
            "shared_artifacts": self._shared_artifacts(),
            "code_fingerprint": self.code_fingerprint,
            "sanity_epochs": outcome["epochs_run"],
            "finite": outcome["finite"],
            "updated": outcome["updated"],
            "validation_improved": outcome["validation_improved"],
            "init_valid_mae": outcome["init_valid_mae"],
            "best_valid_mae": outcome["best_valid_mae"],
            "best_epoch": outcome["best_epoch"],
            "naive_floor": naive_floor,
            "beat_naive": beat_naive,
            "test_evaluation_count": 0,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest


def _with_smoke_dims(spec: PilotRunSpec, hidden_dim: Optional[int], pred_len: int) -> PilotRunSpec:
    """Smoke may shrink hidden width and must honor the provider's pred_len."""

    overrides = {"pred_len": int(pred_len)}
    if hidden_dim is not None:
        overrides["hidden_dim"] = int(hidden_dim)
    return PilotRunSpec(**{**spec.__dict__, **overrides})


def validate_smoke_report(
    report: Dict[str, object],
    expected_ready: int,
    expected_failed: int = 0,
) -> Dict[str, int]:
    """Gate a smoke report: ready/failed counts and zero test metrics."""

    entries = report.get("models", [])
    ready = sum(1 for entry in entries if entry.get("ok") is True)
    failed = sum(1 for entry in entries if entry.get("ok") is not True)
    assert ready == expected_ready, f"expected ready={expected_ready}, got {ready}"
    assert failed == expected_failed, f"expected failed={expected_failed}, got {failed}"
    assert report.get("test_metrics") is None, "smoke report must not contain test metrics"
    for entry in entries:
        checks = entry.get("checks", {})
        assert checks.get("test_metric_count", 0) == 0, (
            f"smoke produced test metrics for {entry.get('model_label')}"
        )
    fingerprints = {
        json.dumps(entry.get("protocol_sha"), sort_keys=True)
        for entry in entries
        if "protocol_sha" in entry
    }
    assert len(fingerprints) <= 1, "smoked models must share one protocol fingerprint"
    return {"ready": ready, "failed": failed, "test_metrics": 0}
