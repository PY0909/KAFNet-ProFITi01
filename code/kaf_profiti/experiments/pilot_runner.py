"""Unified pilot matrix runner (CH2.5-P03-T01).

Expands the tracked FD004 pilot matrices into ordered scientific keys and
executes them under the preregistered rules: no machine-specific paths in
code (roots come from :func:`resolve_runtime_paths`), manifests store only
result-root-relative artifact paths, resume skips only runs whose manifest
and shared-artifact hashes fully match, single-key failures never stop the
batch, and the baseline-first gate is recomputed from manifest evidence of
the expanded keys every time — so it cannot be bypassed by renaming run IDs.

Run levels are strictly separated: ``smoke`` reports live under
``pilot/fd004/smoke/`` and never write run manifests or test metrics, so a
local smoke run can never impersonate a full pilot run with the same key.
"""

import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import yaml

from kaf_profiti.experiments.accumulators import GlobalMetricAccumulator
from kaf_profiti.experiments.datasets import create_protocol_datasets
from kaf_profiti.experiments.masks import (
    TimelineMaskConfig,
    TimelineMaskedWindowDataset,
    generate_or_load_timeline_masks,
)
from kaf_profiti.industrial.batch import IndustrialCollator

_PILOT_ROOT = "pilot/fd004"
SMOKE_MAIN_CONDITION_RATE = 0.30
SMOKE_MAIN_MECHANISM = "mixed"


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
    ):
        from kaf_profiti.experiments.runtime_paths import resolve_runtime_paths

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
        protocol_dir = self.result_root / _PILOT_ROOT / "protocol" / "masks" / dataset
        self.mask_sha: Dict[str, str] = {}
        self._split_datasets: Dict[str, object] = {}
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
                protocol_dir / f"{split}_{mechanism}_{requested_rate:.2f}_seed{mask_seed}.npz",
                config,
                lengths,
                bundle.num_sensors,
            )
            self.mask_sha[split] = mask_bundle.content_sha256
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
        }

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
        for index, (split, dataset) in enumerate(self._split_datasets.items()):
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collator
            )
            out[split] = next(iter(loader))
            del loader
        return out

    def loaders(self, batch_size: int) -> Dict[str, object]:
        from torch.utils.data import DataLoader

        collator = IndustrialCollator()
        return {
            split: DataLoader(
                dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collator
            )
            for split, dataset in self._split_datasets.items()
        }


def _build_provider(spec: PilotRunSpec, data_root: Path, result_root: Path) -> RealProtocolProvider:
    return RealProtocolProvider(
        data_root=data_root,
        result_root=result_root,
        dataset=spec.dataset,
        history_len=spec.history_len,
        pred_len=spec.pred_len,
        stride=spec.stride,
        mechanism=SMOKE_MAIN_MECHANISM,
        requested_rate=SMOKE_MAIN_CONDITION_RATE,
        mask_seed=spec.mask_seed,
        split_seed=spec.split_seed,
    )


# ---------------------------------------------------------------------------
# Default full-run trainer (exercised on the execution machine in P04)
# ---------------------------------------------------------------------------


def _batch_to_device(batch, device):
    """Move every tensor field of an ``IndustrialBatch`` to ``device``."""

    if str(device) == "cpu":
        return batch
    moved = {
        key: (value.to(device) if torch.is_tensor(value) else value)
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
        optimizer.step()
        total += float(loss)
        count += 1
    return total / max(count, 1)


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
                head = model.model.flow_head if hasattr(model, "model") else model.flow_head
                crps = head.crps(batch.y_flat.unsqueeze(1), samples, batch.mq_flat)
                total += float(crps * float(batch.mq_flat.sum()))
                count += float(batch.mq_flat.sum())
    return total / max(count, 1.0)


def _test_metrics(model, loader, device, track, nsamples: int = 20) -> Dict[str, object]:
    accumulator = GlobalMetricAccumulator()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            prediction = model.predict_point(batch)
            accumulator.update_point(batch.y_flat, prediction, batch.mq_flat)
            if track == "probabilistic":
                samples = model.sample_flat(batch, nsamples=nsamples)
                model_object = model.model if hasattr(model, "model") else model
                head = model_object.flow_head
                nll_rows = model.batch_nll_rows(batch)
                accumulator.update_nll(float(nll_rows.sum()), float(batch.mq_flat.sum()))
                per_row = head.crps(batch.y_flat.unsqueeze(1), samples, batch.mq_flat)
                accumulator.update_crps(
                    float((per_row * batch.mq_flat.sum(dim=-1).clamp_min(1.0)).sum()),
                    float(batch.mq_flat.sum()),
                )
                lower, upper = model.interval95_flat(batch)
                mask = batch.mq_flat > 0
                covered = ((batch.y_flat >= lower) & (batch.y_flat <= upper) & mask).sum()
                width = ((upper - lower) * mask).sum()
                accumulator.update_interval_sums(
                    "main", float(covered), float(width), float(mask.sum())
                )
    metrics = accumulator.result()
    if track == "point":
        return accumulator.point_only_result()
    return metrics


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
        score = _valid_score(model, loaders["valid"], device, spec.track)
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
    metrics = _test_metrics(model, loaders["test"], device, spec.track)
    checkpoint_bytes = _serialize_state(model.state_dict())
    return {
        "history": history,
        "metrics": metrics,
        "checkpoint_bytes": checkpoint_bytes,
        "predictions": {},
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
        query_count = spec_query_count = valid_batch.y_flat.shape[1]
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
    ):
        from kaf_profiti.experiments.runtime_paths import resolve_runtime_paths

        paths = resolve_runtime_paths(data_root, str(result_root), {})
        self.result_root = paths.output_root
        self.data_root = paths.data_root
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
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
                        )
                    )
        return specs

    # -- shared artifacts / manifests ---------------------------------------

    def _shared_artifacts(self) -> Dict[str, str]:
        return {matrix.name: matrix.sha256 for matrix in self.matrices}

    def _runs_dir(self) -> Path:
        return self.result_root / _PILOT_ROOT / "runs"

    def _manifest_path(self, key: str) -> Path:
        return self._runs_dir() / key / "manifest.json"

    def _verified_specs(self, specs: List[PilotRunSpec]) -> Dict[str, dict]:
        shared = self._shared_artifacts()
        verified = {}
        for spec in specs:
            path = self._manifest_path(spec.key)
            if not path.exists():
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if manifest.get("status") != "completed":
                continue
            if manifest.get("shared_artifacts") != shared:
                continue
            artifacts = manifest.get("artifacts", {})
            if not artifacts or any(
                not (self.result_root / relative).exists() for relative in artifacts.values()
            ):
                continue
            verified[spec.key] = manifest
        return verified

    # -- dry-run -------------------------------------------------------------

    def dry_run(self) -> Dict[str, object]:
        specs = self.expand()
        verified = self._verified_specs(specs)
        return {
            "run_level": "dry-run",
            "keys": [spec.key for spec in specs],
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

    def _gate_violations(self, specs: List[PilotRunSpec], verified: Dict[str, dict]) -> List[str]:
        """Baseline keys (from the expanded matrix, not stored state) missing."""

        baseline_keys = [spec.key for spec in specs if spec.family == "baseline"]
        return [key for key in baseline_keys if key not in verified]

    # -- execution -------------------------------------------------------------

    def execute(
        self,
        trainer: Optional[Callable] = None,
        gate: bool = True,
        continue_on_error: bool = True,
        provider_factory: Optional[Callable] = None,
    ) -> Dict[str, object]:
        trainer = trainer or pilot_train_and_evaluate
        factory = provider_factory or _build_provider
        specs = self.expand()
        verified = self._verified_specs(specs)
        initially_verified = len(verified)
        completed: List[str] = []
        failed: List[str] = []
        errors: Dict[str, str] = {}
        gate_missing = self._gate_violations(specs, verified)

        for spec in specs:
            if spec.key in verified:
                continue
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
                provider = factory(spec, self.data_root, self.result_root)
                run_start = time.perf_counter()
                model = build_model(
                    spec,
                    provider.num_sensors,
                    provider.context_dim,
                    provider.model_options,
                    device=self.device,
                )
                outcome = trainer(model, provider.loaders(spec.batch_size), spec, provider, self.device)
                self._persist_outcome(spec, outcome)
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
                self._write_manifest(spec, status="completed", error="", outcome=outcome)
                completed.append(spec.key)
                verified[spec.key] = {}
                gate_missing = self._gate_violations(specs, verified)
            except Exception as error:  # noqa: BLE001 — one key failing must not stop the batch
                failed.append(spec.key)
                errors[spec.key] = f"{type(error).__name__}: {error}"
                self._write_manifest(spec, status="failed", error=errors[spec.key], outcome=None)
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
            "skipped": initially_verified,
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
        if outcome.get("predictions"):
            (run_dir / "predictions.json").write_text(
                json.dumps(outcome["predictions"], ensure_ascii=False), encoding="utf-8"
            )
            artifacts["predictions"] = str(
                (run_dir / "predictions.json").relative_to(self.result_root)
            )
        for filename, payload in payloads.items():
            (run_dir / filename).write_text(payload, encoding="utf-8")
            artifacts[filename.split(".")[0]] = str((run_dir / filename).relative_to(self.result_root))

    def _write_manifest(
        self, spec: PilotRunSpec, status: str, error: str, outcome: Optional[Dict]
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
            "status": status,
            "error": error,
            "device": str(outcome.get("device", self.device)) if outcome else self.device,
            "shared_artifacts": self._shared_artifacts(),
            "artifacts": artifacts,
            "test_metric_count": 1 if status == "completed" else 0,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- smoke ---------------------------------------------------------------

    def _smoke_models(self, group: str):
        track, family = SMOKE_GROUPS[group]
        selected = []
        seen = set()
        for spec in self.expand():
            if family != spec.family:
                continue
            if track is not None and spec.track != track:
                continue
            label = spec.model_label
            if label in seen:
                continue
            seen.add(label)
            selected.append(spec)
        return selected

    def _baseline_smoke_ready(self) -> None:
        """T04 gate: both baseline smoke reports must be verified ready."""

        for group in ("point_baselines", "probabilistic_baselines"):
            expected = len(self._smoke_models(group))
            path = self.result_root / _PILOT_ROOT / "smoke" / f"{group}_smoke.json"
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
        entries = []
        fingerprint = None
        for spec_template in model_specs:
            entry: Dict[str, object] = {
                "model_id": spec_template.model_id,
                "head_type": spec_template.head_type,
                "model_label": spec_template.model_label,
            }
            try:
                resolved = provider or _build_provider(spec_template, self.data_root, self.result_root)
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
        report_dir = self.result_root / _PILOT_ROOT / "smoke"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{group}_smoke.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report


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
