"""Test-only evaluation for a frozen TEP classification checkpoint."""

import json
import os
from pathlib import Path
from time import perf_counter

import torch

from .entrypoint import _loader, _normalize_bundle
from ..industrial.protocol import split_identity
from ..models.unified import ModelConfig, UnifiedModel


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))
    os.replace(temporary, path)


def _atomic_torch_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _build_model(config):
    if config.get("dataset") == "tep":
        return UnifiedModel(ModelConfig(**config))
    model_name = config.get("model_name")
    if model_name not in {"tcn", "gru", "patchtst", "timesnet", "itransformer"}:
        raise ValueError("unsupported checkpoint model: {}".format(model_name))
    from compare_code.models.factory import create_baseline

    values = dict(config)
    values.pop("model_name")
    values["num_classes"] = values.pop("num_classes", values.get("num_fault_classes", 1))
    return create_baseline(model_name, **values)


def evaluate_tep_checkpoint(
    bundle, run_dir, batch_size=128, device="auto", normalizer_stats=None,
    training_identity=None,
):
    """Evaluate saved TEP weights once on the official test split without training."""
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "checkpoint.pt"
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    for path in (checkpoint_path, manifest_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError("required checkpoint artifact not found: {}".format(path))
    if bundle.dataset != "tep" or bundle.target_schema != "tep_fault_v1":
        raise ValueError("TEP checkpoint evaluation requires the TEP fault-classification bundle")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    test_samples = bundle.samples.get("test", [])
    if not test_samples:
        raise ValueError("test evaluation requested but test split is empty")
    if any((sample.metadata or {}).get("source_split") != "test" for sample in test_samples):
        raise ValueError("TEP test samples must come from the official testing files")

    if training_identity is None:
        training_identity = split_identity(sorted(
            str(sample.metadata["unit_id"])
            for split in ("train", "validation")
            for sample in bundle.samples[split]
        ))
    manifest = json.loads(manifest_path.read_text())
    run_metrics = json.loads(metrics_path.read_text())
    if run_metrics.get("test_evaluation_count", 0) != 0:
        raise ValueError("this run already has a test evaluation; refusing to evaluate test twice")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    expected = {
        "dataset": "tep",
        "task": "fault_detection_diagnosis",
        "seed": manifest.get("seed"),
        "split_identity": training_identity,
        "target_schema": "tep_fault_v1",
    }
    for key, value in expected.items():
        actual = config.get(key, manifest.get(key))
        if actual != value:
            raise ValueError("checkpoint {} mismatch: expected {!r}, found {!r}".format(key, value, actual))
        if manifest.get(key) != value:
            raise ValueError("manifest {} mismatch: expected {!r}, found {!r}".format(key, value, manifest.get(key)))
    model_name = manifest.get("model_name")
    if config.get("model_name", "unified") != model_name:
        raise ValueError("checkpoint model name does not match its manifest")

    if normalizer_stats is None:
        normalized_bundle = _normalize_bundle(bundle)
        normalizer_identity = normalized_bundle.metadata.get("normalizer_identity")
    else:
        normalizer_identity = normalizer_stats.get("identity")
        if normalizer_stats.get("source_split") != "train" or training_identity != config.get("split_identity"):
            raise ValueError("external normalization statistics must be fitted on the checkpoint training split")
        mean = torch.as_tensor(normalizer_stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(normalizer_stats["std"], dtype=torch.float32)
        if mean.numel() != int(config["input_dim"]) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("external normalization statistics do not match checkpoint input dimensions")
        from dataclasses import replace as dataclass_replace

        test_samples = [dataclass_replace(sample, X_obs=(sample.X_obs.float() - mean) / std)
                        for sample in test_samples]
        normalized_bundle = bundle
    if manifest.get("normalizer_identity") != normalizer_identity:
        raise ValueError("checkpoint normalizer identity does not match train-fitted statistics")
    model = _build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    model.to(device).eval()

    labels, probabilities, run_ids = [], [], []
    losses = []
    elapsed = 0.0
    batches = list(_loader(test_samples, batch_size))
    test_sample_batches = [test_samples[offset:offset + batch_size]
                           for offset in range(0, len(test_samples), batch_size)]
    with torch.no_grad():
        for (batch, targets), sample_batch in zip(batches, test_sample_batches):
            batch = batch.to(device)
            targets = targets.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            prediction = model.predict(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += perf_counter() - started
            if prediction.diagnosis_probability is None or targets.class_label is None:
                raise ValueError("TEP checkpoint must emit diagnosis probabilities")
            loss = model.loss(prediction, targets)
            if isinstance(loss, dict):
                loss = loss["total"]
            losses.append(float(loss.detach().cpu()))
            probabilities.append(prediction.diagnosis_probability.detach().cpu())
            labels.append(targets.class_label.detach().cpu().long().reshape(-1))
            run_ids.extend(str(sample.metadata["unit_id"]) for sample in sample_batch)

    labels = torch.cat(labels)
    probabilities = torch.cat(probabilities)
    predictions = probabilities.argmax(dim=-1)
    from .metrics import classification_metrics

    num_classes = int(probabilities.shape[-1])
    test_metrics = classification_metrics(labels, predictions, num_classes, run_ids)
    test_metrics.update({
        "split": "test",
        "evaluation_task": "fault_classification",
        "loss": sum(losses) / len(losses),
        "batches": len(batches),
        "examples": int(labels.numel()),
        "window_count": int(labels.numel()),
        "class_mapping": {str(index): index for index in range(num_classes)},
        "run_id": manifest["run_id"],
        "seed": manifest["seed"],
        "checkpoint": "checkpoint.pt",
        "dataset": "tep",
        "task": manifest["task"],
        "split_identity": manifest["split_identity"],
        "dataset_schema": manifest.get("dataset_schema"),
        "normalizer_identity": normalizer_identity,
        "target_schema": manifest["target_schema"],
        "selected_testing_runs": normalized_bundle.metadata.get("selected_testing_runs"),
    })
    _atomic_json(run_dir / "test_metrics.json", test_metrics)
    _atomic_torch_save(run_dir / "test_predictions.pt", {
        "run_id": manifest["run_id"],
        "dataset": "tep",
        "task": "fault_classification",
        "split_identity": manifest["split_identity"],
        "target_schema": manifest["target_schema"],
        "class_mapping": {str(index): index for index in range(num_classes)},
        "run_ids": run_ids,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
    })
    run_metrics["test_evaluation_count"] = 1
    _atomic_json(metrics_path, run_metrics)
    efficiency_path = run_dir / "efficiency.json"
    if efficiency_path.is_file():
        efficiency = json.loads(efficiency_path.read_text())
        efficiency.update({
            "test_batch_inference_seconds": elapsed / max(1, len(batches)),
            "test_examples": int(labels.numel()),
        })
        _atomic_json(efficiency_path, efficiency)
    return test_metrics
