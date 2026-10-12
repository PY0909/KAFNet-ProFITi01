"""Re-evaluate frozen checkpoints without retraining or changing source runs."""

import json
import os
import shutil
from pathlib import Path
from time import perf_counter

import torch

from .entrypoint import _loader, _normalize_bundle
from .metrics import classification_metrics, rul_metrics, state_metrics


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))
    os.replace(temporary, path)


def _write_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _make_model(config, manifest):
    if manifest.get("model_name") in (None, "unified") and config.get("variant", "v2") in {"v2", "v2t", "v2tc", "v3a", "v3b", "v3c", "v4"}:
        from ..models.unified import ModelConfig, UnifiedModel

        fields = set(ModelConfig.__dataclass_fields__)
        return UnifiedModel(ModelConfig(**{key: value for key, value in config.items() if key in fields}))
    from compare_code.models.factory import create_baseline

    values = dict(config)
    name = values.pop("model_name", manifest["model_name"])
    return create_baseline(name, **values)


def evaluate_frozen_checkpoint(bundle, source_dir, output_dir, batch_size=128, device="auto"):
    """Evaluate one checkpoint on test samples and write a separate re-evaluation run."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("reevaluation output already exists: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    checkpoint_path = source_dir / "checkpoint.pt"
    manifest = json.loads((source_dir / "manifest.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if manifest.get("dataset") != bundle.dataset or manifest.get("target_schema") != bundle.target_schema:
        raise ValueError("checkpoint and bundle dataset/schema do not match")
    normalized_result = _normalize_bundle(bundle, normalize_targets=(bundle.dataset == "metropt3"))
    if bundle.dataset == "metropt3":
        normalized_bundle, target_stats = normalized_result
    else:
        normalized_bundle, target_stats = normalized_result, None
    if manifest.get("normalizer_identity") != normalized_bundle.metadata.get("normalizer_identity"):
        raise ValueError("checkpoint normalizer does not match train-fitted statistics")
    if manifest.get("split_identity") != normalized_bundle.split_identity:
        raise ValueError("checkpoint split identity does not match the supplied bundle")
    model = _make_model(checkpoint["config"], manifest)
    model.load_state_dict(checkpoint["model"], strict=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    model.to(device).eval()
    test_samples = normalized_bundle.samples.get("test", [])
    if not test_samples:
        raise ValueError("test split must not be empty")
    predictions, targets, units = [], [], []
    class_labels, class_predictions, class_probabilities, run_ids = [], [], [], []
    weighted_loss = 0.0
    loss_examples = 0
    elapsed = 0.0
    batches = list(_loader(test_samples, batch_size))
    sample_batches = [test_samples[i:i + batch_size] for i in range(0, len(test_samples), batch_size)]
    with torch.no_grad():
        for (batch, batch_targets), sample_batch in zip(batches, sample_batches):
            batch = batch.to(device)
            batch_targets = batch_targets.to(device)
            started = perf_counter()
            output = model.predict(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += perf_counter() - started
            loss = model.loss(output, batch_targets)
            if isinstance(loss, dict):
                loss = loss["total"]
            batch_size_actual = int(batch.X_obs.shape[0])
            weighted_loss += float(loss.detach().cpu()) * batch_size_actual
            loss_examples += batch_size_actual
            if bundle.dataset == "metropt3":
                point = output.point.detach().cpu()
                target = batch_targets.values.detach().cpu()
                mean, std = target_stats["mean"], target_stats["std"]
                predictions.append(point * std + mean)
                targets.append(target * std + mean)
            elif bundle.dataset == "cmapss_fd004":
                predictions.append(output.point.detach().cpu().reshape(-1))
                targets.append(batch_targets.values.detach().cpu().reshape(-1))
                units.extend(str(sample.metadata["engine_id"]) for sample in sample_batch)
            else:
                probabilities = output.diagnosis_probability.detach().cpu()
                labels = batch_targets.class_label.detach().cpu().long().reshape(-1)
                class_probabilities.append(probabilities)
                class_predictions.append(probabilities.argmax(dim=-1))
                class_labels.append(labels)
                run_ids.extend(str(sample.metadata["unit_id"]) for sample in sample_batch)

    if bundle.dataset == "metropt3":
        prediction, target = torch.cat(predictions), torch.cat(targets)
        test_metrics = state_metrics(prediction, target, bundle.metadata["features"])
        payload = {
            "run_id": output_dir.name, "source_run_id": manifest["run_id"],
            "dataset": bundle.dataset, "task": manifest["task"],
            "split_identity": manifest["split_identity"], "target_schema": bundle.target_schema,
            "units": "raw_sensor_units", "prediction": prediction, "target": target,
        }
    elif bundle.dataset == "cmapss_fd004":
        prediction, target = torch.cat(predictions), torch.cat(targets)
        test_metrics = rul_metrics(prediction, target, units)
        payload = {
            "run_id": output_dir.name, "source_run_id": manifest["run_id"],
            "dataset": bundle.dataset, "task": manifest["task"],
            "split_identity": manifest["split_identity"], "target_schema": bundle.target_schema,
            "units": "cycles", "engine_ids": units, "prediction": prediction, "target": target,
        }
    else:
        labels = torch.cat(class_labels)
        predicted = torch.cat(class_predictions)
        probabilities = torch.cat(class_probabilities)
        test_metrics = classification_metrics(labels, predicted, probabilities.shape[-1], run_ids)
        payload = {
            "run_id": output_dir.name, "source_run_id": manifest["run_id"],
            "dataset": bundle.dataset, "task": manifest["task"],
            "split_identity": manifest["split_identity"], "target_schema": bundle.target_schema,
            "run_ids": run_ids, "labels": labels, "predictions": predicted, "probabilities": probabilities,
        }
    test_metrics.update({
        "split": "test", "run_id": output_dir.name, "source_run_id": manifest["run_id"],
        "checkpoint": "checkpoint.pt", "dataset": bundle.dataset, "task": manifest["task"],
        "split_identity": manifest["split_identity"], "target_schema": bundle.target_schema,
        "loss": weighted_loss / max(1, loss_examples), "batches": len(batches),
        "examples": len(test_samples), "evaluator_version": "frozen_evaluator_v4",
    })
    output_manifest = dict(manifest)
    output_manifest.update({
        "run_id": output_dir.name, "source_run_id": manifest["run_id"],
        "evaluator_version": "frozen_evaluator_v4", "checkpoint": "checkpoint.pt",
    })
    shutil.copy2(checkpoint_path, output_dir / "checkpoint.pt")
    _write_json(output_dir / "manifest.json", output_manifest)
    _write_json(output_dir / "test_metrics.json", test_metrics)
    _write_torch(output_dir / "test_predictions.pt", payload)
    _write_json(output_dir / "efficiency.json", {
        "run_id": output_dir.name, "source_run_id": manifest["run_id"],
        "test_batch_inference_seconds": elapsed / max(1, len(batches)),
        "test_examples": len(test_samples), "device": str(device),
    })
    return test_metrics
