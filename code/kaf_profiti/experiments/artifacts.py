"""Portable, atomic run artifact writer."""

import json
import os
from pathlib import Path

import torch

from kaf_profiti.industrial.protocol import build_manifest


class ArtifactWriter:
    def __init__(self, result_root, run_id):
        self.result_root = Path(result_root)
        self.run_id = str(run_id)
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be a single directory name")
        self.run_dir = self.result_root / self.run_id
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=False, exist_ok=False)

    def _atomic_json(self, name, value):
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))
        os.replace(temporary, path)
        return path

    def write_manifest(self, **fields):
        fields.setdefault("checkpoint", "checkpoint.pt")
        manifest = build_manifest(
            run_id=self.run_id,
            **fields,
        )
        self._atomic_json("manifest.json", manifest)
        return manifest

    def write_checkpoint(self, state):
        path = self.run_dir / "checkpoint.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        return path

    def write_predictions(self, outputs):
        payload = []
        for output in outputs:
            payload.append({
                "dataset": output.dataset,
                "task": output.task,
                "encoder_version": output.encoder_version,
                "seed": output.seed,
                "split_identity": output.split_identity,
                "target_schema": output.target_schema,
                "point": output.point.detach().cpu() if output.point is not None else None,
                "quantiles": output.quantiles.detach().cpu() if output.quantiles is not None else None,
                "detection_probability": output.detection_probability.detach().cpu() if output.detection_probability is not None else None,
                "diagnosis_probability": output.diagnosis_probability.detach().cpu() if output.diagnosis_probability is not None else None,
                "metadata": output.metadata,
            })
        path = self.run_dir / "predictions.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path

    def write_test_predictions(self, payload):
        path = self.run_dir / "test_predictions.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path

    def write_metrics(self, metrics):
        return self._atomic_json("metrics.json", metrics)

    def write_json(self, name, value):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("artifact name must be a JSON filename")
        return self._atomic_json(name, value)
