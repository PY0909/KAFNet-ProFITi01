import json

import torch

from kaf_profiti.experiments.artifacts import ArtifactWriter
from kaf_profiti.models.unified.output import PredictionOutput


def test_artifact_writer_creates_manifest_checkpoint_and_prediction(tmp_path):
    writer = ArtifactWriter(tmp_path, "run-1")
    manifest = writer.write_manifest(
        dataset="cmapss_fd004", task="rul", seed=2026,
        split_identity="split-a", target_schema="rul-v1",
        encoder_version="unified_temporal_channelwise_v2",
        dataset_schema="cmapss-v1", normalizer_identity="normalizer-x",
        target_convention="rul-v1", evaluator_version="evaluator-v1",
        model_version="model-v1",
    )
    assert manifest["run_id"] == "run-1"
    writer.write_checkpoint({"weight": torch.tensor([1.0])})
    writer.write_predictions(
        [PredictionOutput(
            dataset="cmapss_fd004", task="rul", encoder_version="unified_temporal_channelwise_v2",
            seed=2026, split_identity="split-a", target_schema="rul-v1", point=torch.tensor([2.0]),
        )]
    )
    writer.write_metrics({"split": "validation", "loss": 1.0})
    assert json.loads((tmp_path / "run-1" / "manifest.json").read_text())["checkpoint"] == "checkpoint.pt"
    assert (tmp_path / "run-1" / "predictions.pt").exists()
