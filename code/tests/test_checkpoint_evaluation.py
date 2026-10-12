import json

import torch

from compare_code.models.factory import create_baseline
from compare_code.models.base import BaselineConfig
from kaf_profiti.experiments.checkpoint_evaluation import evaluate_tep_checkpoint
from kaf_profiti.industrial.batch import HistorySample
from kaf_profiti.industrial.datasets import DatasetBundle
from kaf_profiti.industrial.protocol import split_identity


def test_tep_checkpoint_evaluation_uses_saved_weights_and_writes_test_artifacts(tmp_path):
    samples = []
    for index, label in enumerate((0, 1, 2, 0, 1, 2)):
        samples.append(HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.tensor([float(label)]), target_schema="tep_fault_v1",
            class_label=torch.tensor(label), fault_number=label, simulation_run=index,
            metadata={
                "unit_id": "{}:{}".format("test" if index >= 3 else "train", index),
                "source_split": "test" if index >= 3 else "train",
                "window_end": index,
            },
        ))
    training_samples, validation_samples, test_samples = samples[:2], samples[2:3], samples[3:]
    training_identity = split_identity(sorted(
        sample.metadata["unit_id"] for sample in training_samples + validation_samples
    ))
    bundle = DatasetBundle(
        "tep", {"train": training_samples, "validation": validation_samples, "test": test_samples},
        split_identity(sorted(sample.metadata["unit_id"] for sample in samples)),
        "tep_fault_v1", {"features": ["a", "b"], "num_fault_classes": 3},
    )

    config = BaselineConfig(
        model_name="gru", input_dim=2, history_len=4, pred_len=1, seed=12,
        split_identity=training_identity, target_schema="tep_fault_v1", d_model=8,
        layers=1, heads=2, num_classes=3,
    )
    model_values = config.to_dict()
    model_values.pop("model_name")
    model = create_baseline("gru", **model_values)
    run_dir = tmp_path / "tep_gru"
    run_dir.mkdir()
    torch.save({"model": model.state_dict(), "config": config.to_dict(), "best": {"metric": 1.0}}, run_dir / "checkpoint.pt")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "tep_gru", "dataset": "tep", "task": "fault_detection_diagnosis",
        "seed": 12, "split_identity": training_identity, "target_schema": "tep_fault_v1",
        "model_name": "gru", "model_version": "compare_gru_v1", "normalizer_identity": "train-zscore-features2",
    }))
    (run_dir / "metrics.json").write_text(json.dumps({"history": [{"epoch": 1}], "test_evaluation_count": 0}))
    expected = {key: value.clone() for key, value in model.state_dict().items()}

    result = evaluate_tep_checkpoint(bundle, run_dir, batch_size=2, device="cpu")

    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    saved = torch.load(run_dir / "test_predictions.pt", map_location="cpu", weights_only=False)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert result["evaluation_task"] == "fault_classification"
    assert metrics["support"] == 3
    assert metrics["run_summary"]["run_count"] == 3
    assert saved["probabilities"].shape == (3, 3)
    assert all(torch.equal(expected[key], checkpoint["model"][key]) for key in expected)
    assert json.loads((run_dir / "metrics.json").read_text())["test_evaluation_count"] == 1
