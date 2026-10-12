import json

import torch

from compare_code.models.factory import create_baseline
from kaf_profiti.experiments.frozen_evaluation import evaluate_frozen_checkpoint
from kaf_profiti.industrial.batch import HistorySample
from kaf_profiti.industrial.datasets import DatasetBundle


def _sample(value, target, schema, metadata=None):
    return HistorySample(
        X_obs=torch.full((4, 2), float(value)),
        T_obs=torch.arange(4, dtype=torch.float32),
        target=torch.as_tensor(target, dtype=torch.float32),
        target_schema=schema,
        metadata=metadata or {},
    )


def test_frozen_state_checkpoint_writes_reevaluation_predictions(tmp_path):
    samples = [
        _sample(0, [[1.0, 2.0], [1.0, 2.0]], "metropt3_state_v1"),
        _sample(1, [[2.0, 3.0], [2.0, 3.0]], "metropt3_state_v1"),
        _sample(2, [[3.0, 4.0], [3.0, 4.0]], "metropt3_state_v1"),
    ]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "state-split", "metropt3_state_v1", {"features": ["a", "b"]},
    )
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    model = create_baseline(
        "gru", input_dim=2, history_len=4, pred_len=2, seed=4,
        split_identity="state-split", target_schema="metropt3_state_v1", d_model=8,
        layers=1, heads=2, num_classes=1,
    )
    torch.save({"model": model.state_dict(), "config": model.config.to_dict()}, run_dir / "checkpoint.pt")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "source", "dataset": "metropt3", "task": "state", "seed": 4,
        "split_identity": "state-split", "target_schema": "metropt3_state_v1",
        "model_name": "gru", "model_version": "compare_gru_v1",
        "normalizer_identity": "train-zscore-features2",
    }))

    result = evaluate_frozen_checkpoint(bundle, run_dir, tmp_path / "reeval", batch_size=1, device="cpu")

    assert result["split"] == "test"
    assert result["overall"]["mae"] >= 0.0
    assert (tmp_path / "reeval" / "test_predictions.pt").is_file()
    payload = torch.load(tmp_path / "reeval" / "test_predictions.pt", map_location="cpu", weights_only=False)
    assert payload["prediction"].shape == payload["target"].shape == (1, 2, 2)


def test_frozen_rul_checkpoint_writes_raw_cycle_metrics(tmp_path):
    samples = [
        _sample(0, [10.0], "cmapss_fd004_rul_raw_v1", {"engine_id": 1, "raw_rul": 10.0}),
        _sample(1, [20.0], "cmapss_fd004_rul_raw_v1", {"engine_id": 2, "raw_rul": 20.0}),
        _sample(2, [30.0], "cmapss_fd004_rul_raw_v1", {"engine_id": 3, "raw_rul": 30.0}),
    ]
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "rul-split", "cmapss_fd004_rul_raw_v1", {"features": ["a", "b"], "training_cap": 125.0},
    )
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    model = create_baseline(
        "gru", input_dim=2, history_len=4, pred_len=1, seed=4,
        split_identity="rul-split", target_schema="cmapss_fd004_rul_raw_v1", d_model=8,
        layers=1, heads=2, num_classes=1,
    )
    torch.save({"model": model.state_dict(), "config": model.config.to_dict()}, run_dir / "checkpoint.pt")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "source", "dataset": "cmapss_fd004", "task": "rul", "seed": 4,
        "split_identity": "rul-split", "target_schema": "cmapss_fd004_rul_raw_v1",
        "model_name": "gru", "model_version": "compare_gru_v1",
        "normalizer_identity": "train-zscore-features2",
    }))

    result = evaluate_frozen_checkpoint(bundle, run_dir, tmp_path / "reeval", batch_size=1, device="cpu")

    assert result["units"] == "cycles"
    assert result["engines"] == 1
    payload = torch.load(tmp_path / "reeval" / "test_predictions.pt", map_location="cpu", weights_only=False)
    assert payload["engine_ids"] == ["3"]


def test_frozen_evaluation_recognizes_unified_checkpoint_with_legacy_empty_model_name(tmp_path):
    samples = [_sample(0, [[1.0, 2.0], [1.0, 2.0]], "metropt3_state_v1") for _ in range(3)]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "state-split", "metropt3_state_v1", {"features": ["a", "b"]},
    )
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel

    config = ModelConfig(
        dataset="metropt3", task="state", input_dim=2, seed=4,
        split_identity="state-split", target_schema="metropt3_state_v1", pred_len=2,
    )
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    torch.save({"model": UnifiedModel(config).state_dict(), "config": config.to_dict()}, run_dir / "checkpoint.pt")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "source", "dataset": "metropt3", "task": "state", "seed": 4,
        "split_identity": "state-split", "target_schema": "metropt3_state_v1",
        "model_name": None, "model_version": config.encoder_version,
        "normalizer_identity": "train-zscore-features2",
    }))

    result = evaluate_frozen_checkpoint(bundle, run_dir, tmp_path / "reeval", batch_size=1, device="cpu")

    assert result["split"] == "test"
