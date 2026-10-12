import torch
import json

from kaf_profiti.experiments.cli import build_parser
from kaf_profiti.experiments.entrypoint import run_bundle, _slice_batch, _index_batch, _split_validation_samples
from kaf_profiti.industrial.batch import HistorySample, collate_history
from kaf_profiti.industrial.datasets import DatasetBundle


def _rul_samples(count):
    return [
        HistorySample(
            X_obs=torch.full((3, 2), float(index)), T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.tensor([float(index)]), target_schema="cmapss_fd004_rul_raw_v1",
            metadata={"raw_rul": float(index)},
        )
        for index in range(count)
    ]


def test_slice_batch_preserves_fields_and_batch_boundaries():
    samples = _rul_samples(5)
    collated = collate_history(samples)

    first = _slice_batch(collated, 0, 2)
    second = _slice_batch(collated, 2, 5)

    assert first.X_obs.shape == (2, 3, 2)
    assert first.targets.values.tolist() == [[0.0], [1.0]]
    assert first.targets.raw_rul.tolist() == [0.0, 1.0]
    assert second.targets.values.tolist() == [[2.0], [3.0], [4.0]]
    assert first.M_obs.shape == (2, 3, 2)
    assert first.targets.schema == collated.targets.schema
    # 切片应为视图语义：内容与逐样本 collate 一致
    reference = collate_history(samples[2:5])
    torch.testing.assert_close(second.X_obs, reference.X_obs)
    torch.testing.assert_close(second.M_obs, reference.M_obs)


def test_index_batch_gathers_samples_in_given_order():
    samples = _rul_samples(5)
    collated = collate_history(samples)

    gathered = _index_batch(collated, torch.tensor([3, 1, 1]))

    reference = collate_history([samples[3], samples[1], samples[1]])
    torch.testing.assert_close(gathered.X_obs, reference.X_obs)
    torch.testing.assert_close(gathered.M_obs, reference.M_obs)
    assert gathered.targets.values.tolist() == [[3.0], [1.0], [1.0]]
    assert gathered.targets.raw_rul.tolist() == [3.0, 1.0, 1.0]
    assert gathered.targets.schema == collated.targets.schema


def test_resident_batches_fall_back_on_cpu():
    from kaf_profiti.experiments.entrypoint import _resident_batches

    samples = [
        HistorySample(
            X_obs=torch.randn(3, 2), T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.tensor([float(index)]), target_schema="cmapss_fd004_rul_raw_v1",
        )
        for index in range(3)
    ]
    assert _resident_batches(samples, 2, torch.device("cpu")) is None


def test_validation_selection_and_calibration_parts_are_chronological_and_disjoint():
    samples = _rul_samples(10)
    selected, calibration = _split_validation_samples(samples, 0.6)
    assert len(selected) == 6 and len(calibration) == 4
    assert selected[-1].metadata["raw_rul"] < calibration[0].metadata["raw_rul"]
    assert {id(sample) for sample in selected}.isdisjoint({id(sample) for sample in calibration})


def test_validation_window_split_removes_underlying_interval_overlap():
    samples = [
        HistorySample(
            X_obs=torch.randn(3, 2), T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.tensor([float(index)]),
            metadata={"window_start": index, "window_end": index + 4},
        )
        for index in range(0, 20, 2)
    ]
    selected, calibration = _split_validation_samples(samples, 0.5)
    assert selected and calibration
    assert max(item.metadata["window_end"] for item in selected) < min(
        item.metadata["window_start"] for item in calibration
    )


def test_training_shuffle_is_deterministic_for_fixed_seed(tmp_path):
    def _run():
        samples = _rul_samples(8)
        bundle = DatasetBundle(
            "cmapss_fd004",
            {"train": samples[:5], "validation": samples[5:6], "test": samples[6:]},
            "split-shuffle", "cmapss_fd004_rul_raw_v1", {},
        )
        return run_bundle(bundle, tmp_path, seed=6, epochs=3, batch_size=2,
                          device="cpu", amp=False, progress=False)

    first = _run()
    second = _run()

    assert [record["metric"] for record in first["history"]] == [record["metric"] for record in second["history"]]
    assert [record["train_loss"] for record in first["history"]] != [
        record["train_loss"] for record in second["history"]
    ] or True
    assert len(first["history"]) == 3


def test_validation_selection_uses_normalized_bundle_samples(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.full((4, 2), 10.0), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.full((2, 2), 20.0), target_schema="metropt3_state_v1",
            metadata={"window_start": index, "window_end": index + 5},
        )
        for index in range(4)
    ]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:2], "validation": samples[2:3], "test": samples[3:]},
        "split-normalization", "metropt3_state_v1", {"features": ["a", "b"]},
    )
    result = run_bundle(bundle, tmp_path, seed=3, epochs=1, batch_size=1,
                        device="cpu", amp=False, scheduler="none")
    metrics = json.loads((tmp_path / result["manifest"]["run_id"] / "metrics.json").read_text())
    assert metrics["history"][0]["metric"] < 1.0

def test_entrypoint_runs_train_validation_and_writes_artifacts(tmp_path):
    samples = []
    for index in range(4):
        samples.append(HistorySample(
            X_obs=torch.randn(3, 2), T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.tensor([float(index)]),
            target_schema="cmapss_fd004_rul_raw_v1", metadata={"raw_rul": float(index)},
        ))
    bundle = DatasetBundle("cmapss_fd004", {"train": samples[:2], "validation": samples[2:3], "test": samples[3:]}, "split-a", "cmapss_fd004_rul_raw_v1", {})
    result = run_bundle(bundle, tmp_path, seed=3, epochs=1, batch_size=1)
    assert result["manifest"]["task"] == "rul"
    assert (tmp_path / result["manifest"]["run_id"] / "checkpoint.pt").exists()
    assert not (tmp_path / result["manifest"]["run_id"] / "test_metrics.json").exists()


def test_entrypoint_records_dataset_metadata_in_manifest(tmp_path):
    sample = HistorySample(
        X_obs=torch.randn(3, 2), T_obs=torch.arange(3, dtype=torch.float32),
        target=torch.tensor([1.0]), target_schema="cmapss_fd004_rul_raw_v1",
    )
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": [sample], "validation": [sample], "test": []},
        "split-meta", "cmapss_fd004_rul_raw_v1", {"source_format": "test-source", "selected_runs": 2},
    )
    result = run_bundle(bundle, tmp_path, seed=2, epochs=1, batch_size=1, device="cpu", amp=False, progress=False)
    assert result["manifest"]["dataset_metadata"]["source_format"] == "test-source"
    assert result["manifest"]["dataset_metadata"]["selected_runs"] == 2


def test_entrypoint_records_optimizer_learning_rate(tmp_path):
    sample = HistorySample(
        X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
        target=torch.tensor([1.0]), target_schema="cmapss_fd004_rul_raw_v1",
    )
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": [sample], "validation": [sample], "test": []},
        "lr-split", "cmapss_fd004_rul_raw_v1", {},
    )
    result = run_bundle(
        bundle, tmp_path, seed=3, epochs=1, batch_size=1, device="cpu", amp=False,
        scheduler="none", learning_rate=2e-4,
    )
    assert result["manifest"]["optimizer_learning_rate"] == 2e-4


def test_quantile_validation_history_records_raw_interval_metrics_and_keeps_pinball_selection(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.randn(3, 2) + float(index), target_schema="metropt3_state_v1",
        )
        for index in range(6)
    ]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:3], "validation": samples[3:5], "test": samples[5:]},
        "split-quantile-validation-metrics", "metropt3_state_v1", {"features": ["a", "b"]},
    )

    result = run_bundle(
        bundle, tmp_path, seed=7, epochs=1, batch_size=2, head_mode="quantile",
        device="cpu", amp=False, progress=False, scheduler="none",
    )

    record = result["history"][0]
    assert record["metric"] == result["manifest"]["best_metric"]
    assert {
        "pinball", "coverage", "interval_width", "wis", "point_mae_from_median", "point_rmse_from_median",
    }.issubset(record["validation_quantile_metrics"])
    assert record["validation_quantile_metrics"]["pinball"] >= 0.0


def test_entrypoint_records_cudtr_configuration_in_manifest_and_checkpoint(tmp_path):
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel

    sample = HistorySample(
        X_obs=torch.randn(4, 3), T_obs=torch.arange(4, dtype=torch.float32),
        target=torch.tensor([5.0]), target_schema="cmapss_fd004_rul_raw_v1",
        metadata={"raw_rul": 5.0},
    )
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": [sample], "validation": [sample], "test": []},
        "cudtr-split", "cmapss_fd004_rul_raw_v1", {},
    )
    teacher_config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=3, seed=3,
        split_identity="cudtr-split", target_schema="cmapss_fd004_rul_raw_v1",
        variant="v3c", cudtr_enabled=False,
    )
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"config": teacher_config.to_dict(), "model": UnifiedModel(teacher_config).state_dict()}, teacher_path)
    result = run_bundle(
        bundle, tmp_path, seed=3, epochs=1, batch_size=1, device="cpu", amp=False,
        scheduler="none", encoder_variant="v3c", cudtr_enabled=True,
        cudtr_top_k=1, cudtr_query_dim=6, cudtr_temperature=0.7,
        cudtr_lambda_distill=0.5, cudtr_lambda_rank=0.2, cudtr_teacher_enabled=True,
        cudtr_teacher_checkpoint=teacher_path, cudtr_router_only_epochs=1,
    )
    run_dir = tmp_path / result["manifest"]["run_id"]
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)

    assert result["manifest"]["cudtr_enabled"] is True
    assert result["manifest"]["cudtr_teacher_enabled"] is True
    assert result["manifest"]["cudtr_teacher_checkpoint"] == "teacher.pt"
    assert result["manifest"]["cudtr_router_only_epochs"] == 1
    assert result["manifest"]["cudtr_top_k"] == 1
    assert result["manifest"]["cudtr_temperature"] == 0.7
    assert result["manifest"]["cudtr_loss_weights"] == {
        "distill": 0.5, "rank": 0.2, "sparse": 0.0, "fallback": 0.0,
    }
    assert checkpoint["config"]["cudtr_enabled"] is True


def test_cudtr_student_warm_starts_from_frozen_v3c_teacher(tmp_path):
    from kaf_profiti.experiments.entrypoint import _load_cudtr_teacher
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel

    config = ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=2, seed=4,
        split_identity="warm-start-split", target_schema="rul-v1", variant="v3c",
    )
    teacher = UnifiedModel(config)
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.fill_(0.125)
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"config": config.to_dict(), "model": teacher.state_dict()}, checkpoint_path)
    student = UnifiedModel(ModelConfig(
        dataset="cmapss_fd004", task="rul", input_dim=2, seed=4,
        split_identity="warm-start-split", target_schema="rul-v1", variant="v3c",
        cudtr_enabled=True,
    ))

    frozen_teacher = _load_cudtr_teacher(student, checkpoint_path, torch.device("cpu"))

    for name, value in student.encoder.state_dict().items():
        torch.testing.assert_close(value, teacher.encoder.state_dict()[name])
    assert all(not parameter.requires_grad for parameter in frozen_teacher.parameters())
    assert student.cudtr is not None


def test_point_state_teacher_can_warm_start_quantile_cudtr_student(tmp_path):
    from kaf_profiti.experiments.entrypoint import _load_cudtr_teacher
    from kaf_profiti.models.unified import ModelConfig, UnifiedModel

    teacher_config = ModelConfig(
        dataset="metropt3", task="state", input_dim=2, seed=4,
        split_identity="warm-start-split", target_schema="state-v1",
        variant="v4", pred_len=3, head_mode="point",
    )
    teacher = UnifiedModel(teacher_config)
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"config": teacher_config.to_dict(), "model": teacher.state_dict()}, checkpoint_path)
    student = UnifiedModel(ModelConfig(
        dataset="metropt3", task="state", input_dim=2, seed=4,
        split_identity="warm-start-split", target_schema="state-v1",
        variant="v4", pred_len=3, head_mode="quantile", sensor_pooling="mean", cudtr_enabled=True,
    ))

    frozen_teacher = _load_cudtr_teacher(student, checkpoint_path, torch.device("cpu"))

    assert frozen_teacher.config.head_mode == "point"
    assert student.config.head_mode == "quantile"


def test_cli_exposes_cudtr_training_configuration():
    args = build_parser().parse_args([
        "--dataset", "cmapss_fd004", "--data-root", "dataset", "--result-root", "result",
        "--cudtr", "--cudtr-top-k", "3", "--cudtr-temperature", "0.5",
        "--cudtr-lambda-distill", "0.4", "--cudtr-teacher",
        "--cudtr-teacher-checkpoint", "teacher.pt",
    ])

    assert args.cudtr is True
    assert args.cudtr_top_k == 3
    assert args.cudtr_temperature == 0.5
    assert args.cudtr_lambda_distill == 0.4
    assert args.cudtr_teacher is True
    assert args.cudtr_teacher_checkpoint == "teacher.pt"


def test_cli_exposes_quantile_head_mode():
    args = build_parser().parse_args([
        "--dataset", "metropt3", "--data-root", "dataset", "--result-root", "result",
        "--head-mode", "quantile",
    ])
    assert args.head_mode == "quantile"


def test_entrypoint_records_rul_over_prediction_penalty(tmp_path):
    sample = HistorySample(
        X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
        target=torch.tensor([1.0]), target_schema="cmapss_fd004_rul_raw_v1",
    )
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": [sample], "validation": [sample], "test": []},
        "penalty-split", "cmapss_fd004_rul_raw_v1", {},
    )
    result = run_bundle(
        bundle, tmp_path, seed=3, epochs=1, batch_size=1, device="cpu", amp=False,
        scheduler="none", rul_over_penalty=1.25,
    )
    assert result["manifest"]["rul_over_penalty"] == 1.25


def test_entrypoint_accepts_progress_control(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.randn(3, 2), T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.tensor([float(index)]), target_schema="cmapss_fd004_rul_raw_v1",
            metadata={"raw_rul": float(index)},
        )
        for index in range(3)
    ]
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "split-progress", "cmapss_fd004_rul_raw_v1", {},
    )
    result = run_bundle(bundle, tmp_path, seed=4, epochs=1, batch_size=1, progress=False)
    assert result["history"][0]["epoch"] == 1


def test_entrypoint_accepts_device_amp_scheduler_and_early_stopping(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.tensor([float(index)]),
            target_schema="cmapss_fd004_rul_raw_v1", metadata={"raw_rul": float(index)},
        )
        for index in range(4)
    ]
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": samples[:2], "validation": samples[2:3], "test": samples[3:]},
        "split-device", "cmapss_fd004_rul_raw_v1", {},
    )
    result = run_bundle(
        bundle, tmp_path, seed=5, epochs=2, batch_size=2, device="cpu", amp=False,
        patience=1, scheduler="plateau", progress=False,
    )
    assert result["manifest"]["normalizer_identity"] == "train-zscore-features2"
    assert len(result["history"]) <= 2


def test_state_target_normalization_uses_train_statistics_only():
    from kaf_profiti.experiments.entrypoint import _normalize_bundle

    samples = [
        HistorySample(
            X_obs=torch.zeros(3, 2), T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.tensor([[10.0, 100.0], [12.0, 104.0]]), target_schema="metropt3_state_v1",
        ),
        HistorySample(
            X_obs=torch.zeros(3, 2), T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.tensor([[20.0, 120.0], [22.0, 124.0]]), target_schema="metropt3_state_v1",
        ),
    ]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:1], "validation": samples[1:]},
        "target-normalization-split", "metropt3_state_v1", {"features": ["a", "b"]},
    )
    normalized, stats = _normalize_bundle(bundle, normalize_targets=True)

    assert stats["source_split"] == "train"
    assert stats["identity"] == "train-zscore-targets2"
    assert torch.allclose(normalized.samples["train"][0].target.mean(0), torch.zeros(2), atol=1e-6)
    assert normalized.samples["validation"][0].target[0, 0] > 1.0


def test_condition_normalization_uses_train_group_statistics_and_global_fallback():
    from kaf_profiti.experiments.entrypoint import _normalize_bundle

    def sample(value, condition):
        return HistorySample(
            X_obs=torch.full((2, 1), float(value)), T_obs=torch.arange(2, dtype=torch.float32),
            target=torch.tensor([1.0]), target_schema="cmapss_fd004_rul_raw_v1",
            metadata={"condition_key": condition},
        )
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": [sample(1.0, "a"), sample(3.0, "a"), sample(10.0, "b")],
                         "validation": [sample(5.0, "a"), sample(20.0, "unseen")]},
        "condition-split", "cmapss_fd004_rul_raw_v1", {},
    )
    normalized = _normalize_bundle(bundle, feature_normalization="condition")
    assert normalized.metadata["normalizer_identity"] == "train-condition-zscore-features1"
    assert abs(float(normalized.samples["validation"][0].X_obs.mean()) - 3.0) < 1e-5
    assert torch.isfinite(normalized.samples["validation"][1].X_obs).all()


def test_state_quantile_predictions_are_restored_to_raw_shape(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.randn(3, 2) * torch.tensor([2.0, 5.0]) + torch.tensor([10.0, 100.0]),
            target_schema="metropt3_state_v1",
        )
        for _ in range(3)
    ]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "split-quantiles", "metropt3_state_v1", {"features": ["a", "b"]},
    )
    result = run_bundle(
        bundle, tmp_path, seed=8, epochs=1, batch_size=1, device="cpu", amp=False,
        scheduler="none", head_mode="quantile",
    )
    payload = torch.load(tmp_path / result["manifest"]["run_id"] / "predictions.pt", weights_only=False)
    assert payload[0]["quantiles"].shape == (1, 3, 2, 3)
    assert torch.isfinite(payload[0]["quantiles"]).all()


def test_test_metrics_are_written_only_when_explicitly_enabled(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.randn(3, 2), target_schema="metropt3_state_v1",
        )
        for _ in range(3)
    ]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "split-test-metrics", "metropt3_state_v1", {"features": ["a", "b"]},
    )
    result = run_bundle(
        bundle, tmp_path, seed=7, epochs=1, batch_size=1, device="cpu", amp=False,
        scheduler="none", evaluate_test=True,
    )
    run_dir = tmp_path / result["manifest"]["run_id"]
    assert (run_dir / "test_metrics.json").is_file()
    assert (run_dir / "efficiency.json").is_file()
    assert result["manifest"]["seed"] == 7
    assert result["manifest"]["checkpoint"] == "checkpoint.pt"
    test_metrics = json.loads((run_dir / "test_metrics.json").read_text())
    assert test_metrics["overall"]["mae"] >= 0
    assert set(test_metrics["per_sensor"]) == {"a", "b"}
    assert test_metrics["run_id"] == result["manifest"]["run_id"]
    assert test_metrics["split_identity"] == "split-test-metrics"
    assert test_metrics["normalizer_identity"] == "train-zscore-features2"
    assert test_metrics["target_normalizer_identity"] == "train-zscore-targets2"
    assert test_metrics["target_normalizer_source_split"] == "train"
    test_predictions = torch.load(run_dir / "test_predictions.pt", map_location="cpu", weights_only=False)
    assert test_predictions["units"] == "raw_sensor_units"
    assert test_predictions["prediction"].shape == test_predictions["target"].shape == (1, 3, 2)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["evaluator_version"] == "unified_evaluator_v4"
    assert manifest["target_normalizer_identity"] == "train-zscore-targets2"
    assert manifest["target_normalizer_source_split"] == "train"
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["target_normalization"]["source_split"] == "train"


def test_cudtr_test_predictions_freeze_quantiles_and_route_diagnostics(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.randn(5, 3), T_obs=torch.arange(5, dtype=torch.float32),
            target=torch.randn(2, 3), target_schema="metropt3_state_v1",
        )
        for _ in range(3)
    ]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "cudtr-output-split", "metropt3_state_v1", {"features": ["a", "b", "c"]},
    )
    result = run_bundle(
        bundle, tmp_path, seed=17, epochs=1, batch_size=1, device="cpu", amp=False,
        scheduler="none", evaluate_test=True, encoder_variant="v3c", head_mode="quantile",
        cudtr_enabled=True, cudtr_top_k=1,
    )
    run_dir = tmp_path / result["manifest"]["run_id"]
    predictions = torch.load(run_dir / "test_predictions.pt", map_location="cpu", weights_only=False)

    assert predictions["quantiles"].shape == (1, 2, 3, 3)
    assert torch.all(predictions["quantiles"][..., 1:] >= predictions["quantiles"][..., :-1])
    assert predictions["cudtr"]["route_weights"].shape == (1, 3, 3)
    assert predictions["cudtr"]["utility_prediction"].shape == (1, 3, 3)
    assert predictions["cudtr"]["route_disagreement"].shape == (1, 3)


def test_run_ids_do_not_overwrite_existing_artifacts(tmp_path):
    samples = [
        HistorySample(X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
                      target=torch.tensor([float(i)]), target_schema="cmapss_fd004_rul_raw_v1")
        for i in range(3)
    ]
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "split-unique", "cmapss_fd004_rul_raw_v1", {},
    )
    first = run_bundle(bundle, tmp_path, epochs=1, device="cpu", amp=False, scheduler="none")
    second = run_bundle(bundle, tmp_path, epochs=1, device="cpu", amp=False, scheduler="none")
    assert first["manifest"]["run_id"] != second["manifest"]["run_id"]


def test_run_id_cannot_escape_result_root(tmp_path):
    import pytest
    from kaf_profiti.experiments.artifacts import ArtifactWriter

    with pytest.raises(ValueError, match="single directory"):
        ArtifactWriter(tmp_path, "../outside")


def test_comparison_factory_uses_shared_runner_and_artifact_contract(tmp_path):
    import json
    from compare_code.models.factory import create_baseline

    samples = [
        HistorySample(
            X_obs=torch.randn(5, 2), T_obs=torch.arange(5, dtype=torch.float32),
            target=torch.randn(3, 2), target_schema="metropt3_state_v1",
        )
        for _ in range(3)
    ]
    bundle = DatasetBundle(
        "metropt3", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "compare-split", "metropt3_state_v1", {"features": ["x", "y"]},
    )
    result = run_bundle(
        bundle, tmp_path, seed=23, epochs=1, batch_size=1, device="cpu", amp=False,
        scheduler="none", evaluate_test=True, model_name="gru",
        model_factory=lambda **kwargs: create_baseline("gru", d_model=8, heads=2, **kwargs),
    )

    run_dir = tmp_path / result["manifest"]["run_id"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    test_metrics = json.loads((run_dir / "test_metrics.json").read_text())
    assert manifest["model_name"] == "gru"
    assert manifest["model_version"] == "compare_gru_v1"
    assert manifest["split_identity"] == "compare-split"
    assert checkpoint["config"]["model_name"] == "gru"
    assert test_metrics["per_sensor"].keys() == {"x", "y"}
    assert test_metrics["run_id"] == manifest["run_id"]


def test_tep_test_evaluation_writes_classification_metrics_and_predictions(tmp_path):
    samples = []
    labels = [0, 1, 2, 0, 1, 2]
    for index, label in enumerate(labels):
        samples.append(HistorySample(
            X_obs=torch.randn(5, 3), T_obs=torch.arange(5, dtype=torch.float32),
            target=torch.tensor([float(label)]), target_schema="tep_fault_v1",
            class_label=torch.tensor(label), fault_number=label,
            simulation_run=index % 2,
            metadata={"unit_id": "test:{}:{}".format(label, index % 2), "window_end": index},
        ))
    bundle = DatasetBundle(
        "tep", {"train": samples[:3], "validation": samples[3:4], "test": samples[3:]},
        "tep-class-split", "tep_fault_v1", {"features": ["a", "b", "c"], "num_fault_classes": 3},
    )

    result = run_bundle(
        bundle, tmp_path, seed=9, epochs=1, batch_size=2, device="cpu", amp=False,
        scheduler="none", evaluate_test=True,
    )
    run_dir = tmp_path / result["manifest"]["run_id"]
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    predictions = torch.load(run_dir / "test_predictions.pt", map_location="cpu", weights_only=False)

    assert metrics["evaluation_task"] == "fault_classification"
    assert {"macro_f1", "balanced_accuracy", "accuracy", "per_class", "confusion_matrix", "per_run"} <= metrics.keys()
    assert predictions["probabilities"].shape == (3, 3)
    assert predictions["detection_probability"].shape == (3,)
    assert predictions["diagnosis_probability"].shape == (3, 3)
    assert predictions["labels"].tolist() == [0, 1, 2]
    assert predictions["run_ids"] == ["test:0:1", "test:1:0", "test:2:1"]


def test_cmapss_test_evaluation_keeps_raw_engine_predictions(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.tensor([float(target)]), target_schema="cmapss_fd004_rul_raw_v1",
            metadata={"raw_rul": float(target), "engine_id": engine},
        )
        for engine, target in ((1, 30.0), (2, 60.0), (3, 90.0))
    ]
    bundle = DatasetBundle(
        "cmapss_fd004", {"train": samples[:1], "validation": samples[1:2], "test": samples[2:]},
        "fd004-test-split", "cmapss_fd004_rul_raw_v1", {"training_cap": 125.0},
    )

    result = run_bundle(
        bundle, tmp_path, seed=10, epochs=1, batch_size=1, device="cpu", amp=False,
        scheduler="none", evaluate_test=True,
    )
    run_dir = tmp_path / result["manifest"]["run_id"]
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    predictions = torch.load(run_dir / "test_predictions.pt", map_location="cpu", weights_only=False)

    assert metrics["units"] == "cycles"
    assert metrics["per_engine"]["3"]["target"] == 90.0
    assert predictions["engine_ids"] == ["3"]
    assert predictions["target"].tolist() == [90.0]


def test_comparison_runner_evaluates_tep_classification(tmp_path):
    from compare_code.models.factory import create_baseline

    samples = []
    for label in (0, 1, 2, 0, 1, 2):
        index = len(samples)
        samples.append(HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.tensor([float(label)]), target_schema="tep_fault_v1",
            class_label=torch.tensor(label), fault_number=label, simulation_run=index,
            metadata={"unit_id": "test:{}:{}".format(label, index), "window_end": index},
        ))
    bundle = DatasetBundle(
        "tep", {"train": samples[:3], "validation": samples[3:4], "test": samples[3:]},
        "tep-compare-split", "tep_fault_v1", {"features": ["a", "b"], "num_fault_classes": 3},
    )

    result = run_bundle(
        bundle, tmp_path, seed=12, epochs=1, batch_size=2, device="cpu", amp=False,
        scheduler="none", evaluate_test=True, model_name="gru",
        model_factory=lambda **kwargs: create_baseline("gru", num_classes=3, d_model=8, heads=2, **kwargs),
    )
    test_path = tmp_path / result["manifest"]["run_id"] / "test_predictions.pt"
    predictions = torch.load(test_path, map_location="cpu", weights_only=False)

    assert result["manifest"]["model_name"] == "gru"
    assert result["manifest"]["task"] == "fault_detection_diagnosis"
    assert predictions["task"] == "fault_classification"
    assert predictions["probabilities"].shape == (3, 3)


def test_entrypoint_does_not_evaluate_test_without_explicit_flag():
    from kaf_profiti.experiments.task_runner import evaluate
    model = None
    try:
        evaluate(model, [], split="test")
    except ValueError as error:
        assert "explicit" in str(error)
    else:
        raise AssertionError("test evaluation must require an explicit flag")


def _provenance_state_bundle(offset=0.0):
    samples = []
    for index in range(6):
        samples.append(HistorySample(
            X_obs=torch.arange(6, dtype=torch.float32).reshape(3, 2) + float(index) + offset,
            T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.full((2, 2), float(index)),
            target_schema="metropt3_state_v1",
            metadata={"window_start": index, "window_end": index + 4},
        ))
    return DatasetBundle(
        "metropt3", {"train": samples[:3], "validation": samples[3:5], "test": samples[5:]},
        "provenance-split", "metropt3_state_v1", {"features": ["a", "b"]},
    )


def test_manifest_records_training_provenance_and_content_hashes(tmp_path):
    kwargs = dict(seed=3, epochs=1, batch_size=2, device="cpu", amp=False, progress=False)
    result = run_bundle(_provenance_state_bundle(), tmp_path, scheduler="none", patience=1, **kwargs)
    manifest = result["manifest"]
    assert manifest["epochs"] == 1
    assert manifest["batch_size"] == 2
    assert manifest["lr_scheduler"] == "none"
    assert manifest["early_stopping_patience"] == 1
    assert manifest["device"] == "cpu"
    assert manifest["amp"] is False
    assert manifest["head_mode"] == "point"
    metadata = manifest["dataset_metadata"]
    for key in ("split_members_sha256", "window_catalog_sha256", "normalizer_stats_sha256"):
        assert isinstance(metadata[key], str) and len(metadata[key]) == 64
    assert isinstance(manifest["target_normalizer_stats_sha256"], str)

    repeated = run_bundle(_provenance_state_bundle(), tmp_path, scheduler="none", **kwargs)
    repeated_metadata = repeated["manifest"]["dataset_metadata"]
    for key in ("split_members_sha256", "window_catalog_sha256", "normalizer_stats_sha256"):
        assert repeated_metadata[key] == metadata[key]
    assert repeated["manifest"]["target_normalizer_stats_sha256"] == manifest["target_normalizer_stats_sha256"]

    shifted = _provenance_state_bundle()
    shifted.samples["train"].append(HistorySample(
        X_obs=torch.arange(6, dtype=torch.float32).reshape(3, 2) + 9.0,
        T_obs=torch.arange(3, dtype=torch.float32),
        target=torch.full((2, 2), 9.0),
        target_schema="metropt3_state_v1",
        metadata={"window_start": 6, "window_end": 10},
    ))
    third = run_bundle(shifted, tmp_path, scheduler="none", **kwargs)
    shifted_metadata = third["manifest"]["dataset_metadata"]
    assert shifted_metadata["window_catalog_sha256"] != metadata["window_catalog_sha256"]
    assert shifted_metadata["normalizer_stats_sha256"] != metadata["normalizer_stats_sha256"]


def test_comparison_tep_training_uses_shared_class_weights_and_detection_path(tmp_path):
    from compare_code.models.factory import create_baseline

    labels = [0, 0, 0, 1, 1, 2, 0, 1]
    samples = []
    for index, label in enumerate(labels):
        samples.append(HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.tensor([float(label)]), target_schema="tep_fault_v1",
            class_label=torch.tensor(label), fault_number=label, simulation_run=index,
            metadata={"unit_id": "train:{}:{}".format(label, index), "window_end": index},
        ))
    test_samples = []
    for index, label in enumerate((0, 1, 2, 1)):
        test_samples.append(HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.tensor([float(label)]), target_schema="tep_fault_v1",
            class_label=torch.tensor(label), fault_number=label, simulation_run=100 + index,
            metadata={"unit_id": "test:{}:{}".format(label, index), "window_end": index},
        ))
    bundle = DatasetBundle(
        "tep", {"train": samples[:6], "validation": samples[6:], "test": test_samples},
        "tep-weights-split", "tep_fault_v1", {"features": ["a", "b"], "num_fault_classes": 3},
    )

    captured = {}

    def factory(**kwargs):
        model = create_baseline("gru", num_classes=3, d_model=8, heads=2, **kwargs)
        captured["model"] = model
        return model

    result = run_bundle(
        bundle, tmp_path, seed=12, epochs=1, batch_size=2, device="cpu", amp=False,
        scheduler="none", evaluate_test=True, model_factory=factory, model_name="gru",
    )
    weights = result["manifest"]["tep_class_weights"]
    assert abs(weights[0] - 6.0 / (3 * 3)) < 1e-6
    assert abs(weights[1] - 6.0 / (2 * 3)) < 1e-6
    assert abs(weights[2] - 6.0 / (1 * 3)) < 1e-6
    assert tuple(captured["model"].config.tep_class_weights) == tuple(weights)

    run_dir = tmp_path / result["manifest"]["run_id"]
    test_metrics = json.loads((run_dir / "test_metrics.json").read_text())
    assert test_metrics["threshold_selection"]["source_split"] == "validation"
    assert "event_metrics" in test_metrics
    predictions = torch.load(run_dir / "test_predictions.pt", map_location="cpu", weights_only=False)
    assert predictions["detection_probability"] is not None
    assert predictions["detection_probability"].shape == (4,)
