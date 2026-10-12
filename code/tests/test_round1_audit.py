import json

import torch

from kaf_profiti.experiments.round1_protocol_audit import audit_runs


def _write_run(root, run_id, *, dataset, task, model_name, head_mode="point",
               epochs=120, tep=False, split_sha="a" * 64, window_sha="b" * 64,
               normalizer_sha="c" * 64, evaluator="unified_evaluator_v4",
               tep_weights=(0.5, 1.0, 2.0), threshold_source="validation", history_len=80,
               metric_value=10.0, include=("manifest", "checkpoint", "metrics", "test_metrics", "efficiency")):
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "split_members_sha256": split_sha,
        "window_catalog_sha256": window_sha,
        "normalizer_stats_sha256": normalizer_sha,
        "history_len": history_len,
    }
    if dataset == "cmapss_fd004":
        metadata.update({"training_cap": 150, "sensor_filter": "standard",
                         "validation_mode": "all_windows"})
    if tep:
        metadata.update({"requested_max_runs_per_fault": 20, "selected_training_runs": 420})
    manifest = {
        "run_id": run_id, "dataset": dataset, "task": task, "seed": 2026,
        "split_identity": "split-x", "target_schema": "schema-x",
        "encoder_version": "unified_temporal_channelwise_v3c" if model_name == "unified" else "compare_{}_v1".format(model_name),
        "checkpoint": "checkpoint.pt", "dataset_schema": "ds-x",
        "normalizer_identity": "n-x", "target_convention": "raw",
        "evaluator_version": evaluator, "model_version": "v", "model_name": model_name,
        "epochs": epochs, "batch_size": 128, "lr_scheduler": "plateau",
        "early_stopping_patience": None, "device": "cuda", "amp": True,
        "head_mode": head_mode, "dataset_metadata": metadata,
    }
    if tep:
        manifest["tep_class_weights"] = list(tep_weights)
    test_metrics = {"split": "test", "mae": metric_value,
                    "threshold_selection": {"source_split": threshold_source}}
    if "manifest" in include:
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
    if "checkpoint" in include:
        torch.save({"model": {}, "config": {"head_mode": head_mode}, "best": {}}, run_dir / "checkpoint.pt")
    if "metrics" in include:
        (run_dir / "metrics.json").write_text(json.dumps({"test_evaluation_count": 1}))
    if "test_metrics" in include:
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics))
    if "efficiency" in include:
        (run_dir / "efficiency.json").write_text(json.dumps({"parameter_count": 10}))


def _base_runs(root):
    _write_run(root, "round1_fd004_unified_v3c_seed2026", dataset="cmapss_fd004", task="rul", model_name="unified")
    _write_run(root, "round1_metropt3_unified_v3c_seed2026", dataset="metropt3", task="state",
               model_name="unified", split_sha="d" * 64, window_sha="e" * 64, history_len=168)
    _write_run(root, "round1_tep_tcn_seed2026", dataset="tep", task="fault_detection_diagnosis",
               model_name="tcn", head_mode="probability", epochs=50, tep=True,
               split_sha="f" * 64, window_sha="9" * 64)
    _write_run(root, "round1_tep_unified_v3c_seed2026", dataset="tep", task="fault_detection_diagnosis",
               model_name="unified", head_mode="probability", epochs=50, tep=True,
               split_sha="f" * 64, window_sha="9" * 64)


def test_audit_passes_on_consistent_runs(tmp_path):
    _base_runs(tmp_path)
    report = audit_runs(tmp_path, [
        "round1_fd004_unified_v3c_seed2026", "round1_metropt3_unified_v3c_seed2026",
        "round1_tep_tcn_seed2026", "round1_tep_unified_v3c_seed2026",
    ])
    assert report["status"] == "pass", report
    assert report["tep_diagnostic_protocol"] == "max_runs_per_fault=20 (diagnostic subset)"


def test_audit_fails_on_missing_artifact(tmp_path):
    _base_runs(tmp_path)
    (tmp_path / "round1_fd004_unified_v3c_seed2026" / "efficiency.json").unlink()
    report = audit_runs(tmp_path, ["round1_fd004_unified_v3c_seed2026"])
    codes = {issue["code"] for issue in report["runs"]["round1_fd004_unified_v3c_seed2026"]["issues"]}
    assert "missing_artifact" in codes
    assert report["status"] == "fail"


def test_audit_fails_on_tep_head_mode_mismatch(tmp_path):
    _base_runs(tmp_path)
    _write_run(tmp_path, "round1_tep_unified_v3c_seed2026", dataset="tep", task="fault_detection_diagnosis",
               model_name="unified", head_mode="point", epochs=50, tep=True,
               split_sha="f" * 64, window_sha="9" * 64)
    report = audit_runs(tmp_path, ["round1_tep_unified_v3c_seed2026"])
    codes = {issue["code"] for issue in report["runs"]["round1_tep_unified_v3c_seed2026"]["issues"]}
    assert "tep_head_mode_not_probability" in codes


def test_audit_fails_on_cross_run_split_hash_mismatch(tmp_path):
    _base_runs(tmp_path)
    _write_run(tmp_path, "round1_tep_unified_v3c_seed2026", dataset="tep", task="fault_detection_diagnosis",
               model_name="unified", head_mode="probability", epochs=50, tep=True,
               split_sha="0" * 64, window_sha="9" * 64)
    report = audit_runs(tmp_path, ["round1_tep_tcn_seed2026", "round1_tep_unified_v3c_seed2026"])
    codes = {issue["code"] for issue in report["cross_run_issues"]}
    assert "cross_run_hash_mismatch" in codes
    assert report["status"] == "fail"


def test_audit_fails_on_threshold_source_and_nonfinite(tmp_path):
    _base_runs(tmp_path)
    _write_run(tmp_path, "round1_tep_tcn_seed2026", dataset="tep", task="fault_detection_diagnosis",
               model_name="tcn", head_mode="probability", epochs=50, tep=True,
               split_sha="f" * 64, window_sha="9" * 64, threshold_source="test", metric_value=float("nan"))
    report = audit_runs(tmp_path, ["round1_tep_tcn_seed2026"])
    codes = {issue["code"] for issue in report["runs"]["round1_tep_tcn_seed2026"]["issues"]}
    assert "tep_threshold_not_validation" in codes
    assert "nonfinite_metric" in codes


def test_audit_fails_on_missing_provenance_and_class_weight_drift(tmp_path):
    _base_runs(tmp_path)
    run_dir = tmp_path / "round1_tep_unified_v3c_seed2026"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest.pop("epochs")
    manifest["tep_class_weights"] = [1.0, 1.0, 1.0]
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    report = audit_runs(tmp_path, ["round1_tep_tcn_seed2026", "round1_tep_unified_v3c_seed2026"])
    codes = {issue["code"] for issue in report["runs"]["round1_tep_unified_v3c_seed2026"]["issues"]}
    assert "missing_provenance_field" in codes
    cross_codes = {issue["code"] for issue in report["cross_run_issues"]}
    assert "tep_class_weight_mismatch" in cross_codes
