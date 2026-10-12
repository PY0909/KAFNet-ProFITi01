import json

from kaf_profiti.experiments.protocol_checker import (
    EXPECTED_MODELS,
    check_manifests,
)


def _manifest(model="unified", **overrides):
    value = {
        "dataset": "cmapss_fd004",
        "task": "rul",
        "model_name": model,
        "seed": 2026,
        "split_identity": "stable_unit_split_v1-units249",
        "target_schema": "cmapss_fd004_rul_raw_v1",
        "normalizer_identity": "train-zscore-features24",
        "target_convention": "train_val_targets=min(raw_rul,125.0); test_target=official_raw_rul; prediction=nonnegative_raw_cycles",
        "input_padding": "repeat_first_observation_left_padding",
        "evaluator_version": "unified_evaluator_v4",
        "dataset_schema": ["setting1", "setting2", "setting3", "sensor1"],
        "dataset_metadata": {"training_cap": 125.0},
    }
    value.update(overrides)
    return value


def test_checker_accepts_six_matching_manifests(tmp_path):
    for model in EXPECTED_MODELS:
        path = tmp_path / model / "manifest.json"
        path.parent.mkdir()
        path.write_text(json.dumps(_manifest(model)), encoding="utf-8")

    report = check_manifests(tmp_path)

    assert report["ok"] is True
    assert report["models"] == list(EXPECTED_MODELS)
    assert report["issues"] == []


def test_checker_reports_missing_model_and_conflicting_protocol(tmp_path):
    for model in EXPECTED_MODELS[:-1]:
        path = tmp_path / model / "manifest.json"
        path.parent.mkdir()
        fields = {"seed": 7} if model == "gru" else {}
        path.write_text(json.dumps(_manifest(model, **fields)), encoding="utf-8")

    report = check_manifests(tmp_path)

    assert report["ok"] is False
    assert any(issue["code"] == "missing_manifest" for issue in report["issues"])
    assert any(issue["code"] == "field_mismatch" and issue["field"] == "seed"
               for issue in report["issues"])


def test_checker_rejects_capped_test_target_convention(tmp_path):
    for model in EXPECTED_MODELS:
        path = tmp_path / model / "manifest.json"
        path.parent.mkdir()
        convention = "train_val_targets=min(raw_rul,125.0); test_target=min(raw_rul,125.0)"
        path.write_text(json.dumps(_manifest(model, target_convention=convention)), encoding="utf-8")

    report = check_manifests(tmp_path)

    assert report["ok"] is False
    assert any(issue["code"] == "invalid_protocol" for issue in report["issues"])
