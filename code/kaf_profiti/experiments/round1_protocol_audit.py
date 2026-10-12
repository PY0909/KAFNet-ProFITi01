"""Round 1 artifact/protocol audit for the eight seed-2026 diagnostic runs.

Reads explicit run directories (the built-in protocol_checker only covers six
fixed FD004 manifests and cannot audit this round). The audit never modifies
``result/``; it prints a JSON report and exits non-zero on any issue.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

_DATASET_TASK = {
    "fd004": ("cmapss_fd004", "rul"),
    "metropt3": ("metropt3", "state"),
    "tep": ("tep", "fault_detection_diagnosis"),
}
_KNOWN_MODELS = ("unified_v3c", "tcn", "gru", "patchtst", "timesnet", "itransformer")
_ARTIFACTS = ("manifest.json", "checkpoint.pt", "metrics.json", "test_metrics.json", "efficiency.json")
_PROVENANCE_FIELDS = (
    "epochs", "batch_size", "lr_scheduler", "early_stopping_patience", "device", "amp", "head_mode",
)
_METADATA_FIELDS = (
    "split_members_sha256", "window_catalog_sha256", "normalizer_stats_sha256", "history_len",
)


def _walk_nonfinite(value, path=""):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_walk_nonfinite(item, "{}.{}".format(path, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_nonfinite(item, "{}[{}]".format(path, index)))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append(path)
    return found


def _parse_run_id(run_id):
    parts = run_id.split("_")
    if len(parts) < 4 or parts[0] != "round1" or not parts[-1].startswith("seed"):
        return None
    dataset, model, seed = parts[1], "_".join(parts[2:-1]), int(parts[-1][4:])
    if dataset not in _DATASET_TASK or model not in _KNOWN_MODELS:
        return None
    return dataset, model, seed


def _issue(code, **detail):
    return dict(code=code, **detail)


def audit_runs(result_root, run_ids):
    result_root = Path(result_root)
    runs = {}
    cross_issues = []
    for run_id in run_ids:
        issues = []
        parsed = _parse_run_id(run_id)
        if parsed is None:
            runs[run_id] = {"status": "fail", "issues": [_issue("unknown_run_id", run_id=run_id)]}
            continue
        dataset_key, model_token, seed = parsed
        dataset, task = _DATASET_TASK[dataset_key]
        run_dir = result_root / run_id
        missing = [name for name in _ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            issues.append(_issue("missing_artifact", files=missing))
            runs[run_id] = {"status": "fail", "issues": issues}
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        test_metrics = json.loads((run_dir / "test_metrics.json").read_text())
        if manifest.get("run_id") != run_id:
            issues.append(_issue("run_id_mismatch", expected=run_id, found=manifest.get("run_id")))
        if seed != 2026:
            issues.append(_issue("unexpected_seed", seed=seed))
        if manifest.get("dataset") != dataset or manifest.get("task") != task:
            issues.append(_issue("dataset_task_mismatch",
                                 expected=(dataset, task), found=(manifest.get("dataset"), manifest.get("task"))))
        expected_model_name = "unified" if model_token == "unified_v3c" else model_token
        if manifest.get("model_name") != expected_model_name:
            issues.append(_issue("model_name_mismatch", expected=expected_model_name, found=manifest.get("model_name")))
        if model_token == "unified_v3c" and "v3c" not in str(manifest.get("encoder_version", "")):
            issues.append(_issue("encoder_not_v3c", found=manifest.get("encoder_version")))
        for field in _PROVENANCE_FIELDS:
            if field not in manifest:
                issues.append(_issue("missing_provenance_field", field=field))
        metadata = manifest.get("dataset_metadata", {})
        for field in _METADATA_FIELDS:
            if field not in metadata:
                issues.append(_issue("missing_metadata_field", field=field))
        if dataset_key == "cmapss_fd004":
            for field, expected in (("training_cap", 150), ("sensor_filter", "standard"),
                                    ("validation_mode", "all_windows")):
                if metadata.get(field) != expected:
                    issues.append(_issue("fd004_protocol_field", field=field,
                                         expected=expected, found=metadata.get(field)))
        if dataset_key == "metropt3" and metadata.get("history_len") != 168:
            issues.append(_issue("metropt_history_len", expected=168, found=metadata.get("history_len")))
        if dataset_key == "tep":
            if manifest.get("head_mode") != "probability":
                issues.append(_issue("tep_head_mode_not_probability", found=manifest.get("head_mode")))
            if metadata.get("requested_max_runs_per_fault") != 20:
                issues.append(_issue("tep_run_cap_not_diagnostic_20",
                                     found=metadata.get("requested_max_runs_per_fault")))
            if not manifest.get("tep_class_weights"):
                issues.append(_issue("tep_class_weights_missing"))
            threshold = (test_metrics.get("threshold_selection") or {}).get("source_split")
            if threshold != "validation":
                issues.append(_issue("tep_threshold_not_validation", found=threshold))
        if metrics.get("test_evaluation_count") != 1:
            issues.append(_issue("test_evaluation_count", found=metrics.get("test_evaluation_count")))
        nonfinite = _walk_nonfinite(test_metrics)
        if nonfinite:
            issues.append(_issue("nonfinite_metric", fields=nonfinite[:10]))
        runs[run_id] = {
            "status": "fail" if issues else "pass",
            "issues": issues,
            "dataset": dataset,
            "model_name": manifest.get("model_name"),
        }

    loaded = [(run_id, info) for run_id, info in runs.items() if info.get("dataset")]
    by_dataset = defaultdict(list)
    for run_id, info in loaded:
        by_dataset[run_id.split("_")[1]].append(run_id)

    def field_of(run_id, getter):
        manifest = json.loads((result_root / run_id / "manifest.json").read_text())
        return getter(manifest)

    for dataset_key, group in sorted(by_dataset.items()):
        if len(group) < 2:
            continue
        getters = {
            "split_identity": lambda m: m.get("split_identity"),
            "split_members_sha256": lambda m: m.get("dataset_metadata", {}).get("split_members_sha256"),
            "window_catalog_sha256": lambda m: m.get("dataset_metadata", {}).get("window_catalog_sha256"),
            "normalizer_stats_sha256": lambda m: m.get("dataset_metadata", {}).get("normalizer_stats_sha256"),
            "target_schema": lambda m: m.get("target_schema"),
            "dataset_schema": lambda m: m.get("dataset_schema"),
        }
        if dataset_key == "tep":
            getters["tep_class_weights"] = lambda m: m.get("tep_class_weights")
        for field, getter in getters.items():
            values = {run_id: field_of(run_id, getter) for run_id in group}
            if len(set(map(json.dumps, values.values()))) > 1:
                cross_issues.append(_issue("cross_run_hash_mismatch" if "sha256" in field or field == "split_identity"
                                           else "tep_class_weight_mismatch" if field == "tep_class_weights"
                                           else "cross_run_field_mismatch",
                                           dataset=dataset_key, field=field, values=values))
    evaluators = {field_of(run_id, lambda m: m.get("evaluator_version")) for run_id, _ in loaded}
    if len(evaluators) > 1:
        cross_issues.append(_issue("cross_run_field_mismatch", dataset="all", field="evaluator_version",
                                   values=sorted(str(item) for item in evaluators)))

    report = {
        "status": "pass" if all(info["status"] == "pass" for info in runs.values()) and not cross_issues else "fail",
        "runs": runs,
        "cross_run_issues": cross_issues,
    }
    if "tep" in by_dataset:
        report["tep_diagnostic_protocol"] = "max_runs_per_fault=20 (diagnostic subset)"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit Round 1 seed-2026 run artifacts")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-id", action="append", required=True,
                        help="run id to audit; repeat once per run")
    args = parser.parse_args(argv)
    report = audit_runs(args.result_root, args.run_id)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
