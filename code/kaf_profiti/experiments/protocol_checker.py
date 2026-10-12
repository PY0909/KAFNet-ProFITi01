"""Validate that model manifests use one comparable FD004 protocol."""

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


EXPECTED_MODELS = ("unified", "tcn", "gru", "patchtst", "timesnet", "itransformer")
PROTOCOL_FIELDS = (
    "dataset", "task", "seed", "split_identity", "target_schema",
    "normalizer_identity", "target_convention", "input_padding",
    "evaluator_version", "dataset_schema",
)


def _read_manifest(path: Path):
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        return None, str(error)
    if not isinstance(value, dict):
        return None, "manifest root must be an object"
    return value, None


def _default_paths(result_root: Path):
    return {
        model: result_root / "cmapss_fd004_{}_formal_v4_seed2026".format(model) / "manifest.json"
        for model in EXPECTED_MODELS
    }


def check_manifests(result_root: Path, manifest_paths: Mapping[str, Path] = None):
    """Return a protocol report without changing manifests or result artifacts."""
    paths = dict(manifest_paths or _default_paths(Path(result_root)))
    # Small synthetic runs and explicit audit bundles often use MODEL/manifest.json.
    # Keep the formal result naming as the primary convention, with this deterministic fallback.
    if manifest_paths is None:
        root = Path(result_root)
        for model in EXPECTED_MODELS:
            fallback = root / model / "manifest.json"
            if not paths[model].is_file() and fallback.is_file():
                paths[model] = fallback
    manifests = {}
    issues = []
    for model in EXPECTED_MODELS:
        path = Path(paths[model]) if model in paths else None
        if path is None or not path.is_file():
            issues.append({"code": "missing_manifest", "model": model,
                           "path": None if path is None else str(path)})
            continue
        manifest, error = _read_manifest(path)
        if error:
            issues.append({"code": "invalid_manifest", "model": model,
                           "path": str(path), "detail": error})
            continue
        manifests[model] = manifest
        if manifest.get("model_name") != model:
            issues.append({"code": "model_name_mismatch", "model": model,
                           "value": manifest.get("model_name")})
        convention = str(manifest.get("target_convention", ""))
        if "test_target=official_raw_rul" not in convention or "prediction=nonnegative_raw_cycles" not in convention:
            issues.append({"code": "invalid_protocol", "model": model,
                           "field": "target_convention", "value": convention})

    reference = next(iter(manifests.values()), None)
    if reference is not None:
        for model, manifest in manifests.items():
            for field in PROTOCOL_FIELDS:
                if manifest.get(field) != reference.get(field):
                    issues.append({"code": "field_mismatch", "model": model,
                                   "field": field, "value": manifest.get(field),
                                   "reference": reference.get(field)})
            metadata = manifest.get("dataset_metadata")
            reference_metadata = reference.get("dataset_metadata")
            for field in ("training_cap", "history_padding", "raw_rul_source"):
                value = metadata.get(field) if isinstance(metadata, dict) else None
                ref_value = reference_metadata.get(field) if isinstance(reference_metadata, dict) else None
                if value != ref_value:
                    issues.append({"code": "field_mismatch", "model": model,
                                   "field": "dataset_metadata." + field,
                                   "value": value, "reference": ref_value})
    return {
        "ok": not issues and len(manifests) == len(EXPECTED_MODELS),
        "models": list(EXPECTED_MODELS),
        "manifests": {model: str(paths[model]) for model in manifests},
        "issues": issues,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Check FD004 model manifest protocol consistency")
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--run-dir", action="append", metavar="MODEL=PATH",
                        help="explicit model manifest path or run directory; repeat per model")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional JSON report path")
    return parser


def _explicit_paths(entries: Sequence[str]):
    paths = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--run-dir must use MODEL=PATH")
        model, raw_path = entry.split("=", 1)
        if model not in EXPECTED_MODELS:
            raise ValueError("unsupported model: {}".format(model))
        path = Path(raw_path)
        paths[model] = path / "manifest.json" if path.is_dir() else path
    return paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = _explicit_paths(args.run_dir or [])
    report = check_manifests(args.result_root, paths or None)
    payload = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
