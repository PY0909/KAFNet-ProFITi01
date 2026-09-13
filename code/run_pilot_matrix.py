#!/usr/bin/env python
"""CLI for the unified FD004 pilot matrix runner (CH2.5-P03-T01).

Modes:
  dry-run  ordered scientific keys, model order, shared artifact SHAs,
           expected new-training count (default; writes nothing)
  smoke    one-batch per-model validation for a group, validated report under
           ``<result-root>/pilot/fd004/smoke/``; never produces test metrics
  full     full train/validation/test pilot runs (P04; baseline-first gate on)

All roots resolve through ``resolve_runtime_paths`` (CLI flag, then
``KST_DATA_ROOT``/``KST_RESULT_ROOT``, then repository-relative defaults); no
machine-specific path, host, port, or account is ever hardcoded here.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaf_profiti.experiments.pilot_runner import (  # noqa: E402
    PilotRunner,
    load_matrix,
    validate_smoke_report,
)
from kaf_profiti.experiments.runtime_paths import resolve_runtime_paths  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs" / "pilot" / "fd004"

_SMOKE_EXPECTED = {
    "point_baselines": 5,
    "probabilistic_baselines": 6,
    "ours": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified FD004 pilot matrix runner")
    parser.add_argument("--mode", choices=("dry-run", "smoke", "full"), default="dry-run")
    parser.add_argument("--matrix", choices=("point", "probabilistic", "all"), default="all")
    parser.add_argument(
        "--group",
        choices=tuple(_SMOKE_EXPECTED),
        default=None,
        help="smoke group to execute (smoke mode only)",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_matrices(selection: str):
    paths = []
    if selection in ("point", "all"):
        paths.append(_CONFIG_DIR / "point_matrix.yaml")
    if selection in ("probabilistic", "all"):
        paths.append(_CONFIG_DIR / "probabilistic_matrix.yaml")
    return [load_matrix(path) for path in paths]


def main() -> int:
    args = parse_args()
    paths = resolve_runtime_paths(args.data_root, args.result_root, {})
    runner = PilotRunner(
        matrices=_load_matrices(args.matrix),
        result_root=str(paths.output_root),
        data_root=str(paths.data_root),
        device=args.device,
    )

    if args.mode == "dry-run":
        print(json.dumps(runner.dry_run(), ensure_ascii=False, indent=2))
        return 0

    if args.mode == "smoke":
        groups = [args.group] if args.group else list(_SMOKE_EXPECTED)
        for group in groups:
            report = runner.run_smoke(group=group)
            counts = validate_smoke_report(report, expected_ready=_SMOKE_EXPECTED[group])
            print(
                json.dumps(
                    {"event": "smoke", "group": group, **counts, "report": str(
                        paths.output_root / "pilot" / "fd004" / "smoke" / f"{group}_smoke.json"
                    )},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return 0

    summary = runner.execute()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
