"""CH34-S01-T05: MetroPT data-gate (learnability floors + risk labels) tests."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diagnostics"))

metropt_learnability = importlib.import_module("metropt_learnability")

_DATA_ROOT = Path(__import__("os").environ.get(
    "KST_DATA_ROOT", Path(__file__).resolve().parents[3] / "dataset"
))
_METROPT_CSV = _DATA_ROOT / "metropt+3+dataset" / "MetroPT3(AirCompressor).csv"
_requires_metropt = pytest.mark.skipif(
    not _METROPT_CSV.exists(), reason="MetroPT CSV not available under KST_DATA_ROOT"
)


# ---------------------------------------------------------------------------
# gate logic (pure, synthetic)
# ---------------------------------------------------------------------------


def _floor_entry(per_channel_mae):
    return {
        "std_micro": {"mae": float(np.mean(per_channel_mae))},
        "per_channel": {"mae": list(map(float, per_channel_mae))},
    }


def test_gate_requires_ten_percent_and_five_channels():
    zero = _floor_entry([1.0] * 7)
    # persistence improves every channel by 50% -> pass
    strong = _floor_entry([0.5] * 7)
    gate = metropt_learnability.evaluate_learnability_gate(
        {"zero": zero, "persistence": strong}, channel_count=7
    )
    assert gate["result"] == "pass"
    assert gate["best_predictor"] == "persistence"
    assert gate["best_improvement"] == pytest.approx(0.5)

    # improvement below 10% -> fail even with all channels improving
    weak = _floor_entry([0.95] * 7)
    gate = metropt_learnability.evaluate_learnability_gate(
        {"zero": zero, "persistence": weak}, channel_count=7
    )
    assert gate["result"] == "fail"

    # 10% exactly on one channel only (4/7 improved) -> fail on channel count
    mixed = _floor_entry([0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    gate = metropt_learnability.evaluate_learnability_gate(
        {"zero": zero, "persistence": mixed}, channel_count=7
    )
    assert gate["result"] == "fail"

    # 10%+ improvement but only 4 channels -> fail on channel count
    borderline = _floor_entry([0.89, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0])
    gate = metropt_learnability.evaluate_learnability_gate(
        {"zero": zero, "persistence": borderline}, channel_count=7
    )
    assert gate["improved_channels"] == 4
    assert gate["result"] == "fail"


# ---------------------------------------------------------------------------
# risk labels (synthetic timestamps vs fault intervals)
# ---------------------------------------------------------------------------


def _records_with_query_seconds(seconds_per_window):
    from kaf_profiti.industrial.metropt import WindowRecord

    records = []
    for index, seconds in enumerate(seconds_per_window):
        base = pd.Timestamp("2020-04-18 00:00:00").timestamp()
        query = tuple(base + float(s) for s in seconds)
        records.append(
            WindowRecord(
                segment_id=0, start=index, forecast_timestamp=query[0],
                query_timestamps=query, query_row_ids=tuple(range(index * 2, index * 2 + len(query))),
                window_id=f"w{index}",
            )
        )
    return records


def test_risk_labels_intersect_fault_interval_and_sha_stable():
    records = _records_with_query_seconds([(0, 60), (0, 600), (10 ** 6, 10 ** 6 + 60)])
    labels = metropt_learnability.compute_risk_labels(
        records, kaf_profiti_fault_windows()
    )
    assert labels.tolist() == [1, 1, 0]

    summary = metropt_learnability.risk_label_summary(
        {"train": labels}, kaf_profiti_fault_windows(), {"train": records}
    )
    assert summary["per_split"]["train"]["positives"] == 2
    assert summary["per_split"]["train"]["negatives"] == 1

    records_b = _records_with_query_seconds([(0, 60), (0, 600), (10 ** 6, 10 ** 6 + 60)])
    summary_b = metropt_learnability.risk_label_summary(
        {"train": metropt_learnability.compute_risk_labels(records_b, kaf_profiti_fault_windows())},
        kaf_profiti_fault_windows(), {"train": records_b},
    )
    assert summary["per_split"]["train"]["label_sha256"] == summary_b["per_split"]["train"]["label_sha256"]


def kaf_profiti_fault_windows():
    from kaf_profiti.industrial.metropt import METROPT_FAULT_WINDOWS

    return METROPT_FAULT_WINDOWS


def test_risk_evaluable_requires_both_classes():
    records = _records_with_query_seconds([(0, 60), (10 ** 6, 10 ** 6 + 60)])
    labels = metropt_learnability.compute_risk_labels(records, kaf_profiti_fault_windows())
    assert labels.tolist() == [1, 0]
    summary = metropt_learnability.risk_label_summary(
        {"valid": labels}, kaf_profiti_fault_windows(), {"valid": records}
    )
    assert summary["risk_evaluable"]["valid"] is True

    far_records = _records_with_query_seconds(
        [(10 ** 6, 10 ** 6 + 60), (10 ** 6 + 120, 10 ** 6 + 180)]
    )
    all_negative = metropt_learnability.compute_risk_labels(far_records, kaf_profiti_fault_windows())
    summary = metropt_learnability.risk_label_summary(
        {"valid": all_negative}, kaf_profiti_fault_windows(), {"valid": far_records}
    )
    assert summary["risk_evaluable"]["valid"] is False


# ---------------------------------------------------------------------------
# real protocol: determinism, leakage checks, finite metrics
# ---------------------------------------------------------------------------


@_requires_metropt
def test_real_data_gate_is_deterministic_and_leakage_free(tmp_path):
    arguments = dict(max_windows_per_split=120, history_len=168, pred_len=24, stride=60)
    payload_a = metropt_learnability.build_gate_payload(
        data_root=_DATA_ROOT, **arguments
    )
    payload_b = metropt_learnability.build_gate_payload(
        data_root=_DATA_ROOT, **arguments
    )

    assert payload_a["split_sha256"] == payload_b["split_sha256"]
    assert payload_a["floors"] == payload_b["floors"]
    assert payload_a["learnability_gate"] == payload_b["learnability_gate"]
    assert payload_a["risk_labels"]["per_split"] == payload_b["risk_labels"]["per_split"]

    assert all(payload_a["leakage_checks"].values()), payload_a["leakage_checks"]
    assert payload_a["finite"] is True
    assert payload_a["test_floor_role"] == "audit_only"
    # gate structure mirrors the acceptance contract
    assert set(payload_a["learnability_gate"]) >= {
        "best_predictor", "best_improvement", "improved_channels", "result",
    }


@_requires_metropt
def test_real_gate_writes_result_relative_json(tmp_path):
    metropt_learnability.write_gate_payload(
        data_root=_DATA_ROOT, result_root=tmp_path,
        max_windows_per_split=60, history_len=168, pred_len=24, stride=60,
    )
    out = tmp_path / "pilot" / "metropt3" / "diagnostics" / "data_gate.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "metropt3-data-gate-v1"
    assert payload["dataset"] == "metropt3_chrono_502030_v2"
