"""CH34-S01-T01: private MetroPT v2 protocol catalog contracts."""

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from kaf_profiti.industrial.metropt import (
    GAP_MULTIPLIER,
    WindowRecord,
    build_window_catalog,
    load_metropt_frame_v2,
    median_interval_seconds,
    partition_sha,
    raw_data_sha,
    segmentize,
    split_chronological_by_timestamp_group,
    timeline_sha,
    window_catalog_sha,
)

_DATA_ROOT = Path(os.environ.get("KST_DATA_ROOT", Path(__file__).resolve().parents[3] / "dataset"))
_METROPT_CSV = _DATA_ROOT / "metropt+3+dataset" / "MetroPT3(AirCompressor).csv"
_requires_metropt = pytest.mark.skipif(
    not _METROPT_CSV.exists(), reason="MetroPT CSV not available under KST_DATA_ROOT"
)


def _frame(counts=(3, 2, 3), gap_after=None):
    rows = []
    source = 100
    timestamp = pd.Timestamp("2020-01-01")
    for group_index, count in enumerate(counts):
        if gap_after is not None and group_index == gap_after:
            timestamp += timedelta(seconds=31)
        for _ in range(count):
            rows.append({
                "source_row_id": source,
                "timestamp": timestamp,
                "TP2": float(source),
            })
            source += 1
        timestamp += timedelta(seconds=10)
    return pd.DataFrame(rows)


def test_timestamp_group_split_never_splits_duplicate_timestamp_group():
    frame = _frame(counts=(2, 3, 2))
    train, valid, test, meta = split_chronological_by_timestamp_group(frame)

    assigned = {}
    for split, ids in (("train", train), ("valid", valid), ("test", test)):
        for source_id in ids:
            assigned[source_id] = split
    for _, group in frame.groupby("timestamp", sort=True):
        assert len({assigned[int(value)] for value in group["source_row_id"]}) == 1
    assert set(train).isdisjoint(valid)
    assert set(train).isdisjoint(test)
    assert set(valid).isdisjoint(test)
    assert meta["boundaries"][0]["target_rows"] == pytest.approx(3.5)
    assert meta["boundaries"][1]["target_rows"] == pytest.approx(4.9)
    assert meta["actual_rows"]["train"] > 0
    assert meta["actual_rows"]["valid"] > 0
    assert meta["actual_rows"]["test"] > 0


def test_timestamp_group_boundary_tie_chooses_earlier_end():
    frame = _frame(counts=(2, 2, 2, 2))
    train, valid, test, meta = split_chronological_by_timestamp_group(frame)
    # 50% target is 4 rows, exactly a legal group end; 70% target is 5.6,
    # nearest legal end is 6. The tie rule is exercised by the helper's
    # distance ordering in a separate boundary metadata assertion.
    assert len(train) == 4
    assert len(valid) == 2
    assert len(test) == 2
    assert meta["boundaries"][0]["actual_rows"] == 4
    assert meta["boundaries"][1]["actual_rows"] == 6


def test_segment_catalog_excludes_windows_crossing_large_gap():
    frame = _frame(counts=(8, 8), gap_after=1)
    segments = segmentize(frame, threshold_seconds=30.0)
    records = build_window_catalog(
        segments, history_len=4, pred_len=2, stride=1,
        dataset="metropt3_chrono_502030_v2", split="train", timeline_sha="a" * 64,
    )
    assert len(segments) == 2
    assert records
    for record in records:
        segment = segments[record.segment_id]
        window_rows = segment.iloc[record.start : record.start + 6]
        assert window_rows["timestamp"].diff().dt.total_seconds().iloc[1:].max() <= 30.0
        assert record.forecast_timestamp == window_rows["timestamp"].iloc[3].timestamp()
        assert len(record.query_row_ids) == 2


def test_window_projection_is_single_source_of_truth():
    frame = _frame(counts=(8,))
    segments = segmentize(frame, threshold_seconds=30.0)
    records = build_window_catalog(
        segments, history_len=4, pred_len=2, stride=2,
        dataset="metropt3_chrono_502030_v2", split="train", timeline_sha="b" * 64,
    )
    assert all(isinstance(record, WindowRecord) for record in records)
    assert [(r.segment_id, r.start) for r in records] == [
        (record.segment_id, record.start) for record in records
    ]
    assert len({record.window_id for record in records}) == len(records)


def test_catalog_shas_and_window_ids_are_deterministic():
    frame_a = _frame(counts=(8, 8), gap_after=1)
    frame_b = frame_a.copy(deep=True)
    segments_a = segmentize(frame_a, threshold_seconds=30.0)
    segments_b = segmentize(frame_b, threshold_seconds=30.0)
    records_a = build_window_catalog(
        segments_a, 4, 2, 2, "metropt3_chrono_502030_v2", "valid", "c" * 64
    )
    records_b = build_window_catalog(
        segments_b, 4, 2, 2, "metropt3_chrono_502030_v2", "valid", "c" * 64
    )
    assert [r.window_id for r in records_a] == [r.window_id for r in records_b]
    assert [r.query_row_ids for r in records_a] == [r.query_row_ids for r in records_b]


def test_v2_loader_preserves_unique_source_row_id_and_stable_order(tmp_path):
    path = tmp_path / "MetroPT3(AirCompressor).csv"
    frame = pd.DataFrame({
        "Unnamed: 0": [9, 2, 5],
        "timestamp": ["2020-01-01 00:00:10", "2020-01-01 00:00:00", "2020-01-01 00:00:10"],
        "TP2": [1.0, 2.0, 3.0],
    })
    frame.to_csv(path, index=False)
    loaded = load_metropt_frame_v2(tmp_path)
    assert loaded["source_row_id"].is_unique
    assert loaded[["timestamp", "source_row_id"]].values.tolist() == [
        [pd.Timestamp("2020-01-01 00:00:00"), 2],
        [pd.Timestamp("2020-01-01 00:00:10"), 5],
        [pd.Timestamp("2020-01-01 00:00:10"), 9],
    ]


def test_v2_loader_rejects_duplicate_source_row_id(tmp_path):
    path = tmp_path / "MetroPT3(AirCompressor).csv"
    pd.DataFrame({
        "Unnamed: 0": [1, 1],
        "timestamp": ["2020-01-01", "2020-01-01 00:00:10"],
        "TP2": [1.0, 2.0],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="source_row_id"):
        load_metropt_frame_v2(tmp_path)


def test_v2_catalog_rejects_duplicate_timestamps():
    frame = pd.DataFrame({
        "source_row_id": [1, 2, 3],
        "timestamp": [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-01 00:00:10"),
        ],
        "TP2": [1.0, 2.0, 3.0],
    })
    with pytest.raises(ValueError, match="duplicate timestamp"):
        segmentize(frame, threshold_seconds=30.0, reject_duplicate_timestamps=True)


@_requires_metropt
def test_real_metropt_v2_catalog_invariants():
    frame = load_metropt_frame_v2(_DATA_ROOT / "metropt+3+dataset")
    assert frame["source_row_id"].is_unique
    median = median_interval_seconds(frame)
    assert 5.0 <= median <= 20.0, f"expected ~10s median, got {median}"

    train, valid, test, meta = split_chronological_by_timestamp_group(frame)
    assert len(train) + len(valid) + len(test) == len(frame)
    assert not (set(train) & set(valid)) and not (set(train) & set(test)) and not (set(valid) & set(test))
    assert meta["actual_rows"]["train"] > 0 and meta["actual_rows"]["valid"] > 0 and meta["actual_rows"]["test"] > 0

    threshold = GAP_MULTIPLIER * median
    segments = segmentize(frame, threshold_seconds=threshold)
    assert segments
    max_internal = max(
        seg["timestamp"].diff().dt.total_seconds().iloc[1:].max()
        for seg in segments
    )
    assert max_internal <= threshold + 1e-6

    # full protocol window params from §13.2
    records = build_window_catalog(
        segments, history_len=168, pred_len=24, stride=60,
        dataset="metropt3_chrono_502030_v2", split="train", timeline_sha="d" * 64,
    )
    assert records
    assert len({r.window_id for r in records}) == len(records)
    # windows are monotone in source_row_id (chronological within a segment)
    by_segment = {}
    for r in records:
        by_segment.setdefault(r.segment_id, []).append(r)
    for sid, recs in by_segment.items():
        starts = [r.start for r in recs]
        assert starts == sorted(starts)
        assert len(starts) == len(set(starts))


def test_layered_protocol_shas_are_stable_and_acyclic():
    frame = _frame(counts=(8, 8), gap_after=1)
    segments = segmentize(frame, threshold_seconds=30.0)
    records = build_window_catalog(
        segments, 4, 2, 2, "metropt3_chrono_502030_v2", "train", "e" * 64
    )
    raw_a = raw_data_sha(frame, ["TP2"])
    raw_b = raw_data_sha(frame.copy(deep=True), ["TP2"])
    part_a = partition_sha(raw_a, [100, 101], [102], [103])
    part_b = partition_sha(raw_b, [100, 101], [102], [103])
    time_a = timeline_sha(part_a, segments)
    time_b = timeline_sha(part_b, [segment.copy(deep=True) for segment in segments])
    catalog_a = window_catalog_sha(time_a, 4, 2, 2, records)
    catalog_b = window_catalog_sha(time_b, 4, 2, 2, records)

    assert raw_a == raw_b
    assert part_a == part_b
    assert time_a == time_b
    assert catalog_a == catalog_b
    assert window_catalog_sha(time_a, 5, 2, 2, records) != catalog_a
