import numpy as np
import pandas as pd

from kaf_profiti.industrial.datasets import (
    CMapssFD004,
    MetroPT3,
    TEPDataset,
)


def test_metropt_adapter_splits_before_windowing_and_keeps_future_target():
    frame = pd.DataFrame(
        {"timestamp": np.arange(12), "s1": np.arange(12, dtype=float), "s2": np.arange(12, dtype=float) + 1}
    )
    bundle = MetroPT3.from_frame(frame, history_len=3, pred_len=2, protocol="chrono_502030")
    assert bundle.split_identity.startswith("stable_unit_split_v1-")
    assert bundle.samples["train"]
    sample = bundle.samples["train"][0]
    assert sample.X_obs.shape == (3, 2)
    assert sample.target.shape == (2, 2)
    assert sample.target_schema == "metropt3_state_v1"


def test_metropt_adapter_accepts_datetime_timestamps():
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=12, freq="s"), "s1": np.arange(12.0)})
    bundle = MetroPT3.from_frame(frame, history_len=3, pred_len=2, protocol="chrono_502030")
    assert bundle.samples["train"][0].T_obs[0].item() == 0.0


def test_metropt_adapter_supports_window_stride():
    frame = pd.DataFrame({"timestamp": np.arange(20), "s1": np.arange(20.0)})
    dense = MetroPT3.from_frame(frame, history_len=3, pred_len=2, stride=1, protocol="chrono_502030")
    sparse = MetroPT3.from_frame(frame, history_len=3, pred_len=2, stride=2, protocol="chrono_502030")
    assert len(sparse.samples["train"]) < len(dense.samples["train"])


def test_metropt_official_adapter_keeps_test_after_first_month():
    timestamps = pd.date_range("2020-02-01", periods=12, freq="6D")
    frame = pd.DataFrame({"timestamp": timestamps, "s1": np.arange(12.0)})
    bundle = MetroPT3.from_frame(
        frame, history_len=1, pred_len=1, protocol="official_first_month",
        validation_fraction=0.4,
    )
    train_times = [sample.T_obs[-1].item() for sample in bundle.samples["train"]]
    validation_times = [sample.T_obs[-1].item() for sample in bundle.samples["validation"]]
    test_times = [sample.T_obs[-1].item() for sample in bundle.samples["test"]]

    assert train_times and validation_times and test_times
    assert max(train_times + validation_times) < min(test_times)
    assert bundle.metadata["protocol"] == "metropt3_official_first_month_v1"

def test_metropt_excludes_index_columns_and_rejects_nonfinite_features():
    frame = pd.DataFrame({"timestamp": np.arange(12), "Unnamed: 0": np.arange(12), "s1": np.arange(12.0)})
    bundle = MetroPT3.from_frame(frame, history_len=3, pred_len=2, protocol="chrono_502030")
    assert bundle.metadata["features"] == ["s1"]
    frame.loc[0, "s1"] = np.nan
    try:
        MetroPT3.from_frame(frame, history_len=3, pred_len=2, protocol="chrono_502030")
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite MetroPT values must fail")


def test_cmapss_adapter_reports_raw_rul_and_training_cap():
    train = np.array([[1, 0, 0.0, 1.0], [1, 1, 0.2, 1.2], [1, 2, 0.4, 1.4], [2, 0, 0.1, 1.1], [2, 1, 0.3, 1.3]])
    test = np.array([[3, 0, 0.5, 1.5], [3, 1, 0.7, 1.7], [3, 2, 0.9, 1.9]])
    rul = np.array([7.0])
    bundle = CMapssFD004.from_arrays(train, test, rul, history_len=2, training_cap=1.0)
    sample = bundle.samples["test"][0]
    assert sample.target.item() == 7.0
    assert sample.target_schema == "cmapss_fd004_rul_raw_v1"
    assert sample.metadata["raw_rul"] == 7.0
    assert sample.metadata["training_cap"] == 1.0


def test_cmapss_adapter_supports_uncapped_training_targets():
    train = np.array([[1, 0, 0.0], [1, 1, 1.0], [1, 2, 2.0], [1, 3, 3.0],
                      [2, 0, 0.0], [2, 1, 1.0], [2, 2, 2.0], [2, 3, 3.0]])
    test = np.array([[3, 0, 0.0], [3, 1, 1.0]])
    bundle = CMapssFD004.from_arrays(train, test, np.array([7.0]), history_len=2, training_cap=None)
    assert max(sample.target.item() for sample in bundle.samples["train"]) == 2.0
    assert bundle.metadata["training_cap"] is None
    assert "train_val_target=raw_rul" in bundle.metadata["target_convention"]


def test_cmapss_official_test_uses_one_final_window_per_engine():
    train = np.array([[1, 1, 0.0], [1, 2, 1.0], [1, 3, 2.0], [2, 1, 1.0], [2, 2, 2.0]])
    test = np.array([[3, 1, 0.0], [3, 2, 1.0], [3, 3, 2.0], [4, 1, 3.0], [4, 2, 4.0], [4, 3, 5.0]])
    bundle = CMapssFD004.from_arrays(train, test, np.array([7.0, 11.0]), history_len=2, training_cap=2.0)

    assert len(bundle.samples["test"]) == 2
    assert [sample.metadata["engine_id"] for sample in bundle.samples["test"]] == [3, 4]
    assert [sample.target.item() for sample in bundle.samples["test"]] == [7.0, 11.0]
    assert [sample.metadata["raw_rul"] for sample in bundle.samples["test"]] == [7.0, 11.0]
    assert [sample.T_obs[-1].item() for sample in bundle.samples["test"]] == [3.0, 3.0]


def test_cmapss_endpoint_validation_keeps_only_last_window_per_engine():
    train = np.array([[1, cycle, float(cycle)] for cycle in range(1, 7)] +
                     [[2, cycle, float(cycle + 2)] for cycle in range(1, 7)] +
                     [[3, cycle, float(cycle + 4)] for cycle in range(1, 7)])
    test = np.array([[4, cycle, float(cycle)] for cycle in range(1, 4)])
    bundle = CMapssFD004.from_arrays(
        train, test, np.array([7.0]), history_len=3, training_cap=125.0,
        validation_mode="endpoint",
    )
    validation_ids = [sample.metadata["engine_id"] for sample in bundle.samples["validation"]]
    assert validation_ids
    assert len(validation_ids) == len(set(validation_ids))
    assert bundle.metadata["validation_mode"] == "endpoint"
    assert [sample.target.item() for sample in bundle.samples["validation"]] == [3.0]


def test_cmapss_endpoint_validation_uses_observed_prefix_rul_not_failure_endpoint():
    train = np.array([[1, cycle, float(cycle)] for cycle in range(1, 7)] +
                     [[2, cycle, float(cycle + 2)] for cycle in range(1, 7)] +
                     [[3, cycle, float(cycle + 4)] for cycle in range(1, 7)])
    test = np.array([[4, cycle, float(cycle)] for cycle in range(1, 4)])
    bundle = CMapssFD004.from_arrays(
        train, test, np.array([7.0]), history_len=3, training_cap=125.0,
        validation_mode="endpoint", endpoint_horizon=2,
    )
    assert [sample.target.item() for sample in bundle.samples["validation"]] == [2.0]


def test_cmapss_tail_window_validation_uses_multiple_nonconstant_horizons():
    train = np.array([[1, cycle, float(cycle)] for cycle in range(1, 11)] +
                     [[2, cycle, float(cycle + 2)] for cycle in range(1, 11)] +
                     [[3, cycle, float(cycle + 4)] for cycle in range(1, 11)])
    test = np.array([[4, cycle, float(cycle)] for cycle in range(1, 4)])
    bundle = CMapssFD004.from_arrays(
        train, test, np.array([7.0]), history_len=3, training_cap=125.0,
        validation_mode="tail_windows", endpoint_horizon=4, endpoint_horizons=(2, 4, 6),
    )
    targets = [sample.target.item() for sample in bundle.samples["validation"]]
    assert len(targets) == 3
    assert sorted(set(targets)) == [2.0, 4.0, 6.0]
    assert bundle.metadata["validation_mode"] == "tail_windows"


def test_cmapss_short_official_test_engine_is_left_padded_to_history_length():
    train = np.array([[1, cycle, float(cycle)] for cycle in range(1, 8)] +
                     [[2, cycle, float(cycle + 2)] for cycle in range(1, 8)])
    test = np.array([[3, cycle, float(cycle + 4)] for cycle in range(1, 4)])
    bundle = CMapssFD004.from_arrays(train, test, np.array([9.0]), history_len=5, training_cap=125.0)
    sample = bundle.samples["test"][0]

    assert sample.X_obs.shape == (5, 1)
    assert sample.X_obs[:, 0].tolist() == [5.0, 5.0, 5.0, 6.0, 7.0]
    assert sample.T_obs.tolist() == [1.0, 1.0, 1.0, 2.0, 3.0]
    assert sample.metadata["history_padding"] == 2
    assert sample.M_obs is not None
    assert sample.M_obs[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_cmapss_full_length_windows_have_no_explicit_mask():
    train = np.array([[1, cycle, float(cycle)] for cycle in range(1, 8)] +
                     [[2, cycle, float(cycle + 2)] for cycle in range(1, 8)])
    test = np.array([[3, cycle, float(cycle + 4)] for cycle in range(1, 6)])
    bundle = CMapssFD004.from_arrays(train, test, np.array([9.0]), history_len=5, training_cap=125.0)
    sample = bundle.samples["test"][0]

    assert sample.metadata["history_padding"] == 0
    assert sample.M_obs is None


def _fd004_24col_rows(engine):
    # 24 列 FD004 结构：engine, cycle, setting1-3, sensor1-21；sensor 编号编码进取值
    rows = []
    for cycle in range(1, 6):
        features = [100.0, 200.0, 300.0] + [float(sensor) for sensor in range(1, 22)]
        rows.append([engine, cycle] + features)
    return np.array(rows)


def test_cmapss_standard_sensor_filter_keeps_14_degradation_channels():
    train = np.vstack([_fd004_24col_rows(engine) for engine in (1, 2, 3)])
    test = _fd004_24col_rows(4)
    bundle = CMapssFD004.from_arrays(
        train, test, np.array([9.0]), history_len=4, training_cap=125.0, sensor_filter="standard",
    )

    kept = bundle.metadata["features"]
    assert len(kept) == 14
    assert "setting1" not in kept and "sensor1" not in kept and "sensor19" not in kept
    assert "sensor2" in kept and "sensor21" in kept
    sample = bundle.samples["test"][0]
    assert sample.X_obs.shape == (4, 14)
    # sensor2 的原始取值是 2.0，应出现在保留列中
    assert 2.0 in sample.X_obs[0].tolist()
    assert bundle.metadata["feature_selection"].startswith("standard_drop")


def test_cmapss_standard_sensor_filter_rejects_non_24_column_schema():
    train = np.array([[1, cycle, float(cycle)] for cycle in range(1, 6)])
    test = np.array([[2, cycle, float(cycle)] for cycle in range(1, 6)])
    import pytest

    with pytest.raises(ValueError, match="24-column"):
        CMapssFD004.from_arrays(
            train, test, np.array([9.0]), history_len=4, training_cap=125.0, sensor_filter="standard",
        )


def test_tep_adapter_labels_fault_after_onset_and_preserves_run_identity():
    values = np.arange(20, dtype=float).reshape(10, 2)
    bundle = TEPDataset.from_array(values, fault_number=2, simulation_run=7, fault_start=6, history_len=3)
    assert bundle.samples["train"] == []
    assert bundle.samples["test"]
    labels = [sample.class_label.item() for sample in bundle.samples["test"]]
    assert labels[-1] == 2
    assert all(sample.metadata["unit_id"] == "2:7" for sample in bundle.samples["test"])

def test_tep_rejects_negative_fault_label():
    try:
        TEPDataset.from_array(np.ones((4, 2)), fault_number=-1, simulation_run=1, fault_start=1, history_len=2)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative TEP labels must fail")

def test_tep_run_split_uses_explicit_test_entities_and_keeps_entities_disjoint():
    runs = {}
    for fault in (0, 1):
        for run in range(5):
            split = "test" if run == 4 else None
            runs[(fault, run)] = {
                "values": np.ones((8, 2), dtype=float) * run,
                "fault_start": 4,
                "split": split,
            }
    bundle = TEPDataset.from_runs(runs, history_len=2, seed=17)
    unit_sets = {
        split: {sample.metadata["unit_id"] for sample in samples}
        for split, samples in bundle.samples.items()
    }
    assert all(unit_sets.values())
    assert unit_sets["train"].isdisjoint(unit_sets["validation"])
    assert unit_sets["train"].isdisjoint(unit_sets["test"])
    assert unit_sets["validation"].isdisjoint(unit_sets["test"])


def _dataverse_frame(fault, run, length=8, features=2):
    rows = []
    for sample in range(1, length + 1):
        rows.append({
            "faultNumber": fault,
            "simulationRun": run,
            "sample": sample,
            **{"xmeas_{}".format(i + 1): float(sample + i + run) for i in range(features)},
        })
    return pd.DataFrame(rows)


def test_tep_dataverse_frames_use_run_level_splits_and_explicit_test_runs():
    faulty_training = pd.concat([
        _dataverse_frame(1, 1), _dataverse_frame(1, 2),
    ], ignore_index=True)
    fault_free_training = pd.concat([
        _dataverse_frame(0, 1), _dataverse_frame(0, 2),
    ], ignore_index=True)
    faulty_testing = _dataverse_frame(1, 9)
    fault_free_testing = _dataverse_frame(0, 9)

    bundle = TEPDataset.from_dataverse_frames(
        faulty_training=faulty_training,
        fault_free_training=fault_free_training,
        faulty_testing=faulty_testing,
        fault_free_testing=fault_free_testing,
        history_len=3,
        window_stride=2,
        seed=2026,
    )

    unit_sets = {
        split: {sample.metadata["unit_id"] for sample in samples}
        for split, samples in bundle.samples.items()
    }
    assert bundle.metadata["source_format"] == "dataverse_repeated_run"
    assert unit_sets["train"].isdisjoint(unit_sets["validation"])
    assert unit_sets["test"] == {"test:0:9", "test:1:9"}
    assert all(sample.metadata["window_stride"] == 2 for samples in bundle.samples.values() for sample in samples)
    assert all(sample.metadata["fault_start"] == (160 if split == "test" else 20)
               for split, samples in bundle.samples.items() for sample in samples
               if sample.fault_number == 1)


def test_tep_dataverse_frames_reject_missing_run_columns():
    bad = pd.DataFrame({"faultNumber": [1], "sample": [1], "xmeas_1": [0.0]})
    try:
        TEPDataset.from_dataverse_frames(
            faulty_training=bad,
            fault_free_training=bad,
            faulty_testing=bad,
            fault_free_testing=bad,
            history_len=1,
        )
    except ValueError as error:
        assert "simulationRun" in str(error)
    else:
        raise AssertionError("dataverse TEP requires simulationRun")


def test_tep_dataverse_run_limit_is_numeric_not_lexicographic():
    frame = pd.concat([_dataverse_frame(1, 1), _dataverse_frame(1, 2), _dataverse_frame(1, 10)], ignore_index=True)
    records, _ = TEPDataset._dataverse_records(frame, "train", 20, max_runs_per_fault=2)
    assert {record["run"] for record in records} == {"1", "2"}


def test_tep_dataverse_run_limit_filters_before_validating_unselected_features():
    frame = pd.concat([_dataverse_frame(1, 1), _dataverse_frame(1, 2)], ignore_index=True)
    frame.loc[frame["simulationRun"] == 2, "xmeas_1"] = float("nan")

    records, _ = TEPDataset._dataverse_records(frame, "train", 20, max_runs_per_fault=1)

    assert {record["run"] for record in records} == {"1"}


def test_metropt_bundle_records_window_config_and_content_hashes():
    frame = pd.DataFrame(
        {"timestamp": np.arange(30), "s1": np.arange(30, dtype=float), "s2": np.arange(30, dtype=float) + 1}
    )
    first = MetroPT3.from_frame(frame, history_len=4, pred_len=2, stride=2, protocol="chrono_502030")
    second = MetroPT3.from_frame(frame, history_len=4, pred_len=2, stride=2, protocol="chrono_502030")
    assert first.metadata["history_len"] == 4
    assert first.metadata["pred_len"] == 2
    assert first.metadata["stride"] == 2
    for key in ("split_members_sha256", "window_catalog_sha256"):
        assert len(first.metadata[key]) == 64
        assert first.metadata[key] == second.metadata[key]
    wider = MetroPT3.from_frame(frame, history_len=4, pred_len=2, stride=3, protocol="chrono_502030")
    assert wider.metadata["window_catalog_sha256"] != first.metadata["window_catalog_sha256"]
    assert wider.metadata["split_members_sha256"] == first.metadata["split_members_sha256"]


def test_cmapss_bundle_records_window_config_and_content_hashes():
    rng = np.random.default_rng(5)
    train = np.column_stack([
        np.repeat(np.arange(1, 9), 20),
        np.tile(np.arange(1, 21), 8),
        rng.normal(size=(160, 2)),
    ])
    test = np.column_stack([
        np.repeat(np.arange(1, 3), 20),
        np.tile(np.arange(1, 21), 2),
        rng.normal(size=(40, 2)),
    ])
    rul = np.array([10.0, 20.0])
    first = CMapssFD004.from_arrays(train, test, rul, history_len=5, seed=7)
    second = CMapssFD004.from_arrays(train, test, rul, history_len=5, seed=7)
    assert first.metadata["history_len"] == 5
    assert first.metadata["sensor_filter"] == "none"
    assert first.metadata["split_seed"] == 7
    for key in ("split_members_sha256", "window_catalog_sha256"):
        assert first.metadata[key] == second.metadata[key]
    trimmed = CMapssFD004.from_arrays(train[:-20], test, rul, history_len=5, seed=7)
    assert trimmed.metadata["split_members_sha256"] != first.metadata["split_members_sha256"]
    assert trimmed.metadata["window_catalog_sha256"] != first.metadata["window_catalog_sha256"]


def test_tep_dataverse_bundle_records_requested_run_cap_and_content_hashes():
    def frame_for(fault, runs, length=12):
        rows = []
        for run in runs:
            for sample in range(1, length + 1):
                rows.append({"faultNumber": fault, "simulationRun": run, "sample": sample,
                             "x1": float(sample), "x2": float(sample) * 0.5})
        return pd.DataFrame(rows)

    faulty_training = pd.concat([frame_for(1, [1, 2, 3]), frame_for(2, [1, 2, 3])], ignore_index=True)
    fault_free_training = frame_for(0, [1, 2, 3])
    faulty_testing = pd.concat([frame_for(1, [1, 2]), frame_for(2, [1, 2])], ignore_index=True)
    fault_free_testing = frame_for(0, [1, 2])

    def build(max_runs):
        return TEPDataset.from_dataverse_frames(
            faulty_training, fault_free_training, faulty_testing, fault_free_testing,
            history_len=4, window_stride=3, seed=9, valid_fraction=0.25, max_runs_per_fault=max_runs,
        )

    first = build(2)
    second = build(2)
    assert first.metadata["history_len"] == 4
    assert first.metadata["requested_max_runs_per_fault"] == 2
    assert first.metadata["split_seed"] == 9
    assert first.metadata["valid_fraction"] == 0.25
    for key in ("split_members_sha256", "window_catalog_sha256"):
        assert first.metadata[key] == second.metadata[key]
    unbounded = build(None)
    assert unbounded.metadata["requested_max_runs_per_fault"] is None
    assert unbounded.metadata["window_catalog_sha256"] != first.metadata["window_catalog_sha256"]
