"""Native-protocol adapters that produce history-only samples and task targets."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from .batch import HistorySample
from .protocol import split_identity
from .splits import split_cmapss_engines, split_metropt_first_month, split_metropt_timestamps, split_tep_runs


@dataclass
class DatasetBundle:
    dataset: str
    samples: Dict[str, List[HistorySample]]
    split_identity: str
    target_schema: str
    metadata: Dict[str, object]


def _content_sha256(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _split_member_keys(samples):
    """Entity membership of one split: unit/engine ids when present, otherwise
    the split's timeline bounds (MetroPT has no sub-timeline entities)."""
    metadata = [sample.metadata or {} for sample in samples]
    unit_keys = sorted({str(item["unit_id"]) for item in metadata if "unit_id" in item})
    if unit_keys:
        return unit_keys
    engine_keys = sorted({str(item["engine_id"]) for item in metadata if "engine_id" in item})
    if engine_keys:
        return engine_keys
    bounds = [
        (item["window_start"], item["window_end"])
        for item in metadata
        if "window_start" in item and "window_end" in item
    ]
    if bounds:
        return ["timeline:{}:{}".format(min(start for start, _ in bounds), max(end for _, end in bounds))]
    return ["positional:{}".format(len(samples))]


def _window_catalog_entries(samples):
    entries = []
    for index, sample in enumerate(samples):
        metadata = sample.metadata or {}
        if "unit_id" in metadata and "window_end" in metadata:
            entries.append(["unit", str(metadata["unit_id"]), int(metadata["window_end"])])
        elif "engine_id" in metadata and "raw_rul" in metadata:
            entries.append(["engine", str(metadata["engine_id"]), float(metadata["raw_rul"])])
        elif "window_start" in metadata and "window_end" in metadata:
            entries.append(["window", int(metadata["window_start"]), int(metadata["window_end"])])
        else:
            entries.append(["index", index])
    entries.sort()
    return entries


def bundle_content_provenance(samples, members=None):
    """Content-level identity of split membership and the window catalog.

    ``split_identity`` only counts units, so two different engine/run sets of
    equal size share an identity string. These hashes cover the actual member
    sets and every window identity, making silent split or window drift
    auditable from the manifest alone. Adapters may pass explicit ``members``
    (e.g. MetroPT row intervals) when the split membership is independent of
    the window grid.
    """
    if members is None:
        members = {split: _split_member_keys(samples.get(split, []))
                   for split in ("train", "validation", "test")}
    catalog = {split: _window_catalog_entries(samples.get(split, []))
               for split in ("train", "validation", "test")}
    return {
        "split_members_sha256": _content_sha256(members),
        "window_catalog_sha256": _content_sha256(catalog),
    }


def _window_sample(values, times, start, history_len, target=None, target_schema="", **kwargs):
    x = torch.as_tensor(values[start : start + history_len], dtype=torch.float32)
    t = torch.as_tensor(times[start : start + history_len], dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(t).all():
        raise ValueError("dataset window contains non-finite values")
    sample_kwargs = {key: kwargs[key] for key in ("class_label", "fault_number", "simulation_run") if key in kwargs}
    metadata = {key: value for key, value in kwargs.items() if key not in sample_kwargs}
    return HistorySample(
        X_obs=x,
        T_obs=t,
        target=None if target is None else torch.as_tensor(target, dtype=torch.float32),
        target_schema=target_schema,
        metadata=metadata or None,
        **sample_kwargs,
    )


class MetroPT3:
    @staticmethod
    def from_frame(frame: pd.DataFrame, history_len=168, pred_len=24, timestamp="timestamp", feature_columns=None,
                   stride=1, protocol="official_first_month", validation_fraction=0.2):
        if timestamp not in frame or history_len < 1 or pred_len < 1 or stride < 1:
            raise ValueError("MetroPT requires timestamp and positive window lengths")
        ordered = frame.sort_values(timestamp, kind="mergesort").reset_index(drop=True)
        if protocol == "official_first_month":
            split_positions = split_metropt_first_month(ordered[timestamp].tolist(), validation_fraction)
            protocol_identity = "metropt3_official_first_month_v1"
            split_algorithm = "{}_val{}pct".format(protocol_identity, int(round(validation_fraction * 100)))
        elif protocol == "chrono_502030":
            split_positions = split_metropt_timestamps(ordered[timestamp].tolist())
            protocol_identity = "metropt3_chrono_502030_v1"
            split_algorithm = "stable_unit_split_v1"
        else:
            raise ValueError("unsupported MetroPT protocol: {}".format(protocol))
        samples = {"train": [], "validation": [], "test": []}
        features = feature_columns or [column for column in frame.columns if column != timestamp and not str(column).startswith("Unnamed:")]
        values = ordered[features].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("MetroPT feature values must be finite")
        raw_times = ordered[timestamp]
        if not pd.api.types.is_numeric_dtype(raw_times):
            parsed_times = pd.to_datetime(raw_times, errors="raise")
            times = (parsed_times - parsed_times.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32)
        else:
            times = raw_times.to_numpy(dtype=np.float32)
        for split, positions in split_positions.items():
            positions = sorted(positions)
            if not positions:
                continue
            lo, hi = positions[0], positions[-1] + 1
            for start in range(lo, hi - history_len - pred_len + 1, stride):
                target = values[start + history_len : start + history_len + pred_len]
                samples[split].append(_window_sample(
                    values, times, start, history_len, target, "metropt3_state_v1",
                    window_start=start, window_end=start + history_len + pred_len - 1,
                ))
        # 划分成员 = 各 split 的行区间与行数，与窗口网格（stride/history）无关，
        # 因此改变 stride 只影响 window_catalog 哈希，不改变 split_members 哈希。
        explicit_members = {split: [] for split in ("train", "validation", "test")}
        for split, positions in split_positions.items():
            if positions:
                explicit_members[split] = [
                    "rows:{}:{}:{}".format(min(positions), max(positions) + 1, len(positions))
                ]
        return DatasetBundle(
            "metropt3", samples,
            split_identity(ordered[timestamp].tolist(), algorithm=split_algorithm),
            "metropt3_state_v1",
            {"features": features, "timestamp": timestamp, "protocol": protocol_identity,
             "validation_fraction": validation_fraction if protocol == "official_first_month" else None,
             "history_len": history_len, "pred_len": pred_len, "stride": stride,
             **bundle_content_provenance(samples, members=explicit_members)},
        )


class CMapssFD004:
    _FD004_STANDARD_DROP = {
        # 24 列 = setting1-3 + sensor1-21；标准筛选剔除 3 个工况设定列和 7 个
        # 近常数/纯噪声传感器通道（文献常用配置，保留 14 个退化相关传感器）。
        "setting1", "setting2", "setting3",
        "sensor1", "sensor5", "sensor6", "sensor10", "sensor16", "sensor18", "sensor19",
    }

    @staticmethod
    def from_arrays(train, test, official_rul, history_len=50, training_cap=125.0, seed=2026,
                    validation_mode="all_windows", endpoint_horizon=20, sensor_filter=None,
                    endpoint_horizons=None):
        train = np.asarray(train, dtype=float)
        test = np.asarray(test, dtype=float)
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] < 3:
            raise ValueError("C-MAPSS arrays must contain engine, cycle and features")
        if validation_mode not in {"all_windows", "endpoint", "tail_windows"}:
            raise ValueError("validation_mode must be all_windows, endpoint, or tail_windows")
        if endpoint_horizon < 1:
            raise ValueError("endpoint_horizon must be positive")
        if endpoint_horizons is None:
            endpoint_horizons = (20, 40, 60, 80)
        endpoint_horizons = tuple(sorted({int(value) for value in endpoint_horizons if int(value) > 0}))
        if not endpoint_horizons:
            raise ValueError("endpoint_horizons must contain positive values")
        if sensor_filter not in (None, "standard"):
            raise ValueError("sensor_filter must be None or standard")
        feature_count = train.shape[1] - 2
        if sensor_filter == "standard" and feature_count != 24:
            raise ValueError("standard sensor filter requires the 24-column FD004 schema")
        feature_names = (
            ["setting1", "setting2", "setting3"] + ["sensor{}".format(index) for index in range(1, 22)]
            if feature_count == 24
            else ["feature{}".format(index) for index in range(1, feature_count + 1)]
        )
        if sensor_filter == "standard":
            keep_columns = np.array(
                [index for index, name in enumerate(feature_names) if name not in CMapssFD004._FD004_STANDARD_DROP],
                dtype=int,
            )
            selected_features = [feature_names[index] for index in keep_columns]
        else:
            keep_columns = None
            selected_features = feature_names
        train_ids = split_cmapss_engines(np.unique(train[:, 0]).astype(int), seed=seed)
        samples = {"train": [], "validation": [], "test": []}
        partitions = (
            ("train", train, train_ids["train"], None),
            ("validation", train, train_ids["validation"], None),
            ("test", test, np.unique(test[:, 0]).astype(int), np.asarray(official_rul, dtype=float)),
        )
        for split, source, split_ids, rul_values in partitions:
            for engine_index, engine_id in enumerate(split_ids):
                rows = source[source[:, 0] == engine_id]
                rows = rows[np.argsort(rows[:, 1])]
                max_cycle = rows[:, 1].max()
                remaining = 0.0 if source is test else 0.0
                if source is test:
                    remaining = float(rul_values[engine_index])
                if source is test:
                    ends = [len(rows)]
                elif split == "validation" and validation_mode == "endpoint":
                    ends = [max(history_len, len(rows) - endpoint_horizon)]
                elif split == "validation" and validation_mode == "tail_windows":
                    ends = sorted({
                        max(history_len, len(rows) - horizon)
                        for horizon in endpoint_horizons
                    })
                else:
                    ends = range(history_len, len(rows) + 1)
                for end in ends:
                    raw_rul = float(remaining if source is test else max_cycle - rows[end - 1, 1])
                    target = (min(raw_rul, training_cap) if training_cap is not None else raw_rul) if source is not test else raw_rul
                    start = max(0, end - history_len)
                    history_values = rows[start:end, 2:]
                    if keep_columns is not None:
                        history_values = history_values[:, keep_columns]
                    history_times = rows[start:end, 1]
                    padding = history_len - len(history_values)
                    if padding > 0:
                        history_values = np.concatenate((
                            np.repeat(history_values[:1], padding, axis=0), history_values,
                        ))
                        history_times = np.concatenate((
                            np.repeat(history_times[:1], padding), history_times,
                        ))
                    sample = _window_sample(
                        history_values, history_times, 0, history_len, [target],
                        "cmapss_fd004_rul_raw_v1", raw_rul=raw_rul,
                        training_cap=training_cap, target_split=split, engine_id=int(engine_id),
                        history_padding=padding,
                        condition_key="|".join("{:.3f}".format(float(value)) for value in rows[end - 1, 2:5])
                        if rows.shape[1] >= 5 else "global",
                    )
                    if padding > 0:
                        # 左填充位置是重复的首行观测，不是真实测量，必须在掩码中显式置 0，
                        # 否则编码器会把填充平台当作真实退化信号。
                        mask = torch.ones(history_len, history_values.shape[1])
                        mask[:padding] = 0.0
                        sample = replace(sample, M_obs=mask)
                    samples[split].append(sample)
        target_convention = (
            "train_val_targets=min(raw_rul,{}); test_target=official_raw_rul; prediction=nonnegative_raw_cycles".format(training_cap)
            if training_cap is not None
            else "train_val_target=raw_rul; test_target=official_raw_rul; prediction=nonnegative_raw_cycles"
        )
        return DatasetBundle(
            "cmapss_fd004", samples,
            split_identity(sorted(map(int, np.unique(train[:, 0])))),
            "cmapss_fd004_rul_raw_v1",
            {
                "features": selected_features,
                "schema": "engine_id,cycle,setting1-3,sensor1-21",
                "feature_selection": "none" if sensor_filter is None else "standard_drop_settings_and_sensor1,5,6,10,16,18,19",
                "training_cap": training_cap,
                "raw_rul_source": "official_rul",
                "history_padding": "repeat_first_observation_left_padding",
                "target_convention": target_convention,
                "validation_mode": validation_mode,
                "validation_endpoint_horizons": list(endpoint_horizons),
                "history_len": history_len,
                "sensor_filter": "none" if sensor_filter is None else sensor_filter,
                "split_seed": int(seed),
                **bundle_content_provenance(samples),
            },
        )

    @staticmethod
    def from_files(root, history_len=50, training_cap=125.0, seed=2026, validation_mode="all_windows",
                   endpoint_horizon=20, sensor_filter=None, endpoint_horizons=None):
        root = Path(root)
        train = np.loadtxt(root / "train_FD004.txt")
        test = np.loadtxt(root / "test_FD004.txt")
        rul = np.loadtxt(root / "RUL_FD004.txt")
        return CMapssFD004.from_arrays(
            train, test, rul, history_len, training_cap, seed, validation_mode, endpoint_horizon,
            sensor_filter=sensor_filter, endpoint_horizons=endpoint_horizons,
        )


class TEPDataset:
    _DATAVERSE_META_COLUMNS = {"faultNumber", "simulationRun", "sample"}

    @staticmethod
    def _run_sort_key(run):
        try:
            return (0, float(run))
        except (TypeError, ValueError):
            return (1, str(run))

    @staticmethod
    def _dataverse_records(frame, source_split, fault_start, max_runs_per_fault=None):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("TEP dataverse input must be a pandas DataFrame")
        required = {"faultNumber", "simulationRun", "sample"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError("TEP dataverse frame missing columns: {}".format(", ".join(missing)))
        feature_columns = [column for column in frame.columns if column not in TEPDataset._DATAVERSE_META_COLUMNS]
        if not feature_columns:
            raise ValueError("TEP dataverse frame must contain process feature columns")
        selected = frame
        if max_runs_per_fault is not None:
            if max_runs_per_fault < 1:
                raise ValueError("max_runs_per_fault must be positive when provided")
            selected_runs = {}
            for fault, group in frame.groupby("faultNumber", sort=False):
                runs = {
                    str(int(run)) if float(run).is_integer() else str(run)
                    for run in group["simulationRun"].drop_duplicates().tolist()
                }
                selected_runs[int(fault)] = set(sorted(runs, key=TEPDataset._run_sort_key)[:max_runs_per_fault])
            fault_values = frame["faultNumber"].to_numpy()
            run_values = frame["simulationRun"].to_numpy()
            row_mask = np.zeros(len(frame), dtype=bool)
            for fault, runs in selected_runs.items():
                raw_runs = [run for run in frame.loc[fault_values == fault, "simulationRun"].drop_duplicates().tolist()
                            if (str(int(run)) if float(run).is_integer() else str(run)) in runs]
                row_mask |= (fault_values == fault) & np.isin(run_values, raw_runs)
            selected = frame.loc[row_mask]

        values = selected[feature_columns]
        if not all(pd.api.types.is_numeric_dtype(values[column]) for column in feature_columns):
            raise ValueError("TEP dataverse process features must be numeric")
        if not np.isfinite(values.to_numpy(dtype=np.float32)).all():
            raise ValueError("TEP dataverse process features must be finite")
        records = []
        grouped = selected.sort_values(["faultNumber", "simulationRun", "sample"], kind="mergesort").groupby(
            ["faultNumber", "simulationRun"], sort=True
        )
        for (fault, run), group in grouped:
            fault = int(fault)
            run = str(int(run)) if float(run).is_integer() else str(run)
            records.append({
                "fault": fault,
                "run": run,
                "source_split": source_split,
                "fault_start": (fault_start if fault > 0 else 10**9),
                "times": group["sample"].to_numpy(dtype=np.float32),
                "values": group[feature_columns].to_numpy(dtype=np.float32),
            })
        return records, feature_columns

    @staticmethod
    def _from_dataverse_records(
        records, feature_columns, history_len, window_stride, seed, valid_fraction,
        require_test=True, requested_max_runs_per_fault=None,
    ):
        if not records:
            raise ValueError("TEP dataverse contains no runs")
        if history_len < 1 or window_stride < 1:
            raise ValueError("TEP history_len and window_stride must be positive")
        training_records = [record for record in records if record["source_split"] == "train"]
        testing_records = [record for record in records if record["source_split"] == "test"]
        by_fault = {}
        for record in training_records:
            by_fault.setdefault(record["fault"], []).append(record["run"])
        layers = split_tep_runs(by_fault, seed=seed, valid_fraction=valid_fraction)
        samples = {"train": [], "validation": [], "test": []}
        for record in records:
            values = record["values"]
            times = record["times"]
            if len(values) < history_len:
                continue
            if record["source_split"] == "test":
                split = "test"
            else:
                split = "train" if record["run"] in layers[record["fault"]]["train"] else "validation"
            unit_id = "{}:{}:{}".format(record["source_split"], record["fault"], record["run"])
            for start in range(0, len(values) - history_len + 1, window_stride):
                end = start + history_len
                label = record["fault"] if end - 1 >= record["fault_start"] else 0
                samples[split].append(_window_sample(
                    values, times, start, history_len, [label], "tep_fault_v1",
                    class_label=torch.tensor(label), fault_number=record["fault"],
                    simulation_run=int(float(record["run"])) if record["run"].replace(".", "", 1).isdigit() else None,
                    unit_id=unit_id, window_end=end - 1, fault_start=record["fault_start"],
                    source_split=record["source_split"], window_stride=window_stride,
                ))
        required_splits = ("train", "validation", "test") if require_test else ("train", "validation")
        if any(not samples[name] for name in required_splits):
            raise ValueError("TEP dataverse split must produce non-empty {} windows".format(
                ", ".join(required_splits)
            ))
        return DatasetBundle(
            "tep", samples, split_identity(sorted(sample.metadata["unit_id"] for values in samples.values() for sample in values)),
            "tep_fault_v1", {
                "features": feature_columns,
                "num_fault_classes": max(record["fault"] for record in records) + 1,
                "source_format": "dataverse_repeated_run",
                "window_stride": window_stride,
                "training_fault_start": 20,
                "testing_fault_start": 160,
                "selected_training_runs": len(training_records),
                "selected_testing_runs": len(testing_records),
                "history_len": history_len,
                "split_seed": int(seed),
                "valid_fraction": valid_fraction,
                "requested_max_runs_per_fault": (
                    None if requested_max_runs_per_fault is None else int(requested_max_runs_per_fault)
                ),
                **bundle_content_provenance(samples),
            }
        )

    @staticmethod
    def from_dataverse_frames(
        faulty_training, fault_free_training, faulty_testing, fault_free_testing,
        history_len=96, window_stride=12, seed=2026, valid_fraction=0.2,
        max_runs_per_fault=None, include_test=True,
    ):
        records = []
        feature_columns = None
        for frame, source, onset in (
            (faulty_training, "train", 20), (fault_free_training, "train", 20),
            (faulty_testing, "test", 160), (fault_free_testing, "test", 160),
        ):
            current, columns = TEPDataset._dataverse_records(frame, source, onset, max_runs_per_fault)
            if feature_columns is None:
                feature_columns = columns
            elif feature_columns != columns:
                raise ValueError("TEP dataverse frames must use identical feature columns")
            records.extend(current)
        return TEPDataset._from_dataverse_records(
            records, feature_columns, history_len, window_stride, seed, valid_fraction,
            require_test=include_test, requested_max_runs_per_fault=max_runs_per_fault,
        )

    @staticmethod
    def from_dataverse_path(
        root, history_len=96, window_stride=12, seed=2026, valid_fraction=0.2,
        max_runs_per_fault=None, include_test=True,
    ):
        root = Path(root)
        paths = {
            "faulty_training": root / "TEP_Faulty_Training.RData",
            "fault_free_training": root / "TEP_FaultFree_Training.RData",
            "faulty_testing": root / "TEP_Faulty_Testing.RData",
            "fault_free_testing": root / "TEP_FaultFree_Testing.RData",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("TEP dataverse files not found: {}".format(", ".join(missing)))
        try:
            import ctypes
            ctypes.CDLL("libiconv.so", mode=ctypes.RTLD_GLOBAL)
            import pyreadr
        except Exception as error:
            raise RuntimeError("TEP dataverse loading requires pyreadr in the torch23 environment") from error

        records = []
        feature_columns = None
        file_specs = [
            ("faulty_training", "train", 20),
            ("fault_free_training", "train", 20),
        ]
        if include_test:
            file_specs.extend([
                ("faulty_testing", "test", 160),
                ("fault_free_testing", "test", 160),
            ])
        for name, source, onset in file_specs:
            path = paths[name]
            loaded = pyreadr.read_r(str(path))
            if len(loaded) != 1:
                raise ValueError("TEP dataverse file must contain one data frame: {}".format(path.name))
            current, columns = TEPDataset._dataverse_records(
                next(iter(loaded.values())), source, onset, max_runs_per_fault,
            )
            if feature_columns is None:
                feature_columns = columns
            elif feature_columns != columns:
                raise ValueError("TEP dataverse files must use identical feature columns")
            records.extend(current)
            del loaded, current
        return TEPDataset._from_dataverse_records(
            records, feature_columns, history_len, window_stride, seed, valid_fraction,
            require_test=include_test, requested_max_runs_per_fault=max_runs_per_fault,
        )

    @staticmethod
    def from_dataverse_testing_path(root, history_len=96, window_stride=12, max_runs_per_fault=None):
        """Load only official testing runs, avoiding resident training windows during evaluation."""
        root = Path(root)
        paths = (
            (root / "TEP_Faulty_Testing.RData", 160),
            (root / "TEP_FaultFree_Testing.RData", 160),
        )
        try:
            import ctypes
            ctypes.CDLL("libiconv.so", mode=ctypes.RTLD_GLOBAL)
            import pyreadr
        except Exception as error:
            raise RuntimeError("TEP dataverse loading requires pyreadr in the torch23 environment") from error
        samples = {"train": [], "validation": [], "test": []}
        feature_columns = None
        selected_runs = 0
        for path, fault_start in paths:
            if not path.is_file():
                raise FileNotFoundError("TEP dataverse testing file not found: {}".format(path))
            loaded = pyreadr.read_r(str(path))
            if len(loaded) != 1:
                raise ValueError("TEP dataverse file must contain one data frame: {}".format(path.name))
            records, columns = TEPDataset._dataverse_records(
                next(iter(loaded.values())), "test", fault_start, max_runs_per_fault,
            )
            del loaded
            if feature_columns is None:
                feature_columns = columns
            elif feature_columns != columns:
                raise ValueError("TEP dataverse testing files must use identical feature columns")
            selected_runs += len(records)
            for record in records:
                values, times = record["values"], record["times"]
                if len(values) < history_len:
                    continue
                unit_id = "test:{}:{}".format(record["fault"], record["run"])
                for start in range(0, len(values) - history_len + 1, window_stride):
                    end = start + history_len
                    label = record["fault"] if end - 1 >= fault_start else 0
                    samples["test"].append(_window_sample(
                        values, times, start, history_len, [label], "tep_fault_v1",
                        class_label=torch.tensor(label), fault_number=record["fault"],
                        simulation_run=int(float(record["run"])), unit_id=unit_id,
                        window_end=end - 1, fault_start=fault_start,
                        source_split="test", window_stride=window_stride,
                    ))
            del records
        if not samples["test"]:
            raise ValueError("TEP dataverse testing files produced no test windows")
        return DatasetBundle(
            "tep", samples,
            split_identity(sorted(sample.metadata["unit_id"] for sample in samples["test"])),
            "tep_fault_v1", {
                "features": feature_columns,
                "num_fault_classes": max(sample.class_label.item() for sample in samples["test"]) + 1,
                "source_format": "dataverse_repeated_run",
                "window_stride": window_stride,
                "training_fault_start": 20,
                "testing_fault_start": 160,
                "selected_training_runs": 0,
                "selected_testing_runs": selected_runs,
                "history_len": history_len,
                "requested_max_runs_per_fault": (
                    None if max_runs_per_fault is None else int(max_runs_per_fault)
                ),
                **bundle_content_provenance(samples),
            }
        )

    @staticmethod
    def from_array(values, fault_number, simulation_run, fault_start, history_len=96, sample_times=None):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or history_len < 1:
            raise ValueError("TEP values must be [samples, features]")
        if int(fault_number) < 0:
            raise ValueError("TEP fault labels must be non-negative")
        if not np.isfinite(values).all():
            raise ValueError("TEP values must be finite")
        times = np.arange(len(values), dtype=np.float32) if sample_times is None else np.asarray(sample_times, dtype=np.float32)
        samples = {"train": [], "validation": [], "test": []}
        for end in range(history_len, len(values) + 1):
            label = int(fault_number) if end - 1 >= fault_start else 0
            samples["test"].append(_window_sample(values, times, end - history_len, history_len, [label], "tep_fault_v1", class_label=torch.tensor(label), fault_number=int(fault_number), simulation_run=int(simulation_run), unit_id="{}:{}".format(fault_number, simulation_run), window_end=end - 1))
        return DatasetBundle("tep", samples, split_identity(["{}:{}".format(fault_number, simulation_run)]), "tep_fault_v1", {"fault_start": fault_start, "unit_id": "{}:{}".format(fault_number, simulation_run), **bundle_content_provenance(samples)})

    @staticmethod
    def from_runs(runs, history_len=96, seed=2026, valid_fraction=0.2):
        """Build TEP windows after composite-run splitting.

        ``runs`` maps ``(fault_number, simulation_run)`` to a mapping with
        ``values`` and ``fault_start``; optional ``times`` are accepted.
        The split is entity-first, so no window can cross a composite run.
        """
        if not runs:
            raise ValueError("TEP requires at least one composite run")
        normalized = {}
        by_fault = {}
        for key, record in runs.items():
            if len(key) != 2:
                raise ValueError("TEP run keys must be (fault_number, simulation_run)")
            fault, run = int(key[0]), str(key[1])
            if fault < 0:
                raise ValueError("TEP fault labels must be non-negative")
            values = np.asarray(record["values"], dtype=np.float32)
            onset = int(record["fault_start"])
            declared_split = record.get("split")
            if declared_split not in (None, "train", "validation", "test"):
                raise ValueError("TEP split must be train, validation, test or omitted")
            normalized[(fault, run)] = (values, onset, record.get("times"), declared_split)
            if declared_split != "test":
                by_fault.setdefault(fault, []).append(run)
        layers = split_tep_runs(by_fault, seed=seed, valid_fraction=valid_fraction)
        samples = {"train": [], "validation": [], "test": []}
        for (fault, run), (values, onset, times, declared_split) in normalized.items():
            if times is None:
                times = np.arange(len(values), dtype=np.float32)
            unit = "{}:{}".format(fault, run)
            split = "test" if declared_split == "test" else ("train" if run in layers[fault]["train"] else "validation")
            for end in range(history_len, len(values) + 1):
                label = fault if end - 1 >= onset else 0
                sample = _window_sample(values, times, end-history_len, history_len, [label], "tep_fault_v1", class_label=torch.tensor(label), fault_number=fault, simulation_run=int(run) if run.isdigit() else None, unit_id=unit, window_end=end-1)
                samples[split].append(sample)
        if any(not samples[name] for name in samples):
            raise ValueError("TEP split must produce non-empty train, validation and test windows")
        return DatasetBundle("tep", samples, split_identity(["{}:{}".format(f, r) for f, r in normalized]), "tep_fault_v1", {"num_fault_classes": max(f for f, _ in normalized) + 1, "history_len": history_len, **bundle_content_provenance(samples)})

    @staticmethod
    def from_dat(path, fault_number, simulation_run, fault_start, history_len=96):
        values = np.loadtxt(path)
        if values.ndim == 2 and values.shape[0] < values.shape[1]:
            values = values.T
        return TEPDataset.from_array(values, fault_number, simulation_run, fault_start, history_len)
