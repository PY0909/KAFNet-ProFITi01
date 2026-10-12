from typing import Dict, Iterable, List, Mapping
import pandas as pd

import numpy as np


def split_metropt_timestamps(timestamps: Iterable) -> Dict[str, List[int]]:
    """Chronological 50/20/30 split, assigning duplicate timestamps together."""
    values = list(timestamps)
    groups = {}
    for index, timestamp in enumerate(values):
        groups.setdefault(timestamp, []).append(index)
    ordered = sorted(groups, key=lambda value: (value, str(value)))
    n = len(ordered)
    if n < 3:
        raise ValueError("MetroPT split requires at least three unique timestamps")
    boundaries = (int(0.50 * n), int(0.70 * n))
    if boundaries[0] == 0 or boundaries[1] <= boundaries[0] or boundaries[1] >= n:
        raise ValueError("MetroPT split would produce an empty partition")
    result = {"train": [], "validation": [], "test": []}
    for position, timestamp in enumerate(ordered):
        split = "train" if position < boundaries[0] else "validation" if position < boundaries[1] else "test"
        result[split].extend(groups[timestamp])
    return result


def split_metropt_first_month(timestamps: Iterable, validation_fraction=0.2) -> Dict[str, List[int]]:
    """Use the first calendar month as the training period, then test on the remainder.

    The tail of the first month is reserved for validation so that model selection
    never reads the later test period.
    """
    values = list(timestamps)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("MetroPT validation_fraction must be between zero and one")
    if pd.api.types.is_numeric_dtype(pd.Series(values)):
        raise ValueError("official MetroPT split requires datetime timestamps")
    try:
        parsed = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("official MetroPT split requires datetime timestamps") from error
    groups = {}
    for index, timestamp in enumerate(parsed):
        groups.setdefault(timestamp, []).append(index)
    ordered = sorted(groups)
    if len(ordered) < 3:
        raise ValueError("MetroPT split requires at least three unique timestamps")
    first_month = ordered[0].to_period("M")
    month_positions = [index for index, timestamp in enumerate(ordered) if timestamp.to_period("M") == first_month]
    if len(month_positions) < 2 or len(month_positions) == len(ordered):
        raise ValueError("MetroPT first-month split needs train/validation and later test timestamps")
    n_validation = max(1, int(len(month_positions) * validation_fraction + 0.5))
    if n_validation >= len(month_positions):
        raise ValueError("MetroPT validation split would consume the first month")
    validation_start = len(month_positions) - n_validation
    result = {"train": [], "validation": [], "test": []}
    for position, timestamp in enumerate(ordered):
        if timestamp.to_period("M") != first_month:
            split = "test"
        elif position < validation_start:
            split = "train"
        else:
            split = "validation"
        result[split].extend(groups[timestamp])
    return result


def split_cmapss_engines(engine_ids: Iterable[int], seed=2026, valid_fraction=0.2):
    ids = sorted({int(engine) for engine in engine_ids})
    if len(ids) < 2:
        raise ValueError("C-MAPSS split requires at least two engines")
    permutation = np.random.default_rng(seed).permutation(ids).tolist()
    n_valid = max(1, int(len(ids) * valid_fraction + 0.5))
    if n_valid >= len(ids):
        raise ValueError("validation split would consume all engines")
    return {"train": sorted(permutation[:-n_valid]), "validation": sorted(permutation[-n_valid:])}


def split_tep_runs(runs_by_fault: Mapping[int, Iterable[str]], seed=2026, valid_fraction=0.2):
    result = {}
    for fault_number in sorted(runs_by_fault):
        runs = sorted(
            {str(run) for run in runs_by_fault[fault_number]},
            key=lambda run: (0, float(run)) if _is_numeric_run(run) else (1, run),
        )
        if len(runs) < 2:
            raise ValueError("each TEP fault layer needs train and validation runs")
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(fault_number)]))
        order = rng.permutation(runs).tolist()
        n_valid = max(1, int(len(runs) * valid_fraction + 0.5))
        if n_valid >= len(runs):
            raise ValueError("TEP validation split would consume all runs")
        result[int(fault_number)] = {
            "train": sorted(order[:-n_valid]),
            "validation": sorted(order[-n_valid:]),
        }
    return result


def _is_numeric_run(run):
    try:
        float(run)
    except (TypeError, ValueError):
        return False
    return True
