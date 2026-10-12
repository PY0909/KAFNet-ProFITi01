from kaf_profiti.industrial.splits import (
    split_cmapss_engines,
    split_metropt_first_month,
    split_metropt_timestamps,
    split_tep_runs,
)


def test_metropt_keeps_duplicate_timestamps_in_one_split():
    result = split_metropt_timestamps([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert set(result) == {"train", "validation", "test"}
    memberships = {index: split for split, indices in result.items() for index in indices}
    assert memberships[0] == memberships[1]


def test_metropt_official_split_uses_first_calendar_month_for_training_period():
    timestamps = [
        "2020-02-01 00:00:00", "2020-02-01 00:00:00",
        "2020-02-15 00:00:00", "2020-02-29 23:59:00",
        "2020-03-01 00:00:00", "2020-04-01 00:00:00",
    ]
    result = split_metropt_first_month(timestamps, validation_fraction=0.5)
    memberships = {index: split for split, indices in result.items() for index in indices}

    assert memberships[0] == "train"
    assert memberships[1] == "train"
    assert memberships[2] == "validation"
    assert memberships[3] == "validation"
    assert memberships[4] == "test"
    assert memberships[5] == "test"


def test_metropt_official_split_rejects_non_datetime_timestamps():
    try:
        split_metropt_first_month([0, 1, 2, 3])
    except ValueError as error:
        assert "datetime" in str(error)
    else:
        raise AssertionError("official MetroPT split requires datetime timestamps")


def test_cmapss_engine_split_is_deterministic_and_disjoint():
    first = split_cmapss_engines([1, 2, 3, 4, 5], seed=2026)
    second = split_cmapss_engines([1, 2, 3, 4, 5], seed=2026)
    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])


def test_tep_split_is_disjoint_within_each_fault_layer():
    result = split_tep_runs({1: ["1:1", "1:2", "1:3", "1:4"], 2: ["2:1", "2:2", "2:3", "2:4"]})
    for fault, layers in result.items():
        assert set(layers["train"]).isdisjoint(layers["validation"]), fault
