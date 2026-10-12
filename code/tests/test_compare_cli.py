import pytest

from compare_code.cli import build_parser


def test_compare_cli_exposes_model_and_test_switch():
    args = build_parser().parse_args([
        "--model", "patchtst", "--dataset", "metropt3",
        "--data-root", "data", "--result-root", "result", "--evaluate-test",
    ])
    assert args.model == "patchtst"
    assert args.evaluate_test is True


def test_compare_cli_accepts_cmapss_runs():
    from compare_code.cli import validate_dataset

    assert validate_dataset("cmapss_fd004") is None


def test_compare_cli_rejects_unknown_dataset():
    from compare_code.cli import validate_dataset

    with pytest.raises(ValueError, match="dataset"):
        validate_dataset("unknown")


def test_compare_cli_exposes_tep_window_selection_options():
    args = build_parser().parse_args([
        "--model", "tcn", "--dataset", "tep", "--data-root", "data", "--result-root", "result",
    ])
    assert args.tep_window_stride == 12
    assert args.tep_max_runs_per_fault == 20
    assert args.tep_valid_fraction == 0.2


def test_compare_cli_exposes_fd004_matched_preprocessing_and_tail_validation():
    args = build_parser().parse_args([
        "--model", "gru", "--dataset", "cmapss_fd004", "--data-root", "data", "--result-root", "result",
        "--fd004-validation", "tail_windows", "--fd004-endpoint-horizons", "20,40,60,80",
        "--fd004-normalization", "condition", "--fd004-sensor-filter", "standard",
        "--training-cap", "150",
    ])
    assert args.fd004_validation == "tail_windows"
    assert args.fd004_endpoint_horizons == (20, 40, 60, 80)
    assert args.fd004_normalization == "condition"
    assert args.fd004_sensor_filter == "standard"
    assert args.training_cap == 150.0


def test_compare_cli_exposes_learning_rate():
    args = build_parser().parse_args([
        "--model", "tcn", "--dataset", "metropt3",
        "--data-root", "data", "--result-root", "result", "--lr", "3e-4",
    ])
    assert args.lr == 3e-4


def test_compare_cli_forwards_learning_rate_to_runner(monkeypatch, tmp_path):
    from compare_code.cli import main

    captured = {}

    def fake_run_bundle(bundle, result_root, **kwargs):
        captured.update(kwargs)
        return {"manifest": {"run_id": "captured"}}

    monkeypatch.setattr("compare_code.cli.load_bundle", lambda args: object())
    monkeypatch.setattr("compare_code.cli.run_bundle", fake_run_bundle)
    main([
        "--model", "tcn", "--dataset", "metropt3",
        "--data-root", "data", "--result-root", str(tmp_path), "--lr", "5e-4",
    ])
    assert captured["learning_rate"] == 5e-4
