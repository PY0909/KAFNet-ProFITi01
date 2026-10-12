from kaf_profiti.experiments.cli import build_parser, load_bundle


def test_cli_requires_dataset_and_data_root():
    parser = build_parser()
    args = parser.parse_args(["--dataset", "cmapss_fd004", "--data-root", "data", "--result-root", "result"])
    assert args.dataset == "cmapss_fd004"
    assert args.data_root == "data"
    assert args.no_progress is False
    metro_args = parser.parse_args(["--dataset", "metropt3", "--data-root", "data", "--result-root", "result"])
    assert metro_args.metropt_protocol == "official_first_month"


def test_cli_accepts_raw_rul_training_target():
    parser = build_parser()
    args = parser.parse_args([
        "--dataset", "cmapss_fd004", "--data-root", "data", "--result-root", "result",
        "--training-cap", "raw",
    ])
    assert args.training_cap is None


def test_cli_accepts_fd004_endpoint_validation():
    args = build_parser().parse_args([
        "--dataset", "cmapss_fd004", "--data-root", "data", "--result-root", "result",
        "--fd004-validation", "endpoint",
    ])
    assert args.fd004_validation == "endpoint"


def test_cli_parses_patch_lens_and_rul_over_penalty():
    args = build_parser().parse_args([
        "--dataset", "cmapss_fd004", "--data-root", "data", "--result-root", "result",
        "--patch-lens", "5,10,25", "--rul-over-penalty", "1.5",
    ])
    assert args.patch_lens == (5, 10, 25)
    assert args.rul_over_penalty == 1.5


def test_cli_parses_rul_head_input():
    args = build_parser().parse_args([
        "--dataset", "cmapss_fd004", "--data-root", "data", "--result-root", "result",
        "--rul-head-input", "global_trend",
    ])
    assert args.rul_head_input == "global_trend"
def test_cli_rejects_non_positive_patch_lens():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--dataset", "cmapss_fd004", "--data-root", "data", "--result-root", "result",
            "--patch-lens", "5,0",
        ])


def test_compare_models_use_an_independent_cli():
    from compare_code.cli import build_parser

    args = build_parser().parse_args([
        "--model", "tcn", "--dataset", "metropt3", "--data-root", "data", "--result-root", "result",
    ])
    assert args.model == "tcn"


def test_unified_cli_exposes_learning_rate():
    from kaf_profiti.experiments.cli import build_parser

    args = build_parser().parse_args([
        "--dataset", "cmapss_fd004", "--data-root", "data", "--result-root", "result",
        "--lr", "0.0002",
    ])
    assert args.lr == 0.0002


def test_metropt_loader_accepts_canonical_csv_name(tmp_path):
    import pandas as pd

    frame = pd.DataFrame({
        "timestamp": pd.date_range("2020-02-01", periods=40, freq="D"),
        "sensor_a": range(40), "sensor_b": range(40),
    })
    frame.to_csv(tmp_path / "MetroPT3(AirCompressor).csv", index=False)
    args = build_parser().parse_args([
        "--dataset", "metropt3", "--data-root", str(tmp_path), "--result-root", str(tmp_path / "result"),
        "--history-len", "2", "--pred-len", "1",
    ])
    bundle = load_bundle(args)
    assert bundle.dataset == "metropt3"


def test_tep_cli_uses_dataverse_directory(monkeypatch, tmp_path):
    seen = {}

    def fake_loader(root, **kwargs):
        seen["root"] = root
        seen.update(kwargs)
        return "bundle"

    monkeypatch.setattr("kaf_profiti.experiments.cli.TEPDataset.from_dataverse_path", fake_loader)
    args = build_parser().parse_args([
        "--dataset", "tep", "--data-root", str(tmp_path), "--result-root", str(tmp_path / "result"),
        "--seed", "8", "--history-len", "96", "--tep-window-stride", "12", "--tep-max-runs-per-fault", "3",
    ])
    assert load_bundle(args) == "bundle"
    assert seen["root"] == tmp_path
    assert seen["window_stride"] == 12
    assert seen["max_runs_per_fault"] == 3
    assert seen["seed"] == 8
