import argparse
from pathlib import Path

from .entrypoint import run_bundle
from ..industrial.datasets import CMapssFD004, MetroPT3, TEPDataset


def _training_cap(value):
    if str(value).strip().lower() in {"raw", "none", "null"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("training cap must be a number or raw") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("training cap must be positive or raw")
    return parsed


def _patch_lens(value):
    try:
        lengths = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("patch lens must be comma-separated integers") from error
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("patch lens must contain positive lengths")
    return lengths


def _positive_int_list(value):
    try:
        values = tuple(sorted({int(part.strip()) for part in str(value).split(",") if part.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description="Run the unified dataset-specific model")
    parser.add_argument("--dataset", choices=("metropt3", "cmapss_fd004", "tep"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--encoder-variant", choices=("v2", "v2t", "v2tc", "v3a", "v3b", "v3c", "v4", "v5a", "v5b", "v5c"), default="v2")
    parser.add_argument("--patch-lens", type=_patch_lens, default=None,
                        help="comma-separated encoder patch lengths, e.g. 5,10,25 (default 12,24,48)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=60, help="MetroPT window stride")
    parser.add_argument(
        "--metropt-protocol", choices=("official_first_month", "chrono_502030"),
        default="official_first_month",
        help="MetroPT split protocol; official_first_month uses the first calendar month for train/validation",
    )
    parser.add_argument("--training-cap", type=_training_cap, default=125.0,
                        help="RUL training target cap; use raw to disable capping")
    parser.add_argument("--rul-over-penalty", type=float, default=1.0,
                        help="asymmetric L1 multiplier on RUL over-predictions (NASA score punishes over-estimation exponentially)")
    parser.add_argument("--rul-head-input", choices=("global", "global_trend"), default="global",
                        help="RUL head representation: global state only or global state plus masked trend summary")
    parser.add_argument("--sensor-pooling", choices=("attention", "mean", "mean_attention"), default="attention",
                        help="V3C sensor aggregation used by the task head")
    parser.add_argument("--cross-channel-mode", choices=("attention", "cudtr_only"), default="attention")
    parser.add_argument("--state-decoder", choices=("global", "scale_horizon"), default="global")
    parser.add_argument("--head-mode", choices=("point", "quantile"), default="point",
                        help="prediction head output mode for state/RUL tasks")
    parser.add_argument("--quantile-calibration", choices=("median_residual", "cqr"), default="median_residual",
                        help="validation-only interval calibration method")
    parser.add_argument("--validation-selection-fraction", type=float, default=0.5,
                        help="chronological fraction of validation samples reserved for checkpoint selection")
    parser.add_argument("--cudtr", action="store_true", help="enable CUDTR target-conditioned source routing")
    parser.add_argument("--cudtr-teacher", action="store_true",
                        help="compute train-only counterfactual utility labels from a frozen V3C/V4 baseline checkpoint")
    parser.add_argument("--cudtr-teacher-checkpoint", default=None,
                        help="frozen V3C/V4 baseline checkpoint used for train-only counterfactual utilities")
    parser.add_argument("--cudtr-router-only-epochs", type=int, default=0,
                        help="optional initial epochs that train only CUDTR routing/fallback parameters")
    parser.add_argument("--cudtr-top-k", type=int, default=2)
    parser.add_argument("--cudtr-query-dim", type=int, default=8)
    parser.add_argument("--cudtr-temperature", type=float, default=1.0)
    parser.add_argument("--cudtr-fallback-init", type=float, default=0.5,
                        help="initial baseline fallback weight in the bounded fusion gate")
    parser.add_argument("--cudtr-fallback-drop-rate", type=float, default=0.5)
    parser.add_argument("--cudtr-min-observation-rate", type=float, default=0.0,
                        help="minimum historical observation rate for a routable source")
    parser.add_argument("--cudtr-utility-threshold", type=float, default=None,
                        help="optional minimum predicted utility for routing; None keeps relative top-k behavior")
    parser.add_argument("--cudtr-lambda-distill", type=float, default=0.0)
    parser.add_argument("--cudtr-lambda-rank", type=float, default=0.0)
    parser.add_argument("--cudtr-lambda-sparse", type=float, default=0.0)
    parser.add_argument("--cudtr-lambda-fallback", type=float, default=0.0)
    parser.add_argument("--cudtr-rank-margin", type=float, default=0.1)
    parser.add_argument("--cudtr-utility-clip", type=float, default=None)
    parser.add_argument("--fd004-validation", choices=("all_windows", "endpoint", "tail_windows"), default="all_windows",
                        help="FD004 validation samples used for checkpoint selection")
    parser.add_argument("--fd004-endpoint-horizon", type=int, default=20,
                        help="cycles before the training trajectory endpoint used for endpoint validation")
    parser.add_argument("--fd004-endpoint-horizons", type=_positive_int_list, default=(20, 40, 60, 80),
                        help="tail horizons for non-degenerate FD004 validation, e.g. 20,40,60,80")
    parser.add_argument("--fd004-normalization", choices=("global", "condition"), default="condition",
                        help="FD004 feature normalization; condition groups are fit on train only")
    parser.add_argument("--fd004-sensor-filter", choices=("none", "standard"), default="none",
                        help="standard drops setting1-3 and sensor1,5,6,10,16,18,19 (keeps 14 degradation sensors)")
    parser.add_argument("--tep-window-stride", type=int, default=12, help="TEP repeated-run window stride")
    parser.add_argument(
        "--tep-max-runs-per-fault", type=int, default=20,
        help="TEP runs retained per fault (use 0 for all; default keeps the in-memory trainer bounded)",
    )
    parser.add_argument("--tep-valid-fraction", type=float, default=0.2)
    parser.add_argument("--no-progress", action="store_true", help="disable training progress bars")
    parser.add_argument("--evaluate-test", action="store_true", help="evaluate the validation-best model once on test")
    parser.add_argument("--run-id", default=None, help="optional unique artifact directory name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true", help="disable CUDA automatic mixed precision")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--scheduler", choices=("plateau", "cosine", "none"), default="plateau")
    parser.add_argument("--compile", action="store_true", help="wrap the model with torch.compile (CUDA only)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="disable per-epoch training sample shuffling (legacy fixed batch order)")
    return parser


def load_bundle(args):
    root = Path(args.data_root)
    if args.dataset == "cmapss_fd004":
        return CMapssFD004.from_files(
            root, history_len=args.history_len or 50, training_cap=args.training_cap,
            seed=args.seed, validation_mode=args.fd004_validation,
            endpoint_horizon=args.fd004_endpoint_horizon,
            sensor_filter=None if args.fd004_sensor_filter == "none" else args.fd004_sensor_filter,
            endpoint_horizons=args.fd004_endpoint_horizons,
        )
    if args.dataset == "metropt3":
        candidates = (
            root / "metropt3.csv",
            root / "MetroPT3(AirCompressor).csv",
            root / "metropt+3+dataset" / "MetroPT3(AirCompressor).csv",
        )
        csv_path = next((path for path in candidates if path.is_file()), None)
        if csv_path is None:
            raise FileNotFoundError(
                "MetroPT CSV not found; expected metropt3.csv or "
                "MetroPT3(AirCompressor).csv under --data-root"
            )
        frame = __import__("pandas").read_csv(csv_path)
        return MetroPT3.from_frame(
            frame, history_len=args.history_len or 168, pred_len=args.pred_len, stride=args.stride,
            protocol=args.metropt_protocol,
        )
    max_runs = None if args.tep_max_runs_per_fault == 0 else args.tep_max_runs_per_fault
    return TEPDataset.from_dataverse_path(
        root,
        history_len=args.history_len or 96,
        window_stride=args.tep_window_stride,
        seed=args.seed,
        valid_fraction=args.tep_valid_fraction,
        max_runs_per_fault=max_runs,
        include_test=args.evaluate_test,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    bundle = load_bundle(args)
    result = run_bundle(
        bundle, args.result_root, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, progress=not args.no_progress,
        device=args.device, amp=not args.no_amp, patience=args.patience, scheduler=args.scheduler,
        evaluate_test=args.evaluate_test, run_id=args.run_id, head_mode=args.head_mode,
        encoder_variant=args.encoder_variant,
        feature_normalization=args.fd004_normalization if args.dataset == "cmapss_fd004" else "global",
        patch_lens=args.patch_lens, rul_over_penalty=args.rul_over_penalty,
        compile_model=args.compile, shuffle=not args.no_shuffle, learning_rate=args.lr,
        rul_head_input=args.rul_head_input, sensor_pooling=args.sensor_pooling,
        cross_channel_mode=args.cross_channel_mode,
        state_decoder=args.state_decoder,
        cudtr_enabled=args.cudtr,
        cudtr_top_k=args.cudtr_top_k,
        cudtr_query_dim=args.cudtr_query_dim,
        cudtr_temperature=args.cudtr_temperature,
        cudtr_fallback_init=args.cudtr_fallback_init,
        cudtr_fallback_drop_rate=args.cudtr_fallback_drop_rate,
        cudtr_min_observation_rate=args.cudtr_min_observation_rate,
        cudtr_utility_threshold=args.cudtr_utility_threshold,
        cudtr_lambda_distill=args.cudtr_lambda_distill,
        cudtr_lambda_rank=args.cudtr_lambda_rank,
        cudtr_lambda_sparse=args.cudtr_lambda_sparse,
        cudtr_lambda_fallback=args.cudtr_lambda_fallback,
        cudtr_rank_margin=args.cudtr_rank_margin,
        cudtr_utility_clip=args.cudtr_utility_clip,
        cudtr_teacher_enabled=args.cudtr_teacher,
        cudtr_teacher_checkpoint=args.cudtr_teacher_checkpoint,
        cudtr_router_only_epochs=args.cudtr_router_only_epochs,
        quantile_calibration=args.quantile_calibration,
        validation_selection_fraction=args.validation_selection_fraction,
    )
    print(result["manifest"]["run_id"])
    return 0
