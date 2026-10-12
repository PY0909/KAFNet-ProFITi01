import argparse
from pathlib import Path

from kaf_profiti.experiments.cli import _positive_int_list, load_bundle
from kaf_profiti.experiments.entrypoint import run_bundle

from .models.factory import MODEL_NAMES, create_baseline


def build_parser():
    parser = argparse.ArgumentParser(description="Run a comparison baseline")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--dataset", choices=("metropt3", "cmapss_fd004", "tep"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-len", type=int, default=168)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--training-cap", type=float, default=125.0)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument(
        "--metropt-protocol", choices=("official_first_month", "chrono_502030"),
        default="official_first_month",
    )
    parser.add_argument("--tep-window-stride", type=int, default=12, help="TEP repeated-run window stride")
    parser.add_argument(
        "--tep-max-runs-per-fault", type=int, default=20,
        help="TEP runs retained per fault (use 0 for all)",
    )
    parser.add_argument("--tep-valid-fraction", type=float, default=0.2)
    parser.add_argument("--fd004-validation", choices=("all_windows", "endpoint", "tail_windows"), default="all_windows")
    parser.add_argument("--fd004-endpoint-horizon", type=int, default=20)
    parser.add_argument("--fd004-endpoint-horizons", type=_positive_int_list, default=(20, 40, 60, 80))
    parser.add_argument("--fd004-normalization", choices=("global", "condition"), default="condition")
    parser.add_argument("--fd004-sensor-filter", choices=("none", "standard"), default="none")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-len", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3, help="optimizer learning rate for Adam")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--scheduler", choices=("plateau", "cosine", "none"), default="plateau")
    parser.add_argument("--run-id", default=None)
    return parser


def validate_dataset(dataset):
    if dataset not in {"metropt3", "cmapss_fd004", "tep"}:
        raise ValueError("unsupported dataset: {}".format(dataset))


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_dataset(args.dataset)
    bundle = load_bundle(args)

    def factory(**values):
        return create_baseline(
            args.model,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
            patch_len=args.patch_len,
            top_k=args.top_k,
            training_cap=args.training_cap,
            num_classes=int(bundle.metadata.get("num_fault_classes", 1)),
            **values,
        )

    result = run_bundle(
        bundle, Path(args.result_root), seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, progress=not args.no_progress,
        device=args.device, amp=not args.no_amp, patience=args.patience,
        scheduler=args.scheduler, evaluate_test=args.evaluate_test,
        run_id=args.run_id, model_factory=factory, model_name=args.model,
        feature_normalization=args.fd004_normalization if args.dataset == "cmapss_fd004" else "global",
        learning_rate=args.lr,
    )
    print(result["manifest"]["run_id"])
    return 0
