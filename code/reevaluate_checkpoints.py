import argparse
import json
from pathlib import Path

import pandas as pd

from kaf_profiti.experiments.frozen_evaluation import evaluate_frozen_checkpoint
from kaf_profiti.industrial.datasets import CMapssFD004, MetroPT3


def _run_ids(result_root, dataset):
    selected = []
    for path in sorted(Path(result_root).iterdir()):
        if not path.is_dir() or path.name.endswith("_reeval"):
            continue
        manifest_path = path / "manifest.json"
        checkpoint_path = path / "checkpoint.pt"
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dataset") == dataset and manifest.get("model_name") in {
            "unified", "tcn", "gru", "patchtst", "timesnet", "itransformer",
        }:
            selected.append(path.name)
    return selected


def _bundle(dataset, data_root, checkpoint_dir, metropt_protocol="chrono_502030"):
    checkpoint = __import__("torch").load(checkpoint_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if dataset == "cmapss_fd004":
        return CMapssFD004.from_files(
            data_root,
            history_len=int(config.get("history_len", 50)),
            training_cap=float(config.get("training_cap", 125.0)),
            seed=int(config.get("seed", 2026)),
        )
    candidates = (
        Path(data_root) / "metropt3.csv",
        Path(data_root) / "MetroPT3(AirCompressor).csv",
        Path(data_root) / "metropt+3+dataset" / "MetroPT3(AirCompressor).csv",
    )
    csv_path = next((path for path in candidates if path.is_file()), None)
    if csv_path is None:
        raise FileNotFoundError("MetroPT CSV not found under --data-root")
    frame = pd.read_csv(csv_path)
    return MetroPT3.from_frame(
        frame,
        history_len=int(config.get("history_len", 168)),
        pred_len=int(config.get("pred_len", 24)),
        stride=60,
        protocol=metropt_protocol,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-evaluate frozen MetroPT-3 or FD004 checkpoints")
    parser.add_argument("--dataset", choices=("metropt3", "cmapss_fd004"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-ids", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result_root = Path(args.result_root)
    run_ids = args.run_ids or _run_ids(result_root, args.dataset)
    if not run_ids:
        raise ValueError("no checkpoint runs found for {}".format(args.dataset))
    for run_id in run_ids:
        source_dir = result_root / run_id
        output_dir = result_root / (run_id + "_reeval")
        manifest = json.loads((source_dir / "manifest.json").read_text())
        if manifest.get("dataset") != args.dataset:
            raise ValueError("{} is not a {} run".format(run_id, args.dataset))
        dataset_metadata = manifest.get("dataset_metadata", {})
        protocol = (
            "official_first_month"
            if dataset_metadata.get("protocol") == "metropt3_official_first_month_v1"
            else "chrono_502030"
        )
        bundle = _bundle(args.dataset, args.data_root, source_dir, metropt_protocol=protocol)
        metrics = evaluate_frozen_checkpoint(
            bundle, source_dir, output_dir,
            batch_size=args.batch_size, device=args.device,
        )
        summary = {
            "run_id": run_id,
            "output_run_id": output_dir.name,
            "mae": metrics.get("mae", metrics.get("overall", {}).get("mae")),
            "rmse": metrics.get("rmse", metrics.get("overall", {}).get("rmse")),
            "nasa_score": metrics.get("nasa_score"),
            "examples": metrics["examples"],
        }
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
