import argparse
import json
from pathlib import Path

import torch

from kaf_profiti.experiments.checkpoint_evaluation import evaluate_tep_checkpoint
from kaf_profiti.experiments.cli import build_parser as build_experiment_parser, load_bundle
from kaf_profiti.industrial.datasets import TEPDataset
from kaf_profiti.industrial.protocol import TrainNormalizer, split_identity


DEFAULT_RUN_IDS = (
    "tep_dataverse_unified_seed2026",
    "tep_dataverse_tcn_seed2026_20261007",
    "tep_dataverse_gru_seed2026_20261007",
    "tep_dataverse_patchtst_seed2026_20261007",
    "tep_dataverse_timesnet_seed2026_20261007",
    "tep_dataverse_itransformer_seed2026_20261007",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate frozen TEP checkpoints on official test runs")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-ids", nargs="+", default=DEFAULT_RUN_IDS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("prepare", "evaluate"), default="evaluate")
    parser.add_argument("--normalizer-file", required=True)
    args = parser.parse_args(argv)

    result_root = Path(args.result_root)
    normalizer_path = Path(args.normalizer_file)
    if args.phase == "prepare":
        experiment_parser = build_experiment_parser()
        bundle_args = experiment_parser.parse_args([
            "--dataset", "tep", "--data-root", args.data_root,
            "--result-root", args.result_root, "--seed", "2026",
            "--history-len", "96", "--tep-window-stride", "12",
            "--tep-max-runs-per-fault", "20",
        ])
        bundle = load_bundle(bundle_args)
        train_values = torch.cat([
            sample.X_obs.reshape(-1, sample.X_obs.shape[-1])
            for sample in bundle.samples["train"]
        ])
        normalizer = TrainNormalizer().fit(train_values, split="train")
        training_identity = split_identity(sorted(
            sample.metadata["unit_id"]
            for split in ("train", "validation")
            for sample in bundle.samples[split]
        ))
        normalizer_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "mean": normalizer.mean,
            "std": normalizer.std,
            "identity": normalizer.identity,
            "source_split": normalizer.source_split,
            "training_identity": training_identity,
        }, normalizer_path)
        print("prepared train-only normalizer for {}".format(training_identity))
        return 0

    manifests = []
    for run_id in args.run_ids:
        run_dir = result_root / run_id
        manifest = json.loads((run_dir / "manifest.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if metrics.get("test_evaluation_count", 0) != 0:
            raise ValueError("{} already has a test evaluation".format(run_id))
        if manifest.get("dataset") != "tep" or manifest.get("target_schema") != "tep_fault_v1":
            raise ValueError("{} is not a TEP fault-classification run".format(run_id))
        manifests.append(manifest)
    identities = {
        (item["seed"], item["split_identity"], item["target_schema"], item["normalizer_identity"])
        for item in manifests
    }
    if len(identities) != 1:
        raise ValueError("all checkpoints must share the same seed, split, target schema, and normalizer")

    normalizer_stats = torch.load(normalizer_path, map_location="cpu", weights_only=False)
    if normalizer_stats.get("source_split") != "train" or normalizer_stats.get("training_identity") != manifests[0]["split_identity"]:
        raise ValueError("normalizer file does not belong to the shared checkpoint training split")
    bundle = TEPDataset.from_dataverse_testing_path(
        args.data_root, history_len=96, window_stride=12, max_runs_per_fault=20,
    )
    for run_id in args.run_ids:
        metrics = evaluate_tep_checkpoint(
            bundle, result_root / run_id, batch_size=args.batch_size, device=args.device,
            normalizer_stats=normalizer_stats,
            training_identity=normalizer_stats["training_identity"],
        )
        print(json.dumps({
            "run_id": run_id,
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "log_loss": metrics["loss"],
            "test_windows": metrics["window_count"],
            "test_runs": metrics["run_summary"]["run_count"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
