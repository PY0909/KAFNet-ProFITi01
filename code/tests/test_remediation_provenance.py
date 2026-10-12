import torch

from kaf_profiti.experiments.entrypoint import run_bundle
from kaf_profiti.industrial.datasets import DatasetBundle
from kaf_profiti.industrial.batch import HistorySample


def test_remediation_structure_and_tep_weights_are_saved_in_artifacts(tmp_path):
    samples = [
        HistorySample(
            X_obs=torch.randn(4, 2), T_obs=torch.arange(4, dtype=torch.float32),
            target=torch.tensor([float(label)]), target_schema="tep_fault_v1",
            class_label=torch.tensor(label), fault_number=label, simulation_run=index,
            metadata={"unit_id": str(index), "window_end": index},
        )
        for index, label in enumerate((0, 1, 2, 0, 1, 2))
    ]
    bundle = DatasetBundle(
        "tep", {"train": samples[:3], "validation": samples[3:4], "test": samples[4:]},
        "split", "tep_fault_v1", {"num_fault_classes": 3},
    )
    result = run_bundle(
        bundle, tmp_path, epochs=1, batch_size=2, device="cpu", amp=False,
        scheduler="none", evaluate_test=True, cross_channel_mode="cudtr_only",
    )
    run_dir = tmp_path / result["manifest"]["run_id"]
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["cross_channel_mode"] == "cudtr_only"
    assert tuple(checkpoint["config"]["tep_class_weights"]) == (1.0, 1.0, 1.0)
    assert result["manifest"]["cross_channel_mode"] == "cudtr_only"
