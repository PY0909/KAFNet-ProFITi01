import json

import torch

from kaf_profiti.industrial.batch import HistorySample, collate_history
from kaf_profiti.industrial.protocol import TrainNormalizer, build_manifest, split_identity
from kaf_profiti.experiments.registry import create_model, get_model_spec


def test_collate_history_keeps_targets_out_of_prediction_inputs():
    samples = [
        HistorySample(
            X_obs=torch.ones(3, 2),
            T_obs=torch.tensor([0.0, 1.0, 3.0]),
            target=torch.tensor([2.0]),
            target_schema="rul_raw_v1",
        )
    ]
    batch = collate_history(samples)
    assert batch.X_obs.shape == (1, 3, 2)
    assert batch.M_obs.shape == batch.X_obs.shape
    assert torch.equal(batch.M_obs, torch.ones_like(batch.X_obs))
    assert batch.targets.values.shape == (1, 1)


def test_collate_history_preserves_explicit_observation_mask():
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sample = HistorySample(
        X_obs=torch.ones(3, 2), T_obs=torch.arange(3, dtype=torch.float32), M_obs=mask,
    )
    batch = collate_history([sample])
    torch.testing.assert_close(batch.M_obs[0], mask)


def test_history_batch_can_move_inputs_and_targets_to_device():
    samples = [
        HistorySample(
            X_obs=torch.ones(3, 2), T_obs=torch.arange(3, dtype=torch.float32),
            target=torch.tensor([2.0]), target_schema="rul_raw_v1",
        )
    ]
    batch = collate_history(samples).to("cpu")
    assert batch.X_obs.device.type == "cpu"
    assert batch.targets.values.device.type == "cpu"
    assert batch.M_obs.device.type == "cpu"


def test_normalizer_fits_only_training_values():
    normalizer = TrainNormalizer().fit(torch.tensor([[1.0, 3.0], [3.0, 5.0]]))
    transformed = normalizer.transform(torch.tensor([[3.0, 5.0]]))
    assert normalizer.source_split == "train"
    assert normalizer.identity == "train-zscore-features2"
    torch.testing.assert_close(transformed, torch.tensor([[1.0, 1.0]]))


def test_manifest_has_run_identity_and_rejects_absolute_paths():
    manifest = build_manifest(
        run_id="smoke-1", dataset="tep", task="fault_detection_diagnosis", seed=2026,
        split_identity=split_identity(["run-1", "run-2"]), target_schema="tep_fault_v1",
        encoder_version="unified_temporal_channelwise_v2", checkpoint="checkpoints/best.pt",
        dataset_schema="tep-v1", normalizer_identity="normalizer-x",
        target_convention="tep_fault_v1", evaluator_version="evaluator-v1",
        model_version="model-v1",
    )
    assert manifest["run_id"] == "smoke-1"
    json.dumps(manifest)
    try:
        build_manifest(
            run_id="bad", dataset="tep", task="fault_detection_diagnosis", seed=1,
            split_identity="s", target_schema="t", encoder_version="v", checkpoint="/tmp/best.pt",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("absolute checkpoint path must be rejected")


def test_registry_creates_only_new_identity_without_changing_legacy():
    spec = get_model_spec("unified_encoder_task")
    assert spec["status"] == "enabled"
    model = create_model(
        "unified_encoder_task", dataset="cmapss_fd004", task="rul", input_dim=2,
        seed=1, split_identity="s", target_schema="r", pred_len=1,
    )
    assert model.config.parameter_scope == "dataset_independent"
    assert get_model_spec("unified_encoder_task")["status"] == "enabled"
    assert get_model_spec("kst_probflow")["status"] == "not_implemented"
