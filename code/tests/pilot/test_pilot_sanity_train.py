"""CH34-S03-T02: LI+TCN validation-only learnability sanity trainer contracts."""

import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from kaf_profiti.experiments.pilot_runner import (
    PilotRunner,
    build_model,
    load_matrix,
    pilot_sanity_train,
)
from kaf_profiti.industrial.batch import IndustrialBatch


def _sanity_matrix(tmp_path):
    payload = {
        "matrix_id": "tiny_sanity_point",
        "dataset": "metropt3_chrono_502030_v2",
        "seed": 2026,
        "split_seed": 2026,
        "mask_seed": 2026,
        "history_len": 10,
        "pred_len": 3,
        "stride": 1,
        "epochs": 50,
        "batch_size": 4,
        "hidden_dim": 8,
        "conditions": [
            {
                "condition_id": "point_mixed_030",
                "missing_mode": "mixed",
                "target_missing_rate": 0.30,
            },
        ],
        "models": [
            {"model_id": "li_tcn", "head_type": "linear", "family": "baseline", "priority": 1},
        ],
    }
    path = tmp_path / "point.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_matrix(path)


class _ListLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _make_batch(num_sensors, history_len, pred_len, context_dim, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, history_len, num_sensors, generator=generator)
    y = torch.randn(batch_size, pred_len, num_sensors, generator=generator)
    return IndustrialBatch(
        X_obs=x,
        T_obs=torch.arange(history_len, dtype=torch.float32).repeat(batch_size, 1),
        M_obs=torch.ones(batch_size, history_len, num_sensors),
        T_q=torch.arange(1, pred_len + 1, dtype=torch.float32).repeat(batch_size, 1),
        Y_q=y,
        M_q=torch.ones(batch_size, pred_len, num_sensors),
        context=torch.randn(batch_size, context_dim, generator=generator),
        y_flat=y.reshape(batch_size, -1),
        mq_flat=torch.ones(batch_size, pred_len * num_sensors),
        query_channel_ids=torch.arange(num_sensors).repeat(pred_len),
        rul=torch.rand(batch_size, generator=generator) * 100.0,
        unit_id=torch.arange(batch_size),
    )


class _SanityProvider:
    def __init__(self, spec):
        self._spec = spec
        self.num_sensors = 4
        self.context_dim = 3
        self.model_options = {}

    def protocol_fingerprint(self):
        return {
            "mechanism": self._spec.missing_mode,
            "requested_rate": self._spec.target_missing_rate,
            "split_sha256": "s" * 64,
        }

    def loaders(self, batch_size, generator_seed=None):
        spec = self._spec
        return {
            "train": _ListLoader(
                [
                    _make_batch(
                        self.num_sensors, spec.history_len, spec.pred_len,
                        self.context_dim, batch_size, seed=1,
                    ),
                    _make_batch(
                        self.num_sensors, spec.history_len, spec.pred_len,
                        self.context_dim, batch_size, seed=2,
                    ),
                ]
            ),
            "valid": _ListLoader(
                [
                    _make_batch(
                        self.num_sensors, spec.history_len, spec.pred_len,
                        self.context_dim, batch_size, seed=3,
                    )
                ]
            ),
        }


def _factory(spec, data_root, result_root):
    return _SanityProvider(spec)


def _runner(tmp_path):
    return PilotRunner(
        [_sanity_matrix(tmp_path)], tmp_path / "result", profile="metropt3"
    )


def test_sanity_trainer_runs_validation_only_and_reports_flags(tmp_path):
    """The sanity trainer scores init + N epochs and never loads test."""

    spec = _runner(tmp_path).expand()[0]
    provider = _SanityProvider(spec)
    torch.manual_seed(spec.seed)
    model = build_model(spec, provider.num_sensors, provider.context_dim, {}, device="cpu")
    # No "test" key is ever handed to the trainer.
    loaders = provider.loaders(spec.batch_size)

    outcome = pilot_sanity_train(model, loaders, spec, provider, device="cpu", epochs=5)

    assert outcome["epochs_run"] == 5
    assert len(outcome["history"]) == 6  # epoch 0 baseline + 5 training epochs
    assert outcome["history"][0]["epoch"] == 0
    for flag in ("finite", "updated", "validation_improved"):
        assert isinstance(outcome[flag], bool)
    assert outcome["finite"] is True
    assert math.isfinite(outcome["init_valid_mae"])
    assert math.isfinite(outcome["best_valid_mae"])


def test_run_sanity_train_selects_li_tcn_and_writes_isolated_manifest(tmp_path):
    runner = _runner(tmp_path)
    key = "metropt3_chrono_502030_v2|point|li_tcn|linear|point_mixed_030|2026"

    manifest = runner.run_sanity_train(epochs=3, provider_factory=_factory)

    assert manifest["run_level"] == "sanity_train"
    assert manifest["run_id"] == key
    assert manifest["test_evaluation_count"] == 0
    assert manifest["epochs"] == 50  # configured matrix epochs, untouched
    assert manifest["sanity_epochs"] == 3
    assert manifest["finite"] is True
    assert manifest["updated"] is True
    assert isinstance(manifest["validation_improved"], bool)

    # Isolated under sanity/, never under runs/.
    sanity_dir = tmp_path / "result" / "pilot" / "metropt3" / "sanity" / key
    runs_dir = tmp_path / "result" / "pilot" / "metropt3" / "runs"
    assert (sanity_dir / "manifest.json").is_file()
    assert (sanity_dir / "history.json").is_file()
    assert not (sanity_dir / "predictions.json").exists()
    assert not (sanity_dir / "checkpoint.pt").exists()
    assert not runs_dir.exists()

    # The manifest must not look like a resumable complete pilot run.
    manifest_on_disk = json.loads((sanity_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_on_disk["run_level"] == "sanity_train"
    assert manifest_on_disk["test_evaluation_count"] == 0


def test_sanity_train_rejects_when_li_tcn_point_mixed_030_missing(tmp_path):
    runner = _runner(tmp_path)
    matrix = _sanity_matrix(tmp_path)
    matrix.raw["models"] = [{"model_id": "gru_d", "head_type": "linear", "family": "baseline", "priority": 1}]
    # Rebuild a runner over a matrix that lacks li_tcn.
    path = tmp_path / "no_li.yaml"
    path.write_text(yaml.safe_dump(matrix.raw), encoding="utf-8")
    runner = PilotRunner([load_matrix(path)], tmp_path / "result", profile="metropt3")

    with pytest.raises(ValueError, match="exactly one li_tcn"):
        runner.run_sanity_train(epochs=2, provider_factory=_factory)


def test_naive_floor_reference_reads_data_gate_when_present(tmp_path):
    runner = _runner(tmp_path)
    diagnostics = tmp_path / "result" / "pilot" / "metropt3" / "diagnostics"
    diagnostics.mkdir(parents=True)
    (diagnostics / "data_gate.json").write_text(
        json.dumps(
            {
                "floors": {
                    "valid": {
                        "persistence": {"mae": 0.915894, "std_micro": {"mae": 0.402354}}
                    }
                },
                "learnability_gate": {"result": "pass"},
            }
        ),
        encoding="utf-8",
    )

    reference = runner._naive_floor_reference()

    assert reference["available"] is True
    assert reference["persistence_mae_raw"] == 0.915894
    assert reference["persistence_mae_std_micro"] == 0.402354
