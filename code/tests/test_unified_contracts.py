from dataclasses import fields

import pytest
import torch

import inspect

from kaf_profiti.models.unified import ModelConfig, PredictionOutput
from kaf_profiti.models.unified.encoder import EncoderOutput
from kaf_profiti.models.unified.heads.base import TaskTargets
from kaf_profiti.models.unified.model import HistoryBatch, UnifiedModel


def test_history_batch_does_not_expose_future_targets():
    names = {field.name for field in fields(HistoryBatch)}
    forbidden = {"Y_q", "y_flat", "M_q", "mq_flat", "rul"}
    assert names.isdisjoint(forbidden)


def test_model_prediction_api_only_accepts_history():
    parameters = inspect.signature(UnifiedModel.predict).parameters
    assert tuple(parameters) == ("self", "history")
    assert parameters["history"].annotation is HistoryBatch


def test_task_targets_are_separate_from_history():
    targets = TaskTargets(values=torch.tensor([5.0]), schema="rul_raw_v1")
    assert targets.values.shape == (1,)
    assert targets.schema == "rul_raw_v1"


def test_prediction_output_carries_experiment_identity():
    output_fields = {field.name for field in fields(PredictionOutput)}
    assert {
        "dataset",
        "task",
        "encoder_version",
        "seed",
        "split_identity",
        "target_schema",
    } <= output_fields


def test_encoder_output_separates_sensor_and_global_representations():
    output = EncoderOutput(
        z_history=torch.zeros(2, 3, 8),
        z_global=torch.zeros(2, 8),
        patch_tokens=None,
    )
    assert output.z_history.shape == (2, 3, 8)
    assert output.z_global.shape == (2, 8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset": "unknown", "task": "state", "input_dim": 2},
        {"dataset": "tep", "task": "rul", "input_dim": 2},
        {"dataset": "metropt3", "task": "state", "input_dim": 0},
    ],
)
def test_model_config_rejects_invalid_dataset_task_contract(kwargs):
    kwargs.setdefault("seed", 7)
    kwargs.setdefault("split_identity", "fixture-split-v1")
    kwargs.setdefault("target_schema", "fixture-target-v1")
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
