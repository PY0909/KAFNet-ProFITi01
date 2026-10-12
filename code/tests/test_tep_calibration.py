import torch

from kaf_profiti.experiments.metrics import select_tep_threshold
from kaf_profiti.models.unified import ModelConfig, UnifiedModel
from kaf_profiti.models.unified.heads.base import TaskTargets
from kaf_profiti.models.unified.model import HistoryBatch


def test_tep_threshold_is_selected_from_validation_f1():
    threshold, score = select_tep_threshold(
        torch.tensor([0.1, 0.4, 0.6, 0.8]),
        torch.tensor([0, 0, 1, 1]),
        objective="f1",
    )
    assert threshold == 0.5
    assert score > 0.9


def test_tep_head_accepts_train_frequency_class_weights():
    config = ModelConfig(
        dataset="tep", task="fault_detection_diagnosis", input_dim=3, seed=1,
        split_identity="s", target_schema="tep", num_fault_classes=3,
        tep_class_weights=(1.0, 2.0, 3.0),
    )
    model = UnifiedModel(config)
    assert model.head.class_weights.tolist() == [1.0, 2.0, 3.0]
    output = model.predict(HistoryBatch(
        torch.randn(2, 5, 3), torch.arange(5., dtype=torch.float32).expand(2, -1),
    ))
    loss = model.loss(output, TaskTargets(values=None, class_label=torch.tensor([1, 2]), schema="tep"))
    assert torch.isfinite(loss["total"])
