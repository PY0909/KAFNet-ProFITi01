from collections import defaultdict

import torch
from torch import nn

from ..output import PredictionOutput
from .base import TaskHead, TaskTargets


class TEPHead(TaskHead):
    def __init__(self, hidden_dim: int, num_fault_classes: int = 1, class_weights=()):
        super().__init__()
        if num_fault_classes < 1:
            raise ValueError("num_fault_classes must be positive")
        self.num_fault_classes = num_fault_classes
        if class_weights and len(class_weights) != num_fault_classes:
            raise ValueError("class_weights must match num_fault_classes")
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32) if class_weights else torch.ones(num_fault_classes))
        self.detection = nn.Linear(hidden_dim, 1)
        self.diagnosis = nn.Linear(hidden_dim, num_fault_classes)

    def forward(self, encoded, mode="probability") -> PredictionOutput:
        if mode != "probability":
            raise ValueError("TEP head only supports probability mode")
        state = encoded.z_global
        detection = torch.sigmoid(self.detection(state).squeeze(-1))
        diagnosis = torch.softmax(self.diagnosis(state), dim=-1)
        return PredictionOutput(
            dataset="tep", task="fault_detection_diagnosis", encoder_version="unified_temporal_channelwise_v2",
            seed=-1, split_identity="unbound", target_schema="tep_fault_v1",
            detection_probability=detection, diagnosis_probability=diagnosis,
        )

    def loss(self, prediction, targets: TaskTargets):
        if targets.class_label is None:
            raise ValueError("TEP class_label is required")
        labels = targets.class_label.long()
        detection_target = (labels > 0).to(prediction.detection_probability.dtype)
        # BCEWithLogits is autocast-safe; the public output remains a probability.
        detection_logits = torch.logit(prediction.detection_probability.clamp(1e-6, 1.0 - 1e-6))
        detection_loss = nn.functional.binary_cross_entropy_with_logits(detection_logits, detection_target)
        diagnosis_loss = nn.functional.nll_loss(
            prediction.diagnosis_probability.clamp_min(1e-7).log(),
            labels.clamp(0, self.num_fault_classes - 1), weight=self.class_weights,
        )
        return {"detection": detection_loss, "diagnosis": diagnosis_loss, "total": detection_loss + diagnosis_loss}


def detection_delay(probability, labels, run_ids, samples, threshold=0.5):
    delays = []
    for run in torch.unique(run_ids).tolist():
        selected = run_ids == run
        run_samples = samples[selected]
        run_labels = labels[selected]
        run_probability = probability[selected]
        fault_positions = run_samples[run_labels > 0]
        if fault_positions.numel() == 0:
            continue
        onset = fault_positions.min().item()
        alarms = run_samples[(run_samples >= onset) & (run_probability >= threshold)]
        if alarms.numel():
            delays.append(float(alarms.min().item() - onset))
    return float(sum(delays) / len(delays)) if delays else float("nan")


def evaluate_tep_events(probability, labels, run_ids, samples, threshold=0.5):
    """Evaluate fault events over complete composite runs.

    A faulty run is detected only by an alarm at or after its first positive
    label. Normal runs contribute to the false-alarm denominator. The raw
    delay list is retained so callers can report its distribution rather than
    only an average over detected runs.
    """
    probability = probability.detach().flatten()
    labels = labels.detach().flatten().long()
    run_ids = run_ids.detach().flatten()
    samples = samples.detach().flatten()
    if not (probability.numel() == labels.numel() == run_ids.numel() == samples.numel()):
        raise ValueError("TEP event inputs must have equal lengths")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    fault_runs = detected_runs = missed_runs = false_alarm_runs = 0
    delays = []
    false_alarm_delays = []
    for run in torch.unique(run_ids).tolist():
        selected = run_ids == run
        run_labels = labels[selected]
        run_probability = probability[selected]
        run_samples = samples[selected]
        alarms = run_samples[run_probability >= threshold]
        positives = run_samples[run_labels > 0]
        if positives.numel():
            fault_runs += 1
            onset = positives.min()
            post_onset = alarms[alarms >= onset]
            if post_onset.numel():
                detected_runs += 1
                delays.append(float(post_onset.min().item() - onset.item()))
            else:
                missed_runs += 1
        elif alarms.numel():
            false_alarm_runs += 1
            false_alarm_delays.append(float(alarms.min().item()))
    normal_runs = int(torch.unique(run_ids).numel()) - fault_runs
    return {
        "threshold": float(threshold),
        "run_count": int(torch.unique(run_ids).numel()),
        "fault_run_count": fault_runs,
        "normal_run_count": normal_runs,
        "detected_run_count": detected_runs,
        "missed_run_count": missed_runs,
        "false_alarm_run_count": false_alarm_runs,
        "detection_rate": detected_runs / fault_runs if fault_runs else float("nan"),
        "miss_rate": missed_runs / fault_runs if fault_runs else float("nan"),
        "false_alarm_rate": false_alarm_runs / normal_runs if normal_runs else float("nan"),
        "delay_distribution": delays,
        "false_alarm_position_distribution": false_alarm_delays,
        "mean_detection_delay": sum(delays) / len(delays) if delays else float("nan"),
    }
