from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

from kaf_profiti.models.unified.output import PredictionOutput


@dataclass(frozen=True)
class BaselineConfig:
    model_name: str
    input_dim: int
    history_len: int
    pred_len: int
    seed: int
    split_identity: str
    target_schema: str
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    patch_len: int = 12
    top_k: int = 3
    training_cap: float = 125.0
    num_classes: int = 1
    tep_class_weights: tuple = ()

    def __post_init__(self):
        if self.model_name not in {"tcn", "gru", "patchtst", "timesnet", "itransformer"}:
            raise ValueError("unsupported comparison model: {}".format(self.model_name))
        for name in ("input_dim", "history_len", "pred_len", "d_model", "layers", "heads", "patch_len", "top_k", "num_classes"):
            if getattr(self, name) < 1:
                raise ValueError("{} must be positive".format(name))
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by attention heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.seed < 0 or not self.split_identity or not self.target_schema:
            raise ValueError("seed, split identity and target schema must be explicit")
        if self.tep_class_weights and len(self.tep_class_weights) != self.num_classes:
            raise ValueError("tep_class_weights must match num_classes")

    @property
    def version(self):
        return "compare_{}_v1".format(self.model_name)

    @property
    def task(self):
        if self.target_schema.startswith("cmapss_fd004_rul"):
            return "rul"
        if self.target_schema.startswith("tep_fault"):
            return "fault_detection_diagnosis"
        return "state"

    def to_dict(self):
        return asdict(self)


class ForecastBaseline(nn.Module):
    """Point-forecast interface shared by comparison architectures."""

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.model_version = config.version
        if config.task == "rul":
            self.rul_projection = nn.Linear(config.pred_len * config.input_dim, 1)
        elif config.task == "fault_detection_diagnosis":
            self.classifier = nn.Linear(config.pred_len * config.input_dim, config.num_classes)

    def _forecast(self, x):
        raise NotImplementedError

    def forward(self, history):
        x = history if torch.is_tensor(history) else history.X_obs
        if x.ndim != 3 or x.shape[1:] != (self.config.history_len, self.config.input_dim):
            raise ValueError("history must have shape [batch, history_len, input_dim]")
        if not torch.isfinite(x).all():
            raise ValueError("history must contain only finite values")
        forecast = self._forecast(x)
        expected = (x.shape[0], self.config.pred_len, self.config.input_dim)
        if tuple(forecast.shape) != expected:
            raise ValueError("forecast must have shape {}".format(expected))
        if not torch.isfinite(forecast).all():
            raise ValueError("forecast contains non-finite values")
        if self.config.task == "rul":
            return torch.nn.functional.softplus(self.rul_projection(forecast.flatten(start_dim=1))).squeeze(-1)
        if self.config.task == "fault_detection_diagnosis":
            return torch.softmax(self.classifier(forecast.flatten(start_dim=1)), dim=-1)
        return forecast

    def set_tep_class_weights(self, weights):
        """Adopt runner-computed train-frequency class weights.

        Keeps TEP baseline losses comparable with the unified probability
        head, which always trains with the same weights; the weights are
        stored on the config so checkpoint/prediction provenance records them.
        """
        weights = tuple(float(value) for value in weights)
        if not weights or len(weights) != self.config.num_classes:
            raise ValueError("tep class weights must be non-empty and match num_classes")
        self.config = replace(self.config, tep_class_weights=weights)

    def predict(self, history):
        values = self(history)
        if self.config.task == "fault_detection_diagnosis":
            return PredictionOutput(
                dataset="tep", task=self.config.task, encoder_version=self.model_version,
                seed=self.config.seed, split_identity=self.config.split_identity,
                target_schema=self.config.target_schema, point=values.argmax(dim=-1),
                detection_probability=(1.0 - values[:, 0]).clamp(0.0, 1.0),
                diagnosis_probability=values,
                metadata={"model_name": self.model_name, "config": self.config.to_dict()},
            )
        return PredictionOutput(
            dataset="cmapss_fd004" if self.config.task == "rul" else "metropt3",
            task=self.config.task,
            encoder_version=self.model_version,
            seed=self.config.seed,
            split_identity=self.config.split_identity,
            target_schema=self.config.target_schema,
            point=values,
            metadata={
                "model_name": self.model_name,
                "config": self.config.to_dict(),
                **({"target_convention": "raw_output_training_cap_{:g}".format(self.config.training_cap)}
                   if self.config.task == "rul" else {}),
            },
        )

    def loss(self, prediction, targets):
        if self.config.task == "fault_detection_diagnosis":
            if prediction.diagnosis_probability is None or targets.class_label is None:
                raise ValueError("diagnosis probabilities and class labels are required")
            probabilities = prediction.diagnosis_probability.clamp_min(1e-7)
            weight = None
            if self.config.tep_class_weights:
                weight = torch.as_tensor(
                    self.config.tep_class_weights, dtype=probabilities.dtype, device=probabilities.device
                )
            return torch.nn.functional.nll_loss(
                probabilities.log(), targets.class_label.long().reshape(-1), weight=weight
            )
        if prediction.point is None or targets.values is None:
            raise ValueError("point prediction and state targets are required")
        point = prediction.point.reshape(-1)
        target = targets.values.reshape(-1)
        if point.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        if not torch.isfinite(point).all() or not torch.isfinite(target).all():
            raise ValueError("MAE requires finite predictions and targets")
        return (point - target).abs().mean()
