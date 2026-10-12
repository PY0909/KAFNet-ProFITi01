from typing import Iterable


FORBIDDEN_PREDICTION_FIELDS = frozenset({"Y_q", "y_flat", "rul"})


def validate_history_mapping(keys: Iterable[str]) -> None:
    leaked = FORBIDDEN_PREDICTION_FIELDS.intersection(keys)
    if leaked:
        raise ValueError(
            "prediction input contains target-only fields: {}".format(
                ", ".join(sorted(leaked))
            )
        )


def validate_prediction_output(output) -> None:
    required = ("dataset", "task", "encoder_version", "seed", "split_identity", "target_schema")
    absent = [name for name in required if not getattr(output, name, None) and getattr(output, name, None) != 0]
    if absent:
        raise ValueError("prediction output lacks identity: {}".format(", ".join(absent)))
    tensors = [output.point, output.quantiles, output.detection_probability, output.diagnosis_probability]
    for tensor in tensors:
        if tensor is not None:
            if not tensor.is_floating_point():
                raise ValueError("prediction tensors must use a floating dtype")
            if not tensor.isfinite().all():
                raise ValueError("prediction tensors must be finite")
