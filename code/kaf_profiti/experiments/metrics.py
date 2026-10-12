"""Task-specific metrics used by experiment artifacts."""

import torch


def state_quantile_metrics(quantiles, target, sensor_names, quantile_levels=(0.1, 0.5, 0.9)):
    """Report pinball, central coverage, and interval width in raw units."""
    quantiles = torch.as_tensor(quantiles).detach().float().cpu()
    target = torch.as_tensor(target).detach().float().cpu()
    levels = torch.as_tensor(quantile_levels, dtype=quantiles.dtype)
    if quantiles.ndim != target.ndim + 1 or quantiles.shape[:-1] != target.shape:
        raise ValueError("quantiles must have shape target_shape + [quantile]")
    if quantiles.shape[-1] != len(levels) or quantiles.shape[-2] != len(sensor_names):
        raise ValueError("quantile levels and sensor names do not match")
    if not torch.isfinite(quantiles).all() or not torch.isfinite(target).all():
        raise ValueError("quantile metrics require finite values")
    error = target.unsqueeze(-1) - quantiles
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    lower, upper = quantiles[..., 0], quantiles[..., -1]
    median = quantiles[..., len(levels) // 2]
    alpha = float((levels[-1] - levels[0]).item())
    if not 0.0 < alpha < 1.0:
        raise ValueError("central quantile interval must have width in (0, 1)")
    interval_score = upper - lower
    interval_score = interval_score + (2.0 / alpha) * (lower - target).clamp_min(0.0)
    interval_score = interval_score + (2.0 / alpha) * (target - upper).clamp_min(0.0)
    # One central interval plus the median point forecast, using standard WIS weights.
    wis = (0.5 * (median - target).abs() + (alpha / 2.0) * interval_score) / 1.5
    per_sensor = {}
    for index, name in enumerate(sensor_names):
        values = pinball[..., index, :].mean()
        coverage = ((target[..., index] >= lower[..., index]) & (target[..., index] <= upper[..., index])).float().mean()
        width = (upper[..., index] - lower[..., index]).mean()
        per_sensor[str(name)] = {
            "pinball": float(values), "coverage": float(coverage), "interval_width": float(width),
            "wis": float(wis[..., index].mean()),
        }
    return {
        "pinball": float(pinball.mean()),
        "coverage": float(((target >= lower) & (target <= upper)).float().mean()),
        "interval_width": float((upper - lower).mean()),
        "wis": float(wis.mean()),
        "point_mae_from_median": float((median - target).abs().mean()),
        "point_rmse_from_median": float((median - target).square().mean().sqrt()),
        "per_sensor": per_sensor,
        "quantile_levels": [float(level) for level in levels],
    }


def calibrate_state_quantiles(validation_quantiles, validation_target, test_quantiles, coverage=0.8):
    """Apply validation-only split-conformal central interval calibration."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie in (0, 1)")
    validation_quantiles = torch.as_tensor(validation_quantiles).detach().float().cpu()
    validation_target = torch.as_tensor(validation_target).detach().float().cpu()
    test_quantiles = torch.as_tensor(test_quantiles).detach().float().cpu()
    if validation_quantiles.ndim != validation_target.ndim + 1 or test_quantiles.ndim != validation_quantiles.ndim:
        raise ValueError("quantile calibration shapes are invalid")
    median = validation_quantiles[..., 1]
    # Calibrate the central interval around the predicted median. Using the
    # raw interval miss score would yield zero radius whenever the uncalibrated
    # interval contains most validation points, even when test residuals shift.
    nonconformity = (validation_target - median).abs()
    radius = torch.quantile(nonconformity.reshape(-1, nonconformity.shape[-1]), coverage, dim=0)
    calibrated = test_quantiles.clone()
    radius_view = radius.view(1, 1, -1)
    calibrated[..., 0] = calibrated[..., 0] - radius_view
    calibrated[..., -1] = calibrated[..., -1] + radius_view
    return calibrated, radius


def calibrate_state_quantiles_cqr(validation_quantiles, validation_target, test_quantiles, coverage=0.8):
    """Apply validation-only CQR calibration to the central interval.

    The calibration score is the standard interval nonconformity
    ``max(lower - y, y - upper, 0)``.  A channel-wise radius is estimated from
    validation samples and added symmetrically to the lower and upper
    quantiles at test time.  No test target is read by this function.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie in (0, 1)")
    validation_quantiles = torch.as_tensor(validation_quantiles).detach().float().cpu()
    validation_target = torch.as_tensor(validation_target).detach().float().cpu()
    test_quantiles = torch.as_tensor(test_quantiles).detach().float().cpu()
    if validation_quantiles.ndim != validation_target.ndim + 1 or test_quantiles.ndim != validation_quantiles.ndim:
        raise ValueError("quantile calibration shapes are invalid")
    if validation_quantiles.shape[-1] < 3:
        raise ValueError("CQR requires lower, median, and upper quantiles")
    lower = validation_quantiles[..., 0]
    upper = validation_quantiles[..., -1]
    nonconformity = torch.maximum(
        torch.maximum(lower - validation_target, validation_target - upper),
        torch.zeros_like(validation_target),
    )
    radius = torch.quantile(nonconformity.reshape(-1, nonconformity.shape[-1]), coverage, dim=0)
    calibrated = test_quantiles.clone()
    radius_view = radius.view(1, 1, -1)
    calibrated[..., 0] = calibrated[..., 0] - radius_view
    calibrated[..., -1] = calibrated[..., -1] + radius_view
    return calibrated, radius


def state_metrics(prediction, target, sensor_names):
    prediction = torch.as_tensor(prediction).detach().float().cpu()
    target = torch.as_tensor(target).detach().float().cpu()
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("state prediction and target must have equal [..., channels] shapes")
    if prediction.shape[-1] != len(sensor_names):
        raise ValueError("sensor name count must match the final channel dimension")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("state metrics require finite predictions and targets")
    error = prediction - target
    flattened = error.reshape(-1, error.shape[-1])
    per_sensor = {}
    for index, name in enumerate(sensor_names):
        channel_error = flattened[:, index]
        per_sensor[str(name)] = {
            "mae": float(channel_error.abs().mean()),
            "rmse": float(channel_error.square().mean().sqrt()),
        }
    return {
        "overall": {
            "mae": float(flattened.abs().mean()),
            "rmse": float(flattened.square().mean().sqrt()),
        },
        "per_sensor": per_sensor,
        "elements": int(flattened.numel()),
    }


def state_quantile_diagnostics(quantiles, target, sensor_names, num_blocks=4):
    """Report interval metrics per sensor and ordered contiguous test block."""
    quantiles = torch.as_tensor(quantiles).detach().float().cpu()
    target = torch.as_tensor(target).detach().float().cpu()
    if num_blocks < 1 or quantiles.ndim != target.ndim + 1:
        raise ValueError("invalid quantile diagnostic inputs")
    count = target.shape[0]
    if count < num_blocks:
        num_blocks = count
    blocks = []
    for index in range(num_blocks):
        lo = (count * index) // num_blocks
        hi = (count * (index + 1)) // num_blocks
        metrics = state_quantile_metrics(quantiles[lo:hi], target[lo:hi], sensor_names)
        blocks.append({"block": index, "start": lo, "end": hi, "examples": hi - lo, **metrics})
    return {"num_blocks": num_blocks, "examples": count, "per_block": blocks}


def rul_metrics(prediction, target, engine_ids=None):
    prediction = torch.as_tensor(prediction).detach().float().cpu().reshape(-1)
    target = torch.as_tensor(target).detach().float().cpu().reshape(-1)
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("RUL prediction and target must be non-empty and have equal shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("RUL metrics require finite raw-cycle values")
    error = prediction - target
    penalty = torch.where(error < 0, torch.exp(-error / 13.0), torch.exp(error / 10.0)) - 1.0
    result = {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "nasa_score": float(penalty.sum()),
        "units": "cycles",
        "engines": int(error.numel()),
    }
    if engine_ids is not None:
        engine_ids = [str(engine) for engine in engine_ids]
        if len(engine_ids) != error.numel() or len(set(engine_ids)) != len(engine_ids):
            raise ValueError("engine_ids must uniquely identify every RUL target")
        result["per_engine"] = {
            engine: {
                "target": float(target[index]),
                "prediction": float(prediction[index]),
                "absolute_error": float(error[index].abs()),
            }
            for index, engine in enumerate(engine_ids)
        }
    return result


def select_tep_threshold(probability, labels, objective="f1"):
    """Select a detection threshold using validation labels only."""
    probability = torch.as_tensor(probability).detach().float().cpu().reshape(-1)
    labels = torch.as_tensor(labels).detach().long().cpu().reshape(-1)
    if probability.numel() == 0 or probability.shape != labels.shape:
        raise ValueError("validation probabilities and labels must be non-empty and aligned")
    if objective != "f1":
        raise ValueError("unsupported TEP threshold objective")
    candidates = torch.unique(torch.cat((probability, torch.tensor([0.5])))).sort().values
    best = (0.5, -1.0)
    for threshold in candidates.tolist():
        predicted = probability >= threshold
        truth = labels > 0
        tp = float((predicted & truth).sum())
        fp = float((predicted & ~truth).sum())
        fn = float((~predicted & truth).sum())
        score = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        candidate = (float(threshold), score)
        if score > best[1] or (score == best[1] and threshold < best[0]):
            best = candidate
    return best


def classification_metrics(labels, predictions, num_classes, run_ids=None):
    labels = torch.as_tensor(labels).detach().long().cpu().reshape(-1)
    predictions = torch.as_tensor(predictions).detach().long().cpu().reshape(-1)
    if labels.numel() == 0 or labels.shape != predictions.shape:
        raise ValueError("classification labels and predictions must be non-empty and have equal shapes")
    if num_classes < 1 or labels.min() < 0 or predictions.min() < 0:
        raise ValueError("classification labels and class count must be nonnegative and valid")
    if labels.max() >= num_classes or predictions.max() >= num_classes:
        raise ValueError("classification label exceeds num_classes")
    if run_ids is not None and len(run_ids) != labels.numel():
        raise ValueError("run_ids must contain one composite run identity per example")

    def summarize(actual, predicted):
        matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for truth, guess in zip(actual.tolist(), predicted.tolist()):
            matrix[truth, guess] += 1
        support = matrix.sum(dim=1)
        predicted_count = matrix.sum(dim=0)
        class_metrics = {}
        recalls = []
        active_f1_values = []
        for class_id in range(num_classes):
            true_positive = int(matrix[class_id, class_id])
            class_support = int(support[class_id])
            predicted_total = int(predicted_count[class_id])
            precision = true_positive / predicted_total if predicted_total else 0.0
            recall = true_positive / class_support if class_support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_metrics[str(class_id)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": class_support,
            }
            if class_support or predicted_total:
                active_f1_values.append(f1)
            if class_support:
                recalls.append(recall)
        accuracy = float((actual == predicted).sum()) / actual.numel()
        return {
            "accuracy": accuracy,
            "balanced_accuracy": sum(recalls) / len(recalls),
            "macro_f1": sum(active_f1_values) / len(active_f1_values),
            "per_class": class_metrics,
            "confusion_matrix": matrix.tolist(),
            "support": int(actual.numel()),
        }

    result = summarize(labels, predictions)
    result["classes"] = [str(index) for index in range(num_classes)]
    if run_ids is not None:
        grouped = {}
        for index, run_id in enumerate(run_ids):
            grouped.setdefault(str(run_id), []).append(index)
        per_run = {
            run_id: summarize(labels[indices], predictions[indices])
            for run_id, indices in sorted(grouped.items())
        }
        result["per_run"] = per_run
        result["run_summary"] = {
            "run_count": len(per_run),
            "mean_macro_f1": sum(item["macro_f1"] for item in per_run.values()) / len(per_run),
            "mean_balanced_accuracy": sum(item["balanced_accuracy"] for item in per_run.values()) / len(per_run),
            "mean_accuracy": sum(item["accuracy"] for item in per_run.values()) / len(per_run),
        }
    return result
