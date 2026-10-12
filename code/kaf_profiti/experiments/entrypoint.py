"""Small dataset-bundle training entrypoint used by smoke and real adapters."""

from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter

import torch
from tqdm.auto import tqdm

from .artifacts import ArtifactWriter
from .cudtr_diagnostics import summarize_cudtr_diagnostics
from .task_runner import evaluate, precompute_counterfactual_utilities, train_step
from ..industrial.batch import collate_history, HistoryBatch
from ..industrial.datasets import bundle_content_provenance
from ..industrial.protocol import TrainNormalizer
from ..models.unified import ModelConfig, UnifiedModel
from ..models.unified.heads.base import TaskTargets


def _task_for_dataset(dataset):
    return {"metropt3": "state", "cmapss_fd004": "rul", "tep": "fault_detection_diagnosis"}[dataset]


def _loader(samples, batch_size):
    for offset in range(0, len(samples), batch_size):
        batch = collate_history(samples[offset : offset + batch_size])
        yield batch, batch.targets


def _index_batch(batch, index):
    """Gathered view of a collated HistoryBatch at ``index`` (slice or tensor)."""
    targets = batch.targets

    def optional_index(tensor):
        return None if tensor is None else tensor[index]

    sliced_targets = TaskTargets(
        values=optional_index(targets.values),
        schema=targets.schema,
        class_label=optional_index(targets.class_label),
        fault_number=optional_index(targets.fault_number),
        simulation_run=optional_index(targets.simulation_run),
        raw_rul=optional_index(targets.raw_rul),
        capped_rul=optional_index(targets.capped_rul),
    )
    return HistoryBatch(
        X_obs=optional_index(batch.X_obs),
        T_obs=optional_index(batch.T_obs),
        context=optional_index(batch.context),
        targets=sliced_targets,
        M_obs=optional_index(batch.M_obs),
    )


def _slice_batch(batch, lo, hi):
    return _index_batch(batch, slice(lo, hi))


def _split_validation_samples(samples, selection_fraction=0.5):
    """Split validation samples without sharing raw window intervals.

    Native adapters attach ``window_start``/``window_end`` in raw row units.
    Synthetic or legacy samples without those fields retain the historical
    chronological list split.
    """
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("validation selection fraction must lie in (0, 1)")
    count = len(samples)
    if count < 2:
        # Tiny synthetic smoke bundles cannot support isolation; production
        # protocols must provide at least two validation windows.
        return samples, samples
    ordered = sorted(samples, key=lambda sample: (sample.metadata or {}).get("window_start", 0))
    cut = min(count - 1, max(1, int(count * selection_fraction)))
    selected, calibration = ordered[:cut], ordered[cut:]
    metadata = [sample.metadata or {} for sample in ordered]
    if not all("window_start" in item and "window_end" in item for item in metadata):
        return selected, calibration
    selection_end = max(item.metadata["window_end"] for item in selected)
    calibration = [
        sample for sample in calibration
        if sample.metadata["window_start"] > selection_end
    ]
    if not calibration:
        raise ValueError("validation selection/calibration windows overlap; increase validation size")
    return selected, calibration


def _validation_window_provenance(selection, calibration):
    if {id(sample) for sample in selection} & {id(sample) for sample in calibration}:
        return {
            "selection_window_range": None,
            "calibration_window_range": None,
            "window_overlap_checked": False,
            "window_gap": None,
        }
    def bounds(samples):
        metadata = [sample.metadata or {} for sample in samples]
        if not samples or not all("window_start" in item and "window_end" in item for item in metadata):
            return None
        return {
            "start": min(item["window_start"] for item in metadata),
            "end": max(item["window_end"] for item in metadata),
        }

    selected_bounds = bounds(selection)
    calibration_bounds = bounds(calibration)
    if selected_bounds is None or calibration_bounds is None:
        return {
            "selection_window_range": None,
            "calibration_window_range": None,
            "window_overlap_checked": False,
            "window_gap": None,
        }
    return {
        "selection_window_range": selected_bounds,
        "calibration_window_range": calibration_bounds,
        "window_overlap_checked": True,
        "window_gap": calibration_bounds["start"] - selected_bounds["end"] - 1,
    }


def _resident_collated(samples, device):
    """Collate a split once and keep every tensor resident on ``device``.

    Returns None when device is not CUDA or the split does not comfortably
    fit in free device memory, so callers fall back to per-batch CPU loading.
    """
    if device.type != "cuda":
        return None
    first = samples[0]
    per_sample_bytes = first.X_obs.numel() * first.X_obs.element_size()
    per_sample_bytes += first.T_obs.numel() * first.T_obs.element_size()
    for tensor in (first.M_obs, first.context, first.target, first.class_label):
        if tensor is not None:
            per_sample_bytes += tensor.numel() * tensor.element_size()
    estimated_bytes = per_sample_bytes * len(samples)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if estimated_bytes * 1.3 > free_bytes * 0.5:
        return None
    return collate_history(samples).to(device)


def _resident_batches(samples, batch_size, device):
    """Static resident batch list as (HistoryBatch, targets) tuples."""
    resident = _resident_collated(samples, device)
    if resident is None:
        return None
    batches = []
    for offset in range(0, len(samples), batch_size):
        sliced = _slice_batch(resident, offset, min(offset + batch_size, len(samples)))
        batches.append((sliced, sliced.targets))
    return batches


def _stats_sha256(payload):
    """Canonical content hash of normalization statistics.

    ``normalizer_identity`` only carries the feature count; this hash covers
    the fitted mean/std values (and per-condition statistics) so identical
    identities can still be audited for identical statistics.
    """
    import hashlib
    import json as _json

    canonical = _json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_bundle(bundle, normalize_targets=False, feature_normalization="global"):
    from dataclasses import replace

    if feature_normalization not in {"global", "condition"}:
        raise ValueError("feature_normalization must be global or condition")

    train_values = torch.cat([
        sample.X_obs.reshape(-1, sample.X_obs.shape[-1]) for sample in bundle.samples["train"]
    ])
    normalizer = TrainNormalizer().fit(train_values, split="train")
    if feature_normalization == "condition":
        grouped_values = {}
        for sample in bundle.samples["train"]:
            key = (sample.metadata or {}).get("condition_key", "global")
            grouped_values.setdefault(key, []).append(sample.X_obs.reshape(-1, sample.X_obs.shape[-1]))
        condition_stats = {}
        for key, chunks in grouped_values.items():
            values = torch.cat(chunks).float()
            condition_stats[key] = (values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-6))

        def transform(sample):
            key = (sample.metadata or {}).get("condition_key", "global")
            mean, std = condition_stats.get(key, (normalizer.mean, normalizer.std))
            return (sample.X_obs.float() - mean) / std

        normalizer_identity = "train-condition-zscore-features{}".format(normalizer.mean.numel())
    else:
        transform = lambda sample: normalizer.transform(sample.X_obs)
        normalizer_identity = normalizer.identity
    normalized = {
        split: [replace(sample, X_obs=transform(sample)) for sample in samples]
        for split, samples in bundle.samples.items()
    }
    metadata = dict(bundle.metadata)
    metadata["normalizer_identity"] = normalizer_identity
    metadata["feature_normalization"] = feature_normalization
    metadata["normalizer_source_split"] = "train"
    # 手工构造的 bundle 不经过 adapter；runner 层兜底保证任何 run 的 manifest
    # 都带有内容级 split/window 哈希，审计脚本不依赖 bundle 来源。
    for key, value in bundle_content_provenance(bundle.samples).items():
        metadata.setdefault(key, value)
    stats_payload = {"global": {"mean": normalizer.mean.tolist(), "std": normalizer.std.tolist()}}
    if feature_normalization == "condition":
        stats_payload["conditions"] = {
            key: {"mean": pair[0].tolist(), "std": pair[1].tolist()}
            for key, pair in sorted(condition_stats.items())
        }
    metadata["normalizer_stats_sha256"] = _stats_sha256(stats_payload)
    normalized_bundle = type(bundle)(bundle.dataset, normalized, bundle.split_identity, bundle.target_schema, metadata)
    if not normalize_targets:
        return normalized_bundle
    train_targets = [sample.target for sample in normalized_bundle.samples["train"] if sample.target is not None]
    if not train_targets:
        raise ValueError("state target normalization requires train targets")
    flattened = torch.cat([target.reshape(-1, target.shape[-1]) for target in train_targets], dim=0).float()
    if not torch.isfinite(flattened).all():
        raise ValueError("state target normalizer cannot fit non-finite targets")
    target_mean = flattened.mean(dim=0)
    target_std = flattened.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_stats = {
        "source_split": "train",
        "identity": "train-zscore-targets{}".format(target_mean.numel()),
        "mean": target_mean,
        "std": target_std,
        "stats_sha256": _stats_sha256({"targets": {"mean": target_mean.tolist(), "std": target_std.tolist()}}),
    }
    normalized_targets = {
        split: [
            replace(sample, target=None if sample.target is None else (sample.target.float() - target_mean) / target_std)
            for sample in samples
        ]
        for split, samples in normalized_bundle.samples.items()
    }
    metadata["target_normalizer_identity"] = target_stats["identity"]
    metadata["target_normalizer_source_split"] = target_stats["source_split"]
    metadata["target_normalizer_mean"] = target_mean.tolist()
    metadata["target_normalizer_std"] = target_std.tolist()
    return type(normalized_bundle)(normalized_bundle.dataset, normalized_targets, normalized_bundle.split_identity,
                                  normalized_bundle.target_schema, metadata), target_stats


def _load_cudtr_teacher(student, checkpoint_path, device):
    """Warm-start the CUDTR student and return its frozen baseline teacher.

    V4 is the repository's current implementation of the V3C baseline
    configuration.  It has a different serialized variant name because it
    includes the recent/trend residual block, so the teacher must match the
    student's variant exactly rather than being silently treated as V3C.
    """
    if checkpoint_path is None:
        raise ValueError("CUDTR teacher training requires a frozen V3C checkpoint")
    teacher_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_fields = ModelConfig.__dataclass_fields__
    teacher_config_data = {
        key: value for key, value in teacher_payload["config"].items() if key in config_fields
    }
    teacher_config_data["patch_lens"] = tuple(teacher_config_data["patch_lens"])
    teacher_config = ModelConfig(**teacher_config_data)
    expected_fields = (
        "dataset", "task", "input_dim", "seed", "split_identity", "target_schema",
        "training_cap", "rul_over_penalty", "rul_head_input",
        "sensor_pooling", "patch_lens", "cross_channel_mode",
    )
    mismatches = [
        name for name in expected_fields
        if getattr(teacher_config, name) != getattr(student.config, name)
    ]
    if teacher_config.variant not in {"v3c", "v4"} or teacher_config.variant != student.config.variant:
        mismatches.append("teacher/student baseline variant match (v3c or v4)")
    if teacher_config.head_mode != student.config.head_mode and not (
        teacher_config.task == "state"
        and teacher_config.head_mode == "point"
        and student.config.head_mode == "quantile"
    ):
        mismatches.append("head_mode (only point teacher -> quantile state student is allowed)")
    if teacher_config.cudtr_enabled:
        mismatches.append("cudtr_enabled=False")
    if mismatches:
        raise ValueError("frozen utility teacher config mismatch: {}".format(", ".join(mismatches)))

    load_result = student.load_state_dict(teacher_payload["model"], strict=False)
    allowed_missing = {name for name in load_result.missing_keys if name.startswith("cudtr.") or name == "cudtr_queries"}
    if set(load_result.missing_keys) != allowed_missing or load_result.unexpected_keys:
        raise ValueError("V3C checkpoint does not match the CUDTR student base architecture")
    teacher = UnifiedModel(teacher_config).to(device)
    teacher.load_state_dict(teacher_payload["model"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def run_bundle(bundle, result_root, seed=2026, epochs=1, batch_size=128, head_mode="point", evaluate_test=False,
               progress=False, device="auto", amp=None, patience=None, scheduler="plateau", run_id=None,
               model_factory=None, model_name=None, encoder_variant="v2", feature_normalization="global",
               patch_lens=None, rul_over_penalty=1.0, compile_model=False, shuffle=True,
               learning_rate=1e-3, rul_head_input="global", sensor_pooling="attention",
               cross_channel_mode="attention",
               state_decoder="global",
               cudtr_enabled=False, cudtr_top_k=2, cudtr_query_dim=8, cudtr_temperature=1.0,
               cudtr_fallback_init=0.5, cudtr_lambda_distill=0.0, cudtr_lambda_rank=0.0,
               cudtr_lambda_sparse=0.0, cudtr_lambda_fallback=0.0, cudtr_rank_margin=0.1,
               cudtr_utility_clip=None, cudtr_teacher_enabled=False, cudtr_fallback_drop_rate=0.5,
               cudtr_teacher_checkpoint=None, cudtr_router_only_epochs=0,
               cudtr_min_observation_rate=0.0, cudtr_utility_threshold=None,
               quantile_calibration="median_residual", validation_selection_fraction=0.5):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if quantile_calibration not in {"median_residual", "cqr"}:
        raise ValueError("quantile_calibration must be median_residual or cqr")
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    task = _task_for_dataset(bundle.dataset)
    for split in ("train", "validation"):
        if not bundle.samples.get(split):
            raise ValueError("{} split must not be empty".format(split))
    normalized_result = _normalize_bundle(
        bundle, normalize_targets=(task == "state"), feature_normalization=feature_normalization,
    )
    if task == "state":
        bundle, target_stats = normalized_result
    else:
        bundle, target_stats = normalized_result, None
    # Split references must come from the normalized bundle. Splitting before
    # train-only normalization would leave validation selection on raw targets
    # while training uses normalized targets.
    if task == "state" and head_mode == "quantile":
        validation_selection_samples, validation_calibration_samples = _split_validation_samples(
            bundle.samples["validation"], validation_selection_fraction,
        )
    else:
        validation_selection_samples = bundle.samples["validation"]
        validation_calibration_samples = bundle.samples["validation"]
    input_dim = bundle.samples[next(iter(bundle.samples))][0].X_obs.shape[-1]
    tep_class_weights = tuple(bundle.metadata.get("tep_class_weights", ()))
    if task == "fault_detection_diagnosis" and not tep_class_weights:
        labels = [int(sample.class_label.item()) for sample in bundle.samples["train"] if sample.class_label is not None]
        class_count = int(bundle.metadata.get("num_fault_classes", max(labels, default=0) + 1))
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=class_count).float()
        if (counts == 0).any():
            raise ValueError("TEP train split must contain every diagnosis class for class weighting")
        weights = counts.sum() / (counts * class_count)
        tep_class_weights = tuple(float(value) for value in weights.tolist())
    config = ModelConfig(
        dataset=bundle.dataset,
        task=task,
        input_dim=input_dim,
        seed=seed,
        split_identity=bundle.split_identity,
        target_schema=bundle.target_schema,
        pred_len=(bundle.samples["train"][0].target.shape[0] if task == "state" else 1),
        head_mode="probability" if task == "fault_detection_diagnosis" else head_mode,
        num_fault_classes=int(bundle.metadata.get("num_fault_classes", 1 + max(
            (int(sample.class_label.item()) for sample in bundle.samples["train"] if sample.class_label is not None),
            default=0,
        ))),
        training_cap=bundle.metadata.get("training_cap", 125.0),
        variant=encoder_variant,
        patch_lens=patch_lens if patch_lens is not None else (12, 24, 48),
        rul_over_penalty=rul_over_penalty,
        rul_head_input=rul_head_input,
        sensor_pooling=sensor_pooling,
        cross_channel_mode=cross_channel_mode,
        tep_class_weights=tep_class_weights,
        state_decoder=state_decoder,
        cudtr_enabled=cudtr_enabled,
        cudtr_top_k=cudtr_top_k,
        cudtr_query_dim=cudtr_query_dim,
        cudtr_temperature=cudtr_temperature,
        cudtr_fallback_init=cudtr_fallback_init,
        cudtr_lambda_distill=cudtr_lambda_distill,
        cudtr_lambda_rank=cudtr_lambda_rank,
        cudtr_lambda_sparse=cudtr_lambda_sparse,
        cudtr_lambda_fallback=cudtr_lambda_fallback,
        cudtr_rank_margin=cudtr_rank_margin,
        cudtr_utility_clip=cudtr_utility_clip,
        cudtr_fallback_drop_rate=cudtr_fallback_drop_rate,
        cudtr_min_observation_rate=cudtr_min_observation_rate,
        cudtr_utility_threshold=cudtr_utility_threshold,
    )
    if cudtr_teacher_enabled and not cudtr_enabled:
        raise ValueError("cudtr_teacher_enabled requires cudtr_enabled")
    if cudtr_teacher_enabled and encoder_variant not in {"v3c", "v4"}:
        raise ValueError("CUDTR frozen-teacher training requires encoder_variant='v3c' or 'v4'")
    if not 0 <= cudtr_router_only_epochs <= epochs:
        raise ValueError("cudtr_router_only_epochs must be between zero and epochs")
    if cudtr_router_only_epochs and not cudtr_teacher_enabled:
        raise ValueError("router-only warmup requires the frozen CUDTR teacher")
    if (cudtr_lambda_distill > 0 or cudtr_lambda_rank > 0) and not cudtr_teacher_enabled:
        raise ValueError("CUDTR utility loss weights require cudtr_teacher_enabled")
    if not cudtr_enabled and any((cudtr_lambda_distill, cudtr_lambda_rank, cudtr_lambda_sparse, cudtr_lambda_fallback)):
        raise ValueError("CUDTR loss weights require cudtr_enabled")
    if model_factory is None:
        model = UnifiedModel(config).to(device)
        resolved_model_name = "unified"
        resolved_model_version = config.encoder_version
    else:
        model = model_factory(
            input_dim=input_dim,
            history_len=bundle.samples["train"][0].X_obs.shape[0],
            pred_len=config.pred_len,
            seed=seed,
            split_identity=bundle.split_identity,
            target_schema=bundle.target_schema,
        ).to(device)
        resolved_model_name = model_name or getattr(model, "model_name", "comparison")
        resolved_model_version = getattr(model, "model_version", "comparison_v1")
    if task == "fault_detection_diagnosis" and tep_class_weights:
        # Comparison baselines adopt the runner-computed train-frequency class
        # weights so their TEP losses stay comparable with the unified head.
        setter = getattr(model, "set_tep_class_weights", None)
        if callable(setter):
            setter(tep_class_weights)
    utility_teacher_model = None
    if cudtr_teacher_enabled:
        if model_factory is not None:
            raise ValueError("cudtr_teacher_enabled requires the UnifiedModel factory")
        utility_teacher_model = _load_cudtr_teacher(model, cudtr_teacher_checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    elif scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif scheduler in (None, "none"):
        lr_scheduler = None
    else:
        raise ValueError("scheduler must be plateau, cosine, or none")
    use_amp = device.type == "cuda" and (amp is None or bool(amp))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_started = perf_counter()
    train_samples = bundle.samples["train"]
    resident_train = _resident_collated(train_samples, device)
    validation_batches = _resident_batches(validation_selection_samples, batch_size, device)
    validation_calibration_batches = _resident_batches(validation_calibration_samples, batch_size, device)
    if validation_batches is None:
        validation_batches = list(_loader(validation_selection_samples, batch_size))
    if validation_calibration_batches is None:
        validation_calibration_batches = list(_loader(validation_calibration_samples, batch_size))
    utility_cache = None
    if utility_teacher_model is not None:
        # Utility labels are train-only artifacts. Compute them once before
        # optimization and reuse them across shuffled epochs.
        ordered_batches = [
            (batch.to(device), targets.to(device))
            for batch, targets in _loader(train_samples, batch_size)
        ]
        utility_cache = precompute_counterfactual_utilities(
            model, ordered_batches, teacher_model=utility_teacher_model,
        )
    # 样本级 per-epoch shuffle：固定 seed 的独立生成器，保证同 seed 可复现。
    # 不 shuffle 时保持与旧实现完全一致的静态批顺序。
    shuffle_generator = torch.Generator().manual_seed(seed + 9781) if shuffle else None

    def _train_batches():
        count = len(train_samples)
        if resident_train is None:
            if shuffle_generator is None:
                result = []
                for offset, (batch, targets) in enumerate(_loader(train_samples, batch_size)):
                    utility = None if utility_cache is None else utility_cache[offset * batch_size:offset * batch_size + batch.X_obs.shape[0]]
                    result.append((batch, targets, utility))
                return result
            order = torch.randperm(count, generator=shuffle_generator).tolist()
            batches = []
            for offset in range(0, count, batch_size):
                chunk = [train_samples[i] for i in order[offset : offset + batch_size]]
                batch = collate_history(chunk)
                utility = None if utility_cache is None else utility_cache[order[offset:offset + batch_size]]
                batches.append((batch, batch.targets, utility))
            return batches
        order = (
            None if shuffle_generator is None
            else torch.randperm(count, generator=shuffle_generator).to(device)
        )
        batches = []
        for offset in range(0, count, batch_size):
            index = (
                slice(offset, min(offset + batch_size, count)) if order is None
                else order[offset : offset + batch_size]
            )
            batch = _index_batch(resident_train, index)
            utility = None if utility_cache is None else utility_cache[index.cpu() if torch.is_tensor(index) else index]
            batches.append((batch, batch.targets, utility))
        return batches
    # torch.compile shares parameters with the original module; state is saved
    # from / restored into `model` so checkpoint keys stay unprefixed.
    compute_model = model
    if compile_model and device.type == "cuda":
        compute_model = torch.compile(model)
    history = []
    best_state = None
    epoch_iterator = tqdm(range(epochs), desc="training", unit="epoch", disable=not progress)
    for epoch in epoch_iterator:
        router_only = cudtr_teacher_enabled and epoch < cudtr_router_only_epochs
        if cudtr_enabled:
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(not router_only)
            for parameter in model.head.parameters():
                parameter.requires_grad_(not router_only)
        epoch_batches = _train_batches()
        batch_iterator = iter(epoch_batches)
        if progress:
            batch_iterator = tqdm(
                batch_iterator,
                total=len(epoch_batches),
                desc="train",
                unit="batch",
                leave=False,
            )
        train_losses = [train_step(
            compute_model, optimizer, batch, targets, device=device, scaler=scaler, amp=use_amp,
            utility_targets=utility, utility_teacher_model=None if utility_cache is not None else utility_teacher_model,
        )
                        for batch, targets, utility in batch_iterator]
        validation = evaluate(
            compute_model,
            ((batch.to(device), targets.to(device)) for batch, targets in validation_batches),
            split="validation",
        )
        record = {"epoch": epoch + 1, "metric": validation["loss"], "split": "validation", "train_loss": sum(train_losses) / max(1, len(train_losses))}
        if task == "state" and config.head_mode == "quantile":
            from .metrics import state_quantile_metrics

            validation_quantiles = []
            validation_targets = []
            compute_model.eval()
            with torch.no_grad():
                for validation_batch, validation_target in validation_batches:
                    prediction = compute_model.predict(validation_batch.to(device))
                    if prediction.quantiles is None:
                        raise ValueError("quantile validation requires quantile predictions")
                    quantiles = prediction.quantiles
                    targets = validation_target.values.to(device)
                    if target_stats is not None:
                        mean = target_stats["mean"].to(device)
                        std = target_stats["std"].to(device)
                        quantiles = quantiles * std.view(1, 1, -1, 1) + mean.view(1, 1, -1, 1)
                        targets = targets * std + mean
                    validation_quantiles.append(quantiles.detach().cpu())
                    validation_targets.append(targets.detach().cpu())
            sensor_names = bundle.metadata.get("features", list(range(input_dim)))
            record["validation_quantile_metrics"] = state_quantile_metrics(
                torch.cat(validation_quantiles), torch.cat(validation_targets), sensor_names,
            )
        history.append(record)
        if lr_scheduler is not None:
            if scheduler == "plateau":
                lr_scheduler.step(record["metric"])
            else:
                lr_scheduler.step()
        if progress:
            epoch_iterator.set_postfix(train_loss=record["train_loss"], val_loss=record["metric"])
        if best_state is None or record["metric"] < min(item["metric"] for item in history[:-1]):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_index = min(range(len(history)), key=lambda index: history[index]["metric"])
        if patience is not None and len(history) - 1 - best_index >= patience:
            break
    if not history:
        raise ValueError("epochs must be positive")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = perf_counter() - train_started
    best = min(history, key=lambda item: item["metric"])
    model.load_state_dict(best_state)
    if run_id is None:
        run_id = "{}_{}_seed{}_{}".format(
            bundle.dataset, task, seed, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    writer = ArtifactWriter(Path(result_root), run_id)
    checkpoint_config = getattr(getattr(model, "config", None), "to_dict", lambda: config.to_dict())()
    if target_stats is not None:
        checkpoint_config = dict(checkpoint_config)
        checkpoint_config["target_normalization"] = {
            "identity": target_stats["identity"],
            "source_split": target_stats["source_split"],
            "mean": target_stats["mean"].tolist(),
            "std": target_stats["std"].tolist(),
        }
    writer.write_checkpoint({"model": model.state_dict(), "config": checkpoint_config, "best": best})
    manifest = writer.write_manifest(
        dataset=bundle.dataset,
        task=task,
        seed=seed,
        split_identity=bundle.split_identity,
        target_schema=bundle.target_schema,
        encoder_version=resolved_model_version,
        best_metric=best["metric"],
        checkpoint="checkpoint.pt",
        dataset_schema=bundle.metadata.get("features", bundle.metadata.get("schema", "")),
        target_convention=bundle.metadata.get(
            "target_convention",
            "raw_test_rul_training_cap_{}".format(bundle.metadata["training_cap"])
            if task == "rul" and "training_cap" in bundle.metadata
            else bundle.target_schema,
        ),
        normalizer_identity=bundle.metadata.get("normalizer_identity", "unbound"),
        evaluator_version="unified_evaluator_v4",
        model_version=resolved_model_version,
        model_name=resolved_model_name,
        input_padding=bundle.metadata.get("history_padding", "none"),
        optimizer_learning_rate=learning_rate,
        cudtr_enabled=config.cudtr_enabled,
        cudtr_teacher_enabled=bool(cudtr_teacher_enabled),
        cudtr_teacher_type=(
            "fusion_route_level_target_source"
            if cudtr_teacher_enabled and task == "state" and bundle.dataset == "metropt3"
            else "task_level_source" if cudtr_teacher_enabled else None
        ),
        cudtr_teacher_checkpoint=(
            Path(cudtr_teacher_checkpoint).name if cudtr_teacher_checkpoint is not None else None
        ),
        cudtr_router_only_epochs=cudtr_router_only_epochs if cudtr_enabled else None,
        cudtr_top_k=config.cudtr_top_k if cudtr_enabled else None,
        cudtr_query_dim=config.cudtr_query_dim if cudtr_enabled else None,
        cudtr_temperature=config.cudtr_temperature if cudtr_enabled else None,
        cudtr_fallback_init=config.cudtr_fallback_init if cudtr_enabled else None,
        cudtr_loss_weights={
            "distill": config.cudtr_lambda_distill,
            "rank": config.cudtr_lambda_rank,
            "sparse": config.cudtr_lambda_sparse,
            "fallback": config.cudtr_lambda_fallback,
        } if cudtr_enabled else None,
        cudtr_rank_margin=config.cudtr_rank_margin if cudtr_enabled else None,
        cudtr_utility_clip=config.cudtr_utility_clip if cudtr_enabled else None,
        cudtr_fallback_drop_rate=config.cudtr_fallback_drop_rate if cudtr_enabled else None,
        cudtr_min_observation_rate=config.cudtr_min_observation_rate if cudtr_enabled else None,
        cudtr_utility_threshold=config.cudtr_utility_threshold if cudtr_enabled else None,
        rul_over_penalty=rul_over_penalty if task == "rul" else None,
        rul_head_input=rul_head_input if task == "rul" else None,
        sensor_pooling=sensor_pooling,
        cross_channel_mode=cross_channel_mode,
            state_decoder=state_decoder,
            tep_class_weights=tep_class_weights if task == "fault_detection_diagnosis" else None,
        dataset_metadata=dict(bundle.metadata),
        epochs=epochs,
        batch_size=batch_size,
        lr_scheduler=scheduler,
        early_stopping_patience=patience,
        device=str(device),
        amp=bool(use_amp),
        head_mode=config.head_mode,
    )
    manifest["validation_selection_fraction"] = validation_selection_fraction
    manifest["validation_selection_examples"] = len(validation_selection_samples)
    manifest["validation_calibration_examples"] = len(validation_calibration_samples)
    validation_provenance = _validation_window_provenance(
        validation_selection_samples, validation_calibration_samples,
    )
    manifest["validation_calibration_source"] = (
        "chronological_tail_disjoint_from_selection"
        if validation_provenance["window_overlap_checked"]
        else "validation_split_without_window_metadata"
    )
    manifest.update(validation_provenance)
    if target_stats is not None:
        manifest["target_normalizer_identity"] = target_stats["identity"]
        manifest["target_normalizer_source_split"] = target_stats["source_split"]
        manifest["target_normalizer_stats_sha256"] = target_stats["stats_sha256"]
        writer.write_json("manifest.json", manifest)
    metrics = {"validation": validation, "history": history, "test_evaluation_count": 0}
    test_metrics = None
    inference_seconds = None
    test_examples = 0
    if evaluate_test:
        tep_threshold = 0.5
        tep_threshold_score = None
        if task == "fault_detection_diagnosis":
            from .metrics import select_tep_threshold
            validation_probabilities, validation_labels = [], []
            model.eval()
            for validation_batch, validation_target in validation_batches:
                with torch.no_grad():
                    validation_prediction = model.predict(validation_batch.to(device))
                if validation_prediction.detection_probability is not None:
                    validation_probabilities.append(validation_prediction.detection_probability.detach().cpu())
                validation_labels.append((validation_target.class_label > 0).long().detach().cpu())
            if validation_probabilities:
                tep_threshold, tep_threshold_score = select_tep_threshold(
                    torch.cat(validation_probabilities), torch.cat(validation_labels), objective="f1",
                )
        validation_quantiles_eval = None
        validation_targets_eval = None
        if task == "state" and config.head_mode == "quantile":
            validation_quantiles_eval = []
            validation_targets_eval = []
            model.eval()
            for validation_batch, validation_target in validation_calibration_batches:
                with torch.no_grad():
                    validation_prediction = model.predict(validation_batch.to(device))
                validation_quantile = validation_prediction.quantiles
                validation_target_values = validation_target.values.to(device)
                if target_stats is not None:
                    mean = target_stats["mean"].to(device)
                    std = target_stats["std"].to(device)
                    validation_quantile = validation_quantile * std.view(1, 1, -1, 1) + mean.view(1, 1, -1, 1)
                    validation_target_values = validation_target_values * std + mean
                validation_quantiles_eval.append(validation_quantile.detach().cpu())
                validation_targets_eval.append(validation_target_values.detach().cpu())
        test_samples = bundle.samples.get("test", [])
        test_batches = list(_loader(test_samples, batch_size))
        if not test_batches:
            raise ValueError("test evaluation requested but test split is empty")
        predictions, targets_all = [], []
        quantiles_all, detection_probabilities, diagnosis_probabilities = [], [], []
        cudtr_outputs = {}
        weighted_test_loss = 0.0
        test_loss_examples = 0
        class_labels, class_predictions, class_probabilities, class_run_ids = [], [], [], []
        event_probabilities, event_labels, event_run_ids, event_positions = [], [], [], []
        engine_ids = []
        model.eval()
        elapsed = 0.0
        test_sample_batches = [test_samples[offset:offset + batch_size]
                               for offset in range(0, len(test_samples), batch_size)]
        for (batch, targets), sample_batch in zip(test_batches, test_sample_batches):
            batch = batch.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = perf_counter()
            with torch.no_grad():
                prediction = model.predict(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += perf_counter() - started
            if prediction.quantiles is not None:
                quantiles = prediction.quantiles
                if target_stats is not None:
                    quantiles = quantiles * target_stats["std"].to(device).view(1, 1, -1, 1)
                    quantiles = quantiles + target_stats["mean"].to(device).view(1, 1, -1, 1)
                quantiles_all.append(quantiles.detach().cpu())
            if prediction.detection_probability is not None:
                detection_probabilities.append(prediction.detection_probability.detach().cpu())
            if prediction.diagnosis_probability is not None:
                diagnosis_probabilities.append(prediction.diagnosis_probability.detach().cpu())
            route = prediction.metadata.get("cudtr", {})
            if route.get("enabled"):
                for key in ("route_weights", "source_available", "fallback_gate", "utility_prediction", "route_entropy", "route_disagreement"):
                    value = route.get(key)
                    if torch.is_tensor(value):
                        cudtr_outputs.setdefault(key, []).append(value)
            with torch.no_grad():
                loss = model.loss(prediction, targets.to(device))
                if isinstance(loss, dict):
                    loss = loss["total"]
                test_batch_size = int(batch.X_obs.shape[0])
                weighted_test_loss += float(loss.detach().cpu()) * test_batch_size
                test_loss_examples += test_batch_size
            if prediction.point is not None and targets.values is not None:
                point = prediction.point
                target_values = targets.values.to(device)
                if target_stats is not None:
                    mean = target_stats["mean"].to(device)
                    std = target_stats["std"].to(device)
                    point = point * std + mean
                    target_values = target_values * std + mean
                predictions.append(point.detach().cpu())
                targets_all.append(target_values.detach().cpu())
                if task == "rul":
                    engine_ids.extend(str(sample.metadata["engine_id"]) for sample in sample_batch)
            if task == "fault_detection_diagnosis":
                if prediction.diagnosis_probability is None or targets.class_label is None:
                    raise ValueError("TEP test evaluation requires diagnosis probabilities and class labels")
                probabilities = prediction.diagnosis_probability.detach().cpu()
                labels = targets.class_label.detach().cpu().long().reshape(-1)
                class_labels.append(labels)
                class_predictions.append(probabilities.argmax(dim=-1))
                class_probabilities.append(probabilities)
                class_run_ids.extend(str((sample.metadata or {}).get("unit_id", "unknown")) for sample in sample_batch)
                if prediction.detection_probability is not None:
                    event_probabilities.append(prediction.detection_probability.detach().cpu())
                event_labels.append(labels)
                event_run_ids.extend(str((sample.metadata or {}).get("unit_id", "unknown")) for sample in sample_batch)
                event_positions.append(torch.tensor([
                    int((sample.metadata or {}).get("window_end", index))
                    for index, sample in enumerate(sample_batch)
                ]))
            test_examples += int(batch.X_obs.shape[0])
        inference_seconds = elapsed / len(test_batches)
        test_metrics = {"split": "test", "batches": len(test_batches), "examples": test_examples}
        prediction_artifacts = {
            "head_mode": config.head_mode,
            "quantiles": torch.cat(quantiles_all) if quantiles_all else None,
            "detection_probability": torch.cat(detection_probabilities) if detection_probabilities else None,
            "diagnosis_probability": torch.cat(diagnosis_probabilities) if diagnosis_probabilities else None,
            "cudtr": {
                key: torch.cat(values, dim=0) for key, values in cudtr_outputs.items()
            } if cudtr_outputs else {"enabled": False},
        }
        if cudtr_outputs:
            prediction_artifacts["cudtr_diagnostics"] = summarize_cudtr_diagnostics(
                prediction_artifacts["cudtr"]["route_weights"],
                prediction_artifacts["cudtr"]["source_available"],
                prediction_artifacts["cudtr"]["fallback_gate"],
                prediction_artifacts["cudtr"]["route_disagreement"],
            )
            test_metrics["cudtr_diagnostics"] = prediction_artifacts["cudtr_diagnostics"]
        if task == "state":
            from .metrics import calibrate_state_quantiles, calibrate_state_quantiles_cqr, state_metrics, state_quantile_diagnostics, state_quantile_metrics

            sensor_names = bundle.metadata.get("features", list(range(input_dim)))
            test_metrics["point"] = state_metrics(
                torch.cat(predictions), torch.cat(targets_all), sensor_names,
            )
            raw_test_quantiles = torch.cat(quantiles_all) if quantiles_all else None
            if raw_test_quantiles is not None and validation_quantiles_eval:
                validation_quantiles = torch.cat(validation_quantiles_eval)
                validation_targets = torch.cat(validation_targets_eval)
                calibration_fn = calibrate_state_quantiles_cqr if quantile_calibration == "cqr" else calibrate_state_quantiles
                calibrated_quantiles, calibration_radius = calibration_fn(
                    validation_quantiles, validation_targets, raw_test_quantiles, coverage=0.8,
                )
                prediction_artifacts["raw_quantiles"] = raw_test_quantiles
                prediction_artifacts["quantiles"] = calibrated_quantiles
                test_metrics["quantile_raw"] = state_quantile_metrics(
                    raw_test_quantiles, torch.cat(targets_all), sensor_names,
                )
                test_metrics["quantile_calibrated"] = state_quantile_metrics(
                    calibrated_quantiles, torch.cat(targets_all), sensor_names,
                )
                test_metrics["quantile_diagnostics"] = {
                    "raw": state_quantile_diagnostics(raw_test_quantiles, torch.cat(targets_all), sensor_names),
                    "calibrated": state_quantile_diagnostics(calibrated_quantiles, torch.cat(targets_all), sensor_names),
                }
                test_metrics["quantile_calibration"] = {
                    "source_split": "validation",
                    "method": quantile_calibration,
                    "target_coverage": 0.8,
                    "radius": calibration_radius.tolist(),
                }
            else:
                test_metrics.update(state_metrics(torch.cat(predictions), torch.cat(targets_all), sensor_names))
        elif task == "rul":
            from .metrics import rul_metrics

            test_metrics.update(rul_metrics(torch.cat(predictions), torch.cat(targets_all), engine_ids))
            writer.write_test_predictions({
                "run_id": run_id, "dataset": bundle.dataset, "task": task,
                "split_identity": bundle.split_identity, "target_schema": bundle.target_schema,
                "units": "cycles", "engine_ids": engine_ids,
                "prediction": torch.cat(predictions).reshape(-1), "target": torch.cat(targets_all).reshape(-1),
                **prediction_artifacts,
            })
        elif task == "fault_detection_diagnosis":
            from .metrics import classification_metrics

            labels = torch.cat(class_labels)
            predicted = torch.cat(class_predictions)
            probabilities = torch.cat(class_probabilities)
            num_classes = int(probabilities.shape[-1])
            test_metrics.update(classification_metrics(labels, predicted, num_classes, class_run_ids))
            test_metrics["evaluation_task"] = "fault_classification"
            if event_probabilities:
                from ..models.unified.heads.tep import evaluate_tep_events
                run_id_to_int = {run_id: index for index, run_id in enumerate(sorted(set(event_run_ids)))}
                event_metrics = evaluate_tep_events(
                    torch.cat(event_probabilities), torch.cat(event_labels),
                    torch.tensor([run_id_to_int[item] for item in event_run_ids]), torch.cat(event_positions),
                    threshold=tep_threshold,
                )
                test_metrics["event_metrics"] = event_metrics
            test_metrics["threshold_selection"] = {
                "source_split": "validation", "objective": "f1",
                "threshold": tep_threshold, "validation_score": tep_threshold_score,
            }
            test_metrics["class_mapping"] = {str(index): index for index in range(num_classes)}
            test_metrics["window_count"] = int(labels.numel())
            writer.write_test_predictions({
                "run_id": run_id, "dataset": bundle.dataset, "task": "fault_classification",
                "split_identity": bundle.split_identity, "target_schema": bundle.target_schema,
                "class_mapping": {str(index): index for index in range(num_classes)},
                "run_ids": class_run_ids, "labels": labels, "predictions": predicted,
                "probabilities": probabilities,
                **prediction_artifacts,
            })
        else:
            test_metrics["loss"] = weighted_test_loss / max(1, test_loss_examples)
        if task == "state":
            writer.write_test_predictions({
                "run_id": run_id, "dataset": bundle.dataset, "task": task,
                "split_identity": bundle.split_identity, "target_schema": bundle.target_schema,
                "units": bundle.metadata.get("target_units", "raw_sensor_units"),
                "prediction": torch.cat(predictions), "target": torch.cat(targets_all),
                **prediction_artifacts,
            })
        metrics["test_evaluation_count"] = 1
        writer.write_json("test_metrics.json", {
            **test_metrics,
            "run_id": run_id,
            "seed": seed,
            "checkpoint": "checkpoint.pt",
            "dataset": bundle.dataset,
            "task": task,
            "split_identity": bundle.split_identity,
            "dataset_schema": bundle.metadata.get("features", bundle.metadata.get("schema", "")),
            "normalizer_identity": bundle.metadata.get("normalizer_identity", "unbound"),
            "target_normalizer_identity": None if target_stats is None else target_stats["identity"],
            "target_normalizer_source_split": None if target_stats is None else target_stats["source_split"],
            "target_schema": bundle.target_schema,
            "evaluator_version": "unified_evaluator_v4",
        })
    if device.type == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    efficiency = {
        "run_id": run_id,
        "seed": seed,
        "training_seconds": training_seconds,
        "training_examples": len(bundle.samples["train"]) * len(history),
        "training_examples_per_second": len(bundle.samples["train"]) * len(history) / max(training_seconds, 1e-12),
        "test_batch_inference_seconds": inference_seconds,
        "test_examples": test_examples,
        "parameter_count": parameter_count,
        "peak_cuda_memory_bytes": peak_memory_bytes,
        "device": str(device),
    }
    writer.write_json("efficiency.json", efficiency)
    writer.write_metrics(metrics)
    # Keep validation artifacts on CPU so large FD004 runs do not accumulate
    # every batch output on the GPU after training and test evaluation.
    predictions = []
    with torch.no_grad():
        for batch, _ in validation_batches:
            prediction = model.predict(batch.to(device))
            if target_stats is not None:
                mean = target_stats["mean"].to(device)
                std = target_stats["std"].to(device)
                if prediction.point is not None:
                    prediction.point = prediction.point * std + mean
                if prediction.quantiles is not None:
                    quantile_std = std.view(1, 1, -1, 1)
                    quantile_mean = mean.view(1, 1, -1, 1)
                    prediction.quantiles = prediction.quantiles * quantile_std + quantile_mean
            for field in ("point", "quantiles", "detection_probability", "diagnosis_probability"):
                value = getattr(prediction, field)
                if value is not None:
                    setattr(prediction, field, value.detach().cpu())
            predictions.append(prediction)
    writer.write_predictions(predictions)
    return {"manifest": manifest, "validation": validation, "history": history,
            "test_metrics": test_metrics, "efficiency": efficiency}
