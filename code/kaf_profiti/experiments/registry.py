from kaf_profiti.models.unified import ModelConfig, UnifiedModel


_SPECS = {
    "unified_encoder_task": {
        "display_name": "Unified Temporal Encoder + Task Head",
        "status": "enabled",
        "parameter_scope": "dataset_independent",
    },
    "kst_probflow": {"display_name": "KST ProbFlow", "status": "not_implemented"},
}


def get_model_spec(model_id):
    if model_id not in _SPECS:
        raise KeyError("unknown model: {}".format(model_id))
    return dict(_SPECS[model_id])


def create_model(model_id, **kwargs):
    if model_id != "unified_encoder_task":
        raise ValueError("legacy model construction is outside the unified package")
    config = ModelConfig(**kwargs)
    return UnifiedModel(config)
