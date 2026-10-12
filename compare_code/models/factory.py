from .base import BaselineConfig
from .gru import GRU
from .itransformer import ITransformer
from .patchtst import PatchTST
from .tcn import TCN
from .timesnet import TimesNet


MODEL_NAMES = ("tcn", "gru", "patchtst", "timesnet", "itransformer")
_MODEL_TYPES = {
    "tcn": TCN,
    "gru": GRU,
    "patchtst": PatchTST,
    "timesnet": TimesNet,
    "itransformer": ITransformer,
}


def create_baseline(model_name, **config_values):
    if model_name not in MODEL_NAMES:
        raise ValueError("unknown or not-yet-implemented comparison model: {}".format(model_name))
    config = BaselineConfig(model_name=model_name, **config_values)
    return _MODEL_TYPES[model_name](config)
