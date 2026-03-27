"""
Model and config registries for Mechanical MNIST Cahn-Hilliard experiments.

To add a new model:
  1. Create models/my_model.py with a MyModelConfig(Config) dataclass and a
     model class that inherits MechMNISTModel.
  2. Add entries to MODEL_REGISTRY and CONFIG_REGISTRY below.
  3. Pass --model my_model when running train.py.
"""

import json

from models.base import MechMNISTModel
from models.unet import UNetConfig, UNetMultiRegression, MultiTaskLoss, count_parameters
from models.unet_small import UNetSmallConfig, MultiTaskUNet
from models.fno import FNOConfig, MultiTaskFNO
from models.swin import SwinConfig, MultiTaskSwin

MODEL_REGISTRY: dict[str, type[MechMNISTModel]] = {
    "unet": UNetMultiRegression,
    "unet_small": MultiTaskUNet,
    "fno": MultiTaskFNO,
    "swin": MultiTaskSwin,
}

CONFIG_REGISTRY: dict[str, type] = {
    "unet": UNetConfig,
    "unet_small": UNetSmallConfig,
    "fno": FNOConfig,
    "swin": SwinConfig,
}


def get_model(config) -> MechMNISTModel:
            f"Available: {sorted(MODEL_REGISTRY)}"
            return MODEL_REGISTRY[config.model_name].from_config(config)


def default_config(model_name: str = "unet"):
    """Return a default config instance for the given model."""
    if model_name not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {sorted(CONFIG_REGISTRY)}"
        )
    return CONFIG_REGISTRY[model_name]()


def load_config(path):
    """
    Load the correct Config subclass from a JSON file.

    Reads model_name from the file to determine which Config subclass to
    instantiate, so model-specific fields round-trip correctly.
    """
    with open(path) as f:
        data = json.load(f)
    model_name = data.get("model_name", "unet")
    klass = CONFIG_REGISTRY.get(model_name)
    if klass is None:
        from config import Config
        klass = Config
    return klass(**data)


__all__ = [
    "MechMNISTModel",
    "UNetConfig",
    "UNetMultiRegression",
    "UNetSmallConfig",
    "MultiTaskUNet",
    "FNOConfig",
    "MultiTaskFNO",
    "SwinConfig",
    "MultiTaskSwin",
    "MultiTaskLoss",
    "count_parameters",
    "MODEL_REGISTRY",
    "CONFIG_REGISTRY",
    "get_model",
    "default_config",
    "load_config",
]
