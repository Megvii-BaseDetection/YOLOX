import importlib
from typing import Optional

from yolox.config import YoloxConfig


def resolve_config(config_str: str) -> YoloxConfig:
    config = YoloxConfig.get_named_config(config_str)
    if config is not None:
        return config

    config_class: Optional[type[YoloxConfig]] = None
    classpath = config_str.split(":")
    if len(classpath) == 2:
        try:
            module = importlib.import_module(classpath[0])
            config_class = getattr(module, classpath[1], None)
        except ImportError:
            pass
    if config_class is None:
        raise ValueError(f"Unknown config class: {config_str}")
    if not issubclass(config_class, YoloxConfig):
        raise ValueError(f"Invalid config class (does not extend `YoloxConfig`): {config_str}")

    try:
        return config_class()
    except Exception as e:
        raise ValueError(f"Error loading model config: {config_str}") from e


def parse_model_config_opts(kv_opts: list[str]) -> dict[str, str]:
    """
    Parse key-value options from a list of strings.
    """
    kv_dict = {}
    for kv in kv_opts:
        if "=" not in kv:
            raise ValueError(f"Invalid model configuration option (must be of the form OPT=VALUE): {kv}")
        key, value = kv.split("=", 1)
        kv_dict[key] = value
    return kv_dict
