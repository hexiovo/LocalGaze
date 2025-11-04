from importlib import import_module
from pathlib import Path
from typing import Dict, Type

from .base import BaseModel

__all__ = ["BaseModel", "create_model", "AVAILABLE_MODELS"]

AVAILABLE_MODELS: Dict[str, Type[BaseModel]] = {}


def register_model(name: str, cls: Type[BaseModel]) -> None:
    if name in AVAILABLE_MODELS:
        raise ValueError(f"Model name '{name}' already registered")
    AVAILABLE_MODELS[name] = cls


def _auto_discover() -> None:
    pkg_dir = Path(__file__).resolve().parent
    for f in pkg_dir.iterdir():
        if f.name in {"__init__.py", "base.py"} or f.suffix != ".py":
            continue
        mod_name = f"{__name__}.{f.stem}"
        import_module(mod_name)


def create_model(name: str, **kwargs) -> BaseModel:
    ALL_Module = list_available_models()
    if not AVAILABLE_MODELS:
        _auto_discover()
    try:
        cls = AVAILABLE_MODELS[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(AVAILABLE_MODELS)}"
        ) from e
    return cls(**kwargs)

def list_available_models():
    """
    列出 LocalGaze 所有注册的可用模型名称
    """
    # 如果 AVAILABLE_MODELS 为空，先自动发现并注册
    if not AVAILABLE_MODELS:
        _auto_discover()
    # 返回排序后的模型名列表
    return sorted(AVAILABLE_MODELS.keys())