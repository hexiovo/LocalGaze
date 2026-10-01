"""EyeEvaluate package with lazy subpackage imports.

Loading the pure workbook/ROI adapters does not eagerly import unrelated Tk
dialogs; those interfaces are loaded when their subpackage is requested.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
for _import_path in (_PROJECT_ROOT, _PACKAGE_ROOT):
    _path_text = str(_import_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

__all__ = ["Popup", "FileOperate", "Logger", "Evaluate"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals()) | set(__all__))
