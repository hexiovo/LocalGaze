"""File readers and writers, imported on demand to keep data imports light."""

from __future__ import annotations

from importlib import import_module
import os
import sys

package_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(package_root, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_EXPORT_MODULES = {
    "extract_ROI": ".ROIextract",
    "ROIchoose": ".ROIextract",
    "EyeDatachoose": ".EyeDataExtraction",
    "EyeData_Validation": ".EyeDataExtraction",
    "check_columns": ".xlsx_validation",
    "validate_xlsx": ".xlsx_validation",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
