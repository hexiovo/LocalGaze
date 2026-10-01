"""Shared names for new capture requests and output namespaces only.

Existing source IDs are provenance and must not be rewritten with this helper.
"""
from pathlib import Path


def canonical_session_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("session_id must be a string")
    if any(ord(char) < 32 for char in value):
        raise ValueError("session_id contains a control character")
    name = value.strip()
    if not name or name in {".", ".."} or name.endswith("."):
        raise ValueError("session_id must be a nonempty base name without a trailing dot")
    if any(char in name for char in '<>:"/\\|?*') or any(ord(char) < 32 for char in name):
        raise ValueError("session_id contains a Windows-invalid filename character")
    if Path(name).name != name or name.lower().endswith((".txt", ".xlsx", ".manifest.json")):
        raise ValueError("session_id must be a base name without a data file extension")
    if name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("session_id is a reserved Windows device name")
    return name


__all__ = ["canonical_session_id"]
