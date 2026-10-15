"""Exact, symlink-aware paths for LocalGaze capture sidecars."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
import stat
from typing import Any


INTERNAL_LAYOUT = "internal-v1"
LEGACY_LAYOUT = "legacy-root"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def normalize_sidecar_layout(value: Any) -> str:
    """Missing layout markers mean the original root-level layout."""
    if value is None:
        return LEGACY_LAYOUT
    if value not in {INTERNAL_LAYOUT, LEGACY_LAYOUT}:
        raise ValueError("unsupported capture sidecar layout")
    return value


def sidecar_relative_path(name: str, layout: str) -> str:
    """Return the single owned output-root-relative location for a sidecar."""
    layout = normalize_sidecar_layout(layout)
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        raise ValueError("sidecar name must be a direct-child filename")
    return f".internal/{name}" if layout == INTERNAL_LAYOUT else name


def _has_reparse_attribute(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _reject_dotdot(value: str | os.PathLike[str]) -> str:
    text = os.path.expanduser(os.fspath(value))
    if any(part == ".." for part in text.replace("\\", "/").split("/")):
        raise ValueError("capture paths must not contain parent traversal")
    return text


def assert_no_reparse_components(value: str | os.PathLike[str]) -> Path:
    """Reject symlinks and Windows reparse points in every existing component.

    This uses lstat metadata available on Python 3.10; Path.is_junction was
    added later and is deliberately not required by the packaged runtime.
    """
    text = _reject_dotdot(value)
    absolute = Path(os.path.abspath(text))
    parts = absolute.parts
    if not parts:
        raise ValueError("capture path is empty")
    cursor = Path(parts[0])
    for part in parts[1:]:
        cursor = cursor / part
        if _has_reparse_attribute(cursor):
            raise ValueError(f"capture path contains a symbolic link or junction: {cursor}")
        try:
            cursor.lstat()
        except FileNotFoundError:
            # No later component can exist beneath a missing parent.
            break
    return absolute


def _root_path(root: str | os.PathLike[str]) -> Path:
    raw = assert_no_reparse_components(Path(root).expanduser())
    resolved = raw.resolve(strict=False)
    if raw.exists() and (not raw.is_dir() or resolved != raw.resolve(strict=True)):
        raise ValueError("capture output root must be a regular directory")
    return resolved


def output_child_path(
    root: str | os.PathLike[str],
    relative: str | os.PathLike[str],
) -> Path:
    """Resolve a safe output-root-relative file or directory reference."""
    resolved_root = _root_path(root)
    text = os.fspath(relative)
    if not isinstance(text, str) or not text or "\\" in text:
        raise ValueError("capture output reference must be a safe relative path")
    win = PureWindowsPath(text)
    parts = text.split("/")
    if (
        win.is_absolute()
        or win.drive
        or Path(text).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("capture output reference escapes its output root")
    candidate = resolved_root.joinpath(*parts)
    assert_no_reparse_components(candidate)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("capture output reference escapes its output root") from exc
    return resolved


def internal_directory(root: str | os.PathLike[str], *, create: bool = False) -> Path:
    """Return output_root/.internal, rejecting links and non-directory collisions."""
    resolved_root = _root_path(root)
    if create:
        resolved_root.mkdir(parents=True, exist_ok=True)
    directory = resolved_root / ".internal"
    if _has_reparse_attribute(directory):
        raise ValueError("capture .internal directory must not be a symbolic link")
    if directory.exists() and not directory.is_dir():
        raise ValueError("capture .internal path must be a directory")
    if create:
        directory.mkdir(exist_ok=True)
    if directory.exists() and directory.resolve(strict=True) != directory:
        raise ValueError("capture .internal directory resolves outside output_root")
    return directory


def sidecar_path(
    root: str | os.PathLike[str],
    name: str,
    *,
    layout: str = INTERNAL_LAYOUT,
    create_parent: bool = False,
) -> Path:
    """Return an exact sidecar path without accepting caller-supplied traversal."""
    layout = normalize_sidecar_layout(layout)
    relative = sidecar_relative_path(name, layout)
    resolved_root = _root_path(root)
    if layout == INTERNAL_LAYOUT:
        parent = internal_directory(resolved_root, create=create_parent)
        path = parent / name
    else:
        path = resolved_root / name
    if path.is_symlink():
        raise ValueError("capture sidecar must not be a symbolic link")
    assert_no_reparse_components(path)
    if path.resolve(strict=False) != path:
        raise ValueError("capture sidecar resolves outside its owned path")
    if path.relative_to(resolved_root).as_posix() != relative:
        raise ValueError("capture sidecar path does not match its owned layout")
    return path


def owned_sidecar_path(
    root: str | os.PathLike[str],
    value: str | os.PathLike[str],
    *,
    name: str,
    layout: str,
) -> Path:
    """Validate a relative or absolute sidecar reference against its exact layout."""
    resolved_root = _root_path(root)
    expected_relative = sidecar_relative_path(name, layout)
    expected = sidecar_path(resolved_root, name, layout=layout)
    supplied_text = os.fspath(value)
    if not isinstance(supplied_text, str) or not supplied_text:
        raise ValueError("capture sidecar reference is empty")
    supplied = Path(supplied_text).expanduser()
    windows = PureWindowsPath(supplied_text)
    absolute = supplied.is_absolute() or windows.is_absolute() or bool(windows.drive)
    if absolute:
        if not supplied.is_absolute():
            raise ValueError("capture sidecar absolute reference does not match its owned path")
        components = supplied_text.replace("\\", "/").split("/")
        if any(part in {".", ".."} for part in components):
            raise ValueError("capture sidecar absolute reference contains a traversal alias")
        supplied_absolute = assert_no_reparse_components(supplied)
        if supplied_absolute != expected:
            raise ValueError("capture sidecar absolute reference does not match its owned path")
    elif supplied_text.replace("\\", "/") != expected_relative:
        raise ValueError("capture sidecar reference does not match its owned path")
    # sidecar_path already checks .internal and target links; verify each path
    # component again after resolving a potentially absolute caller reference.
    if expected_relative.startswith(".internal/") and internal_directory(resolved_root).is_symlink():
        raise ValueError("capture .internal directory must not be a symbolic link")
    if _has_reparse_attribute(expected) or expected.resolve(strict=False) != expected:
        raise ValueError("capture sidecar path is unsafe")
    return expected


def existing_sidecar_path(
    root: str | os.PathLike[str],
    name: str,
    *,
    layout: str = INTERNAL_LAYOUT,
) -> Path:
    """Return only the layout-owned path; never select a layout by existence."""
    layout = normalize_sidecar_layout(layout)
    expected = sidecar_path(root, name, layout=layout)
    alternate = sidecar_path(
        root,
        name,
        layout=LEGACY_LAYOUT if layout == INTERNAL_LAYOUT else INTERNAL_LAYOUT,
    )
    if (expected.exists() or expected.is_symlink()) and (alternate.exists() or alternate.is_symlink()):
        raise FileExistsError(f"conflicting root and .internal sidecars exist: {name}")
    return expected


__all__ = [
    "INTERNAL_LAYOUT",
    "LEGACY_LAYOUT",
    "assert_no_reparse_components",
    "existing_sidecar_path",
    "internal_directory",
    "normalize_sidecar_layout",
    "output_child_path",
    "owned_sidecar_path",
    "sidecar_path",
    "sidecar_relative_path",
]
