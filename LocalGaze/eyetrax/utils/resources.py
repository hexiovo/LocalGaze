"""Windows resource-path compatibility for MediaPipe 0.10.x solutions."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

_RESOURCE_CACHE = None


def prepare_face_mesh_resources() -> None:
    """Keep native graph/model paths ASCII when the app is in a Unicode folder.

    MediaPipe 0.10.14's SolutionBase computes its resource root from __file__
    and passes it to native narrow-path APIs. Redirect only that resource
    locator to an owned temporary copy. Model bytes and graph options are
    unchanged; no user models, images or gaze data are copied.
    """
    global _RESOURCE_CACHE
    if os.name != "nt" or _RESOURCE_CACHE is not None:
        return
    from mediapipe.python import solution_base
    source_root = Path(solution_base.__file__).resolve().parents[2]
    if str(source_root).isascii():
        return
    modules = source_root / "mediapipe" / "modules"
    if not modules.is_dir():
        raise FileNotFoundError(f"MediaPipe model resources are missing: {modules}")

    candidates = [os.environ.get("LOCALGAZE_RESOURCE_CACHE"), tempfile.gettempdir(),
                  str(Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "Temp")]
    errors = []
    cache = None
    for candidate in candidates:
        if not candidate:
            continue
        directory = Path(candidate).expanduser().resolve()
        if not str(directory).isascii():
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
            cache = tempfile.TemporaryDirectory(prefix="LocalGaze-mp-", dir=directory,
                                                ignore_cleanup_errors=True)
            break
        except OSError as exc:
            errors.append(str(exc))
    if cache is None:
        raise RuntimeError("MediaPipe needs a writable ASCII resource directory. Set "
                           "LOCALGAZE_RESOURCE_CACHE to one, or extract the app into an ASCII path. "
                           + "; ".join(errors))
    try:
        destination = Path(cache.name) / "mediapipe" / "modules"
        shutil.copytree(modules, destination,
                        ignore=shutil.ignore_patterns("*.py", "*.pyc", "__pycache__"))
        # This adapter targets the resource-root calculation in the pinned
        # SolutionBase, without changing its class or native graph execution.
        solution_base.__file__ = str(Path(cache.name) / "mediapipe" / "python" / "solution_base.py")
        _RESOURCE_CACHE = cache  # Keep resources alive for all estimators in this process.
    except Exception:
        cache.cleanup()
        raise


def resource_cache_active() -> bool:
    return _RESOURCE_CACHE is not None
