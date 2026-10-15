"""Crash-safe capture identity, checkpoints, event logs, and recovery parsing.

This module deliberately uses only the Python standard library.  Capture rows
remain in the existing UTF-8 TSV; the sidecars carry identity and durable
watermarks rather than duplicating the raw data.
"""
from __future__ import annotations

import csv
import datetime as _datetime
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from capture_paths import (
    INTERNAL_LAYOUT,
    LEGACY_LAYOUT,
    assert_no_reparse_components,
    internal_directory,
    normalize_sidecar_layout,
    owned_sidecar_path,
    output_child_path,
    sidecar_path,
    sidecar_relative_path,
)


CONTEXT_SCHEMA = "localgaze.capture-recovery-context/1"
CHECKPOINT_SCHEMA = "localgaze.capture-checkpoint/1"
EVENT_SCHEMA = "localgaze.capture-event/1"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_LEGACY_CAPTURE_COLUMN_NAMES = {
    "timestamp": "时间",
    "x": "x",
    "y": "y",
    "blink": "blink",
    "leftpupil": "左瞳孔直径",
    "rightpupil": "右瞳孔直径",
}
USER_MARKER_SCHEMA = "localgaze.capture-user-marker/1"
USER_MARKER_LOG_SCHEMA = "localgaze.capture-user-marker-log/1"
_USER_MARKER_KEYS = frozenset(
    [str(value) for value in range(1, 9)]
    + [f"KP_{value}" for value in range(1, 9)]
)
_USER_MARKER_FIELDS = frozenset({
    "schema", "marker_id", "session_id", "capture_id", "attempt_id",
    "marker_sequence", "literal_key", "label", "monotonic_ns", "elapsed_s",
    "timestamp_local", "source",
})


def marker_key_for_keysym(keysym: Any) -> str | None:
    """Return only unambiguous top-row or NumLock keypad 1–8 keysyms."""
    value = str(keysym) if keysym is not None else ""
    return value if value in _USER_MARKER_KEYS else None


class CapturePersistenceError(RuntimeError):
    """A recovery sidecar or raw capture file is unsafe or inconsistent."""


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def user_marker_stream_sha256(markers: Iterable[Mapping[str, Any]]) -> str:
    """Hash the canonical marker documents independently of other capture events."""
    digest = hashlib.sha256()
    for marker in markers:
        digest.update(_json_bytes(dict(marker)))
    return digest.hexdigest()


def _user_marker_watermark(
    markers: list[dict[str, Any]],
    event_log_path: str,
    event_log_payload: bytes = b"",
) -> dict[str, Any]:
    return {
        "schema": USER_MARKER_LOG_SCHEMA,
        "event": "user_marker",
        "path": event_log_path,
        "count": len(markers),
        "last_sequence": len(markers),
        "last_monotonic_ns": markers[-1]["monotonic_ns"] if markers else None,
        "markers_sha256": user_marker_stream_sha256(markers),
        "events_sha256": hashlib.sha256(event_log_payload).hexdigest(),
        "events_size_bytes": len(event_log_payload),
    }


def user_markers_from_events(
    events: Iterable[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate and extract the ordered, capture-owned marker event stream."""
    markers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_monotonic_ns = -1
    config = context.get("config") if isinstance(context.get("config"), Mapping) else {}
    start_value = config.get("capture_start_monotonic_ns")
    try:
        capture_start_ns = int(start_value) if start_value is not None else None
    except (TypeError, ValueError, OverflowError):
        capture_start_ns = None

    for event_index, event in enumerate(events, start=1):
        if not isinstance(event, Mapping) or event.get("event") != "user_marker":
            continue
        if (
            event.get("session_id") != context.get("session_id")
            or event.get("capture_id") != context.get("capture_id")
            or event.get("attempt_id") != context.get("attempt_id")
            or event.get("job_id") != context.get("job_id")
        ):
            raise CapturePersistenceError(f"user marker event {event_index} belongs to another capture identity")
        marker_value = event.get("marker")
        if not isinstance(marker_value, Mapping):
            raise CapturePersistenceError(f"user marker event {event_index} has no marker document")
        marker = dict(marker_value)
        if set(marker) != _USER_MARKER_FIELDS:
            raise CapturePersistenceError(f"user marker event {event_index} has an invalid marker schema")
        if marker.get("schema") != USER_MARKER_SCHEMA:
            raise CapturePersistenceError(f"user marker event {event_index} uses an unsupported marker schema")
        if (
            marker.get("session_id") != context.get("session_id")
            or marker.get("capture_id") != context.get("capture_id")
            or marker.get("attempt_id") != context.get("attempt_id")
        ):
            raise CapturePersistenceError(f"user marker event {event_index} has a mismatched marker identity")
        marker_id = marker.get("marker_id")
        if not isinstance(marker_id, str) or not re.fullmatch(r"[0-9a-f]{32}", marker_id):
            raise CapturePersistenceError(f"user marker event {event_index} has an invalid marker_id")
        if marker_id in seen_ids:
            raise CapturePersistenceError(f"user marker event {event_index} duplicates marker_id {marker_id}")
        seen_ids.add(marker_id)
        sequence = marker.get("marker_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != len(markers) + 1:
            raise CapturePersistenceError(f"user marker event {event_index} skips or repeats its marker sequence")
        literal_key = marker.get("literal_key")
        if literal_key not in _USER_MARKER_KEYS or marker.get("label") != literal_key:
            raise CapturePersistenceError(f"user marker event {event_index} has an invalid key or label")
        monotonic_value = marker.get("monotonic_ns")
        if not isinstance(monotonic_value, str) or not re.fullmatch(r"[0-9]+", monotonic_value):
            raise CapturePersistenceError(f"user marker event {event_index} has an invalid monotonic timestamp")
        monotonic_ns = int(monotonic_value)
        if monotonic_ns < previous_monotonic_ns:
            raise CapturePersistenceError(f"user marker event {event_index} moves backwards in monotonic time")
        elapsed = marker.get("elapsed_s")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise CapturePersistenceError(f"user marker event {event_index} has an invalid elapsed time")
        if capture_start_ns is not None:
            expected_elapsed = (monotonic_ns - capture_start_ns) / 1_000_000_000.0
            if expected_elapsed < 0 or not math.isclose(float(elapsed), expected_elapsed, rel_tol=0.0, abs_tol=1e-9):
                raise CapturePersistenceError(f"user marker event {event_index} has an elapsed time outside its capture clock")
        if not isinstance(marker.get("timestamp_local"), str) or not marker["timestamp_local"].strip():
            raise CapturePersistenceError(f"user marker event {event_index} has no readable local timestamp")
        if marker.get("source") != "tk_keypress":
            raise CapturePersistenceError(f"user marker event {event_index} has an unsupported source")
        markers.append(marker)
        previous_monotonic_ns = monotonic_ns
    return markers


def _validate_marker_watermark(
    watermark: Mapping[str, Any],
    markers: list[dict[str, Any]],
    *,
    label: str,
    expected_path: str,
) -> int:
    count = watermark.get("count") if label == "context" else watermark.get("user_marker_count")
    sequence = watermark.get("last_sequence") if label == "context" else watermark.get("user_marker_last_sequence")
    last_ns = watermark.get("last_monotonic_ns") if label == "context" else watermark.get("user_marker_last_monotonic_ns")
    digest = watermark.get("markers_sha256") if label == "context" else watermark.get("user_marker_stream_sha256")
    path = watermark.get("path") if label == "context" else watermark.get("user_marker_log_path")
    event_digest = watermark.get("events_sha256") if label == "context" else watermark.get("user_marker_events_sha256")
    event_size = watermark.get("events_size_bytes") if label == "context" else watermark.get("user_marker_events_size_bytes")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > len(markers):
        raise CapturePersistenceError(f"{label} user marker watermark exceeds the durable marker log")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != count:
        raise CapturePersistenceError(f"{label} user marker watermark sequence is inconsistent")
    prefix = markers[:count]
    expected_last = prefix[-1]["monotonic_ns"] if prefix else None
    if last_ns != expected_last:
        raise CapturePersistenceError(f"{label} user marker watermark timestamp is inconsistent")
    if not isinstance(digest, str) or digest != user_marker_stream_sha256(prefix):
        raise CapturePersistenceError(f"{label} user marker watermark hash does not match the event-log prefix")
    if not isinstance(event_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", event_digest):
        raise CapturePersistenceError(f"{label} user marker event-log prefix hash is invalid")
    if isinstance(event_size, bool) or not isinstance(event_size, int) or event_size < 0:
        raise CapturePersistenceError(f"{label} user marker event-log prefix size is invalid")
    if expected_path is not None and path != expected_path:
        raise CapturePersistenceError(f"{label} user marker log reference does not match the capture event log")
    return count


def _validate_marker_event_prefix(
    watermark: Mapping[str, Any],
    markers: list[dict[str, Any]],
    context: Mapping[str, Any],
    event_log_payload: bytes,
    *,
    label: str,
) -> None:
    event_size = watermark.get("events_size_bytes") if label == "context" else watermark.get("user_marker_events_size_bytes")
    event_digest = watermark.get("events_sha256") if label == "context" else watermark.get("user_marker_events_sha256")
    if event_size > len(event_log_payload):
        raise CapturePersistenceError(f"{label} user marker watermark is newer than the durable event log")
    prefix_payload = event_log_payload[:event_size]
    if hashlib.sha256(prefix_payload).hexdigest() != event_digest:
        raise CapturePersistenceError(f"{label} user marker event-log prefix hash changed")
    if prefix_payload and not prefix_payload.endswith((b"\n", b"\r")):
        raise CapturePersistenceError(f"{label} user marker event-log watermark splits a JSONL record")
    parsed, truncated, _ = _parse_jsonl_payload(prefix_payload, f"{label} user marker event-log prefix")
    if truncated:
        raise CapturePersistenceError(f"{label} user marker event-log prefix is truncated")
    prefix_markers = user_markers_from_events(parsed, context)
    count = watermark.get("count") if label == "context" else watermark.get("user_marker_count")
    if len(prefix_markers) != count or prefix_markers != markers[:count]:
        raise CapturePersistenceError(f"{label} user marker event-log prefix conflicts with its marker watermark")


def validate_user_marker_watermarks(
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None,
    markers: list[dict[str, Any]],
    event_log_payload: bytes,
) -> dict[str, Any]:
    """Verify context/checkpoint hashes against prefixes of the durable event log."""
    state = context.get("user_markers")
    if state is None:
        if markers:
            raise CapturePersistenceError("user markers exist but the capture context has no marker-log contract")
        return {"status": "not_recorded", "context_lag": False, "checkpoint_lag": False}
    if not isinstance(state, Mapping):
        raise CapturePersistenceError("capture context user_markers watermark is invalid")
    if state.get("schema") != USER_MARKER_LOG_SCHEMA or state.get("event") != "user_marker":
        raise CapturePersistenceError("capture context user marker log contract is unsupported")
    context_count = _validate_marker_watermark(
        state,
        markers,
        label="context",
        expected_path=str(context.get("events_path") or ""),
    )
    _validate_marker_event_prefix(state, markers, context, event_log_payload, label="context")
    checkpoint_count = 0
    if checkpoint is not None and "user_marker_count" in checkpoint:
        checkpoint_count = _validate_marker_watermark(
            checkpoint,
            markers,
            label="checkpoint",
            expected_path=str(context.get("events_path") or ""),
        )
        _validate_marker_event_prefix(checkpoint, markers, context, event_log_payload, label="checkpoint")
    return {
        "status": "verified",
        "context_lag": context_count < len(markers),
        "checkpoint_lag": checkpoint is not None and checkpoint_count < len(markers),
        "context_marker_count": context_count,
        "checkpoint_marker_count": checkpoint_count,
        "recovered_marker_count": len(markers),
        "markers_sha256": user_marker_stream_sha256(markers),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_directory(value: str | os.PathLike[str]) -> Path:
    try:
        raw = assert_no_reparse_components(Path(value).expanduser())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError("capture output root contains a symbolic link or junction") from exc
    raw.mkdir(parents=True, exist_ok=True)
    root = raw.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise CapturePersistenceError("capture output root must be a regular directory")
    return root


def _direct_child(root: Path, name: Any, *, suffix: str | None = None) -> Path:
    """Resolve only a root child or the exact one-level .internal child."""
    if not isinstance(name, str) or not name:
        raise CapturePersistenceError("capture sidecar path must be a relative owned path")
    relative = name.replace("\\", "/")
    parts = relative.split("/")
    if len(parts) == 1:
        layout = LEGACY_LAYOUT
        leaf = parts[0]
    elif len(parts) == 2 and parts[0] == ".internal":
        layout = INTERNAL_LAYOUT
        leaf = parts[1]
    else:
        raise CapturePersistenceError("capture sidecar path must be a root child or .internal child")
    if leaf in {"", ".", ".."} or (suffix and not leaf.endswith(suffix)):
        raise CapturePersistenceError("capture sidecar path has an invalid name")
    try:
        return owned_sidecar_path(root, name, name=leaf, layout=layout)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError(f"capture sidecar path is unsafe: {name}") from exc


def _metadata_direct_child_matches(
    root: Path,
    value: Any,
    expected_name: str,
    *,
    layout: str = LEGACY_LAYOUT,
) -> bool:
    """Match metadata only to its exact layout-owned sidecar path."""
    if not isinstance(value, str) or not value:
        return False
    try:
        owned_sidecar_path(root, value, name=expected_name, layout=layout)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _publish_exclusive(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if temporary.is_symlink() or not temporary.is_file():
            raise CapturePersistenceError("capture sidecar temporary file changed")
        os.link(temporary, path)
        if not os.path.samefile(temporary, path):
            raise CapturePersistenceError("capture sidecar publication identity changed")
    finally:
        temporary.unlink(missing_ok=True)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while appending capture sidecar")
        view = view[written:]


def _append_bytes(path: Path, payload: bytes, *, durable: bool) -> None:
    if path.is_symlink() or not path.is_file():
        raise CapturePersistenceError(f"capture append target is missing or unsafe: {path.name}")
    flags = os.O_WRONLY | os.O_APPEND
    fd = os.open(path, flags)
    try:
        _write_all(fd, payload)
        if durable:
            os.fsync(fd)
    finally:
        os.close(fd)


def _append_task_jsonl(path: Path, payload: bytes) -> None:
    """Append one independently visible worker/job record."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CapturePersistenceError(f"capture task log is unsafe: {path.name}")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags, 0o600)
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def append_capture_task_event(
    output_root: str | os.PathLike[str],
    session_id: str,
    job_id: str,
    event: str,
    *,
    stage: str,
    layout: str = INTERNAL_LAYOUT,
    capture_id: str | None = None,
    frame_id: int | None = None,
    **fields: Any,
) -> Path:
    """Append a task event to the deterministic per-job JSONL file.

    The desktop parent can append the actual process exit code after the worker
    exits. This log is intentionally separate from snapshot-hashed capture
    events so that the parent's final receipt cannot invalidate an export.
    """
    if not isinstance(session_id, str) or not _SAFE_TOKEN.fullmatch(session_id):
        raise ValueError("session_id is not a safe task-log identifier")
    if not isinstance(job_id, str) or not _SAFE_TOKEN.fullmatch(job_id):
        raise ValueError("job_id is not a safe task-log identifier")
    root = _root_directory(output_root)
    try:
        path = sidecar_path(
            root,
            f"{session_id}.{job_id}.capture-job.jsonl",
            layout=layout,
            create_parent=normalize_sidecar_layout(layout) == INTERNAL_LAYOUT,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError("capture task log path is unsafe") from exc
    document = {
        "schema": "localgaze.capture-task-event/1",
        "event": str(event),
        "timestamp_utc": _utc_now(),
        "stage": str(stage),
        "session_id": session_id,
        "capture_id": capture_id,
        "job_id": job_id,
        "frame_id": frame_id,
        **fields,
    }
    _append_task_jsonl(path, _json_bytes(document))
    return path


def _definitely_incomplete_json(error: json.JSONDecodeError, text: str) -> bool:
    """Recognize EOF-truncated JSON without forgiving complete bad records."""
    stripped = text.rstrip()
    if not stripped:
        return True
    if error.msg.startswith("Unterminated string starting at"):
        return True
    # json.loads reports EOF when a container/value is cut off. An error before
    # the last non-whitespace character identifies a malformed complete value.
    return error.pos >= len(stripped)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CapturePersistenceError(f"capture context is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapturePersistenceError(f"capture context is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise CapturePersistenceError("capture context must contain a JSON object")
    return value


def create_capture_context(
    output_root: str | os.PathLike[str],
    session_id: str,
    *,
    job_id: str | None = None,
    config: Mapping[str, Any] | None = None,
    table_format: str = "csv",
) -> dict[str, Any]:
    """Create the immutable identity/context before a camera worker starts."""
    if not isinstance(session_id, str) or not _SAFE_TOKEN.fullmatch(session_id):
        raise ValueError("session_id is not a safe capture identifier")
    job_id = job_id or os.environ.get("LOCALGAZE_JOB_ID") or uuid.uuid4().hex
    if job_id is not None and (not isinstance(job_id, str) or not _SAFE_TOKEN.fullmatch(job_id)):
        raise ValueError("job_id is not a safe identifier")
    root = _root_directory(output_root)
    if table_format not in {"csv", "txt"}:
        raise ValueError("table_format must be 'csv' or 'txt'")
    internal_directory(root, create=True)
    capture_id = uuid.uuid4().hex
    attempt_id = uuid.uuid4().hex
    stem = f"{session_id}.{capture_id}"
    job_log_id = job_id
    context_name = f"{stem}.capture-context.json"
    checkpoint_name = f"{stem}.capture-checkpoints.jsonl"
    events_name = f"{stem}.capture-events.jsonl"
    job_log_name = f"{session_id}.{job_log_id}.capture-job.jsonl"
    raw_name = f"{session_id}.{table_format}"
    context = {
        "schema": CONTEXT_SCHEMA,
        "sidecar_layout": INTERNAL_LAYOUT,
        "table_format": table_format,
        "session_id": session_id,
        "capture_id": capture_id,
        "attempt_id": attempt_id,
        "job_id": job_id,
        "created_at_utc": _utc_now(),
        "output_root": str(root),
        "raw_path": raw_name,
        "checkpoint_path": sidecar_relative_path(checkpoint_name, INTERNAL_LAYOUT),
        "events_path": sidecar_relative_path(events_name, INTERNAL_LAYOUT),
        "capture_task_log_path": sidecar_relative_path(job_log_name, INTERNAL_LAYOUT),
        "context_path": sidecar_relative_path(context_name, INTERNAL_LAYOUT),
        "config": dict(config or {}),
        "user_markers": _user_marker_watermark(
            [], sidecar_relative_path(events_name, INTERNAL_LAYOUT)
        ),
    }
    context_path = _direct_child(root, context["context_path"], suffix=".capture-context.json")
    checkpoint_path = _direct_child(root, context["checkpoint_path"], suffix=".capture-checkpoints.jsonl")
    events_path = _direct_child(root, context["events_path"], suffix=".capture-events.jsonl")
    _publish_exclusive(context_path, _json_bytes(context))
    _publish_exclusive(checkpoint_path, b"")
    _publish_exclusive(events_path, b"")
    writer = CapturePersistence(context_path, context=context)
    writer.append_event("capture_context_created", stage="prepared", durable=True)
    return dict(context)


def load_capture_context(context_path: str | os.PathLike[str]) -> tuple[dict[str, Any], Path]:
    try:
        raw = assert_no_reparse_components(Path(context_path).expanduser())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError("capture context path contains a symbolic link or junction") from exc
    if not raw.is_file() or not raw.name.endswith(".capture-context.json"):
        raise CapturePersistenceError("select an existing regular *.capture-context.json file")
    context_path = raw.resolve(strict=True)
    context = _read_json(context_path)
    if context.get("schema") != CONTEXT_SCHEMA:
        raise CapturePersistenceError("unsupported capture context schema")
    try:
        layout = normalize_sidecar_layout(context.get("sidecar_layout"))
    except ValueError as exc:
        raise CapturePersistenceError("capture context sidecar_layout is invalid") from exc
    table_format = context.get("table_format", "txt" if layout == LEGACY_LAYOUT else None)
    if table_format not in {"csv", "txt"}:
        raise CapturePersistenceError("capture context table_format is invalid")
    session = context.get("session_id")
    capture = context.get("capture_id")
    attempt = context.get("attempt_id")
    if not isinstance(session, str) or not _SAFE_TOKEN.fullmatch(session):
        raise CapturePersistenceError("capture context session_id is invalid")
    if not isinstance(capture, str) or not re.fullmatch(r"[0-9a-f]{32}", capture):
        raise CapturePersistenceError("capture context capture_id is invalid")
    if not isinstance(attempt, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise CapturePersistenceError("capture context attempt_id is invalid")
    try:
        root = assert_no_reparse_components(Path(context.get("output_root", "")).expanduser())
        root = root.resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError("capture context output root is missing") from exc
    if not root.is_dir():
        raise CapturePersistenceError("capture context output root is not a directory")
    expected = {
        "context_path": sidecar_relative_path(f"{session}.{capture}.capture-context.json", layout),
        "raw_path": f"{session}.{table_format}",
        "checkpoint_path": sidecar_relative_path(f"{session}.{capture}.capture-checkpoints.jsonl", layout),
        "events_path": sidecar_relative_path(f"{session}.{capture}.capture-events.jsonl", layout),
        "capture_task_log_path": sidecar_relative_path(f"{session}.{context.get('job_id') or capture}.capture-job.jsonl", layout),
    }
    for key, filename in expected.items():
        if context.get(key) != filename:
            raise CapturePersistenceError(f"capture context {key} does not match its identity")
    try:
        expected_context = sidecar_path(root, Path(expected["context_path"]).name, layout=layout)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError("capture context path is unsafe") from exc
    if expected_context.resolve(strict=True) != context_path:
        raise CapturePersistenceError("capture context output root does not match its owned location")
    context["sidecar_layout"] = layout
    context["table_format"] = table_format
    for key, suffix in (("checkpoint_path", ".capture-checkpoints.jsonl"), ("events_path", ".capture-events.jsonl"), ("capture_task_log_path", ".capture-job.jsonl")):
        _direct_child(root, context[key], suffix=suffix)
    marker_state = context.get("user_markers")
    if marker_state is not None:
        if (
            not isinstance(marker_state, Mapping)
            or marker_state.get("schema") != USER_MARKER_LOG_SCHEMA
            or marker_state.get("event") != "user_marker"
            or marker_state.get("path") != context.get("events_path")
        ):
            raise CapturePersistenceError("capture context user marker log reference is invalid")
        marker_count = marker_state.get("count")
        marker_sequence = marker_state.get("last_sequence")
        marker_last_ns = marker_state.get("last_monotonic_ns")
        marker_hash = marker_state.get("markers_sha256")
        marker_event_hash = marker_state.get("events_sha256")
        marker_event_size = marker_state.get("events_size_bytes")
        if (
            isinstance(marker_count, bool)
            or not isinstance(marker_count, int)
            or marker_count < 0
            or isinstance(marker_sequence, bool)
            or not isinstance(marker_sequence, int)
            or marker_sequence != marker_count
            or (marker_count == 0 and marker_last_ns is not None)
            or (marker_count > 0 and (not isinstance(marker_last_ns, str) or not re.fullmatch(r"[0-9]+", marker_last_ns)))
            or not isinstance(marker_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", marker_hash)
            or not isinstance(marker_event_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", marker_event_hash)
            or isinstance(marker_event_size, bool)
            or not isinstance(marker_event_size, int)
            or marker_event_size < 0
        ):
            raise CapturePersistenceError("capture context user marker watermark is invalid")
    return context, root.resolve(strict=True)


class CapturePersistence:
    """Append-only stage/checkpoint journal for one capture attempt."""

    def __init__(self, context_path: str | os.PathLike[str], *, context: dict[str, Any] | None = None):
        loaded, root = load_capture_context(context_path)
        if context is not None and loaded != context:
            raise CapturePersistenceError("capture context changed after creation")
        self.context = loaded
        self.root = root
        self.context_path = _direct_child(root, loaded["context_path"], suffix=".capture-context.json")
        self.checkpoint_path = _direct_child(root, loaded["checkpoint_path"], suffix=".capture-checkpoints.jsonl")
        self.events_path = _direct_child(root, loaded["events_path"], suffix=".capture-events.jsonl")
        self.task_log_path = _direct_child(root, loaded["capture_task_log_path"], suffix=".capture-job.jsonl")
        self.raw_path = _direct_child(
            root,
            loaded["raw_path"],
            suffix=".csv" if loaded.get("table_format") == "csv" else ".txt",
        )
        self._log_error: str | None = None
        self._log_errors: dict[str, str] = {}
        self._durability_error: str | None = None
        self._diagnostics: dict[str, int] = {}
        self._last_sync_monotonic = 0.0
        self._append_lock = threading.RLock()
        self._context_bytes = self.context_path.read_bytes()
        self._last_checkpoint, _, _ = read_checkpoints(self.checkpoint_path, self.context)
        event_log = read_event_jsonl(self.events_path, capture_id=self.capture_id)
        self._user_markers = user_markers_from_events(event_log["records"], self.context)
        validate_user_marker_watermarks(
            self.context,
            self._last_checkpoint,
            self._user_markers,
            self.events_path.read_bytes(),
        )

    @property
    def capture_id(self) -> str:
        return str(self.context["capture_id"])

    @property
    def job_id(self) -> str | None:
        value = self.context.get("job_id")
        return value if isinstance(value, str) else None

    @property
    def log_error(self) -> str | None:
        return self._log_error

    @property
    def durability_error(self) -> str | None:
        return self._durability_error

    @property
    def last_checkpoint(self) -> dict[str, Any] | None:
        return dict(self._last_checkpoint) if self._last_checkpoint is not None else None

    @property
    def user_markers(self) -> list[dict[str, Any]]:
        return [dict(marker) for marker in self._user_markers]

    def append_event(self, event: str, *, stage: str, durable: bool = False, **fields: Any) -> bool:
        document = {
            "schema": EVENT_SCHEMA,
            "event": str(event),
            "timestamp_utc": _utc_now(),
            "stage": str(stage),
            "session_id": self.context["session_id"],
            "capture_id": self.capture_id,
            "attempt_id": self.context["attempt_id"],
            "job_id": self.job_id,
            **fields,
        }
        event_logged = True
        try:
            with self._append_lock:
                _append_bytes(self.events_path, _json_bytes(document), durable=durable)
        except Exception as exc:
            event_logged = False
            self._record_log_error("capture_events", exc)
        job_fields = {key: value for key, value in fields.items() if key != "frame_id"}
        job_logged = self.append_job_event(
            event,
            stage=stage,
            frame_id=fields.get("frame_id"),
            capture_event_logged=event_logged,
            **job_fields,
        )
        return event_logged and job_logged

    def _persist_user_marker_watermark(self) -> None:
        updated = dict(self.context)
        updated["user_markers"] = _user_marker_watermark(
            self._user_markers,
            self.context["events_path"],
            self.events_path.read_bytes(),
        )
        payload = _json_bytes(updated)
        temporary = self.context_path.with_name(
            f".{self.context_path.name}.tmp-{uuid.uuid4().hex}"
        )
        try:
            with self._append_lock:
                if self.context_path.is_symlink() or not self.context_path.is_file():
                    raise CapturePersistenceError("capture context is missing or unsafe during marker update")
                if self.context_path.read_bytes() != self._context_bytes:
                    raise CapturePersistenceError("capture context changed before marker watermark update")
                with temporary.open("xb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                if self.context_path.is_symlink() or self.context_path.read_bytes() != self._context_bytes:
                    raise CapturePersistenceError("capture context changed while marker watermark was staged")
                os.replace(temporary, self.context_path)
                self.context = updated
                self._context_bytes = payload
        finally:
            temporary.unlink(missing_ok=True)

    def append_user_marker(self, marker: Mapping[str, Any]) -> dict[str, Any]:
        """Durably append one idempotent keyboard marker to capture events."""
        if not isinstance(marker, Mapping):
            raise TypeError("user marker must be a mapping")
        candidate = dict(marker)
        with self._append_lock:
            before = read_event_jsonl(self.events_path, capture_id=self.capture_id)
            current = user_markers_from_events(before["records"], self.context)
            marker_id = candidate.get("marker_id")
            existing = next((item for item in current if item.get("marker_id") == marker_id), None)
            if existing is not None:
                if existing != candidate:
                    raise CapturePersistenceError("duplicate marker_id has conflicting content")
                self._user_markers = current
                reconciliation = validate_user_marker_watermarks(
                    self.context,
                    self._last_checkpoint,
                    current,
                    self.events_path.read_bytes(),
                )
                if reconciliation.get("context_lag"):
                    self._persist_user_marker_watermark()
                return dict(existing)

            proposed_event = {
                "event": "user_marker",
                "session_id": self.context["session_id"],
                "capture_id": self.capture_id,
                "attempt_id": self.context["attempt_id"],
                "job_id": self.job_id,
                "marker": candidate,
            }
            proposed = user_markers_from_events([*before["records"], proposed_event], self.context)
            if len(proposed) != len(current) + 1:
                raise CapturePersistenceError("user marker sequence did not advance exactly once")

            self.append_event(
                "user_marker",
                stage="capture_marker",
                durable=True,
                marker=candidate,
            )
            after = read_event_jsonl(self.events_path, capture_id=self.capture_id)
            durable_markers = user_markers_from_events(after["records"], self.context)
            if not durable_markers or durable_markers[-1] != candidate:
                raise CapturePersistenceError("user marker was not durably present in the capture event log")
            self._user_markers = durable_markers
            try:
                self._persist_user_marker_watermark()
            except Exception as exc:
                self._record_log_error("capture_context", exc)
                raise CapturePersistenceError(
                    f"marker is durable in capture events but its context watermark could not be updated: {exc}"
                ) from exc
            return dict(candidate)

    def reconcile_user_markers(self) -> dict[str, Any]:
        """Rebuild and persist a validated marker watermark from the event log."""
        with self._append_lock:
            event_log = read_event_jsonl(self.events_path, capture_id=self.capture_id)
            markers = user_markers_from_events(event_log["records"], self.context)
            reconciliation = validate_user_marker_watermarks(
                self.context,
                self._last_checkpoint,
                markers,
                self.events_path.read_bytes(),
            )
            self._user_markers = markers
            if reconciliation.get("context_lag"):
                self._persist_user_marker_watermark()
            return reconciliation

    def _record_log_error(self, channel: str, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self._log_errors.setdefault(channel, message)
        self._log_error = self._log_error or message
        if len(self._log_errors) == 1:
            try:
                sys.stderr.write(f"capture {channel} write failed: {message}\n")
                sys.stderr.flush()
            except Exception:
                pass

    def append_job_event(
        self,
        event: str,
        *,
        stage: str,
        capture_id: str | None = None,
        frame_id: int | None = None,
        **fields: Any,
    ) -> bool:
        try:
            with self._append_lock:
                append_capture_task_event(
                    self.root,
                    self.context["session_id"],
                    str(self.context.get("job_id") or self.capture_id),
                    event,
                    stage=stage,
                    layout=self.context.get("sidecar_layout", LEGACY_LAYOUT),
                    capture_id=self.capture_id if capture_id is None else capture_id,
                    frame_id=frame_id,
                    attempt_id=self.context["attempt_id"],
                    **fields,
                )
            return True
        except Exception as exc:
            self._record_log_error("capture_job_log", exc)
            return False

    def record_duplicate_diagnostic(self, kind: str, *, frame_id: int | None = None) -> None:
        key = str(kind or "duplicate_frame")[:96]
        self._diagnostics[key] = self._diagnostics.get(key, 0) + 1
        if self._diagnostics[key] == 1:
            self._diagnostics[f"{key}:first_frame"] = int(frame_id) if frame_id is not None else -1

    def _flush_diagnostics(self, *, durable: bool) -> None:
        counts = {key: value for key, value in self._diagnostics.items() if not key.endswith(":first_frame")}
        if not counts:
            return
        first = {key[:-12]: value for key, value in self._diagnostics.items() if key.endswith(":first_frame")}
        if self.append_event("diagnostic_summary", stage="capture", durable=durable, counts=counts, first_frame=first):
            self._diagnostics.clear()

    def finish_capture(
        self,
        raw_file,
        *,
        complete_records: int,
        last_frame_time: Any,
        last_frame_id: int | None = None,
        status: str,
        error: str | None = None,
        traceback_text: str | None = None,
        capture_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        checkpoint = None
        if raw_file is not None and not getattr(raw_file, "closed", False):
            checkpoint = self.checkpoint(
                raw_file,
                complete_records=complete_records,
                last_frame_time=last_frame_time,
                last_frame_id=last_frame_id,
                stage="capture_final",
                force=True,
            )
        self._flush_diagnostics(durable=True)
        self.append_event(
            "capture_finished",
            stage="capture_final",
            durable=True,
            status=str(status),
            complete_records=int(complete_records),
            last_frame_time=last_frame_time,
            last_frame_id=last_frame_id,
            error=error,
            traceback=traceback_text,
            capture_metadata=dict(capture_metadata or {}),
            final_checkpoint_sequence=(checkpoint or self._last_checkpoint or {}).get("sequence"),
            durability_error=self._durability_error,
            log_io_errors=dict(self._log_errors),
        )
        return checkpoint

    def bind_schema(self, columns: Iterable[str]) -> bool:
        names = list(columns)
        if not names or any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
            raise CapturePersistenceError("capture TSV schema has invalid or duplicate columns")
        return self.append_event(
            "capture_schema",
            stage="capture_prepared",
            durable=True,
            columns=names,
            column_count=len(names),
        )

    def bind_motion_artifacts(self, *, motion_capture_id: str, event_log: str | None) -> bool:
        if motion_capture_id != self.capture_id:
            raise CapturePersistenceError("motion capture identity must match the capture identity")
        relative = None
        if event_log is not None:
            raw = Path(event_log)
            if raw.is_absolute() or ".." in raw.parts:
                raise CapturePersistenceError("motion event log must be relative to output_root")
            resolved = (self.root / raw).resolve(strict=False)
            if not resolved.is_relative_to(self.root):
                raise CapturePersistenceError("motion event log escapes output_root")
            relative = resolved.relative_to(self.root).as_posix()
        return self.append_event(
            "motion_capture_started",
            stage="motion_started",
            durable=True,
            motion_capture_id=motion_capture_id,
            motion_event_log=relative,
        )

    def checkpoint(
        self,
        raw_file,
        *,
        complete_records: int,
        last_frame_time: Any,
        last_frame_id: int | None = None,
        stage: str,
        force: bool = False,
    ) -> dict[str, Any] | None:
        now = time.monotonic()
        if not force and now - self._last_sync_monotonic < 1.0:
            return self._last_checkpoint
        try:
            raw_file.flush()
            os.fsync(raw_file.fileno())
            st = os.fstat(raw_file.fileno())
            byte_offset = int(st.st_size)
            entry = {
                "schema": CHECKPOINT_SCHEMA,
                "sequence": (self._last_checkpoint or {}).get("sequence", 0) + 1,
                "timestamp_utc": _utc_now(),
                "capture_id": self.capture_id,
                "attempt_id": self.context["attempt_id"],
                "raw_path": self.raw_path.name,
                "raw_device": int(st.st_dev),
                "raw_file_id": int(st.st_ino),
                "byte_offset": byte_offset,
                "complete_records": int(complete_records),
                "last_frame_time": last_frame_time,
                "last_frame_id": last_frame_id,
                "stage": str(stage),
            }
            if "user_markers" in self.context:
                event_log_payload = self.events_path.read_bytes()
                marker_state = _user_marker_watermark(
                    self._user_markers,
                    self.context["events_path"],
                    event_log_payload,
                )
                entry.update({
                    "user_marker_log_path": marker_state["path"],
                    "user_marker_count": marker_state["count"],
                    "user_marker_last_sequence": marker_state["last_sequence"],
                    "user_marker_last_monotonic_ns": marker_state["last_monotonic_ns"],
                    "user_marker_stream_sha256": marker_state["markers_sha256"],
                    "user_marker_events_sha256": marker_state["events_sha256"],
                    "user_marker_events_size_bytes": marker_state["events_size_bytes"],
                })
            _append_bytes(self.checkpoint_path, _json_bytes(entry), durable=True)
            self._last_checkpoint = entry
            self._last_sync_monotonic = now
            self._flush_diagnostics(durable=force)
            self.append_event(
                "durable_checkpoint",
                stage=stage,
                durable=force,
                checkpoint_sequence=entry["sequence"],
                byte_offset=byte_offset,
                complete_records=entry["complete_records"],
                last_frame_time=last_frame_time,
                last_frame_id=last_frame_id,
                frame_id=last_frame_id,
            )
            return entry
        except Exception as exc:
            self._durability_error = f"{type(exc).__name__}: {exc}"
            self.append_event(
                "durability_error",
                stage=stage,
                durable=True,
                error_type=type(exc).__name__,
                error=str(exc),
                checkpoint_advanced=False,
            )
            return None

    def snapshot_metadata(self, *, recovery: Mapping[str, Any] | None = None) -> dict[str, Any]:
        checkpoint, checkpoint_tail, checkpoint_tail_bytes = read_checkpoints(self.checkpoint_path, self.context)
        events = self.events_path
        event_log_payload = events.read_bytes()
        parsed_events = read_event_jsonl(self.events_path, capture_id=self.capture_id)
        events_sha256 = hashlib.sha256(event_log_payload).hexdigest()
        events_size = len(event_log_payload)
        if parsed_events["sha256"] != events_sha256 or parsed_events["size_bytes"] != events_size:
            raise CapturePersistenceError("capture event log changed while snapshot metadata was read")
        parsed_markers = user_markers_from_events(parsed_events["records"], self.context)
        validate_user_marker_watermarks(self.context, checkpoint, parsed_markers, event_log_payload)
        if parsed_markers != self._user_markers:
            raise CapturePersistenceError("capture user marker state changed outside its owned writer")
        marker_state = _user_marker_watermark(
            self._user_markers,
            self.context["events_path"],
            event_log_payload,
        )
        result: dict[str, Any] = {
            "schema": "localgaze.capture-persistence-summary/1",
            "sidecar_layout": self.context.get("sidecar_layout", LEGACY_LAYOUT),
            "table_format": self.context.get("table_format", "txt"),
            "session_id": self.context["session_id"],
            "capture_id": self.capture_id,
            "attempt_id": self.context["attempt_id"],
            "job_id": self.job_id,
            "context_sha256": _sha256(self.context_path),
            "context_path": self.context_path.relative_to(self.root).as_posix(),
            "checkpoint_path": self.checkpoint_path.relative_to(self.root).as_posix(),
            "checkpoint_sha256": _sha256(self.checkpoint_path),
            "last_durable_checkpoint": checkpoint,
            "events_path": events.relative_to(self.root).as_posix(),
            "events_sha256": events_sha256,
            "events_size_bytes": events_size,
            "user_markers": self.user_markers,
            "user_marker_log": {
                **marker_state,
                "events_sha256": events_sha256,
                "events_size_bytes": events_size,
            },
            "capture_task_log_path": self.task_log_path.relative_to(self.root).as_posix(),
            "raw_path": self.raw_path.name,
            "incomplete_checkpoint_records": checkpoint_tail,
            "incomplete_checkpoint_bytes": checkpoint_tail_bytes,
            "log_io_error": self._log_error,
            "log_io_errors": dict(self._log_errors),
            "durability_error": self._durability_error,
            "recovery": dict(recovery or {"status": "normal_capture"}),
        }
        return result

    def cleanup_recovery_sidecars(self, *, completion_verified: bool, expected_metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Delete only this attempt's context/checkpoint after snapshot verification."""
        if not completion_verified:
            return {"status": "retained", "reason": "completion_not_verified", "retained_files": [str(self.context_path), str(self.checkpoint_path)]}
        metadata = expected_metadata.get("capture_persistence") if isinstance(expected_metadata, Mapping) else None
        if not isinstance(metadata, Mapping):
            return {"status": "retained", "reason": "snapshot_missing_persistence_metadata", "retained_files": [str(self.context_path), str(self.checkpoint_path)]}
        expected_context_name = f"{self.context['session_id']}.{self.capture_id}.capture-context.json"
        expected_checkpoint_name = f"{self.context['session_id']}.{self.capture_id}.capture-checkpoints.jsonl"
        if (
            metadata.get("capture_id") != self.capture_id
            or metadata.get("attempt_id") != self.context.get("attempt_id")
            or normalize_sidecar_layout(metadata.get("sidecar_layout", self.context.get("sidecar_layout"))) != self.context.get("sidecar_layout", LEGACY_LAYOUT)
            or not _metadata_direct_child_matches(
                self.root, metadata.get("context_path"), expected_context_name,
                layout=self.context.get("sidecar_layout", LEGACY_LAYOUT),
            )
            or not _metadata_direct_child_matches(
                self.root, metadata.get("checkpoint_path"), expected_checkpoint_name,
                layout=self.context.get("sidecar_layout", LEGACY_LAYOUT),
            )
        ):
            return {
                "status": "retained",
                "reason": "snapshot_sidecar_identity_mismatch",
                "retained_files": [str(self.context_path), str(self.checkpoint_path)],
            }
        # Remove the checkpoint first and the context last so a failed cleanup
        # leaves enough identity to retry safely. Preflight both files first so
        # a replaced context never causes the valid checkpoint to be deleted.
        expected_hashes = (
            (self.checkpoint_path, metadata.get("checkpoint_sha256")),
            (self.context_path, metadata.get("context_sha256")),
        )
        for path, expected_hash in expected_hashes:
            if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                return {
                    "status": "retained",
                    "reason": "snapshot_sidecar_hash_missing_or_invalid",
                    "retained_files": [str(candidate) for candidate, _ in expected_hashes if candidate.exists() or candidate.is_symlink()],
                }
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file():
                return {
                    "status": "retained",
                    "reason": f"{path.name} is missing or replaced",
                    "retained_files": [str(candidate) for candidate, _ in expected_hashes if candidate.exists() or candidate.is_symlink()],
                }
            if _sha256(path) != expected_hash:
                return {
                    "status": "retained",
                    "reason": f"{path.name} content changed after snapshot",
                    "retained_files": [str(candidate) for candidate, _ in expected_hashes if candidate.exists() or candidate.is_symlink()],
                }
        deleted: list[str] = []
        already_absent: list[str] = []
        retained: list[dict[str, str]] = []
        for index, (path, expected_hash) in enumerate(expected_hashes):
            try:
                if not path.exists() and not path.is_symlink():
                    already_absent.append(str(path))
                    continue
                if path.is_symlink() or not path.is_file():
                    raise CapturePersistenceError("sidecar is missing or replaced")
                if _sha256(path) != expected_hash:
                    raise CapturePersistenceError("sidecar content changed after snapshot")
                path.unlink()
                if path.exists() or path.is_symlink():
                    raise CapturePersistenceError("sidecar still exists after deletion")
                deleted.append(str(path))
            except Exception as exc:
                retained.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
                # Keep the context whenever checkpoint cleanup fails. It is the
                # identity needed to verify a later retry of cleanup.
                for later_path, _ in expected_hashes[index + 1:]:
                    if later_path.exists() or later_path.is_symlink():
                        retained.append({
                            "path": str(later_path),
                            "reason": "retained_after_prior_cleanup_failure",
                        })
                break
        if not retained:
            for path, _ in expected_hashes:
                if path.exists() or path.is_symlink():
                    retained.append({"path": str(path), "reason": "sidecar still exists after cleanup"})
        return {
            "status": "completed" if not retained else "partial",
            "deleted": deleted,
            "already_absent": already_absent,
            "retained_files": retained,
        }


def read_checkpoints(path: Path, context: Mapping[str, Any]) -> tuple[dict[str, Any] | None, int, int]:
    if path.is_symlink() or not path.is_file():
        return None, 0, 0
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    last: dict[str, Any] | None = None
    incomplete_tail = 0
    incomplete_tail_bytes = 0
    for index, line in enumerate(lines):
        complete = line.endswith((b"\n", b"\r"))
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            if index == len(lines) - 1 and not complete and exc.reason == "unexpected end of data" and exc.end == len(line):
                incomplete_tail += 1
                incomplete_tail_bytes += len(line)
                break
            raise CapturePersistenceError(f"checkpoint JSONL contains invalid UTF-8 at line {index + 1}") from exc
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not complete and _definitely_incomplete_json(exc, text):
                incomplete_tail += 1
                incomplete_tail_bytes += len(line)
                break
            raise CapturePersistenceError(f"checkpoint JSONL contains a malformed complete/interior record at line {index + 1}") from exc
        if not complete and index != len(lines) - 1:
            raise CapturePersistenceError("checkpoint JSONL has a non-terminal unterminated record")
        if not isinstance(item, dict) or item.get("schema") != CHECKPOINT_SCHEMA:
            raise CapturePersistenceError(f"checkpoint record {index + 1} has an invalid schema")
        if item.get("capture_id") != context.get("capture_id") or item.get("attempt_id") != context.get("attempt_id"):
            raise CapturePersistenceError(f"checkpoint record {index + 1} belongs to another capture")
        if item.get("raw_path") != context.get("raw_path"):
            raise CapturePersistenceError(f"checkpoint record {index + 1} names a different raw file")
        offset = item.get("byte_offset")
        count = item.get("complete_records")
        sequence = item.get("sequence")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise CapturePersistenceError(f"checkpoint record {index + 1} has an invalid byte offset")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CapturePersistenceError(f"checkpoint record {index + 1} has an invalid record count")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise CapturePersistenceError(f"checkpoint record {index + 1} has an invalid sequence")
        if (last is None and sequence != 1) or (last is not None and (sequence != last.get("sequence", 0) + 1 or offset < last.get("byte_offset", 0) or count < last.get("complete_records", 0))):
            raise CapturePersistenceError(f"checkpoint record {index + 1} moves backwards or skips its sequence")
        marker_keys = {
            "user_marker_log_path",
            "user_marker_count",
            "user_marker_last_sequence",
            "user_marker_last_monotonic_ns",
            "user_marker_stream_sha256",
            "user_marker_events_sha256",
            "user_marker_events_size_bytes",
        }
        marker_keys_present = marker_keys.intersection(item)
        if marker_keys_present:
            if marker_keys_present != marker_keys or not isinstance(context.get("user_markers"), Mapping):
                raise CapturePersistenceError(f"checkpoint record {index + 1} has an incomplete user marker watermark")
            marker_count = item.get("user_marker_count")
            marker_sequence = item.get("user_marker_last_sequence")
            marker_last_ns = item.get("user_marker_last_monotonic_ns")
            marker_hash = item.get("user_marker_stream_sha256")
            marker_event_hash = item.get("user_marker_events_sha256")
            marker_event_size = item.get("user_marker_events_size_bytes")
            if (
                item.get("user_marker_log_path") != context.get("events_path")
                or isinstance(marker_count, bool)
                or not isinstance(marker_count, int)
                or marker_count < 0
                or isinstance(marker_sequence, bool)
                or not isinstance(marker_sequence, int)
                or marker_sequence != marker_count
                or (marker_count == 0 and marker_last_ns is not None)
                or (marker_count > 0 and (not isinstance(marker_last_ns, str) or not re.fullmatch(r"[0-9]+", marker_last_ns)))
                or not isinstance(marker_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", marker_hash)
                or not isinstance(marker_event_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", marker_event_hash)
                or isinstance(marker_event_size, bool)
                or not isinstance(marker_event_size, int)
                or marker_event_size < 0
            ):
                raise CapturePersistenceError(f"checkpoint record {index + 1} has an invalid user marker watermark")
            if marker_count == 0 and marker_hash != user_marker_stream_sha256([]):
                raise CapturePersistenceError(f"checkpoint record {index + 1} has a wrong empty marker hash")
            if last is not None and "user_marker_count" in last:
                previous_marker_count = last["user_marker_count"]
                previous_marker_hash = last["user_marker_stream_sha256"]
                previous_marker_last_ns = last["user_marker_last_monotonic_ns"]
                previous_marker_event_hash = last["user_marker_events_sha256"]
                previous_marker_event_size = last["user_marker_events_size_bytes"]
                if (
                    marker_count < previous_marker_count
                    or (marker_count == previous_marker_count and marker_hash != previous_marker_hash)
                    or (marker_count == previous_marker_count and marker_last_ns != previous_marker_last_ns)
                    or (
                        marker_count > previous_marker_count
                        and previous_marker_last_ns is not None
                        and marker_last_ns is not None
                        and int(marker_last_ns) < int(previous_marker_last_ns)
                    )
                    or marker_event_size < previous_marker_event_size
                    or (marker_event_size == previous_marker_event_size and marker_event_hash != previous_marker_event_hash)
                ):
                    raise CapturePersistenceError(f"checkpoint record {index + 1} moves its user marker watermark backwards")
        last = item
    return last, incomplete_tail, incomplete_tail_bytes

def _decode_tsv_value(name: str, value: str) -> Any:
    if value == "":
        return None
    normalized_name = name.casefold()
    if normalized_name in {"x", "y", "leftpupil", "rightpupil", "elapsed_s", "processing_duration_ns", "corrected_x_px", "corrected_y_px", "left_iris_diameter_over_eye_span", "right_iris_diameter_over_eye_span", "iris_eye_span_px", "left_iris_center_x_px", "left_iris_center_y_px", "right_iris_center_x_px", "right_iris_center_y_px"}:
        try:
            return float(value)
        except ValueError:
            return value
    if normalized_name in {"frame_id", "received_monotonic_ns", "process_started_monotonic_ns", "process_finished_monotonic_ns", "frames_dropped_before_frame", "motion_body_frame_id", "motion_body_received_monotonic_ns"}:
        try:
            return int(value)
        except ValueError:
            return value
    if normalized_name in {"duplicate_image", "sample_accepted", "is_pupil_measurement", "legacy_iris_mode", "head_descriptor_outside_training_range", "motion_enabled", "motion_head_fresh", "motion_head_available", "motion_head_valid", "motion_body_fresh", "motion_body_available", "motion_body_valid"}:
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value
    if name.endswith("_json"):
        try:
            json.loads(value)
            return value
        except (json.JSONDecodeError, RecursionError) as exc:
            raise CapturePersistenceError(f"capture TSV contains malformed JSON field {name}: {exc}") from exc
    if normalized_name in {"timestamp", "时间"}:
        return value
    if normalized_name == "blink":
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return value
    return value


def _canonical_capture_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map legacy TSV labels onto the regular in-memory gaze record keys."""
    result: dict[str, Any] = {}
    source_names: dict[str, str] = {}
    for name, value in record.items():
        canonical = _LEGACY_CAPTURE_COLUMN_NAMES.get(name.casefold(), name)
        if canonical in result:
            if result[canonical] != value:
                raise CapturePersistenceError(
                    f"capture TSV contains conflicting columns for {canonical!r}: "
                    f"{source_names[canonical]!r} and {name!r}"
                )
            continue
        result[canonical] = value
        source_names[canonical] = name
    return result


class _Utf8PhysicalLineIterator:
    """Yield universal-newline physical lines while counting source bytes exactly."""

    def __init__(self, source):
        self.source = source
        self.buffer = bytearray()
        self.eof = False
        self.byte_offset = 0
        self.last_line_start = 0
        self.last_line_bytes = 0
        self.last_line_terminated = True

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            lf = self.buffer.find(b"\n")
            cr = self.buffer.find(b"\r")
            choices = [position for position in (lf, cr) if position >= 0]
            if choices:
                delimiter = min(choices)
                if self.buffer[delimiter] == 13 and delimiter + 1 == len(self.buffer) and not self.eof:
                    self._read_more()
                    continue
                end = delimiter + 1
                if self.buffer[delimiter] == 13 and end < len(self.buffer) and self.buffer[end] == 10:
                    end += 1
                raw_line = bytes(self.buffer[:end])
                del self.buffer[:end]
                terminated = True
            elif self.eof:
                if not self.buffer:
                    raise StopIteration
                raw_line = bytes(self.buffer)
                self.buffer.clear()
                terminated = False
            else:
                self._read_more()
                continue

            self.last_line_start = self.byte_offset
            self.last_line_bytes = len(raw_line)
            self.last_line_terminated = terminated
            self.byte_offset += len(raw_line)
            return raw_line.decode("utf-8")

    def _read_more(self):
        block = self.source.read(64 * 1024)
        if block:
            self.buffer.extend(block)
        else:
            self.eof = True


def read_capture_tsv(
    raw_path: str | os.PathLike[str],
    *,
    context: Mapping[str, Any],
    expected_header: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Parse complete TSV logical records with exact UTF-8 source byte offsets.

    A terminal incomplete logical row can be counted and omitted only after all
    checkpointed records have parsed. Complete rows beyond the checkpoint are
    retained and labeled as not previously confirmed durable.
    """
    try:
        path = assert_no_reparse_components(Path(raw_path).expanduser())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError("capture TSV path contains a symbolic link or junction") from exc
    if not path.is_file():
        raise CapturePersistenceError("capture TSV is missing or unsafe")
    root = Path(context["output_root"]).resolve(strict=True)
    if path.resolve(strict=True) != root / context.get("raw_path"):
        raise CapturePersistenceError("capture TSV does not match its context-owned output path")
    stat_before = path.stat()
    checkpoints_path = _direct_child(root, context["checkpoint_path"], suffix=".capture-checkpoints.jsonl")
    checkpoint, checkpoint_tail, checkpoint_tail_bytes = read_checkpoints(checkpoints_path, context)
    if checkpoint is None:
        raise CapturePersistenceError("capture has no valid durable checkpoint")
    if checkpoint.get("raw_file_id") != int(stat_before.st_ino) or checkpoint.get("raw_device") != int(stat_before.st_dev):
        raise CapturePersistenceError("capture TSV identity differs from the durable checkpoint")
    if stat_before.st_size < checkpoint["byte_offset"]:
        raise CapturePersistenceError("capture TSV is shorter than its durable checkpoint")

    records: list[dict[str, Any]] = []
    boundaries: list[tuple[int, int]] = []
    truncated_terminal_records = 0
    truncated_terminal_bytes = 0
    complete_terminal_records_without_newline = 0
    raw_size = int(stat_before.st_size)
    with path.open("rb") as source:
        physical_lines = _Utf8PhysicalLineIterator(source)
        table_format = context.get("table_format", "txt")
        delimiter = "," if table_format == "csv" else "\t"
        reader = csv.reader(physical_lines, delimiter=delimiter, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise CapturePersistenceError("capture TSV has no header") from exc
        except (csv.Error, UnicodeError) as exc:
            raise CapturePersistenceError(f"capture TSV header is damaged: {exc}") from exc
        if not header or len(header) != len(set(header)):
            raise CapturePersistenceError("capture TSV header is empty or contains duplicate columns")
        if expected_header is not None and list(expected_header) != header:
            raise CapturePersistenceError("capture TSV header does not match the active capture schema")
        boundaries.append((physical_lines.byte_offset, 0))
        last_complete_offset = physical_lines.byte_offset

        while True:
            before_record = last_complete_offset
            try:
                row = next(reader)
            except StopIteration:
                break
            except UnicodeDecodeError as exc:
                terminal_partial_utf8 = (
                    physical_lines.eof
                    and not physical_lines.last_line_terminated
                    and physical_lines.last_line_start >= checkpoint["byte_offset"]
                    and exc.reason == "unexpected end of data"
                    and exc.end == len(exc.object)
                )
                if terminal_partial_utf8:
                    truncated_terminal_records += 1
                    truncated_terminal_bytes += max(0, raw_size - before_record)
                    break
                raise CapturePersistenceError(f"capture TSV contains invalid UTF-8 before its terminal tail: {exc}") from exc
            except csv.Error as exc:
                terminal_incomplete_csv = (
                    str(exc) == "unexpected end of data"
                    and physical_lines.eof
                    and physical_lines.byte_offset == raw_size
                    and before_record >= checkpoint["byte_offset"]
                )
                if terminal_incomplete_csv:
                    truncated_terminal_records += 1
                    truncated_terminal_bytes += max(0, raw_size - before_record)
                    break
                raise CapturePersistenceError(f"capture TSV is corrupt before its terminal tail: {exc}") from exc

            row_end = physical_lines.byte_offset
            if len(row) != len(header):
                if (
                    row_end == raw_size
                    and not physical_lines.last_line_terminated
                    and before_record >= checkpoint["byte_offset"]
                ):
                    truncated_terminal_records += 1
                    truncated_terminal_bytes += max(0, raw_size - before_record)
                    break
                raise CapturePersistenceError(f"capture TSV record has {len(row)} fields; expected {len(header)}")
            decoded = {name: _decode_tsv_value(name, value) for name, value in zip(header, row)}
            records.append(_canonical_capture_record(decoded))
            last_complete_offset = row_end
            boundaries.append((row_end, len(records)))
            if row_end == raw_size and not physical_lines.last_line_terminated:
                complete_terminal_records_without_newline += 1

    durable_boundary = next((count for offset, count in boundaries if offset == checkpoint["byte_offset"]), None)
    if durable_boundary is None or durable_boundary != checkpoint["complete_records"]:
        raise CapturePersistenceError("durable checkpoint does not match a complete TSV logical-record byte boundary")
    if len(records) < checkpoint["complete_records"]:
        raise CapturePersistenceError("parsed capture TSV is shorter than its durable record count")
    if checkpoint["byte_offset"] == raw_size and complete_terminal_records_without_newline:
        raise CapturePersistenceError("unterminated TSV record intersects the durable checkpoint")
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns, stat_before.st_ino) != (stat_after.st_size, stat_after.st_mtime_ns, stat_after.st_ino):
        raise CapturePersistenceError("capture TSV changed while recovery was reading it")
    extra = len(records) - checkpoint["complete_records"]
    return {
        "header": header,
        "records": records,
        "last_durable_checkpoint": checkpoint,
        "complete_records_after_checkpoint": extra,
        "complete_tail_without_record_terminator": complete_terminal_records_without_newline,
        "truncated_terminal_records": truncated_terminal_records,
        "truncated_terminal_bytes": truncated_terminal_bytes,
        "incomplete_checkpoint_records": checkpoint_tail,
        "incomplete_checkpoint_bytes": checkpoint_tail_bytes,
        "raw_size_bytes": stat_after.st_size,
        "raw_sha256": _sha256(path),
        "raw_path": str(path.resolve()),
    }

def read_event_jsonl(path: str | os.PathLike[str], *, capture_id: str) -> dict[str, Any]:
    """Read JSONL events; ignore only an explicitly unterminated final line."""
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise CapturePersistenceError("capture event log is missing or unsafe")
    payload = raw_path.read_bytes()
    parsed, truncated_tail, truncated_tail_bytes = _parse_jsonl_payload(payload, "capture event JSONL")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict) or item.get("schema") != EVENT_SCHEMA:
            raise CapturePersistenceError(f"capture event {index + 1} has an invalid schema")
        if item.get("capture_id") != capture_id:
            raise CapturePersistenceError(f"capture event {index + 1} belongs to another capture")
        records.append(item)
    return {
        "records": records,
        "truncated_terminal_records": truncated_tail,
        "truncated_terminal_bytes": truncated_tail_bytes,
        "sha256": _sha256(raw_path),
        "size_bytes": len(payload),
    }


def _parse_jsonl_payload(payload: bytes, label: str) -> tuple[list[Any], int, int]:
    """Keep valid JSON records without a newline; skip only EOF-truncated data."""
    lines = payload.splitlines(keepends=True)
    records: list[Any] = []
    truncated_records = 0
    truncated_bytes = 0
    for index, line in enumerate(lines):
        terminated = line.endswith((b"\n", b"\r"))
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            if index == len(lines) - 1 and not terminated and exc.reason == "unexpected end of data" and exc.end == len(line):
                truncated_records += 1
                truncated_bytes += len(line)
                break
            raise CapturePersistenceError(f"{label} has invalid UTF-8 at line {index + 1}") from exc
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not terminated and _definitely_incomplete_json(exc, text):
                truncated_records += 1
                truncated_bytes += len(line)
                break
            raise CapturePersistenceError(f"{label} has a malformed complete/interior record at line {index + 1}") from exc
        if not terminated and index != len(lines) - 1:
            raise CapturePersistenceError(f"{label} has a non-terminal unterminated record")
        records.append(item)
    return records, truncated_records, truncated_bytes


def motion_recovery_summary(
    context: Mapping[str, Any],
    capture_events: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild movement event state from MovementArtifactWriter's JSONL updates."""
    capture_id = context.get("capture_id")
    try:
        output_root = assert_no_reparse_components(Path(context["output_root"]).expanduser()).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError("capture output root contains a symbolic link or junction") from exc
    layout = normalize_sidecar_layout(context.get("sidecar_layout"))
    session = context.get("session_id")
    event_name = None
    for item in capture_events.get("records", []):
        if item.get("event") == "motion_capture_started":
            if item.get("motion_capture_id") != capture_id:
                raise CapturePersistenceError("motion capture event identity does not match capture context")
            event_name = item.get("motion_event_log")
    if event_name is None:
        return {"status": "not_recorded", "open_events": [], "event_log": None}
    expected_event = (
        sidecar_relative_path(f"{session}.{capture_id}.movement-events.jsonl", layout)
        if layout == INTERNAL_LAYOUT
        else f"motion_artifacts/{session}/{capture_id}/movement-events.jsonl"
    )
    if event_name != expected_event:
        raise CapturePersistenceError("motion event log path does not match its declared layout and capture identity")
    try:
        motion_path = output_child_path(output_root, event_name)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CapturePersistenceError("motion event log reference is unsafe") from exc
    if not motion_path.is_file():
        raise CapturePersistenceError("motion event log is missing or unsafe")
    payload = motion_path.read_bytes()
    lines, truncated_tail, truncated_tail_bytes = _parse_jsonl_payload(payload, "motion event JSONL")
    latest_events: dict[str, dict[str, Any]] = {}
    active_ids: set[str] = set()
    closed_ids: set[str] = set()
    keyframes: list[dict[str, Any]] = []
    finish_record: Mapping[str, Any] | None = None
    valid_rows = 0
    for index, item in enumerate(lines):
        if not isinstance(item, dict) or item.get("record_type") != "event_update":
            raise CapturePersistenceError(f"motion event {index + 1} has an invalid record type")
        if item.get("capture_id") != capture_id:
            raise CapturePersistenceError(f"motion event {index + 1} belongs to another capture")
        kind = item.get("kind")
        written_at = item.get("written_at_monotonic_ns")
        if not isinstance(written_at, str) or not re.fullmatch(r"\d+", written_at):
            raise CapturePersistenceError(f"motion event {index + 1} has an invalid writer timestamp")
        if finish_record is not None:
            raise CapturePersistenceError("motion event log contains records after capture_finish")
        if kind in {"start", "trigger", "peak", "end"}:
            event_id = item.get("event_id")
            channel = item.get("channel")
            event = item.get("event")
            if (
                not isinstance(event_id, str) or not event_id
                or channel not in {"head", "upper_body"}
                or not isinstance(event, Mapping)
                or event.get("event_id") != event_id
                or event.get("capture_id") != capture_id
                or event.get("channel") != channel
                or not isinstance(event.get("type"), str)
                or event.get("type") not in {"motion", "tracking_lost"}
            ):
                raise CapturePersistenceError(f"motion event {index + 1} has an invalid event schema")
            if kind == "start":
                if event_id in latest_events or event.get("status") not in {"active", "completed"}:
                    raise CapturePersistenceError(f"motion event {event_id} has an invalid start transition")
                if event.get("status") == "completed":
                    if "end" not in event or "end_reason" not in event:
                        raise CapturePersistenceError(f"motion event {event_id} has an incomplete closed start state")
                    closed_ids.add(event_id)
                else:
                    active_ids.add(event_id)
            elif event_id not in active_ids:
                if kind == "end" and event_id in closed_ids and latest_events.get(event_id) == dict(event):
                    # Some instantaneous updates are serialized after their
                    # in-memory event has already reached its closed state.
                    pass
                else:
                    raise CapturePersistenceError(f"motion event {event_id} updates without a matching start")
            elif kind in {"trigger", "peak"} and event.get("status") != "active":
                raise CapturePersistenceError(f"motion event {event_id} has an invalid {kind} state")
            elif kind == "end":
                if event.get("status") != "completed" or "end" not in event or "end_reason" not in event:
                    raise CapturePersistenceError(f"motion event {event_id} has an invalid end state")
                active_ids.remove(event_id)
                closed_ids.add(event_id)
            latest_events[event_id] = dict(event)
        elif kind == "keyframe":
            event_id = item.get("event_id")
            frame = item.get("keyframe")
            if (
                not isinstance(event_id, str) or event_id not in latest_events
                or not isinstance(frame, Mapping)
                or frame.get("event_id") != event_id
                or frame.get("capture_id") != capture_id
                or not isinstance(frame.get("keyframe_type"), str)
                or frame.get("status") not in {"saved", "not_saved", "failed"}
            ):
                raise CapturePersistenceError(f"motion event {index + 1} has an invalid keyframe schema")
            if frame.get("status") == "saved" and not isinstance(frame.get("relative_path"), str):
                raise CapturePersistenceError(f"motion keyframe {event_id} has no saved file reference")
            keyframes.append(dict(frame))
        elif kind == "capture_finish":
            event_count = item.get("event_count")
            if (
                not isinstance(item.get("reason"), str)
                or isinstance(event_count, bool)
                or not isinstance(event_count, int)
                or event_count < 0
                or event_count != len(closed_ids)
            ):
                raise CapturePersistenceError("motion capture_finish record has an invalid schema or event count")
            finish_record = item
        else:
            raise CapturePersistenceError(f"motion event {index + 1} has an unknown update kind")
        valid_rows += 1
    interrupted: list[dict[str, Any]] = []
    rebuilt_events: list[dict[str, Any]] = []
    for event_id, event in latest_events.items():
        if event_id in active_ids:
            censored = dict(event)
            censored.update({
                "end": None,
                "end_sample": None,
                "end_reason": "interrupted",
                "status": "interrupted_censored",
                "censored": True,
            })
            interrupted.append(censored)
            rebuilt_events.append(censored)
        else:
            rebuilt_events.append(dict(event))
    return {
        "status": "interrupted" if interrupted or truncated_tail or finish_record is None else "complete_event_log",
        "events": rebuilt_events,
        "open_events": interrupted,
        "keyframes": keyframes,
        "capture_finished": finish_record is not None,
        "event_log": motion_path.relative_to(output_root).as_posix(),
        "event_log_sha256": _sha256(motion_path),
        "event_log_size_bytes": len(payload),
        "parsed_event_records": valid_rows,
        "truncated_terminal_records": truncated_tail,
        "truncated_terminal_bytes": truncated_tail_bytes,
    }


def cleanup_capture_sidecars(
    context_path: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] | None = None,
    completion_verified: bool,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    supplied_context = Path(context_path).expanduser()
    raw_context = supplied_context
    sidecar_metadata = (
        expected_metadata.get("capture_persistence")
        if isinstance(expected_metadata, Mapping)
        else None
    )
    if not completion_verified:
        return {
            "status": "retained",
            "reason": "completion_not_verified",
            "retained_files": [str(raw_context)],
        }
    if not isinstance(sidecar_metadata, Mapping):
        return {
            "status": "retained",
            "reason": "snapshot_missing_persistence_metadata",
            "retained_files": [str(raw_context)],
        }
    session = sidecar_metadata.get("session_id")
    capture_id = sidecar_metadata.get("capture_id")
    attempt_id = sidecar_metadata.get("attempt_id")
    expected_context_name = f"{session}.{capture_id}.capture-context.json"
    expected_checkpoint_name = f"{session}.{capture_id}.capture-checkpoints.jsonl"
    try:
        layout = normalize_sidecar_layout(sidecar_metadata.get("sidecar_layout"))
    except ValueError as exc:
        return {"status": "retained", "reason": str(exc), "retained_files": [str(raw_context)]}
    raw_root: Path | None
    path_error = None
    try:
        if output_root is None:
            if not supplied_context.is_absolute():
                raise CapturePersistenceError("output_root is required for a relative context path")
            parent = raw_context.parent
            raw_root = parent.parent if parent.name == ".internal" else parent
            raw_root = raw_root.resolve(strict=False)
        else:
            supplied_root = Path(output_root).expanduser()
            if supplied_root.is_symlink():
                raise CapturePersistenceError("capture output root is a symbolic link")
            raw_root = supplied_root.resolve(strict=True)
            if not raw_root.is_dir():
                raise CapturePersistenceError("capture output root is not a directory")
        expected_context_path = sidecar_path(raw_root, expected_context_name, layout=layout)
        raw_context = owned_sidecar_path(
            raw_root,
            supplied_context,
            name=expected_context_name,
            layout=layout,
        )
        if raw_context != expected_context_path:
            raise CapturePersistenceError("selected context path does not match the declared sidecar layout")
    except (OSError, RuntimeError, CapturePersistenceError):
        raw_root = None
    except (TypeError, ValueError) as exc:
        raw_root = None
        path_error = str(exc)
    else:
        path_error = None
    if (
        not isinstance(session, str) or not _SAFE_TOKEN.fullmatch(session)
        or not isinstance(capture_id, str) or not re.fullmatch(r"[0-9a-f]{32}", capture_id)
        or not isinstance(attempt_id, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt_id)
        or raw_context.name != expected_context_name
        or raw_root is None
        or not _metadata_direct_child_matches(raw_root, sidecar_metadata.get("context_path"), expected_context_name, layout=layout)
        or not _metadata_direct_child_matches(raw_root, sidecar_metadata.get("checkpoint_path"), expected_checkpoint_name, layout=layout)
        or sidecar_metadata.get("sidecar_layout", LEGACY_LAYOUT) != layout
    ):
        return {
            "status": "retained",
            "reason": path_error or "snapshot_sidecar_identity_mismatch",
            "retained_files": [str(raw_context)],
        }

    checkpoint_owned_path = sidecar_path(raw_root, expected_checkpoint_name, layout=layout)

    if raw_context.is_symlink():
        retained = [str(raw_context)]
        checkpoint_candidate = checkpoint_owned_path
        if checkpoint_candidate.exists() or checkpoint_candidate.is_symlink():
            retained.append(str(checkpoint_candidate))
        return {
            "status": "retained",
            "reason": "capture context is a symbolic link",
            "retained_files": retained,
        }
    if not raw_context.exists():
        try:
            context_sha = sidecar_metadata.get("context_sha256")
            checkpoint_sha = sidecar_metadata.get("checkpoint_sha256")
            if (
                not isinstance(context_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", context_sha)
                or not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha)
            ):
                raise CapturePersistenceError("snapshot sidecar hash is missing or invalid")
            checkpoint_path = _direct_child(
                raw_root,
                sidecar_relative_path(expected_checkpoint_name, layout),
                suffix=".capture-checkpoints.jsonl",
            )
            if checkpoint_path.exists() or checkpoint_path.is_symlink():
                if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
                    raise CapturePersistenceError("checkpoint is missing or replaced")
                if _sha256(checkpoint_path) != checkpoint_sha:
                    raise CapturePersistenceError("checkpoint content changed after snapshot")
                checkpoint_path.unlink()
                if (
                    raw_context.exists() or raw_context.is_symlink()
                    or checkpoint_path.exists() or checkpoint_path.is_symlink()
                ):
                    raise CapturePersistenceError("capture sidecars still exist after cleanup")
                return {
                    "status": "completed",
                    "deleted": [str(checkpoint_path)],
                    "already_absent": [str(raw_context)],
                    "retained_files": [],
                }
            if (
                raw_context.exists() or raw_context.is_symlink()
                or checkpoint_path.exists() or checkpoint_path.is_symlink()
            ):
                raise CapturePersistenceError("capture sidecars reappeared during cleanup")
            return {
                "status": "completed",
                "deleted": [],
                "already_absent": [str(raw_context), str(checkpoint_path)],
                "retained_files": [],
            }
        except Exception as exc:
            retained = [str(raw_context)]
            checkpoint_candidate = checkpoint_owned_path
            if checkpoint_candidate.exists() or checkpoint_candidate.is_symlink():
                retained.append(str(checkpoint_candidate))
            return {
                "status": "retained",
                "reason": f"{type(exc).__name__}: {exc}",
                "retained_files": retained,
            }
    try:
        context, _ = load_capture_context(raw_context)
        if (
            context.get("session_id") != session
            or context.get("capture_id") != capture_id
            or context.get("attempt_id") != attempt_id
        ):
            raise CapturePersistenceError("capture context identity does not match snapshot")
        writer = CapturePersistence(raw_context, context=context)
    except Exception as exc:
        retained = [str(raw_context)]
        checkpoint_candidate = checkpoint_owned_path
        if checkpoint_candidate.exists() or checkpoint_candidate.is_symlink():
            retained.append(str(checkpoint_candidate))
        return {"status": "retained", "reason": f"{type(exc).__name__}: {exc}", "retained_files": retained}
    result = writer.cleanup_recovery_sidecars(
        completion_verified=completion_verified,
        expected_metadata=expected_metadata,
    )
    if result.get("status") == "completed":
        remaining = [
            str(path)
            for path in (writer.context_path, writer.checkpoint_path)
            if path.exists() or path.is_symlink()
        ]
        if remaining:
            result = dict(result)
            result["status"] = "partial"
            result["retained_files"] = [
                *result.get("retained_files", []),
                *({"path": path, "reason": "sidecar still exists after cleanup"} for path in remaining),
            ]
    return result


__all__ = [
    "CapturePersistence",
    "CapturePersistenceError",
    "CONTEXT_SCHEMA",
    "create_capture_context",
    "cleanup_capture_sidecars",
    "load_capture_context",
    "marker_key_for_keysym",
    "motion_recovery_summary",
    "read_capture_tsv",
    "read_checkpoints",
    "read_event_jsonl",
    "user_marker_stream_sha256",
    "user_markers_from_events",
    "validate_user_marker_watermarks",
]
