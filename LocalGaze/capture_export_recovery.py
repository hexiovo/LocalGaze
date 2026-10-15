"""Durable capture snapshots and exclusive, retryable workbook publication.

This module deliberately has no camera, UI, pandas, or pipeline dependency.  A
caller supplies the small writer callback that turns a validated JSON payload
into an XLSX workbook and manifest inside an attempt-owned staging directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Mapping

from session_names import canonical_session_id
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


SNAPSHOT_SCHEMA = "localgaze.capture-export-snapshot/1"
_OWNER_SCHEMA = "localgaze.capture-export-owner/1"
_STAGING_SCHEMA = "localgaze.capture-export-staging/1"
_COMPLETE_SCHEMA = "localgaze.capture-export-complete/1"
_FLOAT_TAG = "$localgaze_snapshot_float"
_ATTEMPT_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MARKER_NS_STRING_KEYS = frozenset({
    "monotonic_ns",
    "last_monotonic_ns",
    "user_marker_last_monotonic_ns",
})

SnapshotPayload = dict[str, Any]
ExportResult = Mapping[str, Mapping[str, str]]
OutputWriter = Callable[[Path, SnapshotPayload], ExportResult]


class SnapshotError(ValueError):
    """The capture data cannot be represented as a safe, strict JSON snapshot."""


class RecoveryConflict(RuntimeError):
    """An existing path does not match this attempt's ownership and content."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ns_key(key: Any) -> bool:
    return isinstance(key, str) and (
        key == "received_ns" or key.endswith("_ns") or "monotonic_ns" in key
    )


def _json_safe(value: Any, *, key: Any = None, depth: int = 0) -> Any:
    """Convert capture values to strict JSON without flattening image arrays."""

    if depth > 64:
        raise SnapshotError("snapshot value nesting exceeds 64 levels")
    if value is None or isinstance(value, (str, bool)):
        if _ns_key(key) and isinstance(value, str):
            if not re.fullmatch(r"-?\d+", value):
                raise SnapshotError(f"{key} must be an integer nanosecond value")
            return str(int(value))
        return value

    value_type = type(value)
    module_name = getattr(value_type, "__module__", "").split(".", 1)[0]
    if module_name == "numpy":
        dimensions = getattr(value, "ndim", None)
        if dimensions is not None and int(dimensions) > 0:
            raise SnapshotError("snapshot does not store NumPy arrays or image pixels")
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return _json_safe(item(), key=key, depth=depth + 1)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SnapshotError(f"unsupported NumPy scalar at {key!r}") from exc

    if isinstance(value, int):
        return str(value) if _ns_key(key) else value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        label = "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
        return {_FLOAT_TAG: label}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise SnapshotError("snapshot object keys must be strings")
            if raw_key == _FLOAT_TAG:
                raise SnapshotError(f"input uses reserved snapshot key {_FLOAT_TAG!r}")
            result[raw_key] = _json_safe(item, key=raw_key, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value]

    # Many array/tensor types are not NumPy subclasses.  Refuse array-like
    # objects rather than accidentally calling tolist() on a camera image.
    if (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and callable(getattr(value, "tolist", None))
    ):
        dimensions = getattr(value, "ndim", None)
        if dimensions is None or int(dimensions) > 0:
            raise SnapshotError("snapshot does not store array or image pixels")

    raise SnapshotError(f"unsupported snapshot value at {key!r}: {value_type.__name__}")


def _from_json_safe(value: Any, *, key: Any = None) -> Any:
    if isinstance(value, list):
        return [_from_json_safe(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            token = value[_FLOAT_TAG]
            if token == "nan":
                return float("nan")
            if token == "inf":
                return float("inf")
            if token == "-inf":
                return float("-inf")
            raise SnapshotError("snapshot contains an invalid tagged float")
        return {name: _from_json_safe(item, key=name) for name, item in value.items()}
    # Motion-event samples and user-marker documents/watermarks keep
    # decimal-string timestamps in their schemas.
    if (
        key not in {"received_ns", *_MARKER_NS_STRING_KEYS}
        and _ns_key(key)
        and isinstance(value, str)
        and re.fullmatch(r"-?\d+", value)
    ):
        return int(value)
    return value


def _restore_record_order(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore the original row-column order recorded beside canonical JSON."""

    columns = payload.get("record_columns")
    records = payload.get("records")
    if not isinstance(columns, list) or not isinstance(records, list):
        raise RecoveryConflict("snapshot gaze columns or records are invalid")
    if any(not isinstance(column, str) for column in columns):
        raise RecoveryConflict("snapshot gaze column name is invalid")
    reordered: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RecoveryConflict("snapshot gaze row is not an object")
        ordered = {column: record[column] for column in columns if column in record}
        ordered.update((key, value) for key, value in record.items() if key not in ordered)
        reordered.append(ordered)
    payload["records"] = reordered
    return payload


def _output_root(value: str | os.PathLike[str]) -> Path:
    root = assert_no_reparse_components(Path(value).expanduser())
    if not root.exists() or not root.is_dir():
        raise ValueError("output_root must be an existing directory")
    return root.resolve(strict=True)


def _declared_layout(capture_metadata: Any) -> str:
    if not isinstance(capture_metadata, Mapping):
        return LEGACY_LAYOUT
    persistence = capture_metadata.get("capture_persistence")
    declared_values = (
        capture_metadata.get("sidecar_layout"),
        persistence.get("sidecar_layout") if isinstance(persistence, Mapping) else None,
    )
    if any(value is not None and not isinstance(value, str) for value in declared_values):
        raise SnapshotError("capture sidecar_layout must be a string")
    try:
        layouts = [
            normalize_sidecar_layout(value)
            for value in declared_values
            if value is not None
        ]
    except ValueError as exc:
        raise SnapshotError("capture sidecar_layout is invalid") from exc
    if len(set(layouts)) > 1:
        raise SnapshotError("capture sidecar_layout declarations conflict")
    return layouts[0] if layouts else LEGACY_LAYOUT


def _expected_snapshot_path(
    root: Path,
    session: str,
    layout: str,
    *,
    create_parent: bool = False,
) -> Path:
    return sidecar_path(
        root,
        f"{session}.capture-snapshot.json",
        layout=layout,
        create_parent=create_parent,
    )


def _relative_sidecar(root: Path, name: str, layout: str) -> str:
    return sidecar_path(root, name, layout=layout).relative_to(root).as_posix()


def _confined_path(value: str | os.PathLike[str], root: Path, *, relative_to_root: bool) -> Path:
    raw_path = Path(value).expanduser()
    if relative_to_root and not raw_path.is_absolute():
        path = output_child_path(root, os.fspath(raw_path).replace("\\", "/"))
    else:
        path = assert_no_reparse_components(raw_path)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("snapshot and recovery paths must remain inside output_root") from exc
    return resolved


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        path = assert_no_reparse_components(path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict(f"JSON path contains a symbolic link or junction: {path}") from exc
    if not path.is_file():
        raise RecoveryConflict(f"expected a regular owned JSON file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryConflict(f"invalid JSON at {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryConflict(f"JSON object expected at {path}")
    return value, raw


def _same_identity(path: Path, identity: os.stat_result) -> bool:
    try:
        current = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return current.st_dev == identity.st_dev and current.st_ino == identity.st_ino


def _write_exclusive_atomic(path: Path, content: bytes) -> None:
    """Atomically publish a new file without replacing an existing path."""

    path = assert_no_reparse_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = assert_no_reparse_components(path)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    assert_no_reparse_components(temp)
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    identity: os.stat_result | None = None
    try:
        identity = os.fstat(descriptor)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if not _same_identity(temp, identity):
            raise RecoveryConflict("exclusive temporary file identity changed")
        # A hard link is an atomic create-if-absent operation on the same
        # volume.  os.replace is intentionally never used for final paths.
        os.link(temp, path)
        if not os.path.samefile(temp, path):
            raise RecoveryConflict("published file identity differs from its staging file")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if identity is not None and _same_identity(temp, identity):
            try:
                temp.unlink()
            except OSError:
                pass


def _file_reference(
    root: Path,
    relative: str,
    role: str,
    *,
    hash_content: bool = False,
    expected_type: str = "file",
) -> dict[str, Any]:
    if not isinstance(relative, str) or not relative:
        return {"role": role, "relative_path": None, "status": "unavailable"}
    win_path = PureWindowsPath(relative)
    parts = relative.replace("\\", "/").split("/")
    if win_path.is_absolute() or win_path.drive or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        return {"role": role, "relative_path": relative, "status": "unsafe_reference"}
    try:
        path = output_child_path(root, relative)
    except (OSError, RuntimeError, TypeError, ValueError):
        try:
            assert_no_reparse_components(root.joinpath(*parts))
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"role": role, "relative_path": relative, "status": "symlink_reference"}
        return {"role": role, "relative_path": relative, "status": "unsafe_reference"}
    if expected_type == "directory":
        if not path.is_dir():
            return {"role": role, "relative_path": relative, "status": "missing"}
        return {
            "role": role,
            "relative_path": path.relative_to(root).as_posix(),
            "status": "present",
            "reference_type": "directory",
        }
    if not path.is_file():
        return {"role": role, "relative_path": relative, "status": "missing"}
    result: dict[str, Any] = {
        "role": role,
        "relative_path": path.relative_to(root).as_posix(),
        "status": "present",
        "size_bytes": path.stat().st_size,
    }
    if hash_content:
        result["sha256"] = _sha256_file(path)
    return result


def _normalize_capture_metadata_path(
    root: Path,
    value: Any,
    expected_relative: str,
    *,
    role: str,
    layout: str = LEGACY_LAYOUT,
) -> str:
    """Accept only the exact declared root-level or .internal sidecar path."""
    if not isinstance(value, str) or not value or not expected_relative:
        raise RecoveryConflict(f"capture {role} path identity is invalid")
    try:
        leaf = expected_relative.rsplit("/", 1)[-1]
        expected = sidecar_path(root, leaf, layout=layout)
        if expected.relative_to(root).as_posix() != expected_relative:
            raise ValueError("expected path does not match sidecar layout")
        owned_sidecar_path(root, value, name=leaf, layout=layout)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict(f"capture {role} path does not match its output-root identity") from exc
    return expected_relative


def _capture_file_references(root: Path, session_id: str, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    persistence = metadata.get("capture_persistence")
    layout = _declared_layout(metadata)
    table_format = (
        persistence.get("table_format")
        if isinstance(persistence, Mapping)
        else metadata.get("table_format")
    )
    if table_format is None:
        table_format = "txt" if layout == LEGACY_LAYOUT else "csv"
    if table_format not in {"csv", "txt"}:
        raise SnapshotError("capture table_format is invalid")
    sample_name = f"{session_id}.{table_format}"
    if isinstance(persistence, Mapping) and persistence.get("raw_path") is not None:
        if persistence.get("raw_path") != sample_name:
            raise SnapshotError("capture raw_path does not match its table_format")
    references = [_file_reference(root, sample_name, "samples_tsv", hash_content=True)]
    if isinstance(persistence, Mapping):
        capture_id = metadata.get("capture_id")
        owns_capture = (
            isinstance(capture_id, str)
            and _ATTEMPT_RE.fullmatch(capture_id) is not None
            and persistence.get("capture_id") == capture_id
        )
        events_path = persistence.get("events_path")
        if isinstance(events_path, str) and owns_capture:
            events_path = _normalize_capture_metadata_path(
                root,
                events_path,
                sidecar_relative_path(f"{session_id}.{capture_id}.capture-events.jsonl", layout),
                role="event log",
                layout=layout,
            )
        references.append(
            _file_reference(root, events_path, "capture_events", hash_content=True)
            if isinstance(events_path, str)
            else {"role": "capture_events", "relative_path": None, "status": "unavailable"}
        )
        task_log_path = persistence.get("capture_task_log_path")
        job_id = persistence.get("job_id") or capture_id
        if (
            isinstance(task_log_path, str)
            and owns_capture
            and isinstance(job_id, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", job_id)
        ):
            task_log_path = _normalize_capture_metadata_path(
                root,
                task_log_path,
                sidecar_relative_path(f"{session_id}.{job_id}.capture-job.jsonl", layout),
                role="task log",
                layout=layout,
            )
        task_reference = (
            _file_reference(root, task_log_path, "capture_task_log")
            if isinstance(task_log_path, str)
            else {"role": "capture_task_log", "relative_path": None, "status": "unavailable"}
        )
        # Job receipts may still be appended by the parent after the snapshot;
        # bind their owned path without hashing live content.
        task_reference.pop("size_bytes", None)
        task_reference.pop("sha256", None)
        references.append(task_reference)

    motion = metadata.get("motion_monitor")
    result = motion.get("result") if isinstance(motion, Mapping) else None
    artifacts = result.get("artifacts") if isinstance(result, Mapping) else None
    recovery = result.get("recovery") if isinstance(result, Mapping) else None
    event_log = artifacts.get("event_log") if isinstance(artifacts, Mapping) else None
    if not isinstance(event_log, str) and isinstance(recovery, Mapping):
        event_log = recovery.get("event_log")
    if isinstance(event_log, str):
        references.append(
            _file_reference(root, event_log, "motion_event_log", hash_content=True)
        )
    if isinstance(artifacts, Mapping):
        for name in ("capture_directory", "keyframe_directory"):
            value = artifacts.get(name)
            if isinstance(value, str):
                references.append(
                    _file_reference(
                        root,
                        value,
                        f"motion_{name}",
                        expected_type="directory",
                    )
                )
    keyframes = result.get("keyframes") if isinstance(result, Mapping) else None
    if isinstance(keyframes, list):
        has_keyframe_directory = isinstance(artifacts, Mapping) and isinstance(artifacts.get("keyframe_directory"), str)
        for index, item in enumerate(keyframes):
            if isinstance(item, Mapping) and item.get("status") == "saved":
                relative_path = item.get("relative_path")
                references.append(
                    _file_reference(root, relative_path, f"motion_keyframe_{index}", hash_content=True)
                    if isinstance(relative_path, str)
                    else {"role": f"motion_keyframe_{index}", "relative_path": None, "status": "unavailable"}
                )
                if not has_keyframe_directory and isinstance(relative_path, str):
                    keyframe_directory = Path(relative_path).parent.as_posix()
                    references.append(
                        _file_reference(
                            root,
                            keyframe_directory,
                            "motion_keyframe_directory",
                            expected_type="directory",
                        )
                    )
                    has_keyframe_directory = True
    return references


def _normalize_capture_payload(
    *,
    session_id: str,
    root: Path,
    records: Any,
    experiment_info: Any,
    capture_metadata: Any,
    model_metadata: Any,
) -> dict[str, Any]:
    if not isinstance(records, (list, tuple)):
        raise SnapshotError("records must be a list or tuple of mappings")
    if not isinstance(experiment_info, Mapping):
        raise SnapshotError("experiment_info must be a mapping")
    if not isinstance(capture_metadata, Mapping):
        raise SnapshotError("capture_metadata must be a mapping")
    if model_metadata is not None and not isinstance(model_metadata, Mapping):
        raise SnapshotError("model_metadata must be a mapping or None")
    metadata_session = capture_metadata.get("session_id")
    if metadata_session is not None and metadata_session != session_id:
        raise SnapshotError("capture metadata session_id does not match session_id")
    metadata_root = capture_metadata.get("output_dir")
    if metadata_root is not None:
        try:
            recorded_root = Path(str(metadata_root)).expanduser().resolve(strict=False)
        except (TypeError, ValueError, OSError) as exc:
            raise SnapshotError("capture metadata output_dir is invalid") from exc
        if recorded_root != root:
            raise SnapshotError("capture metadata output_dir does not match output_root")

    normalized_records: list[dict[str, Any]] = []
    columns: list[str] = []
    seen_columns: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SnapshotError(f"records[{index}] must be a mapping")
        normalized_record = _json_safe(record)
        if not isinstance(normalized_record, dict):
            raise SnapshotError(f"records[{index}] is not a JSON object")
        normalized_records.append(normalized_record)
        for column in record:
            if not isinstance(column, str):
                raise SnapshotError("gaze record columns must be strings")
            if column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)

    metadata_json = _json_safe(capture_metadata)
    if not isinstance(metadata_json, dict):
        raise SnapshotError("capture_metadata is not a JSON object")
    references = _capture_file_references(root, session_id, capture_metadata)
    samples_ref = references[0]
    if samples_ref["status"] in {"unsafe_reference", "symlink_reference"}:
        raise SnapshotError("samples TSV reference is unsafe")

    model_contract = (
        {key: value for key, value in model_metadata.items() if key != "training"}
        if isinstance(model_metadata, Mapping)
        else None
    )
    return {
        "session_id": session_id,
        "output_root": str(root),
        "record_columns": columns,
        "records": normalized_records,
        "experiment_info": _json_safe(experiment_info),
        "capture_metadata": metadata_json,
        # The saver intentionally excludes training arrays from the workbook
        # contract; omitting them here avoids duplicating large model state.
        "model_metadata": _json_safe(model_contract),
        "file_references": references,
    }


def _snapshot_response(
    status: str,
    *,
    snapshot: Path | None = None,
    attempt_id: str | None = None,
    snapshot_sha256: str | None = None,
    recoverable: bool,
    reason: str | None = None,
    error: str | None = None,
    file_references: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    output_root: Path | None = None,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    outputs = {"snapshot": str(snapshot)} if snapshot is not None else {}
    if file_references:
        samples = next((item for item in file_references if item.get("role") == "samples_tsv"), None)
        if samples and samples.get("status") == "present":
            outputs["samples"] = str((output_root or snapshot.parent) / samples["relative_path"])
    details: dict[str, Any] = {
        "snapshot_saved": status == "prepared" and snapshot is not None,
        "recoverable": recoverable,
        "retryable": recoverable,
        "reason": reason,
        "attempt_id": attempt_id,
        "snapshot_sha256": snapshot_sha256,
        "session_id": session_id,
        "output_root": str(output_root) if output_root is not None else None,
        "snapshot_path": str(snapshot_path or snapshot) if (snapshot_path or snapshot) is not None else None,
        "file_references": file_references or [],
    }
    if snapshot is not None:
        details["staging_path"] = str(output_root or snapshot.parent)
        if attempt_id and session_id:
            details["attempt_staging_path"] = str(
                (internal_directory(output_root, create=True) if output_root is not None and snapshot.parent.name == ".internal" else output_root or snapshot.parent)
                / f".{session_id}.capture-export-{attempt_id}"
            )
    elif output_root is not None:
        details["staging_path"] = str(output_root)
    if error:
        details["error"] = error
    return {"status": status, "outputs": outputs, "details": details}


def persist_capture_snapshot(
    snapshot_path: Path,
    *,
    session_id: str,
    output_root: str | os.PathLike[str],
    records: Any,
    experiment_info: Any,
    capture_metadata: Any,
    model_metadata: Any = None,
) -> dict[str, Any]:
    """Persist a strict, image-free snapshot before attempting workbook export.

    A successful return has ``status='prepared'``; this is deliberately not an
    export success.  Only :func:`recover_capture_export` can report completed.
    Existing snapshot paths are reused only when the validated payload matches
    byte-for-byte after canonical JSON normalization.
    """

    snapshot: Path | None = None
    root: Path | None = None
    session: str | None = None
    references: list[dict[str, Any]] | None = None
    try:
        session = canonical_session_id(session_id)
        root = _output_root(output_root)
        snapshot = _confined_path(snapshot_path, root, relative_to_root=True)
        layout = _declared_layout(capture_metadata)
        expected_snapshot = _expected_snapshot_path(
            root,
            session,
            layout,
            create_parent=layout == INTERNAL_LAYOUT,
        )
        if snapshot != expected_snapshot:
            raise SnapshotError("snapshot_path does not match the declared sidecar layout")
        alternate_snapshot = sidecar_path(
            root,
            f"{session}.capture-snapshot.json",
            layout=LEGACY_LAYOUT if layout == INTERNAL_LAYOUT else INTERNAL_LAYOUT,
            create_parent=False,
        )
        if alternate_snapshot.exists() or alternate_snapshot.is_symlink():
            raise SnapshotError("a conflicting capture snapshot exists in the alternate layout")
        if snapshot == root or snapshot.name in {
            f"{session}.txt",
            f"{session}.csv",
            f"{session}.xlsx",
            f"{session}.manifest.json",
            f".{session}.capture-export-owner.json",
            f".{session}.capture-export-complete.json",
        }:
            raise SnapshotError("snapshot_path collides with a reserved session output")
        payload = _normalize_capture_payload(
            session_id=session,
            root=root,
            records=records,
            experiment_info=experiment_info,
            capture_metadata=capture_metadata,
            model_metadata=model_metadata,
        )
        references = payload["file_references"]

        if snapshot.exists():
            existing, raw = _read_json(snapshot)
            _validate_snapshot_envelope(existing)
            existing_payload = existing["payload"]
            if (
                existing_payload == payload
                and existing.get("session_id") == session
                and existing.get("output_root") == str(root)
            ):
                return _snapshot_response(
                    "prepared",
                    snapshot=snapshot,
                    attempt_id=existing["attempt_id"],
                    snapshot_sha256=_sha256_bytes(raw),
                    recoverable=True,
                    reason="existing_snapshot_matches",
                    file_references=references,
                    session_id=session,
                    output_root=root,
                )
            raise RecoveryConflict("snapshot path exists with different or unknown content")

        attempt_id = uuid.uuid4().hex
        envelope = {
            "schema": SNAPSHOT_SCHEMA,
            "session_id": session,
            "output_root": str(root),
            "attempt_id": attempt_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
            "payload_sha256": _sha256_bytes(_json_bytes(payload)),
        }
        content = _json_bytes(envelope) + b"\n"
        _write_exclusive_atomic(snapshot, content)
        return _snapshot_response(
            "prepared",
            snapshot=snapshot,
            attempt_id=attempt_id,
            snapshot_sha256=_sha256_bytes(content),
            recoverable=True,
            file_references=references,
            session_id=session,
            output_root=root,
        )
    except Exception as exc:
        return _snapshot_response(
            "failed",
            recoverable=False,
            reason="snapshot_write_failed",
            error=f"{type(exc).__name__}: {exc}",
            file_references=references,
            session_id=session,
            output_root=root,
            snapshot_path=snapshot,
        )


def _validate_snapshot_envelope(envelope: Mapping[str, Any]) -> None:
    if envelope.get("schema") != SNAPSHOT_SCHEMA:
        raise RecoveryConflict("snapshot schema is unsupported")
    try:
        session = canonical_session_id(envelope.get("session_id"))
    except (TypeError, ValueError) as exc:
        raise RecoveryConflict("snapshot session_id is invalid") from exc
    attempt_id = envelope.get("attempt_id")
    if not isinstance(attempt_id, str) or not _ATTEMPT_RE.fullmatch(attempt_id):
        raise RecoveryConflict("snapshot attempt_id is invalid")
    if not isinstance(envelope.get("output_root"), str):
        raise RecoveryConflict("snapshot output_root is invalid")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RecoveryConflict("snapshot payload is missing")
    if payload.get("session_id") != session or payload.get("output_root") != envelope.get("output_root"):
        raise RecoveryConflict("snapshot identity does not match its payload")
    digest = envelope.get("payload_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise RecoveryConflict("snapshot payload hash is invalid")
    try:
        actual = _sha256_bytes(_json_bytes(payload))
    except (TypeError, ValueError) as exc:
        raise RecoveryConflict("snapshot payload is not strict JSON") from exc
    if actual != digest:
        raise RecoveryConflict("snapshot payload hash mismatch")


def _load_snapshot(snapshot_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path, bytes]:
    try:
        snapshot_path = assert_no_reparse_components(snapshot_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict("snapshot path contains a symbolic link or junction") from exc
    if not snapshot_path.is_file():
        raise RecoveryConflict("snapshot is missing or is not a regular file")
    envelope, raw = _read_json(snapshot_path)
    _validate_snapshot_envelope(envelope)
    root = _output_root(envelope["output_root"])
    resolved_snapshot = _confined_path(snapshot_path, root, relative_to_root=True)
    if resolved_snapshot != snapshot_path.resolve(strict=True):
        raise RecoveryConflict("snapshot path changed during validation")
    payload = _from_json_safe(envelope["payload"])
    if not isinstance(payload, dict):
        raise RecoveryConflict("decoded snapshot payload is not an object")
    layout = _declared_layout(payload.get("capture_metadata"))
    try:
        expected_snapshot = _expected_snapshot_path(root, envelope["session_id"], layout)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict("snapshot layout path is unsafe") from exc
    if resolved_snapshot != expected_snapshot:
        raise RecoveryConflict("snapshot path does not match its declared sidecar layout")
    return envelope, _restore_record_order(payload), root, raw


def _artifact_paths(root: Path, session: str, layout: str) -> dict[str, Path]:
    layout = normalize_sidecar_layout(layout)
    manifest_name = f"{session}.manifest.json"
    owner_name = f".{session}.capture-export-owner.json"
    complete_name = f".{session}.capture-export-complete.json"

    def exact_sidecar(name: str) -> Path:
        alternate = sidecar_path(
            root,
            name,
            layout=LEGACY_LAYOUT if layout == INTERNAL_LAYOUT else INTERNAL_LAYOUT,
        )
        if alternate.exists() or alternate.is_symlink():
            raise RecoveryConflict(f"conflicting sidecar exists in the alternate layout: {name}")
        return sidecar_path(
            root,
            name,
            layout=layout,
            create_parent=layout == INTERNAL_LAYOUT,
        )

    return {
        "xlsx": output_child_path(root, f"{session}.xlsx"),
        "manifest": exact_sidecar(manifest_name),
        "owner": exact_sidecar(owner_name),
        "completion_marker": exact_sidecar(complete_name),
    }


def _expected_final_artifacts(session: str, layout: str) -> dict[str, str]:
    layout = normalize_sidecar_layout(layout)
    return {
        "xlsx": f"{session}.xlsx",
        "manifest": sidecar_relative_path(f"{session}.manifest.json", layout),
    }


def _writer_artifact_names(session: str) -> dict[str, str]:
    return {"xlsx": f"{session}.xlsx", "manifest": f"{session}.manifest.json"}


def _validate_stage_base(staging_path: Path, root: Path, layout: str) -> Path:
    try:
        raw_path = assert_no_reparse_components(Path(staging_path).expanduser())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict("staging base contains a symbolic link or junction") from exc
    supplied = raw_path.resolve(strict=False)
    expected = internal_directory(root, create=True) if layout == INTERNAL_LAYOUT else root
    if supplied != expected:
        raise RecoveryConflict("staging_path must match the declared sidecar layout")
    if not supplied.is_dir():
        raise RecoveryConflict("staging base must be an existing regular directory")
    return supplied


def _safe_child(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RecoveryConflict("staging ownership contains an invalid relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.name != relative or relative in {".", ".."}:
        raise RecoveryConflict("staging ownership path is not a direct child")
    lexical_child = root / candidate
    try:
        child = assert_no_reparse_components(lexical_child)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict("staging or output child contains a symbolic link or junction") from exc
    if child.parent != root:
        raise RecoveryConflict("staging ownership path escaped its directory")
    return child


def _safe_reference_child(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RecoveryConflict("snapshot artifact reference is not a safe relative path")
    win_path = PureWindowsPath(relative)
    parts = relative.split("/")
    if (
        win_path.is_absolute()
        or win_path.drive
        or Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RecoveryConflict("snapshot artifact reference escapes its output root")
    try:
        return output_child_path(root, relative)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict("snapshot artifact reference contains a link or escapes its output root") from exc


def _verify_file_references(payload: Mapping[str, Any], root: Path) -> dict[str, Path]:
    references = payload.get("file_references")
    if not isinstance(references, list):
        raise RecoveryConflict("snapshot file_references are missing")
    session = payload.get("session_id")
    metadata = payload.get("capture_metadata")
    if not isinstance(session, str) or not isinstance(metadata, Mapping):
        raise RecoveryConflict("snapshot capture identity or metadata is invalid")
    capture_id = metadata.get("capture_id")
    layout = _declared_layout(metadata)
    persistence = metadata.get("capture_persistence")
    table_format = (
        persistence.get("table_format")
        if isinstance(persistence, Mapping)
        else metadata.get("table_format")
    )
    if table_format is None:
        table_format = "txt" if layout == LEGACY_LAYOUT else "csv"
    if table_format not in {"csv", "txt"}:
        raise RecoveryConflict("capture table_format is invalid")
    expected_paths: dict[str, str] = {"samples_tsv": f"{session}.{table_format}"}
    directory_roles: set[str] = set()
    path_only_roles: set[str] = set()
    expected_metadata_hashes: dict[str, tuple[str, int]] = {}

    if isinstance(persistence, Mapping):
        if not isinstance(capture_id, str) or not _ATTEMPT_RE.fullmatch(capture_id):
            raise RecoveryConflict("capture persistence has an invalid capture identity")
        if persistence.get("capture_id") != capture_id:
            raise RecoveryConflict("capture persistence belongs to another capture")
        if persistence.get("session_id") is not None and persistence.get("session_id") != session:
            raise RecoveryConflict("capture persistence belongs to another session")
        context_name = f"{session}.{capture_id}.capture-context.json"
        checkpoint_name = f"{session}.{capture_id}.capture-checkpoints.jsonl"
        event_name = f"{session}.{capture_id}.capture-events.jsonl"
        task_job_id = persistence.get("job_id") or capture_id
        if not isinstance(task_job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", task_job_id):
            raise RecoveryConflict("capture persistence has an invalid job identity")
        task_name = f"{session}.{task_job_id}.capture-job.jsonl"
        _normalize_capture_metadata_path(
            root, persistence.get("events_path"),
            sidecar_relative_path(event_name, layout), role="event log", layout=layout,
        )
        _normalize_capture_metadata_path(
            root, persistence.get("capture_task_log_path"),
            sidecar_relative_path(task_name, layout), role="task log", layout=layout,
        )
        if "context_path" in persistence:
            _normalize_capture_metadata_path(
                root, persistence.get("context_path"),
                sidecar_relative_path(context_name, layout), role="context", layout=layout,
            )
        if "checkpoint_path" in persistence:
            _normalize_capture_metadata_path(
                root, persistence.get("checkpoint_path"),
                sidecar_relative_path(checkpoint_name, layout), role="checkpoint", layout=layout,
            )
        event_sha = persistence.get("events_sha256")
        event_size = persistence.get("events_size_bytes")
        if not isinstance(event_sha, str) or not _SHA256_RE.fullmatch(event_sha):
            raise RecoveryConflict("capture event log snapshot hash is invalid")
        if isinstance(event_size, bool) or not isinstance(event_size, int) or event_size < 0:
            raise RecoveryConflict("capture event log snapshot size is invalid")
        expected_paths.update({
            "capture_events": sidecar_relative_path(event_name, layout),
            "capture_task_log": sidecar_relative_path(task_name, layout),
        })
        expected_metadata_hashes["capture_events"] = (event_sha, event_size)
        path_only_roles.add("capture_task_log")

    motion = metadata.get("motion_monitor")
    result = motion.get("result") if isinstance(motion, Mapping) else None
    if isinstance(result, Mapping):
        motion_id = result.get("capture_id") or (motion.get("capture_id") if isinstance(motion, Mapping) else None)
        if motion_id is not None and (motion_id != capture_id or not isinstance(capture_id, str)):
            raise RecoveryConflict("motion artifacts belong to another capture")
        artifacts = result.get("artifacts")
        recovery = result.get("recovery")
        event_log = artifacts.get("event_log") if isinstance(artifacts, Mapping) else None
        if not isinstance(event_log, str) and isinstance(recovery, Mapping):
            event_log = recovery.get("event_log")
        if bool((isinstance(motion, Mapping) and motion.get("enabled")) or result.get("enabled")) and not isinstance(event_log, str):
            raise RecoveryConflict("enabled motion capture has no event log reference")
        expected_motion_dir = f"motion_artifacts/{session}/{capture_id}"
        if isinstance(event_log, str):
            expected_event_log = (
                sidecar_relative_path(f"{session}.{capture_id}.movement-events.jsonl", layout)
                if layout == INTERNAL_LAYOUT
                else f"{expected_motion_dir}/movement-events.jsonl"
            )
            if event_log != expected_event_log:
                raise RecoveryConflict("motion event log path does not match its capture identity")
            expected_paths["motion_event_log"] = expected_event_log
        if isinstance(artifacts, Mapping):
            for key, role, expected in (
                ("capture_directory", "motion_capture_directory", expected_motion_dir),
                ("keyframe_directory", "motion_keyframe_directory", f"{expected_motion_dir}/keyframes"),
            ):
                value = artifacts.get(key)
                if isinstance(value, str):
                    if value != expected:
                        raise RecoveryConflict(f"motion {key} does not match its capture identity")
                    expected_paths[role] = expected
                    directory_roles.add(role)
        keyframes = result.get("keyframes")
        events = result.get("events")
        event_ids = {
            item.get("event_id") for item in events
            if isinstance(item, Mapping) and isinstance(item.get("event_id"), str)
        } if isinstance(events, list) else set()
        if isinstance(keyframes, list):
            for index, item in enumerate(keyframes):
                if not isinstance(item, Mapping) or item.get("status") != "saved":
                    continue
                if item.get("capture_id") != capture_id or item.get("event_id") not in event_ids:
                    raise RecoveryConflict("saved motion keyframe has an invalid event identity")
                relative = item.get("relative_path")
                expected_parent = f"{expected_motion_dir}/keyframes"
                if not isinstance(relative, str) or Path(relative).name == relative or not relative.startswith(expected_parent + "/") or not relative.lower().endswith(".jpg"):
                    raise RecoveryConflict("saved motion keyframe path is outside its owned JPEG directory")
                role = f"motion_keyframe_{index}"
                expected_paths[role] = relative
                if "motion_keyframe_directory" not in expected_paths:
                    expected_paths["motion_keyframe_directory"] = expected_parent
                    directory_roles.add("motion_keyframe_directory")

    by_role: dict[str, Mapping[str, Any]] = {}
    for item in references:
        if not isinstance(item, Mapping) or not isinstance(item.get("role"), str):
            raise RecoveryConflict("snapshot contains an invalid file reference")
        role = item["role"]
        if role in by_role:
            raise RecoveryConflict(f"snapshot contains duplicate file reference role {role!r}")
        by_role[role] = item
    if set(by_role) != set(expected_paths):
        raise RecoveryConflict("snapshot file references do not match the required capture artifacts")

    verified: dict[str, Path] = {}
    for role, relative in expected_paths.items():
        reference = by_role[role]
        # R1 snapshots created a small number of persistence references from
        # Path objects before they were normalized to output-root-relative
        # names. Accept only that exact old representation: the absolute path
        # must be the canonical expected child, with no unbound file metadata.
        legacy_absolute_reference = (
            role in {"capture_events", "capture_task_log"}
            and set(reference) == {"role", "relative_path", "status"}
            and reference.get("status") == "unsafe_reference"
            and reference.get("relative_path") == str(root / relative)
        )
        if not legacy_absolute_reference and (
            reference.get("status") != "present" or reference.get("relative_path") != relative
        ):
            raise RecoveryConflict(f"required {role} reference is missing or changed")
        path = _safe_reference_child(root, relative)
        if role in directory_roles:
            if reference.get("reference_type") != "directory" or not path.is_dir():
                raise RecoveryConflict(f"required {role} directory is missing or replaced")
            if "sha256" in reference or "size_bytes" in reference:
                raise RecoveryConflict(f"directory reference {role} contains file-only metadata")
            verified[role] = path
            continue
        if not path.is_file():
            raise RecoveryConflict(f"required {role} file is missing or replaced")
        if role in path_only_roles:
            if "sha256" in reference or "size_bytes" in reference:
                raise RecoveryConflict("capture task log must bind only its owned path")
            verified[role] = path
            continue
        if legacy_absolute_reference:
            expected_sha, expected_size = expected_metadata_hashes.get(role, (None, None))
        else:
            expected_size = reference.get("size_bytes")
            expected_sha = reference.get("sha256")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise RecoveryConflict(f"snapshot {role} size is invalid")
        if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
            raise RecoveryConflict(f"snapshot {role} hash is invalid")
        stat = path.stat()
        if stat.st_size != expected_size or _sha256_file(path) != expected_sha:
            raise RecoveryConflict(f"required {role} file changed after the snapshot was saved")
        if role in expected_metadata_hashes and expected_metadata_hashes[role] != (expected_sha, expected_size):
            raise RecoveryConflict(f"snapshot {role} reference conflicts with capture metadata")
        verified[role] = path

    marker_metadata_present = "user_markers" in metadata or "user_marker_log" in metadata
    if marker_metadata_present:
        markers = metadata.get("user_markers")
        marker_log = metadata.get("user_marker_log")
        if (
            not isinstance(persistence, Mapping)
            or not isinstance(markers, list)
            or not isinstance(marker_log, Mapping)
            or "capture_events" not in verified
        ):
            raise RecoveryConflict("snapshot user marker metadata or capture event reference is missing")
        try:
            from capture_persistence import (
                read_event_jsonl,
                user_marker_stream_sha256,
                user_markers_from_events,
            )

            event_path = verified["capture_events"]
            event_data = read_event_jsonl(event_path, capture_id=capture_id)
            marker_context = {
                "session_id": session,
                "capture_id": capture_id,
                "attempt_id": persistence.get("attempt_id"),
                "job_id": persistence.get("job_id"),
                "events_path": persistence.get("events_path"),
                "config": {"capture_start_monotonic_ns": metadata.get("capture_start_monotonic_ns")},
            }
            parsed_markers = user_markers_from_events(event_data["records"], marker_context)
            marker_digest = user_marker_stream_sha256(parsed_markers)
            expected_relative = sidecar_relative_path(
                f"{session}.{capture_id}.capture-events.jsonl", layout
            )
            expected_event_hash = persistence.get("events_sha256")
            expected_event_size = persistence.get("events_size_bytes")
            if markers != parsed_markers:
                raise RecoveryConflict("snapshot user marker list differs from its durable capture events")
            if (
                marker_log.get("schema") != "localgaze.capture-user-marker-log/1"
                or marker_log.get("event") != "user_marker"
                or marker_log.get("path") != expected_relative
                or marker_log.get("count") != len(parsed_markers)
                or marker_log.get("last_sequence") != len(parsed_markers)
                or marker_log.get("last_monotonic_ns") != (parsed_markers[-1]["monotonic_ns"] if parsed_markers else None)
                or marker_log.get("markers_sha256") != marker_digest
                or marker_log.get("events_sha256") != expected_event_hash
                or marker_log.get("events_size_bytes") != expected_event_size
                or marker_log != persistence.get("user_marker_log")
            ):
                raise RecoveryConflict("snapshot user marker log reference or hash is inconsistent")
        except RecoveryConflict:
            raise
        except Exception as exc:
            raise RecoveryConflict(f"snapshot user marker event log is invalid: {type(exc).__name__}: {exc}") from exc
    return verified


def _verify_samples_reference(payload: Mapping[str, Any], root: Path) -> tuple[Path, str]:
    verified = _verify_file_references(payload, root)
    sample = next(
        item for item in payload["file_references"]
        if isinstance(item, Mapping) and item.get("role") == "samples_tsv"
    )
    return verified["samples_tsv"], str(sample["sha256"])


def _staging_owner(envelope: Mapping[str, Any], snapshot_sha: str) -> dict[str, Any]:
    session = envelope["session_id"]
    attempt = envelope["attempt_id"]
    return {
        "schema": _STAGING_SCHEMA,
        "session_id": session,
        "attempt_id": attempt,
        "snapshot_sha256": snapshot_sha,
    }


def _ensure_staging_dir(root: Path, envelope: Mapping[str, Any], snapshot_sha: str) -> Path:
    session = envelope["session_id"]
    attempt = envelope["attempt_id"]
    staging = root / f".{session}.capture-export-{attempt}"
    try:
        staging = assert_no_reparse_components(staging)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict("attempt staging path contains a symbolic link or junction") from exc
    if staging.exists() or staging.is_symlink():
        if not staging.is_dir() or staging.resolve(strict=True).parent != root:
            raise RecoveryConflict("attempt staging path is not an owned directory")
        marker = staging / ".owner.json"
        if not marker.exists() and not marker.is_symlink() and not any(staging.iterdir()):
            # A crash after creating the unique attempt directory but before
            # its ledger is harmless only while the directory remains empty.
            expected = _staging_owner(envelope, snapshot_sha)
            _write_exclusive_atomic(marker, _json_bytes(expected) + b"\n")
            return staging
        existing, raw = _read_json(marker)
        expected = _staging_owner(envelope, snapshot_sha)
        if existing != expected or raw != _json_bytes(expected) + b"\n":
            raise RecoveryConflict("attempt staging ownership marker does not match")
        return staging
    staging.mkdir(mode=0o700)
    marker = staging / ".owner.json"
    expected = _staging_owner(envelope, snapshot_sha)
    _write_exclusive_atomic(marker, _json_bytes(expected) + b"\n")
    return staging


def _validate_writer_result(
    result: Any,
    build_dir: Path,
    session: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(result, Mapping) or set(result) != {"xlsx", "manifest"}:
        raise RecoveryConflict("writer must return exactly xlsx and manifest artifacts")
    expected_names = {
        "xlsx": f"{session}.xlsx",
        "manifest": f"{session}.manifest.json",
    }
    validated: dict[str, dict[str, str]] = {}
    for kind, expected_name in expected_names.items():
        item = result.get(kind)
        if not isinstance(item, Mapping):
            raise RecoveryConflict(f"writer result {kind} is invalid")
        relative = item.get("relative_path")
        digest = item.get("sha256")
        if relative != expected_name:
            raise RecoveryConflict(f"writer {kind} must be the session's direct-child filename")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RecoveryConflict(f"writer {kind} SHA-256 is invalid")
        path = _safe_child(build_dir, relative)
        if path.is_symlink() or not path.is_file():
            raise RecoveryConflict(f"writer {kind} is missing or not a regular file")
        actual = _sha256_file(path)
        if actual != digest:
            raise RecoveryConflict(f"writer {kind} hash does not match its staged content")
        validated[kind] = {"relative_path": relative, "sha256": digest}
    return validated


def _load_owner(
    path: Path,
    *,
    session: str,
    attempt: str,
    snapshot_sha: str,
    layout: str,
) -> tuple[dict[str, Any] | None, bytes | None]:
    if not path.exists() and not path.is_symlink():
        return None, None
    owner, raw = _read_json(path)
    if (
        owner.get("schema") != _OWNER_SCHEMA
        or owner.get("session_id") != session
        or owner.get("attempt_id") != attempt
        or owner.get("snapshot_sha256") != snapshot_sha
    ):
        raise RecoveryConflict("export owner marker belongs to another or unknown attempt")
    artifacts = owner.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"xlsx", "manifest"}:
        raise RecoveryConflict("export owner marker is incomplete")
    expected = _expected_final_artifacts(session, layout)
    for kind, name in expected.items():
        entry = artifacts.get(kind)
        if (
            not isinstance(entry, Mapping)
            or entry.get("relative_path") != name
            or not isinstance(entry.get("sha256"), str)
            or not _SHA256_RE.fullmatch(entry["sha256"])
        ):
            raise RecoveryConflict("export owner marker contains an invalid artifact identity")
    return dict(owner), raw


def _verify_final_file(path: Path, expected_sha: str) -> bool:
    try:
        path = assert_no_reparse_components(path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RecoveryConflict(f"existing export path contains a link or junction: {path.name}") from exc
    if not path.exists() and not path.is_symlink():
        return False
    if not path.is_file():
        raise RecoveryConflict(f"existing export is not an owned regular file: {path.name}")
    if _sha256_file(path) != expected_sha:
        raise RecoveryConflict(f"existing export was changed or replaced: {path.name}")
    return True


def _marker_document(
    *,
    session: str,
    attempt: str,
    snapshot_sha: str,
    artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema": _COMPLETE_SCHEMA,
        "session_id": session,
        "attempt_id": attempt,
        "snapshot_sha256": snapshot_sha,
        "artifacts": {
            name: {
                "relative_path": artifacts[name]["relative_path"],
                "sha256": artifacts[name]["sha256"],
            }
            for name in ("xlsx", "manifest")
        },
    }


def _success_result(
    *,
    paths: Mapping[str, Path],
    snapshot: Path,
    samples: Path | None,
    attempt: str,
    snapshot_sha: str,
    session: str,
    root: Path,
    stage_base: Path,
    attempt_staging: Path,
    idempotent: bool,
) -> dict[str, Any]:
    outputs = {
        "xlsx": str(paths["xlsx"]),
        "manifest": str(paths["manifest"]),
        "snapshot": str(snapshot),
        "completion_marker": str(paths["completion_marker"]),
    }
    if samples is not None:
        outputs["samples"] = str(samples)
    return {
        "status": "completed",
        "outputs": outputs,
        "details": {
            "recoverable": True,
            "export_complete": True,
            "idempotent": idempotent,
            "retryable": False,
            "reason": None,
            "attempt_id": attempt,
            "session_id": session,
            "output_root": str(root),
            "snapshot_path": str(snapshot),
            "staging_path": str(stage_base),
            "attempt_staging_path": str(attempt_staging),
            "snapshot_sha256": snapshot_sha,
            "partial_outputs": [],
            "retained_files": [],
        },
    }


def _failure_result(
    reason: str,
    *,
    recoverable: bool,
    attempt: str | None = None,
    snapshot_sha: str | None = None,
    snapshot: Path | None = None,
    error: str | None = None,
    partial_outputs: list[str] | None = None,
    retained_files: list[str] | None = None,
    session: str | None = None,
    root: Path | None = None,
    stage_base: Path | None = None,
    attempt_staging: Path | None = None,
) -> dict[str, Any]:
    outputs = {"snapshot": str(snapshot)} if snapshot is not None else {}
    details: dict[str, Any] = {
        "recoverable": recoverable,
        "retryable": recoverable,
        "export_complete": False,
        "idempotent": False,
        "reason": reason,
        "attempt_id": attempt,
        "session_id": session,
        "output_root": str(root) if root is not None else None,
        "snapshot_path": str(snapshot) if snapshot is not None else None,
        "staging_path": str(stage_base or root) if (stage_base or root) is not None else None,
        "attempt_staging_path": str(attempt_staging) if attempt_staging is not None else None,
        "snapshot_sha256": snapshot_sha,
        "partial_outputs": partial_outputs or [],
        "retained_files": retained_files or [],
    }
    if error:
        details["error"] = error
    return {"status": "failed", "outputs": outputs, "details": details}


def _completion_is_valid(
    paths: Mapping[str, Path],
    marker_doc: Mapping[str, Any],
    expected_doc: Mapping[str, Any],
) -> bool:
    if marker_doc != expected_doc:
        raise RecoveryConflict("completion marker belongs to another or changed export")
    artifacts = expected_doc["artifacts"]
    for name in ("xlsx", "manifest"):
        if not _verify_final_file(paths[name], artifacts[name]["sha256"]):
            raise RecoveryConflict(f"completed export is missing {name}")
    return True


def verify_capture_archive(snapshot_path: Path) -> dict[str, Any]:
    """Revalidate a saved snapshot and every content-bound archive reference."""
    envelope, payload, root, raw = _load_snapshot(Path(snapshot_path).expanduser())
    verified = _verify_file_references(payload, root)
    capture_metadata = payload.get("capture_metadata")
    persistence = capture_metadata.get("capture_persistence") if isinstance(capture_metadata, Mapping) else None
    return {
        "status": "verified",
        "session_id": envelope["session_id"],
        "attempt_id": envelope["attempt_id"],
        "snapshot_sha256": _sha256_bytes(raw),
        "verified_roles": sorted(verified),
        "capture_persistence": dict(persistence) if isinstance(persistence, Mapping) else None,
    }


def _publish_exclusive(source: Path, final: Path, expected_sha: str) -> bool:
    """Create a final hard link or accept an exact already-published file."""

    if _verify_final_file(final, expected_sha):
        return False
    if source.is_symlink() or not source.is_file() or _sha256_file(source) != expected_sha:
        raise RecoveryConflict(f"owned staged source is missing or changed: {source.name}")
    try:
        os.link(source, final)
    except FileExistsError:
        if _verify_final_file(final, expected_sha):
            return False
        raise RecoveryConflict(f"unknown file appeared during publication: {final.name}")
    if not os.path.samefile(source, final) or _sha256_file(final) != expected_sha:
        raise RecoveryConflict(f"published file identity changed: {final.name}")
    return True


def recover_capture_export(
    snapshot_path: Path,
    *,
    output_writer: OutputWriter,
    staging_path: Path,
) -> dict[str, Any]:
    """Rebuild/publish XLSX and manifest without camera or model execution.

    ``staging_path`` is the declared sidecar-layout directory: output_root for
    legacy snapshots and output_root/.internal for new snapshots. The writer
    returns exact direct-child filenames and SHA-256 values for the two
    regular files it created there.
    """

    snapshot = Path(snapshot_path).expanduser()
    attempt: str | None = None
    snapshot_sha: str | None = None
    session: str | None = None
    root: Path | None = None
    stage_base: Path | None = None
    staging: Path | None = None
    samples: Path | None = None
    paths: dict[str, Path] | None = None
    published: list[str] = []
    owner_raw: bytes | None = None
    try:
        envelope, payload, root, snapshot_raw = _load_snapshot(snapshot)
        attempt = envelope["attempt_id"]
        snapshot_sha = _sha256_bytes(snapshot_raw)
        session = envelope["session_id"]
        layout = _declared_layout(payload.get("capture_metadata"))
        stage_base = _validate_stage_base(Path(staging_path), root, layout)
        if not callable(output_writer):
            raise RecoveryConflict("output_writer must be callable")
        samples, _ = _verify_samples_reference(payload, root)
        paths = _artifact_paths(root, session, layout)

        owner, owner_raw = _load_owner(
            paths["owner"],
            session=session,
            attempt=attempt,
            snapshot_sha=snapshot_sha,
            layout=layout,
        )
        if paths["completion_marker"].exists() or paths["completion_marker"].is_symlink():
            marker, _ = _read_json(paths["completion_marker"])
            if owner is None:
                raise RecoveryConflict("completion marker exists without a matching owner marker")
            expected_marker = _marker_document(
                session=session,
                attempt=attempt,
                snapshot_sha=snapshot_sha,
                artifacts=owner["artifacts"],
            )
            _completion_is_valid(paths, marker, expected_marker)
            if snapshot.is_symlink() or _sha256_file(snapshot) != snapshot_sha:
                raise RecoveryConflict("snapshot changed while validating its completion marker")
            _verify_samples_reference(payload, root)
            return _success_result(
                paths=paths,
                snapshot=snapshot,
                samples=samples,
                attempt=attempt,
                snapshot_sha=snapshot_sha,
                session=session,
                root=root,
                stage_base=stage_base,
                attempt_staging=stage_base / f".{session}.capture-export-{attempt}",
                idempotent=True,
            )

        if owner is None and any(
            path.exists() or path.is_symlink()
            for path in (paths["xlsx"], paths["manifest"])
        ):
            raise RecoveryConflict("an export exists without this attempt's owner marker")
        staging = _ensure_staging_dir(stage_base, envelope, snapshot_sha)

        staged_artifacts: dict[str, dict[str, str]] | None = None
        build_dir: Path | None = None
        if owner is None:
            build_dir = staging / f"build-{uuid.uuid4().hex}"
            build_dir.mkdir(mode=0o700)
            writer_result = output_writer(build_dir, payload)
            staged_artifacts = _validate_writer_result(writer_result, build_dir, session)
            if snapshot.is_symlink() or _sha256_file(snapshot) != snapshot_sha:
                raise RecoveryConflict("snapshot changed while the writer prepared outputs")
            _verify_samples_reference(payload, root)
            owner = {
                "schema": _OWNER_SCHEMA,
                "session_id": session,
                "attempt_id": attempt,
                "snapshot_sha256": snapshot_sha,
                "artifacts": {
                    kind: {
                        "relative_path": _expected_final_artifacts(session, layout)[kind],
                        "sha256": staged_artifacts[kind]["sha256"],
                    }
                    for kind in ("xlsx", "manifest")
                },
            }
            owner_content = _json_bytes(owner) + b"\n"
            _write_exclusive_atomic(paths["owner"], owner_content)
            owner_raw = owner_content
        else:
            staged_artifacts = dict(owner["artifacts"])
            # Validate every existing final before publishing either member of
            # the pair.  A replaced manifest must not leave a new workbook
            # published as a side effect of this retry.
            for kind in ("xlsx", "manifest"):
                _verify_final_file(paths[kind], staged_artifacts[kind]["sha256"])
            # Prefer an existing build only after the private staging marker and
            # the owner hashes validate.  Otherwise ask the callback to rebuild;
            # no failed or unknown staging files are deleted.
            for child in staging.iterdir():
                if not child.is_dir() or child.is_symlink() or not child.name.startswith("build-"):
                    continue
                try:
                    candidate = _validate_writer_result(
                        {
                            kind: {
                                "relative_path": _writer_artifact_names(session)[kind],
                                "sha256": staged_artifacts[kind]["sha256"],
                            }
                            for kind in ("xlsx", "manifest")
                        },
                        child,
                        session,
                    )
                except (OSError, RecoveryConflict):
                    continue
                if all(candidate[kind]["sha256"] == staged_artifacts[kind]["sha256"] for kind in ("xlsx", "manifest")):
                    build_dir = child
                    break
            if build_dir is None:
                build_dir = staging / f"build-{uuid.uuid4().hex}"
                build_dir.mkdir(mode=0o700)
                writer_result = output_writer(build_dir, payload)
                candidate = _validate_writer_result(writer_result, build_dir, session)
                if not all(candidate[kind]["sha256"] == staged_artifacts[kind]["sha256"] for kind in ("xlsx", "manifest")):
                    raise RecoveryConflict("regenerated staged output hashes differ from owner marker")

        assert owner is not None and staged_artifacts is not None and build_dir is not None
        expected_owner_bytes = _json_bytes(owner) + b"\n"
        if owner_raw != expected_owner_bytes:
            raise RecoveryConflict("owner marker bytes changed during recovery")
        current_owner, current_owner_raw = _read_json(paths["owner"])
        if current_owner != owner or current_owner_raw != expected_owner_bytes:
            raise RecoveryConflict("owner marker changed before publication")
        if snapshot.is_symlink() or _sha256_file(snapshot) != snapshot_sha:
            raise RecoveryConflict("snapshot changed before publication")
        _verify_samples_reference(payload, root)

        for kind in ("xlsx", "manifest"):
            final = paths[kind]
            source = _safe_child(build_dir, _writer_artifact_names(session)[kind])
            _publish_exclusive(source, final, staged_artifacts[kind]["sha256"])
            _verify_final_file(final, staged_artifacts[kind]["sha256"])
            published.append(str(final))

        # Re-read ownership and content just before the final completion marker.
        current_owner, current_owner_raw = _read_json(paths["owner"])
        if current_owner != owner or current_owner_raw != expected_owner_bytes:
            raise RecoveryConflict("owner marker changed before completion")
        if snapshot.is_symlink() or _sha256_file(snapshot) != snapshot_sha:
            raise RecoveryConflict("snapshot changed before completion")
        _verify_samples_reference(payload, root)
        marker_doc = _marker_document(
            session=session,
            attempt=attempt,
            snapshot_sha=snapshot_sha,
            artifacts=owner["artifacts"],
        )
        marker_content = _json_bytes(marker_doc) + b"\n"
        try:
            _write_exclusive_atomic(paths["completion_marker"], marker_content)
        except FileExistsError:
            existing_marker, _ = _read_json(paths["completion_marker"])
            _completion_is_valid(paths, existing_marker, marker_doc)
        actual_marker, _ = _read_json(paths["completion_marker"])
        if snapshot.is_symlink() or _sha256_file(snapshot) != snapshot_sha:
            raise RecoveryConflict("snapshot changed after publishing completion")
        _verify_samples_reference(payload, root)
        _completion_is_valid(paths, actual_marker, marker_doc)
        return _success_result(
            paths=paths,
            snapshot=snapshot,
            samples=samples,
            attempt=attempt,
            snapshot_sha=snapshot_sha,
            session=session,
            root=root,
            stage_base=stage_base,
            attempt_staging=staging,
            idempotent=False,
        )
    except Exception as exc:
        reason = "recovery_conflict" if isinstance(exc, RecoveryConflict) else "export_recovery_failed"
        recoverable = reason != "recovery_conflict"
        retained: list[str] = []
        for path in (
            snapshot,
            paths.get("xlsx") if paths else None,
            paths.get("manifest") if paths else None,
            paths.get("owner") if paths else None,
            paths.get("completion_marker") if paths else None,
            staging,
        ):
            if path is not None and (path.exists() or path.is_symlink()):
                retained.append(str(path))
        if samples is not None and (samples.exists() or samples.is_symlink()):
            retained.append(str(samples))
        return _failure_result(
            reason,
            recoverable=recoverable,
            attempt=attempt,
            snapshot_sha=snapshot_sha,
            snapshot=snapshot if snapshot.is_file() else None,
            error=f"{type(exc).__name__}: {exc}",
            partial_outputs=published,
            retained_files=retained,
            session=session,
            root=root,
            stage_base=stage_base,
            attempt_staging=staging,
        )


def recover_capture_from_context(
    context_path: str | os.PathLike[str],
    *,
    output_writer: OutputWriter,
) -> dict[str, Any]:
    """Rebuild a snapshot and verified export from a durable capture context.

    This recovery path reads only the saved context, TSV, checkpoint journal,
    and event logs. It never opens a camera or loads a gaze model. A snapshot
    that was already prepared is reused only after its capture identity is
    matched to the selected context.
    """

    from capture_persistence import (
        CapturePersistence,
        CapturePersistenceError,
        cleanup_capture_sidecars,
        load_capture_context,
        motion_recovery_summary,
        read_capture_tsv,
        read_event_jsonl,
    )

    supplied_context = Path(context_path).expanduser()
    context: dict[str, Any] | None = None
    root: Path | None = None
    writer: CapturePersistence | None = None
    raw_path: Path | None = None
    snapshot_path: Path | None = None
    parsed: dict[str, Any] | None = None
    recovery_result: dict[str, Any] | None = None
    snapshot_result: dict[str, Any] | None = None
    capture_metadata: dict[str, Any] | None = None
    task_log_started = False
    layout = LEGACY_LAYOUT

    def retained_paths() -> list[str]:
        def declared_sidecar(name: str) -> Path | None:
            if root is None:
                return None
            return root / ".internal" / name if layout == INTERNAL_LAYOUT else root / name

        paths = [
            supplied_context,
            raw_path,
            (root / context["checkpoint_path"]) if root is not None and context is not None else None,
            (root / context["events_path"]) if root is not None and context is not None else None,
            (root / context["capture_task_log_path"]) if root is not None and context is not None else None,
            snapshot_path,
            (root / f"{context['session_id']}.xlsx") if root is not None and context is not None else None,
            declared_sidecar(f"{context['session_id']}.manifest.json") if context is not None else None,
            declared_sidecar(f".{context['session_id']}.capture-export-owner.json") if context is not None else None,
            declared_sidecar(f".{context['session_id']}.capture-export-complete.json") if context is not None else None,
        ]
        return list(dict.fromkeys(str(path) for path in paths if path is not None and (path.exists() or path.is_symlink())))

    def failure(reason: str, error: BaseException | None = None) -> dict[str, Any]:
        identity = context or {}
        checkpoint = parsed.get("last_durable_checkpoint") if isinstance(parsed, dict) else None
        if checkpoint is None and writer is not None:
            checkpoint = writer.last_checkpoint
        record_count = len(parsed.get("records", [])) if isinstance(parsed, dict) else 0
        details: dict[str, Any] = {
            "recoverable": bool(raw_path and raw_path.is_file() and writer and writer.checkpoint_path.is_file()),
            "retryable": False,
            "export_complete": False,
            "idempotent": False,
            "reason": reason,
            "session_id": identity.get("session_id"),
            "capture_id": identity.get("capture_id"),
            "attempt_id": identity.get("attempt_id"),
            "job_id": identity.get("job_id"),
            "context_path": str(supplied_context),
            "output_root": str(root) if root is not None else None,
            "raw_path": str(raw_path) if raw_path is not None else None,
            "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
            "last_durable_checkpoint": checkpoint,
            "recovered_record_count": record_count,
            "partial_outputs": (
                list(recovery_result.get("details", {}).get("partial_outputs", []))
                if isinstance(recovery_result, dict) and isinstance(recovery_result.get("details"), dict)
                else []
            ),
            "retained_files": retained_paths(),
            "log_io_errors": dict(writer._log_errors) if writer is not None else {},
        }
        if isinstance(snapshot_result, dict):
            details["snapshot_result"] = snapshot_result.get("details", {})
        if isinstance(recovery_result, dict):
            details["export_recovery"] = recovery_result.get("details", {})
        if isinstance(capture_metadata, Mapping):
            if capture_metadata.get("error") is not None:
                details["original_capture_error"] = capture_metadata["error"]
            if capture_metadata.get("traceback") is not None:
                details["original_capture_traceback"] = capture_metadata["traceback"]
            recovery = capture_metadata.get("recovery")
            if isinstance(recovery, Mapping):
                details["recovery"] = dict(recovery)
        if error is not None:
            details["error"] = f"{type(error).__name__}: {error}"
        return {
            "status": "failed",
            "message": str(details.get("error") or reason),
            "outputs": {},
            "details": details,
        }

    try:
        context, root = load_capture_context(supplied_context)
        layout = normalize_sidecar_layout(context.get("sidecar_layout"))
        raw_path = root / context["raw_path"]
        snapshot_path = sidecar_path(
            root,
            f"{context['session_id']}.capture-snapshot.json",
            layout=layout,
            create_parent=layout == INTERNAL_LAYOUT,
        )
        writer = CapturePersistence(supplied_context, context=context)
        task_log_started = writer.append_job_event(
            "capture_recovery_started",
            stage="recovery_start",
            context_path=str(supplied_context.resolve()),
            raw_path=str(raw_path),
        )

        if snapshot_path.exists() or snapshot_path.is_symlink():
            if snapshot_path.is_symlink():
                raise RecoveryConflict("capture snapshot is a symbolic link")
            envelope, payload, snapshot_root, _ = _load_snapshot(snapshot_path)
            saved_metadata = payload.get("capture_metadata")
            saved_persistence = saved_metadata.get("capture_persistence") if isinstance(saved_metadata, Mapping) else None
            if (
                snapshot_root != root
                or envelope.get("session_id") != context.get("session_id")
                or not isinstance(saved_metadata, Mapping)
                or saved_metadata.get("capture_id") != context.get("capture_id")
                or not isinstance(saved_persistence, Mapping)
                or saved_persistence.get("attempt_id") != context.get("attempt_id")
            ):
                raise RecoveryConflict("existing snapshot does not belong to the selected capture context")
            capture_metadata = dict(saved_metadata)
            snapshot_result = {
                "status": "prepared",
                "outputs": {"snapshot": str(snapshot_path)},
                "details": {
                    "recoverable": True,
                    "attempt_id": envelope.get("attempt_id"),
                    "snapshot_sha256": _sha256_bytes(snapshot_path.read_bytes()),
                },
            }
        else:
            config = context.get("config") if isinstance(context.get("config"), Mapping) else {}
            expected_header = config.get("capture_columns")
            parsed = read_capture_tsv(
                raw_path,
                context=context,
                expected_header=expected_header if isinstance(expected_header, list) else None,
            )
            records = parsed["records"]
            for index, record in enumerate(records, start=1):
                row_capture_id = record.get("capture_id")
                if row_capture_id is not None and row_capture_id != context["capture_id"]:
                    raise CapturePersistenceError(f"capture TSV record {index} belongs to another capture")
                motion_id = record.get("motion_capture_id")
                if motion_id is not None and motion_id != context["capture_id"]:
                    raise CapturePersistenceError(f"capture TSV record {index} has a mismatched motion capture identity")

            marker_reconciliation = writer.reconcile_user_markers()
            context = writer.context
            events = read_event_jsonl(writer.events_path, capture_id=context["capture_id"])
            motion = motion_recovery_summary(context, events)
            completed_events = [
                item for item in events["records"] if item.get("event") == "capture_finished"
            ]
            final_event = completed_events[-1] if completed_events else None
            final_fields = final_event if isinstance(final_event, Mapping) else {}
            final_capture_metadata = (
                final_fields.get("capture_metadata")
                if isinstance(final_fields.get("capture_metadata"), Mapping)
                else {}
            )
            prior_failure = next(
                (
                    item for item in reversed(events["records"])
                    if item.get("error") or item.get("traceback")
                ),
                {},
            )
            original_error = final_fields.get("error") or prior_failure.get("error")
            original_traceback = final_fields.get("traceback") or prior_failure.get("traceback")
            interrupted = not bool(final_event) or final_fields.get("status") != "completed"
            recovery_reason = original_error or (
                "capture_finished_event_missing" if not final_event else final_fields.get("error")
            )
            recovery_status = "capture_interrupted" if not final_event else (
                "capture_finished_export_recovery" if not interrupted else "partial_capture_export_recovery"
            )
            recovery = {
                "status": recovery_status,
                "interrupted": interrupted,
                "capture_complete": not interrupted,
                "reason": recovery_reason,
                "original_error": original_error,
                "original_traceback": original_traceback,
                "last_durable_checkpoint": parsed["last_durable_checkpoint"],
                "complete_records_after_checkpoint": parsed["complete_records_after_checkpoint"],
                "truncated_terminal_records": parsed["truncated_terminal_records"],
                "truncated_terminal_bytes": parsed["truncated_terminal_bytes"],
                "incomplete_checkpoint_records": parsed["incomplete_checkpoint_records"],
                "incomplete_checkpoint_bytes": parsed["incomplete_checkpoint_bytes"],
                "event_log_sha256": events["sha256"],
                "event_log_size_bytes": events["size_bytes"],
                "truncated_event_records": events["truncated_terminal_records"],
                "truncated_event_bytes": events["truncated_terminal_bytes"],
                "user_marker_reconciliation": marker_reconciliation,
                "motion": motion,
            }
            persistence_summary = writer.snapshot_metadata(recovery=recovery)
            persistence_summary.update({
                "recovered_raw_sha256": parsed["raw_sha256"],
                "recovered_raw_size_bytes": parsed["raw_size_bytes"],
                "recovered_record_count": len(records),
                "event_log_parse": {
                    "truncated_terminal_records": events["truncated_terminal_records"],
                    "truncated_terminal_bytes": events["truncated_terminal_bytes"],
                    "parsed_records": len(events["records"]),
                },
            })
            status_counts: dict[str, int] = {}
            for record in records:
                status = str(record.get("sample_status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            start_event = next(
                (item for item in events["records"] if item.get("event") == "capture_prepared"),
                None,
            )
            motion_requested = bool(config.get("motion_monitor_enabled", False))
            motion_events = list(motion.get("events", []))
            motion_keyframes = list(motion.get("keyframes", []))
            capture_metadata = {
                "session_id": context["session_id"],
                "capture_id": context["capture_id"],
                "sidecar_layout": layout,
                "table_format": context.get("table_format", "txt"),
                "raw_path": context["raw_path"],
                "files": {
                    "xlsx": f"{context['session_id']}.xlsx",
                    context.get("table_format", "txt"): context["raw_path"],
                    "samples": context["raw_path"],
                    "table_format": context.get("table_format", "txt"),
                    "manifest": sidecar_relative_path(
                        f"{context['session_id']}.manifest.json", layout
                    ),
                },
                "job_id": context.get("job_id"),
                "workflow_mode": config.get("workflow_mode", "capture"),
                "source": "recovered_from_durable_capture_context",
                "started_at_local": config.get("started_at_local"),
                "ended_at_local": final_capture_metadata.get("ended_at_local"),
                "capture_start_monotonic_ns": config.get("capture_start_monotonic_ns"),
                "capture_end_monotonic_ns": final_capture_metadata.get("capture_end_monotonic_ns"),
                "stop_reason": (
                    final_capture_metadata.get("stop_reason")
                    if final_event
                    else "capture_interrupted"
                ),
                "error": original_error,
                "traceback": original_traceback,
                "frames_received": None,
                "frames_processed": len(records),
                "frames_dropped_unprocessed": None,
                "records_written": len(records),
                "accepted_samples": sum(1 for record in records if record.get("sample_accepted") is True),
                "sample_status_counts": status_counts,
                "model_binding": config.get("model_binding"),
                "offset_correction": config.get("offset_correction", {"enabled": False, "status": "unavailable_after_interruption"}),
                "motion_monitor": {
                    "requested": motion_requested,
                    "enabled": bool(motion.get("event_log")),
                    "save_keyframes_requested": bool(config.get("save_motion_keyframes", False)),
                    "reason": "recovered_after_interruption" if interrupted else "capture_finished_before_export",
                    "result": {
                        "capture_id": context["capture_id"],
                        "events": motion_events,
                        "keyframes": motion_keyframes,
                        "recovery": motion,
                    },
                },
                "capture_persistence": persistence_summary,
                "user_markers": writer.user_markers,
                "user_marker_log": dict(persistence_summary["user_marker_log"]),
                "recovery": recovery,
            }
            if start_event is not None:
                capture_metadata["capture_context_created_at_utc"] = context.get("created_at_utc")
            experiment_info = {
                "会话ID": [context["session_id"]],
                "恢复状态": [recovery_status],
                "恢复记录数": [len(records)],
                "开始时间": [context.get("created_at_utc")],
                "结束时间": [final_capture_metadata.get("ended_at_local")],
                "capture_id": [context["capture_id"]],
                "用户标记数量": [len(writer.user_markers)],
                "标记事件日志": [persistence_summary["user_marker_log"].get("path")],
                "标记事件日志 SHA-256": [persistence_summary["user_marker_log"].get("events_sha256")],
                "标记序列 SHA-256": [persistence_summary["user_marker_log"].get("markers_sha256")],
            }
            snapshot_result = persist_capture_snapshot(
                snapshot_path,
                session_id=context["session_id"],
                output_root=root,
                records=records,
                experiment_info=experiment_info,
                capture_metadata=capture_metadata,
                model_metadata=(
                    config.get("model_metadata_contract")
                    if isinstance(config.get("model_metadata_contract"), Mapping)
                    else None
                ),
            )
            if not isinstance(snapshot_result, dict) or snapshot_result.get("status") != "prepared":
                return failure("snapshot_persist_failed")
            payload_envelope, payload, _, _ = _load_snapshot(snapshot_path)
            capture_metadata = dict(payload.get("capture_metadata") or capture_metadata)

        recovery_result = recover_capture_export(
            snapshot_path,
            output_writer=output_writer,
            staging_path=internal_directory(root, create=True) if layout == INTERNAL_LAYOUT else root,
        )
        if not isinstance(recovery_result, dict) or recovery_result.get("status") != "completed":
            return failure("export_recovery_failed")

        archive_verification = verify_capture_archive(snapshot_path)

        saved_envelope, saved_payload, _, _ = _load_snapshot(snapshot_path)
        saved_metadata = saved_payload.get("capture_metadata")
        if not isinstance(saved_metadata, Mapping) or saved_metadata.get("capture_id") != context["capture_id"]:
            raise RecoveryConflict("verified snapshot capture identity changed before cleanup")
        final_capture_metadata = dict(saved_metadata)
        job_log_finished = writer.append_job_event(
            "capture_recovery_completed",
            stage="recovery_complete",
            capture_id=context["capture_id"],
            snapshot_path=str(snapshot_path),
            export_complete=True,
            recovery_status=(final_capture_metadata.get("recovery") or {}).get("status"),
            recovered_record_count=final_capture_metadata.get("records_written"),
        )
        cleanup = cleanup_capture_sidecars(
            supplied_context,
            output_root=root,
            completion_verified=True,
            expected_metadata=saved_metadata,
        )
        outputs = dict(recovery_result.get("outputs") or {})
        outputs.update({
            "samples": str(raw_path.resolve()),
            "snapshot": str(snapshot_path.resolve()),
            "capture_events": str(writer.events_path),
            "capture_task_log": str(writer.task_log_path),
        })
        details = dict(recovery_result.get("details") or {})
        details.update({
            "recoverable": True,
            "retryable": False,
            "export_complete": True,
            "session_id": context["session_id"],
            "capture_id": context["capture_id"],
            "attempt_id": context["attempt_id"],
            "job_id": context.get("job_id"),
            "context_path": str(supplied_context),
            "last_durable_checkpoint": final_capture_metadata.get("capture_persistence", {}).get("last_durable_checkpoint"),
            "recovered_record_count": final_capture_metadata.get("records_written"),
            "recovery": final_capture_metadata.get("recovery"),
            "cleanup": cleanup,
            "archive_verification": {
                key: value for key, value in archive_verification.items()
                if key != "capture_persistence"
            },
            "log_io_errors": dict(writer._log_errors),
            "task_log_write_succeeded": bool(task_log_started and job_log_finished),
            "retained_files": [
                str(path) for path in (raw_path, writer.events_path, writer.task_log_path, snapshot_path)
                if path.exists()
            ],
        })
        return {
            "status": "completed",
            "message": "Capture records recovered and export files verified without opening a camera.",
            "outputs": outputs,
            "details": details,
        }
    except Exception as exc:
        reason = "recovery_conflict" if isinstance(exc, RecoveryConflict) else (
            "capture_recovery_validation_failed" if isinstance(exc, CapturePersistenceError) else "capture_recovery_failed"
        )
        if writer is not None and context is not None:
            writer.append_job_event(
                "capture_recovery_failed",
                stage="recovery_failed",
                capture_id=context["capture_id"],
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
        return failure(reason, exc)


__all__ = [
    "ExportResult",
    "OutputWriter",
    "SNAPSHOT_SCHEMA",
    "SnapshotError",
    "SnapshotPayload",
    "persist_capture_snapshot",
    "recover_capture_from_context",
    "recover_capture_export",
    "verify_capture_archive",
]
