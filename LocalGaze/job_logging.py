"""Durable, append-only task logs shared by the desktop parent and worker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Mapping

from session_names import canonical_session_id
from capture_paths import (
    INTERNAL_LAYOUT,
    LEGACY_LAYOUT,
    assert_no_reparse_components,
    normalize_sidecar_layout,
    owned_sidecar_path,
    sidecar_path,
)

LOG_ERROR_PREFIX = "LOCALGAZE_JOB_LOG_ERROR "
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_FORCE_SYNC_EVENTS = {
    "worker_started",
    "worker_terminal",
    "python_exception",
    "parent_terminal",
    "durable_checkpoint",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def worker_log_path(
    output_root: str | os.PathLike[str],
    session_id: str | None,
    job_id: str,
    *,
    sidecar_layout: str = INTERNAL_LAYOUT,
) -> Path:
    """Return the stable log path without creating its parent directory."""
    if not isinstance(job_id, str) or not _SAFE_TOKEN.fullmatch(job_id):
        raise ValueError("job_id is not a safe worker-log identifier")
    if session_id is not None:
        session_id = canonical_session_id(session_id)
    root = assert_no_reparse_components(Path(output_root).expanduser()).resolve(strict=False)
    stem = f"{session_id}.{job_id}" if session_id else f"job.{job_id}"
    layout = normalize_sidecar_layout(sidecar_layout)
    return sidecar_path(
        root,
        f"{stem}.worker.jsonl",
        layout=layout,
        create_parent=layout == INTERNAL_LAYOUT,
    )


def capture_context_metadata(context_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read only enough identity from a context file to associate task logs.

    The capture recovery API performs the authoritative schema and path checks
    before using this input. This helper deliberately returns partial metadata
    for damaged contexts so the worker can still preserve an error log nearby.
    """
    path = Path(context_path).expanduser()
    result: dict[str, Any] = {"context_path": str(path.resolve())}
    try:
        if path.is_symlink() or not path.is_file():
            return result
        from capture_persistence import load_capture_context

        document, root = load_capture_context(path)
        for key in ("session_id", "capture_id", "job_id", "capture_task_log_path", "sidecar_layout"):
            value = document.get(key)
            if isinstance(value, str):
                result[key] = value
        result["output_root"] = str(root)
    except (OSError, UnicodeError, ValueError, TypeError):
        pass
    return result


def job_log_context(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Derive persistent-log identity from the approved worker payload."""
    output_root = payload.get("output_root")
    session_id = payload.get("session_id")
    capture_id = None
    capture_task_log_path = None
    source_capture_job_id = None
    capture_kind = kind in {"capture", "capture_export_retry", "capture_recovery", "known_target_offset", "independent_validation"}
    sidecar_layout = INTERNAL_LAYOUT if capture_kind else LEGACY_LAYOUT
    worker_log_layout = INTERNAL_LAYOUT
    context_path = payload.get("context_path")

    if isinstance(context_path, (str, os.PathLike)):
        metadata = capture_context_metadata(context_path)
        context_candidate = assert_no_reparse_components(Path(context_path).expanduser())
        fallback_root = context_candidate.parent.parent if context_candidate.parent.name == ".internal" else context_candidate.parent
        output_root = metadata.get("output_root") or fallback_root
        session_id = metadata.get("session_id") or session_id
        capture_id = metadata.get("capture_id")
        source_capture_job_id = metadata.get("job_id")
        capture_task_log_path = metadata.get("capture_task_log_path")
        if isinstance(metadata.get("sidecar_layout"), str):
            sidecar_layout = normalize_sidecar_layout(metadata["sidecar_layout"])
    elif kind == "capture_export_retry":
        snapshot_path = payload.get("snapshot_path")
        if isinstance(snapshot_path, (str, os.PathLike)):
            snapshot = assert_no_reparse_components(Path(snapshot_path).expanduser())
            output_root = snapshot.parent.parent if snapshot.parent.name == ".internal" else snapshot.parent
            try:
                from capture_export_recovery import _declared_layout, _load_snapshot

                envelope, document, validated_root, _ = _load_snapshot(snapshot)
                metadata = document.get("capture_metadata")
                persistence = metadata.get("capture_persistence") if isinstance(metadata, Mapping) else None
                output_root = validated_root
                session_id = envelope.get("session_id")
                capture_id = metadata.get("capture_id") if isinstance(metadata, Mapping) else None
                sidecar_layout = _declared_layout(metadata)
                if isinstance(persistence, Mapping):
                    source_capture_job_id = persistence.get("job_id")
                    task_name = (
                        f"{session_id}.{source_capture_job_id or capture_id}.capture-job.jsonl"
                        if isinstance(session_id, str) and isinstance(source_capture_job_id or capture_id, str)
                        else None
                    )
                    if task_name is not None:
                        capture_task_log_path = str(sidecar_path(output_root, task_name, layout=sidecar_layout))
            except (OSError, UnicodeError, ValueError, TypeError, RuntimeError):
                pass

    if output_root is None:
        raise ValueError("worker task has no persistent output directory")
    resolved_root = assert_no_reparse_components(Path(output_root).expanduser()).resolve(strict=False)
    task_log = capture_task_log_path if isinstance(capture_task_log_path, str) else None
    if isinstance(session_id, str):
        session_id = canonical_session_id(session_id)
    if isinstance(source_capture_job_id, str) and isinstance(session_id, str):
        expected_source_log = sidecar_path(
            resolved_root,
            f"{session_id}.{source_capture_job_id}.capture-job.jsonl",
            layout=sidecar_layout,
        )
        if task_log is not None:
            checked_task_log = owned_sidecar_path(
                resolved_root,
                task_log,
                name=f"{session_id}.{source_capture_job_id}.capture-job.jsonl",
                layout=sidecar_layout,
            )
            if checked_task_log != expected_source_log:
                raise ValueError("capture task log path does not match its declared sidecar layout")
            task_log = str(checked_task_log)
        else:
            task_log = str(expected_source_log)
    result = {
        "output_root": str(resolved_root),
        "session_id": session_id if isinstance(session_id, str) else None,
        "capture_id": capture_id if isinstance(capture_id, str) else None,
        "capture_task_log_path": task_log,
        "source_capture_job_id": source_capture_job_id if isinstance(source_capture_job_id, str) else None,
        "context_path": str(Path(context_path).expanduser().resolve()) if isinstance(context_path, (str, os.PathLike)) else None,
        "sidecar_layout": sidecar_layout,
        "worker_log_layout": worker_log_layout,
    }
    return result


class JobLogWriter:
    """Small append-only JSONL writer; no buffered records survive a process."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        job_id: str,
        kind: str,
        actor: str,
        session_id: str | None = None,
        capture_id: str | None = None,
        context_path: str | None = None,
    ):
        # Keep the final path component unresolved so a pre-existing symlink
        # can be rejected before opening it.
        self.path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        self.job_id = job_id
        self.kind = kind
        self.actor = actor
        self.session_id = session_id
        self.capture_id = capture_id
        self.context_path = context_path
        self._sequence = 0
        self._last_fsync = 0.0
        self._last_durable_sequence: int | None = None
        self._errors: list[str] = []
        self._error_lock = threading.Lock()

    @property
    def errors(self) -> list[str]:
        with self._error_lock:
            return list(self._errors)

    @property
    def error(self) -> str | None:
        errors = self.errors
        return errors[0] if errors else None

    @property
    def last_durable_sequence(self) -> int | None:
        return self._last_durable_sequence

    def update_capture_id(self, capture_id: str | None) -> None:
        if isinstance(capture_id, str):
            self.capture_id = capture_id

    def _remember_error(self, operation: str, exc: BaseException) -> None:
        message = f"{operation}: {type(exc).__name__}: {exc}"
        with self._error_lock:
            is_first = not self._errors
            if message not in self._errors:
                self._errors.append(message)
        if is_first:
            self._report_error(message)

    def _report_error(self, message: str) -> None:
        try:
            stream = getattr(sys, "__stderr__", None)
            if stream is not None:
                notice = {"path": str(self.path), "error": message, "job_id": self.job_id}
                stream.write(LOG_ERROR_PREFIX + json.dumps(notice, ensure_ascii=False) + "\n")
                stream.flush()
        except Exception:
            pass

    @staticmethod
    def _write_all(fd: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short write while appending task log")
            view = view[count:]

    def append(
        self,
        event: str,
        *,
        stage: str,
        message: str | None = None,
        frame_id: int | None = None,
        durable: bool = False,
        force_sync: bool = False,
        **fields: Any,
    ) -> bool:
        self._sequence += 1
        document: dict[str, Any] = {
            "schema": "localgaze.worker-task-event/1",
            "event": str(event),
            "timestamp_utc": utc_now(),
            "stage": str(stage),
            "job_id": self.job_id,
            "job_kind": self.kind,
            "actor": self.actor,
            "process_id": os.getpid(),
            "sequence": self._sequence,
            "session_id": self.session_id,
            "capture_id": self.capture_id,
            "frame_id": frame_id,
            "last_confirmed_durable_sequence_by_actor": {self.actor: self._last_durable_sequence},
        }
        if self.context_path:
            document["context_path"] = self.context_path
        if message is not None:
            document["message"] = str(message)
        document.update(fields)
        try:
            now = time.monotonic()
            must_sync = force_sync or event in _FORCE_SYNC_EVENTS or durable
            # Ordinary output is synced at most once per second even while the
            # worker remains in the same stage. Critical records always sync.
            timer_sync = now - self._last_fsync >= 1.0
            should_sync = must_sync or timer_sync
            document["fsync_requested"] = should_sync
            document["fsync_policy"] = "critical" if must_sync else "one_second_throttle" if timer_sync else "buffered_append"
            payload = (json.dumps(document, ensure_ascii=False, separators=(",", ":"), default=str, allow_nan=False) + "\n").encode("utf-8", errors="replace")
            assert_no_reparse_components(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            assert_no_reparse_components(self.path)
            flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            fd = os.open(self.path, flags, 0o600)
            try:
                self._write_all(fd, payload)
                if should_sync:
                    os.fsync(fd)
                    self._last_fsync = now
                    self._last_durable_sequence = self._sequence
            finally:
                os.close(fd)
            return True
        except Exception as exc:
            self._remember_error("append", exc)
            return False


class TeeTextIO:
    """Mirror text to the worker pipe and append it to the durable task log."""

    def __init__(self, stream, writer: JobLogWriter, *, stream_name: str):
        self._stream = stream
        self._writer = writer
        self._stream_name = stream_name
        self._pipe_error_reported = False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None) or "utf-8"

    @property
    def errors(self):
        return getattr(self._stream, "errors", None) or "replace"

    def write(self, text: str) -> int:
        try:
            written = self._stream.write(text)
        except (OSError, ValueError) as exc:
            written = len(text)
            if not self._pipe_error_reported:
                self._pipe_error_reported = True
                self._writer.append(
                    "worker_pipe_error",
                    stage="pipe",
                    message=f"{type(exc).__name__}: {exc}",
                    stream=self._stream_name,
                    force_sync=True,
                )
        if text:
            self._writer.append(
                "worker_output",
                stage="worker_output",
                message=text,
                stream=self._stream_name,
            )
        return written

    def flush(self) -> None:
        try:
            self._stream.flush()
        except (OSError, ValueError) as exc:
            if not self._pipe_error_reported:
                self._pipe_error_reported = True
                self._writer.append(
                    "worker_pipe_error",
                    stage="pipe",
                    message=f"{type(exc).__name__}: {exc}",
                    stream=self._stream_name,
                    force_sync=True,
                )

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._stream.fileno()

