"""Owned worker processes for the desktop view; importing starts no UI/camera."""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent
LOCALGAZE_ROOT = PROJECT_ROOT / "LocalGaze"
if str(LOCALGAZE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCALGAZE_ROOT))

from session_names import canonical_session_id
from capture_paths import (
    INTERNAL_LAYOUT,
    LEGACY_LAYOUT,
    assert_no_reparse_components,
    normalize_sidecar_layout,
    sidecar_path,
)
from EyeEvaluate.Evaluate.analysis_options import (
    normalize_analysis_config,
    normalize_selected_metrics,
    normalize_table_format,
)

from job_logging import LOG_ERROR_PREFIX, JobLogWriter, TeeTextIO, job_log_context, worker_log_path

RESULT_PREFIX = "LOCALGAZE_JOB_RESULT "
PROGRESS_PREFIX = "LOCALGAZE_JOB_PROGRESS "
JOB_KINDS = {"calibrate", "capture", "capture_export_retry", "capture_recovery", "known_target_offset", "independent_validation", "roi", "preprocess", "analyze"}
ANALYSIS_INPUT_SUFFIXES = {".xlsx", ".xlsm", ".csv", ".tsv", ".txt"}
ANALYSIS_BATCH_SUMMARY_NAME = "analysis_batch_summary.xlsx"
ANALYSIS_BATCH_MANIFEST_NAME = "analysis_batch_manifest.json"
ANALYSIS_BATCH_JOURNAL_NAME = "analysis_batch_journal.jsonl"
CALIBRATION_MODEL_NAMES = {"ridge", "elastic_net", "linear_svr", "svr", "tiny_mlp"}
CALIBRATION_FORMULA_VERSIONS = {
    "eyetrax-landmark-features-v1",
    "eyetrax-compact-eye-head-v2",
}
DEFAULT_CALIBRATION_MODEL_NAME = "ridge"
DEFAULT_CALIBRATION_FORMULA_VERSION = "eyetrax-compact-eye-head-v2"


def calibration_export_recovery(directory, session_id):
    """Read-only ownership proof for an interrupted calibration publication."""
    directory = assert_no_reparse_components(Path(directory).expanduser()).resolve(strict=False)
    session_id = canonical_session_id(session_id)
    internal_checkpoint = sidecar_path(
        directory, f"{session_id}.calibration-checkpoint.json", layout=INTERNAL_LAYOUT
    )
    legacy_checkpoint = sidecar_path(
        directory, f"{session_id}.calibration-checkpoint.json", layout=LEGACY_LAYOUT
    )
    internal_exists = internal_checkpoint.exists() or internal_checkpoint.is_symlink()
    legacy_exists = legacy_checkpoint.exists() or legacy_checkpoint.is_symlink()
    if internal_exists and legacy_exists:
        raise ValueError("calibration checkpoint exists in conflicting layouts")
    if not internal_exists and not legacy_exists:
        return None
    checkpoint_path = internal_checkpoint if internal_exists else legacy_checkpoint
    checkpoint_path = assert_no_reparse_components(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError("calibration checkpoint is missing or unsafe")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    try:
        declared_layout = normalize_sidecar_layout(
            checkpoint.get("sidecar_layout") if isinstance(checkpoint, dict) else None
        )
        expected_checkpoint = sidecar_path(directory, f"{session_id}.calibration-checkpoint.json", layout=declared_layout)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration checkpoint sidecar layout is invalid") from exc
    if expected_checkpoint != checkpoint_path:
        raise ValueError("calibration checkpoint path does not match its declared layout")
    attempt = checkpoint.get("export_attempt") if isinstance(checkpoint, dict) else None
    if attempt is None:
        return None
    if (
        checkpoint.get("schema") != "localgaze.calibration-checkpoint/1"
        or checkpoint.get("session_id") != session_id
        or checkpoint.get("status") not in {"in_progress", "incomplete", "samples_complete", "training_complete"}
        or not isinstance(attempt, dict)
        or attempt.get("schema") != "localgaze.calibration-export-attempt/1"
        or attempt.get("session_id") != session_id
    ):
        raise ValueError("calibration export ownership marker is incompatible")
    attempt_id = attempt.get("attempt_id")
    if not isinstance(attempt_id, str) or len(attempt_id) != 8 or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in attempt_id):
        raise ValueError("calibration export attempt identifier is invalid")
    staging = assert_no_reparse_components(directory / f".{session_id}-calibration-{attempt_id}")
    if attempt.get("staging_directory") != str(staging) or staging.resolve() != staging:
        raise ValueError("calibration export staging ownership is invalid")
    entries = attempt.get("outputs")
    expected = [
        assert_no_reparse_components(directory / f"{session_id}{suffix}")
        for suffix in (".pkl", ".xlsx")
    ]
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError("calibration export output ownership is incomplete")
    owned = []
    for path, entry in zip(expected, entries):
        if not isinstance(entry, dict) or entry.get("path") != str(path):
            raise ValueError("calibration export output path is incompatible")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("calibration export content hash is invalid")
        if path.resolve() != path:
            raise ValueError("calibration export output is not an owned regular path")
        if path.exists():
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise FileExistsError("calibration export output no longer matches its ownership marker")
            owned.append(path)
    return checkpoint, owned, staging


def _payload(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind: {kind}")
    if not isinstance(payload, Mapping):
        raise TypeError("job payload must be a mapping")
    allowed = {"source_path", "source_paths", "recursive", "analysis_output_base", "data_path", "model_path", "mapping_table_path", "output_root", "time_unit", "session_id", "roi_path", "correction_path", "target_xy_px", "model_name", "feature_formula_version", "motion_monitor_enabled", "save_motion_keyframes", "snapshot_path", "context_path", "selected_metrics", "analysis_config", "table_format"}
    if set(payload) - allowed:
        raise ValueError("unsupported job payload fields: " + ", ".join(sorted(set(payload) - allowed)))
    if kind == "capture_export_retry":
        if set(payload) != {"snapshot_path"}:
            raise ValueError("capture_export_retry accepts only a snapshot_path; it does not open camera or calibration")
        raw_snapshot = assert_no_reparse_components(Path(payload["snapshot_path"]).expanduser())
        snapshot = raw_snapshot.resolve(strict=True)
        if not snapshot.is_file() or not snapshot.name.endswith(".capture-snapshot.json"):
            raise ValueError("capture_export_retry requires an existing *.capture-snapshot.json file")
        return {"snapshot_path": str(snapshot)}
    if kind == "capture_recovery":
        if set(payload) != {"context_path"}:
            raise ValueError("capture_recovery accepts only a context_path; it does not open a camera")
        raw_context = assert_no_reparse_components(Path(payload["context_path"]).expanduser())
        context_path = raw_context.resolve(strict=True)
        if not context_path.is_file() or not context_path.name.endswith(".capture-context.json"):
            raise ValueError("capture_recovery requires an existing *.capture-context.json file")
        return {"context_path": str(context_path)}
    if "context_path" in payload:
        raise ValueError("context_path is only accepted by capture_recovery")
    if "snapshot_path" in payload:
        raise ValueError("snapshot_path is only accepted by capture_export_retry")
    if "mapping_table_path" in payload and kind != "preprocess":
        raise ValueError("mapping_table_path is only accepted by preprocess")
    if kind != "analyze" and any(name in payload for name in ("source_paths", "recursive", "analysis_output_base")):
        raise ValueError(f"{kind} does not accept batch-analysis inputs")
    if kind == "known_target_offset" and any(payload.get(name) is not None for name in ("source_path", "data_path")):
        raise ValueError("known_target_offset captures its own target window; external source paths are not accepted")
    result = dict(payload)
    if kind == "analyze":
        result["selected_metrics"] = list(normalize_selected_metrics(result.get("selected_metrics")))
        result["analysis_config"] = normalize_analysis_config(result.get("analysis_config"))
        result["table_format"] = normalize_table_format(result.get("table_format"))
        recursive = result.get("recursive", False)
        if not isinstance(recursive, bool):
            raise ValueError("analyze recursive must be a bool")
        result["recursive"] = recursive
        if result.get("analysis_output_base") is not None:
            raw_base = assert_no_reparse_components(Path(result["analysis_output_base"]).expanduser())
            if raw_base.exists() and not raw_base.is_dir():
                raise ValueError("analysis_output_base must be a directory")
            result["analysis_output_base"] = str(raw_base.resolve(strict=False))
        if "data_path" in result:
            raise ValueError("analyze accepts source_path or source_paths, not data_path")
        if "source_paths" in result:
            if "source_path" in result:
                raise ValueError("analyze accepts either source_path or source_paths, not both")
            raw_paths = result["source_paths"]
            if isinstance(raw_paths, (str, bytes, os.PathLike)) or not isinstance(raw_paths, (list, tuple)) or not raw_paths:
                raise ValueError("analyze source_paths must be a nonempty list of files")
            normalized_paths = []
            for raw_path in raw_paths:
                if not isinstance(raw_path, (str, os.PathLike)):
                    raise ValueError("analyze source_paths entries must be file paths")
                candidate = assert_no_reparse_components(Path(raw_path).expanduser())
                if not candidate.is_file() or candidate.suffix.lower() not in ANALYSIS_INPUT_SUFFIXES:
                    raise ValueError(f"analyze source_paths entries must be existing supported data files: {candidate}")
                normalized_paths.append(str(candidate.resolve(strict=True)))
            # Preserve order and duplicates so batch reporting can explicitly mark repeats as skipped.
            result["source_paths"] = normalized_paths
        elif result.get("source_path"):
            candidate = assert_no_reparse_components(Path(result["source_path"]).expanduser())
            if not candidate.exists() or not (candidate.is_file() or candidate.is_dir()):
                raise ValueError("analyze source_path must be an existing data file or folder")
            if candidate.is_file() and candidate.suffix.lower() not in ANALYSIS_INPUT_SUFFIXES:
                raise ValueError(f"analyze source_path has an unsupported data file format: {candidate.suffix}")
            result["source_path"] = str(candidate.resolve(strict=True))
    elif kind in {"capture", "known_target_offset", "independent_validation"}:
        result["table_format"] = normalize_table_format(result.get("table_format"))
    elif any(name in result for name in ("selected_metrics", "analysis_config", "table_format")):
        raise ValueError(f"{kind} does not accept analysis options")
    motion_option_names = ("motion_monitor_enabled", "save_motion_keyframes")
    if kind == "capture":
        for name in motion_option_names:
            value = result.get(name, True)
            if not isinstance(value, bool):
                raise ValueError(f"capture {name} must be a bool")
            result[name] = value
    elif any(name in result for name in motion_option_names):
        raise ValueError(f"{kind} does not accept movement-monitor options")
    if kind == "calibrate":
        model_name = result.get("model_name")
        formula_version = result.get("feature_formula_version")
        model_name = DEFAULT_CALIBRATION_MODEL_NAME if model_name is None else model_name
        formula_version = DEFAULT_CALIBRATION_FORMULA_VERSION if formula_version is None else formula_version
        if not isinstance(model_name, str) or model_name not in CALIBRATION_MODEL_NAMES:
            raise ValueError("calibrate model_name must be a supported estimator type")
        if not isinstance(formula_version, str) or formula_version not in CALIBRATION_FORMULA_VERSIONS:
            raise ValueError("calibrate feature_formula_version must be a supported formula version")
        result["model_name"] = model_name
        result["feature_formula_version"] = formula_version
    elif "model_name" in result or "feature_formula_version" in result:
        raise ValueError(f"{kind} does not accept calibration model/formula overrides")
    if result.get("session_id") is not None:
        result["session_id"] = canonical_session_id(result["session_id"])
    required = {"output_root"}
    if kind in {"calibrate", "capture", "known_target_offset", "independent_validation"}:
        required.add("session_id")
    if kind in {"capture", "known_target_offset", "independent_validation"}:
        required.add("model_path")
    if kind == "preprocess":
        mapping_table_path = result.get("mapping_table_path")
        if mapping_table_path is not None:
            if any(name in result for name in ("source_path", "data_path")) or result.get("model_path") is not None or result.get("session_id") is not None:
                raise ValueError("mapping-table preprocess cannot inherit page-level data, model or session choices")
            if not isinstance(mapping_table_path, (str, os.PathLike)):
                raise ValueError("mapping_table_path must be a CSV/XLSX path")
            raw_table = assert_no_reparse_components(Path(mapping_table_path).expanduser())
            table = raw_table.resolve(strict=True)
            if not table.is_file() or table.suffix.lower() not in {".csv", ".xlsx"}:
                raise ValueError("mapping_table_path must be an existing CSV or XLSX file")
            from pre.mapping_table import read_mapping_table

            rows = read_mapping_table(table)
            if not rows or not any(not row.get("error") for row in rows):
                raise ValueError("mapping table has no valid data rows")
            result["mapping_table_path"] = str(table)
        else:
            if result.get("model_path") is None or result.get("model_path") == "":
                result["model_path"] = None
            elif not isinstance(result.get("model_path"), (str, os.PathLike)):
                raise ValueError("model_path must be an existing model file/folder or None")
            else:
                raw_model = assert_no_reparse_components(Path(result["model_path"]).expanduser())
                model = raw_model.resolve(strict=True)
                if not (model.is_file() or model.is_dir()):
                    raise ValueError("model_path must be an existing model file/folder or None")
                result["model_path"] = str(model)
            if not result.get("source_path"):
                result["source_path"] = result.get("data_path")
            required.add("source_path")
    if kind == "analyze" and not result.get("source_paths"):
        required.add("source_path")
    for key in required:
        if not result.get(key):
            raise ValueError(f"{kind} requires {key}")
    for key in ("source_path", "data_path", "model_path", "mapping_table_path", "output_root", "roi_path", "correction_path"):
        if result.get(key) is not None:
            path = Path(result[key]).expanduser()
            if key == "output_root" and kind in {"capture", "known_target_offset", "independent_validation", "calibrate"}:
                path = assert_no_reparse_components(path)
            result[key] = str(path.resolve())
    if Path(result["output_root"]).exists() and not Path(result["output_root"]).is_dir():
        raise ValueError("output_root must be a directory")
    if result.get("time_unit") not in {None, "seconds", "milliseconds", "microseconds"}:
        raise ValueError("time_unit must be seconds, milliseconds or microseconds")
    if kind == "preprocess" and result.get("time_unit") is None:
        raise ValueError("preprocess requires an explicit numeric time_unit; use seconds for elapsed_s")
    if kind == "preprocess" and not result.get("mapping_table_path"):
        source = Path(result["source_path"])
        if source.is_symlink() or not source.exists() or not (source.is_file() or source.is_dir()):
            raise ValueError("preprocess source_path must be an existing file or folder")
    if kind in {"capture", "known_target_offset", "independent_validation"}:
        model = Path(result["model_path"])
        if not model.is_file() or model.suffix.lower() != ".pkl":
            raise ValueError(f"{kind} model_path must be an existing .pkl model")
    if kind == "known_target_offset":
        if result.get("correction_path") is None:
            result["correction_path"] = str(sidecar_path(
                Path(result["output_root"]),
                f"{result['session_id']}.offset-correction.json",
                layout=INTERNAL_LAYOUT,
                create_parent=True,
            ))
        correction = Path(result["correction_path"])
        if correction.suffix.lower() != ".json":
            raise ValueError("known_target_offset correction_path must end in .json")
        if correction.exists():
            raise FileExistsError("correction_path already exists; choose a new path")
        target = result.get("target_xy_px")
        if target is not None:
            if not isinstance(target, (list, tuple)) or len(target) != 2:
                raise ValueError("target_xy_px must contain two coordinates")
            normalized = []
            for coordinate in target:
                if isinstance(coordinate, bool):
                    raise ValueError("target_xy_px must contain finite numbers")
                try:
                    value = float(coordinate)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("target_xy_px must contain finite numbers") from exc
                if not math.isfinite(value):
                    raise ValueError("target_xy_px must contain finite numbers")
                normalized.append(value)
            result["target_xy_px"] = normalized
    if kind in {"calibrate", "capture", "known_target_offset", "independent_validation"}:
        directory = Path(result["output_root"])
        recovery = calibration_export_recovery(directory, result["session_id"]) if kind == "calibrate" else None
        session = result["session_id"]
        if kind == "calibrate":
            outputs = [directory / f"{session}.pkl", directory / f"{session}.xlsx"]
        elif kind == "independent_validation":
            internal = directory / ".internal"
            outputs = [directory / f"{session}.csv", directory / f"{session}.txt", directory / f"{session}.xlsx"]
            outputs.extend(internal / f"{session}{suffix}" for suffix in (
                ".manifest.json",
                ".capture-snapshot.json",
                ".independent-validation.json",
            ))
            outputs.extend(directory / f"{session}{suffix}" for suffix in (
                ".manifest.json",
                ".capture-snapshot.json",
                ".independent-validation.json",
            ))
        else:
            internal = directory / ".internal"
            suffixes = (
                (".xlsx", ".offset-correction.json")
                if kind == "known_target_offset"
                else (".xlsx",)
            )
            outputs = [directory / f"{session}.csv", directory / f"{session}.txt"]
            outputs.extend(directory / f"{session}{suffix}" for suffix in suffixes)
            outputs.extend(
                internal / f"{session}{suffix}"
                for suffix in ((".manifest.json", ".offset-correction.json") if kind == "known_target_offset" else (".manifest.json", ".capture-snapshot.json"))
            )
            outputs.extend(
                directory / f"{session}{suffix}"
                for suffix in ((".manifest.json", ".offset-correction.json") if kind == "known_target_offset" else (".manifest.json", ".capture-snapshot.json"))
            )
        if kind != "independent_validation" and recovery is None and any(path.exists() for path in outputs):
            raise FileExistsError("session outputs already exist; choose a new session_id")
    json.dumps(result, allow_nan=False)
    return result


CAPTURE_JOB_KINDS = {"capture", "capture_export_retry", "capture_recovery"}


def _allocate_preprocess_run_root(base_root: str | os.PathLike[str], job_id: str) -> Path:
    """Create a unique direct-preprocess output directory beneath the UI root."""

    base = assert_no_reparse_components(Path(base_root).expanduser())
    if base.exists() and not base.is_dir():
        raise ValueError("preprocess output root must be a directory")
    base.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for attempt in range(10):
        suffix = job_id[:8] if attempt == 0 else uuid.uuid4().hex[:8]
        candidate = base / f"preprocessing_{timestamp}_{suffix}"
        assert_no_reparse_components(candidate)
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate.resolve(strict=True)
    raise FileExistsError(f"Unable to allocate a unique preprocessing run under {base}")


def _allocate_analysis_batch_root(base_root: str | os.PathLike[str], job_id: str) -> Path:
    """Create a unique run folder for batch outputs and cancellation state."""

    base = assert_no_reparse_components(Path(base_root).expanduser())
    if base.exists() and not base.is_dir():
        raise ValueError("analysis output root must be a directory")
    base.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for attempt in range(10):
        suffix = job_id[:8] if attempt == 0 else uuid.uuid4().hex[:8]
        candidate = base / f"analysis_batch_{timestamp}_{suffix}"
        assert_no_reparse_components(candidate)
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate.resolve(strict=True)
    raise FileExistsError(f"Unable to allocate a unique analysis batch under {base}")


def _analysis_batch_requested(payload: Mapping[str, Any]) -> bool:
    if payload.get("source_paths"):
        return True
    source = payload.get("source_path")
    return bool(source and Path(source).is_dir())


def _analysis_cancel_marker(output_root: str | os.PathLike[str], job_id: str) -> Path:
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Fa-f0-9]{32}", job_id):
        raise ValueError("analysis batch job id is invalid")
    return sidecar_path(
        Path(output_root).expanduser(),
        f"analysis-batch-{job_id}.cancel",
        layout=INTERNAL_LAYOUT,
        create_parent=True,
    )


def _first_report_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None
CAPTURE_RECOVERY_JOB_KINDS = {"capture_export_retry", "capture_recovery"}


def _capture_job_identity_fields(kind, current_job_id, source_job_id):
    """Keep source-capture identity separate from this owned worker attempt."""
    fields = {"current_job_id": current_job_id}
    if isinstance(source_job_id, str) and source_job_id:
        fields.update(
            source_job_id=source_job_id,
            source_capture_job_id=source_job_id,
            capture_job_id=source_job_id,
        )
    if kind in CAPTURE_RECOVERY_JOB_KINDS:
        fields["recovery_job_id"] = current_job_id
    return fields


def _nested_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        if field in value and value[field] is not None:
            return value[field]
        for nested in value.values():
            found = _nested_field(nested, field)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _nested_field(nested, field)
            if found is not None:
                return found
    return None


def _capture_state_for_job(kind, payload, job_id, details, log_context):
    """Recover the last checkpoint only from capture-owned durable metadata."""
    state = {
        "session_id": log_context.get("session_id"),
        "capture_id": log_context.get("capture_id"),
        "context_path": log_context.get("context_path"),
        "capture_task_log_path": log_context.get("capture_task_log_path"),
        "output_root": log_context.get("output_root"),
        "sidecar_layout": log_context.get("sidecar_layout", LEGACY_LAYOUT),
        "last_durable_checkpoint": _nested_field(details, "last_durable_checkpoint"),
        "checkpoint_error": None,
        "source_capture_job_id": log_context.get("source_capture_job_id"),
    }
    candidate = None
    for key in ("context_path", "capture_context_path"):
        value = details.get(key)
        if isinstance(value, (str, os.PathLike)):
            candidate = Path(value).expanduser()
            break
    if candidate is None and isinstance(payload.get("context_path"), str):
        candidate = Path(payload["context_path"])
    if candidate is None and kind == "capture_export_retry":
        snapshot_path = payload.get("snapshot_path")
        if isinstance(snapshot_path, str):
            try:
                snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
                candidate_value = _nested_field(snapshot, "context_path")
                if isinstance(candidate_value, str):
                    candidate = Path(candidate_value)
                    if not candidate.is_absolute():
                        candidate = Path(snapshot_path).parent / candidate
            except (OSError, UnicodeError, ValueError, TypeError):
                pass
    if candidate is None and kind == "capture":
        root_text = payload.get("output_root")
        session_id = payload.get("session_id")
        if isinstance(root_text, str) and isinstance(session_id, str):
            root = Path(root_text) / ".internal"
            try:
                for path in root.iterdir():
                    if path.is_file() and path.name.startswith(session_id + ".") and path.name.endswith(".capture-context.json"):
                        try:
                            from capture_persistence import load_capture_context
                            context, _ = load_capture_context(path)
                        except Exception:
                            continue
                        if context.get("job_id") == job_id:
                            candidate = path
                            break
            except OSError:
                pass

    if candidate is not None:
        try:
            if not candidate.is_absolute():
                root_text = details.get("output_root") or log_context.get("output_root")
                if isinstance(root_text, (str, os.PathLike)):
                    candidate = Path(root_text).expanduser() / candidate
            from capture_persistence import load_capture_context, read_checkpoints
            context, root = load_capture_context(candidate)
            checkpoint, _, _ = read_checkpoints(root / context["checkpoint_path"], context)
            state.update({
                "session_id": context.get("session_id"),
                "capture_id": context.get("capture_id"),
                "context_path": str(Path(candidate).resolve()),
                "capture_task_log_path": str(root / context["capture_task_log_path"]),
                "source_capture_job_id": context.get("job_id"),
                "output_root": str(root),
                "sidecar_layout": context.get("sidecar_layout", LEGACY_LAYOUT),
            })
            if checkpoint is not None:
                state["last_durable_checkpoint"] = checkpoint
        except Exception as exc:
            state["checkpoint_error"] = f"{type(exc).__name__}: {exc}"

    if state["capture_task_log_path"] is None:
        task_log = details.get("capture_task_log_path") or details.get("job_log_path")
        if isinstance(task_log, (str, os.PathLike)):
            task_log_path = Path(task_log).expanduser()
            if not task_log_path.is_absolute():
                task_log_path = Path(log_context["output_root"]) / task_log_path
            state["capture_task_log_path"] = str(task_log_path.resolve())
    if state["capture_id"] is None:
        capture_id = details.get("capture_id") or _nested_field(details, "capture_id")
        if isinstance(capture_id, str):
            state["capture_id"] = capture_id
    state["source_job_id"] = state.get("source_capture_job_id")
    state["capture_job_id"] = state.get("source_capture_job_id")
    return state


def _append_capture_parent_event(output_root, session_id, job_id, event, *, stage, capture_id=None, frame_id=None, layout=INTERNAL_LAYOUT, **fields):
    from capture_persistence import append_capture_task_event
    return append_capture_task_event(
        output_root,
        session_id,
        job_id,
        event,
        stage=stage,
        layout=layout,
        capture_id=capture_id,
        frame_id=frame_id,
        **fields,
    )


class JobManager:
    """One explicitly started owned process; all methods are nonblocking.

    The Tk thread calls start/poll/cancel/shutdown. A private reader thread
    handles pipe I/O only; it never calls Tk or controls another process.
    """
    def __init__(self, python_executable: str | None = None):
        self.python_executable = str(python_executable or sys.executable)
        self.state = "idle"
        self.job_id: str | None = None
        self._process = None
        self._reader_done = threading.Event()
        self._reader_done.set()
        self._messages = deque(maxlen=512)
        self._lock = threading.Lock()
        self._result = None
        self._dropped_logs = 0
        self._progress_sequence = 0
        self._cancel_requested = False
        self._cancel_at = None
        self._cancel_marker_path: Path | None = None
        self._closing = False
        self._pending = []
        self._log_context: dict[str, Any] | None = None
        self._worker_log_path: Path | None = None
        self._parent_log_writer: JobLogWriter | None = None
        self._log_io_errors: list[str] = []
        self._worker_log_io_errors: list[str] = []
        self._capture_task_log_io_errors: list[str] = []
        self._capture_task_log_path: Path | None = None

    @property
    def busy(self) -> bool:
        return self._process is not None

    @property
    def closed(self) -> bool:
        return self._closing and not self.busy and self._reader_done.is_set()

    def start(self, kind: str, payload: Mapping[str, Any]) -> str:
        if self._closing:
            raise ValueError("job manager is closing")
        if self.busy:
            raise ValueError("an owned job is already running")
        prepared = _payload(kind, payload)
        job_id = uuid.uuid4().hex
        if kind == "preprocess" and not prepared.get("mapping_table_path"):
            # Legacy direct preprocessing keeps its exact output-path contract.
            # The desktop worker gets a fresh child so fixed report and
            # manifest names remain safe across repeated GUI runs.
            prepared["output_root"] = str(_allocate_preprocess_run_root(prepared["output_root"], job_id))
        if kind == "analyze" and _analysis_batch_requested(prepared):
            analysis_output_base = prepared["output_root"]
            prepared["output_root"] = str(_allocate_analysis_batch_root(analysis_output_base, job_id))
            prepared["analysis_output_base"] = str(Path(analysis_output_base).resolve(strict=False))
            self._cancel_marker_path = _analysis_cancel_marker(prepared["output_root"], job_id)
        else:
            self._cancel_marker_path = None
        self._log_context = None
        self._worker_log_path = None
        self._parent_log_writer = None
        self._capture_task_log_path = None
        self._log_io_errors = []
        self._worker_log_io_errors = []
        self._capture_task_log_io_errors = []
        try:
            self._log_context = {**job_log_context(kind, prepared), "kind": kind, "payload": prepared}
            self._worker_log_path = worker_log_path(
                self._log_context["output_root"],
                self._log_context.get("session_id"),
                job_id,
                sidecar_layout=self._log_context.get("worker_log_layout", INTERNAL_LAYOUT),
            )
            self._parent_log_writer = JobLogWriter(
                self._worker_log_path,
                job_id=job_id,
                kind=kind,
                actor="desktop_parent",
                session_id=self._log_context.get("session_id"),
                capture_id=self._log_context.get("capture_id"),
                context_path=self._log_context.get("context_path"),
            )
            self._parent_log_writer.append(
                "desktop_job_started",
                stage="starting",
                force_sync=True,
                worker_log_path=str(self._worker_log_path),
            )
            if self._parent_log_writer.errors:
                self._log_io_errors.extend(self._parent_log_writer.errors)
                self._worker_log_io_errors.extend(self._parent_log_writer.errors)
            if kind in CAPTURE_JOB_KINDS and self._log_context.get("session_id"):
                root = Path(self._log_context["output_root"])
                self._capture_task_log_path = sidecar_path(
                    root,
                    f"{self._log_context['session_id']}.{job_id}.capture-job.jsonl",
                    layout=INTERNAL_LAYOUT,
                    create_parent=True,
                )
                source_job_id = self._log_context.get("source_capture_job_id")
                identity_fields = _capture_job_identity_fields(kind, job_id, source_job_id)
                _append_capture_parent_event(
                    root,
                    self._log_context["session_id"],
                    job_id,
                    "desktop_job_started",
                    stage="starting",
                    capture_id=self._log_context.get("capture_id"),
                    layout=INTERNAL_LAYOUT,
                    worker_log_path=str(self._worker_log_path),
                    related_capture_job_id=self._log_context.get("source_capture_job_id"),
                    **identity_fields,
                )
        except Exception as exc:
            error = f"日志准备失败：{type(exc).__name__}: {exc}"
            self._log_io_errors.append(error)
            if self._capture_task_log_path is not None:
                self._capture_task_log_io_errors.append(error)
            else:
                self._worker_log_io_errors.append(error)
        arguments = ["--worker", kind, "--job-id", job_id, "--payload-json", json.dumps(prepared, ensure_ascii=False)]
        if getattr(sys, "frozen", False):
            worker = Path(sys.executable).resolve().parent / "LocalGazeWorker.exe"
            if not worker.is_file():
                raise FileNotFoundError(f"packaged worker is missing: {worker}; keep the complete application folder")
            command = [str(worker), *arguments]
        else:
            command = [self.python_executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), *arguments]
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        env["LOCALGAZE_JOB_ID"] = job_id
        if self._worker_log_path is not None:
            env["LOCALGAZE_WORKER_LOG_PATH"] = str(self._worker_log_path)
        try:
            process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace",
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        except Exception as exc:
            if self._parent_log_writer is not None:
                self._parent_log_writer.append(
                    "parent_start_failed",
                    stage="failed",
                    message=f"{type(exc).__name__}: {exc}",
                    force_sync=True,
                )
            if self._capture_task_log_path is not None and self._log_context and self._log_context.get("session_id"):
                try:
                    _append_capture_parent_event(
                        self._log_context["output_root"],
                        self._log_context["session_id"],
                        job_id,
                        "worker_process_start_failed",
                        stage="failed",
                        capture_id=self._log_context.get("capture_id"),
                        layout=INTERNAL_LAYOUT,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                except Exception as log_exc:
                    error = f"capture task log: {type(log_exc).__name__}: {log_exc}"
                    self._log_io_errors.append(error)
                    self._capture_task_log_io_errors.append(error)
            self._cleanup_cancel_marker()
            raise
        self._process, self.job_id = process, job_id
        self.state = "running"
        self._cancel_requested = False
        self._cancel_at = None
        with self._lock:
            self._messages.clear()
            self._result = None
            self._dropped_logs = 0
            self._progress_sequence = 0
        self._reader_done.clear()
        context = self._log_context or {}
        started_details = {
            "kind": kind,
            "session_id": context.get("session_id") or prepared.get("session_id"),
            "output_root": context.get("output_root") or prepared.get("output_root"),
            "snapshot_path": prepared.get("snapshot_path"),
            "context_path": prepared.get("context_path"),
            "worker_log_path": str(self._worker_log_path) if self._worker_log_path else None,
            "capture_task_log_path": str(self._capture_task_log_path) if self._capture_task_log_path else context.get("capture_task_log_path"),
            "source_capture_task_log_path": context.get("capture_task_log_path"),
            "log_io_errors": list(self._log_io_errors),
        }
        if kind in CAPTURE_JOB_KINDS:
            started_details.update(_capture_job_identity_fields(kind, job_id, context.get("source_capture_job_id")))
        self._pending = [dict(job_id=job_id, type="started", message=f"{kind} started", returncode=None, outputs={}, details=started_details)]
        threading.Thread(target=self._read, args=(process, job_id), name="desktop-job-pipe", daemon=True).start()
        return job_id

    def _read(self, process, job_id):
        try:
            for line in process.stdout:
                text = line.rstrip("\r\n")
                if text.startswith(LOG_ERROR_PREFIX):
                    try:
                        notice = json.loads(text[len(LOG_ERROR_PREFIX):])
                    except (TypeError, ValueError):
                        notice = {"error": text[len(LOG_ERROR_PREFIX):]}
                    error = str(notice.get("error") or "持久日志写入失败") if isinstance(notice, dict) else str(notice)
                    with self._lock:
                        if job_id != self.job_id:
                            continue
                        wrapped_error = f"worker: {error}"
                        self._log_io_errors.append(wrapped_error)
                        self._worker_log_io_errors.append(wrapped_error)
                        self._messages.append(dict(
                            job_id=job_id,
                            type="log_error",
                            message=f"持久任务日志写入失败：{error}",
                            returncode=None,
                            outputs={},
                            details=notice if isinstance(notice, dict) else {"error": error},
                        ))
                    continue
                if text.startswith(PROGRESS_PREFIX):
                    try:
                        candidate_progress = json.loads(text[len(PROGRESS_PREFIX):])
                    except (TypeError, ValueError):
                        candidate_progress = None
                    if (
                        not isinstance(candidate_progress, dict)
                        or candidate_progress.get("schema") != "localgaze.job-progress/1"
                        or candidate_progress.get("type") != "capture_quality"
                        or candidate_progress.get("job_id") != job_id
                    ):
                        # Progress belongs to a structured channel; a stale or
                        # malformed record must never update a later job's UI.
                        continue
                    sequence = candidate_progress.get("sequence")
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= self._progress_sequence:
                        continue
                    self._progress_sequence = sequence
                    with self._lock:
                        if job_id != self.job_id:
                            continue
                        self._messages.append(dict(
                            job_id=job_id,
                            type="progress",
                            message="capture quality update",
                            returncode=None,
                            outputs={},
                            details=candidate_progress,
                        ))
                    continue
                result = None
                if text.startswith(RESULT_PREFIX):
                    try:
                        candidate = json.loads(text[len(RESULT_PREFIX):])
                        if isinstance(candidate, dict) and candidate.get("job_id") == job_id:
                            result = candidate
                    except (TypeError, ValueError):
                        pass
                with self._lock:
                    if job_id != self.job_id:
                        continue
                    if result is not None:
                        self._result = result
                    else:
                        if len(self._messages) == self._messages.maxlen:
                            self._dropped_logs += 1
                        self._messages.append(dict(job_id=job_id, type="log", message=text[:8192], returncode=None, outputs={}, details={}))
        except (OSError, ValueError) as exc:
            with self._lock:
                self._messages.append(dict(job_id=job_id, type="log", message=f"worker pipe: {exc}", returncode=None, outputs={}, details={}))
        finally:
            if process.stdout is not None:
                process.stdout.close()
            self._reader_done.set()

    def poll(self) -> list[dict[str, Any]]:
        current = self.job_id
        events, self._pending = self._pending, []
        with self._lock:
            events.extend(self._messages)
            self._messages.clear()
            if self._dropped_logs:
                events.append(dict(job_id=current, type="log", message=f"{self._dropped_logs} older log lines omitted", returncode=None, outputs={}, details={"truncated_log_lines": self._dropped_logs}))
                self._dropped_logs = 0
            result = self._result
        process = self._process
        if process is not None:
            returncode = process.poll()
            if returncode is None and self._cancel_requested and self._cancel_at is not None and time.monotonic() - self._cancel_at >= 2.:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            if returncode is not None and self._reader_done.is_set():
                # The pipe reader can finish after the first lock was released.
                # Re-read its final result only after observing its done event.
                with self._lock:
                    result = self._result
                outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
                details = result.get("details", {}) if isinstance(result, dict) else {}
                if not isinstance(outputs, dict):
                    outputs = {}
                if not isinstance(details, dict):
                    details = {}
                else:
                    details = dict(details)
                if self._cancel_requested:
                    terminal, message = "cancelled", "owned worker exited after cancellation; any partial outputs are retained"
                    details = {**details, "partial_outputs_possible": True}
                    context = self._log_context or {}
                    payload = context.get("payload")
                    if (
                        context.get("kind") == "analyze"
                        and isinstance(payload, Mapping)
                        and _analysis_batch_requested(payload)
                    ):
                        try:
                            batch_finalization = _finalize_interrupted_analysis_batch(
                                payload,
                                job_id=current,
                                cancel_marker_path=self._cancel_marker_path,
                                worker_exitcode=returncode,
                            )
                            if batch_finalization is not None:
                                outputs = batch_finalization["outputs"]
                                details.update(batch_finalization["details"])
                                message = "owned worker exited after cancellation; completed outputs were retained and the batch record was finalized"
                            else:
                                details["analysis_batch_finalization"] = "no matching durable batch state"
                        except Exception as exc:
                            error = f"analysis batch cancellation finalization: {type(exc).__name__}: {exc}"
                            details["analysis_batch_finalization_error"] = error
                            with self._lock:
                                self._log_io_errors.append(error)
                elif returncode != 0:
                    terminal, message = "failed", result.get("message", f"worker exited with code {returncode}") if isinstance(result, dict) else f"worker exited with code {returncode}"
                elif not isinstance(result, dict) or result.get("status") not in {"completed", "failed", "cancelled"}:
                    terminal, message = "failed", "worker exited without a valid final result"
                else:
                    terminal = result["status"]
                    message = result.get("message", terminal)
                checkpoint = None
                capture_state = {}
                if self._log_context and self._log_context.get("kind") in CAPTURE_JOB_KINDS:
                    capture_state = _capture_state_for_job(
                        self._log_context["kind"],
                        self._log_context.get("payload", {}),
                        current,
                        details,
                        self._log_context,
                    )
                    checkpoint = capture_state.get("last_durable_checkpoint")
                else:
                    checkpoint = _nested_field(details, "last_durable_checkpoint")
                if self._parent_log_writer is not None:
                    self._parent_log_writer.append(
                        "parent_terminal",
                        stage=terminal,
                        message=message,
                        exitcode=returncode,
                        task_status=terminal,
                        last_durable_checkpoint=checkpoint,
                        force_sync=True,
                    )
                    if self._parent_log_writer.errors:
                        self._log_io_errors.extend(self._parent_log_writer.errors)
                        self._worker_log_io_errors.extend(self._parent_log_writer.errors)
                if capture_state and self._capture_task_log_path is not None and self._log_context:
                    session_id = capture_state.get("session_id") or self._log_context.get("session_id")
                    source_job_id = capture_state.get("source_capture_job_id") or capture_state.get("source_job_id")
                    identity_fields = _capture_job_identity_fields(
                        self._log_context.get("kind"), current, source_job_id
                    )
                    try:
                        capture_event_path = _append_capture_parent_event(
                            self._log_context["output_root"],
                            session_id,
                            current,
                            "worker_process_exited",
                            stage=terminal,
                            capture_id=capture_state.get("capture_id"),
                            layout=INTERNAL_LAYOUT,
                            frame_id=checkpoint.get("last_frame_id") if isinstance(checkpoint, dict) else None,
                            exitcode=returncode,
                            task_status=terminal,
                            last_durable_checkpoint=checkpoint,
                            worker_log_path=str(self._worker_log_path) if self._worker_log_path else None,
                            **identity_fields,
                        )
                        self._capture_task_log_path = capture_event_path
                    except Exception as exc:
                        error = f"capture task log: {type(exc).__name__}: {exc}"
                        self._log_io_errors.append(error)
                        self._capture_task_log_io_errors.append(error)
                    if (
                        self._log_context.get("kind") in CAPTURE_RECOVERY_JOB_KINDS
                        and isinstance(source_job_id, str)
                        and source_job_id != current
                    ):
                        source_task_log = capture_state.get("capture_task_log_path")
                        source_root = Path(capture_state.get("output_root") or self._log_context["output_root"])
                        try:
                            _append_capture_parent_event(
                                source_root,
                                session_id,
                                source_job_id,
                                "worker_process_exited",
                                stage=terminal,
                                capture_id=capture_state.get("capture_id"),
                                layout=capture_state.get("sidecar_layout", LEGACY_LAYOUT),
                                frame_id=checkpoint.get("last_frame_id") if isinstance(checkpoint, dict) else None,
                                exitcode=returncode,
                                task_status=terminal,
                                last_durable_checkpoint=checkpoint,
                                worker_log_path=str(self._worker_log_path) if self._worker_log_path else None,
                                source_capture_job_id=source_job_id,
                                source_job_id=source_job_id,
                                capture_job_id=source_job_id,
                                recovery_job_id=current,
                                current_job_id=current,
                                recovery_task_log_path=str(self._capture_task_log_path),
                            )
                        except Exception as exc:
                            error = f"source capture task log: {type(exc).__name__}: {exc}"
                            self._log_io_errors.append(error)
                            self._capture_task_log_io_errors.append(error)
                if self._worker_log_path is not None:
                    details["worker_log_path"] = str(self._worker_log_path)
                    details["parent_log_last_durable_sequence"] = (
                        self._parent_log_writer.last_durable_sequence if self._parent_log_writer else None
                    )
                if self._capture_task_log_path is not None:
                    details["capture_task_log_path"] = str(self._capture_task_log_path)
                if capture_state:
                    details["last_durable_checkpoint"] = checkpoint
                    details["context_path"] = capture_state.get("context_path") or details.get("context_path")
                    details["capture_id"] = capture_state.get("capture_id") or details.get("capture_id")
                    details.update(_capture_job_identity_fields(
                        self._log_context.get("kind") if self._log_context else None,
                        current,
                        capture_state.get("source_capture_job_id") or capture_state.get("source_job_id"),
                    ))
                    source_log = capture_state.get("capture_task_log_path")
                    if source_log and str(self._capture_task_log_path) != str(source_log):
                        details["source_capture_task_log_path"] = source_log
                    if capture_state.get("checkpoint_error"):
                        details["checkpoint_error"] = capture_state["checkpoint_error"]
                with self._lock:
                    log_errors = list(self._log_io_errors)
                worker_errors = details.get("worker_log_io_errors")
                if isinstance(worker_errors, list):
                    log_errors.extend(str(error) for error in worker_errors)
                    self._worker_log_io_errors.extend(str(error) for error in worker_errors)
                else:
                    worker_error = details.get("worker_log_error")
                    if worker_error:
                        log_errors.append(str(worker_error))
                        self._worker_log_io_errors.append(str(worker_error))
                capture_errors = details.get("log_io_errors")
                if isinstance(capture_errors, dict):
                    self._capture_task_log_io_errors.extend(f"{key}: {value}" for key, value in capture_errors.items() if value)
                    log_errors.extend(f"capture {key}: {value}" for key, value in capture_errors.items() if value)
                elif isinstance(capture_errors, list):
                    self._capture_task_log_io_errors.extend(str(error) for error in capture_errors)
                    log_errors.extend(str(error) for error in capture_errors)
                elif isinstance(capture_errors, str) and capture_errors:
                    self._capture_task_log_io_errors.append(capture_errors)
                    log_errors.append(capture_errors)
                details["log_io_errors"] = list(dict.fromkeys(log_errors))
                worker_log_errors = list(dict.fromkeys(self._worker_log_io_errors))
                capture_task_errors = list(dict.fromkeys(self._capture_task_log_io_errors))
                details["worker_log_io_errors"] = worker_log_errors
                details["worker_log_status"] = "failed" if worker_log_errors and self._worker_log_path else "ok" if self._worker_log_path else "unavailable"
                details["capture_task_log_io_errors"] = capture_task_errors
                details["capture_task_log_status"] = "failed" if capture_task_errors else "ok" if self._capture_task_log_path else "not_applicable"
                details["log_status"] = "error" if log_errors else "ok" if self._worker_log_path else "unavailable"
                details["worker_exitcode"] = returncode
                self.state = terminal
                self._process = None
                self._cleanup_cancel_marker()
                events.append(dict(job_id=current, type=terminal, message=message, returncode=returncode, outputs=outputs, details=details))
        return [event for event in events if event.get("job_id") == current]

    def cancel(self) -> bool:
        if self._process is None or self._process.poll() is not None:
            return False
        if self._cancel_requested:
            return True
        self._cancel_requested = True
        self._cancel_at = time.monotonic()
        self.state = "cancelling"
        if self._cancel_marker_path is not None:
            try:
                marker = assert_no_reparse_components(self._cancel_marker_path)
                marker.parent.mkdir(parents=True, exist_ok=True)
                with marker.open("x", encoding="ascii", newline="\n") as stream:
                    stream.write(f"{self.job_id}\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass
            except (OSError, ValueError):
                # A missing cooperative signal must still fall back to the
                # established owned-process termination path.
                try:
                    self._process.terminate()
                except ProcessLookupError:
                    pass
        else:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass
        return True

    def _cleanup_cancel_marker(self) -> None:
        marker, self._cancel_marker_path = self._cancel_marker_path, None
        if marker is None:
            return
        try:
            safe = assert_no_reparse_components(marker)
            if safe.is_file() and not safe.is_symlink():
                safe.unlink()
        except (OSError, ValueError):
            pass

    def shutdown(self) -> None:
        self._closing = True
        self.cancel()


def _roi_job(payload):
    """Track writes performed by this app, rather than scanning others' files."""
    # Package-qualified imports keep the ROI globals separate from camera globals.
    from ROIDrawing.roi_drawer import ROIApp
    import tkinter as tk
    root = tk.Tk()
    outputs = {}
    try:
        app = ROIApp(root)
        original_writer = app._write_workbook_exclusive
        def write_owned(path, first, metadata):
            original_writer(path, first, metadata)
            outputs["roi_workbook"] = str(Path(path).resolve())
        app._write_workbook_exclusive = write_owned
        # The old save callback accepts its base directory from sys.argv.
        sys.argv = [sys.argv[0], payload["output_root"]]
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return dict(status="completed" if outputs else "cancelled", outputs=outputs, details={"reason": "roi_saved" if outputs else "roi_closed_without_save"})


def _analysis_path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _is_analysis_output_directory(path: Path) -> bool:
    """Recognize output folders created by LocalGaze before walking them."""

    if path.name.casefold().startswith(("analysis_batch_", "preprocessing_")):
        return True
    if (path / ANALYSIS_BATCH_MANIFEST_NAME).is_file() or (path / "preprocess_manifest.json").is_file():
        return True
    internal = path / ".internal"
    try:
        internal = assert_no_reparse_components(internal)
    except (OSError, ValueError):
        return False
    if internal.is_dir():
        try:
            for manifest in internal.iterdir():
                if not manifest.is_file() or not manifest.name.endswith("_manifest.json"):
                    continue
                stem = manifest.name[:-len("_manifest.json")]
                if (path / f"{stem}_summary.xlsx").is_file():
                    return True
        except OSError:
            return False
    return False


def _is_analysis_export_artifact(path: Path) -> bool:
    name = path.name.casefold()
    if name in {ANALYSIS_BATCH_SUMMARY_NAME, ANALYSIS_BATCH_MANIFEST_NAME.casefold(), ANALYSIS_BATCH_JOURNAL_NAME.casefold()}:
        return True
    if name == "preprocess_report.csv" or name.endswith("_report.csv"):
        return True
    if name.endswith(("_summary.xlsx", "_metadata.txt", "_metrics.csv", "_metrics.txt", "_samples.csv", "_samples.txt", "_events.csv", "_events.txt", "_roi_hits.csv", "_roi_hits.txt")):
        return True
    # Trust a declared analysis-export manifest to identify generated tables,
    # while leaving capture CSV/TXT input files eligible for analysis.
    manifest_stems = [path.stem]
    if name.endswith(("_markers.csv", "_markers.txt")) and path.stem.casefold().endswith("_markers"):
        # Marker tables share the session export manifest; confirm the exact
        # declared output path so ordinary files with similar names remain valid.
        manifest_stems.append(path.stem[:-len("_markers")])
    candidates = tuple(
        candidate
        for stem in manifest_stems
        for candidate in (
            path.parent / ".internal" / f"{stem}_manifest.json",
            path.parent / f"{stem}_manifest.json",
        )
    )
    for manifest_path in candidates:
        try:
            manifest_path = assert_no_reparse_components(manifest_path)
            if not manifest_path.is_file():
                continue
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or not str(document.get("schema", "")).startswith("LocalGaze.analysis-export/"):
            continue
        outputs = document.get("outputs")
        if isinstance(outputs, dict):
            target = str(path.resolve(strict=False))
            for entry in outputs.values():
                if isinstance(entry, dict) and entry.get("path"):
                    try:
                        if str(Path(entry["path"]).resolve(strict=False)) == target:
                            return True
                    except (OSError, TypeError, ValueError):
                        continue
    return False


def _is_mapping_template(path: Path) -> bool:
    if "mapping_template" in path.stem.casefold() or "mapping-template" in path.stem.casefold() or "映射模板" in path.stem:
        return True
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = csv.reader(stream)
                header = next(rows, None)
                if header is None or [value.strip().casefold() for value in header] != ["data_path", "model_path", "session_id"]:
                    return False
                return not any(any(value.strip() for value in row) for row in rows)
        except (OSError, UnicodeError, csv.Error):
            return False
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            import pandas as pd

            with pd.ExcelFile(path) as excel:
                names = set(excel.sheet_names)
            if not {"Input", "Example", "Instructions"}.issubset(names):
                return False
            header = pd.read_excel(path, sheet_name="Input", nrows=0)
            return [str(value).strip().casefold() for value in header.columns] == ["data_path", "model_path", "session_id"]
        except Exception:
            return False
    return False


def _append_analysis_batch_candidate(
    records: list[dict[str, Any]],
    path: Path,
    *,
    output_roots: list[Path],
    seen: dict[str, int],
) -> None:
    raw_source = path.expanduser()
    source = raw_source.resolve(strict=False)
    record: dict[str, Any] = {
        "item_index": len(records) + 1,
        "source_path": str(source),
        "status": "pending",
        "reason": None,
        "duplicate_of": None,
        "source_sha256": None,
        "source_manifest_path": None,
        "source_manifest_sha256": None,
        "model_path": None,
        "model_feature_contract": None,
        "session_id": None,
        "output_session_id": None,
        "output_directory": None,
        "output_paths": {},
        "valid_sample_count": None,
        "selected_metrics": [],
        "marker_artifact_references": {},
        "error": None,
    }
    records.append(record)
    if source.suffix.lower() not in ANALYSIS_INPUT_SUFFIXES:
        record.update(status="skipped", reason="unsupported_extension")
        return
    if any(_analysis_path_within(raw_source, root) for root in output_roots):
        record.update(status="skipped", reason="inside_batch_output_directory")
        return
    try:
        safe = assert_no_reparse_components(raw_source)
    except (OSError, ValueError) as exc:
        record.update(status="skipped", reason=f"unsafe_path:{type(exc).__name__}", error=str(exc))
        return
    if safe.is_symlink() or not safe.is_file():
        record.update(status="skipped", reason="not_a_regular_file")
        return
    if _is_analysis_export_artifact(safe):
        record.update(status="skipped", reason="generated_analysis_output")
        return
    if _is_mapping_template(safe):
        record.update(status="skipped", reason="mapping_template")
        return
    identity = os.path.normcase(os.path.normpath(str(safe.resolve(strict=True))))
    if identity in seen:
        record.update(status="skipped", reason="duplicate_input", duplicate_of=seen[identity])
        return
    seen[identity] = record["item_index"]
    record["source_path"] = str(safe.resolve(strict=True))


def _enumerate_analysis_batch(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    output_root = Path(request["output_root"]).resolve(strict=False)
    records: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    explicit = request.get("source_paths")
    if explicit:
        for value in explicit:
            _append_analysis_batch_candidate(records, Path(value), output_roots=[output_root], seen=seen)
        return records

    source_root = Path(request["source_path"]).resolve(strict=True)
    if source_root.is_file():
        _append_analysis_batch_candidate(records, source_root, output_roots=[output_root], seen=seen)
        return records
    if not source_root.is_dir():
        raise ValueError("analysis folder source is no longer an existing directory")
    output_roots = [output_root]
    configured_output_base = Path(request.get("analysis_output_base") or output_root).resolve(strict=False)
    if configured_output_base != source_root and _analysis_path_within(configured_output_base, source_root):
        output_roots.append(configured_output_base)
    if any(_analysis_path_within(source_root, root) for root in output_roots) or _is_analysis_output_directory(source_root):
        _append_analysis_batch_directory_skip(records, source_root, "output_directory")
        return records

    pending_dirs = [source_root]
    visited_dirs: set[str] = set()
    recursive = bool(request.get("recursive", False))
    while pending_dirs:
        directory = pending_dirs.pop()
        identity = os.path.normcase(os.path.normpath(str(directory.resolve(strict=False))))
        if identity in visited_dirs:
            continue
        visited_dirs.add(identity)
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda entry: (entry.name.casefold(), entry.name))
        except (OSError, ValueError) as exc:
            _append_analysis_batch_directory_skip(records, directory, f"unreadable_directory:{type(exc).__name__}")
            continue
        descend: list[Path] = []
        for entry in children:
            child = Path(entry.path)
            try:
                if entry.is_dir(follow_symlinks=False):
                    if not recursive:
                        continue
                    try:
                        assert_no_reparse_components(child)
                    except (OSError, ValueError):
                        _append_analysis_batch_directory_skip(records, child, "unsafe_directory")
                        continue
                    inside_output = any(_analysis_path_within(child, root) for root in output_roots)
                    generated_output = _is_analysis_output_directory(child)
                    if inside_output or generated_output:
                        _append_analysis_batch_directory_skip(records, child, "output_directory")
                        continue
                    descend.append(child)
                elif entry.is_file(follow_symlinks=False) and child.suffix.lower() in ANALYSIS_INPUT_SUFFIXES:
                    _append_analysis_batch_candidate(records, child, output_roots=output_roots, seen=seen)
                elif entry.is_symlink() and child.suffix.lower() in ANALYSIS_INPUT_SUFFIXES:
                    _append_analysis_batch_candidate(records, child, output_roots=output_roots, seen=seen)
            except (OSError, ValueError) as exc:
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                except OSError:
                    is_directory = False
                if is_directory:
                    _append_analysis_batch_directory_skip(records, child, f"unreadable_or_unsafe_directory:{type(exc).__name__}")
                elif child.suffix.lower() in ANALYSIS_INPUT_SUFFIXES:
                    _append_analysis_batch_candidate(records, child, output_roots=output_roots, seen=seen)
                    records[-1].update(status="skipped", reason=f"unreadable_or_unsafe:{type(exc).__name__}", error=str(exc))
        if recursive:
            pending_dirs.extend(reversed(descend))
    return records


def _append_analysis_batch_directory_skip(records: list[dict[str, Any]], path: Path, reason: str) -> None:
    records.append({
        "item_index": len(records) + 1,
        "source_path": str(path.resolve(strict=False)),
        "status": "skipped",
        "reason": reason,
        "duplicate_of": None,
        "source_sha256": None,
        "source_manifest_path": None,
        "source_manifest_sha256": None,
        "model_path": None,
        "model_feature_contract": None,
        "session_id": None,
        "output_session_id": None,
        "output_directory": None,
        "output_paths": {},
        "valid_sample_count": None,
        "selected_metrics": [],
        "marker_artifact_references": {},
        "error": None,
    })


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _analysis_marker_references(metadata: Mapping[str, Any]) -> dict[str, Any]:
    references: dict[str, Any] = {}

    def visit(prefix: str, value: Any, marker_context: bool = False, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                label = f"{prefix}.{key}" if prefix else str(key)
                is_marker = marker_context or "marker" in str(key).casefold()
                key_name = str(key).casefold()
                if is_marker and any(token in key_name for token in ("path", "log", "artifact", "sha256", "count", "sequence")):
                    if child is None or isinstance(child, (str, int, float, bool)):
                        references[label] = child
                visit(label, child, is_marker, depth + 1)

    visit("", metadata)
    capture_manifest = metadata.get("source_capture_manifest")
    if isinstance(capture_manifest, Mapping):
        visit("source_capture_manifest", capture_manifest)
    return references


def _batch_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _batch_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _batch_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_batch_json_value(child) for child in value]
    return str(value)


def _write_analysis_batch_checkpoint(
    root: Path,
    job_id: str,
    document: dict[str, Any],
) -> dict[str, Path]:
    """Durably journal and atomically refresh the manifest and workbook."""

    import pandas as pd

    document["checkpoint_sequence"] = int(document.get("checkpoint_sequence", 0)) + 1
    document["updated_utc"] = _batch_utc_now()
    safe_document = _batch_json_value(document)
    encoded = json.dumps(safe_document, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    manifest_path = root / ANALYSIS_BATCH_MANIFEST_NAME
    summary_path = root / ANALYSIS_BATCH_SUMMARY_NAME
    journal_path = root / ANALYSIS_BATCH_JOURNAL_NAME
    with journal_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())

    manifest_temp = root / f".{ANALYSIS_BATCH_MANIFEST_NAME}.{job_id}.tmp"
    with manifest_temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(safe_document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(manifest_temp, manifest_path)

    items = safe_document.get("items", [])
    columns = [
        "item_index", "status", "reason", "source_path", "source_sha256",
        "source_manifest_path", "source_manifest_sha256", "model_path", "session_id",
        "model_feature_contract", "output_session_id", "output_directory", "output_paths", "valid_sample_count",
        "selected_metrics", "marker_artifact_references", "duplicate_of", "error",
    ]
    item_rows = []
    for item in items:
        row = {key: item.get(key) for key in columns}
        for key, value in tuple(row.items()):
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False, allow_nan=False)
        item_rows.append(row)
    overview_rows = [
        {"field": "status", "value": safe_document.get("status")},
        {"field": "job_id", "value": job_id},
        {"field": "source_mode", "value": safe_document.get("source_mode")},
        {"field": "recursive", "value": safe_document.get("recursive")},
        {"field": "item_count", "value": len(items)},
        {"field": "completed_count", "value": sum(item.get("status") == "completed" for item in items)},
        {"field": "failed_count", "value": sum(item.get("status") == "failed" for item in items)},
        {"field": "skipped_count", "value": sum(item.get("status") == "skipped" for item in items)},
        {"field": "created_utc", "value": safe_document.get("created_utc")},
        {"field": "updated_utc", "value": safe_document.get("updated_utc")},
        {"field": "selected_metrics", "value": json.dumps(safe_document.get("selected_metrics", []), ensure_ascii=False)},
        {"field": "analysis_config", "value": json.dumps(safe_document.get("analysis_config", {}), ensure_ascii=False)},
    ]
    summary_temp = root / f".{ANALYSIS_BATCH_SUMMARY_NAME}.{job_id}.tmp"
    with pd.ExcelWriter(summary_temp, engine="openpyxl") as writer:
        pd.DataFrame(overview_rows, columns=["field", "value"]).to_excel(writer, sheet_name="Batch", index=False)
        pd.DataFrame(item_rows, columns=columns).to_excel(writer, sheet_name="Items", index=False)
    os.replace(summary_temp, summary_path)
    return {"summary": summary_path, "manifest": manifest_path, "journal": journal_path}


def _batch_path_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))


def _analysis_batch_partial_item_outputs(root: Path, item: Mapping[str, Any]) -> dict[str, str]:
    """List files already written for an interrupted item without treating them as complete."""

    raw_item_root = item.get("output_directory")
    if not isinstance(raw_item_root, str) or not raw_item_root.strip():
        return {}
    try:
        item_root = assert_no_reparse_components(Path(raw_item_root).expanduser())
        resolved_item_root = item_root.resolve(strict=True)
        items_root = assert_no_reparse_components(root / "items").resolve(strict=True)
        if resolved_item_root.parent != items_root or not resolved_item_root.is_dir():
            return {}
    except (OSError, RuntimeError, ValueError):
        return {}

    partial: dict[str, str] = {}
    for candidate in sorted(resolved_item_root.rglob("*")):
        try:
            safe_candidate = assert_no_reparse_components(candidate)
            if safe_candidate.is_symlink() or not safe_candidate.is_file():
                continue
            resolved_candidate = safe_candidate.resolve(strict=True)
            if not resolved_candidate.is_relative_to(resolved_item_root):
                continue
            partial[resolved_candidate.relative_to(resolved_item_root).as_posix()] = str(resolved_candidate)
        except (OSError, RuntimeError, ValueError):
            continue
    return partial


def _truncate_incomplete_analysis_batch_journal_tail(path: Path) -> None:
    """Remove only an unterminated tail left if the worker stopped mid-append."""

    if not path.is_file():
        return
    with path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end == 0:
            return
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        position = end
        while position > 0:
            position -= 1
            stream.seek(position)
            if stream.read(1) == b"\n":
                stream.truncate(position + 1)
                break
        else:
            stream.truncate(0)
        stream.flush()
        os.fsync(stream.fileno())


def _finalize_interrupted_analysis_batch(
    payload: Mapping[str, Any],
    *,
    job_id: str,
    cancel_marker_path: Path | None,
    worker_exitcode: int | None,
) -> dict[str, Any] | None:
    """Finalize the current batch after cancellation stops its worker mid-item."""

    if not _analysis_batch_requested(payload):
        return None
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Fa-f0-9]{32}", job_id):
        raise ValueError("analysis batch cancellation job id is invalid")
    raw_root = payload.get("output_root")
    raw_base = payload.get("analysis_output_base")
    if not isinstance(raw_root, str) or not raw_root.strip() or not isinstance(raw_base, str) or not raw_base.strip():
        raise ValueError("analysis batch cancellation is missing its allocated output root")

    root = assert_no_reparse_components(Path(raw_root).expanduser()).resolve(strict=True)
    base = assert_no_reparse_components(Path(raw_base).expanduser()).resolve(strict=False)
    if not root.is_dir() or root.parent != base or not root.name.casefold().startswith("analysis_batch_"):
        raise ValueError("analysis batch cancellation root does not match this allocated job")
    if cancel_marker_path is None:
        raise ValueError("analysis batch cancellation marker path is unavailable")
    marker = assert_no_reparse_components(Path(cancel_marker_path).expanduser())
    expected_marker = sidecar_path(root, f"analysis-batch-{job_id}.cancel", layout=INTERNAL_LAYOUT)
    if _batch_path_key(marker) != _batch_path_key(expected_marker):
        raise ValueError("analysis batch cancellation marker does not belong to this job")
    if marker.exists() or marker.is_symlink():
        if marker.is_symlink() or marker.read_text(encoding="ascii").strip() != job_id:
            raise ValueError("analysis batch cancellation marker identity does not match this job")

    manifest_path = assert_no_reparse_components(root / ANALYSIS_BATCH_MANIFEST_NAME)
    summary_path = assert_no_reparse_components(root / ANALYSIS_BATCH_SUMMARY_NAME)
    journal_path = assert_no_reparse_components(root / ANALYSIS_BATCH_JOURNAL_NAME)
    for path in (manifest_path, summary_path, journal_path):
        if path.is_symlink():
            raise ValueError("analysis batch output artifact must not be a symbolic link")

    def matching_document(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if value.get("schema") != "LocalGaze.analysis-batch/1" or value.get("job_id") != job_id:
            return False
        recorded_root = value.get("output_root")
        if not isinstance(recorded_root, str) or not recorded_root.strip():
            return False
        try:
            safe_recorded_root = assert_no_reparse_components(Path(recorded_root).expanduser())
            return _batch_path_key(safe_recorded_root) == _batch_path_key(root)
        except (OSError, RuntimeError, ValueError):
            return False

    candidates: list[tuple[int, dict[str, Any]]] = []
    try:
        if manifest_path.is_file():
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if matching_document(value):
                sequence = value.get("checkpoint_sequence")
                if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence >= 0:
                    candidates.append((sequence, value))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    try:
        if journal_path.is_file():
            with journal_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        value = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if not matching_document(value):
                        continue
                    sequence = value.get("checkpoint_sequence")
                    if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence >= 0:
                        candidates.append((sequence, value))
    except (OSError, UnicodeDecodeError):
        pass
    if not candidates:
        return None

    _, document = max(candidates, key=lambda candidate: candidate[0])
    items = document.get("items")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError("durable analysis batch state has an invalid item list")

    changed = False
    interrupted_utc = _batch_utc_now()
    for item in items:
        status = item.get("status")
        if status == "running":
            item.update(
                status="failed",
                reason="cancelled_during_processing",
                finished_utc=interrupted_utc,
                output_paths=_analysis_batch_partial_item_outputs(root, item),
                error=(
                    "worker exited after cancellation before this item's completion was durably recorded; "
                    "listed output paths, if any, may be partial"
                ),
            )
            changed = True
        elif status == "pending":
            item.update(
                status="skipped",
                reason="cancelled_before_start",
                finished_utc=interrupted_utc,
            )
            changed = True

    if document.get("status") == "running" or changed:
        document["status"] = "cancelled"
        document["reason"] = "worker exited after cancellation; completed item outputs are retained"
        document["worker_exitcode"] = worker_exitcode
        document["finalized_by"] = "desktop_parent_after_worker_exit"
        changed = True

    if changed:
        _truncate_incomplete_analysis_batch_journal_tail(journal_path)
        paths = _write_analysis_batch_checkpoint(root, job_id, document)
    else:
        paths = {
            "summary": summary_path,
            "manifest": manifest_path,
            "journal": journal_path,
        }

    outputs: dict[str, str] = {}
    for key, path in paths.items():
        safe_path = assert_no_reparse_components(path)
        if safe_path.is_file() and not safe_path.is_symlink():
            outputs[key] = str(safe_path.resolve(strict=True))

    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "outputs": outputs,
        "details": {
            "output_root": str(root),
            "analysis_batch_status": document.get("status"),
            "analysis_batch_finalization": "written_by_parent" if changed else "already_terminal",
            "item_counts": counts,
            "partial_outputs": document.get("status") == "cancelled",
            "summary_path": outputs.get("summary"),
            "manifest_path": outputs.get("manifest"),
            "journal_path": outputs.get("journal"),
        },
    }


def _analysis_batch_cancelled(output_root: Path) -> bool:
    job_id = os.environ.get("LOCALGAZE_JOB_ID")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Fa-f0-9]{32}", job_id):
        return False
    marker = sidecar_path(output_root, f"analysis-batch-{job_id}.cancel", layout=INTERNAL_LAYOUT)
    try:
        safe = assert_no_reparse_components(marker)
        return safe.is_file() and not safe.is_symlink()
    except (OSError, ValueError):
        return False


def _run_analysis_batch(request: Mapping[str, Any]) -> dict[str, Any]:
    from EyeEvaluate.Evaluate.analysis_pipeline import analyze_file

    root = assert_no_reparse_components(Path(request["output_root"]).expanduser())
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if any((root / name).exists() or (root / name).is_symlink() for name in (
        ANALYSIS_BATCH_SUMMARY_NAME, ANALYSIS_BATCH_MANIFEST_NAME, ANALYSIS_BATCH_JOURNAL_NAME,
    )):
        raise FileExistsError("analysis batch output root already contains a batch summary or manifest")
    job_id = os.environ.get("LOCALGAZE_JOB_ID")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Fa-f0-9]{32}", job_id):
        job_id = uuid.uuid4().hex
    items = _enumerate_analysis_batch(request)
    selected_metrics = list(request["selected_metrics"])
    for item in items:
        item["selected_metrics"] = selected_metrics
        if item["status"] == "pending":
            source = Path(item["source_path"])
            identity_hash = hashlib.sha256(os.path.normcase(str(source.resolve(strict=False))).encode("utf-8")).hexdigest()[:10]
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._-")[:48] or "input"
            item["output_directory"] = str(root / "items" / f"item_{item['item_index']:04d}_{safe_stem}_{identity_hash}")
    items_root = root / "items"
    if items_root.exists() or items_root.is_symlink():
        raise FileExistsError("analysis batch output root already contains an items directory")
    items_root.mkdir(exist_ok=True)
    source_mode = "multi_file" if request.get("source_paths") else "folder"
    document: dict[str, Any] = {
        "schema": "LocalGaze.analysis-batch/1",
        "job_id": job_id,
        "status": "running",
        "created_utc": _batch_utc_now(),
        "updated_utc": None,
        "checkpoint_sequence": 0,
        "source_mode": source_mode,
        "source_path": request.get("source_path"),
        "source_paths": list(request.get("source_paths", [])),
        "recursive": bool(request.get("recursive", False)),
        "selected_metrics": selected_metrics,
        "analysis_config": request.get("analysis_config", {}),
        "time_unit": request.get("time_unit"),
        "roi_path": request.get("roi_path"),
        "table_format": request.get("table_format", "csv"),
        "output_root": str(root),
        "items": items,
    }
    paths = _write_analysis_batch_checkpoint(root, job_id, document)
    print(f"批次分析已发现 {len(items)} 个候选项；输出目录：{root}", flush=True)
    cancellation_requested = False
    for item in items:
        if item["status"] != "pending":
            continue
        if _analysis_batch_cancelled(root):
            cancellation_requested = True
            for not_started in items:
                if not_started["status"] == "pending":
                    not_started.update(status="skipped", reason="cancelled_before_start")
            break
        item.update(status="running", started_utc=_batch_utc_now())
        _write_analysis_batch_checkpoint(root, job_id, document)
        item_root = Path(item["output_directory"])
        try:
            item_root.mkdir(parents=False, exist_ok=False)
            item["source_sha256"] = _sha256_path(Path(item["source_path"]))
            run = analyze_file(
                item["source_path"], item_root,
                roi_path=request.get("roi_path"), time_unit=request.get("time_unit"),
                config=request.get("analysis_config"), selected_metrics=selected_metrics,
                table_format=request.get("table_format", "csv"),
            )
            reported = {str(key): str(Path(value).resolve()) for key, value in run.outputs.items()}
            if not reported or any(not Path(value).exists() for value in reported.values()):
                raise ValueError("analysis completed without a complete output bundle")
            metadata = run.metadata if isinstance(run.metadata, Mapping) else {}
            source_manifest = metadata.get("source_manifest_path")
            capture_manifest = metadata.get("source_capture_manifest")
            capture_manifest = capture_manifest if isinstance(capture_manifest, Mapping) else {}
            item.update(
                status="completed",
                reason=None,
                finished_utc=_batch_utc_now(),
                source_sha256=metadata.get("source_sha256") or item.get("source_sha256"),
                source_manifest_path=str(source_manifest) if source_manifest else None,
                source_manifest_sha256=metadata.get("source_manifest_sha256"),
                model_path=metadata.get("model_path") or capture_manifest.get("model_path") or capture_manifest.get("model_file"),
                model_feature_contract=metadata.get("model_feature_contract"),
                session_id=metadata.get("session_id"),
                output_session_id=Path(reported.get("directory", item["output_directory"])).name,
                output_paths=reported,
                valid_sample_count=int(run.result.summary.get("valid_measured_count", 0)),
                marker_artifact_references=_analysis_marker_references(metadata),
                error=None,
            )
        except Exception as exc:
            partial_files = {}
            if item_root.exists():
                for output in sorted(item_root.rglob("*")):
                    if output.is_file() and not output.is_symlink():
                        partial_files[output.relative_to(item_root).as_posix()] = str(output.resolve())
            item.update(
                status="failed",
                reason="analysis_failed",
                finished_utc=_batch_utc_now(),
                source_sha256=item.get("source_sha256"),
                output_paths=partial_files,
                error=f"{type(exc).__name__}: {exc}",
            )
        _write_analysis_batch_checkpoint(root, job_id, document)
        print(f"[{item['item_index']}/{len(items)}] {item['status']}: {item['source_path']}", flush=True)

    if _analysis_batch_cancelled(root):
        cancellation_requested = True
    if cancellation_requested:
        for item in items:
            if item["status"] == "pending":
                item.update(status="skipped", reason="cancelled_before_start")
        document["status"] = "cancelled"
        document["reason"] = "cancellation_requested; completed outputs are retained"
    else:
        completed_count = sum(item["status"] == "completed" for item in items)
        if completed_count:
            document["status"] = "completed"
            document["reason"] = None
        else:
            document["status"] = "failed"
            document["reason"] = "no input file completed analysis"
    paths = _write_analysis_batch_checkpoint(root, job_id, document)
    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    failed_rows = [
        {"source_path": item["source_path"], "status": item["status"], "reason": item.get("reason"), "error": item.get("error")}
        for item in items if item["status"] in {"failed", "skipped"}
    ]
    outputs = {key: str(path.resolve()) for key, path in paths.items()}
    return {
        "status": document["status"],
        "message": document.get("reason") or f"batch completed: {counts.get('completed', 0)} succeeded, {counts.get('failed', 0)} failed, {counts.get('skipped', 0)} skipped",
        "outputs": outputs,
        "details": {
            "source_path": request.get("source_path"),
            "source_paths": list(request.get("source_paths", [])),
            "output_root": str(root),
            "item_counts": counts,
            "partial_outputs": bool(counts.get("failed") or counts.get("skipped") or cancellation_requested),
            "row_errors": failed_rows,
            "selected_metrics": selected_metrics,
            "summary_path": outputs["summary"],
            "manifest_path": outputs["manifest"],
            "journal_path": outputs["journal"],
        },
    }


def run_worker_task(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Lazy API dispatch; called only inside a deliberately started worker."""
    request = _payload(kind, payload)
    if kind == "capture_export_retry":
        sys.path.insert(0, str(PROJECT_ROOT / "LocalGaze"))
        from eyetracking_pipeline import retry_capture_export
        return retry_capture_export(request["snapshot_path"])
    if kind == "capture_recovery":
        from eyetracking_pipeline import retry_capture_from_context
        return retry_capture_from_context(request["context_path"])
    if kind in {"calibrate", "capture"}:
        sys.path.insert(0, str(PROJECT_ROOT / "LocalGaze"))
        from eyetracking_pipeline import run_eye_tracking_workflow
        return run_eye_tracking_workflow(
            session_id=request["session_id"], save_dir=request["output_root"],
            workflow_mode=kind, model_path=request.get("model_path"),
            model_name=request.get("model_name") if kind == "calibrate" else None,
            feature_formula_version=request.get("feature_formula_version") if kind == "calibrate" else None,
            correction_path=request.get("correction_path") if kind == "capture" else None,
            motion_monitor_enabled=request.get("motion_monitor_enabled", True) if kind == "capture" else False,
            save_motion_keyframes=request.get("save_motion_keyframes", True) if kind == "capture" else False,
            table_format=request.get("table_format", "csv") if kind == "capture" else "csv",
        )
    if kind == "known_target_offset":
        sys.path.insert(0, str(PROJECT_ROOT / "LocalGaze"))
        from eyetracking_pipeline import run_known_target_offset_workflow
        return run_known_target_offset_workflow(
            session_id=request["session_id"], save_dir=request["output_root"],
            model_path=request["model_path"], correction_path=request["correction_path"],
            target_xy_px=request.get("target_xy_px"),
            table_format=request.get("table_format", "csv"),
        )
    if kind == "independent_validation":
        sys.path.insert(0, str(PROJECT_ROOT / "LocalGaze"))
        from eyetracking_pipeline import run_independent_validation_workflow
        return run_independent_validation_workflow(
            session_id=request["session_id"],
            save_dir=request["output_root"],
            model_path=request["model_path"],
            table_format=request.get("table_format", "csv"),
        )
    if kind == "roi":
        return _roi_job(request)
    if kind == "preprocess":
        from pre import PreprocessConfig

        cfg = PreprocessConfig(time_unit=request["time_unit"])
        if request.get("mapping_table_path"):
            from pre.pipeline import preprocess_mapping_table

            report = preprocess_mapping_table(
                request["mapping_table_path"], request["output_root"], config=cfg, preset="analysis"
            )
            output_root = Path(report.attrs.get("output_root") or request["output_root"]).resolve()
            report_path = Path(report.attrs.get("summary_path") or output_root / "preprocess_report.csv").resolve()
            manifest_path = Path(report.attrs.get("manifest_path") or output_root / "preprocess_manifest.json").resolve()
        else:
            from pre import preprocess_analysis_paths

            report = preprocess_analysis_paths(
                request.get("model_path"), request["source_path"], request["output_root"], config=cfg
            )
            output_root = Path(request["output_root"]).resolve()
            report_path = output_root / "preprocess_report.csv"
            manifest_path = output_root / "preprocess_manifest.json"
        counts = report["status"].value_counts().to_dict() if not report.empty else {}
        processed_count = int(counts.get("processed", 0))
        failed = report.empty or processed_count == 0
        partial = any(key != "processed" for key in counts)
        processed_paths = []
        if not report.empty and "output_file" in report.columns:
            processed_paths = [
                str(Path(value).resolve())
                for value in report.loc[report["status"] == "processed", "output_file"].dropna().tolist()
                if Path(value).is_file()
            ]
        outputs = {
            "directory": str(output_root),
            "report": str(report_path),
            "manifest": str(manifest_path),
        }
        if len(processed_paths) == 1:
            outputs["processed_file"] = processed_paths[0]
        row_errors = []
        if not report.empty and "status" in report.columns:
            failed_rows = report.loc[report["status"] != "processed"]
            for row in failed_rows.to_dict("records"):
                row_number = row.get("row_number")
                try:
                    row_number = int(row_number) if row_number is not None and not math.isnan(float(row_number)) else None
                except (TypeError, ValueError, OverflowError):
                    row_number = None
                item = {"row_number": row_number}
                values = {
                    "data_path": _first_report_value(row, "input_path", "data_file", "data_path"),
                    "model_path": _first_report_value(row, "model_file", "model_path"),
                    "session_id": _first_report_value(row, "session_id"),
                    "status": _first_report_value(row, "status"),
                    "error": _first_report_value(row, "message", "error"),
                    "reason": _first_report_value(row, "calibration_reason", "reason"),
                }
                for key, value in values.items():
                    if value is None or (isinstance(value, float) and not math.isfinite(value)):
                        item[key] = None
                    else:
                        item[key] = str(value)
                row_errors.append(item)
        mapping_table_path = request.get("mapping_table_path")
        return dict(status="failed" if failed else "completed",
                    outputs=outputs,
                    details={"status_counts": counts, "partial_outputs": partial, "source_path": request.get("source_path"), "mapping_table_path": mapping_table_path, "output_root": str(output_root), "session_id": request.get("session_id"), "row_errors": row_errors})
    if kind == "analyze" and _analysis_batch_requested(request):
        return _run_analysis_batch(request)
    from EyeEvaluate.Evaluate.analysis_pipeline import analyze_file
    run = analyze_file(
        request["source_path"], request["output_root"],
        roi_path=request.get("roi_path"), time_unit=request.get("time_unit"),
        config=request.get("analysis_config"), selected_metrics=request.get("selected_metrics"),
        table_format=request.get("table_format", "csv"), session_id=request.get("session_id"),
    )
    return dict(status="completed", outputs={key: str(path) for key, path in run.outputs.items()}, details={"source_path": request["source_path"], "output_root": request["output_root"], "session_id": request.get("session_id"), "source_session_id": run.metadata.get("session_id"), "analysis_method_version": run.result.methods.get("method_version")})


def main(argv=None) -> int:
    # Frozen Python does not necessarily honor PYTHONIOENCODING.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Internal owned desktop worker")
    parser.add_argument("--worker", required=True, choices=sorted(JOB_KINDS))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--payload-json", required=True)
    args = parser.parse_args(argv)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    worker_logger = None
    log_context = None
    log_setup_error = None
    payload = None
    try:
        payload = json.loads(args.payload_json)
        log_context = {**job_log_context(args.worker, payload), "kind": args.worker, "payload": payload}
        expected_path = worker_log_path(
            log_context["output_root"],
            log_context.get("session_id"),
            args.job_id,
            sidecar_layout=log_context.get("worker_log_layout", INTERNAL_LAYOUT),
        )
        supplied_path = os.environ.get("LOCALGAZE_WORKER_LOG_PATH")
        log_path = assert_no_reparse_components(Path(supplied_path).expanduser()) if supplied_path else expected_path
        if log_path != expected_path:
            raise ValueError("worker log path does not match the validated job output directory")
        worker_logger = JobLogWriter(
            log_path,
            job_id=args.job_id,
            kind=args.worker,
            actor="worker",
            session_id=log_context.get("session_id"),
            capture_id=log_context.get("capture_id"),
            context_path=log_context.get("context_path"),
        )
        worker_identity = (
            _capture_job_identity_fields(args.worker, args.job_id, log_context.get("source_capture_job_id"))
            if args.worker in CAPTURE_JOB_KINDS else {}
        )
        worker_logger.append("worker_started", stage="starting", force_sync=True, **worker_identity)
        sys.stdout = TeeTextIO(original_stdout, worker_logger, stream_name="stdout")
        sys.stderr = TeeTextIO(original_stderr, worker_logger, stream_name="stderr")
    except Exception as exc:
        log_setup_error = f"{type(exc).__name__}: {exc}"
        sys.stdout, sys.stderr = original_stdout, original_stderr
    try:
        result = run_worker_task(args.worker, payload if payload is not None else json.loads(args.payload_json))
        if not isinstance(result, dict) or result.get("status") not in {"completed", "failed", "cancelled"}:
            raise ValueError("backend did not return a valid outcome")
        if result["status"] == "completed":
            outputs = result.get("outputs")
            if not isinstance(outputs, dict) or not outputs or any(not Path(path).exists() for path in outputs.values()):
                raise ValueError("backend completed without all reported output paths")
    except Exception as exc:
        trace = traceback.format_exc()
        if worker_logger is not None:
            worker_logger.append(
                "python_exception",
                stage="worker_exception",
                message=f"{type(exc).__name__}: {exc}",
                traceback=trace,
                force_sync=True,
            )
        result = dict(
            status="failed",
            message=f"{type(exc).__name__}: {exc}",
            outputs={},
            details={"error_type": type(exc).__name__, "traceback": trace, "partial_outputs_possible": True},
        )
    if not isinstance(result.get("details"), dict):
        result["details"] = {}
    details = result["details"]
    if log_setup_error:
        details["worker_log_setup_error"] = log_setup_error
    capture_state = {}
    if log_context and args.worker in CAPTURE_JOB_KINDS:
        capture_state = _capture_state_for_job(args.worker, payload or {}, args.job_id, details, log_context)
        if worker_logger is not None:
            worker_logger.update_capture_id(capture_state.get("capture_id"))
        if capture_state.get("last_durable_checkpoint") is not None:
            details.setdefault("last_durable_checkpoint", capture_state["last_durable_checkpoint"])
        if capture_state.get("context_path"):
            details.setdefault("context_path", capture_state["context_path"])
        if capture_state.get("capture_id"):
            details.setdefault("capture_id", capture_state["capture_id"])
        if capture_state.get("capture_task_log_path"):
            details.setdefault("source_capture_task_log_path", capture_state["capture_task_log_path"])
        details.update(_capture_job_identity_fields(
            args.worker,
            args.job_id,
            capture_state.get("source_capture_job_id") or capture_state.get("source_job_id") or log_context.get("source_capture_job_id"),
        ))
        if capture_state.get("checkpoint_error"):
            details["checkpoint_error"] = capture_state["checkpoint_error"]
    if worker_logger is not None:
        details["worker_log_path"] = str(worker_logger.path)
        worker_logger.append(
            "worker_terminal",
            stage=str(result.get("status", "failed")),
            message=str(result.get("message", result.get("status", "failed"))),
            task_status=result.get("status"),
            last_durable_checkpoint=details.get("last_durable_checkpoint"),
            force_sync=True,
            **(
                _capture_job_identity_fields(
                    args.worker,
                    args.job_id,
                    capture_state.get("source_capture_job_id") or capture_state.get("source_job_id") or log_context.get("source_capture_job_id") if log_context else None,
                )
                if args.worker in CAPTURE_JOB_KINDS else {}
            ),
        )
        details["worker_log_last_durable_sequence"] = worker_logger.last_durable_sequence
        if worker_logger.errors:
            details["worker_log_io_errors"] = worker_logger.errors
            details["worker_log_status"] = "failed"
        else:
            details["worker_log_status"] = "ok"
    elif log_setup_error:
        details["worker_log_status"] = "failed"
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout, sys.stderr = original_stdout, original_stderr
    result["job_id"] = args.job_id
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["JobManager", "run_worker_task", "JOB_KINDS"]
