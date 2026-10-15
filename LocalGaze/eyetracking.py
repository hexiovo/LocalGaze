from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys
import threading
import time
import traceback
import tkinter as tk
import uuid
import warnings
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
from session_names import canonical_session_id
from capture_paths import INTERNAL_LAYOUT, LEGACY_LAYOUT, internal_directory
from capture_quality import summarize_quality_window

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from misc import draw_gaze_cloud_overlay
from Global_data import *
from eyetrax.data_contract import (
    DATA_CONTRACT_VERSION,
    IRIS_PROXY_FORMULA_VERSION,
    FramePacket,
    FrameReceptionTracker,
    IrisProxyMetrics,
    SampleStatus,
    derived_field,
    raw_field,
)
from eyetrax.gaze import LEGACY_IRIS_FORMULA_VERSION


warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


LEGACY_TSV_COLUMNS = (
    "Timestamp",
    "X",
    "Y",
    "Blink",
    "LeftPupil",
    "RightPupil",
)
CAPTURE_COLUMNS = (
    "capture_id",
    "frame_id",
    "duplicate_image",
    "elapsed_s",
    "received_at_local",
    "received_monotonic_ns",
    "process_started_monotonic_ns",
    "process_finished_monotonic_ns",
    "processing_duration_ns",
    "frames_dropped_before_frame",
    "sample_status",
    "sample_status_reason",
    "sample_accepted",
    "feature_formula_version",
    "iris_proxy_formula_version",
    "iris_measurement_kind",
    "is_pupil_measurement",
    "left_iris_diameter_over_eye_span",
    "right_iris_diameter_over_eye_span",
    "iris_eye_span_px",
    "left_iris_center_x_px",
    "left_iris_center_y_px",
    "right_iris_center_x_px",
    "right_iris_center_y_px",
    "legacy_iris_mode",
    "head_descriptor_outside_training_range",
    "processing_error",
    "corrected_x_px",
    "corrected_y_px",
    "offset_correction_status",
    "offset_correction_id",
    "motion_enabled",
    "motion_capture_id",
    "motion_head_fresh",
    "motion_head_available",
    "motion_head_valid",
    "motion_head_reason",
    "motion_body_fresh",
    "motion_body_available",
    "motion_body_valid",
    "motion_body_reason",
    "motion_body_frame_id",
    "motion_body_received_monotonic_ns",
    "motion_head_observation_json",
    "motion_body_observation_json",
    "motion_frame_state_json",
    "motion_metrics_json",
    "motion_raw_json",
    "motion_current_event_ids_json",
    "motion_event_updates_json",
)
TSV_COLUMNS = LEGACY_TSV_COLUMNS + CAPTURE_COLUMNS


def _capture_model_metadata_contract(estimator: Any) -> dict[str, Any]:
    """Copy the stable model contract into recovery context without training rows."""

    model = getattr(estimator, "model", None)
    metadata = getattr(model, "metadata", {})
    if not isinstance(metadata, dict):
        return {}
    excluded = {"training"}

    def normalize(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if np.isfinite(value) else str(value)
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items() if isinstance(key, str)}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return normalize(value.item())
        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return normalize(value.tolist())
            except Exception:
                pass
        return str(value)

    return {
        key: normalize(value)
        for key, value in metadata.items()
        if key not in excluded
    }


def _frame_pixel_signature(frame: Any) -> bytes | None:
    """Return a stable signature for frame pixels, or None when unavailable."""

    try:
        pixels = np.asarray(frame)
        if pixels.ndim < 2 or pixels.size == 0:
            return None
        pixels = np.ascontiguousarray(pixels)
        digest = hashlib.blake2b(digest_size=12)
        digest.update(pixels.dtype.str.encode("ascii"))
        digest.update(repr(pixels.shape).encode("ascii"))
        digest.update(memoryview(pixels).cast("B"))
        return digest.digest()
    except Exception:
        # This is metadata only: an unsupported frame must not interrupt capture.
        return None


def _frame_pixel_duplicate(
    current_signature: bytes | None,
    previous_signature: bytes | None,
) -> bool | None:
    """Compare against the immediately previous received frame when possible."""

    if current_signature is None or previous_signature is None:
        return None
    return current_signature == previous_signature


def _local_timestamp() -> str:
    # Keep the historical wall-clock format used by the TSV consumers.
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _validate_session_id(save_name: str, session_id: str | None) -> str:
    name = canonical_session_id(session_id if session_id is not None else save_name)
    if session_id is not None and save_name and canonical_session_id(save_name) != name:
        raise ValueError("save_name and session_id must identify the same capture session")
    return name


def _prepare_capture_paths(
    save_name: str,
    *,
    save_dir: str | os.PathLike[str] | None,
    session_id: str | None,
    session_dir: str | os.PathLike[str] | None,
    table_format: str = "csv",
) -> tuple[str, Path, Path]:
    if save_dir is not None and session_dir is not None:
        if Path(save_dir).resolve() != Path(session_dir).resolve():
            raise ValueError("save_dir and session_dir refer to different directories")
    directory_value = save_dir if save_dir is not None else session_dir
    directory = (
        Path(directory_value)
        if directory_value is not None
        else Path(__file__).resolve().parent / "Data"
    )
    name = _validate_session_id(save_name, session_id)
    if table_format not in {"csv", "txt"}:
        raise ValueError("table_format must be 'csv' or 'txt'")
    directory.mkdir(parents=True, exist_ok=True)
    internal = internal_directory(directory, create=True)
    companions = (
        directory / f"{name}.csv",
        directory / f"{name}.txt",
        directory / f"{name}.xlsx",
        internal / f"{name}.manifest.json",
        internal / f"{name}.capture-snapshot.json",
        directory / f"{name}.manifest.json",
        directory / f"{name}.capture-snapshot.json",
    )
    existing = [path for path in companions if path.exists()]
    if existing:
        raise FileExistsError(
            "capture session files already exist; choose a new session name: "
            + ", ".join(str(path) for path in existing)
        )
    return name, directory, directory / f"{name}.{table_format}"


def _classify_frame_status(
    features,
    blink_value: bool,
    eye_state: str | None,
    ear_metadata: dict[str, Any] | None = None,
) -> tuple[SampleStatus, str | None]:
    """Classify only explicit eye states; missing or contradictory state is invalid."""

    if features is None or eye_state == "no_face":
        return SampleStatus.NO_FACE, "face_mesh_no_face_detected"
    if eye_state == "warmup":
        count = (ear_metadata or {}).get("warmup_samples")
        return SampleStatus.WARMUP, f"open_eye_baseline_warmup; samples={count}"
    if eye_state == "invalid":
        ear = ear_metadata or {}
        return (
            SampleStatus.INVALID,
            "invalid_ear_or_eye_state; "
            f"left_ear={ear.get('left_ear')!r}; right_ear={ear.get('right_ear')!r}",
        )
    if eye_state == "closed":
        thresholds = (ear_metadata or {}).get("thresholds") or {}
        return (
            SampleStatus.BLINK,
            "eye_closed; "
            f"close_threshold={thresholds.get('close')!r}; "
            f"open_threshold={thresholds.get('open')!r}",
        )
    if eye_state == "open":
        if blink_value:
            return SampleStatus.INVALID, "blink_flag_conflicts_with_open_eye_state"
        return SampleStatus.ACCEPTED, None
    if eye_state is None:
        return SampleStatus.INVALID, "eye_state_unavailable"
    return SampleStatus.INVALID, f"unknown_eye_state:{eye_state!r}"


def run_realtime_gaze(
    estimator,
    save_name,
    draw_cloud=True,
    screen_width=1920,
    screen_height=1080,
    cam_width=320,
    cam_height=240,
    *,
    camera_index=None,
    save_dir=None,
    session_id=None,
    session_dir=None,
    iris_formula_version=IRIS_PROXY_FORMULA_VERSION,
    correction_context=None,
    model_binding=None,
    known_target_xy_px=None,
    independent_validation_targets=None,
    independent_validation_capture_id: str | None = None,
    validation_settle_s: float = 0.5,
    validation_observe_s: float = 1.0,
    motion_monitor_enabled=False,
    save_motion_keyframes=True,
    capture_job_id: str | None = None,
    capture_persistence_enabled=True,
    table_format: str = "csv",
):
    """Capture frames once, process each consumed packet once, and return rich rows.

    The six Chinese keys are the stable in-memory/Excel fields. The matching
    historical English column names remain at the front of the TSV header for
    old readers. New sessions keep the pupil columns empty unless callers
    explicitly select the historical iris formula.
    """

    if iris_formula_version not in {
        IRIS_PROXY_FORMULA_VERSION,
        LEGACY_IRIS_FORMULA_VERSION,
    }:
        raise ValueError(f"unsupported iris formula version: {iris_formula_version}")
    if not isinstance(motion_monitor_enabled, bool):
        raise TypeError("motion_monitor_enabled must be a bool")
    if not isinstance(save_motion_keyframes, bool):
        raise TypeError("save_motion_keyframes must be a bool")
    if not isinstance(capture_persistence_enabled, bool):
        raise TypeError("capture_persistence_enabled must be a bool")
    if table_format not in {"csv", "txt"}:
        raise ValueError("table_format must be 'csv' or 'txt'")

    selected_camera = camera_num if camera_index is None else int(camera_index)
    capture_job_id = capture_job_id or os.environ.get("LOCALGAZE_JOB_ID")
    session_name = _validate_session_id(save_name, session_id)
    validation_enabled = independent_validation_targets is not None
    if validation_enabled and known_target_xy_px is not None:
        raise ValueError("known-target offset and independent validation are separate capture modes")
    target_capture_enabled = known_target_xy_px is not None or validation_enabled
    if validation_enabled:
        if model_binding is None:
            raise ValueError("independent validation requires an active model binding")
        if not isinstance(independent_validation_targets, (list, tuple)) or len(independent_validation_targets) != 9:
            raise ValueError("independent validation requires exactly nine target records")
        if (
            isinstance(validation_settle_s, bool)
            or not isinstance(validation_settle_s, (int, float))
            or not np.isfinite(validation_settle_s)
            or validation_settle_s < 0
            or isinstance(validation_observe_s, bool)
            or not isinstance(validation_observe_s, (int, float))
            or not np.isfinite(validation_observe_s)
            or validation_observe_s <= 0
        ):
            raise ValueError("independent validation timing parameters are invalid")
        normalized_targets = []
        for index, item in enumerate(independent_validation_targets):
            if not isinstance(item, dict) or not isinstance(item.get("target_id"), str):
                raise ValueError("independent validation target records are invalid")
            try:
                coordinates = np.asarray(item.get("target_xy_px"), dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("independent validation target coordinates are invalid") from exc
            if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
                raise ValueError("independent validation target coordinates are invalid")
            if not (0 <= coordinates[0] < screen_width and 0 <= coordinates[1] < screen_height):
                raise ValueError("independent validation target is outside the current canvas")
            normalized_targets.append({
                **dict(item),
                "target_id": str(item["target_id"]),
                "target_xy_px": [float(coordinates[0]), float(coordinates[1])],
            })
        independent_validation_targets = normalized_targets
        if not isinstance(independent_validation_capture_id, str) or len(independent_validation_capture_id) != 32:
            raise ValueError("independent validation requires a 32-character capture identity")
    target_xy = None
    if known_target_xy_px is not None:
        if model_binding is None:
            raise ValueError("known-target capture requires an active model binding")
        target_canvas = getattr(model_binding, "calibration_canvas_size_px", None)
        if target_canvas is None:
            raise ValueError("known-target capture requires a recorded model canvas")
        try:
            target_array = np.asarray(known_target_xy_px, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("target_xy_px must contain two finite coordinates") from exc
        if target_array.shape != (2,) or not np.isfinite(target_array).all():
            raise ValueError("target_xy_px must contain two finite coordinates")
        target_xy = (float(target_array[0]), float(target_array[1]))
        if not (0 <= target_xy[0] < target_canvas[0] and 0 <= target_xy[1] < target_canvas[1]):
            raise ValueError("target_xy_px is outside the recorded model canvas")
        if (int(screen_width), int(screen_height)) != tuple(target_canvas):
            raise ValueError("capture canvas does not match the recorded model canvas")

    def binding_metadata(binding):
        if binding is None:
            return None
        canvas = getattr(binding, "calibration_canvas_size_px", None)
        return {
            "model_sha256": getattr(binding, "model_sha256", None),
            "feature_formula_version": getattr(binding, "feature_formula_version", None),
            "calibration_canvas_size_px": list(canvas) if canvas is not None else None,
        }

    metadata: dict[str, Any] = {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "session_id": session_name,
        "capture_id": independent_validation_capture_id if validation_enabled else None,
        "source": "OpenCV VideoCapture; frame_id is the successful-read sequence in this session",
        "field_provenance": {
            "duplicate_image": {
                "source": "derived",
                "derived_from": [
                    "current_frame_pixels",
                    "previous_successful_read_frame_pixels",
                ],
                "comparison": "BLAKE2b signature of dtype, shape, and contiguous pixel bytes; compares only the immediately previous successful read",
                "null_means": "NA: no previous frame or pixel comparison was unavailable",
                "sampling_effect": "none; duplicate pixels are retained and do not imply a repeated exposure",
            }
        },
        "camera_index": selected_camera,
        "local_timezone": dt.datetime.now().astimezone().tzname(),
        "camera_reported_fps": None,
        "started_at_local": _local_timestamp(),
        "ended_at_local": None,
        "capture_start_monotonic_ns": time.monotonic_ns(),
        "capture_end_monotonic_ns": None,
        "feature_formula_version": getattr(estimator, "feature_formula_version", None),
        "iris_formula_version": iris_formula_version,
        "iris_measurement_kind": (
            "legacy_iris_ring_ratio"
            if iris_formula_version == LEGACY_IRIS_FORMULA_VERSION
            else "iris_ring_diameter_proxy"
        ),
        "is_pupil_measurement": False,
        "legacy_iris_mode": iris_formula_version == LEGACY_IRIS_FORMULA_VERSION,
        "stop_reason": "initializing",
        "error": None,
        "frames_received": 0,
        "frames_processed": 0,
        "frames_dropped_unprocessed": 0,
        "records_written": 0,
        "accepted_samples": 0,
        "sample_status_counts": {status.value: 0 for status in SampleStatus},
        "model_binding": binding_metadata(model_binding),
        "offset_correction": (
            correction_context.manifest_metadata()
            if correction_context is not None
            else {"enabled": False, "status": "disabled", "reason": None}
        ),
        "workflow_mode": (
            "independent_validation"
            if validation_enabled
            else "known_target_offset"
            if target_capture_enabled
            else "capture"
        ),
        "motion_monitor": {
            "requested": motion_monitor_enabled,
            "enabled": False,
            "save_keyframes_requested": bool(motion_monitor_enabled and save_motion_keyframes),
            "reason": (
                "disabled_for_known_target_capture"
                if target_capture_enabled
                else "disabled_not_requested"
                if not motion_monitor_enabled
                else "initializing"
            ),
        },
        "target_capture": ({
            "target_xy_px": list(target_xy),
            "target_display_monotonic_ns": None,
            "clock": "time.monotonic_ns; successful frame-read receipt clock",
            "duration_limit_ns": 4_000_000_000,
            "stop_reason": None,
            "context_valid": True,
            "context_violations": [],
            "participant_fixation_independently_verified": False,
            "physical_dpi_verified": False,
        } if known_target_xy_px is not None else None),
        "independent_validation": ({
            "schema": "localgaze.independent-validation/1",
            "targets": independent_validation_targets,
            "settle_seconds": float(validation_settle_s),
            "observe_seconds": float(validation_observe_s),
            "target_display_times_monotonic_ns": [],
            "target_sequence_deadline_monotonic_ns": None,
            "samples": [],
            "context_valid": True,
            "context_violations": [],
            "coordinate_source": "Tk screen canvas in pixels; target positions are the 10%, 50%, and 90% interior grid fractions",
            "sample_gate": "existing accepted/open-eye status and finite locked-model prediction; no additional accuracy threshold",
        } if validation_enabled else None),
        "files": {},
        "capture_persistence": {
            "enabled": capture_persistence_enabled,
            "status": "pending" if capture_persistence_enabled else "disabled",
        },
    }
    estimator.last_capture_metadata = metadata

    try:
        session_name, output_dir, save_path = _prepare_capture_paths(
            save_name,
            save_dir=save_dir,
            session_id=session_id,
            session_dir=session_dir,
            table_format=table_format,
        )
    except (FileExistsError, TypeError, ValueError) as exc:
        metadata["stop_reason"] = "output_path_rejected"
        metadata["error"] = str(exc)
        metadata["ended_at_local"] = _local_timestamp()
        metadata["capture_end_monotonic_ns"] = time.monotonic_ns()
        raise

    metadata["session_id"] = session_name
    metadata["output_dir"] = str(output_dir.resolve())
    metadata["files"] = {
        "txt": save_path.name,
        "xlsx": f"{session_name}.xlsx",
        "samples": save_path.name,
        "table_format": table_format,
        "manifest": f".internal/{session_name}.manifest.json",
    }
    metadata["table_format"] = table_format
    metadata["sidecar_layout"] = INTERNAL_LAYOUT
    metadata["raw_path"] = save_path.name

    gaze_data: list[dict[str, Any]] = []
    output_file = None
    writer = None
    cap = None
    root = None
    capture_thread = None
    stop_event = threading.Event()
    capture_done = threading.Event()
    state_lock = threading.Lock()
    reception = FrameReceptionTracker()
    state: dict[str, Any] = {
        "latest": None,
        "pending": deque(),
        "frames_received": 0,
        "frames_processed": 0,
        "frames_dropped_unprocessed": 0,
        "records_written": 0,
        "accepted_samples": 0,
        "sample_status_counts": Counter(),
        "capture_error": None,
        "capture_traceback": None,
        "capture_stop_reason": None,
        "frame_receipt_times_ns": deque(),
        "quality_samples": deque(),
        "reader_started_monotonic_ns": None,
        "last_valid_sample_ns": None,
        "tracking_lost_started_ns": None,
        "tracking_lost_seen": False,
        "last_quality_publish_ns": 0,
        "quality_sequence": 0,
    }
    after_ids: set[str] = set()
    gaze_history: list[tuple[float, float]] = []
    camera_visible = True
    camera_canvas = None
    main_error: str | None = None
    stop_reason = "user_exit"
    motion_runtime = None
    capture_persistence = None
    main_traceback: str | None = None
    motion_status_label = None
    persistence_status_label = None
    quality_status_label = None
    marker_status_label = None
    held_marker_keys: set[str] = set()
    last_motion_status = None
    last_motion_status_refresh_ns = 0
    last_persistence_status = None
    last_persistence_status_refresh_ns = 0

    def refresh_metadata() -> None:
        with state_lock:
            metadata.update(
                {
                    "frames_received": int(state["frames_received"]),
                    "frames_processed": int(state["frames_processed"]),
                    "frames_dropped_unprocessed": int(state["frames_dropped_unprocessed"]),
                    "records_written": int(state["records_written"]),
                    "accepted_samples": int(state["accepted_samples"]),
                    "sample_status_counts": {
                        status.value: int(state["sample_status_counts"].get(status.value, 0))
                        for status in SampleStatus
                    },
                }
            )
            if state["capture_error"]:
                metadata["capture_error"] = state["capture_error"]
            if state["capture_traceback"]:
                metadata["capture_traceback"] = state["capture_traceback"]
        if capture_persistence is not None:
            persistence_state = metadata.setdefault("capture_persistence", {})
            persistence_state.update(
                {
                    "enabled": True,
                    "capture_id": capture_persistence.capture_id,
                    "context_path": str(capture_persistence.context_path),
                    "checkpoint_path": str(capture_persistence.checkpoint_path),
                    "events_path": str(capture_persistence.events_path),
                    "capture_task_log_path": str(capture_persistence.task_log_path),
                    "last_durable_checkpoint": capture_persistence.last_checkpoint,
                    "log_io_error": capture_persistence.log_error,
                    "log_io_errors": dict(capture_persistence._log_errors),
                    "durability_error": capture_persistence.durability_error,
                    "status": "error" if capture_persistence.log_error or capture_persistence.durability_error else "ok",
                }
            )
        estimator.last_capture_metadata = metadata

    def schedule_safe(callback, delay_ms: int) -> None:
        if stop_event.is_set() or root is None:
            return
        holder: dict[str, str] = {}

        def invoke() -> None:
            callback_id = holder.get("id")
            if callback_id is not None:
                after_ids.discard(callback_id)
            callback()

        try:
            callback_id = root.after(delay_ms, invoke)
            holder["id"] = callback_id
            after_ids.add(callback_id)
        except tk.TclError:
            pass

    def finish_window(reason: str) -> None:
        nonlocal stop_reason
        if stop_event.is_set():
            return
        held_marker_keys.clear()
        stop_reason = reason
        if capture_persistence is not None:
            capture_persistence.append_event("capture_stop_requested", stage="capture_stop", reason=reason)
        stop_event.set()
        for callback_id in list(after_ids):
            try:
                root.after_cancel(callback_id)
            except (tk.TclError, AttributeError):
                pass
        after_ids.clear()
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if root is not None and root.winfo_exists():
                root.quit()
                root.destroy()
        except tk.TclError:
            pass

    def capture_loop() -> None:
        frame_id = 0
        previous_pixel_signature: bytes | None = None
        try:
            if not target_capture_enabled:
                with state_lock:
                    state["reader_started_monotonic_ns"] = time.monotonic_ns()
            while not stop_event.is_set():
                if target_capture_enabled:
                    # Backpressure keeps the four-second window bounded without
                    # dropping a successfully read frame from its raw receipt set.
                    while True:
                        with state_lock:
                            queue_has_room = len(state["pending"]) < 48
                        if queue_has_room or stop_event.is_set():
                            break
                        stop_event.wait(0.005)
                    if stop_event.is_set():
                        break
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not stop_event.is_set():
                        with state_lock:
                            state["capture_stop_reason"] = "camera_read_failed"
                            state["capture_error"] = "VideoCapture.read returned no frame"
                        if capture_persistence is not None:
                            capture_persistence.append_event(
                                "camera_read_failed",
                                stage="camera_read",
                                frame_id=frame_id + 1,
                                error="VideoCapture.read returned no frame",
                            )
                    break
                frame_id += 1
                received_ns = time.monotonic_ns()
                if target_capture_enabled:
                    if validation_enabled:
                        validation = metadata.get("independent_validation") or {}
                        deadline_ns = validation.get("target_sequence_deadline_monotonic_ns")
                        if deadline_ns is not None and received_ns >= int(deadline_ns):
                            with state_lock:
                                state["capture_stop_reason"] = "target_sequence_complete"
                            break
                    else:
                        display_ns = (metadata.get("target_capture") or {}).get("target_display_monotonic_ns")
                        if display_ns is None or received_ns > display_ns + 4_000_000_000:
                            with state_lock:
                                state["capture_stop_reason"] = "target_capture_window_complete"
                            break
                received_at_local = _local_timestamp()
                pixel_signature = _frame_pixel_signature(frame)
                duplicate_image = _frame_pixel_duplicate(
                    pixel_signature,
                    previous_pixel_signature,
                )
                previous_pixel_signature = pixel_signature
                packet = reception.receive(
                    frame_id,
                    received_ns,
                    has_face=True,
                    fields={
                        "frame_id": raw_field(frame_id),
                        "received_monotonic_ns": raw_field(received_ns),
                        "received_at_local": raw_field(received_at_local),
                        "camera_index": raw_field(selected_camera),
                        "duplicate_image": derived_field(
                            duplicate_image,
                            "current_frame_pixels",
                            "previous_successful_read_frame_pixels",
                        ),
                    },
                )
                with state_lock:
                    state["frames_received"] += 1
                    if not target_capture_enabled:
                        receipt_times = state["frame_receipt_times_ns"]
                        receipt_times.append(received_ns)
                        while receipt_times and receipt_times[0] < received_ns - 5_000_000_000:
                            receipt_times.popleft()
                    if target_capture_enabled:
                        state["pending"].append((packet, frame))
                    else:
                        if state["latest"] is not None:
                            state["frames_dropped_unprocessed"] += 1
                        state["latest"] = (packet, frame)
        except Exception as exc:
            with state_lock:
                state["capture_stop_reason"] = "capture_error"
                state["capture_error"] = f"{type(exc).__name__}: {exc}"
                state["capture_traceback"] = traceback.format_exc()
            if capture_persistence is not None:
                capture_persistence.append_event(
                    "camera_capture_exception",
                    stage="camera_read",
                    frame_id=frame_id + 1,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
        finally:
            capture_done.set()
            try:
                cap.release()
            except Exception:
                pass

    def iris_fields(iris_result, returned_metadata) -> dict[str, Any]:
        extracted = dict(returned_metadata or {})
        if not extracted:
            extracted = dict(getattr(estimator, "last_iris_measurement_metadata", {}) or {})
        if iris_formula_version == LEGACY_IRIS_FORMULA_VERSION:
            return {
                "legacy_left_value": iris_result[0],
                "legacy_right_value": iris_result[1],
                "left_iris_diameter_over_eye_span": None,
                "right_iris_diameter_over_eye_span": None,
                "iris_eye_span_px": None,
                "left_iris_center_x_px": None,
                "left_iris_center_y_px": None,
                "right_iris_center_x_px": None,
                "right_iris_center_y_px": None,
                "iris_proxy_formula_version": LEGACY_IRIS_FORMULA_VERSION,
                "iris_measurement_kind": extracted.get("measurement_kind", "legacy_iris_ring_ratio"),
                "is_pupil_measurement": False,
            }
        if not isinstance(iris_result, IrisProxyMetrics):
            return {
                "legacy_left_value": None,
                "legacy_right_value": None,
                "left_iris_diameter_over_eye_span": None,
                "right_iris_diameter_over_eye_span": None,
                "iris_eye_span_px": None,
                "left_iris_center_x_px": None,
                "left_iris_center_y_px": None,
                "right_iris_center_x_px": None,
                "right_iris_center_y_px": None,
                "iris_proxy_formula_version": extracted.get("formula_version", IRIS_PROXY_FORMULA_VERSION),
                "iris_measurement_kind": extracted.get("measurement_kind", "iris_ring_diameter_proxy"),
                "is_pupil_measurement": False,
            }
        return {
            "legacy_left_value": None,
            "legacy_right_value": None,
            "left_iris_diameter_over_eye_span": iris_result.left_iris_diameter_over_eye_span,
            "right_iris_diameter_over_eye_span": iris_result.right_iris_diameter_over_eye_span,
            "iris_eye_span_px": iris_result.eye_span_px,
            "left_iris_center_x_px": iris_result.left_center_px[0],
            "left_iris_center_y_px": iris_result.left_center_px[1],
            "right_iris_center_x_px": iris_result.right_center_px[0],
            "right_iris_center_y_px": iris_result.right_center_px[1],
            "iris_proxy_formula_version": iris_result.formula_version,
            "iris_measurement_kind": extracted.get("measurement_kind", "iris_ring_diameter_proxy"),
            "is_pupil_measurement": False,
        }

    def observe_target_context(stage: str) -> bool:
        """Persist any formula/canvas drift for this owned target capture."""
        if not target_capture_enabled:
            return False
        target_metadata = (
            metadata.get("independent_validation")
            if validation_enabled
            else metadata.get("target_capture")
        )
        if not isinstance(target_metadata, dict):
            return True
        violations = target_metadata.setdefault("context_violations", [])
        if not isinstance(violations, list):
            violations = []
            target_metadata["context_violations"] = violations

        observed_formula = getattr(estimator, "feature_formula_version", None)
        try:
            observed_canvas = (
                (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
                if root is not None
                else None
            )
        except Exception:
            observed_canvas = None

        reasons = []
        if observed_formula != model_binding.feature_formula_version:
            reasons.append("feature_formula_changed")
        if observed_canvas is None:
            reasons.append("capture_canvas_unavailable")
        elif observed_canvas != model_binding.calibration_canvas_size_px:
            reasons.append("capture_canvas_changed")

        existing_reasons = {
            item.get("reason") for item in violations if isinstance(item, dict)
        }
        for reason in reasons:
            if reason in existing_reasons:
                continue
            violations.append(
                {
                    "reason": reason,
                    "stage": stage,
                    "observed_feature_formula_version": observed_formula,
                    "observed_canvas_size_px": list(observed_canvas) if observed_canvas is not None else None,
                }
            )
            existing_reasons.add(reason)
        target_metadata["context_valid"] = not violations
        return bool(violations)

    def process_packet(packet: FramePacket, frame) -> dict[str, Any]:
        process_start_ns = time.monotonic_ns()
        status = packet.status
        blink_value = False
        x_value = y_value = None
        legacy_left = legacy_right = None
        processing_error = None
        status_reason = None
        iris_result = None
        returned_iris_metadata: dict[str, Any] = {}
        head_observation_fresh = False
        feature_formula = getattr(estimator, "feature_formula_version", None)
        head_range_flag = None
        corrected_x = corrected_y = None
        correction_status = "disabled"
        correction_id = None
        packet_drops = None

        with state_lock:
            packet_drops = int(state["frames_dropped_unprocessed"])

        if status is not SampleStatus.DUPLICATE_FRAME_ID:
            try:
                target_context_invalid = observe_target_context("before_feature_extraction")
                features, blink_value, iris_result, returned_iris_metadata = estimator.extract_features(
                    frame,
                    return_more=True,
                    iris_formula_version=iris_formula_version,
                )
                head_observation_fresh = True
                target_context_invalid = (
                    observe_target_context("after_feature_extraction")
                    or target_context_invalid
                )
                eye_state = getattr(estimator, "last_eye_state", None)
                ear_metadata = getattr(estimator, "last_ear_metadata", {}) or {}
                if iris_formula_version == LEGACY_IRIS_FORMULA_VERSION:
                    legacy_left, legacy_right = iris_result, returned_iris_metadata
                    returned_iris_metadata = getattr(estimator, "last_iris_measurement_metadata", {})
                    iris_result = (legacy_left, legacy_right)
                if target_context_invalid:
                    status = SampleStatus.INVALID
                    status_reason = "target_capture_context_changed"
                else:
                    status, status_reason = _classify_frame_status(
                        features,
                        bool(blink_value),
                        eye_state,
                        ear_metadata,
                    )

                if status is SampleStatus.ACCEPTED:
                    prediction = np.asarray(estimator.predict([features]), dtype=float)
                    if prediction.ndim != 2 or prediction.shape[0] != 1 or prediction.shape[1] < 2:
                        raise ValueError(f"gaze prediction must have shape (1, 2+), got {prediction.shape}")
                    if not np.all(np.isfinite(prediction[0, :2])):
                        raise ValueError("gaze prediction contains non-finite coordinates")
                    x_value, y_value = float(prediction[0, 0]), float(prediction[0, 1])
                    flags = getattr(estimator, "last_runtime_flags", {})
                    head_range_flag = flags.get("head_descriptor_outside_training_range") if isinstance(flags, dict) else None
                    if isinstance(head_range_flag, (list, tuple, np.ndarray)):
                        head_range_flag = head_range_flag[0] if len(head_range_flag) else None
            except Exception as exc:
                status = SampleStatus.PROCESSING_ERROR
                processing_error = f"{type(exc).__name__}: {exc}"
                status_reason = "feature_or_prediction_error"
                returned_iris_metadata = getattr(estimator, "last_iris_measurement_metadata", {}) or {}

        elif status is SampleStatus.DUPLICATE_FRAME_ID:
            status_reason = "duplicate_frame_id"

        if correction_context is not None:
            from offset_workflow import apply_capture_correction

            raw_xy = (x_value, y_value) if x_value is not None and y_value is not None else None
            active_context = correction_context
            if correction_context.status == "ready":
                if (
                    model_binding is None
                    or correction_context.active_binding != model_binding
                    or getattr(estimator, "feature_formula_version", None)
                    != model_binding.feature_formula_version
                ):
                    active_context = replace(correction_context, status="binding_mismatch")
                else:
                    try:
                        live_canvas = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
                    except Exception:
                        live_canvas = None
                    if live_canvas is None:
                        active_context = replace(correction_context, status="capture_canvas_unavailable")
                    elif live_canvas != model_binding.calibration_canvas_size_px:
                        active_context = replace(correction_context, status="capture_canvas_mismatch")
                if active_context.status != correction_context.status:
                    offset_metadata = metadata.setdefault("offset_correction", {})
                    refusals = offset_metadata.setdefault("runtime_refusals", [])
                    if active_context.status not in refusals:
                        refusals.append(active_context.status)
            applied = apply_capture_correction(raw_xy, active_context)
            correction_status = applied.status
            correction_id = applied.correction_id
            if applied.applied and applied.corrected_xy_px is not None:
                corrected_x, corrected_y = applied.corrected_xy_px

        final_packet = replace(packet, status=status)
        measured = iris_fields(
            (legacy_left, legacy_right) if iris_formula_version == LEGACY_IRIS_FORMULA_VERSION else iris_result,
            returned_iris_metadata,
        )
        received_field = packet.fields.get("received_at_local")
        duplicate_image_field = packet.fields.get("duplicate_image")
        timestamp = (
            received_field.value
            if received_field is not None
            else _local_timestamp()
        )
        blink_column = (
            1
            if status is SampleStatus.BLINK
            else 0
            if status is SampleStatus.ACCEPTED
            else None
        )
        record: dict[str, Any] = {
            "时间": timestamp,
            "x": x_value,
            "y": y_value,
            "blink": blink_column,
            "左瞳孔直径": measured["legacy_left_value"] if iris_formula_version == LEGACY_IRIS_FORMULA_VERSION else None,
            "右瞳孔直径": measured["legacy_right_value"] if iris_formula_version == LEGACY_IRIS_FORMULA_VERSION else None,
            "左虹膜尺寸代理": measured["left_iris_diameter_over_eye_span"],
            "右虹膜尺寸代理": measured["right_iris_diameter_over_eye_span"],
            "capture_id": capture_persistence.capture_id if capture_persistence is not None else metadata.get("capture_id"),
            "frame_id": final_packet.frame_id,
            "duplicate_image": (
                duplicate_image_field.value
                if duplicate_image_field is not None
                else None
            ),
            "elapsed_s": final_packet.elapsed_s,
            "received_at_local": timestamp,
            "received_monotonic_ns": final_packet.received_monotonic_ns,
            "process_started_monotonic_ns": process_start_ns,
            "process_finished_monotonic_ns": time.monotonic_ns(),
            "processing_duration_ns": None,
            "frames_dropped_before_frame": packet_drops,
            "sample_status": final_packet.status.value,
            "sample_status_reason": status_reason,
            "sample_accepted": final_packet.sample_accepted,
            "feature_formula_version": feature_formula,
            "iris_proxy_formula_version": measured["iris_proxy_formula_version"],
            "iris_measurement_kind": measured["iris_measurement_kind"],
            "is_pupil_measurement": measured["is_pupil_measurement"],
            "left_iris_diameter_over_eye_span": measured["left_iris_diameter_over_eye_span"],
            "right_iris_diameter_over_eye_span": measured["right_iris_diameter_over_eye_span"],
            "iris_eye_span_px": measured["iris_eye_span_px"],
            "left_iris_center_x_px": measured["left_iris_center_x_px"],
            "left_iris_center_y_px": measured["left_iris_center_y_px"],
            "right_iris_center_x_px": measured["right_iris_center_x_px"],
            "right_iris_center_y_px": measured["right_iris_center_y_px"],
            "legacy_iris_mode": iris_formula_version == LEGACY_IRIS_FORMULA_VERSION,
            "head_descriptor_outside_training_range": head_range_flag,
            "processing_error": processing_error,
            "corrected_x_px": corrected_x,
            "corrected_y_px": corrected_y,
            "offset_correction_status": correction_status,
            "offset_correction_id": correction_id,
        }
        if motion_runtime is None:
            motion_row = {
                "motion_enabled": False,
                "motion_capture_id": None,
                "motion_head_fresh": False,
                "motion_head_available": False,
                "motion_head_valid": False,
                "motion_head_reason": metadata.get("motion_monitor", {}).get("reason"),
                "motion_body_fresh": False,
                "motion_body_available": False,
                "motion_body_valid": False,
                "motion_body_reason": "motion_monitor_unavailable",
                "motion_body_frame_id": None,
                "motion_body_received_monotonic_ns": None,
                "motion_head_observation_json": "{}",
                "motion_body_observation_json": "{}",
                "motion_frame_state_json": "{}",
                "motion_metrics_json": "{}",
                "motion_raw_json": "{}",
                "motion_current_event_ids_json": "{}",
                "motion_event_updates_json": "[]",
            }
        else:
            try:
                if head_observation_fresh:
                    head_getter = getattr(estimator, "get_head_motion_observation", None)
                    head_observation = (
                        head_getter()
                        if callable(head_getter)
                        else getattr(estimator, "last_head_motion_observation", None)
                    )
                else:
                    head_observation = {
                        "available": False,
                        "valid": False,
                        "reason": "head_feature_extraction_failed_or_frame_rejected",
                    }
                motion_row = motion_runtime.observe(frame, packet, head_observation)
            except Exception as exc:
                motion_error = f"{type(exc).__name__}: {exc}"
                motion_metadata = metadata.setdefault("motion_monitor", {})
                motion_metadata["last_processing_error"] = motion_error
                motion_metadata["processing_error_count"] = int(
                    motion_metadata.get("processing_error_count", 0)
                ) + 1
                motion_row = {
                    "motion_enabled": True,
                    "motion_capture_id": motion_runtime.capture_id,
                    "motion_head_fresh": True,
                    "motion_head_available": False,
                    "motion_head_valid": False,
                    "motion_head_reason": motion_error,
                    "motion_body_fresh": False,
                    "motion_body_available": False,
                    "motion_body_valid": False,
                    "motion_body_reason": "movement_processing_error",
                    "motion_body_frame_id": None,
                    "motion_body_received_monotonic_ns": None,
                    "motion_head_observation_json": json.dumps(
                        {"available": False, "valid": False, "reason": motion_error},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "motion_body_observation_json": "{\"available\":false,\"valid\":false,\"reason\":\"movement_processing_error\"}",
                    "motion_frame_state_json": "{}",
                    "motion_metrics_json": "{}",
                    "motion_raw_json": "{}",
                    "motion_current_event_ids_json": "{}",
                    "motion_event_updates_json": "[]",
                }
        record.update(motion_row)
        record["processing_duration_ns"] = (
            record["process_finished_monotonic_ns"] - process_start_ns
        )
        return record

    def record_validation_sample(packet: FramePacket, record: dict[str, Any]) -> None:
        if not validation_enabled:
            return
        validation = metadata.get("independent_validation")
        if not isinstance(validation, dict):
            return
        received_ns = int(packet.received_monotonic_ns)
        settle_ns = int(float(validation["settle_seconds"]) * 1_000_000_000)
        observe_ns = int(float(validation["observe_seconds"]) * 1_000_000_000)
        display_times = validation.get("target_display_times_monotonic_ns", [])
        for index, display_ns in enumerate(display_times):
            start_ns = int(display_ns) + settle_ns
            end_ns = start_ns + observe_ns
            if start_ns <= received_ns < end_ns:
                target = validation["targets"][index]
                accepted = bool(record.get("sample_accepted"))
                target_xy = list(target["target_xy_px"])
                predicted_xy = (
                    [float(record["x"]), float(record["y"])]
                    if accepted
                    and record.get("x") is not None
                    and record.get("y") is not None
                    else None
                )
                error_px = None
                if predicted_xy is not None:
                    error_px = float(np.hypot(
                        predicted_xy[0] - target_xy[0],
                        predicted_xy[1] - target_xy[1],
                    ))
                validation.setdefault("samples", []).append({
                    "target_id": target["target_id"],
                    "target_xy_px": target_xy,
                    "predicted_xy_px": predicted_xy,
                    "pixel_error_px": error_px,
                    "received_monotonic_ns": str(received_ns),
                    "elapsed_s": float(packet.elapsed_s),
                    "timestamp_local": record.get("received_at_local"),
                    "frame_id": int(packet.frame_id),
                    "accepted": accepted,
                    "sample_status": record.get("sample_status"),
                    "sample_status_reason": record.get("sample_status_reason"),
                })
                return

    try:
        if capture_persistence_enabled and not target_capture_enabled:
            from capture_persistence import (
                CapturePersistence,
                create_capture_context,
                marker_key_for_keysym,
            )

            capture_context = create_capture_context(
                output_dir,
                session_name,
                job_id=capture_job_id,
                config={
                    "workflow_mode": metadata["workflow_mode"],
                    "camera_index": selected_camera,
                    "motion_monitor_enabled": bool(motion_monitor_enabled),
                    "save_motion_keyframes": bool(motion_monitor_enabled and save_motion_keyframes),
                    "capture_columns": list(TSV_COLUMNS),
                    "table_format": table_format,
                    "feature_formula_version": metadata.get("feature_formula_version"),
                    "iris_formula_version": iris_formula_version,
                    "started_at_local": metadata.get("started_at_local"),
                    "capture_start_monotonic_ns": metadata.get("capture_start_monotonic_ns"),
                    "capture_canvas_size_px": [int(screen_width), int(screen_height)],
                    "model_binding": metadata.get("model_binding"),
                    "offset_correction": metadata.get("offset_correction"),
                    "model_metadata_contract": _capture_model_metadata_contract(estimator),
                },
            )
            capture_persistence = CapturePersistence(
                output_dir / capture_context["context_path"],
                context=capture_context,
            )
            metadata["capture_id"] = capture_persistence.capture_id
            metadata["user_markers"] = capture_persistence.user_markers
            metadata["user_marker_log"] = dict(capture_context["user_markers"])
            metadata["capture_persistence"].update(
                {
                    "status": "prepared",
                    "capture_id": capture_persistence.capture_id,
                    "attempt_id": capture_context["attempt_id"],
                    "context_path": str(capture_persistence.context_path),
                    "checkpoint_path": str(capture_persistence.checkpoint_path),
                    "events_path": str(capture_persistence.events_path),
                    "capture_task_log_path": str(capture_persistence.task_log_path),
                    "job_id": capture_persistence.job_id,
                }
            )
            capture_persistence.append_event(
                "capture_prepared",
                stage="capture_prepared",
                durable=True,
                motion_monitor_enabled=bool(motion_monitor_enabled),
                raw_path=save_path.name,
            )
            capture_persistence.bind_schema(TSV_COLUMNS)

        # Exclusive creation prevents accidentally replacing any past session.
        output_file = save_path.open("x", encoding="utf-8", newline="")
        writer = csv.writer(output_file, delimiter="," if table_format == "csv" else "\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        output_file.flush()
        if capture_persistence is not None:
            header_checkpoint = capture_persistence.checkpoint(
                output_file,
                complete_records=0,
                last_frame_time=None,
                last_frame_id=None,
                stage="capture_header",
                force=True,
            )
            capture_persistence.append_event(
                "raw_capture_opened",
                stage="capture_header",
                durable=True,
                byte_offset=(header_checkpoint or {}).get("byte_offset"),
                checkpoint_sequence=(header_checkpoint or {}).get("sequence"),
            )
        metadata["stop_reason"] = "starting_camera"
        if capture_persistence is not None:
            capture_persistence.append_event("camera_open_start", stage="camera_open", durable=True)

        if cv2 is None:
            raise RuntimeError("OpenCV is unavailable; camera capture cannot start")
        cap = cv2.VideoCapture(selected_camera)
        if capture_persistence is not None:
            capture_persistence.append_event("camera_opened", stage="camera_open", durable=True)
        try:
            metadata["camera_reported_fps"] = float(cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            metadata["camera_reported_fps"] = None
        try:
            metadata["camera_reported_width"] = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata["camera_reported_height"] = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            metadata["camera_reported_width"] = None
            metadata["camera_reported_height"] = None
        if hasattr(cap, "isOpened") and not cap.isOpened():
            raise RuntimeError(f"camera {selected_camera} could not be opened")

        root = tk.Tk()
        root.attributes("-topmost", True)
        root.overrideredirect(True)
        if target_capture_enabled:
            actual_canvas = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
            target_metadata = (
                metadata.get("independent_validation")
                if validation_enabled
                else metadata.get("target_capture")
            )
            if isinstance(target_metadata, dict):
                target_metadata["current_canvas_size_px"] = list(actual_canvas)
            if actual_canvas != tuple(model_binding.calibration_canvas_size_px):
                raise ValueError("current display canvas does not match the recorded model canvas")
        root.geometry(f"{screen_width}x{screen_height}+0+0")
        root.attributes("-transparentcolor", "white")
        root.config(bg="white")

        canvas = tk.Canvas(
            root,
            width=screen_width,
            height=screen_height,
            bg="white",
            highlightthickness=0,
        )
        canvas.pack()
        validation_target_canvas_ids: dict[str, int] = {}

        def draw_validation_target(index: int) -> None:
            target = independent_validation_targets[index]
            target_x, target_y = target["target_xy_px"]
            label_y = min(screen_height - 18, target_y + 31)
            if not validation_target_canvas_ids:
                validation_target_canvas_ids["outer"] = canvas.create_oval(
                    target_x - 13, target_y - 13, target_x + 13, target_y + 13,
                    fill="#ffd400", outline="#202020", width=2,
                )
                validation_target_canvas_ids["inner"] = canvas.create_oval(
                    target_x - 4, target_y - 4, target_x + 4, target_y + 4,
                    fill="#d62828", outline="#ffffff", width=1,
                )
                validation_target_canvas_ids["label"] = canvas.create_text(
                    target_x, label_y, text=f"{target['target_id']} · 请持续注视圆点",
                    fill="#202020", font=("Arial", 15, "bold"),
                )
            else:
                canvas.coords(
                    validation_target_canvas_ids["outer"],
                    target_x - 13, target_y - 13, target_x + 13, target_y + 13,
                )
                canvas.coords(
                    validation_target_canvas_ids["inner"],
                    target_x - 4, target_y - 4, target_x + 4, target_y + 4,
                )
                canvas.coords(validation_target_canvas_ids["label"], target_x, label_y)
                canvas.itemconfigure(
                    validation_target_canvas_ids["label"],
                    text=f"{target['target_id']} · 请持续注视圆点",
                )

        if validation_enabled:
            draw_validation_target(0)
        elif target_capture_enabled:
            target_x, target_y = target_xy
            canvas.create_oval(
                target_x - 13, target_y - 13, target_x + 13, target_y + 13,
                fill="#ffd400", outline="#202020", width=2,
            )
            canvas.create_oval(
                target_x - 4, target_y - 4, target_x + 4, target_y + 4,
                fill="#d62828", outline="#ffffff", width=1,
            )
            label_y = min(screen_height - 18, target_y + 31)
            canvas.create_text(
                target_x, label_y, text="请持续注视圆点",
                fill="#202020", font=("Arial", 15, "bold"),
            )
        if target_capture_enabled:
            camera_canvas = None
            root.bind("<Escape>", lambda _event: finish_window("user_exit"))
        else:
            camera_canvas = tk.Canvas(root, width=cam_width, height=cam_height)
            camera_canvas.place(x=0, y=0)
            quality_status_label = tk.Label(
                root,
                text=(
                    "近期采集质量 · 接收帧率 NA · 有效帧率 NA · "
                    "有效/无效比例 NA/NA · warmup 0 · 连续无有效样本 NA · Head NA · Body NA"
                ),
                anchor="w",
                bg="#ffffff",
                fg="#203b39",
                font=("Arial", 9),
            )
            quality_status_label.place(x=10, y=max(98, screen_height - 54))
            if capture_persistence is not None and metadata.get("workflow_mode") == "capture":
                marker_status_label = tk.Label(
                    root,
                    text=(
                        "采集标记 0 · 按 1–8 或 NumLock 小键盘 1–8 · "
                        "请保持采集控制窗口聚焦；失焦会清除按下状态"
                    ),
                    anchor="w",
                    bg="#ffffff",
                    fg="#203b39",
                    font=("Arial", 9),
                )
                marker_status_label.place(x=10, y=max(120, screen_height - 30))

        def on_exit() -> None:
            finish_window("user_exit")

        def toggle_camera() -> None:
            nonlocal camera_visible
            camera_visible = not camera_visible
            if not camera_visible:
                camera_canvas.place_forget()
            else:
                camera_canvas.place(x=0, y=0)

        marker_enabled = bool(
            capture_persistence is not None
            and metadata.get("workflow_mode") == "capture"
            and marker_status_label is not None
        )

        def on_marker_key_press(event) -> str | None:
            if not marker_enabled or stop_event.is_set():
                return None
            try:
                widget_class = str(event.widget.winfo_class()).lower()
                if (
                    event.widget.winfo_toplevel() is not root
                    or root.focus_displayof() is None
                    or widget_class in {
                        "entry", "text", "spinbox", "tentry", "tspinbox", "tcombobox",
                    }
                ):
                    return None
            except (AttributeError, tk.TclError):
                return None
            literal_key = marker_key_for_keysym(event.keysym)
            if literal_key is None:
                return None
            if literal_key in held_marker_keys:
                return "break"
            held_marker_keys.add(literal_key)
            marker_ns = time.monotonic_ns()
            capture_start_ns = int(metadata["capture_start_monotonic_ns"])
            marker = {
                "schema": "localgaze.capture-user-marker/1",
                "marker_id": uuid.uuid4().hex,
                "session_id": session_name,
                "capture_id": capture_persistence.capture_id,
                "attempt_id": capture_persistence.context["attempt_id"],
                "marker_sequence": len(capture_persistence.user_markers) + 1,
                "literal_key": literal_key,
                "label": literal_key,
                "monotonic_ns": str(marker_ns),
                "elapsed_s": (marker_ns - capture_start_ns) / 1_000_000_000.0,
                "timestamp_local": _local_timestamp(),
                "source": "tk_keypress",
            }
            try:
                saved_marker = capture_persistence.append_user_marker(marker)
                metadata["user_markers"] = capture_persistence.user_markers
                metadata["user_marker_log"] = dict(capture_persistence.context["user_markers"])
                with state_lock:
                    complete_records = int(state["records_written"])
                last_record = gaze_data[-1] if gaze_data else {}
                checkpoint = capture_persistence.checkpoint(
                    output_file,
                    complete_records=complete_records,
                    last_frame_time=last_record.get("时间"),
                    last_frame_id=last_record.get("frame_id"),
                    stage="capture_user_marker",
                    force=True,
                )
                recent = f" · 最近 {saved_marker['label']} · {saved_marker['timestamp_local']}"
                checkpoint_note = "" if checkpoint is not None else " · 检查点写入失败"
                marker_status_label.configure(
                    text=f"采集标记 {len(capture_persistence.user_markers)}{recent}{checkpoint_note}"
                )
            except Exception as exc:
                markers = capture_persistence.user_markers
                metadata["user_markers"] = markers
                metadata["user_marker_log"] = dict(capture_persistence.context.get("user_markers") or {})
                recent = f" · 最近 {markers[-1]['label']}" if markers else ""
                marker_status_label.configure(
                    text=(
                        f"采集标记 {len(markers)}{recent} · 标记日志同步异常："
                        f"{type(exc).__name__}: {exc}"
                    )
                )
                metadata.setdefault("user_marker_errors", []).append(
                    f"{type(exc).__name__}: {exc}"
                )
            return "break"

        def on_marker_key_release(event) -> str | None:
            if not marker_enabled:
                return None
            literal_key = marker_key_for_keysym(getattr(event, "keysym", None))
            if literal_key is not None:
                held_marker_keys.discard(literal_key)
                return "break"
            return None

        def on_marker_focus_out(_event) -> None:
            held_marker_keys.clear()

        if marker_enabled:
            root.bind("<KeyPress>", on_marker_key_press, add="+")
            root.bind("<KeyRelease>", on_marker_key_release, add="+")
            root.bind("<FocusOut>", on_marker_focus_out, add="+")
            root.bind("<Destroy>", on_marker_focus_out, add="+")

        if not target_capture_enabled:
            tk.Button(root, text="退出", command=on_exit).place(x=10, y=10)
            tk.Button(root, text="隐藏/显示摄像头", command=toggle_camera).place(x=80, y=10)

        if motion_monitor_enabled and not target_capture_enabled:
            motion_status_label = tk.Label(
                root,
                text="运动监测初始化中…",
                anchor="w",
                bg="#ffffff",
                fg="#203b39",
                font=("Arial", 9),
            )
            motion_status_label.place(x=10, y=42)
            root.update_idletasks()
            root.update()
            try:
                from eyetrax.movement_runtime import MovementRuntime

                motion_runtime = MovementRuntime(
                    session_id=session_name,
                    output_dir=output_dir,
                    capture_id=(capture_persistence.capture_id if capture_persistence is not None else None),
                    save_images=save_motion_keyframes,
                    sidecar_layout=(
                        capture_persistence.context.get("sidecar_layout", LEGACY_LAYOUT)
                        if capture_persistence is not None else INTERNAL_LAYOUT
                    ),
                )
                if capture_persistence is not None:
                    event_log = getattr(motion_runtime.writer, "event_log_path", None)
                    relative_log = (
                        Path(event_log).resolve().relative_to(output_dir.resolve()).as_posix()
                        if event_log is not None and Path(event_log).exists()
                        else None
                    )
                    capture_persistence.bind_motion_artifacts(
                        motion_capture_id=motion_runtime.capture_id,
                        event_log=relative_log,
                    )
                metadata["motion_monitor"] = {
                    "requested": True,
                    "enabled": True,
                    "capture_id": motion_runtime.capture_id,
                    "save_keyframes_requested": save_motion_keyframes,
                    "algorithm_version": "head-upper-motion-dual-channel-v1",
                    "coordinate_space": "camera_image_px",
                    "reason": None,
                }
            except Exception as exc:
                motion_error = f"{type(exc).__name__}: {exc}"
                metadata["motion_monitor"] = {
                    "requested": True,
                    "enabled": False,
                    "save_keyframes_requested": save_motion_keyframes,
                    "reason": "motion_monitor_initialization_failed",
                    "detail": motion_error,
                }
                motion_status_label.configure(text=f"运动监测不可用：{motion_error}")

        if capture_persistence is not None:
            persistence_status_label = tk.Label(
                root,
                text="本地日志与耐久检查点：正常",
                anchor="w",
                bg="#ffffff",
                fg="#203b39",
                font=("Arial", 9),
            )
            persistence_status_label.place(x=10, y=(68 if motion_status_label is not None else 42))

        def advance_validation_target() -> None:
            if stop_event.is_set() or not validation_enabled or root is None:
                return
            validation = metadata.get("independent_validation")
            display_times = validation.get("target_display_times_monotonic_ns", []) if isinstance(validation, dict) else []
            next_index = len(display_times)
            if next_index >= len(independent_validation_targets):
                return
            draw_validation_target(next_index)
            root.update_idletasks()
            display_ns = time.monotonic_ns()
            display_times.append(str(display_ns))
            validation["current_target_index"] = next_index
            observe_target_context("target_display_event")
            if next_index == len(independent_validation_targets) - 1:
                slot_ns = int((float(validation_settle_s) + float(validation_observe_s)) * 1_000_000_000)
                validation["target_sequence_deadline_monotonic_ns"] = str(display_ns + slot_ns)
            else:
                schedule_safe(
                    advance_validation_target,
                    max(1, int(round((float(validation_settle_s) + float(validation_observe_s)) * 1000))),
                )

        if target_capture_enabled:
            # Map and draw the first target before starting the acquisition clock.
            root.update_idletasks()
            root.update()
            if not root.winfo_viewable():
                raise RuntimeError("known target window did not become visible")
            root.focus_force()
            if validation_enabled:
                validation = metadata["independent_validation"]
                display_ns = time.monotonic_ns()
                validation["target_display_times_monotonic_ns"].append(str(display_ns))
                validation["current_target_index"] = 0
                observe_target_context("target_display_event")
                schedule_safe(
                    advance_validation_target,
                    max(1, int(round((float(validation_settle_s) + float(validation_observe_s)) * 1000))),
                )
            else:
                display_ns = time.monotonic_ns()
                metadata["target_capture"]["target_display_monotonic_ns"] = display_ns
                metadata["target_capture"]["target_display_event"] = "tk_window_visible_after_update"
                observe_target_context("target_display_event")

        def update_quality_dashboard() -> None:
            if target_capture_enabled:
                return
            now_ns = time.monotonic_ns()
            with state_lock:
                if now_ns - int(state["last_quality_publish_ns"]) < 500_000_000:
                    return
                state["last_quality_publish_ns"] = now_ns
                state["quality_sequence"] += 1
                sequence = int(state["quality_sequence"])
                frame_receipts = list(state["frame_receipt_times_ns"])
                samples = list(state["quality_samples"])
                reader_started_ns = state["reader_started_monotonic_ns"]
                last_valid_ns = state["last_valid_sample_ns"]
                tracking_lost_started_ns = state["tracking_lost_started_ns"]
                tracking_lost_seen = bool(state["tracking_lost_seen"])
            quality = summarize_quality_window(
                frame_receipts,
                samples,
                now_ns=now_ns,
                reader_started_ns=reader_started_ns,
                last_valid_ns=last_valid_ns,
            )
            if tracking_lost_seen:
                quality["tracking_lost_seconds"] = (
                    max(0.0, (now_ns - int(tracking_lost_started_ns)) / 1_000_000_000.0)
                    if tracking_lost_started_ns is not None
                    else 0.0
                )
            metadata["quality_dashboard"] = quality

            def shown(value: Any, suffix: str = "") -> str:
                if value is None:
                    return "NA"
                if isinstance(value, float):
                    return f"{value:.2f}{suffix}"
                return f"{value}{suffix}"

            head = quality.get("head_state", {}).get("status", "NA")
            body = quality.get("body_state", {}).get("status", "NA")
            label_text = (
                "近期采集质量 · "
                f"接收帧率 {shown(quality.get('read_frame_rate_hz'), ' 帧/s')} · "
                f"有效帧率 {shown(quality.get('effective_sample_rate_hz'), ' 个/s')} · "
                f"有效/无效比例 {shown(quality.get('valid_ratio'), '')}/"
                f"{shown(quality.get('invalid_ratio'), '')} · "
                f"warmup {quality.get('warmup_count', 0)} · "
                "连续无有效样本 "
                f"{shown(quality.get('continuous_no_valid_sample_seconds'), ' s')} · "
                f"Head {head} · Body {body}"
            )
            if quality.get("tracking_lost_seconds") is not None:
                label_text += f" · tracking_lost {shown(quality.get('tracking_lost_seconds'), ' s')}"
            if quality_status_label is not None:
                try:
                    quality_status_label.configure(text=label_text)
                except tk.TclError:
                    pass
            if capture_job_id:
                progress = {
                    "schema": "localgaze.job-progress/1",
                    "type": "capture_quality",
                    "job_id": capture_job_id,
                    "sequence": sequence,
                    "monotonic_ns": str(now_ns),
                    "quality": quality,
                }
                print("LOCALGAZE_JOB_PROGRESS " + json.dumps(progress, ensure_ascii=False, separators=(",", ":")), flush=True)

        def update_gui() -> None:
            nonlocal stop_reason, last_motion_status, last_motion_status_refresh_ns
            nonlocal last_persistence_status, last_persistence_status_refresh_ns
            if stop_event.is_set():
                return
            if target_capture_enabled:
                observe_target_context("capture_poll")
            entry = None
            with state_lock:
                if target_capture_enabled:
                    if state["pending"]:
                        entry = state["pending"].popleft()
                else:
                    entry = state["latest"]
                    state["latest"] = None

            if entry is not None:
                packet, frame = entry
                record = process_packet(packet, frame)
                record_validation_sample(packet, record)
                try:
                    legacy_values = [
                        record["时间"],
                        record["x"],
                        record["y"],
                        record["blink"],
                        record["左瞳孔直径"],
                        record["右瞳孔直径"],
                    ]
                    writer.writerow(legacy_values + [record[column] for column in CAPTURE_COLUMNS])
                    output_file.flush()
                    gaze_data.append(record)
                    with state_lock:
                        state["frames_processed"] += 1
                        state["records_written"] += 1
                        state["sample_status_counts"][record["sample_status"]] += 1
                        if record["sample_accepted"]:
                            state["accepted_samples"] += 1
                            state["last_valid_sample_ns"] = int(packet.received_monotonic_ns)
                        if "tracking_lost" in record and isinstance(record.get("tracking_lost"), bool):
                            state["tracking_lost_seen"] = True
                            state["tracking_lost_started_ns"] = (
                                int(packet.received_monotonic_ns)
                                if record.get("tracking_lost") and state["tracking_lost_started_ns"] is None
                                else None
                                if not record.get("tracking_lost")
                                else state["tracking_lost_started_ns"]
                            )
                        quality_samples = state["quality_samples"]
                        quality_samples.append(record)
                        while (
                            quality_samples
                            and int(quality_samples[0].get("received_monotonic_ns") or 0)
                            < int(packet.received_monotonic_ns) - 5_000_000_000
                        ):
                            quality_samples.popleft()
                        records_written = int(state["records_written"])
                    if capture_persistence is not None:
                        if record.get("duplicate_image") is True:
                            capture_persistence.record_duplicate_diagnostic(
                                "duplicate_image",
                                frame_id=record.get("frame_id"),
                            )
                        capture_persistence.checkpoint(
                            output_file,
                            complete_records=records_written,
                            last_frame_time=record.get("时间"),
                            last_frame_id=record.get("frame_id"),
                            stage="capture",
                        )
                    if record["sample_accepted"] and draw_cloud and not target_capture_enabled:
                        x_value, y_value = record["x"], record["y"]
                        gaze_history.append((x_value, y_value))
                        if len(gaze_history) > MAX_POINTS:
                            gaze_history.pop(0)
                        try:
                            draw_gaze_cloud_overlay(
                                canvas,
                                gaze_history,
                                radius=20,
                                color_gray="#B4B4B4",
                                color_center="#FF0000",
                                border_color="#000000",
                                center_radius=4,
                                alpha=0.4,
                            )
                        except Exception:
                            pass
                except Exception as exc:
                    metadata["error"] = f"{type(exc).__name__}: {exc}"
                    metadata["traceback"] = traceback.format_exc()
                    stop_reason = "record_write_error"
                    if capture_persistence is not None:
                        capture_persistence.append_event(
                            "raw_frame_write_failed",
                            stage="capture_write",
                            frame_id=getattr(packet, "frame_id", None),
                            error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc(),
                        )
                    finish_window(stop_reason)
                    refresh_metadata()
                    return

                if camera_visible and camera_canvas is not None:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (cam_width, cam_height))
                        encoded = cv2.imencode(".ppm", frame_resized)[1]
                        image = tk.PhotoImage(
                            master=camera_canvas,
                            width=cam_width,
                            height=cam_height,
                            data=encoded.tobytes(),
                        )
                        camera_canvas.create_image(0, 0, anchor="nw", image=image)
                        camera_canvas.image = image
                    except Exception:
                        # Rendering the preview must not interrupt saved samples.
                        pass

            if motion_status_label is not None:
                try:
                    motion_metadata = metadata.get("motion_monitor", {})
                    if motion_runtime is not None:
                        motion_status = motion_runtime.status_text()
                    else:
                        detail = motion_metadata.get("detail") or motion_metadata.get("reason") or "不可用"
                        motion_status = f"运动监测不可用：{detail}"
                    now_ns = time.monotonic_ns()
                    changed = motion_status != last_motion_status
                    if changed and (
                        last_motion_status is None
                        or now_ns - last_motion_status_refresh_ns >= 500_000_000
                    ):
                        motion_status_label.configure(text=motion_status)
                        motion_metadata["status"] = motion_status
                        last_motion_status = motion_status
                        last_motion_status_refresh_ns = now_ns
                except Exception as exc:
                    motion_error = f"{type(exc).__name__}: {exc}"
                    motion_metadata = metadata.setdefault("motion_monitor", {})
                    motion_metadata.setdefault("ui_error", motion_error)

            if persistence_status_label is not None and capture_persistence is not None:
                try:
                    if capture_persistence.log_error:
                        persistence_status = f"采集日志写入失败：{capture_persistence.log_error}"
                    elif capture_persistence.durability_error:
                        persistence_status = f"耐久检查点失败：{capture_persistence.durability_error}"
                    else:
                        persistence_status = "本地日志与耐久检查点：正常"
                    now_ns = time.monotonic_ns()
                    changed = persistence_status != last_persistence_status
                    if changed and (
                        last_persistence_status is None
                        or now_ns - last_persistence_status_refresh_ns >= 500_000_000
                    ):
                        persistence_status_label.configure(text=persistence_status)
                        metadata.setdefault("capture_persistence", {})["status_text"] = persistence_status
                        last_persistence_status = persistence_status
                        last_persistence_status_refresh_ns = now_ns
                except Exception as exc:
                    persistence_state = metadata.setdefault("capture_persistence", {})
                    persistence_state["ui_status_error"] = f"{type(exc).__name__}: {exc}"

            update_quality_dashboard()
            refresh_metadata()
            if capture_done.is_set():
                with state_lock:
                    camera_stop_reason = state["capture_stop_reason"]
                    pending_remains = bool(state["pending"]) if target_capture_enabled else False
                if pending_remains:
                    schedule_safe(update_gui, 0)
                    return
                finish_window(camera_stop_reason or "camera_read_failed")
                return
            schedule_safe(update_gui, 0 if target_capture_enabled else 10)

        capture_thread = threading.Thread(target=capture_loop, name="gaze-frame-capture", daemon=True)
        capture_thread.start()
        schedule_safe(update_gui, 10)
        root.mainloop()
    except (FileExistsError, TypeError, ValueError) as exc:
        main_error = f"{type(exc).__name__}: {exc}"
        main_traceback = traceback.format_exc()
        metadata["error"] = main_error
        stop_reason = "runtime_error"
        raise
    except Exception as exc:
        main_error = f"{type(exc).__name__}: {exc}"
        main_traceback = traceback.format_exc()
        metadata["error"] = main_error
        stop_reason = "runtime_error"
    finally:
        stop_event.set()
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        if capture_thread is not None:
            capture_thread.join(timeout=2.0)
        with state_lock:
            if state["latest"] is not None:
                state["latest"] = None
                state["frames_dropped_unprocessed"] += 1
            if state["capture_stop_reason"] and stop_reason == "user_exit":
                stop_reason = state["capture_stop_reason"]
        if motion_runtime is not None:
            try:
                motion_result = motion_runtime.finish(
                    last_packet=motion_runtime._last_packet,
                    reason=stop_reason,
                )
                motion_metadata = metadata.setdefault("motion_monitor", {})
                motion_metadata["enabled"] = True
                motion_metadata["result"] = motion_result
                motion_metadata["status"] = motion_runtime.status_text()
            except Exception as exc:
                motion_metadata = metadata.setdefault("motion_monitor", {})
                motion_metadata["finalization_error"] = f"{type(exc).__name__}: {exc}"
        if capture_persistence is not None and output_file is not None and not getattr(output_file, "closed", False):
            with state_lock:
                final_records_written = int(state["records_written"])
            last_record = gaze_data[-1] if gaze_data else {}
            capture_persistence.checkpoint(
                output_file,
                complete_records=final_records_written,
                last_frame_time=last_record.get("时间"),
                last_frame_id=last_record.get("frame_id"),
                stage="capture_final",
                force=True,
            )
        if output_file is not None:
            try:
                output_file.flush()
                output_file.close()
            except Exception as exc:
                metadata["error"] = metadata.get("error") or f"{type(exc).__name__}: {exc}"
                stop_reason = "record_close_error"
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if root is not None:
            for callback_id in list(after_ids):
                try:
                    root.after_cancel(callback_id)
                except Exception:
                    pass
            try:
                root.destroy()
            except Exception:
                pass
        metadata["stop_reason"] = stop_reason
        metadata["error"] = metadata.get("error") or metadata.get("capture_error") or main_error
        metadata["traceback"] = (
            main_traceback
            or metadata.get("traceback")
            or metadata.get("capture_traceback")
        )
        metadata["ended_at_local"] = _local_timestamp()
        metadata["capture_end_monotonic_ns"] = time.monotonic_ns()
        if validation_enabled:
            validation = metadata.get("independent_validation")
            if isinstance(validation, dict):
                validation["stop_reason"] = stop_reason
                validation["capture_end_monotonic_ns"] = str(metadata["capture_end_monotonic_ns"])
                validation["target_sequence_complete"] = stop_reason == "target_sequence_complete"
                from independent_validation import summarize_independent_validation

                validation["summary"] = summarize_independent_validation(
                    independent_validation_targets,
                    validation.get("samples", []),
                    capture_status=stop_reason,
                    context_valid=bool(validation.get("context_valid", True)),
                )
        elif target_capture_enabled:
            metadata["target_capture"]["stop_reason"] = stop_reason
            metadata["target_capture"]["capture_end_monotonic_ns"] = metadata["capture_end_monotonic_ns"]
        if capture_persistence is not None:
            with state_lock:
                final_records_written = int(state["records_written"])
            last_record = gaze_data[-1] if gaze_data else {}
            final_status = "completed" if (
                stop_reason == "user_exit"
                and not metadata.get("error")
                and not capture_persistence.durability_error
            ) else "interrupted"
            try:
                capture_persistence.finish_capture(
                    None,
                    complete_records=final_records_written,
                    last_frame_time=last_record.get("时间"),
                    last_frame_id=last_record.get("frame_id"),
                    status=final_status,
                    error=metadata.get("error"),
                    traceback_text=metadata.get("traceback"),
                    capture_metadata={
                        key: metadata.get(key)
                        for key in (
                            "started_at_local", "ended_at_local", "capture_start_monotonic_ns",
                            "capture_end_monotonic_ns", "stop_reason", "error", "frames_received",
                            "frames_processed", "frames_dropped_unprocessed", "records_written",
                            "accepted_samples", "capture_id", "user_markers", "user_marker_log",
                        )
                    },
                )
                persistence_summary = capture_persistence.snapshot_metadata(
                    recovery={"status": "normal_capture", "capture_status": final_status}
                )
                metadata.setdefault("capture_persistence", {}).update(persistence_summary)
                metadata["user_markers"] = capture_persistence.user_markers
                metadata["user_marker_log"] = dict(persistence_summary["user_marker_log"])
            except Exception as exc:
                persistence_state = metadata.setdefault("capture_persistence", {})
                persistence_state["finalization_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    sys.stderr.write(f"capture persistence finalization failed: {type(exc).__name__}: {exc}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
        refresh_metadata()

    return gaze_data


__all__ = [
    "CAPTURE_COLUMNS",
    "LEGACY_TSV_COLUMNS",
    "TSV_COLUMNS",
    "_classify_frame_status",
    "run_realtime_gaze",
]
