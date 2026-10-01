from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os
import sys
import threading
import time
import tkinter as tk
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
)
TSV_COLUMNS = LEGACY_TSV_COLUMNS + CAPTURE_COLUMNS


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
    directory.mkdir(parents=True, exist_ok=True)
    companions = (
        directory / f"{name}.txt",
        directory / f"{name}.xlsx",
        directory / f"{name}.manifest.json",
    )
    existing = [path for path in companions if path.exists()]
    if existing:
        raise FileExistsError(
            "capture session files already exist; choose a new session name: "
            + ", ".join(str(path) for path in existing)
        )
    return name, directory, companions[0]


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

    selected_camera = camera_num if camera_index is None else int(camera_index)
    session_name = _validate_session_id(save_name, session_id)
    target_capture_enabled = known_target_xy_px is not None
    target_xy = None
    if target_capture_enabled:
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
        "workflow_mode": "known_target_offset" if target_capture_enabled else "capture",
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
        } if target_capture_enabled else None),
        "files": {},
    }
    estimator.last_capture_metadata = metadata

    try:
        session_name, output_dir, save_path = _prepare_capture_paths(
            save_name,
            save_dir=save_dir,
            session_id=session_id,
            session_dir=session_dir,
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
        "manifest": f"{session_name}.manifest.json",
    }

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
        "capture_stop_reason": None,
    }
    after_ids: set[str] = set()
    gaze_history: list[tuple[float, float]] = []
    camera_visible = True
    camera_canvas = None
    main_error: str | None = None
    stop_reason = "user_exit"

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
        stop_reason = reason
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
                    break
                frame_id += 1
                received_ns = time.monotonic_ns()
                if target_capture_enabled:
                    display_ns = metadata["target_capture"].get("target_display_monotonic_ns")
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
        target_metadata = metadata.get("target_capture")
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
        record["processing_duration_ns"] = (
            record["process_finished_monotonic_ns"] - process_start_ns
        )
        return record

    try:
        # Exclusive creation prevents accidentally replacing any past session.
        output_file = save_path.open("x", encoding="utf-8", newline="")
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        output_file.flush()
        metadata["stop_reason"] = "starting_camera"

        if cv2 is None:
            raise RuntimeError("OpenCV is unavailable; camera capture cannot start")
        cap = cv2.VideoCapture(selected_camera)
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
            metadata["target_capture"]["current_canvas_size_px"] = list(actual_canvas)
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
        if target_capture_enabled:
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

        def on_exit() -> None:
            finish_window("user_exit")

        def toggle_camera() -> None:
            nonlocal camera_visible
            camera_visible = not camera_visible
            if not camera_visible:
                camera_canvas.place_forget()
            else:
                camera_canvas.place(x=0, y=0)

        if not target_capture_enabled:
            tk.Button(root, text="退出", command=on_exit).place(x=10, y=10)
            tk.Button(root, text="隐藏/显示摄像头", command=toggle_camera).place(x=80, y=10)

        if target_capture_enabled:
            # Map and draw the target before starting the acquisition clock.
            root.update_idletasks()
            root.update()
            if not root.winfo_viewable():
                raise RuntimeError("known target window did not become visible")
            display_ns = time.monotonic_ns()
            metadata["target_capture"]["target_display_monotonic_ns"] = display_ns
            metadata["target_capture"]["target_display_event"] = "tk_window_visible_after_update"
            observe_target_context("target_display_event")

        def update_gui() -> None:
            nonlocal stop_reason
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
                    stop_reason = "record_write_error"
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
    except (FileExistsError, TypeError, ValueError):
        raise
    except Exception as exc:
        main_error = f"{type(exc).__name__}: {exc}"
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
        metadata["error"] = metadata.get("error") or main_error
        metadata["ended_at_local"] = _local_timestamp()
        metadata["capture_end_monotonic_ns"] = time.monotonic_ns()
        if target_capture_enabled:
            metadata["target_capture"]["stop_reason"] = stop_reason
            metadata["target_capture"]["capture_end_monotonic_ns"] = metadata["capture_end_monotonic_ns"]
        refresh_metadata()

    return gaze_data


__all__ = [
    "CAPTURE_COLUMNS",
    "LEGACY_TSV_COLUMNS",
    "TSV_COLUMNS",
    "_classify_frame_status",
    "run_realtime_gaze",
]
