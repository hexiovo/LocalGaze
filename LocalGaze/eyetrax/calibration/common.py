from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from eyetrax.gaze import FIT_RESIDUAL_METHOD_VERSION
from eyetrax.stability import (
    CalibrationStabilityConfig,
    CalibrationStabilityGate,
    StabilityObservation,
)


CALIBRATION_SETTLE_NS = 300_000_000
MIN_OPEN_EYE_FRAMES = 5
MIN_OPEN_EYE_SPAN_NS = 800_000_000
MAX_TARGET_CAPTURE_NS = 6_000_000_000


@dataclass(frozen=True)
class TargetFitResidual:
    target_id: tuple[float, float]
    mean_residual_px: float
    sample_count: int
    frame_residuals_px: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationResidualResult:
    """In-sample fitting residuals, grouped by calibration target and visit."""

    stage: str
    visit_id: str
    targets: tuple[TargetFitResidual, ...]
    residual_kind: str = "training_fit_residual"
    method_version: str = FIT_RESIDUAL_METHOD_VERSION

    @property
    def mean_residual_px(self) -> float | None:
        if not self.targets:
            return None
        return float(np.mean([target.mean_residual_px for target in self.targets]))

    @property
    def sample_count(self) -> int:
        return sum(target.sample_count for target in self.targets)


def compute_grid_points(order, sw: int, sh: int, margin_ratio: float = 0.10):
    """Translate grid (row, col) indices into absolute pixel locations."""

    if not order:
        return []
    max_r = max(r for r, _ in order)
    max_c = max(c for _, c in order)
    mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
    gw, gh = sw - 2 * mx, sh - 2 * my
    step_x = 0 if max_c == 0 else gw / max_c
    step_y = 0 if max_r == 0 else gh / max_r
    return [(mx + int(c * step_x), my + int(r * step_y)) for r, c in order]


def _frame_id(frame: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(frame)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=12).hexdigest()


def _coverage_complete(sample_times_ns: list[int]) -> bool:
    return (
        len(sample_times_ns) >= MIN_OPEN_EYE_FRAMES
        and all(right > left for left, right in zip(sample_times_ns, sample_times_ns[1:]))
        and sample_times_ns[-1] - sample_times_ns[0] >= MIN_OPEN_EYE_SPAN_NS
    )


def _valid_open_features(gaze_estimator, features, blink) -> bool:
    return (
        features is not None and not blink
        and getattr(gaze_estimator, "last_eye_state", None) == "open"
        and np.asarray(features).ndim == 1 and np.asarray(features).size > 0
        and bool(np.all(np.isfinite(features)))
    )


def calibration_gate_config_record(config=None) -> dict[str, Any]:
    """Return the A gate settings plus B's fixed-target collection bounds."""

    config = config or CalibrationStabilityConfig()
    return {
        **asdict(config),
        "target_settle_ns": CALIBRATION_SETTLE_NS,
        "minimum_qualified_receipts": MIN_OPEN_EYE_FRAMES,
        "minimum_post_ready_segment_ns": MIN_OPEN_EYE_SPAN_NS,
        "maximum_target_capture_ns": MAX_TARGET_CAPTURE_NS,
        "duplicate_raw_frame_policy": "exclude_sample_and_restart_commit_coverage",
    }


def record_capture_metadata(gaze_estimator, metadata, residuals=None):
    gaze_estimator.last_calibration_capture_metadata = metadata
    if residuals is not None:
        for record, residual in zip(metadata["samples"], residuals):
            record["training_fit_residual_px"] = float(residual)
        training = gaze_estimator.model.metadata.setdefault("training", {})
        training.setdefault("calibration_captures", []).append(metadata)


def summarize_fit_residuals(
    targets: np.ndarray,
    residuals_px: np.ndarray,
    *,
    stage: str,
    visit_id: str,
) -> CalibrationResidualResult:
    """Produce explicit target/visit fit residuals for reports and callers."""

    target_rows = np.asarray(targets, dtype=np.float64)
    residual_rows = np.asarray(residuals_px, dtype=np.float64).reshape(-1)
    if len(target_rows) != len(residual_rows):
        raise ValueError("targets and residuals must have matching rows")
    grouped: dict[tuple[float, float], list[float]] = {}
    for target, residual in zip(target_rows, residual_rows):
        target_id = (float(target[0]), float(target[1]))
        grouped.setdefault(target_id, []).append(float(residual))
    groups = tuple(
        TargetFitResidual(
            target_id=target_id,
            mean_residual_px=float(np.mean(values)),
            sample_count=len(values),
            frame_residuals_px=tuple(values),
        )
        for target_id, values in grouped.items()
    )
    return CalibrationResidualResult(stage=stage, visit_id=visit_id, targets=groups)


def wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, dur: int = 2) -> bool:
    """Wait for an open-eye face, then show a monotonic countdown."""

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fd_start_ns = None
    countdown = False
    head_box_w, head_box_h = sw // 4, sh // 3
    head_box_x, head_box_y = (sw - head_box_w) // 2, (sh - head_box_h) // 2
    cam_w, cam_h = 480, 360

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            frame_small = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            face_detected = False
        else:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame_small = cv2.resize(frame, (cam_w, cam_h))
            features, blink = gaze_estimator.extract_features(frame)
            face_detected = _valid_open_features(gaze_estimator, features, blink)

        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        canvas[0:cam_h, 0:cam_w] = frame_small
        cv2.rectangle(
            canvas,
            (head_box_x, head_box_y),
            (head_box_x + head_box_w, head_box_y + head_box_h),
            (0, 0, 0),
            3,
        )
        now_ns = time.monotonic_ns()

        if face_detected:
            if not countdown:
                fd_start_ns = now_ns
                countdown = True
            elapsed = (now_ns - fd_start_ns) / 1_000_000_000
            if elapsed >= dur:
                return True
            progress = elapsed / dur
            eased = progress * progress * (3 - 2 * progress)
            angle = 360 * (1 - eased)
            cv2.ellipse(
                canvas,
                (sw // 2, sh // 2),
                (50, 50),
                0,
                -90,
                -90 + angle,
                (0, 255, 0),
                -1,
            )
        else:
            countdown = False
            fd_start_ns = None
            text = "Face not detected or eye state warming up"
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            cv2.putText(
                canvas,
                text,
                ((sw - size[0]) // 2, (sh + size[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) == 27:
            return False


def _pulse_and_capture(
    gaze_estimator,
    cap,
    pts,
    sw: int,
    sh: int,
    pulse_d: float = 1.0,
    cd_d: float = 1.0,
    *,
    visit_id: str | None = None,
    return_metadata: bool = False,
    window_name: str = "Calibration",
    show_real_time: bool = False,
    show_camera: bool = False,
):
    """Capture fixed targets through A's gate and B's post-ready commit rule."""

    visit_id = visit_id or uuid.uuid4().hex
    capture_id = uuid.uuid4().hex
    gate_config = CalibrationStabilityConfig()
    gate_config_record = calibration_gate_config_record(gate_config)
    gate = CalibrationStabilityGate(gate_config)
    feats: list[np.ndarray] = []
    targs: list[list[float]] = []
    records: list[dict[str, Any]] = []
    target_statuses: list[dict[str, Any]] = []
    receipt_statuses: list[dict[str, Any]] = []
    received_count = 0
    read_attempt_count = 0
    final_radius = 20
    abort_reason = None

    def frame_receipt_id() -> str:
        nonlocal read_attempt_count
        read_attempt_count += 1
        return f"{visit_id}:{capture_id}:read-{read_attempt_count}"

    def finish():
        metadata = {
            "visit_id": visit_id,
            "capture_id": capture_id,
            "samples": records,
            "target_statuses": target_statuses,
            "receipts": receipt_statuses,
            "gate_config": gate_config_record,
            "minimum_open_eye_frames": MIN_OPEN_EYE_FRAMES,
            "minimum_open_eye_span_ms": MIN_OPEN_EYE_SPAN_NS / 1_000_000,
            "maximum_target_capture_ms": MAX_TARGET_CAPTURE_NS / 1_000_000,
            "received_count": received_count,
            "read_attempt_count": read_attempt_count,
            "timestamp_basis": "read_completion_not_exposure",
            "failure_reason": abort_reason,
        }
        record_capture_metadata(gaze_estimator, metadata)
        if return_metadata:
            return feats, targs, metadata
        return None if abort_reason else (feats, targs)

    for target_index, (x, y) in enumerate(pts):
        target_id = (int(x), int(y))
        target_capture_id = uuid.uuid4().hex
        pulse_start_ns = time.monotonic_ns()
        while (time.monotonic_ns() - pulse_start_ns) / 1_000_000_000 < pulse_d:
            ok, frame = cap.read()
            receipt_ns = time.monotonic_ns()
            frame_id = frame_receipt_id()
            if ok and frame is not None:
                received_count += 1
            elapsed = (receipt_ns - pulse_start_ns) / 1_000_000_000
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            final_radius = 15 + int(15 * abs(np.sin(2 * np.pi * elapsed)))
            cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
            cv2.imshow(window_name, canvas)
            receipt_statuses.append({
                "frame_id": frame_id, "visit_id": visit_id, "target_id": target_id,
                "capture_id": target_capture_id, "received_monotonic_ns": receipt_ns,
                "phase": "target_pulse", "status": "not_qualified",
                "reason": "target_not_stationary", "gate_ready": False,
                "reset_required": False, "qualified": False, "committed": False,
            })
            if cv2.waitKey(1) == 27:
                abort_reason = "cancelled_during_target_pulse"
                target_statuses.append({
                    "visit_id": visit_id, "target_id": target_id,
                    "capture_id": target_capture_id, "accepted": False,
                    "status": "cancelled", "reason": abort_reason,
                    "qualified_receipt_count": 0,
                })
                break
        if abort_reason:
            break

        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
        cv2.imshow(window_name, canvas)
        target_display_ns = time.monotonic_ns()
        gate.reset(target_id, visit_id)
        target_receipts: list[dict[str, Any]] = []
        provisional: list[tuple[np.ndarray, int, dict[str, Any]]] = []
        point_times_ns: list[int] = []
        provisional_start_ns = None
        target_seen_images: set[str] = set()
        accepted = False
        target_reason = "target_timeout"

        while True:
            now_ns = time.monotonic_ns()
            elapsed_ns = now_ns - target_display_ns
            if elapsed_ns >= MAX_TARGET_CAPTURE_NS:
                target_reason = "target_timeout"
                break
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
            progress = min(1.0, max(0.0, elapsed_ns / MAX_TARGET_CAPTURE_NS))
            eased = progress * progress * (3 - 2 * progress)
            cv2.ellipse(
                canvas, (x, y), (40, 40), 0, -90, -90 + 360 * (1 - eased),
                (255, 255, 255), 4,
            )
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) == 27:
                abort_reason = "cancelled_during_target_capture"
                target_reason = abort_reason
                break

            if elapsed_ns >= CALIBRATION_SETTLE_NS:
                ok, frame = cap.read()
                received_ns = time.monotonic_ns()
                frame_id = frame_receipt_id()
                if received_ns - target_display_ns >= MAX_TARGET_CAPTURE_NS:
                    target_reason = "target_timeout"
                    receipt_statuses.append({
                        "frame_id": frame_id, "visit_id": visit_id,
                        "target_id": target_id, "capture_id": target_capture_id,
                        "received_monotonic_ns": received_ns,
                        "phase": "target_capture", "status": "rejected",
                        "reason": "target_timeout", "gate_ready": False,
                        "reset_required": False, "qualified": False,
                        "committed": False,
                    })
                    break
                feature = None
                blink = None
                valid_open = False
                observables = None
                image_id = None
                duplicate_image = False
                extraction_error = None

                if ok and frame is not None:
                    received_count += 1
                    image_id = _frame_id(frame)
                    duplicate_image = image_id in target_seen_images
                    target_seen_images.add(image_id)
                    try:
                        feature, blink = gaze_estimator.extract_features(frame)
                        valid_open = _valid_open_features(gaze_estimator, feature, blink)
                        feature_metadata = getattr(gaze_estimator, "last_feature_metadata", {}) or {}
                        observables = feature_metadata.get("stability_observables")
                    except Exception as exc:
                        extraction_error = f"feature_extraction_error:{type(exc).__name__}"
                else:
                    extraction_error = "camera_read_failed"

                observable_values = observables or {}
                decision = gate.update(StabilityObservation(
                    received_monotonic_ns=received_ns,
                    frame_id=frame_id,
                    valid_open=bool(valid_open),
                    eye_uv=observable_values.get("eye_uv"),
                    head_basis_x=observable_values.get("head_basis_x"),
                    head_basis_y=observable_values.get("head_basis_y"),
                    face_center_xy=observable_values.get("face_center_xy"),
                    outer_eye_span_px=observable_values.get("outer_eye_span_px"),
                ))

                if decision.reset_required or duplicate_image or (
                    provisional_start_ns is not None and not decision.ready
                ):
                    provisional.clear()
                    provisional_start_ns = None

                qualified = bool(decision.ready and valid_open and not duplicate_image)
                if qualified:
                    if provisional_start_ns is None:
                        provisional_start_ns = received_ns
                    sample_record = {
                        "frame_id": frame_id, "target_id": target_id, "visit_id": visit_id,
                        "capture_id": target_capture_id,
                        "received_monotonic_ns": received_ns,
                        "target_display_monotonic_ns": target_display_ns,
                        "duplicate_image": False,
                        "timestamp_basis": "read_completion_not_exposure",
                    }
                    provisional.append((np.asarray(feature), received_ns, sample_record))
                if decision.reset_required:
                    receipt_reason = decision.reason
                elif not valid_open:
                    receipt_reason = extraction_error or "invalid_or_blink"
                elif duplicate_image:
                    receipt_reason = "duplicate_raw_frame_not_qualified"
                elif not decision.ready:
                    receipt_reason = decision.reason or "stability_not_ready"
                else:
                    receipt_reason = "provisional_stable_segment"

                receipt = {
                    "frame_id": frame_id, "visit_id": visit_id, "target_id": target_id,
                    "capture_id": target_capture_id, "received_monotonic_ns": received_ns,
                    "phase": "target_capture", "status": "qualified" if qualified else "rejected",
                    "reason": receipt_reason, "gate_reason": decision.reason,
                    "valid_open": bool(valid_open), "blink": blink,
                    "gate_ready": decision.ready, "reset_required": decision.reset_required,
                    "qualified": qualified, "committed": False,
                    "duplicate_image": duplicate_image,
                    "gate_statistics": decision.statistics,
                }
                target_receipts.append(receipt)
                receipt_statuses.append(receipt)

                if show_camera and ok and frame is not None:
                    width, height = min(360, sw), min(270, sh)
                    canvas[:height, :width] = cv2.resize(frame, (width, height))
                if show_real_time and valid_open:
                    from .misc import draw_gaze_cloud
                    predicted = gaze_estimator.predict([feature])[0]
                    if np.all(np.isfinite(predicted)):
                        draw_gaze_cloud(canvas, *predicted)
                if show_camera or show_real_time:
                    cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
                    cv2.imshow(window_name, canvas)

                segment_span_ns = (
                    received_ns - provisional_start_ns if provisional_start_ns is not None else 0
                )
                if len(provisional) >= MIN_OPEN_EYE_FRAMES and segment_span_ns >= MIN_OPEN_EYE_SPAN_NS:
                    accepted = True
                    target_reason = None
                    selected_ids = {sample[2]["frame_id"] for sample in provisional}
                    for receipt in target_receipts:
                        if receipt["frame_id"] in selected_ids:
                            receipt["committed"] = True
                            receipt["status"] = "committed"
                    point_features = [sample[0] for sample in provisional]
                    point_records = [sample[2] for sample in provisional]
                    feats.extend(point_features)
                    targs.extend([[x, y] for _ in point_features])
                    records.extend(point_records)
                    point_times_ns = [sample[1] for sample in provisional]
                    break

        segment_span_ns = (
            point_times_ns[-1] - provisional_start_ns
            if accepted and point_times_ns and provisional_start_ns is not None
            else 0
        )
        target_statuses.append({
            "visit_id": visit_id, "target_id": target_id,
            "capture_id": target_capture_id, "accepted": accepted,
            "status": "accepted" if accepted else ("cancelled" if abort_reason else "failed"),
            "reason": target_reason,
            "qualified_receipt_count": len(point_times_ns) if accepted else len(provisional),
            "continuous_stable_span_ms": segment_span_ns / 1_000_000,
            "first_provisional_monotonic_ns": provisional_start_ns,
            "target_display_monotonic_ns": target_display_ns,
            "target_timeout_ns": MAX_TARGET_CAPTURE_NS,
        })
        if abort_reason:
            break
        if not accepted:
            abort_reason = target_reason or "target_capture_failed"
            break

    if abort_reason:
        for x, y in pts[len(target_statuses):]:
            target_statuses.append({
                "visit_id": visit_id, "target_id": (int(x), int(y)),
                "accepted": False, "status": "not_attempted",
                "reason": "visit_aborted_after_target_failure",
                "qualified_receipt_count": 0,
            })
    return finish()


__all__ = [
    "CALIBRATION_SETTLE_NS",
    "FIT_RESIDUAL_METHOD_VERSION",
    "MIN_OPEN_EYE_FRAMES",
    "MIN_OPEN_EYE_SPAN_NS",
    "MAX_TARGET_CAPTURE_NS",
    "CalibrationResidualResult",
    "TargetFitResidual",
    "_coverage_complete",
    "_pulse_and_capture",
    "calibration_gate_config_record",
    "compute_grid_points",
    "summarize_fit_residuals",
    "wait_for_face_and_countdown",
]
