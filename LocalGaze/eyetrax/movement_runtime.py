"""Camera-frame adapters and bounded evidence storage for movement events.

This module stays outside the gaze feature/training path. It consumes the
current successful capture frame and a detached FaceMesh side-channel value.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from capture_paths import (
    INTERNAL_LAYOUT,
    assert_no_reparse_components,
    normalize_sidecar_layout,
    output_child_path,
    sidecar_path,
)

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


MOTION_ALGORITHM_VERSION = "head-upper-motion-dual-channel-v1"
POSE_SAMPLE_INTERVAL_NS = 100_000_000
POSE_SHOULDER_VISIBILITY_MIN = 0.5
KEYFRAME_MAX_LONG_SIDE = 960
KEYFRAME_JPEG_QUALITY = 88
_CHANNELS = ("head", "upper_body")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return _json_safe(enum_value)
    return str(value)


def _json_line(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class FrameImage:
    frame_id: Any
    received_monotonic_ns: int
    elapsed_s: float | None
    source_width: int
    source_height: int
    image: np.ndarray
    scale: float

    @property
    def stored_width(self) -> int:
        return int(self.image.shape[1])

    @property
    def stored_height(self) -> int:
        return int(self.image.shape[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "source_frame_id": self.frame_id,
            "source_received_monotonic_ns": str(int(self.received_monotonic_ns)),
            "source_elapsed_s": self.elapsed_s,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "stored_width": self.stored_width,
            "stored_height": self.stored_height,
            "scale": self.scale,
            "coordinate_space": "camera_image_px",
        }


class PoseObserver:
    """Throttle MediaPipe Pose on successful source-frame receipt time."""

    def __init__(
        self,
        *,
        sample_interval_ns: int = POSE_SAMPLE_INTERVAL_NS,
        shoulder_visibility_min: float = POSE_SHOULDER_VISIBILITY_MIN,
    ) -> None:
        if isinstance(sample_interval_ns, bool) or int(sample_interval_ns) <= 0:
            raise ValueError("sample_interval_ns must be a positive integer")
        if not 0 <= float(shoulder_visibility_min) <= 1:
            raise ValueError("shoulder_visibility_min must be between zero and one")
        self.sample_interval_ns = int(sample_interval_ns)
        self.shoulder_visibility_min = float(shoulder_visibility_min)
        self._pose = None
        self.available = False
        self.reason: str | None = None
        self.last_sample_ns: int | None = None
        self.latest: dict[str, Any] = {
            "available": False,
            "valid": False,
            "reason": "pose_not_initialized",
            "center_xy_px": None,
            "scale_px": None,
            "shoulder_span_px": None,
            "tilt_deg": None,
            "visibility": {},
            "raw_landmarks_px": {},
            "confidence": None,
        }
        self.process_count = 0
        self.failure_count = 0
        self.processing_durations_ns: list[int] = []
        try:
            import mediapipe as mp

            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.available = True
            self.reason = None
            self.latest = {
                **self.latest,
                "available": True,
                "reason": "awaiting_first_pose_sample",
            }
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            self.latest = {
                **self.latest,
                "reason": "pose_initialization_failed",
                "detail": self.reason,
            }

    @staticmethod
    def _invalid(reason: str, *, available: bool = True, detail: str | None = None) -> dict[str, Any]:
        result = {
            "available": available,
            "valid": False,
            "reason": reason,
            "center_xy_px": None,
            "scale_px": None,
            "shoulder_span_px": None,
            "tilt_deg": None,
            "visibility": {},
            "raw_landmarks_px": {},
            "confidence": None,
        }
        if detail:
            result["detail"] = detail
        return result

    def observe(
        self,
        frame: np.ndarray,
        *,
        frame_id: Any,
        received_monotonic_ns: int,
        elapsed_s: float | None,
    ) -> tuple[dict[str, Any], bool]:
        timestamp = int(received_monotonic_ns)
        if not self.available or self._pose is None:
            result = copy.deepcopy(self.latest)
            result.update(
                attempt_frame_id=frame_id,
                attempt_received_monotonic_ns=str(timestamp),
                attempt_elapsed_s=elapsed_s,
            )
            return result, False

        if self.last_sample_ns is not None:
            if timestamp < self.last_sample_ns or timestamp == self.last_sample_ns:
                result = copy.deepcopy(self.latest)
                result.update(
                    reason="non_monotonic_or_duplicate_pose_time",
                    attempt_frame_id=frame_id,
                    attempt_received_monotonic_ns=str(timestamp),
                    attempt_elapsed_s=elapsed_s,
                )
                return result, False
            if timestamp - self.last_sample_ns < self.sample_interval_ns:
                result = copy.deepcopy(self.latest)
                result.update(
                    attempt_frame_id=frame_id,
                    attempt_received_monotonic_ns=str(timestamp),
                    attempt_elapsed_s=elapsed_s,
                )
                return result, False

        self.last_sample_ns = timestamp
        start_ns = time.perf_counter_ns()
        fresh = True
        try:
            if cv2 is None:
                raise RuntimeError("OpenCV is unavailable")
            if frame is None or getattr(frame, "ndim", 0) < 2:
                raise ValueError("camera frame is invalid")
            height, width = frame.shape[:2]
            if width <= 0 or height <= 0:
                raise ValueError("camera frame dimensions are invalid")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)
            pose = getattr(result, "pose_landmarks", None)
            landmarks = getattr(pose, "landmark", None) if pose is not None else None
            if not landmarks:
                observation = self._invalid("pose_not_detected")
            else:
                indices = {"left_shoulder": 11, "right_shoulder": 12, "left_hip": 23, "right_hip": 24}
                pixels: dict[str, tuple[float, float, float]] = {}
                visibility: dict[str, float | None] = {}
                for name, index in indices.items():
                    landmark = landmarks[index]
                    values = (
                        float(landmark.x) * width,
                        float(landmark.y) * height,
                        float(getattr(landmark, "z", 0.0)) * width,
                    )
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"non-finite Pose landmark: {name}")
                    pixels[name] = values
                    visibility[name] = _float_or_none(getattr(landmark, "visibility", None))

                left = pixels["left_shoulder"]
                right = pixels["right_shoulder"]
                shoulder_span = math.hypot(right[0] - left[0], right[1] - left[1])
                shoulder_vis = (
                    visibility["left_shoulder"],
                    visibility["right_shoulder"],
                )
                center = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
                tilt = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
                raw = {name: [float(value) for value in point] for name, point in pixels.items()}
                confidence = {
                    "left_shoulder_visibility": shoulder_vis[0],
                    "right_shoulder_visibility": shoulder_vis[1],
                    "threshold": self.shoulder_visibility_min,
                }
                if shoulder_span <= 1e-6:
                    observation = self._invalid("degenerate_shoulder_span")
                elif any(value is None or value < self.shoulder_visibility_min for value in shoulder_vis):
                    observation = {
                        **self._invalid("shoulder_visibility_below_threshold"),
                        "visibility": visibility,
                        "raw_landmarks_px": raw,
                        "confidence": confidence,
                    }
                else:
                    hips = (pixels["left_hip"], pixels["right_hip"])
                    torso_center = (
                        (center[0] + (hips[0][0] + hips[1][0]) / 2.0) / 2.0,
                        (center[1] + (hips[0][1] + hips[1][1]) / 2.0) / 2.0,
                    )
                    observation = {
                        "available": True,
                        "valid": True,
                        "reason": None,
                        "center_xy_px": [float(center[0]), float(center[1])],
                        "torso_center_xy_px": [float(torso_center[0]), float(torso_center[1])],
                        "scale_px": float(shoulder_span),
                        "shoulder_span_px": float(shoulder_span),
                        "tilt_deg": float(tilt),
                        "visibility": visibility,
                        "raw_landmarks_px": raw,
                        "confidence": confidence,
                    }
        except Exception as exc:
            self.failure_count += 1
            observation = self._invalid(
                "pose_processing_error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        duration_ns = time.perf_counter_ns() - start_ns
        self.process_count += 1
        self.processing_durations_ns.append(int(duration_ns))
        if len(self.processing_durations_ns) > 4096:
            del self.processing_durations_ns[:2048]
        observation.update(
            source_frame_id=frame_id,
            source_received_monotonic_ns=str(timestamp),
            source_elapsed_s=elapsed_s,
        )
        self.latest = observation
        return copy.deepcopy(observation), fresh

    def summary(self) -> dict[str, Any]:
        durations = self.processing_durations_ns
        ordered = sorted(durations)
        p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)] if ordered else None
        return {
            "available": self.available,
            "reason": self.reason,
            "configuration": {
                "model": "MediaPipe Pose",
                "model_complexity": 1,
                "static_image_mode": False,
                "enable_segmentation": False,
                "sample_interval_ns": self.sample_interval_ns,
                "shoulder_visibility_min": self.shoulder_visibility_min,
                "coordinate_space": "camera_image_px",
            },
            "process_count": self.process_count,
            "failure_count": self.failure_count,
            "processing_duration_ns": {
                "median": int(statistics.median(durations)) if durations else None,
                "p95": int(p95) if p95 is not None else None,
                "max": max(durations) if durations else None,
            },
        }

    def close(self) -> None:
        pose, self._pose = self._pose, None
        if pose is not None:
            try:
                pose.close()
            except Exception:
                pass


class MovementArtifactWriter:
    """Append event changes and save a bounded set of scaled JPEG keyframes."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        session_id: str,
        capture_id: str,
        *,
        save_images: bool = True,
        max_long_side: int = KEYFRAME_MAX_LONG_SIDE,
        jpeg_quality: int = KEYFRAME_JPEG_QUALITY,
        sidecar_layout: str = INTERNAL_LAYOUT,
    ) -> None:
        self.output_dir = assert_no_reparse_components(Path(output_dir).expanduser()).resolve(strict=False)
        self.session_id = str(session_id)
        self.capture_id = str(capture_id)
        if (
            not self.session_id
            or self.session_id in {".", ".."}
            or Path(self.session_id).name != self.session_id
            or not self.capture_id
            or self.capture_id in {".", ".."}
            or Path(self.capture_id).name != self.capture_id
        ):
            raise ValueError("movement artifact identity must use direct-child filename components")
        self.sidecar_layout = normalize_sidecar_layout(sidecar_layout)
        self.save_images = bool(save_images)
        self.max_long_side = int(max_long_side)
        self.jpeg_quality = int(jpeg_quality)
        if self.max_long_side <= 0:
            raise ValueError("max_long_side must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.capture_dir: Path | None = None
        self.keyframe_dir: Path | None = None
        self.event_log_path: Path | None = None
        self._event_log = None
        self.storage_error: str | None = None
        self.keyframes: list[dict[str, Any]] = []
        self._event_keyframes: dict[str, list[dict[str, Any]]] = {}
        self._event_counter = 0
        self._lock = None
        try:
            base = output_child_path(self.output_dir, f"motion_artifacts/{self.session_id}")
            assert_no_reparse_components(base)
            base.mkdir(parents=True, exist_ok=True)
            capture_dir = output_child_path(self.output_dir, f"motion_artifacts/{self.session_id}/{self.capture_id}")
            assert_no_reparse_components(capture_dir)
            capture_dir.mkdir(exist_ok=False)
            self.capture_dir = capture_dir
            event_log_name = f"{self.session_id}.{self.capture_id}.movement-events.jsonl"
            self.event_log_path = (
                sidecar_path(
                    self.output_dir,
                    event_log_name,
                    layout=self.sidecar_layout,
                    create_parent=self.sidecar_layout == INTERNAL_LAYOUT,
                )
                if self.sidecar_layout == INTERNAL_LAYOUT
                else output_child_path(
                    self.output_dir,
                    f"motion_artifacts/{self.session_id}/{self.capture_id}/movement-events.jsonl",
                )
            )
            self._event_log = self.event_log_path.open("x", encoding="utf-8", newline="\n")
            if self.save_images:
                self.keyframe_dir = output_child_path(
                    self.output_dir,
                    f"motion_artifacts/{self.session_id}/{self.capture_id}/keyframes",
                )
                assert_no_reparse_components(self.keyframe_dir)
                self.keyframe_dir.mkdir(exist_ok=False)
        except Exception as exc:
            self.storage_error = f"{type(exc).__name__}: {exc}"
            self.close()

    def make_snapshot(self, frame: np.ndarray, packet) -> FrameImage | None:
        if frame is None or cv2 is None:
            return None
        try:
            height, width = frame.shape[:2]
            if width <= 0 or height <= 0:
                return None
            scale = min(1.0, self.max_long_side / float(max(width, height)))
            if scale < 1.0:
                resized = cv2.resize(
                    frame,
                    (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                resized = frame.copy()
            return FrameImage(
                frame_id=packet.frame_id,
                received_monotonic_ns=int(packet.received_monotonic_ns),
                elapsed_s=_float_or_none(packet.elapsed_s),
                source_width=int(width),
                source_height=int(height),
                image=resized,
                scale=float(scale),
            )
        except Exception:
            return None

    @staticmethod
    def _safe_token(value: Any) -> str:
        token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
        return token[:64] or "event"

    def append_update(self, update: dict[str, Any]) -> None:
        if self._event_log is None:
            return
        try:
            self._event_log.write(_json_line({
                "record_type": "event_update",
                "capture_id": self.capture_id,
                "written_at_monotonic_ns": str(time.monotonic_ns()),
                **update,
            }) + "\n")
            self._event_log.flush()
            if update.get("kind") == "start":
                os.fsync(self._event_log.fileno())
        except Exception as exc:
            self.storage_error = self.storage_error or f"{type(exc).__name__}: {exc}"

    def record_keyframe(
        self,
        event_id: Any,
        keyframe_type: str,
        snapshot: FrameImage | None,
        *,
        reason: str | None = None,
        event_start: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_token = str(event_id)
        rows = self._event_keyframes.setdefault(event_token, [])
        if len(rows) >= 3:
            row = {
                "event_id": event_id,
                "capture_id": self.capture_id,
                "keyframe_type": keyframe_type,
                "status": "not_saved",
                "reason": "event_keyframe_limit_reached",
            }
        elif not self.save_images:
            row = {
                "event_id": event_id,
                "capture_id": self.capture_id,
                "keyframe_type": keyframe_type,
                "status": "not_saved",
                "reason": "disabled_by_user",
            }
        elif self.keyframe_dir is None:
            row = {
                "event_id": event_id,
                "capture_id": self.capture_id,
                "keyframe_type": keyframe_type,
                "status": "failed",
                "reason": self.storage_error or "keyframe_directory_unavailable",
            }
        elif snapshot is None:
            row = {
                "event_id": event_id,
                "capture_id": self.capture_id,
                "keyframe_type": keyframe_type,
                "status": "failed",
                "reason": reason or "source_frame_not_available",
            }
        else:
            self._event_counter += 1
            filename = (
                f"{self._safe_token(event_id)}-{self._event_counter:06d}-"
                f"{self._safe_token(keyframe_type)}.jpg"
            )
            path = self.keyframe_dir / filename
            row = {
                "event_id": event_id,
                "capture_id": self.capture_id,
                "keyframe_type": keyframe_type,
                "filename": filename,
                "relative_path": path.relative_to(self.output_dir).as_posix(),
                "status": "failed",
                **snapshot.metadata(),
            }
            try:
                if cv2 is None:
                    raise RuntimeError("OpenCV is unavailable for image encoding")
                ok, encoded = cv2.imencode(
                    ".jpg",
                    snapshot.image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("OpenCV JPEG encoding returned false")
                with path.open("xb") as output:
                    output.write(encoded.tobytes())
                    output.flush()
                    os.fsync(output.fileno())
                row["status"] = "saved"
                row["reason"] = None
            except Exception as exc:
                row["reason"] = f"{type(exc).__name__}: {exc}"
            if event_start:
                start = event_start.get("start") if isinstance(event_start, dict) else None
                start = start if isinstance(start, dict) else {}
                row["event_start_frame_id"] = start.get("frame_id")
                row["event_start_received_monotonic_ns"] = start.get("received_ns")
                start_elapsed = _float_or_none(start.get("elapsed_s"))
                if start_elapsed is not None and snapshot.elapsed_s is not None:
                    row["image_lag_from_onset_s"] = max(0.0, snapshot.elapsed_s - start_elapsed)
                row["image_after_onset"] = (
                    start.get("frame_id") != snapshot.frame_id
                    or (start_elapsed is not None and snapshot.elapsed_s is not None and snapshot.elapsed_s > start_elapsed)
                )
        rows.append(row)
        self.keyframes.append(row)
        self.append_update({
            "kind": "keyframe",
            "event_id": event_id,
            "keyframe": row,
        })
        return copy.deepcopy(row)

    def event_keyframes(self, event_id: Any) -> list[dict[str, Any]]:
        return copy.deepcopy(self._event_keyframes.get(str(event_id), []))

    def summary(self) -> dict[str, Any]:
        failures = [item for item in self.keyframes if item.get("status") == "failed"]
        if not self.save_images:
            status = "disabled_by_user"
        elif self.storage_error or failures:
            status = "partial_or_failed"
        else:
            status = "enabled"
        return {
            "capture_id": self.capture_id,
            "status": status,
            "save_images": self.save_images,
            "image_format": "JPEG",
            "jpeg_quality": self.jpeg_quality,
            "max_long_side_px": self.max_long_side,
            "coordinate_space": "camera_image_px",
            "capture_directory": (
                self.capture_dir.relative_to(self.output_dir).as_posix()
                if self.capture_dir is not None
                else None
            ),
            "event_log": (
                self.event_log_path.relative_to(self.output_dir).as_posix()
                if self.event_log_path is not None and self.event_log_path.exists()
                else None
            ),
            "keyframe_directory": (
                self.keyframe_dir.relative_to(self.output_dir).as_posix()
                if self.keyframe_dir is not None
                else None
            ),
            "keyframes": copy.deepcopy(self.keyframes),
            "write_failures": failures,
            "storage_error": self.storage_error,
        }

    def close(self) -> None:
        if self._event_log is not None:
            try:
                self._event_log.flush()
                self._event_log.close()
            except Exception:
                pass
            self._event_log = None


class MovementRuntime:
    """Connect camera/head/body observations to the pure MotionMonitor."""

    def __init__(
        self,
        *,
        session_id: str,
        output_dir: str | os.PathLike[str],
        capture_id: str | None = None,
        config=None,
        save_images: bool = True,
        sidecar_layout: str = INTERNAL_LAYOUT,
    ) -> None:
        from eyetrax.movement import MotionConfig, MotionMonitor

        self.session_id = str(session_id)
        self.capture_id = str(capture_id or uuid.uuid4().hex)
        self.config = config if config is not None else MotionConfig()
        self.monitor = MotionMonitor(
            self.session_id,
            self.config,
            capture_id=self.capture_id,
        )
        self.pose = PoseObserver()
        self.writer = MovementArtifactWriter(
            output_dir,
            self.session_id,
            self.capture_id,
            save_images=save_images,
            sidecar_layout=sidecar_layout,
        )
        self._recent_frames: dict[str, FrameImage] = {}
        self._peak_frames: dict[str, FrameImage | None] = {}
        self._events_by_id: dict[str, dict[str, Any]] = {}
        self._active_ids: dict[str, set[str]] = {channel: set() for channel in _CHANNELS}
        self._last_frame_state: dict[str, Any] = {}
        self._last_metrics: dict[str, Any] = {}
        self._last_raw: dict[str, Any] = {}
        self._last_packet = None
        self._finished = False
        self._final_result: dict[str, Any] | None = None
        self._error: str | None = None
        self._last_status_signature = None

    @staticmethod
    def _sample_head(observation: Any, packet) -> dict[str, Any]:
        if hasattr(observation, "to_dict"):
            observation = observation.to_dict()
        elif not isinstance(observation, dict):
            observation = {}
        head = copy.deepcopy(observation)
        head.setdefault("available", True)
        head.setdefault("valid", False)
        head.setdefault("reason", "head_observation_unavailable")
        if head.get("reason") == "no_face":
            # FaceMesh is available, but it did not observe a face on this
            # source frame. Keep this as tracking evidence rather than a
            # session-level capability failure.
            head["available"] = True
        if "raw_landmarks" not in head:
            head["raw_landmarks"] = copy.deepcopy(head.get("raw_points_px", {}))
        head["source_frame_id"] = packet.frame_id
        head["source_received_monotonic_ns"] = str(int(packet.received_monotonic_ns))
        head["source_elapsed_s"] = _float_or_none(packet.elapsed_s)
        head["confidence"] = None
        head["landmark_confidence"] = None
        head["confidence_reason"] = (
            "FaceMesh side-channel exposes no per-landmark confidence for cached head indices"
        )
        return head

    @staticmethod
    def _event_value(update: dict[str, Any]) -> dict[str, Any]:
        event = update.get("event")
        if isinstance(event, dict):
            return copy.deepcopy(event)
        return {}

    @staticmethod
    def _frame_id_of(sample: Any) -> Any:
        return sample.get("frame_id") if isinstance(sample, dict) else None

    def _matching_frame(
        self,
        sample: Any,
        *,
        current: FrameImage | None,
        previous: FrameImage | None,
    ) -> FrameImage | None:
        frame_id = self._frame_id_of(sample)
        if current is not None and current.frame_id == frame_id:
            return current
        if previous is not None and previous.frame_id == frame_id:
            return previous
        return None

    @staticmethod
    def _is_instant(event: dict[str, Any]) -> bool:
        modes = event.get("trigger_modes")
        if isinstance(modes, (list, tuple, set)):
            if any("瞬时" in str(mode) or "instant" in str(mode).lower() for mode in modes):
                return True
        mode = str(event.get("trigger_mode", "")).lower()
        return "瞬时" in mode or "instant" in mode

    def _record_instant_pair(
        self,
        event_id: Any,
        event: dict[str, Any],
        *,
        current: FrameImage | None,
        previous: FrameImage | None,
    ) -> None:
        already = {row.get("keyframe_type") for row in self.writer.event_keyframes(event_id)}
        trigger = event.get("trigger_sample")
        trigger_samples = event.get("trigger_samples")
        if isinstance(trigger_samples, list):
            instant = next(
                (
                    item for item in trigger_samples
                    if isinstance(item, dict)
                    and "instant" in str(item.get("trigger_mode", "")).lower()
                ),
                None,
            )
            if instant is not None:
                trigger = instant
        if not isinstance(trigger, dict):
            trigger = {}
        before = trigger.get("before")
        after = trigger.get("after")
        if "before" not in already:
            self.writer.record_keyframe(
                event_id,
                "before",
                self._matching_frame(before, current=current, previous=previous),
                reason="before_source_frame_not_cached",
                event_start=event,
            )
        if "after" not in already:
            self.writer.record_keyframe(
                event_id,
                "after",
                self._matching_frame(after, current=current, previous=previous),
                reason="after_source_frame_not_cached",
                event_start=event,
            )

    def _handle_update(
        self,
        update: dict[str, Any],
        *,
        current: FrameImage | None,
        previous_by_channel: dict[str, FrameImage],
    ) -> None:
        kind = str(update.get("kind", ""))
        event_id = update.get("event_id")
        channel = str(update.get("channel", ""))
        event = self._event_value(update)
        if not event and isinstance(update.get("event"), dict):
            event = update["event"]
        if event_id is not None:
            token = str(event_id)
            if event:
                self._events_by_id[token] = event
            if kind in {"start", "trigger", "peak", "end"} and channel in self._active_ids:
                if kind == "start":
                    self._active_ids[channel].add(token)
                elif kind == "end":
                    self._active_ids[channel].discard(token)
        self.writer.append_update(update)

        if kind == "start" and event_id is not None:
            if self._is_instant(event):
                self._record_instant_pair(
                    event_id,
                    event,
                    current=current,
                    previous=previous_by_channel.get(channel),
                )
            else:
                self.writer.record_keyframe(
                    event_id,
                    "start",
                    current,
                    reason="trigger_frame_image_unavailable",
                    event_start=event,
                )
        elif kind == "trigger" and event_id is not None and self._is_instant(event):
            self._record_instant_pair(
                event_id,
                event,
                current=current,
                previous=previous_by_channel.get(channel),
            )
        elif kind == "peak" and event_id is not None:
            sample = event.get("peak_sample")
            peak_frame = self._matching_frame(
                sample,
                current=current,
                previous=previous_by_channel.get(channel),
            )
            if peak_frame is not None:
                self._peak_frames[str(event_id)] = peak_frame
        elif kind == "end" and event_id is not None:
            peak = self._peak_frames.pop(str(event_id), None)
            rows = self.writer.event_keyframes(event_id)
            if self._is_instant(event):
                if not any(row.get("keyframe_type") == "before" for row in rows):
                    self._record_instant_pair(
                        event_id,
                        event,
                        current=current,
                        previous=previous_by_channel.get(channel),
                    )
            elif peak is not None:
                self.writer.record_keyframe(
                    event_id, "peak", peak, event_start=event
                )
            rows = self.writer.event_keyframes(event_id)
            if len(rows) < 3:
                self.writer.record_keyframe(
                    event_id,
                    "end",
                    current or previous_by_channel.get(channel),
                    reason="end_source_frame_image_unavailable",
                    event_start=event,
                )

    @staticmethod
    def _state_text(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("state", "status", "motion", "tracking"):
                if value.get(key) is not None:
                    return str(value[key])
        return str(value) if value is not None else "unknown"

    def _channel_status(self, channel: str) -> str:
        state_key = "upper_body" if channel == "upper_body" else "head"
        state = self._last_frame_state.get(state_key)
        observation = (
            self._last_raw.get("upper_body")
            if channel == "upper_body"
            else self._last_raw.get("head")
        )
        state_text = self._state_text(state).lower()
        if "lost" in state_text or "丢失" in state_text or "tracking_lost" in state_text:
            return "丢失"
        if (
            "unavailable" in state_text
            or (isinstance(observation, dict) and observation.get("available") is False)
        ):
            return "不可用"
        active = self._active_ids.get(channel, set())
        if active:
            if any("tracking" in str(event.get("type", "")) for key, event in self._events_by_id.items() if key in active):
                return "丢失"
            return "运动"
        if isinstance(observation, dict) and observation.get("valid") is False and observation.get("reason") not in {
            None, "awaiting_first_pose_sample"
        }:
            return "丢失"
        return "正常"

    def status_text(self) -> str:
        if self._error:
            return f"运动监测不可用：{self._error}"
        event_ids = set()
        for values in self._active_ids.values():
            event_ids.update(values)
        completed = getattr(self.monitor, "events", [])
        if callable(completed):
            try:
                completed = completed()
            except Exception:
                completed = []
        if isinstance(completed, list):
            event_ids.update(str(item.get("event_id")) for item in completed if isinstance(item, dict) and item.get("event_id") is not None)
        image_summary = self.writer.summary()
        image_status = "关闭" if not image_summary["save_images"] else "有失败" if image_summary["write_failures"] or image_summary["storage_error"] else "开启"
        return (
            f"头部：{self._channel_status('head')}  "
            f"上身：{self._channel_status('upper_body')}  "
            f"事件数：{len(event_ids)}  截图：{image_status}"
        )

    def _row_fields(
        self,
        *,
        head: dict[str, Any],
        body: dict[str, Any],
        body_fresh: bool,
        frame_state: dict[str, Any],
        metrics: dict[str, Any],
        raw: dict[str, Any],
        current_event_ids: dict[str, Any],
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "motion_enabled": True,
            "motion_capture_id": self.capture_id,
            "motion_head_fresh": True,
            "motion_head_available": bool(head.get("available")),
            "motion_head_valid": bool(head.get("valid")),
            "motion_head_reason": head.get("reason"),
            "motion_body_fresh": bool(body_fresh),
            "motion_body_available": bool(body.get("available")),
            "motion_body_valid": bool(body.get("valid")),
            "motion_body_reason": body.get("reason"),
            "motion_body_frame_id": body.get("source_frame_id"),
            "motion_body_received_monotonic_ns": body.get("source_received_monotonic_ns"),
            "motion_head_observation_json": _json_line(head),
            "motion_body_observation_json": _json_line(body),
            "motion_frame_state_json": _json_line(frame_state),
            "motion_metrics_json": _json_line(metrics),
            "motion_raw_json": _json_line(raw),
            "motion_current_event_ids_json": _json_line(current_event_ids),
            "motion_event_updates_json": _json_line([
                {
                    "kind": item.get("kind"),
                    "event_id": item.get("event_id"),
                    "capture_id": item.get("capture_id", self.capture_id),
                    "channel": item.get("channel"),
                    "event": (
                        copy.deepcopy(item.get("event"))
                        if isinstance(item.get("event"), dict)
                        else None
                    ),
                    "event_type": (
                        item.get("event", {}).get("type")
                        if isinstance(item.get("event"), dict)
                        else None
                    ),
                    "trigger_mode": (
                        item.get("event", {}).get("trigger_mode")
                        if isinstance(item.get("event"), dict)
                        else None
                    ),
                    "candidate_label": (
                        item.get("event", {}).get("candidate_label")
                        if isinstance(item.get("event"), dict)
                        else None
                    ),
                }
                for item in updates
            ]),
        }

    def observe(
        self,
        frame: np.ndarray,
        packet,
        head_observation: Any,
    ) -> dict[str, Any]:
        if self._finished:
            return self._row_fields(
                head={}, body={}, body_fresh=False, frame_state={},
                metrics={}, raw={}, current_event_ids={}, updates=[],
            )
        try:
            head = self._sample_head(head_observation, packet)
            body, body_fresh = self.pose.observe(
                frame,
                frame_id=packet.frame_id,
                received_monotonic_ns=packet.received_monotonic_ns,
                elapsed_s=packet.elapsed_s,
            )
            if body.get("reason") == "pose_not_detected":
                body["source_observation_reason"] = body["reason"]
                body["reason"] = "no_pose"
            body.setdefault("raw_landmarks", copy.deepcopy(body.get("raw_landmarks_px", {})))
            snapshot = (
                self.writer.make_snapshot(frame, packet)
                if self.writer.save_images
                else None
            )
            previous_by_channel = dict(self._recent_frames)
            result = self.monitor.observe(
                packet.frame_id,
                packet.received_monotonic_ns,
                packet.elapsed_s,
                head,
                body,
                body_fresh=body_fresh,
            )
            result = result if isinstance(result, dict) else {}
            frame_state = result.get("frame_state") or {}
            metrics = result.get("metrics") or {}
            raw = result.get("raw") or {"head": head, "upper_body": body, "body_fresh": body_fresh}
            current_event_ids = result.get("current_event_ids") or {}
            updates = result.get("event_updates") or []
            if not isinstance(updates, list):
                updates = []

            for update in updates:
                if isinstance(update, dict):
                    self._handle_update(
                        update,
                        current=snapshot,
                        previous_by_channel=previous_by_channel,
                    )
            for channel, observation, is_fresh in (
                ("head", head, bool(head.get("available") and head.get("valid"))),
                ("upper_body", body, bool(body_fresh and body.get("available") and body.get("valid"))),
            ):
                if is_fresh and snapshot is not None:
                    self._recent_frames[channel] = snapshot

            self._last_frame_state = copy.deepcopy(frame_state)
            self._last_metrics = copy.deepcopy(metrics)
            self._last_raw = copy.deepcopy(raw)
            self._last_packet = packet
            return self._row_fields(
                head=head,
                body=body,
                body_fresh=body_fresh,
                frame_state=frame_state,
                metrics=metrics,
                raw=raw,
                current_event_ids=current_event_ids,
                updates=updates,
            )
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self.writer.storage_error = self.writer.storage_error or self._error
            return {
                "motion_enabled": True,
                "motion_capture_id": self.capture_id,
                "motion_head_fresh": True,
                "motion_head_available": False,
                "motion_head_valid": False,
                "motion_head_reason": self._error,
                "motion_body_fresh": False,
                "motion_body_available": False,
                "motion_body_valid": False,
                "motion_body_reason": "movement_processing_error",
                "motion_body_frame_id": None,
                "motion_body_received_monotonic_ns": None,
                "motion_head_observation_json": _json_line({"available": False, "valid": False, "reason": self._error}),
                "motion_body_observation_json": _json_line({"available": False, "valid": False, "reason": "movement_processing_error"}),
                "motion_frame_state_json": "{}",
                "motion_metrics_json": "{}",
                "motion_raw_json": "{}",
                "motion_current_event_ids_json": "{}",
                "motion_event_updates_json": "[]",
            }

    def _handle_finish_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        reason: str,
    ) -> None:
        # Finish events at the last valid source frame for their own channel.
        # The head and Pose channels are sampled at different rates, so sharing
        # one "current" image here can attach a head frame to an upper-body end.
        current = None
        previous = dict(self._recent_frames)
        for update in updates:
            if isinstance(update, dict):
                self._handle_update(update, current=current, previous_by_channel=previous)

    def finish(
        self,
        *,
        last_packet=None,
        reason: str = "capture_stopped",
    ) -> dict[str, Any]:
        try:
            return self._finish_capture(last_packet=last_packet, reason=reason)
        finally:
            self._peak_frames.clear()

    def _finish_capture(
        self,
        *,
        last_packet=None,
        reason: str = "capture_stopped",
    ) -> dict[str, Any]:
        if self._final_result is not None:
            return copy.deepcopy(self._final_result)
        updates: list[dict[str, Any]] = []
        if not self._finished:
            self._finished = True
            try:
                updates = self.monitor.finish(
                    last_frame_id=getattr(last_packet, "frame_id", None),
                    received_ns=getattr(last_packet, "received_monotonic_ns", None),
                    elapsed_s=getattr(last_packet, "elapsed_s", None),
                    reason=reason,
                ) or []
            except Exception as exc:
                self._error = self._error or f"{type(exc).__name__}: {exc}"
            if isinstance(updates, list):
                self._handle_finish_updates(updates, reason=reason)
        self.pose.close()
        events = getattr(self.monitor, "events", [])
        if callable(events):
            try:
                events = events()
            except Exception:
                events = []
        if not isinstance(events, list):
            events = []
        completed: list[dict[str, Any]] = []
        for item in events:
            if not isinstance(item, dict):
                continue
            event = copy.deepcopy(item)
            event_id = event.get("event_id")
            event["keyframes"] = self.writer.event_keyframes(event_id) if event_id is not None else []
            completed.append(event)
        summary_fn = getattr(self.monitor, "summary", None)
        try:
            monitor_summary = summary_fn() if callable(summary_fn) else {}
        except Exception as exc:
            monitor_summary = {"error": f"{type(exc).__name__}: {exc}"}
        config_fn = getattr(self.config, "to_dict", None)
        try:
            config_data = config_fn() if callable(config_fn) else {}
        except Exception:
            config_data = {}
        self._final_result = {
            "capture_id": self.capture_id,
            "enabled": True,
            "algorithm_version": MOTION_ALGORITHM_VERSION,
            "coordinate_space": "camera_image_px",
            "config": config_data,
            "monitor": _json_safe(monitor_summary),
            "pose": self.pose.summary(),
            "artifacts": self.writer.summary(),
            "events": completed,
            "keyframes": copy.deepcopy(self.writer.keyframes),
            "event_count": len(completed),
            "keyframe_failure_count": sum(
                item.get("status") == "failed" for item in self.writer.keyframes
            ),
            "error": self._error,
            "stop_reason": reason,
        }
        self.writer.append_update({
            "kind": "capture_finish",
            "capture_id": self.capture_id,
            "reason": reason,
            "event_count": len(completed),
        })
        self.writer.close()
        self._final_result["artifacts"] = self.writer.summary()
        return copy.deepcopy(self._final_result)


__all__ = [
    "KEYFRAME_MAX_LONG_SIDE",
    "MOTION_ALGORITHM_VERSION",
    "POSE_SAMPLE_INTERVAL_NS",
    "MovementArtifactWriter",
    "MovementRuntime",
    "PoseObserver",
]
