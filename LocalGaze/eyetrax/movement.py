"""Pure head and upper-body motion metrics and event state machine.

This module consumes already-computed landmarks and timestamps. It has no
camera, MediaPipe, OpenCV, filesystem, or UI dependencies. Thresholds are
adjustable engineering starting points; they are not research or clinical
cutoffs. Head orientation is a landmark-basis direction proxy, not yaw/pitch
ground truth.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from math import acos, degrees, hypot, isclose, isfinite, log
from numbers import Integral, Real
from typing import Any, Mapping


MOVEMENT_SCHEMA_VERSION = "movement/1.0"
MOVEMENT_ALGORITHM_VERSION = "head-upper-motion-dual-channel-v1"
_NANOSECONDS_PER_SECOND = 1_000_000_000
_CHANNELS = ("head", "upper_body")
_MISSING_REASONS = {
    "no_detection", "no_face", "face_not_found", "no_landmarks",
    "landmarks_missing", "detection_missing", "tracking_lost", "no_pose",
}


def _meets(value: float, threshold: float) -> bool:
    """Compare threshold boundaries without losing exact decimal intervals to float rounding."""

    return value >= threshold or isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


@dataclass(frozen=True)
class MotionConfig:
    """Versioned motion thresholds with explicit units.

    All defaults are adjustable initial engineering values, not validated
    thresholds for human movement or diagnosis.
    """

    version: str = MOVEMENT_SCHEMA_VERSION
    head_translation_ref_width_per_s: float = 1.0
    body_translation_shoulder_width_per_s: float = 0.75
    head_orientation_proxy_deg_per_s: float = 80.0
    body_tilt_deg_per_s: float = 60.0
    head_abs_log_scale_per_s: float = 0.8
    body_abs_log_scale_per_s: float = 0.8
    enter_score: float = 1.0
    enter_duration_s: float = 0.20
    enter_min_fresh_samples: int = 2
    exit_score: float = 0.65
    exit_duration_s: float = 0.30
    tracking_lost_after_s: float = 0.30
    max_baseline_gap_s: float = 0.75
    instant_enabled: bool = True
    instant_min_score: float = 3.0
    instant_max_dt_s: float = 0.25
    instant_center_step_ref_width: float = 0.40
    instant_head_orientation_step_deg: float = 25.0
    instant_body_tilt_step_deg: float = 20.0
    instant_abs_log_scale_step: float = 0.25
    recent_frame_id_cache_size: int = 2048

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty string")
        positive = (
            "head_translation_ref_width_per_s",
            "body_translation_shoulder_width_per_s",
            "head_orientation_proxy_deg_per_s",
            "body_tilt_deg_per_s",
            "head_abs_log_scale_per_s",
            "body_abs_log_scale_per_s",
            "enter_score",
            "enter_duration_s",
            "exit_duration_s",
            "tracking_lost_after_s",
            "max_baseline_gap_s",
            "instant_min_score",
            "instant_max_dt_s",
            "instant_center_step_ref_width",
            "instant_head_orientation_step_deg",
            "instant_body_tilt_step_deg",
            "instant_abs_log_scale_step",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite number")
            if not isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if (
            isinstance(self.exit_score, bool)
            or not isinstance(self.exit_score, Real)
            or not isfinite(float(self.exit_score))
            or float(self.exit_score) < 0
            or float(self.exit_score) >= float(self.enter_score)
        ):
            raise ValueError("exit_score must be finite, nonnegative, and below enter_score")
        if (
            isinstance(self.enter_min_fresh_samples, bool)
            or not isinstance(self.enter_min_fresh_samples, Integral)
            or int(self.enter_min_fresh_samples) < 2
        ):
            raise ValueError("enter_min_fresh_samples must be an integer of at least 2")
        if (
            isinstance(self.recent_frame_id_cache_size, bool)
            or not isinstance(self.recent_frame_id_cache_size, Integral)
            or int(self.recent_frame_id_cache_size) < 1
        ):
            raise ValueError("recent_frame_id_cache_size must be a positive integer")
        if not isinstance(self.instant_enabled, bool):
            raise TypeError("instant_enabled must be a bool")

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-compatible algorithm/config manifest."""

        thresholds = {
            "head_translation_ref_width_per_s": float(self.head_translation_ref_width_per_s),
            "body_translation_shoulder_width_per_s": float(
                self.body_translation_shoulder_width_per_s
            ),
            "head_orientation_proxy_deg_per_s": float(
                self.head_orientation_proxy_deg_per_s
            ),
            "body_tilt_deg_per_s": float(self.body_tilt_deg_per_s),
            "head_abs_log_scale_per_s": float(self.head_abs_log_scale_per_s),
            "body_abs_log_scale_per_s": float(self.body_abs_log_scale_per_s),
        }
        return {
            "schema_version": MOVEMENT_SCHEMA_VERSION,
            "version": self.version,
            "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
            "defaults_status": "adjustable_engineering_initial_values_not_research_thresholds",
            "units": {
                "head_translation_ref_width_per_s": "outer_eye_span_reference_width/s",
                "body_translation_shoulder_width_per_s": "shoulder_span_reference_width/s",
                "head_orientation_proxy_deg_per_s": "landmark_basis_direction_proxy_deg/s",
                "body_tilt_deg_per_s": "shoulder_line_angle_deg/s",
                "head_abs_log_scale_per_s": "absolute_log_scale_ratio/s",
                "body_abs_log_scale_per_s": "absolute_log_scale_ratio/s",
                "instant_center_step": "reference_widths",
                "instant_head_orientation_step": "landmark_basis_direction_proxy_deg",
                "instant_body_tilt_step": "shoulder_line_angle_deg",
                "instant_abs_log_scale_step": "absolute_log_scale_ratio",
                "time": "seconds",
            },
            "thresholds": thresholds,
            "sustained": {
                "enter_score": float(self.enter_score),
                "enter_duration_s": float(self.enter_duration_s),
                "enter_min_fresh_samples": int(self.enter_min_fresh_samples),
                "exit_score": float(self.exit_score),
                "exit_duration_s": float(self.exit_duration_s),
            },
            "tracking_lost": {"confirm_after_s": float(self.tracking_lost_after_s)},
            "reset": {
                "max_baseline_gap_s": float(self.max_baseline_gap_s),
                "nonmonotonic_time_resets_baseline": True,
                "duplicate_or_nonmonotonic_frame_resets_baseline": True,
                "reacquisition_resets_baseline": True,
            },
            "instantaneous": {
                "enabled": self.instant_enabled,
                "min_score": float(self.instant_min_score),
                "max_dt_s": float(self.instant_max_dt_s),
                "center_step_ref_width": float(self.instant_center_step_ref_width),
                "head_orientation_proxy_step_deg": float(
                    self.instant_head_orientation_step_deg
                ),
                "body_tilt_step_deg": float(self.instant_body_tilt_step_deg),
                "abs_log_scale_step": float(self.instant_abs_log_scale_step),
                "label": "瞬时大幅变化/待复核",
                "requires_two_adjacent_valid_fresh_samples": True,
            },
            "event_merge": {"strategy": "exit_hysteresis_only", "merge_gap_s": None},
            "cache_limits": {
                "recent_frame_id_cache_size": int(self.recent_frame_id_cache_size),
                "stored_raw_frame_history": False,
                "trigger_samples_per_event_mode": 1,
            },
        }


@dataclass(frozen=True)
class _Clock:
    frame_id: Any
    received_ns: int
    elapsed_s: float

    def stamp(self) -> dict[str, Any]:
        return {
            "frame_id": _json_safe(self.frame_id),
            "received_ns": str(self.received_ns),
            "elapsed_s": float(self.elapsed_s),
        }


def _json_safe(value: Any, *, _depth: int = 0, _budget: list[int] | None = None) -> Any:
    """Convert a bounded slice of common Python/NumPy values to JSON primitives."""

    if _budget is None:
        _budget = [4096]
    _budget[0] -= 1
    if _budget[0] < 0 or _depth > 8:
        return "[truncated]"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        number = float(value)
        return number if isfinite(number) else None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 256 or _budget[0] <= 0:
                result["_truncated"] = True
                break
            result[str(key)] = _json_safe(item, _depth=_depth + 1, _budget=_budget)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            if index >= 2048 or _budget[0] <= 0:
                result.append("[truncated]")
                break
            result.append(_json_safe(item, _depth=_depth + 1, _budget=_budget))
        return result
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist(), _depth=_depth + 1, _budget=_budget)
        except (TypeError, ValueError, OverflowError):
            return str(value)
    return str(value)


def _safe_id(value: Any) -> Any:
    if value is None:
        raise ValueError("frame_id must not be None")
    try:
        hash(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("frame_id must be hashable") from exc
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("frame_id numeric value must be finite")
        return value
    return str(value)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _vector(value: Any, *, dimensions: tuple[int, ...] = (2,)) -> tuple[float, ...] | None:
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            value = value.tolist()
        except (TypeError, ValueError, OverflowError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) not in dimensions:
        return None
    numbers = tuple(_finite_number(item) for item in value)
    if any(item is None for item in numbers):
        return None
    return tuple(float(item) for item in numbers if item is not None)


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return hypot(*(x - y for x, y in zip(a, b)))


def _wrap_degrees(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def _unit_basis(raw: Any) -> tuple[float, ...] | None:
    values = _vector(raw, dimensions=(2, 3))
    if values is None:
        return None
    norm = hypot(*values)
    if not isfinite(norm) or norm <= 1e-12:
        return None
    return tuple(value / norm for value in values)


def _angle_between(a: tuple[float, ...], b: tuple[float, ...]) -> float | None:
    if len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    if not isfinite(dot):
        return None
    return degrees(acos(max(-1.0, min(1.0, dot))))


def _clock_from_inputs(
    frame_id: Any, received_ns: Any, elapsed_s: Any
) -> tuple[_Clock | None, str | None]:
    try:
        safe_frame_id = _safe_id(frame_id)
    except (TypeError, ValueError) as exc:
        return None, "invalid_frame_id:" + str(exc)
    if isinstance(received_ns, bool) or not isinstance(received_ns, Integral):
        return None, "received_ns_must_be_an_integer"
    received_value = int(received_ns)
    if received_value < 0:
        return None, "received_ns_must_be_nonnegative"
    elapsed_value = _finite_number(elapsed_s)
    if elapsed_value is None or elapsed_value < 0:
        return None, "elapsed_s_must_be_finite_and_nonnegative"
    return _Clock(safe_frame_id, received_value, elapsed_value), None


def _canonical_values(
    channel: str, raw: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Extract geometry components separately so one bad metric need not erase others."""

    errors: dict[str, str] = {}
    center_raw = raw.get("center_xy_px", raw.get("center_xy"))
    if channel == "upper_body":
        center_raw = raw.get(
            "shoulder_midpoint_xy_px",
            raw.get("center_xy_px", raw.get("center_xy", center_raw)),
        )
    center = _vector(center_raw, dimensions=(2,))
    if center is None:
        errors["translation"] = "invalid_or_missing_center_xy_px"

    scale_key = "outer_eye_span_px" if channel == "head" else "shoulder_width_px"
    scale_value = _finite_number(raw.get("scale_px", raw.get(scale_key)))
    if scale_value is None or scale_value <= 0:
        scale_value = None
        errors["scale"] = "scale_px_must_be_finite_and_positive"
    values: dict[str, Any] = {"center": center, "scale": scale_value}

    if channel == "head":
        basis_x = _unit_basis(raw.get("basis_x"))
        basis_y = _unit_basis(raw.get("basis_y"))
        if basis_x is None or basis_y is None:
            errors["orientation_proxy"] = "invalid_or_degenerate_basis_x_or_basis_y"
        elif len(basis_x) != len(basis_y):
            basis_x = basis_y = None
            errors["orientation_proxy"] = "basis_dimension_mismatch"
        values.update({"basis_x": basis_x, "basis_y": basis_y})
    else:
        tilt_value = _finite_number(raw.get("tilt_deg", raw.get("shoulder_tilt_deg")))
        if tilt_value is None:
            errors["tilt"] = "invalid_or_missing_tilt_deg"
        values["tilt"] = tilt_value
    return values, errors


class MotionMonitor:
    """Track motion and tracking-loss events from fresh head/body observations.

    Only completed events grow with session length. The monitor retains bounded
    frame-ID metadata and samples needed by active events and baselines.
    """

    def __init__(
        self,
        session_id: Any,
        config: MotionConfig | Mapping[str, Any] | None = None,
        *,
        capture_id: str | None = None,
    ) -> None:
        if session_id is None:
            raise ValueError("session_id must not be None")
        if config is None:
            self.config = MotionConfig()
        elif isinstance(config, MotionConfig):
            self.config = config
        elif isinstance(config, Mapping):
            self.config = MotionConfig(**dict(config))
        else:
            raise TypeError("config must be MotionConfig, a config mapping, or None")
        self.session_id = str(session_id)
        self.capture_id = str(capture_id) if capture_id is not None else None
        self._states = {name: self._new_channel_state() for name in _CHANNELS}
        self._completed_events: list[dict[str, Any]] = []
        self._recent_frame_ids: OrderedDict[Any, None] = OrderedDict()
        self._last_global_clock: _Clock | None = None
        self._max_elapsed_seen: float | None = None
        self._max_received_seen: int | None = None
        self._max_numeric_frame_id: float | None = None
        self._event_sequence = 0
        self._finished = False
        self._finish_reason: str | None = None
        self._finish_stamp: dict[str, Any] | None = None

    @staticmethod
    def _new_channel_state() -> dict[str, Any]:
        return {
            "baseline": None,
            "last_measurement": None,
            "last_measurement_result": None,
            "last_measurement_output": None,
            "last_fresh_clock": None,
            "ever_valid": False,
            "missing_start": None,
            "missing_reason": None,
            "active_motion": None,
            "active_tracking_lost": None,
            "entry_candidate": None,
            "exit_candidate": None,
            "await_reacquisition_baseline": False,
            "segment": 0,
        }

    def _remember_frame_id(self, frame_id: Any) -> None:
        try:
            self._recent_frame_ids[frame_id] = None
            self._recent_frame_ids.move_to_end(frame_id)
        except TypeError:
            return
        while len(self._recent_frame_ids) > int(self.config.recent_frame_id_cache_size):
            self._recent_frame_ids.popitem(last=False)

    @staticmethod
    def _numeric_frame_id(frame_id: Any) -> float | None:
        if isinstance(frame_id, bool):
            return None
        if isinstance(frame_id, Real):
            number = float(frame_id)
            return number if isfinite(number) else None
        if isinstance(frame_id, str) and frame_id.isdigit():
            try:
                return float(int(frame_id))
            except (ValueError, OverflowError):
                return None
        return None

    @staticmethod
    def _stamp_sample(sample: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if sample is None:
            return None
        return {
            "frame_id": deepcopy(sample.get("frame_id")),
            "received_ns": str(sample.get("received_ns")),
            "elapsed_s": float(sample.get("elapsed_s", 0.0)),
        }

    def _sample(
        self,
        clock: _Clock,
        metrics: Mapping[str, Any],
        raw: Mapping[str, Any],
        channel: str,
    ) -> dict[str, Any]:
        source = raw.get(channel, {})
        head_input = raw.get("head", {})
        visibility = source.get("visibility") if isinstance(source, Mapping) else None
        confidence = None
        confidence_reason = "not_provided_by_observer"
        if isinstance(head_input, Mapping):
            if "landmark_confidence" in head_input:
                confidence = _json_safe(head_input.get("landmark_confidence"))
                confidence_reason = _json_safe(
                    head_input.get("confidence_reason")
                    or "provided_by_observer_without_validation"
                )
            elif "confidence" in head_input:
                confidence = _json_safe(head_input.get("confidence"))
                confidence_reason = _json_safe(
                    head_input.get("confidence_reason")
                    or "provided_by_observer_without_validation"
                )
            elif head_input.get("confidence_reason") is not None:
                confidence_reason = _json_safe(head_input.get("confidence_reason"))
        return {
            **clock.stamp(),
            "channel": channel,
            "metrics": _json_safe(metrics),
            "raw": _json_safe(source),
            "confidence": {
                "head_landmark_confidence": confidence,
                "head_confidence_reason": confidence_reason,
                "upper_body_visibility": _json_safe(visibility),
                "upper_body_visibility_reason": (
                    "provided_by_pose_observer"
                    if visibility is not None
                    else "not_provided_by_observer"
                ),
            },
        }

    def _thresholds_for(self, channel: str) -> dict[str, Any]:
        config = self.config
        if channel == "head":
            continuous = {
                "translation": {
                    "value": float(config.head_translation_ref_width_per_s),
                    "unit": "outer_eye_span_reference_width/s",
                },
                "orientation_proxy": {
                    "value": float(config.head_orientation_proxy_deg_per_s),
                    "unit": "landmark_basis_direction_proxy_deg/s",
                },
                "abs_log_scale": {
                    "value": float(config.head_abs_log_scale_per_s),
                    "unit": "absolute_log_scale_ratio/s",
                },
            }
            instant_amplitudes = {
                "center_step_ref_width": float(config.instant_center_step_ref_width),
                "orientation_proxy_step_deg": float(
                    config.instant_head_orientation_step_deg
                ),
                "abs_log_scale_step": float(config.instant_abs_log_scale_step),
            }
        else:
            continuous = {
                "translation": {
                    "value": float(config.body_translation_shoulder_width_per_s),
                    "unit": "shoulder_span_reference_width/s",
                },
                "tilt": {
                    "value": float(config.body_tilt_deg_per_s),
                    "unit": "shoulder_line_angle_deg/s",
                },
                "abs_log_scale": {
                    "value": float(config.body_abs_log_scale_per_s),
                    "unit": "absolute_log_scale_ratio/s",
                },
            }
            instant_amplitudes = {
                "center_step_ref_width": float(config.instant_center_step_ref_width),
                "tilt_step_deg": float(config.instant_body_tilt_step_deg),
                "abs_log_scale_step": float(config.instant_abs_log_scale_step),
            }
        return {
            "continuous": continuous,
            "sustained": {
                "enter_score": float(config.enter_score),
                "enter_duration_s": float(config.enter_duration_s),
                "enter_min_fresh_samples": int(config.enter_min_fresh_samples),
                "exit_score": float(config.exit_score),
                "exit_duration_s": float(config.exit_duration_s),
            },
            "instantaneous": {
                "enabled": bool(config.instant_enabled),
                "min_score": float(config.instant_min_score),
                "max_dt_s": float(config.instant_max_dt_s),
                "amplitude_conditions": instant_amplitudes,
            },
            "tracking_lost_after_s": float(config.tracking_lost_after_s),
            "max_baseline_gap_s": float(config.max_baseline_gap_s),
        }

    def _event_update(
        self, kind: str, event: Mapping[str, Any], sample: Any = None
    ) -> dict[str, Any]:
        update = {
            "kind": kind,
            "event_id": event.get("event_id"),
            "channel": event.get("channel"),
            "event": deepcopy(dict(event)),
        }
        if self.capture_id is not None:
            update["capture_id"] = self.capture_id
        if sample is not None:
            update["sample"] = self._stamp_sample(sample)
        return update

    def _new_event_id(self, event_type: str, channel: str) -> str:
        self._event_sequence += 1
        if self.capture_id is not None:
            return (
                f"{self.session_id}:capture:{self.capture_id}:{channel}:"
                f"{event_type}:{self._event_sequence:06d}"
            )
        return f"{self.session_id}:{channel}:{event_type}:{self._event_sequence:06d}"

    def _capture_metadata(self) -> dict[str, str]:
        return {"capture_id": self.capture_id} if self.capture_id is not None else {}

    def _end_event(
        self,
        state: dict[str, Any],
        field_name: str,
        sample: Mapping[str, Any] | None,
        reason: str,
        censored: bool,
        updates: list[dict[str, Any]],
    ) -> None:
        event = state[field_name]
        if event is None:
            return
        end_sample = deepcopy(dict(sample)) if sample is not None else None
        event["end"] = self._stamp_sample(end_sample)
        event["end_sample"] = end_sample
        event["end_reason"] = str(reason)
        event["censored"] = bool(censored)
        event["status"] = "completed"
        start_time = event.get("start", {}).get("elapsed_s")
        end_time = event.get("end", {}).get("elapsed_s")
        event["duration_s"] = (
            max(0.0, float(end_time) - float(start_time))
            if start_time is not None and end_time is not None
            else None
        )
        completed = deepcopy(event)
        self._completed_events.append(completed)
        updates.append(self._event_update("end", completed, end_sample))
        state[field_name] = None

    def _close_channel(
        self,
        channel: str,
        reason: str,
        updates: list[dict[str, Any]],
        *,
        end_sample: Mapping[str, Any] | None = None,
        censored: bool = True,
    ) -> None:
        state = self._states[channel]
        sample = end_sample or state.get("last_measurement")
        self._end_event(state, "active_motion", sample, reason, censored, updates)
        self._end_event(
            state, "active_tracking_lost", sample, reason, censored, updates
        )
        state["baseline"] = None
        state["last_measurement"] = None
        state["last_measurement_result"] = None
        state["last_measurement_output"] = None
        state["last_fresh_clock"] = None
        state["entry_candidate"] = None
        state["exit_candidate"] = None
        state["missing_start"] = None
        state["missing_reason"] = None
        state["await_reacquisition_baseline"] = True
        state["segment"] += 1

    def _reset_all(
        self,
        reason: str,
        updates: list[dict[str, Any]],
        *,
        end_sample: Mapping[str, Any] | None = None,
    ) -> None:
        for channel in _CHANNELS:
            self._close_channel(
                channel,
                reason,
                updates,
                end_sample=end_sample,
                censored=True,
            )

    def _motion_metrics(
        self,
        channel: str,
        previous_values: Mapping[str, Any],
        current_values: Mapping[str, Any],
        dt_s: float,
        current_errors: Mapping[str, str],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pair_valid": True,
            "dt_s": float(dt_s),
            "components": {},
            "reasons": dict(current_errors),
            "score": None,
            "score_component": None,
            "instant_amplitudes": {},
            "instant_qualifiers": [],
        }
        components = result["components"]
        old_center, new_center = previous_values.get("center"), current_values.get("center")
        old_scale, new_scale = previous_values.get("scale"), current_values.get("scale")
        if old_center is not None and new_center is not None:
            center_delta_px = _distance(old_center, new_center)
            if not isfinite(center_delta_px):
                result["reasons"]["translation"] = "nonfinite_center_delta"
            elif old_scale is not None and new_scale is not None:
                mean_scale = (float(old_scale) + float(new_scale)) / 2.0
                if mean_scale > 0 and isfinite(mean_scale):
                    step = center_delta_px / mean_scale
                    threshold = (
                        self.config.head_translation_ref_width_per_s
                        if channel == "head"
                        else self.config.body_translation_shoulder_width_per_s
                    )
                    rate = step / dt_s
                    score_ratio = rate / threshold
                    if all(isfinite(value) for value in (step, rate, score_ratio)):
                        components["translation"] = {
                            "delta_px": float(center_delta_px),
                            "normalized_step": float(step),
                            "rate_per_s": float(rate),
                            "threshold_per_s": float(threshold),
                            "ratio": float(score_ratio),
                            "unit": (
                                "outer_eye_span_reference_width/s"
                                if channel == "head"
                                else "shoulder_span_reference_width/s"
                            ),
                        }
                        result["instant_amplitudes"]["center_step_ref_width"] = float(step)
                    else:
                        result["reasons"]["translation"] = "nonfinite_derived_translation"
                else:
                    result["reasons"]["translation"] = "invalid_mean_reference_scale"
            else:
                result["reasons"]["translation"] = "current_or_previous_scale_invalid"
        else:
            result["reasons"].setdefault("translation", "current_or_previous_center_invalid")

        if old_scale is not None and new_scale is not None:
            if float(old_scale) > 0 and float(new_scale) > 0:
                signed_log = log(float(new_scale)) - log(float(old_scale))
                absolute_log = abs(signed_log)
                threshold = (
                    self.config.head_abs_log_scale_per_s
                    if channel == "head"
                    else self.config.body_abs_log_scale_per_s
                )
                rate = absolute_log / dt_s
                score_ratio = rate / threshold
                if all(isfinite(value) for value in (signed_log, absolute_log, rate, score_ratio)):
                    components["abs_log_scale"] = {
                        "log_scale_ratio": float(signed_log),
                        "delta_abs": float(absolute_log),
                        "rate_per_s": float(rate),
                        "threshold_per_s": float(threshold),
                        "ratio": float(score_ratio),
                        "unit": "absolute_log_scale_ratio/s",
                    }
                    result["instant_amplitudes"]["abs_log_scale_step"] = float(absolute_log)
                else:
                    result["reasons"]["scale"] = "nonfinite_derived_log_scale"
            else:
                result["reasons"]["scale"] = "invalid_scale_ratio"
        else:
            result["reasons"].setdefault("scale", "current_or_previous_scale_invalid")

        if channel == "head":
            angles: list[float] = []
            axes = (
                ("basis_x", previous_values.get("basis_x"), current_values.get("basis_x")),
                ("basis_y", previous_values.get("basis_y"), current_values.get("basis_y")),
            )
            for label, old_axis, new_axis in axes:
                if old_axis is None or new_axis is None:
                    result["reasons"].setdefault(
                        "orientation_proxy",
                        f"invalid_{label}_in_current_or_previous_sample",
                    )
                    continue
                angle = _angle_between(old_axis, new_axis)
                if angle is None:
                    result["reasons"]["orientation_proxy"] = (
                        "basis_dimension_changed_between_samples"
                    )
                    angles = []
                    break
                angles.append(angle)
            if angles:
                delta_deg = max(angles)
                threshold = self.config.head_orientation_proxy_deg_per_s
                rate = delta_deg / dt_s
                score_ratio = rate / threshold
                if all(isfinite(value) for value in (delta_deg, rate, score_ratio)):
                    components["orientation_proxy"] = {
                        "delta_deg": float(delta_deg),
                        "rate_per_s": float(rate),
                        "threshold_per_s": float(threshold),
                        "ratio": float(score_ratio),
                        "unit": "landmark_basis_direction_proxy_deg/s",
                        "interpretation": "basis-vector angle proxy; not yaw/pitch ground truth",
                    }
                    result["instant_amplitudes"]["orientation_proxy_step_deg"] = float(
                        delta_deg
                    )
                else:
                    result["reasons"]["orientation_proxy"] = "nonfinite_derived_orientation_proxy"
        else:
            old_tilt, new_tilt = previous_values.get("tilt"), current_values.get("tilt")
            if old_tilt is not None and new_tilt is not None:
                signed_delta = _wrap_degrees(float(new_tilt) - float(old_tilt))
                delta_deg = abs(signed_delta)
                threshold = self.config.body_tilt_deg_per_s
                rate = delta_deg / dt_s
                score_ratio = rate / threshold
                if all(isfinite(value) for value in (signed_delta, delta_deg, rate, score_ratio)):
                    components["tilt"] = {
                        "signed_delta_deg": float(signed_delta),
                        "delta_deg": float(delta_deg),
                        "rate_per_s": float(rate),
                        "threshold_per_s": float(threshold),
                        "ratio": float(score_ratio),
                        "unit": "shoulder_line_angle_deg/s",
                    }
                    result["instant_amplitudes"]["tilt_step_deg"] = float(delta_deg)
                else:
                    result["reasons"]["tilt"] = "nonfinite_derived_tilt"
            else:
                result["reasons"]["tilt"] = "current_or_previous_tilt_invalid"

        ratios = [
            (name, float(component["ratio"]))
            for name, component in components.items()
            if isfinite(float(component["ratio"]))
        ]
        if ratios:
            winner, score = max(ratios, key=lambda item: item[1])
            result["score"] = score
            result["score_component"] = winner

        amplitudes = result["instant_amplitudes"]
        qualifiers: list[str] = []
        if _meets(
            amplitudes.get("center_step_ref_width", -1.0),
            self.config.instant_center_step_ref_width,
        ):
            qualifiers.append("center_step_ref_width")
        if channel == "head":
            if _meets(
                amplitudes.get("orientation_proxy_step_deg", -1.0),
                self.config.instant_head_orientation_step_deg,
            ):
                qualifiers.append("orientation_proxy_step_deg")
        elif _meets(
            amplitudes.get("tilt_step_deg", -1.0),
            self.config.instant_body_tilt_step_deg,
        ):
            qualifiers.append("tilt_step_deg")
        if _meets(
            amplitudes.get("abs_log_scale_step", -1.0),
            self.config.instant_abs_log_scale_step,
        ):
            qualifiers.append("abs_log_scale_step")
        result["instant_qualifiers"] = qualifiers
        return result

    @staticmethod
    def _peak_metric(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
        name = metrics.get("score_component")
        component = metrics.get("components", {}).get(name)
        if not isinstance(component, Mapping):
            return None
        return {
            "name": name,
            "value": component.get("rate_per_s"),
            "threshold": component.get("threshold_per_s"),
            "unit": component.get("unit"),
        }

    def _trigger_record(
        self,
        mode: str,
        before: Mapping[str, Any] | None,
        after: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        if mode == "instantaneous":
            return {
                "trigger_mode": "instantaneous",
                "candidate_label": "瞬时大幅变化/待复核",
                "before": deepcopy(dict(before)) if before is not None else None,
                "after": deepcopy(dict(after)),
                "dt_s": float(metrics.get("dt_s", 0.0)),
                "deltas": deepcopy(metrics.get("components", {})),
                "score": metrics.get("score"),
                "score_component": metrics.get("score_component"),
                "amplitude_values": deepcopy(metrics.get("instant_amplitudes", {})),
                "qualifying_amplitudes": list(metrics.get("instant_qualifiers", [])),
                "thresholds": self._thresholds_for(str(after["channel"])),
                "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
                "config_version": self.config.version,
                "confidence": deepcopy(after.get("confidence")),
            }
        return {
            "trigger_mode": "sustained",
            "candidate_label": "持续变化",
            "crossing_sample": deepcopy(dict(before)) if before is not None else None,
            "confirmation_sample": deepcopy(dict(after)),
            "score": after.get("metrics", {}).get("score"),
            "score_component": after.get("metrics", {}).get("score_component"),
            "thresholds": self._thresholds_for(str(after["channel"])),
            "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
            "config_version": self.config.version,
            "confidence": deepcopy(after.get("confidence")),
        }

    def _open_or_add_trigger(
        self,
        channel: str,
        mode: str,
        trigger_record: dict[str, Any],
        start_sample: dict[str, Any],
        peak_sample: dict[str, Any],
        peak_metrics: Mapping[str, Any],
        confirmation_sample: dict[str, Any],
        updates: list[dict[str, Any]],
    ) -> None:
        state = self._states[channel]
        event = state["active_motion"]
        score = peak_metrics.get("score")
        if score is None:
            score = 0.0
        if event is None:
            instant = mode == "instantaneous"
            event = {
                "event_id": self._new_event_id("motion", channel),
                "session_id": self.session_id,
                **self._capture_metadata(),
                "type": "motion",
                "channel": channel,
                "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
                "config_version": self.config.version,
                "config": self.config.to_dict(),
                "thresholds": self._thresholds_for(channel),
                "trigger_mode": mode,
                "trigger_modes": [mode],
                "trigger_samples": [deepcopy(trigger_record)],
                "trigger_sample": deepcopy(trigger_record),
                "confidence": deepcopy(trigger_record.get("confidence")),
                "candidate_label": (
                    "瞬时大幅变化/待复核" if instant else "持续变化"
                ),
                "needs_review": instant,
                "start": self._stamp_sample(start_sample),
                "start_sample": deepcopy(start_sample),
                "confirmed_at": self._stamp_sample(confirmation_sample),
                "peak_score": float(score),
                "peak_metric": self._peak_metric(peak_metrics),
                "peak_sample": deepcopy(peak_sample),
                "end": None,
                "end_sample": None,
                "end_reason": None,
                "duration_s": None,
                "censored": None,
                "status": "active",
                "trigger_count": {mode: 1},
            }
            state["active_motion"] = event
            updates.append(self._event_update("start", event, start_sample))
            return

        counts = event.setdefault("trigger_count", {})
        counts[mode] = int(counts.get(mode, 0)) + 1
        if mode not in event["trigger_modes"]:
            event["trigger_modes"].append(mode)
            event["trigger_samples"].append(deepcopy(trigger_record))
        else:
            recorded = next(
                (
                    item
                    for item in event["trigger_samples"]
                    if item.get("trigger_mode") == mode
                ),
                None,
            )
            if recorded is not None and float(trigger_record.get("score") or 0.0) > float(
                recorded.get("score") or 0.0
            ):
                event["trigger_samples"][
                    event["trigger_samples"].index(recorded)
                ] = deepcopy(trigger_record)
        if mode == "instantaneous":
            event["needs_review"] = True
            event["candidate_label"] = "瞬时大幅变化/待复核"
            event["trigger_sample"] = deepcopy(trigger_record)
            if event.get("confidence") is None:
                event["confidence"] = deepcopy(trigger_record.get("confidence"))
        updates.append(self._event_update("trigger", event, confirmation_sample))

    def _process_valid_channel(
        self,
        channel: str,
        clock: _Clock,
        source: Mapping[str, Any],
        safe_raw: Mapping[str, Any],
        updates: list[dict[str, Any]],
        *,
        fresh: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self._states[channel]
        available = bool(source.get("available", False))
        valid = bool(source.get("valid", False))
        reason = str(source["reason"]) if source.get("reason") is not None else None

        if not fresh:
            last = state.get("last_measurement")
            old_output = state.get("last_measurement_output")
            reused_metrics = (
                deepcopy(old_output.get("metrics"))
                if isinstance(old_output, Mapping)
                else None
            )
            if isinstance(reused_metrics, dict):
                reused_metrics["measurement_fresh"] = False
                reused_metrics["last_measurement"] = self._stamp_sample(last)
                reused_metrics["reused_measurement"] = True
            last_frame_state = (
                deepcopy(old_output.get("frame_state"))
                if isinstance(old_output, Mapping)
                else None
            )
            prior_available = (
                bool(last_frame_state.get("available"))
                if isinstance(last_frame_state, Mapping)
                else available
            )
            prior_valid = (
                bool(last_frame_state.get("valid"))
                if isinstance(last_frame_state, Mapping)
                else valid
            )
            return (
                {
                    "available": prior_available,
                    "valid": prior_valid,
                    "status": "reused",
                    "reason": "measurement_not_fresh",
                    "measurement_fresh": False,
                    "last_measurement": self._stamp_sample(last),
                    "motion_event_id": (
                        state["active_motion"].get("event_id")
                        if state["active_motion"] is not None
                        else None
                    ),
                    "tracking_event_id": (
                        state["active_tracking_lost"].get("event_id")
                        if state["active_tracking_lost"] is not None
                        else None
                    ),
                },
                reused_metrics or {
                    "measurement_fresh": False,
                    "last_measurement": self._stamp_sample(last),
                    "score": None,
                    "reason": "no_fresh_measurement",
                },
            )

        if not available:
            self._close_channel(
                channel,
                "channel_unavailable",
                updates,
                end_sample=state.get("last_measurement"),
                censored=True,
            )
            unavailable_state = {
                "available": False,
                "valid": False,
                "status": "unavailable",
                "reason": reason or "channel_unavailable",
                "measurement_fresh": True,
                "last_measurement": None,
                "motion_event_id": None,
                "tracking_event_id": None,
            }
            unavailable_metrics = {
                "measurement_fresh": True,
                "pair_valid": False,
                "score": None,
                "reason": reason or "channel_unavailable",
                "components": {},
                "reasons": {"channel": reason or "channel_unavailable"},
            }
            state["last_measurement_output"] = {
                "metrics": deepcopy(unavailable_metrics),
                "frame_state": deepcopy(unavailable_state),
            }
            return unavailable_state, unavailable_metrics

        previous_fresh_clock = state.get("last_fresh_clock")
        fresh_gap_s = (
            float(clock.elapsed_s - previous_fresh_clock.elapsed_s)
            if previous_fresh_clock is not None
            else None
        )
        observation_gap = (
            fresh_gap_s is not None
            and fresh_gap_s > self.config.max_baseline_gap_s
        )

        if not valid:
            missing_reason = reason or "no_detection"
            is_tracking_missing = missing_reason.strip().lower() in _MISSING_REASONS
            if observation_gap:
                if state["active_tracking_lost"] is not None:
                    self._end_event(
                        state,
                        "active_tracking_lost",
                        state.get("last_measurement"),
                        "tracking_observation_gap",
                        True,
                        updates,
                    )
                state["missing_start"] = None
                state["missing_reason"] = None
                state["entry_candidate"] = None
                state["exit_candidate"] = None
            sample = {
                **clock.stamp(),
                "channel": channel,
                "metrics": {
                    "measurement_fresh": True,
                    "pair_valid": False,
                    "score": None,
                    "reason": missing_reason,
                    "components": {},
                },
                "raw": _json_safe(source),
                "confidence": self._sample(clock, {}, safe_raw, channel).get("confidence"),
            }
            state["baseline"] = None
            state["last_measurement"] = None
            state["last_measurement_result"] = None
            state["last_measurement_output"] = None
            state["entry_candidate"] = None
            state["exit_candidate"] = None
            state["await_reacquisition_baseline"] = True
            state["segment"] += 1
            state["last_fresh_clock"] = clock
            if is_tracking_missing and state["ever_valid"]:
                if state["missing_start"] is None:
                    state["missing_start"] = sample
                    state["missing_reason"] = missing_reason
                missing_start = state["missing_start"]
                missing_duration = max(
                    0.0, float(clock.elapsed_s) - float(missing_start["elapsed_s"])
                )
                if (
                    state["active_tracking_lost"] is None
                    and _meets(missing_duration, self.config.tracking_lost_after_s)
                ):
                    self._end_event(
                        state, "active_motion", missing_start, "tracking_lost", True, updates
                    )
                    event = {
                        "event_id": self._new_event_id("tracking_lost", channel),
                        "session_id": self.session_id,
                        **self._capture_metadata(),
                        "type": "tracking_lost",
                        "channel": channel,
                        "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
                        "config_version": self.config.version,
                        "config": self.config.to_dict(),
                        "thresholds": {
                            "confirm_after_s": float(self.config.tracking_lost_after_s)
                        },
                        "trigger_mode": None,
                        "trigger_modes": [],
                        "trigger_sample": deepcopy(missing_start),
                        "trigger_samples": [],
                        "candidate_label": "tracking_lost",
                        "needs_review": False,
                        "start": self._stamp_sample(missing_start),
                        "start_sample": deepcopy(missing_start),
                        "confirmed_at": clock.stamp(),
                        "peak_score": None,
                        "peak_metric": None,
                        "peak_sample": None,
                        "end": None,
                        "end_sample": None,
                        "end_reason": None,
                        "duration_s": None,
                        "censored": None,
                        "status": "active",
                        "reason": state["missing_reason"],
                    }
                    state["active_tracking_lost"] = event
                    updates.append(self._event_update("start", event, missing_start))
            status = (
                "tracking_lost"
                if state["active_tracking_lost"] is not None
                else "tracking_lost_pending"
                if state["missing_start"] is not None
                else "no_detection"
                if is_tracking_missing
                else "invalid_observation"
            )
            state["last_measurement"] = sample
            missing_state = {
                "available": True,
                "valid": False,
                "status": status,
                "reason": missing_reason,
                "measurement_fresh": True,
                "last_measurement": self._stamp_sample(sample),
                "motion_event_id": (
                    state["active_motion"].get("event_id")
                    if state["active_motion"] is not None
                    else None
                ),
                "tracking_event_id": (
                    state["active_tracking_lost"].get("event_id")
                    if state["active_tracking_lost"] is not None
                    else None
                ),
            }
            state["last_measurement_output"] = {
                "metrics": deepcopy(sample["metrics"]),
                "frame_state": deepcopy(missing_state),
            }
            return missing_state, sample["metrics"]

        if observation_gap:
            state["entry_candidate"] = None
            state["exit_candidate"] = None
            if state["active_tracking_lost"] is not None:
                self._end_event(
                    state,
                    "active_tracking_lost",
                    state.get("last_measurement"),
                    "tracking_observation_gap",
                    True,
                    updates,
                )
                state["missing_start"] = None
                state["missing_reason"] = None
                state["await_reacquisition_baseline"] = True
        was_reacquisition = (
            state["missing_start"] is not None
            or state["active_tracking_lost"] is not None
            or state["await_reacquisition_baseline"]
        )
        if state["active_tracking_lost"] is not None:
            reacquired_sample = {
                **clock.stamp(),
                "channel": channel,
                "metrics": {"measurement_fresh": True, "status": "reacquired"},
                "raw": _json_safe(source),
            }
            self._end_event(
                state,
                "active_tracking_lost",
                reacquired_sample,
                "tracking_reacquired",
                False,
                updates,
            )
        state["missing_start"] = None
        state["missing_reason"] = None
        state["ever_valid"] = True
        values, value_errors = _canonical_values(channel, source)
        previous = state.get("baseline")
        dt_s = (
            float(clock.elapsed_s - previous["clock"].elapsed_s)
            if previous is not None
            else None
        )
        if previous is not None and (
            dt_s is None or dt_s <= 0 or dt_s > self.config.max_baseline_gap_s
        ):
            self._close_channel(
                channel,
                "channel_baseline_gap",
                updates,
                end_sample=state.get("last_measurement"),
                censored=True,
            )
            state = self._states[channel]
            previous = None
            was_reacquisition = True
        if was_reacquisition:
            previous = None
            state["entry_candidate"] = None
            state["exit_candidate"] = None
        if previous is None:
            metrics = {
                "measurement_fresh": True,
                "pair_valid": False,
                "score": None,
                "score_component": None,
                "components": {},
                "reasons": value_errors or {"baseline": "first_fresh_sample_in_segment"},
                "baseline": "established",
                "instant_qualifiers": [],
                "instant_amplitudes": {},
            }
        else:
            metrics = self._motion_metrics(
                channel, previous["values"], values, float(dt_s), value_errors
            )
            metrics["measurement_fresh"] = True
            metrics["baseline"] = "compared"
        current_sample = self._sample(clock, metrics, safe_raw, channel)
        current_sample["values"] = _json_safe(values)
        state["baseline"] = {
            "clock": clock,
            "values": values,
            "sample": current_sample,
        }
        state["last_measurement"] = current_sample
        state["last_measurement_result"] = metrics
        state["last_fresh_clock"] = clock
        state["await_reacquisition_baseline"] = False
        score = metrics.get("score")
        pair_is_usable = bool(metrics.get("pair_valid")) and score is not None
        instant_fired = False

        if (
            pair_is_usable
            and self.config.instant_enabled
            and float(metrics.get("dt_s", 0.0)) > 0
            and float(metrics.get("dt_s", 0.0)) <= self.config.instant_max_dt_s
            and _meets(float(score), self.config.instant_min_score)
            and bool(metrics.get("instant_qualifiers"))
        ):
            before_sample = previous.get("sample") if previous is not None else None
            trigger = self._trigger_record(
                "instantaneous", before_sample, current_sample, metrics
            )
            self._open_or_add_trigger(
                channel,
                "instantaneous",
                trigger,
                current_sample,
                current_sample,
                metrics,
                current_sample,
                updates,
            )
            instant_fired = True

        if pair_is_usable and _meets(float(score), self.config.enter_score):
            candidate = state.get("entry_candidate")
            if candidate is None:
                candidate = {
                    "start_sample": current_sample,
                    "count": 1,
                    "best_sample": current_sample,
                    "best_score": float(score),
                    "best_metrics": metrics,
                }
                state["entry_candidate"] = candidate
            else:
                candidate["count"] += 1
                if float(score) > float(candidate["best_score"]):
                    candidate["best_score"] = float(score)
                    candidate["best_sample"] = current_sample
                    candidate["best_metrics"] = metrics
            elapsed_high_s = float(clock.elapsed_s) - float(
                candidate["start_sample"]["elapsed_s"]
            )
            if (
                int(candidate["count"]) >= int(self.config.enter_min_fresh_samples)
                and _meets(elapsed_high_s, self.config.enter_duration_s)
            ):
                crossing = candidate["start_sample"]
                trigger = self._trigger_record(
                    "sustained", crossing, current_sample, candidate["best_metrics"]
                )
                self._open_or_add_trigger(
                    channel,
                    "sustained",
                    trigger,
                    crossing,
                    candidate["best_sample"],
                    candidate["best_metrics"],
                    current_sample,
                    updates,
                )
                state["entry_candidate"] = None
        else:
            state["entry_candidate"] = None

        active = state.get("active_motion")
        if active is not None and pair_is_usable:
            if float(score) > float(active["peak_score"]):
                active["peak_score"] = float(score)
                active["peak_metric"] = self._peak_metric(metrics)
                active["peak_sample"] = deepcopy(current_sample)
                updates.append(self._event_update("peak", active, current_sample))
            if float(score) <= self.config.exit_score:
                exit_start = state.get("exit_candidate")
                if exit_start is None:
                    state["exit_candidate"] = current_sample
                    exit_start = current_sample
                if (
                    float(clock.elapsed_s) - float(exit_start["elapsed_s"])
                    >= self.config.exit_duration_s
                    or isclose(
                        float(clock.elapsed_s) - float(exit_start["elapsed_s"]),
                        self.config.exit_duration_s,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    self._end_event(
                        state,
                        "active_motion",
                        current_sample,
                        "below_exit_hysteresis",
                        False,
                        updates,
                    )
                    state["exit_candidate"] = None
            else:
                state["exit_candidate"] = None
        elif active is not None:
            state["exit_candidate"] = None

        frame_state = {
            "available": True,
            "valid": True,
            "status": "reacquired_baseline" if was_reacquisition else "valid",
            "reason": reason,
            "measurement_fresh": True,
            "last_measurement": self._stamp_sample(current_sample),
            "motion_event_id": (
                state["active_motion"].get("event_id")
                if state["active_motion"] is not None
                else None
            ),
            "tracking_event_id": (
                state["active_tracking_lost"].get("event_id")
                if state["active_tracking_lost"] is not None
                else None
            ),
            "instant_triggered": instant_fired,
        }
        state["last_measurement_output"] = {
            "metrics": deepcopy(metrics),
            "frame_state": deepcopy(frame_state),
        }
        return frame_state, metrics

    @staticmethod
    def _raw_inputs(
        head: Any, body: Any, body_fresh: bool
    ) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
        head_source = dict(head) if isinstance(head, Mapping) else {}
        body_source = dict(body) if isinstance(body, Mapping) else {}
        raw = {
            "head": _json_safe(head_source),
            "upper_body": _json_safe(body_source),
            "body_fresh": bool(body_fresh),
        }
        return raw, {"head": head_source, "upper_body": body_source}

    def _current_event_ids(self) -> dict[str, dict[str, str | None]]:
        result: dict[str, dict[str, str | None]] = {}
        for channel, state in self._states.items():
            motion = state.get("active_motion")
            lost = state.get("active_tracking_lost")
            result[channel] = {
                "motion": motion.get("event_id") if motion is not None else None,
                "tracking_lost": lost.get("event_id") if lost is not None else None,
            }
        return result

    def _frame_result(
        self,
        clock: _Clock | None,
        frame_state: Mapping[str, Any],
        metrics: Mapping[str, Any],
        raw: Mapping[str, Any],
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "frame_id": _json_safe(clock.frame_id) if clock is not None else None,
            "received_ns": str(clock.received_ns) if clock is not None else None,
            "elapsed_s": float(clock.elapsed_s) if clock is not None else None,
            "frame_state": _json_safe(frame_state),
            "metrics": _json_safe(metrics),
            "raw": _json_safe(raw),
            "current_event_ids": self._current_event_ids(),
            "event_updates": deepcopy(updates),
        }

    def observe(
        self,
        frame_id: Any,
        received_ns: Any,
        elapsed_s: Any,
        head: Mapping[str, Any] | None,
        body: Mapping[str, Any] | None,
        body_fresh: bool = True,
    ) -> dict[str, Any]:
        """Consume a capture result and return its JSON-compatible motion state."""

        safe_raw, source_inputs = self._raw_inputs(head, body, body_fresh)
        updates: list[dict[str, Any]] = []
        if self._finished:
            state = {
                "head": {"status": "finished", "reason": self._finish_reason},
                "upper_body": {"status": "finished", "reason": self._finish_reason},
                "baseline_reset": False,
                "reset_reason": None,
            }
            return self._frame_result(
                None, state, {"head": None, "upper_body": None}, safe_raw, updates
            )

        clock, clock_error = _clock_from_inputs(frame_id, received_ns, elapsed_s)
        if clock is None:
            self._reset_all(
                clock_error or "invalid_clock",
                updates,
                end_sample=(
                    self._last_global_clock.stamp()
                    if self._last_global_clock is not None
                    else None
                ),
            )
            state = {
                "head": {"status": "invalid_observation", "reason": clock_error},
                "upper_body": {"status": "invalid_observation", "reason": clock_error},
                "baseline_reset": True,
                "reset_reason": clock_error,
            }
            return self._frame_result(
                None, state, {"head": None, "upper_body": None}, safe_raw, updates
            )

        frame_id_value = clock.frame_id
        numeric_id = self._numeric_frame_id(frame_id_value)
        duplicate = frame_id_value in self._recent_frame_ids
        nonincreasing_frame = (
            numeric_id is not None
            and self._max_numeric_frame_id is not None
            and numeric_id <= self._max_numeric_frame_id
        )
        time_error = None
        if self._max_elapsed_seen is not None and clock.elapsed_s <= self._max_elapsed_seen:
            time_error = "nonmonotonic_elapsed_s"
        if self._max_received_seen is not None and clock.received_ns <= self._max_received_seen:
            time_error = "nonmonotonic_received_ns"
        if duplicate or nonincreasing_frame or time_error is not None:
            reset_reason = (
                "duplicate_frame_id"
                if duplicate
                else "nonmonotonic_frame_id"
                if nonincreasing_frame
                else time_error
            )
            self._reset_all(
                str(reset_reason),
                updates,
                end_sample=(
                    self._last_global_clock.stamp()
                    if self._last_global_clock is not None
                    else None
                ),
            )
            state = {
                "head": {"status": "baseline_reset", "reason": reset_reason},
                "upper_body": {"status": "baseline_reset", "reason": reset_reason},
                "baseline_reset": True,
                "reset_reason": reset_reason,
            }
            return self._frame_result(clock, state, {"head": None, "upper_body": None}, safe_raw, updates)

        global_reset_reason = None
        if self._last_global_clock is not None:
            elapsed_gap = clock.elapsed_s - self._last_global_clock.elapsed_s
            received_gap = (
                clock.received_ns - self._last_global_clock.received_ns
            ) / _NANOSECONDS_PER_SECOND
            if (
                elapsed_gap > self.config.max_baseline_gap_s
                or received_gap > self.config.max_baseline_gap_s
            ):
                global_reset_reason = "observation_gap"
                self._reset_all(
                    global_reset_reason,
                    updates,
                    end_sample=self._last_global_clock.stamp(),
                )

        self._remember_frame_id(frame_id_value)
        self._last_global_clock = clock
        self._max_elapsed_seen = (
            clock.elapsed_s
            if self._max_elapsed_seen is None
            else max(self._max_elapsed_seen, clock.elapsed_s)
        )
        self._max_received_seen = (
            clock.received_ns
            if self._max_received_seen is None
            else max(self._max_received_seen, clock.received_ns)
        )
        if numeric_id is not None:
            self._max_numeric_frame_id = (
                numeric_id
                if self._max_numeric_frame_id is None
                else max(self._max_numeric_frame_id, numeric_id)
            )

        frame_channels: dict[str, Any] = {}
        metric_channels: dict[str, Any] = {}
        for channel in _CHANNELS:
            fresh = bool(body_fresh) if channel == "upper_body" else True
            frame_channels[channel], metric_channels[channel] = self._process_valid_channel(
                channel,
                clock,
                source_inputs[channel],
                safe_raw,
                updates,
                fresh=fresh,
            )
        frame_state = {
            **frame_channels,
            "baseline_reset": global_reset_reason is not None,
            "reset_reason": global_reset_reason,
        }
        return self._frame_result(
            clock, frame_state, metric_channels, safe_raw, updates
        )

    def finish(
        self,
        last_frame_id: Any = None,
        received_ns: Any = None,
        elapsed_s: Any = None,
        reason: str = "capture_stopped",
        **time_aliases: Any,
    ) -> list[dict[str, Any]]:
        """Close open events and mark endings without a low-motion sample censored.

        The aliases last_received_ns and last_elapsed_s are accepted for callers
        that name the final clock fields explicitly.
        """

        if self._finished:
            return []
        if received_ns is None:
            received_ns = time_aliases.pop("last_received_ns", None)
        if elapsed_s is None:
            elapsed_s = time_aliases.pop("last_elapsed_s", None)
        if time_aliases:
            raise TypeError(
                "unexpected finish arguments: " + ", ".join(sorted(time_aliases))
            )
        final_sample: dict[str, Any] | None = None
        if last_frame_id is not None and received_ns is not None and elapsed_s is not None:
            final_clock, error = _clock_from_inputs(last_frame_id, received_ns, elapsed_s)
            if final_clock is not None:
                final_sample = final_clock.stamp()
            else:
                reason = f"{reason};invalid_final_clock:{error}"
        if final_sample is None and self._last_global_clock is not None:
            final_sample = self._last_global_clock.stamp()

        updates: list[dict[str, Any]] = []
        for channel in _CHANNELS:
            state = self._states[channel]
            ending = final_sample or state.get("last_measurement")
            self._end_event(
                state, "active_motion", ending, str(reason), True, updates
            )
            self._end_event(
                state, "active_tracking_lost", ending, str(reason), True, updates
            )
            state["entry_candidate"] = None
            state["exit_candidate"] = None
        self._finished = True
        self._finish_reason = str(reason)
        self._finish_stamp = deepcopy(final_sample)
        return updates

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return completed events as detached JSON-compatible dictionaries."""

        return deepcopy(self._completed_events)

    def summary(self) -> dict[str, Any]:
        """Return a serializable session manifest without frame history."""

        event_counts = {"motion": 0, "tracking_lost": 0}
        channel_counts = {
            channel: {"motion": 0, "tracking_lost": 0} for channel in _CHANNELS
        }
        for event in self._completed_events:
            event_type = str(event.get("type"))
            if event_type in event_counts:
                event_counts[event_type] += 1
                channel = event.get("channel")
                if channel in channel_counts:
                    channel_counts[channel][event_type] += 1
        return {
            "session_id": self.session_id,
            **self._capture_metadata(),
            "schema_version": MOVEMENT_SCHEMA_VERSION,
            "algorithm_version": MOVEMENT_ALGORITHM_VERSION,
            "config": self.config.to_dict(),
            "completed_event_count": len(self._completed_events),
            "event_counts_by_type": event_counts,
            "event_counts_by_channel": channel_counts,
            "active_event_ids": self._current_event_ids(),
            "finished": self._finished,
            "finish_reason": self._finish_reason,
            "finish_at": deepcopy(self._finish_stamp),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export the session summary and completed event list."""

        return {"summary": self.summary(), "events": self.events}


__all__ = [
    "MOVEMENT_ALGORITHM_VERSION",
    "MOVEMENT_SCHEMA_VERSION",
    "MotionConfig",
    "MotionMonitor",
]
