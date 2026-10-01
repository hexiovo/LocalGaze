"""Calibration-only geometry stability decisions; callers supply every receipt."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationStabilityConfig:
    window_ns: int = 600_000_000
    min_window_frames: int = 5
    min_window_span_ns: int = 300_000_000
    max_receipt_gap_ns: int = 250_000_000
    max_eye_uv_deviation: float = .04
    max_basis_deviation: float = .04
    max_face_center_deviation: float = .02
    max_relative_eye_span_deviation: float = .05

    def __post_init__(self):
        for name in ("window_ns", "min_window_frames", "min_window_span_ns", "max_receipt_gap_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_window_frames < 2 or self.min_window_span_ns > self.window_ns:
            raise ValueError("window coverage configuration is inconsistent")
        for name in ("max_eye_uv_deviation", "max_basis_deviation", "max_face_center_deviation", "max_relative_eye_span_deviation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class StabilityObservation:
    received_monotonic_ns: int
    frame_id: str
    valid_open: bool
    eye_uv: Any = None
    head_basis_x: Any = None
    head_basis_y: Any = None
    face_center_xy: Any = None
    outer_eye_span_px: float | None = None


@dataclass(frozen=True)
class StabilityDecision:
    ready: bool
    reset_required: bool
    reason: str | None
    stable_since_ns: int | None
    qualified_frame_ids: tuple[str, ...] = ()
    statistics: dict[str, Any] = field(default_factory=dict)


class CalibrationStabilityGate:
    """Window readiness is not an 800-ms target commit or fixation truth.

    The collector must discard provisional rows on reset_required and enforce
    its own target settle and continuous final coverage. Never use this motion
    gate to delete ordinary gaze samples during experimental capture.
    """
    def __init__(self, config: CalibrationStabilityConfig | None = None):
        self.config = config or CalibrationStabilityConfig()
        self.reset(None, None)

    def reset(self, target_id, visit_id):
        self.target_id, self.visit_id = target_id, visit_id
        self._window = deque()
        self._last_ns = None
        self._seen_ids = set()
        self._stable_since = None

    def _reject(self, reason, statistics=None):
        self._window.clear()
        self._stable_since = None
        return StabilityDecision(False, True, reason, None, (), statistics or {})

    def update(self, observation: StabilityObservation) -> StabilityDecision:
        if not isinstance(observation, StabilityObservation):
            return self._reject("invalid_observation")
        t = observation.received_monotonic_ns
        if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 0:
            return self._reject("invalid_receipt_time")
        if self._last_ns is not None and t <= self._last_ns:
            return self._reject("nonincreasing_receipt_time")
        gap = t - self._last_ns if self._last_ns is not None else None
        self._last_ns = int(t)
        if not isinstance(observation.frame_id, str) or not observation.frame_id:
            return self._reject("invalid_frame_id")
        if observation.frame_id in self._seen_ids:
            return self._reject("duplicate_frame_id")
        self._seen_ids.add(observation.frame_id)
        if gap is not None and gap > self.config.max_receipt_gap_ns:
            return self._reject("receipt_gap")
        if observation.valid_open is not True:
            return self._reject("not_valid_open")
        try:
            uv = np.asarray(observation.eye_uv, dtype=float)
            hx = np.asarray(observation.head_basis_x, dtype=float)
            hy = np.asarray(observation.head_basis_y, dtype=float)
            face = np.asarray(observation.face_center_xy, dtype=float)
            span = float(observation.outer_eye_span_px)
            if uv.shape != (2, 2) or hx.shape != (3,) or hy.shape != (3,) or face.shape != (2,) or not all(np.isfinite(a).all() for a in (uv, hx, hy, face)) or not np.isfinite(span) or span <= 0:
                return self._reject("invalid_geometry")
            if abs(np.linalg.norm(hx) - 1) > 1e-4 or abs(np.linalg.norm(hy) - 1) > 1e-4 or abs(hx @ hy) > 1e-4:
                return self._reject("invalid_head_basis")
        except (TypeError, ValueError, OverflowError):
            return self._reject("invalid_geometry")
        self._window.append((int(t), observation.frame_id, uv, hx, hy, face, span))
        while self._window and t - self._window[0][0] > self.config.window_ns:
            self._window.popleft()
        count = len(self._window)
        elapsed = t - self._window[0][0]
        stats = {"window_frame_count": count, "window_span_ns": int(elapsed)}
        if count < self.config.min_window_frames or elapsed < self.config.min_window_span_ns:
            if self._stable_since is not None:
                return self._reject("window_coverage_lost", stats)
            return StabilityDecision(False, False, "window_warmup", None, (), stats)
        arrays = [np.asarray([row[index] for row in self._window]) for index in range(2, 7)]
        eye, x, y, centers, spans = arrays
        stats["max_eye_uv_deviation"] = float(np.max(np.linalg.norm(eye - np.median(eye, axis=0), axis=2)))
        stats["max_basis_deviation"] = float(max(np.max(np.linalg.norm(a - np.median(a, axis=0), axis=1)) for a in (x, y)))
        stats["max_face_center_deviation"] = float(np.max(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)))
        stats["max_relative_eye_span_deviation"] = float(np.max(np.abs(spans / np.median(spans) - 1)))
        for name in ("max_eye_uv_deviation", "max_basis_deviation", "max_face_center_deviation", "max_relative_eye_span_deviation"):
            # Decimal thresholds can land one rounding step above their
            # float representation. This is comparison tolerance only.
            limit = getattr(self.config, name)
            if stats[name] > limit and not np.isclose(stats[name], limit, rtol=1e-12, atol=1e-12):
                return self._reject(name + "_exceeded", stats)
        if self._stable_since is None:
            self._stable_since = self._window[0][0]
        return StabilityDecision(True, False, None, self._stable_since, tuple(row[1] for row in self._window), stats)
