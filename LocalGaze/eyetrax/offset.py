"""Optional target-instructed translation correction; pure math and contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .features import FEATURE_FORMULA_V1, FEATURE_FORMULA_V2, LEGACY_UNVERSIONED_FORMULA

OFFSET_SCHEMA = "LocalGaze.offset-correction/1"
FIXATION_ASSUMPTION = "participant_instructed_fixation_not_independently_verified"
REFERENCE_STATUS = "canvas_dimensions_only_monitor_and_dpi_unverified"


@dataclass(frozen=True)
class OffsetConfig:
    settle_ns: int = 300_000_000
    min_frames: int = 15
    min_span_ns: int = 800_000_000
    max_duration_ns: int = 4_000_000_000
    max_receipt_gap_ns: int = 250_000_000
    max_p90_radius_fraction: float = .01
    max_offset_fraction: float = .05

    def __post_init__(self):
        for name in ("settle_ns", "min_frames", "min_span_ns", "max_duration_ns", "max_receipt_gap_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_frames < 2 or self.settle_ns + self.min_span_ns > self.max_duration_ns:
            raise ValueError("offset time limits are inconsistent")
        for name in ("max_p90_radius_fraction", "max_offset_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite fraction in [0,1]")


@dataclass(frozen=True)
class OffsetBinding:
    model_sha256: str
    feature_formula_version: str
    calibration_canvas_size_px: tuple[int, int] | None


def _check_binding(binding, *, allow_unknown=False):
    if not isinstance(binding, OffsetBinding):
        raise ValueError("invalid offset binding")
    sha = binding.model_sha256
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("model_sha256 must be 64 lowercase hex digits")
    if binding.feature_formula_version not in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2, LEGACY_UNVERSIONED_FORMULA}:
        raise ValueError("unsupported binding feature formula")
    size = binding.calibration_canvas_size_px
    if size is None and allow_unknown:
        return
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size):
        raise ValueError("calibration canvas must have two positive integer dimensions")


def _xy(value):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("coordinate must be two finite numbers") from exc
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("coordinate must be two finite numbers")
    return result


@dataclass(frozen=True)
class OffsetObservation:
    received_monotonic_ns: int
    frame_id: str
    raw_xy_px: Any
    accepted: bool
    eye_state: str


@dataclass(frozen=True)
class OffsetEstimate:
    status: str
    reason: str | None
    offset_xy_px: tuple[float, float] | None
    raw_median_xy_px: tuple[float, float] | None
    sample_count: int
    span_ms: float
    p90_radius_px: float | None
    limits: dict[str, Any]
    binding: OffsetBinding
    target_xy_px: tuple[float, float] | None


def estimate_offset(
    observations: Sequence[OffsetObservation], target_xy_px, binding: OffsetBinding,
    config: OffsetConfig | None = None, *, target_display_monotonic_ns: int,
) -> OffsetEstimate:
    """Use the final contiguous eligible run, never concatenate across bad rows.

    Target fixation is instructed, not independently observed. Receipt timing
    is not exposure timing. Limits are configurable engineering defaults.
    """
    cfg = config or OffsetConfig()
    limits = asdict(cfg)
    target = None
    def failure(reason, rows=(), median=None, radius=None):
        span = (rows[-1][0] - rows[0][0]) / 1e6 if len(rows) > 1 else 0.
        return OffsetEstimate("rejected", reason, None, tuple(median) if median is not None else None, len(rows), span, radius, limits, binding, tuple(target) if target is not None else None)
    try:
        _check_binding(binding, allow_unknown=True)
        if binding.calibration_canvas_size_px is None:
            return failure("model_canvas_unavailable")
        target = _xy(target_xy_px)
        w, h = binding.calibration_canvas_size_px
        if not (0 <= target[0] < w and 0 <= target[1] < h):
            return failure("target_outside_canvas")
        if isinstance(target_display_monotonic_ns, bool) or not isinstance(target_display_monotonic_ns, (int, np.integer)) or target_display_monotonic_ns < 0:
            return failure("invalid_target_display_time")
    except (TypeError, ValueError):
        return failure("invalid_binding_or_target")
    start = int(target_display_monotonic_ns) + cfg.settle_ns
    end = int(target_display_monotonic_ns) + cfg.max_duration_ns
    rows = []
    last_ns = None
    seen_ids = set()
    last_rejection = "insufficient_samples"
    for obs in observations:
        if not isinstance(obs, OffsetObservation):
            rows.clear(); last_rejection = "invalid_observation"; continue
        t = obs.received_monotonic_ns
        if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 0:
            rows.clear(); last_rejection = "invalid_receipt_time"; continue
        if t > end:
            break
        gap = t - last_ns if last_ns is not None else None
        if last_ns is not None and t <= last_ns:
            rows.clear(); last_rejection = "nonincreasing_receipt_time"; continue
        last_ns = int(t)
        if not isinstance(obs.frame_id, str) or not obs.frame_id or obs.frame_id in seen_ids:
            rows.clear(); last_rejection = "invalid_or_duplicate_frame_id"; continue
        seen_ids.add(obs.frame_id)
        if gap is not None and gap > cfg.max_receipt_gap_ns:
            rows.clear(); last_rejection = "receipt_gap"
            # Current good receipt may start a new run, but not join the old one.
        if t < start:
            continue
        if obs.accepted is not True or obs.eye_state != "open":
            rows.clear(); last_rejection = "not_valid_open"; continue
        try:
            raw = _xy(obs.raw_xy_px)
        except ValueError:
            rows.clear(); last_rejection = "invalid_raw_prediction"; continue
        rows.append((int(t), raw.copy()))
    if len(rows) < cfg.min_frames:
        return failure(last_rejection, rows)
    span = rows[-1][0] - rows[0][0]
    if span < cfg.min_span_ns:
        return failure("insufficient_continuous_span", rows)
    values = np.asarray([raw for _, raw in rows])
    median = np.median(values, axis=0)
    radius = float(np.percentile(np.linalg.norm(values - median, axis=1), 90))
    diagonal = float(np.hypot(w, h))
    delta = target - median
    if radius > cfg.max_p90_radius_fraction * diagonal:
        return failure("unstable_predictions", rows, median, radius)
    if np.linalg.norm(delta) > cfg.max_offset_fraction * diagonal:
        return failure("offset_too_large_recalibrate", rows, median, radius)
    return OffsetEstimate("eligible", None, tuple(float(v) for v in delta), tuple(float(v) for v in median), len(rows), span / 1e6, radius, limits, binding, tuple(float(v) for v in target))


@dataclass(frozen=True)
class OffsetCorrectionRecord:
    schema: str
    correction_id: str
    binding: OffsetBinding
    target_xy_px: tuple[float, float]
    offset_xy_px: tuple[float, float]
    raw_median_xy_px: tuple[float, float]
    sample_count: int
    span_ms: float
    p90_radius_px: float
    config: OffsetConfig
    fixation_assumption: str = FIXATION_ASSUMPTION
    coordinate_reference_status: str = REFERENCE_STATUS

    def to_dict(self):
        return asdict(self)


def validate_correction_record(value: Mapping[str, Any]) -> OffsetCorrectionRecord:
    """Strictly validate portable data without opening or creating any file."""
    if not isinstance(value, Mapping) or set(value) != set(OffsetCorrectionRecord.__dataclass_fields__):
        raise ValueError("invalid correction record fields")
    try:
        if value["schema"] != OFFSET_SCHEMA or value["fixation_assumption"] != FIXATION_ASSUMPTION or value["coordinate_reference_status"] != REFERENCE_STATUS:
            raise ValueError("unsupported correction semantics")
        correction_id = value["correction_id"]
        if not isinstance(correction_id, str) or not correction_id.strip() or any(ord(c) < 32 for c in correction_id):
            raise ValueError("invalid correction_id")
        b = value["binding"]
        if not isinstance(b, Mapping) or set(b) != set(OffsetBinding.__dataclass_fields__):
            raise ValueError("invalid binding fields")
        binding = OffsetBinding(b["model_sha256"], b["feature_formula_version"], tuple(b["calibration_canvas_size_px"]) if b["calibration_canvas_size_px"] is not None else None)
        _check_binding(binding)
        config = OffsetConfig(**value["config"])
        target, delta, median = (_xy(value[name]) for name in ("target_xy_px", "offset_xy_px", "raw_median_xy_px"))
        count, span, radius = value["sample_count"], value["span_ms"], value["p90_radius_px"]
        w, h = binding.calibration_canvas_size_px
        diagonal = float(np.hypot(w, h))
        if isinstance(count, bool) or not isinstance(count, int) or count < config.min_frames:
            raise ValueError("insufficient correction sample count")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) for v in (span, radius)) or span < config.min_span_ns / 1e6 or span > (config.max_duration_ns - config.settle_ns) / 1e6 or not 0 <= radius <= config.max_p90_radius_fraction * diagonal:
            raise ValueError("correction quality limits not met")
        if not (0 <= target[0] < w and 0 <= target[1] < h) or np.linalg.norm(delta) > config.max_offset_fraction * diagonal or not np.allclose(delta, target - median, rtol=1e-12, atol=1e-9):
            raise ValueError("correction translation or target inconsistent")
        return OffsetCorrectionRecord(OFFSET_SCHEMA, correction_id, binding, tuple(target), tuple(delta), tuple(median), count, float(span), float(radius), config)
    except (TypeError, KeyError, OverflowError) as exc:
        raise ValueError("malformed correction record") from exc


def correction_record_from_estimate(estimate: OffsetEstimate, correction_id: str) -> OffsetCorrectionRecord:
    if not isinstance(estimate, OffsetEstimate) or estimate.status != "eligible" or estimate.offset_xy_px is None:
        raise ValueError("a rejected estimate cannot create an eligible correction")
    record = OffsetCorrectionRecord(OFFSET_SCHEMA, correction_id, estimate.binding, estimate.target_xy_px, estimate.offset_xy_px, estimate.raw_median_xy_px, estimate.sample_count, estimate.span_ms, estimate.p90_radius_px, OffsetConfig(**estimate.limits))
    return validate_correction_record(record.to_dict())


@dataclass(frozen=True)
class CorrectionApplication:
    corrected_xy_px: tuple[float, float] | None
    applied: bool
    status: str
    correction_id: str | None


def apply_offset(raw_xy_px, record: OffsetCorrectionRecord, active_binding: OffsetBinding) -> CorrectionApplication:
    if not isinstance(record, OffsetCorrectionRecord):
        raise ValueError("expected a validated correction record")
    record = validate_correction_record(record.to_dict())
    try:
        _check_binding(active_binding)
    except ValueError:
        return CorrectionApplication(None, False, "binding_unavailable", record.correction_id)
    if active_binding != record.binding:
        return CorrectionApplication(None, False, "binding_mismatch", record.correction_id)
    try:
        raw = _xy(raw_xy_px)
    except ValueError:
        return CorrectionApplication(None, False, "raw_unavailable", record.correction_id)
    corrected = raw + np.asarray(record.offset_xy_px)
    if not np.isfinite(corrected).all():
        return CorrectionApplication(None, False, "corrected_nonfinite", record.correction_id)
    return CorrectionApplication(tuple(float(v) for v in corrected), True, "applied", record.correction_id)
