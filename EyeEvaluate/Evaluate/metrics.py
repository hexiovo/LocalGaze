"""Pure metrics for complete normalized samples; engineering candidates only.

Display smoothing and interpolation never replace measured gaze. Iris ring
ratios and EAR-marked closures are not physical pupil or eyelid ground truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from roi_contract import ROI, classify_point, validate_coordinate_reference_compatibility, validate_roi_reference_compatibility, validate_rois
from .analysis_options import ANALYSIS_METRIC_GROUP_IDS, normalize_selected_metrics

METHOD_VERSION = "LocalGaze-analysis-1.2.0"
METRIC_COLUMNS = ["metric_id", "value", "unit", "status", "reason", "method", "input_source", "n", "denominator"]
EVENT_COLUMNS = ["event_id", "event_type", "status", "reason", "method", "input_source", "segment_id", "trial_id", "stimulus_id", "start_row", "end_row", "start_time_s", "end_time_s", "duration_s", "sample_count", "left_censored", "right_censored", "roi_id", "roi_name", "center_x_px", "center_y_px", "dispersion_px", "x_range_px", "y_range_px", "amplitude_px", "path_length_px", "peak_speed_px_s", "mean_speed_px_s"]
ROI_COLUMNS = ["roi_id", "roi_name", "shape", "boundary_mode", "priority", "status", "reason", "sample_count", "all_hit_sample_count", "observed_duration_s", "all_hit_duration_s", "observed_duration_share", "observed_entry_count", "visit_count", "complete_visit_count", "left_censored_count", "right_censored_count", "complete_dwell_mean_s", "complete_dwell_median_s", "first_observed_time_s", "first_observed_entry_time_s", "method", "input_source"]
HIT_COLUMNS = ["analysis_row_index", "time_s", "gaze_valid", "status", "reason", "primary_roi_id", "primary_roi_name", "all_hit_ids", "all_hit_names", "pair_valid", "represented_duration_s"]


@dataclass(frozen=True)
class AnalysisConfig:
    max_time_gap_s: float | None = None
    gap_median_multiplier: float = 3.0
    fixation_min_samples: int = 3
    fixation_min_duration_s: float = 0.120
    fixation_dispersion_diagonal_ratio: float = 0.03
    fast_speed_diagonals_s: float = 1.0
    ivt_speed_diagonals_s: float = 1.0
    ivt_min_duration_s: float = 0.020
    ivt_max_duration_s: float = 0.200
    ivt_min_samples: int = 3
    blink_min_duration_s: float = 0.050
    blink_max_duration_s: float = 0.500
    entropy_grid: tuple[int, int] = (32, 18)

    def __post_init__(self):
        for key in ("gap_median_multiplier", "fixation_min_duration_s", "fixation_dispersion_diagonal_ratio", "fast_speed_diagonals_s", "ivt_speed_diagonals_s", "ivt_min_duration_s", "ivt_max_duration_s", "blink_min_duration_s", "blink_max_duration_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.max_time_gap_s is not None and (isinstance(self.max_time_gap_s, bool) or not math.isfinite(float(self.max_time_gap_s)) or self.max_time_gap_s <= 0):
            raise ValueError("max_time_gap_s must be positive and finite")
        if isinstance(self.fixation_min_samples, bool) or not isinstance(self.fixation_min_samples, int) or self.fixation_min_samples < 3:
            raise ValueError("fixation_min_samples must be an integer >= 3")
        if isinstance(self.ivt_min_samples, bool) or not isinstance(self.ivt_min_samples, int) or self.ivt_min_samples < 3:
            raise ValueError("ivt_min_samples must be an integer >= 3")
        if self.blink_max_duration_s < self.blink_min_duration_s:
            raise ValueError("blink duration limits are reversed")
        if self.ivt_max_duration_s < self.ivt_min_duration_s:
            raise ValueError("I-VT duration limits are reversed")
        if len(self.entropy_grid) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.entropy_grid):
            raise ValueError("entropy_grid must contain two positive integers")


@dataclass
class AnalysisResult:
    metrics: pd.DataFrame
    roi: pd.DataFrame
    events: pd.DataFrame
    roi_hits: pd.DataFrame
    samples: pd.DataFrame
    methods: dict[str, Any]
    summary: dict[str, Any]

    @property
    def roi_metrics(self):
        return self.roi


def _numbers(frame, name):
    if name not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _flags(frame, name, *, missing=False):
    if name not in frame:
        return np.full(len(frame), missing, dtype=bool)
    def flag(value):
        if value is None or pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1"}
        return value is True or value == 1
    return np.array([flag(v) for v in frame[name]], dtype=bool)


def _keys(frame, name):
    if name not in frame:
        return [None] * len(frame)
    return [None if v is None or pd.isna(v) else str(v) for v in frame[name]]


def _runs(valid, pair):
    """Original-row runs: pair[i] joins rows i-1 and i."""
    runs, start = [], None
    for i in range(len(valid)):
        if valid[i]:
            if start is None:
                start = i
            elif not pair[i]:
                runs.append((start, i - 1)); start = i
        elif start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(valid) - 1))
    return runs


def _at_least(value, threshold):
    """Include exact boundaries despite a few arithmetic rounding ulps."""
    return value >= threshold or math.isclose(value, threshold, rel_tol=0., abs_tol=8 * max(math.ulp(float(value)), math.ulp(float(threshold))))


def _screen(metadata):
    reference = metadata.get("coordinate_reference") or {}
    if not isinstance(reference, Mapping):
        return None
    if reference.get("space") != "reference_pixel_centers":
        return None
    try:
        width, height = float(reference["width"]), float(reference["height"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) and v > 1 for v in (width, height)):
        return None
    return width, height


def _ivt_screen_reason(metadata, screen):
    reference = metadata.get("coordinate_reference")
    if not isinstance(reference, Mapping):
        return "missing_trusted_screen_reference"
    if reference.get("space") != "reference_pixel_centers":
        return "incompatible_screen_reference" if reference.get("space") not in {None, "unknown"} else "missing_trusted_screen_reference"
    if metadata.get("coordinate_reference_status") != "verified":
        return "unverified_screen_reference"
    if screen is None:
        return "invalid_screen_dimensions"
    return None


def analyze_samples(
    samples: pd.DataFrame,
    *,
    rois: Sequence[ROI] = (),
    metadata: Mapping[str, Any] | None = None,
    config: AnalysisConfig | Mapping[str, Any] | None = None,
    selected_metrics: Sequence[str] | None = None,
) -> AnalysisResult:
    """Analyze normalized rows without modifying inputs or their metadata."""
    required = {"analysis_x_raw", "analysis_y_raw", "analysis_time_s", "analysis_gaze_valid"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"normalized analysis columns missing: {', '.join(missing)}")
    cfg = config if isinstance(config, AnalysisConfig) else AnalysisConfig(**dict(config or {}))
    selected_groups = normalize_selected_metrics(selected_metrics)
    selected_group_set = set(selected_groups)
    roi_selected = "roi" in selected_group_set
    records = list(rois) if roi_selected else []
    meta = dict(metadata if metadata is not None else samples.attrs.get("analysis_metadata", {}))
    if roi_selected:
        validate_rois(records)
        validate_roi_reference_compatibility(records)
    n = len(samples)
    frame = samples.copy(deep=True)
    x, y, t = (_numbers(frame, key) for key in ("analysis_x_raw", "analysis_y_raw", "analysis_time_s"))
    valid = _flags(frame, "analysis_gaze_valid") & np.isfinite(x) & np.isfinite(y)
    valid &= ~_flags(frame, "analysis_interpolated") & ~_flags(frame, "analysis_bad_segment")
    brk = _flags(frame, "analysis_break_before", missing=True)
    if n:
        brk[0] = True
    segment, trial, stimulus, statuses = (_keys(frame, key) for key in ("analysis_segment_id", "analysis_trial_id", "analysis_stimulus_id", "analysis_sample_status"))
    # Even a contradictory normalized validity flag cannot authorize known
    # rejected capture states as measured gaze.
    valid &= np.array([status not in {"no_face", "blink", "warmup", "invalid", "duplicate_frame_id", "processing_error"} for status in statuses], bool)
    same = np.zeros(n, dtype=bool)
    for i in range(1, n):
        same[i] = not brk[i] and (segment[i], trial[i], stimulus[i]) == (segment[i-1], trial[i-1], stimulus[i-1])
    time_valid = np.isfinite(t)
    clock_quality = _keys(frame, "analysis_clock_quality")
    for i, quality in enumerate(clock_quality):
        if quality is not None and quality.lower() in {"unknown", "missing", "invalid", "unreliable", "synthetic", "nominal_fps"}:
            time_valid[i] = False
    dt = np.full(n, np.nan)
    if n > 1:
        dt[1:] = np.diff(t)
    previous_time = np.r_[False, time_valid[:-1]] if n else np.zeros(0, bool)
    previous_valid = np.r_[False, valid[:-1]] if n else np.zeros(0, bool)
    clock_pair = same & time_valid & previous_time
    positive = clock_pair & (dt > 0)
    base_pair = positive & valid & previous_valid
    reference_dt = dt[base_pair] if base_pair.any() else dt[positive]
    median_dt = float(np.median(reference_dt)) if reference_dt.size else None
    gap = cfg.max_time_gap_s
    if gap is None and median_dt is not None:
        gap = cfg.gap_median_multiplier * median_dt
    gap_mask = positive & (dt > gap) if gap is not None else np.zeros(n, bool)
    pair = base_pair & ~gap_mask
    runs = _runs(valid, pair)
    step, speed = np.full(n, np.nan), np.full(n, np.nan)
    for i in np.flatnonzero(pair):
        step[i] = math.hypot(float(x[i] - x[i-1]), float(y[i] - y[i-1]))
        speed[i] = step[i] / dt[i]
    finite_motion = pair & np.isfinite(step) & np.isfinite(speed)
    weight = np.zeros(n)
    for i in np.flatnonzero(pair):
        weight[i-1] = dt[i]
    valid_duration = float(dt[pair].sum())
    duration_available = bool((valid & time_valid).any())
    effective_duration = valid_duration if duration_available else None
    screen = _screen(meta)
    diagonal = math.hypot(*screen) if screen is not None else None
    metrics, events, roi_rows, hit_rows = [], [], [], []
    ivt_summary = None
    source = str(meta.get("source_path", meta.get("source", "normalized_samples")))
    input_source = f"{source}:analysis_x_raw,analysis_y_raw,analysis_gaze_valid"

    def metric(key, value, unit, *, status=None, reason=None, method="sampled-descriptive", count=None, denominator=None, field_source=None):
        if value is not None and isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            value, reason = None, reason or "nonfinite_derived_value"
        metrics.append(dict(metric_id=key, value=value, unit=unit, status=status or ("computed" if value is not None else "not_computable"), reason=reason, method=method, input_source=field_source or input_source, n=count, denominator=denominator))

    def distribution(prefix, values, unit, *, field_source=None):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        count = len(values)
        metric(prefix + "_count", count, "count", count=count, field_source=field_source)
        for suffix, function in (("mean", np.mean), ("median", np.median), ("p95", lambda a: np.percentile(a, 95)), ("iqr", lambda a: np.percentile(a, 75) - np.percentile(a, 25)), ("min", np.min), ("max", np.max)):
            metric(prefix + "_" + suffix, float(function(values)) if count else None, unit, reason=None if count else "no_eligible_observations", count=count, field_source=field_source)
        metric(prefix + "_sd", float(np.std(values, ddof=1)) if count >= 2 else None, unit, reason=None if count >= 2 else "fewer_than_two_observations", count=count, field_source=field_source)

    def event(kind, a, b, *, count, duration=None, status="estimated_candidate", reason=None, left=False, right=False, **extra):
        row = dict.fromkeys(EVENT_COLUMNS)
        row.update(event_id=f"event-{len(events)+1:06d}", event_type=kind, status=status, reason=reason, method=METHOD_VERSION + ":" + kind, input_source=input_source, segment_id=segment[a], trial_id=trial[a], stimulus_id=stimulus[a], start_row=int(a), end_row=int(b), start_time_s=float(t[a]) if time_valid[a] else None, end_time_s=float(t[b]) if time_valid[b] else None, duration_s=duration if duration is not None else (float(t[b] - t[a]) if time_valid[a] and time_valid[b] else None), sample_count=int(count), left_censored=bool(left), right_censored=bool(right))
        row.update(extra)
        for key, value in row.items():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                row[key] = None
                row["status"] = "invalid_input"
                row["reason"] = "nonfinite_derived_value"
        events.append(row)
        return row

    metric("input_sample_count", n, "count", count=n)
    metric("valid_measured_sample_count", int(valid.sum()), "count", count=n)
    metric("invalid_sample_count", int((~valid).sum()), "count", count=n)
    metric("valid_measured_sample_ratio", float(valid.mean()) if n else None, "ratio", reason=None if n else "empty_input", count=int(valid.sum()), denominator=n)
    for name, column in (("interpolated_sample_count", "analysis_interpolated"), ("bad_segment_sample_count", "analysis_bad_segment"), ("break_before_count", "analysis_break_before")):
        metric(name, int(_flags(frame, column).sum()), "count")
    for status_name in sorted({value for value in statuses if value is not None}):
        metric("status_" + status_name + "_count", statuses.count(status_name), "count", field_source=source + ":analysis_sample_status")
    reason_column = "analysis_invalid_reason" if "analysis_invalid_reason" in frame else "sample_status_reason"
    invalid_reasons = _keys(frame, reason_column)
    for invalid_reason in sorted({invalid_reasons[i] for i in range(n) if not valid[i] and invalid_reasons[i] is not None}):
        metric("invalid_reason:" + invalid_reason, sum(not valid[i] and invalid_reasons[i] == invalid_reason for i in range(n)), "count", field_source=source + ":" + reason_column)
    if "head_descriptor_outside_training_range" in frame:
        metric("head_descriptor_outside_training_range_count", int(_flags(frame, "head_descriptor_outside_training_range").sum()), "count", method="calibration coverage flag; not accuracy measurement")
    full_clock = n > 0 and time_valid.all() and (n == 1 or np.all(np.diff(t) > 0))
    wall_span = float(t[-1] - t[0]) if full_clock else None
    metric("session_wall_span_s", wall_span, "s", reason=None if full_clock else "missing_or_nonmonotonic_session_clock", field_source=source + ":analysis_time_s")
    metric("valid_pair_observed_duration_s", effective_duration, "s", reason=None if duration_available else "no_valid_timed_observations", method="sum valid original-row adjacent dt; singleton contributes zero", count=int(pair.sum()))
    metric("valid_timed_pair_count", int(pair.sum()), "count")
    metric("clock_nonpositive_pair_count", int((clock_pair & (dt <= 0)).sum()), "count")
    metric("long_gap_count", int(gap_mask.sum()), "count")
    distribution("adjacent_dt_ms", dt[pair] * 1000., "ms", field_source=source + ":analysis_time_s")
    metric("observed_sampling_rate_hz", 1. / float(np.median(dt[pair])) if pair.any() else None, "Hz", reason=None if pair.any() else "no_valid_timed_adjacent_pairs", method="reciprocal median valid adjacent dt; reception clock, not exposure rate", count=int(pair.sum()))
    if "frames_dropped_before_frame" in frame:
        dropped = _numbers(frame, "frames_dropped_before_frame")
        usable = dropped[np.isfinite(dropped) & (dropped >= 0)]
        monotonic = usable.size > 0 and np.all(np.diff(usable) >= 0)
        metric("last_cumulative_unprocessed_drop_count", int(usable[-1]) if monotonic else None, "count", reason=None if monotonic else "missing_or_nonmonotonic_cumulative_drop_counter", method="last cumulative value; never sum rows")
    if "duplicate_image" in frame:
        metric("duplicate_image_marked_count", int(_flags(frame, "duplicate_image").sum()), "count", method="pixel equality QC only; not repeated-exposure detection")
    if "trajectory" in selected_group_set:
        distribution("x_px", x[valid], "px")
        distribution("y_px", y[valid], "px")
        metric("path_length_px", float(step[finite_motion].sum()) if finite_motion.any() and np.all(np.isfinite(step[pair])) else None, "px", reason=None if finite_motion.any() and np.all(np.isfinite(step[pair])) else "no_valid_pairs_or_nonfinite_displacement", method="sum adjacent measured displacement within valid runs", count=int(pair.sum()))
        distribution("speed_px_s", speed[finite_motion], "px/s")
        distribution("segment_displacement_px", [math.hypot(x[b] - x[a], y[b] - y[a]) for a, b in runs if b > a], "px")

    roi_reason = "not_selected"
    if roi_selected:
        roi_reason = None
        if not records:
            roi_reason = "no_roi_input"
        elif meta.get("coordinate_reference_status") != "verified" or screen is None:
            roi_reason = "coordinate_reference_unverified"
        elif any(roi.coordinate_reference.get("space") == "unknown" for roi in records):
            roi_reason = "roi_coordinate_reference_unknown"
        else:
            try:
                validate_coordinate_reference_compatibility([meta["coordinate_reference"], *(roi.coordinate_reference for roi in records)])
            except ValueError:
                roi_reason = "roi_gaze_coordinate_reference_conflict"
    primary, hits = [None] * n, [[] for _ in range(n)]
    if roi_selected and roi_reason is None:
        for i in np.flatnonzero(valid):
            classified = classify_point(records, float(x[i]), float(y[i]))
            primary[i] = classified.primary.id if classified.primary is not None else None
            hits[i] = list(classified.hits)
    by_id = {roi.id: roi for roi in records}
    for i in (range(n) if roi_selected else ()):
        row_index = frame["analysis_row_index"].iloc[i] if "analysis_row_index" in frame else i
        hit_rows.append(dict(analysis_row_index=row_index, time_s=float(t[i]) if time_valid[i] else None, gaze_valid=bool(valid[i]), status="not_computable" if roi_reason else "invalid_coordinates" if not valid[i] else "matched" if primary[i] else "outside", reason=roi_reason, primary_roi_id=primary[i], primary_roi_name=by_id[primary[i]].name if primary[i] else None, all_hit_ids=[roi.id for roi in hits[i]], all_hit_names=[roi.name for roi in hits[i]], pair_valid=bool(pair[i]), represented_duration_s=float(weight[i]) if valid[i] and roi_reason is None else None))
    transitions = {}
    if roi_selected and roi_reason is None:
        for i in np.flatnonzero(pair):
            before, after = primary[i-1], primary[i]
            if before != after:
                key = (before, after)
                transitions[key] = transitions.get(key, 0) + 1
        for run_start, run_end in runs:
            if not time_valid[run_start] or not time_valid[run_end]:
                continue
            i = run_start
            while i <= run_end:
                identity = primary[i]
                end = i
                while end < run_end and primary[end+1] == identity:
                    end += 1
                if identity is not None:
                    exit_row = end + 1 if end < run_end else end
                    left, right = i == run_start, end == run_end
                    event("roi_visit", i, exit_row, count=end-i+1, status="boundary_truncated" if left or right else "computed", left=left, right=right, roi_id=identity, roi_name=by_id[identity].name, reason="sampled_left_hold_visit; not exact crossing time")
                i = end + 1
    visits = [row for row in events if row["event_type"] == "roi_visit"]
    for roi in sorted(records, key=lambda item: item.priority):
        row = dict.fromkeys(ROI_COLUMNS)
        row.update(roi_id=roi.id, roi_name=roi.name, shape=roi.shape, boundary_mode=roi.boundary_mode, priority=roi.priority, status="not_computable" if roi_reason else "computed", reason=roi_reason, method="primary priority; sampled left-hold dt; censored visits excluded from complete dwell", input_source=input_source)
        if roi_reason is None:
            chosen = np.array([identity == roi.id for identity in primary], dtype=bool)
            all_chosen = np.array([any(hit.id == roi.id for hit in row_hits) for row_hits in hits], dtype=bool)
            own_visits = [v for v in visits if v["roi_id"] == roi.id]
            complete = [v["duration_s"] for v in own_visits if not v["left_censored"] and not v["right_censored"]]
            entries = [i for i in np.flatnonzero(pair) if primary[i] == roi.id and primary[i-1] != roi.id]
            observed = np.flatnonzero(chosen & time_valid)
            row.update(sample_count=int(chosen.sum()), all_hit_sample_count=int(all_chosen.sum()), observed_duration_s=float(weight[chosen].sum()) if duration_available else None, all_hit_duration_s=float(weight[all_chosen].sum()) if duration_available else None, observed_duration_share=float(weight[chosen].sum()) / valid_duration if valid_duration > 0 else None, observed_entry_count=len(entries) if duration_available else None, visit_count=len(own_visits) if duration_available else None, complete_visit_count=len(complete) if duration_available else None, left_censored_count=sum(v["left_censored"] for v in own_visits) if duration_available else None, right_censored_count=sum(v["right_censored"] for v in own_visits) if duration_available else None, complete_dwell_mean_s=float(np.mean(complete)) if complete else None, complete_dwell_median_s=float(np.median(complete)) if complete else None, first_observed_time_s=float(t[observed[0]]) if observed.size else None, first_observed_entry_time_s=float(t[entries[0]]) if entries else None)
        roi_rows.append(row)
        for field_name, unit in (("sample_count", "count"), ("all_hit_sample_count", "count"), ("observed_duration_s", "s"), ("all_hit_duration_s", "s"), ("observed_duration_share", "ratio"), ("observed_entry_count", "count"), ("complete_dwell_mean_s", "s"), ("complete_dwell_median_s", "s")):
            value = row[field_name]
            reason = roi_reason
            if value is None and reason is None:
                reason = "no_complete_uncensored_visits" if field_name.startswith("complete_dwell") and pair.any() else "no_valid_timed_adjacent_pairs"
            metric("roi:" + roi.id + ":" + field_name, value, unit, reason=reason, method=row["method"], denominator=valid_duration if field_name == "observed_duration_share" and pair.any() else None)
    if roi_selected:
        metric("roi_primary_observed_duration_s", float(sum(weight[i] for i in range(n) if primary[i] is not None)) if roi_reason is None and duration_available else None, "s", reason=roi_reason or (None if duration_available else "no_valid_timed_observations"), method="priority primary left-hold; valid pair time denominator", denominator=effective_duration)

    if "fixation" in selected_group_set:
        candidate_reason = "missing_known_screen_dimensions" if screen is None else "no_valid_timed_adjacent_pairs" if not pair.any() else None
        fixation_assigned = np.zeros(n, bool)
        if candidate_reason is None:
            threshold = cfg.fixation_dispersion_diagonal_ratio * diagonal
            for a, b in runs:
                start = a
                while start <= b:
                    end = start + cfg.fixation_min_samples - 1
                    while end <= b and not _at_least(float(t[end] - t[start]), cfg.fixation_min_duration_s):
                        end += 1
                    if end > b:
                        break
                    xmin, xmax = float(np.min(x[start:end+1])), float(np.max(x[start:end+1]))
                    ymin, ymax = float(np.min(y[start:end+1])), float(np.max(y[start:end+1]))
                    if _at_least(threshold, xmax-xmin+ymax-ymin):
                        while end < b:
                            nxmin, nxmax = min(xmin, x[end+1]), max(xmax, x[end+1])
                            nymin, nymax = min(ymin, y[end+1]), max(ymax, y[end+1])
                            if not _at_least(threshold, nxmax-nxmin+nymax-nymin):
                                break
                            xmin, xmax, ymin, ymax = nxmin, nxmax, nymin, nymax
                            end += 1
                        fixation_assigned[start:end+1] = True
                        event("fixation_candidate", start, end, count=end-start+1, left=start == a, right=end == b, center_x_px=float(np.mean(x[start:end+1])), center_y_px=float(np.mean(y[start:end+1])), dispersion_px=float(xmax-xmin+ymax-ymin), x_range_px=float(xmax-xmin), y_range_px=float(ymax-ymin))
                        start = end + 1
                    else:
                        start += 1
        selected_fixations = [row for row in events if row["event_type"] == "fixation_candidate"]
        metric("fixation_candidate_count", len(selected_fixations) if candidate_reason is None else None, "count", status="estimated_candidate" if selected_fixations else "not_observed" if candidate_reason is None else "not_computable", reason=candidate_reason, method=METHOD_VERSION + ":fixation_candidate")
        if candidate_reason is None:
            distribution("fixation_candidate_duration_s", [row["duration_s"] for row in selected_fixations], "s")
        else:
            metric("fixation_candidate_duration_s_mean", None, "s", reason=candidate_reason)
        metric("fixation_unassigned_valid_sample_count", int((valid & ~fixation_assigned).sum()) if candidate_reason is None else None, "count", reason=candidate_reason, method="not a physiological event classification")

    if "fast_movement" in selected_group_set:
        candidate_reason = "missing_known_screen_dimensions" if screen is None else "no_valid_timed_adjacent_pairs" if not pair.any() else None
        if candidate_reason is None:
            fast = np.array([bool(finite_motion[i] and _at_least(float(speed[i]), cfg.fast_speed_diagonals_s * diagonal)) for i in range(n)], bool)
            for first, last in _runs(fast, pair):
                start = first - 1
                path = float(step[first:last+1].sum())
                duration = float(t[last] - t[start])
                event("fast_movement_candidate", start, last, count=last-start+1, amplitude_px=float(math.hypot(x[last]-x[start], y[last]-y[start])), path_length_px=path, peak_speed_px_s=float(np.max(speed[first:last+1])), mean_speed_px_s=path / duration)
        selected_movements = [row for row in events if row["event_type"] == "fast_movement_candidate"]
        metric("fast_movement_candidate_count", len(selected_movements) if candidate_reason is None else None, "count", status="estimated_candidate" if selected_movements else "not_observed" if candidate_reason is None else "not_computable", reason=candidate_reason, method=METHOD_VERSION + ":fast_movement_candidate")
        if candidate_reason is None:
            distribution("fast_movement_candidate_duration_s", [row["duration_s"] for row in selected_movements], "s")
        else:
            metric("fast_movement_candidate_duration_s_mean", None, "s", reason=candidate_reason)

    if "ivt_saccade" in selected_group_set:
        ivt_screen_reason = _ivt_screen_reason(meta, screen)
        candidate_reason = ivt_screen_reason or ("no_valid_timed_adjacent_pairs" if not pair.any() else None)
        threshold_px_s = None
        if candidate_reason is None:
            threshold_px_s = cfg.ivt_speed_diagonals_s * diagonal
            if not math.isfinite(threshold_px_s):
                candidate_reason = "nonfinite_screen_speed_threshold"

        accepted_candidates = []
        filtered_clusters = []
        filtered_reason_counts = {
            "too_few_samples": 0,
            "duration_below_min": 0,
            "duration_above_max": 0,
        }
        if candidate_reason is None:
            ivt_qualifies = np.array(
                [bool(finite_motion[i] and _at_least(float(speed[i]), threshold_px_s)) for i in range(n)],
                dtype=bool,
            )
            for first, last in _runs(ivt_qualifies, pair):
                start = first - 1
                sample_count = last - start + 1
                duration = float(t[last] - t[start])
                reasons = []
                if sample_count < cfg.ivt_min_samples:
                    reasons.append("too_few_samples")
                if not _at_least(duration, cfg.ivt_min_duration_s):
                    reasons.append("duration_below_min")
                if not _at_least(cfg.ivt_max_duration_s, duration):
                    reasons.append("duration_above_max")

                path_length = float(step[first:last + 1].sum())
                event_fields = dict(
                    center_x_px=float(np.mean(x[start:last + 1])),
                    center_y_px=float(np.mean(y[start:last + 1])),
                    amplitude_px=float(math.hypot(x[last] - x[start], y[last] - y[start])),
                    path_length_px=path_length,
                    peak_speed_px_s=float(np.max(speed[first:last + 1])),
                    mean_speed_px_s=path_length / duration,
                )
                if reasons:
                    for reason_code in reasons:
                        filtered_reason_counts[reason_code] += 1
                    filtered_clusters.append(event(
                        "ivt_saccade_filtered_cluster",
                        start,
                        last,
                        count=sample_count,
                        duration=duration,
                        status="filtered_candidate",
                        reason=";".join(reasons),
                        **event_fields,
                    ))
                else:
                    accepted_candidates.append(event(
                        "ivt_saccade_candidate",
                        start,
                        last,
                        count=sample_count,
                        duration=duration,
                        status="estimated_candidate",
                        **event_fields,
                    ))

        ivt_status = (
            "not_computable"
            if candidate_reason is not None
            else "estimated_candidate"
            if accepted_candidates
            else "not_observed"
        )
        accepted_count = len(accepted_candidates) if candidate_reason is None else None
        filtered_count = len(filtered_clusters) if candidate_reason is None else None
        metric(
            "ivt_valid_pair_exposure_s",
            effective_duration,
            "s",
            reason=None if duration_available else "no_valid_timed_observations",
            method="sum actual dt for all existing strict valid adjacent pairs; no additional I-VT pair window",
            count=int(pair.sum()),
        )
        metric(
            "invalid_sample_loss_ratio",
            float((~valid).mean()) if n else None,
            "ratio",
            reason=None if n else "empty_input",
            method="invalid measured samples / input sample count",
            count=int((~valid).sum()),
            denominator=n,
        )
        metric(
            "ivt_saccade_candidate_count",
            accepted_count,
            "count",
            status=ivt_status,
            reason=candidate_reason,
            method=METHOD_VERSION + ":ivt_saccade_candidate",
            count=accepted_count,
        )
        rate = (
            float(accepted_count) * 60.0 / effective_duration
            if candidate_reason is None and effective_duration is not None and effective_duration > 0
            else None
        )
        rate_reason = candidate_reason
        if rate is None and rate_reason is None:
            rate_reason = "no_valid_timed_adjacent_pairs"
        metric(
            "ivt_saccade_candidate_rate_per_min",
            rate,
            "candidate/min",
            status=ivt_status if rate is not None else "not_computable",
            reason=rate_reason,
            method="accepted I-VT candidates / all existing strict valid-pair exposure in minutes",
            count=accepted_count,
            denominator=effective_duration,
        )
        metric(
            "ivt_saccade_filtered_cluster_count",
            filtered_count,
            "count",
            status="computed" if candidate_reason is None else "not_computable",
            reason=candidate_reason,
            method="maximal over-threshold clusters rejected by sample-count or whole-candidate duration limits",
            count=filtered_count,
        )
        for reason_code, rejected_count in filtered_reason_counts.items():
            metric(
                "ivt_saccade_filtered_" + reason_code + "_count",
                rejected_count if candidate_reason is None else None,
                "count",
                status="computed" if candidate_reason is None else "not_computable",
                reason=candidate_reason,
                method="filtered I-VT maximal-cluster reason count",
            )
        if candidate_reason is None:
            distribution(
                "ivt_saccade_candidate_duration_s",
                [row["duration_s"] for row in accepted_candidates],
                "s",
            )
        else:
            metric(
                "ivt_saccade_candidate_duration_s_count",
                None,
                "count",
                status="not_computable",
                reason=candidate_reason,
            )
            metric(
                "ivt_saccade_candidate_duration_s_mean",
                None,
                "s",
                status="not_computable",
                reason=candidate_reason,
            )

        ivt_summary = {
            "ivt_saccade_status": ivt_status,
            "ivt_saccade_reason": candidate_reason,
            "ivt_saccade_candidate_count": accepted_count,
            "ivt_saccade_filtered_cluster_count": filtered_count,
            "ivt_valid_pair_exposure_s": effective_duration,
            "valid_measured_sample_ratio": float(valid.mean()) if n else None,
            "invalid_sample_loss_ratio": float((~valid).mean()) if n else None,
        }

    if "blink" in selected_group_set:
        # Closed frames have no measured gaze: use an independent EAR-status stream.
        closure_valid = np.array([status in {"accepted", "blink"} for status in statuses], dtype=bool) & time_valid
        closure_valid &= ~_flags(frame, "analysis_bad_segment") & ~_flags(frame, "analysis_interpolated")
        closure_previous = np.r_[False, closure_valid[:-1]] if n else np.zeros(0, bool)
        closure_pair = positive & ~gap_mask & closure_valid & closure_previous
        closure_available = closure_pair.any() and any(status == "accepted" for status in statuses)
        if closure_available:
            for a, b in _runs(closure_valid, closure_pair):
                i = a
                while i <= b:
                    if statuses[i] != "blink":
                        i += 1; continue
                    end = i
                    while end < b and statuses[end+1] == "blink":
                        end += 1
                    recovery = end + 1 if end < b else end
                    left, right = i == a, end == b
                    duration = float(t[recovery] - t[i])
                    kind = "blink_candidate" if not left and not right and _at_least(duration, cfg.blink_min_duration_s) and _at_least(cfg.blink_max_duration_s, duration) else "long_closure_candidate" if not left and not right and duration > cfg.blink_max_duration_s else "unclassified_closure_episode"
                    event(kind, i, recovery, count=end-i+1, left=left, right=right, status="boundary_truncated" if left or right else "estimated_candidate", reason="EAR heuristic marked status; not independently observed eyelid truth", method=METHOD_VERSION + ":EAR-marked-closure", input_source=source + ":analysis_sample_status")
                    i = end + 1
        for kind in ("blink_candidate", "long_closure_candidate", "unclassified_closure_episode"):
            count = sum(row["event_type"] == kind for row in events)
            metric(kind + "_count", count if closure_available else None, "count", status="estimated_candidate" if count else "not_observed" if closure_available else "not_computable", reason=None if closure_available else "no_usable_EAR_status_clock_sequence", method="EAR-marked status episodes; separate from iris-ratio detector", field_source=source + ":analysis_sample_status")
    if "iris_proxy" in selected_group_set:
        formulas = _keys(frame, "iris_proxy_formula_version")
        for side in ("left", "right"):
            column = side + "_iris_diameter_over_eye_span"
            proxy = _numbers(frame, column)
            eligible = np.isfinite(proxy) & (proxy >= 0) & np.array([status == "accepted" for status in statuses], bool)
            versions = sorted({formulas[i] for i in np.flatnonzero(eligible) if formulas[i] is not None})
            if len(versions) == 1 and all(formulas[i] == versions[0] for i in np.flatnonzero(eligible)):
                distribution(side + "_iris_proxy", proxy[eligible], "ratio", field_source=source + ":" + column + ";formula=" + versions[0])
            else:
                metric(side + "_iris_proxy_mean", None, "ratio", reason="missing_or_mixed_proxy_formula_or_no_eligible_values", field_source=source + ":" + column)
        metric("iris_ratio_blink_candidate_count", None, "count", reason="no_explicit_proxy_baseline_contract", method="requires separate same-formula iris baseline; 0.70/0.85 hysteresis not applied to raw diameter ratios")
    if "calibration" in selected_group_set:
        residuals = meta.get("calibration_fit_residuals_px")
        if residuals is not None:
            residual_values = np.asarray(residuals, dtype=float)
            residual_values = residual_values[np.isfinite(residual_values) & (residual_values >= 0)]
        else:
            residual_values = np.array([], dtype=float)
        for key, function in (("mean", np.mean), ("median", np.median), ("p95", lambda a: np.percentile(a, 95)), ("rmse", lambda a: np.sqrt(np.mean(a*a)))):
            metric("calibration_fit_residual_px_" + key, float(function(residual_values)) if residual_values.size else None, "px", reason=None if residual_values.size else "calibration_training_residuals_not_supplied", method="explicit training-fit residual array; not independent accuracy", count=len(residual_values), field_source=source + ":metadata.calibration_fit_residuals_px")

    if "entropy" in selected_group_set:
        entropy_reason = "missing_known_screen_dimensions" if screen is None else "no_valid_timed_adjacent_pairs" if not pair.any() else None
        entropy = None
        if entropy_reason is None:
            width, height = screen
            inside = valid & (x >= 0) & (x <= width-1) & (y >= 0) & (y <= height-1) & (weight > 0)
            counts = np.zeros(cfg.entropy_grid)
            for i in np.flatnonzero(inside):
                ix = min(cfg.entropy_grid[0]-1, int(x[i] / width * cfg.entropy_grid[0]))
                iy = min(cfg.entropy_grid[1]-1, int(y[i] / height * cfg.entropy_grid[1]))
                counts[ix, iy] += weight[i]
            if counts.sum() > 0:
                probabilities = counts[counts > 0] / counts.sum()
                entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            else:
                entropy_reason = "no_time_weighted_observations_inside_screen"
        metric("time_weighted_grid_entropy_bits", entropy, "bits", reason=entropy_reason, method="fixed grid; valid left-hold observed time; spatial description, not attention")

    if "capabilities" in selected_group_set:
        for key, unit, reason in (("visual_angle_displacement_deg", "deg", "physical_screen_size_and_viewing_distance_not_supplied"), ("stimulus_first_fixation_latency_ms", "ms", "complete_stimulus_onset_observation_contract_not_supplied"), ("stimulus_first_roi_entry_latency_ms", "ms", "complete_stimulus_onset_observation_contract_not_supplied"), ("binocular_vergence_deg", "deg", "independent_left_right_gaze_not_supplied"), ("physical_pupil_diameter_mm", "mm", "physical_pupil_measurement_not_available"), ("independent_gaze_accuracy_px", "px", "independent_validation_data_not_supplied")):
            metric(key, None, unit, reason=reason)
    frame["analysis_effective_gaze_valid"] = valid
    frame["analysis_pair_valid"] = pair
    frame["analysis_pair_dt_s"] = np.where(pair, dt, np.nan)
    frame["analysis_speed_px_s"] = speed
    frame["analysis_represented_duration_s"] = weight if duration_available else np.full(n, np.nan)
    frame["analysis_outgoing_pair_valid"] = np.r_[pair[1:], False] if n else np.zeros(0, bool)
    frame["analysis_long_gap_before"] = gap_mask
    frame["analysis_primary_roi_id"] = primary
    methods = dict(method_version=METHOD_VERSION, config=asdict(cfg), max_time_gap_s=gap, selected_metric_groups=[group for group in selected_groups if group != "capabilities"], retained_capability_metadata=True, screen_diagonal_px=diagonal, screen_diagonal_rule="hypot(reference width, reference height)", analysis_coordinates="raw measured coordinates; display smoothing/interpolation excluded", continuity="adjacent original rows; both valid; same segment/trial/stimulus; no break; positive clock dt <= maximum gap", roi_time="left endpoint hold on valid pairs; priority primary and separate all_hits; crossing times are sampled estimates", roi_reference_status="not_computable:" + roi_reason if roi_reason else "compatible verified reference", transitions=[dict(from_roi_id=a, to_roi_id=b, count=count) for (a, b), count in transitions.items()], fixation="I-DT duration-qualified first window, extended within dispersion threshold, greedy nonoverlapping candidates", fast_movement="adjacent qualifying pair runs; engineering candidates", closure="EAR marked closed status episodes; first closed sample to observed recovery; censored/unclassified preserved", iris="per-eye same-formula descriptive ring ratios; never physical pupil or proxy-baseline ratios", evidence_boundary="software checks and engineering estimates do not establish real gaze accuracy", calibration_residual_interpretation="training fit residual, never independent gaze accuracy")
    if ivt_summary is not None:
        methods["ivt_saccade"] = {
            "algorithm": "I-VT maximal uninterrupted raw-XY adjacent-pair runs at or above screen-diagonal speed threshold",
            "pair_eligibility": "existing strict valid pair, positive actual dt, and effective maximum-gap rule; no additional I-VT pair-dt window",
            "speed_threshold_diagonals_s": cfg.ivt_speed_diagonals_s,
            "speed_threshold_px_s": threshold_px_s,
            "duration_bounds_s_inclusive": {
                "minimum": cfg.ivt_min_duration_s,
                "maximum": cfg.ivt_max_duration_s,
            },
            "minimum_sample_count": cfg.ivt_min_samples,
            "duration_source": "actual end_time_s - start_time_s",
            "candidate_status": ivt_summary["ivt_saccade_status"],
            "candidate_reason": ivt_summary["ivt_saccade_reason"],
            "filtered_clusters": "retained in the events table with filtered_candidate status and reason; excluded from accepted count/rate",
            "interpretation": "engineering candidate only; not independently validated physiological saccade ground truth",
        }
    methods["numerical_boundary_rule"] = "inclusive thresholds with up to 8 arithmetic ulps; config controls all thresholds"
    methods["weight_direction"] = "analysis_pair_valid/roi_hits.pair_valid are incoming; represented_duration_s and analysis_outgoing_pair_valid refer to the outgoing interval"
    summary = dict(sample_count=n, valid_measured_count=int(valid.sum()), invalid_sample_count=int((~valid).sum()), interpolated_sample_count=int(_flags(frame, "analysis_interpolated").sum()), bad_segment_sample_count=int(_flags(frame, "analysis_bad_segment").sum()), valid_pair_count=int(pair.sum()), effective_pair_coverage_s=effective_duration, wall_span_s=wall_span, time_source=sorted({v for v in _keys(frame, "analysis_time_source") if v is not None}), clock_quality=sorted({v for v in clock_quality if v is not None}), event_count=len(events), roi_status="not_computable" if roi_reason else "computed", roi_reason=roi_reason, coordinate_reference=meta.get("coordinate_reference"), coordinate_reference_status=meta.get("coordinate_reference_status", "unknown"), known_screen_dimensions=screen, method_version=METHOD_VERSION, session_id=meta.get("session_id"))
    if ivt_summary is not None:
        summary.update(ivt_summary)
    return AnalysisResult(pd.DataFrame(metrics, columns=METRIC_COLUMNS), pd.DataFrame(roi_rows, columns=ROI_COLUMNS), pd.DataFrame(events, columns=EVENT_COLUMNS), pd.DataFrame(hit_rows, columns=HIT_COLUMNS), frame, methods, summary)


__all__ = ["AnalysisConfig", "AnalysisResult", "METHOD_VERSION", "analyze_samples"]
