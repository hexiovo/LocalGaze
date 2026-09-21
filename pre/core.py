"""Vector-level eye-tracking preprocessing adapted from AIPT.main.Eyedata_Pre.

The public function in this module performs no file I/O.  File discovery,
name matching and format-preserving output live in pipeline.py and fileio.py.
Project candidate thresholds are recorded in the returned QC dictionary and
must not be described as universally validated thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


METHOD_VERSION = "LocalGaze-pre-1.0.0-AIPT-adapted"


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for the LocalGaze unstable-data preset."""

    time_unit: str = "seconds"
    coordinate_bounds: tuple[float, float, float, float] | None = (0.0, 2559.0, 0.0, 1439.0)
    blink_padding_s: float = 0.050
    max_time_gap_s: float | None = None
    max_interpolation_gap_s: float = 0.050
    max_interpolation_distance: float | None = None
    smooth_window_s: float = 0.060
    smooth_max_points: int = 5
    velocity_mad_multiplier: float = 6.0
    quality_window_s: float = 0.500
    quality_step_s: float = 0.100
    missing_ratio_threshold: float = 0.50
    round_trip_ratio_threshold: float = 0.30
    noise_mad_multiplier: float = 6.0
    min_bad_segment_s: float = 0.200
    bad_segment_merge_gap_s: float = 0.100
    pupil_max_interpolation_gap_s: float = 0.075
    pupil_artifact_padding_s: float = 0.050
    pupil_dilation_mad_multiplier: float = 6.0
    pupil_filter_cutoff_hz: float = 4.0

    def validate(self) -> None:
        if self.time_unit not in {"seconds", "milliseconds", "microseconds"}:
            raise ValueError("time_unit must be seconds, milliseconds, or microseconds")
        positive = {
            "blink_padding_s": self.blink_padding_s,
            "max_interpolation_gap_s": self.max_interpolation_gap_s,
            "smooth_window_s": self.smooth_window_s,
            "quality_window_s": self.quality_window_s,
            "quality_step_s": self.quality_step_s,
            "pupil_max_interpolation_gap_s": self.pupil_max_interpolation_gap_s,
            "pupil_filter_cutoff_hz": self.pupil_filter_cutoff_hz,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.smooth_max_points < 1:
            raise ValueError("smooth_max_points must be at least 1")
        if self.coordinate_bounds is not None:
            if len(self.coordinate_bounds) != 4:
                raise ValueError("coordinate_bounds must be (xmin, xmax, ymin, ymax)")
            xmin, xmax, ymin, ymax = self.coordinate_bounds
            if xmin > xmax or ymin > ymax:
                raise ValueError("coordinate_bounds minima must not exceed maxima")


@dataclass(slots=True)
class PreprocessResult:
    data: pd.DataFrame
    qc: dict[str, Any]
    windows: pd.DataFrame
    bad_segments: pd.DataFrame
    suspect_segments: pd.DataFrame


def preprocess_frame(
    frame: pd.DataFrame,
    *,
    time_column: str,
    x_column: str,
    y_column: str,
    blink_column: str | None = None,
    left_pupil_column: str | None = None,
    right_pupil_column: str | None = None,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Preprocess one gaze table while preserving its row count and order."""

    cfg = config or PreprocessConfig()
    cfg.validate()
    if frame.empty:
        raise ValueError("The gaze table is empty")

    t = _time_seconds(frame[time_column], cfg.time_unit)
    x = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
    n = len(frame)
    if not (len(t) == len(x) == len(y)):
        raise ValueError("Time, X, and Y columns must have equal lengths")
    if np.any(np.diff(t) < 0):
        raise ValueError("Time moves backwards; split trials before preprocessing")
    if np.any(np.diff(t) == 0):
        raise ValueError("Duplicate timestamps are not accepted because output rows must stay aligned")

    blink = _bool_column(frame, blink_column, n)
    pupil, pupil_names = _pupil_matrix(frame, left_pupil_column, right_pupil_column, n)
    dt = np.diff(t)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else np.nan
    sample_rate = float(1.0 / median_dt) if np.isfinite(median_dt) and median_dt > 0 else np.nan
    max_time_gap = cfg.max_time_gap_s
    if max_time_gap is None:
        max_time_gap = 3.0 * median_dt if np.isfinite(median_dt) else np.inf

    initial_break = np.zeros(n, dtype=bool)
    initial_break[0] = True
    if n > 1:
        initial_break[1:] = dt > max_time_gap
    initial_segment = np.cumsum(initial_break)
    blink_padding = _expand_mask(blink, t, cfg.blink_padding_s, initial_segment)

    nonfinite = ~np.isfinite(x) | ~np.isfinite(y)
    out_of_bounds = np.zeros(n, dtype=bool)
    if cfg.coordinate_bounds is not None:
        xmin, xmax, ymin, ymax = cfg.coordinate_bounds
        finite = np.isfinite(x) & np.isfinite(y)
        out_of_bounds[finite] = (
            (x[finite] < xmin) | (x[finite] > xmax) |
            (y[finite] < ymin) | (y[finite] > ymax)
        )
    protected = out_of_bounds | blink_padding
    break_before = _add_mask_boundaries(initial_break, protected)
    segment = np.cumsum(break_before)
    base_invalid = nonfinite | protected
    base_valid = ~base_invalid

    spike, spike_thresholds = _detect_spikes(x, y, base_valid, segment, cfg.velocity_mad_multiplier)
    velocity, velocity_threshold = _velocity_candidates(x, y, t, segment, base_valid, cfg.velocity_mad_multiplier)
    velocity_candidate = np.isfinite(velocity) & (velocity > velocity_threshold)

    windows = _quality_windows(
        x, y, t, segment, base_invalid, out_of_bounds,
        velocity_candidate, spike, cfg,
    )
    bad_mask, suspect_mask, bad_segments, suspect_segments, quality_thresholds = _quality_masks(
        windows, t, segment, cfg
    )
    protected = protected | bad_mask
    break_before = _add_mask_boundaries(break_before, bad_mask)
    segment = np.cumsum(break_before)
    valid_measured = base_valid & ~spike & ~bad_mask
    invalid = ~valid_measured

    x_artifact = x.copy()
    y_artifact = y.copy()
    x_artifact[invalid] = np.nan
    y_artifact[invalid] = np.nan

    interpolation_enabled = cfg.max_interpolation_distance is not None
    if interpolation_enabled:
        x_clean, y_clean, interpolated = _interpolate_xy(
            x_artifact, y_artifact, t, segment, protected,
            cfg.max_interpolation_gap_s, float(cfg.max_interpolation_distance),
        )
        interpolation_status = "enabled_model_calibration_distance"
    else:
        x_clean, y_clean = x_artifact.copy(), y_artifact.copy()
        interpolated = np.zeros(n, dtype=bool)
        interpolation_status = "disabled_no_model_calibration_distance"

    x_smoothed = _median_smooth(x_clean, t, segment, cfg.smooth_window_s, cfg.smooth_max_points)
    y_smoothed = _median_smooth(y_clean, t, segment, cfg.smooth_window_s, cfg.smooth_max_points)
    valid_final = np.isfinite(x_smoothed) & np.isfinite(y_smoothed)

    pupil_clean, pupil_valid, pupil_artifact, pupil_interpolated, pupil_status = _clean_pupil(
        pupil, t, segment, bad_mask, blink_padding, sample_rate, cfg
    )

    output = frame.copy()
    appended: dict[str, Any] = {
        "Pre_TimeSeconds": t,
        "Pre_XRaw": x,
        "Pre_YRaw": y,
        "Pre_XArtifactRemoved": x_artifact,
        "Pre_YArtifactRemoved": y_artifact,
        "Pre_XClean": x_clean,
        "Pre_YClean": y_clean,
        "Pre_X": x_smoothed,
        "Pre_Y": y_smoothed,
        "Pre_ValidMeasured": valid_measured,
        "Pre_ValidFinal": valid_final,
        "Pre_Invalid": invalid,
        "Pre_BlinkMask": blink,
        "Pre_BlinkPaddingMask": blink_padding,
        "Pre_OutOfBoundsMask": out_of_bounds,
        "Pre_SpikeMask": spike,
        "Pre_VelocityCandidateMask": velocity_candidate,
        "Pre_BadSegmentMask": bad_mask,
        "Pre_SuspectSegmentMask": suspect_mask,
        "Pre_InterpolatedMask": interpolated,
        "Pre_BreakBefore": break_before,
        "Pre_SegmentID": segment,
    }
    for index, source_name in enumerate(pupil_names):
        label = "Left" if index == 0 else "Right"
        if len(pupil_names) == 1:
            prefix = "Pre_Pupil"
        else:
            prefix = f"Pre_{label}Pupil"
        appended[f"{prefix}Clean"] = pupil_clean[:, index]
        appended[f"{prefix}ValidMask"] = pupil_valid[:, index]
        appended[f"{prefix}ArtifactMask"] = pupil_artifact[:, index]
        appended[f"{prefix}InterpolatedMask"] = pupil_interpolated[:, index]
    for name, values in appended.items():
        output[name] = values

    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    qc = {
        "method_version": METHOD_VERSION,
        "sample_count": int(n),
        "duration_seconds": duration,
        "estimated_sample_rate_hz": sample_rate,
        "median_dt_seconds": median_dt,
        "valid_measured_count": int(valid_measured.sum()),
        "valid_final_count": int(valid_final.sum()),
        "valid_final_proportion": float(valid_final.mean()),
        "interpolated_count": int(interpolated.sum()),
        "spike_count": int(spike.sum()),
        "bad_segment_count": int(len(bad_segments)),
        "suspect_segment_count": int(len(suspect_segments)),
        "bad_sample_count": int(bad_mask.sum()),
        "suspect_sample_count": int(suspect_mask.sum()),
        "coordinate_interpolation": interpolation_status,
        "pupil_status": pupil_status,
        "thresholds": {
            "max_time_gap_seconds": _json_number(max_time_gap),
            "max_interpolation_gap_seconds": cfg.max_interpolation_gap_s,
            "max_interpolation_distance_pixels": _json_number(cfg.max_interpolation_distance),
            "spike_minimum_deviation": _json_number(spike_thresholds[0]),
            "spike_maximum_endpoint_distance": _json_number(spike_thresholds[1]),
            "velocity_candidate_per_second": _json_number(velocity_threshold),
            **quality_thresholds,
        },
        "config": asdict(cfg),
        "parameter_note": (
            "Window lengths, MAD multipliers, missing ratios, smoothing settings, and bad-segment "
            "rules are project candidates adapted from the prior AIPT method; they are not universal "
            "literature-validated thresholds."
        ),
    }
    return PreprocessResult(output, qc, windows, bad_segments, suspect_segments)


def _time_seconds(series: pd.Series, unit: str) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Could not parse {int(parsed.isna().sum())} timestamp value(s)")
        return (parsed - parsed.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    nonmissing = series.notna()
    if nonmissing.any() and numeric[nonmissing].notna().all():
        values = numeric.to_numpy(dtype=float)
        scale = {"seconds": 1.0, "milliseconds": 1e-3, "microseconds": 1e-6}[unit]
        values = values * scale
        if not np.all(np.isfinite(values)):
            raise ValueError("Numeric time contains missing or non-finite values")
        return values - values[0]
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"Could not parse {bad} timestamp value(s)")
    # Pandas may use ns or us resolution depending on the Excel reader.
    # Timedelta.total_seconds() is resolution-independent.
    return (parsed - parsed.iloc[0]).dt.total_seconds().to_numpy(dtype=float)


def _bool_column(frame: pd.DataFrame, name: str | None, n: int) -> np.ndarray:
    if name is None:
        return np.zeros(n, dtype=bool)
    values = pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(dtype=float)
    invalid = ~np.isin(values, [0.0, 1.0])
    if invalid.any():
        raise ValueError(f"Blink column {name!r} must contain only 0, 1, or missing")
    return values == 1.0


def _pupil_matrix(
    frame: pd.DataFrame, left: str | None, right: str | None, n: int
) -> tuple[np.ndarray, list[str]]:
    names = [name for name in (left, right) if name is not None]
    if not names:
        return np.empty((n, 0), dtype=float), []
    columns = [pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float) for name in names]
    return np.column_stack(columns), names


def _robust_upper(values: np.ndarray, multiplier: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.inf
    median = float(np.median(values))
    raw_mad = float(np.median(np.abs(values - median)))
    threshold = median + multiplier * 1.4826 * raw_mad
    if threshold == 0:
        threshold = np.sqrt(np.finfo(float).eps) * max(1.0, float(np.max(np.abs(values))))
    return float(threshold)


def _expand_mask(mask: np.ndarray, t: np.ndarray, padding: float, segment: np.ndarray) -> np.ndarray:
    result = mask.copy()
    if padding <= 0 or not mask.any():
        return result
    starts, ends = _mask_runs(mask, segment)
    for start, end in zip(starts, ends):
        sid = segment[start]
        result |= (segment == sid) & (t >= t[start] - padding) & (t <= t[end] + padding)
    return result


def _add_mask_boundaries(break_before: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = break_before.copy()
    starts = mask & np.r_[True, ~mask[:-1]]
    after = np.r_[False, mask[:-1] & ~mask[1:]]
    result |= starts | after
    result[0] = True
    return result


def _detect_spikes(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray, segment: np.ndarray, multiplier: float
) -> tuple[np.ndarray, tuple[float, float]]:
    n = len(x)
    result = np.zeros(n, dtype=bool)
    if n < 3:
        return result, (np.nan, np.nan)
    candidate = (
        valid[:-2] & valid[1:-1] & valid[2:] &
        (segment[:-2] == segment[1:-1]) & (segment[1:-1] == segment[2:])
    )
    centers = np.flatnonzero(candidate) + 1
    if not centers.size:
        return result, (np.nan, np.nan)
    deviation = np.hypot(
        x[centers] - (x[centers - 1] + x[centers + 1]) / 2.0,
        y[centers] - (y[centers - 1] + y[centers + 1]) / 2.0,
    )
    endpoint = np.hypot(x[centers + 1] - x[centers - 1], y[centers + 1] - y[centers - 1])
    minimum = _robust_upper(deviation, multiplier)
    maximum = _robust_upper(endpoint, multiplier)
    result[centers[(deviation > minimum) & (endpoint <= maximum)]] = True
    return result, (minimum, maximum)


def _velocity_candidates(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, segment: np.ndarray,
    valid: np.ndarray, multiplier: float,
) -> tuple[np.ndarray, float]:
    velocity = np.full(len(x), np.nan)
    if len(x) > 1:
        delta = np.diff(t)
        usable = valid[:-1] & valid[1:] & (segment[:-1] == segment[1:]) & (delta > 0)
        pair = np.full(len(x) - 1, np.nan)
        pair[usable] = np.hypot(np.diff(x)[usable], np.diff(y)[usable]) / delta[usable]
        velocity[1:] = pair
    reference = velocity[np.isfinite(velocity)]
    threshold = _robust_upper(reference, multiplier) if reference.size >= 3 else np.inf
    return velocity, threshold


def _quality_windows(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, segment: np.ndarray,
    base_invalid: np.ndarray, out_of_bounds: np.ndarray,
    velocity_candidate: np.ndarray, spike: np.ndarray, cfg: PreprocessConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    combined_missing = base_invalid | out_of_bounds
    segment_starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    segment_ends = np.r_[segment_starts[1:] - 1, len(segment) - 1]
    for segment_start, segment_end in zip(segment_starts, segment_ends):
        sid = segment[segment_start]
        local_time = t[segment_start:segment_end + 1]
        first, last = local_time[0], local_time[-1]
        if last - first <= cfg.quality_window_s:
            starts = np.array([first])
        else:
            starts = np.arange(first, last - cfg.quality_window_s + np.finfo(float).eps, cfg.quality_step_s)
            if not starts.size:
                starts = np.array([first])
        for start in starts:
            end = min(last, start + cfg.quality_window_s)
            point_lo = segment_start + int(np.searchsorted(local_time, start, side="left"))
            point_hi = segment_start + int(np.searchsorted(local_time, end, side="right"))
            points = np.arange(point_lo, point_hi)
            candidate_lo = max(segment_start, point_lo - 1)
            candidate_hi = min(segment_end, point_hi)
            left = np.arange(candidate_lo, candidate_hi)
            left = left[(t[left] < end) & (t[left + 1] > start)]
            weight = np.maximum(0.0, np.minimum(t[left + 1], end) - np.maximum(t[left], start))
            positive = weight > 0
            left, weight = left[positive], weight[positive]
            observed = float(weight.sum())
            missing_ratio = np.nan
            velocity_ratio = np.nan
            round_trip_ratio = np.nan
            if observed > 0:
                missing_ratio = float(weight[combined_missing[left]].sum() / observed)
                velocity_ratio = float(weight[velocity_candidate[left + 1]].sum() / observed)
                round_trip_ratio = float(weight[spike[left]].sum() / observed)
            usable = ~base_invalid[left] & ~base_invalid[left + 1]
            rms = np.nan
            if usable.any():
                dx = x[left[usable] + 1] - x[left[usable]]
                dy = y[left[usable] + 1] - y[left[usable]]
                rms = float(np.sqrt(np.sum((dx * dx + dy * dy) * weight[usable]) / weight[usable].sum()))
            valid_points = points[~base_invalid[points]]
            local_mad = np.nan
            if valid_points.size:
                cx, cy = np.median(x[valid_points]), np.median(y[valid_points])
                local_mad = float(np.median(np.hypot(x[valid_points] - cx, y[valid_points] - cy)))
            records.append({
                "Start": float(start), "End": float(end), "SegmentID": int(sid),
                "ObservedDuration": observed, "SampleCount": int(points.size),
                "ValidSampleCount": int(valid_points.size), "PairCount": int(usable.sum()),
                "MissingOutRatio": missing_ratio, "VelocityCandidateRatio": velocity_ratio,
                "RoundTripSpikeRatio": round_trip_ratio, "RMSS2S": rms, "LocalMAD": local_mad,
            })
    return pd.DataFrame.from_records(records)


def _quality_masks(
    windows: pd.DataFrame, t: np.ndarray, segment: np.ndarray, cfg: PreprocessConfig
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n = len(t)
    bad = np.zeros(n, dtype=bool)
    suspect = np.zeros(n, dtype=bool)
    if windows.empty:
        return bad, suspect, _segment_frame(bad, t, segment, ""), _segment_frame(suspect, t, segment, ""), {
            "quality_rms_s2s": None, "quality_local_mad": None,
        }
    eligible = (windows["PairCount"] >= 2) & windows["RMSS2S"].notna() & windows["LocalMAD"].notna()
    rms_threshold = _robust_upper(windows.loc[eligible, "RMSS2S"].to_numpy(), cfg.noise_mad_multiplier)
    mad_threshold = _robust_upper(windows.loc[eligible, "LocalMAD"].to_numpy(), cfg.noise_mad_multiplier)
    missing_candidate = windows["MissingOutRatio"] > cfg.missing_ratio_threshold
    high_variation = (windows["RMSS2S"] > rms_threshold) | (windows["LocalMAD"] > mad_threshold)
    noise_candidate = high_variation & (windows["RoundTripSpikeRatio"] > cfg.round_trip_ratio_threshold)
    windows["MissingCandidate"] = missing_candidate
    windows["NoiseCandidate"] = noise_candidate
    windows["BadWindow"] = missing_candidate
    windows["SuspectWindow"] = noise_candidate & ~missing_candidate
    windows["ThresholdSource"] = "internal_all_windows_no_external_reference"
    for _, row in windows.loc[windows["BadWindow"]].iterrows():
        bad |= (segment == int(row["SegmentID"])) & (t >= row["Start"]) & (t <= row["End"])
    for _, row in windows.loc[windows["SuspectWindow"]].iterrows():
        suspect |= (segment == int(row["SegmentID"])) & (t >= row["Start"]) & (t <= row["End"])
    suspect = _remove_short_runs(suspect, t, segment, cfg.min_bad_segment_s)
    bad = _merge_mask_gaps(bad, t, segment, cfg.bad_segment_merge_gap_s)
    suspect = _merge_mask_gaps(suspect & ~bad, t, segment, cfg.bad_segment_merge_gap_s)
    thresholds = {
        "quality_rms_s2s": _json_number(rms_threshold),
        "quality_local_mad": _json_number(mad_threshold),
        "quality_window_seconds": cfg.quality_window_s,
        "quality_step_seconds": cfg.quality_step_s,
        "missing_ratio_candidate": cfg.missing_ratio_threshold,
        "round_trip_ratio_candidate": cfg.round_trip_ratio_threshold,
    }
    return (
        bad, suspect,
        _segment_frame(bad, t, segment, "MissingInvalid"),
        _segment_frame(suspect, t, segment, "NoiseJointRuleNoReliableReference"),
        thresholds,
    )


def _mask_runs(mask: np.ndarray, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous_same = np.r_[False, segment[1:] == segment[:-1]]
    next_same = np.r_[segment[:-1] == segment[1:], False]
    starts = np.flatnonzero(mask & (~np.r_[False, mask[:-1]] | ~previous_same))
    ends = np.flatnonzero(mask & (~np.r_[mask[1:], False] | ~next_same))
    return starts, ends


def _remove_short_runs(mask: np.ndarray, t: np.ndarray, segment: np.ndarray, minimum: float) -> np.ndarray:
    result = mask.copy()
    if minimum <= 0:
        return result
    starts, ends = _mask_runs(result, segment)
    for start, end in zip(starts, ends):
        if t[end] - t[start] < minimum:
            result[start:end + 1] = False
    return result


def _merge_mask_gaps(mask: np.ndarray, t: np.ndarray, segment: np.ndarray, maximum: float) -> np.ndarray:
    result = mask.copy()
    if maximum <= 0 or result.sum() < 2:
        return result
    starts, ends = _mask_runs(result, segment)
    for left_end, right_start in zip(ends[:-1], starts[1:]):
        if segment[left_end] == segment[right_start] and t[right_start] - t[left_end] <= maximum:
            result[left_end:right_start + 1] = True
    return result


def _segment_frame(mask: np.ndarray, t: np.ndarray, segment: np.ndarray, reason: str) -> pd.DataFrame:
    starts, ends = _mask_runs(mask, segment)
    if not starts.size:
        return pd.DataFrame(columns=["Start", "End", "Duration", "Reasons"])
    return pd.DataFrame({
        "Start": t[starts], "End": t[ends], "Duration": t[ends] - t[starts],
        "Reasons": [reason] * len(starts),
    })


def _interpolate_xy(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, segment: np.ndarray,
    protected: np.ndarray, max_gap: float, max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xo, yo = x.copy(), y.copy()
    interpolated = np.zeros(len(x), dtype=bool)
    missing = ~np.isfinite(xo) | ~np.isfinite(yo)
    starts, ends = _mask_runs(missing, segment)
    for start, end in zip(starts, ends):
        left, right = start - 1, end + 1
        if left < 0 or right >= len(xo):
            continue
        gap = np.arange(start, end + 1)
        if protected[gap].any() or segment[left] != segment[right]:
            continue
        if not np.all(np.isfinite([xo[left], yo[left], xo[right], yo[right]])):
            continue
        if t[right] - t[left] > max_gap:
            continue
        if np.hypot(xo[right] - xo[left], yo[right] - yo[left]) > max_distance:
            continue
        alpha = (t[gap] - t[left]) / (t[right] - t[left])
        xo[gap] = xo[left] + alpha * (xo[right] - xo[left])
        yo[gap] = yo[left] + alpha * (yo[right] - yo[left])
        interpolated[gap] = True
    return xo, yo, interpolated


def _median_smooth(values: np.ndarray, t: np.ndarray, segment: np.ndarray, window: float, max_points: int) -> np.ndarray:
    result = values.copy()
    valid = np.isfinite(values)
    starts, ends = _mask_runs(valid, segment)
    keep = max_points if max_points % 2 == 1 else max(1, max_points - 1)
    for start, end in zip(starts, ends):
        run = np.arange(start, end + 1)
        for center in run:
            local = run[np.abs(t[run] - t[center]) <= window / 2.0]
            if local.size > keep:
                nearest = np.argsort(np.abs(t[local] - t[center]), kind="stable")[:keep]
                local = local[nearest]
            result[center] = float(np.median(values[local]))
    return result


def _clean_pupil(
    pupil: np.ndarray, t: np.ndarray, segment: np.ndarray, protected: np.ndarray,
    blink_padding: np.ndarray, sample_rate: float, cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if pupil.shape[1] == 0:
        shape = pupil.shape
        return pupil.copy(), np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, bool), "not_available"
    blocked = protected | blink_padding
    base_valid = np.isfinite(pupil) & (pupil > 0) & (~blocked[:, None])
    artifact = np.zeros_like(base_valid)
    clean = pupil.copy()
    interpolated = np.zeros_like(base_valid)
    for eye in range(pupil.shape[1]):
        speed = np.full(len(t), np.nan)
        if len(t) > 1:
            delta = np.diff(t)
            pair_valid = base_valid[:-1, eye] & base_valid[1:, eye] & (segment[:-1] == segment[1:]) & (delta > 0)
            local = np.full(len(t) - 1, np.nan)
            local[pair_valid] = np.abs(np.diff(pupil[:, eye])[pair_valid]) / delta[pair_valid]
            speed[1:] = local
        reference = speed[np.isfinite(speed)]
        if reference.size >= 3:
            threshold = _robust_upper(reference, cfg.pupil_dilation_mad_multiplier)
            high = np.isfinite(speed) & (speed > threshold)
            artifact[:, eye] = high | np.r_[high[1:], False]
            artifact[:, eye] = _expand_mask(artifact[:, eye], t, cfg.pupil_artifact_padding_s, segment)
        clean[~base_valid[:, eye] | artifact[:, eye], eye] = np.nan
        clean[:, eye], interpolated[:, eye] = _interpolate_signal(
            clean[:, eye], t, segment, blocked, cfg.pupil_max_interpolation_gap_s
        )
        if np.isfinite(sample_rate) and sample_rate > 2 * cfg.pupil_filter_cutoff_hz:
            clean[:, eye] = _gaussian_filter_actual_time(clean[:, eye], t, segment, cfg.pupil_filter_cutoff_hz)
    status = "available_filtered" if np.isfinite(sample_rate) and sample_rate > 2 * cfg.pupil_filter_cutoff_hz else "available_filter_disabled_low_sample_rate"
    return clean, np.isfinite(clean), artifact, interpolated, status


def _interpolate_signal(
    values: np.ndarray, t: np.ndarray, segment: np.ndarray, blocked: np.ndarray, max_gap: float
) -> tuple[np.ndarray, np.ndarray]:
    result = values.copy()
    interpolated = np.zeros(len(values), dtype=bool)
    starts, ends = _mask_runs(~np.isfinite(result), segment)
    for start, end in zip(starts, ends):
        left, right = start - 1, end + 1
        if left < 0 or right >= len(result):
            continue
        gap = np.arange(start, end + 1)
        if blocked[gap].any() or segment[left] != segment[right]:
            continue
        if not (np.isfinite(result[left]) and np.isfinite(result[right])):
            continue
        if t[right] - t[left] > max_gap:
            continue
        alpha = (t[gap] - t[left]) / (t[right] - t[left])
        result[gap] = result[left] + alpha * (result[right] - result[left])
        interpolated[gap] = True
    return result, interpolated


def _gaussian_filter_actual_time(values: np.ndarray, t: np.ndarray, segment: np.ndarray, cutoff: float) -> np.ndarray:
    result = values.copy()
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * cutoff)
    valid = np.isfinite(values)
    starts, ends = _mask_runs(valid, segment)
    for start, end in zip(starts, ends):
        run = np.arange(start, end + 1)
        if run.size < 5 or t[end] - t[start] < 6.0 * sigma:
            continue
        for center in run:
            local = run[np.abs(t[run] - t[center]) <= 3.0 * sigma]
            weights = np.exp(-0.5 * ((t[local] - t[center]) / sigma) ** 2)
            result[center] = float(np.sum(weights * values[local]) / np.sum(weights))
    return result


def _json_number(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None
