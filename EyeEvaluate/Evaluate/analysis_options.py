"""Lightweight, shared validation for user-selectable analysis outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


ANALYSIS_METRIC_GROUPS = (
    ("trajectory", "坐标与轨迹统计"),
    ("fixation", "注视候选"),
    ("fast_movement", "快速移动候选"),
    ("ivt_saccade", "眼跳候选（I-VT）"),
    ("blink", "EAR 闭合候选"),
    ("roi", "ROI 区域统计"),
    ("iris_proxy", "虹膜比例代理"),
    ("entropy", "空间分布熵"),
    ("calibration", "校准训练残差"),
)
ANALYSIS_METRIC_GROUP_IDS = tuple(group_id for group_id, _ in ANALYSIS_METRIC_GROUPS)
ANALYSIS_METRIC_GROUP_IDS_WITH_METADATA = (*ANALYSIS_METRIC_GROUP_IDS, "capabilities")
LEGACY_METRIC_GROUP_ALIASES = {"events": ("fixation", "fast_movement", "blink")}


def normalize_selected_metrics(value: Iterable[str] | None) -> tuple[str, ...]:
    """Expand stable metric-group IDs; capability notes are always retained."""
    if value is None:
        selected = set(ANALYSIS_METRIC_GROUP_IDS)
    else:
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
            raise ValueError("selected_metrics must be an iterable of metric-group IDs or None")
        selected: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("selected_metrics entries must be nonempty strings")
            key = item.strip()
            if key == "events":
                selected.update(("fixation", "fast_movement", "blink"))
            else:
                selected.add(key)
        unknown = selected.difference(ANALYSIS_METRIC_GROUP_IDS_WITH_METADATA)
        if unknown:
            raise ValueError("unknown selected metric groups: " + ", ".join(sorted(unknown)))
    selected.add("capabilities")
    return tuple(group_id for group_id in ANALYSIS_METRIC_GROUP_IDS_WITH_METADATA if group_id in selected)


def normalize_table_format(value: str | None) -> str:
    """Return the selected delimited-table format."""
    selected = "csv" if value is None else value
    if not isinstance(selected, str):
        raise ValueError("table_format must be 'csv' or 'txt'")
    selected = selected.strip().lower()
    if selected not in {"csv", "txt"}:
        raise ValueError("table_format must be 'csv' or 'txt'")
    return selected


_ANALYSIS_CONFIG_FIELDS = {
    "max_time_gap_s",
    "gap_median_multiplier",
    "fixation_min_samples",
    "fixation_min_duration_s",
    "fixation_dispersion_diagonal_ratio",
    "fast_speed_diagonals_s",
    "ivt_speed_diagonals_s",
    "ivt_min_duration_s",
    "ivt_max_duration_s",
    "ivt_min_samples",
    "blink_min_duration_s",
    "blink_max_duration_s",
    "entropy_grid",
}


def normalize_analysis_config(value: Mapping[str, object] | None) -> dict[str, object]:
    """Validate an AnalysisConfig-shaped mapping without importing analytics."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("analysis_config must be an object")
    unknown = set(value).difference(_ANALYSIS_CONFIG_FIELDS)
    if unknown:
        raise ValueError("unknown analysis_config fields: " + ", ".join(sorted(str(key) for key in unknown)))
    result: dict[str, object] = {}
    for key, raw in value.items():
        if key == "entropy_grid":
            if not isinstance(raw, (list, tuple)) or len(raw) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in raw):
                raise ValueError("entropy_grid must contain two positive integers")
            result[key] = tuple(raw)
            continue
        if key in {"fixation_min_samples", "ivt_min_samples"}:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 3:
                raise ValueError(f"{key} must be an integer >= 3")
            result[key] = raw
            continue
        if raw is None and key == "max_time_gap_s":
            result[key] = None
            continue
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be positive and finite")
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be positive and finite") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{key} must be positive and finite")
        result[key] = number
    for minimum_key, maximum_key, default_minimum, default_maximum, label in (
        ("blink_min_duration_s", "blink_max_duration_s", 0.050, 0.500, "blink"),
        ("ivt_min_duration_s", "ivt_max_duration_s", 0.020, 0.200, "I-VT"),
    ):
        minimum = result.get(minimum_key, default_minimum)
        maximum = result.get(maximum_key, default_maximum)
        if maximum < minimum:
            raise ValueError(f"{label} duration limits are reversed")
    return result


__all__ = [
    "ANALYSIS_METRIC_GROUPS",
    "ANALYSIS_METRIC_GROUP_IDS",
    "ANALYSIS_METRIC_GROUP_IDS_WITH_METADATA",
    "LEGACY_METRIC_GROUP_ALIASES",
    "normalize_selected_metrics",
    "normalize_table_format",
    "normalize_analysis_config",
]
