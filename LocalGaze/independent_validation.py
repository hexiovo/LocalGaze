"""Pure helpers for locked-model, independent nine-point validation."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


GRID_FRACTIONS = (0.1, 0.5, 0.9)
SETTLE_SECONDS = 0.5
OBSERVE_SECONDS = 1.0


def nine_point_targets(width: int, height: int) -> list[dict[str, Any]]:
    """Return a three-by-three interior grid in screen pixels."""
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 1
        or height < 1
    ):
        raise ValueError("validation canvas dimensions must be positive integers")
    return [
        {
            "target_id": f"P{row * 3 + column + 1:02d}",
            "grid_row": row + 1,
            "grid_column": column + 1,
            "fraction_xy": [x_fraction, y_fraction],
            "target_xy_px": [float(round((width - 1) * x_fraction)), float(round((height - 1) * y_fraction))],
        }
        for row, y_fraction in enumerate(GRID_FRACTIONS)
        for column, x_fraction in enumerate(GRID_FRACTIONS)
    ]


def _finite_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, OverflowError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def summarize_independent_validation(
    targets: Sequence[Mapping[str, Any]],
    samples: Iterable[Mapping[str, Any]],
    *,
    capture_status: str,
    context_valid: bool = True,
) -> dict[str, Any]:
    """Summarize accepted pixel predictions without inventing a pass threshold.

    Coverage uses every processed sample in that target's observation window as
    its denominator. Weighted statistics pool valid samples; target-equal
    statistics give each target with a valid prediction equal weight.
    """
    target_map = {str(item["target_id"]): dict(item) for item in targets}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unassigned = 0
    for sample in samples:
        target_id = sample.get("target_id")
        if isinstance(target_id, str) and target_id in target_map:
            grouped[target_id].append(sample)
        else:
            unassigned += 1

    per_target: list[dict[str, Any]] = []
    valid_errors: list[float] = []
    target_means: list[float] = []
    target_medians: list[float] = []
    target_mean_squares: list[float] = []
    incomplete_reasons: list[str] = []
    total_count = 0

    for target in targets:
        target_id = str(target["target_id"])
        target_samples = grouped.get(target_id, [])
        total = len(target_samples)
        errors: list[float] = []
        for sample in target_samples:
            target_xy = _finite_xy(sample.get("target_xy_px"))
            predicted_xy = _finite_xy(sample.get("predicted_xy_px"))
            if (
                sample.get("accepted") is True
                and target_xy is not None
                and predicted_xy is not None
            ):
                dx = predicted_xy[0] - target_xy[0]
                dy = predicted_xy[1] - target_xy[1]
                errors.append(math.hypot(dx, dy))

        valid = len(errors)
        total_count += total
        valid_errors.extend(errors)
        coverage = (valid / total) if total else None
        mean = (sum(errors) / valid) if valid else None
        median = _median(errors) if valid else None
        rmse = math.sqrt(sum(error * error for error in errors) / valid) if valid else None
        if mean is not None:
            target_means.append(mean)
            target_medians.append(median)
            target_mean_squares.append(rmse * rmse)
        if total == 0:
            incomplete_reasons.append(f"{target_id}:no_observation_samples")
        elif valid == 0:
            incomplete_reasons.append(f"{target_id}:no_valid_predictions")
        per_target.append({
            **dict(target),
            "valid": valid,
            "total": total,
            "coverage": coverage,
            "mean_error_px": mean,
            "median_error_px": median,
            "rmse_px": rmse,
            "sample_status_counts": _status_counts(target_samples),
        })

    if unassigned:
        incomplete_reasons.append("samples_without_target_assignment")
    if capture_status != "target_sequence_complete":
        incomplete_reasons.append(f"capture_stopped:{capture_status or 'unknown'}")
    if not context_valid:
        incomplete_reasons.append("validation_context_changed")

    weighted = _stats(valid_errors)
    equal_target = {
        "targets_with_valid_predictions": len(target_means),
        "mean_error_px": _mean(target_means),
        "median_of_target_medians_px": _median(target_medians) if target_medians else None,
        "rmse_px": math.sqrt(sum(target_mean_squares) / len(target_mean_squares)) if target_mean_squares else None,
    }
    valid_total = len(valid_errors)
    return {
        "status": "complete" if not incomplete_reasons else "incomplete",
        "incomplete_reasons": list(dict.fromkeys(incomplete_reasons)),
        "unassigned_sample_count": unassigned,
        "valid": valid_total,
        "total": total_count,
        "coverage": valid_total / total_count if total_count else None,
        "overall_sample_weighted": weighted,
        "overall_target_equal_weighted": equal_target,
        "per_target": per_target,
        "error_unit": "screen_px_euclidean",
        "accuracy_threshold": None,
        "interpretation_limit": "Pixel error only; no visual-angle, distance, or pass/fail inference.",
    }


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_error_px": None, "median_error_px": None, "rmse_px": None}
    return {
        "mean_error_px": sum(values) / len(values),
        "median_error_px": _median(values),
        "rmse_px": math.sqrt(sum(value * value for value in values) / len(values)),
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _status_counts(samples: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        status = str(sample.get("sample_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


__all__ = [
    "GRID_FRACTIONS",
    "SETTLE_SECONDS",
    "OBSERVE_SECONDS",
    "nine_point_targets",
    "summarize_independent_validation",
]
