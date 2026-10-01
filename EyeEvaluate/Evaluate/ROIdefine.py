"""Point-to-ROI adapters backed by the shared, non-executable ROI contract."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

_FILE = Path(__file__).resolve()
_REPOSITORY_ROOT = _FILE.parents[2]
_EYE_EVALUATE_ROOT = _FILE.parents[1]
for _path in (_REPOSITORY_ROOT, _EYE_EVALUATE_ROOT):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from roi_contract import (
    ROI,
    classify_point as _classify_point,
    parse_legacy_description,
    validate_rois,
)


def identify_roi(roi_list, x, y):
    """Return the first hit's name, preserving the legacy single-hit API.

    Invalid or absent coordinates return ``None`` and are never labelled as a
    blink. Descriptions are accepted only through the shared legacy whitelist;
    they are never evaluated as Python code.
    """
    classification = classify_roi_point(roi_list, x, y)
    if classification["status"] != "matched":
        return None
    return classification["primary"]


def all_hits(roi_list, x, y):
    """Return ordered ROI names, or ``None`` when the point is not valid."""
    classification = classify_roi_point(roi_list, x, y)
    if classification["status"] in {"missing", "invalid_coordinates"}:
        return None
    return list(classification["all_hits"])


def classify_roi_point(roi_list, x, y):
    """Return ``status``, primary name, and ordered hit names for one point."""
    rois = _normalise_roi_list(roi_list)
    result = _classify_point(rois, x, y)
    status = result.status
    if status == "invalid_coordinates" and (_is_missing(x) or _is_missing(y)):
        status = "missing"
    primary = getattr(result, "primary", None)
    return {
        "status": status,
        "primary": primary.name if primary is not None else None,
        "all_hits": tuple(roi.name for roi in result.hits),
    }


def process_roi_lists(x_list, y_list, roi_list):
    """Classify parallel coordinate lists without inferring blink status.

    Valid points return the first ROI name or the historical ``"None"``
    outside marker. Missing and invalid coordinates return explicit strings
    ``"missing"`` and ``"invalid"``; missing coordinates are not blinks.
    """
    if len(x_list) != len(y_list):
        raise ValueError("x_list 和 y_list 必须长度相同")

    rois = _normalise_roi_list(roi_list)
    result: list[str] = []
    for x, y in zip(x_list, y_list):
        classification = _classify_point(rois, x, y)
        if classification.status == "missing":
            result.append("missing")
        elif classification.status == "invalid_coordinates":
            result.append("invalid")
        elif classification.status == "outside":
            result.append("None")
        else:
            primary = getattr(classification, "primary", None)
            result.append(primary.name if primary is not None else "None")
    return result


def _normalise_roi_list(roi_list) -> list[ROI]:
    normalised: list[ROI] = []
    for index, item in enumerate(roi_list or []):
        if isinstance(item, ROI):
            normalised.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(f"ROI item {index} must be an ROI or mapping, got {type(item).__name__}")

        cached = item.get("_roi")
        if isinstance(cached, ROI):
            normalised.append(cached)
            continue

        if "shape" in item and "geometry" in item and "name" in item:
            fields = {
                key: item[key]
                for key in (
                    "schema_version",
                    "id",
                    "name",
                    "shape",
                    "geometry",
                    "priority",
                    "boundary_mode",
                    "source",
                    "coordinate_reference",
                    "mapping",
                )
                if key in item
            }
            normalised.append(ROI.from_dict(fields))
            continue

        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"ROI item {index} has neither structured geometry nor a safe legacy description.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"ROI item {index} has an empty name.")
        try:
            parsed = parse_legacy_description(
                description.strip(),
                shape_hint=item.get("shape"),
                identity_seed=f"evaluate-api:{index}:{name}:{description}",
                reference=item.get("coordinate_reference"),
                name=name,
                priority=int(item.get("priority", index)),
            )
        except Exception as exc:
            reason = getattr(exc, "reason", str(exc))
            raise ValueError(f"ROI {name!r} has an unsafe or malformed legacy description: {reason}") from exc
        normalised.append(parsed)

    validate_rois(normalised)
    ordered = sorted(enumerate(normalised), key=lambda pair: (pair[1].priority, pair[0]))
    return [roi for _, roi in ordered]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    try:
        return isinstance(value, (int, float)) and math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False
