"""Explicit gaze feature formulas. No camera, UI, clock or file access."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .constants import LEFT_EYE_INDICES, MUTUAL_INDICES, RIGHT_EYE_INDICES

FEATURE_FORMULA_V1 = "eyetrax-landmark-features-v1"
FEATURE_FORMULA_V2 = "eyetrax-compact-eye-head-v2"
LEGACY_UNVERSIONED_FORMULA = "legacy-unversioned"
DEFAULT_FEATURE_FORMULA_VERSION = FEATURE_FORMULA_V2
GEOMETRY_RELATIVE_EPSILON = 1e-6

_EYE_FIELDS = ("u", "v", "ear", "iris_diameter_over_eye_width", "eye_width_over_outer_span")
V2_FIELDS = tuple(
    f"eye_{tag}_{field}"
    for tag in ("33_133_iris468", "362_263_iris473")
    for field in _EYE_FIELDS
) + tuple(
    f"head_basis_{axis}_{component}"
    for axis in ("x", "y") for component in ("image_x", "image_y", "pseudo_z")
) + (
    "nose_x_over_image_width", "nose_y_over_image_height",
    "outer_eye_span_over_image_width", "face_height_over_outer_eye_span",
)


def get_feature_schema(formula_version: str) -> dict[str, Any]:
    """Return an independent canonical schema; legacy assumes the old formula."""
    if formula_version in {FEATURE_FORMULA_V1, LEGACY_UNVERSIONED_FORMULA}:
        ordered = [f"landmark_{i}_{axis}" for i in LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES for axis in ("x", "y", "z")]
        ordered += ["head_basis_descriptor_1", "head_basis_descriptor_2", "head_basis_descriptor_3"]
        schema = {
            "schema_version": "eyetrax-feature-schema-v1",
            "feature_formula_version": FEATURE_FORMULA_V1,
            "ordered_fields": ordered, "feature_count": len(ordered),
            "coordinate_source": "refined face landmarks normalized in image coordinates",
            "landmark_transform": "nose-anchored local basis, normalized by 3-D eye-corner span",
            "head_descriptor_note": "coarse basis descriptors; not ground-truth yaw/pitch angles",
        }
    elif formula_version == FEATURE_FORMULA_V2:
        schema = {
            "schema_version": "eyetrax-feature-schema-v2",
            "feature_formula_version": FEATURE_FORMULA_V2,
            "ordered_fields": list(V2_FIELDS), "feature_count": 20,
            "coordinate_source": "refined 478-point landmarks; x*image_width, y*image_height, z*image_width",
            "landmark_transform": "2-D eye-corner midpoint tangent/perpendicular divided by eye width; 3-D pseudo-pixel head basis",
            "head_descriptor_note": "unit basis components, not physical yaw/pitch or calibrated camera depth",
            "field_units": {name: "dimensionless" for name in V2_FIELDS},
            "head_descriptor_fields": list(V2_FIELDS[10:16]),
            "geometry_relative_epsilon": GEOMETRY_RELATIVE_EPSILON,
            "calculation_dtype": "float64", "feature_dtype": "float32",
            "image_orientation": "decoded camera image without an added mirror transform",
            "eye_axes": "33->133 and 362->263; perpendicular=(-tangent_y,tangent_x), image y increases down",
            "iris_diameter": "mean of two opposite ring-pair pixel distances divided by same-eye corner width",
        }
    else:
        raise ValueError(f"unsupported feature formula: {formula_version!r}")
    return copy.deepcopy(schema)


def head_descriptor_indices(formula_version: str) -> tuple[int, ...]:
    schema = get_feature_schema(formula_version)
    fields = schema.get("head_descriptor_fields", schema["ordered_fields"][-3:])
    return tuple(schema["ordered_fields"].index(field) for field in fields)


@dataclass(frozen=True)
class CompactFeatureResult:
    features: np.ndarray | None
    formula_version: str
    reason: str | None
    stability_observables: dict[str, Any] | None


def extract_compact_features(
    landmarks: Sequence[Any], image_width: int | float, image_height: int | float,
) -> CompactFeatureResult:
    """Compute v2; local-eye ratios are similarity invariant, global cues are not.

    z*width follows MediaPipe's weak-perspective convention; it is not depth.
    Numerical epsilon guards division, and is not a camera quality certificate.
    Invalid inputs yield no feature vector, rather than imputed zeros.
    """
    def invalid(reason):
        return CompactFeatureResult(None, FEATURE_FORMULA_V2, reason, None)
    try:
        width, height = float(image_width), float(image_height)
        if isinstance(image_width, bool) or isinstance(image_height, bool) or not np.isfinite([width, height]).all() or min(width, height) <= 0:
            return invalid("invalid_image_dimensions")
        if len(landmarks) < 478:
            return invalid("missing_refined_landmarks")
        indices = {4, 10, 152, 33, 133, 362, 263, 159, 145, 386, 374, *range(468, 478)}
        points = {}
        for index in indices:
            lm = landmarks[index]
            point = np.asarray((lm.x, lm.y, lm.z) if hasattr(lm, "x") else lm[:3], dtype=np.float64)
            if point.shape != (3,) or not np.isfinite(point).all():
                return invalid("nonfinite_or_invalid_landmark")
            points[index] = point * [width, height, width]
        outer_span = float(np.linalg.norm(points[263][:2] - points[33][:2]))
        pixel_epsilon = GEOMETRY_RELATIVE_EPSILON * max(width, height)
        if outer_span <= pixel_epsilon:
            return invalid("degenerate_outer_eye_span")
        local = []
        eye_uv = []
        for a, b, iris, upper, lower, ring in (
            (33, 133, 468, 159, 145, (469, 470, 471, 472)),
            (362, 263, 473, 386, 374, (474, 475, 476, 477)),
        ):
            origin = (points[a][:2] + points[b][:2]) / 2
            axis = points[b][:2] - points[a][:2]
            length = float(np.linalg.norm(axis))
            if length <= GEOMETRY_RELATIVE_EPSILON * outer_span:
                return invalid("degenerate_eye_corner_width")
            tangent = axis / length
            perpendicular = np.array([-tangent[1], tangent[0]])
            relative = points[iris][:2] - origin
            uv = (float(relative @ tangent / length), float(relative @ perpendicular / length))
            ear = np.linalg.norm(points[upper][:2] - points[lower][:2]) / length
            diameter = (np.linalg.norm(points[ring[0]][:2] - points[ring[2]][:2]) + np.linalg.norm(points[ring[1]][:2] - points[ring[3]][:2])) / 2
            local.extend((*uv, float(ear), float(diameter / length), length / outer_span))
            eye_uv.append(uv)
        x = points[263] - points[33]
        x_length = np.linalg.norm(x)
        x /= x_length
        y = points[10] - points[4]
        y -= (y @ x) * x
        y_length = np.linalg.norm(y)
        if y_length <= GEOMETRY_RELATIVE_EPSILON * x_length:
            return invalid("degenerate_head_basis")
        y /= y_length
        center = (float(points[4][0] / width), float(points[4][1] / height))
        face = (*center, outer_span / width, float(np.linalg.norm(points[152][:2] - points[10][:2]) / outer_span))
        vector = np.asarray((*local, *x, *y, *face), dtype=np.float32)
        if vector.shape != (20,) or not np.isfinite(vector).all():
            return invalid("nonfinite_feature_vector")
        observables = {"eye_uv": tuple(eye_uv), "head_basis_x": tuple(float(v) for v in x),
                       "head_basis_y": tuple(float(v) for v in y), "face_center_xy": center,
                       "outer_eye_span_px": outer_span}
        return CompactFeatureResult(vector, FEATURE_FORMULA_V2, None, observables)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return invalid("invalid_landmark_input")
