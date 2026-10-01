"""Pure-Python ROI geometry, coordinate mapping, and safe legacy migration.

This module intentionally has no Tk, OpenCV, pandas, or application-global
dependencies so the drawing and analysis entry points can share one contract.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import numbers
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
SHAPES = frozenset({"rectangle", "circle", "ellipse"})
BOUNDARY_MODES = frozenset({"inclusive", "strict"})
_UNKNOWN_REFERENCE = {"space": "unknown", "source": "unknown"}
ROI_METADATA_COLUMNS = (
    "schema_version",
    "row_index",
    "roi_id",
    "priority",
    "boundary_mode",
    "geometry_json",
    "source_json",
    "reference_space",
    "reference_width",
    "reference_height",
    "reference_source",
    "monitor_id",
    "monitor_origin_px_json",
    "dpi_scale_json",
    "dpi_source",
    "reference_extra_json",
    "mapping_json",
)


class ROIValidationError(ValueError):
    """Invalid ROI geometry or metadata with a user-readable reason."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class LegacyROIParseError(ValueError):
    """A legacy description is outside the supported non-executable grammar."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class CoordinateMappingError(ValueError):
    """Invalid or unavailable mapping between canvas and reference coordinates."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ROIValidationError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ROIValidationError(f"{label} must be finite")
    return result


def _ast_number(node: ast.AST, label: str) -> float:
    """Read a single numeric AST literal, allowing unary +/- only."""
    sign = 1.0
    value_node = node
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        value_node = node.operand
    if not isinstance(value_node, ast.Constant) or isinstance(value_node.value, bool):
        raise LegacyROIParseError(f"{label} is not a numeric literal")
    if not isinstance(value_node.value, (int, float)):
        raise LegacyROIParseError(f"{label} is not a numeric literal")
    result = sign * float(value_node.value)
    if not math.isfinite(result):
        raise LegacyROIParseError(f"{label} must be finite")
    return result


def _normalise_shape(shape: str) -> str:
    value = str(shape).strip().lower()
    if value not in SHAPES:
        raise ROIValidationError(f"unsupported ROI shape: {shape!r}")
    return value


def _normalise_geometry(shape: str, geometry: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(geometry, Mapping):
        raise ROIValidationError("ROI geometry must be a mapping")
    required = {
        "rectangle": ("left", "top", "right", "bottom"),
        "circle": ("cx", "cy", "r"),
        "ellipse": ("cx", "cy", "a", "b"),
    }[shape]
    if set(geometry) != set(required):
        missing = sorted(set(required) - set(geometry))
        extra = sorted(set(geometry) - set(required))
        raise ROIValidationError(f"{shape} geometry keys mismatch (missing={missing}, extra={extra})")
    values = {key: _number(geometry[key], f"geometry.{key}") for key in required}
    if shape == "rectangle":
        if values["right"] <= values["left"] or values["bottom"] <= values["top"]:
            raise ROIValidationError("rectangle must have positive width and height")
    elif shape == "circle":
        if values["r"] <= 0:
            raise ROIValidationError("circle radius must be positive")
    elif values["a"] <= 0 or values["b"] <= 0:
        raise ROIValidationError("ellipse radii must be positive")
    return values


def _normalise_reference(reference: Mapping[str, Any] | None) -> dict[str, Any]:
    if reference is None:
        return dict(_UNKNOWN_REFERENCE)
    if not isinstance(reference, Mapping):
        raise ROIValidationError("coordinate_reference must be a mapping or None")
    result = dict(reference)
    space = result.get("space", "unknown")
    if space not in {"unknown", "reference_pixel_centers"}:
        raise ROIValidationError(f"unsupported coordinate reference space: {space!r}")
    width = result.get("width")
    height = result.get("height")
    if (width is None) != (height is None):
        raise ROIValidationError("reference width and height must both be known or both be unknown")
    if width is not None:
        if isinstance(width, bool) or not isinstance(width, numbers.Integral):
            raise ROIValidationError("reference width must be an integer")
        if isinstance(height, bool) or not isinstance(height, numbers.Integral):
            raise ROIValidationError("reference height must be an integer")
        width, height = int(width), int(height)
        if width <= 1 or height <= 1:
            raise ROIValidationError("reference width and height must exceed one pixel center")
        result["width"], result["height"] = width, height
        if space == "unknown":
            raise ROIValidationError("known reference dimensions require space='reference_pixel_centers'")
    elif space != "unknown":
        raise ROIValidationError("reference_pixel_centers requires width and height")

    monitor_id = result.get("monitor_id")
    if monitor_id is not None and not isinstance(monitor_id, str):
        raise ROIValidationError("monitor_id must be a string or None")
    result.setdefault("monitor_id", None)

    origin = result.get("monitor_origin_px")
    if origin is not None:
        if not isinstance(origin, (tuple, list)) or len(origin) != 2:
            raise ROIValidationError("monitor_origin_px must be a two-value sequence or None")
        result["monitor_origin_px"] = [
            _number(origin[0], "monitor_origin_px[0]"),
            _number(origin[1], "monitor_origin_px[1]"),
        ]
    else:
        result["monitor_origin_px"] = None

    dpi_scale = result.get("dpi_scale")
    if dpi_scale is not None:
        if not isinstance(dpi_scale, (tuple, list)) or len(dpi_scale) != 2:
            raise ROIValidationError("dpi_scale must be a two-value sequence or None")
        parsed = [_number(dpi_scale[0], "dpi_scale[0]"), _number(dpi_scale[1], "dpi_scale[1]")]
        if any(value <= 0 for value in parsed):
            raise ROIValidationError("dpi_scale values must be positive")
        result["dpi_scale"] = parsed
    else:
        result["dpi_scale"] = None

    result.setdefault("dpi_source", "unknown")
    result.setdefault("source", "unknown")
    # Check that metadata can be persisted deterministically and safely.
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ROIValidationError(f"coordinate_reference is not JSON-safe: {exc}") from exc
    return result


_REFERENCE_PROVENANCE_KEYS = frozenset(
    {"source", "dpi_source", "description", "source_description", "label", "comment", "notes"}
)


def validate_coordinate_reference_compatibility(
    references: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] | None = None,
) -> None:
    """Reject conflicts among known coordinate frames without coercing unknowns.

    References in ``unknown`` space are preserved but do not assert a frame.
    For known pixel-center frames, canonical space and dimensions must match.
    Monitor, origin, DPI, and extra identity values conflict only when both
    records declare them. Provenance labels and canvas-to-reference mappings
    are intentionally excluded from frame identity.
    """
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise ROIValidationError("references must be an ordered sequence")
    normalized = [_normalise_reference(reference) for reference in references]
    if labels is None:
        display_labels = [f"reference #{index + 1}" for index in range(len(normalized))]
    else:
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ROIValidationError("labels must be an ordered sequence")
        if len(labels) != len(normalized):
            raise ROIValidationError("labels must match the number of references")
        display_labels = [str(label) for label in labels]

    def is_known(value: Any) -> bool:
        if value is None:
            return False
        return not (
            isinstance(value, str)
            and value.strip().lower() in {"", "unknown", "unspecified", "none", "null"}
        )

    def same_declared_value(left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
            return _close(float(left), float(right))
        if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
            return len(left) == len(right) and all(
                same_declared_value(a, b) for a, b in zip(left, right)
            )
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            shared = set(left).intersection(right)
            return all(
                not (is_known(left[key]) and is_known(right[key]))
                or same_declared_value(left[key], right[key])
                for key in shared
            )
        return type(left) is type(right) and left == right

    def identity_fields(reference: Mapping[str, Any]) -> dict[str, Any]:
        fields = {
            key: value
            for key, value in reference.items()
            if key not in _REFERENCE_PROVENANCE_KEYS
        }
        return fields

    known = [
        (index, reference, identity_fields(reference))
        for index, reference in enumerate(normalized)
        if reference["space"] == "reference_pixel_centers"
    ]
    for position, (left_index, left_reference, left_fields) in enumerate(known):
        for right_index, right_reference, right_fields in known[position + 1 :]:
            keys = set(left_fields).intersection(right_fields)
            for key in keys:
                left_value, right_value = left_fields[key], right_fields[key]
                if not is_known(left_value) or not is_known(right_value):
                    continue
                if not same_declared_value(left_value, right_value):
                    raise ROIValidationError(
                        "coordinate reference conflict between "
                        f"{display_labels[left_index]!r} and {display_labels[right_index]!r}: "
                        f"field {key!r} differs ({left_value!r} vs {right_value!r})"
                    )


def _normalise_source(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return {"kind": "new"}
    if not isinstance(source, Mapping):
        raise ROIValidationError("source must be a mapping or None")
    result = dict(source)
    result.setdefault("kind", "new")
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ROIValidationError(f"source is not JSON-safe: {exc}") from exc
    return result


def _normalise_mapping(mapping: "CanvasMapping | Mapping[str, Any] | None") -> dict[str, Any] | None:
    if mapping is None:
        return None
    if isinstance(mapping, CanvasMapping):
        return mapping.to_dict()
    if isinstance(mapping, Mapping):
        return CanvasMapping.from_dict(mapping).to_dict()
    raise ROIValidationError("mapping must be a CanvasMapping, mapping, or None")


@dataclass(frozen=True)
class ROI:
    """Validated, immutable ROI record; nested mappings are copied on creation."""

    id: str
    name: str
    shape: str
    geometry: dict[str, float]
    priority: int = 0
    boundary_mode: str = "inclusive"
    source: dict[str, Any] = field(default_factory=lambda: {"kind": "new"})
    coordinate_reference: dict[str, Any] = field(default_factory=lambda: dict(_UNKNOWN_REFERENCE))
    mapping: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ROIValidationError(f"unsupported ROI schema version: {self.schema_version!r}")
        name = str(self.name).strip()
        if not name:
            raise ROIValidationError("ROI name must not be empty")
        roi_id = str(self.id).strip()
        if not roi_id:
            raise ROIValidationError("ROI id must not be empty")
        shape = _normalise_shape(self.shape)
        geometry = _normalise_geometry(shape, self.geometry)
        if isinstance(self.priority, bool) or not isinstance(self.priority, numbers.Integral):
            raise ROIValidationError("priority must be an integer")
        priority = int(self.priority)
        if priority < 0:
            raise ROIValidationError("priority must be non-negative")
        boundary = str(self.boundary_mode).strip().lower()
        if boundary not in BOUNDARY_MODES:
            raise ROIValidationError(f"unsupported boundary mode: {self.boundary_mode!r}")
        if boundary == "strict":
            if shape != "rectangle" or not str((self.source or {}).get("kind", "")).startswith("legacy"):
                raise ROIValidationError("strict boundaries are reserved for migrated legacy rectangles")
        object.__setattr__(self, "id", roi_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "boundary_mode", boundary)
        object.__setattr__(self, "source", _normalise_source(self.source))
        reference = _normalise_reference(self.coordinate_reference)
        mapping = _normalise_mapping(self.mapping)
        if mapping is not None and reference["space"] == "reference_pixel_centers":
            expected_target = (
                0.0,
                0.0,
                float(reference["width"] - 1),
                float(reference["height"] - 1),
            )
            actual_target = tuple(mapping["target_bounds"])
            if any(
                not _close(actual, expected)
                for actual, expected in zip(actual_target, expected_target)
            ):
                raise ROIValidationError(
                    "mapping target bounds must match the coordinate reference pixel-center extents"
                )
        object.__setattr__(self, "coordinate_reference", reference)
        object.__setattr__(self, "mapping", mapping)

    @classmethod
    def create(
        cls,
        name: str,
        shape: str,
        geometry: Mapping[str, Any],
        *,
        id: str | None = None,
        priority: int = 0,
        boundary_mode: str = "inclusive",
        source: Mapping[str, Any] | None = None,
        coordinate_reference: Mapping[str, Any] | None = None,
        mapping: "CanvasMapping | Mapping[str, Any] | None" = None,
    ) -> "ROI":
        return cls(
            id=id or str(uuid.uuid4()),
            name=name,
            shape=shape,
            geometry=dict(geometry),
            priority=priority,
            boundary_mode=boundary_mode,
            source=dict(source) if source is not None else {"kind": "new"},
            coordinate_reference=dict(coordinate_reference) if coordinate_reference is not None else dict(_UNKNOWN_REFERENCE),
            mapping=mapping,
            schema_version=SCHEMA_VERSION,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ROI":
        if not isinstance(record, Mapping):
            raise ROIValidationError("ROI record must be a mapping")
        return cls(
            id=record.get("id", record.get("roi_id", "")),
            name=record.get("name", ""),
            shape=record.get("shape", ""),
            geometry=record.get("geometry", {}),
            priority=record.get("priority", 0),
            boundary_mode=record.get("boundary_mode", "inclusive"),
            source=record.get("source"),
            coordinate_reference=record.get("coordinate_reference"),
            mapping=record.get("mapping"),
            schema_version=record.get("schema_version", SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "shape": self.shape,
            "geometry": dict(self.geometry),
            "priority": self.priority,
            "boundary_mode": self.boundary_mode,
            "source": dict(self.source),
            "coordinate_reference": dict(self.coordinate_reference),
            "mapping": dict(self.mapping) if self.mapping is not None else None,
        }


def convert_strict_rectangle_to_inclusive(roi: ROI) -> ROI:
    """Explicitly retain a legacy rectangle's geometry while including its edges.

    The operation changes only boundary semantics and records that choice in
    provenance. Callers must invoke this named conversion rather than relying
    on implicit migration behavior.
    """
    if not isinstance(roi, ROI):
        raise ROIValidationError("roi must be a validated ROI record")
    if roi.shape != "rectangle" or roi.boundary_mode != "strict":
        raise ROIValidationError("only a strict-boundary rectangle can be converted to inclusive")
    source = dict(roi.source)
    source["boundary_conversion"] = {"from": "strict", "to": "inclusive"}
    return ROI.create(
        name=roi.name,
        shape=roi.shape,
        geometry=roi.geometry,
        id=roi.id,
        priority=roi.priority,
        boundary_mode="inclusive",
        source=source,
        coordinate_reference=roi.coordinate_reference,
        mapping=roi.mapping,
    )


@dataclass(frozen=True)
class CanvasMapping:
    """Axis-aligned affine mapping between two pixel-center coordinate frames."""

    source_bounds: tuple[float, float, float, float]
    target_bounds: tuple[float, float, float, float]
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise CoordinateMappingError(f"unsupported mapping version: {self.schema_version!r}")
        source = _bounds(self.source_bounds, "source_bounds")
        target = _bounds(self.target_bounds, "target_bounds")
        sx = _number(self.scale_x, "scale_x")
        sy = _number(self.scale_y, "scale_y")
        ox = _number(self.offset_x, "offset_x")
        oy = _number(self.offset_y, "offset_y")
        if sx == 0 or sy == 0:
            raise CoordinateMappingError("mapping scales must be non-zero")
        expected_sx = (target[2] - target[0]) / (source[2] - source[0])
        expected_sy = (target[3] - target[1]) / (source[3] - source[1])
        expected_ox = target[0] - expected_sx * source[0]
        expected_oy = target[1] - expected_sy * source[1]
        if not _close(sx, expected_sx) or not _close(sy, expected_sy) or not _close(ox, expected_ox) or not _close(oy, expected_oy):
            raise CoordinateMappingError("mapping bounds do not match its stored scale and offset")
        object.__setattr__(self, "source_bounds", source)
        object.__setattr__(self, "target_bounds", target)
        object.__setattr__(self, "scale_x", sx)
        object.__setattr__(self, "scale_y", sy)
        object.__setattr__(self, "offset_x", ox)
        object.__setattr__(self, "offset_y", oy)

    @classmethod
    def from_endpoints(
        cls,
        source_bounds: Sequence[float],
        target_bounds: Sequence[float],
    ) -> "CanvasMapping":
        source = _bounds(source_bounds, "source_bounds")
        target = _bounds(target_bounds, "target_bounds")
        sx = (target[2] - target[0]) / (source[2] - source[0])
        sy = (target[3] - target[1]) / (source[3] - source[1])
        if sx == 0 or sy == 0:
            raise CoordinateMappingError("mapping scales must be non-zero")
        return cls(
            source_bounds=source,
            target_bounds=target,
            scale_x=sx,
            scale_y=sy,
            offset_x=target[0] - sx * source[0],
            offset_y=target[1] - sy * source[1],
        )

    @classmethod
    def from_sizes(
        cls,
        canvas_width: int,
        canvas_height: int,
        reference_width: int,
        reference_height: int,
        *,
        canvas_origin: Sequence[float] = (0, 0),
        target_origin: Sequence[float] = (0, 0),
    ) -> "CanvasMapping":
        cw, ch = _dimension(canvas_width, "canvas_width"), _dimension(canvas_height, "canvas_height")
        rw, rh = _dimension(reference_width, "reference_width"), _dimension(reference_height, "reference_height")
        cox, coy = _pair(canvas_origin, "canvas_origin")
        tox, toy = _pair(target_origin, "target_origin")
        return cls.from_endpoints(
            (cox, coy, cox + cw - 1, coy + ch - 1),
            (tox, toy, tox + rw - 1, toy + rh - 1),
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "CanvasMapping":
        if not isinstance(record, Mapping):
            raise CoordinateMappingError("mapping record must be a mapping")
        return cls(
            source_bounds=tuple(record["source_bounds"]),
            target_bounds=tuple(record["target_bounds"]),
            scale_x=record["scale_x"],
            scale_y=record["scale_y"],
            offset_x=record["offset_x"],
            offset_y=record["offset_y"],
            schema_version=record.get("schema_version", SCHEMA_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_bounds": list(self.source_bounds),
            "target_bounds": list(self.target_bounds),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "coordinate_rule": "pixel-center endpoints [0, width-1] and [0, height-1]",
        }

    def map_point(self, x: Any, y: Any, *, inverse: bool = False) -> tuple[float, float]:
        px, py = _number(x, "x"), _number(y, "y")
        if inverse:
            return ((px - self.offset_x) / self.scale_x, (py - self.offset_y) / self.scale_y)
        return (px * self.scale_x + self.offset_x, py * self.scale_y + self.offset_y)

    def map_geometry(
        self,
        shape: str,
        geometry: Mapping[str, Any],
        *,
        inverse: bool = False,
    ) -> tuple[str, dict[str, float]]:
        shape = _normalise_shape(shape)
        values = _normalise_geometry(shape, geometry)
        sx, sy, ox, oy = self.scale_x, self.scale_y, self.offset_x, self.offset_y
        if inverse:
            sx, sy = 1.0 / sx, 1.0 / sy
            ox, oy = -self.offset_x / self.scale_x, -self.offset_y / self.scale_y
        if shape == "rectangle":
            points = [
                self._map(values["left"], values["top"], sx, sy, ox, oy),
                self._map(values["right"], values["bottom"], sx, sy, ox, oy),
            ]
            x0, x1 = sorted((points[0][0], points[1][0]))
            y0, y1 = sorted((points[0][1], points[1][1]))
            return "rectangle", {"left": x0, "top": y0, "right": x1, "bottom": y1}
        cx, cy = self._map(values["cx"], values["cy"], sx, sy, ox, oy)
        if shape == "circle":
            rx = values["r"] * abs(sx)
            ry = values["r"] * abs(sy)
            if _close(rx, ry):
                return "circle", {"cx": cx, "cy": cy, "r": (rx + ry) / 2.0}
            return "ellipse", {"cx": cx, "cy": cy, "a": rx, "b": ry}
        return "ellipse", {
            "cx": cx,
            "cy": cy,
            "a": values["a"] * abs(sx),
            "b": values["b"] * abs(sy),
        }

    @staticmethod
    def _map(x: float, y: float, sx: float, sy: float, ox: float, oy: float) -> tuple[float, float]:
        return (x * sx + ox, y * sy + oy)


@dataclass(frozen=True)
class PointClassification:
    status: str
    primary: ROI | None
    hits: tuple[ROI, ...]

    @property
    def all_hits(self) -> tuple[ROI, ...]:
        return self.hits

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "primary": self.primary.name if self.primary is not None else None,
            "all_hits": [roi.name for roi in self.hits],
            "all_hit_ids": [roi.id for roi in self.hits],
        }


def geometry_from_bounds(
    shape: str,
    bounds: Sequence[float],
    *,
    constrain_circle: bool = False,
) -> dict[str, float]:
    """Build normalized geometry from two canvas/reference corners.

    ``bounds`` is ``[x0, y0, x1, y1]`` and may be in either drag direction.
    Ellipse mode is never converted to a circle. Circle mode rejects unequal
    radii unless the caller explicitly requests a centered inscribed circle.
    """
    shape = _normalise_shape(shape)
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
        raise ROIValidationError("bounds must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (_number(value, f"bounds[{i}]") for i, value in enumerate(bounds))
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if shape == "rectangle":
        return _normalise_geometry(
            shape, {"left": left, "top": top, "right": right, "bottom": bottom}
        )
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    rx, ry = (right - left) / 2.0, (bottom - top) / 2.0
    if shape == "circle":
        if constrain_circle:
            radius = min(rx, ry)
        else:
            if not _close(rx, ry):
                raise ROIValidationError("circle must have equal radii in the canonical reference frame")
            radius = (rx + ry) / 2.0
        return _normalise_geometry(shape, {"cx": cx, "cy": cy, "r": radius})
    return _normalise_geometry(shape, {"cx": cx, "cy": cy, "a": rx, "b": ry})


def format_legacy_description(
    shape: str,
    geometry: Mapping[str, Any],
    boundary_mode: str = "inclusive",
) -> str:
    """Format the compatibility description stored in the legacy first sheet."""
    shape = _normalise_shape(shape)
    values = _normalise_geometry(shape, geometry)
    boundary = str(boundary_mode).strip().lower()
    if boundary not in BOUNDARY_MODES:
        raise ROIValidationError(f"unsupported boundary mode: {boundary_mode!r}")
    if boundary == "strict" and shape != "rectangle":
        raise ROIValidationError("strict boundaries are supported only for rectangles")
    n = {key: _format_number(value) for key, value in values.items()}
    if shape == "rectangle":
        if boundary == "strict":
            return f"x>{n['left']} and x<{n['right']} and y>{n['top']} and y<{n['bottom']}"
        return f"x>={n['left']} and x<={n['right']} and y>={n['top']} and y<={n['bottom']}"
    if shape == "circle":
        return f"(x-{n['cx']})**2 + (y-{n['cy']})**2 <= {n['r']}**2"
    return f"((x-{n['cx']})/{n['a']})**2 + ((y-{n['cy']})/{n['b']})**2 <= 1"


def roi_metadata_record(roi: ROI, row_index: int) -> dict[str, Any]:
    """Serialize one ROI using the canonical ``ROI_Metadata`` workbook layout.

    The first sheet remains a three-column compatibility view. This function
    defines the metadata sheet's field names and JSON encoding so workbook
    producers do not independently invent a second schema.
    """
    if not isinstance(roi, ROI):
        raise ROIValidationError("roi must be a validated ROI record")
    if isinstance(row_index, bool) or not isinstance(row_index, numbers.Integral) or row_index < 0:
        raise ROIValidationError("row_index must be a non-negative integer")
    reference = roi.coordinate_reference
    standard = {
        "space",
        "width",
        "height",
        "source",
        "monitor_id",
        "monitor_origin_px",
        "dpi_scale",
        "dpi_source",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "row_index": int(row_index),
        "roi_id": roi.id,
        "priority": roi.priority,
        "boundary_mode": roi.boundary_mode,
        "geometry_json": _json_text(roi.geometry),
        "source_json": _json_text(roi.source),
        "reference_space": reference.get("space", "unknown"),
        "reference_width": reference.get("width"),
        "reference_height": reference.get("height"),
        "reference_source": reference.get("source", "unknown"),
        "monitor_id": reference.get("monitor_id"),
        "monitor_origin_px_json": _json_text(reference.get("monitor_origin_px")),
        "dpi_scale_json": _json_text(reference.get("dpi_scale")),
        "dpi_source": reference.get("dpi_source", "unknown"),
        "reference_extra_json": _json_text(
            {key: value for key, value in reference.items() if key not in standard}
        ),
        "mapping_json": _json_text(roi.mapping),
    }


def serialize_roi_workbook(rois: Sequence[ROI]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return canonical legacy-view rows and versioned metadata rows."""
    ordered = list(rois)
    validate_rois(ordered)
    validate_roi_reference_compatibility(ordered)
    first_sheet: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    ordered_with_priority = sorted(
        enumerate(ordered), key=lambda pair: (pair[1].priority, pair[0])
    )
    for row_index, (_, record) in enumerate(ordered_with_priority):
        first_sheet.append(
            {
                "命名": record.name,
                "形状": record.shape,
                "描述": format_legacy_description(
                    record.shape, record.geometry, record.boundary_mode
                ),
            }
        )
        metadata.append(roi_metadata_record(record, row_index))
    return first_sheet, metadata


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_legacy_description(
    text: str,
    shape_hint: str | None,
    identity_seed: str,
    reference: Mapping[str, Any] | None = None,
    *,
    name: str,
    priority: int = 0,
) -> ROI:
    """Convert one exact historical producer expression into structured ROI.

    Only the old strict rectangle, inclusive circle, and inclusive ellipse
    expressions are accepted. Parsing inspects a restricted AST; it never calls
    ``eval`` or executes any part of the input.
    """
    if not isinstance(text, str) or not text.strip():
        raise LegacyROIParseError("description is empty")
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError) as exc:
        raise LegacyROIParseError(f"invalid expression syntax: {exc.msg if isinstance(exc, SyntaxError) else exc}") from exc

    try:
        shape, geometry, boundary = _parse_rectangle(expression)
        if shape is None:
            shape, geometry, boundary = _parse_circle(expression)
        if shape is None:
            shape, geometry, boundary = _parse_ellipse(expression)
    except LegacyROIParseError:
        raise
    if shape is None:
        raise LegacyROIParseError("expression does not match a supported legacy ROI form")

    if shape_hint is not None and str(shape_hint).strip().lower() != shape:
        raise LegacyROIParseError(
            f"shape column {shape_hint!r} conflicts with the {shape} description"
        )
    if not isinstance(identity_seed, str) or not identity_seed:
        raise LegacyROIParseError("identity_seed is required for stable migration IDs")
    stable_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity_seed))
    try:
        return ROI.create(
            name=name,
            shape=shape,
            geometry=geometry,
            id=stable_id,
            priority=priority,
            boundary_mode=boundary,
            source={"kind": "legacy_migrated", "parser": "legacy-roi-ast-v1"},
            coordinate_reference=reference or dict(_UNKNOWN_REFERENCE),
            mapping=None,
        )
    except ROIValidationError as exc:
        raise LegacyROIParseError(str(exc)) from exc


def _parse_rectangle(node: ast.AST) -> tuple[str | None, dict[str, float] | None, str | None]:
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And) or len(node.values) != 4:
        return None, None, None
    expected = (("x", ast.Gt), ("x", ast.Lt), ("y", ast.Gt), ("y", ast.Lt))
    values: list[float] = []
    for expression, (variable, op_type) in zip(node.values, expected):
        if not isinstance(expression, ast.Compare) or len(expression.ops) != 1 or len(expression.comparators) != 1:
            return None, None, None
        if not isinstance(expression.left, ast.Name) or expression.left.id != variable:
            return None, None, None
        if not isinstance(expression.ops[0], op_type):
            return None, None, None
        values.append(_ast_number(expression.comparators[0], "rectangle bound"))
    geometry = {"left": values[0], "right": values[1], "top": values[2], "bottom": values[3]}
    return "rectangle", _normalise_geometry("rectangle", geometry), "strict"


def _parse_circle(node: ast.AST) -> tuple[str | None, dict[str, float] | None, str | None]:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.LtE):
        return None, None, None
    if len(node.comparators) != 1 or not _is_add(node.left):
        return None, None, None
    first, second = node.left.left, node.left.right
    x_center = _parse_squared_difference(first, "x")
    y_center = _parse_squared_difference(second, "y")
    if x_center is None or y_center is None:
        return None, None, None
    radius_squared = node.comparators[0]
    if not _is_power_two(radius_squared):
        return None, None, None
    radius = _ast_number(radius_squared.left, "circle radius")
    if radius <= 0:
        raise LegacyROIParseError("circle radius must be positive")
    geometry = {"cx": x_center, "cy": y_center, "r": radius}
    return "circle", _normalise_geometry("circle", geometry), "inclusive"


def _parse_ellipse(node: ast.AST) -> tuple[str | None, dict[str, float] | None, str | None]:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.LtE):
        return None, None, None
    if len(node.comparators) != 1 or not _is_add(node.left):
        return None, None, None
    x_center, a = _parse_squared_normalized_difference(node.left.left, "x")
    y_center, b = _parse_squared_normalized_difference(node.left.right, "y")
    if x_center is None or y_center is None or a is None or b is None:
        return None, None, None
    if _ast_number(node.comparators[0], "ellipse threshold") != 1.0:
        return None, None, None
    geometry = {"cx": x_center, "cy": y_center, "a": a, "b": b}
    return "ellipse", _normalise_geometry("ellipse", geometry), "inclusive"


def _is_add(node: ast.AST) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)


def _is_power_two(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and _is_constant_two(node.right)
    )


def _is_constant_two(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int and node.value == 2


def _parse_squared_difference(node: ast.AST, variable: str) -> float | None:
    if not _is_power_two(node):
        return None
    base = node.left
    if not isinstance(base, ast.BinOp) or not isinstance(base.op, ast.Sub):
        return None
    if not isinstance(base.left, ast.Name) or base.left.id != variable:
        return None
    return _ast_number(base.right, f"{variable} center")


def _parse_squared_normalized_difference(node: ast.AST, variable: str) -> tuple[float | None, float | None]:
    if not _is_power_two(node):
        return None, None
    quotient = node.left
    if not isinstance(quotient, ast.BinOp) or not isinstance(quotient.op, ast.Div):
        return None, None
    difference = quotient.left
    if not isinstance(difference, ast.BinOp) or not isinstance(difference.op, ast.Sub):
        return None, None
    if not isinstance(difference.left, ast.Name) or difference.left.id != variable:
        return None, None
    center = _ast_number(difference.right, f"{variable} center")
    radius = _ast_number(quotient.right, f"{variable} radius")
    if radius <= 0:
        raise LegacyROIParseError(f"{variable} radius must be positive")
    return center, radius


def classify_point(rois: Sequence[ROI], x: Any, y: Any) -> PointClassification:
    """Classify a point; ``None``/NaN are missing, other bad values invalid."""
    if _is_missing_coordinate(x) or _is_missing_coordinate(y):
        return PointClassification("missing", None, ())
    try:
        px, py = _number(x, "x"), _number(y, "y")
    except ROIValidationError:
        return PointClassification("invalid_coordinates", None, ())
    ordered = _ordered_rois(rois)
    hits = tuple(roi for roi in ordered if _contains(roi, px, py))
    if not hits:
        return PointClassification("outside", None, ())
    return PointClassification("matched", hits[0], hits)


def all_hits(rois: Sequence[ROI], x: Any, y: Any) -> list[ROI]:
    """Return every matching ROI in priority order; inspect status via classify_point."""
    return list(classify_point(rois, x, y).hits)


def identify_roi(rois: Sequence[ROI], x: Any, y: Any) -> str | None:
    """Compatibility helper returning the first matching ROI name."""
    result = classify_point(rois, x, y)
    return result.primary.name if result.primary is not None else None


def validate_rois(rois: Sequence[ROI]) -> None:
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, roi in enumerate(rois):
        if not isinstance(roi, ROI):
            raise ROIValidationError(f"ROI at index {index} is not a validated ROI record")
        if roi.id in seen_ids:
            raise ROIValidationError(f"duplicate ROI id: {roi.id}")
        if roi.name in seen_names:
            raise ROIValidationError(f"duplicate ROI name: {roi.name}")
        seen_ids.add(roi.id)
        seen_names.add(roi.name)


def validate_roi_reference_compatibility(rois: Sequence[ROI]) -> None:
    """Validate all known coordinate frames in one ROI collection."""
    if not isinstance(rois, Sequence) or isinstance(rois, (str, bytes)):
        raise ROIValidationError("rois must be an ordered sequence")
    validate_rois(rois)
    validate_coordinate_reference_compatibility(
        [roi.coordinate_reference for roi in rois],
        labels=[roi.name for roi in rois],
    )


def _ordered_rois(rois: Sequence[ROI]) -> list[ROI]:
    if not isinstance(rois, Sequence):
        raise ROIValidationError("rois must be an ordered sequence")
    validate_rois(rois)
    return [roi for _, roi in sorted(enumerate(rois), key=lambda pair: (pair[1].priority, pair[0]))]


def _contains(roi: ROI, x: float, y: float) -> bool:
    g = roi.geometry
    inclusive = roi.boundary_mode == "inclusive"
    if roi.shape == "rectangle":
        if inclusive:
            return g["left"] <= x <= g["right"] and g["top"] <= y <= g["bottom"]
        return g["left"] < x < g["right"] and g["top"] < y < g["bottom"]
    if roi.shape == "circle":
        return (x - g["cx"]) ** 2 + (y - g["cy"]) ** 2 <= g["r"] ** 2
    return ((x - g["cx"]) / g["a"]) ** 2 + ((y - g["cy"]) / g["b"]) ** 2 <= 1.0


def _is_missing_coordinate(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    return isinstance(value, numbers.Real) and math.isnan(float(value))


def _bounds(values: Sequence[float], label: str) -> tuple[float, float, float, float]:
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise CoordinateMappingError(f"{label} must be [left, top, right, bottom]")
    left, top, right, bottom = (_number(value, f"{label}[{i}]") for i, value in enumerate(values))
    if right <= left or bottom <= top:
        raise CoordinateMappingError(f"{label} must have positive width and height")
    return left, top, right, bottom


def _pair(values: Sequence[float], label: str) -> tuple[float, float]:
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise CoordinateMappingError(f"{label} must contain two coordinates")
    return _number(values[0], f"{label}[0]"), _number(values[1], f"{label}[1]")


def _dimension(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise CoordinateMappingError(f"{label} must be an integer")
    result = int(value)
    if result <= 1:
        raise CoordinateMappingError(f"{label} must exceed one pixel center")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _format_number(value: float) -> str:
    return str(float(value))
