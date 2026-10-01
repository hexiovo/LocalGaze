"""Small, explicit contracts for captured frames and derived eye features.

This module is independent of camera/UI code so its timing and geometry rules
can be checked with synthetic coordinates. Iris measurements here are proxies;
they do not measure the pupil.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot, isfinite
from typing import Any, Hashable, Mapping, Sequence


DATA_CONTRACT_VERSION = "1.0.0"
IRIS_PROXY_FORMULA_VERSION = "iris-proxy-2d-v1"
NANOSECONDS_PER_SECOND = 1_000_000_000


class FieldSource(str, Enum):
    """Whether a field came directly from capture or was computed later."""

    RAW = "raw"
    DERIVED = "derived"


@dataclass(frozen=True)
class ContractField:
    """A field value with its provenance recorded alongside it."""

    value: Any
    source: FieldSource
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source is FieldSource.DERIVED and not self.derived_from:
            raise ValueError("derived fields must name their source fields")
        if self.source is FieldSource.RAW and self.derived_from:
            raise ValueError("raw fields cannot declare derived_from fields")


def raw_field(value: Any) -> ContractField:
    """Mark a value that was captured or supplied by the source unchanged."""

    return ContractField(value=value, source=FieldSource.RAW)


def derived_field(value: Any, *derived_from: str) -> ContractField:
    """Mark a computed value and list the field names it depends on."""

    return ContractField(
        value=value,
        source=FieldSource.DERIVED,
        derived_from=tuple(derived_from),
    )


class SampleStatus(str, Enum):
    ACCEPTED = "accepted"
    NO_FACE = "no_face"
    BLINK = "blink"
    WARMUP = "warmup"
    INVALID = "invalid"
    DUPLICATE_FRAME_ID = "duplicate_frame_id"
    PROCESSING_ERROR = "processing_error"


@dataclass(frozen=True)
class FramePacket:
    """Metadata and provenance for one camera read attempt."""

    frame_id: Hashable
    received_monotonic_ns: int
    elapsed_s: float
    status: SampleStatus
    fields: Mapping[str, ContractField] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_id is None:
            raise ValueError("frame_id must not be None")
        try:
            hash(self.frame_id)
        except TypeError as exc:
            raise TypeError("frame_id must be hashable") from exc
        if isinstance(self.received_monotonic_ns, bool) or not isinstance(
            self.received_monotonic_ns, int
        ):
            raise TypeError("received_monotonic_ns must be an integer")
        if self.received_monotonic_ns < 0:
            raise ValueError("received_monotonic_ns must be nonnegative")
        if not isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        if not isinstance(self.status, SampleStatus):
            raise TypeError("status must be a SampleStatus")
        if any(not isinstance(value, ContractField) for value in self.fields.values()):
            raise TypeError("all frame fields must be ContractField values")

    @property
    def sample_accepted(self) -> bool:
        """False for repeated frame IDs and reads with no detected face."""

        return self.status is SampleStatus.ACCEPTED


class FrameReceptionTracker:
    """Assign nonnegative session-relative times and reject repeated IDs."""

    def __init__(self) -> None:
        self._seen_frame_ids: set[Hashable] = set()
        self._session_start_ns: int | None = None
        self._last_elapsed_s = 0.0

    def receive(
        self,
        frame_id: Hashable,
        received_monotonic_ns: int,
        *,
        has_face: bool = True,
        fields: Mapping[str, ContractField] | None = None,
    ) -> FramePacket:
        """Record a successful read; repeated IDs are retained but not sampled."""

        if frame_id is None:
            raise ValueError("frame_id must not be None")
        try:
            hash(frame_id)
        except TypeError as exc:
            raise TypeError("frame_id must be hashable") from exc
        if isinstance(received_monotonic_ns, bool) or not isinstance(
            received_monotonic_ns, int
        ):
            raise TypeError("received_monotonic_ns must be an integer")
        if received_monotonic_ns < 0:
            raise ValueError("received_monotonic_ns must be nonnegative")
        packet_fields = dict(fields or {})
        if any(not isinstance(value, ContractField) for value in packet_fields.values()):
            raise TypeError("all frame fields must be ContractField values")

        if self._session_start_ns is None:
            self._session_start_ns = received_monotonic_ns
        candidate_elapsed_s = max(
            0.0,
            (received_monotonic_ns - self._session_start_ns)
            / NANOSECONDS_PER_SECOND,
        )
        elapsed_s = max(self._last_elapsed_s, candidate_elapsed_s)
        self._last_elapsed_s = elapsed_s

        duplicate = frame_id in self._seen_frame_ids
        self._seen_frame_ids.add(frame_id)
        if duplicate:
            status = SampleStatus.DUPLICATE_FRAME_ID
        elif not has_face:
            status = SampleStatus.NO_FACE
        else:
            status = SampleStatus.ACCEPTED

        return FramePacket(
            frame_id=frame_id,
            received_monotonic_ns=received_monotonic_ns,
            elapsed_s=elapsed_s,
            status=status,
            fields=packet_fields,
        )


@dataclass(frozen=True)
class IrisProxyMetrics:
    """2-D iris-ring size proxies normalized by the outer eye-corner span.

    These values describe the visible iris landmark ring. They are not pupil
    diameters and must not be named or interpreted as pupil measurements.
    """

    left_iris_diameter_over_eye_span: float
    right_iris_diameter_over_eye_span: float
    eye_span_px: float
    left_center_px: tuple[float, float]
    right_center_px: tuple[float, float]
    formula_version: str = IRIS_PROXY_FORMULA_VERSION


def _xy(point: Any) -> tuple[float, float] | None:
    try:
        if hasattr(point, "x") and hasattr(point, "y"):
            x, y = point.x, point.y
        else:
            x, y = point[0], point[1]
        x_value, y_value = float(x), float(y)
    except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError):
        return None
    if not isfinite(x_value) or not isfinite(y_value):
        return None
    return x_value, y_value


def compute_iris_proxy(
    landmarks: Sequence[Any] | Mapping[int, Any],
    image_width: int | float,
    image_height: int | float,
) -> IrisProxyMetrics | None:
    """Compute visible iris-ring diameter proxies from normalized landmarks.

    The refined 478-point layout uses 473/468 as the left/right centers and
    474-477 / 469-472 as their four ring points. Each normalized coordinate is
    converted using its corresponding image dimension before measuring. The
    denominator is the 2-D pixel distance between outer eye corners 33 and
    263. Returns ``None`` for missing, degenerate, or non-finite geometry.
    """

    try:
        width, height = float(image_width), float(image_height)
    except (OverflowError, TypeError, ValueError):
        return None
    if not isfinite(width) or not isfinite(height) or width <= 0 or height <= 0:
        return None

    def get_xy(index: int) -> tuple[float, float] | None:
        try:
            normalized = _xy(landmarks[index])
        except (IndexError, KeyError, TypeError):
            return None
        if normalized is None:
            return None
        return normalized[0] * width, normalized[1] * height

    left_corner = get_xy(33)
    right_corner = get_xy(263)
    left_center = get_xy(473)
    right_center = get_xy(468)
    if any(point is None for point in (left_corner, right_corner, left_center, right_center)):
        return None

    eye_span_px = hypot(
        right_corner[0] - left_corner[0], right_corner[1] - left_corner[1]
    )
    if not isfinite(eye_span_px) or eye_span_px <= 0:
        return None

    def normalized_diameter(center: tuple[float, float], indices: range) -> float | None:
        radii: list[float] = []
        for index in indices:
            edge = get_xy(index)
            if edge is None:
                return None
            radius = hypot(edge[0] - center[0], edge[1] - center[1])
            if not isfinite(radius):
                return None
            radii.append(radius)
        value = 2.0 * (sum(radii) / len(radii)) / eye_span_px
        return value if isfinite(value) else None

    left_value = normalized_diameter(left_center, range(474, 478))
    right_value = normalized_diameter(right_center, range(469, 473))
    if left_value is None or right_value is None:
        return None

    return IrisProxyMetrics(
        left_iris_diameter_over_eye_span=left_value,
        right_iris_diameter_over_eye_span=right_value,
        eye_span_px=eye_span_px,
        left_center_px=left_center,
        right_center_px=right_center,
    )


__all__ = [
    "DATA_CONTRACT_VERSION",
    "IRIS_PROXY_FORMULA_VERSION",
    "ContractField",
    "FieldSource",
    "FramePacket",
    "FrameReceptionTracker",
    "IrisProxyMetrics",
    "SampleStatus",
    "compute_iris_proxy",
    "derived_field",
    "raw_field",
]
