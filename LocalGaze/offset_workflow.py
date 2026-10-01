"""File and capture adapters for the pure eyetrax offset API."""
from __future__ import annotations

import csv
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import uuid
from dataclasses import dataclass
from typing import Any

from eyetrax.features import (
    FEATURE_FORMULA_V1,
    FEATURE_FORMULA_V2,
    LEGACY_UNVERSIONED_FORMULA,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.offset import (
    FIXATION_ASSUMPTION,
    REFERENCE_STATUS,
    CorrectionApplication,
    OffsetBinding,
    OffsetCorrectionRecord,
    OffsetObservation,
    apply_offset,
    correction_record_from_estimate,
    estimate_offset,
    validate_correction_record,
)


_SUPPORTED_FORMULAS = {
    FEATURE_FORMULA_V1,
    FEATURE_FORMULA_V2,
    LEGACY_UNVERSIONED_FORMULA,
}

_TARGET_DISPLAY_EVENT = "tk_window_visible_after_update"
_TARGET_CAPTURE_CLOCK = "time.monotonic_ns; successful frame-read receipt clock"


def _positive_canvas(value: Any) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        if "width" not in value or "height" not in value:
            return None
        values = (value["width"], value["height"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        values = value
    else:
        return None
    result = []
    for item in values:
        if isinstance(item, bool):
            return None
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            return None
        result.append(int(numeric))
    return tuple(result)


def _binding_dict(binding: OffsetBinding | None) -> dict[str, Any] | None:
    if binding is None:
        return None
    return {
        "model_sha256": binding.model_sha256,
        "feature_formula_version": binding.feature_formula_version,
        "calibration_canvas_size_px": (
            list(binding.calibration_canvas_size_px)
            if binding.calibration_canvas_size_px is not None
            else None
        ),
    }


def _binding_from_dict(value: Any) -> OffsetBinding:
    expected = set(OffsetBinding.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("model_binding_fields_invalid")
    canvas = value.get("calibration_canvas_size_px")
    return OffsetBinding(
        value.get("model_sha256"),
        value.get("feature_formula_version"),
        _positive_canvas(canvas) if canvas is not None else None,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_binding_for_estimator(model_path: str | Path, estimator: Any) -> OffsetBinding:
    """Bind an already loaded model to its recorded formula and calibration canvas."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".pkl":
        raise ValueError("model_path_must_be_existing_pkl")
    model = getattr(estimator, "model", None)
    metadata = getattr(model, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    formula = getattr(estimator, "feature_formula_version", None)
    if formula is None:
        formula = metadata.get("feature_formula_version") or LEGACY_UNVERSIONED_FORMULA
    if formula not in _SUPPORTED_FORMULAS:
        raise ValueError("model_feature_formula_unavailable")
    canvas_value = metadata.get(
        "calibration_canvas_size_px",
        metadata.get("screen_info_px", getattr(estimator, "screen_info", None)),
    )
    canvas = _positive_canvas(canvas_value) if canvas_value is not None else None
    return OffsetBinding(_sha256_file(path), formula, canvas)


def load_model_binding(model_path: str | Path) -> OffsetBinding:
    """Load and validate model metadata without opening a camera or FaceMesh."""
    estimator = GazeEstimator(initialize_face_mesh=False)
    estimator.load_model(Path(model_path).expanduser().resolve())
    return model_binding_for_estimator(model_path, estimator)


@dataclass(frozen=True)
class OffsetCaptureContext:
    record: OffsetCorrectionRecord | None
    active_binding: OffsetBinding | None
    status: str
    reason: str | None = None
    correction_path: str | None = None
    correction_sha256: str | None = None

    @property
    def correction_id(self) -> str | None:
        return self.record.correction_id if self.record is not None else None

    def manifest_metadata(self) -> dict[str, Any]:
        record = self.record
        return {
            "enabled": self.correction_path is not None,
            "status": self.status,
            "reason": self.reason,
            "correction_path": self.correction_path,
            "correction_sha256": self.correction_sha256,
            "correction_id": self.correction_id,
            "active_binding": _binding_dict(self.active_binding),
            "recorded_binding": _binding_dict(record.binding) if record else None,
            "fixation_assumption": record.fixation_assumption if record else FIXATION_ASSUMPTION,
            "coordinate_reference_status": record.coordinate_reference_status if record else REFERENCE_STATUS,
            "participant_fixation_independently_verified": False,
            "physical_dpi_verified": False,
        }


def load_capture_correction(
    correction_path: str | Path | None,
    active_binding: OffsetBinding,
    current_canvas_size_px: Any,
) -> OffsetCaptureContext:
    """Load correction only for an explicit path and reject unsafe bindings."""
    if correction_path is None:
        return OffsetCaptureContext(None, active_binding, "disabled")

    path = Path(correction_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        record = validate_correction_record(json.loads(raw.decode("utf-8-sig")))
    except FileNotFoundError:
        return OffsetCaptureContext(None, active_binding, "correction_file_missing", correction_path=str(path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return OffsetCaptureContext(
            None, active_binding, "correction_record_invalid",
            reason=f"{type(exc).__name__}: {exc}", correction_path=str(path),
        )

    digest = hashlib.sha256(raw).hexdigest()
    current_canvas = _positive_canvas(current_canvas_size_px)
    if active_binding.calibration_canvas_size_px is None:
        return OffsetCaptureContext(
            record, active_binding, "model_canvas_unavailable",
            correction_path=str(path), correction_sha256=digest,
        )
    if current_canvas is None:
        return OffsetCaptureContext(
            record, active_binding, "capture_canvas_unavailable",
            correction_path=str(path), correction_sha256=digest,
        )
    if current_canvas != active_binding.calibration_canvas_size_px:
        return OffsetCaptureContext(
            record, active_binding, "capture_canvas_mismatch",
            correction_path=str(path), correction_sha256=digest,
        )
    if record.binding != active_binding:
        return OffsetCaptureContext(
            record, active_binding, "binding_mismatch",
            correction_path=str(path), correction_sha256=digest,
        )
    return OffsetCaptureContext(
        record, active_binding, "ready",
        correction_path=str(path), correction_sha256=digest,
    )


def apply_capture_correction(raw_xy_px: Any, context: OffsetCaptureContext | None) -> CorrectionApplication:
    if context is None or context.status == "disabled":
        return CorrectionApplication(None, False, "disabled", None)
    if context.status != "ready" or context.record is None or context.active_binding is None:
        return CorrectionApplication(None, False, context.status, context.correction_id)
    return apply_offset(raw_xy_px, context.record, context.active_binding)


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip().casefold() in {"", "none", "nan", "null"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool_field(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _read_capture_observations(path: Path) -> list[OffsetObservation]:
    required = {
        "X", "Y", "frame_id", "received_monotonic_ns", "sample_status", "sample_accepted",
    }
    observations: list[OffsetObservation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("capture_columns_missing_for_offset")
        for row in reader:
            try:
                receipt_ns = int(row.get("received_monotonic_ns", ""))
            except (TypeError, ValueError, OverflowError):
                receipt_ns = -1
            status = str(row.get("sample_status", "")).strip()
            accepted = _bool_field(row.get("sample_accepted", "")) and status == "accepted"
            x_value = _float_or_none(row.get("X"))
            y_value = _float_or_none(row.get("Y"))
            raw_xy = (x_value, y_value) if x_value is not None and y_value is not None else None
            observations.append(OffsetObservation(
                received_monotonic_ns=receipt_ns,
                frame_id=str(row.get("frame_id", "")),
                raw_xy_px=raw_xy,
                accepted=accepted,
                eye_state="open" if accepted else status or "invalid",
            ))
    return observations


def create_correction_from_target_capture(
    *,
    capture_path: str | Path,
    model_path: str | Path,
    correction_path: str | Path,
) -> dict[str, Any]:
    """Estimate only from a completed known-target job's own capture and manifest."""
    source = Path(capture_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    destination = Path(correction_path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".txt":
        raise ValueError("capture_path_must_be_existing_capture_txt")
    if not destination.name or destination.suffix.lower() != ".json":
        raise ValueError("correction_path_must_end_in_json")

    binding = load_model_binding(model)
    if binding.calibration_canvas_size_px is None:
        return {
            "status": "failed", "message": "model calibration canvas is unavailable",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "model_canvas_unavailable", "correction_json_written": False},
        }

    manifest_path = source.with_suffix(".manifest.json")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw.decode("utf-8-sig"))
        captured_binding = _binding_from_dict(manifest.get("model_binding"))
        capture_metadata = manifest.get("target_capture")
        files = manifest.get("files")
        if (
            manifest.get("workflow_mode") != "known_target_offset"
            or not isinstance(capture_metadata, dict)
            or not isinstance(files, dict)
            or files.get("txt") != source.name
            or manifest.get("session_id") != source.stem
        ):
            raise ValueError("capture_not_owned_known_target_job")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        return {
            "status": "failed", "message": "known-target capture manifest is unavailable or invalid",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "capture_manifest_invalid", "binding_error": f"{type(exc).__name__}: {exc}", "correction_json_written": False},
        }
    if captured_binding != binding:
        return {
            "status": "failed", "message": "capture and selected model bindings differ",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "binding_mismatch", "correction_json_written": False},
        }

    if manifest.get("feature_formula_version") != binding.feature_formula_version:
        return {
            "status": "failed", "message": "known-target capture formula metadata is incomplete or inconsistent",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "capture_formula_mismatch", "correction_json_written": False},
        }

    current_canvas = _positive_canvas(capture_metadata.get("current_canvas_size_px"))
    if current_canvas is None or current_canvas != binding.calibration_canvas_size_px:
        return {
            "status": "failed", "message": "known-target capture canvas metadata is unavailable or inconsistent",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "capture_canvas_mismatch", "correction_json_written": False},
        }

    if (
        capture_metadata.get("target_display_event") != _TARGET_DISPLAY_EVENT
        or capture_metadata.get("clock") != _TARGET_CAPTURE_CLOCK
    ):
        return {
            "status": "failed", "message": "known-target display event or shared monotonic clock contract is missing",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "target_display_event_unverified", "correction_json_written": False},
        }

    violations = capture_metadata.get("context_violations")
    if capture_metadata.get("context_valid") is not True or violations != []:
        return {
            "status": "failed", "message": "known-target model or canvas context changed during capture",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "target_context_changed", "context_violations": violations, "correction_json_written": False},
        }

    if capture_metadata.get("stop_reason") != "target_capture_window_complete":
        reason = capture_metadata.get("stop_reason") or "target_capture_incomplete"
        return {
            "status": "failed", "message": "known-target capture did not finish its timed window",
            "outputs": {},
            "details": {
                "offset_status": "rejected", "offset_reason": reason,
                "target_display_monotonic_ns": capture_metadata.get("target_display_monotonic_ns"),
                "correction_json_written": False,
                "participant_fixation_independently_verified": False,
                "physical_dpi_verified": False,
            },
        }

    target_xy_px = capture_metadata.get("target_xy_px")
    target_display_monotonic_ns = capture_metadata.get("target_display_monotonic_ns")
    if (
        isinstance(target_display_monotonic_ns, bool)
        or not isinstance(target_display_monotonic_ns, int)
        or target_display_monotonic_ns < 0
    ):
        return {
            "status": "failed", "message": "known-target display event timestamp is invalid",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "invalid_target_display_time", "correction_json_written": False},
        }

    observations = _read_capture_observations(source)
    estimate = estimate_offset(
        observations,
        target_xy_px,
        binding,
        target_display_monotonic_ns=target_display_monotonic_ns,
    )
    if estimate.status != "eligible":
        return {
            "status": "failed", "message": estimate.reason or "offset estimate rejected",
            "outputs": {},
            "details": {
                "offset_status": estimate.status,
                "offset_reason": estimate.reason,
                "sample_count": estimate.sample_count,
                "span_ms": estimate.span_ms,
                "target_xy_px": estimate.target_xy_px,
                "target_display_monotonic_ns": target_display_monotonic_ns,
                "binding": _binding_dict(binding),
                "correction_json_written": False,
                "participant_fixation_independently_verified": False,
                "physical_dpi_verified": False,
            },
        }

    record = correction_record_from_estimate(estimate, uuid.uuid4().hex)
    serialized = json.dumps(record.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("x", encoding="utf-8", newline="") as stream:
            created = True
            stream.write(serialized)
            stream.flush()
    except Exception:
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise

    return {
        "status": "completed",
        "message": "eligible correction record created",
        "outputs": {"correction": str(destination)},
        "details": {
            "offset_status": estimate.status,
            "offset_reason": estimate.reason,
            "correction_id": record.correction_id,
            "source_path": str(source),
            "source_sha256": _sha256_file(source),
            "capture_manifest_path": str(manifest_path),
            "capture_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "target_xy_px": list(record.target_xy_px),
            "target_display_monotonic_ns": target_display_monotonic_ns,
            "sample_count": record.sample_count,
            "span_ms": record.span_ms,
            "binding": _binding_dict(record.binding),
            "fixation_assumption": record.fixation_assumption,
            "coordinate_reference_status": record.coordinate_reference_status,
            "participant_fixation_independently_verified": False,
            "physical_dpi_verified": False,
        },
    }


__all__ = [
    "OffsetCaptureContext",
    "apply_capture_correction",
    "create_correction_from_target_capture",
    "load_capture_correction",
    "load_model_binding",
    "model_binding_for_estimator",
]
