"""Non-overwriting exports for headless eye-tracking analysis results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .metrics import AnalysisResult
from session_names import canonical_session_id


EXPORT_SCHEMA_VERSION = "1.0.0"


def export_analysis_result(
    result: AnalysisResult,
    output_root: str | Path,
    *,
    session_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    rois=(),
) -> dict[str, Path]:
    """Write one analysis bundle under a new session directory.

    Existing session directories are never reused. The manifest excludes its
    own hash but records hashes for the input and every other output file.
    """

    meta = dict(metadata or getattr(result.samples, "attrs", {}).get("analysis_metadata", {}))
    selected_session = session_id or meta.get("session_id") or _source_stem(meta) or "analysis"
    selected_session = _safe_session_id(str(selected_session))
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / selected_session

    paths = {
        "directory": session_dir,
        "summary": session_dir / f"{selected_session}_summary.xlsx",
        "samples": session_dir / f"{selected_session}_samples.csv",
        "events": session_dir / f"{selected_session}_events.csv",
        "roi_hits": session_dir / f"{selected_session}_roi_hits.csv",
        "manifest": session_dir / f"{selected_session}_manifest.json",
    }
    frames = {
        "samples": _as_frame(result.samples),
        "events": _as_frame(result.events),
        "roi_hits": _as_frame(result.roi_hits),
    }
    summary_frames = _summary_sheets(result, meta, selected_session)
    session_dir.mkdir(exist_ok=False)

    _write_csv_exclusive(paths["samples"], frames["samples"])
    _write_csv_exclusive(paths["events"], frames["events"])
    _write_csv_exclusive(paths["roi_hits"], frames["roi_hits"])
    _write_summary_exclusive(paths["summary"], summary_frames)

    roi_records = []
    for roi in rois or ():
        if hasattr(roi, "to_dict"):
            roi_records.append(roi.to_dict())
        elif isinstance(roi, Mapping):
            roi_records.append(dict(roi))
        else:
            roi_records.append({"name": str(roi), "status": "unserialized"})
    output_hashes = {
        key: _sha256_file(path)
        for key, path in paths.items()
        if key not in {"directory", "manifest"}
    }
    manifest = {
        "schema": "LocalGaze.analysis-export/" + EXPORT_SCHEMA_VERSION,
        "software": {
            "name": "LocalGaze",
            "contract_version": meta.get("data_contract_version"),
            "analysis_method_version": result.methods.get("method_version"),
        },
        "session_id": selected_session,
        "input": {
            "path": meta.get("source_path"),
            "sha256": meta.get("source_sha256"),
            "format": meta.get("kind"),
            "sheet": meta.get("sheet"),
            "manifest_path": meta.get("source_manifest_path"),
            "session_id": meta.get("session_id"),
            "field_mapping": meta.get("field_mapping", {}),
            "source_field_provenance": meta.get("source_field_provenance", {}),
            "model_feature_contract": meta.get("model_feature_contract"),
            "calibration_fit_residuals_px": meta.get("calibration_fit_residuals_px"),
            "analysis_field_provenance": meta.get("analysis_field_provenance", {}),
            "time_source": meta.get("time_source"),
            "clock_quality": meta.get("clock_quality"),
            "roi_source_path": meta.get("roi_source_path"),
            "roi_source_sha256": meta.get("roi_source_sha256"),
            "roi_migration_status": meta.get("roi_migration_status"),
        },
        "coordinate_reference": {
            "status": meta.get("coordinate_reference_status", "unknown"),
            "value": meta.get("coordinate_reference", {"space": "unknown", "source": "unknown"}),
        },
        "analysis": {
            "preset": meta.get("analysis_preset", "raw_measured_full_contract_v1"),
            "config": result.methods.get("config", {}),
            "methods": result.methods,
            "summary": result.summary,
        },
        "rois": roi_records,
        "outputs": {
            key: {"path": str(path), "sha256": output_hashes[key]}
            for key, path in paths.items()
            if key in output_hashes
        },
        "limitations": [
            "Engineering event candidates and sampled ROI crossing times are not validated against independent ground truth.",
            "Missing or unknown clock, validity, screen, stimulus, or physical measurement prerequisites remain unavailable rather than inferred.",
        ],
    }
    _write_json_exclusive(paths["manifest"], manifest)
    return paths


def _summary_sheets(result: AnalysisResult, metadata: Mapping[str, Any], session_id: str) -> dict[str, pd.DataFrame]:
    summary = dict(result.summary)
    session_rows = [{"field": "session_id", "value": session_id}]
    session_rows.extend({"field": key, "value": value} for key, value in summary.items())
    session_rows.extend([
        {"field": "source_path", "value": metadata.get("source_path")},
        {"field": "source_sha256", "value": metadata.get("source_sha256")},
        {"field": "data_contract_version", "value": metadata.get("data_contract_version")},
    ])
    metrics = _as_frame(result.metrics)
    if "metric_id" in metrics:
        qc_prefixes = (
            "input_", "valid_", "invalid_", "break_", "interpolated_",
            "bad_segment_", "status_", "session_", "adjacent_dt_", "clock_",
            "long_gap_", "observed_sampling_rate_", "last_cumulative_",
            "duplicate_image_",
        )
        qc = metrics[metrics["metric_id"].astype(str).str.startswith(qc_prefixes)].copy()
        calibration = metrics[metrics["metric_id"].astype(str).str.startswith("calibration_")].copy()
    else:
        qc = pd.DataFrame()
        calibration = pd.DataFrame()
    if calibration.empty:
        calibration = pd.DataFrame([{
            "metric_id": "calibration_fit_residuals",
            "value": None,
            "unit": "px",
            "status": "not_computable",
            "reason": "training_fit_residuals_not_supplied_in_input_or_analysis_metadata",
            "method": "training fit residuals are not independent accuracy",
            "input_source": metadata.get("source_path"),
        }])
    return {
        "Session": pd.DataFrame(session_rows),
        "QC": qc,
        "Calibration": calibration,
        "Metrics": metrics,
        "ROI": _as_frame(result.roi),
        "Events": _as_frame(result.events),
        "Methods": _flatten_mapping(result.methods),
    }


def _flatten_mapping(values: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        else:
            rows.append({"method": prefix, "value": value})

    visit("", values)
    return pd.DataFrame(rows, columns=["method", "value"])


def _as_frame(value: Any) -> pd.DataFrame:
    return value.copy(deep=True) if isinstance(value, pd.DataFrame) else pd.DataFrame(value)


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        clean = frame.apply(lambda column: column.map(_csv_value))
        clean.to_csv(stream, index=False, lineterminator="\n")


def _write_summary_exclusive(path: Path, sheets: Mapping[str, pd.DataFrame]) -> None:
    with path.open("xb") as stream:
        with pd.ExcelWriter(stream, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                clean = frame.apply(lambda column: column.map(_excel_value))
                clean.to_excel(writer, sheet_name=name[:31], index=False)


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _excel_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _csv_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    try:
        missing = pd.isna(value)
        if np.isscalar(missing) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _source_stem(metadata: Mapping[str, Any]) -> str | None:
    source = metadata.get("source_path")
    return Path(str(source)).stem if source else None


def _safe_session_id(value: str) -> str:
    # Only the new output namespace changes; input IDs remain in metadata.
    return canonical_session_id(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["EXPORT_SCHEMA_VERSION", "export_analysis_result"]
