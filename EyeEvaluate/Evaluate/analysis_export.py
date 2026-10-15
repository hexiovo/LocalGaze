"""Non-overwriting exports for headless eye-tracking analysis results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .metrics import AnalysisResult
from .analysis_options import normalize_selected_metrics, normalize_table_format
from session_names import canonical_session_id


EXPORT_SCHEMA_VERSION = "1.1.0"
_MARKER_TABLE_COLUMNS = (
    "schema", "marker_id", "session_id", "capture_id", "attempt_id",
    "marker_sequence", "literal_key", "label", "monotonic_ns", "elapsed_s",
    "timestamp_local", "source", "last_monotonic_ns",
    "user_marker_last_monotonic_ns", "association_sample_index",
    "association_status", "association_reason", "association_method",
    "source_data_path", "source_data_sha256", "source_manifest_path",
    "source_manifest_sha256", "source_marker_log_path",
    "source_capture_manifest_path", "source_capture_manifest_sha256",
    "source_capture_data_path", "source_capture_data_sha256",
    "source_marker_events_sha256", "source_marker_stream_sha256",
    "source_workbook_path", "source_workbook_sha256",
)


def export_analysis_result(
    result: AnalysisResult,
    output_root: str | Path,
    *,
    session_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    rois=(),
    table_format: str = "csv",
) -> dict[str, Path]:
    """Write one analysis bundle under a new session directory.

    Existing session directories are never reused. The manifest excludes its
    own hash but records hashes for the input and every other output file.
    """

    meta = dict(metadata or getattr(result.samples, "attrs", {}).get("analysis_metadata", {}))
    selected_session = session_id or meta.get("session_id") or _source_stem(meta) or "analysis"
    selected_session = _safe_session_id(str(selected_session))
    selected_format = normalize_table_format(table_format)
    delimiter = "," if selected_format == "csv" else "\t"
    selected_groups = normalize_selected_metrics(
        result.methods.get("selected_metric_groups")
        if isinstance(result.methods, Mapping) and "selected_metric_groups" in result.methods
        else None
    )
    has_events = bool(set(selected_groups).intersection({"fixation", "fast_movement", "ivt_saccade", "blink"}))
    has_roi = "roi" in selected_groups
    marker_details = _marker_details(meta, _as_frame(result.samples))
    marker_frame = _marker_frame(marker_details)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / selected_session

    internal_dir = session_dir / ".internal"
    paths = {
        "directory": session_dir,
        "summary": session_dir / f"{selected_session}_summary.xlsx",
        "metadata": session_dir / f"{selected_session}_metadata.txt",
        "metrics": session_dir / f"{selected_session}_metrics.{selected_format}",
        "samples": session_dir / f"{selected_session}_samples.{selected_format}",
        "markers": session_dir / f"{selected_session}_markers.{selected_format}",
        "manifest": internal_dir / f"{selected_session}_manifest.json",
    }
    if has_events:
        paths["events"] = session_dir / f"{selected_session}_events.{selected_format}"
    if has_roi:
        paths["roi_hits"] = session_dir / f"{selected_session}_roi_hits.{selected_format}"
    frames = {
        "metrics": _as_frame(result.metrics),
        "samples": _as_frame(result.samples),
        "markers": marker_frame,
    }
    if has_events:
        frames["events"] = _as_frame(result.events)
    if has_roi:
        frames["roi_hits"] = _as_frame(result.roi_hits)
    summary_frames = _summary_sheets(
        result, meta, selected_session, selected_groups, marker_details, marker_frame
    )
    session_dir.mkdir(exist_ok=False)
    internal_dir.mkdir()

    for key in frames:
        _write_table_exclusive(paths[key], frames[key], delimiter=delimiter)
    _write_summary_exclusive(paths["summary"], summary_frames)
    _write_metadata_exclusive(
        paths["metadata"], result, meta, selected_session, selected_format, marker_details
    )

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
            "manifest_sha256": meta.get("source_manifest_sha256"),
            "session_id": meta.get("session_id"),
            "capture_id": marker_details["capture_id"],
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
            "table_format": selected_format,
            "config": result.methods.get("config", {}),
            "methods": result.methods,
            "summary": result.summary,
        },
        "rois": roi_records,
        "markers": {
            "schema": "LocalGaze.analysis-markers/1",
            "status": marker_details["status"],
            "count": marker_details["count"],
            "associated_sample_count": marker_details["associated_sample_count"],
            "unassociated_count": marker_details["unassociated_count"],
            "association_method": marker_details["association_method"],
            "association_policy": "exact monotonic-nanosecond timestamp equality only; no nearest-sample inference",
            "last_monotonic_ns": marker_details["last_monotonic_ns"],
            "user_marker_last_monotonic_ns": marker_details["user_marker_last_monotonic_ns"],
            "source": marker_details["source"],
            "source_monotonic_fields": marker_details["source_monotonic_fields"],
            "source_marker_log": marker_details["marker_log"],
            "records": marker_details["records"],
        },
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


def _summary_sheets(
    result: AnalysisResult,
    metadata: Mapping[str, Any],
    session_id: str,
    selected_groups: tuple[str, ...],
    marker_details: Mapping[str, Any],
    marker_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    summary = dict(result.summary)
    session_rows = [{"field": "session_id", "value": session_id}]
    session_rows.extend({"field": key, "value": value} for key, value in summary.items())
    session_rows.extend([
        {"field": "source_path", "value": metadata.get("source_path")},
        {"field": "source_sha256", "value": metadata.get("source_sha256")},
        {"field": "data_contract_version", "value": metadata.get("data_contract_version")},
        {"field": "source_marker_count", "value": marker_details["count"]},
        {"field": "marker_associated_sample_count", "value": marker_details["associated_sample_count"]},
        {"field": "marker_unassociated_count", "value": marker_details["unassociated_count"]},
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
    if "calibration" in selected_groups and calibration.empty:
        calibration = pd.DataFrame([{
            "metric_id": "calibration_fit_residuals",
            "value": None,
            "unit": "px",
            "status": "not_computable",
            "reason": "training_fit_residuals_not_supplied_in_input_or_analysis_metadata",
            "method": "training fit residuals are not independent accuracy",
            "input_source": metadata.get("source_path"),
        }])
    sheets = {"Session": pd.DataFrame(session_rows), "QC": qc}
    if "calibration" in selected_groups:
        sheets["Calibration"] = calibration
    sheets["Metrics"] = metrics
    if "roi" in selected_groups:
        sheets["ROI"] = _as_frame(result.roi)
    if set(selected_groups).intersection({"fixation", "fast_movement", "ivt_saccade", "blink"}):
        sheets["Events"] = _as_frame(result.events)
    sheets["UserMarkers"] = marker_frame
    sheets["MarkerSummary"] = _marker_summary_frame(marker_details)
    sheets["Methods"] = _flatten_mapping(result.methods)
    return sheets


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


def _marker_details(metadata: Mapping[str, Any], samples: pd.DataFrame) -> dict[str, Any]:
    source_manifest = metadata.get("source_capture_manifest")
    containers = list(_walk_mappings(source_manifest))
    candidates = [item for item in containers if isinstance(item.get("user_markers"), list)]
    marker_lists: list[list[dict[str, Any]]] = []
    for item in candidates:
        values = item["user_markers"]
        if any(not isinstance(marker, Mapping) for marker in values):
            raise ValueError("source capture manifest contains a non-object user marker")
        marker_lists.append([dict(marker) for marker in values])

    records_by_id: dict[str, dict[str, Any]] = {}
    for marker_list in marker_lists:
        for marker in marker_list:
            marker_id = marker.get("marker_id")
            identity = str(marker_id) if marker_id is not None else json.dumps(
                _json_nanosecond_safe(marker), ensure_ascii=False, sort_keys=True
            )
            previous = records_by_id.get(identity)
            if previous is not None and _json_nanosecond_safe(previous) != _json_nanosecond_safe(marker):
                raise ValueError(f"source manifests disagree about user marker {identity}")
            records_by_id[identity] = marker
    marker_records = list(records_by_id.values())

    logs: list[dict[str, Any]] = []
    for item in containers:
        direct_log = item.get("user_marker_log")
        persistence = item.get("capture_persistence")
        nested_log = persistence.get("user_marker_log") if isinstance(persistence, Mapping) else None
        for value in (direct_log, nested_log):
            if not isinstance(value, Mapping):
                continue
            log = dict(value)
            identity = (str(log.get("path") or ""), str(log.get("events_sha256") or ""))
            if not any((str(old.get("path") or ""), str(old.get("events_sha256") or "")) == identity for old in logs):
                logs.append(log)
    marker_log = logs[0] if logs else {}
    expected_count = marker_log.get("count")
    if isinstance(expected_count, bool):
        expected_count = None
    if isinstance(expected_count, int) and expected_count != len(marker_records):
        raise ValueError(
            f"source user marker count {expected_count} does not match the {len(marker_records)} preserved marker records"
        )
    if isinstance(expected_count, int) and expected_count > 0 and not marker_records:
        raise ValueError("source marker log declares events but contains no marker records")

    source = _marker_source_references(metadata, source_manifest, marker_log)
    capture_ids = {str(item.get("capture_id")) for item in marker_records if item.get("capture_id") is not None}
    capture_id = next(iter(capture_ids)) if len(capture_ids) == 1 else None
    if len(capture_ids) > 1:
        raise ValueError("one analysis input contains marker records from multiple captures")

    log_last_ns = _exact_nanosecond(marker_log.get("last_monotonic_ns"), "last_monotonic_ns")
    if log_last_ns is None and marker_records:
        log_last_ns = _exact_nanosecond(marker_records[-1].get("monotonic_ns"), "last_monotonic_ns")
    user_last_ns = _find_nanosecond_field(source_manifest, "user_marker_last_monotonic_ns") or log_last_ns
    source_ns_fields = _collect_monotonic_fields(source_manifest)
    if log_last_ns is not None:
        source_ns_fields.setdefault("user_marker_log.last_monotonic_ns", log_last_ns)
    if user_last_ns is not None:
        source_ns_fields.setdefault("user_marker_last_monotonic_ns", user_last_ns)

    association_method = "exact_received_monotonic_ns_equality"
    records: list[dict[str, Any]] = []
    for marker in marker_records:
        row = _json_nanosecond_safe(marker)
        row.update({
            "last_monotonic_ns": log_last_ns,
            "user_marker_last_monotonic_ns": user_last_ns,
            "source_data_path": source["analysis_input"]["path"],
            "source_data_sha256": source["analysis_input"]["sha256"],
            "source_manifest_path": source["source_manifest"]["path"],
            "source_manifest_sha256": source["source_manifest"]["sha256"],
            "source_capture_manifest_path": source["capture_manifest"]["path"],
            "source_capture_manifest_sha256": source["capture_manifest"]["sha256"],
            "source_capture_data_path": source["capture_source_data"]["path"],
            "source_capture_data_sha256": source["capture_source_data"]["sha256"],
            "source_marker_log_path": source["marker_event_log"]["path"],
            "source_marker_events_sha256": source["marker_event_log"]["sha256"],
            "source_marker_stream_sha256": source["marker_event_log"]["stream_sha256"],
            "source_workbook_path": source["capture_workbook"]["path"],
            "source_workbook_sha256": source["capture_workbook"]["sha256"],
        })
        row.update(_associate_marker(row, samples, metadata, association_method))
        records.append(row)

    associated_count = sum(row.get("association_status") == "associated" for row in records)
    return {
        "status": "available" if marker_lists or logs else "not_present",
        "count": len(records),
        "associated_sample_count": associated_count,
        "unassociated_count": len(records) - associated_count,
        "association_method": association_method,
        "last_monotonic_ns": log_last_ns,
        "user_marker_last_monotonic_ns": user_last_ns,
        "capture_id": capture_id,
        "records": records,
        "source": source,
        "source_monotonic_fields": source_ns_fields,
        "marker_log": _json_nanosecond_safe(marker_log),
    }


def _walk_mappings(value: Any, *, depth: int = 0):
    if depth > 16 or not isinstance(value, Mapping):
        return
    yield value
    for child in value.values():
        if isinstance(child, Mapping):
            yield from _walk_mappings(child, depth=depth + 1)
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, Mapping):
                    yield from _walk_mappings(item, depth=depth + 1)


def _marker_source_references(
    metadata: Mapping[str, Any], source_manifest: Any, marker_log: Mapping[str, Any]
) -> dict[str, Any]:
    input_path = str(metadata.get("source_path")) if metadata.get("source_path") else None
    input_hash = metadata.get("source_sha256")
    manifest_path = str(metadata.get("source_manifest_path")) if metadata.get("source_manifest_path") else None
    manifest_hash = metadata.get("source_manifest_sha256")
    roots: list[Path] = []
    for value in (input_path, manifest_path):
        if value:
            path = Path(value).expanduser()
            root = path.parent.parent if path.parent.name.casefold() == ".internal" else path.parent
            roots.append(root)
    capture_manifest_path = None
    capture_manifest_hash = None
    for item in _walk_mappings(source_manifest):
        candidate_path = item.get("source_manifest_path")
        candidate_hash = item.get("source_manifest_sha256")
        if isinstance(candidate_path, str) and candidate_path:
            capture_manifest_path = candidate_path
            capture_manifest_hash = candidate_hash if isinstance(candidate_hash, str) else None
            break
    if capture_manifest_path is None:
        capture_manifest_path = manifest_path
        capture_manifest_hash = manifest_hash
    if capture_manifest_path:
        candidate = Path(capture_manifest_path).expanduser()
        if candidate.is_absolute():
            roots.append(candidate.parent)
        if candidate.is_file() and not candidate.is_symlink():
            computed_hash = _sha256_file(candidate)
            if capture_manifest_hash is not None and capture_manifest_hash != computed_hash:
                raise ValueError("source capture manifest hash does not match its file")
            capture_manifest_hash = computed_hash
    for item in _walk_mappings(source_manifest):
        for key in ("output_root", "output_dir"):
            value = item.get(key)
            if isinstance(value, str) and value:
                candidate = Path(value).expanduser()
                if candidate.is_absolute():
                    roots.append(candidate)
        for key in ("source_data_file", "raw_path", "source_path"):
            value = item.get(key)
            if isinstance(value, str) and value:
                candidate = Path(value).expanduser()
                if candidate.is_absolute():
                    roots.append(candidate.parent)
    deduped_roots = []
    seen_roots = set()
    for root in roots:
        resolved = root.resolve(strict=False)
        if str(resolved).casefold() not in seen_roots:
            seen_roots.add(str(resolved).casefold())
            deduped_roots.append(resolved)

    workbook_reference = None
    for item in _walk_mappings(source_manifest):
        files = item.get("files")
        if isinstance(files, Mapping) and isinstance(files.get("xlsx"), str):
            workbook_reference = files["xlsx"]
            break
    workbook_path = input_path if input_path and Path(input_path).suffix.casefold() in {".xlsx", ".xlsm"} else None
    workbook_sha = input_hash if workbook_path == input_path else None
    if workbook_path is None and workbook_reference:
        workbook_path, computed_hash = _resolve_source_reference(workbook_reference, deduped_roots)
        workbook_sha = computed_hash or _find_file_hash(source_manifest, "xlsx", workbook_reference)

    marker_log_reference = marker_log.get("path")
    marker_log_path = None
    if isinstance(marker_log_reference, str) and marker_log_reference:
        marker_log_path, resolved_hash = _resolve_source_reference(marker_log_reference, deduped_roots)
    else:
        resolved_hash = None
    marker_log_hash = marker_log.get("events_sha256") or resolved_hash
    if marker_log_path and resolved_hash and marker_log.get("events_sha256") not in (None, resolved_hash):
        raise ValueError("source marker event-log hash does not match its file")

    original_data = _find_source_data_reference(source_manifest)
    original_data_path = original_data.get("path")
    original_data_hash = original_data.get("sha256")
    if isinstance(original_data_path, str) and original_data_path:
        resolved_data, computed_data_hash = _resolve_source_reference(original_data_path, deduped_roots)
        original_data["path"] = resolved_data
        if computed_data_hash:
            if original_data_hash is not None and original_data_hash != computed_data_hash:
                raise ValueError("source capture data hash does not match its file")
            original_data["sha256"] = computed_data_hash
    return {
        "analysis_input": {"path": input_path, "sha256": input_hash},
        "source_manifest": {"path": manifest_path, "sha256": manifest_hash},
        "capture_manifest": {"path": capture_manifest_path, "sha256": capture_manifest_hash},
        "capture_source_data": original_data,
        "marker_event_log": {
            "path": marker_log_path or marker_log_reference,
            "sha256": resolved_hash or marker_log_hash,
            "stream_sha256": marker_log.get("markers_sha256"),
            "size_bytes": marker_log.get("events_size_bytes"),
        },
        "capture_workbook": {
            "path": workbook_path or workbook_reference,
            "sha256": workbook_sha,
        },
    }


def _resolve_source_reference(reference: str, roots: list[Path]) -> tuple[str, str | None]:
    path = Path(reference).expanduser()
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        try:
            if resolved.is_file() and not resolved.is_symlink():
                return str(resolved), _sha256_file(resolved)
        except OSError:
            continue
    return reference, None


def _find_file_hash(value: Any, file_key: str, reference: str) -> str | None:
    for item in _walk_mappings(value):
        candidate = item.get(file_key)
        if isinstance(candidate, Mapping) and candidate.get("sha256"):
            path = candidate.get("path") or candidate.get("relative_path")
            if path is None or str(path) == reference:
                return str(candidate["sha256"])
    return None


def _find_source_data_reference(value: Any) -> dict[str, Any]:
    for item in _walk_mappings(value):
        path = item.get("source_data_file") or item.get("raw_path")
        if path:
            return {"path": str(path), "sha256": item.get("source_sha256") or item.get("recovered_raw_sha256")}
    return {"path": None, "sha256": None}


def _associate_marker(
    marker: Mapping[str, Any], samples: pd.DataFrame, metadata: Mapping[str, Any], method: str
) -> dict[str, Any]:
    marker_time = _exact_nanosecond(marker.get("monotonic_ns"), "marker.monotonic_ns")
    mapping = metadata.get("field_mapping")
    mapping = mapping if isinstance(mapping, Mapping) else {}
    time_column = mapping.get("received_monotonic_ns")
    if not isinstance(time_column, str) or time_column not in samples.columns:
        time_column = next((name for name in ("received_monotonic_ns", "monotonic_ns") if name in samples.columns), None)
    if marker_time is None:
        return {"association_sample_index": None, "association_status": "unassociated", "association_reason": "marker_timestamp_unavailable", "association_method": method}
    if time_column is None:
        return {"association_sample_index": None, "association_status": "unassociated", "association_reason": "sample_monotonic_timestamp_unavailable", "association_method": method}

    matches = []
    identity_conflict = False
    for position, (_, sample) in enumerate(samples.iterrows()):
        sample_time = _exact_nanosecond(sample.get(time_column), f"sample.{time_column}")
        if sample_time != marker_time:
            continue
        conflict = False
        for key in ("session_id", "capture_id"):
            marker_value = marker.get(key)
            sample_value = sample.get(key)
            if marker_value is not None and not _is_missing(sample_value) and str(marker_value) != str(sample_value):
                conflict = True
        if conflict:
            identity_conflict = True
            continue
        if not _analysis_sample_valid(sample):
            continue
        row_index = sample.get("analysis_row_index", position)
        if isinstance(row_index, np.generic):
            row_index = row_index.item()
        matches.append(row_index)
    if len(matches) == 1:
        return {"association_sample_index": matches[0], "association_status": "associated", "association_reason": None, "association_method": method}
    if len(matches) > 1:
        reason = "multiple_exact_valid_sample_matches"
    elif identity_conflict:
        reason = "capture_identity_mismatch"
    else:
        reason = "no_exact_valid_sample_timestamp_match"
    return {"association_sample_index": None, "association_status": "unassociated", "association_reason": reason, "association_method": method}


def _analysis_sample_valid(sample: pd.Series) -> bool:
    value = sample.get("analysis_gaze_valid")
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "y"}
    return bool(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if np.isscalar(missing) else False
    except (TypeError, ValueError):
        return False


def _exact_nanosecond(value: Any, label: str) -> str | None:
    if value is None or _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a decimal nanosecond value, not boolean")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{label} must be non-negative")
        return str(value)
    if isinstance(value, str):
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError(f"{label} must be a base-10 integer string")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer() or abs(value) > 2**53:
            raise ValueError(f"{label} arrived as a float with unsafe nanosecond precision")
        return str(int(value))
    raise ValueError(f"{label} has unsupported nanosecond type {type(value).__name__}")


def _is_nanosecond_field(key: Any) -> bool:
    return str(key).casefold().endswith("_ns")


def _json_nanosecond_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _exact_nanosecond(child, str(key)) if _is_nanosecond_field(key) else _json_nanosecond_safe(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_nanosecond_safe(child) for child in value]
    return _json_safe(value)


def _find_nanosecond_field(value: Any, field: str) -> str | None:
    for item in _walk_mappings(value):
        if field in item:
            result = _exact_nanosecond(item.get(field), field)
            if result is not None:
                return result
    return None


def _collect_monotonic_fields(value: Any) -> dict[str, str]:
    output: dict[str, str] = {}

    def visit(current: Any, prefix: str = "", depth: int = 0) -> None:
        if depth > 16:
            return
        if isinstance(current, Mapping):
            for key, child in current.items():
                label = f"{prefix}.{key}" if prefix else str(key)
                if _is_nanosecond_field(key):
                    normalized = _exact_nanosecond(child, label)
                    if normalized is not None:
                        output[label] = normalized
                else:
                    visit(child, label, depth + 1)
        elif isinstance(current, (list, tuple)):
            for index, child in enumerate(current):
                visit(child, f"{prefix}[{index}]", depth + 1)

    visit(value)
    return output


def _marker_frame(details: Mapping[str, Any]) -> pd.DataFrame:
    records = details["records"]
    columns = list(_MARKER_TABLE_COLUMNS)
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    return pd.DataFrame(records, columns=columns)


def _marker_summary_frame(details: Mapping[str, Any]) -> pd.DataFrame:
    rows = [
        {"field": "status", "value": details["status"]},
        {"field": "marker_count", "value": details["count"]},
        {"field": "associated_sample_count", "value": details["associated_sample_count"]},
        {"field": "unassociated_count", "value": details["unassociated_count"]},
        {"field": "association_method", "value": details["association_method"]},
        {"field": "source_marker_log_path", "value": details["source"]["marker_event_log"]["path"]},
        {"field": "source_capture_manifest_path", "value": details["source"]["capture_manifest"]["path"]},
        {"field": "source_capture_manifest_sha256", "value": details["source"]["capture_manifest"]["sha256"]},
        {"field": "source_capture_data_path", "value": details["source"]["capture_source_data"]["path"]},
        {"field": "source_capture_data_sha256", "value": details["source"]["capture_source_data"]["sha256"]},
        {"field": "source_marker_events_sha256", "value": details["source"]["marker_event_log"]["sha256"]},
        {"field": "source_marker_stream_sha256", "value": details["source"]["marker_event_log"]["stream_sha256"]},
        {"field": "source_workbook_path", "value": details["source"]["capture_workbook"]["path"]},
        {"field": "source_workbook_sha256", "value": details["source"]["capture_workbook"]["sha256"]},
        {"field": "last_monotonic_ns", "value": details["last_monotonic_ns"]},
        {"field": "user_marker_last_monotonic_ns", "value": details["user_marker_last_monotonic_ns"]},
    ]
    rows.extend({"field": key, "value": value} for key, value in details["source_monotonic_fields"].items())
    return pd.DataFrame(rows, columns=("field", "value"))


def _write_table_exclusive(path: Path, frame: pd.DataFrame, *, delimiter: str) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        clean = frame.apply(lambda column: column.map(_csv_value))
        clean.to_csv(stream, index=False, sep=delimiter, lineterminator="\n")


def _write_metadata_exclusive(
    path: Path,
    result: AnalysisResult,
    metadata: Mapping[str, Any],
    session_id: str,
    table_format: str,
    marker_details: Mapping[str, Any],
) -> None:
    config = result.methods.get("config", {})
    selected = result.methods.get("selected_metric_groups", [])
    lines = [
        "LocalGaze analysis metadata",
        f"session_id: {session_id}",
        f"source_path: {metadata.get('source_path', '')}",
        f"source_sha256: {metadata.get('source_sha256', '')}",
        f"analysis_method_version: {result.methods.get('method_version', '')}",
        f"table_format: {table_format}",
        f"source_marker_count: {marker_details.get('count', 0)}",
        f"marker_associated_sample_count: {marker_details.get('associated_sample_count', 0)}",
        f"marker_unassociated_count: {marker_details.get('unassociated_count', 0)}",
        "marker_association_policy: exact received_monotonic_ns equality; no nearest-sample inference",
        "selected_metric_groups: " + (", ".join(selected) if selected else "none"),
        "retained_metadata: input quality and non-computable capability notes",
        "analysis_config:",
    ]
    for key, value in config.items():
        lines.append(f"  {key}: {_metadata_value(value)}")
    if "ivt_saccade" in selected and isinstance(result.methods.get("ivt_saccade"), Mapping):
        ivt = result.methods["ivt_saccade"]
        metric_values = {}
        metric_frame = _as_frame(result.metrics)
        if {"metric_id", "value"}.issubset(metric_frame.columns):
            for row in metric_frame[["metric_id", "value"]].itertuples(index=False):
                metric_id = str(row.metric_id)
                if metric_id.startswith("ivt_saccade_"):
                    metric_values[metric_id] = None if row.value is None or pd.isna(row.value) else row.value
        lines.extend([
            "",
            "ivt_saccade_method:",
            f"  status: {_metadata_value(result.summary.get('ivt_saccade_status'))}",
            f"  reason: {_metadata_value(result.summary.get('ivt_saccade_reason'))}",
            f"  candidate_count: {_metadata_value(result.summary.get('ivt_saccade_candidate_count'))}",
            f"  candidate_rate_per_min: {_metadata_value(metric_values.get('ivt_saccade_candidate_rate_per_min'))}",
            f"  filtered_cluster_count: {_metadata_value(result.summary.get('ivt_saccade_filtered_cluster_count'))}",
            f"  valid_pair_exposure_s: {_metadata_value(result.summary.get('ivt_valid_pair_exposure_s'))}",
            f"  filtered_too_few_samples_count: {_metadata_value(metric_values.get('ivt_saccade_filtered_too_few_samples_count'))}",
            f"  filtered_duration_below_min_count: {_metadata_value(metric_values.get('ivt_saccade_filtered_duration_below_min_count'))}",
            f"  filtered_duration_above_max_count: {_metadata_value(metric_values.get('ivt_saccade_filtered_duration_above_max_count'))}",
            f"  speed_threshold_diagonals_s: {_metadata_value(ivt.get('speed_threshold_diagonals_s'))}",
            f"  speed_threshold_px_s_internal: {_metadata_value(ivt.get('speed_threshold_px_s'))}",
            f"  duration_bounds_s_inclusive: {_metadata_value(ivt.get('duration_bounds_s_inclusive'))}",
            f"  minimum_sample_count: {_metadata_value(ivt.get('minimum_sample_count'))}",
            f"  pair_eligibility: {_metadata_value(ivt.get('pair_eligibility'))}",
            f"  interpretation: {_metadata_value(ivt.get('interpretation'))}",
        ])
    lines.extend([
        "",
        "Interpretation: event candidates are engineering estimates; calibration training residuals are not independent accuracy measurements.",
        "Camera checks and software diagnostics do not establish real gaze accuracy.",
    ])
    with path.open("x", encoding="utf-8-sig", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def _metadata_value(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return ", ".join(_metadata_value(item) for item in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{key}={_metadata_value(child)}" for key, child in value.items())
    if value is None:
        return "not set"
    return str(value)


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
