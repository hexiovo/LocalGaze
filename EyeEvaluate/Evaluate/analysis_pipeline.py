"""Headless adapter connecting full-contract inputs, metrics, and exports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from roi_contract import ROI

from pre.fileio import _read_analysis_manifest, read_analysis_gaze
from .analysis_options import normalize_analysis_config, normalize_selected_metrics, normalize_table_format
from .analysis_export import export_analysis_result
from .metrics import AnalysisConfig, AnalysisResult, analyze_samples


@dataclass(frozen=True)
class AnalysisRun:
    result: AnalysisResult
    outputs: dict[str, Path]
    metadata: dict[str, Any]


def analyze_file(
    source_path: str | Path,
    output_root: str | Path,
    *,
    rois: Sequence[ROI] = (),
    roi_path: str | Path | None = None,
    time_unit: str | None = None,
    config: AnalysisConfig | Mapping[str, Any] | None = None,
    selected_metrics: Sequence[str] | None = None,
    table_format: str = "csv",
    session_id: str | None = None,
) -> AnalysisRun:
    """Run analysis on one source file without opening the application UI.

    ``roi_path`` uses the existing safe S02 workbook reader. Callers may pass
    validated ROI objects directly instead; supplying both forms is rejected.
    """

    cfg = config if isinstance(config, AnalysisConfig) else AnalysisConfig(**normalize_analysis_config(config))
    selected_groups = normalize_selected_metrics(selected_metrics)
    selected_format = normalize_table_format(table_format)
    source = Path(source_path).expanduser().resolve()
    samples, metadata = read_analysis_gaze(source, time_unit=time_unit)
    # The reader normalizes analysis fields, while this adapter retains the
    # complete source manifest so capture markers and their durable references
    # survive both direct capture input and preprocessing output.
    source_manifest, source_manifest_path = _read_analysis_manifest(source)
    # Text preprocessing writes its complete provenance to <file>.qc.json.
    # Read that documented sidecar here rather than changing the frozen pre/*
    # owner files; the nested source_capture_manifest carries the original
    # capture marker events and durable hashes.
    if not source_manifest:
        qc_path = source.with_suffix(source.suffix + ".qc.json")
        if qc_path.is_file() and not qc_path.is_symlink():
            candidate = json.loads(qc_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict) and "source_capture_manifest" in candidate:
                source_manifest = candidate
                source_manifest_path = qc_path
    if source_manifest:
        metadata["source_capture_manifest"] = source_manifest
    if source_manifest_path is not None:
        metadata["source_manifest_path"] = str(source_manifest_path)
        metadata["source_manifest_sha256"] = _sha256_file(source_manifest_path)
    roi_selected = "roi" in selected_groups
    selected_rois = list(rois) if roi_selected else []
    if roi_selected and roi_path is not None and rois:
        raise ValueError("supply either rois or roi_path, not both")
    if roi_selected and roi_path is not None:
        from ..FileOperate.ROIextract import extract_ROI

        roi_source = Path(roi_path).expanduser().resolve()
        records = extract_ROI(roi_source)
        selected_rois = [record["_roi"] for record in records]
        metadata["roi_source_path"] = str(roi_source)
        metadata["roi_source_sha256"] = _sha256_file(roi_source)
        metadata["roi_migration_status"] = sorted({record.get("migration_status", "unknown") for record in records})
    result = analyze_samples(samples, rois=selected_rois, metadata=metadata, config=cfg, selected_metrics=selected_groups)
    outputs = export_analysis_result(
        result,
        output_root,
        session_id=session_id,
        metadata=metadata,
        rois=selected_rois,
        table_format=selected_format,
    )
    marker_path = outputs.get("markers")
    if marker_path is not None:
        metadata["analysis_marker_artifact_path"] = str(marker_path)
        metadata["analysis_marker_artifact_sha256"] = _sha256_file(marker_path)
    return AnalysisRun(result=result, outputs=outputs, metadata=metadata)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["AnalysisRun", "analyze_file"]
