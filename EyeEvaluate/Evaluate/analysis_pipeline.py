"""Headless adapter connecting full-contract inputs, metrics, and exports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from roi_contract import ROI

from pre.fileio import read_analysis_gaze
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
    session_id: str | None = None,
) -> AnalysisRun:
    """Run analysis on one source file without opening the application UI.

    ``roi_path`` uses the existing safe S02 workbook reader. Callers may pass
    validated ROI objects directly instead; supplying both forms is rejected.
    """

    if roi_path is not None and rois:
        raise ValueError("supply either rois or roi_path, not both")
    samples, metadata = read_analysis_gaze(source_path, time_unit=time_unit)
    selected_rois = list(rois)
    if roi_path is not None:
        from ..FileOperate.ROIextract import extract_ROI

        roi_source = Path(roi_path).expanduser().resolve()
        records = extract_ROI(roi_source)
        selected_rois = [record["_roi"] for record in records]
        metadata["roi_source_path"] = str(roi_source)
        metadata["roi_source_sha256"] = _sha256_file(roi_source)
        metadata["roi_migration_status"] = sorted({record.get("migration_status", "unknown") for record in records})
    result = analyze_samples(samples, rois=selected_rois, metadata=metadata, config=config)
    outputs = export_analysis_result(
        result,
        output_root,
        session_id=session_id,
        metadata=metadata,
        rois=selected_rois,
    )
    return AnalysisRun(result=result, outputs=outputs, metadata=metadata)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["AnalysisRun", "analyze_file"]
