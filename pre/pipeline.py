"""Path-level matching and batch preprocessing for gaze data."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

from .core import PreprocessConfig, preprocess_frame
from .fileio import (
    DATA_SUFFIXES,
    MODEL_SUFFIXES,
    discover_files,
    read_calibration_error,
    read_gaze,
    write_result,
)
from .mapping_table import read_mapping_table


REPORT_COLUMNS = [
    "name",
    "data_file",
    "model_file",
    "output_file",
    "status",
    "message",
    "sample_count",
    "valid_final_proportion",
    "calibration_error_pixels",
    "calibration_status",
    "row_number",
    "session_id",
    "calibration_reason",
    "input_path",
]


@dataclass(frozen=True, slots=True)
class _InputSpec:
    input_path: Path | None
    model_path: Path | None
    session_id: str | None = None
    row_number: int | None = None
    error: str | None = None


def preprocess_paths(
    model_path: str | Path | None,
    data_path: str | Path,
    output_path: str | Path | None = None,
    *,
    config: PreprocessConfig | None = None,
    preset: Literal["legacy", "analysis"] = "legacy",
) -> pd.DataFrame:
    """Preprocess data files under one path, optionally associating models by stem.

    ``model_path`` may be ``None``. In that case preprocessing still performs
    data cleaning and QC, while gaze XY interpolation remains disabled. The
    default ``legacy`` preset preserves calibration-distance interpolation
    when a matching model and usable calibration error are both available.
    The explicit ``analysis`` preset always disables coordinate interpolation.
    """

    _validate_preset(preset)
    data_root = Path(data_path).expanduser().resolve()
    model_root = None if model_path is None else Path(model_path).expanduser().resolve()
    output_root = _default_output(data_root) if output_path is None else Path(output_path).expanduser().resolve()
    spec = _InputSpec(input_path=data_root, model_path=model_root)
    return _run_batch(
        [spec],
        output_root,
        config=config,
        preset=preset,
        mapping_mode=False,
        excluded_paths=set(),
        excluded_roots=[output_root],
        manifest_fields={
            "model_path": str(model_root) if model_root is not None else None,
            "data_path": str(data_root),
            "matching_rule": "case-insensitive exact filename stem",
        },
    )


def preprocess_mapping_table(
    table_path: str | Path,
    output_path: str | Path,
    *,
    config: PreprocessConfig | None = None,
    preset: Literal["legacy", "analysis"] = "legacy",
) -> pd.DataFrame:
    """Preprocess data referenced by a CSV/XLSX mapping table.

    ``output_path`` is required and is treated as an output directory. Paths in
    the table are resolved relative to the table file. Each valid table row may
    point to one data file or a folder; folders expand to supported gaze files.
    """

    _validate_preset(preset)
    table = Path(table_path).expanduser().resolve()
    requested_output_root = Path(output_path).expanduser().resolve()
    records = read_mapping_table(table)
    output_root = _new_mapping_run_root(requested_output_root)
    specs = [
        _InputSpec(
            input_path=Path(record["data_path"]) if record["data_path"] is not None else None,
            model_path=Path(record["model_path"]) if record["model_path"] is not None else None,
            session_id=record["session_id"],
            row_number=record["row_number"],
            error=record["error"],
        )
        for record in records
    ]
    return _run_batch(
        specs,
        output_root,
        config=config,
        preset=preset,
        mapping_mode=True,
        excluded_paths={table},
        excluded_roots=[requested_output_root, output_root],
        manifest_fields={
            "mapping_table": str(table),
            "requested_output_root": str(requested_output_root),
            "matching_rule": "case-insensitive exact filename stem; unmatched data runs without a model",
        },
    )


def preprocess_analysis_paths(
    model_path: str | Path | None,
    data_path: str | Path,
    output_path: str | Path | None = None,
    *,
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """Run the compatibility preprocessor with its non-interpolating preset."""

    return preprocess_paths(model_path, data_path, output_path, config=config, preset="analysis")


def _validate_preset(preset: str) -> None:
    if preset not in {"legacy", "analysis"}:
        raise ValueError("preset must be 'legacy' or 'analysis'")


def _run_batch(
    specs: list[_InputSpec],
    output_root: Path,
    *,
    config: PreprocessConfig | None,
    preset: Literal["legacy", "analysis"],
    mapping_mode: bool,
    excluded_paths: set[Path],
    excluded_roots: list[Path],
    manifest_fields: dict[str, Any],
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "preprocess_report.csv"
    manifest_path = output_root / "preprocess_manifest.json"
    existing = [path for path in (report_path, manifest_path) if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"Output already contains a preprocessing report or manifest: {existing}")

    rows: list[dict[str, Any]] = []
    first_mapping_row: dict[tuple[str, str | None, str | None], int] = {}
    for spec in specs:
        if mapping_mode and spec.error is None and spec.input_path is not None and spec.row_number is not None:
            mapping_key = (
                str(spec.input_path.resolve()).casefold(),
                str(spec.model_path.resolve()).casefold() if spec.model_path is not None else None,
                spec.session_id.casefold() if spec.session_id is not None else None,
            )
            if mapping_key in first_mapping_row:
                original = first_mapping_row[mapping_key]
                rows.append(_report_row(
                    spec,
                    status="duplicate_row",
                    message=f"Duplicate input/model/session row; skipped because it was listed in mapping row {original}",
                    data_file=str(spec.input_path),
                    model_file=str(spec.model_path) if spec.model_path else "",
                ))
                continue
            first_mapping_row[mapping_key] = spec.row_number
        if spec.error:
            rows.append(_report_row(
                spec,
                status="invalid_mapping_row",
                message=spec.error,
                data_file="",
                model_file=str(spec.model_path) if spec.model_path else "",
            ))
            continue
        if spec.input_path is None:
            rows.append(_report_row(spec, status="invalid_mapping_row", message="Missing data_path"))
            continue

        try:
            _validate_data_input(spec.input_path)
            sources = _discover_data_files(spec.input_path, excluded_paths, excluded_roots)
        except Exception as exc:
            rows.append(_report_row(
                spec,
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
                data_file=str(spec.input_path),
                model_file=str(spec.model_path) if spec.model_path else "",
            ))
            continue
        if not sources:
            rows.append(_report_row(
                spec,
                status="failed",
                message="No supported gaze data files found",
                data_file=str(spec.input_path),
                model_file=str(spec.model_path) if spec.model_path else "",
            ))
            continue

        try:
            model_files = _model_files(spec.model_path)
            model_index, model_errors = _index_models(model_files)
        except Exception as exc:
            rows.append(_report_row(
                spec,
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
                data_file=str(spec.input_path),
                model_file=str(spec.model_path) if spec.model_path else "",
            ))
            continue

        for source in sources:
            key = source.stem.casefold()
            if key in model_errors:
                rows.append(_report_row(
                    spec,
                    status="failed",
                    message=model_errors[key],
                    data_file=str(source),
                ))
                continue
            model = model_index.get(key)
            try:
                if model is not None:
                    _assert_readable_model(model)
                destination = (
                    _mapping_destination(source, spec.input_path, output_root, spec.row_number)
                    if mapping_mode
                    else _destination(source, spec.input_path, output_root)
                )
                _ensure_output_is_available(destination)
                gaze, metadata = read_gaze(source)
                if model is None:
                    error_pixels = None
                    calibration_status = "no_model"
                    calibration_reason = ""
                    calibration_workbook = None
                else:
                    error_pixels, calibration_detail, calibration_workbook = read_calibration_error(model)
                    if error_pixels is None:
                        calibration_status = "calibration_error_unavailable"
                        calibration_reason = calibration_detail
                    else:
                        calibration_status = calibration_detail
                        calibration_reason = ""

                local_config = copy.deepcopy(config or PreprocessConfig())
                if preset == "analysis" or error_pixels is None:
                    local_config.max_interpolation_distance = None
                else:
                    local_config.max_interpolation_distance = error_pixels

                columns = metadata["columns"]
                result = preprocess_frame(
                    gaze,
                    time_column=columns["time"],
                    x_column=columns["x"],
                    y_column=columns["y"],
                    blink_column=columns["blink"],
                    left_pupil_column=columns["left_pupil"],
                    right_pupil_column=columns["right_pupil"],
                    config=local_config,
                )
                result.qc.update({
                    "source_data_file": str(source),
                    "source_sha256": metadata["source_sha256"],
                    "source_manifest_path": metadata["source_manifest_path"] or "",
                    "source_manifest_sha256": metadata["source_manifest_sha256"] or "",
                    "source_capture_manifest": metadata["source_capture_manifest"],
                    "source_model_file": str(model) if model is not None else "",
                    "calibration_workbook": str(calibration_workbook) if calibration_workbook else "",
                    "calibration_status": calibration_status,
                    "calibration_reason": calibration_reason,
                    "calibration_error_pixels": error_pixels,
                    "preprocessing_preset": preset,
                    "calibration_distance_used_for_interpolation": preset == "legacy" and error_pixels is not None,
                    "session_id": spec.session_id,
                    "model_compatibility_verified": False,
                })
                if preset == "analysis":
                    result.qc["coordinate_interpolation"] = "disabled_analysis_preset"
                elif error_pixels is None:
                    result.qc["coordinate_interpolation"] = "disabled_no_calibration_error"
                write_result(
                    source,
                    destination,
                    result.data,
                    metadata,
                    result.qc,
                    result.windows,
                    result.bad_segments,
                    result.suspect_segments,
                )
                rows.append(_report_row(
                    spec,
                    status="processed",
                    message="OK" if model is not None else "Processed without a matching model",
                    data_file=str(source),
                    model_file=str(model) if model is not None else "",
                    output_file=str(destination),
                    sample_count=result.qc["sample_count"],
                    valid_final_proportion=result.qc["valid_final_proportion"],
                    calibration_error_pixels=error_pixels,
                    calibration_status=calibration_status,
                    calibration_reason=calibration_reason,
                    name=source.stem,
                ))
                print(f"processed {source.name} -> {destination}")
            except Exception as exc:
                rows.append(_report_row(
                    spec,
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                    data_file=str(source),
                    model_file=str(model) if model is not None else "",
                    calibration_status="calibration_error_unavailable" if model is not None else "no_model",
                ))
                print(f"failed {source.name}: {type(exc).__name__}: {exc}")

    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    manifest_config = asdict(config or PreprocessConfig())
    if preset == "analysis":
        manifest_config["max_interpolation_distance"] = None
    manifest = {
        **manifest_fields,
        "output_path": str(output_root),
        "config": manifest_config,
        "preset": preset,
        "coordinate_interpolation": "disabled" if preset == "analysis" else "calibration_error_distance_when_available",
        "mapping_mode": mapping_mode,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report.attrs["summary_path"] = str(report_path)
    report.attrs["manifest_path"] = str(manifest_path)
    report.attrs["output_root"] = str(output_root)
    return report


def _report_row(spec: _InputSpec, **values: Any) -> dict[str, Any]:
    row = {
        "row_number": spec.row_number,
        "session_id": spec.session_id,
        "input_path": str(spec.input_path) if spec.input_path is not None else "",
        "name": spec.input_path.stem if spec.input_path is not None and spec.input_path.is_file() else "",
        "data_file": "",
        "model_file": "",
        "output_file": "",
        "status": "",
        "message": "",
        "sample_count": None,
        "valid_final_proportion": None,
        "calibration_error_pixels": None,
        "calibration_status": "",
        "calibration_reason": "",
    }
    row.update(values)
    return row


def _validate_data_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.suffix.lower() not in DATA_SUFFIXES:
        raise ValueError(f"Unsupported gaze data file type: {path.suffix}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"Data path is not a regular file or folder: {path}")


def _discover_data_files(path: Path, excluded_paths: set[Path], excluded_roots: list[Path]) -> list[Path]:
    excluded_files = {item.resolve() for item in excluded_paths}
    data_root = path.resolve()
    excluded_dirs = []
    if path.is_dir():
        excluded_dirs = [
            item.resolve()
            for item in excluded_roots
            if item.resolve() != data_root and _is_relative_to(item.resolve(), data_root)
        ]
    discovered = discover_files(path, DATA_SUFFIXES)
    result = []
    for source in discovered:
        resolved = source.resolve()
        if resolved in excluded_files:
            continue
        if path.is_dir():
            relative = source.relative_to(path)
            if any(part.casefold().startswith("preprocessing_") for part in relative.parts[:-1]):
                continue
        if any(_is_relative_to(resolved, excluded_root) for excluded_root in excluded_dirs):
            continue
        result.append(source)
    return result


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _model_files(model_path: Path | None) -> list[Path]:
    if model_path is None:
        return []
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if model_path.is_file():
        if model_path.suffix.lower() not in MODEL_SUFFIXES:
            raise ValueError(f"Unsupported model file type: {model_path.suffix}")
        if model_path.suffix.lower() in {".xlsx", ".xlsm"}:
            paired_model = model_path.with_suffix(".pkl")
            if paired_model.is_file():
                return [paired_model, model_path]
        return [model_path]
    if not model_path.is_dir():
        raise ValueError(f"Model path is not a regular file or folder: {model_path}")
    return discover_files(model_path, MODEL_SUFFIXES)


def _index_models(files: list[Path]) -> tuple[dict[str, Path], dict[str, str]]:
    """Prefer a same-stem PKL as the model, leaving XLSX as its evidence file."""

    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(path.stem.casefold(), []).append(path)
    selected: dict[str, Path] = {}
    errors: dict[str, str] = {}
    for key, candidates in groups.items():
        models = [p for p in candidates if p.suffix.lower() == ".pkl"]
        workbooks = [p for p in candidates if p.suffix.lower() in {".xlsx", ".xlsm"}]
        if len(models) > 1:
            errors[key] = f"Duplicate model files for {key}: {models}"
        elif len(workbooks) > 1:
            errors[key] = f"Duplicate calibration workbooks for {key}: {workbooks}"
        elif models:
            selected[key] = models[0]
        elif workbooks:
            selected[key] = workbooks[0]
    return selected, errors


def _assert_readable_model(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Model file is not a regular file: {path}")
    with path.open("rb") as stream:
        stream.read(1)


def _ensure_output_is_available(destination: Path) -> None:
    sidecar = destination.with_suffix(destination.suffix + ".qc.json")
    existing = [path for path in (destination, sidecar) if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"Output already exists: {existing}")


def _default_output(data_root: Path) -> Path:
    if data_root.is_file():
        return data_root.parent / "preprocessed"
    return data_root.parent / f"{data_root.name}_preprocessed"


def _destination(source: Path, data_root: Path, output_root: Path) -> Path:
    if data_root.is_file():
        return output_root / source.name
    return output_root / source.relative_to(data_root)


def _mapping_destination(source: Path, data_root: Path, output_root: Path, row_number: int | None) -> Path:
    row_label = f"row_{row_number:04d}" if row_number is not None else "row_unknown"
    if data_root.is_file():
        relative = Path(source.name)
    else:
        relative = source.relative_to(data_root)
    return output_root / row_label / relative


def _new_mapping_run_root(base_root: Path) -> Path:
    """Allocate one unique child directory for every mapping-table run."""

    base_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(10):
        candidate = base_root / f"preprocessing_{timestamp}_{uuid4().hex[:8]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Unable to allocate a unique preprocessing run under {base_root}")
