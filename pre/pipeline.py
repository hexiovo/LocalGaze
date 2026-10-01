"""Path-level one-to-one model/data matching and batch preprocessing."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

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


def preprocess_paths(
    model_path: str | Path,
    data_path: str | Path,
    output_path: str | Path | None = None,
    *,
    config: PreprocessConfig | None = None,
    preset: Literal["legacy", "analysis"] = "legacy",
) -> pd.DataFrame:
    """Match same-named model/data files and preprocess every unambiguous pair.

    The default ``legacy`` preset preserves calibration-distance interpolation.
    The explicit ``analysis`` preset records measured masks while forcibly
    disabling coordinate interpolation, regardless of the supplied config.
    """

    if preset not in {"legacy", "analysis"}:
        raise ValueError("preset must be 'legacy' or 'analysis'")

    model_root = Path(model_path).expanduser().resolve()
    data_root = Path(data_path).expanduser().resolve()
    output_root = _default_output(data_root) if output_path is None else Path(output_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if preset == "analysis":
        reserved = [output_root / "preprocess_report.csv", output_root / "preprocess_manifest.json"]
        existing = [path for path in reserved if path.exists()]
        if existing:
            raise FileExistsError(f"Analysis output already contains a report or manifest: {existing}")

    model_files = discover_files(model_root, MODEL_SUFFIXES)
    data_files = discover_files(data_root, DATA_SUFFIXES)
    model_index = _index_models(model_files)
    data_index = _unique_index(data_files, "data")
    rows: list[dict[str, Any]] = []

    all_keys = sorted(set(data_index) | set(model_index))
    for number, key in enumerate(all_keys, start=1):
        source = data_index.get(key)
        model = model_index.get(key)
        display_name = source.stem if source is not None else model.stem
        row: dict[str, Any] = {
            "name": display_name,
            "data_file": str(source) if source else "",
            "model_file": str(model) if model else "",
            "output_file": "",
            "status": "",
            "message": "",
            "sample_count": None,
            "valid_final_proportion": None,
            "calibration_error_pixels": None,
            "calibration_status": "",
        }
        if source is None:
            row.update(status="model_without_data", message="No same-named data file")
            rows.append(row)
            continue
        if model is None:
            row.update(status="data_without_model", message="No same-named model file")
            rows.append(row)
            continue
        try:
            destination = _destination(source, data_root, output_root)
            if destination.exists():
                raise FileExistsError(f"Output already exists: {destination}")
            gaze, metadata = read_gaze(source)
            error_pixels, calibration_status, calibration_workbook = read_calibration_error(model)
            local_config = copy.deepcopy(config or PreprocessConfig())
            if preset == "analysis":
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
            result.qc["source_data_file"] = str(source)
            result.qc["source_model_file"] = str(model)
            result.qc["calibration_workbook"] = str(calibration_workbook) if calibration_workbook else ""
            result.qc["calibration_status"] = calibration_status
            result.qc["calibration_error_pixels"] = error_pixels
            result.qc["preprocessing_preset"] = preset
            result.qc["calibration_distance_used_for_interpolation"] = preset == "legacy" and error_pixels is not None
            if preset == "analysis":
                result.qc["coordinate_interpolation"] = "disabled_analysis_preset"
            write_result(
                source, destination, result.data, metadata, result.qc,
                result.windows, result.bad_segments, result.suspect_segments,
            )
            row.update(
                output_file=str(destination), status="processed", message="OK",
                sample_count=result.qc["sample_count"],
                valid_final_proportion=result.qc["valid_final_proportion"],
                calibration_error_pixels=error_pixels,
                calibration_status=calibration_status,
            )
            print(f"[{number}/{len(all_keys)}] processed {display_name} -> {destination}")
        except Exception as exc:
            row.update(status="failed", message=f"{type(exc).__name__}: {exc}")
            print(f"[{number}/{len(all_keys)}] failed {display_name}: {type(exc).__name__}: {exc}")
        rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(output_root / "preprocess_report.csv", index=False, encoding="utf-8-sig")
    manifest_config = asdict(config or PreprocessConfig())
    if preset == "analysis":
        manifest_config["max_interpolation_distance"] = None
    manifest = {
        "model_path": str(model_root), "data_path": str(data_root),
        "output_path": str(output_root), "config": manifest_config,
        "preset": preset,
        "coordinate_interpolation": "disabled" if preset == "analysis" else "calibration_error_distance_when_available",
        "matching_rule": "case-insensitive exact filename stem",
    }
    (output_root / "preprocess_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def preprocess_analysis_paths(
    model_path: str | Path,
    data_path: str | Path,
    output_path: str | Path | None = None,
    *,
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """Run the compatibility preprocessor with its non-interpolating preset."""

    return preprocess_paths(model_path, data_path, output_path, config=config, preset="analysis")


def _index_models(files: list[Path]) -> dict[str, Path]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(path.stem.casefold(), []).append(path)
    result: dict[str, Path] = {}
    for key, candidates in groups.items():
        calibration = [p for p in candidates if p.suffix.lower() in {".xlsx", ".xlsm"}]
        if len(calibration) == 1:
            result[key] = calibration[0]
        elif len(calibration) > 1:
            raise ValueError(f"Duplicate model calibration workbooks for {key}: {calibration}")
        elif len(candidates) == 1:
            result[key] = candidates[0]
        else:
            raise ValueError(f"Duplicate model files for {key}: {candidates}")
    return result


def _unique_index(files: list[Path], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in files:
        key = path.stem.casefold()
        if key in result:
            raise ValueError(f"Duplicate {label} filename stem {path.stem!r}: {result[key]} and {path}")
        result[key] = path
    return result


def _default_output(data_root: Path) -> Path:
    if data_root.is_file():
        return data_root.parent / "preprocessed"
    return data_root.parent / f"{data_root.name}_preprocessed"


def _destination(source: Path, data_root: Path, output_root: Path) -> Path:
    if data_root.is_file():
        return output_root / source.name
    return output_root / source.relative_to(data_root)
