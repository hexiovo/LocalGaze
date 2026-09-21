"""Input discovery and format-preserving output helpers."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill


DATA_SUFFIXES = {".xlsx", ".xlsm", ".csv", ".tsv", ".txt"}
MODEL_SUFFIXES = {".xlsx", ".xlsm", ".pkl"}

ALIASES = {
    "time": ("时间", "时间戳", "timestamp", "time", "datetime"),
    "x": ("x", "gazex", "gaze_x", "水平坐标", "横坐标"),
    "y": ("y", "gazey", "gaze_y", "垂直坐标", "纵坐标"),
    "blink": ("blink", "眨眼", "眨眼标记"),
    "left_pupil": ("左瞳孔直径", "leftpupil", "left_pupil", "pupil_left"),
    "right_pupil": ("右瞳孔直径", "rightpupil", "right_pupil", "pupil_right"),
}


def discover_files(path: Path, suffixes: set[str]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        (p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes),
        key=lambda p: str(p).casefold(),
    )


def read_gaze(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        with pd.ExcelFile(path) as excel:
            sheet_names = list(excel.sheet_names)
        preferred = ["眼动数据", "EyeData", "GazeData", "gaze_data"]
        candidates = preferred + [name for name in sheet_names if name not in preferred]
        errors: list[str] = []
        for sheet in candidates:
            if sheet not in sheet_names:
                continue
            try:
                frame = pd.read_excel(path, sheet_name=sheet)
                columns = resolve_columns(frame)
                return frame, {"kind": "excel", "sheet": sheet, "columns": columns}
            except ValueError as exc:
                errors.append(f"{sheet}: {exc}")
        raise ValueError("No Excel sheet contains recognizable time, X, and Y columns; " + "; ".join(errors))
    delimiter, encoding = _detect_text_format(path)
    frame = pd.read_csv(path, sep=delimiter, encoding=encoding)
    return frame, {
        "kind": "text", "delimiter": delimiter, "encoding": encoding,
        "columns": resolve_columns(frame),
    }


def resolve_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    normalized = {_normalize(name): str(name) for name in frame.columns}
    found: dict[str, str | None] = {}
    for role, aliases in ALIASES.items():
        found[role] = next((normalized[_normalize(alias)] for alias in aliases if _normalize(alias) in normalized), None)
    missing = [role for role in ("time", "x", "y") if found[role] is None]
    if missing:
        raise ValueError(f"Missing required column roles: {', '.join(missing)}")
    return found


def read_calibration_error(model_path: Path) -> tuple[float | None, str, Path | None]:
    workbook = model_path
    if model_path.suffix.lower() == ".pkl":
        workbook = model_path.with_suffix(".xlsx")
    if workbook.suffix.lower() not in {".xlsx", ".xlsm"} or not workbook.is_file():
        return None, "calibration_workbook_not_found", None
    try:
        with pd.ExcelFile(workbook) as excel:
            sheets = list(excel.sheet_names)
        additional = _case_sheet(sheets, "AdditionalCalibration")
        if additional:
            table = pd.read_excel(workbook, sheet_name=additional)
            column = _case_column(table.columns, "MeanErrorPerRound")
            if column:
                values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values) & (values >= 0)]
                if values.size:
                    return float(values[-1]), "additional_calibration_final", workbook
        nine = _case_sheet(sheets, "NinePointCalibration")
        if nine:
            table = pd.read_excel(workbook, sheet_name=nine)
            column = _case_column(table.columns, "Error")
            if column:
                values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values) & (values >= 0)]
                if values.size:
                    return float(values.mean()), "nine_point_mean_fallback", workbook
        return None, "no_usable_calibration_error", workbook
    except Exception as exc:  # retained in the batch report
        return None, f"calibration_read_error: {type(exc).__name__}: {exc}", workbook


def write_result(
    source: Path,
    destination: Path,
    processed: pd.DataFrame,
    metadata: dict[str, Any],
    qc: dict[str, Any],
    windows: pd.DataFrame,
    bad_segments: pd.DataFrame,
    suspect_segments: pd.DataFrame,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if metadata["kind"] == "excel":
        shutil.copy2(source, destination)
        keep_vba = source.suffix.lower() == ".xlsm"
        workbook = load_workbook(destination, keep_vba=keep_vba)
        sheet = workbook[metadata["sheet"]]
        source_columns = list(pd.read_excel(source, sheet_name=metadata["sheet"], nrows=0).columns)
        appended = [name for name in processed.columns if name not in source_columns]
        start_column = sheet.max_column + 1
        header_fill = PatternFill("solid", fgColor="D9EAF7")
        for offset, name in enumerate(appended):
            cell = sheet.cell(row=1, column=start_column + offset, value=name)
            cell.font = Font(bold=True)
            cell.fill = header_fill
            values = processed[name].tolist()
            for row, value in enumerate(values, start=2):
                if pd.isna(value):
                    value = None
                elif isinstance(value, np.generic):
                    value = value.item()
                sheet.cell(row=row, column=start_column + offset, value=value)
        _replace_sheet(workbook, "Pre_QC", _qc_table(qc))
        _replace_sheet(workbook, "Pre_Windows", windows)
        _replace_sheet(workbook, "Pre_BadSegments", bad_segments)
        _replace_sheet(workbook, "Pre_SuspectSegments", suspect_segments)
        workbook.save(destination)
    else:
        processed.to_csv(
            destination, sep=metadata["delimiter"], encoding=metadata["encoding"],
            index=False, lineterminator="\n",
        )
        sidecar = destination.with_suffix(destination.suffix + ".qc.json")
        sidecar.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")


def _replace_sheet(workbook, title: str, frame: pd.DataFrame) -> None:
    if title in workbook.sheetnames:
        del workbook[title]
    sheet = workbook.create_sheet(title)
    for column, name in enumerate(frame.columns, start=1):
        cell = sheet.cell(row=1, column=column, value=str(name))
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
    for row_index, row in enumerate(frame.itertuples(index=False, name=None), start=2):
        for column, value in enumerate(row, start=1):
            if pd.isna(value):
                value = None
            elif isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, np.generic):
                value = value.item()
            sheet.cell(row=row_index, column=column, value=value)
    sheet.freeze_panes = "A2"


def _qc_table(qc: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        else:
            rows.append({"Key": prefix, "Value": value})
    visit("", qc)
    return pd.DataFrame(rows)


def _detect_text_format(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()[:65536]
    encoding = "utf-8-sig"
    try:
        sample = raw.decode(encoding)
    except UnicodeDecodeError:
        encoding = "gb18030"
        sample = raw.decode(encoding)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return delimiter, encoding


def _normalize(value: Any) -> str:
    return "".join(ch for ch in str(value).casefold().strip() if ch not in " _-()[]（）")


def _case_sheet(sheets: list[str], requested: str) -> str | None:
    return next((name for name in sheets if name.casefold() == requested.casefold()), None)


def _case_column(columns, requested: str) -> str | None:
    return next((name for name in columns if str(name).casefold() == requested.casefold()), None)
