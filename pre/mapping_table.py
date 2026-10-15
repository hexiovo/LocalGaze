"""Read and create small CSV/XLSX preprocessing mapping tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


INPUT_COLUMNS = ["data_path", "model_path", "session_id"]


def mapping_template_frame() -> pd.DataFrame:
    """Return the blank input table schema used by template files."""

    return pd.DataFrame(columns=INPUT_COLUMNS)


def mapping_template_example_frame() -> pd.DataFrame:
    """Return an illustrative row for previews; never used as real input."""

    return pd.DataFrame([
        {
            "data_path": "data/session_001.csv",
            "model_path": "models/session_001.pkl",
            "session_id": "session_001",
        }
    ], columns=INPUT_COLUMNS)


def save_mapping_template(path: str | Path) -> Path:
    """Save a non-overwriting blank CSV or XLSX mapping template."""

    destination = Path(path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Mapping template already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        mapping_template_frame().to_csv(destination, index=False, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        with pd.ExcelWriter(destination, engine="openpyxl") as writer:
            mapping_template_frame().to_excel(writer, sheet_name="Input", index=False)
            mapping_template_example_frame().to_excel(writer, sheet_name="Example", index=False)
            pd.DataFrame({
                "Field": INPUT_COLUMNS,
                "Description": [
                    "Required. A gaze data file or a folder containing gaze data files.",
                    "Optional. A model file or folder; blank means run QC without a model.",
                    "Optional. A session label copied into QC and the report.",
                ],
                "Path rule": [
                    "Relative paths are resolved from the mapping file's folder.",
                    "A folder is matched to data files by case-insensitive filename stem.",
                    "Blank rows are ignored; invalid rows are reported independently.",
                ],
            }).to_excel(writer, sheet_name="Instructions", index=False)
    else:
        raise ValueError("Mapping templates support only .csv and .xlsx files")
    return destination


def read_mapping_table(path: str | Path) -> list[dict[str, Any]]:
    """Read mappings and isolate invalid data rows without aborting the batch.

    Returned ``row_number`` values are one-based physical table row numbers,
    including the header row. Table-level errors, including a missing required
    ``data_path`` column or an unreadable workbook, raise ``ValueError``.
    """

    table_path = Path(path).expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"Mapping table is not a regular file: {table_path}")
    frame = _read_frame(table_path)
    normalized: dict[str, str] = {}
    for column in frame.columns:
        key = str(column).strip().casefold()
        if key in normalized:
            raise ValueError(f"Mapping table has duplicate column names after normalization: {column!r}")
        normalized[key] = str(column)
    if "data_path" not in normalized:
        raise ValueError("Mapping table is missing the required 'data_path' column")

    data_column = normalized["data_path"]
    model_column = normalized.get("model_path")
    session_column = normalized.get("session_id")
    records: list[dict[str, Any]] = []
    base = table_path.parent
    for offset, (_, row) in enumerate(frame.iterrows(), start=2):
        data_value = _cell_text(row[data_column])
        model_value = _cell_text(row[model_column]) if model_column is not None else ""
        session_id = _cell_text(row[session_column]) if session_column is not None else ""
        if not data_value and not model_value and not session_id:
            continue

        data_path = _resolve_path(data_value, base) if data_value else None
        model_path = _resolve_path(model_value, base) if model_value else None
        error = None
        if data_path is None:
            error = "Missing data_path"
        else:
            error = _path_error(data_path, allowed_suffixes={".xlsx", ".xlsm", ".csv", ".tsv", ".txt"}, label="data_path")
        if error is None and model_path is not None:
            error = _path_error(model_path, allowed_suffixes={".xlsx", ".xlsm", ".pkl"}, label="model_path")
        records.append({
            "row_number": offset,
            "data_path": str(data_path) if data_path is not None else None,
            "model_path": str(model_path) if model_path is not None else None,
            "session_id": session_id or None,
            "error": error,
        })
    return records


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        try:
            with pd.ExcelFile(path) as excel:
                sheets = list(excel.sheet_names)
            if not sheets:
                raise ValueError("Mapping workbook contains no worksheets")
            sheet = "Input" if "Input" in sheets else sheets[0]
            return pd.read_excel(path, sheet_name=sheet, dtype=object)
        except Exception as exc:
            if isinstance(exc, ValueError) and "data_path" in str(exc):
                raise
            raise ValueError(f"Unable to read mapping workbook {path}: {type(exc).__name__}: {exc}") from exc
    if suffix not in {".csv", ".tsv"}:
        raise ValueError("Mapping table must be a .csv, .tsv, .xlsx, or .xlsm file")
    delimiter = "\t" if suffix == ".tsv" else ","
    try:
        return pd.read_csv(
            path,
            sep=delimiter,
            dtype=object,
            encoding="utf-8-sig",
            keep_default_na=False,
            skip_blank_lines=False,
        )
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                sep=delimiter,
                dtype=object,
                encoding="gb18030",
                keep_default_na=False,
                skip_blank_lines=False,
            )
        except Exception as exc:
            raise ValueError(f"Unable to read mapping table {path}: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Unable to read mapping table {path}: {type(exc).__name__}: {exc}") from exc


def _path_error(path: Path, *, allowed_suffixes: set[str], label: str) -> str | None:
    if not path.exists():
        return f"{label} does not exist: {path}"
    if path.is_file() and path.suffix.lower() not in allowed_suffixes:
        return f"Unsupported {label} file type: {path.suffix}"
    if not path.is_file() and not path.is_dir():
        return f"{label} is not a regular file or folder: {path}"
    return None


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _cell_text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
