"""Read legacy and versioned ROI workbooks without evaluating expressions."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

_FILE = Path(__file__).resolve()
_EYE_EVALUATE_ROOT = _FILE.parents[1]
_REPOSITORY_ROOT = _FILE.parents[2]
for _path in (_REPOSITORY_ROOT, _EYE_EVALUATE_ROOT):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from roi_contract import (
    LegacyROIParseError,
    ROI,
    ROI_METADATA_COLUMNS,
    ROIValidationError,
    SCHEMA_VERSION,
    format_legacy_description,
    parse_legacy_description,
    validate_roi_reference_compatibility,
)
from .xlsx_validation import check_columns

import pandas as pd


_ROI_SHEET = "ROI_Metadata"
_REQUIRED_COLUMNS = ["命名", "形状", "描述"]
_METADATA_COLUMNS = set(ROI_METADATA_COLUMNS)
_REFERENCE_KEYS = {
    "space",
    "width",
    "height",
    "source",
    "monitor_id",
    "monitor_origin_px",
    "dpi_scale",
    "dpi_source",
}


def ROIchoose(logger):
    """Prompt for an ROI workbook and report any legacy-migration semantics."""
    try:
        from ..Popup.tkClass import custom_file_dialog, custom_messagebox
    except ImportError:
        from Popup.tkClass import custom_file_dialog, custom_messagebox

    while True:
        result_filedialog, path = custom_file_dialog(
            title="请选择ROI兴趣区所在的文件(.xlsx文件)", logger=logger
        )
        if result_filedialog == 1 and path != "":
            try:
                roi_data = extract_ROI(path)
                _report_import_hints(roi_data, logger)
                return roi_data
            except Exception as exc:
                print(exc)
                logger.info("ROI输入有误，输入为错误文件")
                custom_messagebox(
                    title="ROI 文件无法读取",
                    message=str(exc),
                    autowh=True,
                )
        elif result_filedialog == 2 and path != "":
            logger.info("ROI输入有误，输入为文件夹")
            custom_messagebox(
                title="提示",
                message="所选择为文件夹",
                autowh=False,
                width=280,
                height=160,
            )
        elif result_filedialog == 0 or path == "":
            logger.info("用户取消输入")
            return []
        else:
            logger.info("未知错误")
            return []


def extract_ROI(file_path):
    """Read a versioned ROI workbook or safely migrate the legacy three-column form.

    The first sheet remains exactly ``命名``, ``形状``, ``描述``. Versioned
    workbooks add an ``ROI_Metadata`` sheet with structured geometry and
    coordinate-reference data. Files without that sheet are parsed through the
    strict legacy whitelist in :mod:`roi_contract`.
    """
    path = Path(file_path)
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"文件必须是 .xlsx 格式: {path}")
    if not path.is_file():
        raise ValueError(f"ROI 文件不存在: {path}")

    try:
        workbook = pd.read_excel(path, sheet_name=None)
    except Exception as exc:
        raise ValueError(f"读取 Excel 文件失败: {exc}") from exc
    if not workbook:
        raise ValueError("ROI 工作簿没有工作表。")

    first_sheet_name = next(iter(workbook))
    first_sheet = workbook[first_sheet_name]
    column_rule = check_columns(
        sheet_index=1, expected_cols=_REQUIRED_COLUMNS, mode="exact"
    )
    if not column_rule(workbook):
        actual = list(first_sheet.columns)
        raise ValueError(
            f"Excel 首张工作表列名必须精确为 {_REQUIRED_COLUMNS}，当前为 {actual}"
        )

    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = workbook.get(_ROI_SHEET)
    if metadata is None:
        if any(str(sheet).startswith("ROI_") for sheet in workbook if sheet != first_sheet_name):
            raise ValueError("工作簿包含未知 ROI_* 版本工作表，不能按旧格式回退解析。")
        rois = _read_legacy_rows(first_sheet, source_hash)
    else:
        rois = _read_versioned_rows(first_sheet, metadata, source_hash)

    if not rois:
        raise ValueError("ROI 工作簿没有有效的兴趣区行。")
    return rois


def _read_legacy_rows(frame: pd.DataFrame, source_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row, (_, row) in enumerate(frame.iterrows()):
        name_value = row["命名"]
        shape_value = row["形状"]
        description_value = row["描述"]
        if all(_cell_is_missing(value) for value in (name_value, shape_value, description_value)):
            continue
        name = _required_text(name_value, "命名", source_row + 2)
        shape = _required_text(shape_value, "形状", source_row + 2).lower()
        description = _required_text(description_value, "描述", source_row + 2)
        seed = f"sha256:{source_hash}:row:{source_row}:name:{name}"
        try:
            parsed = parse_legacy_description(
                description,
                shape_hint=shape,
                identity_seed=seed,
                reference=_unknown_reference(),
                name=name,
                priority=len(rows),
            )
        except (LegacyROIParseError, ROIValidationError) as exc:
            reason = getattr(exc, "reason", str(exc))
            raise ValueError(
                f"旧 ROI 第 {source_row + 2} 行（{name}）的描述无法安全迁移：{reason}"
            ) from exc

        record = parsed.to_dict()
        record["source"] = {
            **(record.get("source") or {}),
            "kind": "legacy_excel",
            "workbook_sha256": source_hash,
            "row_index": source_row,
            "original_description": description,
        }
        roi = ROI.from_dict(record)
        record = _consumer_record(roi, description, legacy=True, row_index=source_row)
        record.pop("_row_index", None)
        rows.append(record)
    return rows


def _read_versioned_rows(
    first_sheet: pd.DataFrame,
    metadata: pd.DataFrame,
    source_hash: str,
) -> list[dict[str, Any]]:
    missing_columns = sorted(_METADATA_COLUMNS - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"ROI_Metadata 缺少必需列：{missing_columns}")

    first_rows: list[tuple[int, str, str, str]] = []
    for source_row, (_, row) in enumerate(first_sheet.iterrows()):
        values = (row["命名"], row["形状"], row["描述"])
        if all(_cell_is_missing(value) for value in values):
            continue
        first_rows.append(
            (
                source_row,
                _required_text(values[0], "命名", source_row + 2),
                _required_text(values[1], "形状", source_row + 2).lower(),
                _required_text(values[2], "描述", source_row + 2),
            )
        )

    metadata_by_index: dict[int, pd.Series] = {}
    for _, row in metadata.iterrows():
        if all(_cell_is_missing(value) for value in row.tolist()):
            continue
        row_index = _required_int(row["row_index"], "row_index")
        if row_index in metadata_by_index:
            raise ValueError(f"ROI_Metadata 中 row_index {row_index} 重复。")
        metadata_by_index[row_index] = row

    first_indices = {source_row for source_row, *_ in first_rows}
    if set(metadata_by_index) != first_indices:
        missing = sorted(first_indices - set(metadata_by_index))
        extra = sorted(set(metadata_by_index) - first_indices)
        raise ValueError(
            f"首张表与 ROI_Metadata 行数或位置不一致（缺少元数据行={missing}，额外行={extra}）。"
        )

    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for source_row, name, shape, description in first_rows:
        metadata_row = metadata_by_index[source_row]
        schema_version = _required_int(metadata_row["schema_version"], "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"ROI_Metadata 第 {source_row + 2} 行版本不受支持：{schema_version}"
            )
        roi_id = _required_text(metadata_row["roi_id"], "roi_id", source_row + 2)
        priority = _required_int(metadata_row["priority"], "priority")
        boundary_mode = _required_text(
            metadata_row["boundary_mode"], "boundary_mode", source_row + 2
        )
        if roi_id in seen_ids:
            raise ValueError(f"ROI_Metadata 中 ROI ID 重复：{roi_id}")
        if name in seen_names:
            raise ValueError(f"ROI 名称重复：{name}")
        seen_ids.add(roi_id)
        seen_names.add(name)

        geometry = _json_object(metadata_row["geometry_json"], "geometry_json")
        source = _json_object(metadata_row["source_json"], "source_json")
        reference = _reference_from_row(metadata_row)
        mapping = _json_value(metadata_row["mapping_json"], "mapping_json", allow_blank=True)
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError(f"ROI_Metadata 第 {source_row + 2} 行 mapping_json 必须是对象或 null。")

        try:
            roi = ROI.from_dict(
                {
                    "schema_version": schema_version,
                    "id": roi_id,
                    "name": name,
                    "shape": shape,
                    "geometry": geometry,
                    "priority": priority,
                    "boundary_mode": boundary_mode,
                    "source": source,
                    "coordinate_reference": reference,
                    "mapping": mapping,
                }
            )
        except ROIValidationError as exc:
            raise ValueError(
                f"版本化 ROI 第 {source_row + 2} 行无效：{exc}"
            ) from exc

        expected_description = format_legacy_description(
            roi.shape, roi.geometry, roi.boundary_mode
        )
        if description != expected_description:
            raise ValueError(
                f"第 {source_row + 2} 行的“描述”与 ROI_Metadata 几何不一致；请核对后重新保存。"
            )
        if roi.name != name:
            raise ValueError(f"第 {source_row + 2} 行名称与 ROI_Metadata 不一致。")
        result.append(
            _consumer_record(roi, description, legacy=False, row_index=source_row)
        )

    result.sort(key=lambda item: (item["priority"], item["_row_index"]))
    for item in result:
        item["_roi"] = ROI.create(
            name=item["name"],
            shape=item["shape"],
            geometry=item["geometry"],
            id=item["id"],
            priority=item["priority"],
            boundary_mode=item["boundary_mode"],
            source=item["source"],
            coordinate_reference=item["coordinate_reference"],
            mapping=item["mapping"],
        )
        item.pop("_row_index", None)
    validate_roi_reference_compatibility([item["_roi"] for item in result])
    return result


def _consumer_record(
    roi: ROI,
    description: str,
    *,
    legacy: bool,
    row_index: int,
) -> dict[str, Any]:
    record = roi.to_dict()
    record["description"] = description
    record["_roi"] = roi
    record["_row_index"] = row_index
    record["migration_status"] = "legacy_migrated" if legacy else "versioned"
    record["migration_hint"] = (
        "旧矩形保持严格边界；旧圆和椭圆保持包含边界。显示器和校准参考尺寸未知，未进行缩放。"
        if legacy
        else "已读取版本化 ROI 几何和坐标元数据。"
    )
    return record


def _report_import_hints(roi_data: list[dict[str, Any]], logger) -> None:
    legacy_count = sum(item.get("migration_status") == "legacy_migrated" for item in roi_data)
    if not legacy_count:
        return
    message = (
        f"已安全迁移 {legacy_count} 个旧格式 ROI。旧矩形继续使用严格边界，旧圆/椭圆继续包含边界。"
        "旧文件没有显示器/校准坐标参考 metadata，因此坐标参考标记为 unknown，未按当前显示器自动缩放。"
    )
    if logger is not None:
        logger.info(message)
    try:
        from ..Popup.tkClass import custom_messagebox
    except ImportError:
        from Popup.tkClass import custom_messagebox
    custom_messagebox(title="旧 ROI 格式提示", message=message, autowh=True)


def _reference_from_row(row: pd.Series) -> dict[str, Any]:
    extras = _json_object(row["reference_extra_json"], "reference_extra_json", allow_blank=True)
    reference = {
        "space": _required_text(row["reference_space"], "reference_space"),
        "width": _optional_int(row["reference_width"], "reference_width"),
        "height": _optional_int(row["reference_height"], "reference_height"),
        "monitor_id": _optional_text(row["monitor_id"]),
        "monitor_origin_px": _json_value(
            row["monitor_origin_px_json"], "monitor_origin_px_json", allow_blank=True
        ),
        "dpi_scale": _json_value(row["dpi_scale_json"], "dpi_scale_json", allow_blank=True),
        "dpi_source": _required_text(row["dpi_source"], "dpi_source"),
        "source": _required_text(row["reference_source"], "reference_source"),
    }
    if reference["monitor_origin_px"] is not None and not _is_pair(reference["monitor_origin_px"]):
        raise ValueError("monitor_origin_px_json 必须是二元数组或 null。")
    if reference["dpi_scale"] is not None:
        if not _is_pair(reference["dpi_scale"]):
            raise ValueError("dpi_scale_json 必须是二元数组或 null。")
        if any(not _finite_positive(value) for value in reference["dpi_scale"]):
            raise ValueError("DPI scale 必须是有限正数。")
    overlap = _REFERENCE_KEYS.intersection(extras)
    if overlap:
        raise ValueError(f"reference_extra_json 不得覆盖标准字段：{sorted(overlap)}")
    reference.update(extras)
    return reference


def _unknown_reference() -> dict[str, Any]:
    return {
        "space": "unknown",
        "width": None,
        "height": None,
        "monitor_id": None,
        "monitor_origin_px": None,
        "dpi_scale": None,
        "dpi_source": "unknown",
        "source": "unknown",
    }


def _json_object(value: Any, field: str, *, allow_blank: bool = False) -> dict[str, Any]:
    parsed = _json_value(value, field, allow_blank=allow_blank)
    if parsed is None and allow_blank:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} 必须是 JSON 对象。")
    return parsed


def _json_value(value: Any, field: str, *, allow_blank: bool = False) -> Any:
    if _cell_is_missing(value):
        if allow_blank:
            return None
        raise ValueError(f"{field} 不能为空。")
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} 不是有效 JSON：{exc}") from exc


def _required_text(value: Any, field: str, row_number: int | None = None) -> str:
    if _cell_is_missing(value):
        location = f"第 {row_number} 行的" if row_number is not None else ""
        raise ValueError(f"ROI {location}{field} 不能为空。")
    text = str(value).strip()
    if not text:
        location = f"第 {row_number} 行的" if row_number is not None else ""
        raise ValueError(f"ROI {location}{field} 不能为空。")
    return text


def _optional_text(value: Any) -> str | None:
    if _cell_is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _required_int(value: Any, field: str) -> int:
    if _cell_is_missing(value):
        raise ValueError(f"{field} 不能为空。")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是整数。") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} 必须是有限整数。")
    return int(numeric)


def _optional_int(value: Any, field: str) -> int | None:
    if _cell_is_missing(value):
        return None
    result = _required_int(value, field)
    if result <= 0:
        raise ValueError(f"{field} 必须是正数或空值。")
    return result


def _cell_is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0
