"""Versioned local JSON presets for the desktop workbench.

This module stays at the Tk boundary: it reads and writes variable-like
objects without importing the preprocessing runtime or starting any work.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from EyeEvaluate.Evaluate.analysis_options import (
    ANALYSIS_METRIC_GROUP_IDS,
    normalize_analysis_config,
    normalize_selected_metrics,
    normalize_table_format,
)


PREPROCESS_PARAMETER_DEFAULTS = {
    "blink_padding_s": "0.050",
    "max_time_gap_s": "",
    "smooth_window_s": "0.060",
    "smooth_max_points": "5",
    "quality_window_s": "0.500",
    "quality_step_s": "0.100",
    "missing_ratio_threshold": "0.50",
}

PREPROCESS_PARAMETER_LABELS = {
    "blink_padding_s": "闭眼扩展（秒）",
    "max_time_gap_s": "最大时间间隔（秒；留空自动）",
    "smooth_window_s": "平滑窗口（秒）",
    "smooth_max_points": "平滑最多样本点数（个）",
    "quality_window_s": "质量窗口（秒）",
    "quality_step_s": "质量步长（秒）",
    "missing_ratio_threshold": "缺失比例阈值（0–1）",
}

_SCHEMA = "localgaze.desktop-config"
_VERSION = 1
_PREPROCESS_UNIT_IDS = {"seconds", "milliseconds", "microseconds"}
_ANALYSIS_TIME_UNIT_IDS = {"auto", "seconds", "milliseconds", "microseconds"}
_TABLE_FORMAT_IDS = {"csv", "txt"}
_CALIBRATION_ESTIMATOR_IDS = {"ridge", "elastic_net", "linear_svr", "svr", "tiny_mlp"}
_CALIBRATION_FORMULA_IDS = {
    "eyetrax-compact-eye-head-v2",
    "eyetrax-landmark-features-v1",
}
_ANALYSIS_UI_PARAMETER_FIELDS = {
    "max_time_gap_s",
    "gap_median_multiplier",
    "fixation_min_samples",
    "fixation_min_duration_s",
    "fixation_dispersion_diagonal_ratio",
    "fast_speed_diagonals_s",
    "ivt_speed_diagonals_s",
    "ivt_min_duration_s",
    "ivt_max_duration_s",
    "ivt_min_samples",
    "blink_min_duration_s",
    "blink_max_duration_s",
    "entropy_grid_width",
    "entropy_grid_height",
}
_ANALYSIS_PARAMETER_FIELDS = {
    "max_time_gap_s",
    "gap_median_multiplier",
    "fixation_min_samples",
    "fixation_min_duration_s",
    "fixation_dispersion_diagonal_ratio",
    "fast_speed_diagonals_s",
    "ivt_speed_diagonals_s",
    "ivt_min_duration_s",
    "ivt_max_duration_s",
    "ivt_min_samples",
    "blink_min_duration_s",
    "blink_max_duration_s",
    "entropy_grid",
}
_ANALYSIS_INTEGER_FIELDS = {"fixation_min_samples", "ivt_min_samples"}
_TIME_UNIT_LABEL_TO_ID = {
    "自动识别": "auto",
    "秒 (seconds)": "seconds",
    "毫秒 (milliseconds)": "milliseconds",
    "微秒 (microseconds)": "microseconds",
}
_TIME_UNIT_ID_TO_LABEL = {value: key for key, value in _TIME_UNIT_LABEL_TO_ID.items()}
_CALIBRATION_ESTIMATOR_LABEL_TO_ID = {
    "Ridge（默认）": "ridge",
    "Elastic Net": "elastic_net",
    "Linear SVR": "linear_svr",
    "SVR": "svr",
    "Tiny MLP": "tiny_mlp",
}
_CALIBRATION_ESTIMATOR_ID_TO_LABEL = {
    value: key for key, value in _CALIBRATION_ESTIMATOR_LABEL_TO_ID.items()
}
_CALIBRATION_FORMULA_LABEL_TO_ID = {
    "v2（默认，紧凑眼部与头部特征）": "eyetrax-compact-eye-head-v2",
    "v1（旧版地标特征）": "eyetrax-landmark-features-v1",
}
_CALIBRATION_FORMULA_ID_TO_LABEL = {
    value: key for key, value in _CALIBRATION_FORMULA_LABEL_TO_ID.items()
}
_TABLE_FORMAT_LABEL_TO_ID = {
    "CSV（逗号分隔）": "csv",
    "TXT（制表符分隔）": "txt",
}
_TABLE_FORMAT_ID_TO_LABEL = {value: key for key, value in _TABLE_FORMAT_LABEL_TO_ID.items()}


def normalize_preprocess_parameters(
    values: Mapping[str, object], *, allow_strings: bool = True
) -> dict[str, float | int | None]:
    """Validate GUI or typed worker values once and return numeric parameters."""
    if not isinstance(values, Mapping):
        raise ValueError("preprocess parameters must be an object")
    unknown = set(values).difference(PREPROCESS_PARAMETER_DEFAULTS)
    if unknown:
        raise ValueError("unknown preprocess parameters: " + ", ".join(sorted(str(key) for key in unknown)))
    values = {**PREPROCESS_PARAMETER_DEFAULTS, **values}

    result: dict[str, float | int | None] = {}
    for key in ("blink_padding_s", "smooth_window_s", "quality_window_s", "quality_step_s"):
        result[key] = _positive_number(values[key], key, allow_strings=allow_strings)

    max_gap = values["max_time_gap_s"]
    if max_gap is None or (allow_strings and isinstance(max_gap, str) and not max_gap.strip()):
        result["max_time_gap_s"] = None
    else:
        result["max_time_gap_s"] = _positive_number(
            max_gap, "max_time_gap_s", allow_strings=allow_strings
        )

    raw_points = values["smooth_max_points"]
    if type(raw_points) is int:
        points = raw_points
    elif allow_strings and isinstance(raw_points, str):
        try:
            points = int(raw_points.strip())
        except ValueError as exc:
            raise ValueError("smooth_max_points must be an integer >= 1") from exc
    else:
        raise ValueError("smooth_max_points must be an integer >= 1")
    if points < 1:
        raise ValueError("smooth_max_points must be an integer >= 1")
    result["smooth_max_points"] = points

    ratio = _finite_number(
        values["missing_ratio_threshold"], "missing_ratio_threshold", allow_strings=allow_strings
    )
    if not 0 <= ratio <= 1:
        raise ValueError("missing_ratio_threshold must be between 0 and 1")
    result["missing_ratio_threshold"] = ratio
    return result


def snapshot_configuration(app: Any) -> dict[str, object]:
    """Read and validate the current workbench values as a JSON-ready document."""
    preprocess_variables = _read_variable_mapping(
        app.preprocess_config_vars,
        set(PREPROCESS_PARAMETER_DEFAULTS),
        "preprocess_config_vars",
    )
    preprocess_parameters = normalize_preprocess_parameters(preprocess_variables)
    analysis_variables = _read_variable_mapping(
        app.analysis_config_vars,
        _ANALYSIS_UI_PARAMETER_FIELDS,
        "analysis_config_vars",
    )
    analysis_parameters = _analysis_parameters_from_variables(analysis_variables)

    selected_files = app._analysis_selected_files
    if not isinstance(selected_files, list) or any(not isinstance(path, Path) for path in selected_files):
        raise ValueError("_analysis_selected_files must be a list of paths")
    raw_metrics = _read_boolean_mapping(
        app.analysis_metric_vars,
        set(ANALYSIS_METRIC_GROUP_IDS),
        "analysis_metric_vars",
    )
    analysis_metrics = {metric_id: raw_metrics[metric_id] for metric_id in ANALYSIS_METRIC_GROUP_IDS}
    normalize_selected_metrics(
        [metric_id for metric_id in ANALYSIS_METRIC_GROUP_IDS if analysis_metrics[metric_id]]
    )

    return {
        "schema": _SCHEMA,
        "version": _VERSION,
        "session": {
            "session_id": _string_variable(app.session_id, "session.session_id"),
            "output_root": _string_variable(app.output_root, "session.output_root"),
        },
        "capture": {
            "snapshot_path": _string_variable(app.capture_snapshot_path, "capture.snapshot_path"),
            "context_path": _string_variable(app.capture_context_path, "capture.context_path"),
            "model_path": _string_variable(app.model_path, "capture.model_path"),
            "correction_path": _string_variable(app.correction_path, "capture.correction_path"),
            "use_correction_for_capture": _boolean_variable(
                app.use_correction_for_capture, "capture.use_correction_for_capture"
            ),
            "motion_monitor_enabled": _boolean_variable(
                app.motion_monitor_enabled, "capture.motion_monitor_enabled"
            ),
            "save_motion_keyframes": _boolean_variable(
                app.save_motion_keyframes, "capture.save_motion_keyframes"
            ),
            "target_x_px": _string_variable(app.target_x_px, "capture.target_x_px"),
            "target_y_px": _string_variable(app.target_y_px, "capture.target_y_px"),
            "calibration_estimator": _enum_from_label(
                app.calibration_estimator_label.get(),
                _CALIBRATION_ESTIMATOR_LABEL_TO_ID,
                "capture.calibration_estimator",
            ),
            "calibration_formula": _enum_from_label(
                app.calibration_formula_label.get(),
                _CALIBRATION_FORMULA_LABEL_TO_ID,
                "capture.calibration_formula",
            ),
            "table_format": _table_format_from_label(
                app.capture_table_format_label.get(), "capture.table_format"
            ),
        },
        "preprocessing": {
            "source_path": _string_variable(app.preprocess_source_path, "preprocessing.source_path"),
            "mapping_table_path": _string_variable(
                app.preprocess_mapping_table_path, "preprocessing.mapping_table_path"
            ),
            "time_unit": _enum_value(
                app.preprocess_time_unit.get(),
                _PREPROCESS_UNIT_IDS,
                "preprocessing.time_unit",
            ),
            "parameters": preprocess_parameters,
        },
        "roi": {"path": _string_variable(app.roi_path, "roi.path")},
        "analysis": {
            "source_path": _string_variable(app.analysis_source_path, "analysis.source_path"),
            "selected_files": [str(path) for path in selected_files],
            "recursive": _boolean_variable(app.analysis_recursive, "analysis.recursive"),
            "time_unit": _enum_from_label(
                app.analysis_time_unit_label.get(), _TIME_UNIT_LABEL_TO_ID, "analysis.time_unit"
            ),
            "table_format": _table_format_from_label(
                app.analysis_table_format_label.get(), "analysis.table_format"
            ),
            "metrics": analysis_metrics,
            "parameters": analysis_parameters,
        },
    }


def apply_configuration(app: Any, document: Mapping[str, object]) -> None:
    """Validate a complete preset before applying any value to the workbench."""
    if app._busy or app._closing or app._dependency_install_active:
        raise ValueError("configuration cannot be applied while another workbench operation is active")
    normalized = _validate_document(document)

    app.session_id.set(normalized["session"]["session_id"])
    app.output_root.set(normalized["session"]["output_root"])

    capture = normalized["capture"]
    app.capture_snapshot_path.set(capture["snapshot_path"])
    app.capture_context_path.set(capture["context_path"])
    app.model_path.set(capture["model_path"])
    app.correction_path.set(capture["correction_path"])
    app.use_correction_for_capture.set(capture["use_correction_for_capture"])
    app.motion_monitor_enabled.set(capture["motion_monitor_enabled"])
    app.save_motion_keyframes.set(capture["save_motion_keyframes"])
    app.target_x_px.set(capture["target_x_px"])
    app.target_y_px.set(capture["target_y_px"])
    app.calibration_estimator_label.set(
        _CALIBRATION_ESTIMATOR_ID_TO_LABEL[capture["calibration_estimator"]]
    )
    app.calibration_formula_label.set(_CALIBRATION_FORMULA_ID_TO_LABEL[capture["calibration_formula"]])
    app.capture_table_format_label.set(_TABLE_FORMAT_ID_TO_LABEL[capture["table_format"]])

    preprocessing = normalized["preprocessing"]
    app.preprocess_source_path.set(preprocessing["source_path"])
    app.preprocess_mapping_table_path.set(preprocessing["mapping_table_path"])
    app.preprocess_time_unit.set(preprocessing["time_unit"])
    for key, value in preprocessing["parameters"].items():
        app.preprocess_config_vars[key].set("" if value is None else str(value))

    app.roi_path.set(normalized["roi"]["path"])

    analysis = normalized["analysis"]
    # This StringVar has a write trace that clears the current multi-file selection.
    app.analysis_source_path.set(analysis["source_path"])
    app._analysis_selected_files = [Path(value) for value in analysis["selected_files"]]
    app.analysis_source_selection_text.set(
        f"多文件模式 · 已选择 {len(app._analysis_selected_files)} 个文件"
        if app._analysis_selected_files
        else "单文件或文件夹模式"
    )
    app.analysis_recursive.set(analysis["recursive"])
    app.analysis_time_unit_label.set(_TIME_UNIT_ID_TO_LABEL[analysis["time_unit"]])
    app.analysis_table_format_label.set(_TABLE_FORMAT_ID_TO_LABEL[analysis["table_format"]])
    for metric_id, selected in analysis["metrics"].items():
        app.analysis_metric_vars[metric_id].set(selected)
    for key, value in _analysis_parameters_to_variables(analysis["parameters"]).items():
        app.analysis_config_vars[key].set(value)

    app._refresh_analysis_parameter_states()
    app._sync_capture_retry_button()


def load_configuration(path: str | os.PathLike[str], app: Any) -> None:
    """Read a UTF-8 JSON preset and apply it only after full validation."""
    with Path(path).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    apply_configuration(app, document)


def save_configuration(path: str | os.PathLike[str], app: Any) -> None:
    """Atomically save the current validated workbench values as UTF-8 JSON."""
    document = snapshot_configuration(app)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_document(document: Mapping[str, object]) -> dict[str, Any]:
    document = _require_object(document, "configuration")
    _require_exact_keys(
        document,
        {"schema", "version", "session", "capture", "preprocessing", "roi", "analysis"},
        "configuration",
    )
    schema = _require_string(document["schema"], "configuration.schema")
    if schema != _SCHEMA:
        raise ValueError(f"unsupported configuration schema: {schema!r}")
    if type(document["version"]) is not int or document["version"] != _VERSION:
        raise ValueError(f"unsupported configuration version: {document['version']!r}")

    session = _require_object(document["session"], "session")
    _require_exact_keys(session, {"session_id", "output_root"}, "session")
    normalized_session = {
        "session_id": _require_string(session["session_id"], "session.session_id"),
        "output_root": _require_string(session["output_root"], "session.output_root"),
    }

    capture = _require_object(document["capture"], "capture")
    _require_exact_keys(
        capture,
        {
            "snapshot_path",
            "context_path",
            "model_path",
            "correction_path",
            "use_correction_for_capture",
            "motion_monitor_enabled",
            "save_motion_keyframes",
            "target_x_px",
            "target_y_px",
            "calibration_estimator",
            "calibration_formula",
            "table_format",
        },
        "capture",
    )
    normalized_capture = {
        key: _require_string(capture[key], f"capture.{key}")
        for key in ("snapshot_path", "context_path", "model_path", "correction_path", "target_x_px", "target_y_px")
    }
    for key in ("use_correction_for_capture", "motion_monitor_enabled", "save_motion_keyframes"):
        normalized_capture[key] = _require_bool(capture[key], f"capture.{key}")
    normalized_capture["calibration_estimator"] = _enum_value(
        capture["calibration_estimator"], _CALIBRATION_ESTIMATOR_IDS, "capture.calibration_estimator"
    )
    normalized_capture["calibration_formula"] = _enum_value(
        capture["calibration_formula"], _CALIBRATION_FORMULA_IDS, "capture.calibration_formula"
    )
    normalized_capture["table_format"] = _table_format_id(capture["table_format"], "capture.table_format")

    preprocessing = _require_object(document["preprocessing"], "preprocessing")
    _require_exact_keys(
        preprocessing, {"source_path", "mapping_table_path", "time_unit", "parameters"}, "preprocessing"
    )
    raw_preprocess_parameters = _require_object(preprocessing["parameters"], "preprocessing.parameters")
    _require_exact_keys(raw_preprocess_parameters, set(PREPROCESS_PARAMETER_DEFAULTS), "preprocessing.parameters")
    normalized_preprocessing = {
        "source_path": _require_string(preprocessing["source_path"], "preprocessing.source_path"),
        "mapping_table_path": _require_string(
            preprocessing["mapping_table_path"], "preprocessing.mapping_table_path"
        ),
        "time_unit": _enum_value(
            preprocessing["time_unit"], _PREPROCESS_UNIT_IDS, "preprocessing.time_unit"
        ),
        "parameters": normalize_preprocess_parameters(raw_preprocess_parameters, allow_strings=False),
    }

    roi = _require_object(document["roi"], "roi")
    _require_exact_keys(roi, {"path"}, "roi")
    normalized_roi = {"path": _require_string(roi["path"], "roi.path")}

    analysis = _require_object(document["analysis"], "analysis")
    _require_exact_keys(
        analysis,
        {"source_path", "selected_files", "recursive", "time_unit", "table_format", "metrics", "parameters"},
        "analysis",
    )
    selected_files = analysis["selected_files"]
    if not isinstance(selected_files, list):
        raise ValueError("analysis.selected_files must be an array of path strings")
    normalized_files = [
        _require_string(value, f"analysis.selected_files[{index}]")
        for index, value in enumerate(selected_files)
    ]
    metrics = _require_object(analysis["metrics"], "analysis.metrics")
    _require_exact_keys(metrics, set(ANALYSIS_METRIC_GROUP_IDS), "analysis.metrics")
    normalized_metrics = {
        metric_id: _require_bool(metrics[metric_id], f"analysis.metrics.{metric_id}")
        for metric_id in ANALYSIS_METRIC_GROUP_IDS
    }
    normalize_selected_metrics(
        [metric_id for metric_id in ANALYSIS_METRIC_GROUP_IDS if normalized_metrics[metric_id]]
    )
    normalized_analysis = {
        "source_path": _require_string(analysis["source_path"], "analysis.source_path"),
        "selected_files": normalized_files,
        "recursive": _require_bool(analysis["recursive"], "analysis.recursive"),
        "time_unit": _enum_value(analysis["time_unit"], _ANALYSIS_TIME_UNIT_IDS, "analysis.time_unit"),
        "table_format": _table_format_id(analysis["table_format"], "analysis.table_format"),
        "metrics": normalized_metrics,
        "parameters": _validate_analysis_parameters(analysis["parameters"]),
    }
    return {
        "session": normalized_session,
        "capture": normalized_capture,
        "preprocessing": normalized_preprocessing,
        "roi": normalized_roi,
        "analysis": normalized_analysis,
    }


def _analysis_parameters_from_variables(values: Mapping[str, object]) -> dict[str, object]:
    def number(key: str, *, optional: bool = False) -> float | None:
        raw = _require_string(values[key], f"analysis_config_vars.{key}")
        if optional and not raw.strip():
            return None
        return _parse_gui_float(raw, f"analysis_config_vars.{key}")

    def integer(key: str) -> int:
        raw = _require_string(values[key], f"analysis_config_vars.{key}")
        try:
            return int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"analysis_config_vars.{key} must be an integer") from exc

    parameters = {
        "max_time_gap_s": number("max_time_gap_s", optional=True),
        "gap_median_multiplier": number("gap_median_multiplier"),
        "fixation_min_samples": integer("fixation_min_samples"),
        "fixation_min_duration_s": number("fixation_min_duration_s"),
        "fixation_dispersion_diagonal_ratio": number("fixation_dispersion_diagonal_ratio"),
        "fast_speed_diagonals_s": number("fast_speed_diagonals_s"),
        "ivt_speed_diagonals_s": number("ivt_speed_diagonals_s"),
        "ivt_min_duration_s": number("ivt_min_duration_s"),
        "ivt_max_duration_s": number("ivt_max_duration_s"),
        "ivt_min_samples": integer("ivt_min_samples"),
        "blink_min_duration_s": number("blink_min_duration_s"),
        "blink_max_duration_s": number("blink_max_duration_s"),
        "entropy_grid": [integer("entropy_grid_width"), integer("entropy_grid_height")],
    }
    return _validate_analysis_parameters(parameters)


def _validate_analysis_parameters(value: object) -> dict[str, object]:
    parameters = _require_object(value, "analysis.parameters")
    _require_exact_keys(parameters, _ANALYSIS_PARAMETER_FIELDS, "analysis.parameters")
    for key, raw in parameters.items():
        if key == "entropy_grid":
            if not isinstance(raw, (list, tuple)) or len(raw) != 2 or any(type(item) is not int for item in raw):
                raise ValueError("analysis.parameters.entropy_grid must contain two integers")
        elif key in _ANALYSIS_INTEGER_FIELDS:
            if type(raw) is not int:
                raise ValueError(f"analysis.parameters.{key} must be an integer")
        elif key == "max_time_gap_s":
            if raw is not None and type(raw) not in (int, float):
                raise ValueError("analysis.parameters.max_time_gap_s must be a number or null")
        elif type(raw) not in (int, float):
            raise ValueError(f"analysis.parameters.{key} must be a number")
    normalized = normalize_analysis_config(parameters)
    normalized["entropy_grid"] = list(normalized["entropy_grid"])
    return normalized


def _analysis_parameters_to_variables(parameters: Mapping[str, object]) -> dict[str, str]:
    return {
        "max_time_gap_s": "" if parameters["max_time_gap_s"] is None else str(parameters["max_time_gap_s"]),
        "gap_median_multiplier": str(parameters["gap_median_multiplier"]),
        "fixation_min_samples": str(parameters["fixation_min_samples"]),
        "fixation_min_duration_s": str(parameters["fixation_min_duration_s"]),
        "fixation_dispersion_diagonal_ratio": str(parameters["fixation_dispersion_diagonal_ratio"]),
        "fast_speed_diagonals_s": str(parameters["fast_speed_diagonals_s"]),
        "ivt_speed_diagonals_s": str(parameters["ivt_speed_diagonals_s"]),
        "ivt_min_duration_s": str(parameters["ivt_min_duration_s"]),
        "ivt_max_duration_s": str(parameters["ivt_max_duration_s"]),
        "ivt_min_samples": str(parameters["ivt_min_samples"]),
        "blink_min_duration_s": str(parameters["blink_min_duration_s"]),
        "blink_max_duration_s": str(parameters["blink_max_duration_s"]),
        "entropy_grid_width": str(parameters["entropy_grid"][0]),
        "entropy_grid_height": str(parameters["entropy_grid"][1]),
    }


def _read_variable_mapping(variables: object, expected_keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(variables, Mapping):
        raise ValueError(f"{label} must be an object of variables")
    _require_exact_keys(variables, expected_keys, label)
    return {key: variables[key].get() for key in expected_keys}


def _read_boolean_mapping(variables: object, expected_keys: set[str], label: str) -> dict[str, bool]:
    if not isinstance(variables, Mapping):
        raise ValueError(f"{label} must be an object of variables")
    _require_exact_keys(variables, expected_keys, label)
    return {key: _boolean_variable(variables[key], f"{label}.{key}") for key in expected_keys}


def _string_variable(variable: object, label: str) -> str:
    return _require_string(variable.get(), label)


def _boolean_variable(variable: object, label: str) -> bool:
    return _require_bool(variable.get(), label)


def _require_object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        unknown = sorted((str(key) for key in actual.difference(expected)))
        missing = sorted(expected.difference(actual))
        details = []
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        raise ValueError(f"{label} has " + "; ".join(details))


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _enum_value(value: object, choices: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return value


def _enum_from_label(value: object, labels: Mapping[str, str], label: str) -> str:
    if not isinstance(value, str) or value not in labels:
        raise ValueError(f"{label} has an unsupported selection")
    return labels[value]


def _table_format_from_label(value: object, label: str) -> str:
    if not isinstance(value, str) or value not in _TABLE_FORMAT_LABEL_TO_ID:
        raise ValueError(f"{label} has an unsupported selection")
    return normalize_table_format(_TABLE_FORMAT_LABEL_TO_ID[value])


def _table_format_id(value: object, label: str) -> str:
    selected = _enum_value(value, _TABLE_FORMAT_IDS, label)
    return normalize_table_format(selected)


def _positive_number(value: object, label: str, *, allow_strings: bool = True) -> float:
    number = _finite_number(value, label, allow_strings=allow_strings)
    if number <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _finite_number(value: object, label: str, *, allow_strings: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        if allow_strings and isinstance(value, str):
            number = float(value.strip())
        elif type(value) in (int, float):
            number = float(value)
        else:
            raise ValueError
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _parse_gui_float(value: str, label: str) -> float:
    if not value.strip():
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


__all__ = [
    "PREPROCESS_PARAMETER_DEFAULTS",
    "PREPROCESS_PARAMETER_LABELS",
    "normalize_preprocess_parameters",
    "snapshot_configuration",
    "apply_configuration",
    "load_configuration",
    "save_configuration",
]
