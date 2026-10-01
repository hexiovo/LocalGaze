from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import CalibrationResidualResult, calibration_gate_config_record


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item))


def write_calibration_status(output_root, session_id, failure, *, feature_formula_version=None):
    """Persist a failed calibration without replacing any prior session file."""

    if not isinstance(session_id, str) or not session_id or Path(session_id).name != session_id or session_id in {".", ".."}:
        raise ValueError("session_id must be a single filename component")
    output_root = Path(output_root).expanduser().resolve()
    path = output_root / f"{session_id}.calibration-status.json"
    failure_data = dict(failure or {})
    payload = {
        "schema": "localgaze.calibration-status/1",
        "session_id": session_id,
        "status": "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "feature_formula_version": feature_formula_version,
        "gate_config": failure_data.get("gate_config") or calibration_gate_config_record(),
        "failure": failure_data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create is authoritative: never replace another session's status.
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return path.resolve()


def _point_rows(result, *, mean, round_number=None):
    rows = []
    if isinstance(result, (list, tuple)) and all(isinstance(item, CalibrationResidualResult) for item in result):
        for item in result:
            rows.extend(_point_rows(item, mean=item.mean_residual_px, round_number=round_number))
        return rows
    if isinstance(result, CalibrationResidualResult):
        for index, target in enumerate(result.targets, 1):
            rows.append({
                "Index": index, "Point": f"Point{index}", "Error": target.mean_residual_px,
                "MeanError": result.mean_residual_px, "MeanErrorPerRound": result.mean_residual_px,
                "Round": round_number, "TargetX": target.target_id[0], "TargetY": target.target_id[1],
                "VisitID": result.visit_id, "SampleCount": target.sample_count,
                "FrameResidualsPx": _json(target.frame_residuals_px),
                "ResidualKind": result.residual_kind, "MethodVersion": result.method_version,
            })
        return rows
    # A list/array is a point's complete frame residual vector, never (index,error).
    values_by_point = [] if result is None else ([result] if np.isscalar(result) else result)
    for index, value in enumerate(values_by_point, 1):
        values = np.asarray(value, dtype=float).reshape(-1)
        if values.size and not np.all(np.isfinite(values)):
            raise ValueError("Calibration residuals must be finite")
        rows.append({
            "Index": index, "Point": f"Point{index}",
            "Error": float(np.mean(values)) if values.size else None,
            "MeanError": mean, "MeanErrorPerRound": mean, "Round": round_number,
            "SampleCount": int(values.size), "FrameResidualsPx": _json(values),
            "ResidualKind": "legacy_residual_unversioned", "MethodVersion": "legacy_grouping_unspecified",
        })
    return rows


def save_calibration_results(model_path: Path, errors_per_point_9, errors_per_point_add,
                             mean_error_9, mean_errors_add, model_names=None, model_kwargs=None,
                             *, model_metadata=None, overwrite=False):
    """Save complete residual vectors and explicit groups; protect prior files.

    Historical sheets/columns remain readable. Error/MeanError are fit residuals
    for new results and never independent webcam accuracy.
    """
    excel_path = Path(model_path).with_suffix(".xlsx")
    if excel_path.exists() and not overwrite:
        raise FileExistsError(f"Calibration results already exist: {excel_path}")
    nine = _point_rows(errors_per_point_9, mean=mean_error_9)
    additional = []
    means = [] if mean_errors_add is None else np.atleast_1d(mean_errors_add).tolist()
    rounds = [] if errors_per_point_add is None else errors_per_point_add
    if isinstance(rounds, CalibrationResidualResult):
        rounds = [rounds]
    for number, result in enumerate(rounds, 1):
        round_mean = means[number - 1] if number <= len(means) else None
        additional.extend(_point_rows(result, mean=round_mean, round_number=number))
    columns = ["Index", "Point", "Error", "MeanError", "MeanErrorPerRound", "Round",
               "TargetX", "TargetY", "VisitID", "SampleCount", "FrameResidualsPx", "ResidualKind", "MethodVersion"]
    nine_df, additional_df = pd.DataFrame(nine, columns=columns), pd.DataFrame(additional, columns=columns)
    if not additional_df.empty:
        additional_df["Index"] = range(1, len(additional_df) + 1)
    names = model_names if isinstance(model_names, (list, tuple)) else [model_names]
    kwargs = model_kwargs if isinstance(model_kwargs, (list, tuple)) else [model_kwargs]
    info_rows = [{"Model Name": name, "Model kwargs": _json(kwargs[min(index, len(kwargs) - 1)]) if kwargs else "null"}
                 for index, name in enumerate(names)]
    metadata = dict(model_metadata or {})
    training = dict(metadata.get("training", {}))
    ledger = training.pop("calibration_ledger", [])
    captures = training.pop("calibration_captures", [])
    metadata["training"] = training
    methods = [{"Key": key, "Value": _json(value)} for key, value in metadata.items()]
    methods.extend([
        {"Key": "residual_interpretation", "Value": "training_fit_residual; not independently measured accuracy"},
        {"Key": "internal_holdout", "Value": _json(training.get("visit_holdout", {"value_px": None, "reason": "not_available"}))},
    ])
    frame_rows = []
    receipt_rows = []
    coverage_rows = []
    for capture in captures:
        for sample in capture.get("samples", []):
            frame_rows.append({
                key: _json(value) if isinstance(value, (dict, list, tuple, np.ndarray)) else value
                for key, value in sample.items()
            })
        for receipt in capture.get("receipts", []):
            receipt_rows.append({
                key: _json(value) if isinstance(value, (dict, list, tuple, np.ndarray)) else value
                for key, value in receipt.items()
            })
        for status in capture.get("target_statuses", []):
            row = dict(status)
            row.setdefault("visit_id", capture.get("visit_id"))
            row.setdefault("visit_index", capture.get("visit_index"))
            row["gate_config"] = _json(capture.get("gate_config", {}))
            coverage_rows.append({
                key: _json(value) if isinstance(value, (dict, list, tuple, np.ndarray)) else value
                for key, value in row.items()
            })
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive open avoids a race between the name check and final output.
    with excel_path.open("wb" if overwrite else "xb") as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            nine_df.to_excel(writer, sheet_name="NinePointCalibration", index=False)
            additional_df.to_excel(writer, sheet_name="AdditionalCalibration", index=False)
            pd.DataFrame(info_rows, columns=["Model Name", "Model kwargs"]).to_excel(writer, sheet_name="model_info", index=False)
            pd.DataFrame(methods).to_excel(writer, sheet_name="CalibrationMethods", index=False)
            if ledger:
                pd.DataFrame([{key: _json(value) if isinstance(value, (list, tuple, dict)) else value
                               for key, value in row.items()} for row in ledger]).to_excel(writer, sheet_name="CalibrationLedger", index=False)
            if frame_rows:
                frame_df = pd.DataFrame(frame_rows)
                for name in frame_df.columns:
                    if name.endswith("monotonic_ns"):
                        frame_df[name] = pd.Series([
                            str(int(row[name])) if pd.notna(row.get(name)) else None
                            for row in frame_rows
                        ], dtype=object)
                frame_df.to_excel(writer, sheet_name="CalibrationFrames", index=False)
            if receipt_rows:
                pd.DataFrame(receipt_rows).to_excel(writer, sheet_name="CalibrationReceipts", index=False)
            if coverage_rows:
                pd.DataFrame(coverage_rows).to_excel(writer, sheet_name="CalibrationCoverage", index=False)
    print(f"校准拟合残差与方法记录已保存至：{excel_path}")
    return excel_path


def draw_gaze_cloud(frame, x, y, radius=30, color_gray=(180, 180, 180),
                    color_center=(0, 0, 255), border_color=(0, 0, 0), center_radius=4, alpha=0.4):
    overlay = frame.copy()
    cv2.circle(overlay, (int(x), int(y)), radius, color_gray, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.circle(frame, (int(x), int(y)), radius, border_color, 1)
    cv2.circle(frame, (int(x), int(y)), center_radius, color_center, -1)
