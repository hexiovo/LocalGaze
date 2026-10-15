from eyetrax.gaze import GazeEstimator
from eyetrax.calibration import run_9_point_calibration,run_additional_random_calibration,save_calibration_results
from eyetrax.calibration.misc import write_calibration_status
from eyetrax.features import DEFAULT_FEATURE_FORMULA_VERSION, FEATURE_FORMULA_V1, FEATURE_FORMULA_V2
import tkinter.font as tkFont
import tkinter as tk
from tkinter import filedialog
from misc import *
from eyetracking import *
from Global_data import *
import numpy as np
import pandas as pd
import json
import datetime as datetime_module
from pathlib import Path
import sys
import os
import tempfile
import zipfile
import hashlib
import shutil
import traceback
from typing import Any, Mapping

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
from session_names import canonical_session_id
from capture_paths import (
    INTERNAL_LAYOUT,
    LEGACY_LAYOUT,
    assert_no_reparse_components,
    internal_directory,
    normalize_sidecar_layout,
    sidecar_path,
)
from desktop_jobs import calibration_export_recovery


# ---------- 封装工作流程 ----------
def run_eye_tracking_workflow(*, session_id=None, save_dir=None, workflow_mode="legacy", model_path=None,
                              model_name=None, feature_formula_version=None, correction_path=None,
                              motion_monitor_enabled=True, save_motion_keyframes=True,
                              table_format="csv"):
    """Keep the legacy UI, optionally accepting a canonical external request."""
    from tkinter import TclError
    requested_session = canonical_session_id(session_id) if session_id is not None else None
    requested_directory = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    if workflow_mode not in {"legacy", "calibrate", "capture"}:
        raise ValueError("workflow_mode must be legacy, calibrate or capture")
    if not isinstance(motion_monitor_enabled, bool) or not isinstance(save_motion_keyframes, bool):
        raise TypeError("motion monitor options must be bool values")
    if table_format not in {"csv", "txt"}:
        raise ValueError("table_format must be 'csv' or 'txt'")
    if model_name is not None and workflow_mode != "calibrate":
        raise ValueError("model_name can only select the estimator for calibration")
    if workflow_mode == "calibrate":
        model_name = "ridge" if model_name is None else model_name
        if not isinstance(model_name, str) or model_name not in MODEL_KWARGS:
            raise ValueError(f"unsupported calibration estimator: {model_name!r}")
        feature_formula_version = DEFAULT_FEATURE_FORMULA_VERSION if feature_formula_version is None else feature_formula_version
        if not isinstance(feature_formula_version, str) or feature_formula_version not in {FEATURE_FORMULA_V1, FEATURE_FORMULA_V2}:
            raise ValueError(f"unsupported calibration feature formula: {feature_formula_version!r}")
    if workflow_mode == "capture" and not model_path:
        raise ValueError("capture requires an existing model_path")
    if workflow_mode == "capture" and feature_formula_version is not None:
        raise ValueError("capture uses the loaded model's recorded feature formula")
    if correction_path is not None and workflow_mode != "capture":
        raise ValueError("correction_path is only supported for capture")
    if workflow_mode == "capture":
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file() or model_path.suffix.lower() != ".pkl":
            raise ValueError("capture requires an existing .pkl model")
    if requested_session is not None and requested_directory is not None:
        if workflow_mode == "calibrate":
            _check_calibration_output_names(requested_directory, requested_session)
        else:
            _check_output_names(requested_directory, requested_session, (".csv", ".txt", ".xlsx", ".manifest.json"))
    root = tk.Tk()
    try:
        root.withdraw()
        outcome = _run_eye_tracking_workflow(
            root, requested_session_id=requested_session,
            requested_save_dir=requested_directory,
            workflow_mode=workflow_mode, requested_model_path=model_path,
            requested_model_name=model_name,
            feature_formula_version=feature_formula_version,
            correction_path=correction_path,
            motion_monitor_enabled=motion_monitor_enabled if workflow_mode == "capture" else False,
            save_motion_keyframes=save_motion_keyframes if workflow_mode == "capture" else False,
            table_format=table_format,
        )
        return outcome or {"status": "cancelled", "outputs": {}, "details": {"reason": "workflow_cancelled_before_capture"}}
    except (FileExistsError, ValueError, NotImplementedError) as exc:
        messagebox.showerror("采集已停止", str(exc))
        if session_id is not None or save_dir is not None or workflow_mode != "legacy":
            raise
        return {"status": "failed", "outputs": {}, "details": {"reason": str(exc)}}
    finally:
        try:
            root.destroy()
        except TclError:
            # Calibration may already have closed the shared Tk application.
            pass


def run_known_target_offset_workflow(*, session_id, save_dir, model_path,
                                     correction_path=None, target_xy_px=None,
                                     table_format="csv"):
    """Capture this job's displayed target and create a correction only if eligible."""
    from offset_workflow import (
        create_correction_from_target_capture,
        model_binding_for_estimator,
    )

    session_id = canonical_session_id(session_id)
    output_root = Path(save_dir).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file() or model_path.suffix.lower() != ".pkl":
        raise ValueError("known-target offset requires an existing .pkl model")

    # Query the actual current canvas before any capture files are created.
    probe = tk.Tk()
    try:
        probe.withdraw()
        current_canvas = (int(probe.winfo_screenwidth()), int(probe.winfo_screenheight()))
    finally:
        try:
            probe.destroy()
        except tk.TclError:
            pass

    estimator = GazeEstimator(model_name="ridge", model_kwargs=MODEL_KWARGS.get("ridge", {}))
    estimator.load_model(model_path)
    binding = model_binding_for_estimator(model_path, estimator)
    recorded_canvas = binding.calibration_canvas_size_px
    if recorded_canvas is None:
        return {
            "status": "failed", "message": "model calibration canvas is unavailable",
            "outputs": {}, "details": {"offset_status": "rejected", "offset_reason": "model_canvas_unavailable", "correction_json_written": False},
        }
    if current_canvas != tuple(recorded_canvas):
        return {
            "status": "failed", "message": "current display canvas differs from the model calibration canvas",
            "outputs": {}, "details": {
                "offset_status": "rejected", "offset_reason": "capture_canvas_mismatch",
                "model_canvas_size_px": list(recorded_canvas), "current_canvas_size_px": list(current_canvas),
                "correction_json_written": False,
            },
        }

    width, height = recorded_canvas
    if target_xy_px is None:
        target_xy_px = (width / 2.0, height / 2.0)
    try:
        target = np.asarray(target_xy_px, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target_xy_px must contain two finite coordinates") from exc
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target_xy_px must contain two finite coordinates")
    target_xy_px = (float(target[0]), float(target[1]))
    if not (0 <= target_xy_px[0] < width and 0 <= target_xy_px[1] < height):
        raise ValueError("target_xy_px is outside the recorded model canvas")

    if table_format not in {"csv", "txt"}:
        raise ValueError("table_format must be 'csv' or 'txt'")
    _check_output_names(
        output_root,
        session_id,
        (".csv", ".txt", ".xlsx", ".manifest.json", ".offset-correction.json"),
    )
    correction_destination = (
        Path(correction_path).expanduser().resolve()
        if correction_path is not None
        else sidecar_path(
            output_root,
            f"{session_id}.offset-correction.json",
            layout=INTERNAL_LAYOUT,
            create_parent=True,
        )
    )
    if correction_destination.suffix.lower() != ".json":
        raise ValueError("correction_path must end in .json")
    if correction_destination.exists():
        raise FileExistsError(f"correction file already exists: {correction_destination}")

    samples_path = output_root / f"{session_id}.{table_format}"
    try:
        eye_data = run_realtime_gaze(
            estimator=estimator,
            save_name=session_id,
            draw_cloud=False,
            screen_width=width,
            screen_height=height,
            camera_index=camera_num,
            save_dir=output_root,
            session_id=session_id,
            model_binding=binding,
            known_target_xy_px=target_xy_px,
            motion_monitor_enabled=False,
            save_motion_keyframes=False,
            table_format=table_format,
        )
    except Exception:
        # Capture exceptions are surfaced to the owned worker; no correction is
        # considered unless the completed manifest is later read and validated.
        raise

    capture_metadata = dict(getattr(estimator, "last_capture_metadata", {}))
    experiment_info = {
        "workflow_mode": ["known_target_offset"],
        "model_path": [str(model_path)],
        "session_id": [session_id],
        "target_xy_px": [list(target_xy_px)],
    }
    capture_export = save_capture_result_with_recovery(
        output_root,
        session_id,
        eye_data,
        experiment_info,
        capture_metadata,
        model_metadata=estimator.model.metadata,
    )
    capture_outputs = dict(capture_export.get("outputs") or {})
    if capture_export.get("status") != "completed":
        return {
            "status": "failed",
            "message": capture_export.get("message", "known-target capture export could not be verified"),
            "outputs": capture_outputs,
            "details": {
                **dict(capture_export.get("details") or {}),
                "offset_status": "rejected",
                "offset_reason": "capture_export_not_verified",
                "correction_json_written": False,
            },
        }

    try:
        correction_result = create_correction_from_target_capture(
            capture_path=samples_path,
            model_path=model_path,
            correction_path=correction_destination,
        )
    except Exception as exc:
        return {
            "status": "failed", "message": f"{type(exc).__name__}: {exc}",
            "outputs": capture_outputs,
            "details": {"offset_status": "rejected", "offset_reason": "correction_write_failed", "correction_json_written": False, "partial_outputs": True},
        }

    if correction_result.get("status") != "completed":
        return {
            "status": "failed",
            "message": correction_result.get("message", "known-target estimate was rejected"),
            "outputs": capture_outputs,
            "details": {**correction_result.get("details", {}), "capture_completed": capture_metadata.get("stop_reason") == "target_capture_window_complete", "partial_outputs": True},
        }
    return {
        "status": "completed",
        "message": "eligible correction record created",
        "outputs": {**capture_outputs, **correction_result["outputs"]},
        "details": {**correction_result.get("details", {}), "capture_stop_reason": capture_metadata.get("stop_reason")},
    }


def run_independent_validation_workflow(*, session_id, save_dir, model_path, table_format="csv"):
    """Collect one nine-point validation pass with an existing locked model."""
    import uuid

    from independent_validation import (
        OBSERVE_SECONDS,
        SETTLE_SECONDS,
        nine_point_targets,
        summarize_independent_validation,
    )
    from offset_workflow import model_binding_for_estimator
    from eyetrax.features import get_feature_schema

    session_id = canonical_session_id(session_id)
    output_root = assert_no_reparse_components(Path(save_dir).expanduser()).resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    if table_format not in {"csv", "txt"}:
        raise ValueError("table_format must be csv or txt")

    requested_model_path = Path(model_path).expanduser()
    capture_id = uuid.uuid4().hex
    report_path = sidecar_path(
        output_root,
        f"{session_id}.{capture_id}.independent-validation.json",
        layout=INTERNAL_LAYOUT,
        create_parent=True,
    )
    estimator = None
    binding = None
    model_canvas = None
    formula = None
    targets: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema": "localgaze.independent-validation-report/1",
        "status": "incomplete",
        "incomplete_reasons": ["capture_not_started"],
        "session_id": session_id,
        "capture_id": capture_id,
        "model": {
            "path": str(requested_model_path.resolve(strict=False)),
            "sha256": None,
            "feature_formula_version": None,
            "feature_schema": None,
            "calibration_canvas_size_px": None,
        },
        "canvas": {
            "current_screen_size_px": None,
            "coordinate_source": "Tk screen canvas; no visual-angle or viewing-distance conversion",
        },
        "parameters": {
            "target_count": 9,
            "grid_fractions": [0.1, 0.5, 0.9],
            "settle_seconds": SETTLE_SECONDS,
            "observe_seconds": OBSERVE_SECONDS,
            "sample_gate": "existing accepted/open-eye status and finite locked-model prediction",
            "offset_correction": "disabled",
            "model_refit": False,
            "accuracy_threshold": None,
        },
        "targets": [],
        "samples": [],
        "summary": None,
        "capture_metadata": {},
    }

    report_identity: tuple[int, int] | None = None
    capture_metadata: dict[str, Any] = {}

    def persist_report() -> None:
        nonlocal report_identity
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n"
        ).encode("utf-8")
        if report_identity is None:
            with report_path.open("xb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            stat = report_path.stat()
            report_identity = (int(stat.st_dev), int(stat.st_ino))
            return

        temporary = report_path.with_name(f".{report_path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            current = report_path.stat(follow_symlinks=False)
            if report_path.is_symlink() or not report_path.is_file() or report_identity != (int(current.st_dev), int(current.st_ino)):
                raise RuntimeError("independent validation report ownership changed; existing file was preserved")
            os.replace(temporary, report_path)
            stat = report_path.stat(follow_symlinks=False)
            report_identity = (int(stat.st_dev), int(stat.st_ino))
        finally:
            temporary.unlink(missing_ok=True)

    try:
        persist_report()
    except FileExistsError:
        collision_path = report_path
        last_error: Exception | None = None
        for _ in range(3):
            capture_id = uuid.uuid4().hex
            report["capture_id"] = capture_id
            report["status"] = "incomplete"
            report["incomplete_reasons"] = ["report_path_collision"]
            report["capture_metadata"] = {
                "report_path_collision": {
                    "path": str(collision_path.resolve(strict=False)),
                    "preserved": True,
                }
            }
            report_path = sidecar_path(
                output_root,
                f"{session_id}.{capture_id}.independent-validation.json",
                layout=INTERNAL_LAYOUT,
                create_parent=True,
            )
            try:
                persist_report()
            except FileExistsError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                break
            return {
                "status": "failed",
                "message": "independent validation report path collision; existing report was preserved",
                "outputs": {"validation_report": str(report_path.resolve())},
                "details": {
                    "validation_status": "incomplete",
                    "reason": "report_path_collision",
                    "collision_path": str(collision_path.resolve(strict=False)),
                    "validation_report_path": str(report_path.resolve()),
                    "report_persisted": True,
                    "session_id": session_id,
                    "capture_id": capture_id,
                },
            }
        existing_report_path = (
            collision_path.resolve(strict=False)
            if collision_path.exists() or collision_path.is_symlink()
            else None
        )
        return {
            "status": "failed",
            "message": "independent validation report path collisions prevented a failure report",
            "outputs": (
                {"existing_validation_report": str(existing_report_path)}
                if existing_report_path is not None
                else {}
            ),
            "details": {
                "validation_status": "incomplete",
                "reason": "report_path_collision_retry_failed",
                "collision_path": str(collision_path.resolve(strict=False)),
                "existing_validation_report": (
                    str(existing_report_path) if existing_report_path is not None else None
                ),
                "validation_report_path": None,
                "report_persisted": False,
                "session_id": session_id,
                "capture_id": capture_id,
                "error": f"{type(last_error).__name__}: {last_error}" if last_error else None,
            },
        }

    def failed_result(reason: str, message: str, *, capture_metadata: Mapping[str, Any] | None = None):
        metadata_value = capture_metadata
        if metadata_value is None:
            candidate = getattr(estimator, "last_capture_metadata", {})
            metadata_value = candidate if isinstance(candidate, Mapping) else {}
        validation_metadata = metadata_value.get("independent_validation")
        if isinstance(validation_metadata, Mapping):
            recovered_samples = validation_metadata.get("samples")
            if isinstance(recovered_samples, list):
                report["samples"] = recovered_samples
        stop_reason = metadata_value.get("stop_reason") or reason
        report["status"] = "incomplete"
        report["incomplete_reasons"] = [reason]
        report["summary"] = summarize_independent_validation(
            targets,
            report.get("samples", []),
            capture_status=str(stop_reason),
            context_valid=(
                bool(validation_metadata.get("context_valid", True))
                if isinstance(validation_metadata, Mapping)
                else True
            ),
        ) if targets else None
        if report["summary"] is not None:
            report["incomplete_reasons"] = list(dict.fromkeys(
                [*report["incomplete_reasons"], *report["summary"].get("incomplete_reasons", [])]
            ))
        if metadata_value:
            report["capture_metadata"] = {
                key: metadata_value.get(key)
                for key in (
                    "started_at_local", "ended_at_local", "capture_start_monotonic_ns",
                    "capture_end_monotonic_ns", "stop_reason", "error", "traceback",
                    "frames_received", "frames_processed", "accepted_samples",
                    "sample_status_counts", "current_canvas_size_px", "feature_formula_version",
                    "model_binding", "offset_correction",
                )
                if key in metadata_value
            }
        try:
            persist_report()
        except Exception as exc:
            message = f"{message}; report finalization failed: {type(exc).__name__}: {exc}"
        return {
            "status": "failed",
            "message": message,
            "outputs": {"validation_report": str(report_path.resolve())},
            "details": {
                "validation_status": "incomplete",
                "reason": reason,
                "session_id": session_id,
                "capture_id": capture_id,
                "validation_report_path": str(report_path.resolve()),
                "report_persisted": report_path.is_file(),
            },
        }

    try:
        model_path = assert_no_reparse_components(requested_model_path).resolve(strict=True)
        if not model_path.is_file() or model_path.suffix.lower() != ".pkl":
            raise ValueError("independent validation requires an existing .pkl model")
        estimator = GazeEstimator(model_name="ridge", model_kwargs=MODEL_KWARGS.get("ridge", {}))
        estimator.load_model(model_path)
        binding = model_binding_for_estimator(model_path, estimator)
        model_canvas = binding.calibration_canvas_size_px
        formula = binding.feature_formula_version
        report["model"].update({
            "path": str(model_path),
            "sha256": binding.model_sha256,
            "feature_formula_version": formula,
            "feature_schema": get_feature_schema(formula),
            "calibration_canvas_size_px": list(model_canvas) if model_canvas is not None else None,
        })
        persist_report()
        probe = tk.Tk()
        try:
            probe.withdraw()
            current_canvas = (int(probe.winfo_screenwidth()), int(probe.winfo_screenheight()))
        finally:
            try:
                probe.destroy()
            except tk.TclError:
                pass
        report["canvas"]["current_screen_size_px"] = list(current_canvas)
        if model_canvas is None:
            return failed_result("model_canvas_unavailable", "model calibration canvas is unavailable")
        if current_canvas != tuple(model_canvas):
            return failed_result(
                "capture_canvas_mismatch",
                "current display canvas differs from the model calibration canvas",
            )

        targets = nine_point_targets(*current_canvas)
        report["targets"] = targets
        report["parameters"]["grid_coordinate_rule"] = "rounded (width-1,height-1) multiplied by each interior grid fraction"
        _check_output_names(
            output_root,
            session_id,
            (".csv", ".txt", ".xlsx", ".manifest.json", ".capture-snapshot.json"),
        )
        samples_path = output_root / f"{session_id}.{table_format}"
        eye_data = run_realtime_gaze(
            estimator=estimator,
            save_name=session_id,
            draw_cloud=False,
            screen_width=current_canvas[0],
            screen_height=current_canvas[1],
            camera_index=camera_num,
            save_dir=output_root,
            session_id=session_id,
            model_binding=binding,
            independent_validation_targets=targets,
            independent_validation_capture_id=capture_id,
            validation_settle_s=SETTLE_SECONDS,
            validation_observe_s=OBSERVE_SECONDS,
            motion_monitor_enabled=False,
            save_motion_keyframes=False,
            capture_persistence_enabled=False,
            table_format=table_format,
        )
        capture_metadata = dict(getattr(estimator, "last_capture_metadata", {}))
        validation_metadata = capture_metadata.get("independent_validation")
        validation_metadata = validation_metadata if isinstance(validation_metadata, Mapping) else {}
        samples = list(validation_metadata.get("samples", []))
        summary = validation_metadata.get("summary")
        if not isinstance(summary, Mapping):
            summary = summarize_independent_validation(
                targets,
                samples,
                capture_status=str(capture_metadata.get("stop_reason") or "capture_failed"),
                context_valid=bool(validation_metadata.get("context_valid", False)),
            )
        report["status"] = summary.get("status", "incomplete")
        report["incomplete_reasons"] = list(summary.get("incomplete_reasons", []))
        report["samples"] = samples
        report["summary"] = dict(summary)
        report["capture_metadata"] = {
            key: capture_metadata.get(key)
            for key in (
                "started_at_local",
                "ended_at_local",
                "capture_start_monotonic_ns",
                "capture_end_monotonic_ns",
                "stop_reason",
                "frames_received",
                "frames_processed",
                "accepted_samples",
                "sample_status_counts",
                "current_canvas_size_px",
                "feature_formula_version",
            )
            if key in capture_metadata
        }
        capture_metadata["capture_id"] = capture_id
        capture_metadata["capture_persistence"] = None
        capture_metadata["independent_validation_report"] = {
            "relative_path": report_path.relative_to(output_root).as_posix(),
            "schema": report["schema"],
        }
        persist_report()
        report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
        capture_metadata["independent_validation_report"]["sha256"] = report_sha
        capture_metadata["independent_validation_report"]["size_bytes"] = report_path.stat().st_size
        experiment_info = {
            "workflow_mode": ["independent_validation"],
            "session_id": [session_id],
            "capture_id": [capture_id],
            "model_path": [str(model_path)],
            "model_sha256": [binding.model_sha256],
            "validation_report": [report_path.relative_to(output_root).as_posix()],
        }
        export = save_capture_result_with_recovery(
            output_root,
            session_id,
            eye_data,
            experiment_info,
            capture_metadata,
            model_metadata=estimator.model.metadata,
        )
        outputs = dict(export.get("outputs") or {})
        outputs["validation_report"] = str(report_path.resolve())
        stop_reason = capture_metadata.get("stop_reason")
        capture_error = capture_metadata.get("error")
        if stop_reason == "user_exit":
            status = "cancelled"
            message = "independent validation cancelled; incomplete report was retained"
        elif capture_error or stop_reason != "target_sequence_complete":
            status = "failed"
            message = str(capture_error or f"validation capture stopped: {stop_reason}")
        elif export.get("status") != "completed":
            status = "failed"
            message = export.get("message", "validation report retained but workbook export failed")
        else:
            status = "completed"
            message = (
                "nine-point validation capture and export completed"
                if report["status"] == "complete"
                else "validation run ended with incomplete target coverage; report has no pass/fail threshold"
            )
        return {
            "status": status,
            "message": message,
            "outputs": outputs,
            "details": {
                **dict(export.get("details") or {}),
                "validation_status": report["status"],
                "incomplete_reasons": report["incomplete_reasons"],
                "validation_report_path": str(report_path.resolve()),
                "validation_report_sha256": report_sha,
                "session_id": session_id,
                "capture_id": capture_id,
                "model_sha256": binding.model_sha256,
                "stop_reason": stop_reason,
            },
        }
    except Exception as exc:
        return failed_result(
            f"{type(exc).__name__}:{exc}",
            f"independent validation could not be completed: {type(exc).__name__}: {exc}",
            capture_metadata=capture_metadata or None,
        )


def _resolve_model_type(workflow_mode, requested_model_name=None):
    if workflow_mode == "capture":
        return "ridge"
    if workflow_mode == "calibrate":
        return requested_model_name or "ridge"
    return select_model()


def _run_eye_tracking_workflow(root, *, requested_session_id=None, requested_save_dir=None,
                              workflow_mode="legacy", requested_model_path=None,
                              requested_model_name=None, feature_formula_version=None, correction_path=None,
                              motion_monitor_enabled=False, save_motion_keyframes=False,
                              table_format="csv"):

    root.attributes("-topmost", True)  # 确保弹窗在最前

    # ---------- 设置全局字体 ----------
    default_font = tkFont.nametofont("TkDefaultFont")
    default_font.configure(family="Arial", size=12)  # 设置字体为 Arial，大小 20

    text_font = tkFont.nametofont("TkTextFont")
    text_font.configure(family="Arial", size=14)

    model_type = _resolve_model_type(workflow_mode, requested_model_name)
    model_kwargs = MODEL_KWARGS.get(model_type, {})
    print(model_kwargs)

    if workflow_mode == "capture":
        # load_model selects the recorded formula/schema for historical models.
        estimator = GazeEstimator(model_name=model_type, model_kwargs=model_kwargs)
    else:
        selected_formula = feature_formula_version or DEFAULT_FEATURE_FORMULA_VERSION
        estimator = GazeEstimator(
            model_name=model_type,
            model_kwargs=model_kwargs,
            feature_formula_version=selected_formula,
        )


    use_existing = True if workflow_mode == "capture" else False if workflow_mode == "calibrate" else messagebox.askyesno("模型选择", "是否已有历史模型？\nYes: 选择已有模型\nNo: 创建新模型")

    if use_existing:
        model_path = requested_model_path or filedialog.askopenfilename(
            title="选择已有模型 (.pkl)",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=str(Path.cwd() / "Model_history")
        )
        if not model_path:
            print("未选择模型，程序终止")
            return
        estimator.load_model(model_path)
        print(f"已加载历史模型: {model_path}")
    else:
        model_name = requested_session_id if workflow_mode == "calibrate" and requested_session_id is not None else simpledialog.askstring("新模型", "请输入新模型文件名（不含扩展名）")
        if not model_name:
            print("未输入模型名，程序终止")
            return
        model_dir = requested_save_dir if workflow_mode == "calibrate" and requested_save_dir is not None else Path("Model_history")
        model_dir.mkdir(parents=True, exist_ok=True)
        _check_calibration_output_names(model_dir, model_name)
        model_path = model_dir / f"{model_name}.pkl"
        checkpoint_path = sidecar_path(
            model_dir,
            f"{model_name}.calibration-checkpoint.json",
            layout=INTERNAL_LAYOUT,
            create_parent=True,
        )
        if not estimator.model.supports_sample_weight:
            choice = messagebox.askyesnocancel(
                "校准训练方式", "当前模型不支持目标/访问均衡权重。\n是：改用Ridge（推荐）\n否：明确继续无权重兼容模式\n取消：结束")
            if choice is None:
                return
            if choice:
                model_type, model_kwargs = "ridge", MODEL_KWARGS.get("ridge", {})
                estimator = GazeEstimator(
                    model_name=model_type,
                    model_kwargs=model_kwargs,
                    feature_formula_version=selected_formula,
                )
            else:
                estimator.calibration_use_group_weights = False
        if not hasattr(estimator, "calibration_use_group_weights"):
            estimator.calibration_use_group_weights = True

        # ---- 九点校准 ----
        checkpoint_write_error = None
        try:
            errors_per_point_9, mean_error_9 = run_9_point_calibration(
                estimator,
                camera_index=camera_num,
                checkpoint_path=checkpoint_path,
                session_id=model_name,
            )
        except Exception as exc:
            checkpoint_write_error = f"{type(exc).__name__}: {exc}"
            estimator.last_calibration_failure = {
                "stage": "calibration_checkpoint_or_capture",
                "reason": "checkpoint_write_or_capture_error",
                "error": checkpoint_write_error,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_may_be_stale": checkpoint_path.is_file(),
            }
            errors_per_point_9, mean_error_9 = None, None
        if errors_per_point_9 is None or mean_error_9 is None:
            failure = getattr(estimator, "last_calibration_failure", None) or {
                "stage": "fixed_nine_point", "reason": "three_complete_visits_required",
            }
            failure = dict(failure)
            failure["calibration_checkpoint_path"] = str(checkpoint_path.resolve())
            failure["calibration_checkpoint_may_be_stale"] = checkpoint_path.is_file()
            if checkpoint_write_error:
                failure["calibration_checkpoint_write_error"] = checkpoint_write_error
            expected_status_path = sidecar_path(
                Path(model_dir).expanduser().resolve(),
                f"{model_name}.calibration-status.json",
                layout=INTERNAL_LAYOUT,
                create_parent=True,
            )
            status_path = None
            status_error = None
            try:
                status_path = write_calibration_status(
                    model_dir,
                    model_name,
                    failure,
                    feature_formula_version=estimator.feature_formula_version,
                )
            except (FileExistsError, OSError, ValueError) as exc:
                # Preserve the exclusive-create contract; report the attempted path and error.
                status_error = f"{type(exc).__name__}: {exc}"
            messagebox.showinfo("校准未完成", "三次九点 visit 未全部完成；没有保存新模型或校准工作簿。")
            return {
                "status": "failed",
                "outputs": {
                    **({"calibration_status": str(status_path)} if status_path else {}),
                    "calibration_checkpoint": str(checkpoint_path.resolve()),
                },
                "details": {
                    "session_id": model_name,
                    "workflow_mode": workflow_mode,
                    "reason": failure.get("reason"),
                    "stage": failure.get("stage"),
                    "calibration_status_path": str(expected_status_path),
                    "calibration_status_written": status_path is not None,
                    "calibration_status_error": status_error,
                    "calibration_checkpoint_path": str(checkpoint_path.resolve()),
                    "calibration_checkpoint_exists": checkpoint_path.is_file(),
                    "calibration_checkpoint_write_error": checkpoint_write_error,
                    "failure": failure,
                },
            }
        error_text = f"{mean_error_9:.2f}"
        messagebox.showinfo("九点校准完成", f"九点校准训练内拟合残差（不是独立精度）: {error_text}")

        # ---- 进一步校准循环 ----
        all_errors_per_point_add = []
        all_mean_error_add = []

        if messagebox.askyesno("进一步校准", "是否愿意进入进一步校准？"):
            continue_calibration = True
            while continue_calibration:
                # 弹出参数输入框
                dialog = CalibrationDialog(root, title="进一步校准参数设置")
                if dialog.canceled:
                    print("用户取消了输入，程序退出")
                    return  # 或设置默认值继续
                error_threshold = dialog.error_threshold
                max_rounds = dialog.max_rounds
                points_per_round = dialog.points_per_round
                print(f"用户设置 - 阈值: {error_threshold}, 最大轮数: {max_rounds}, 每轮点数: {points_per_round}")

                # 执行随机补充校准
                errors_per_point_add, mean_error_add = run_additional_random_calibration(
                    estimator,
                    camera_index=camera_num,
                    points_per_round=points_per_round,
                    error_threshold=error_threshold,
                    max_rounds=max_rounds
                )

                # 累计保存每轮结果
                all_errors_per_point_add.extend(errors_per_point_add)
                if isinstance(mean_error_add, list):
                    all_mean_error_add.extend(mean_error_add)
                elif isinstance(mean_error_add, (int, float)):
                    all_mean_error_add.append(mean_error_add)

                # 获取最低误差
                if all_mean_error_add:
                    min_error = all_mean_error_add[-1]
                    error_text_add = f"{min_error:.2f}"
                else:
                    min_error = None
                    error_text_add = "未计算"

                messagebox.showinfo("随机补充校准完成", f"随机补充校准完成，最新训练内拟合残差（不是独立精度）: {error_text_add}")

                # 判断是否满足阈值
                if error_threshold is not None and isinstance(min_error, (int, float)):
                    if min_error < error_threshold:
                        print(f"最新拟合残差 {min_error:.2f} 小于阈值 {error_threshold:.2f}，校准结束。")
                        continue_calibration = False
                        break  # 满足阈值，退出循环

                # 阈值未达到或未设置，询问是否继续
                continue_calibration = messagebox.askyesno(
                    "继续补充校准",
                    "最新训练内拟合残差未达到阈值（不代表真实精度），是否再次进行补充校准？"
                )
        else:
            messagebox.showinfo("跳过校准", "用户选择不进行进一步校准。")
            errors_per_point_add, mean_error_add =[], []
            all_errors_per_point_add.extend(errors_per_point_add)
            all_mean_error_add.extend(mean_error_add)

        # ---- 保存模型与结果 ----
        if errors_per_point_9 is None:
            errors_per_point_9 = []
        _save_calibration_outputs(
            estimator=estimator,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            errors_per_point_9 = errors_per_point_9,
            errors_per_point_add=all_errors_per_point_add,
            mean_error_9=mean_error_9,
            mean_errors_add=all_mean_error_add,
            model_names=model_type,
            model_kwargs=model_kwargs,
            model_metadata=estimator.model.metadata
        )

    if workflow_mode == "calibrate":
        return {
            "status": "completed",
            "outputs": {
                "model": str(Path(model_path).resolve()),
                "calibration": str(Path(model_path).with_suffix(".xlsx").resolve()),
                "calibration_checkpoint": str(sidecar_path(
                    Path(model_dir).expanduser().resolve(),
                    f"{model_name}.calibration-checkpoint.json",
                    layout=INTERNAL_LAYOUT,
                ).resolve()),
            },
            "details": {"session_id": model_name, "workflow_mode": "calibrate"},
        }
    start_experiment = True if workflow_mode == "capture" else messagebox.askyesno("开始实验", "是否进入正式实验？")

    if start_experiment:

        data_name = requested_session_id if requested_session_id is not None else simpledialog.askstring("数据名称", "请输入数据名称（不含扩展名）")
        if not data_name:
            print("未输入模型名，程序终止")
            return

        session_id = canonical_session_id(data_name)
        data_name = session_id
        save_dir = requested_save_dir if requested_save_dir is not None else Path.cwd() / "Data"
        output_suffixes = (".xlsx", ".csv", ".txt", ".manifest.json")
        if workflow_mode == "capture":
            output_suffixes += (".capture-snapshot.json",)
        _check_output_names(save_dir, session_id, output_suffixes)

        # 实验开始时间
        start_time = datetime_module.datetime.utcnow() + datetime_module.timedelta(hours=8)
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

        print("开始时间为:",start_time_str)

        draw_cloud = messagebox.askyesno("预测点显示", "是否开启预测点显示功能？")

        model_binding = None
        correction_context = None
        capture_canvas = (screen_width, screen_height)
        if workflow_mode == "capture":
            from offset_workflow import load_capture_correction, model_binding_for_estimator

            model_binding = model_binding_for_estimator(model_path, estimator)
            capture_canvas = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
            correction_context = load_capture_correction(
                correction_path,
                model_binding,
                capture_canvas,
            )

        eye_data = run_realtime_gaze(estimator = estimator, save_name = data_name,
                                     draw_cloud = draw_cloud,
                                     screen_width=capture_canvas[0],
                                     screen_height=capture_canvas[1],
                                     camera_index=camera_num,
                                     save_dir=save_dir,
                                     session_id=session_id,
                                     correction_context=correction_context,
                                     model_binding=model_binding,
                                     motion_monitor_enabled=(workflow_mode == "capture" and motion_monitor_enabled),
                                     save_motion_keyframes=(workflow_mode == "capture" and save_motion_keyframes),
                                     table_format=table_format,
                                     )

        # 实验结束时间
        end_time = datetime_module.datetime.utcnow() + datetime_module.timedelta(hours=8)
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        print("结束时间为:",end_time_str)

        # ---------- 保存实验数据 ----------
        # 确定最终平均误差
        if use_existing:
            # 历史模型，尝试从对应文件读取
            # 假设 save_calibration_results 或模型保存路径有记录
            # 这里简单示意读取 pickle 文件里的 mean_error

            Calibrationdata_path = Path(model_path).with_suffix(".xlsx")
            final_mean_error = get_min_calibration_mean(Calibrationdata_path)
            model_type , model_kwargs = get_model_info(Calibrationdata_path)
            if final_mean_error is None:
                final_mean_error = "未知"
            all_errors_per_point_add =  get_additional_calibration_mean(Calibrationdata_path)


        else:
            # 新模型
            if all_mean_error_add:
                final_mean_error = all_mean_error_add[-1]  # 最后一次补充校准的平均误差
            elif mean_error_9 is not None:
                final_mean_error = mean_error_9
            else:
                final_mean_error = "未计算"

        if 'all_errors_per_point_add' not in locals():
            all_errors_per_point_add = []

        capture_metadata = dict(getattr(estimator, "last_capture_metadata", {}))
        fps = capture_metadata.get("camera_reported_fps")
        marker_rows = capture_metadata.get("user_markers", [])
        marker_log = capture_metadata.get("user_marker_log", {})
        marker_log = marker_log if isinstance(marker_log, Mapping) else {}

        # 实验信息表
        experiment_info = {
            "使用已有模型": [use_existing],
            "模型名称": [Path(model_path).name if use_existing else model_name],
            "是否进行额外矫正": [len(all_errors_per_point_add) > 0],
            "矫正模型路径": [str(model_path) if len(all_errors_per_point_add) > 0 else ""],
            "开始时间": [start_time_str],
            "结束时间": [end_time_str],
            "最终平均误差": [final_mean_error],
            "误差含义": ["training_fit_residual; not independently measured accuracy"],
            "会话ID": [session_id],
            "计算模型":[model_type],
            "计算模型参数":[model_kwargs],
            "摄像头理论帧率": [fps],
            "用户标记数量": [len(marker_rows) if isinstance(marker_rows, list) else 0],
            "标记事件日志": [marker_log.get("path")],
            "标记事件日志 SHA-256": [marker_log.get("events_sha256")],
            "标记序列 SHA-256": [marker_log.get("markers_sha256")],
        }
        df_info = pd.DataFrame(experiment_info)

        stopped = capture_metadata.get("stop_reason")
        persistence_state = capture_metadata.get("capture_persistence")
        persistence_failed = bool(
            isinstance(persistence_state, Mapping)
            and (
                persistence_state.get("log_io_error")
                or persistence_state.get("durability_error")
                or persistence_state.get("finalization_error")
            )
        )
        capture_failed = bool(capture_metadata.get("error")) or stopped not in {"user_exit", None} or persistence_failed
        if workflow_mode == "capture":
            context_path = (
                persistence_state.get("context_path")
                if isinstance(persistence_state, Mapping)
                else None
            )
            if capture_failed and isinstance(context_path, str):
                result = retry_capture_from_context(context_path)
            else:
                result = save_capture_result_with_recovery(
                    save_dir,
                    session_id,
                    eye_data,
                    experiment_info,
                    capture_metadata,
                    model_metadata=estimator.model.metadata,
                )
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            details = result.get("details") if isinstance(result.get("details"), dict) else {}
            details.update({
                "session_id": session_id,
                "output_root": str(Path(save_dir).resolve()),
                "stop_reason": stopped,
                "records_written": capture_metadata.get("records_written"),
                "capture_complete": not capture_failed,
                "capture_status": "interrupted" if capture_failed else "completed",
            })
            if capture_failed:
                details.setdefault("retryable", bool(details.get("recoverable")))
                details.setdefault("reason", capture_metadata.get("error") or f"capture_stopped:{stopped}")
                details["partial_capture"] = True
                details["capture_persistence_error"] = persistence_failed
                if capture_metadata.get("error"):
                    details.setdefault("original_capture_error", capture_metadata.get("error"))
                    details.setdefault("error", capture_metadata.get("error"))
                if capture_metadata.get("traceback"):
                    details.setdefault("original_capture_traceback", capture_metadata.get("traceback"))
                    details.setdefault("traceback", capture_metadata.get("traceback"))
            result["details"] = details
            if capture_failed or result.get("status") != "completed":
                result["status"] = "failed"
            else:
                print("实验结果快照、XLSX 与 manifest 已校验并发布。")
                result["status"] = "completed"
            result["outputs"] = outputs
            return result

        save_path = save_experiment_outputs(
            save_dir, session_id, eye_data, experiment_info, capture_metadata,
            model_metadata=estimator.model.metadata,
        )
        print(f"实验数据与同名清单已保存到: {save_path}")
        outputs = {
            "xlsx": str(save_path.resolve()),
            "samples": str((save_dir / f"{session_id}.{capture_metadata.get('table_format', table_format)}").resolve()),
            "manifest": str(sidecar_path(save_dir, f"{session_id}.manifest.json", layout=INTERNAL_LAYOUT).resolve()),
        }
        return {
            "status": "failed" if capture_failed else "completed",
            "outputs": outputs,
            "details": {
                "session_id": session_id, "output_root": str(save_dir.resolve()),
                "stop_reason": stopped, "error": capture_metadata.get("error"),
                "records_written": capture_metadata.get("records_written"),
                "partial_outputs": failed,
            },
        }

    else:
        print("已取消实验")
        return {"status": "cancelled", "outputs": {}, "details": {"reason": "experiment_not_started"}}


# ---------- 安全调用保存函数 ----------
def safe_save_calibration_results(model_path, errors_per_point_9, errors_per_point_add,
                                  mean_error_9, mean_errors_add, model_names=None,
                                  model_kwargs=None, *, model_metadata=None,
                                  model_reference_path=None):
    """Pass explicit new results or full legacy residual vectors without guessing pairs."""
    return save_calibration_results(
        Path(model_path), errors_per_point_9, errors_per_point_add, mean_error_9,
        mean_errors_add, model_names=model_names, model_kwargs=model_kwargs,
        model_metadata=model_metadata,
        model_reference_path=model_reference_path,
    )


def _save_calibration_outputs(*, estimator, model_path, checkpoint_path, **results):
    """Stage a complete output pair; roll back this attempt on caught failures."""
    model_path = Path(model_path).expanduser().resolve()
    workbook_path = model_path.with_suffix(".xlsx")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    _check_calibration_final_outputs(model_path.parent, model_path.stem)
    # A private directory on the same volume owns all staging files. Hard links
    # publish each finished file exclusively, so a concurrent file is never replaced.
    with tempfile.TemporaryDirectory(
        prefix=f".{model_path.stem}-calibration-", dir=model_path.parent,
        ignore_cleanup_errors=True,
    ) as staging_directory:
        staged_model = Path(staging_directory) / model_path.name
        staged_workbook = staged_model.with_suffix(".xlsx")
        staged_checkpoint = Path(staging_directory) / checkpoint_path.name
        published = []
        checkpoint_before_export = None
        intent_saved = False
        try:
            estimator.save_model(staged_model)
            safe_save_calibration_results(
                model_path=staged_model, model_reference_path=model_path, **results,
            )
            if not staged_model.is_file() or staged_model.stat().st_size == 0:
                raise ValueError("staged calibration model is empty or missing")
            with zipfile.ZipFile(staged_workbook) as workbook:
                if (
                    workbook.testzip() is not None
                    or not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(workbook.namelist())
                ):
                    raise ValueError("staged calibration workbook is incomplete")
            if checkpoint_path.is_file():
                from eyetrax.calibration.misc import mark_calibration_checkpoint_completed, save_calibration_checkpoint
                staged_checkpoint.write_bytes(checkpoint_path.read_bytes())
                # Finish a copy first: any completion-write error leaves the
                # recovery checkpoint untouched and no final output published.
                mark_calibration_checkpoint_completed(
                    staged_checkpoint, model_path=model_path, workbook_path=workbook_path,
                )
                checkpoint_before_export = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                intent = dict(checkpoint_before_export)
                intent["export_attempt"] = {
                    "schema": "localgaze.calibration-export-attempt/1",
                    "session_id": model_path.stem,
                    "attempt_id": Path(staging_directory).name[len(f".{model_path.stem}-calibration-"):],
                    "staging_directory": str(Path(staging_directory)),
                    "outputs": [
                        {"path": str(final), "sha256": hashlib.sha256(staged.read_bytes()).hexdigest()}
                        for staged, final in ((staged_model, model_path), (staged_workbook, workbook_path))
                    ],
                }
                # Persist ownership before either link: a later process can
                # safely reclaim a partial pair using strict path/hash checks.
                save_calibration_checkpoint(checkpoint_path, intent)
                intent_saved = True
            for staged, final in (
                (staged_model, model_path), (staged_workbook, workbook_path),
            ):
                os.link(staged, final)
                published.append((staged, final))
            if staged_checkpoint.is_file():
                os.replace(staged_checkpoint, checkpoint_path)
        except BaseException:
            rollback_complete = True
            for staged, final in reversed(published):
                try:
                    # Do not delete a file replaced by another writer since publication.
                    if final.exists() and os.path.samefile(staged, final):
                        final.unlink()
                except OSError as cleanup_error:
                    rollback_complete = False
                    print(f"Calibration output rollback failed for {final}: {cleanup_error}", file=sys.stderr)
            if intent_saved and rollback_complete:
                try:
                    save_calibration_checkpoint(checkpoint_path, checkpoint_before_export)
                except OSError as cleanup_error:
                    # Keep the persisted ownership marker for the next retry.
                    print(f"Calibration checkpoint rollback failed: {cleanup_error}", file=sys.stderr)
            raise


def _check_output_names(directory, name, suffixes):
    if not isinstance(name, str) or not name or name in {".", ".."} or name[-1] in {".", " "}:
        raise ValueError("文件名必须是非空的普通名称，不能以空格或句点结尾")
    if any(char in name for char in '<>:"/\\|?*') or any(ord(char) < 32 for char in name):
        raise ValueError("文件名不能包含路径分隔符或Windows非法字符")
    if name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("文件名不能是Windows设备保留名称")
    directory = Path(directory)
    for suffix in suffixes:
        path = directory / f"{name}{suffix}"
        candidates = [path]
        if suffix in {
            ".manifest.json", ".capture-snapshot.json", ".offset-correction.json",
            ".calibration-checkpoint.json", ".calibration-status.json",
        }:
            candidates.insert(0, directory / ".internal" / f"{name}{suffix}")
        existing = next((candidate for candidate in candidates if candidate.exists() or candidate.is_symlink()), None)
        if existing is not None:
            raise FileExistsError(f"文件已存在，请使用新名称：{existing}")
    directory.mkdir(parents=True, exist_ok=True)


def _check_calibration_output_names(directory, name):
    """Allow an existing recoverable checkpoint, but never replace final outputs."""
    _check_output_names(directory, name, ())
    directory = Path(directory)
    recovery = calibration_export_recovery(directory, name)
    if recovery is not None:
        checkpoint, owned, staging = recovery
        # Validation checks the whole pair before any mutation; a mismatch
        # leaves all user files intact. This directory is the strict owned
        # sibling under the requested output root, never a supplied arbitrary path.
        for path in owned:
            path.unlink()
        if staging.is_dir():
            shutil.rmtree(staging)
        from eyetrax.calibration.misc import save_calibration_checkpoint
        checkpoint.pop("export_attempt")
        declared_layout = normalize_sidecar_layout(checkpoint.get("sidecar_layout"))
        save_calibration_checkpoint(
            sidecar_path(directory, f"{name}.calibration-checkpoint.json", layout=declared_layout, create_parent=declared_layout == INTERNAL_LAYOUT),
            checkpoint,
        )
    for suffix in (".pkl", ".xlsx"):
        path = directory / f"{name}{suffix}"
        if path.exists():
            raise FileExistsError(f"文件已存在，请使用新名称：{path}")
    checkpoint_paths = [
        sidecar_path(directory, f"{name}.calibration-checkpoint.json", layout=INTERNAL_LAYOUT),
        sidecar_path(directory, f"{name}.calibration-checkpoint.json", layout=LEGACY_LAYOUT),
    ]
    status_paths = [
        sidecar_path(directory, f"{name}.calibration-status.json", layout=INTERNAL_LAYOUT),
        sidecar_path(directory, f"{name}.calibration-status.json", layout=LEGACY_LAYOUT),
    ]
    if sum(path.exists() for path in checkpoint_paths) > 1 or sum(path.exists() for path in status_paths) > 1:
        raise FileExistsError("校准 sidecar 在多个布局中冲突")
    if any(path.exists() for path in status_paths) and not any(path.exists() for path in checkpoint_paths):
        raise FileExistsError(
            f"存在失败记录但没有可恢复的校准检查点，请使用新名称：{next(path for path in status_paths if path.exists())}"
        )


def _check_calibration_final_outputs(directory, name):
    directory = Path(directory)
    for suffix in (".pkl", ".xlsx"):
        path = directory / f"{name}{suffix}"
        if path.exists():
            raise FileExistsError(f"文件已存在，请使用新名称：{path}")


_EXCEL_CELL_TEXT_LIMIT = 32767
_EXCEL_PART_TEXT_CHUNK_CHARS = 16000
_EXCEL_CELL_PARTS_SCHEMA = "localgaze.excel-cell-parts/1"
_EXCEL_CELL_PARTS_SHEET = "ExportCellParts"
_EXCEL_CELL_PARTS_COLUMNS = (
    "source_sheet",
    "source_row",
    "source_column",
    "cell_id",
    "part_index",
    "part_count",
    "text_chunk",
    "source_utf8_sha256",
    "chunk_utf8_sha256",
)


def _excel_utf16_code_units(value):
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _excel_json_cell_overflows(value):
    if not isinstance(value, str):
        return False
    if max(len(value), _excel_utf16_code_units(value)) <= _EXCEL_CELL_TEXT_LIMIT:
        return False
    try:
        json.loads(value)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def _excel_json_identity(value, key):
    if isinstance(value, dict):
        direct = value.get(key)
        if direct is not None and not isinstance(direct, (dict, list, tuple)):
            return direct
        for nested in value.values():
            found = _excel_json_identity(nested, key)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _excel_json_identity(nested, key)
            if found is not None:
                return found
    return None


def _excel_row_identity(frame, row_index, parsed_value, columns, json_keys):
    for column in columns:
        if column not in frame.columns:
            continue
        value = frame.iloc[row_index][column]
        try:
            missing_value = pd.isna(value)
        except (TypeError, ValueError):
            missing_value = False
        if value is None or (isinstance(missing_value, (bool, np.bool_)) and missing_value):
            continue
        return str(value)
    for key in json_keys:
        value = _excel_json_identity(parsed_value, key)
        if value is not None:
            return str(value)
    return None


def _excel_frame_with_json_cell_parts(frame, sheet_name, first_cell_number):
    """Replace only over-limit serialized JSON cells with workbook-local references."""
    safe_frame = frame.copy(deep=True)
    part_rows = []
    cell_number = first_cell_number
    for row_index, values in enumerate(frame.itertuples(index=False, name=None)):
        excel_row = row_index + 2  # one-based Excel row after the header
        for column_index, value in enumerate(values):
            if not _excel_json_cell_overflows(value):
                continue
            encoded = value.encode("utf-8")
            source_sha256 = hashlib.sha256(encoded).hexdigest()
            chunks = [
                value[offset:offset + _EXCEL_PART_TEXT_CHUNK_CHARS]
                for offset in range(0, len(value), _EXCEL_PART_TEXT_CHUNK_CHARS)
            ]
            cell_id = f"excel-cell-{cell_number:06d}"
            source_column = str(frame.columns[column_index])
            parsed_value = json.loads(value)
            capture_id = _excel_row_identity(
                frame, row_index, parsed_value,
                ("motion_capture_id", "capture_id"), ("capture_id",),
            )
            frame_id = _excel_row_identity(
                frame, row_index, parsed_value,
                ("frame_id", "source_frame_id"), ("frame_id", "source_frame_id"),
            )
            for part_index, chunk in enumerate(chunks, start=1):
                part_rows.append({
                    "source_sheet": sheet_name,
                    "source_row": excel_row,
                    "source_column": source_column,
                    "cell_id": cell_id,
                    "part_index": part_index,
                    "part_count": len(chunks),
                    "text_chunk": chunk,
                    "source_utf8_sha256": source_sha256,
                    "chunk_utf8_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                })
            descriptor = {
                "schema": _EXCEL_CELL_PARTS_SCHEMA,
                "kind": "lossless_json_cell_reference",
                "complete_length_chars": len(value),
                "complete_length_utf16_code_units": _excel_utf16_code_units(value),
                "complete_length_utf8_bytes": len(encoded),
                "original_utf8_sha256": source_sha256,
                "capture_id": capture_id,
                "frame_id": frame_id,
                "source": {
                    "sheet": sheet_name,
                    "row": excel_row,
                    "column": source_column,
                },
                "parts": {
                    "sheet": _EXCEL_CELL_PARTS_SHEET,
                    "cell_id": cell_id,
                    "part_count": len(chunks),
                },
            }
            safe_frame.iat[row_index, column_index] = json.dumps(
                descriptor, ensure_ascii=False, separators=(",", ":")
            )
            cell_number += 1
    return safe_frame, part_rows, cell_number


def save_experiment_outputs(save_dir, session_id, records, experiment_info, capture_metadata,
                            *, model_metadata=None, final_output_root=None):
    """Save the workbook and its layout-owned manifest for one capture session."""
    save_dir = assert_no_reparse_components(Path(save_dir).expanduser())
    session_id = canonical_session_id(session_id)
    _check_output_names(save_dir, session_id, (".xlsx", ".manifest.json"))
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("新版实时记录必须使用dict，以保留状态和时间字段")
    metadata = dict(capture_metadata)
    if metadata.get("session_id", session_id) != session_id:
        raise ValueError("实时采集与导出会话ID不一致")
    metadata["session_id"] = session_id
    persistence = metadata.get("capture_persistence")
    persistence = persistence if isinstance(persistence, Mapping) else {}
    from capture_export_recovery import _declared_layout

    layout = _declared_layout(metadata)
    top_table_format = metadata.get("table_format")
    nested_table_format = persistence.get("table_format")
    if (
        top_table_format is not None
        and nested_table_format is not None
        and top_table_format != nested_table_format
    ):
        raise ValueError("capture table_format conflicts with capture_persistence")
    table_format = nested_table_format if nested_table_format is not None else top_table_format
    if table_format is None:
        table_format = "txt" if layout == LEGACY_LAYOUT else "csv"
    if table_format not in {"csv", "txt"}:
        raise ValueError("capture table_format is invalid")
    file_metadata = metadata.get("files")
    file_metadata = file_metadata if isinstance(file_metadata, Mapping) else {}
    if file_metadata.get("table_format") not in (None, table_format):
        raise ValueError("capture files.table_format conflicts with capture metadata")
    metadata["sidecar_layout"] = layout
    metadata["table_format"] = table_format
    expected_samples_name = f"{session_id}.{table_format}"
    for label, value in (
        ("raw_path", metadata.get("raw_path")),
        ("capture_persistence.raw_path", persistence.get("raw_path")),
        ("files.samples", file_metadata.get("samples")),
        (f"files.{table_format}", file_metadata.get(table_format)),
    ):
        if value is not None and value != expected_samples_name:
            raise ValueError(f"capture {label} does not match its session and table_format")
    samples_name = expected_samples_name
    metadata["raw_path"] = samples_name
    sample_file_key = table_format
    manifest_relative = sidecar_path(
        save_dir,
        f"{session_id}.manifest.json",
        layout=layout,
        create_parent=layout == INTERNAL_LAYOUT and final_output_root is None,
    ).relative_to(save_dir.resolve()).as_posix()
    metadata["files"] = {
        "xlsx": f"{session_id}.xlsx",
        sample_file_key: samples_name,
        "samples": samples_name,
        "table_format": table_format,
        "manifest": manifest_relative,
    }
    if final_output_root is not None:
        # Recovery writes the workbook and manifest in an attempt-owned
        # staging directory, but the manifest must describe the final paths.
        # Set these references before serialization and hashing; never patch a
        # published manifest after the completion marker is written.
        final_root = assert_no_reparse_components(Path(final_output_root).expanduser()).resolve(strict=True)
        metadata["output_dir"] = str(final_root)
        metadata["files"] = {
            "xlsx": str(final_root / f"{session_id}.xlsx"),
            sample_file_key: str(final_root / samples_name),
            "samples": str(final_root / samples_name),
            "table_format": table_format,
            "manifest": str(sidecar_path(
                final_root,
                f"{session_id}.manifest.json",
                layout=layout,
                create_parent=layout == INTERNAL_LAYOUT,
            )),
        }
    metadata["calibration_residual_interpretation"] = "training_fit_residual; not independently measured accuracy"
    metadata["model_feature_contract"] = {
        key: value for key, value in (model_metadata or {}).items() if key != "training"
    }
    eye_frame = pd.DataFrame(records)
    for name in ("时间", "x", "y", "blink", "左瞳孔直径", "右瞳孔直径"):
        if name not in eye_frame:
            eye_frame[name] = pd.Series(dtype=object)
    # Excel numeric cells cannot preserve arbitrary 64-bit nanosecond integers.
    for name in eye_frame.columns:
        if name.endswith("monotonic_ns"):
            # Read original dict values before pandas can coerce nullable
            # int64 data to float64 and lose the low nanosecond digits.
            eye_frame[name] = pd.Series([
                str(int(record[name])) if pd.notna(record.get(name)) else None
                for record in records
            ], dtype=object)
    movement_metadata = metadata.get("motion_monitor")
    movement_result = (
        movement_metadata.get("result")
        if isinstance(movement_metadata, dict)
        else None
    )

    def excel_rows(rows):
        normalized = []
        for item in rows if isinstance(rows, list) else []:
            if not isinstance(item, dict):
                continue
            row = {}
            for key, value in item.items():
                if key.endswith("monotonic_ns") and value is not None:
                    row[key] = str(value)
                elif isinstance(value, (dict, list, tuple, set)):
                    row[key] = json.dumps(value, ensure_ascii=False, default=str)
                else:
                    row[key] = value
            normalized.append(row)
        return normalized

    include_movement_sheets = bool(
        isinstance(movement_metadata, dict)
        and (movement_metadata.get("requested") or movement_metadata.get("enabled"))
    )
    movement_events = excel_rows(
        movement_result.get("events", []) if isinstance(movement_result, dict) else []
    )
    movement_keyframes = excel_rows(
        movement_result.get("keyframes", []) if isinstance(movement_result, dict) else []
    )
    workbook_frames = [
        ("实验信息", pd.DataFrame(experiment_info)),
        ("眼动数据", eye_frame),
        ("CaptureMetadata", pd.DataFrame([
            {"Key": key, "Value": json.dumps(value, ensure_ascii=False, default=str)}
            for key, value in metadata.items()
        ])),
    ]
    if "user_markers" in metadata:
        marker_rows = excel_rows(metadata.get("user_markers"))
        marker_frame = pd.DataFrame(marker_rows)
        if marker_frame.empty:
            marker_frame = pd.DataFrame(columns=(
                "schema", "marker_id", "session_id", "capture_id", "attempt_id",
                "marker_sequence", "literal_key", "label", "monotonic_ns",
                "elapsed_s", "timestamp_local", "source",
            ))
        workbook_frames.append(("UserMarkers", marker_frame))
    if include_movement_sheets:
        event_frame = pd.DataFrame(movement_events)
        if event_frame.empty:
            event_frame = pd.DataFrame(columns=("event_id", "capture_id", "channel", "type", "start", "end", "duration_s", "end_reason", "candidate_label", "needs_review"))
        keyframe_frame = pd.DataFrame(movement_keyframes)
        if keyframe_frame.empty:
            keyframe_frame = pd.DataFrame(columns=("event_id", "capture_id", "keyframe_type", "status", "relative_path", "reason"))
        workbook_frames.extend((
            ("MovementEvents", event_frame),
            ("MovementKeyframes", keyframe_frame),
        ))
    export_cell_parts = []
    next_cell_number = 1
    safe_workbook_frames = []
    for sheet_name, frame in workbook_frames:
        safe_frame, parts, next_cell_number = _excel_frame_with_json_cell_parts(
            frame, sheet_name, next_cell_number
        )
        safe_workbook_frames.append((sheet_name, safe_frame))
        export_cell_parts.extend(parts)
    metadata["excel_cell_parts"] = {
        "schema": _EXCEL_CELL_PARTS_SCHEMA,
        "sheet": _EXCEL_CELL_PARTS_SHEET,
        "sheet_present": bool(export_cell_parts),
        "cell_reference_count": next_cell_number - 1,
        "part_row_count": len(export_cell_parts),
        "max_chunk_unicode_chars": _EXCEL_PART_TEXT_CHUNK_CHARS,
        "chunk_length_checks": ["unicode_code_points", "utf16_code_units"],
        "source_encoding": "UTF-8",
        "hash_algorithm": "SHA-256",
    }
    text = json.dumps(metadata, ensure_ascii=False, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
    path = save_dir / f"{session_id}.xlsx"
    with path.open("xb") as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_name, frame in safe_workbook_frames:
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
            if export_cell_parts:
                sheet = writer.book.create_sheet(_EXCEL_CELL_PARTS_SHEET)
                for column_index, column_name in enumerate(_EXCEL_CELL_PARTS_COLUMNS, start=1):
                    sheet.cell(row=1, column=column_index, value=column_name)
                for row_index, part in enumerate(export_cell_parts, start=2):
                    for column_index, column_name in enumerate(_EXCEL_CELL_PARTS_COLUMNS, start=1):
                        value = part[column_name]
                        cell = sheet.cell(row=row_index, column=column_index, value=value)
                        if isinstance(value, str):
                            # Force literal text, including chunks beginning with '='.
                            cell.data_type = "s"
    manifest_path = (
        save_dir / f"{session_id}.manifest.json"
        if final_output_root is not None
        else sidecar_path(
            save_dir,
            f"{session_id}.manifest.json",
            layout=layout,
            create_parent=layout == INTERNAL_LAYOUT,
        )
    )
    with manifest_path.open("x", encoding="utf-8") as output:
        output.write(text)
    return path


def _capture_export_output_writer(staging_path, payload):
    """Write one snapshot payload into the recovery module's attempt staging directory."""
    if not isinstance(payload, dict):
        raise TypeError("capture recovery payload must be a JSON object")
    staging_path = assert_no_reparse_components(Path(staging_path).expanduser())
    if not staging_path.is_dir():
        raise ValueError("capture recovery writer requires its owned staging directory")
    session_id = canonical_session_id(payload.get("session_id"))
    workbook_path = save_experiment_outputs(
        staging_path,
        session_id,
        payload.get("records"),
        payload.get("experiment_info"),
        payload.get("capture_metadata"),
        model_metadata=payload.get("model_metadata"),
        final_output_root=payload.get("output_root"),
    )
    manifest_path = staging_path / f"{session_id}.manifest.json"
    return {
        "xlsx": {
            "relative_path": workbook_path.name,
            "sha256": hashlib.sha256(workbook_path.read_bytes()).hexdigest(),
        },
        "manifest": {
            "relative_path": manifest_path.name,
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    }


def _capture_recovery_details(value):
    if not isinstance(value, dict):
        return {}
    details = value.get("details")
    details = dict(details) if isinstance(details, dict) else {}
    for key in (
        "recoverable", "retryable", "reason", "snapshot_path", "snapshot_sha256",
        "attempt_id", "staging_path", "retained_files", "partial_outputs",
    ):
        if key in value and key not in details:
            details[key] = value[key]
    return details


def _capture_recovery_failure(
    *, snapshot_path, samples_path=None, output_root=None, session_id=None,
    snapshot_result=None, recovery_result=None, error=None, reason=None,
):
    outputs = {}
    if isinstance(snapshot_result, dict) and isinstance(snapshot_result.get("outputs"), dict):
        outputs.update(snapshot_result["outputs"])
    if isinstance(recovery_result, dict) and isinstance(recovery_result.get("outputs"), dict):
        outputs.update(recovery_result["outputs"])
    details = _capture_recovery_details(snapshot_result)
    details.update(_capture_recovery_details(recovery_result))
    exception_details = getattr(error, "details", None)
    if isinstance(exception_details, dict):
        details.update(exception_details)
    for key in ("recoverable", "retryable", "reason", "attempt_id", "snapshot_sha256", "retained_files"):
        value = getattr(error, key, None) if error is not None else None
        if value is not None and key not in details:
            details[key] = value

    snapshot_path = Path(snapshot_path)
    details.setdefault("snapshot_path", str(snapshot_path))
    if session_id is not None:
        details.setdefault("session_id", session_id)
    if output_root is not None:
        details.setdefault("output_root", str(Path(output_root).resolve()))
    details.setdefault("recoverable", snapshot_path.is_file())
    details.setdefault("retryable", bool(details.get("recoverable")))
    details.setdefault("reason", reason or (f"{type(error).__name__}: {error}" if error else "export_recovery_failed"))
    if error is not None:
        details.setdefault("error", f"{type(error).__name__}: {error}")
        if error.__traceback__ is not None:
            details.setdefault(
                "traceback",
                "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            )

    retained = []
    for item in details.get("retained_files", []) if isinstance(details.get("retained_files"), list) else []:
        candidate = item.get("path") if isinstance(item, dict) else item
        if isinstance(candidate, (str, os.PathLike)) and str(candidate) not in retained:
            retained.append(str(candidate))
    for candidate in [snapshot_path, samples_path, *outputs.values()]:
        if not isinstance(candidate, (str, os.PathLike)):
            continue
        candidate = Path(candidate)
        if candidate.is_file() and str(candidate.resolve()) not in retained:
            retained.append(str(candidate.resolve()))
    details["retained_files"] = retained
    details["partial_outputs"] = bool(retained)

    if snapshot_path.is_file():
        outputs.setdefault("snapshot", str(snapshot_path.resolve()))
    if samples_path is not None and Path(samples_path).is_file():
        outputs.setdefault("samples", str(Path(samples_path).resolve()))
    outputs = {
        key: str(value) if isinstance(value, os.PathLike) else value
        for key, value in outputs.items()
    }
    return {
        "status": "failed",
        "message": str(details.get("reason") or "capture export recovery failed"),
        "outputs": outputs,
        "details": details,
    }


def save_capture_result_with_recovery(
    save_dir, session_id, records, experiment_info, capture_metadata, *, model_metadata=None,
):
    """Persist a durable capture snapshot, then stage and verify its export pair."""
    from capture_export_recovery import (
        _declared_layout,
        persist_capture_snapshot,
        recover_capture_export,
    )

    output_root = assert_no_reparse_components(Path(save_dir).expanduser()).resolve(strict=True)
    session_id = canonical_session_id(session_id)
    persistence = capture_metadata.get("capture_persistence") if isinstance(capture_metadata, Mapping) else None
    layout = _declared_layout(capture_metadata)
    table_format = (
        persistence.get("table_format")
        if isinstance(persistence, Mapping)
        else capture_metadata.get("table_format") if isinstance(capture_metadata, Mapping) else None
    )
    if table_format is None:
        table_format = "txt" if layout == LEGACY_LAYOUT else "csv"
    if table_format not in {"csv", "txt"}:
        raise ValueError("capture table_format is invalid")
    snapshot_path = sidecar_path(
        output_root,
        f"{session_id}.capture-snapshot.json",
        layout=layout,
        create_parent=layout == INTERNAL_LAYOUT,
    )
    samples_path = output_root / f"{session_id}.{table_format}"
    snapshot_result = None
    try:
        snapshot_result = persist_capture_snapshot(
            snapshot_path,
            session_id=session_id,
            output_root=output_root,
            records=records,
            experiment_info=experiment_info,
            capture_metadata=capture_metadata,
            model_metadata=model_metadata,
        )
        if not isinstance(snapshot_result, dict) or snapshot_result.get("status") == "failed":
            return _capture_recovery_failure(
                snapshot_path=snapshot_path,
                samples_path=samples_path,
                output_root=output_root,
                session_id=session_id,
                snapshot_result=snapshot_result,
                reason="snapshot_persist_failed",
            )
        if not snapshot_path.is_file():
            raise FileNotFoundError("snapshot persistence returned without a durable snapshot")
    except Exception as exc:
        return _capture_recovery_failure(
            snapshot_path=snapshot_path,
            samples_path=samples_path,
            output_root=output_root,
            session_id=session_id,
            snapshot_result=snapshot_result,
            error=exc,
            reason="snapshot_persist_failed",
        )

    try:
        recovery_result = recover_capture_export(
            snapshot_path,
            output_writer=_capture_export_output_writer,
            staging_path=internal_directory(output_root, create=True) if layout == INTERNAL_LAYOUT else output_root,
        )
    except Exception as exc:
        return _capture_recovery_failure(
            snapshot_path=snapshot_path,
            samples_path=samples_path,
            output_root=output_root,
            session_id=session_id,
            snapshot_result=snapshot_result,
            error=exc,
            reason="export_recovery_failed",
        )
    if not isinstance(recovery_result, dict) or recovery_result.get("status") != "completed":
        return _capture_recovery_failure(
            snapshot_path=snapshot_path,
            samples_path=samples_path,
            output_root=output_root,
            session_id=session_id,
            snapshot_result=snapshot_result,
            recovery_result=recovery_result,
            reason="export_recovery_failed",
        )

    try:
        from capture_export_recovery import verify_capture_archive

        archive_verification = verify_capture_archive(snapshot_path)
    except Exception as exc:
        return _capture_recovery_failure(
            snapshot_path=snapshot_path,
            samples_path=samples_path,
            output_root=output_root,
            session_id=session_id,
            snapshot_result=snapshot_result,
            recovery_result=recovery_result,
            error=exc,
            reason="archive_validation_failed",
        )

    cleanup = {"status": "not_applicable"}
    persistence = capture_metadata.get("capture_persistence") if isinstance(capture_metadata, Mapping) else None
    context_path = persistence.get("context_path") if isinstance(persistence, Mapping) else None
    if isinstance(context_path, str):
        try:
            from capture_persistence import cleanup_capture_sidecars

            cleanup = cleanup_capture_sidecars(
                context_path,
                output_root=output_root,
                completion_verified=True,
                expected_metadata={"capture_persistence": persistence},
            )
        except Exception as exc:
            cleanup = {
                "status": "retained",
                "reason": f"{type(exc).__name__}: {exc}",
                "retained_files": [context_path],
            }

    outputs = dict(recovery_result.get("outputs") or {})
    if samples_path.is_file():
        outputs.setdefault("samples", str(samples_path.resolve()))
    outputs.setdefault("snapshot", str(snapshot_path.resolve()))
    if isinstance(persistence, Mapping):
        for output_key, metadata_key in (
            ("capture_events", "events_path"),
            ("capture_task_log", "capture_task_log_path"),
        ):
            value = persistence.get(metadata_key)
            if isinstance(value, str):
                candidate = Path(value)
                if candidate.is_file():
                    outputs.setdefault(output_key, str(candidate.resolve()))
    details = _capture_recovery_details(snapshot_result)
    details.update(_capture_recovery_details(recovery_result))
    details.update({
        "session_id": session_id,
        "output_root": str(output_root),
        "snapshot_path": str(snapshot_path.resolve()),
        "recoverable": True,
        "retryable": False,
        "cleanup": cleanup,
        "archive_verification": {
            key: value for key, value in archive_verification.items()
            if key != "capture_persistence"
        },
    })
    return {
        "status": "completed",
        "message": "Capture export recovered and verified.",
        "outputs": {key: str(value) if isinstance(value, os.PathLike) else value for key, value in outputs.items()},
        "details": details,
    }


def retry_capture_export(snapshot_path):
    """Retry only the export from an existing snapshot; this path opens no camera."""
    raw_path = assert_no_reparse_components(Path(snapshot_path).expanduser())
    guessed_root = raw_path.parent.parent if raw_path.parent.name == ".internal" else raw_path.parent
    if raw_path.is_symlink():
        return _capture_recovery_failure(
            snapshot_path=raw_path,
            output_root=guessed_root,
            reason="snapshot_path_is_symlink",
        )
    try:
        snapshot_path = raw_path.resolve(strict=True)
        if not snapshot_path.is_file() or not snapshot_path.name.endswith(".capture-snapshot.json"):
            raise ValueError("retry requires an existing *.capture-snapshot.json file")
        from capture_export_recovery import _declared_layout, _load_snapshot
        envelope, payload, output_root, _ = _load_snapshot(snapshot_path)
        capture_metadata = payload.get("capture_metadata")
        persistence = capture_metadata.get("capture_persistence") if isinstance(capture_metadata, Mapping) else None
        layout = _declared_layout(capture_metadata)
    except Exception as exc:
        return _capture_recovery_failure(
            snapshot_path=raw_path,
            output_root=guessed_root,
            error=exc,
            reason="snapshot_unavailable",
        )
    try:
        from capture_export_recovery import recover_capture_export
        result = recover_capture_export(
            snapshot_path,
            output_writer=_capture_export_output_writer,
            staging_path=internal_directory(output_root, create=True) if layout == INTERNAL_LAYOUT else output_root,
        )
    except Exception as exc:
        return _capture_recovery_failure(
            snapshot_path=snapshot_path,
            output_root=output_root,
            error=exc,
            reason="export_recovery_failed",
        )
    if not isinstance(result, dict) or result.get("status") != "completed":
        return _capture_recovery_failure(
            snapshot_path=snapshot_path,
            output_root=output_root,
            recovery_result=result,
            reason="export_recovery_failed",
        )
    try:
        from capture_export_recovery import verify_capture_archive
        from capture_persistence import cleanup_capture_sidecars

        archive_verification = verify_capture_archive(snapshot_path)
    except Exception as exc:
        return _capture_recovery_failure(
            snapshot_path=snapshot_path,
            output_root=output_root,
            recovery_result=result,
            error=exc,
            reason="archive_validation_failed",
        )
    cleanup = {"status": "not_applicable"}
    persistence = archive_verification.get("capture_persistence")
    context_path = persistence.get("context_path") if isinstance(persistence, Mapping) else None
    if isinstance(context_path, str):
        cleanup = cleanup_capture_sidecars(
            context_path,
            output_root=output_root,
            completion_verified=True,
            expected_metadata={"capture_persistence": persistence},
        )
    outputs = dict(result.get("outputs") or {})
    outputs.setdefault("snapshot", str(snapshot_path))
    details = _capture_recovery_details(result)
    details.update({
        "snapshot_path": str(snapshot_path),
        "output_root": str(output_root),
        "recoverable": True,
        "retryable": False,
        "cleanup": cleanup,
        "archive_verification": {
            key: value for key, value in archive_verification.items()
            if key != "capture_persistence"
        },
    })
    return {
        "status": "completed",
        "message": "Capture export recovered and verified without capture.",
        "outputs": {key: str(value) if isinstance(value, os.PathLike) else value for key, value in outputs.items()},
        "details": details,
    }


def retry_capture_from_context(context_path):
    """Recover capture records and publish XLSX/manifest from a saved context."""
    raw_path = Path(context_path).expanduser()
    try:
        from capture_export_recovery import recover_capture_from_context

        return recover_capture_from_context(
            raw_path,
            output_writer=_capture_export_output_writer,
        )
    except Exception as exc:
        retained = [str(raw_path)] if raw_path.exists() or raw_path.is_symlink() else []
        return {
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "outputs": {},
            "details": {
                "recoverable": bool(raw_path.is_file() and not raw_path.is_symlink()),
                "retryable": False,
                "export_complete": False,
                "reason": "capture_recovery_unavailable",
                "context_path": str(raw_path),
                "retained_files": retained,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
