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

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
from session_names import canonical_session_id


# ---------- 封装工作流程 ----------
def run_eye_tracking_workflow(*, session_id=None, save_dir=None, workflow_mode="legacy", model_path=None,
                              model_name=None, feature_formula_version=None, correction_path=None):
    """Keep the legacy UI, optionally accepting a canonical external request."""
    from tkinter import TclError
    requested_session = canonical_session_id(session_id) if session_id is not None else None
    requested_directory = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    if workflow_mode not in {"legacy", "calibrate", "capture"}:
        raise ValueError("workflow_mode must be legacy, calibrate or capture")
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
        suffixes = (".pkl", ".xlsx", ".calibration-status.json") if workflow_mode == "calibrate" else (".txt", ".xlsx", ".manifest.json")
        _check_output_names(requested_directory, requested_session, suffixes)
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
                                     correction_path=None, target_xy_px=None):
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

    _check_output_names(output_root, session_id, (".txt", ".xlsx", ".manifest.json"))
    correction_destination = (
        Path(correction_path).expanduser().resolve()
        if correction_path is not None
        else output_root / f"{session_id}.offset-correction.json"
    )
    if correction_destination.suffix.lower() != ".json":
        raise ValueError("correction_path must end in .json")
    if correction_destination.exists():
        raise FileExistsError(f"correction file already exists: {correction_destination}")

    samples_path = output_root / f"{session_id}.txt"
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
    workbook_path = save_experiment_outputs(
        output_root, session_id, eye_data, experiment_info, capture_metadata,
        model_metadata=estimator.model.metadata,
    )
    capture_outputs = {
        "xlsx": str(workbook_path.resolve()),
        "samples": str(samples_path.resolve()),
        "manifest": str((output_root / f"{session_id}.manifest.json").resolve()),
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


def _resolve_model_type(workflow_mode, requested_model_name=None):
    if workflow_mode == "capture":
        return "ridge"
    if workflow_mode == "calibrate":
        return requested_model_name or "ridge"
    return select_model()


def _run_eye_tracking_workflow(root, *, requested_session_id=None, requested_save_dir=None,
                              workflow_mode="legacy", requested_model_path=None,
                              requested_model_name=None, feature_formula_version=None, correction_path=None):

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
        _check_output_names(model_dir, model_name, (".pkl", ".xlsx", ".calibration-status.json"))
        model_path = model_dir / f"{model_name}.pkl"
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
        errors_per_point_9, mean_error_9 = run_9_point_calibration(estimator,camera_index = camera_num)
        if errors_per_point_9 is None or mean_error_9 is None:
            failure = getattr(estimator, "last_calibration_failure", None) or {
                "stage": "fixed_nine_point", "reason": "three_complete_visits_required",
            }
            expected_status_path = (Path(model_dir).expanduser().resolve() / f"{model_name}.calibration-status.json")
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
                "outputs": {"calibration_status": str(status_path)} if status_path else {},
                "details": {
                    "session_id": model_name,
                    "workflow_mode": workflow_mode,
                    "reason": failure.get("reason"),
                    "stage": failure.get("stage"),
                    "calibration_status_path": str(expected_status_path),
                    "calibration_status_written": status_path is not None,
                    "calibration_status_error": status_error,
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
        _check_output_names(model_dir, model_name, (".pkl", ".xlsx", ".calibration-status.json"))
        estimator.save_model(model_path)

        if errors_per_point_9 is None:
            errors_per_point_9 = []


        safe_save_calibration_results(
            model_path=model_path,
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
            "outputs": {"model": str(Path(model_path).resolve()), "calibration": str(Path(model_path).with_suffix(".xlsx").resolve())},
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
        _check_output_names(save_dir, session_id, (".xlsx", ".txt", ".manifest.json"))

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
            "摄像头理论帧率":[fps]
        }
        df_info = pd.DataFrame(experiment_info)

        save_path = save_experiment_outputs(
            save_dir, session_id, eye_data, experiment_info, capture_metadata,
            model_metadata=estimator.model.metadata,
        )
        print(f"实验数据与同名清单已保存到: {save_path}")
        outputs = {
            "xlsx": str(save_path.resolve()),
            "samples": str((save_dir / f"{session_id}.txt").resolve()),
            "manifest": str((save_dir / f"{session_id}.manifest.json").resolve()),
        }
        stopped = capture_metadata.get("stop_reason")
        failed = bool(capture_metadata.get("error")) or stopped not in {"user_exit", None}
        return {
            "status": "failed" if failed else "completed",
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
                                  model_kwargs=None, *, model_metadata=None):
    """Pass explicit new results or full legacy residual vectors without guessing pairs."""
    return save_calibration_results(
        Path(model_path), errors_per_point_9, errors_per_point_add, mean_error_9,
        mean_errors_add, model_names=model_names, model_kwargs=model_kwargs,
        model_metadata=model_metadata,
    )


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
        if path.exists():
            raise FileExistsError(f"文件已存在，请使用新名称：{path}")
    directory.mkdir(parents=True, exist_ok=True)


def save_experiment_outputs(save_dir, session_id, records, experiment_info, capture_metadata,
                            *, model_metadata=None):
    """Save expanded dictionary rows and associate them with the same TXT session."""
    save_dir = Path(save_dir)
    session_id = canonical_session_id(session_id)
    _check_output_names(save_dir, session_id, (".xlsx", ".manifest.json"))
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("新版实时记录必须使用dict，以保留状态和时间字段")
    metadata = dict(capture_metadata)
    if metadata.get("session_id", session_id) != session_id:
        raise ValueError("实时采集与导出会话ID不一致")
    metadata["session_id"] = session_id
    metadata["files"] = {"xlsx": f"{session_id}.xlsx", "txt": f"{session_id}.txt", "manifest": f"{session_id}.manifest.json"}
    metadata["calibration_residual_interpretation"] = "training_fit_residual; not independently measured accuracy"
    metadata["model_feature_contract"] = {
        key: value for key, value in (model_metadata or {}).items() if key != "training"
    }
    text = json.dumps(metadata, ensure_ascii=False, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
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
    path = save_dir / f"{session_id}.xlsx"
    with path.open("xb") as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(experiment_info).to_excel(writer, sheet_name="实验信息", index=False)
            eye_frame.to_excel(writer, sheet_name="眼动数据", index=False)
            pd.DataFrame([{"Key": key, "Value": json.dumps(value, ensure_ascii=False, default=str)} for key, value in metadata.items()]).to_excel(writer, sheet_name="CaptureMetadata", index=False)
    with (save_dir / f"{session_id}.manifest.json").open("x", encoding="utf-8") as output:
        output.write(text)
    return path
