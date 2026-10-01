from __future__ import annotations

import random
import time
import uuid
from typing import List, Tuple

import cv2
import numpy as np

from eyetrax.calibration.common import (
    CalibrationResidualResult,
    _pulse_and_capture,
    compute_grid_points,
    calibration_gate_config_record,
    summarize_fit_residuals,
    record_capture_metadata,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size


_VISIT_PROMPTS = (
    "Visit 1 of 3: Keep your head centered and comfortable.",
    "Visit 2 of 3: Keep your head comfortably slightly left.",
    "Visit 3 of 3: Keep your head comfortably slightly right.",
)


def _show_visit_prompt(sw: int, sh: int, prompt: str, *, window_name="Calibration") -> bool:
    """Show a posture instruction; it is never recorded as a sample label."""

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
    scale = max(0.7, min(sw, sh) / 1400)
    text_size, _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(
        canvas, prompt, ((sw - text_size[0]) // 2, sh // 2),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2,
    )
    cv2.imshow(window_name, canvas)
    return cv2.waitKey(1800) != 27


class BlueNoiseSampler:
    def __init__(self, w: int, h: int, margin: float = 0.08):
        self.w, self.h = w, h
        self.mx, self.my = int(w * margin), int(h * margin)

    def sample(self, n: int, k: int = 30) -> List[Tuple[int, int]]:
        pts: List[Tuple[int, int]] = []
        for _ in range(n):
            best, best_d2 = None, -1
            for _ in range(k):
                x = random.randint(self.mx, self.w - self.mx)
                y = random.randint(self.my, self.h - self.my)
                d2 = (
                    min((x - px) ** 2 + (y - py) ** 2 for px, py in pts) if pts else 1e9
                )
                if d2 > best_d2:
                    best, best_d2 = (x, y), d2
            pts.append(best)
        return pts


def run_9_point_calibration(gaze_estimator, camera_index: int = 0):
    """Run three complete nine-point visits and return in-sample fit residuals."""

    sw, sh = get_screen_size()
    if hasattr(gaze_estimator, "set_screen_info"):
        gaze_estimator.set_screen_info(sw, sh)
    gate_config = calibration_gate_config_record()
    gaze_estimator.last_calibration_failure = None
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("摄像头未成功打开")
        cap.release()
        gaze_estimator.last_calibration_failure = {
            "stage": "camera_open", "reason": "camera_open_failed",
            "gate_config": gate_config, "visits": [],
        }
        return None, None

    try:
        import tkinter as tk

        root = tk._default_root
        if root:
            root.update()
            root.destroy()
    except Exception:
        pass

    order = [
        (1, 1),
        (0, 0),
        (2, 0),
        (0, 2),
        (2, 2),
        (1, 0),
        (0, 1),
        (2, 1),
        (1, 2),
    ]
    points = compute_grid_points(order, sw, sh)
    visit_metadata = []
    all_features = []
    all_targets = []
    all_samples = []
    try:
        for visit_index, prompt in enumerate(_VISIT_PROMPTS, 1):
            if not _show_visit_prompt(sw, sh, prompt):
                gaze_estimator.last_calibration_failure = {
                    "stage": "visit_prompt", "reason": "cancelled_by_user",
                    "failed_visit_index": visit_index, "gate_config": gate_config,
                    "visits": visit_metadata,
                }
                return None, None
            if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
                gaze_estimator.last_calibration_failure = {
                    "stage": "visit_readiness", "reason": "cancelled_before_visit",
                    "failed_visit_index": visit_index, "gate_config": gate_config,
                    "visits": visit_metadata,
                }
                return None, None
            visit_id = uuid.uuid4().hex
            try:
                result = _pulse_and_capture(
                    gaze_estimator, cap, points, sw, sh,
                    visit_id=visit_id, return_metadata=True,
                )
            except Exception as exc:
                gaze_estimator.last_calibration_failure = {
                    "stage": "fixed_target_capture",
                    "reason": f"capture_error:{type(exc).__name__}",
                    "failed_visit_index": visit_index,
                    "error": str(exc), "gate_config": gate_config,
                    "visits": visit_metadata,
                }
                return None, None
            if result is None:
                gaze_estimator.last_calibration_failure = {
                    "stage": "fixed_target_capture", "reason": "capture_returned_no_metadata",
                    "failed_visit_index": visit_index, "gate_config": gate_config,
                    "visits": visit_metadata,
                }
                return None, None
            features, targets, capture_metadata = result
            capture_metadata["visit_index"] = visit_index
            capture_metadata["instruction_only"] = prompt
            capture_metadata["posture_label"] = None
            capture_metadata["posture_measured"] = False
            visit_metadata.append(capture_metadata)

            expected_targets = [tuple(point) for point in points]
            observed_targets = [tuple(status.get("target_id", ())) for status in capture_metadata.get("target_statuses", [])]
            visit_complete = (
                len(observed_targets) == len(expected_targets)
                and observed_targets == expected_targets
                and all(status.get("accepted") is True for status in capture_metadata.get("target_statuses", []))
            )
            if not visit_complete:
                failed_target = next(
                    (status for status in capture_metadata.get("target_statuses", []) if not status.get("accepted")),
                    None,
                )
                gaze_estimator.last_calibration_failure = {
                    "stage": "fixed_target_capture",
                    "reason": (failed_target or {}).get("reason", "incomplete_visit"),
                    "failed_visit_index": visit_index,
                    "failed_target": failed_target,
                    "gate_config": gate_config,
                    "visits": visit_metadata,
                }
                print(f"九点校准第 {visit_index} 次 visit 未完成；不会生成成功模型或工作簿。")
                return None, None
            if len(features) != len(capture_metadata.get("samples", [])) or len(targets) != len(features):
                gaze_estimator.last_calibration_failure = {
                    "stage": "capture_metadata_validation",
                    "reason": "sample_metadata_count_mismatch",
                    "failed_visit_index": visit_index, "gate_config": gate_config,
                    "visits": visit_metadata,
                }
                return None, None
            all_features.extend(features)
            all_targets.extend(targets)
            all_samples.extend(capture_metadata["samples"])
    except Exception as exc:
        gaze_estimator.last_calibration_failure = {
            "stage": "fixed_nine_point_workflow",
            "reason": f"workflow_error:{type(exc).__name__}",
            "error": str(exc), "gate_config": gate_config,
            "visits": visit_metadata,
        }
        return None, None
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(visit_metadata) != len(_VISIT_PROMPTS) or not all_features:
        if gaze_estimator.last_calibration_failure is None:
            gaze_estimator.last_calibration_failure = {
                "stage": "visit_coverage", "reason": "three_complete_visits_required",
                "gate_config": gate_config, "visits": visit_metadata,
            }
        return None, None

    X = np.asarray(all_features, dtype=np.float32)
    Y = np.asarray(all_targets, dtype=np.float64)
    if X.ndim != 2 or Y.shape != (len(X), 2) or len(all_samples) != len(X):
        gaze_estimator.last_calibration_failure = {
            "stage": "training_input_validation", "reason": "sample_shape_mismatch",
            "gate_config": gate_config, "visits": visit_metadata,
        }
        return None, None
    try:
        gaze_estimator.train(
            X,
            Y,
            target_ids=[tuple(sample["target_id"]) for sample in all_samples],
            visit_ids=[sample["visit_id"] for sample in all_samples],
            frame_ids=[sample["frame_id"] for sample in all_samples],
            use_group_weights=getattr(gaze_estimator, "calibration_use_group_weights", True),
        )
        predictions = np.asarray(gaze_estimator.predict(X), dtype=np.float64)
        if predictions.shape != Y.shape or not np.all(np.isfinite(predictions)):
            raise ValueError("training predictions are not finite Nx2 coordinates")
    except Exception as exc:
        gaze_estimator.last_calibration_failure = {
            "stage": "model_training", "reason": f"training_error:{type(exc).__name__}",
            "error": str(exc), "gate_config": gate_config, "visits": visit_metadata,
        }
        return None, None

    frame_residuals = np.linalg.norm(predictions - Y, axis=1)
    summaries = []
    sample_offset = 0
    for metadata in visit_metadata:
        visit_samples = metadata["samples"]
        visit_count = len(visit_samples)
        visit_residuals = frame_residuals[sample_offset:sample_offset + visit_count]
        visit_targets = Y[sample_offset:sample_offset + visit_count]
        record_capture_metadata(gaze_estimator, metadata, visit_residuals)
        summaries.append(summarize_fit_residuals(
            visit_targets,
            visit_residuals,
            stage="nine_point",
            visit_id=metadata["visit_id"],
        ))
        sample_offset += visit_count
    mean_residual = float(np.mean([summary.mean_residual_px for summary in summaries]))
    gaze_estimator.last_calibration_failure = None
    print(f"九点校准训练内拟合残差均值: {mean_residual:.2f} px")
    return summaries, mean_residual


def run_additional_random_calibration(
    gaze_estimator,
    camera_index: int = 0,
    points_per_round: int = 5,
    error_threshold: float = 30.0,
    max_rounds: int = 10,
):
    """Append each additional visit and refit using all retained calibration samples.

    Returns one explicit ``CalibrationResidualResult`` per completed round and
    the corresponding target-balanced in-sample fit residual means.
    """

    from eyetrax.calibration.adaptive import _pulse_and_capture_live

    sw, sh = get_screen_size()
    if hasattr(gaze_estimator, "set_screen_info"):
        gaze_estimator.set_screen_info(sw, sh)
    sampler = BlueNoiseSampler(sw, sh)
    round_count = 0
    current_mean_residual = float("inf")
    residual_results: list[CalibrationResidualResult] = []
    mean_residuals_per_round: list[float] = []

    print("开始随机补充校准（报告训练内拟合残差）...")
    while current_mean_residual > error_threshold and round_count < max_rounds:
        round_count += 1
        print(f"\n—— 第 {round_count} 轮 ——")
        points = sampler.sample(points_per_round)
        visit_id = uuid.uuid4().hex
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("无法打开摄像头。")
            cap.release()
            break

        try:
            result = _pulse_and_capture_live(
                gaze_estimator,
                cap,
                points,
                sw,
                sh,
                show_real_time=True,
                visit_id=visit_id,
                return_metadata=True,
            )
        except Exception as exc:
            print(f"第 {round_count} 轮运行出错: {exc}")
            result = None
        finally:
            cap.release()
            cv2.waitKey(200)
            cv2.destroyAllWindows()
            time.sleep(0.3)

        if result is None:
            print(f"第 {round_count} 轮已取消或失败，停止补充校准。")
            break
        features, targets, capture_metadata = result
        if not features or not all(status["accepted"] for status in capture_metadata["target_statuses"]):
            print(f"第 {round_count} 轮没有满足帧数/覆盖时长条件的开放眼样本，跳过。")
            continue

        X = np.asarray(features, dtype=np.float32)
        Y = np.asarray(targets, dtype=np.float64)
        samples = capture_metadata["samples"]
        gaze_estimator.train(
            X,
            Y,
            target_ids=[tuple(sample["target_id"]) for sample in samples],
            visit_ids=[sample["visit_id"] for sample in samples],
            frame_ids=[sample["frame_id"] for sample in samples],
            use_group_weights=getattr(gaze_estimator, "calibration_use_group_weights", True),
        )
        predictions = gaze_estimator.predict(X)
        frame_residuals = np.linalg.norm(predictions - Y, axis=1)
        record_capture_metadata(gaze_estimator, capture_metadata, frame_residuals)
        summary = summarize_fit_residuals(
            Y,
            frame_residuals,
            stage="additional_random",
            visit_id=visit_id,
        )
        current_mean_residual = summary.mean_residual_px if summary.mean_residual_px is not None else float("inf")
        residual_results.append(summary)
        mean_residuals_per_round.append(current_mean_residual)
        print(f"第 {round_count} 轮训练内拟合残差均值 = {current_mean_residual:.2f} px")

        if current_mean_residual <= error_threshold:
            print(f"达到拟合残差阈值（{current_mean_residual:.2f}px），结束校准。")
            break

    print("随机补充校准结束。")
    return residual_results, mean_residuals_per_round


__all__ = [
    "BlueNoiseSampler",
    "run_9_point_calibration",
    "run_additional_random_calibration",
]
