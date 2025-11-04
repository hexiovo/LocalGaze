import random
import cv2
import numpy as np
from typing import List, Tuple
import time

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size


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
    """
    九点校准，带实时注视点概率云显示
    Returns:
        errors_per_point: 每个校准点误差列表
        mean_error: 平均误差
    """

    sw, sh = get_screen_size()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("摄像头未成功打开")
        return None, None

    # 如果之前有 Tkinter 弹窗，确保销毁，避免阻塞
    try:
        import tkinter as tk
        root = tk._default_root
        if root:
            root.update()
            root.destroy()
    except Exception:
        pass

    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return None, None

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
    pts = compute_grid_points(order, sw, sh)

    # 捕获特征和目标坐标，实时显示概率云
    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)

    # 确保释放摄像头和窗口
    cap.release()
    cv2.destroyAllWindows()

    if res is None:
        return None, None

    feats, targs = res
    if feats:
        feats = np.array(feats)
        targs = np.array(targs)

        # 训练模型
        gaze_estimator.train(feats, targs)

        # 预测训练点
        preds = gaze_estimator.predict(feats)

        # 每帧误差
        errors_per_frame = np.linalg.norm(preds - targs, axis=1)

        # 按目标点划分，每个目标点对应的帧索引
        unique_points, indices = np.unique(targs, axis=0, return_inverse=True)

        # 存储每个点的所有帧误差
        every_point_errors = [[] for _ in range(len(unique_points))]
        for frame_idx, point_idx in enumerate(indices):
            every_point_errors[point_idx].append(errors_per_frame[frame_idx])

        # 计算每个点平均误差
        errors_per_point = [np.mean(errors) if errors else np.nan for errors in every_point_errors]

        # 总平均误差（所有帧误差的平均）
        mean_error = np.mean(errors_per_frame)

        print("Per-point average errors:", errors_per_point)
        print(f"Mean calibration error: {mean_error:.2f} pixels")

        return every_point_errors, mean_error

    return None, None


def run_additional_random_calibration(
    gaze_estimator,
    camera_index: int = 0,
    points_per_round: int = 5,
    error_threshold: float = 30.0,
    max_rounds: int = 10
):
    """
    随机点补充校准流程：
    - 每轮重新打开摄像头并采集若干点；
    - 计算每个点的逐帧误差与每轮平均误差；
    - 返回：
        every_point_errors_per_round: list[list[np.ndarray]]，每轮每点的帧误差
        mean_errors_per_round: list[float]，每轮平均误差
    """
    from eyetrax.calibration.adaptive import _pulse_and_capture_live
    import cv2, time
    import numpy as np

    sw, sh = get_screen_size()
    sampler = BlueNoiseSampler(sw, sh)

    round_count = 0
    current_mean_error = float('inf')

    every_point_errors_per_round = []  # 每轮每点的误差帧
    mean_errors_per_round = []         # 每轮平均误差

    print("🟢 开始随机补充校准...")

    while current_mean_error > error_threshold and round_count < max_rounds:
        round_count += 1
        print(f"\n—— 第 {round_count} 轮 ——")

        pts = sampler.sample(points_per_round)

        # 每轮单独打开摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ 无法打开摄像头。")
            break

        try:
            res = _pulse_and_capture_live(
                gaze_estimator, cap, pts, sw, sh, show_real_time=True
            )
        except Exception as e:
            print(f"⚠ 第 {round_count} 轮运行出错: {e}")
            res = None
        finally:
            cap.release()
            cv2.waitKey(200)
            cv2.destroyAllWindows()
            time.sleep(0.3)

        # 数据有效性检查
        if not res or res[0] is None or len(res[0]) == 0:
            print(f"  ❌ 第 {round_count} 轮无有效数据，跳过。")
            mean_errors_per_round.append(np.nan)
            continue

        feats, targs = map(np.array, res)
        if feats.shape[0] == 0:
            print(f"  ⚠ 第 {round_count} 轮采集为空，继续下一轮。")
            mean_errors_per_round.append(np.nan)
            continue

        # ✅ 训练并预测
        gaze_estimator.train(feats, targs)
        preds = gaze_estimator.predict(feats)

        # ✅ 计算逐帧误差
        errors_per_frame = np.linalg.norm(preds - targs, axis=1)

        # ✅ 按目标点聚类
        unique_points, indices = np.unique(targs, axis=0, return_inverse=True)
        every_point_errors = [errors_per_frame[indices == i] for i in range(len(unique_points))]

        # ✅ 计算平均误差
        errors_per_point = [np.mean(e) for e in every_point_errors if len(e) > 0]
        current_mean_error = np.nanmean(errors_per_point)

        # ✅ 保存结果
        every_point_errors_per_round.append(every_point_errors)
        mean_errors_per_round.append(current_mean_error)

        print(f"  ✅ 平均误差 = {current_mean_error:.2f} 像素")

        if current_mean_error <= error_threshold:
            print(f"🎯 达到目标精度（平均误差 {current_mean_error:.2f}px），结束校准。")
            break

    print("\n✅ 随机补充校准完成。")
    return every_point_errors_per_round, mean_errors_per_round




