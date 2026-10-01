from __future__ import annotations

import random
import time
import uuid
from typing import List, Tuple

import cv2
import numpy as np

from eyetrax.calibration.common import (
    CALIBRATION_SETTLE_NS,
    MIN_OPEN_EYE_FRAMES,
    MIN_OPEN_EYE_SPAN_NS,
    _coverage_complete,
    _frame_id,
)
from eyetrax.calibration.nine_point import run_9_point_calibration
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.draw import draw_cursor
from eyetrax.utils.screen import get_screen_size
from eyetrax.calibration.misc import draw_gaze_cloud


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


def _pulse_and_capture_live(
    gaze_estimator, cap, pts, sw, sh, show_real_time=True,
    *, visit_id=None, return_metadata=False,
):
    """Use the shared receipt/settling/coverage rules for additional targets."""
    from eyetrax.calibration.common import _pulse_and_capture
    window_name = "Adaptive Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return _pulse_and_capture(
        gaze_estimator, cap, pts, sw, sh, visit_id=visit_id,
        return_metadata=return_metadata, window_name=window_name,
        show_real_time=show_real_time, show_camera=True,
    )


def run_adaptive_calibration(
    gaze_estimator: GazeEstimator,
    *,
    num_random_points: int = 60,
    retrain_every: int = 10,
    show_predictions: bool = True,
    camera_index: int = 0,
) -> None:
    summary, _ = run_9_point_calibration(gaze_estimator, camera_index=camera_index)
    if summary is None:
        return
    sw, sh = get_screen_size()
    if hasattr(gaze_estimator, "set_screen_info"):
        gaze_estimator.set_screen_info(sw, sh)
    sampler = BlueNoiseSampler(sw, sh)
    if retrain_every <= 0:
        raise ValueError("retrain_every must be positive")
    points = sampler.sample(num_random_points)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return

    all_feats: list[np.ndarray] = []
    all_targs: list[list[float]] = []
    all_samples: list[dict] = []
    visit_id = uuid.uuid4().hex
    try:
        for chunk_start in range(0, len(points), retrain_every):
            chunk = points[chunk_start:chunk_start + retrain_every]
            result = _pulse_and_capture_live(
                gaze_estimator,
                cap,
                chunk,
                sw,
                sh,
                show_real_time=show_predictions,
                visit_id=visit_id,
                return_metadata=True,
            )
            if not result:
                break
            features, targets, capture_metadata = result
            if not features or not all(item["accepted"] for item in capture_metadata["target_statuses"]):
                continue
            all_feats.extend(features)
            all_targs.extend(targets)
            all_samples.extend(capture_metadata["samples"])
            gaze_estimator.train(
                np.asarray(all_feats),
                np.asarray(all_targs),
                target_ids=[tuple(sample["target_id"]) for sample in all_samples],
                visit_ids=[sample["visit_id"] for sample in all_samples],
                frame_ids=[sample["frame_id"] for sample in all_samples],
                use_group_weights=getattr(gaze_estimator, "calibration_use_group_weights", True),
            )
    finally:
        cap.release()
        cv2.destroyAllWindows()


__all__ = [
    "BlueNoiseSampler",
    "_pulse_and_capture_live",
    "run_adaptive_calibration",
]
