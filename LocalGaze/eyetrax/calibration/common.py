import time

import cv2
import numpy as np


def compute_grid_points(order, sw: int, sh: int, margin_ratio: float = 0.10):
    """
    Translate grid (row, col) indices into absolute pixel locations
    """
    if not order:
        return []

    max_r = max(r for r, _ in order)
    max_c = max(c for _, c in order)

    mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
    gw, gh = sw - 2 * mx, sh - 2 * my

    step_x = 0 if max_c == 0 else gw / max_c
    step_y = 0 if max_r == 0 else gh / max_r

    return [(mx + int(c * step_x), my + int(r * step_y)) for r, c in order]


def wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, dur: int = 2) -> bool:
    """
    等待检测到人脸（未眨眼），然后显示倒计时圆环
    左上角显示摄像头画面，中间有黑色矩形用于标定人头位置
    """
    import cv2
    import numpy as np
    import time

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fd_start = None
    countdown = False

    # 中间黑框大小 (可调)
    head_box_w, head_box_h = sw // 4, sh // 3
    head_box_x, head_box_y = (sw - head_box_w) // 2, (sh - head_box_h) // 2

    # 左上角摄像头窗口大小
    cam_w, cam_h = 480, 360

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # 摄像头丢帧，黑色占位
            frame_small = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            face_detected = False
        else:
            # 保证三通道
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame_small = cv2.resize(frame, (cam_w, cam_h))

            # 特征提取，不影响显示
            f, blink = gaze_estimator.extract_features(frame)
            face_detected = f is not None and not blink

        # ---------------- 创建主画布 ----------------
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

        # ---------------- 左上角显示摄像头 ----------------
        canvas[0:cam_h, 0:cam_w] = frame_small

        # ---------------- 中间黑框 ----------------
        cv2.rectangle(
            canvas,
            (head_box_x, head_box_y),
            (head_box_x + head_box_w, head_box_y + head_box_h),
            (0, 0, 0),
            3,
        )

        now = time.time()

        if face_detected:
            if not countdown:
                fd_start = now
                countdown = True

            elapsed = now - fd_start
            if elapsed >= dur:
                return True

            # Ease-in-out 倒计时动画
            t = elapsed / dur
            e = t * t * (3 - 2 * t)
            ang = 360 * (1 - e)

            # 中心倒计时圆环
            center_x, center_y = sw // 2, sh // 2
            radius = 50
            cv2.ellipse(
                canvas,
                (center_x, center_y),
                (radius, radius),
                0,
                -90,
                -90 + ang,
                (0, 255, 0),
                -1,
            )
        else:
            countdown = False
            fd_start = None
            # 提示文字
            txt = "Face not detected"
            fs = 3  # 字体大小
            thick = 4
            size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            tx = (sw - size[0]) // 2
            ty = (sh + size[1]) // 2
            cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), thick)

        # 显示画布
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) == 27:  # 按 Esc 退出
            return False



def _pulse_and_capture(
    gaze_estimator,
    cap,
    pts,
    sw: int,
    sh: int,
    pulse_d: float = 1.0,
    cd_d: float = 1.0,
):
    """
    Shared pulse-and-capture loop for each calibration point
    """
    feats, targs = [], []

    for x, y in pts:
        # pulse
        ps = time.time()
        final_radius = 20
        while True:
            e = time.time() - ps
            if e > pulse_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
            final_radius = radius
            cv2.circle(canvas, (x, y), radius, (0, 255, 0), -1)
            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None
        # capture
        cs = time.time()
        while True:
            e = time.time() - cs
            if e > cd_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
            t = e / cd_d
            ease = t * t * (3 - 2 * t)
            ang = 360 * (1 - ease)
            cv2.ellipse(canvas, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None
            ft, blink = gaze_estimator.extract_features(frame)
            if ft is not None and not blink:
                feats.append(ft)
                targs.append([x, y])

    return feats, targs
