import cv2
import datetime
from misc import draw_gaze_cloud_overlay
import warnings
import threading
import time
import tkinter as tk

from Global_data import *

import os

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def run_realtime_gaze(estimator, save_name,
                      draw_cloud=True, screen_width=1920, screen_height=1080,
                      cam_width=320, cam_height=240):
    """
    实时预测眼动位置并在桌面全局显示注视点，同时左上角显示摄像头画面。
    实时将 gaze 数据保存为 Data/<save_name>.txt 文件。
    """
    gaze_data = []
    running = True
    cam_visible = True

    # ---------- 路径准备 ----------
    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f"{save_name}.txt")

    # 若文件已存在则清空重写
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Timestamp\tX\tY\tBlink\tLeftPupil\tRightPupil\n")

    # ---------- Tkinter 初始化 ----------
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.attributes("-transparentcolor", "white")
    root.config(bg="white")

    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white", highlightthickness=0)
    canvas.pack()

    cam_canvas = tk.Canvas(root, width=cam_width, height=cam_height)
    cam_canvas.place(x=0, y=0)

    def on_exit():
        nonlocal running
        running = False

        # ---------- 立即停止 Tkinter 的 after 调度 ----------
        if hasattr(root, "_after_ids"):
            for after_id in root._after_ids:
                try:
                    root.after_cancel(after_id)
                except:
                    pass
            root._after_ids.clear()

        # ---------- 强制关闭摄像头 ----------
        try:
            if cap.isOpened():
                cap.release()
        except:
            pass

        # ---------- 关闭所有 OpenCV 窗口（保险起见） ----------
        try:
            cv2.destroyAllWindows()
        except:
            pass

        # ---------- 安全销毁 Tkinter ----------
        try:
            root.quit()  # 让 mainloop() 停止
            root.destroy()  # 销毁窗口
        except:
            pass

    def schedule_safe(func, delay):
        """安全封装 root.after：在 root 销毁时不报错"""
        if running:
            after_id = root.after(delay, func)
            # 记录所有 after ID，方便退出时取消
            if not hasattr(root, "_after_ids"):
                root._after_ids = []
            root._after_ids.append(after_id)

    def toggle_camera():
        nonlocal cam_visible
        cam_visible = not cam_visible
        cam_canvas.place_forget() if not cam_visible else cam_canvas.place(x=0, y=0)

    exit_button = tk.Button(root, text="退出", command=on_exit)
    exit_button.place(x=10, y=10)

    toggle_button = tk.Button(root, text="隐藏/显示摄像头", command=toggle_camera)
    toggle_button.place(x=80, y=10)

    # ---------- 摄像头与预测 ----------
    cap = cv2.VideoCapture(camera_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    gaze_history = []

    latest_frame = [None]
    frame_lock = threading.Lock()

    def capture_loop():
        """后台线程：读取摄像头帧"""
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            with frame_lock:
                latest_frame[0] = frame.copy()
            time.sleep(0.01)
        cap.release()

    def update_gui():
        """主线程：更新界面 + 实时预测 + 实时写入文件"""
        if not running or not root.winfo_exists():
            return

        with frame_lock:
            frame = latest_frame[0]

        if frame is not None:
            features, blink, left_pupil_diameter, right_pupil_diameter = estimator.extract_features(
                frame, return_more=True)
            if features is not None:
                timestamp = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                if blink:
                    row = [timestamp_str, None, None, 1, None, None]
                else:
                    x, y = estimator.predict([features])[0]
                    row = [timestamp_str, x, y, 0, left_pupil_diameter, right_pupil_diameter]

                    if draw_cloud:
                        gaze_history.append((x, y))
                        if len(gaze_history) > MAX_POINTS:
                            gaze_history.pop(0)
                        draw_gaze_cloud_overlay(canvas, gaze_history,
                                                radius=20, color_gray="#B4B4B4",
                                                color_center="#FF0000", border_color="#000000",
                                                center_radius=4, alpha=0.4)

                # 记录到内存
                gaze_data.append(row)

                # 实时写入文件
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write("\t".join(str(v) if v is not None else "" for v in row) + "\n")

            # 摄像头画面显示
            if cam_visible:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (cam_width, cam_height))
                img = tk.PhotoImage(master=cam_canvas,
                                    width=cam_width, height=cam_height,
                                    data=cv2.imencode(".ppm", frame_resized)[1].tobytes())
                cam_canvas.create_image(0, 0, anchor="nw", image=img)
                cam_canvas.image = img

        # 每10ms刷新一次
        schedule_safe(update_gui, 10)

    # 启动
    threading.Thread(target=capture_loop, daemon=True).start()
    root.after(10, update_gui)
    root.mainloop()

    running = False
    return gaze_data
