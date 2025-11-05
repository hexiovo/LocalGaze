import cv2
import datetime
from misc import draw_gaze_cloud_overlay
import warnings
import threading
import time
import tkinter as tk


warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def run_realtime_gaze(estimator, draw_cloud=True, screen_width=1920, screen_height=1080,
                      cam_width=320, cam_height=240):
    """
    实时预测眼动位置并在桌面全局显示注视点，同时左上角显示摄像头画面（可隐藏/显示），左上角有退出和切换按钮。
    点击退出后彻底关闭线程和摄像头。
    """
    gaze_data = []

    root = tk.Tk()
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.attributes("-transparentcolor", "white")
    root.config(bg="white")

    canvas = tk.Canvas(root, width=screen_width, height=screen_height,
                       bg="white", highlightthickness=0)
    canvas.pack()

    running = True
    cam_visible = True  # 控制摄像头显示状态

    cam_canvas = tk.Canvas(root, width=cam_width, height=cam_height)
    cam_canvas.place(x=0, y=0)

    # ---------- 按钮事件 ----------
    def on_exit():
        nonlocal running
        running = False
        try:
            root.quit()     # 停止 mainloop
            root.destroy()  # 销毁窗口
        except:
            pass

    def toggle_camera():
        nonlocal cam_visible
        cam_visible = not cam_visible
        cam_canvas.place_forget() if not cam_visible else cam_canvas.place(x=0, y=0)

    # ---------- 左上角按钮 ----------
    exit_button = tk.Button(root, text="退出", command=on_exit)
    exit_button.place(x=10, y=10)

    toggle_button = tk.Button(root, text="隐藏/显示摄像头", command=toggle_camera)
    toggle_button.place(x=80, y=10)

    # ---------- 摄像头预测线程 ----------
    def predict_loop():
        nonlocal running
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while running:
                ret, frame = cap.read()
                if not ret:
                    print("摄像头已关闭，退出循环")
                    break

                features, blink,left_pupil_diameter, right_pupil_diameter = estimator.extract_features(frame,return_more=True)
                if features is not None:
                    timestamp = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    if blink:
                        # 若检测到眨眼，记录为 blink
                        gaze_data.append([timestamp_str,None, None, "1", None,None])
                    else:
                        # 若未眨眼，预测 gaze 坐标并记录 pupil 直径
                        x, y = estimator.predict([features])[0]
                        gaze_data.append([timestamp_str, x, y, 0,left_pupil_diameter, right_pupil_diameter])

                        if draw_cloud:
                            canvas.delete("all")
                            draw_gaze_cloud_overlay(canvas, x, y, radius=20, color_gray="#B4B4B4",
                                                    color_center="#FF0000", border_color="#000000",
                                                    center_radius=4, alpha=0.4)

                # 摄像头画面显示（可隐藏）
                if cam_visible:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (cam_width, cam_height))
                    img = tk.PhotoImage(master=cam_canvas,
                                        width=cam_width, height=cam_height,
                                        data=cv2.imencode(".ppm", frame_resized)[1].tobytes())
                    cam_canvas.create_image(0, 0, anchor="nw", image=img)
                    cam_canvas.image = img

                # 安全更新 Tkinter
                try:
                    root.update_idletasks()
                    root.update()
                except tk.TclError:
                    # Tkinter 已被销毁，安全退出循环
                    break

                time.sleep(0.01)
        finally:
            cap.release()
            running = False  # 确保线程退出

    t = threading.Thread(target=predict_loop, daemon=True)
    t.start()

    root.mainloop()
    return gaze_data

