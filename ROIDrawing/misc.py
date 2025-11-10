import tkinter as tk
from Global_data import *
import sys
import logging
import os
import datetime

def show_instructions(root):
    import tkinter.font as font
    # 半透明或普通提示窗口
    instr_win = tk.Toplevel(root)
    instr_win.title("操作提示")
    instr_win.resizable(False, False)
    instr_win.attributes("-topmost", True)
    instr_win.grab_set()  # 阻塞主窗口，必须先点击确认

    # 弹窗大小
    win_w, win_h = 400, 150

    # 屏幕中心坐标
    x = (SCREEN_W - win_w) // 4
    y = (SCREEN_H - win_h) // 4
    instr_win.geometry(f"{win_w}x{win_h}+{x}+{y}")

    # 标签提示文字
    label_font = font.Font(size=14)  # 字号14
    label = tk.Label(
        instr_win,
        text="操作说明：\n左键拖拽已绘制的 ROI\n右键绘制新的 ROI",
        font=label_font,
        justify='center'
    )
    label.pack(expand=True, pady=20)

    # 确认按钮
    def on_confirm():
        instr_win.destroy()

    btn = tk.Button(instr_win, text="确定", command=on_confirm)
    btn.pack(pady=10)


def setup_logger():
    """
    初始化日志记录器，返回 logger 对象及日志文件路径。
    - 文件日志保存所有输出
    - 控制台只显示 WARNING 及以上
    - stdout/stderr 重定向到 logger 文件，捕获 C++ 层输出
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "ROIDrawing_log")  # 修改日志文件夹名称
    os.makedirs(log_dir, exist_ok=True)

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{start_time}.txt")

    # ---------- Logger 配置 ----------
    logger = logging.getLogger("ROIDrawing")
    logger.setLevel(logging.DEBUG)  # 捕获所有日志

    # 文件 handler 保存所有日志
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # 控制台 handler 只显示 WARNING+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    if not logger.hasHandlers():  # 防止重复添加 handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # ---------- 重定向 stdout 到 logger ----------
    class StreamToLogger:
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level
        def write(self, message):
            message = message.strip()
            if message:
                self.logger.log(self.level, message)
        def flush(self):
            pass

    sys.stdout = StreamToLogger(logger, logging.INFO)

    # ---------- 捕获未处理异常 ----------
    def excepthook(type, value, tb):
        import traceback
        logger.error("未捕获异常:\n" + "".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = excepthook

    # ---------- 重定向 C++ 层 stderr ----------
    sys.__stderr__.flush()
    f_stderr = open(log_file, 'a', encoding='utf-8')
    os.dup2(f_stderr.fileno(), sys.stderr.fileno())

    return logger, log_file

