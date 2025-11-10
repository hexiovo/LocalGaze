from eyetracking_pipeline import run_eye_tracking_workflow
from misc import setup_logger

import tkinter as tk
from tkinter import messagebox
import os
import sys

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    logger, log_file = setup_logger()
    logger.info("程序启动：EyeTrax 实时眼动追踪系统")

    # ---------- 重定向 C++ 层 stderr 到日志文件 ----------
    # 打开同一个日志文件追加模式
    sys.__stderr__.flush()
    f_stderr = open(log_file, 'a', encoding='utf-8')
    os.dup2(f_stderr.fileno(), sys.stderr.fileno())

    # ---------- Tkinter 弹窗 ----------
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    logger.info("Tkinter 根窗口初始化完成，弹窗即将显示")

    try:
        messagebox.showinfo(
            title="EyeTrax 实时眼动追踪系统",
            message=(
                "=== EyeTrax 实时眼动追踪系统 ===\n\n"
                "流程说明：\n"
                "1. 选择是否已有历史模型\n"
                "2. 九点校准（如果是新模型）\n"
                "3. 随机点补充校准\n"
                "4. 实时眼动追踪窗口\n\n"
            )
        )
        logger.info("系统说明弹窗已显示，用户已确认")
    except Exception as e:
        logger.exception(f"弹窗显示失败: {e}")

    # ---------- 调用工作流程 ----------
    try:
        run_eye_tracking_workflow()  # 保持原样，不改动
    except Exception as e:
        logger.exception(f"工作流程执行异常: {e}")

    logger.info("程序结束")