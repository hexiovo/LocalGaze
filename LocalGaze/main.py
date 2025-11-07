from eyetracking_pipeline import run_eye_tracking_workflow

import tkinter as tk
from tkinter import messagebox
from Global_data import *
if __name__ == "__main__":
    """
    EyeTrax 眼动追踪主程序入口
    """
    # 创建 Tkinter 根窗口（用于弹窗）
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes("-topmost", True)  # 确保弹窗在最前

    # 弹窗展示系统说明
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


    # 调用封装好的工作流程函数
    run_eye_tracking_workflow()