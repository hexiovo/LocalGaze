from pathlib import Path
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk
import pandas as pd
from Global_data import *
import sys
import numpy as np


kalman_state = {
    "x": None,  # [x, y] 上一次滤波值
    "P": None   # 协方差矩阵 [[Px, 0],[0, Py]]
}


class CalibrationDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None, default_threshold=30, default_max_rounds=10, default_points_per_round=5):
        self.error_threshold = default_threshold
        self.max_rounds = default_max_rounds
        self.points_per_round = default_points_per_round
        self.canceled = True  # 默认取消
        super().__init__(parent, title)

    def body(self, master):
        label_font = ("Arial", 16)
        entry_font = ("Arial", 16)

        tk.Label(master, text="请输入误差阈值（默认30）:", font=label_font).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.entry_threshold = tk.Entry(master, width=10, font=entry_font)
        self.entry_threshold.insert(0, str(self.error_threshold))
        self.entry_threshold.grid(row=0, column=1, pady=5, padx=5, ipady=5)

        tk.Label(master, text="请输入期待最大轮数（默认10）:", font=label_font).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_max_rounds = tk.Entry(master, width=10, font=entry_font)
        self.entry_max_rounds.insert(0, str(self.max_rounds))
        self.entry_max_rounds.grid(row=1, column=1, pady=5, padx=5, ipady=5)

        tk.Label(master, text="请输入每轮点数（默认5）:", font=label_font).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.entry_points = tk.Entry(master, width=10, font=entry_font)
        self.entry_points.insert(0, str(self.points_per_round))
        self.entry_points.grid(row=2, column=1, pady=5, padx=5, ipady=5)

        return self.entry_threshold

    def validate(self):
        # 尝试转换输入，如果失败就提示错误
        try:
            float(self.entry_threshold.get())
            int(self.entry_max_rounds.get())
            int(self.entry_points.get())
            return True
        except ValueError:
            tk.messagebox.showerror("输入错误", "请确保输入为数字")
            return False

    def apply(self):
        # 点击确定时执行
        self.canceled = False  # 只有确定才改为 False
        self.error_threshold = float(self.entry_threshold.get())
        self.max_rounds = int(self.entry_max_rounds.get())
        self.points_per_round = int(self.entry_points.get())

def get_min_calibration_mean(excel_path: Path) -> float:
    """
    读取包含 'NinePointCalibration' 和 'AdditionalCalibration' 两个 sheet 的 Excel 文件，
    计算每张表的平均误差，并返回最小的平均误差值。

    参数:
        excel_path (Path 或 str): Excel 文件路径

    返回:
        float: 两张表平均误差中的最小值，如果无法计算返回 None
    """
    excel_path = Path(excel_path)

    try:
        # 读取两个 sheet
        nine_df = pd.read_excel(excel_path, sheet_name="NinePointCalibration")
        add_df = pd.read_excel(excel_path, sheet_name="AdditionalCalibration")
    except Exception as e:
        print("读取 Excel 出错:", e)
        return None

    # 获取每张表的平均误差列
    nine_mean = nine_df["MeanError"].mean() if "MeanError" in nine_df.columns else None
    add_mean = add_df["MeanErrorPerRound"].mean() if "MeanErrorPerRound" in add_df.columns else None

    # 排除 None，找最小值
    means = [m for m in [nine_mean, add_mean] if m is not None]
    return min(means) if means else None

def get_additional_calibration_mean(excel_path: Path):
    """
    读取 Excel 文件中的 'AdditionalCalibration' sheet，
    返回 'MeanErrorPerRound' 列（含列名）。

    参数:
        excel_path (Path or str): Excel 文件路径

    返回:
        pd.Series 或 None: 如果列存在返回列数据，否则返回 None
    """
    excel_path = Path(excel_path)
    try:
        # 读取 AdditionalCalibration sheet
        add_df = pd.read_excel(excel_path, sheet_name="AdditionalCalibration")

        # 返回 MeanErrorPerRound 列
        if "MeanErrorPerRound" in add_df.columns:
            return add_df["MeanErrorPerRound"]
        else:
            print("⚠ 'MeanErrorPerRound' 列不存在")
            return None
    except Exception as e:
        print(f"⚠ 无法读取 Excel 文件: {e}")
        return None


def draw_gaze_cloud_overlay(canvas, points, radius=30, color_gray="#B4B4B4",
                            color_center="#FF0000", border_color="#000000",
                            center_radius=4, alpha=0.4):
    """
    在 Tkinter Canvas 上绘制最新注视点可视化标记
    - 灰色半透明大圆（注视范围）
    - 红色小圆（注视中心）
    - 黑色外圈边框

    Args:
        canvas: Tkinter Canvas 对象
        points: [(x1, y1), (x2, y2), ...] 最近若干点列表，按屏幕坐标
        radius: 灰色圆半径
        color_gray: 灰色圆颜色 (hex)
        color_center: 中心红点颜色 (hex)
        border_color: 外圈边框颜色 (hex)
        center_radius: 中心红点半径
        alpha: 灰色圆透明度 (0~1)
    """
    if not points:
        return  # 没有点则直接返回

    canvas.delete("all")

    # 获取最新点
    x, y = smooth_points_multi(points)

    # 模拟半透明灰色圆：调整颜色亮度
    def hex_with_alpha(hex_color, alpha_val):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r * alpha_val + 255 * (1 - alpha_val))
        g = int(g * alpha_val + 255 * (1 - alpha_val))
        b = int(b * alpha_val + 255 * (1 - alpha_val))
        return f"#{r:02x}{g:02x}{b:02x}"

    gray_alpha_color = hex_with_alpha(color_gray, alpha)

    # 绘制灰色圆（半透明效果）
    canvas.create_oval(
        x - radius, y - radius,
        x + radius, y + radius,
        fill=gray_alpha_color,
        outline=border_color
    )

    # 绘制红色中心点
    canvas.create_oval(
        x - center_radius, y - center_radius,
        x + center_radius, y + center_radius,
        fill=color_center,
        outline=""
    )


def select_model():
    """弹窗选择模型并返回选择的模型名称，如果取消或关闭则停止程序"""
    result = {}

    def on_confirm():
        # 直接通过 combobox.get() 获取当前选中值
        result['model'] = dropdown.get()
        top.destroy()

    def on_cancel():
        print("⚠ 用户取消输入，程序终止")
        top.destroy()
        sys.exit()

    root = tk.Tk()
    root.withdraw()

    # 创建 Toplevel 窗口
    top = tk.Toplevel(root)
    top.title("选择模型")

    # 设置弹窗的宽度和高度
    window_width = 250
    window_height = 120

    # 计算使弹窗居中的位置
    position_top = int(screen_height/ 4)
    position_left = int(screen_width/ 4)

    # 设置弹窗位置和大小
    top.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")
    top.attributes("-topmost", True)
    top.protocol("WM_DELETE_WINDOW", on_cancel)

    # 添加标签
    tk.Label(top, text="请选择模型（默认为{}）:".format(ALL_MODELS[0])).pack(pady=5)

    # 创建下拉框
    dropdown = ttk.Combobox(top, values=ALL_MODELS, state="readonly")
    dropdown.current(0)  # 设置默认选中第一个
    dropdown.pack(pady=5)

    # 添加确认和取消按钮
    button_frame = tk.Frame(top)
    button_frame.pack(pady=10)
    tk.Button(button_frame, text="确认", width=10, command=on_confirm).pack(side="left", padx=5)
    tk.Button(button_frame, text="取消", width=10, command=on_cancel).pack(side="right", padx=5)

    root.wait_window(top)

    chosen_model = result.get('model')
    print("选择了：", chosen_model)
    return chosen_model


def get_model_info(excel_path: Path):
    """
    从 Excel 文件的 'model_info' sheet 读取 'Model Name' 和 'Model kwargs'，
    分别返回这两列的合并结果（不含列名）。

    参数:
        excel_path (Path or str): Excel 文件路径

    返回:
        tuple(str, str) 或 None: (model_type, model_kwargs)
    """

    excel_path = Path(excel_path)
    try:
        # 读取 model_info sheet
        model_df = pd.read_excel(excel_path, sheet_name="model_info")

        # 检查列是否存在
        required_cols = ["Model Name", "Model kwargs"]
        if not all(col in model_df.columns for col in required_cols):
            print(f"⚠ Excel 中缺少列: {required_cols}")
            return None, None

        # 去除空行
        model_df = model_df[required_cols].dropna(how="all")

        # 将两列分别拼接为单行字符串
        model_type = " ".join(str(v) for v in model_df["Model Name"].tolist())
        model_kwargs = " ".join(str(v) for v in model_df["Model kwargs"].tolist())

        return model_type, model_kwargs

    except Exception as e:
        print(f"⚠ 无法读取 Excel 文件: {e}")
        return None, None

def smooth_points_multi(points, max_tail=10, alpha=0.2, method=None, kalman_Q=0.0001, kalman_R=0.01):
    """
    多平滑方案函数，包括指数平滑、加权平均、低通滤波、简单卡尔曼滤波

    Args:
        points: [(x1, y1), ...] 最近点列表
        max_tail: 最近多少个点参与平滑
        alpha: 指数平滑系数
        method: 指定平滑方法，可覆盖 global_data['smooth_type']
        kalman_Q: 卡尔曼滤波过程噪声方差
        kalman_R: 卡尔曼滤波测量噪声方差

    Returns:
        (x_smooth, y_smooth)
    """
    if not points:
        return None, None

    tail_points = points[-max_tail:]
    if len(tail_points) == 1:
        return tail_points[0]

    smooth_method = method if method else smooth_type

    if smooth_method == "exponential":
        # 指数平滑
        x_smooth, y_smooth = tail_points[0]
        for px, py in tail_points[1:]:
            x_smooth = alpha * px + (1 - alpha) * x_smooth
            y_smooth = alpha * py + (1 - alpha) * y_smooth
        return x_smooth, y_smooth

    elif smooth_method == "weighted_average":
        # 权重随时间递增，越新点权重越大
        n = len(tail_points)
        weights = [i / n for i in range(1, n+1)]
        x_smooth = sum(px * w for (px, _), w in zip(tail_points, weights)) / sum(weights)
        y_smooth = sum(py * w for (_, py), w in zip(tail_points, weights)) / sum(weights)
        return x_smooth, y_smooth

    elif smooth_method == "lowpass":
        # 一阶低通滤波
        x_smooth, y_smooth = tail_points[0]
        for px, py in tail_points[1:]:
            x_smooth += alpha * (px - x_smooth)
            y_smooth += alpha * (py - y_smooth)
        return x_smooth, y_smooth



    elif smooth_method == "kalman":

        # 简单卡尔曼滤波（单维分别处理 x, y）
        global kalman_state

        x_measure, y_measure = tail_points[-1]
        if kalman_state["x"] is None:
            kalman_state["x"] = np.array([x_measure, y_measure], dtype=float)
            kalman_state["P"] = np.array([1.0, 1.0], dtype=float)  # 初始协方差

        x_prior = kalman_state["x"]
        P_prior = kalman_state["P"]

        # 卡尔曼增益（单维）
        K = P_prior / (P_prior + kalman_R)

        # 更新估计（逐元素）
        x_post = x_prior + K * (np.array([x_measure, y_measure]) - x_prior)

        # 更新协方差（逐元素）
        P_post = (1 - K) * P_prior + kalman_Q
        kalman_state["x"] = x_post
        kalman_state["P"] = P_post

        return float(x_post[0]), float(x_post[1])


    else:
        # 未知方案，直接返回最新点
        return tail_points[-1]