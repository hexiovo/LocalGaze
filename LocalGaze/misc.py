from pathlib import Path
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk
import pandas as pd
from Global_data import *
import sys

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


def draw_gaze_cloud_overlay(canvas, x, y, radius=30, color_gray="#B4B4B4", color_center="#FF0000", border_color="#000000", center_radius=4, alpha=0.4):
    """
    在 Tkinter Canvas 上绘制注视点可视化标记（模拟原 draw_gaze_cloud 功能）
    - 灰色半透明大圆（注视范围）
    - 红色小圆（注视中心）
    - 黑色外圈边框

    Args:
        canvas: Tkinter Canvas 对象
        x, y: 注视点中心坐标
        radius: 灰色圆半径
        color_gray: 灰色圆颜色 (hex)
        color_center: 中心红点颜色 (hex)
        border_color: 外圈边框颜色 (hex)
        center_radius: 中心红点半径
        alpha: 灰色圆透明度 (0~1)
    """
    # 模拟半透明灰色圆：调整颜色亮度
    def hex_with_alpha(hex_color, alpha):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r * alpha + 255 * (1 - alpha))
        g = int(g * alpha + 255 * (1 - alpha))
        b = int(b * alpha + 255 * (1 - alpha))
        return f"#{r:02x}{g:02x}{b:02x}"

    gray_alpha_color = hex_with_alpha(color_gray, alpha)

    # 绘制灰色圆（半透明效果）
    canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=gray_alpha_color, outline=border_color)

    # 绘制红色中心点
    canvas.create_oval(x - center_radius, y - center_radius, x + center_radius, y + center_radius, fill=color_center, outline="")

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

    top = tk.Toplevel(root)
    top.title("选择模型")
    top.geometry("250x120")
    top.attributes("-topmost", True)
    top.protocol("WM_DELETE_WINDOW", on_cancel)

    tk.Label(top, text="请选择模型（默认为{}）:".format(ALL_MODELS[0])).pack(pady=5)

    dropdown = ttk.Combobox(top, values=ALL_MODELS, state="readonly")
    dropdown.current(0)  # 设置默认选中第一个
    dropdown.pack(pady=5)

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
    读取 Excel 文件中的 'model_info' sheet，
    返回 'model_type' 和 'model_kwargs' 列（含列名）。

    参数:
        excel_path (Path or str): Excel 文件路径

    返回:
        pd.DataFrame 或 None: 如果 sheet 和列存在返回包含两列的 DataFrame，否则返回 None
    """

    excel_path = Path(excel_path)
    try:
        # 读取 model_info sheet
        model_df = pd.read_excel(excel_path, sheet_name="model_info")

        # 检查列是否存在
        required_cols = ["Model Name", "Model kwargs"]
        if all(col in model_df.columns for col in required_cols):
            return model_df[required_cols]
        else:
            print(f"⚠ Excel 中缺少列: {required_cols}")
            return None

    except Exception as e:
        print(f"⚠ 无法读取 Excel 文件: {e}")
        return None
