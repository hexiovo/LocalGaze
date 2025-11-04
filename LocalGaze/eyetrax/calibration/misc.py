import pandas as pd
from pathlib import Path
import cv2
import numpy as np


def save_calibration_results(model_path: Path, 
                             errors_per_point_9, 
                             errors_per_point_add, 
                             mean_error_9, 
                             mean_errors_add,
                             model_names=None,
                             model_kwargs=None):
    """
    保存校准误差到 Excel 文件，路径与模型文件相同，只是扩展名为 .xlsx。
    包含三个 sheet：
        1️⃣ NinePointCalibration
        2️⃣ AdditionalCalibration
        3️⃣ model_info （包含 model_names 与 model_kwargs）
    第一列均为从 1 开始的序号。
    """

    # Excel 文件路径
    excel_path = model_path.with_suffix(".xlsx")

    # -------- 九点校准 --------
    nine_point_df = pd.DataFrame({
        "Index": range(1, len(errors_per_point_9) + 1),
        "Point": [f"Point{i+1}" for i in range(len(errors_per_point_9))],
        "Error": errors_per_point_9
    })
    nine_point_df["MeanError"] = mean_error_9  # 每行都加上均值方便查看

    # -------- 随机补充校准 --------
    additional_df = pd.DataFrame({
        "Index": range(1, len(errors_per_point_add) + 1),
        "Error": errors_per_point_add
    })
    additional_df["MeanErrorPerRound"] = None  # 预留列

    # -------- 填充每轮平均误差 --------
    start_idx = 0
    if mean_errors_add and isinstance(mean_errors_add[0], list):
        # 兼容旧格式（list of list）
        points_per_round = len(mean_errors_add[0])
    else:
        points_per_round = 5  # 默认一轮 5 点

    for round_idx, mean_err in enumerate(mean_errors_add, start=1):
        end_idx = start_idx + points_per_round
        additional_df.loc[start_idx:end_idx-1, "MeanErrorPerRound"] = mean_err
        start_idx = end_idx

    # -------- model_info --------
    if model_names is None:
        model_names = []
    if model_kwargs is None:
        model_kwargs = []

    model_info = {
        "Model Name": model_names,
        "Model kwargs": model_kwargs
    }

    model_info_df = pd.DataFrame(model_info)

    # -------- 写入 Excel --------
    with pd.ExcelWriter(excel_path) as writer:
        nine_point_df.to_excel(writer, sheet_name="NinePointCalibration", index=False)
        additional_df.to_excel(writer, sheet_name="AdditionalCalibration", index=False)
        model_info_df.to_excel(writer, sheet_name="model_info", index=False)

    print(f"✅ 校准结果已保存至：{excel_path}")




def draw_gaze_cloud(frame, x, y, radius=30, color_gray=(180, 180, 180), color_center=(0, 0, 255), border_color=(0, 0, 0), center_radius=4, alpha=0.4):
    """
    在图像上绘制注视点可视化标记：
    - 灰色半透明实心圆（表示注视范围）
    - 红色实心小圆（表示注视中心）
    - 黑色外圈边界线（增强视觉对比）

    Args:
        frame: np.array, 摄像头帧
        x, y: 注视点中心坐标
        radius: 外圈半径（像素）
        color_gray: 灰色圆的颜色 (BGR)
        color_center: 中心红点颜色 (BGR)
        border_color: 外圈边界线颜色 (BGR)
        center_radius: 中心红点半径（像素）
        alpha: 灰色圆的透明度 (0~1)
    """
    overlay = frame.copy()

    # 绘制灰色实心圆（半透明）
    cv2.circle(overlay, (int(x), int(y)), radius, color_gray, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 绘制黑色外边框（细线）
    cv2.circle(frame, (int(x), int(y)), radius, border_color, 1)

    # 绘制中心红色小圆
    cv2.circle(frame, (int(x), int(y)), center_radius, color_center, -1)
