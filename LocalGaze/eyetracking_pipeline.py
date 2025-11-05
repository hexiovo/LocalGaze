from eyetrax.gaze import GazeEstimator
from eyetrax.calibration import run_9_point_calibration,run_additional_random_calibration,save_calibration_results
import tkinter.font as tkFont
from tkinter import filedialog
from misc import *
from eyetracking import *
from Global_data import *
import numpy as np


# ---------- 封装工作流程 ----------
def run_eye_tracking_workflow():

    root = tk.Tk()
    root.withdraw()

    root.attributes("-topmost", True)  # 确保弹窗在最前

    # ---------- 设置全局字体 ----------
    default_font = tkFont.nametofont("TkDefaultFont")
    default_font.configure(family="Arial", size=12)  # 设置字体为 Arial，大小 20

    text_font = tkFont.nametofont("TkTextFont")
    text_font.configure(family="Arial", size=14)

    model_type = select_model()
    model_kwargs = MODEL_KWARGS.get(model_type, {})
    print(model_kwargs)

    estimator = GazeEstimator(model_name = model_type,
                            model_kwargs = model_kwargs)  # 假设你已经有这个类


    use_existing = messagebox.askyesno("模型选择", "是否已有历史模型？\nYes: 选择已有模型\nNo: 创建新模型")

    if use_existing:
        model_path = filedialog.askopenfilename(
            title="选择已有模型 (.pkl)",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=str(Path.cwd() / "Model_history")
        )
        if not model_path:
            print("未选择模型，程序终止")
            return
        estimator.load_model(model_path)
        print(f"已加载历史模型: {model_path}")
    else:
        model_name = simpledialog.askstring("新模型", "请输入新模型文件名（不含扩展名）")
        if not model_name:
            print("未输入模型名，程序终止")
            return
        model_dir = Path("Model_history")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_name}.pkl"

        # ---- 九点校准 ----
        errors_per_point_9, mean_error_9 = run_9_point_calibration(estimator)
        error_text = f"{mean_error_9:.2f}" if mean_error_9 is not None else "未计算"
        messagebox.showinfo("九点校准完成", f"九点校准完成，平均误差: {error_text}")

        # ---- 进一步校准循环 ----
        all_errors_per_point_add = []
        all_mean_error_add = []

        if messagebox.askyesno("进一步校准", "是否愿意进入进一步校准？"):
            continue_calibration = True
            while continue_calibration:
                # 弹出参数输入框
                dialog = CalibrationDialog(root, title="进一步校准参数设置")
                if dialog.canceled:
                    print("用户取消了输入，程序退出")
                    return  # 或设置默认值继续
                error_threshold = dialog.error_threshold
                max_rounds = dialog.max_rounds
                points_per_round = dialog.points_per_round
                print(f"用户设置 - 阈值: {error_threshold}, 最大轮数: {max_rounds}, 每轮点数: {points_per_round}")

                # 执行随机补充校准
                errors_per_point_add, mean_error_add = run_additional_random_calibration(
                    estimator,
                    points_per_round=points_per_round,
                    error_threshold=error_threshold,
                    max_rounds=max_rounds
                )

                # 累计保存每轮结果
                all_errors_per_point_add.extend(errors_per_point_add)
                if isinstance(mean_error_add, list):
                    all_mean_error_add.extend(mean_error_add)
                elif isinstance(mean_error_add, (int, float)):
                    all_mean_error_add.append(mean_error_add)

                # 获取最低误差
                if all_mean_error_add:
                    min_error = min(all_mean_error_add)
                    error_text_add = f"{min_error:.2f}"
                else:
                    error_text_add = "未计算"

                messagebox.showinfo("随机补充校准完成", f"随机补充校准完成，最低误差: {error_text_add}")

                # 判断是否满足阈值
                if error_threshold is not None and isinstance(min_error, (int, float)):
                    if min_error < error_threshold:
                        print(f"最低误差 {min_error:.2f} 小于阈值 {error_threshold:.2f}，校准结束。")
                        continue_calibration = False
                        break  # 满足阈值，退出循环

                # 阈值未达到或未设置，询问是否继续
                continue_calibration = messagebox.askyesno(
                    "继续补充校准",
                    "最低误差未达到目标阈值，是否再次进行补充校准？"
                )
        else:
            messagebox.showinfo("跳过校准", "用户选择不进行进一步校准。")
            errors_per_point_add, mean_error_add =[], []
            all_errors_per_point_add.extend(errors_per_point_add)
            all_mean_error_add.extend(mean_error_add)

        # ---- 保存模型与结果 ----
        estimator.save_model(model_path)

        if errors_per_point_9 is None:
            errors_per_point_9 = []


        safe_save_calibration_results(
            model_path=model_path,
            errors_per_point_9 = errors_per_point_9,
            errors_per_point_add=all_errors_per_point_add,
            mean_error_9=mean_error_9,
            mean_errors_add=all_mean_error_add,
            model_names=model_type,
            model_kwargs=model_kwargs
        )

    start_experiment = messagebox.askyesno("开始实验", "是否进入正式实验？")

    if start_experiment:

        data_name = simpledialog.askstring("数据名称", "请输入数据名称（不含扩展名）")
        if not data_name:
            print("未输入模型名，程序终止")
            return

        # 实验开始时间
        start_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

        print("开始时间为:",start_time_str)

        draw_cloud = messagebox.askyesno("预测点显示", "是否开启预测点显示功能？")

        eye_data = run_realtime_gaze(estimator, draw_cloud)

        # 实验结束时间
        end_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        print("结束时间为:",end_time_str)

        # ---------- 保存实验数据 ----------
        # 确定最终平均误差
        if use_existing:
            # 历史模型，尝试从对应文件读取
            # 假设 save_calibration_results 或模型保存路径有记录
            # 这里简单示意读取 pickle 文件里的 mean_error
            try:
                Calibrationdata_path = model_path.replace(".pkl", ".xlsx")
                final_mean_error = get_min_calibration_mean(Calibrationdata_path)
                model_type , model_kwargs = get_model_info(Calibrationdata_path)
                if final_mean_error is None:
                    final_mean_error = "未知"
                all_errors_per_point_add =  get_additional_calibration_mean(Calibrationdata_path)

            except Exception:
                final_mean_error = "未知"

        else:
            # 新模型
            if all_mean_error_add:
                final_mean_error = all_mean_error_add[-1]  # 最后一次补充校准的平均误差
            elif mean_error_9 is not None:
                final_mean_error = mean_error_9
            else:
                final_mean_error = "未计算"

        if 'all_errors_per_point_add' not in locals():
            all_errors_per_point_add = []

        # 实验信息表
        experiment_info = {
            "使用已有模型": [use_existing],
            "模型名称": [Path(model_path).name if use_existing else model_name],
            "是否进行额外矫正": [len(all_errors_per_point_add) > 0],
            "矫正模型路径": [str(model_path) if len(all_errors_per_point_add) > 0 else ""],
            "开始时间": [start_time_str],
            "结束时间": [end_time_str],
            "最终平均误差": [final_mean_error],
            "计算模型":[model_type],
            "计算模型参数":[model_kwargs]
        }
        df_info = pd.DataFrame(experiment_info)

        # eye_data 表
        df_eye = pd.DataFrame(eye_data, columns=["时间", "x", "y","blink","左瞳孔直径","右瞳孔直径"])

        # 保存路径
        main_path = Path.cwd()
        save_dir = main_path / "Data"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{data_name}.xlsx"

        with pd.ExcelWriter(save_path) as writer:
            df_info.to_excel(writer, sheet_name="实验信息", index=False)
            df_eye.to_excel(writer, sheet_name="眼动数据", index=False)

        print(f"实验数据已保存到: {save_path}")

    else:
        print("已取消实验")

    root.destroy()

# ---------- 安全调用保存函数 ----------
def safe_save_calibration_results(model_path,
                                  errors_per_point_9,
                                  errors_per_point_add,
                                  mean_error_9,
                                  mean_errors_add,
                                  model_names=None,
                                  model_kwargs=None):
    """
    通用保存封装：
    兼容直接传入 run_9_point_calibration / run_additional_random_calibration 的输出。
    可额外传入 model_names 和 model_kwargs，用于保存 model_info sheet。
    """


    # --- 1️⃣ 九点校准结果 ---
    if isinstance(errors_per_point_9, list):
        # 若是 [(index, error), ...] 则提取误差值
        if len(errors_per_point_9) > 0 and isinstance(errors_per_point_9[0], (list, tuple)):
            errors_per_point_9_values = [e[1] for e in errors_per_point_9]
        else:
            errors_per_point_9_values = errors_per_point_9
    else:
        errors_per_point_9_values = []

    # --- 2️⃣ 随机补充校准结果 ---
    if isinstance(errors_per_point_add, list):
        # 若是多轮结构 [[...], [...]]
        if len(errors_per_point_add) > 0 and isinstance(errors_per_point_add[0], list):
            errors_per_point_add_values = [err for round_errs in errors_per_point_add for err in round_errs]
        else:
            errors_per_point_add_values = errors_per_point_add
    else:
        errors_per_point_add_values = []

    # --- 3️⃣ 计算或验证均值 ---
    mean_error_9_value = (
        float(mean_error_9)
        if mean_error_9 is not None and not isinstance(mean_error_9, (list, tuple))
        else (np.nanmean(mean_error_9) if isinstance(mean_error_9, (list, np.ndarray)) else np.nan)
    )

    mean_errors_add_value = mean_errors_add if isinstance(mean_errors_add, list) else [mean_errors_add]
    model_kwargs_list = [model_kwargs]

    # --- 4️⃣ 调用原保存函数 ---
    save_calibration_results(
        model_path=model_path,
        errors_per_point_9=errors_per_point_9_values,
        errors_per_point_add=errors_per_point_add_values,
        mean_error_9=mean_error_9_value,
        mean_errors_add=mean_errors_add_value,
        model_names=model_names,
        model_kwargs=model_kwargs_list
    )


