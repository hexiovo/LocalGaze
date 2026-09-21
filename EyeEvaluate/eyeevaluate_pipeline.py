from Popup.tkClass import *
from FileOperate import *
import tkinter as tk


def eyeEvaluate_pipeline(logger):

    #读取ROI
    ROI_data = ROIchoose(logger)
    if ROI_data:
        logger.important("ROI识别完成")
    else:
        logger.important("ROI识别失败")
        raise SystemExit("程序终止：ROI识别失败，请检查输入文件。")

    #读取眼动数据
    EyeData_Type,EyeData_Path,path = EyeDatachoose(logger)
    if EyeData_Type == 1 :
        logger.important("EyeData文件识别完成")
    elif EyeData_Type == 2:
        logger.important("EyeData文件夹识别完成")
    else:
        logger.important("EyeData文件识别失败")
        raise SystemExit("程序终止：眼动数据识别失败，请检查输入文件。")


    #处理眼动数据





def run_EyeData_Evaluate(logger):
    """
    启动EyeEvaluate系统时显示流程说明弹窗，并调用主pipeline。
    该函数对异常进行了完整捕获与日志记录。
    """
    # ---------- 初始化 Tkinter 根窗口 ----------
    try:
        root = tk.Tk()
        root.withdraw()                # 隐藏主窗口
        root.attributes("-topmost", True)  # 保持窗口置顶
        logger.info("Tkinter 根窗口初始化完成，弹窗即将显示")
    except Exception as e:
        logger.exception(f"Tkinter 根窗口初始化失败: {e}")
        return

    # ---------- 显示系统流程说明 ----------
    try:
        result_messagebox = custom_messagebox(
            title="EyeEvaluate 眼动指标计算系统",
            message=(
                "流程说明：\n"
                "1. 选择ROI兴趣区文件\n"
                "2. 选择对应数据/包含数据的文件夹\n"
                "3. 选择功能\n"
                "4. 选择输出位置并计算\n"
            ),
            font_size=12,
            autowh=True
        )

        if result_messagebox:
            logger.info("系统说明弹窗已显示，用户已确认")

            # ---------- 执行主流程 ----------
            try:
                eyeEvaluate_pipeline(logger)
                logger.info("程序运行完成，顺利退出")

            except Exception as e:
                logger.exception(f"程序运行失败: {e}")
                custom_messagebox(
                    title="程序运行错误",
                    message=str(e),
                    font_size=12,
                    autowh=True
                )

        else:
            logger.info("系统说明弹窗已显示，用户已取消")

    except Exception as e:
        logger.exception(f"弹窗显示失败: {e}")
        custom_messagebox(
            title="错误",
            message=f"弹窗显示失败：{e}",
            font_size=12,
            autowh=True
        )