from eyeevaluate_pipeline import *
from Logger import init_logger_and_redirect
import tkinter as tk


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    logger = init_logger_and_redirect("程序启动：EyeEvaluate")

    # ---------- Tkinter 弹窗 ----------
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    logger.info("Tkinter 根窗口初始化完成，弹窗即将显示")

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

            try:
                eyeEvaluate_pipeline(logger)
                logger.info("程序运行完成，顺利退出")
            except Exception as e:
                logger.exception(f"程序运行失败: {e}")

        else:
            logger.info("系统说明弹窗已显示，用户已取消")
    except Exception as e:
        logger.exception(f"弹窗显示失败: {e}")

    logger.info("程序结束")
