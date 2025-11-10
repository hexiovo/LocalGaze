from roi_drawer import ROIApp
from misc import *


if __name__ == "__main__":
    logger, log_file = setup_logger()
    logger.info("程序启动：ROIDrawing ROI 绘制工具")

    # ---------- Tkinter GUI ----------
    root = tk.Tk()
    show_instructions(root)  # 启动时显示操作提示
    logger.info("操作提示弹窗已显示")

    app = ROIApp(root)
    logger.info("ROIApp GUI 初始化完成")

    root.mainloop()
    logger.info("程序退出")