"""Standalone Windows entry point."""
import ctypes
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox


def configure_windows():
    if sys.platform != 'win32':
        return
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('LocalGaze.ROIDrawing.1')
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


def main():
    configure_windows()
    from misc import setup_logger
    logger, log_file = setup_logger()
    root = tk.Tk()
    root.withdraw()
    icon = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent)) / 'roi.ico'
    if icon.exists():
        root.iconbitmap(default=str(icon))

    def report_error(kind, value, tb):
        logger.error('界面操作异常', exc_info=(kind, value, tb))
        messagebox.showerror('操作失败', f'{value}\n\n日志：{log_file}', parent=root)

    root.report_callback_exception = report_error
    try:
        from roi_drawer import ROIApp
        app = ROIApp(root)
        root.deiconify()
        logger.info('ROI 绘制工具启动')
        if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
            from smoke_test import run
            root.after(250, lambda: run(root, app, Path(sys.argv[2])))
        root.mainloop()
    except Exception:
        logger.exception('程序启动失败')
        messagebox.showerror('启动失败', f'请查看日志：\n{log_file}', parent=root)
        raise
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == '__main__':
    main()
