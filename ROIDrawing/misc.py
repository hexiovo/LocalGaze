"""Writable per-user logging for windowed PyInstaller applications."""
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import tempfile
from tkinter import messagebox


def show_instructions(root):
    messagebox.showinfo('操作提示', '选择形状后，按住鼠标右键绘制。\n左键拖动黄色控制点调整。\nEnter 确认，Esc 返回。', parent=root)


def setup_logger():
    base = Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'ROIDrawing' / 'logs'
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        base = Path(tempfile.gettempdir()) / 'ROIDrawing-logs'
        base.mkdir(parents=True, exist_ok=True)
    log_file = base / 'ROIDrawing.log'
    logger = logging.getLogger('ROIDrawing')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
        handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(handler)
    return logger, str(log_file)
