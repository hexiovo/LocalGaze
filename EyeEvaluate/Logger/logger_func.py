import os
import logging
import datetime
import sys


from GlobalData import *


class StreamToLogger:
    """
    将 stdout/stderr 输出重定向到 logger
    """
    def __init__(self, logger, level=logging.ERROR):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        self.linebuf += buf
        while '\n' in self.linebuf:
            line, self.linebuf = self.linebuf.split('\n', 1)
            if line.strip():
                self.logger.log(self.level, line.strip())

    def flush(self):
        if self.linebuf.strip():
            self.logger.log(self.level, self.linebuf.strip())
            self.linebuf = ''
        for handler in self.logger.handlers:
            handler.flush()

def get_log_file(base_dir):
    """
    根据时间戳生成日志文件，如果已存在且超过大小限制，则增加序号
    """
    log_dir = os.path.join(base_dir, "EyeEvaluate_log")
    os.makedirs(log_dir, exist_ok=True)

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{start_time}.txt")

    if os.path.exists(log_file) and os.path.getsize(log_file) / (1024*1024) > MAX_LOG:
        # 检查已有序号文件，取最大序号 +1
        i = 2
        while os.path.exists(os.path.join(log_dir, f"{start_time}-{i}.txt")):
            i += 1
        log_file = os.path.join(log_dir, f"{start_time}-{i}.txt")

    return log_file

def setup_logger():
    """
    初始化 logger，并将 stdout/stderr 重定向到日志文件
    返回 logger 对象
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = get_log_file(base_dir)

    # 创建 logger
    logger = logging.getLogger("EyeEvaluate_log")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 避免重复打印

    # 文件 handler 保存所有日志
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # 防止重复添加 handler
    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    # 重定向 stdout 和 stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # 捕获未处理异常
    def excepthook(type_, value, tb):
        import traceback
        logger.error("未捕获异常:\n" + "".join(traceback.format_exception(type_, value, tb)))
    sys.excepthook = excepthook

    # ---- 新增 logger.important 功能 ----
    def important(msg,width = 100):
        # 直接写到文件 handler，而不走 logger 格式化

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        middle_line = f"[{timestamp}] {msg}"
        # 如果需要居中，可以用 str.center 对齐到 width
        middle_line = middle_line.center(width, " ")

        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                with open(handler.baseFilename, 'a', encoding='utf-8') as f:
                    f.write("*" * width + "\n")  # 上一行
                    f.write(middle_line + "\n")        # 中间信息
                    f.write("*" * width + "\n")  # 下一行

    # 动态绑定方法到 logger 对象
    logger.important = important

    return logger

def init_logger_and_redirect(message="程序启动"):
    """
    初始化日志系统并输出启动信息，只返回 logger 对象
    """
    logger = setup_logger()
    logger.info(message)
    return logger