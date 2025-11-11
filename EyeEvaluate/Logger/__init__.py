from .logger_func import *

__all__ = ['setup_logger','init_logger_and_redirect','get_log_file']

"""
------------------------------------------------------------
函数说明（仅用于提醒自己）：

StreamToLogger(logger, level=logging.ERROR):
    将标准输出 stdout 或标准错误 stderr 重定向到指定 logger。
    - logger：logging.Logger 对象
    - level：日志等级，默认 ERROR
    用法：
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

get_log_file(base_dir):
    根据当前时间戳生成日志文件路径。
    - 如果文件已存在且超过 MAX_LOG 大小限制，会自动增加序号。
    返回：
        log_file：日志文件完整路径字符串

setup_logger():
    初始化 logger 并配置日志文件、格式及重定向 stdout/stderr。
    - 文件 handler 保存所有日志
    - stdout/stderr 会重定向到 logger
    - 捕获未处理异常并记录到 logger
    返回：
        logger：logging.Logger 对象

init_logger_and_redirect(message="程序启动"):
    调用 setup_logger 初始化日志系统，并在日志中写入启动信息。
    返回：
        logger：logging.Logger 对象
------------------------------------------------------------
"""

