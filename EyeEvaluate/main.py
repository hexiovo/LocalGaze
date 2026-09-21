from eyeevaluate_pipeline import *
from Logger import init_logger_and_redirect



# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    logger = init_logger_and_redirect("程序启动：EyeEvaluate")

    try:
        run_EyeData_Evaluate(logger)
        logger.info("程序顺利结束")
    except Exception as e:
        print(e)
        logger.info("程序报错结束")

