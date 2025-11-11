from Popup.tkClass import *
from FileOperate import *

def eyeEvaluate_pipeline(logger):
    ROI_data = ROIchoose(logger)
    logger.important("ROI识别完成")