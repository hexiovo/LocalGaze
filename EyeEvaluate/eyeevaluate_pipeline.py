from Popup.tkClass import *
from FileOperate import *

def eyeEvaluate_pipeline(logger):
    #ROI_data = ROIchoose(logger)
    #logger.important("ROI识别完成")
    EyeData_Type = EyeDatachoose(logger)
    logger.important("EyeData识别完成")