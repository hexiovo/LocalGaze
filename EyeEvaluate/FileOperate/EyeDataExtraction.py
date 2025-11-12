from Popup.tkClass import *
from GlobalData import *
from .xlsx_validation import *

import pandas as pd

def EyeDatachoose(logger):
    while True:
        result_filedialog, path = custom_file_dialog(title="请选择EyeData所在的文件(.xlsx文件)或包含对应文件的文件夹",
                                                     logger=logger)
        if result_filedialog == 1 and path != "":
            try:
                if EyeData_Validation(path):
                    EyeData_Type = 1
                    break
            except Exception as e:
                print(e)
                logger.info("眼动数据输入有误，输入为错误文件")
                custom_messagebox(
                    title="提示",
                    message="所选择文件不符合要求眼动数据文件要求",
                    autowh=False,
                    width=330,
                    height=160)
        elif result_filedialog == 2 and path != "":
            logger.info("ROI输入有误，输入为文件夹")
            custom_messagebox(
                title="提示",
                message="所选择为文件夹",
                autowh=False,
                width=280,
                height=160)
        elif result_filedialog == 0 or path == "":
            logger.info("用户取消输入")
            break
        else:
            logger.info("未知错误")
            break

    return EyeData_Type


def EyeData_Validation(file_path):
    """
        读取 Excel 文件，只提取名称和描述列。

        参数:
            file_path (str): Excel 文件路径

        返回:
            逻辑值，True表示符合要求，False表示不符合要求

        异常处理:
            - 如果不是 .xlsx 文件，抛出 ValueError
            - 如果不符合 要求 ，抛出 ValueError
        """
    # 1. 检查文件扩展名
    if not file_path.lower().endswith(".xlsx"):
        raise ValueError(f"文件必须是 .xlsx 格式: {file_path}")

    # 2. 读取 Excel 文件（保留所有sheet）
    try:
        df_dict = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        raise ValueError(f"读取 Excel 文件失败: {e}")

    EyeData_rule = EyeDataRule()

    # 3. 构建规则：第1个sheet必须exact匹配
    rule = check_columns(
        sheet_index=EyeData_rule.sheet_index,
        expected_cols=EyeData_rule.expected_cols,
        mode=EyeData_rule.mode)

    # 4. 检查规则
    if not rule(df_dict):
        first_sheet_name = list(df_dict.keys())[0]
        actual_cols = list(df_dict[first_sheet_name].columns)
        raise ValueError(f"Excel 列名不符合要求，应为 {EyeData_rule.expected_cols}，当前列为 {actual_cols}")


    return True
