from Popup.tkClass import *
from GlobalData import *
from .xlsx_validation import *


import pandas as pd


def Infochoose(logger):
    while True:
        result_filedialog, path = custom_file_dialog(title="请选择实验Info文件(.xlsx文件)", logger=logger)
        if result_filedialog == 1 and path != "":
            try:
                info_Data = extract_Info(path)
                break
            except Exception as e:
                print(e)
                logger.info("Info输入有误，输入为错误文件")
                custom_messagebox(
                    title="提示",
                    message="所选择文件不符合要求Info文件要求",
                    autowh=False,
                    width=330,
                    height=160)
        elif result_filedialog == 2 and path != "":
            logger.info("Info输入有误，输入为文件夹")
            custom_messagebox(
                title="提示",
                message="所选择为文件夹",
                autowh=False,
                width=280,
                height=160)
        elif result_filedialog == 0 or path == "":
            logger.info("用户取消输入")
            info_Data = []
            break
        else:
            logger.info("未知错误")
            info_Data = []
            break

    return info_Data


def extract_Info(file_path):
    """
    读取 Info Excel 文件，只提取名称和描述列。

    参数:
        file_path (str): Excel 文件路径

    返回:
        Info信息

    异常处理:
        - 如果不是 .xlsx 文件，抛出 ValueError
        - 如果列名不符合 规则，抛出 ValueError
    """
    # 1. 检查文件扩展名
    if not file_path.lower().endswith(".xlsx"):
        raise ValueError(f"文件必须是 .xlsx 格式: {file_path}")

    # 2. 读取 Excel 文件（保留所有sheet）
    try:
        df_dict = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        raise ValueError(f"读取 Excel 文件失败: {e}")

    inforule = InfoRule()

    # 3. 构建规则：第1个sheet必须匹配
    rule = check_columns(
        sheet_index=inforule.sheet_index,
        expected_cols=inforule.expected_cols,
        mode=inforule.mode
    )

    # 4. 检查规则
    if not rule(df_dict):
        first_sheet_name = list(df_dict.keys())[0]
        actual_cols = list(df_dict[first_sheet_name].columns)
        raise ValueError(
            f"Excel 列名不符合要求，应为 {inforule.expected_cols}，当前列为 {actual_cols}"
        )

    # 5. 返回 DataFrame（规则指定的 sheet）
    target_sheet_name = list(df_dict.keys())[inforule.sheet_index]
    
    return df_dict[target_sheet_name]