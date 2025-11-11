from Popup.tkClass import *


import pandas as pd


def ROIchoose(logger):
    while True:
        result_filedialog, path = custom_file_dialog(title="请选择ROI兴趣区所在的文件(.xlsx文件)", logger=logger)
        if result_filedialog == 1 and path != "":
            try:
                ROI_data = extract_ROI(path)
                break
            except Exception as e:
                print(e)
                logger.info("ROI输入有误，输入为错误文件")
                custom_messagebox(
                    title="提示",
                    message="所选择文件不符合要求ROI文件要求",
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

    return ROI_data


def extract_ROI(file_path):
    """
    读取 ROI Excel 文件，只提取名称和描述列

    参数:
        file_path (str): Excel 文件路径

    返回:
        List[Dict]: 每个 ROI 的字典列表，每个字典包含 'name' 和 'description'

    异常处理:
        - 如果不是 .xlsx 文件，抛出 ValueError
        - 如果列名不符合 ['命名','形状','描述']，抛出 ValueError
    """
    # 1. 检查文件扩展名
    if not file_path.lower().endswith(".xlsx"):
        raise ValueError(f"文件必须是 .xlsx 格式: {file_path}")

    # 2. 读取 Excel 文件
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"读取 Excel 文件失败: {e}")

    # 3. 检查列名
    expected_columns = ['命名', '形状', '描述']
    if list(df.columns) != expected_columns:
        raise ValueError(f"Excel 列名不符合要求，应为 {expected_columns}, 当前列为 {list(df.columns)}")

    # 4. 提取命名和描述列
    roi_list = []
    for _, row in df.iterrows():
        name = str(row['命名']).strip()
        description = str(row['描述']).strip()  # 保留原字符串
        roi_list.append({
            "name": name,
            "description": description
        })

    return roi_list