import os

def get_xlsxpath(folder_path):
    """
    递归遍历指定文件夹及其所有子文件夹，返回所有 .xlsx 文件的完整路径列表。

    参数：
    - folder_path : str
        要遍历的根目录路径

    返回：
    - xlsx_files : list
        包含所有 .xlsx 文件绝对路径的列表
    """

    xlsx_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.xlsx'):
                full_path = os.path.join(root, file)
                xlsx_files.append(full_path)

    return xlsx_files



