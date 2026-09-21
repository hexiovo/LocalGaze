import os


def get_savepath(src_file, src_root, dst_root, func_name=None):
    """
    根据源文件路径与根路径计算相对路径，并在目标路径下生成对应结构。
    可在目标路径下增加一个功能性子文件夹。

    参数：
    ----------
    src_file : str
        源文件的绝对路径，例如 "F:/程序/test/1.xlsx"
    src_root : str
        源文件的根路径，例如 "F:/程序"
    dst_root : str
        目标根路径，例如 "F:/test"
    func_name : str or None, default=None
        可选参数，用于在dst_root下增加功能性文件夹，
        例如 func_name="123" → "F:/test/123/..."

    返回：
    ----------
    dst_file : str
        目标文件应保存的完整路径（不复制文件）
    """

    # 计算相对路径（src_file 相对于 src_root）
    rel_path = os.path.relpath(src_file, src_root)

    # 如果提供了功能性子目录，则嵌入其中
    if func_name:
        dst_root = os.path.join(dst_root, str(func_name))

    # 组合出目标完整路径
    dst_file = os.path.join(dst_root, rel_path)

    # 确保路径合法（创建中间目录结构）
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    return dst_file