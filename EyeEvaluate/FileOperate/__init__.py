from .dataextraction import *
from .ROIextract import *

__all__ = ['extract_ROI','ROIchoose']


"""
--------------------------ROIextrct-------------------------
ROIchoose(logger):
    弹出文件选择对话框，选择 ROI Excel 文件并提取 ROI 数据。
    - logger: 日志对象
    逻辑：
        - 选择文件（.xlsx）则调用 extract_ROI 提取数据
        - 选择非 .xlsx 文件或文件夹，弹窗提示错误
        - 取消选择或未知错误，退出循环
    返回：
        ROI_data: 提取的 ROI 列表（List[Dict]），每个字典包含 'name' 和 'description'

extract_ROI(file_path):
    从 Excel 文件中提取 ROI 信息
    - file_path: ROI Excel 文件路径，必须为 .xlsx
    返回：
        List[Dict]: 每个 ROI 的字典列表，每个字典包含:
            - 'name': ROI 名称
            - 'description': ROI 描述（可用于 eval 判断）
    异常处理：
        - 文件不是 .xlsx -> ValueError
        - 列名不符合 ['命名','形状','描述'] -> ValueError
        - Excel 读取失败 -> ValueError
------------------------------------------------------------
"""
