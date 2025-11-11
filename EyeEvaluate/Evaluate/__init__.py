from .ROIdefine import *

__all__ = ['identify_roi','process_roi_lists']

"""
-----------------------ROIdefine-----------------------------
identify_roi(roi_list, x, y):
    判断给定点 (x, y) 是否位于任意 ROI 内。
    - roi_list: ROI 列表，每个元素为字典 {'name': ..., 'description': ...}
      description 为可 eval 的条件表达式
    - x, y: 待判断点坐标
    返回：
        ROI 名称（字符串），若不属于任何 ROI 返回 None

process_roi_lists(x_list, y_list, roi_list):
    批量处理坐标列表，生成 ROI 标记列
    - x_list, y_list: 坐标列表，长度必须相同
    - roi_list: ROI 描述列表，同 identify_roi
    返回：
        result: 与输入列表等长的标记列表
            - 'blink' 表示 x 或 y 为空
            - ROI 名称表示命中 ROI
            - 'None' 表示未命中任何 ROI
------------------------------------------------------------
"""
