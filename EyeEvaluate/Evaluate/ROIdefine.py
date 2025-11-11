def identify_roi(roi_list, x, y):
    """
    roi_list: [{'name': ..., 'description': ...}, ...]
    x, y: 待判断点坐标
    返回值: ROI 名称，如果不属于任何 ROI 返回 None
    """
    for roi in roi_list:
        desc = roi['description']
        name = roi['name']
        try:
            # 使用 eval 判断点是否在 ROI 内
            if eval(desc):
                return name
        except Exception as e:
            print(f"ROI {name} 描述解析失败: {e}")
            continue
    return None


def process_roi_lists(x_list, y_list, roi_list):
    """
    输入两列数据列表，返回 ROI 标记列
    - x 或 y 为空 -> 'blink'
    - 有值 -> 调用 identify_roi 判断
    """
    if len(x_list) != len(y_list):
        raise ValueError("x_list 和 y_list 必须长度相同")

    result = []
    for x, y in zip(x_list, y_list):
        if x is None or y is None:
            result.append('blink')
        else:
            roi_name = identify_roi(roi_list, x, y)
            result.append(roi_name if roi_name is not None else 'None')
    return result