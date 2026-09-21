"""
该部分用于负责为全局变量提供相关信息，包括且不限于屏幕设置，显示设置等
"""

#-----------显示设置--------------
###屏幕分辨率
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440

###提示框相关设置
max_length_per_line = 30    #每行最多的字符数


#-----------文件设置--------------
###日志设置
MAX_LOG = 10  # 单位 MB


###EyeData格式(默认为本包LocalGaze格式)
class EyeDataRule:
    """
    EyeDataRule 配置类，用于定义眼动数据的基本规则
    属性：
        sheet_index (int)       : 眼动数据所在子表格编号，从 1 开始
        expected_cols (list)    : 眼动数据表中应包含的列名列表
        mode (str)              : 列名匹配模式
                                  "contain_any" -> 只需包含任意一个期望列名
                                  "contain_all" -> 必须包含所有期望列名（可多）
                                  "exact"       -> 列名集合必须与期望集合完全一致（顺序无关）
    """
    def __init__(self):
        self.sheet_index = 2  # 眼动数据所在子表格编号
        self.expected_cols = ["时间", "x", "y", "blink", "左瞳孔直径", "右瞳孔直径"]  # 所包含的列名
        self.mode = "exact"  # 列名匹配模式


###InfoRule格式(默认为本包LocalGaze格式)
class InfoRule:
    """
   InfoRule 配置类，用于定义实验相关数据的基本规则
    属性：
        sheet_index (int)       : 实验数据所在子表格编号，从 1 开始
        expected_cols (list)    : 实验数据表中应包含的列名列表
        mode (str)              : 列名匹配模式
                                  "contain_any" -> 只需包含任意一个期望列名
                                  "contain_all" -> 必须包含所有期望列名（可多）
                                  "exact"       -> 列名集合必须与期望集合完全一致（顺序无关）
    """
    def __init__(self):
        self.sheet_index = 1  # 眼动数据所在子表格编号
        self.expected_cols = ["文件名", "开始时间", "结束时间"]  # 所包含的列名
        self.mode = "contain_all"  # 列名匹配模式


#------------模型设置------------
###频率
camera_fps = 30 #实际采样率，单位：Hz