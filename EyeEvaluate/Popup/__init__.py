from .tkClass import *
from .askopenpathname import *
from .misc_utils import *


__all__ = ['get_drives', 'sort_entries', 'askopenpathname','wrap_message',
           'custom_messagebox','custom_file_dialog']
"""
----------------------askopenpathname----------------------------
get_drives()：
    获取当前系统中所有存在的盘符（如 C:/、D:/ 等），返回列表。

sort_entries(entries, parent_path)：
    对指定目录下的文件和文件夹进行排序：
    - 文件夹在前、文件在后；
    - 均按中文名的拼音顺序排序。

askopenpathname(title="请选择文件或文件夹", width=400, height=400, logger=None)：
    打开一个自定义的 Tkinter 文件/文件夹浏览窗口。
    用户可在树形结构中展开磁盘、桌面、文档等目录选择目标。
    返回 (path, datatype)：
        path：用户选择的路径；
        datatype：0=未选择，1=文件，2=文件夹。
------------------------------------------------------------
"""

"""
------------------misc_utils------------------------------
wrap_message(message)：
    将输入的文本 message 按指定的最大行宽 max_length_per_line 进行自动换行。
    处理逻辑：
        - 保留原有的换行符；
        - 对每一行按照 max_length_per_line 分割；
        - 返回换行处理后的新字符串；
    用途：
        在弹窗或日志显示时，避免文本过长导致显示超出窗口范围。
------------------------------------------------------------
"""

"""
------------------tkClass------------------------------
custom_messagebox(title, message, font_size=12, autowh=True, width=420, height=280, font_name="微软雅黑")：
    自定义消息弹窗。
    输入：
        title: 弹窗标题
        message: 弹窗内容文本
        font_size: 文本字体大小
        autowh: 是否自动根据文本内容调整窗口宽高
        width, height: 初始宽高
        font_name: 字体名称
    作用：
        显示一个带“确定”和“取消”的消息框。
        点击“确定”返回 True，继续执行；
        点击“取消”或关闭窗口打印“用户退出”，返回 False。
    返回：
        bool: True=点击确定, False=点击取消或关闭窗口

custom_file_dialog(title="选择文件/文件夹", font_size=12, font_name="微软雅黑", width=420, height=160, logger=None)：
    自定义文件/文件夹选择对话框。
    输入：
        title: 窗口标题
        font_size, font_name: 字体相关设置
        width, height: 窗口初始宽高
        logger: 可选日志对象，用于记录操作
    作用：
        弹出一个窗口，用户可点击“浏览”选择文件或文件夹，也可直接点击“确定”使用 Entry 中路径。
        支持路径验证、连续多次选择。
    返回：
        tuple(datatype, path)
            datatype: 0=取消/未选, 1=文件, 2=文件夹
            path: 用户选择的路径字符串，取消则为空
------------------------------------------------------------
"""