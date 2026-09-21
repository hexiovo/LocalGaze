from .dataextraction import *
from .ROIextract import *
from .xlsx_validation import *
from .EyeDataExtraction import *
from .datasave import *


__all__ = ['extract_ROI','ROIchoose','EyeDatachoose','check_columns','validate_xlsx',
           'EyeData_Validation','EyeDatachoose']


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

"""
--------------------------Infoextrct-------------------------
Infochoose(logger):
    弹出文件选择对话框，选择 ROI Excel 文件并提取 ROI 数据。
    - logger: 日志对象
    逻辑：
        - 选择文件（.xlsx）则调用 extract_ROI 提取数据
        - 选择非 .xlsx 文件或文件夹，弹窗提示错误
        - 取消选择或未知错误，退出循环
    返回：
        path : Info信息
        
extract_Info(file_path):
    从 Excel 文件中提取 Info 信息
    - file_path: ROI Excel 文件路径，必须为 .xlsx
    异常处理：
        - 文件不是 .xlsx -> ValueError
        - 列名不符合 条件 -> ValueError
        - Excel 读取失败 -> ValueError
------------------------------------------------------------
"""


"""
--------------------------xlsx_validation----------------------
check_columns(sheet_index, expected_cols, mode="contain_all"):
    通用规则函数：检查 Excel 第 n 个子表的列名是否符合要求。
    
    参数：
    - sheet_index: int，从 1 开始计数的子表序号
    - expected_cols: list[str]，期望列名列表
    - mode: str，匹配模式，可选：
        "contain_any"  -> 只需包含任意一个期望列名
        "contain_all"  -> 必须包含所有期望列名（可多）
        "exact"        -> 列名集合必须与期望集合完全一致（顺序无关）
    
    返回：
    - rule_func(df_dict): 函数，用于 validate_xlsx 调用
    
    
validate_xlsx(file_path, rule_funcs, logic="and"):
    对指定的 Excel 文件执行多重规则验证，并根据逻辑方式返回整体检查结果。

    功能：
        读取指定 Excel 文件并依次执行多个用户自定义规则函数（rule_func），
        每个规则函数接收 DataFrame 并返回布尔值。函数会按给定逻辑 ("and"/"or")
        组合所有结果，用于判断该文件是否符合预期格式或内容要求。
    
    参数：
        - file_path (str): Excel 文件路径，需为 .xlsx 格式。
        - rule_funcs (list[function]): 待执行的规则函数列表。
            每个函数形式应为：
                def rule_func(df: pd.DataFrame) -> bool
            若函数未返回布尔值或执行出错，将记为 False。
        - logic (str, 默认 "and"): 规则组合逻辑。
            "and" → 所有规则通过才返回 True；
            "or"  → 任意规则通过即返回 True。
    
    返回：
        bool：True 表示文件通过验证；False 表示不符合规则或读取失败。

"""

"""
-------------------------dataextraction------------------
get_xlsxpath(folder_path):
    递归遍历指定文件夹及其所有子文件夹，返回所有 .xlsx 文件的完整路径列表。

    参数：
    - folder_path : str
        要遍历的根目录路径

    返回：
    - xlsx_files : list
        包含所有 .xlsx 文件绝对路径的列表


"""


"""
-------------------------EyeDataExtraction-----------------
def EyeData_Validation_folder(file_list):
    接收一个包含 .xlsx 文件路径的列表，逐个调用 EyeData_Validation 进行验证。

    参数:
        file_list (list): 包含 Excel 文件完整路径的列表

    返回:
        valid_files (list): 验证通过的文件路径列表
def EyeData_Validation(file_path):
    读取 Excel 文件，只提取名称和描述列。

    参数:
        file_path (str): Excel 文件路径

    返回:
        逻辑值，True表示符合要求，False表示不符合要求

    异常处理:
        - 如果不是 .xlsx 文件，抛出 ValueError
        - 如果不符合 要求 ，抛出 ValueError

def EyeDatachoose(logger):
    选择并验证眼动数据文件或文件夹。
    
    该函数通过自定义文件选择对话框让用户选择一个单个的 .xlsx 文件或包含多个此类文件的文件夹。
    根据选择内容，自动调用相应的验证函数（单文件使用 EyeData_Validation，多文件使用 EyeData_Validation_folder）。
    若验证通过，返回对应的文件类型标识、有效路径及原始路径；若用户取消或文件不符合要求，则记录日志并给出提示信息。
    
    返回:
        tuple:
            EyeData_Type (int): 输入类型标识（1 = 单文件，2 = 文件夹，0 = 用户取消或错误）
            EyeData_Path (list 或 str): 验证通过的文件路径（文件夹模式下为列表）
            path (str): 用户选择的原始路径


"""


"""
-----------------datasave---------------------
def get_savepath(src_file, src_root, dst_root, func_name=None):

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