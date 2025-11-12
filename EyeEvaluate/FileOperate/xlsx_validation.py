import pandas as pd


def check_columns(sheet_index, expected_cols, mode="contain_all"):
    """
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
    """

    def rule_func(df_dict):
        # 索引修正（从1计数 -> Python从0计数）
        idx = sheet_index - 1

        # 检查索引有效性
        if idx < 0 or idx >= len(df_dict):
            return False

        # 获取对应 sheet
        sheet_name = list(df_dict.keys())[idx]
        df = df_dict[sheet_name]

        # 转为集合比较
        actual_cols = set(map(str, df.columns))
        expected = set(map(str, expected_cols))

        if mode == "contain_any":
            return len(actual_cols & expected) > 0
        elif mode == "contain_all":
            return expected.issubset(actual_cols)
        elif mode == "exact":
            return actual_cols == expected
        else:
            raise ValueError(f"未知模式: {mode}")

    return rule_func


def validate_xlsx(file_path, rule_funcs, logic="and"):
    """
    支持多规则检查。
    - rule_funcs: list[function]
    - logic: "and" 或 "or" 决定逻辑组合方式
    """
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return False

    results = []
    for rule_func in rule_funcs:
        try:
            res = rule_func(df)
            if not isinstance(res, bool):
                raise ValueError(f"{rule_func.__name__} 未返回布尔值,在检查{file_path}时")
            results.append(res)
        except Exception as e:
            print(f"规则 {rule_func.__name__} 执行失败: {e}，在检查{file_path}时")
            results.append(False)

    if logic == "and":
        return all(results)
    else:
        return any(results)

