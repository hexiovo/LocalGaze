import tkinter as tk
from tkinter import font as tkfont
from .askopenpathname import askopenpathname
from GlobalData import *



import os
from .misc_utils import *


def custom_messagebox(title, message, font_size=12, autowh = True, width=420, height=280,font_name="微软雅黑",):
    """
    自定义消息弹窗，可控制字体大小与窗口尺寸。
    - 点击“确定”：返回 True，继续执行
    - 点击“取消”或关闭窗口：打印“用户退出”，返回 False
    """
    win = tk.Toplevel()
    win.title(title)
    win.attributes("-topmost", True)
    win.resizable(False, False)

    # --- 字体设置 ---
    text_font = tkfont.Font(family=font_name, size=font_size)
    btn_font = tkfont.Font(family=font_name, size=font_size)

    if autowh:
        # 将 message 按行分割
        lines = message.split("\n")
        num_lines = len(lines)
        max_line_len = max((len(line) for line in lines if line.strip()), default=1)

        title_lines =(len(title) + 19) // 20

        lines = [title[i:i + max_length_per_line] for i in range(0, len(title), max_length_per_line)]
        max_title_length = max(len(line) for line in lines)

        max_len = max(max_title_length, max_line_len)
        total_lines = num_lines + title_lines

        # ========== 动态计算宽高 ==========
        # 估算每个字符的平均宽度（单位像素）
        avg_char_width = text_font.measure("一")  # 中文基准
        # 基础宽度取最长行的文字长度 + 两边空白
        width = int(avg_char_width * max_len * 0.9) + 10
        # 基础高度取行数、字体高度、边距综合
        line_height = text_font.metrics("linespace")
        height = int(line_height * total_lines * 1.6) + 20  # 额外留空间给按钮

    # 居中显示
    win.update_idletasks()
    x = (SCREEN_WIDTH - width) // 4
    y = (SCREEN_HEIGHT - height) // 4
    win.geometry(f"{width}x{height}+{x}+{y}")
    win.configure(bg="white")

    print(f"提示框『{title}』的尺寸为：高度 {height}，宽度 {width}")

    title_font = tkfont.Font(family=font_name, size=font_size + 2, weight="bold")
    title_label = tk.Label(
        win,
        text=wrap_message(title),
        font=title_font,
        bg="white",
        fg="black",
        anchor="center",
    )
    title_label.pack(pady=(15, 5))  # 上下留一点空间

    # === 文本内容 ===
    label = tk.Label(
        win,
        text=wrap_message(message),
        justify="left",
        font=text_font,
        bg="white",
        anchor="nw",
        wraplength=width - 40  # 自动换行
    )
    label.pack(fill="both", expand=True, padx=20, pady=10)

    # ======= 状态控制变量 =======
    result = {"confirmed": None}

    # 点击确定
    def on_confirm():
        result["confirmed"] = True
        win.destroy()

    # 点击取消或关闭窗口
    def on_cancel():
        result["confirmed"] = False
        print("用户退出文件确认")
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_cancel)  # 关闭按钮时的处理

    # ======= 按钮区 =======
    btn_frame = tk.Frame(win, bg="white")
    btn_frame.pack(pady=(0, 15))

    btn_ok = tk.Button(
        btn_frame, text="确定",
        font=btn_font,
        width=8,
        command=on_confirm
    )
    btn_ok.pack(side="left", padx=10)

    btn_cancel = tk.Button(
        btn_frame, text="取消",
        font=btn_font,
        width=8,
        command=on_cancel
    )
    btn_cancel.pack(side="left", padx=10)

    # 阻塞主窗口，等待用户选择
    win.grab_set()
    win.wait_window(win)

    return result["confirmed"]


def custom_file_dialog(title="选择文件/文件夹",  font_size=12, font_name="微软雅黑",
                       width=420, height=160,logger = None):
    """
    自定义文件/文件夹选择对话框。
    - select_type: "file" -> 文件, "folder" -> 文件夹
    - 返回: (datatype, path)
        datatype: 1 文件，2 文件夹，0 取消/未选
        path:  用户选择的路径，取消则为空字符串
    说明：
      - 用户可以先点击“浏览”选择路径，路径会显示在只读 Entry 中；
      - 用户也可以直接点击“确定”，若 Entry 中已有路径则直接使用该路径；
      - 若 Entry 为空则“确定”会触发文件/文件夹选择对话框。
    """
    win = tk.Toplevel()
    win.title(title)
    win.attributes("-topmost", False)  # 不固定最上层
    win.lift()  # 将窗口抬到当前应用的最上层
    win.resizable(False, False)

    # 字体设置
    text_font = tkfont.Font(family=font_name, size=font_size)
    btn_font = tkfont.Font(family=font_name, size=font_size)

    # 居中显示（保留你原来的分母 4，如果想居中于屏幕请改为 //2）
    win.update_idletasks()
    x = (SCREEN_WIDTH - width) // 4
    y = (SCREEN_HEIGHT - height) // 4
    win.geometry(f"{width}x{height}+{x}+{y}")
    win.configure(bg="white")

    # 标题
    title_font = tkfont.Font(family=font_name, size=font_size + 2, weight="bold")
    title_label = tk.Label(
        win,
        text=title,
        font=title_font,
        bg="white",
        fg="black",
        anchor="center"
    )
    title_label.pack(pady=(20, 10))

    # 状态控制变量（外部可读）
    result = {"datatype": 0, "path": ""}  # 0=取消/未选, 1=file, 2=folder

    # Entry 显示变量
    entry_var = tk.StringVar()
    entry = tk.Entry(win, textvariable=entry_var, font=text_font, width=40, state="readonly")
    entry.pack(pady=(5, 10), padx=20)

    def open_dialog():
        """
        弹出一个对话框，可选择文件或文件夹，选择后更新 entry 与 result。
        支持连续多次选择。
        """
        # 释放当前窗口 grab
        win.grab_release()

        try:
            # 调用你自定义的 askopenpathname（支持文件或文件夹）
            try:
                path,type = askopenpathname(title="请选择文件或文件夹",logger = logger)
                print(f"[被试选择] 路径: {path} | 类型: {type}")
            except Exception as e:
                logger.exception(f"选择程序运行失败: {e}")

            if path:
                # 判断类型
                if os.path.isdir(path):
                    datatype = 2  # 文件夹
                elif os.path.isfile(path):
                    datatype = 1  # 文件
                else:
                    datatype = 0
                    path = ""
            else:
                datatype = 0

            # 更新 Entry 和 result
            if path:
                entry_var.set(path)
                result["path"] = path
                result["datatype"] = datatype
                type_str = "文件" if datatype == 1 else "文件夹"
                print(f"[被试选择] 路径: {path} | 类型: {type_str}（{datatype}）")
            else:
                entry_var.set("")
                result["path"] = ""
                result["datatype"] = 0
                print(f"[被试选择] 未选择任何路径 | 类别: 0（无）")
                custom_messagebox(
                    "提示",
                    "未选择,请先选择文件或文件夹！",
                    autowh=False,
                    width=330,
                    height=160
                )

        except Exception as e:
            print(f"[错误] 打开文件选择器时出错: {e}")

    # ----- 确认逻辑（供“确定”按钮使用） -----
    def on_confirm():
        """
        确认时优先使用 Entry 中已显示的路径；若为空则直接赋空，不弹窗。
        选择到路径则关闭窗口并写入 result。
        """

        current = entry_var.get().strip()
        if current:
            if os.path.isfile(current):
                result["datatype"] = 1
            elif os.path.isdir(current):
                result["datatype"] = 2
            else:
                result["datatype"] = 0
            result["path"] = current
            print(f"[被试确认] 路径: {current} | 类别: {result['datatype']}（1=文件, 2=文件夹）")
            win.destroy()
        else:
            # 路径无效或为空
            result["path"] = ""
            result["datatype"] = 0
            print(f"[被试确认] 路径无效或未选择 | 类别: 0（无）")
            custom_messagebox(
                "提示",
                "路径无效，请选择有效的文件或文件夹！",
                autowh=False,
                width=330,
                height=160
            )

    # ----- 取消逻辑 -----
    def on_cancel():
        result["datatype"] = 0
        result["path"] = ""
        print("用户退出文件选择")
        win.destroy()

    # 按钮区
    btn_frame = tk.Frame(win, bg="white")
    btn_frame.pack(pady=(0, 20))

    btn_open = tk.Button(btn_frame, text="浏览", font=btn_font, width=8, command=open_dialog)
    btn_open.pack(side="left", padx=5)

    btn_ok = tk.Button(btn_frame, text="确定", font=btn_font, width=8, command=on_confirm)
    btn_ok.pack(side="left", padx=5)

    btn_cancel = tk.Button(btn_frame, text="取消", font=btn_font, width=8, command=on_cancel)
    btn_cancel.pack(side="left", padx=5)

    # 关闭时行为与取消一致
    win.protocol("WM_DELETE_WINDOW", on_cancel)

    # 阻塞主窗口
    win.grab_set()
    win.wait_window(win)

    return result["datatype"], result["path"]


