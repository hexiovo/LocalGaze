import tkinter as tk
from tkinter import ttk, messagebox
import os
import string
from pypinyin import lazy_pinyin

# -------------------- 工具函数 --------------------
def get_drives():
    """获取系统所有盘符"""
    drives = []
    for d in string.ascii_uppercase:
        drive = f"{d}:/"
        if os.path.exists(drive):
            drives.append(drive)
    return drives

def sort_entries(entries, parent_path):
    """文件夹在前，文件在后，按拼音排序"""
    dirs = []
    files = []
    for name in entries:
        abspath = os.path.join(parent_path, name)
        if os.path.isdir(abspath):
            dirs.append(name)
        else:
            files.append(name)
    dirs.sort(key=lambda x: lazy_pinyin(x))
    files.sort(key=lambda x: lazy_pinyin(x))
    return dirs + files

def askopenpathname(title="请选择文件或文件夹", width=400, height=400, logger=None):
    selected_path = ""
    selected_type = 0  # 0=无, 1=文件, 2=文件夹

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # ---------- 初始化 Tkinter ----------
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    log("[初始化] Tkinter 主窗口隐藏完成")

    top = tk.Toplevel(root)
    top.title(title)
    top.geometry(f"{width}x{height}")
    top.grab_set()
    log(f"[窗口创建] 标题: {title}, 尺寸: {width}x{height}")

    path_var = tk.StringVar()

    # ---------- Treeview ----------
    tree = ttk.Treeview(top)
    tree["columns"] = ("fullpath",)
    tree.column("fullpath", width=0, stretch=False)
    tree.heading("fullpath", text="Full Path")
    tree.pack(fill="both", expand=True, side="top")

    lbl = tk.Label(top, textvariable=path_var, anchor="w")
    lbl.pack(fill="x")

    def populate_tree(parent, path):
        try:
            entries = os.listdir(path)
            entries = [e for e in entries if not e.startswith(("$", "."))]
            entries = sort_entries(entries, path)  # 你原来的排序逻辑
            log(f"[填充目录] {path} 下 {len(entries)} 个条目")
            for name in entries:
                abspath = os.path.join(path, name)
                isdir = os.path.isdir(abspath)
                node = tree.insert(parent, "end", text=name, values=[abspath])
                if isdir:
                    tree.insert(node, "end", text="")  # 占位
        except PermissionError:
            log(f"[权限错误] 无法访问目录: {path}")

    def open_node(event):
        node = tree.focus()
        children = tree.get_children(node)
        if len(children) == 1 and tree.item(children[0], "text") == "":
            tree.delete(children[0])
            path = tree.set(node, "fullpath")
            log(f"[展开节点] {path}")
            populate_tree(node, path)

    def on_select(event):
        item = tree.focus()
        path = tree.set(item, "fullpath")
        if os.path.exists(path):
            path_var.set(path)
            log(f"[选中路径] {path}")
        else:
            path_var.set("")

    tree.bind("<<TreeviewOpen>>", open_node)
    tree.bind("<<TreeviewSelect>>", on_select)

    # ---------- 顶层节点 ----------
    user = os.path.expanduser("~")
    desktop = os.path.join(user, "Desktop")
    documents = os.path.join(user, "Documents")

    parent_computer = tree.insert("", "end", text="此电脑", values=[""])
    drives = get_drives()  # 你原来的获取驱动器函数
    for d in drives:
        drive_node = tree.insert(parent_computer, "end", text=d, values=[d])
        populate_tree(drive_node, d)

    if os.path.exists(desktop):
        parent_desktop = tree.insert("", "end", text="桌面", values=[desktop])
        populate_tree(parent_desktop, desktop)

    if os.path.exists(documents):
        parent_docs = tree.insert("", "end", text="文档", values=[documents])
        populate_tree(parent_docs, documents)

    # ---------- 按钮 ----------
    btn_frame = tk.Frame(top)
    btn_frame.pack(fill="x", side="bottom")

    def on_confirm():
        nonlocal selected_path, selected_type
        path = path_var.get()
        if path and os.path.exists(path):
            selected_path = path
            selected_type = 2 if os.path.isdir(path) else 1
            type_str = "文件夹" if selected_type == 2 else "文件"
            log(f"[选择完成] 路径: {selected_path} | 类型: {type_str}")
            top.destroy()  # 只销毁 top
            log("[选择结束] 用户确认选择")
        else:
            messagebox.showwarning("提示", "请选择有效的文件或文件夹！")
            log("[警告] 用户未选择有效路径点击确认")

    def on_cancel():
        nonlocal selected_path, selected_type
        selected_path = ""
        selected_type = 0
        log("[取消] 用户取消选择")
        top.destroy()  # 只销毁 top

    confirm_btn = tk.Button(btn_frame, text="确认", command=on_confirm)
    confirm_btn.pack(side="left", expand=True, fill="x", padx=5, pady=5)
    cancel_btn = tk.Button(btn_frame, text="取消", command=on_cancel)
    cancel_btn.pack(side="right", expand=True, fill="x", padx=5, pady=5)

    # ---------- 等待窗口关闭 ----------
    root.wait_window(top)  # 阻塞直到 top 被销毁

    log("[函数结束] 正在返回选择结果")
    return selected_path, selected_type


