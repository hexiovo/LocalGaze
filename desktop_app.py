"""Lightweight Chinese desktop entry point for LocalGaze.

The module builds the Tk view only. Capture, preprocessing, ROI drawing and
analysis code are started in an owned worker after an explicit button click.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from session_names import canonical_session_id


_COLORS = {
    "canvas": "#F4F6F5",
    "surface": "#FFFFFF",
    "ink": "#183333",
    "muted": "#617373",
    "line": "#D6E2DE",
    "teal": "#087F76",
    "teal_dark": "#06665F",
    "teal_pale": "#E5F3F0",
    "orange": "#E88345",
    "danger": "#B63C3C",
}

_TIME_UNITS = {
    "自动识别": None,
    "秒 (seconds)": "seconds",
    "毫秒 (milliseconds)": "milliseconds",
    "微秒 (microseconds)": "microseconds",
}
_PREPROCESS_TIME_UNITS = ("seconds", "milliseconds", "microseconds")
_CALIBRATION_ESTIMATORS = {
    "Ridge（默认）": "ridge",
    "Elastic Net": "elastic_net",
    "Linear SVR": "linear_svr",
    "SVR": "svr",
    "Tiny MLP": "tiny_mlp",
}
_CALIBRATION_FORMULAS = {
    "v2（默认，紧凑眼部与头部特征）": "eyetrax-compact-eye-head-v2",
    "v1（旧版地标特征）": "eyetrax-landmark-features-v1",
}


class DesktopApp:
    """Responsive Tk shell whose only asynchronous work is an owned process."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        manager_factory: Callable[[], object] | None = None,
        auto_poll: bool = True,
    ) -> None:
        self.root = root
        self._manager_factory = manager_factory or self._default_manager_factory
        self._manager = None
        self._active_job_id: str | None = None
        self._active_job_kind: str | None = None
        self._closing = False
        self._auto_poll = auto_poll
        self._poll_after_id = None
        self._input_controls: list[tk.Widget] = []
        self._action_controls: list[tk.Widget] = []
        self._primary_actions: dict[str, tk.Widget] = {}
        self._browse_controls: list[tk.Widget] = []
        self._busy = False
        self._last_success_outputs: dict[str, Path] = {}
        self._last_success_job_id: str | None = None

        self.session_id = tk.StringVar(root, "")
        self.model_path = tk.StringVar(root, "")
        self.correction_path = tk.StringVar(root, "")
        self.use_correction_for_capture = tk.BooleanVar(root, False)
        self.target_x_px = tk.StringVar(root, "")
        self.target_y_px = tk.StringVar(root, "")
        self.calibration_estimator_label = tk.StringVar(root, "Ridge（默认）")
        self.calibration_formula_label = tk.StringVar(root, "v2（默认，紧凑眼部与头部特征）")
        self.preprocess_source_path = tk.StringVar(root, "")
        self.analysis_source_path = tk.StringVar(root, "")
        self.output_root = tk.StringVar(root, "")
        self.roi_path = tk.StringVar(root, "")
        self.preprocess_time_unit = tk.StringVar(root, "seconds")
        self.analysis_time_unit_label = tk.StringVar(root, "自动识别")
        self.status_text = tk.StringVar(root, "就绪 · 选择一个操作开始")

        self._configure_root()
        self._configure_styles()
        self._build_view()
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        if self._auto_poll:
            self._schedule_poll()

    def _configure_root(self) -> None:
        self.root.title("LocalGaze · 眼动实验工作台")
        self.root.configure(background=_COLORS["canvas"])
        self.root.minsize(780, 570)
        self.root.geometry("1080x760")
        try:
            self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        except tk.TclError:
            pass

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=_COLORS["canvas"])
        style.configure("Card.TFrame", background=_COLORS["surface"], relief="flat")
        style.configure("TLabel", background=_COLORS["canvas"], foreground=_COLORS["ink"])
        style.configure("Card.TLabel", background=_COLORS["surface"], foreground=_COLORS["ink"])
        style.configure("Muted.TLabel", background=_COLORS["surface"], foreground=_COLORS["muted"])
        style.configure("Header.TLabel", background=_COLORS["canvas"], foreground=_COLORS["ink"], font=("Microsoft YaHei UI", 19, "bold"))
        style.configure("Eyebrow.TLabel", background=_COLORS["canvas"], foreground=_COLORS["teal"], font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Status.TLabel", background=_COLORS["teal_pale"], foreground=_COLORS["teal_dark"], padding=(11, 6))
        style.configure("Section.TLabel", background=_COLORS["surface"], foreground=_COLORS["ink"], font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("TNotebook", background=_COLORS["canvas"], borderwidth=0)
        style.configure("TNotebook.Tab", background="#E7EEEB", foreground=_COLORS["muted"], padding=(15, 9), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", _COLORS["surface"])], foreground=[("selected", _COLORS["teal_dark"])])
        style.configure("TEntry", padding=(7, 6), fieldbackground="#FBFCFC")
        style.configure("TCombobox", padding=(7, 5), fieldbackground="#FBFCFC")
        style.configure("Primary.TButton", background=_COLORS["teal"], foreground="#FFFFFF", padding=(14, 9), borderwidth=0, font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", _COLORS["teal_dark"]), ("disabled", "#A9C4BF")], foreground=[("disabled", "#F4F7F6")])
        style.configure("Secondary.TButton", background=_COLORS["teal_pale"], foreground=_COLORS["teal_dark"], padding=(12, 8), borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#D4EBE6"), ("disabled", "#EDF3F1")])
        style.configure("Quiet.TButton", background=_COLORS["surface"], foreground=_COLORS["muted"], padding=(10, 7), borderwidth=1)
        style.configure("Danger.TButton", background="#FBEDEC", foreground=_COLORS["danger"], padding=(11, 7), borderwidth=0)

    def _build_view(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0, minsize=42)

        header = ttk.Frame(self.root, padding=(24, 19, 24, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="LOCALGAZE  /  DESKTOP", style="Eyebrow.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="眼动实验工作台", style="Header.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.status_label = ttk.Label(header, textvariable=self.status_text, style="Status.TLabel")
        self.status_label.grid(row=0, column=1, sticky="e", padx=(14, 0))
        self.guide_button = ttk.Button(
            header, text="查看整体指南", style="Secondary.TButton", command=self._show_guide
        )
        self.guide_button.grid(row=1, column=1, sticky="e", padx=(14, 0), pady=(5, 0))

        main = ttk.Frame(self.root, padding=(18, 0, 18, 0))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self._build_session_tab()
        self._build_roi_tab()
        self._build_preprocess_tab()
        self._build_analysis_tab()
        self._build_guide_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._update_tab_state, add="+")

        self._build_log_panel()

    def _new_tab(self, title: str) -> ttk.Frame:
        page = ttk.Frame(self.notebook)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        canvas = tk.Canvas(page, background=_COLORS["canvas"], borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content = ttk.Frame(canvas, padding=(22, 14, 22, 18))
        content.columnconfigure(0, weight=1)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event, target=canvas: target.configure(scrollregion=target.bbox("all")))
        canvas.bind("<Configure>", lambda event, target=canvas, item=window: target.itemconfigure(item, width=event.width))
        self.notebook.add(page, text=title)
        return content

    def _card(self, parent: ttk.Frame, row: int, title: str, subtitle: str | None = None) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(18, 16, 18, 17))
        card.grid(row=row, column=0, sticky="ew", pady=(0, 12))
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text=title, style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        if subtitle:
            subtitle_label = ttk.Label(card, text=subtitle, style="Muted.TLabel", justify="left")
            subtitle_label.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 12))
            self._wrap_to_width(subtitle_label, card, inset=36)
        return card

    def _wrap_to_width(self, label: ttk.Label, container: ttk.Widget, *, inset: int = 0, minimum: int = 160) -> None:
        def resize(event: tk.Event | None = None) -> None:
            width = container.winfo_width() if event is None else event.width
            if width > 1:
                label.configure(wraplength=max(minimum, width - inset))

        container.bind("<Configure>", resize, add="+")
        self.root.after_idle(resize)

    def _advanced_section(self, parent: ttk.Frame, row: int, title: str) -> ttk.Frame:
        button = ttk.Button(parent, text=f"高级设置 ▸ {title}", style="Quiet.TButton")
        button.grid(row=row, column=0, columnspan=4, sticky="w", pady=(10, 0))
        panel = ttk.Frame(parent, style="Card.TFrame", padding=(0, 7, 0, 0))
        panel.columnconfigure(1, weight=1)
        panel.grid(row=row + 1, column=0, columnspan=4, sticky="ew", pady=(2, 0))
        panel.grid_remove()
        button.configure(command=lambda: self._toggle_advanced(panel, button, title))
        return panel

    @staticmethod
    def _toggle_advanced(panel: ttk.Frame, button: ttk.Button, title: str) -> None:
        if panel.winfo_ismapped():
            panel.grid_remove()
            button.configure(text=f"高级设置 ▸ {title}")
        else:
            panel.grid()
            button.configure(text=f"高级设置 ▾ {title}")

    def _build_session_tab(self) -> None:
        page = self._new_tab("会话与采集")
        page.columnconfigure(0, weight=1)
        page.rowconfigure(2, weight=1)
        card = self._card(page, 0, "本次会话", "为校准和采集设置会话、模型与输出位置；预处理和分析的数据文件请在各自页面选择。")
        self._entry_row(card, 2, "会话 ID", self.session_id, browse=None, hint="例如：P001_session1")
        self._entry_row(card, 3, "输出位置", self.output_root, browse="directory")
        self._entry_row(card, 4, "模型路径", self.model_path, browse="model", hint="采集使用 .pkl；预处理使用模型工作簿或文件夹")
        advanced = self._advanced_section(card, 5, "校准、目标与修正参数")
        ttk.Label(advanced, text="校准估计器", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=(8, 0))
        estimator_box = ttk.Combobox(
            advanced,
            textvariable=self.calibration_estimator_label,
            values=tuple(_CALIBRATION_ESTIMATORS),
            state="readonly",
            width=25,
        )
        estimator_box.grid(row=0, column=1, sticky="w", padx=(12, 8), pady=(8, 0))
        self._input_controls.append(estimator_box)
        ttk.Label(advanced, text="model_name 表示估计器类型。", style="Muted.TLabel").grid(row=0, column=2, columnspan=2, sticky="w", padx=(10, 0), pady=(8, 0))
        ttk.Label(advanced, text="校准特征公式", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        formula_box = ttk.Combobox(
            advanced,
            textvariable=self.calibration_formula_label,
            values=tuple(_CALIBRATION_FORMULAS),
            state="readonly",
            width=34,
        )
        formula_box.grid(row=1, column=1, sticky="w", padx=(12, 8), pady=(8, 0))
        self._input_controls.append(formula_box)
        ttk.Label(advanced, text="采集会读取模型中记录的公式。", style="Muted.TLabel").grid(row=1, column=2, columnspan=2, sticky="w", padx=(10, 0), pady=(8, 0))
        ttk.Label(advanced, text="目标位置（屏幕像素）", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 0))
        target_x = ttk.Entry(advanced, textvariable=self.target_x_px, width=12)
        target_x.grid(row=2, column=1, sticky="w", padx=(12, 8), pady=(8, 0))
        ttk.Label(advanced, text="X", style="Card.TLabel").grid(row=2, column=2, sticky="e", pady=(8, 0))
        target_y = ttk.Entry(advanced, textvariable=self.target_y_px, width=12)
        target_y.grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(8, 0))
        self._input_controls.extend((target_x, target_y))
        ttk.Label(advanced, text="留空时使用屏幕中心。", style="Muted.TLabel").grid(row=3, column=1, columnspan=3, sticky="w", padx=(12, 0), pady=(2, 0))
        self._entry_row(advanced, 4, "修正文件", self.correction_path, browse="correction", hint="已知目标成功后自动填入；采集默认不使用")
        use_correction = ttk.Checkbutton(
            advanced,
            text="在正式采集中显式应用此修正文件",
            variable=self.use_correction_for_capture,
        )
        use_correction.grid(row=5, column=1, columnspan=3, sticky="w", padx=(12, 0), pady=(8, 0))
        self._input_controls.append(use_correction)
        actions = self._card(page, 1, "采集操作", "只有点击下方按钮后才会启动相机采集流程。运行期间可以取消任务。")
        action_row = ttk.Frame(actions, style="Card.TFrame")
        action_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        action_row.columnconfigure(0, weight=1, uniform="capture-actions")
        action_row.columnconfigure(1, weight=1, uniform="capture-actions")
        action_row.columnconfigure(2, weight=1, uniform="capture-actions")
        calibrate = ttk.Button(action_row, text="校准并保存模型", style="Primary.TButton", command=self.start_calibration)
        calibrate.grid(row=0, column=0, sticky="ew", padx=(0, 7))
        capture = ttk.Button(action_row, text="使用已有模型采集", style="Primary.TButton", command=self.start_capture)
        capture.grid(row=0, column=1, sticky="ew", padx=(7, 0))
        known_target = ttk.Button(action_row, text="已知目标修正", style="Secondary.TButton", command=self.start_known_target_offset)
        known_target.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self._action_controls.extend((calibrate, capture, known_target))
        self._primary_actions.update(calibrate=calibrate, capture=capture, known_target_offset=known_target)
        footer = ttk.Label(page, text="校准和正式采集沿用旧有流程；此工作台负责统一设置会话与输出位置。", style="Muted.TLabel")
        footer.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self._wrap_to_width(footer, page, inset=8)

    def _build_roi_tab(self) -> None:
        page = self._new_tab("ROI 绘制")
        page.columnconfigure(0, weight=1)
        card = self._card(page, 0, "绘制感兴趣区域", "ROI 工具会在独立窗口中叠加到选中的显示器；保存的工作簿可在分析页选择。")
        ttk.Label(card, text="背景图片或视频", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 0))
        roi_background_hint = ttk.Label(card, text="请先在目标显示器打开并显示要标注的图片或视频，再启动绘制窗口。ROI 会叠加在当前屏幕上绘制；背景素材不作为眼动数据来源。", style="Muted.TLabel", justify="left")
        roi_background_hint.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self._wrap_to_width(roi_background_hint, card, inset=36)
        self._entry_row(card, 4, "输出位置", self.output_root, browse="directory", hint="ROI 工作簿保存在此目录下的 ROIdata 文件夹")
        button = ttk.Button(card, text="打开 ROI 绘制工具", style="Primary.TButton", command=self.start_roi)
        button.grid(row=5, column=0, sticky="w", pady=(15, 0))
        self._action_controls.append(button)
        self._primary_actions["roi"] = button

    def _build_preprocess_tab(self) -> None:
        page = self._new_tab("预处理与 QC")
        page.columnconfigure(0, weight=1)
        card = self._card(page, 0, "预处理数据", "按文件名匹配模型与数据，运行分析预设的预处理和 QC；原始文件不会被覆盖。")
        self._entry_row(card, 2, "模型路径", self.model_path, browse="model", hint="按文件名匹配模型工作簿或文件夹")
        self._entry_row(card, 3, "眼动数据文件/文件夹", self.preprocess_source_path, browse="source", hint="可选单个文件或含多个文件的文件夹")
        self._entry_row(card, 4, "输出位置", self.output_root, browse="directory")
        button = ttk.Button(card, text="运行预处理与 QC", style="Primary.TButton", command=self.start_preprocess)
        button.grid(row=5, column=0, sticky="w", pady=(13, 0))
        self._action_controls.append(button)
        self._primary_actions["preprocess"] = button
        advanced = self._advanced_section(card, 6, "时间单位")
        ttk.Label(advanced, text="数值时间单位（新格式 elapsed_s 为秒）", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=(3, 0))
        self.preprocess_time_box = ttk.Combobox(advanced, textvariable=self.preprocess_time_unit, values=_PREPROCESS_TIME_UNITS, state="readonly", width=24)
        self.preprocess_time_box.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(3, 0))
        self._input_controls.append(self.preprocess_time_box)
        ttk.Label(advanced, text="旧格式纯数值时间请按来源单位选择。", style="Muted.TLabel").grid(row=0, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(3, 0))
        footer = ttk.Label(page, text="输出包含逐文件结果、批次报告及运行清单。", style="Muted.TLabel")
        footer.grid(row=1, column=0, sticky="ew")
        self._wrap_to_width(footer, page, inset=8)

    def _build_analysis_tab(self) -> None:
        page = self._new_tab("分析与导出")
        page.columnconfigure(0, weight=1)
        card = self._card(page, 0, "分析单个数据文件", "读取一个眼动数据文件，计算摘要、样本标记、事件和可用的 ROI 命中，并导出结果包。")
        self._entry_row(card, 2, "眼动数据文件", self.analysis_source_path, browse="analysis_source", hint="每次分析一个文件")
        self._entry_row(card, 3, "输出位置", self.output_root, browse="directory")
        self._entry_row(card, 4, "ROI 工作簿（可选）", self.roi_path, browse="roi", hint="留空时不计算 ROI 命中")
        button = ttk.Button(card, text="分析并导出结果", style="Primary.TButton", command=self.start_analysis)
        button.grid(row=5, column=0, sticky="w", pady=(8, 0))
        self._action_controls.append(button)
        self._primary_actions["analyze"] = button
        advanced = self._advanced_section(card, 6, "时间单位")
        ttk.Label(advanced, text="数值时间单位", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=(11, 0))
        self.analysis_time_box = ttk.Combobox(advanced, textvariable=self.analysis_time_unit_label, values=tuple(_TIME_UNITS), state="readonly", width=28)
        self.analysis_time_box.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(11, 0))
        self._input_controls.append(self.analysis_time_box)
        ttk.Label(advanced, text="自动识别会读取文件元数据；有歧义的数值时间请明确选择。", style="Muted.TLabel", wraplength=400).grid(row=0, column=2, columnspan=2, sticky="w", padx=(12, 0), pady=(11, 0))
        footer = ttk.Label(page, text="成功的预处理结果会预填为本页的数据文件；仍需检查后手动启动分析。", style="Muted.TLabel")
        footer.grid(row=1, column=0, sticky="ew")
        self._wrap_to_width(footer, page, inset=8)

    def _build_guide_tab(self) -> None:
        page = ttk.Frame(self.notebook, padding=(18, 8, 18, 8))
        self.notebook.add(page, text="使用指南")
        self._guide_tab_id = str(page)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        card = ttk.Frame(page, style="Card.TFrame", padding=(18, 10, 18, 10))
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(2, weight=1)
        ttk.Label(card, text="LocalGaze 整体使用指南", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        subtitle = ttk.Label(card, text="先看快速流程，再按步骤操作。", style="Muted.TLabel", justify="left")
        subtitle.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self._wrap_to_width(subtitle, card, inset=36)
        filename = "LocalGaze使用指导语及流程.txt"
        guide_path = Path(__file__).resolve().parent / filename
        if getattr(sys, "frozen", False):
            adjacent_guide = Path(sys.executable).resolve().parent / filename
            if adjacent_guide.is_file():
                guide_path = adjacent_guide
        self.guide_path = guide_path
        self.guide_status = tk.StringVar(self.root, "正在读取正式指导语…")
        self.guide_text = ScrolledText(
            card,
            height=10,
            wrap="word",
            background=_COLORS["surface"],
            foreground=_COLORS["ink"],
            insertbackground=_COLORS["ink"],
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=_COLORS["line"],
            padx=10,
            pady=8,
        )
        self.guide_text.grid(row=2, column=0, sticky="nsew", pady=(2, 10))
        self._load_guide_text()
        toolbar = ttk.Frame(card, style="Card.TFrame")
        toolbar.grid(row=3, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)
        status = ttk.Label(toolbar, textvariable=self.guide_status, style="Muted.TLabel", justify="left")
        status.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self._wrap_to_width(status, toolbar, inset=150)
        open_button = ttk.Button(
            toolbar,
            text="打开 TXT",
            style="Secondary.TButton",
            command=lambda: self._open_guide(guide_path),
        )
        open_button.grid(row=0, column=1, sticky="e")

    def _show_guide(self) -> None:
        self.notebook.select(self._guide_tab_id)
        self._load_guide_text()
        self.guide_text.yview_moveto(0)
        self.guide_text.focus_set()

    def _load_guide_text(self) -> None:
        try:
            text = self.guide_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            text = f"无法读取完整指导语。\n\n文件：{self.guide_path}\n\n原因：{exc}"
            self.guide_status.set("正式指导语未能加载；请检查来源文件路径和 UTF-8 编码。")
        else:
            self.guide_status.set("正式指导语已加载。")
        self.guide_text.configure(state="normal")
        self.guide_text.delete("1.0", "end")
        self.guide_text.insert("1.0", text)
        self.guide_text.configure(state="disabled")

    def _build_log_panel(self) -> None:
        panel = ttk.Frame(self.root, padding=(18, 4, 18, 7))
        panel.grid(row=2, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        head = ttk.Frame(panel)
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)
        self._log_expanded = False
        self.log_toggle = ttk.Button(head, text="展开任务记录 ▸", style="Quiet.TButton", command=self._toggle_log)
        self.log_toggle.grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(head, text="取消当前任务", style="Danger.TButton", command=self.cancel_job, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="e", padx=(0, 8))
        self.outputs_menu = tk.Menu(self.root, tearoff=0)
        self.open_outputs_button = ttk.Menubutton(head, text="打开本次输出 ▾", menu=self.outputs_menu, state="disabled")
        self.open_outputs_button.grid(row=0, column=2, sticky="e", padx=(0, 8))
        self.clear_log_button = ttk.Button(head, text="清空", style="Quiet.TButton", command=self.clear_log)
        self.clear_log_button.grid(row=0, column=3, sticky="e")
        self.clear_log_button.grid_remove()
        self.log = ScrolledText(panel, height=4, wrap="word", background="#FFFFFF", foreground=_COLORS["ink"], insertbackground=_COLORS["ink"], relief="flat", borderwidth=1, highlightthickness=1, highlightbackground=_COLORS["line"], padx=10, pady=8, font=("Consolas", 9))
        self.log.grid(row=1, column=0, sticky="ew")
        self.log.grid_remove()
        self.log.configure(state="disabled")
        self._append_log("就绪。任务仅会在你点击对应操作后启动。")

    def _toggle_log(self) -> None:
        if self._log_expanded:
            self.log.grid_remove()
            self.clear_log_button.grid_remove()
            self.log_toggle.configure(text="展开任务记录 ▸")
            self._log_expanded = False
        else:
            self.log.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            self.clear_log_button.grid()
            self.log_toggle.configure(text="收起任务记录 ▾")
            self._log_expanded = True

    def _entry_row(self, card: ttk.Frame, row: int, label: str, variable: tk.StringVar, *, browse: str | None, hint: str | None = None) -> tuple[ttk.Entry, ttk.Button | None]:
        field = ttk.Frame(card, style="Card.TFrame")
        field.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        field.columnconfigure(0, weight=1)
        ttk.Label(field, text=label, style="Card.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        entry = ttk.Entry(field, textvariable=variable, width=24)
        entry.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._input_controls.append(entry)
        if browse:
            button = ttk.Button(field, text="浏览…", style="Quiet.TButton", command=lambda b=browse, v=variable, l=label: self._browse(b, v, l))
            button.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
            self._browse_controls.append(button)
        else:
            button = None
        if hint:
            hint_label = ttk.Label(field, text=hint, style="Muted.TLabel", justify="left")
            hint_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(3, 0))
            self._wrap_to_width(hint_label, field, minimum=160)
        return entry, button

    def _browse(self, kind: str, variable: tk.StringVar, label: str) -> None:
        current = variable.get().strip()
        initial = current if Path(current).exists() else str(Path(current).parent) if current else str(Path.cwd())
        selected = ""
        if kind == "directory":
            selected = filedialog.askdirectory(parent=self.root, title=f"选择{label}", initialdir=initial, mustexist=False)
        elif kind == "source":
            choice = messagebox.askyesnocancel("选择数据来源", "选择“是”浏览单个文件；选择“否”浏览文件夹。", parent=self.root)
            if choice is True:
                selected = filedialog.askopenfilename(parent=self.root, title="选择数据文件", initialdir=initial, filetypes=(("眼动数据", "*.xlsx *.xlsm *.csv *.tsv *.txt"), ("所有文件", "*.*")))
            elif choice is False:
                selected = filedialog.askdirectory(parent=self.root, title="选择数据文件夹", initialdir=initial, mustexist=True)
        elif kind == "analysis_source":
            selected = filedialog.askopenfilename(parent=self.root, title="选择眼动数据文件", initialdir=initial, filetypes=(("眼动数据", "*.xlsx *.xlsm *.csv *.tsv *.txt"), ("所有文件", "*.*")))
        elif kind == "model":
            choice = messagebox.askyesnocancel("选择模型路径", "选择“是”浏览模型文件；选择“否”浏览模型文件夹。", parent=self.root)
            if choice is True:
                selected = filedialog.askopenfilename(parent=self.root, title="选择模型文件", initialdir=initial, filetypes=(("模型文件", "*.pkl *.xlsx *.xlsm *.csv *.tsv *.txt"), ("所有文件", "*.*")))
            elif choice is False:
                selected = filedialog.askdirectory(parent=self.root, title="选择模型文件夹", initialdir=initial, mustexist=True)
        elif kind == "correction":
            selected = filedialog.askopenfilename(parent=self.root, title="选择修正 JSON", initialdir=initial, filetypes=(("修正 JSON", "*.json"), ("所有文件", "*.*")))
        elif kind == "roi":
            selected = filedialog.askopenfilename(parent=self.root, title="选择 ROI 工作簿", initialdir=initial, filetypes=(("Excel 工作簿", "*.xlsx *.xlsm"), ("所有文件", "*.*")))
        if selected:
            variable.set(selected)

    def _payload(self, kind: str) -> dict[str, object]:
        session = canonical_session_id(self.session_id.get())
        output_text = self.output_root.get().strip()
        if not output_text:
            raise ValueError("请先设置输出位置。")
        output = Path(output_text).expanduser()
        if output.exists() and not output.is_dir():
            raise ValueError("输出位置必须是文件夹。")
        payload: dict[str, object] = {"output_root": str(output.resolve()), "session_id": session}

        if kind == "calibrate":
            estimator_label = self.calibration_estimator_label.get()
            formula_label = self.calibration_formula_label.get()
            try:
                payload["model_name"] = _CALIBRATION_ESTIMATORS[estimator_label]
                payload["feature_formula_version"] = _CALIBRATION_FORMULAS[formula_label]
            except KeyError as exc:
                raise ValueError("请选择有效的校准估计器和特征公式。") from exc
        if kind == "capture" and self.use_correction_for_capture.get():
            correction_text = self.correction_path.get().strip()
            if not correction_text:
                raise ValueError("已勾选修正文件；请先选择一个已知目标修正 JSON。")
            correction = Path(correction_text).expanduser()
            if not correction.is_file() or correction.suffix.lower() != ".json":
                raise FileNotFoundError(f"修正文件不存在或不是 JSON：{correction}")
            payload["correction_path"] = str(correction.resolve())
        if kind == "known_target_offset":
            x_text, y_text = self.target_x_px.get().strip(), self.target_y_px.get().strip()
            if bool(x_text) != bool(y_text):
                raise ValueError("目标 X 和 Y 需要同时填写，或同时留空以使用屏幕中心。")
            if x_text:
                try:
                    payload["target_xy_px"] = [float(x_text), float(y_text)]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("目标 X 和 Y 必须是有效数字。") from exc
        elif kind == "preprocess":
            unit = self.preprocess_time_unit.get()
            if unit not in _PREPROCESS_TIME_UNITS:
                raise ValueError("预处理必须选择 seconds、milliseconds 或 microseconds。")
            payload["time_unit"] = unit
        elif kind == "analyze":
            unit_label = self.analysis_time_unit_label.get()
            if unit_label not in _TIME_UNITS:
                raise ValueError("请选择有效的分析时间单位。")
            payload["time_unit"] = _TIME_UNITS[unit_label]

        if kind in {"calibrate", "capture", "known_target_offset", "preprocess", "analyze"}:
            if kind in {"capture", "known_target_offset", "preprocess"}:
                model_text = self.model_path.get().strip()
                if not model_text:
                    raise ValueError("请先设置模型路径。")
                model = Path(model_text).expanduser()
                if not model.exists():
                    raise FileNotFoundError(f"模型路径不存在：{model}")
                if kind in {"capture", "known_target_offset"} and (not model.is_file() or model.suffix.lower() != ".pkl"):
                    raise ValueError("采集和已知目标修正需要选择一个已存在的 .pkl 模型文件。")
                payload["model_path"] = str(model.resolve())
            if kind in {"preprocess", "analyze"}:
                source_field = self.preprocess_source_path if kind == "preprocess" else self.analysis_source_path
                source_text = source_field.get().strip()
                if not source_text:
                    raise ValueError("请先设置数据来源。")
                source = Path(source_text).expanduser()
                if not source.exists() or not (source.is_file() or source.is_dir()):
                    raise FileNotFoundError(f"数据来源不存在：{source}")
                if kind == "analyze" and not source.is_file():
                    raise ValueError("分析与导出一次处理一个文件，请在数据来源处选择单个数据文件。")
                payload["source_path"] = str(source.resolve())
            if kind == "analyze":
                roi_text = self.roi_path.get().strip()
                if roi_text:
                    roi = Path(roi_text).expanduser()
                    if not roi.is_file():
                        raise FileNotFoundError(f"ROI 工作簿不存在：{roi}")
                    payload["roi_path"] = str(roi.resolve())
        return payload

    def start_calibration(self) -> None:
        self._start_job("calibrate", "校准并保存模型")

    def start_capture(self) -> None:
        self._start_job("capture", "使用已有模型采集")

    def start_known_target_offset(self) -> None:
        self._start_job("known_target_offset", "采集已知目标并估计修正")

    def start_roi(self) -> None:
        self._start_job("roi", "打开 ROI 绘制工具")

    def start_preprocess(self) -> None:
        self._start_job("preprocess", "预处理与 QC")

    def start_analysis(self) -> None:
        self._start_job("analyze", "分析与导出")

    def _start_job(self, kind: str, display_name: str) -> None:
        if self._closing or self._busy:
            return
        try:
            payload = self._payload(kind)
            manager = self._get_manager()
            job_id = manager.start(kind, payload)
        except Exception as exc:
            messagebox.showerror("无法启动任务", str(exc), parent=self.root)
            return
        self._active_job_id = job_id
        self._active_job_kind = kind
        self._clear_last_success_outputs()
        self._set_busy(True)
        self.status_text.set(f"运行中 · {display_name}")
        self._append_log(f"已启动：{display_name}（会话 {payload['session_id']}）")

    def _get_manager(self):
        if self._manager is None:
            self._manager = self._manager_factory()
        return self._manager

    @staticmethod
    def _default_manager_factory():
        # This import is deliberately deferred until the user starts a task.
        from desktop_jobs import JobManager
        return JobManager()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in (*self._input_controls, *self._browse_controls, *self._action_controls):
            try:
                widget.configure(state="disabled" if busy or self._closing else "normal")
            except tk.TclError:
                pass
        try:
            self.cancel_button.configure(state="normal" if busy and not self._closing else "disabled")
        except (AttributeError, tk.TclError):
            pass

    def cancel_job(self) -> None:
        if self._manager is not None and self._busy:
            if self._manager.cancel():
                self.status_text.set("正在取消 · 等待当前任务退出")
                self._append_log("已请求取消；正在等待任务进程退出。")
                self.cancel_button.configure(state="disabled")

    def _schedule_poll(self) -> None:
        if self._closing and self._manager is None:
            return
        self._poll_after_id = self.root.after(250, self._poll_jobs)

    def _poll_jobs(self) -> None:
        self._poll_after_id = None
        manager = self._manager
        if manager is not None:
            for event in manager.poll():
                if event.get("job_id") != self._active_job_id:
                    continue
                self._handle_event(event)
        if self._closing:
            if manager is None or manager.closed:
                self.root.destroy()
                return
            self._schedule_poll()
        elif self._auto_poll:
            self._schedule_poll()
        elif manager is not None and manager.busy:
            self._schedule_poll()

    def _handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        message = str(event.get("message", ""))
        if event_type == "log":
            self._append_log(message)
            return
        if event_type == "started":
            return
        if event_type in {"completed", "failed", "cancelled"}:
            self._set_busy(False)
            outputs = event.get("outputs")
            known_outputs = outputs if isinstance(outputs, dict) else {}
            active_kind = self._active_job_kind
            if self._active_job_kind == "roi" and event_type in {"completed", "cancelled"} and not known_outputs:
                self.status_text.set("工具会话结束 · 可开始下一项")
                self._append_log("ROI 工具会话结束。")
            else:
                labels = {"completed": "已完成", "failed": "失败", "cancelled": "已取消"}
                label = labels.get(str(event_type), str(event_type))
                self.status_text.set(f"{label} · 可开始下一项")
                self._append_log(f"任务{label}：{message or label}")
            if isinstance(outputs, dict):
                for name, path in outputs.items():
                    if path:
                        self._append_log(f"  {name}：{path}")
            if event_type == "completed":
                self._prefill_success_outputs(active_kind, known_outputs)
                self._remember_success_outputs(known_outputs, event.get("job_id") or self._active_job_id)
            details = event.get("details")
            if event_type == "failed" and isinstance(details, dict):
                reason = details.get("reason") or details.get("error")
                if reason:
                    self._append_log(f"原因：{reason}")
                planned_path = details.get("calibration_status_path")
                if planned_path and not details.get("calibration_status_written"):
                    self._append_log(f"计划路径/未写入：{planned_path}")
            self._active_job_id = None
            self._active_job_kind = None

    def _prefill_success_outputs(self, kind: str | None, outputs: dict[str, object]) -> None:
        def fill(field: tk.StringVar, output_key: str, description: str) -> bool:
            value = outputs.get(output_key)
            if not value or not Path(str(value)).is_file():
                return False
            field.set(str(Path(str(value)).resolve()))
            self._append_log(f"已预填{description}；请检查内容后再启动下一项。")
            return True

        if kind == "calibrate":
            fill(self.model_path, "model", "模型路径")
        elif kind == "capture":
            fill(self.preprocess_source_path, "samples", "预处理数据来源")
            fill(self.analysis_source_path, "samples", "分析数据文件")
        elif kind == "known_target_offset":
            if fill(self.correction_path, "correction", "修正文件"):
                self._append_log("修正文件仍处于未启用状态；需要勾选后才会用于正式采集。")
        elif kind == "preprocess":
            if not fill(self.analysis_source_path, "processed_file", "分析数据文件"):
                self._append_log("本次预处理没有唯一的已完成文件可预填；请选择要分析的单个文件。")
        elif kind == "roi":
            fill(self.roi_path, "roi_workbook", "ROI 工作簿")

    @staticmethod
    def _existing_output_paths(outputs: dict[str, object]) -> dict[str, Path]:
        existing: dict[str, Path] = {}
        for name, value in outputs.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(value, (str, os.PathLike)):
                continue
            try:
                raw_path = os.fspath(value)
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue
                path = Path(raw_path).expanduser().resolve(strict=True)
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            if path.is_file() or path.is_dir():
                existing[name] = path
        return existing

    def _clear_last_success_outputs(self) -> None:
        self._last_success_outputs = {}
        self._last_success_job_id = None
        if hasattr(self, "outputs_menu"):
            self.outputs_menu.delete(0, "end")
        if hasattr(self, "open_outputs_button"):
            self.open_outputs_button.configure(state="disabled")

    def _remember_success_outputs(self, outputs: dict[str, object], job_id: object) -> None:
        self._last_success_outputs = self._existing_output_paths(outputs)
        self._last_success_job_id = str(job_id) if job_id else None
        self.outputs_menu.delete(0, "end")
        for name, path in self._last_success_outputs.items():
            self.outputs_menu.add_command(
                label=f"{name}：{path.name}",
                command=lambda output_name=name: self._open_job_output(output_name),
            )
        self.open_outputs_button.configure(state="normal" if self._last_success_outputs else "disabled")
        if self._last_success_outputs:
            self._append_log("本次成功输出可从底部“打开本次输出”查看。")
        else:
            self._append_log("任务完成，但没有仍存在的本次输出路径可打开。")

    def _open_job_output(self, name: str) -> None:
        path = self._last_success_outputs.get(name)
        if path is None:
            return
        existing = self._existing_output_paths({name: str(path)})
        if not existing:
            self._last_success_outputs.pop(name, None)
            self.outputs_menu.delete(0, "end")
            for output_name, output_path in self._last_success_outputs.items():
                self.outputs_menu.add_command(
                    label=f"{output_name}：{output_path.name}",
                    command=lambda selected_name=output_name: self._open_job_output(selected_name),
                )
            self.open_outputs_button.configure(state="normal" if self._last_success_outputs else "disabled")
            self._append_log(f"本次输出已不存在，无法打开：{path}")
            return
        current_path = existing[name]
        try:
            self._open_path_with_os(current_path)
        except OSError as exc:
            messagebox.showerror("无法打开本次输出", f"{current_path}\n\n{exc}", parent=self.root)

    def _open_path_with_os(self, path: Path) -> None:
        if hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("打开本次输出", f"请在文件管理器中打开：\n{path}", parent=self.root)

    def request_close(self) -> None:
        if self._closing:
            return
        manager = self._manager
        if manager is None or not manager.busy:
            self.root.destroy()
            return
        self._closing = True
        self._set_busy(True)
        self.status_text.set("正在安全关闭 · 等待任务进程退出")
        self._append_log("窗口关闭时已请求停止当前任务；将在所属进程退出后关闭窗口。")
        manager.shutdown()
        if self._poll_after_id is None:
            self._schedule_poll()

    def _update_tab_state(self, _event=None) -> None:
        # Do not let tab changes re-enable a control while a task is active.
        if self._busy:
            self._set_busy(True)

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, message: str) -> None:
        if not hasattr(self, "log"):
            return
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        try:
            line_count = int(self.log.index("end-1c").split(".")[0])
            if line_count > 600:
                self.log.delete("1.0", f"{line_count - 500}.0")
        except (ValueError, tk.TclError):
            pass
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open_guide(self, path: Path) -> None:
        if not path.is_file():
            messagebox.showerror("找不到指导语", f"未找到说明文件：\n{path}", parent=self.root)
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                messagebox.showinfo("使用指南", f"请打开此文件查看完整说明：\n{path}", parent=self.root)
        except OSError as exc:
            messagebox.showerror("无法打开指导语", str(exc), parent=self.root)


def main() -> int:
    """Start the desktop application after enabling Windows DPI awareness."""
    _enable_dpi_awareness()
    root = tk.Tk()
    DesktopApp(root)
    root.mainloop()
    return 0


def _enable_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


__all__ = ["DesktopApp", "main"]
