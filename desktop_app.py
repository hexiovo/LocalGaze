"""Lightweight Chinese desktop entry point for LocalGaze.

The module builds the Tk view only. Capture, preprocessing, ROI drawing and
analysis code are started in an owned worker after an explicit button click.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from session_names import canonical_session_id
from EyeEvaluate.Evaluate.analysis_options import ANALYSIS_METRIC_GROUPS, normalize_table_format


_COLORS = {
    "canvas": "#FCF8F2",
    "surface": "#FFFFFF",
    "ink": "#27324A",
    "navy": "#27324A",
    "navy_light": "#394662",
    "muted": "#667085",
    "line": "#D7DCE5",
    "teal": "#20C46A",
    "teal_dark": "#13894A",
    "teal_pale": "#E9FBF1",
    "orange": "#FFB4AA",
    "sky": "#E6F7FB",
    "lavender": "#E6E6FA",
    "danger": "#A62F3A",
}

_TIME_UNITS = {
    "自动识别": None,
    "秒 (seconds)": "seconds",
    "毫秒 (milliseconds)": "milliseconds",
    "微秒 (microseconds)": "microseconds",
}
_PREPROCESS_TIME_UNITS = ("seconds", "milliseconds", "microseconds")
_QUICK_START_FILENAME = "快速开始.txt"
_SKILL_GUIDE_FILENAME = "README.txt"
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
        self._dependency_process: subprocess.Popen[str] | None = None
        self._dependency_events: queue.Queue[tuple[str, str | int]] = queue.Queue()
        self._dependency_install_active = False
        self._dependency_cancel_requested = False
        self._environment_dialog: tk.Toplevel | None = None
        self._environment_refresh: Callable[[], None] | None = None
        self._environment_install_button: ttk.Button | None = None
        self._environment_install_status = tk.StringVar(root, "")
        self._configuration_path: Path | None = None
        self._auto_poll = auto_poll
        self._poll_after_id = None
        self._input_controls: list[tk.Widget] = []
        self._action_controls: list[tk.Widget] = []
        self._primary_actions: dict[str, tk.Widget] = {}
        self._browse_controls: list[tk.Widget] = []
        self._page_ids: list[str] = []
        self._page_buttons: list[ttk.Button] = []
        self._advanced_sections: list[tuple[ttk.Frame, ttk.Button]] = []
        self._capture_action_buttons: tuple[ttk.Button, ...] = ()
        self._capture_actions_stacked: bool | None = None
        self._active_page_index = 0
        self._busy = False
        self._last_success_outputs: dict[str, Path] = {}
        self._last_success_job_id: str | None = None
        self._scroll_pages: list[tuple[tk.Widget, tk.Canvas, tk.Widget]] = []
        self._scrollregion_after: dict[tk.Canvas, str] = {}
        self._canvas_widths: dict[tk.Canvas, int] = {}
        self._wheel_binding_ids: dict[str, str] = {}

        self.session_id = tk.StringVar(root, "")
        self.capture_snapshot_path = tk.StringVar(root, "")
        self.capture_context_path = tk.StringVar(root, "")
        self.model_path = tk.StringVar(root, "")
        self.correction_path = tk.StringVar(root, "")
        self.use_correction_for_capture = tk.BooleanVar(root, False)
        self.motion_monitor_enabled = tk.BooleanVar(root, True)
        self.save_motion_keyframes = tk.BooleanVar(root, True)
        self.target_x_px = tk.StringVar(root, "")
        self.target_y_px = tk.StringVar(root, "")
        self.calibration_estimator_label = tk.StringVar(root, "Ridge（默认）")
        self.calibration_formula_label = tk.StringVar(root, "v2（默认，紧凑眼部与头部特征）")
        self.preprocess_source_path = tk.StringVar(root, "")
        self.preprocess_mapping_table_path = tk.StringVar(root, "")
        self.analysis_source_path = tk.StringVar(root, "")
        self.analysis_source_selection_text = tk.StringVar(root, "单文件或文件夹模式")
        self.analysis_recursive = tk.BooleanVar(root, False)
        self._analysis_selected_files: list[Path] = []
        self.output_root = tk.StringVar(root, "")
        self.roi_path = tk.StringVar(root, "")
        self.preprocess_time_unit = tk.StringVar(root, "seconds")
        from desktop_configuration import PREPROCESS_PARAMETER_DEFAULTS

        self.preprocess_config_vars = {
            name: tk.StringVar(root, value=value)
            for name, value in PREPROCESS_PARAMETER_DEFAULTS.items()
        }
        self.analysis_time_unit_label = tk.StringVar(root, "自动识别")
        self.analysis_table_format_label = tk.StringVar(root, "CSV（逗号分隔）")
        self.capture_table_format_label = tk.StringVar(root, "CSV（逗号分隔）")
        self.analysis_metric_vars = {group_id: tk.BooleanVar(root, True) for group_id, _ in ANALYSIS_METRIC_GROUPS}
        self.analysis_config_vars = {
            "max_time_gap_s": tk.StringVar(root, ""),
            "gap_median_multiplier": tk.StringVar(root, "3.0"),
            "fixation_min_samples": tk.StringVar(root, "3"),
            "fixation_min_duration_s": tk.StringVar(root, "0.120"),
            "fixation_dispersion_diagonal_ratio": tk.StringVar(root, "0.03"),
            "fast_speed_diagonals_s": tk.StringVar(root, "1.0"),
            "ivt_speed_diagonals_s": tk.StringVar(root, "1.0"),
            "ivt_min_duration_s": tk.StringVar(root, "0.020"),
            "ivt_max_duration_s": tk.StringVar(root, "0.200"),
            "ivt_min_samples": tk.StringVar(root, "3"),
            "blink_min_duration_s": tk.StringVar(root, "0.050"),
            "blink_max_duration_s": tk.StringVar(root, "0.500"),
            "entropy_grid_width": tk.StringVar(root, "32"),
            "entropy_grid_height": tk.StringVar(root, "18"),
        }
        self._analysis_parameter_widgets: dict[str, tuple[str, list[tk.Widget]]] = {}
        self.analysis_source_path.trace_add("write", self._clear_analysis_file_selection)
        self.status_text = tk.StringVar(root, "就绪 · 选择一个操作开始")
        self.capture_quality_text = tk.StringVar(root, "采集质量 · 等待采集")
        self.capture_snapshot_path.trace_add("write", lambda *_: self._sync_capture_retry_button())
        self.capture_context_path.trace_add("write", lambda *_: self._sync_capture_retry_button())
        self._last_job_logs: dict[str, Path] = {}

        self._configure_root()
        self._configure_styles()
        self._build_view()
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            binding_id = self.root.bind(sequence, self._on_mousewheel, add="+")
            if binding_id:
                self._wheel_binding_ids[sequence] = binding_id
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        if self._auto_poll:
            self._schedule_poll()

    def _configure_root(self) -> None:
        self.root.title("LocalGaze · 眼动实验工作台")
        icon_path = Path(__file__).resolve().parent / "assets" / "localgaze.ico"
        if os.name == "nt" and icon_path.is_file():
            self.root.iconbitmap(default=str(icon_path))
        self.root.configure(background=_COLORS["canvas"])
        self.root.minsize(780, 570)
        initial_height = min(900, max(570, self.root.winfo_screenheight() - 40))
        self.root.geometry(f"1320x{initial_height}")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        body_font = ("Microsoft YaHei UI", -20)
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name, root=self.root).configure(family=body_font[0], size=body_font[1])
        self.root.option_add("*TCombobox*Listbox.font", body_font)
        style.configure(".", font=body_font)
        style.configure("TFrame", background=_COLORS["canvas"])
        style.configure("App.TFrame", background=_COLORS["canvas"])
        style.configure("Card.TFrame", background=_COLORS["surface"], relief="flat", borderwidth=0)
        style.configure("OuterCard.TFrame", background=_COLORS["surface"], relief="solid", borderwidth=1)
        style.configure("NavRail.TFrame", background=_COLORS["navy"])
        style.configure("NavDivider.TFrame", background=_COLORS["navy_light"])
        style.configure("PageHeader.TFrame", background=_COLORS["sky"], relief="solid", borderwidth=1)
        style.configure("TButton", font=body_font)
        style.configure("TCheckbutton", background=_COLORS["surface"], font=body_font)
        style.configure("TLabel", background=_COLORS["canvas"], foreground=_COLORS["ink"], font=body_font)
        style.configure("Card.TLabel", background=_COLORS["surface"], foreground=_COLORS["ink"], font=body_font)
        style.configure("Muted.TLabel", background=_COLORS["surface"], foreground=_COLORS["muted"], font=("Microsoft YaHei UI", -17))
        style.configure("Header.TLabel", background=_COLORS["canvas"], foreground=_COLORS["ink"], font=("Microsoft YaHei UI", -24, "bold"))
        style.configure("Eyebrow.TLabel", background=_COLORS["canvas"], foreground=_COLORS["teal_dark"], font=("Microsoft YaHei UI", -17, "bold"))
        style.configure("Brand.TLabel", background=_COLORS["navy"], foreground="#FFFFFF", font=("Microsoft YaHei UI", -24, "bold"))
        style.configure("RailHint.TLabel", background=_COLORS["navy"], foreground="#D4DAE5", font=("Microsoft YaHei UI", -18))
        style.configure("PageNumber.TLabel", background=_COLORS["teal"], foreground=_COLORS["navy"], padding=(9, 5), font=("Microsoft YaHei UI", -18, "bold"))
        style.configure("PageTitle.TLabel", background=_COLORS["sky"], foreground=_COLORS["ink"], font=("Microsoft YaHei UI", -30, "bold"))
        style.configure("PageHint.TLabel", background=_COLORS["sky"], foreground=_COLORS["muted"], font=("Microsoft YaHei UI", -17))
        style.configure("Status.TLabel", background=_COLORS["navy"], foreground="#FFFFFF", padding=(11, 7), font=("Microsoft YaHei UI", -18, "bold"))
        style.configure("Section.TLabel", background=_COLORS["surface"], foreground=_COLORS["ink"], font=("Microsoft YaHei UI", -24, "bold"))
        style.configure("TNotebook", background=_COLORS["canvas"], borderwidth=0)
        style.layout("Hidden.TNotebook", [("Notebook.client", {"sticky": "nswe"})])
        style.layout("Hidden.TNotebook.Tab", [])
        style.configure("Hidden.TNotebook", background=_COLORS["canvas"], borderwidth=0, tabmargins=0)
        style.configure("TEntry", padding=(6, 5), fieldbackground="#FFFFFF", font=body_font)
        style.configure("TCombobox", padding=(6, 5), fieldbackground="#FFFFFF", font=body_font)
        style.configure("Primary.TButton", background=_COLORS["teal"], foreground=_COLORS["navy"], padding=(14, 9), borderwidth=2, relief="raised", font=("Microsoft YaHei UI", -20, "bold"))
        style.map("Primary.TButton", background=[("pressed", _COLORS["teal_dark"]), ("active", "#59D68F"), ("disabled", "#CAD8CE")], foreground=[("disabled", "#68736D")])
        style.configure("StartAction.TButton", background=_COLORS["teal"], foreground=_COLORS["navy"], padding=(15, 11), borderwidth=2, relief="raised", font=("Microsoft YaHei UI", -20, "bold"))
        style.map("StartAction.TButton", background=[("pressed", _COLORS["teal_dark"]), ("active", "#59D68F"), ("disabled", "#CAD8CE")], foreground=[("disabled", "#68736D")])
        style.configure("CalibrationAction.TButton", background=_COLORS["orange"], foreground=_COLORS["ink"], padding=(13, 11), borderwidth=2, relief="raised", font=("Microsoft YaHei UI", -20, "bold"))
        style.map("CalibrationAction.TButton", background=[("pressed", "#F18F83"), ("active", "#FFD0C9"), ("disabled", "#E7D7D4")], foreground=[("disabled", "#766D6B")])
        style.configure("Secondary.TButton", background=_COLORS["teal_pale"], foreground=_COLORS["teal_dark"], padding=(11, 7), borderwidth=1, relief="solid")
        style.map("Secondary.TButton", background=[("active", "#D2F2DE"), ("disabled", "#EDF3EF")])
        style.configure("Quiet.TButton", background=_COLORS["surface"], foreground=_COLORS["muted"], padding=(9, 6), borderwidth=1, relief="solid")
        style.configure("Danger.TButton", background="#FBEDEC", foreground=_COLORS["danger"], padding=(10, 6), borderwidth=1, relief="solid")
        style.configure("Nav.TButton", background=_COLORS["navy"], foreground="#E9EDF4", padding=(9, 6), anchor="w", borderwidth=1, relief="flat", font=("Microsoft YaHei UI", -18, "bold"))
        style.map("Nav.TButton", background=[("active", _COLORS["navy_light"]), ("disabled", _COLORS["navy"])], foreground=[("disabled", "#98A2B3")])
        style.configure("NavActive.TButton", background=_COLORS["navy_light"], foreground="#FFFFFF", padding=(9, 6), anchor="w", borderwidth=2, relief="raised", font=("Microsoft YaHei UI", -18, "bold"))
        style.map("NavActive.TButton", background=[("active", "#465573"), ("disabled", _COLORS["navy_light"])], foreground=[("disabled", "#D4DAE5")])
        style.configure("RailQuiet.TButton", background=_COLORS["navy"], foreground="#D4DAE5", padding=(8, 3), anchor="w", borderwidth=1, relief="flat", font=("Microsoft YaHei UI", -18))
        style.map("RailQuiet.TButton", background=[("active", _COLORS["navy_light"])], foreground=[("active", "#FFFFFF")])

    def _build_view(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        workspace = ttk.Frame(self.root, style="App.TFrame")
        workspace.grid(row=0, column=0, sticky="nsew")
        workspace.columnconfigure(0, weight=0, minsize=232)
        workspace.columnconfigure(1, weight=1)
        workspace.rowconfigure(0, weight=1)

        rail = ttk.Frame(workspace, style="NavRail.TFrame", padding=(12, 12, 12, 8), width=232)
        rail.grid(row=0, column=0, sticky="nsw")
        rail.grid_propagate(False)
        rail.columnconfigure(0, weight=1)
        rail.rowconfigure(10, weight=1)
        ttk.Label(rail, text="LOCALGAZE", style="Brand.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(rail, text="眼动实验工作台", style="RailHint.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 10))
        divider = ttk.Frame(rail, style="NavDivider.TFrame", height=1)
        divider.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(rail, text="工作步骤", style="RailHint.TLabel").grid(row=3, column=0, sticky="w", pady=(0, 5))

        self._page_info = (
            ("01", "会话与采集", "设置本次会话，然后校准、采集或检查采集结果。"),
            ("02", "ROI 绘制", "在目标显示器上标注感兴趣区域，并保存到 ROI 工作簿。"),
            ("03", "预处理 QC", "选择眼动数据，执行预处理并查看质量检查结果。"),
            ("04", "分析导出", "选择指标与 ROI，运行分析并导出结果表格。"),
            ("05", "使用指南", "查看快速开始、实验设计资料和环境检查。"),
        )
        self._page_buttons.clear()
        for index, (number, title, _hint) in enumerate(self._page_info):
            button = ttk.Button(
                rail,
                text=f"{number}   {title}",
                style="Nav.TButton",
                takefocus=True,
                command=lambda target=index: self._select_page(target),
            )
            button.grid(row=5 + index, column=0, sticky="ew", pady=(0, 3))
            button.bind("<Up>", lambda event, target=index: self._focus_adjacent_page(target, -1), add="+")
            button.bind("<Down>", lambda event, target=index: self._focus_adjacent_page(target, 1), add="+")
            button.bind("<Return>", lambda _event, target=index: self._select_page(target), add="+")
            self._page_buttons.append(button)

        rail_tools = ttk.Frame(rail, style="NavRail.TFrame")
        rail_tools.grid(row=11, column=0, sticky="sew")
        rail_tools.columnconfigure(0, weight=1)
        ttk.Frame(rail_tools, style="NavDivider.TFrame", height=1).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        rail_help_label = ttk.Label(rail_tools, text="帮助与环境", style="RailHint.TLabel")
        rail_help_label.grid(row=1, column=0, sticky="w", pady=(0, 4))
        self.guide_button = ttk.Button(rail_tools, text="查看使用指南", style="RailQuiet.TButton", command=self._show_guide)
        self.guide_button.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        self.materials_button = ttk.Button(rail_tools, text="实验设计资料", style="RailQuiet.TButton", command=self._open_study_materials)
        self.materials_button.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        self.environment_button = ttk.Button(rail_tools, text="环境检查", style="RailQuiet.TButton", command=self.show_environment_check)
        self.environment_button.grid(row=4, column=0, sticky="ew")
        self._action_controls.append(self.environment_button)

        def fit_rail(event: tk.Event) -> None:
            # The guide remains available as page 05 when vertical space is tight.
            if event.height < 600:
                rail_help_label.grid_remove()
                self.guide_button.grid_remove()
            else:
                rail_help_label.grid()
                self.guide_button.grid()

        rail.bind("<Configure>", fit_rail, add="+")

        main = ttk.Frame(workspace, style="App.TFrame", padding=(16, 15, 16, 7))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        page_header = ttk.Frame(main, style="PageHeader.TFrame", padding=(14, 11, 14, 11))
        page_header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        page_header.columnconfigure(1, weight=1)
        self.page_number_label = ttk.Label(page_header, text="01", style="PageNumber.TLabel")
        self.page_number_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 12))
        self.page_title_label = ttk.Label(page_header, text="会话与采集", style="PageTitle.TLabel")
        self.page_title_label.grid(row=0, column=1, sticky="w")
        self.page_hint_label = ttk.Label(page_header, text="", style="PageHint.TLabel", justify="left")
        self.page_hint_label.grid(row=1, column=1, sticky="ew", pady=(2, 0))
        self._wrap_to_width(self.page_hint_label, page_header, inset=160, minimum=200)

        config_bar = ttk.Frame(main, style="Card.TFrame", padding=(12, 7))
        config_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        config_bar.columnconfigure(1, weight=1)
        self.configuration_title_label = ttk.Label(config_bar, text="本地自定义配置", style="Card.TLabel")
        self.configuration_title_label.grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.configuration_status_text = tk.StringVar(self.root, "未载入配置")
        self.configuration_status_label = ttk.Label(
            config_bar, textvariable=self.configuration_status_text, style="Muted.TLabel", justify="left"
        )
        self.configuration_status_label.grid(row=0, column=1, sticky="ew")
        self.configuration_status_label.bind(
            "<Configure>",
            lambda event: self.configuration_status_label.configure(wraplength=max(180, event.width - 8)),
            add="+",
        )
        self.load_configuration_button = ttk.Button(config_bar, text="加载配置…", style="Secondary.TButton", command=self.load_local_configuration)
        self.load_configuration_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.save_configuration_button = ttk.Button(config_bar, text="保存配置…", style="Secondary.TButton", command=self.save_local_configuration)
        self.save_configuration_button.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self._configuration_toolbar_compact: bool | None = None
        config_bar.bind("<Configure>", self._layout_configuration_toolbar, add="+")
        self._action_controls.extend((self.load_configuration_button, self.save_configuration_button))

        self.notebook = ttk.Notebook(main, style="Hidden.TNotebook")
        self.notebook.grid(row=2, column=0, sticky="nsew")

        self._build_session_tab()
        self._build_roi_tab()
        self._build_preprocess_tab()
        self._build_analysis_tab()
        self._build_guide_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._update_tab_state, add="+")
        for index, key in enumerate(("<Alt-Key-1>", "<Alt-Key-2>", "<Alt-Key-3>", "<Alt-Key-4>", "<Alt-Key-5>")):
            self.root.bind(key, lambda _event, target=index: self._select_page(target), add="+")
        self._update_tab_state()

        self._build_log_panel()

    def _layout_configuration_toolbar(self, event: tk.Event) -> None:
        compact = event.width < 860
        if compact == self._configuration_toolbar_compact:
            return
        if compact:
            self.configuration_title_label.grid_remove()
            self.configuration_status_label.grid_configure(row=0, column=0, columnspan=4, sticky="ew")
            self.load_configuration_button.grid_configure(row=1, column=0, sticky="w", padx=(0, 8), pady=(5, 0))
            self.save_configuration_button.grid_configure(row=1, column=1, sticky="w", padx=(0, 0), pady=(5, 0))
        else:
            self.configuration_title_label.grid(row=0, column=0, sticky="w", padx=(0, 12))
            self.configuration_status_label.grid_configure(row=0, column=1, columnspan=1, sticky="ew")
            self.load_configuration_button.grid_configure(row=0, column=2, sticky="e", padx=(8, 0), pady=(0, 0))
            self.save_configuration_button.grid_configure(row=0, column=3, sticky="e", padx=(8, 0), pady=(0, 0))
        self._configuration_toolbar_compact = compact

    def _select_page(self, index: int) -> None:
        if 0 <= index < len(self._page_ids):
            self.notebook.select(self._page_ids[index])
            self._update_tab_state()

    def _focus_adjacent_page(self, index: int, direction: int) -> str:
        target = (index + direction) % len(self._page_buttons)
        self._page_buttons[target].focus_set()
        return "break"

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
        self._scroll_pages.append((page, canvas, content))
        self._page_ids.append(str(page))
        content.bind("<Configure>", lambda _event, target=canvas: self._queue_scrollregion_refresh(target), add="+")
        canvas.bind(
            "<Configure>",
            lambda event, target=canvas, item=window: self._resize_canvas_content(target, item, event.width),
            add="+",
        )
        self.notebook.add(page, text=title)
        return content

    def _resize_canvas_content(self, canvas: tk.Canvas, window: int, width: int) -> None:
        if width < 1:
            return
        if self._canvas_widths.get(canvas) != width:
            self._canvas_widths[canvas] = width
            try:
                canvas.itemconfigure(window, width=width)
            except tk.TclError:
                return
        self._queue_scrollregion_refresh(canvas)

    def _queue_scrollregion_refresh(self, canvas: tk.Canvas) -> None:
        if canvas in self._scrollregion_after:
            return
        try:
            self._scrollregion_after[canvas] = self.root.after_idle(
                lambda target=canvas: self._refresh_scrollregion(target)
            )
        except tk.TclError:
            pass

    def _refresh_scrollregion(self, canvas: tk.Canvas) -> None:
        self._scrollregion_after.pop(canvas, None)
        try:
            bounds = canvas.bbox("all")
            if bounds is None:
                return
            current_value = canvas.cget("scrollregion").strip()
            current = tuple(int(value) for value in current_value.split()) if current_value else None
            if current != bounds:
                canvas.configure(scrollregion=bounds)
        except (tk.TclError, ValueError):
            pass

    @staticmethod
    def _is_descendant(widget: tk.Widget, ancestor: tk.Widget) -> bool:
        current = widget
        while current is not None:
            if current is ancestor:
                return True
            current = getattr(current, "master", None)
        return False

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        widget = event.widget
        widget_class = widget.winfo_class()
        if widget_class in {"Entry", "TEntry", "TCombobox", "Combobox", "TSpinbox", "Spinbox", "Text", "Listbox", "Scale", "TScale", "Scrollbar"}:
            return None
        selected = self.notebook.select()
        target = next(
            (
                canvas
                for page, canvas, content in self._scroll_pages
                if selected == str(page)
                and (widget is page or widget is canvas or self._is_descendant(widget, content))
            ),
            None,
        )
        if target is None:
            return None
        event_number = getattr(event, "num", None)
        if event_number == 4:
            units = -1
        elif event_number == 5:
            units = 1
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return None
            units = -int(delta / 120)
            if units == 0:
                units = -1 if delta > 0 else 1
        first, last = target.yview()
        if last - first >= 0.999:
            return None
        target.yview_scroll(units, "units")
        return "break"

    def _unbind_mousewheel(self) -> None:
        for sequence, binding_id in self._wheel_binding_ids.items():
            try:
                self.root.unbind(sequence, binding_id)
            except tk.TclError:
                pass
        self._wheel_binding_ids.clear()

    def _card(self, parent: ttk.Frame, row: int, title: str, subtitle: str | None = None) -> ttk.Frame:
        card = ttk.Frame(parent, style="OuterCard.TFrame", padding=(18, 16, 18, 17))
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
            if not label.winfo_exists() or not container.winfo_exists():
                return
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
        self._advanced_sections.append((panel, button))
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
        settings = self._card(page, 0, "本次会话", "先设置会话 ID、输出位置和模型；例如 P001_session1。模型用于采集或关联预处理质量检查。")
        self._compact_entry_row(settings, 2, "会话 ID", self.session_id, browse=None)
        self._compact_entry_row(settings, 3, "输出位置", self.output_root, browse="directory")
        self._compact_entry_row(settings, 4, "模型路径", self.model_path, browse="model")
        advanced = self._advanced_section(settings, 5, "校准、目标与修正参数")
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
        self._entry_row(advanced, 2, "修正文件", self.correction_path, browse="correction", hint="已知目标成功后自动填入；采集默认不使用")
        use_correction = ttk.Checkbutton(
            advanced,
            text="在正式采集中显式应用此修正文件",
            variable=self.use_correction_for_capture,
        )
        use_correction.grid(row=3, column=1, columnspan=3, sticky="w", padx=(12, 0), pady=(8, 0))
        self._input_controls.append(use_correction)
        actions = self._card(page, 1, "开始操作", "选择要执行的工作。只有点击采集或校准按钮后才会启动相机；任务运行期间可以取消。")
        action_row = ttk.Frame(actions, style="Card.TFrame")
        action_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(3, 0))
        for column in range(3):
            action_row.columnconfigure(column, weight=1, uniform="capture-actions")
        calibrate = ttk.Button(action_row, text="校准并保存模型", style="CalibrationAction.TButton", command=self.start_calibration)
        calibrate.grid(row=0, column=0, sticky="ew", padx=(0, 7))
        capture = ttk.Button(action_row, text="使用已有模型采集", style="StartAction.TButton", command=self.start_capture)
        capture.grid(row=0, column=1, sticky="ew", padx=7)
        validate = ttk.Button(
            action_row,
            text="锁定模型九点独立验证",
            style="Secondary.TButton",
            command=self.start_independent_validation,
        )
        validate.grid(row=0, column=2, sticky="ew", padx=(7, 0))
        self._capture_action_buttons = (calibrate, capture, validate)
        action_row.bind("<Configure>", self._layout_capture_actions, add="+")
        self._action_controls.extend((calibrate, capture, validate))
        self._primary_actions.update(calibrate=calibrate, capture=capture, independent_validation=validate)
        capture_options = self._advanced_section(actions, 4, "采集选项与表格格式")
        motion_option = ttk.Checkbutton(
            capture_options,
            text="正式采集中监测头部与上身运动",
            variable=self.motion_monitor_enabled,
        )
        motion_option.grid(row=0, column=0, columnspan=4, sticky="w", padx=(12, 0), pady=(5, 0))
        keyframe_option = ttk.Checkbutton(
            capture_options,
            text="保存事件关键帧截图（每个事件最多三张）",
            variable=self.save_motion_keyframes,
        )
        keyframe_option.grid(row=1, column=0, columnspan=4, sticky="w", padx=(12, 0), pady=(4, 0))
        self._input_controls.extend((motion_option, keyframe_option))
        ttk.Label(capture_options, text="采集表格格式", style="Card.TLabel").grid(row=2, column=0, sticky="w", padx=(12, 0), pady=(8, 0))
        capture_format_box = ttk.Combobox(
            capture_options,
            textvariable=self.capture_table_format_label,
            values=("CSV（逗号分隔）", "TXT（制表符分隔）"),
            state="readonly",
            width=19,
        )
        capture_format_box.grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(8, 0))
        self._input_controls.append(capture_format_box)
        action_hint = ttk.Label(actions, text="推荐顺序：先校准并保存模型，再用对应模型开始采集。", style="Muted.TLabel")
        action_hint.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self._wrap_to_width(action_hint, actions, inset=36)

        recovery = self._advanced_section(actions, 6, "快照恢复与已知目标修正")
        self._entry_row(
            recovery, 0, "采集快照文件", self.capture_snapshot_path,
            browse="snapshot",
            hint="仅从已保存的 *.capture-snapshot.json 重试 XLSX/manifest 导出，不会启动相机。",
        )
        self.capture_export_retry_button = ttk.Button(
            recovery,
            text="仅从快照重试导出",
            style="Secondary.TButton",
            command=self.start_capture_export_retry,
            state="disabled",
        )
        self.capture_export_retry_button.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._action_controls.append(self.capture_export_retry_button)
        ttk.Label(recovery, text="已知目标修正", style="Section.TLabel").grid(row=2, column=0, columnspan=4, sticky="w", pady=(12, 0))
        ttk.Label(recovery, text="在已知屏幕位置显示目标后，采集一次目标样本并保存修正文件。", style="Muted.TLabel", wraplength=560).grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(recovery, text="目标位置（屏幕像素）", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=(8, 0))
        target_x = ttk.Entry(recovery, textvariable=self.target_x_px, width=12)
        target_x.grid(row=4, column=1, sticky="w", padx=(12, 8), pady=(8, 0))
        ttk.Label(recovery, text="X", style="Card.TLabel").grid(row=4, column=2, sticky="e", pady=(8, 0))
        target_y = ttk.Entry(recovery, textvariable=self.target_y_px, width=12)
        target_y.grid(row=4, column=3, sticky="w", padx=(8, 0), pady=(8, 0))
        self._input_controls.extend((target_x, target_y))
        ttk.Label(recovery, text="留空时使用屏幕中心。", style="Muted.TLabel").grid(row=5, column=1, columnspan=3, sticky="w", padx=(12, 0), pady=(2, 0))
        known_target = ttk.Button(recovery, text="采集已知目标并生成修正", style="Secondary.TButton", command=self.start_known_target_offset)
        known_target.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self._action_controls.append(known_target)
        self._primary_actions["known_target_offset"] = known_target
        self._entry_row(
            recovery, 7, "采集恢复上下文", self.capture_context_path,
            browse="capture_context",
            hint="从已保存的采集上下文、原始 TSV 与耐久检查点恢复导出；不会启动相机或校准。",
        )
        self.capture_recovery_button = ttk.Button(
            recovery,
            text="从上下文恢复采集结果",
            style="Secondary.TButton",
            command=self.start_capture_recovery,
            state="disabled",
        )
        self.capture_recovery_button.grid(row=8, column=0, sticky="w", pady=(8, 0))
        self._action_controls.append(self.capture_recovery_button)
        footer = ttk.Label(page, text="校准和正式采集流程保持不变；本页统一设置会话与输出位置。", style="Muted.TLabel")
        footer.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self._wrap_to_width(footer, page, inset=8)

    def _layout_capture_actions(self, event: tk.Event | None = None) -> None:
        action_row = self._capture_action_buttons[0].master if self._capture_action_buttons else None
        if action_row is None:
            return
        width = action_row.winfo_width() if event is None else event.width
        if width <= 1:
            return
        stacked = width < 650
        if stacked == self._capture_actions_stacked:
            return
        for button in self._capture_action_buttons:
            button.grid_forget()
        if stacked:
            for row, button in enumerate(self._capture_action_buttons):
                button.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 5))
        else:
            for column, button in enumerate(self._capture_action_buttons):
                button.grid(
                    row=0,
                    column=column,
                    sticky="ew",
                    padx=(0, 7) if column == 0 else (7, 0) if column == 2 else 7,
                )
        self._capture_actions_stacked = stacked

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
        card = self._card(page, 0, "预处理数据", "模型可选；可直接选择文件/文件夹，也可用 CSV/XLSX 映射表逐行指定数据与模型。原始文件不会被覆盖。")
        self._entry_row(card, 2, "模型文件/文件夹（可选）", self.model_path, browse="model", hint="模型只用于关联同名校准误差工作簿；不预测新坐标。留空时执行无模型 QC。")
        self._entry_row(card, 3, "眼动数据文件/文件夹", self.preprocess_source_path, browse="source", hint="可选单个文件或含多个文件的文件夹")
        mapping_options = self._advanced_section(card, 4, "映射表与模板")
        mapping = ttk.Frame(mapping_options, style="Card.TFrame")
        mapping.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        mapping.columnconfigure(0, weight=1)
        ttk.Label(mapping, text="映射表（可选）", style="Card.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        mapping_entry = ttk.Entry(mapping, textvariable=self.preprocess_mapping_table_path, width=24)
        mapping_entry.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._input_controls.append(mapping_entry)
        mapping_browse = ttk.Button(
            mapping, text="浏览…",
            command=lambda: self._browse("mapping_table", self.preprocess_mapping_table_path, "映射表"),
        )
        mapping_browse.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._browse_controls.append(mapping_browse)
        mapping_clear = ttk.Button(mapping, text="清除", command=self._clear_preprocess_mapping_table)
        mapping_clear.grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=(4, 0))
        self._input_controls.append(mapping_clear)
        mapping_hint = ttk.Label(
            mapping,
            text="选中映射表后，其数据/模型/session 行优先；空模型严格表示 None，不继承上方模型。相对路径以表格所在目录为准。",
            style="Muted.TLabel", justify="left", wraplength=560,
        )
        mapping_hint.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(3, 0))
        self._wrap_to_width(mapping_hint, mapping, inset=24)
        template_actions = ttk.Frame(mapping_options, style="Card.TFrame")
        template_actions.grid(row=3, column=0, columnspan=4, sticky="w", pady=(7, 0))
        preview_button = ttk.Button(template_actions, text="预览模板示例", command=self._preview_mapping_template)
        preview_button.grid(row=0, column=0, sticky="w")
        save_button = ttk.Button(template_actions, text="保存空白模板…", command=self._save_mapping_template)
        save_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self._action_controls.extend((preview_button, save_button))
        template_hint = ttk.Label(
            template_actions,
            text="预览示例仅供参考，不会作为任务输入；保存的模板输入页为空。",
            style="Muted.TLabel",
        )
        template_hint.grid(row=0, column=2, sticky="w", padx=(12, 0))
        self._wrap_to_width(template_hint, template_actions, inset=24)
        self._entry_row(card, 6, "输出位置", self.output_root, browse="directory")
        button = ttk.Button(card, text="运行预处理与 QC", style="Primary.TButton", command=self.start_preprocess)
        button.grid(row=7, column=0, sticky="w", pady=(13, 0))
        self._action_controls.append(button)
        self._primary_actions["preprocess"] = button
        advanced = self._advanced_section(card, 8, "时间单位与预处理参数")
        time_unit_label = ttk.Label(advanced, text="数值时间单位（新格式 elapsed_s 为秒）", style="Card.TLabel", justify="left")
        time_unit_label.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(3, 0))
        self.preprocess_time_box = ttk.Combobox(advanced, textvariable=self.preprocess_time_unit, values=_PREPROCESS_TIME_UNITS, state="readonly", width=24)
        self.preprocess_time_box.grid(row=1, column=0, sticky="w", pady=(5, 0))
        self._input_controls.append(self.preprocess_time_box)
        time_unit_hint = ttk.Label(advanced, text="旧格式纯数值时间请按来源单位选择。", style="Muted.TLabel", justify="left")
        time_unit_hint.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self._wrap_to_width(time_unit_label, advanced, inset=12, minimum=180)
        self._wrap_to_width(time_unit_hint, advanced, inset=12, minimum=180)
        from desktop_configuration import PREPROCESS_PARAMETER_LABELS

        parameter_grid = ttk.Frame(advanced, style="Card.TFrame")
        parameter_grid.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        parameter_grid.columnconfigure(0, weight=1)
        for index, (name, variable) in enumerate(self.preprocess_config_vars.items()):
            label = ttk.Label(parameter_grid, text=PREPROCESS_PARAMETER_LABELS[name], style="Card.TLabel")
            label.grid(row=index * 2, column=0, columnspan=4, sticky="w", pady=(4, 2))
            entry = ttk.Entry(parameter_grid, textvariable=variable, width=13)
            entry.grid(row=index * 2 + 1, column=0, sticky="w", pady=(0, 5))
            self._input_controls.append(entry)
        footer = ttk.Label(page, text="输出包含逐文件结果、批次报告及运行清单；映射表坏行会单独记录，其他有效行继续处理。", style="Muted.TLabel")
        footer.grid(row=1, column=0, sticky="ew")
        self._wrap_to_width(footer, page, inset=8)

    def _clear_analysis_file_selection(self, *_args) -> None:
        if self._analysis_selected_files:
            self._analysis_selected_files = []
            self.analysis_source_selection_text.set("单文件或文件夹模式")

    def _browse_analysis_multiple(self) -> None:
        current = self.analysis_source_path.get().strip()
        initial = current if Path(current).is_dir() else str(Path(current).parent) if current else str(Path.cwd())
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="选择一个或多个眼动数据文件",
            initialdir=initial,
            filetypes=(("眼动数据", "*.xlsx *.xlsm *.csv *.tsv *.txt"), ("所有文件", "*.*")),
        )
        paths = [Path(value).expanduser().resolve(strict=True) for value in selected]
        if not paths:
            return
        self.analysis_source_path.set(str(paths[0]))
        self._analysis_selected_files = paths
        self.analysis_source_selection_text.set(f"多文件模式 · 已选择 {len(paths)} 个文件")

    def _browse_analysis_folder(self) -> None:
        current = self.analysis_source_path.get().strip()
        initial = current if Path(current).is_dir() else str(Path(current).parent) if current else str(Path.cwd())
        selected = filedialog.askdirectory(
            parent=self.root, title="选择眼动数据文件夹", initialdir=initial, mustexist=True
        )
        if selected:
            self.analysis_source_path.set(selected)

    def _build_analysis_tab(self) -> None:
        page = self._new_tab("分析与导出")
        page.columnconfigure(0, weight=1)
        card = self._card(page, 0, "分析眼动数据", "可分析单个文件、多选文件或文件夹；所有文件共用当前指标、参数、ROI 和时间单位。")
        self._entry_row(card, 2, "数据文件或文件夹", self.analysis_source_path, browse="analysis_source", hint="支持 CSV、TSV、TXT、XLSX 和 XLSM；多文件或文件夹模式会生成独立批次汇总。")
        source_actions = ttk.Frame(card, style="Card.TFrame")
        source_actions.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        choose_many = ttk.Button(source_actions, text="多选文件…", command=self._browse_analysis_multiple)
        choose_many.grid(row=0, column=0, sticky="w")
        choose_folder = ttk.Button(source_actions, text="选择文件夹…", command=self._browse_analysis_folder)
        choose_folder.grid(row=0, column=1, sticky="w", padx=(8, 0))
        recursive = ttk.Checkbutton(
            source_actions, text="递归读取子文件夹", variable=self.analysis_recursive,
        )
        recursive.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._browse_controls.extend((choose_many, choose_folder))
        self._input_controls.append(recursive)
        selection_hint = ttk.Label(source_actions, textvariable=self.analysis_source_selection_text, style="Muted.TLabel")
        selection_hint.grid(row=2, column=0, columnspan=4, sticky="ew", padx=(0, 8), pady=(4, 0))
        self._wrap_to_width(selection_hint, source_actions, inset=12)
        self._entry_row(card, 4, "输出位置", self.output_root, browse="directory")
        self._entry_row(card, 5, "ROI 工作簿（可选）", self.roi_path, browse="roi", hint="留空时不计算 ROI 命中")

        output_options = ttk.Frame(card, style="Card.TFrame")
        output_options.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(output_options, text="表格格式", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.analysis_table_format_box = ttk.Combobox(
            output_options, textvariable=self.analysis_table_format_label,
            values=("CSV（逗号分隔）", "TXT（制表符分隔）"), state="readonly", width=18,
        )
        self.analysis_table_format_box.grid(row=0, column=1, sticky="w", padx=(12, 12))
        self._input_controls.append(self.analysis_table_format_box)
        output_hint = ttk.Label(output_options, text="TXT 使用制表符分隔；运行清单保留在本次结果文件夹。", style="Muted.TLabel", justify="left")
        output_hint.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self._wrap_to_width(output_hint, output_options, inset=12)

        metric_frame = ttk.Frame(card, style="Card.TFrame")
        metric_frame.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        metric_frame.columnconfigure(0, weight=1, uniform="metric-columns")
        metric_frame.columnconfigure(1, weight=1, uniform="metric-columns")
        ttk.Label(metric_frame, text="计算指标（默认全选）", style="Card.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        for index, (group_id, label) in enumerate(ANALYSIS_METRIC_GROUPS):
            check = ttk.Checkbutton(
                metric_frame, text=label, variable=self.analysis_metric_vars[group_id],
                command=self._refresh_analysis_parameter_states,
            )
            check.grid(row=1 + index // 2, column=index % 2, sticky="w", padx=(0, 10), pady=(5, 0))
            self._input_controls.append(check)
        ttk.Label(
            metric_frame,
            text="数据质量与输入来源说明始终保留；注视、快速移动、I-VT 与闭合候选可分别选择。",
            style="Muted.TLabel",
        ).grid(row=1 + (len(ANALYSIS_METRIC_GROUPS) + 1) // 2, column=0, columnspan=4, sticky="w", pady=(5, 0))

        button = ttk.Button(card, text="分析并导出结果", style="Primary.TButton", command=self.start_analysis)
        button.grid(row=8, column=0, sticky="w", pady=(8, 0))
        self._action_controls.append(button)
        self._primary_actions["analyze"] = button
        advanced = self._advanced_section(card, 9, "时间单位与指标参数")
        ttk.Label(advanced, text="数值时间单位", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=(11, 0))
        self.analysis_time_box = ttk.Combobox(advanced, textvariable=self.analysis_time_unit_label, values=tuple(_TIME_UNITS), state="readonly", width=24)
        self.analysis_time_box.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(11, 0))
        self._input_controls.append(self.analysis_time_box)
        time_hint = ttk.Label(advanced, text="自动识别会读取文件元数据；有歧义的数值时间请明确选择。", style="Muted.TLabel", justify="left")
        time_hint.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(3, 0))
        self._wrap_to_width(time_hint, advanced, inset=12)
        parameter_rows = (
            ("max_time_gap_s", "最大有效时间间隔（秒）", "留空按中位间隔倍数" , "timed"),
            ("gap_median_multiplier", "最大间隔中位数倍数", "默认 3.0", "timed_multiplier"),
            ("fixation_min_samples", "注视最少样本数", "默认 3", "fixation"),
            ("fixation_min_duration_s", "注视最短时长（秒）", "默认 0.120", "fixation"),
            ("fixation_dispersion_diagonal_ratio", "注视离散度／屏幕对角线", "默认 0.03", "fixation"),
            ("fast_speed_diagonals_s", "快速移动速度阈值（屏幕对角线／秒）", "默认 1.0", "fast_movement"),
            ("ivt_speed_diagonals_s", "I-VT 速度阈值（屏幕对角线／秒）", "默认 1.0；任意正值", "ivt_saccade"),
            ("ivt_min_duration_s", "I-VT 候选最短时长（秒）", "默认 0.020；整个候选", "ivt_saccade"),
            ("ivt_max_duration_s", "I-VT 候选最长时长（秒）", "默认 0.200；整个候选", "ivt_saccade"),
            ("ivt_min_samples", "I-VT 候选最少样本数", "默认 3", "ivt_saccade"),
            ("blink_min_duration_s", "闭合最短时长（秒）", "默认 0.050", "blink"),
            ("blink_max_duration_s", "闭合最长时长（秒）", "默认 0.500", "blink"),
            ("entropy_grid_width", "熵网格宽度（格）", "默认 32", "entropy"),
            ("entropy_grid_height", "熵网格高度（格）", "默认 18", "entropy"),
        )
        for index, (field, label, hint, group) in enumerate(parameter_rows):
            row = 2 + index * 2
            label_widget = ttk.Label(advanced, text=label, style="Card.TLabel", justify="left", wraplength=280)
            label_widget.grid(row=row, column=0, sticky="w", pady=(7, 0))
            entry = ttk.Entry(advanced, textvariable=self.analysis_config_vars[field], width=12)
            entry.grid(row=row, column=1, sticky="w", padx=(12, 8), pady=(7, 0))
            hint_label = ttk.Label(advanced, text=hint, style="Muted.TLabel", justify="left")
            hint_label.grid(row=row + 1, column=0, columnspan=4, sticky="ew", padx=(0, 8), pady=(2, 0))
            self._wrap_to_width(hint_label, advanced, inset=12)
            self._analysis_parameter_widgets[field] = (group, [entry, hint_label])
            self._input_controls.append(entry)
        self.analysis_config_vars["max_time_gap_s"].trace_add("write", lambda *_: self._refresh_analysis_parameter_states())
        self._refresh_analysis_parameter_states()
        footer = ttk.Label(page, text="成功的预处理结果会预填为本页的数据文件；仍需检查后手动启动分析。", style="Muted.TLabel")
        footer.grid(row=1, column=0, sticky="ew")
        self._wrap_to_width(footer, page, inset=8)

    def _refresh_analysis_parameter_states(self) -> None:
        selected = {key for key, variable in self.analysis_metric_vars.items() if variable.get()}
        timed = bool(selected.intersection({"trajectory", "fixation", "fast_movement", "ivt_saccade", "blink", "roi", "entropy"}))
        for field, (group, widgets) in self._analysis_parameter_widgets.items():
            enabled = timed if group == "timed" else timed and bool(selected) if group == "timed_multiplier" else group in selected
            if field == "gap_median_multiplier" and self.analysis_config_vars["max_time_gap_s"].get().strip():
                enabled = False
            state = "normal" if enabled else "disabled"
            for widget in widgets:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass

    def _build_guide_tab(self) -> None:
        page = ttk.Frame(self.notebook, padding=(18, 8, 18, 8))
        self.notebook.add(page, text="使用指南")
        self._guide_tab_id = str(page)
        self._page_ids.append(str(page))
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)
        card = ttk.Frame(page, style="OuterCard.TFrame", padding=(18, 10, 18, 10))
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(2, weight=1)
        ttk.Label(card, text="LocalGaze 使用指南", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        subtitle = ttk.Label(card, text="先看快速流程，再按步骤操作。", style="Muted.TLabel", justify="left")
        subtitle.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self._wrap_to_width(subtitle, card, inset=36)
        source_root = Path(__file__).resolve().parent
        guide_candidates = [source_root / "docs" / _QUICK_START_FILENAME, source_root / "README.md"]
        if getattr(sys, "frozen", False):
            executable_dir = Path(sys.executable).resolve().parent
            runtime_root = Path(getattr(sys, "_MEIPASS", source_root))
            guide_candidates = [executable_dir / "docs" / _QUICK_START_FILENAME,
                                runtime_root / "docs" / _QUICK_START_FILENAME, *guide_candidates]
        guide_path = next((candidate for candidate in guide_candidates if candidate.is_file()), guide_candidates[0])
        self.guide_path = guide_path
        self.guide_status = tk.StringVar(self.root, "正在读取快速开始…")
        self.guide_text = ScrolledText(
            card,
            height=10,
            wrap="word",
            font=("Microsoft YaHei UI", 10),
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
            text="打开快速开始",
            style="Secondary.TButton",
            command=lambda: self._open_guide(guide_path),
        )
        open_button.grid(row=0, column=1, sticky="e")

    def _environment_check_rows(self) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        runtime_version = ".".join(str(part) for part in sys.version_info[:3])
        runtime_status = "通过" if sys.version_info[:2] in {(3, 10), (3, 12)} else "不支持"
        runtime_kind = "便携运行时" if getattr(sys, "frozen", False) else "当前 Python 环境"
        rows.append(("Python", runtime_status, f"{runtime_kind} {runtime_version}；源码支持 Python 3.10/3.12 x64；解释器：{sys.executable}"))
        rows.append(("Tkinter 桌面界面", "通过", "工作台已启动；不会检测或改变系统显示设置。"))

        from desktop_environment import RUNTIME_DEPENDENCIES

        for label, module_name in RUNTIME_DEPENDENCIES:
            try:
                available = importlib.util.find_spec(module_name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                available = False
            detail = (
                f"模块 {module_name} 可定位；尚未实际导入，对应功能运行时仍会继续校验。"
                if available
                else f"未找到模块 {module_name}；对应功能可能不可用，请检查当前环境或完整便携包。"
            )
            rows.append((label, "通过" if available else "缺少", detail))

        guide_ok = self.guide_path.is_file()
        rows.append(("使用指南", "通过" if guide_ok else "缺少", str(self.guide_path)))
        materials_path = self._study_materials_path()
        try:
            materials_ok = materials_path.is_file() and materials_path.stat().st_size > 0
        except OSError:
            materials_ok = False
        rows.append(("实验设计资料", "通过" if materials_ok else "缺少", str(materials_path)))

        session = self.session_id.get().strip()
        if not session:
            rows.append(("会话 ID", "未配置", "校准或采集前填写；仅查看环境无需填写。"))
        else:
            try:
                canonical_session_id(session)
            except (TypeError, ValueError) as exc:
                rows.append(("会话 ID", "不可用", f"{exc}；请更换会话 ID 后再启动校准或采集。"))
            else:
                rows.append(("会话 ID", "通过", "名称可用于校准或采集输出。"))
        model_text = self.model_path.get().strip()
        if not model_text:
            rows.append(("模型路径", "未配置", "采集前选择已保存模型；首次使用可先运行校准。"))
        else:
            model_path = Path(model_text).expanduser()
            model_ok = model_path.exists() and not model_path.is_symlink()
            rows.append(("模型路径", "通过" if model_ok else "缺少", str(model_path) if model_ok else f"路径不存在或不可用：{model_path}"))
        output_text = self.output_root.get().strip()
        if not output_text:
            rows.append(("输出位置", "未配置", "开始校准、采集或分析前选择输出文件夹。"))
        else:
            output_path = Path(output_text).expanduser()
            if output_path.exists() and not output_path.is_dir():
                rows.append(("输出位置", "缺少", f"此路径已存在但不是文件夹：{output_path}"))
            elif output_path.is_dir():
                rows.append(("输出位置", "通过", f"文件夹已存在；实际写入权限由任务启动时验证：{output_path}"))
            else:
                rows.append(("输出位置", "待创建", f"任务启动时会尝试创建文件夹：{output_path}"))
        rows.append(("摄像头", "未验证", "检查不会打开摄像头；只有用户明确启动采集/校准后才能验证实际设备。"))
        return rows

    def save_local_configuration(self) -> None:
        if self._busy or self._dependency_install_active or self._closing:
            return
        initial_dir = self._configuration_path.parent if self._configuration_path else Path(__file__).resolve().parent
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存 LocalGaze 本地配置",
            initialdir=str(initial_dir),
            initialfile=self._configuration_path.name if self._configuration_path else "LocalGaze-config.json",
            defaultextension=".json",
            filetypes=(("LocalGaze 配置（JSON）", "*.json"),),
        )
        if not selected:
            return
        from desktop_configuration import save_configuration

        path = Path(selected).expanduser()
        try:
            save_configuration(path, self)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("无法保存配置", str(exc), parent=self.root)
            return
        self._configuration_path = path.resolve()
        self.configuration_status_text.set(f"已保存：{self._configuration_path.name}")
        self._append_log(f"已保存本地自定义配置：{self._configuration_path}")

    def load_local_configuration(self) -> None:
        if self._busy or self._dependency_install_active or self._closing:
            return
        initial_dir = self._configuration_path.parent if self._configuration_path else Path(__file__).resolve().parent
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="加载 LocalGaze 本地配置",
            initialdir=str(initial_dir),
            filetypes=(("LocalGaze 配置（JSON）", "*.json"),),
        )
        if not selected:
            return
        from desktop_configuration import load_configuration

        path = Path(selected).expanduser()
        try:
            load_configuration(path, self)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("无法加载配置", str(exc), parent=self.root)
            return
        self._configuration_path = path.resolve()
        self.configuration_status_text.set(f"已载入：{self._configuration_path.name}")
        self._append_log(f"已载入本地自定义配置：{self._configuration_path}；没有启动任何任务。")

    def show_environment_check(self) -> None:
        if self._environment_dialog is not None and self._environment_dialog.winfo_exists():
            self._environment_dialog.lift()
            return
        dialog = tk.Toplevel(self.root)
        self._environment_dialog = dialog
        dialog.title("环境与配置检查")
        dialog.transient(self.root)
        dialog.configure(background=_COLORS["canvas"])
        dialog.minsize(740, 520)
        dialog.geometry("980x740")
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(2, weight=1)
        ttk.Label(dialog, text="环境与配置检查", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", padx=18, pady=(16, 4)
        )
        summary = tk.StringVar(dialog)
        overview = ttk.Frame(dialog)
        overview.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))
        totals = ttk.Label(overview, textvariable=summary, font=("Microsoft YaHei UI", 13, "bold"))
        totals.pack(anchor="w", fill="x")
        self._wrap_to_width(totals, overview)
        legend = ttk.Frame(overview)
        legend.pack(anchor="w", pady=(6, 0))
        appearances = {
            "pass": ("✓", "#21834A", "#EAF5EE", "通过"),
            "fail": ("×", _COLORS["danger"], "#FBEDEC", "未通过"),
            "pending": ("○", "#788585", "#EDF1F0", "未检查 / 不适用"),
        }
        for symbol, color, _, label in appearances.values():
            tk.Label(legend, text=f"{symbol}  {label}", foreground=color,
                     background=_COLORS["canvas"], font=("Microsoft YaHei UI", 12)).pack(side="left", padx=(0, 20))

        body = ttk.Frame(dialog)
        body.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 10))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        canvas = tk.Canvas(body, background=_COLORS["canvas"], highlightthickness=0, yscrollincrement=24)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        items = ttk.Frame(canvas)
        items.columnconfigure(0, weight=1)
        window = canvas.create_window((0, 0), window=items, anchor="nw")
        items.bind("<Configure>", lambda _event: self._queue_scrollregion_refresh(canvas), add="+")
        canvas.bind("<Configure>", lambda event: self._resize_canvas_content(canvas, window, event.width), add="+")

        def scroll(event: tk.Event) -> str | None:
            delta = getattr(event, "delta", 0)
            number = getattr(event, "num", None)
            units = -1 if number == 4 else 1 if number == 5 else -int(delta / 120)
            if not units and delta:
                units = -1 if delta > 0 else 1
            if units:
                canvas.yview_scroll(units, "units")
                return "break"
            return None

        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            dialog.bind(sequence, scroll, add="+")

        def dispose(event: tk.Event) -> None:
            if event.widget is dialog:
                after_id = self._scrollregion_after.pop(canvas, None)
                if after_id:
                    self.root.after_cancel(after_id)
                self._canvas_widths.pop(canvas, None)
                if self._environment_dialog is dialog:
                    self._environment_dialog = None
                    self._environment_refresh = None
                    self._environment_install_button = None

        dialog.bind("<Destroy>", dispose, add="+")

        def refresh() -> None:
            rows = self._environment_check_rows()
            passed = sum(status == "通过" for _, status, _ in rows)
            failed = sum(status in {"缺少", "不可用"} for _, status, _ in rows)
            summary.set(f"共 {len(rows)} 项   ·   通过 {passed} 项   ·   未通过 {failed} 项   ·   待确认 {len(rows) - passed - failed} 项")
            position = canvas.yview()[0]
            for child in items.winfo_children():
                child.destroy()
            for index, (name, status, detail) in enumerate(rows):
                kind = "pass" if status == "通过" else "fail" if status in {"缺少", "不可用"} else "pending"
                symbol, color, pale, _ = appearances[kind]
                card = tk.Frame(items, background=_COLORS["surface"], highlightbackground=_COLORS["line"], highlightthickness=1)
                card.grid(row=index, column=0, sticky="ew", pady=(0, 7), padx=(0, 8))
                card.columnconfigure(1, weight=1)
                tk.Label(card, text=symbol, font=("Segoe UI Symbol", 22, "bold"), foreground=color,
                         background=pale, width=2).grid(row=0, column=0, rowspan=2, padx=(12, 14), pady=10)
                tk.Label(card, text=name, font=("Microsoft YaHei UI", 13, "bold"), foreground=_COLORS["ink"],
                         background=_COLORS["surface"], anchor="w").grid(row=0, column=1, sticky="ew", pady=(10, 2))
                tk.Label(card, text=status, font=("Microsoft YaHei UI", 11), foreground=color,
                         background=pale, padx=9, pady=3).grid(row=0, column=2, padx=12, pady=(10, 2), sticky="e")
                description = tk.Label(card, text=detail, font=("Microsoft YaHei UI", 11), foreground=_COLORS["muted"],
                                       background=_COLORS["surface"], anchor="w", justify="left")
                description.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 12), pady=(0, 10))
                self._wrap_to_width(description, description, minimum=1)
            canvas.yview_moveto(position)

        refresh()
        self._environment_refresh = refresh
        footer = ttk.Frame(dialog, padding=(18, 0, 18, 14))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        install_command, install_detail = self._dependency_install_target()
        self._environment_install_status.set(
            "正在安装项目 requirements.txt 中的依赖…"
            if self._dependency_install_active else install_detail
        )
        install_status = ttk.Label(footer, textvariable=self._environment_install_status, style="Muted.TLabel", justify="left")
        install_status.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 7))
        self._wrap_to_width(install_status, footer, minimum=1)
        note = ttk.Label(footer, text="检查不会打开摄像头。", justify="left")
        note.grid(row=1, column=0, sticky="ew", padx=(0, 12))
        self._wrap_to_width(note, note, minimum=1)
        self._environment_install_button = ttk.Button(
            footer, text="一键安装项目依赖", style="Primary.TButton",
            command=lambda: self._start_dependency_install(refresh),
        )
        self._environment_install_button.grid(row=1, column=1, padx=(0, 8))
        self._environment_install_button.configure(
            state="normal" if install_command is not None and not self._dependency_install_active else "disabled"
        )
        ttk.Button(footer, text="重新检查", style="Secondary.TButton", command=refresh).grid(row=1, column=2, padx=(0, 8))
        ttk.Button(footer, text="关闭", style="Primary.TButton", command=dialog.destroy).grid(row=1, column=3)

    def _dependency_install_target(self) -> tuple[tuple[str, ...] | None, str]:
        from desktop_environment import dependency_install_command

        return dependency_install_command(
            Path(__file__).resolve().parent,
            sys.prefix,
            sys.executable,
            frozen=bool(getattr(sys, "frozen", False)),
        )

    def _start_dependency_install(self, refresh: Callable[[], None]) -> None:
        if self._busy or self._dependency_install_active or self._closing:
            return
        command, detail = self._dependency_install_target()
        if command is None:
            self._environment_install_status.set(detail)
            if self._environment_install_button is not None:
                self._environment_install_button.configure(state="disabled")
            return
        try:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parent,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        except OSError as exc:
            self._environment_install_status.set(f"无法启动依赖安装：{exc}")
            self._append_log(f"启动项目依赖安装失败：{exc}")
            messagebox.showerror("无法安装项目依赖", str(exc), parent=self._environment_dialog or self.root)
            return

        self._dependency_process = process
        self._dependency_events = queue.Queue()
        self._dependency_install_active = True
        self._dependency_cancel_requested = False
        self._environment_refresh = refresh
        self._environment_install_status.set("正在安装项目 requirements.txt 中的依赖…")
        if self._environment_install_button is not None:
            self._environment_install_button.configure(state="disabled")
        self.status_text.set("正在安装项目依赖")
        self._append_log("启动项目依赖安装：" + subprocess.list2cmdline(command))
        self._append_log("目标仅为当前项目 .venv；正在接收 pip 输出。")
        self._set_busy(self._busy)

        def collect_output() -> None:
            if process.stdout is not None:
                with process.stdout:
                    for line in process.stdout:
                        self._dependency_events.put(("line", line.rstrip("\r\n")))
            self._dependency_events.put(("exit", process.wait()))

        threading.Thread(target=collect_output, name="LocalGazeDependencyLog", daemon=True).start()
        if self._poll_after_id is None:
            self._schedule_poll()

    def _poll_dependency_install(self) -> None:
        while True:
            try:
                event, value = self._dependency_events.get_nowait()
            except queue.Empty:
                break
            if event == "line":
                line = str(value)
                if line:
                    self._append_log("依赖安装：" + line)
                    if self._environment_dialog is not None and self._environment_dialog.winfo_exists():
                        self._environment_install_status.set(line)
                continue

            exit_code = int(value)
            cancelled = self._dependency_cancel_requested
            self._dependency_process = None
            self._dependency_install_active = False
            self._dependency_cancel_requested = False
            self._set_busy(self._busy)
            if cancelled:
                status = "依赖安装已取消；当前 Python 包可能只安装了一部分。"
                result = "已取消"
            elif exit_code == 0:
                importlib.invalidate_caches()
                status = "依赖安装成功；建议重新启动 LocalGaze 后继续使用。"
                result = "完成，退出码 0；建议重启软件"
            else:
                status = f"依赖安装失败，退出码 {exit_code}；请查看底部运行日志。"
                result = f"失败，退出码 {exit_code}"
            self._environment_install_status.set(status)
            self._append_log("项目依赖安装" + result + "。")
            if self._environment_install_button is not None and not self._closing:
                command, _detail = self._dependency_install_target()
                self._environment_install_button.configure(state="normal" if command is not None else "disabled")
            if self._environment_refresh is not None and not self._closing:
                self._environment_refresh()
            if not self._closing:
                self.status_text.set("依赖安装完成 · 建议重启软件" if exit_code == 0 else "依赖安装结束 · 查看运行日志")
            self._environment_refresh = None

    def _show_guide(self) -> None:
        try:
            self._select_page(self._page_ids.index(self._guide_tab_id))
        except ValueError:
            self.notebook.select(self._guide_tab_id)
        self._load_guide_text()
        self.guide_text.yview_moveto(0)
        self.guide_text.focus_set()

    def _load_guide_text(self) -> None:
        try:
            text = self.guide_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            text = f"无法读取使用指南。\n\n文件：{self.guide_path}\n\n原因：{exc}"
            self.guide_status.set("快速开始未能加载；请检查文件路径和 UTF-8 编码。")
        else:
            self.guide_status.set("快速开始已加载。")
        self.guide_text.configure(state="normal")
        self.guide_text.delete("1.0", "end")
        self.guide_text.insert("1.0", text)
        self.guide_text.configure(state="disabled")

    def _build_log_panel(self) -> None:
        self.root.rowconfigure(1, weight=0, minsize=62)
        panel = ttk.Frame(self.root, style="App.TFrame", padding=(14, 4, 14, 7))
        panel.grid(row=1, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        head = ttk.Frame(panel)
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(1, weight=1)
        self._log_expanded = False
        self.status_label = ttk.Label(head, textvariable=self.status_text, style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w", padx=(0, 9))
        self.capture_quality_label = ttk.Label(head, textvariable=self.capture_quality_text, style="PageHint.TLabel")
        self.capture_quality_label.grid(row=0, column=1, sticky="ew", padx=(0, 9))
        self.log_toggle = ttk.Button(head, text="任务记录 ▸", style="Quiet.TButton", command=self._toggle_log)
        self.log_toggle.grid(row=0, column=2, sticky="w")
        actions = ttk.Frame(panel)
        actions.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        actions.columnconfigure(0, weight=1)
        self.cancel_button = ttk.Button(actions, text="取消当前任务", style="Danger.TButton", command=self.cancel_job, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.outputs_menu = tk.Menu(self.root, tearoff=0)
        self.open_outputs_button = ttk.Menubutton(actions, text="打开本次输出 ▾", menu=self.outputs_menu, state="disabled")
        self.open_outputs_button.grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.job_logs_menu = tk.Menu(self.root, tearoff=0)
        self.open_job_logs_button = ttk.Menubutton(actions, text="持久日志 ▾", menu=self.job_logs_menu, state="disabled")
        self.open_job_logs_button.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self.clear_log_button = ttk.Button(actions, text="清空", style="Quiet.TButton", command=self.clear_log)
        self.clear_log_button.grid(row=0, column=4, sticky="e", padx=(8, 0))
        self.clear_log_button.grid_remove()
        self.log = ScrolledText(panel, height=4, wrap="word", background="#FFFFFF", foreground=_COLORS["ink"], insertbackground=_COLORS["ink"], relief="flat", borderwidth=1, highlightthickness=1, highlightbackground=_COLORS["line"], padx=10, pady=8, font=("Consolas", -17))
        self.log.grid(row=2, column=0, sticky="ew")
        self.log.grid_remove()
        self.log.configure(state="disabled")
        self._append_log("就绪。任务仅会在你点击对应操作后启动。")

    def _toggle_log(self) -> None:
        if self._log_expanded:
            self.log.grid_remove()
            self.clear_log_button.grid_remove()
            self.log_toggle.configure(text="任务记录 ▸")
            self._log_expanded = False
        else:
            self.log.grid(row=2, column=0, sticky="ew", pady=(4, 0))
            self.clear_log_button.grid()
            self.log_toggle.configure(text="任务记录 ▾")
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

    def _compact_entry_row(
        self,
        card: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        browse: str | None,
    ) -> tuple[ttk.Entry, ttk.Button | None]:
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text=label, style="Card.TLabel", width=8).grid(
            row=row, column=0, sticky="w", pady=(7, 0)
        )
        entry = ttk.Entry(card, textvariable=variable, width=24)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(7, 0))
        self._input_controls.append(entry)
        button = None
        if browse:
            button = ttk.Button(
                card,
                text="浏览…",
                style="Quiet.TButton",
                command=lambda b=browse, v=variable, l=label: self._browse(b, v, l),
            )
            button.grid(row=row, column=2, sticky="e", pady=(7, 0))
            self._browse_controls.append(button)
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
        elif kind == "mapping_table":
            selected = filedialog.askopenfilename(parent=self.root, title="选择预处理映射表", initialdir=initial, filetypes=(("映射表（CSV/XLSX）", "*.csv *.xlsx"), ("所有文件", "*.*")))
        elif kind == "model":
            choice = messagebox.askyesnocancel("选择模型路径", "选择“是”浏览模型文件；选择“否”浏览模型文件夹。", parent=self.root)
            if choice is True:
                selected = filedialog.askopenfilename(parent=self.root, title="选择模型文件", initialdir=initial, filetypes=(("模型文件", "*.pkl *.xlsx *.xlsm *.csv *.tsv *.txt"), ("所有文件", "*.*")))
            elif choice is False:
                selected = filedialog.askdirectory(parent=self.root, title="选择模型文件夹", initialdir=initial, mustexist=True)
        elif kind == "correction":
            selected = filedialog.askopenfilename(parent=self.root, title="选择修正 JSON", initialdir=initial, filetypes=(("修正 JSON", "*.json"), ("所有文件", "*.*")))
        elif kind == "snapshot":
            selected = filedialog.askopenfilename(parent=self.root, title="选择采集快照", initialdir=initial, filetypes=(("采集快照", "*.capture-snapshot.json"), ("所有文件", "*.*")))
        elif kind == "capture_context":
            selected = filedialog.askopenfilename(parent=self.root, title="选择采集恢复上下文", initialdir=initial, filetypes=(("采集上下文", "*.capture-context.json"), ("所有文件", "*.*")))
        elif kind == "roi":
            selected = filedialog.askopenfilename(parent=self.root, title="选择 ROI 工作簿", initialdir=initial, filetypes=(("Excel 工作簿", "*.xlsx *.xlsm"), ("所有文件", "*.*")))
        if selected:
            variable.set(selected)

    def _clear_preprocess_mapping_table(self) -> None:
        self.preprocess_mapping_table_path.set("")
        self._append_log("已清除预处理映射表；将使用上方直接文件/文件夹与可选模型设置。")

    def _preview_mapping_template(self) -> None:
        try:
            from pre.mapping_table import mapping_template_example_frame

            example = mapping_template_example_frame()
            columns = [str(column) for column in example.columns]
        except Exception as exc:
            messagebox.showerror("无法预览映射模板", str(exc), parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("映射表模板示例（仅预览）")
        window.transient(self.root)
        window.geometry("820x340")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        ttk.Label(
            window,
            text="以下示例路径仅用于说明，不会作为任务输入。请保存空白模板后填写真实路径。",
            justify="left",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        tree = ttk.Treeview(window, columns=columns, show="headings", height=8)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=235, stretch=True)
        for row in example.itertuples(index=False, name=None):
            tree.insert("", "end", values=["" if value is None else str(value) for value in row])
        tree.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 12))
        tree.configure(yscrollcommand=scrollbar.set)

    def _save_mapping_template(self) -> None:
        target = filedialog.asksaveasfilename(
            parent=self.root,
            title="保存空白映射表模板",
            initialdir=self.output_root.get().strip() or str(Path.cwd()),
            defaultextension=".xlsx",
            filetypes=(("Excel 映射表", "*.xlsx"), ("CSV 映射表", "*.csv")),
        )
        if not target:
            return
        try:
            from pre.mapping_table import mapping_template_frame, save_mapping_template

            frame = mapping_template_frame()
            if list(frame.columns) != ["data_path", "model_path", "session_id"] or not frame.empty:
                raise ValueError("空白模板必须只包含 data_path、model_path、session_id 三列表头。")
            saved = save_mapping_template(target)
        except Exception as exc:
            messagebox.showerror("无法保存映射模板", str(exc), parent=self.root)
            return
        path = str(Path(saved).resolve())
        self._append_log(f"已保存空白映射表模板（不会覆盖已有文件）：{path}")
        messagebox.showinfo("模板已保存", f"空白输入模板已保存：\n{path}\n\n示例页仅供预览，不会作为任务输入。", parent=self.root)

    def _validated_preprocess_mapping_table(self) -> tuple[Path | None, list[dict[str, object]]]:
        text = self.preprocess_mapping_table_path.get().strip()
        if not text:
            return None, []
        path = Path(text).expanduser()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"映射表不存在或不是普通文件：{path}")
        if path.suffix.lower() not in {".csv", ".xlsx"}:
            raise ValueError("映射表只支持 CSV 或 XLSX 文件。")
        from pre.mapping_table import read_mapping_table

        rows = read_mapping_table(path)
        if not rows:
            raise ValueError("映射表没有可处理的数据行。")
        row_errors = [row for row in rows if row.get("error")]
        valid_rows = [row for row in rows if not row.get("error")]
        if not valid_rows:
            raise ValueError("映射表中的数据行均无效；请先修正后再启动。")
        if row_errors:
            self._append_log(f"映射表预检：{len(row_errors)} 行有错误；这些行会在批次报告中单独记录，其余 {len(valid_rows)} 行继续处理。")
            for row in row_errors[:10]:
                self._append_log(f"  第 {row.get('row_number', '?')} 行：{row.get('error')}")
            if len(row_errors) > 10:
                self._append_log(f"  另有 {len(row_errors) - 10} 行错误，详见批次报告。")
        return path.resolve(strict=True), rows

    def _payload(self, kind: str) -> dict[str, object]:
        if kind == "capture_recovery":
            context_text = self.capture_context_path.get().strip()
            if not context_text:
                raise ValueError("请先选择一个采集恢复上下文文件。")
            context = Path(context_text).expanduser()
            if context.is_symlink() or not context.is_file() or not context.name.endswith(".capture-context.json"):
                raise ValueError("请选择一个存在且名称以 .capture-context.json 结尾的恢复上下文。")
            return {"context_path": str(context.resolve(strict=True))}
        if kind == "capture_export_retry":
            snapshot_text = self.capture_snapshot_path.get().strip()
            if not snapshot_text:
                raise ValueError("请先选择一个采集快照文件。")
            snapshot = Path(snapshot_text).expanduser()
            if snapshot.is_symlink() or not snapshot.is_file() or not snapshot.name.endswith(".capture-snapshot.json"):
                raise ValueError("请选择一个存在且名称以 .capture-snapshot.json 结尾的采集快照文件。")
            return {"snapshot_path": str(snapshot.resolve(strict=True))}

        # ROI workbooks are named in the drawing tool and do not belong to a
        # capture session. Ignore the capture page's session field entirely.
        # Direct preprocessing has no page-level session contract, and a
        # selected mapping table owns its row sessions. Never require or leak
        # the unrelated global session field for either route.
        session_text = self.session_id.get().strip()
        session = (
            None
            if kind in {"roi", "preprocess"} or (kind == "analyze" and not session_text)
            else canonical_session_id(session_text)
        )
        output_text = self.output_root.get().strip()
        if not output_text:
            raise ValueError("请先设置输出位置。")
        output = Path(output_text).expanduser()
        if output.exists() and not output.is_dir():
            raise ValueError("输出位置必须是文件夹。")
        payload: dict[str, object] = {"output_root": str(output.resolve())}
        if session is not None:
            payload["session_id"] = session

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
        if kind == "capture":
            payload["motion_monitor_enabled"] = bool(self.motion_monitor_enabled.get())
            payload["save_motion_keyframes"] = bool(self.save_motion_keyframes.get())
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
            from desktop_configuration import normalize_preprocess_parameters

            payload["preprocess_config"] = normalize_preprocess_parameters(
                {name: variable.get() for name, variable in self.preprocess_config_vars.items()}
            )
        elif kind == "analyze":
            unit_label = self.analysis_time_unit_label.get()
            if unit_label not in _TIME_UNITS:
                raise ValueError("请选择有效的分析时间单位。")
            payload["time_unit"] = _TIME_UNITS[unit_label]
            format_label = self.analysis_table_format_label.get()
            format_values = {"CSV（逗号分隔）": "csv", "TXT（制表符分隔）": "txt"}
            if format_label not in format_values:
                raise ValueError("请选择 CSV 或 TXT 表格格式。")
            payload["table_format"] = normalize_table_format(format_values[format_label])
            payload["recursive"] = bool(self.analysis_recursive.get())
            payload["selected_metrics"] = [
                group_id for group_id, variable in self.analysis_metric_vars.items() if variable.get()
            ]
            try:
                config = {
                    "max_time_gap_s": None if not self.analysis_config_vars["max_time_gap_s"].get().strip() else float(self.analysis_config_vars["max_time_gap_s"].get()),
                    "gap_median_multiplier": float(self.analysis_config_vars["gap_median_multiplier"].get()),
                    "fixation_min_samples": int(self.analysis_config_vars["fixation_min_samples"].get()),
                    "fixation_min_duration_s": float(self.analysis_config_vars["fixation_min_duration_s"].get()),
                    "fixation_dispersion_diagonal_ratio": float(self.analysis_config_vars["fixation_dispersion_diagonal_ratio"].get()),
                    "fast_speed_diagonals_s": float(self.analysis_config_vars["fast_speed_diagonals_s"].get()),
                    "ivt_speed_diagonals_s": float(self.analysis_config_vars["ivt_speed_diagonals_s"].get()),
                    "ivt_min_duration_s": float(self.analysis_config_vars["ivt_min_duration_s"].get()),
                    "ivt_max_duration_s": float(self.analysis_config_vars["ivt_max_duration_s"].get()),
                    "ivt_min_samples": int(self.analysis_config_vars["ivt_min_samples"].get()),
                    "blink_min_duration_s": float(self.analysis_config_vars["blink_min_duration_s"].get()),
                    "blink_max_duration_s": float(self.analysis_config_vars["blink_max_duration_s"].get()),
                    "entropy_grid": (
                        int(self.analysis_config_vars["entropy_grid_width"].get()),
                        int(self.analysis_config_vars["entropy_grid_height"].get()),
                    ),
                }
                from EyeEvaluate.Evaluate.metrics import AnalysisConfig

                payload["analysis_config"] = vars(AnalysisConfig(**config))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"分析参数无效：{exc}") from exc
        elif kind in {"capture", "independent_validation"}:
            format_values = {"CSV（逗号分隔）": "csv", "TXT（制表符分隔）": "txt"}
            format_label = self.capture_table_format_label.get()
            if format_label not in format_values:
                raise ValueError("请选择 CSV 或 TXT 表格格式。")
            payload["table_format"] = normalize_table_format(format_values[format_label])

        if kind in {"calibrate", "capture", "known_target_offset", "independent_validation", "analyze"}:
            if kind in {"capture", "known_target_offset", "independent_validation"}:
                model_text = self.model_path.get().strip()
                if not model_text:
                    raise ValueError("请先设置模型路径。")
                model = Path(model_text).expanduser()
                if not model.exists():
                    raise FileNotFoundError(f"模型路径不存在：{model}")
                if kind in {"capture", "known_target_offset", "independent_validation"} and (not model.is_file() or model.suffix.lower() != ".pkl"):
                    raise ValueError("采集、已知目标修正和独立验证需要选择一个已存在的 .pkl 模型文件。")
                payload["model_path"] = str(model.resolve())
            if kind == "analyze":
                if self._analysis_selected_files:
                    paths = []
                    for selected_path in self._analysis_selected_files:
                        source = Path(selected_path).expanduser()
                        if source.is_symlink() or not source.is_file():
                            raise FileNotFoundError(f"眼动数据文件不存在或不可用：{source}")
                        if source.suffix.lower() not in {".xlsx", ".xlsm", ".csv", ".tsv", ".txt"}:
                            raise ValueError(f"不支持的眼动数据文件格式：{source.suffix}")
                        paths.append(str(source.resolve(strict=True)))
                    payload["source_paths"] = paths
                else:
                    source_text = self.analysis_source_path.get().strip()
                    if not source_text:
                        raise ValueError("请先设置数据文件/文件夹，或选择多个数据文件。")
                    source = Path(source_text).expanduser()
                    if source.is_symlink() or not source.exists() or not (source.is_file() or source.is_dir()):
                        raise FileNotFoundError(f"数据来源不存在或不可用：{source}")
                    if source.is_file() and source.suffix.lower() not in {".xlsx", ".xlsm", ".csv", ".tsv", ".txt"}:
                        raise ValueError(f"不支持的眼动数据文件格式：{source.suffix}")
                    payload["source_path"] = str(source.resolve(strict=True))
            if kind == "analyze":
                roi_text = self.roi_path.get().strip()
                roi_selected = "roi" in payload["selected_metrics"]
                if roi_selected and roi_text:
                    roi = Path(roi_text).expanduser()
                    if not roi.is_file():
                        raise FileNotFoundError(f"ROI 工作簿不存在：{roi}")
                    payload["roi_path"] = str(roi.resolve())
        if kind == "preprocess":
            mapping_path, _ = self._validated_preprocess_mapping_table()
            if mapping_path is not None:
                # A selected table is authoritative. Do not leak page-level
                # data, model, or session selections into its rows.
                payload.pop("session_id", None)
                payload["mapping_table_path"] = str(mapping_path)
                self._append_log("本次预处理使用映射表；上方直接数据/模型/会话设置不会继承到表格行。")
            else:
                model_text = self.model_path.get().strip()
                if model_text:
                    model = Path(model_text).expanduser()
                    if model.is_symlink() or not model.exists() or not (model.is_file() or model.is_dir()):
                        raise FileNotFoundError(f"模型路径不存在或不可用：{model}")
                    payload["model_path"] = str(model.resolve(strict=True))
                else:
                    payload["model_path"] = None
                source_text = self.preprocess_source_path.get().strip()
                if not source_text:
                    raise ValueError("请先设置眼动数据文件/文件夹，或选择映射表。")
                source = Path(source_text).expanduser()
                if source.is_symlink() or not source.exists() or not (source.is_file() or source.is_dir()):
                    raise FileNotFoundError(f"数据来源不存在或不可用：{source}")
                payload["source_path"] = str(source.resolve(strict=True))
        return payload

    def start_calibration(self) -> None:
        self._start_job("calibrate", "校准并保存模型")

    def start_capture(self) -> None:
        self._start_job("capture", "使用已有模型采集")

    def start_independent_validation(self) -> None:
        self._start_job("independent_validation", "锁定模型九点独立验证")

    def start_capture_export_retry(self) -> None:
        self._start_job("capture_export_retry", "从采集快照重试导出")

    def start_capture_recovery(self) -> None:
        self._start_job("capture_recovery", "从采集上下文恢复结果")

    def start_known_target_offset(self) -> None:
        self._start_job("known_target_offset", "采集已知目标并估计修正")

    def start_roi(self) -> None:
        self._start_job("roi", "打开 ROI 绘制工具")

    def start_preprocess(self) -> None:
        self._start_job("preprocess", "预处理与 QC")

    def start_analysis(self) -> None:
        self._start_job("analyze", "分析与导出")

    def _start_job(self, kind: str, display_name: str) -> None:
        if self._closing or self._busy or self._dependency_install_active:
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
        self._clear_last_job_logs()
        self._set_busy(True)
        self.status_text.set(f"运行中 · {display_name}")
        self.capture_quality_text.set("采集质量 · 等待采集数据" if kind in {"capture", "calibrate"} else "采集质量 · 当前任务无质量流")
        session = payload.get("session_id")
        suffix = f"（会话 {session}）" if session else "（不启动相机）" if kind in {"capture_export_retry", "capture_recovery"} else ""
        self._append_log(f"已启动：{display_name}{suffix}")

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
        locked = busy or self._closing or self._dependency_install_active
        for widget in (*self._input_controls, *self._browse_controls, *self._action_controls):
            try:
                widget.configure(state="disabled" if locked else "normal")
            except tk.TclError:
                pass
        try:
            self.cancel_button.configure(state="normal" if busy and not self._closing else "disabled")
        except (AttributeError, tk.TclError):
            pass
        if not busy and not self._closing and not self._dependency_install_active:
            self._refresh_analysis_parameter_states()
        self._sync_capture_retry_button()
        install_button = self._environment_install_button
        if install_button is not None:
            command, _detail = self._dependency_install_target()
            install_button.configure(
                state="normal" if command is not None and not busy and not self._closing and not self._dependency_install_active else "disabled"
            )

    def _sync_capture_retry_button(self) -> None:
        button = getattr(self, "capture_export_retry_button", None)
        if button is None:
            return
        snapshot = Path(self.capture_snapshot_path.get().strip()).expanduser()
        enabled = (
            not self._busy
            and not self._dependency_install_active
            and not self._closing
            and snapshot.name.endswith(".capture-snapshot.json")
            and snapshot.is_file()
            and not snapshot.is_symlink()
        )
        try:
            button.configure(state="normal" if enabled else "disabled")
        except tk.TclError:
            pass
        context_button = getattr(self, "capture_recovery_button", None)
        if context_button is not None:
            context = Path(self.capture_context_path.get().strip()).expanduser()
            context_enabled = (
                not self._busy
                and not self._dependency_install_active
                and not self._closing
                and context.name.endswith(".capture-context.json")
                and context.is_file()
                and not context.is_symlink()
            )
            try:
                context_button.configure(state="normal" if context_enabled else "disabled")
            except tk.TclError:
                pass

    def cancel_job(self) -> None:
        if self._manager is not None and self._busy:
            if self._manager.cancel():
                self.status_text.set("正在取消 · 等待当前任务退出")
                self._append_log("已请求取消；正在等待任务进程退出。")
                self.cancel_button.configure(state="disabled")

    def _schedule_poll(self) -> None:
        if self._closing and (self._manager is None or not self._manager.busy) and not self._dependency_install_active:
            return
        if self._poll_after_id is None:
            self._poll_after_id = self.root.after(250, self._poll_jobs)

    def _poll_jobs(self) -> None:
        self._poll_after_id = None
        manager = self._manager
        if manager is not None:
            for event in manager.poll():
                if event.get("job_id") != self._active_job_id:
                    continue
                self._handle_event(event)
        self._poll_dependency_install()
        if self._closing:
            if (manager is None or manager.closed or not manager.busy) and not self._dependency_install_active:
                self.root.destroy()
                return
            self._schedule_poll()
        elif self._auto_poll:
            self._schedule_poll()
        elif (manager is not None and manager.busy) or self._dependency_install_active:
            self._schedule_poll()

    def _handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        message = str(event.get("message", ""))
        if event_type == "progress":
            details = event.get("details")
            quality = details.get("quality") if isinstance(details, dict) else None
            if isinstance(quality, dict):
                def value_text(key: str, suffix: str = "") -> str:
                    value = quality.get(key)
                    if value is None:
                        return "NA"
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        return f"{float(value):.2f}{suffix}"
                    return f"{value}{suffix}"

                def ratio_text(key: str) -> str:
                    value = quality.get(key)
                    return "NA" if value is None else f"{float(value):.0%}"

                head = quality.get("head_state", {})
                body = quality.get("body_state", {})
                text = (
                    "采集质量 · "
                    f"读帧率 {value_text('read_frame_rate_hz', ' 帧/s')} · "
                    f"有效帧率 {value_text('effective_sample_rate_hz', ' 个/s')} · "
                    f"有效/无效比例 {ratio_text('valid_ratio')}/{ratio_text('invalid_ratio')} · "
                    f"warmup {quality.get('warmup_count', 0)} · "
                    "连续无有效样本 "
                    f"{value_text('continuous_no_valid_sample_seconds', ' s')} · "
                    f"Head {head.get('status', 'NA') if isinstance(head, dict) else 'NA'} · "
                    f"Body {body.get('status', 'NA') if isinstance(body, dict) else 'NA'}"
                )
                if quality.get("tracking_lost_seconds") is not None:
                    text += f" · tracking_lost {value_text('tracking_lost_seconds', ' s')}"
                self.capture_quality_text.set(text)
            return
        if event_type == "log":
            self._append_log(message)
            return
        if event_type == "log_error":
            self.status_text.set("持久日志写入异常 · 查看任务记录")
            self._append_log(message or "持久日志写入失败；任务仍会继续运行。")
            return
        if event_type == "started":
            details = event.get("details")
            if isinstance(details, dict):
                worker_log = details.get("worker_log_path")
                capture_log = details.get("capture_task_log_path")
                if worker_log:
                    self._append_log(f"工作进程持久日志：{worker_log}")
                if capture_log:
                    self._append_log(f"采集任务持久日志：{capture_log}")
                for error in details.get("log_io_errors", []) if isinstance(details.get("log_io_errors"), list) else []:
                    self._append_log(f"日志状态错误：{error}")
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
            if isinstance(details, dict):
                status_counts = details.get("status_counts")
                if isinstance(status_counts, dict) and status_counts:
                    counts_text = "，".join(f"{key}={value}" for key, value in sorted(status_counts.items()))
                    self._append_log(f"逐项处理结果：{counts_text}")
                if details.get("partial_outputs"):
                    self._append_log("本批次含失败或跳过条目；已完成文件已保留，详见逐行错误和运行报告。")
                row_errors = details.get("row_errors")
                if isinstance(row_errors, list):
                    for row_error in row_errors[:20]:
                        if isinstance(row_error, dict):
                            row_number = row_error.get("row_number", "?")
                            reason = row_error.get("error") or row_error.get("reason") or "处理失败"
                            self._append_log(f"映射表第 {row_number} 行：{reason}")
                    if len(row_errors) > 20:
                        self._append_log(f"另有 {len(row_errors) - 20} 行错误，详见运行报告。")
                self._remember_job_logs(details)
                for key, label in (
                    ("worker_log_path", "工作进程持久日志"),
                    ("capture_task_log_path", "采集任务持久日志"),
                    ("source_capture_task_log_path", "原采集任务日志"),
                ):
                    value = details.get(key)
                    if value:
                        self._append_log(f"{label}：{value}")
                log_status = details.get("log_status")
                if log_status and log_status != "ok":
                    self._append_log(f"持久日志状态：{log_status}")
                for error in details.get("log_io_errors", []) if isinstance(details.get("log_io_errors"), list) else []:
                    self._append_log(f"日志写入错误：{error}")
                if details.get("traceback"):
                    worker_log = details.get("worker_log_path")
                    traceback_location = f"：{worker_log}" if worker_log else "中"
                    self._append_log(f"Python异常的完整 traceback 已保存在工作进程日志{traceback_location}。")
                exitcode = details.get("worker_exitcode")
                if exitcode is not None:
                    self._append_log(f"工作进程 exitcode：{exitcode}")
                checkpoint = details.get("last_durable_checkpoint")
                if isinstance(checkpoint, dict):
                    summary = ", ".join(
                        f"{label}={checkpoint[field]}"
                        for field, label in (("sequence", "序号"), ("stage", "阶段"), ("last_frame_id", "末帧"), ("complete_records", "完整记录"), ("byte_offset", "字节位置"))
                        if checkpoint.get(field) is not None
                    )
                    self._append_log(f"最后确认的持久检查点：{summary or checkpoint}")
                elif details.get("checkpoint_error"):
                    self._append_log(f"无法读取最后的持久检查点：{details['checkpoint_error']}")
                for key in ("retryable", "reason", "snapshot_path", "retained_files"):
                    if key not in details:
                        continue
                    value = details.get(key)
                    if value is not None:
                        self._append_log(f"{key}：{value}")
                snapshot_value = details.get("snapshot_path")
                if details.get("retryable") and isinstance(snapshot_value, (str, os.PathLike)):
                    snapshot_candidate = Path(snapshot_value).expanduser()
                    if snapshot_candidate.is_file() and snapshot_candidate.name.endswith(".capture-snapshot.json"):
                        self.capture_snapshot_path.set(str(snapshot_candidate.resolve()))
                context_value = details.get("context_path") or details.get("capture_context_path")
                if details.get("retryable") and isinstance(context_value, (str, os.PathLike)):
                    context_candidate = Path(context_value).expanduser()
                    if context_candidate.is_file() and context_candidate.name.endswith(".capture-context.json"):
                        self.capture_context_path.set(str(context_candidate.resolve()))
            if event_type == "failed" and isinstance(details, dict):
                reason = details.get("reason") or details.get("error")
                if reason and "reason" not in details:
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

    def _clear_last_job_logs(self) -> None:
        self._last_job_logs = {}
        if hasattr(self, "job_logs_menu"):
            self.job_logs_menu.delete(0, "end")
        if hasattr(self, "open_job_logs_button"):
            self.open_job_logs_button.configure(state="disabled")

    def _remember_job_logs(self, details: dict[str, object]) -> None:
        candidates = (
            ("工作进程日志", "worker_log_path"),
            ("采集任务日志", "capture_task_log_path"),
            ("原采集任务日志", "source_capture_task_log_path"),
        )
        paths: dict[str, Path] = {}
        seen: set[Path] = set()
        for label, key in candidates:
            raw = details.get(key)
            if not isinstance(raw, (str, os.PathLike)):
                continue
            try:
                path = Path(raw).expanduser().resolve(strict=True)
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            if path.is_file() and path not in seen:
                paths[label] = path
                seen.add(path)
        self._last_job_logs = paths
        self.job_logs_menu.delete(0, "end")
        for label, path in paths.items():
            self.job_logs_menu.add_command(
                label=f"{label}：{path.name}",
                command=lambda selected_label=label: self._open_job_log(selected_label),
            )
        self.open_job_logs_button.configure(state="normal" if paths else "disabled")

    def _open_job_log(self, label: str) -> None:
        path = self._last_job_logs.get(label)
        if path is None:
            return
        try:
            current_path = path.resolve(strict=True)
            if not current_path.is_file():
                raise FileNotFoundError(current_path)
            self._open_path_with_os(current_path)
        except OSError as exc:
            messagebox.showerror("无法打开持久日志", f"{path}\n\n{exc}", parent=self.root)
            self._append_log(f"无法打开持久日志：{path}；{exc}")

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
        self._unbind_mousewheel()
        for canvas, after_id in tuple(self._scrollregion_after.items()):
            try:
                canvas.after_cancel(after_id)
            except tk.TclError:
                pass
        self._scrollregion_after.clear()
        manager = self._manager
        if (manager is None or not manager.busy) and not self._dependency_install_active:
            self.root.destroy()
            return
        self._closing = True
        self._set_busy(True)
        self.status_text.set("正在安全关闭 · 等待任务和依赖安装进程退出")
        if manager is not None and manager.busy:
            self._append_log("窗口关闭时已请求停止当前任务；将在所属进程退出后关闭窗口。")
            manager.shutdown()
        process = self._dependency_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                self._append_log("依赖安装进程已退出；正在读取最终日志和退出状态。")
            except OSError as exc:
                self._append_log(f"无法停止本程序启动的依赖安装进程：{exc}；窗口将继续等待它退出。")
            else:
                self._dependency_cancel_requested = True
                self._append_log("窗口关闭时已请求停止本程序启动的依赖安装；正在等待该进程退出。")
        if self._poll_after_id is None:
            self._schedule_poll()

    def _update_tab_state(self, _event=None) -> None:
        try:
            index = self.notebook.index(self.notebook.select())
        except (tk.TclError, ValueError):
            index = 0
        if 0 <= index < len(self._page_info):
            self._active_page_index = index
            number, title, hint = self._page_info[index]
            self.page_number_label.configure(text=number)
            self.page_title_label.configure(text=title)
            self.page_hint_label.configure(text=hint)
            for position, button in enumerate(self._page_buttons):
                button.configure(style="NavActive.TButton" if position == index else "Nav.TButton")
        # Do not let page changes re-enable a control while a task is active.
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
        self._open_guide_file(path, "快速开始")

    @staticmethod
    def _study_materials_path() -> Path:
        source_root = Path(__file__).resolve().parent
        candidates = []
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).resolve().parent / "docs" / "skills" / _SKILL_GUIDE_FILENAME)
            runtime_root = Path(getattr(sys, "_MEIPASS", source_root))
            candidates.append(runtime_root / "docs" / "skills" / _SKILL_GUIDE_FILENAME)
        candidates.append(source_root / "docs" / "skills" / _SKILL_GUIDE_FILENAME)
        return next((path for path in candidates if path.is_file()), candidates[0])

    def _open_study_materials(self) -> None:
        self._open_guide_file(self._study_materials_path(), "实验设计资料")

    def _open_guide_file(self, path: Path, label: str) -> None:
        if not path.is_file():
            messagebox.showerror(f"找不到{label}", f"未找到说明文件：\n{path}", parent=self.root)
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                messagebox.showinfo("使用指南", f"请打开此文件查看完整说明：\n{path}", parent=self.root)
        except OSError as exc:
            messagebox.showerror(f"无法打开{label}", str(exc), parent=self.root)


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
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                ctypes.c_wchar_p("LocalGaze.Workbench")
            )
        except (AttributeError, OSError):
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


__all__ = ["DesktopApp", "main"]
