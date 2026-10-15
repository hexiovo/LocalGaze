"""Tk application for drawing and saving versioned screen-coordinate ROIs."""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import pandas as pd

if __package__:
    from .ClassType import (
        CanvasMapping,
        DraggableHandle,
        ROI,
        ROIValidationError,
        ScreenInfo,
        enumerate_screens,
        geometry_from_bounds,
    )
    from .Global_data import TRANSPARENCY
else:
    from ClassType import (
        CanvasMapping,
        DraggableHandle,
        ROI,
        ROIValidationError,
        ScreenInfo,
        enumerate_screens,
        geometry_from_bounds,
    )
    from Global_data import TRANSPARENCY

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from roi_contract import (
    ROI_METADATA_COLUMNS,
    convert_strict_rectangle_to_inclusive,
    serialize_roi_workbook,
    validate_coordinate_reference_compatibility,
    validate_roi_reference_compatibility,
    validate_rois,
)


class ROIApp:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.config(bg="gray25")
        self.root.geometry("520x44+0+0")

        self.root.update_idletasks()
        self.screens = enumerate_screens(
            self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        )
        self.selected_screen = next(
            (screen for screen in self.screens if screen.is_primary), self.screens[0]
        )
        self.coordinate_reference = self._reference_for_screen(self.selected_screen)

        self.toolbar = tk.Frame(root, bg="gray25")
        self.toolbar.pack(padx=4, pady=4)
        self._add_button("圆形", lambda: self.open_overlay("circle"))
        self._add_button("矩形", lambda: self.open_overlay("rectangle"))
        self._add_button("椭圆", lambda: self.open_overlay("ellipse"))
        self._add_button("屏幕/坐标", self.edit_coordinate_reference)
        self._add_button("ROI管理", self.manage_rois)
        self._add_button("保存", self.save_all_prompt)
        self._add_button("退出", self.root.destroy)

        self.toolbar.bind("<Button-1>", self._start_move)
        self.toolbar.bind("<B1-Motion>", self._do_move)
        self._drag_start_x = self._drag_start_y = 0

        self.overlay = None
        self.button_window = None
        self.canvas = None
        self.current_shape_id = None
        self.temp_center_id = None
        self.handles = []
        self.center_handle = None
        self.roi_list: list[ROI] = []
        self._canvas_start = (0.0, 0.0)
        self._canvas_bounds: list[float] | None = None
        self._active_canvas_size: tuple[int, int] | None = None

    def _add_button(self, text, command):
        tk.Button(self.toolbar, text=text, command=command).pack(side="left", padx=2)

    def _start_move(self, event):
        self._drag_start_x, self._drag_start_y = event.x, event.y

    def _do_move(self, event):
        x = self.root.winfo_x() + event.x - self._drag_start_x
        y = self.root.winfo_y() + event.y - self._drag_start_y
        self.root.geometry(f"+{x}+{y}")

    def _ordered_rois(self) -> list[ROI]:
        return [
            roi
            for _, roi in sorted(
                enumerate(self.roi_list), key=lambda pair: (pair[1].priority, pair[0])
            )
        ]

    def manage_rois(self):
        """Show saved ROIs and provide explicit import/order/boundary controls."""
        dialog = tk.Toplevel(self.root)
        dialog.title("ROI 列表与优先级")
        dialog.transient(self.root)
        dialog.geometry("720x380")

        columns = ("priority", "name", "shape", "boundary", "geometry", "reference")
        tree = ttk.Treeview(dialog, columns=columns, show="headings", selectmode="browse")
        headings = {
            "priority": "优先级（小者优先）",
            "name": "名称",
            "shape": "形状",
            "boundary": "边界",
            "geometry": "几何参数",
            "reference": "坐标参考",
        }
        widths = {"priority": 125, "name": 125, "shape": 75, "boundary": 75, "geometry": 190, "reference": 120}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], stretch=column in {"geometry", "reference"})
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        def refresh(selected_id: str | None = None):
            for item in tree.get_children():
                tree.delete(item)
            for roi in self._ordered_rois():
                reference = roi.coordinate_reference
                reference_label = reference.get("monitor_id") or reference.get("space", "unknown")
                tree.insert(
                    "",
                    "end",
                    iid=roi.id,
                    values=(
                        roi.priority,
                        roi.name,
                        roi.shape,
                        roi.boundary_mode,
                        str(roi.geometry),
                        reference_label,
                    ),
                )
            if selected_id and tree.exists(selected_id):
                tree.selection_set(selected_id)
                tree.focus(selected_id)

        def move_selected(offset: int):
            selection = tree.selection()
            if not selection:
                return
            ordered = self._ordered_rois()
            current = next((i for i, roi in enumerate(ordered) if roi.id == selection[0]), None)
            if current is None:
                return
            target = current + offset
            if not 0 <= target < len(ordered):
                return
            ordered[current], ordered[target] = ordered[target], ordered[current]
            self.roi_list = [replace(roi, priority=index) for index, roi in enumerate(ordered)]
            refresh(selection[0])

        def import_workbook():
            path = filedialog.askopenfilename(
                parent=dialog,
                title="导入旧版或版本化 ROI 工作簿",
                filetypes=(("Excel 工作簿", "*.xlsx"),),
            )
            if not path:
                return
            try:
                # The reader is imported only when requested and does not load
                # the EyeEvaluate GUI modules for workbook-only migration.
                from EyeEvaluate.FileOperate.ROIextract import extract_ROI

                records = extract_ROI(path)
                imported = [
                    record.get("_roi") if isinstance(record.get("_roi"), ROI) else ROI.from_dict(record)
                    for record in records
                ]
                current = self._ordered_rois()
                validate_rois([*current, *imported])
                validate_roi_reference_compatibility([*current, *imported])
                imported.sort(key=lambda roi: roi.priority)
                next_priority = max((roi.priority for roi in current), default=-1) + 1
                self.roi_list = [
                    *current,
                    *(
                        replace(roi, priority=next_priority + index)
                        for index, roi in enumerate(imported)
                    ),
                ]
            except Exception as exc:
                messagebox.showerror("导入失败", f"ROI 工作簿无法导入：\n{exc}\n不同已知坐标参考请分别建立 ROI 集合。", parent=dialog)
                return
            refresh(imported[0].id if imported else None)
            messagebox.showinfo("导入完成", f"已导入 {len(imported)} 个 ROI；旧工作簿的边界语义和未知坐标参考均予以保留。", parent=dialog)

        def convert_selected_boundary():
            selection = tree.selection()
            if not selection:
                return
            selected_id = selection[0]
            roi = next((item for item in self.roi_list if item.id == selected_id), None)
            if roi is None:
                return
            if roi.boundary_mode != "strict":
                messagebox.showinfo("边界未更改", "只有严格边界的矩形可以转换为包含边界。", parent=dialog)
                return
            if not messagebox.askyesno(
                "确认边界转换",
                f"将 {roi.name} 的判断条件从严格边界改为包含边界，并保留几何边缘。是否继续？",
                parent=dialog,
            ):
                return
            converted = convert_strict_rectangle_to_inclusive(roi)
            self.roi_list = [converted if item.id == selected_id else item for item in self.roi_list]
            refresh(selected_id)

        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="上移优先级", command=lambda: move_selected(-1)).pack(side="left", padx=3)
        ttk.Button(buttons, text="下移优先级", command=lambda: move_selected(1)).pack(side="left", padx=3)
        ttk.Button(buttons, text="导入工作簿", command=import_workbook).pack(side="left", padx=3)
        ttk.Button(buttons, text="严格→包含", command=convert_selected_boundary).pack(side="left", padx=3)
        ttk.Button(buttons, text="关闭", command=dialog.destroy).pack(side="right", padx=3)
        refresh()

    @staticmethod
    def _reference_for_screen(screen: ScreenInfo) -> dict:
        return {
            "space": "reference_pixel_centers",
            "width": screen.width,
            "height": screen.height,
            "source": screen.source,
            "monitor_id": screen.monitor_id,
            "monitor_origin_px": [screen.left, screen.top],
            "dpi_scale": None,
            "dpi_source": "unknown",
        }

    def edit_coordinate_reference(self):
        """Choose a monitor and edit the saved coordinate-reference metadata."""
        dialog = tk.Toplevel(self.root)
        dialog.title("屏幕与坐标参考")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()

        screen_names = [
            f"{i + 1}: {screen.monitor_id} ({screen.width}×{screen.height}, {screen.left},{screen.top})"
            for i, screen in enumerate(self.screens)
        ]
        selected_index = self.screens.index(self.selected_screen)
        screen_var = tk.StringVar(value=screen_names[selected_index])
        ttk.Label(dialog, text="屏幕").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        chooser = ttk.Combobox(
            dialog,
            textvariable=screen_var,
            values=screen_names,
            state="readonly",
            width=50,
        )
        chooser.grid(row=0, column=1, columnspan=3, sticky="w", padx=6, pady=4)

        fields = {
            "monitor_id": tk.StringVar(),
            "width": tk.StringVar(),
            "height": tk.StringVar(),
            "origin_x": tk.StringVar(),
            "origin_y": tk.StringVar(),
            "dpi_x": tk.StringVar(),
            "dpi_y": tk.StringVar(),
            "dpi_source": tk.StringVar(value="unknown"),
            "source": tk.StringVar(),
        }
        field_labels = [
            ("monitor_id", "显示器标识"),
            ("width", "参考宽度（像素中心数）"),
            ("height", "参考高度（像素中心数）"),
            ("origin_x", "桌面原点 X"),
            ("origin_y", "桌面原点 Y"),
            ("dpi_x", "DPI 比例 X（未知留空）"),
            ("dpi_y", "DPI 比例 Y（未知留空）"),
            ("dpi_source", "DPI 信息来源"),
            ("source", "坐标参考来源"),
        ]
        for row, (key, label) in enumerate(field_labels, start=1):
            ttk.Label(dialog, text=label).grid(row=row, column=0, sticky="e", padx=6, pady=3)
            ttk.Entry(dialog, textvariable=fields[key], width=36).grid(
                row=row, column=1, columnspan=3, sticky="w", padx=6, pady=3
            )

        def load_screen(index: int):
            screen = self.screens[index]
            reference = self._reference_for_screen(screen)
            for key in ("monitor_id", "width", "height", "source"):
                fields[key].set(str(reference[key]))
            fields["origin_x"].set(str(screen.left))
            fields["origin_y"].set(str(screen.top))
            fields["dpi_x"].set("")
            fields["dpi_y"].set("")
            fields["dpi_source"].set("unknown")

        def on_screen_change(_event=None):
            index = chooser.current()
            if index >= 0:
                load_screen(index)

        chooser.bind("<<ComboboxSelected>>", on_screen_change)
        load_screen(selected_index)

        def parse_finite(text: str, label: str) -> float:
            try:
                number = float(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} 必须是数字。") from exc
            if not math.isfinite(number):
                raise ValueError(f"{label} 必须是有限数值。")
            return number

        def accept():
            monitor_id = fields["monitor_id"].get().strip()
            source = fields["source"].get().strip()
            if not monitor_id or not source:
                messagebox.showerror("坐标元数据无效", "显示器标识和坐标参考来源不能为空。", parent=dialog)
                return
            try:
                width = int(fields["width"].get())
                height = int(fields["height"].get())
                if width <= 1 or height <= 1:
                    raise ValueError("参考宽度和高度必须大于 1。")
                origin = [
                    parse_finite(fields["origin_x"].get(), "桌面原点 X"),
                    parse_finite(fields["origin_y"].get(), "桌面原点 Y"),
                ]
                dpi_x, dpi_y = fields["dpi_x"].get().strip(), fields["dpi_y"].get().strip()
                if bool(dpi_x) != bool(dpi_y):
                    raise ValueError("DPI 比例 X 和 Y 必须同时填写或同时留空。")
                if dpi_x:
                    dpi_scale = [parse_finite(dpi_x, "DPI 比例 X"), parse_finite(dpi_y, "DPI 比例 Y")]
                    if any(value <= 0 for value in dpi_scale):
                        raise ValueError("DPI 比例必须为正数。")
                    dpi_source = fields["dpi_source"].get().strip()
                    if not dpi_source or dpi_source.lower() == "unknown":
                        raise ValueError("填写 DPI 比例时，请说明 DPI 信息来源。")
                else:
                    dpi_scale = None
                    dpi_source = "unknown"
            except ValueError as exc:
                messagebox.showerror("坐标元数据无效", str(exc), parent=dialog)
                return

            index = chooser.current()
            if index < 0:
                messagebox.showerror("坐标元数据无效", "请选择显示器。", parent=dialog)
                return
            selected_screen = self.screens[index]
            if source == selected_screen.source and (
                monitor_id != selected_screen.monitor_id
                or width != selected_screen.width
                or height != selected_screen.height
                or origin != [selected_screen.left, selected_screen.top]
            ):
                source = "user_supplied_override"
            candidate_reference = {
                "space": "reference_pixel_centers",
                "width": width,
                "height": height,
                "source": source,
                "monitor_id": monitor_id,
                "monitor_origin_px": origin,
                "dpi_scale": dpi_scale,
                "dpi_source": dpi_source,
            }
            try:
                validate_coordinate_reference_compatibility(
                    [*(roi.coordinate_reference for roi in self.roi_list), candidate_reference],
                    labels=[*(roi.name for roi in self.roi_list), "新坐标参考"],
                )
            except ROIValidationError as exc:
                messagebox.showerror(
                    "坐标参考冲突",
                    f"{exc}\n现有 ROI 保持原坐标；请为不同参考建立新的 ROI 集合。",
                    parent=dialog,
                )
                return
            self.selected_screen = selected_screen
            self.coordinate_reference = candidate_reference
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.grid(row=len(field_labels) + 1, column=0, columnspan=4, pady=8)
        ttk.Button(buttons, text="应用", command=accept).pack(side="left", padx=5)
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="left", padx=5)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.root.wait_window(dialog)

    def open_overlay(self, mode):
        if mode not in {"rectangle", "circle", "ellipse"}:
            raise ValueError(f"Unsupported ROI shape: {mode}")
        self.root.withdraw()
        self.mode = mode
        screen = self.selected_screen
        position = f"{screen.left:+d}{screen.top:+d}"

        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", TRANSPARENCY)
        self.overlay.geometry(f"{screen.width}x{screen.height}{position}")
        self.overlay.config(bg="gray")
        self.overlay.protocol("WM_DELETE_WINDOW", self.back_from_overlay)

        # An owned toolbar stays above its overlay when drawing gives the
        # canvas focus. Independent topmost windows can cover one another.
        self.button_window = tk.Toplevel(self.overlay)
        self.button_window.transient(self.overlay)
        # Keep a native title bar: on Windows, overrideredirect tool windows
        # lose the owner stacking guarantee when the canvas takes focus.
        self.button_window.title("ROI 绘制")
        self.button_window.resizable(False, False)
        self.button_window.attributes("-topmost", True)
        self.button_window.config(bg="gray25")
        tk.Label(self.button_window, text="右键绘制 · 左键拖动黄色控制点", bg="gray25", fg="white").pack(padx=8, pady=(5, 0))
        buttons = tk.Frame(self.button_window, bg="gray25")
        buttons.pack(padx=4, pady=4)
        tk.Button(buttons, text="确认 (Enter)", command=self.confirm_current).pack(side="left", padx=4)
        tk.Button(buttons, text="返回 (Esc)", command=self.back_from_overlay).pack(side="left", padx=4)
        self.button_window.geometry(position)
        self.button_window.protocol("WM_DELETE_WINDOW", self.back_from_overlay)
        for window in (self.overlay, self.button_window):
            window.bind("<Escape>", self._cancel_overlay)
            window.bind("<Return>", self._confirm_overlay)
            window.bind("<KP_Enter>", self._confirm_overlay)

        self.canvas = tk.Canvas(self.overlay, bg="gray", highlightthickness=0, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-3>", self.on_mouse_down)
        self.canvas.bind("<B3-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up)
        self.overlay.update_idletasks()
        self.canvas.update_idletasks()
        self._active_canvas_size = (
            max(self.canvas.winfo_width(), 2),
            max(self.canvas.winfo_height(), 2),
        )
        self.current_shape_id = None
        self.temp_center_id = None
        self._canvas_bounds = None
        self._clear_handles()
        self.button_window.update_idletasks()
        self.button_window.lift(self.overlay)
        self.canvas.focus_force()

    def _cancel_overlay(self, _event):
        self.back_from_overlay()
        return "break"

    def _confirm_overlay(self, _event):
        self.confirm_current()
        return "break"

    def back_from_overlay(self):
        self._clear_handles()
        if self.button_window is not None:
            self.button_window.destroy()
            self.button_window = None
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None
        self.canvas = None
        self._active_canvas_size = None
        self.current_shape_id = None
        self.temp_center_id = None
        self._canvas_bounds = None
        self.root.deiconify()
        self.toolbar.pack(padx=4, pady=4)

    def on_mouse_down(self, event):
        self._canvas_start = (event.x, event.y)
        self._canvas_bounds = [event.x, event.y, event.x, event.y]
        self._clear_handles()
        if self.current_shape_id:
            self.canvas.delete(self.current_shape_id)
        if self.temp_center_id:
            self.canvas.delete(self.temp_center_id)
            self.temp_center_id = None
        self.current_shape_id = self._create_shape(*self._canvas_bounds)

    def on_mouse_move(self, event):
        if self.current_shape_id is None or self._canvas_bounds is None:
            return
        self._canvas_bounds = [self._canvas_start[0], self._canvas_start[1], event.x, event.y]
        self._render_current_shape()

    def on_mouse_up(self, event):
        if self.current_shape_id is None or self._canvas_bounds is None:
            return
        self._canvas_bounds = [self._canvas_start[0], self._canvas_start[1], event.x, event.y]
        self._render_current_shape()
        self._create_handles_for_current()

    def _create_shape(self, x0, y0, x1, y1):
        if self.mode == "rectangle":
            return self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)
        return self.canvas.create_oval(x0, y0, x1, y1, outline="red", width=2)

    def _canonical_candidate(self):
        """Build the selected shape in local reference pixel-center coordinates."""
        canvas_width, canvas_height = self._active_canvas_size or (
            self.canvas.winfo_width(), self.canvas.winfo_height()
        )
        # Desktop origin locates the overlay; it is not a gaze-frame offset.
        mapping = CanvasMapping.from_sizes(
            max(int(canvas_width), 2),
            max(int(canvas_height), 2),
            self.coordinate_reference["width"],
            self.coordinate_reference["height"],
            target_origin=(0, 0),
        )
        x0, y0, x1, y1 = self._canvas_bounds
        left, top = mapping.map_point(x0, y0)
        right, bottom = mapping.map_point(x1, y1)
        geometry = geometry_from_bounds(
            self.mode,
            (left, top, right, bottom),
            constrain_circle=self.mode == "circle",
        )
        return mapping, self.mode, geometry

    def _render_current_shape(self):
        if self._canvas_bounds is None or self.current_shape_id is None:
            return
        x0, y0, x1, y1 = self._canvas_bounds
        try:
            mapping, shape, geometry = self._canonical_candidate()
            display_shape, display = mapping.map_geometry(shape, geometry, inverse=True)
            if display_shape == "rectangle":
                x0, y0, x1, y1 = (
                    display["left"], display["top"], display["right"], display["bottom"]
                )
            else:
                rx = display["r"] if display_shape == "circle" else display["a"]
                ry = display["r"] if display_shape == "circle" else display["b"]
                x0, y0 = display["cx"] - rx, display["cy"] - ry
                x1, y1 = display["cx"] + rx, display["cy"] + ry
        except (ROIValidationError, ValueError, KeyError):
            # A zero-width/height drag has no valid ROI yet; display the gesture.
            # Confirmation still validates and rejects this candidate.
            pass
        self.canvas.coords(self.current_shape_id, x0, y0, x1, y1)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        cx, cy = (left + right) / 2, (top + bottom) / 2
        if self.temp_center_id:
            self.canvas.delete(self.temp_center_id)
        self.temp_center_id = self.canvas.create_oval(
            cx - 3, cy - 3, cx + 3, cy + 3, fill="yellow", outline="black"
        )
        self.canvas.tag_raise(self.temp_center_id)
        for handle in self.handles:
            self.canvas.tag_raise(handle.id)
        if self.center_handle:
            self.canvas.tag_raise(self.center_handle.id)

    def confirm_current(self):
        if self.current_shape_id is None or self._canvas_bounds is None:
            messagebox.showinfo("提示", "请先按住鼠标右键绘制 ROI。", parent=self.overlay)
            return
        name = simpledialog.askstring("ROI 名称", "请输入 ROI 名称：", parent=self.overlay)
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showerror("ROI 无效", "ROI 名称不能为空。", parent=self.overlay)
            return
        if any(roi.name == name for roi in self.roi_list):
            messagebox.showerror("ROI 无效", f"ROI 名称已存在：{name}", parent=self.overlay)
            return
        next_priority = max((roi.priority for roi in self.roi_list), default=-1) + 1
        priority = simpledialog.askinteger(
            "ROI 优先级",
            "重叠时优先级较小者先命中：",
            initialvalue=next_priority,
            minvalue=0,
            parent=self.overlay,
        )
        if priority is None:
            return

        try:
            mapping, shape, geometry = self._canonical_candidate()
            roi = ROI.create(
                name=name,
                shape=shape,
                geometry=geometry,
                priority=priority,
                boundary_mode="inclusive",
                source={"kind": "new", "producer": "ROIDrawing"},
                coordinate_reference=self.coordinate_reference,
                mapping=mapping,
            )
            validate_rois([*self.roi_list, roi])
            validate_roi_reference_compatibility([*self.roi_list, roi])
        except (ROIValidationError, ValueError, KeyError) as exc:
            messagebox.showerror("ROI 无效", f"{exc}\n不同已知坐标参考请分别建立 ROI 集合。", parent=self.overlay)
            return

        self.roi_list.append(roi)
        messagebox.showinfo("已保存", f"ROI {roi.name} 已加入列表，优先级 {roi.priority}。", parent=self.overlay)
        self.back_from_overlay()

    def _create_handles_for_current(self):
        self._clear_handles()
        if self._canvas_bounds is None:
            return
        x0, y0, x1, y1 = self._canvas_bounds
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        cx, cy = (left + right) / 2, (top + bottom) / 2

        def resize(corner, x, y):
            bounds = list(self._canvas_bounds)
            if corner == "tl":
                bounds[0], bounds[1] = x, y
            elif corner == "tr":
                bounds[2], bounds[1] = x, y
            elif corner == "bl":
                bounds[0], bounds[3] = x, y
            else:
                bounds[2], bounds[3] = x, y
            self._canvas_bounds = bounds
            self._render_current_shape()
            self._move_handles_positions()

        self.handles = [
            DraggableHandle(self.canvas, left, top, callback=lambda x, y: resize("tl", x, y)),
            DraggableHandle(self.canvas, right, top, callback=lambda x, y: resize("tr", x, y)),
            DraggableHandle(self.canvas, left, bottom, callback=lambda x, y: resize("bl", x, y)),
            DraggableHandle(self.canvas, right, bottom, callback=lambda x, y: resize("br", x, y)),
        ]

        def move_center(x, y):
            bounds = list(self._canvas_bounds)
            width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            bounds[0], bounds[2] = x - width / 2, x + width / 2
            bounds[1], bounds[3] = y - height / 2, y + height / 2
            self._canvas_bounds = bounds
            self._render_current_shape()
            self._move_handles_positions()

        self.center_handle = DraggableHandle(
            self.canvas, cx, cy, size=10, callback=move_center
        )
        self._move_handles_positions()

    def _move_handles_positions(self):
        if self._canvas_bounds is None:
            return
        x0, y0, x1, y1 = self._canvas_bounds
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        cx, cy = (left + right) / 2, (top + bottom) / 2
        positions = [(left, top), (right, top), (left, bottom), (right, bottom)]
        for handle, (x, y) in zip(self.handles, positions):
            handle.move_to(x, y)
        if self.center_handle:
            self.center_handle.move_to(cx, cy)

    def _clear_handles(self):
        for handle in getattr(self, "handles", []):
            try:
                handle.destroy()
            except tk.TclError:
                pass
        self.handles = []
        if getattr(self, "center_handle", None):
            try:
                self.center_handle.destroy()
            except tk.TclError:
                pass
            self.center_handle = None

    def save_all_prompt(self):
        if not self.roi_list:
            messagebox.showwarning("无 ROI", "没有可保存的数据。", parent=self.root)
            return
        try:
            validate_roi_reference_compatibility(self.roi_list)
        except ROIValidationError as exc:
            messagebox.showerror(
                "坐标参考冲突",
                f"{exc}\n不同已知坐标参考请分别建立 ROI 集合。",
                parent=self.root,
            )
            return
        name = simpledialog.askstring("文件名", "请输入保存文件名：", parent=self.root)
        if name is None:
            return
        filename = self._normalise_filename(name)
        if filename is None:
            messagebox.showerror("文件名无效", "请输入不含路径分隔符的文件名（可选 .xlsx 扩展名）。", parent=self.root)
            return

        if len(sys.argv) > 1:
            main_path = Path(sys.argv[1])
        else:
            main_path = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
        save_dir = main_path / "ROIdata"
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / filename
            first_rows, metadata_rows = serialize_roi_workbook(self.roi_list)
            self._write_workbook_exclusive(save_path, first_rows, metadata_rows)
        except FileExistsError:
            messagebox.showerror("文件已存在", f"不会覆盖已有文件：\n{save_path}", parent=self.root)
            return
        except Exception as exc:
            messagebox.showerror("保存失败", f"ROI 工作簿保存失败：\n{exc}", parent=self.root)
            return

        messagebox.showinfo("保存成功", f"ROI 工作簿已保存至：\n{save_path}", parent=self.root)
        self.root.quit()

    @staticmethod
    def _normalise_filename(name: str) -> str | None:
        candidate = name.strip()
        if not candidate or Path(candidate).name != candidate or "/" in candidate or "\\" in candidate:
            return None
        suffix = Path(candidate).suffix
        if suffix and suffix.lower() != ".xlsx":
            return None
        if not suffix:
            candidate += ".xlsx"
        return candidate

    @staticmethod
    def _write_workbook_exclusive(save_path: Path, first_rows, metadata_rows):
        """Write a two-sheet workbook without replacing any existing path."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".roi-", suffix=".xlsx", dir=str(save_path.parent)
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
                pd.DataFrame(first_rows, columns=["命名", "形状", "描述"]).to_excel(
                    writer, index=False, sheet_name="ROI"
                )
                pd.DataFrame(metadata_rows, columns=ROI_METADATA_COLUMNS).to_excel(
                    writer, index=False, sheet_name="ROI_Metadata"
                )
            try:
                os.link(temp_path, save_path)
            except FileExistsError:
                raise
            except OSError:
                if save_path.exists():
                    raise FileExistsError(save_path)
                target_created = False
                try:
                    with temp_path.open("rb") as source, save_path.open("xb") as target:
                        target_created = True
                        shutil.copyfileobj(source, target)
                except Exception:
                    if target_created:
                        try:
                            save_path.unlink()
                        except OSError:
                            pass
                    raise
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


__all__ = ["ROIApp"]
