"""Shared records and small Tk helpers for the ROI drawing application."""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from roi_contract import (  # noqa: E402
    CanvasMapping,
    ROI,
    ROIValidationError,
    format_legacy_description,
    geometry_from_bounds,
)


@dataclass(frozen=True)
class ScreenInfo:
    """A monitor's desktop rectangle and the source used to discover it."""

    monitor_id: str
    left: int
    top: int
    width: int
    height: int
    source: str
    is_primary: bool = False


def enumerate_screens(fallback_width: int, fallback_height: int) -> list[ScreenInfo]:
    """Enumerate Windows monitors, falling back to Tk's single-screen view."""
    if sys.platform == "win32":
        try:
            return _enumerate_windows_screens()
        except (AttributeError, OSError, ctypes.ArgumentError):
            pass
    return [
        ScreenInfo(
            monitor_id="primary",
            left=0,
            top=0,
            width=max(int(fallback_width), 2),
            height=max(int(fallback_height), 2),
            source="tkinter_primary_screen",
            is_primary=True,
        )
    ]


def _enumerate_windows_screens() -> list[ScreenInfo]:
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )
    found: list[ScreenInfo] = []

    def on_monitor(handle, _hdc, _rect, _data):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if not user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            return True
        rect = info.rcMonitor
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width > 1 and height > 1:
            found.append(
                ScreenInfo(
                    monitor_id=info.szDevice or f"monitor-{len(found) + 1}",
                    left=int(rect.left),
                    top=int(rect.top),
                    width=int(width),
                    height=int(height),
                    source="windows_GetMonitorInfoW",
                    is_primary=bool(info.dwFlags & 1),
                )
            )
        return True

    callback = callback_type(on_monitor)
    user32.EnumDisplayMonitors(None, None, callback, 0)
    found.sort(key=lambda monitor: (not monitor.is_primary, monitor.left, monitor.top))
    if not found:
        raise OSError("Windows did not report any monitors")
    return found


class DraggableHandle:
    """Canvas resize/move handle with an optional geometry callback."""

    def __init__(self, canvas, x, y, size=8, callback=None):
        self.canvas = canvas
        self.callback = callback
        self.size = size
        self.id = canvas.create_oval(
            x - size,
            y - size,
            x + size,
            y + size,
            fill="yellow",
            outline="black",
            tags=("handle",),
        )
        self._drag_start = (0, 0)
        canvas.tag_bind(self.id, "<Button-1>", self._on_start)
        canvas.tag_bind(self.id, "<B1-Motion>", self._on_drag)
        canvas.tag_bind(self.id, "<ButtonRelease-1>", self._on_release)

    def _on_start(self, event):
        self._drag_start = (event.x, event.y)
        self.canvas.config(cursor="fleur")
        return "break"

    def _on_drag(self, event):
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        x0, y0, x1, y1 = self.canvas.coords(self.id)
        cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
        self.move_to(cx, cy)
        if self.callback:
            self.callback(cx, cy)
        self._drag_start = (event.x, event.y)
        return "break"

    def _on_release(self, _event):
        self.canvas.config(cursor="cross")
        return "break"

    def move_to(self, x, y):
        self.canvas.coords(self.id, x - self.size, y - self.size, x + self.size, y + self.size)

    def destroy(self):
        self.canvas.delete(self.id)


__all__ = [
    "CanvasMapping",
    "DraggableHandle",
    "ROI",
    "ROIValidationError",
    "ScreenInfo",
    "enumerate_screens",
    "format_legacy_description",
    "geometry_from_bounds",
]
