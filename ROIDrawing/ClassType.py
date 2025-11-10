import tkinter as tk


class ROI:
    def __init__(self, name: str, shape: str, data: dict):
        self.name = name
        self.shape = shape  # 'rectangle'|'circle'|'ellipse'
        self.data = data  # dict with geometry info

    def description(self) -> str:
        """返回 python 可解析的表达式字符串"""
        if self.shape == 'rectangle':
            left = self.data['left']; right = self.data['right']
            top = self.data['top']; bottom = self.data['bottom']
            return f"x>{left} and x<{right} and y>{top} and y<{bottom}"
        elif self.shape == 'circle':
            cx = self.data['cx']; cy = self.data['cy']; r = self.data['r']
            return f"(x-{cx})**2 + (y-{cy})**2 <= {r}**2"
        elif self.shape == 'ellipse':
            cx = self.data['cx']; cy = self.data['cy']; a = self.data['a']; b = self.data['b']
            return f"((x-{cx})/{a})**2 + ((y-{cy})/{b})**2 <= 1"
        else:
            return ""


class DraggableHandle:
    def __init__(self, canvas, x, y, size=8, callback=None):
        self.canvas = canvas
        self.callback = callback
        self.size = size
        self.id = canvas.create_oval(x-size, y-size, x+size, y+size, fill='yellow', outline='black', tags='handle')
        self._drag_start = (0, 0)
        # 绑定拖动事件
        canvas.tag_bind(self.id, "<Button-1>", self._on_start)
        canvas.tag_bind(self.id, "<B1-Motion>", self._on_drag)
        canvas.tag_bind(self.id, "<ButtonRelease-1>", self._on_release)  # 拖动结束恢复光标

    def _on_start(self, event):
        self._drag_start = (event.x, event.y)
        self.canvas.config(cursor='fleur')  # 开始拖动，光标变成十字手形

    def _on_drag(self, event):
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        x0, y0, x1, y1 = self.canvas.coords(self.id)
        cx, cy = (x0 + x1)/2 + dx, (y0 + y1)/2 + dy
        self.move_to(cx, cy)
        if self.callback:
            self.callback(cx, cy)
        self._drag_start = (event.x, event.y)

    def _on_release(self, event):
        self.canvas.config(cursor='cross')  # 拖动结束，恢复默认绘制光标

    def move_to(self, x, y):
        self.canvas.coords(self.id, x-self.size, y-self.size, x+self.size, y+self.size)

    def destroy(self):
        self.canvas.delete(self.id)
