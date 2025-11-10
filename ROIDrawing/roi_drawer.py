import os
import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd
from ClassType import *
from Global_data import *
import sys
from pathlib import Path


class ROIApp:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.config(bg='gray25')
        self.root.geometry("230x40+0+0")

        # === Toolbar ===
        self.toolbar = tk.Frame(root, bg='gray25')
        self.toolbar.pack(padx=5, pady=5)

        self._add_button("圆形", lambda: self.open_overlay('circle'))
        self._add_button("矩形", lambda: self.open_overlay('rectangle'))
        self._add_button("椭圆", lambda: self.open_overlay('ellipse'))
        self._add_button("保存", self.save_all_prompt)
        self._add_button("退出", self.root.destroy)

        # === 拖动窗口 ===
        self.toolbar.bind('<Button-1>', self._start_move)
        self.toolbar.bind('<B1-Motion>', self._do_move)
        self._drag_start_x = self._drag_start_y = 0

        # 状态与数据
        self.overlay = None
        self.canvas = None
        self.current_shape_id = None
        self.temp_center_id = None
        self.handles = []
        self.center_handle = None
        self.roi_list = []
        self.button_drag_start_x = 0
        self.button_drag_start_y = 0

    def _add_button(self, text, cmd):
        btn = tk.Button(self.toolbar, text=text, command=cmd)
        btn.pack(side='left', padx=5)

    def _start_move(self, e):
        self._drag_start_x, self._drag_start_y = e.x, e.y

    def _do_move(self, e):
        x = self.root.winfo_x() + e.x - self._drag_start_x
        y = self.root.winfo_y() + e.y - self._drag_start_y
        self.root.geometry(f"+{x}+{y}")

    # ========== Overlay ==========
    def open_overlay(self, mode):
        print(f"[ROIApp] 打开 overlay, 模式: {mode}")
        self.root.withdraw()
        self.mode = mode

        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.lift()
        self.overlay.geometry(f'{SCREEN_W}x{SCREEN_H}+0+0')
        self.overlay.attributes('-alpha', TRANSPARENCY)
        self.overlay.config(bg='gray')

        self.button_window = tk.Toplevel(self.root)
        self.button_window.overrideredirect(True)
        self.button_window.attributes('-topmost', True)
        self.button_window.geometry("100x40+0+0")
        self.button_window.config(bg='gray25')
        print("[ROIApp] 按钮窗口创建完成")

        # 拖动按钮窗口
        def start_button_drag(event):
            self.button_drag_start_x = event.x
            self.button_drag_start_y = event.y

        def do_button_drag(event):
            x = self.button_window.winfo_x() + event.x - self.button_drag_start_x
            y = self.button_window.winfo_y() + event.y - self.button_drag_start_y
            self.button_window.geometry(f"+{x}+{y}")

        self.button_window.bind('<Button-1>', start_button_drag)
        self.button_window.bind('<B1-Motion>', do_button_drag)

        tk.Button(self.button_window, text='确认', command=self.confirm_current, bg='lightgray').pack(side='left', padx=4)
        tk.Button(self.button_window, text='返回', command=self.back_from_overlay, bg='lightgray').pack(side='left', padx=4)
        print("[ROIApp] overlay 按钮添加完成")

        self.canvas = tk.Canvas(self.overlay, bg='gray', highlightthickness=0, cursor='cross')
        self.canvas.pack(fill='both', expand=True)
        self.current_shape_id = None
        self.temp_center_id = None
        self._clear_handles()
        print("[ROIApp] 画布初始化完成")

        self.canvas.bind('<ButtonPress-3>', self.on_right_mouse_down)
        self.canvas.bind('<B3-Motion>', self.on_right_mouse_move)
        self.canvas.bind('<ButtonRelease-3>', self.on_right_mouse_up)

    def back_from_overlay(self):
        print("[ROIApp] 返回主界面")
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        if hasattr(self, 'button_window') and self.button_window:
            self.button_window.destroy()
            self.button_window = None
        self.canvas = None
        self._clear_handles()
        self.root.deiconify()
        self.toolbar.pack(padx=5, pady=5)

    # ========== 绘制逻辑 ==========
    def on_mouse_down(self, e):
        self.start_x, self.start_y = e.x, e.y
        if self.current_shape_id:
            self.canvas.delete(self.current_shape_id)
        self.current_shape_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline='red', width=2) if self.mode=='rectangle' else self.canvas.create_oval(e.x, e.y, e.x, e.y, outline='red', width=2)
        print(f"[ROIApp] 鼠标按下坐标: ({e.x},{e.y})")

    def on_mouse_move(self, e):
        if not self.current_shape_id: return
        self.canvas.coords(self.current_shape_id, self.start_x, self.start_y, e.x, e.y)
        cx, cy = (self.start_x+e.x)/2, (self.start_y+e.y)/2
        if self.temp_center_id:
            self.canvas.delete(self.temp_center_id)
        self.temp_center_id = self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill='yellow')
        print(f"[ROIApp] 鼠标移动, shape coords: {self.canvas.coords(self.current_shape_id)}")

    def on_mouse_up(self, e):
        if self.current_shape_id:
            print(f"[ROIApp] 鼠标释放, final coords: {self.canvas.coords(self.current_shape_id)}")
            self._create_handles_for_current()

    def on_right_mouse_down(self, e):
        self.start_x, self.start_y = e.x, e.y
        if self.current_shape_id:
            self.canvas.delete(self.current_shape_id)
        self.current_shape_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline='red', width=2) if self.mode=='rectangle' else self.canvas.create_oval(e.x, e.y, e.x, e.y, outline='red', width=2)
        print(f"[ROIApp] 右键按下坐标: ({e.x},{e.y})")

    def on_right_mouse_move(self, e):
        if not self.current_shape_id: return
        self.canvas.coords(self.current_shape_id, self.start_x, self.start_y, e.x, e.y)
        cx, cy = (self.start_x+e.x)/2, (self.start_y+e.y)/2
        if self.temp_center_id:
            self.canvas.delete(self.temp_center_id)
        self.temp_center_id = self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill='yellow')
        print(f"[ROIApp] 右键移动, shape coords: {self.canvas.coords(self.current_shape_id)}")

    def on_right_mouse_up(self, e):
        if self.current_shape_id:
            print(f"[ROIApp] 右键释放, final coords: {self.canvas.coords(self.current_shape_id)}")
            self._create_handles_for_current()

    # ========== 保存 ==========
    def confirm_current(self):
        if not self.current_shape_id:
            messagebox.showinfo("提示", "请先绘制 ROI。")
            return
        if not messagebox.askyesno("保存", "是否保存当前 ROI？"):
            return
        name = simpledialog.askstring("输入名称", "请输入 ROI 名称：")
        if not name:
            return
        x0, y0, x1, y1 = self.canvas.coords(self.current_shape_id)
        left, right, top, bottom = min(x0,x1), max(x0,x1), min(y0,y1), max(y0,y1)
        if self.mode=='rectangle':
            roi = ROI(name,'rectangle',{'left':left,'right':right,'top':top,'bottom':bottom})
        elif self.mode=='circle':
            cx,cy=(left+right)/2,(top+bottom)/2
            r=max((right-left)/2,(bottom-top)/2)
            roi = ROI(name,'circle',{'cx':cx,'cy':cy,'r':r})
        else:
            cx,cy=(left+right)/2,(top+bottom)/2
            a,b=(right-left)/2,(bottom-top)/2
            roi = ROI(name,'ellipse',{'cx':cx,'cy':cy,'a':a,'b':b})
        self.roi_list.append(roi)
        print(f"[ROIApp] ROI {name} 已保存, 类型: {roi.shape}, 描述: {roi.description()}")
        messagebox.showinfo("已保存", f"ROI {name} 已加入列表。")
        self.back_from_overlay()

    def save_all_prompt(self):
        if not self.roi_list:
            messagebox.showwarning("无ROI", "没有可保存的数据。")
            return
        name = simpledialog.askstring("文件名", "请输入保存文件名：")
        if not name:
            return
        if len(sys.argv)>1:
            main_path = Path(sys.argv[1])
        else:
            main_path = Path(sys.executable).parent if getattr(sys,'frozen',False) else Path.cwd()
        save_dir = main_path / "ROIdata"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{name}.xlsx"
        data = [{'命名':r.name,'形状':r.shape,'描述':r.description()} for r in self.roi_list]
        df=pd.DataFrame(data)
        df.to_excel(save_path,index=False)
        print(f"[ROIApp] 所有 ROI 已保存至 {save_path}")
        messagebox.showinfo("成功", f"文件已保存至：\n{save_path}")
        self.root.quit()

    #-------------------- ROI 移动逻辑 ----------------
    def _create_handles_for_current(self):
        self._clear_handles()
        coords = self.canvas.coords(self.current_shape_id)
        if not coords or len(coords)<4: return
        x0,y0,x1,y1 = coords
        left,right = min(x0,x1), max(x0,x1)
        top,bottom = min(y0,y1), max(y0,y1)
        cx,cy = (left+right)/2, (top+bottom)/2
        print(f"[ROIApp] 创建 handles, 初始坐标: {coords}")

        def top_left_move(nx,ny):
            nonlocal left,top
            left,top=nx,ny
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 左上角移动到 ({nx},{ny})")

        def top_right_move(nx,ny):
            nonlocal right,top
            right,top=nx,ny
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 右上角移动到 ({nx},{ny})")

        def bottom_left_move(nx,ny):
            nonlocal left,bottom
            left,bottom=nx,ny
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 左下角移动到 ({nx},{ny})")

        def bottom_right_move(nx,ny):
            nonlocal right,bottom
            right,bottom=nx,ny
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 右下角移动到 ({nx},{ny})")

        def top_middle_move(nx,ny):
            nonlocal top
            top=ny
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 上中点移动到 ({nx},{ny})")

        def bottom_middle_move(nx,ny):
            nonlocal bottom
            bottom=ny
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 下中点移动到 ({nx},{ny})")

        def left_middle_move(nx,ny):
            nonlocal left
            left=nx
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 左中点移动到 ({nx},{ny})")

        def right_middle_move(nx,ny):
            nonlocal right
            right=nx
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 右中点移动到 ({nx},{ny})")

        def center_move(nx,ny):
            nonlocal left,right,top,bottom
            w,h=right-left,bottom-top
            left,right=nx-w/2,nx+w/2
            top,bottom=ny-h/2,ny+h/2
            self.canvas.coords(self.current_shape_id,left,top,right,bottom)
            self._move_handles_positions()
            print(f"[ROIApp] 中心点移动到 ({nx},{ny})")

        self.handles = [
            DraggableHandle(self.canvas,left,top,callback=top_left_move),
            DraggableHandle(self.canvas,right,top,callback=top_right_move),
            DraggableHandle(self.canvas,left,bottom,callback=bottom_left_move),
            DraggableHandle(self.canvas,right,bottom,callback=bottom_right_move),
            DraggableHandle(self.canvas,cx,top,callback=top_middle_move,size=6),
            DraggableHandle(self.canvas,cx,bottom,callback=bottom_middle_move,size=6),
            DraggableHandle(self.canvas,left,cy,callback=left_middle_move,size=6),
            DraggableHandle(self.canvas,right,cy,callback=right_middle_move,size=6)
        ]
        self.center_handle=DraggableHandle(self.canvas,cx,cy,size=10,callback=center_move)

    def _move_handles_positions(self):
        if not self.current_shape_id: return
        coords=self.canvas.coords(self.current_shape_id)
        if not coords or len(coords)<4: return
        x0,y0,x1,y1=coords
        left,right=min(x0,x1),max(x0,x1)
        top,bottom=min(y0,y1),max(y0,y1)
        cx,cy=(left+right)/2,(top+bottom)/2
        positions=[
            (left,top),(right,top),(left,bottom),(right,bottom),
            (cx,top),(cx,bottom),(left,cy),(right,cy)
        ]
        for h,(hx,hy) in zip(self.handles,positions):
            h.move_to(hx,hy)
        if self.center_handle:
            self.center_handle.move_to(cx,cy)
        print(f"[ROIApp] 更新 handles 位置, shape coords: {coords}")

    def _clear_handles(self):
        for h in getattr(self, "handles", []):
            try:
                h.destroy()
            except Exception:
                pass
        self.handles=[]
        if hasattr(self,"center_handle") and self.center_handle:
            try:
                self.center_handle.destroy()
            except Exception:
                pass
            self.center_handle=None
        print("[ROIApp] 清空所有 handles")
