"""Repeatable GUI/workbook smoke test, also runnable inside the frozen EXE."""
import json
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch


def run(root, app, output):
    from roi_drawer import ROIApp
    from roi_contract import serialize_roi_workbook, classify_point
    from EyeEvaluate.FileOperate.ROIextract import extract_ROI
    output.mkdir(parents=True, exist_ok=True)
    results = {'frozen': bool(getattr(sys, 'frozen', False)), 'checks': []}
    def capture(window, name):
        # Capture only this application's window, for local visual verification.
        from PIL import ImageGrab
        root.update()
        window.lift()
        window.update()
        import time
        time.sleep(0.4)
        root.update()
        x, y = window.winfo_rootx(), window.winfo_rooty()
        ImageGrab.grab(bbox=(x, y, x + window.winfo_width(), y + window.winfo_height())).save(output / f'{name}.png')
    def children(widget):
        for child in widget.winfo_children():
            yield child
            yield from children(child)
    try:
        capture(root, '01-toolbar')
        assert root.winfo_width() >= root.winfo_reqwidth()
        results['checks'].append('toolbar fits requested content')
        for index, shape in enumerate(('rectangle', 'circle', 'ellipse')):
            app.open_overlay(shape)
            capture(app.button_window, f'02-{shape}-controls')
            app.on_mouse_down(SimpleNamespace(x=250, y=250))
            app.on_mouse_move(SimpleNamespace(x=550, y=450))
            app.on_mouse_up(SimpleNamespace(x=550, y=450))
            assert app._canonical_candidate()[1] == shape
            geometry = app._canonical_candidate()[2]
            if shape == 'rectangle':
                assert abs(geometry['left'] - 250) < 1, geometry
                assert abs(geometry['right'] - 550) < 1, geometry
            with patch('roi_drawer.simpledialog.askstring', return_value=f'测试区域-{shape}'), patch('roi_drawer.simpledialog.askinteger', return_value=index), patch('roi_drawer.messagebox.showinfo'):
                app.confirm_current()
            assert len(app.roi_list) == index + 1
        original = list(app.roi_list)
        rows, metadata = serialize_roi_workbook(original)
        workbook = output / '中文路径 ROI 往返.xlsx'
        ROIApp._write_workbook_exclusive(workbook, rows, metadata)
        restored = [record['_roi'] for record in extract_ROI(workbook)]
        assert [r.to_dict() for r in restored] == [r.to_dict() for r in original]
        results['checks'].append('all three shapes drawn, confirmed, exported and imported losslessly')
        before = workbook.read_bytes()
        try:
            ROIApp._write_workbook_exclusive(workbook, rows, metadata)
            raise AssertionError('overwrite was allowed')
        except FileExistsError:
            assert workbook.read_bytes() == before
        results['checks'].append('existing workbook protected')
        app.roi_list = []
        app.manage_rois()
        dialog = next(w for w in root.winfo_children() if w.winfo_class() == 'Toplevel')
        with patch('roi_drawer.filedialog.askopenfilename', return_value=str(workbook)), patch('roi_drawer.messagebox.showinfo'):
            next(w for w in children(dialog) if w.winfo_class() == 'TButton' and w.cget('text') == '导入工作簿').invoke()
        assert len(app.roi_list) == 3
        capture(dialog, '03-manager')
        dialog.destroy()
        results['checks'].append('actual manager import button')
        def close_coordinate_dialog():
            dialog = next(w for w in root.winfo_children() if w.winfo_class() == 'Toplevel')
            capture(dialog, '04-coordinates')
            dialog.destroy()
        root.after(300, close_coordinate_dialog)
        app.edit_coordinate_reference()
        app.open_overlay('rectangle')
        app._cancel_overlay(None)
        assert app.overlay is None and root.state() == 'normal'
        results['checks'].append('coordinate dialog and overlay cancellation')
        export = output / '另存为.xlsx'
        with patch('roi_drawer.filedialog.asksaveasfilename', return_value=str(export)), patch('roi_drawer.messagebox.showinfo'):
            app.save_all_prompt()
        assert len(extract_ROI(export)) == 3
        results['checks'].append('save dialog export')
        results['ok'] = True
    except Exception:
        results['ok'] = False
        results['error'] = traceback.format_exc()
    finally:
        (output / 'result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        root.destroy()
