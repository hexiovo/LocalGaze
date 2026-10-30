# ROI 提取与绘制工具

独立构建的输出目录默认是仓库下 `build-artifacts/windows-2026.10.31.0/roidrawing/`，也可用 `-Destination` 指定其他位置。

这是 Windows x64 单文件程序，内嵌 Python、Tcl/Tk、ROI 函数、pandas、NumPy 和 openpyxl。
复制 **ROIDrawing.exe 一个文件** 到另一台 Windows 10/11 64 位电脑即可双击运行，无需安装 Python、Excel 或复制项目文件夹。不能用于 macOS、Linux、32 位 Windows；其他电脑尚未实测。
首次启动需解压内嵌运行库至系统临时目录，可能稍慢；关闭后由运行库清理。无需安装程序。

## 操作

1. 将要标注的页面、图片或视频画面显示在屏幕上。
2. 双击 exe；需要时在“屏幕/坐标”中选择显示器和坐标参考。
3. 选择圆形、矩形或椭圆，按住鼠标**右键**拖拽绘制。
4. 用左键拖动黄色控制点调整；Enter 确认，输入名称和优先级。Esc 返回工具栏。
5. 可在“ROI管理”中导入 `.xlsx`、调整优先级、转换旧矩形边界；表格可横向滚动并调整列宽。
6. 点击“保存”，选择 `.xlsx` 保存位置。保存成功后程序退出；已有同名文件不会被覆盖。

输出保留旧版的“命名、形状、描述”表及带结构化几何参数的 `ROI_Metadata` 表，与 LocalGaze 的 ROI 读取逻辑兼容。
坐标取决于标注时选择的显示器及参考分辨率；换电脑可以运行程序，已有 ROI 则需核对坐标参考后再使用。
日志：`%LOCALAPPDATA%\ROIDrawing\logs\ROIDrawing.log`。源代码和构建文件保存在本目录。
依赖许可证嵌入 exe 的 `licenses` 资源目录，构建时也保存到 `build\licenses`。

## 重新构建

从仓库根目录运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\ROIDrawing\build.ps1`。需要改输出位置时添加 `-Destination <目录>`。
构建机需要 Python 3.12 x64 和网络下载依赖；使用者电脑不需要这些。
`build_exe.py` 从父项目收集 `roi_contract.py` 和 `EyeEvaluate.FileOperate.ROIextract`，一并编入 exe，无运行时外部源码依赖。

## 验证

运行 `ROIDrawing.exe --self-test <新的输出文件夹绝对路径>` 可执行 GUI/Excel 冒烟检查并生成 `result.json` 和界面截图。
验证三种形状、实际像素坐标、Excel 往返、管理页导入、原文件防覆盖、坐标设置窗口、Esc 返回和另存为。
