# LocalGaze

LocalGaze 是面向 Windows 的本地眼动采集、预处理和分析工作台。源码下载后可通过启动器准备依赖；Windows 便携 ZIP 解压后可直接运行。

## 快速开始

### 便携版

1. 从 [GitHub Releases](https://github.com/hexiovo/LocalGaze/releases) 下载 Windows ZIP 并解压到有写入权限的文件夹。
2. 双击 `LocalGaze.exe`。保留 ZIP 中的 `LocalGazeWorker.exe`、`ROIDrawing.exe`、`_internal` 和 `docs` 文件夹。
3. 在工作台选择会话 ID、输入文件和输出位置。程序不会把用户数据写进运行环境目录。

也可使用下面的源码启动方式。

### 源码版（Windows 10/11，64 位）

1. 安装 64 位 Python 3.12，或 Python 3.10，并确保 `py` 启动器可用。
2. 从 GitHub 克隆仓库或下载并解压源码 ZIP。
3. 双击 `launch_localgaze.bat`。首次运行会在仓库中创建 `.venv` 并安装项目依赖；之后会检查依赖并启动工作台。

也可在 PowerShell 中手动运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe launch.py
```

需要 Python 3.10 时，将 `-3.12` 改为 `-3.10`。安装过程需要网络连接；摄像头功能需要系统可访问的摄像头。

## 基本工作流程

1. 填写可重复使用的会话 ID，并分别选择模型、原始采集和分析输出位置。
2. 按界面提示校准；采集前选择匹配的校准模型。
3. 可选使用 ROI 绘制、预处理与质量检查，再选择输入文件运行分析和导出。
4. 通过任务状态、输出清单和日志确认任务是否完成；取消的任务可能保留已生成的部分文件。

产品手册位于 [docs/LocalGaze产品介绍与使用手册.pdf](docs/LocalGaze产品介绍与使用手册.pdf)，也可从工作台顶部打开。

## 从源码构建 Windows 便携包

建议使用 Python 3.12.14 或更高的 Python 3.12 维护版本构建。CPython 3.12.0 会让 PyInstaller 生成无法正常导入 SciPy 的冻结代码，构建脚本会拒绝该版本；源码启动仍可使用 Python 3.10 或 3.12。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_tools\windows\build.ps1
```

构建输出写入被 Git 忽略的 `build-artifacts/` 和 `release/`。GitHub Actions 可在手动触发时生成可下载构件；推送 `v*` 标签时才创建 GitHub Release。

## 第三方许可与来源

仓库没有为 LocalGaze 自有代码附加项目级开源许可证。`THIRD_PARTY_NOTICES.md` 和 `third_party/licenses/` 列出随源码及便携包分发的第三方组件来源与原始许可文本；这些许可仅适用于其中列明的组件。Windows 便携包也会在 `third_party/` 中携带这些通知。

## 数据与限制

- 仓库不附带参与者数据、校准模型或真实采集样本。用户应自行保留原始数据和模型，并按需要备份输出。
- 软件自检和合成数据检查不能证明真实摄像头或独立眼动精度；多显示器 DPI 行为尚未验证，也未覆盖特定电脑上的效果。
- I-VT 眼跳结果是工程候选事件。屏幕尺寸、时间单位和坐标参考系必须有可信来源；结果不代表生理真值或独立精度。
- 相机、光照、姿势、屏幕和安装路径都会影响采集；正式使用前请按产品手册检查当前设备与流程。

## 版本

当前程序与分析方法版本见 [version.md](version.md)。
