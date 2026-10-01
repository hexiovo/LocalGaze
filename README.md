# LocalGaze 眼动工作台

当前源代码的统一桌面入口是项目根目录的 `launch.py`。当前新版便携包的统一桌面入口位于 `release/LocalGaze-20261002-guide/LocalGaze.exe`，同目录的 `ROIDrawing.exe` 是新版独立 ROI 入口。项目根目录的旧 EXE 和此前 `LocalGaze-20261001` 发布均予以保留；请使用 LocalGaze-20261002-guide 文件夹中的新版。

## 快速启动

在项目根目录打开终端，使用包含 Tk 的 Python 环境运行：

```powershell
python -B launch.py
```

`-B` 可避免这次启动写入 `.pyc` 缓存。默认启动只建立桌面界面，不会自动初始化摄像头或启动任务。选择“校准并保存模型”或“使用已有模型采集”后，后端才会启动相应任务。

统一入口有五页：会话与采集、ROI 绘制、预处理与 QC、分析与导出、使用指南。校准默认使用 v2 紧凑眼部与头部特征；固定九点校准按提示完成三轮有效访问并通过稳定门控。软件检查不构成参与者精度基准，不能据此声称注视精度提升。窗口顶部“查看整体指南”按钮可随时打开内置阅读页，任务运行时也可使用；同页“打开 TXT”可在系统编辑器查看。整体指南涵盖首次流程、文件夹示例、校准/采集、可选修正与 ROI、预处理匹配、导出和常见问题。正式步骤与指标含义见 [LocalGaze使用指导语及流程.txt](LocalGaze使用指导语及流程.txt)。

使用便携包时解压整个 `LocalGaze-20261002-guide.zip`，双击其中的 `LocalGaze.exe`。它带 Python 与任务依赖，无需运行旧环境安装脚本；必须保留 `_internal` 和 `LocalGazeWorker.exe`，不能只复制主 EXE。用户模型、数据和输出目录由工作台选择。

## Python 环境与功能依赖

仓库没有根目录的统一 `requirements.txt` 或 `pyproject.toml`。现有环境文件是分组件维护的，版本锁定反映仓库中旧组件和当前源代码的依赖线索，不构成所有机器均可复现的兼容性声明。

- `env_requirement/environment.yml` 固定 Python 3.10.19、Tk 和较完整的 pip 组件集；它还含有指向特定机器的 `prefix`，直接复制到其他电脑前需要先检查该路径。
- 桌面壳使用 Python 标准库 Tkinter。预处理、分析和 ROI 数据路径会用到 NumPy、pandas、openpyxl 等；校准/采集路径还依赖 OpenCV、MediaPipe 和 scikit-learn 等相机/模型组件。只有运行相应任务时才需要其后端依赖。
- `env_requirement/requirements_LocalGaze.txt` 和 `LocalGaze/requirements.txt` 是包含相机与模型库的较宽清单；其中同时列有 `opencv-python` 与 `opencv-contrib-python`。它不是为新统一界面审定的最小依赖集合，当前文档没有验证这两个发行包并存的安装结果。
- `env_requirement/requirements_ROIDrawing.txt` 与 `ROIDrawing/requirements.txt` 服务于独立旧 ROI 组件，主要列有 NumPy、pandas、openpyxl 和日期/时区库。
- 当前 manifests 未声明统一入口在 Python 3.10 以外版本的完整支持范围；Python 3.10.19 是仓库现存 Conda 文件所 pin 的参考基线。这里没有安装依赖，也没有为新环境执行完整兼容性验证。

旧脚本有额外限制：`SetForLocalGaze.bat` 会创建/激活名为 `LocalGaze` 的 Conda 环境，并从 `environment.yml` 安装宽依赖，但脚本还尝试读取 `env_requirement/requirements.txt`；仓库文件名实际是 `requirements_LocalGaze.txt`。它使用 pip 镜像并会修改本机环境，不建议把它当作已核验的统一入口安装器。`SetForROIDrawing.bat` 单独创建 Python 3.13 的 `ROIDrawing` 环境并安装旧 ROI 清单；它不安装或配置根目录统一桌面入口。

如果环境缺少 Tk、数值/表格依赖或相机组件，按实际报错和功能清单核对相应环境文件。不要把某个模块的清单当成全部功能都已验证可运行的证明。

## 工作流程与数据

1. 会话与采集页显示会话 ID、共享输出位置、模型路径和采集按钮；预处理页显示自己的数据文件/文件夹输入，分析页显示独立的单个数据文件输入及可选 ROI 工作簿。预处理和分析输入相互独立，成功的唯一预处理文件只会预填分析页。工作台规范化会话名并将同一 session ID 传给任务。
2. 新建模型请显式点击“校准并保存模型”；已有模型采集请先选择现存 `.pkl`，再点击“使用已有模型采集”。采集不会自动重新校准。已知目标偏移修正是单独的可选流程；普通采集默认不应用修正，显式启用时仍保留原始 `X`/`Y` 并添加修正坐标列。
3. ROI 绘制在现有独立工具窗口进行，区域叠加在选定显示器的当前屏幕画面上。请先在目标显示器显示要标注的图片或视频，再启动工具；没有背景文件选择器，屏幕背景也不是眼动输入。ROI 工作簿写入所选输出目录的 `ROIdata` 子目录，分析时可选择该工作簿。
4. 预处理的时间单位必须明确选择 seconds、milliseconds 或 microseconds；新格式字段 `elapsed_s` 总是秒。分析页的自动识别选项则把 `time_unit=None` 传给 reader；如果纯数值时间的来源有歧义，请选明真实单位。
5. 预处理 QC、分析产物与原始数据应分开保存。analysis preset 不插补 gaze X/Y；处理结果中的派生字段不能替代原始观测。工作台根据后端真实报告并确认的路径显示输出，分析导出使用会话子目录和不覆写策略。
6. 任务有就绪、运行、取消、完成和失败状态。取消会等待自有子进程退出；失败/取消的目录可能含部分输出，需先看状态和日志再决定后续操作。

分析导出会生成摘要工作簿、samples/events/ROI-hit CSV 和哈希 manifest。预处理还会生成处理文件与 QC/批次记录，具体输出取决于所选输入格式和任务。v2 便携包的 `version.md` 与 `release-manifest.json` 记录版本、来源和包内文件校验；ZIP 自身的 SHA-256 写入包外 delivery manifest。

## 指标的含义与限制

结果表里的 `status`、`reason` 和 `method` 与数值同样重要。`not_computable` 表示该指标依赖的输入或测量前提不足；`null` 不等于零。已实现统计在可信时间、有效样本、已知屏幕尺寸或兼容 ROI 坐标引用缺失时可保留不可计算原因。当前 `fixation_candidate`、`fast_movement_candidate` 和 EAR 状态相关 blink/closure 是工程候选，不是经参与者真值验证的生理事件。

训练拟合 residual 不是独立 gaze accuracy；iris ring/eye-span 是二维 ratio 代理，不是物理瞳孔或毫米；帧接收时间不是相机曝光时刻。ROI 坐标引用未知时不推断显示器坐标、DPI 或映射。当前代码对 `visual_angle_displacement_deg`、`stimulus_first_fixation_latency_ms`、`stimulus_first_roi_entry_latency_ms`、`binocular_vergence_deg`、`physical_pupil_diameter_mm` 和 `independent_gaze_accuracy_px` 只输出 `null` 与原因，尚无可用计算或可靠来源；仅补元数据不会自动增加这些算法或测量来源。

所有软件检查都不能替代真实采集设备、参与者准确度基准、可见窗口/DPI 使用性或研究方案验证。当前没有宣称上述真实场景已经完成验证。

## 旧入口与独立组件

| 入口 | 适用范围 | 说明 |
|---|---|---|
| `python -B launch.py` | 当前统一源代码桌面工作台 | 推荐入口；UI 将采集、预处理、ROI 和分析作为按需任务连接。 |
| `LocalGaze/main.py` / `LocalGaze.exe` | 既有独立 LocalGaze 采集流程 | 保留的旧入口，不代表含有根目录新增的统一界面。 |
| `ROIDrawing/main.py` / `ROIDrawing.exe` | 既有 ROI 绘制流程 | 统一界面仍通过该独立工具提供 ROI 绘制。 |
| `EyeEvaluate/main.py` | 旧分析图形界面 | 新桌面分析页走 headless `analyze_file` 路径；旧 GUI 的可选界面依赖需单独核对。 |
| `python -B pre/run_preprocessing.py` | 预处理命令行 | 可独立运行现有预处理流程；查看该脚本参数和本指南的时间单位规则。 |

## Windows 打包与检查

2026-10-01 的打包工作使用现有 LocalGaze Python 3.10.19 环境和 PyInstaller 6.16.0，采用目录式发布，三个入口共享完整依赖。主程序与 ROI 入口不显示终端；工作台通过隐藏控制台任务程序保持 UTF-8 日志和结果管道。模型内置类型使用显式模块发现，能够从 EXE 归档加载。没有调整眼动特征或训练公式。

Windows 中文安装路径下，MediaPipe 0.10.14 的原生资源加载需要 ASCII 路径。程序仅将其自带图/模型资源复制到本进程的系统临时目录，结束时清理；不复制用户数据或已校准模型。可用 `LOCALGAZE_RESOURCE_CACHE` 指定有写权限的 ASCII 临时目录。构建时也检查中文路径下的无终端主程序。

构建入口：`build_tools/windows/build.ps1`。必须提供 `-PythonExe`（包含功能依赖及 PyInstaller 的 Python）；`-ArtifactsRoot` 默认是 `E:\CodexData\localgaze-builds\20261001-v2`，`-ReleaseName` 默认是 `LocalGaze-20261001-v2`。目标目录、发布目录或 ZIP 已存在时，脚本会为构建目录和发布名一起选择未占用的 `-r1`、`-r2` 等后缀，不覆盖旧结果。构建日志、安装报告和外置交付清单保留在外部构建目录。

`release-manifest.json` 记录包内文件、相关源码哈希、版本与安装检查结果。检查包含隐藏 Tk 界面/指南、所有模型类型发现、模型保存/加载、FaceMesh 图及原生库、ROI 工作簿，以及真实任务子进程对合成数据的预处理和统一导出。它不代表真实摄像头、独立眼动精度、可见窗口/DPI 或全新电脑兼容性已通过。发布包中的 `LocalGazeWorker.exe --check-installation 新报告.json` 可重复执行相同检查，报告路径必须不存在。

## 项目记录

S01–S04 的已完成范围、验证证据和限制见 [version.md](version.md)。分析方法目录与数据字段以对应源代码和实际导出为准。
