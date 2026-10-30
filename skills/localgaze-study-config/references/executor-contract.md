# LocalGaze 执行器真实契约

合同依据当前项目源码中的 `desktop_configuration.py`、`desktop_jobs.py`、`pre/core.py`、`pre/pipeline.py`、`pre/fileio.py`、`pre/mapping_table.py`、`EyeEvaluate/Evaluate/metrics.py`、`EyeEvaluate/Evaluate/analysis_pipeline.py` 和 `roi_contract.py`。在其他版本或代码副本中使用前先重新核实。

## 桌面配置 JSON

`desktop_configuration._validate_document(document)` 严格要求 `schema: "localgaze.desktop-config"`、`version: 1`，并且顶层键恰为：`schema`、`version`、`session`、`capture`、`preprocessing`、`roi`、`analysis`。未知顶层/嵌套键会被拒绝。完整结构见 `assets/example.desktop-config.demo.json`，可用 `scripts/validate_config.py` 对真实源码函数验证。没有 `statistics` 键；统计必须放在独立计划与脚本中。GUI 导入仅验证并填充控件，不启动采集、预处理或分析。

严格配置校验不检查路径是否存在，也不验证科学设计。空字符串可通过结构检查，但不能运行。`analysis.metrics` 必须含精确的布尔指标组键；`analysis.parameters` 必须使用精确参数集，其中 `entropy_grid: [width,height]`。预处理 JSON 只有 GUI 暴露的 7 个字段（见 `eye-workflow.md`），不能添加底层隐藏参数。

## 实际预处理 CLI

真实命令：`python -m pre.run_preprocessing`（或 `pre/run_preprocessing.py`），互斥输入 `--data PATH` / `--mapping PATH`，可选 `--model PATH`、`--output PATH`、`--width INT`、`--height INT`、`--time-unit seconds|milliseconds|microseconds`、`--preset legacy|analysis`。`--mapping` 必须配 `--output`。无 `--config`、`--validate`、`--plan` 参数。`--width/--height` 默认 2560×1440 并形成坐标边界，仅为工程默认，不是屏幕事实。未知时不得把默认写成已知设备事实。

CLI 和公开 pipeline 默认 `legacy`；桌面 job 用 `analysis` 禁用 XY 插值。要与桌面路线一致时必须显式选 `analysis`，不能因参数工具不支持而静默换路线。

## 高级 API 边界

`desktop_jobs` 内部 worker payload 进入 `preprocess` 时，将 `time_unit` 与 `preprocess_config` 交给 `PreprocessConfig`。这是内部 Python job API，不是 CLI，也不是承诺稳定的跨版本接口。只有核实目标 checkout 的 `_payload` 校验和 dispatch 后才能使用；记录确切 payload 和 preset。否则保留 UI 暴露参数或走明确验证过的 Python API，并披露实际默认值。绝不声称严格 GUI JSON 携带了隐藏值。

`EyeEvaluate.Evaluate.analysis_pipeline.analyze_file(source_path, output_root, *, rois=(), roi_path=None, time_unit=None, config=None, selected_metrics=None, table_format="csv", session_id=None)` 读取原始规范化眼动数据、调用 `analyze_samples` 并导出分析表；它不读取通用配置路径，也不做推断统计。统计脚本必须接入真实方法/库和明确表契约。

## 唯一可用结构校验命令

从目标项目根目录执行：

运行时提供技能脚本、配置文件和目标项目根目录的实际路径：`python <技能目录>/scripts/validate_config.py <配置文件> --project-root <LocalGaze项目根目录>`。脚本会从指定项目根目录导入真实配置校验函数，不依赖固定安装位置。

脚本用严格 JSON 解析（重复键和 NaN/Infinity 均拒绝），再调用目标源码的 `_validate_document`。输出 `STRUCTURE_LOADABLE` 只表示结构可加载；不表示 GUI 或数据执行成功。可加 `--ledger file.csv` 做台账列与状态格式检查，台账状态不是验证证据。

## 每项转换的执行文件

清单只用于指定方法，不能代替转换本身。若转换超出目标版 LocalGaze 的原生能力（如设备时钟漂移同步、视觉角度换算、基线差值、宽长表重整或眼间融合），生成可独立运行的 `prepare_data.py` 或小型分阶段脚本，明确输入/输出 schema、列和单位、算法/顺序、参数来源、版本/依赖、错误/缺失策略和新输出路径。脚本必须在合成的设计形状样本上验证边界和不变量，并在有数据时用真实脱敏 pilot 核对；二者任一未完成就标 `NOT_RUN`，禁止把统计启动步骤称为已验证。不得只生成一张转换清单而省略实际转换代码。

便携提示：文中的路径仅为本机示例。解压后请将技能目录换为实际位置，将项目根目录换为目标 LocalGaze checkout；不要把本机绝对路径写死到其他用户的包中。
