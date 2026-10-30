# 每个实验包的依赖执行编排

每次生成研究包都需要一份研究特定的 `execution_plan.json`（这是本 skill 的项目计划格式，不是 LocalGaze 原生 schema）和中文操作卡。使用有向无环步骤列表；每个步骤含 ID、依赖、状态、负责人、确切输入路径/schema/hash、确切输出路径/schema、真实命令或 GUI 动作、运行时/环境、参数来源、预期现象、成功判据、失败判据、失败后的下一步、保留的日志和人工核对点。任何 `missing` 阻断项会把后续依赖设为 `BLOCKED`，不能输出启动步骤。

按真实设计选择/合并阶段，不适用阶段标 `not_applicable` 并说明理由：

1. `intake`: 固定源文件、设备与设计事实、匿名 ID 和只读边界；生成源文件清单/hash。
2. `prepare`: 对原生不支持的转换运行 `prepare_data.py`；对原生支持的字段/时间读入，明确由 LocalGaze 的哪个真实入口处理。用合成 fixture 检查行数、键、时钟、单位和转换守恒；未检查前置后续阶段保持 `BLOCKED`。
3. `preprocess`（如适用）: 运行目标版已确认的 GUI 或真实 CLI/API，指定 `analysis` 或其他实际 preset；将 `preprocess_report.csv`、`preprocess_manifest.json`、文件 sidecar/QC 表列为可观察输出并核对内容。
4. `features`: 调用真实桌面任务或 `analysis_pipeline.analyze_file`，列出确切输入和导出的 metric/event/ROI 文件。`desktop-config.json` 是控件预设，不是分析数据文件。确认原始实测数据、显式有效状态、ID、屏幕参考和 QC。
5. `derive`: 若主要结果需组合/汇总（例如 ROI 与 fixation candidate 交集），生成有 schema 的真实脚本及规则，不把现有描述性输出误称为该结果。
6. `statistics`: 仅调用研究计划指定的独立脚本，输入为前序确切结果表，输出包括估计、CI、模型诊断、计数、排除和 provenance。
7. `report`: 汇总真实产物、方法、限制、失败和未运行阶段。合成示例不混入实证结果。

依赖链必须直连实际文件，例如：`original/raw.csv -> prepare_data.py -> prepared/long_gaze.csv -> LocalGaze preprocessing -> processed/session.csv + QC manifest -> analyze_file -> analysis/tables/*.csv -> derive_roi_fixations.py -> analysis/subject_summary.csv -> run_statistics.py -> statistics/results.csv`。这是结构示意，实际包须从真实输出源码/日志确认文件名，不能照抄示意中的占位路径。

命令只列目标版本的真实已支持入口。预处理 CLI 必须使用源码文档化参数；桌面操作写逐步菜单/文件动作。若某条 CLI/API 尚未在当前环境成功运行，只写“待验证入口”和预期，状态 `NOT_RUN`，不作为可执行命令。给出每阶段开始前的备份/目录确认、成功/失败现象、停止条件和下一步，不要求主试自行选统计方法。
