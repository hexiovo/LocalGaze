---
name: localgaze-study-config
description: 将眼动实验设计转成可追溯的 LocalGaze 配置、统计执行脚本和零基础操作包；其他模态须先核实其执行器契约。
---

# LocalGaze 实验配置

把实验设计转成适用于当前 LocalGaze 项目的眼动工作流。其他传感器或模态不得沿用眼动参数；先核实目标执行器的真实输入、配置和运行契约。

## 必读资料

按需阅读 [眼动流程与指标契约](references/eye-workflow.md)、[数据转换清单](references/data-preparation.md)、[统计计划与脚本要求](references/statistical-execution.md)、[执行器真实合同](references/executor-contract.md) 和 [零基础操作手册](references/operator-runbook.md)。具体命令或 JSON 结构必须以执行器真实合同为准。

## 生成流程

1. 用匿名输入表 `assets/study-intake-template.csv` 收集实验信息。只追问会阻断安全执行或正确解释的关键问题，其余先写成草案并标明 `proposed` / `missing` / `not_applicable`。
2. 生成严格符合 `localgaze.desktop-config` v1 的完整桌面 JSON、逐参数台账、转换台账、独立统计计划和仅针对该设计的统计脚本。台账对每个值记录单位、来源、风险和状态：`known`、`proposed`、`missing`、`not_applicable`。统计计划不得写入桌面 JSON。
3. 保持原始数据只读；所有输出写入新的空目录。每一步写清操作、预期现象、通过/失败条件和失败后的下一步。关键事实缺失时只给 `DRAFT / NOT_RUN`，不提供启动步骤。
4. 对目标项目运行 `scripts/validate_config.py`。它检查严格 JSON 并调用当前项目的 `desktop_configuration._validate_document`。这只证明结构可加载，不证明方法科学、数据质量、GUI 执行或统计执行。
5. 如无可用工具或运行环境，只输出文件/明确内容和 `NOT_RUN`，说明主试下一步。不得声称已运行，也不得承诺零错误。

## 证据等级

只能按真实证据报告：`DRAFT`、`STRUCTURE_LOADABLE`、`ENGINEERING_DEMO_ONLY`、`REAL_PILOT_PASSED` 或 `REAL_EXECUTABLE`。参数台账和结构检查不能冒充执行验证。当前 LocalGaze 有描述统计与眼动特征管线，但没有通用推断统计引擎；因此每个研究须单独生成统计计划与真实可运行脚本，验证不过就阻断统计执行。

## 运行边界

桌面配置导入只填充控件，不会启动任务。CLI 只接受项目实际文档化的参数；不得编造 `--config`、`validate`、`plan` 等命令。高级参数必须通过实际 GUI 或经过目标版本源码验证的 API 传入，不能静默丢弃或回退。

默认用中文输出方案、台账说明和逐步操作卡；参数名、JSON键及工具命令保持原样。单文件粘贴稿必须内含全部必要参考正文、示例与附件规则，不依赖未附带文件。

## 每次生成的研究包

除上面的桌面 JSON 与统计计划外，必须给研究包一份有依赖关系的阶段执行编排、中文操作卡、转换台账、环境记录和数据/结果文件衔接图。凡原生执行器不支持的同步、重整、基线、单位或数据结构转换，必须按真实输入输出合同生成独立 `prepare_data.py`（或分阶段转换脚本），写出每步输入、输出、参数、依赖、命令、预期现象、验收和失败动作；用设计形状的合成数据检查脚本，再用 pilot 检查真实设备链路。不能只给转换清单或伪造 CLI。合成检查和 pilot 都未完成时，清楚标记 `NOT_RUN`，统计不可称为可执行。

所有文件和命令必须串成实际链路：转换产物 → LocalGaze 预处理/分析实际可用入口 → 真实导出表 → 研究专用统计脚本 → 整理结果。每阶段写明依赖和确切文件名/位置，不支持的步骤标阻断，不能把配置文件误当数据输入。
必读 [阶段编排](references/stage-orchestration.md)：每个研究包须逐步连接真实转换产物、LocalGaze 输入、统计脚本和结果文件。
核对源码差异时也读 references/source-contract-fingerprint.md；版本/文件指纹改变就先重新检查。
