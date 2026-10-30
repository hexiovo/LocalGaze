# 实验设计资料

本目录随 LocalGaze Windows 便携包提供。`localgaze-study-config/` 包含完整技能源、参考资料、配置校验脚本和模板；`实验设计填写模板.md` 适合先整理研究信息，`localgaze-study-config/assets/study-intake-template.csv` 可用于逐项登记。

LocalGaze 本身不需要安装此技能。若要在 Codex 中使用，可将 `localgaze-study-config/` 文件夹复制到你自己的技能目录，再从技能选择器调用。目录位置由你的 Codex 设置决定，本资料不写入固定机器路径。

先填写研究问题、主要结局、实验单位、设备和数据格式。每项记录数值、单位、来源、状态和风险；状态使用 `known`、`proposed`、`missing` 或 `not_applicable`。缺少会影响正确执行或结果解释的信息时，应停在草案并补充资料。

技能会依据当前 LocalGaze 源码核对配置与执行入口。严格 JSON 检查只说明配置结构可加载，不证明方法科学、数据质量、GUI 执行或统计结果正确。LocalGaze 不提供通用推断统计引擎；每项研究都需要独立统计计划和匹配设计的脚本。不要把候选指标或软件自检写成真实精度证据。

发布附件中的单文件版包含本目录说明、填写模板和技能的全部参考正文及资源说明；ZIP 包含相同的可维护文件树。两者都不依赖本机绝对路径。
