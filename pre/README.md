# LocalGaze 独立眼动预处理

本目录把原 AIPT 眼动预处理方法改写为可独立运行的 Python 模块。只需给出模型位置和数据位置，程序会按文件名（不含扩展名、忽略大小写）严格一一匹配并逐个处理。

## 快速运行

```powershell
python .\pre\run_preprocessing.py `
  --model "F:\数据\被试1\Model_history" `
  --data "F:\数据\被试1\Data"
```

单文件也可以直接处理：

```powershell
python .\pre\run_preprocessing.py `
  --model "F:\数据\被试1\Model_history\A10-AUT.xlsx" `
  --data "F:\数据\被试1\Data\A10-AUT.xlsx" `
  --output "F:\数据\被试1\Data_preprocessed"
```

默认屏幕范围为 2560 × 1440。若采集屏幕不同，请增加 `--width` 和 `--height`。纯数值时间列默认按秒解释，可用 `--time-unit milliseconds` 或 `microseconds` 修改。

## 匹配和输出规则

- `A10-AUT.xlsx` 只匹配模型目录中的 `A10-AUT.xlsx` 或同名 `A10-AUT.pkl`。
- 如果同一目录树出现两个同名数据文件，程序停止并报告歧义，不猜测匹配。
- 默认输出到数据目录旁的 `Data_preprocessed`，不覆盖原始数据。
- `.xlsx/.xlsm` 保持 Excel 格式和原工作表，在眼动数据表右侧追加 `Pre_` 列，并增加 `Pre_QC`、`Pre_Windows`、`Pre_BadSegments` 和 `Pre_SuspectSegments` 工作表。
- `.csv/.tsv/.txt` 保持原扩展名和分隔符，在右侧追加 `Pre_` 列，并另存同名 `.qc.json`。
- 每批输出 `preprocess_report.csv` 和 `preprocess_manifest.json`。

## 当前支持的输入字段

必需字段会自动识别中英文别名：时间、x、y。可选字段为 blink、左瞳孔直径和右瞳孔直径。Excel 会优先读取 `眼动数据` 工作表，也会自动检查其他工作表。

## 方法范围

当前实现包括时间检查、非有限值和屏幕外标记、眨眼保护区、孤立往返尖峰识别、速度候选、500 ms/100 ms 质量窗口、短缺口线性插值、60 ms 最多五点中值平滑、瞳孔突变标记、短缺口插值和 4 Hz 对称高斯低通处理。

模型校准 Excel 的 `AdditionalCalibration.MeanErrorPerRound` 最后一个有效值优先用作坐标插值端点距离；缺少该值时回退为 `NinePointCalibration.Error` 均值。若模型没有可用校准误差，坐标插值关闭，其余预处理仍可执行并在 QC 中记录原因。

质量窗口长度、比例、MAD 倍数、平滑窗口和自拟坏段规则属于项目候选参数，需要结合人工标注片段和真实数据质量检查冻结，不能表述为文献已验证的通用阈值。

