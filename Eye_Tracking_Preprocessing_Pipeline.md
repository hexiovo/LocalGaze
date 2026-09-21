# Comprehensive Preprocessing Pipeline for Eye Tracking and Pupillometry Data (22-Step Detailed Version)

以下流程综合了主流眼动与瞳孔信号处理框架（如 *PupilMetrics*, *PUPILS*, *GraFIX*, *Pupil Labs*, *EyeLink DataViewer*, *iMotion Sensors* 等），融合了认知心理学、神经科学和人机交互领域的最新研究成果，旨在提供一个从原始信号到可计算指标前的完整预处理体系。

---

### Step 1. 原始数据导入与结构检查
**目标：** 确认输入数据完整、时间同步无误、字段名称规范。  
**方法：**
- 导入时间序列（x, y, pupil_left, pupil_right, blink）  
- 检查采样率一致性与时间戳间隔（Δt稳定性）  
- 校验缺失值比例与单位一致性（像素或度量单位）  
- 自动检测文件格式（.csv, .edf, .asc, .tsv等）  

**参考文献：** Fink et al. (2023); Coe et al. (2024); Mathôt & Vilotijević (2022)

---

### Step 2. 时间同步与时钟漂移修正
**目标：** 确保眼动信号与实验事件同步。  
**方法：**
- 使用事件标记（Triggers/TTL）进行时间对齐  
- 校正采样时间漂移（线性插值或多项式漂移模型）  
- 当多模态信号存在（EEG/fNIRS）时使用交叉相关法校准  

**参考文献：** Ong et al. (2025); Nobukawa et al. (2024)

---

### Step 3. 异常时间段剔除与设备初始化噪声去除
**目标：** 去除启动/结束阶段的非稳定采样信号。  
**方法：**
- 根据采样速率与瞳孔直径稳定区间自动检测  
- 去除记录前后数秒或无效数据块  
- 可视化确认设备稳定点  

**参考文献：** Fink et al. (2023); Kret & Sjak-Shie (2019)

---

### Step 4. 缺失值检测与分类
**目标：** 区分自然眨眼引起的缺失与技术丢帧。  
**方法：**
- 依据blink标志或瞳孔直径突降识别眨眼段  
- 记录缺失数据比例并绘制分布直方图  
- 标记“短时缺失（≤150 ms）”与“长时缺失（>150 ms）”  

**参考文献：** Mathôt & Vilotijević (2022); Hershman et al. (2018)

---

### Step 5. 缺失值插值
**目标：** 恢复连续时间序列，为滤波和平滑准备。  
**可选方法：**
- 线性插值（短眨眼段）  
- 样条插值（平滑度高）  
- Savitzky–Golay滤波插值  
- Blink-aware插值（结合blink边界延伸处理）  

**参考文献：** Fink et al. (2023); Mathôt (2018); Saez de Urabain et al. (2015)

---

### Step 6. 去眨眼边缘伪影
**目标：** 消除眨眼前后边缘瞳孔突变伪影。  
**方法：**
- 眨眼前后各延伸 ±100 ms 设为NaN再插值  
- 使用滑动窗检测瞳孔突降速率  
- 限制恢复段变化率  

**参考文献：** Kret & Sjak-Shie (2019); Hershman et al. (2018)

---

### Step 7. 异常值与漂移检测
**目标：** 检测由于光照或校准误差造成的异常漂移。  
**方法：**
- 使用移动中位数与MAD（Median Absolute Deviation）检测异常点  
- 基于时间窗的局部线性回归校正  
- 可选：Loess平滑模型评估全程趋势  

**参考文献：** Fink et al. (2023); Coe et al. (2024)

---

### Step 8. 平滑与去噪
**目标：** 降低高频噪声，保持信号真实波动。  
**可选方法：**
- 滑动平均（window=5–11 samples）  
- Savitzky–Golay滤波器  
- Butterworth低通滤波（cutoff≈4–6 Hz）  
- Wavelet小波平滑（DWT/DB4）  

**参考文献：** Amiot et al. (2024); Fink et al. (2023); Mathôt & Vilotijević (2022)

---

### Step 9. 瞳孔直径双眼整合
**目标：** 合并双眼信号获得稳健估计。  
**方法：**
- 均值法：`(left + right)/2`  
- 加权法：依据信号质量（丢失比例）加权  
- 异常眼剔除法：若一眼异常则仅使用另一眼  

**参考文献：** Mathôt & Vilotijević (2022); Ong et al. (2025)

---

### Step 10. 瞳孔直径单位转换与标准化
**目标：** 统一单位（mm或相对值）。  
**方法：**
- 若为像素单位，依据校准转换为mm  
- 归一化至个体内Z分数或比例尺度  

**参考文献：** Fink et al. (2023); Hershman et al. (2018)

---

### Step 11. 瞳孔基线校正
**目标：** 消除任务前个体差异影响。  
**方法：**
- 以刺激前500 ms平均为基线  
- 或以任务间歇平均为基线  
- 输出Δpupil或相对变化百分比  

**参考文献：** Mathôt & Vilotijević (2022); Fink et al. (2023)

---

### Step 12. 坐标空间校准
**目标：** 将(x, y)像素坐标转换为视角度坐标。  
**方法：**
- 根据屏幕分辨率与观看距离计算每像素角度  
- 应用9点或13点校准模型  
- 平面几何变换（仿射或多项式）  

**参考文献：** Coe et al. (2024); Holmqvist et al. (2011)

---

### Step 13. 失准（Drift）校正
**目标：** 修正随时间推移的视线偏移。  
**方法：**
- 基于固定点重标定（recalibration）  
- 拟合漂移模型（线性/二次）并平移补偿  
- 事件锁定校正：以刺激中央点重对齐  

**参考文献：** Fink et al. (2023); Dalmaijer et al. (2014)

---

### Step 14. 注视点检测（Fixation Detection）
**目标：** 划分注视与扫视阶段。  
**方法：**
- I-DT（Dispersion-Threshold）算法  
- I-VT（Velocity-Threshold）算法  
- Adaptive I-VT（动态阈值）  
- Cluster-based方法（DBSCAN/K-means）  

**参考文献：** Saez de Urabain et al. (2015); Holmqvist et al. (2011)

---

### Step 15. 扫视检测（Saccade Detection）
**目标：** 识别快速眼跳。  
**方法：**
- 速度阈值法（>30°/s）  
- 角加速度法（结合速度与方向变化）  
- 基于机器学习模型分类（如SVM）  

**参考文献：** Coe et al. (2024); Fink et al. (2023)

---

### Step 16. 注视持续时间与瞳孔联合特征提取
**目标：** 计算平均注视时长、注视内瞳孔直径均值。  
**方法：**
- 按注视段计算平均瞳孔值、持续时间  
- 计算个体内变异系数（CV）  
- 绘制注视热图与瞳孔分布图  

**参考文献：** Fink et al. (2023); Amiot et al. (2024)

---

### Step 17. ROI映射与空间标注
**目标：** 将注视点映射到ROI范围。  
**方法：**
- 使用多边形包含判断（point-in-polygon）  
- 统计ROI内注视次数、停留时间、瞳孔平均值  
- 计算ROI转移矩阵（transition matrix）  

**参考文献：** Saez de Urabain et al. (2015); Mathôt & Vilotijević (2022)

---

### Step 18. Blink特征分析
**目标：** 提取眨眼频率、持续时间等参数。  
**方法：**
- 基于blink标记段统计频率（次/分钟）  
- 眨眼恢复时间与幅度评估  
- 可选：blink-triggered pupil response  

**参考文献：** Coe et al. (2024); Hershman et al. (2018)

---

### Step 19. 信号标准化与个体间对齐
**目标：** 消除个体差异，实现群体比较。  
**方法：**
- Z-score标准化或百分位变换  
- 使用个体内最大/最小归一化  
- 对齐事件时序（时间归零）  

**参考文献：** Fink et al. (2023); Mathôt & Vilotijević (2022)

---

### Step 20. 多模态数据同步（若存在）
**目标：** 对齐脑电/fNIRS等信号。  
**方法：**
- 事件触发信号交叉相关法  
- 时间戳插值对齐  
- 统一采样率重采样  

**参考文献：** Nobukawa et al. (2024); Ong et al. (2025)

---

### Step 21. 数据可视化与质量评估
**目标：** 直观检验预处理效果。  
**方法：**
- 绘制瞳孔时间序列、注视轨迹  
- 热图与密度图可视化  
- 生成QC报告（丢失比例、插值次数、噪声水平）  

**参考文献：** Amiot et al. (2024); Fink et al. (2023)

---

### Step 22. 最终输出与指标计算准备
**目标：** 输出清洁、平滑、对齐的时间序列数据。  
**方法：**
- 导出标准化DataFrame（时间、x、y、pupil、blink、fixID、ROI）  
- 确认时间对齐与信号质量  
- 准备输入计算指标模块（如注视热图、瞳孔变化率、扫视特征等）  

**参考文献：** Fink et al. (2023); Mathôt & Vilotijević (2022); Coe et al. (2024)

---

**附加经典参考文献：**
- Kret, M. E., & Sjak-Shie, E. E. (2019). Preprocessing pupil size data: Guidelines and code. *Behavior Research Methods*, 51, 1336–1342.  
- Hershman, R., Henik, A., & Cohen, N. (2018). A novel blink detection method based on pupillometry noise. *Behavior Research Methods*, 50, 107–114.  
- Holmqvist, K., Nyström, M., Andersson, R., et al. (2011). *Eye Tracking: A Comprehensive Guide to Methods and Measures.* Oxford University Press.  
- Dalmaijer, E. S., Mathôt, S., & Van der Stigchel, S. (2014). PyGaze: An open-source toolbox for eye tracking. *Behavior Research Methods*, 46, 913–921.
