# Gaze–PSM fixed-instrument v2

打开 `report.html` 查看中文完整报告。图像内嵌，离线可用。

- GT 来源：各 User/npy/*_gaze.npy 的 gaze_x/gaze_y；仅坐标有限且 x、y 均在 [0,1] 内，忽略 valid/Status。
- gaze_raw 从 NPY 与原模型预测重新生成 41 Hz 输入；原方法、录像集合及全部 8 分支不变。
- 主分析：position / PSM in stage，左右器械分别固定。
- DLC likelihood 和跳变只用于诊断；保留低 likelihood 点，采用轻度平滑。
- 主比较集合仅按技术输入支持确定，不按峰正负、边界或模型可定位性删除条件。
- 曲线 MAE 评价相关曲线复现；signed/absolute lag 差同时报告可定位覆盖及边界比例。
- 组汇总以参与者为单位；跨阶段值明确称 observed-stage average。
- lag_identifiable是操作性规则，尚未验证为确证判定。
- 重采样范围是小样本探索性范围，未验证为95%置信区间。
- 模型原始时序保留，不用测试集最佳平移校准。
- 运行清单、输入/代码哈希、验证与敏感性结果一并保存。

## 结果入口

- `paired_summary.csv`：参与者等权配对汇总与重采样范围。
- `paired_metrics.csv`：全部逐条件同侧比较和状态。
- `record_summary.csv`：每条相关曲线的峰、支持和可靠性。
- `group_summary.csv` / `group_curves.csv`：组曲线及参与者支持。
- `paired_group_summary.csv`：共同完整参与者上的组曲线峰差，区别于逐记录MAE。
- `sensitivity_summary.csv`：每一分支与其matched primary的共同条件比较。
- `unpaired_reference.csv`：他人gaze参照及实际有效抽样数。
- `direct_model_delay.csv`：直接GT–模型相对时序审计。
- `identity_reference.csv`：参与者整体身份重配参照。
- `model_comparisons.csv`：参与者整块模型标签交换及Holm校正。
- `run_manifest.json` / `validation.json`：版本和验收记录。

## 推断边界

当前每任务最多5名参与者，A/C合计7名独立参与者。不得把重复阶段当作独立样本。
完整敏感性方案须一起报告，不能挑选最有利分支。新参与者验证、独立校准和独立器械标注尚需额外数据。


## 本次输入修正与独立核验

- GT 来源统一为 User/npy 中逐帧 NPY；坐标有限且 x、y 均在 [0,1]，不读取 valid/Status 作为筛选条件。
- 保留旧 10 个录像、7 名参与者、GT 与五模型、8 个处理分支和完整阶段集合；组/配对汇总及非配对参照计划均为 5000 次。
- 原模型预测与原生模型均值不变，41 Hz gaze 输入重新构建，消除旧 GT 筛选掩码；共同支持按新 GT 重新确定。
- 1824 条 primary 曲线中，与旧版共享有效点的数值发生改变 1456 条，支持变化 0 条。输入已改变，因此不要求与旧版数值相等。
- 独立 legacy core 在新输入上重算 888 条曲线，最大相关差 0；独立从记录曲线重建 432 条 primary 组曲线，最大差 2.22e-16。
- 原有 39 项单元测试全部通过。逐点原因、源文件指纹和规范分别见 preprocessing_reason_schema.json、input_manifest.json、run_manifest.json。
- 非配对参考完整支持组实际有效次数 906–5000；低于 2000 次目标的组共 12 个，均保留并在 reference_coverage_note.json 披露。
