# Gaze–PSM fixed-instrument v2

打开 `report.html` 查看中文完整报告。图像内嵌，离线可用。

- 主分析：position / psm_stage，左右器械分别固定。
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

## 中文分析报告

请先打开 analysis_report.html 阅读基于本次数据的分析与结论；report.html 为完整技术表图入口。

## 阶段范围修订

本次按用户要求在阶段曲线和随机参照计算之前排除 drop ring，并重新生成所有活动统计与图件。
共移除240条阶段曲线和200条模型配对；剩余8,880条曲线与修订前的对应曲线逐条完全一致，2,000条保留汇总的主要指标和重采样范围最大差为0。
主position/psm_stage现在共有74个record-stage-side单元，均有共同完整输入支持，构成370条模型配对。
整视频时序审计保留全视频定义；原始视频、gaze、DLC和prediction不变。
逐文件核验见stage_removal_audit.json。

## 验证与复现

34项测试通过；888条旧规则曲线回归最大差0；1/2 worker结果一致。
在requirements.txt记录的包版本环境中，运行code_snapshot/analyze_model_xcorr_dlc_v2.py，传入--source原始max-DLC结果目录和--output新空目录，即按本次默认范围与参数计算。
验证脚本为code_snapshot/validate_model_xcorr_dlc_v2.py。
中文分析报告生成器为analysis_code_snapshot/build_gaze_psm_analysis_report_v2.py，传入--output数值运行目录。该报告针对已审计十视频结果撰写，若主要数值改变会拒绝复用结论。
DLC likelihood与跳变只作QC，未剔除或降权；区间仅为未校准的探索性重采样范围。
