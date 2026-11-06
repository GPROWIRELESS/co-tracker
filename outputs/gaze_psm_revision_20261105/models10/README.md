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

## 本次扩展与独立核验

- 10个录制、7名独立参与者、GT与五模型；8个预处理/窗口分支。
- 主分析明确命名 fixed_left / fixed_right；GT选侧和独立max的补充曲线、配对及组结果位于 selected_rule_*。
- position主配置3点移动平均；velocity主配置5点二阶SG后差分，完整保留7/11点SG对照。
- 10,032条固定侧曲线、8,360条固定侧配对；另有1,824条补充曲线和1,520条补充配对。
- 38项既有测试与1项补充选侧测试通过；测试记录见 unit_test_results.txt。
- 与上一版model v2的1,824条primary曲线逐项完全一致（最大相关差0）；888条旧已选侧曲线回归最大差0。
- 独立从保存记录曲线重算432条primary组曲线，最大差2.23e-16。
- 质量规则删除数0。逐点原因和时间轴见 native_qc_flags、preprocessing_provenance、preprocessing_reason_schema.json。
- 各组参考计划5000次；完整支持组有效次数906–5000，部分组未达到2000目标，详见 reference_coverage_note.json，不隐瞒或补选掉失败抽样。
- 新验证代码在 validation_code；数值执行快照 code_snapshot 保持运行时原样。independent_validation.json记录独立核验。
