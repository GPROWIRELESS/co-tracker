# Target Locking v2：概率质量主分析与独立解码分析

运行：`20261105T063658759189Z_v2_full`。主半径 target=gripper=50 px；主 gaze 状态为 2/3，3-only 与旧1/2/3规则另作敏感性。

## 本次变化及评价边界

- 主分析直接统计每帧 K=64 个原始模型 samples 的四区域命中比例。它估计模型采样分布的质量，不将热图压成一个点。期望点和 argmax 来自保存的 probability；有限 K 样本的经验分布与保存的概率图不必完全一致，解码差异不能全部归因于点选择规则。
- 期望点、argmax 和固定随机种子的逐帧采样点均在本次相同区域、相同共同有效帧上重算。旧运行数值没有拼入新表。
- K列在不同帧间没有联合轨迹身份。decoded_sample 的驻留/切换只是独立随机解码诊断，不代表模型预测分布的真实时间连贯性。
- 原始DLC点被如实称为腕部/铰接邻域，不能冒称jaw tip或完整器械。target保留人工/跟踪槽位算术平均，不以未经确认的物体身份重新指定坐标。区域候选与语义审计另见附带文件。
- 50 px仍是预设空间容差，当前没有独立眼动校准误差95百分位证据。不能把投影校准矩阵当作定位准确率验证。
- train_heatmap直接保留并积分完整训练平均热图；train_rate仅由训练录像拟合。center_point是确定性点，uniform_area是均匀随机落点的软概率，二者不同。
- GT_smoothed由真实gaze构建，是标签平滑的诊断参照，不是严格上界，不参与BSS排名。
- 所有模型仅按留出录像评价，同人其他录像可在训练集。参与者bootstrap条件于当前模型；模型采样MC误差没有由参与者bootstrap完整覆盖。K=64单帧概率步长1/64，二项抽样标准差理论最大0.0625。

## 主概率分析：全任务均值

| 集合 | 任务 | 来源 | Target | Gripper | Overlap | D | C |
|---|---|---|---:|---:|---:|---:|---:|
| paired_test | A | all_ABCD | 0.0644 | 0.0713 | 0.0061 | 0.1234 | 0.4715 |
| paired_test | A | gaze | 0.2268 | 0.0978 | 0.0213 | 0.2819 | 0.7505 |
| paired_test | A | leave_out | 0.0494 | 0.0494 | 0.0049 | 0.0890 | 0.5060 |
| paired_test | A | single_task | 0.0682 | 0.0732 | 0.0064 | 0.1286 | 0.4810 |
| paired_test | A | spd_finetune | 0.0652 | 0.0705 | 0.0060 | 0.1236 | 0.4786 |
| paired_test | A | spd_zero_shot | 0.0518 | 0.0388 | 0.0040 | 0.0826 | 0.5813 |
| paired_test | C | all_ABCD | 0.1185 | 0.1466 | 0.0554 | 0.1543 | 0.4122 |
| paired_test | C | gaze | 0.2595 | 0.2131 | 0.1152 | 0.2422 | 0.5723 |
| paired_test | C | leave_out | 0.0629 | 0.0949 | 0.0292 | 0.0993 | 0.3489 |
| paired_test | C | single_task | 0.1208 | 0.1467 | 0.0567 | 0.1542 | 0.4200 |
| paired_test | C | spd_finetune | 0.1115 | 0.1365 | 0.0515 | 0.1449 | 0.4155 |
| paired_test | C | spd_zero_shot | 0.0870 | 0.1383 | 0.0413 | 0.1427 | 0.3216 |
| real_all | A | gaze | 0.2498 | 0.0946 | 0.0274 | 0.2897 | 0.7679 |
| real_all | C | gaze | 0.2468 | 0.2335 | 0.1184 | 0.2435 | 0.5141 |

## 概率事件误差

|任务|模型|ΔTarget|ΔC|Target Brier|Gripper Brier|
|---|---|---:|---:|---:|---:|
| A | all_ABCD | -0.1624 | -0.2789 | 0.1927 | 0.0868 |
| A | leave_out | -0.1774 | -0.2444 | 0.1997 | 0.0885 |
| A | single_task | -0.1586 | -0.2695 | 0.1920 | 0.0870 |
| A | spd_finetune | -0.1616 | -0.2719 | 0.1933 | 0.0869 |
| A | spd_zero_shot | -0.1750 | -0.1691 | 0.2032 | 0.0912 |
| C | all_ABCD | -0.1410 | -0.1601 | 0.1931 | 0.1543 |
| C | leave_out | -0.1966 | -0.2234 | 0.2240 | 0.1730 |
| C | single_task | -0.1386 | -0.1523 | 0.1885 | 0.1518 |
| C | spd_finetune | -0.1480 | -0.1568 | 0.1959 | 0.1556 |
| C | spd_zero_shot | -0.1724 | -0.2507 | 0.2142 | 0.1638 |

## 结果导航

- [详细中文分析报告](detailed_report_zh.md)：逐任务、阶段、模型、表现相关及敏感性结果。
- [主概率分析](main_probability/)：参与者/阶段/模型比较、表现相关、bootstrap和图件。
- [期望点解码](decoded_expectation/) · [argmax解码](decoded_argmax/) · [随机采样点解码](decoded_sample/)。
- [覆盖](coverage.csv) · [来源审计](recording_audit.csv) · [预测审计](prediction_audit.csv) · [训练参照](baseline_input_audit.json)。
- [GT平滑诊断](gt_smoothed_occupancy.csv) · [验证](validation.json) · [参数和代码哈希](run_config.json)。

原始与敏感性结果必须按analysis_branch/scenario分开使用；不得用旧运行均值替代新运行gaze对照。
表现相关继续是小样本描述分析；特别是Task C测试分数仅两个不同取值，不据此主张技能识别泛化。
