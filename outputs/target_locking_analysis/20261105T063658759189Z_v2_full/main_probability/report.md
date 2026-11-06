# Target Locking 分支结果

分析分支为 probability_mc。

L_gripper / gripper 事件表示双侧 DLC 腕部或铰接点邻域占用（DLC wrist-region occupancy），并非经过验证的器械尖端。L_P 是 wrist-only 独占区域，L_O 是 target 与该腕部邻域的重叠。

模型占用由逐帧区域概率质量计算。target 与 DLC wrist-region 事件概率包含 overlap；C 使用独占 target 概率的时长加权和除以两个独占区域概率的时长加权和。先在每个试次内计算，再在参与者内对试次等权平均，参与者之间等权。

事件 Brier 使用模型的区域概率与真实 gaze 的二元命中比较。本分支只输出概率占用、面积校正、事件损失及其统计关系。

统计使用 10,000 次参与者 bootstrap，随机种子为 42。模型差值按共同试次配对后在参与者内平均。BSS 使用参与者等权平均 Brier 的比值。相关分析先在指标与分数均有效的同一组试次内形成参与者均值；相关及阶段、轮次结果为描述和探索性结果。

BSS 的参照为 train_heatmap、center_point、uniform_area 和 train_rate。各参照的逐试次 Brier 单独保存在 baseline_reference_bytrial.csv，其中 GT_smoothed 列是平滑真实 gaze 的诊断参考。GT_smoothed 的独立汇总位于 GT_smoothed_diagnostic_summary.csv，不作为训练基线或模型上限，也不进入 BSS 分母和模型参照优劣图。

## 输出索引

- [trial_stage.csv](trial_stage.csv)，5,040 行
- [model_trial_errors.csv](model_trial_errors.csv)，1,750 行
- [participant_stage_summary.csv](participant_stage_summary.csv)，2,870 行
- [group_stage_summary.csv](group_stage_summary.csv)，8,820 行
- [participant_model_agreement.csv](participant_model_agreement.csv)，6,230 行
- [performance_correlations.csv](performance_correlations.csv)，3,920 行
- [performance_participants.csv](performance_participants.csv)，20,754 行
- [performance_leave_one_out.csv](performance_leave_one_out.csv)，20,754 行
- [participant_model_errors.csv](participant_model_errors.csv)，1,750 行
- [model_error_summary.csv](model_error_summary.csv)，8,400 行
- [model_paired_contrasts.csv](model_paired_contrasts.csv)，5,600 行
- [participant_round_summary.csv](participant_round_summary.csv)，4,410 行
- [repetition_contrasts.csv](repetition_contrasts.csv)，14,700 行
- [baseline_reference_bytrial.csv](baseline_reference_bytrial.csv)，350 行
- [baseline_reference_summary.csv](baseline_reference_summary.csv)，560 行
- [GT_smoothed_diagnostic_summary.csv](GT_smoothed_diagnostic_summary.csv)，140 行

## 图表

- [figures/paired_test/task_A_regions.png](figures/paired_test/task_A_regions.png)
- [figures/paired_test/task_A_performance_C.png](figures/paired_test/task_A_performance_C.png)
- [figures/paired_test/task_A_agreement_C.png](figures/paired_test/task_A_agreement_C.png)
- [figures/paired_test/task_A_performance_L_target.png](figures/paired_test/task_A_performance_L_target.png)
- [figures/paired_test/task_A_agreement_L_target.png](figures/paired_test/task_A_agreement_L_target.png)
- [figures/paired_test/task_A_performance_L_gripper.png](figures/paired_test/task_A_performance_L_gripper.png)
- [figures/paired_test/task_A_agreement_L_gripper.png](figures/paired_test/task_A_agreement_L_gripper.png)
- [figures/paired_test/task_A_repetition.png](figures/paired_test/task_A_repetition.png)
- [figures/paired_test/task_A_sensitivity.png](figures/paired_test/task_A_sensitivity.png)
- [figures/paired_test/task_A_brier_skill.png](figures/paired_test/task_A_brier_skill.png)
- [figures/paired_test/task_C_regions.png](figures/paired_test/task_C_regions.png)
- [figures/paired_test/task_C_performance_C.png](figures/paired_test/task_C_performance_C.png)
- [figures/paired_test/task_C_agreement_C.png](figures/paired_test/task_C_agreement_C.png)
- [figures/paired_test/task_C_performance_L_target.png](figures/paired_test/task_C_performance_L_target.png)
- [figures/paired_test/task_C_agreement_L_target.png](figures/paired_test/task_C_agreement_L_target.png)
- [figures/paired_test/task_C_performance_L_gripper.png](figures/paired_test/task_C_performance_L_gripper.png)
- [figures/paired_test/task_C_agreement_L_gripper.png](figures/paired_test/task_C_agreement_L_gripper.png)
- [figures/paired_test/task_C_repetition.png](figures/paired_test/task_C_repetition.png)
- [figures/paired_test/task_C_sensitivity.png](figures/paired_test/task_C_sensitivity.png)
- [figures/paired_test/task_C_brier_skill.png](figures/paired_test/task_C_brier_skill.png)
- [figures/real_all/task_A_regions.png](figures/real_all/task_A_regions.png)
- [figures/real_all/task_A_performance_C.png](figures/real_all/task_A_performance_C.png)
- [figures/real_all/task_A_performance_L_target.png](figures/real_all/task_A_performance_L_target.png)
- [figures/real_all/task_A_performance_L_gripper.png](figures/real_all/task_A_performance_L_gripper.png)
- [figures/real_all/task_A_repetition.png](figures/real_all/task_A_repetition.png)
- [figures/real_all/task_A_sensitivity.png](figures/real_all/task_A_sensitivity.png)
- [figures/real_all/task_C_regions.png](figures/real_all/task_C_regions.png)
- [figures/real_all/task_C_performance_C.png](figures/real_all/task_C_performance_C.png)
- [figures/real_all/task_C_performance_L_target.png](figures/real_all/task_C_performance_L_target.png)
- [figures/real_all/task_C_performance_L_gripper.png](figures/real_all/task_C_performance_L_gripper.png)
- [figures/real_all/task_C_repetition.png](figures/real_all/task_C_repetition.png)
- [figures/real_all/task_C_sensitivity.png](figures/real_all/task_C_sensitivity.png)
