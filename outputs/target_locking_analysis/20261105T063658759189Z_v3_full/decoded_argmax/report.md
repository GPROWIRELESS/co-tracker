# Target Locking 分支结果

分析分支为 decoded_argmax。

L_gripper / gripper 事件表示双侧 DLC 腕部或铰接点邻域占用（DLC wrist-region occupancy），并非经过验证的器械尖端。L_P 是 wrist-only 独占区域，L_O 是 target 与该腕部邻域的重叠。

本分支使用 argmax 解码得到的模型位置，计算点占用、硬事件错误、混淆矩阵以及解码轨迹的驻留和切换。gaze 来源保留实际观测轨迹。

所有时间指标在导出列名及 metric 值中带有 decoded_argmax_ 前缀。前缀标识本次比较的解码分支；gaze 行仍表示真实 gaze 的时间指标。

统计使用 10,000 次参与者 bootstrap，随机种子为 42。模型差值按共同试次配对后在参与者内平均。BSS 使用参与者等权平均 Brier 的比值。相关分析先在指标与分数均有效的同一组试次内形成参与者均值；相关及阶段、轮次结果为描述和探索性结果。

BSS 的参照为 train_heatmap、center_point、uniform_area 和 train_rate。各参照的逐试次 Brier 单独保存在 baseline_reference_bytrial.csv，其中 GT_smoothed 列是平滑真实 gaze 的诊断参考。GT_smoothed 的独立汇总位于 GT_smoothed_diagnostic_summary.csv，不作为训练基线或模型上限，也不进入 BSS 分母和模型参照优劣图。

## 输出索引

- [trial_stage.csv](trial_stage.csv)，3,600 行
- [model_trial_errors.csv](model_trial_errors.csv)，1,250 行
- [participant_stage_summary.csv](participant_stage_summary.csv)，2,050 行
- [group_stage_summary.csv](group_stage_summary.csv)，8,050 行
- [participant_model_agreement.csv](participant_model_agreement.csv)，4,479 行
- [performance_correlations.csv](performance_correlations.csv)，2,800 行
- [performance_participants.csv](performance_participants.csv)，14,854 行
- [performance_leave_one_out.csv](performance_leave_one_out.csv)，14,854 行
- [participant_model_errors.csv](participant_model_errors.csv)，1,250 行
- [model_error_summary.csv](model_error_summary.csv)，8,250 行
- [model_paired_contrasts.csv](model_paired_contrasts.csv)，8,500 行
- [participant_round_summary.csv](participant_round_summary.csv)，3,150 行
- [repetition_contrasts.csv](repetition_contrasts.csv)，15,750 行
- [baseline_reference_bytrial.csv](baseline_reference_bytrial.csv)，250 行
- [baseline_reference_summary.csv](baseline_reference_summary.csv)，400 行
- [GT_smoothed_diagnostic_summary.csv](GT_smoothed_diagnostic_summary.csv)，100 行
- [trial_confusions.csv](trial_confusions.csv)，20,000 行
- [confusion_summary.csv](confusion_summary.csv)，4,000 行

## 图表

- [figures/paired_test/task_A_regions.png](figures/paired_test/task_A_regions.png)
- [figures/paired_test/task_A_performance_C.png](figures/paired_test/task_A_performance_C.png)
- [figures/paired_test/task_A_agreement_C.png](figures/paired_test/task_A_agreement_C.png)
- [figures/paired_test/task_A_performance_L_target.png](figures/paired_test/task_A_performance_L_target.png)
- [figures/paired_test/task_A_agreement_L_target.png](figures/paired_test/task_A_agreement_L_target.png)
- [figures/paired_test/task_A_performance_L_gripper.png](figures/paired_test/task_A_performance_L_gripper.png)
- [figures/paired_test/task_A_agreement_L_gripper.png](figures/paired_test/task_A_agreement_L_gripper.png)
- [figures/paired_test/task_A_dwell_switch.png](figures/paired_test/task_A_dwell_switch.png)
- [figures/paired_test/task_A_repetition.png](figures/paired_test/task_A_repetition.png)
- [figures/paired_test/task_A_sensitivity.png](figures/paired_test/task_A_sensitivity.png)
- [figures/paired_test/task_A_brier_skill.png](figures/paired_test/task_A_brier_skill.png)
- [figures/paired_test/task_A_confusion.png](figures/paired_test/task_A_confusion.png)
- [figures/paired_test/task_C_regions.png](figures/paired_test/task_C_regions.png)
- [figures/paired_test/task_C_performance_C.png](figures/paired_test/task_C_performance_C.png)
- [figures/paired_test/task_C_agreement_C.png](figures/paired_test/task_C_agreement_C.png)
- [figures/paired_test/task_C_performance_L_target.png](figures/paired_test/task_C_performance_L_target.png)
- [figures/paired_test/task_C_agreement_L_target.png](figures/paired_test/task_C_agreement_L_target.png)
- [figures/paired_test/task_C_performance_L_gripper.png](figures/paired_test/task_C_performance_L_gripper.png)
- [figures/paired_test/task_C_agreement_L_gripper.png](figures/paired_test/task_C_agreement_L_gripper.png)
- [figures/paired_test/task_C_dwell_switch.png](figures/paired_test/task_C_dwell_switch.png)
- [figures/paired_test/task_C_repetition.png](figures/paired_test/task_C_repetition.png)
- [figures/paired_test/task_C_sensitivity.png](figures/paired_test/task_C_sensitivity.png)
- [figures/paired_test/task_C_brier_skill.png](figures/paired_test/task_C_brier_skill.png)
- [figures/paired_test/task_C_confusion.png](figures/paired_test/task_C_confusion.png)
- [figures/real_all/task_A_regions.png](figures/real_all/task_A_regions.png)
- [figures/real_all/task_A_performance_C.png](figures/real_all/task_A_performance_C.png)
- [figures/real_all/task_A_performance_L_target.png](figures/real_all/task_A_performance_L_target.png)
- [figures/real_all/task_A_performance_L_gripper.png](figures/real_all/task_A_performance_L_gripper.png)
- [figures/real_all/task_A_dwell_switch.png](figures/real_all/task_A_dwell_switch.png)
- [figures/real_all/task_A_repetition.png](figures/real_all/task_A_repetition.png)
- [figures/real_all/task_A_sensitivity.png](figures/real_all/task_A_sensitivity.png)
- [figures/real_all/task_C_regions.png](figures/real_all/task_C_regions.png)
- [figures/real_all/task_C_performance_C.png](figures/real_all/task_C_performance_C.png)
- [figures/real_all/task_C_performance_L_target.png](figures/real_all/task_C_performance_L_target.png)
- [figures/real_all/task_C_performance_L_gripper.png](figures/real_all/task_C_performance_L_gripper.png)
- [figures/real_all/task_C_dwell_switch.png](figures/real_all/task_C_dwell_switch.png)
- [figures/real_all/task_C_repetition.png](figures/real_all/task_C_repetition.png)
- [figures/real_all/task_C_sensitivity.png](figures/real_all/task_C_sensitivity.png)
