# Target Locking 分析结果

本次成功读取 84 段录像及 50 份预测。主半径 50 px，独立单位为参与者。

| 集合 | 任务 | 人数 | 录像数 | 有效帧 | 有效秒数 |
|---|---|---:|---:|---:|---:|
| paired_test | A | 5 | 5 | 58893 | 1256.0 |
| paired_test | C | 5 | 5 | 36267 | 837.8 |
| real_all | A | 11 | 33 | 369366 | 8668.1 |
| real_all | C | 11 | 51 | 338323 | 8057.6 |

## 主结果

表中 occupancy、D、C 均先在参与者内对可用 trial 等权，再对参与者等权。各列分别按可估计数据汇总。

| 集合 | 任务 | 来源 | Target occupancy | Gripper occupancy | Overlap | D | C |
|---|---|---|---:|---:|---:|---:|---:|
| paired_test | A | all_ABCD | 0.0820 | 0.0821 | 0.0119 | 0.1402 | 0.5140 |
| paired_test | A | gaze | 0.2261 | 0.0976 | 0.0213 | 0.2811 | 0.7505 |
| paired_test | A | leave_out | 0.0673 | 0.0452 | 0.0076 | 0.0973 | 0.6476 |
| paired_test | A | single_task | 0.1153 | 0.0714 | 0.0168 | 0.1531 | 0.6301 |
| paired_test | A | spd_finetune | 0.1006 | 0.0733 | 0.0180 | 0.1380 | 0.5936 |
| paired_test | A | spd_zero_shot | 0.0427 | 0.0113 | 0.0005 | 0.0530 | 0.7831 |
| paired_test | C | all_ABCD | 0.1988 | 0.2070 | 0.1028 | 0.2003 | 0.4667 |
| paired_test | C | gaze | 0.2592 | 0.2129 | 0.1151 | 0.2419 | 0.5722 |
| paired_test | C | leave_out | 0.0874 | 0.1152 | 0.0346 | 0.1333 | 0.4676 |
| paired_test | C | single_task | 0.2430 | 0.2276 | 0.1274 | 0.2158 | 0.5306 |
| paired_test | C | spd_finetune | 0.2009 | 0.1770 | 0.0884 | 0.2012 | 0.5164 |
| paired_test | C | spd_zero_shot | 0.1800 | 0.1511 | 0.0745 | 0.1822 | 0.5780 |
| real_all | A | gaze | 0.2489 | 0.0945 | 0.0273 | 0.2889 | 0.7674 |
| real_all | C | gaze | 0.2460 | 0.2328 | 0.1180 | 0.2428 | 0.5142 |

## 十段测试录像上的模型误差

有符号误差按模型减真实 gaze 计算。Brier 为时间加权的二值事件不一致率。

| 任务 | 模型 | ΔC | Target Brier | Gripper Brier |
|---|---|---:|---:|---:|
| A | all_ABCD | -0.2365 | 0.2412 | 0.1348 |
| A | leave_out | -0.1028 | 0.2441 | 0.1262 |
| A | single_task | -0.1203 | 0.2430 | 0.1306 |
| A | spd_finetune | -0.1568 | 0.2466 | 0.1319 |
| A | spd_zero_shot | 0.0327 | 0.2441 | 0.1045 |
| C | all_ABCD | -0.1056 | 0.2658 | 0.2488 |
| C | leave_out | -0.1047 | 0.2891 | 0.2628 |
| C | single_task | -0.0417 | 0.2704 | 0.2526 |
| C | spd_finetune | -0.0558 | 0.2677 | 0.2375 |
| C | spd_zero_shot | 0.0058 | 0.3106 | 0.2714 |

## 阅读与复核

- 全量真人行为与共同测试录像分 scope 保存。模型评估为 held-out recordings，同一参与者的其他录像可在训练集中。
- Target 为 NPY 有效点的算术平均，gripper 为双侧 DLC 跟踪点的并集。输入没有可用的独立目标身份字段时，驻留按连续阶段 occurrence 分段。图上位置重合时，C 仅表示空间偏好。
- 50 px 是预设空间容差。标注来源、点数和覆盖率可核查；本次没有新增人工定位准确率测量。
- 视频原时钟与帧号用于全部时长。超过 100 ms 或原始帧不连续时中断。中点法对相邻有效采样间隔各分配一半。观察到的 120 ms 以上驻留包含截断片段并单报截断计数。
- uniform 面积参照与固定画面中心点是不同预测器。训练参照采用同任务 train 清单，与全部被比较模型使用同一评价标尺。训练空间热图为 sigma=1 网格单位的 gaze 核，按有效帧时长平均。
- 完成及失败且有实际分数的 trials 用于 performance，缺失分数和零方差原因在表中标明。测试集每项任务最多五人，Spearman、bootstrap 区间和模型对比均作样本内描述。CI 条件于现有模型，未校正多重比较。
- 对应 static、better-novice、新参与者留出及模型 repetition 配对的预测不在当前导出清单。真实 gaze 的轮次用相异 TaskNum 排序，先合并重试。

## 输出导航

- [输入审计](recording_audit.csv) · [模型审计](prediction_audit.csv) · [覆盖表](coverage.csv) · [标注点数与来源](target_labels.csv)
- [Trial 与阶段指标](trial_stage.csv) · [配对误差](model_trial_errors.csv) · [训练参照](baseline_parameters.csv)
- [运行配置](run_config.json) · [完整性核验](validation.json) · [汇总与图件目录](output_index.csv)

帧级 NPZ 的坐标为像素；坐标列顺序 (x,y)。状态码 -1 无效、0 other、1 target-only、2 gripper-only、3 overlap。valid_real / valid_paired 标记对应情景的分母。
