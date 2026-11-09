# Passive gaze 与配对 Active gaze：novice–novice 16对分析

本报告仅纳入 source skill 与 observer skill 均为 novice 的指定配对。每个 passive 的指标、图表与对应 active 一起呈现；所有数值来自本次逐行配对重新测量。

## 数据与共同支持

|任务|配对数|源录像数|源操作者数|观察者数|
|---|---:|---:|---:|---:|
|A|7|7|7|7|
|C|9|9|8|8|

匹配规则为对应 active/passive NPY 行数相等，即第 i 行对应同一视频帧；不要求两份时间戳数值相等，不另做时间重配准。阶段、target 槽位、双侧 DLC 腕部位置、原始视频时间与尺寸均继承对应 active 内容。每对在两者共同有效帧上同时计算两种 gaze；有效性只要求坐标有限且归一化坐标在 [0,1]，不以 valid 状态码筛选。

[配对逐阶段原始指标](pair_metrics.csv) · [覆盖审计](coverage.csv) · [运行配置](run_config.json)

## 完全复用的 v3 指标

直接调用冻结的 v3 classify、areas、summarize、temporal 与时间支持代码。主条件 target 半径与双侧腕部邻域半径均为 50 px。Target 中心仍是现有有效槽位的算术平均；器械区域仍是 DLC 腕部/铰接邻域，未改称器械尖端。

令 T/P/O/N 分别表示仅 target、仅腕部、两者重叠及其他区域。每对/阶段以原始视频时间戳计算的帧支持时长 dt 加权得到 L_T、L_P、L_O、L_N；L_target=L_T+L_O，L_gripper=L_P+L_O，D=L_T+L_P，C=L_T/(L_T+L_P)。分母为零时保留 NA。面积参照及 C_excess、E_target、E_gripper、B、B_excess 同样调用 v3。whole 按纳入阶段的时间合并，不平均阶段比例。

驻留与切换沿用 v3 的连续段、时间断点、最短驻留与端点删失规则。敏感性保留 main、r25、r100、gripper100、smooth；smooth 为原三样本平滑及短缺口规则，两类 gaze 分别平滑后重建共同支持。

## 汇总与统计

逐指标先筛选同一 pair 中 active/passive 均可估的配对，再在同一 content 内对观察者等权平均；随后同一源操作者内录像等权，最后源操作者等权。因共同有效帧随观察者变化，active 也保留各配对重新测量的结果，在 content 内与 passive 对称平均；不把重复 active 当作新增独立样本。Δ 一律为 Passive−Active。

95% CI 直接复用 v3 ParticipantBootstrap，seed=42、10,000次、参与者重抽及百分位区间。此处参与者是源录像操作者；CI 条件于本次观察者集合，不表示对新观察者总体的双重抽样推断。每个指标分别记录有效配对/录像/参与者数。少于2个源操作者不计算均值CI；不新增显著性检验。

[内容层汇总](content_stage_summary.csv) · [源操作者层汇总](participant_stage_summary.csv) · [组汇总（宽表）](group_stage_summary.csv) · [组汇总（长表）](group_stage_summary_long.csv) · [观察者描述表](observer_stage_summary.csv)

## 主条件全任务结果

占用与 C 以百分比显示；差值为百分点，方括号为95%CI。

|任务|指标|Active [95%CI]|Passive [95%CI]|Passive−Active [95%CI]|源操作者/录像/配对|
|---|---|---:|---:|---:|---:|
|A|L_target|26.79 [24.29, 29.03]|9.50 [5.94, 12.99]|-17.29 [-20.73, -13.68]|7/7/7|
|A|L_gripper|9.68 [4.97, 15.16]|4.39 [2.36, 6.56]|-5.30 [-9.59, -1.31]|7/7/7|
|A|C|79.01 [69.15, 87.46]|72.28 [64.66, 79.48]|-6.73 [-16.08, 2.15]|7/7/7|

### Task A 图表

**Occupancy and preference (95% participant CI)**

![Occupancy and preference (95% participant CI)](figures/task_A/core_metrics.png)

**Area-corrected metrics (95% participant CI)**

![Area-corrected metrics (95% participant CI)](figures/task_A/area_corrected.png)

**Observed dwell and switching (95% participant CI)**

![Observed dwell and switching (95% participant CI)](figures/task_A/dwell_switch.png)

**四区域占用：Active与Passive并排**

![四区域占用：Active与Passive并排](figures/task_A/regions.png)

**配对差值：Passive−Active**

![配对差值：Passive−Active](figures/task_A/paired_differences.png)

**全任务Active–Passive一致性散点**

![全任务Active–Passive一致性散点](figures/task_A/agreement.png)

**五种条件敏感性：两种gaze与CI**

![五种条件敏感性：两种gaze与CI](figures/task_A/sensitivity.png)

**源操作者成绩关联：同内容Active与Passive**

![源操作者成绩关联：同内容Active与Passive](figures/task_A/performance.png)

**源录像练习轮次：配对gaze轨迹**

![源录像练习轮次：配对gaze轨迹](figures/task_A/rounds.png)

**Active×Passive区域联合及条件矩阵**

![Active×Passive区域联合及条件矩阵](figures/task_A/region_correspondence.png)

**每对Active、Passive及共同有效帧**

![每对Active、Passive及共同有效帧](figures/task_A/coverage.png)


|任务|指标|Active [95%CI]|Passive [95%CI]|Passive−Active [95%CI]|源操作者/录像/配对|
|---|---|---:|---:|---:|---:|
|C|L_target|29.17 [21.22, 37.51]|8.96 [5.99, 12.37]|-20.21 [-27.64, -13.07]|8/9/9|
|C|L_gripper|23.14 [16.54, 30.18]|17.67 [12.43, 23.63]|-5.47 [-12.50, 3.77]|8/9/9|
|C|C|61.15 [47.36, 72.08]|27.04 [15.76, 39.59]|-34.11 [-46.24, -22.04]|8/9/9|

### Task C 图表

**Occupancy and preference (95% participant CI)**

![Occupancy and preference (95% participant CI)](figures/task_C/core_metrics.png)

**Area-corrected metrics (95% participant CI)**

![Area-corrected metrics (95% participant CI)](figures/task_C/area_corrected.png)

**Observed dwell and switching (95% participant CI)**

![Observed dwell and switching (95% participant CI)](figures/task_C/dwell_switch.png)

**四区域占用：Active与Passive并排**

![四区域占用：Active与Passive并排](figures/task_C/regions.png)

**配对差值：Passive−Active**

![配对差值：Passive−Active](figures/task_C/paired_differences.png)

**全任务Active–Passive一致性散点**

![全任务Active–Passive一致性散点](figures/task_C/agreement.png)

**五种条件敏感性：两种gaze与CI**

![五种条件敏感性：两种gaze与CI](figures/task_C/sensitivity.png)

**源操作者成绩关联：同内容Active与Passive**

![源操作者成绩关联：同内容Active与Passive](figures/task_C/performance.png)

**源录像练习轮次：配对gaze轨迹**

![源录像练习轮次：配对gaze轨迹](figures/task_C/rounds.png)

**Active×Passive区域联合及条件矩阵**

![Active×Passive区域联合及条件矩阵](figures/task_C/region_correspondence.png)

**每对Active、Passive及共同有效帧**

![每对Active、Passive及共同有效帧](figures/task_C/coverage.png)

## 源操作者成绩与练习轮次

成绩关联使用源操作者成绩，不是 passive observer 自身表现。每个指标先在分数及两种 gaze 同时有效的同一组源录像内形成操作者均值，再算 Spearman ρ、参与者bootstrap CI及逐人留一法。少于3人、常量成绩或常量指标保留NA；ties与无定义bootstrap数量保留在CSV。

|任务|指标|n|Active ρ [95%CI]|Passive ρ [95%CI]|
|---|---|---:|---:|---:|
|A|L_target|7|0.197 [-0.814, 0.923]|0.236 [-0.828, 0.921]|
|A|L_gripper|7|-0.217 [-0.961, 0.623]|-0.374 [-0.907, 0.642]|
|A|C|7|0.217 [-0.626, 0.948]|0.315 [-0.828, 0.971]|
|C|L_target|8|0.592 [0.085, 0.877]|0.592 [0.085, 0.877]|
|C|L_gripper|8|0.655 [0.175, 0.956]|0.234 [-0.521, 0.798]|
|C|C|8|-0.156 [-0.869, 0.513]|0.483 [-0.088, 0.871]|

round 表示源操作者/任务内的练习轮次，不是观察者观看顺序或 Attempt。轮次差只在同一操作者两轮及两类 gaze 均有效时计算；可用样本不足时保留NA，描述曲线不表示纵向因果效应。

[成绩相关](performance_correlations.csv) · [相关用参与者数据](performance_participants.csv) · [逐人留一法](performance_leave_one_out.csv) · [轮次描述](participant_round_summary.csv) · [轮次配对差](repetition_contrasts.csv)

## 模型子集与原四分析分支

模型比较使用本次配对与原v3测试录像的交集：3对、3个源内容、2个源操作者。仍使用原五模型、原保存的64 samples概率质量主分支及expectation/argmax/sample解码分支；每个分支同时附active、passive实测参考。

Brier 对同一个模型分别与 active/passive 的二元事件计算；BSS=1−参与者等权平均模型Brier/参与者等权平均参照Brier。训练参照冻结为原训练平均热图、中心点、均匀面积及训练事件频率。GT_smoothed只是诊断，不进入BSS分母。独立逐帧sample解码的驻留/切换只描述所抽路径，不表示概率分布的时间连贯性。

[模型逐配对指标](model_pair_metrics.csv) · [模型内容汇总](model_content_stage_summary.csv) · [模型参与者汇总](model_participant_stage_summary.csv) · [模型组汇总](model_group_stage_summary.csv) · [模型BSS](model_brier_skill.csv)

- [all_ABCD: decoded_argmax](figures/models/decoded_argmax/all_ABCD/task_A_metrics.png)
- [all_ABCD: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/all_ABCD/task_A_temporal.png)
- [all_ABCD: Brier against active / passive](figures/models/decoded_argmax/all_ABCD/task_A_brier.png)
- [模型BSS：decoded_argmax/all_ABCD，两类gaze参照](figures/models/decoded_argmax/all_ABCD/task_A_bss.png)
- [all_ABCD: decoded_argmax](figures/models/decoded_argmax/all_ABCD/task_C_metrics.png)
- [all_ABCD: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/all_ABCD/task_C_temporal.png)
- [all_ABCD: Brier against active / passive](figures/models/decoded_argmax/all_ABCD/task_C_brier.png)
- [模型BSS：decoded_argmax/all_ABCD，两类gaze参照](figures/models/decoded_argmax/all_ABCD/task_C_bss.png)
- [leave_out: decoded_argmax](figures/models/decoded_argmax/leave_out/task_A_metrics.png)
- [leave_out: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/leave_out/task_A_temporal.png)
- [leave_out: Brier against active / passive](figures/models/decoded_argmax/leave_out/task_A_brier.png)
- [模型BSS：decoded_argmax/leave_out，两类gaze参照](figures/models/decoded_argmax/leave_out/task_A_bss.png)
- [leave_out: decoded_argmax](figures/models/decoded_argmax/leave_out/task_C_metrics.png)
- [leave_out: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/leave_out/task_C_temporal.png)
- [leave_out: Brier against active / passive](figures/models/decoded_argmax/leave_out/task_C_brier.png)
- [模型BSS：decoded_argmax/leave_out，两类gaze参照](figures/models/decoded_argmax/leave_out/task_C_bss.png)
- [single_task: decoded_argmax](figures/models/decoded_argmax/single_task/task_A_metrics.png)
- [single_task: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/single_task/task_A_temporal.png)
- [single_task: Brier against active / passive](figures/models/decoded_argmax/single_task/task_A_brier.png)
- [模型BSS：decoded_argmax/single_task，两类gaze参照](figures/models/decoded_argmax/single_task/task_A_bss.png)
- [single_task: decoded_argmax](figures/models/decoded_argmax/single_task/task_C_metrics.png)
- [single_task: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/single_task/task_C_temporal.png)
- [single_task: Brier against active / passive](figures/models/decoded_argmax/single_task/task_C_brier.png)
- [模型BSS：decoded_argmax/single_task，两类gaze参照](figures/models/decoded_argmax/single_task/task_C_bss.png)
- [spd_finetune: decoded_argmax](figures/models/decoded_argmax/spd_finetune/task_A_metrics.png)
- [spd_finetune: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/spd_finetune/task_A_temporal.png)
- [spd_finetune: Brier against active / passive](figures/models/decoded_argmax/spd_finetune/task_A_brier.png)
- [模型BSS：decoded_argmax/spd_finetune，两类gaze参照](figures/models/decoded_argmax/spd_finetune/task_A_bss.png)
- [spd_finetune: decoded_argmax](figures/models/decoded_argmax/spd_finetune/task_C_metrics.png)
- [spd_finetune: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/spd_finetune/task_C_temporal.png)
- [spd_finetune: Brier against active / passive](figures/models/decoded_argmax/spd_finetune/task_C_brier.png)
- [模型BSS：decoded_argmax/spd_finetune，两类gaze参照](figures/models/decoded_argmax/spd_finetune/task_C_bss.png)
- [spd_zero_shot: decoded_argmax](figures/models/decoded_argmax/spd_zero_shot/task_A_metrics.png)
- [spd_zero_shot: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/spd_zero_shot/task_A_temporal.png)
- [spd_zero_shot: Brier against active / passive](figures/models/decoded_argmax/spd_zero_shot/task_A_brier.png)
- [模型BSS：decoded_argmax/spd_zero_shot，两类gaze参照](figures/models/decoded_argmax/spd_zero_shot/task_A_bss.png)
- [spd_zero_shot: decoded_argmax](figures/models/decoded_argmax/spd_zero_shot/task_C_metrics.png)
- [spd_zero_shot: decoded_argmax temporal diagnostic](figures/models/decoded_argmax/spd_zero_shot/task_C_temporal.png)
- [spd_zero_shot: Brier against active / passive](figures/models/decoded_argmax/spd_zero_shot/task_C_brier.png)
- [模型BSS：decoded_argmax/spd_zero_shot，两类gaze参照](figures/models/decoded_argmax/spd_zero_shot/task_C_bss.png)
- [all_ABCD: decoded_expectation](figures/models/decoded_expectation/all_ABCD/task_A_metrics.png)
- [all_ABCD: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/all_ABCD/task_A_temporal.png)
- [all_ABCD: Brier against active / passive](figures/models/decoded_expectation/all_ABCD/task_A_brier.png)
- [模型BSS：decoded_expectation/all_ABCD，两类gaze参照](figures/models/decoded_expectation/all_ABCD/task_A_bss.png)
- [all_ABCD: decoded_expectation](figures/models/decoded_expectation/all_ABCD/task_C_metrics.png)
- [all_ABCD: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/all_ABCD/task_C_temporal.png)
- [all_ABCD: Brier against active / passive](figures/models/decoded_expectation/all_ABCD/task_C_brier.png)
- [模型BSS：decoded_expectation/all_ABCD，两类gaze参照](figures/models/decoded_expectation/all_ABCD/task_C_bss.png)
- [leave_out: decoded_expectation](figures/models/decoded_expectation/leave_out/task_A_metrics.png)
- [leave_out: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/leave_out/task_A_temporal.png)
- [leave_out: Brier against active / passive](figures/models/decoded_expectation/leave_out/task_A_brier.png)
- [模型BSS：decoded_expectation/leave_out，两类gaze参照](figures/models/decoded_expectation/leave_out/task_A_bss.png)
- [leave_out: decoded_expectation](figures/models/decoded_expectation/leave_out/task_C_metrics.png)
- [leave_out: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/leave_out/task_C_temporal.png)
- [leave_out: Brier against active / passive](figures/models/decoded_expectation/leave_out/task_C_brier.png)
- [模型BSS：decoded_expectation/leave_out，两类gaze参照](figures/models/decoded_expectation/leave_out/task_C_bss.png)
- [single_task: decoded_expectation](figures/models/decoded_expectation/single_task/task_A_metrics.png)
- [single_task: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/single_task/task_A_temporal.png)
- [single_task: Brier against active / passive](figures/models/decoded_expectation/single_task/task_A_brier.png)
- [模型BSS：decoded_expectation/single_task，两类gaze参照](figures/models/decoded_expectation/single_task/task_A_bss.png)
- [single_task: decoded_expectation](figures/models/decoded_expectation/single_task/task_C_metrics.png)
- [single_task: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/single_task/task_C_temporal.png)
- [single_task: Brier against active / passive](figures/models/decoded_expectation/single_task/task_C_brier.png)
- [模型BSS：decoded_expectation/single_task，两类gaze参照](figures/models/decoded_expectation/single_task/task_C_bss.png)
- [spd_finetune: decoded_expectation](figures/models/decoded_expectation/spd_finetune/task_A_metrics.png)
- [spd_finetune: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/spd_finetune/task_A_temporal.png)
- [spd_finetune: Brier against active / passive](figures/models/decoded_expectation/spd_finetune/task_A_brier.png)
- [模型BSS：decoded_expectation/spd_finetune，两类gaze参照](figures/models/decoded_expectation/spd_finetune/task_A_bss.png)
- [spd_finetune: decoded_expectation](figures/models/decoded_expectation/spd_finetune/task_C_metrics.png)
- [spd_finetune: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/spd_finetune/task_C_temporal.png)
- [spd_finetune: Brier against active / passive](figures/models/decoded_expectation/spd_finetune/task_C_brier.png)
- [模型BSS：decoded_expectation/spd_finetune，两类gaze参照](figures/models/decoded_expectation/spd_finetune/task_C_bss.png)
- [spd_zero_shot: decoded_expectation](figures/models/decoded_expectation/spd_zero_shot/task_A_metrics.png)
- [spd_zero_shot: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/spd_zero_shot/task_A_temporal.png)
- [spd_zero_shot: Brier against active / passive](figures/models/decoded_expectation/spd_zero_shot/task_A_brier.png)
- [模型BSS：decoded_expectation/spd_zero_shot，两类gaze参照](figures/models/decoded_expectation/spd_zero_shot/task_A_bss.png)
- [spd_zero_shot: decoded_expectation](figures/models/decoded_expectation/spd_zero_shot/task_C_metrics.png)
- [spd_zero_shot: decoded_expectation temporal diagnostic](figures/models/decoded_expectation/spd_zero_shot/task_C_temporal.png)
- [spd_zero_shot: Brier against active / passive](figures/models/decoded_expectation/spd_zero_shot/task_C_brier.png)
- [模型BSS：decoded_expectation/spd_zero_shot，两类gaze参照](figures/models/decoded_expectation/spd_zero_shot/task_C_bss.png)
- [all_ABCD: decoded_sample](figures/models/decoded_sample/all_ABCD/task_A_metrics.png)
- [all_ABCD: decoded_sample temporal diagnostic](figures/models/decoded_sample/all_ABCD/task_A_temporal.png)
- [all_ABCD: Brier against active / passive](figures/models/decoded_sample/all_ABCD/task_A_brier.png)
- [模型BSS：decoded_sample/all_ABCD，两类gaze参照](figures/models/decoded_sample/all_ABCD/task_A_bss.png)
- [all_ABCD: decoded_sample](figures/models/decoded_sample/all_ABCD/task_C_metrics.png)
- [all_ABCD: decoded_sample temporal diagnostic](figures/models/decoded_sample/all_ABCD/task_C_temporal.png)
- [all_ABCD: Brier against active / passive](figures/models/decoded_sample/all_ABCD/task_C_brier.png)
- [模型BSS：decoded_sample/all_ABCD，两类gaze参照](figures/models/decoded_sample/all_ABCD/task_C_bss.png)
- [leave_out: decoded_sample](figures/models/decoded_sample/leave_out/task_A_metrics.png)
- [leave_out: decoded_sample temporal diagnostic](figures/models/decoded_sample/leave_out/task_A_temporal.png)
- [leave_out: Brier against active / passive](figures/models/decoded_sample/leave_out/task_A_brier.png)
- [模型BSS：decoded_sample/leave_out，两类gaze参照](figures/models/decoded_sample/leave_out/task_A_bss.png)
- [leave_out: decoded_sample](figures/models/decoded_sample/leave_out/task_C_metrics.png)
- [leave_out: decoded_sample temporal diagnostic](figures/models/decoded_sample/leave_out/task_C_temporal.png)
- [leave_out: Brier against active / passive](figures/models/decoded_sample/leave_out/task_C_brier.png)
- [模型BSS：decoded_sample/leave_out，两类gaze参照](figures/models/decoded_sample/leave_out/task_C_bss.png)
- [single_task: decoded_sample](figures/models/decoded_sample/single_task/task_A_metrics.png)
- [single_task: decoded_sample temporal diagnostic](figures/models/decoded_sample/single_task/task_A_temporal.png)
- [single_task: Brier against active / passive](figures/models/decoded_sample/single_task/task_A_brier.png)
- [模型BSS：decoded_sample/single_task，两类gaze参照](figures/models/decoded_sample/single_task/task_A_bss.png)
- [single_task: decoded_sample](figures/models/decoded_sample/single_task/task_C_metrics.png)
- [single_task: decoded_sample temporal diagnostic](figures/models/decoded_sample/single_task/task_C_temporal.png)
- [single_task: Brier against active / passive](figures/models/decoded_sample/single_task/task_C_brier.png)
- [模型BSS：decoded_sample/single_task，两类gaze参照](figures/models/decoded_sample/single_task/task_C_bss.png)
- [spd_finetune: decoded_sample](figures/models/decoded_sample/spd_finetune/task_A_metrics.png)
- [spd_finetune: decoded_sample temporal diagnostic](figures/models/decoded_sample/spd_finetune/task_A_temporal.png)
- [spd_finetune: Brier against active / passive](figures/models/decoded_sample/spd_finetune/task_A_brier.png)
- [模型BSS：decoded_sample/spd_finetune，两类gaze参照](figures/models/decoded_sample/spd_finetune/task_A_bss.png)
- [spd_finetune: decoded_sample](figures/models/decoded_sample/spd_finetune/task_C_metrics.png)
- [spd_finetune: decoded_sample temporal diagnostic](figures/models/decoded_sample/spd_finetune/task_C_temporal.png)
- [spd_finetune: Brier against active / passive](figures/models/decoded_sample/spd_finetune/task_C_brier.png)
- [模型BSS：decoded_sample/spd_finetune，两类gaze参照](figures/models/decoded_sample/spd_finetune/task_C_bss.png)
- [spd_zero_shot: decoded_sample](figures/models/decoded_sample/spd_zero_shot/task_A_metrics.png)
- [spd_zero_shot: decoded_sample temporal diagnostic](figures/models/decoded_sample/spd_zero_shot/task_A_temporal.png)
- [spd_zero_shot: Brier against active / passive](figures/models/decoded_sample/spd_zero_shot/task_A_brier.png)
- [模型BSS：decoded_sample/spd_zero_shot，两类gaze参照](figures/models/decoded_sample/spd_zero_shot/task_A_bss.png)
- [spd_zero_shot: decoded_sample](figures/models/decoded_sample/spd_zero_shot/task_C_metrics.png)
- [spd_zero_shot: decoded_sample temporal diagnostic](figures/models/decoded_sample/spd_zero_shot/task_C_temporal.png)
- [spd_zero_shot: Brier against active / passive](figures/models/decoded_sample/spd_zero_shot/task_C_brier.png)
- [模型BSS：decoded_sample/spd_zero_shot，两类gaze参照](figures/models/decoded_sample/spd_zero_shot/task_C_bss.png)
- [all_ABCD: probability_mc](figures/models/probability_mc/all_ABCD/task_A_metrics.png)
- [all_ABCD: Brier against active / passive](figures/models/probability_mc/all_ABCD/task_A_brier.png)
- [模型BSS：probability_mc/all_ABCD，两类gaze参照](figures/models/probability_mc/all_ABCD/task_A_bss.png)
- [all_ABCD: probability_mc](figures/models/probability_mc/all_ABCD/task_C_metrics.png)
- [all_ABCD: Brier against active / passive](figures/models/probability_mc/all_ABCD/task_C_brier.png)
- [模型BSS：probability_mc/all_ABCD，两类gaze参照](figures/models/probability_mc/all_ABCD/task_C_bss.png)
- [leave_out: probability_mc](figures/models/probability_mc/leave_out/task_A_metrics.png)
- [leave_out: Brier against active / passive](figures/models/probability_mc/leave_out/task_A_brier.png)
- [模型BSS：probability_mc/leave_out，两类gaze参照](figures/models/probability_mc/leave_out/task_A_bss.png)
- [leave_out: probability_mc](figures/models/probability_mc/leave_out/task_C_metrics.png)
- [leave_out: Brier against active / passive](figures/models/probability_mc/leave_out/task_C_brier.png)
- [模型BSS：probability_mc/leave_out，两类gaze参照](figures/models/probability_mc/leave_out/task_C_bss.png)
- [single_task: probability_mc](figures/models/probability_mc/single_task/task_A_metrics.png)
- [single_task: Brier against active / passive](figures/models/probability_mc/single_task/task_A_brier.png)
- [模型BSS：probability_mc/single_task，两类gaze参照](figures/models/probability_mc/single_task/task_A_bss.png)
- [single_task: probability_mc](figures/models/probability_mc/single_task/task_C_metrics.png)
- [single_task: Brier against active / passive](figures/models/probability_mc/single_task/task_C_brier.png)
- [模型BSS：probability_mc/single_task，两类gaze参照](figures/models/probability_mc/single_task/task_C_bss.png)
- [spd_finetune: probability_mc](figures/models/probability_mc/spd_finetune/task_A_metrics.png)
- [spd_finetune: Brier against active / passive](figures/models/probability_mc/spd_finetune/task_A_brier.png)
- [模型BSS：probability_mc/spd_finetune，两类gaze参照](figures/models/probability_mc/spd_finetune/task_A_bss.png)
- [spd_finetune: probability_mc](figures/models/probability_mc/spd_finetune/task_C_metrics.png)
- [spd_finetune: Brier against active / passive](figures/models/probability_mc/spd_finetune/task_C_brier.png)
- [模型BSS：probability_mc/spd_finetune，两类gaze参照](figures/models/probability_mc/spd_finetune/task_C_bss.png)
- [spd_zero_shot: probability_mc](figures/models/probability_mc/spd_zero_shot/task_A_metrics.png)
- [spd_zero_shot: Brier against active / passive](figures/models/probability_mc/spd_zero_shot/task_A_brier.png)
- [模型BSS：probability_mc/spd_zero_shot，两类gaze参照](figures/models/probability_mc/spd_zero_shot/task_A_bss.png)
- [spd_zero_shot: probability_mc](figures/models/probability_mc/spd_zero_shot/task_C_metrics.png)
- [spd_zero_shot: Brier against active / passive](figures/models/probability_mc/spd_zero_shot/task_C_brier.png)
- [模型BSS：probability_mc/spd_zero_shot，两类gaze参照](figures/models/probability_mc/spd_zero_shot/task_C_bss.png)

### 解码区域矩阵与GT平滑诊断

下列图均在同一张图内并列active、passive两种观测参照。GT平滑仅为观测平滑诊断，不作模型上限或训练参照。

[补充报告及数据索引](model_supplement_report.md) · [区域矩阵组汇总](model_confusion_group_summary.csv) · [补充图索引](model_supplement_figure_index.csv)

- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/all_ABCD_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/all_ABCD_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/leave_out_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/leave_out_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/single_task_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/single_task_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/spd_finetune_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/spd_finetune_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/spd_zero_shot_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_argmax/spd_zero_shot_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/all_ABCD_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/all_ABCD_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/leave_out_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/leave_out_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/single_task_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/single_task_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/spd_finetune_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/spd_finetune_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/spd_zero_shot_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_expectation/spd_zero_shot_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/all_ABCD_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/all_ABCD_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/leave_out_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/leave_out_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/single_task_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/single_task_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/spd_finetune_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/spd_finetune_task_C_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/spd_zero_shot_task_A_confusion.png)
- [Active/passive reference joint and conditional model-region matrices](figures/model_supplement/decoded_sample/spd_zero_shot_task_C_confusion.png)
- [Paired active/passive GT-smoothed occupancy diagnostic; no BSS baseline](figures/model_supplement/task_A_gt_smoothed.png)
- [Paired active/passive GT-smoothed occupancy diagnostic; no BSS baseline](figures/model_supplement/task_C_gt_smoothed.png)

## 每对视频帧与时间序列

每个配对均提供同帧active/passive坐标叠加及双轨迹时间序列。示例帧按主条件共同有效帧的中位索引确定，未按差异大小筛选。

[示例帧选择及坐标索引](example_index.csv)

|配对|任务|视频帧对照|时间序列|
|---|---|---|---|
|User1_4_User11_2_1|A|[同帧叠加](figures/examples/User1_4_User11_2_1_overlay.png)|[双轨迹](figures/examples/User1_4_User11_2_1_timeline.png)|
|User1_6_User9_5_2|C|[同帧叠加](figures/examples/User1_6_User9_5_2_overlay.png)|[双轨迹](figures/examples/User1_6_User9_5_2_timeline.png)|
|User2_2_User11_3_1|C|[同帧叠加](figures/examples/User2_2_User11_3_1_overlay.png)|[双轨迹](figures/examples/User2_2_User11_3_1_timeline.png)|
|User2_6_User12_3_1|A|[同帧叠加](figures/examples/User2_6_User12_3_1_overlay.png)|[双轨迹](figures/examples/User2_6_User12_3_1_timeline.png)|
|User3_2_User6_3_1|A|[同帧叠加](figures/examples/User3_2_User6_3_1_overlay.png)|[双轨迹](figures/examples/User3_2_User6_3_1_timeline.png)|
|User3_4_User12_5_2|C|[同帧叠加](figures/examples/User3_4_User12_5_2_overlay.png)|[双轨迹](figures/examples/User3_4_User12_5_2_timeline.png)|
|User6_2_User1_3_1|C|[同帧叠加](figures/examples/User6_2_User1_3_1_overlay.png)|[双轨迹](figures/examples/User6_2_User1_3_1_timeline.png)|
|User6_6_User2_2_1|C|[同帧叠加](figures/examples/User6_6_User2_2_1_overlay.png)|[双轨迹](figures/examples/User6_6_User2_2_1_timeline.png)|
|User8_4_User10_1_1|A|[同帧叠加](figures/examples/User8_4_User10_1_1_overlay.png)|[双轨迹](figures/examples/User8_4_User10_1_1_timeline.png)|
|User8_6_User5_3_1|C|[同帧叠加](figures/examples/User8_6_User5_3_1_overlay.png)|[双轨迹](figures/examples/User8_6_User5_3_1_timeline.png)|
|User9_2_User8_6_1|C|[同帧叠加](figures/examples/User9_2_User8_6_1_overlay.png)|[双轨迹](figures/examples/User9_2_User8_6_1_timeline.png)|
|User9_6_User5_2_1|A|[同帧叠加](figures/examples/User9_6_User5_2_1_overlay.png)|[双轨迹](figures/examples/User9_6_User5_2_1_timeline.png)|
|User10_2_User7_1_1|A|[同帧叠加](figures/examples/User10_2_User7_1_1_overlay.png)|[双轨迹](figures/examples/User10_2_User7_1_1_timeline.png)|
|User10_4_User2_2_2|C|[同帧叠加](figures/examples/User10_4_User2_2_2_overlay.png)|[双轨迹](figures/examples/User10_4_User2_2_2_timeline.png)|
|User12_2_User7_3_1|C|[同帧叠加](figures/examples/User12_2_User7_3_1_overlay.png)|[双轨迹](figures/examples/User12_2_User7_3_1_timeline.png)|
|User12_6_User9_2_1|A|[同帧叠加](figures/examples/User12_6_User9_2_1_overlay.png)|[双轨迹](figures/examples/User12_6_User9_2_1_timeline.png)|

## 解读范围与输出检查

- 本次仅16个novice–novice配对，Task A 7对、Task C 9对；结论不外推到其他技能组。
- 每个passive仅与其对应active比较，图上active基线来自本次相同配对、共同帧及相同层级汇总。
- 区域矩阵描述两种人类gaze的对应关系，active不是passive应复制的唯一真值。
- 同一内容的观察者重复及源操作者重复均保留身份，未把每行配对当作独立参与者。
- 平滑和半径敏感性单独报告；缺失值保留NA，不以零补值。

[区域联合逐对数据](region_joint.csv) · [区域联合汇总](region_joint_summary.csv) · [图表索引](figure_index.csv) · [统计配置](statistics_manifest.json)
