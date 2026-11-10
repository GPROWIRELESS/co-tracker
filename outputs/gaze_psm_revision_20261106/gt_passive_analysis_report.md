# Passive 与 Active GT gaze–PSM 结果分析报告

更新日期：2026-11-09。数值来自 `gt_passive16` 及其 `active_passive_comparison` 已有结果，本次整理报告，不重新计算或修改冻结分析。全文沿用 [GT84 分析报告](gt84_analysis_report.md) 与 [GT84 论文摘要](gt84_paper_summary.md) 的字段和阶段名称。

`passive` 指观看者观看来源操作视频时的人工 gaze；`active` 指同一来源视频中操作者的人工 GT gaze。两者分别与该来源视频的固定左、右 PSM 轨迹进行交叉相关（xcorr）。完整 GT84 包含 84 条录制、11 名操作者；本报告的直接比较仅使用与 passive 按来源视频匹配的 16 条 active GT，不将完整 GT84 组结果作为配对对照。

纳入 16 次 novice–novice 观看、8 名观看者及 16 个不同来源视频；匹配 active 涉及 10 名来源操作者，两种身份的并集为 11 人。Task A 为 7 对/7 名观看者，Task C 为 9 对/8 名观看者。配对单位是来源视频，不是同一个人的 active/passive 两种实验状态；16 也不是独立人数。全部 122 次观看中排除任一端为 intermediate 的 98 次，再排除 A/C 之外的 8 次。

passive gaze 来自观看者 NPY，active GT 来自 `D:/Dataset/dvss/Active_Data/User*/npy/<record>_gaze.npy`；stage、PSM 与来源视频一致。双方均仅按两个坐标有限且位于归一化闭区间 `[0,1]` 筛选，不用 stored `valid` 或 CSV `Status` 排除样本。passive 的 205,023 个输入 NPY 样本全部通过当前坐标规则；历史 NPY 已包含 clamp/最近邻填充，不能将其解释为所有原始眼动样本均有效。坐标按 1280×1024 图像换算为像素。

按既有确认规则使用 NPY 第 i 行、视频第 i 帧及 DLC 第 i 行对应；active/passive 依据来源映射及 NPY 行数相等配对，不增加时间戳匹配条件。分析沿用精确源采集 `video_ts` 的秒数及原 41 Hz 网格；未经验校准真实播放时序，lag 不能直接解释为观看者生理反应时。

## 1. 主要结果与边界

passive 主分支预定 72 个固定侧条件，其中 56 个具有支持充分的组曲线。原规则方向状态为 `direction_uncertain`=31、`window_limited`=23、`negative_peak_resampling_range`=2、`positive_peak_resampling_range`=0。其余 16 个条件中，A / drop ring 的 8 个条件没有阶段标注；C / threading string through tower 的 8 个条件只有 1 名完整支持观看者、2 次观看，不满足至少 3 人规则，均保留 NA。

原规则标为负向峰位重采样范围的条件为：A / grab ring / l / position / PSM in stage：-1.073 s，范围 [-1.732, -0.317] s；A / place ring / r / position / Both in stage：-0.561 s，范围 [-1.244, -0.341] s。这属于描述性方向证据，不是正式生理滞后检验。


全部 8 个 variant 有 2420 条逐录制条件曲线，primary 为 440 条（同时包含两侧、两种 signal 和两种 mode），440 条均具完整 lag 支持，只有 3 条通过 `lag_identifiable`。28 个支持充分位置组峰的 `peak_r` 为 −0.0620–0.2532，28 个速度组峰为 0.0091–0.0598；有符号最大相关也可能仍为负值。零 lag 空间指标在 28 个支持充分条件中的 `distance_median_px` 为 135.9–419.1 px，`proximity_50px` 为 0.34%–12.29%。

在来源视频配对比较中，440 个逐录制主条件有 436 个双方均完整支持。该集合中 passive 的 3 条、active 的 14 条分别通过可定位规则，双方同时通过的仅 1 条。因此不能据峰位差给出稳定的 passive 相对 active 生理延迟。

论文限定五阶段、position、PSM in stage 的十个固定侧条件中，逐录制 `peak_r` 分层均值的 passive − active 点估计全部为负，7 个探索性范围完全位于零下；`r_zero` 点估计也全部为负，7 个范围完全位于零下。这描述了该匹配样本中的位置轨迹协变差异，不是正式显著性或因果效应。空间结果随阶段、器械侧和指标而异，不能把相关降低统一表述为 gaze 更远或更差。

## 2. 预处理、相关和汇总方法

主分析为 `variant=primary`、固定物理器械侧 `side=l/r`，包含 `signal=position/velocity` 和 `mode=PSM in stage/Both in stage`。网格为 41 Hz，最大插值间隙 0.1 s，不外推或跨长缺口插值；主搜索窗为 ±2 s（165 个 lag，步长 1/41 s）。位置在有限连续段内做 3 点居中平均；速度从未做位置平滑的网格坐标独立进行二阶 SG5 平滑，再相邻差分，时间取中点。速度是有符号二维速度，不是速率大小。PSM 的全部有限坐标保留，likelihood、跳变及图外点仅作质量标记。

配对为 `gaze(t−τ)` 与 `PSM(t)`；统一用 `τ` 表示 lag，正 `τ` 表示较早的 gaze 对应当前 PSM，负值方向相反。每个 lag、每个 occurrence 单独按有效配对去均值，再汇总二维协方差与平方和：

$$
r(\tau)=\frac{\sum_o\sum_{i\in I_o(\tau)}(\mathbf g_i-\bar{\mathbf g}_o)\cdot(\mathbf p_i-\bar{\mathbf p}_o)}{\sqrt{\left[\sum_o\sum_{i\in I_o(\tau)}\|\mathbf g_i-\bar{\mathbf g}_o\|^2\right]\left[\sum_o\sum_{i\in I_o(\tau)}\|\mathbf p_i-\bar{\mathbf p}_o\|^2\right]}}.
$$

这里 `o` 为 occurrence，`I_o(τ)` 为该 lag 下的有效配对集合，`g_i=gaze(t_i−τ)`、`p_i=PSM(t_i)`，均值也随 lag 的配对集合变化。不是 x/y Pearson 系数的平均。每个贡献 occurrence 至少 2 对，总配对至少 82 点（实际 `min_overlap=2.0 s`），双方中心化二维 RMS 至少 1，且分母非退化。

`PSM in stage` 仅要求 PSM 端属于该阶段 occurrence；`Both in stage` 要求两端均属于同一次 occurrence。两种 mode 改变配对支持及统计对象，不能把其峰位差直接称为预视时长。相关组曲线只纳入全 lag 窗均有有限相关的录制，先在观看者内等权平均 Fisher z，再在观看者间等权，最后反变换；至少 3 名观看者才报告组曲线。`n_stable_users` 计数的是提供完整 lag 支持的观看者，不表示他们的峰位已通过稳定性检验。

配对比较在每个条件固定 active/passive 共同完整支持的记录对，两域使用相同权重。组曲线均按 **passive 观看者** 分层，active 不在主比较中另按来源操作者分层。统一以 `passive − active` 定义所有差值。组曲线的 `peak_r=max r(τ)`、`peak_lag_s=argmax r(τ)` 与逐录制峰值的分层算术平均是不同统计量；后者在本报告始终标为“逐录制指标分层均值”，并保留 `metric`、`passive_value`、`active_value`、`delta` 字段。

5000 次整观看者重采样给出 2.5–97.5 百分位范围；配对比较同步重采样两域，保留记录配对关系。范围仅作探索性描述，没有校准为生理时延置信区间，也未完整处理观看者/来源操作者的交叉身份依赖。逐 lag 阴影不是同时置信带。保留原峰形、近峰、边界、方向与 `lag_identifiable` 规则；`positive_peak` 仅指峰值相关大于零，不表示正 lag。边界峰、宽峰和不稳定峰均限制精确定位。

保留全部 8 个 variant：`primary`、`position_unsmoothed`、`position_smooth5`、`velocity_box3`、`velocity_sg7`、`velocity_sg11`、`trim_edges`、`window4`（±4 s，329 lag）。配对补充还包括 `observer_equal`、`pair_equal`、`operator_equal` 权重及从观看者/来源操作者两种身份同时删除某人的敏感性。`common_time_primary` 是既有文件技术名称，实际为共同可用坐标样本敏感性，不代表新增时间同步或播放时钟校准。

数值方法沿用冻结 `gt84-fixed-v2.1-npy-20261106`。passive 分析 manifest 记录 Python 3.12.12、依赖清单记录 NumPy 1.26.4；配对比较 manifest 记录 Python 3.11.5/NumPy 1.24.3。方法与参数一致不等于运行环境完全相同。本次文档没有重跑分析。

## 3. 主分析固定侧条件与配对比较

### 3.1 Passive 全部 72 个预定条件

从 `gt_passive16/group_summary.csv` 读取 primary 的 64 行，并依据 `stage_coverage.csv` 补列没有写入该组表的 8 个 A / drop ring 单元。缺失值为“—”，不填零；仅支持人数/记录数在确定为空时记 0。以下峰值均为 **passive 组曲线峰**。

| task | stage | side | signal | mode | n_stable_users | n_complete_records | peak_r | peak_lag_s | peak_lag_resample_low | peak_lag_resample_high | lag_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | drop ring | l | position | PSM in stage | 0 | 0 | — | — | — | — | insufficient_support |
| A | drop ring | l | position | Both in stage | 0 | 0 | — | — | — | — | insufficient_support |
| A | grab ring | l | position | PSM in stage | 7 | 7 | 0.1482 | -1.0732 | -1.7317 | -0.3171 | negative_peak_resampling_range |
| A | grab ring | l | position | Both in stage | 7 | 7 | 0.1520 | -1.8049 | -2.0000 | 1.1951 | window_limited |
| A | place ring | l | position | PSM in stage | 7 | 7 | 0.1249 | -2.0000 | -2.0000 | -1.0000 | window_limited |
| A | place ring | l | position | Both in stage | 7 | 7 | 0.0600 | -0.9512 | -1.1713 | 2.0000 | window_limited |
| A | reach ring | l | position | PSM in stage | 7 | 7 | 0.1047 | -0.1707 | -1.9024 | 0.5122 | direction_uncertain |
| A | reach ring | l | position | Both in stage | 7 | 7 | 0.1024 | -0.1463 | -1.8780 | 0.4390 | direction_uncertain |
| A | transfer ring | l | position | PSM in stage | 7 | 7 | 0.1279 | -0.0732 | -1.6585 | 0.8780 | direction_uncertain |
| A | transfer ring | l | position | Both in stage | 7 | 7 | 0.1279 | -0.0732 | -1.6341 | 0.4945 | direction_uncertain |
| A | drop ring | l | velocity | PSM in stage | 0 | 0 | — | — | — | — | insufficient_support |
| A | drop ring | l | velocity | Both in stage | 0 | 0 | — | — | — | — | insufficient_support |
| A | grab ring | l | velocity | PSM in stage | 7 | 7 | 0.0334 | -1.7561 | -1.8780 | 0.9268 | direction_uncertain |
| A | grab ring | l | velocity | Both in stage | 7 | 7 | 0.0598 | -1.7561 | -2.0000 | 1.1707 | window_limited |
| A | place ring | l | velocity | PSM in stage | 7 | 7 | 0.0331 | 0.1951 | -1.8780 | 0.1951 | direction_uncertain |
| A | place ring | l | velocity | Both in stage | 7 | 7 | 0.0352 | 0.1707 | -1.6098 | 0.1951 | direction_uncertain |
| A | reach ring | l | velocity | PSM in stage | 7 | 7 | 0.0124 | -0.1220 | -0.7073 | 1.7805 | direction_uncertain |
| A | reach ring | l | velocity | Both in stage | 7 | 7 | 0.0125 | -0.1220 | -1.3415 | 1.7805 | direction_uncertain |
| A | transfer ring | l | velocity | PSM in stage | 7 | 7 | 0.0140 | -1.4878 | -1.6585 | 1.4634 | direction_uncertain |
| A | transfer ring | l | velocity | Both in stage | 7 | 7 | 0.0188 | -1.4878 | -1.6585 | 0.9268 | direction_uncertain |
| A | drop ring | r | position | PSM in stage | 0 | 0 | — | — | — | — | insufficient_support |
| A | drop ring | r | position | Both in stage | 0 | 0 | — | — | — | — | insufficient_support |
| A | grab ring | r | position | PSM in stage | 7 | 7 | 0.0476 | -1.5610 | -2.0000 | 1.8293 | window_limited |
| A | grab ring | r | position | Both in stage | 7 | 7 | 0.0607 | -1.4634 | -2.0000 | 2.0000 | window_limited |
| A | place ring | r | position | PSM in stage | 7 | 7 | 0.1215 | -0.4146 | -2.0000 | -0.3415 | window_limited |
| A | place ring | r | position | Both in stage | 7 | 7 | 0.1364 | -0.5610 | -1.2439 | -0.3415 | negative_peak_resampling_range |
| A | reach ring | r | position | PSM in stage | 7 | 7 | 0.2358 | -0.4878 | -2.0000 | 0.6098 | window_limited |
| A | reach ring | r | position | Both in stage | 7 | 7 | 0.2264 | -0.2683 | -2.0000 | 0.4634 | window_limited |
| A | transfer ring | r | position | PSM in stage | 7 | 7 | 0.2503 | 0.3902 | -1.0000 | 0.7805 | direction_uncertain |
| A | transfer ring | r | position | Both in stage | 7 | 7 | 0.2532 | -0.3902 | -1.0732 | 0.0000 | direction_uncertain |
| A | drop ring | r | velocity | PSM in stage | 0 | 0 | — | — | — | — | insufficient_support |
| A | drop ring | r | velocity | Both in stage | 0 | 0 | — | — | — | — | insufficient_support |
| A | grab ring | r | velocity | PSM in stage | 7 | 7 | 0.0306 | -1.5610 | -1.9268 | 1.8537 | direction_uncertain |
| A | grab ring | r | velocity | Both in stage | 7 | 7 | 0.0393 | -1.5610 | -2.0000 | 1.1463 | window_limited |
| A | place ring | r | velocity | PSM in stage | 7 | 7 | 0.0311 | -0.9512 | -0.9756 | 1.4878 | direction_uncertain |
| A | place ring | r | velocity | Both in stage | 7 | 7 | 0.0314 | -0.9512 | -2.0000 | 1.0488 | window_limited |
| A | reach ring | r | velocity | PSM in stage | 7 | 7 | 0.0127 | -0.6829 | -1.7073 | 2.0000 | window_limited |
| A | reach ring | r | velocity | Both in stage | 7 | 7 | 0.0123 | -0.6585 | -1.9756 | 2.0000 | window_limited |
| A | transfer ring | r | velocity | PSM in stage | 7 | 7 | 0.0156 | 0.1463 | -1.7073 | 0.9268 | direction_uncertain |
| A | transfer ring | r | velocity | Both in stage | 7 | 7 | 0.0161 | 0.1463 | -1.7073 | 1.8780 | direction_uncertain |
| C | pull through and cinch | l | position | PSM in stage | 7 | 7 | -0.0151 | -1.9756 | -2.0000 | 0.7317 | window_limited |
| C | pull through and cinch | l | position | Both in stage | 7 | 7 | -0.0620 | -2.0000 | -2.0000 | 1.6585 | window_limited |
| C | reach string | l | position | PSM in stage | 8 | 9 | 0.0845 | -0.2195 | -2.0000 | 1.9512 | window_limited |
| C | reach string | l | position | Both in stage | 8 | 9 | 0.0817 | -0.1951 | -1.7805 | 1.1220 | direction_uncertain |
| C | threading string through tower | l | position | PSM in stage | 1 | 2 | — | — | — | — | insufficient_support |
| C | threading string through tower | l | position | Both in stage | 1 | 2 | — | — | — | — | insufficient_support |
| C | wrap around psm | l | position | PSM in stage | 8 | 9 | 0.0506 | 1.9268 | -2.0000 | 2.0000 | window_limited |
| C | wrap around psm | l | position | Both in stage | 8 | 9 | 0.0214 | 1.9268 | -2.0000 | 2.0000 | window_limited |
| C | pull through and cinch | l | velocity | PSM in stage | 7 | 7 | 0.0168 | -0.1951 | -1.9268 | 1.9756 | direction_uncertain |
| C | pull through and cinch | l | velocity | Both in stage | 7 | 7 | 0.0172 | -1.4146 | -1.9268 | 1.9756 | direction_uncertain |
| C | reach string | l | velocity | PSM in stage | 8 | 9 | 0.0152 | -0.2195 | -1.5366 | 1.8293 | direction_uncertain |
| C | reach string | l | velocity | Both in stage | 8 | 9 | 0.0162 | -1.5366 | -1.5610 | 1.8293 | direction_uncertain |
| C | threading string through tower | l | velocity | PSM in stage | 1 | 2 | — | — | — | — | insufficient_support |
| C | threading string through tower | l | velocity | Both in stage | 1 | 2 | — | — | — | — | insufficient_support |
| C | wrap around psm | l | velocity | PSM in stage | 8 | 9 | 0.0091 | 0.8537 | -1.9512 | 1.9512 | direction_uncertain |
| C | wrap around psm | l | velocity | Both in stage | 8 | 9 | 0.0100 | 1.9268 | -1.9512 | 1.9512 | direction_uncertain |
| C | pull through and cinch | r | position | PSM in stage | 7 | 7 | 0.1417 | -0.9512 | -1.1951 | 2.0000 | window_limited |
| C | pull through and cinch | r | position | Both in stage | 7 | 7 | 0.1634 | -1.0000 | -2.0000 | 0.7317 | window_limited |
| C | reach string | r | position | PSM in stage | 8 | 9 | 0.1320 | -0.2439 | -1.6341 | 2.0000 | window_limited |
| C | reach string | r | position | Both in stage | 8 | 9 | 0.1295 | -0.2439 | -1.6098 | 1.9756 | direction_uncertain |
| C | threading string through tower | r | position | PSM in stage | 1 | 2 | — | — | — | — | insufficient_support |
| C | threading string through tower | r | position | Both in stage | 1 | 2 | — | — | — | — | insufficient_support |
| C | wrap around psm | r | position | PSM in stage | 8 | 9 | 0.1670 | 1.0000 | -2.0000 | 1.4878 | window_limited |
| C | wrap around psm | r | position | Both in stage | 8 | 9 | 0.1389 | 0.0000 | -2.0000 | 0.0000 | window_limited |
| C | pull through and cinch | r | velocity | PSM in stage | 7 | 7 | 0.0306 | -1.8293 | -1.8293 | 1.8293 | direction_uncertain |
| C | pull through and cinch | r | velocity | Both in stage | 7 | 7 | 0.0241 | 1.9512 | -1.4146 | 1.9756 | direction_uncertain |
| C | reach string | r | velocity | PSM in stage | 8 | 9 | 0.0228 | -1.6829 | -1.6829 | 1.8537 | direction_uncertain |
| C | reach string | r | velocity | Both in stage | 8 | 9 | 0.0246 | -1.6829 | -1.8780 | 1.8537 | direction_uncertain |
| C | threading string through tower | r | velocity | PSM in stage | 1 | 2 | — | — | — | — | insufficient_support |
| C | threading string through tower | r | velocity | Both in stage | 1 | 2 | — | — | — | — | insufficient_support |
| C | wrap around psm | r | velocity | PSM in stage | 8 | 9 | 0.0167 | 0.9512 | -1.0244 | 1.3415 | direction_uncertain |
| C | wrap around psm | r | velocity | Both in stage | 8 | 9 | 0.0183 | 1.3415 | -1.6341 | 1.3659 | direction_uncertain |

### 3.2 与来源 Active GT 的匹配位置组曲线峰

以下为 `group_peak_comparison.csv` 的 primary、position、28 个支持充分条件；两域采用相同完整配对集合和观看者权重。表中的 `passive_peak_r`/`active_peak_r` 是组曲线峰，`passive_peak_lag_s`/`active_peak_lag_s` 为其位置（s）。这不是完整 GT84 的组峰，也不是逐录制峰值的均值。不足支持的 16 个预定条件见上表；速度组峰与全部 variant 见 [完整组峰比较](gt_passive16/active_passive_comparison/group_peak_comparison.csv)。

| task | stage | side | mode | n_pairs | n_observers | passive_peak_r | active_peak_r | passive_peak_lag_s | active_peak_lag_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | grab ring | l | PSM in stage | 7 | 7 | 0.1482 | 0.1693 | -1.0732 | -0.3415 |
| A | grab ring | l | Both in stage | 7 | 7 | 0.1520 | 0.1619 | -1.8049 | -1.1463 |
| A | place ring | l | PSM in stage | 7 | 7 | 0.1249 | 0.1217 | -2.0000 | -1.9268 |
| A | place ring | l | Both in stage | 7 | 7 | 0.0600 | 0.0623 | -0.9512 | -0.9024 |
| A | reach ring | l | PSM in stage | 7 | 7 | 0.1047 | 0.2354 | -0.1707 | 0.3902 |
| A | reach ring | l | Both in stage | 7 | 7 | 0.1024 | 0.2317 | -0.1463 | 0.3415 |
| A | transfer ring | l | PSM in stage | 7 | 7 | 0.1279 | 0.3174 | -0.0732 | 0.3902 |
| A | transfer ring | l | Both in stage | 7 | 7 | 0.1279 | 0.2890 | -0.0732 | 0.1951 |
| A | grab ring | r | PSM in stage | 7 | 7 | 0.0476 | 0.1080 | -1.5610 | -0.1951 |
| A | grab ring | r | Both in stage | 7 | 7 | 0.0607 | 0.1859 | -1.4634 | -0.6098 |
| A | place ring | r | PSM in stage | 7 | 7 | 0.1215 | 0.1361 | -0.4146 | -0.2195 |
| A | place ring | r | Both in stage | 7 | 7 | 0.1364 | 0.1167 | -0.5610 | 0.1951 |
| A | reach ring | r | PSM in stage | 7 | 7 | 0.2358 | 0.3599 | -0.4878 | 0.3659 |
| A | reach ring | r | Both in stage | 7 | 7 | 0.2264 | 0.3419 | -0.2683 | 0.1463 |
| A | transfer ring | r | PSM in stage | 7 | 7 | 0.2503 | 0.4182 | 0.3902 | 0.3171 |
| A | transfer ring | r | Both in stage | 7 | 7 | 0.2532 | 0.3967 | -0.3902 | 0.0000 |
| C | pull through and cinch | l | PSM in stage | 7 | 7 | -0.0151 | -0.0292 | -1.9756 | -1.0000 |
| C | pull through and cinch | l | Both in stage | 7 | 7 | -0.0620 | -0.0542 | -2.0000 | -1.0244 |
| C | reach string | l | PSM in stage | 9 | 8 | 0.0845 | 0.3030 | -0.2195 | 0.6098 |
| C | reach string | l | Both in stage | 9 | 8 | 0.0817 | 0.2791 | -0.1951 | 0.1707 |
| C | wrap around psm | l | PSM in stage | 9 | 8 | 0.0506 | 0.1250 | 1.9268 | -2.0000 |
| C | wrap around psm | l | Both in stage | 9 | 8 | 0.0214 | 0.1056 | 1.9268 | -1.9756 |
| C | pull through and cinch | r | PSM in stage | 7 | 7 | 0.1417 | 0.2595 | -0.9512 | 2.0000 |
| C | pull through and cinch | r | Both in stage | 7 | 7 | 0.1634 | 0.1304 | -1.0000 | -2.0000 |
| C | reach string | r | PSM in stage | 9 | 8 | 0.1320 | 0.3052 | -0.2439 | 0.6341 |
| C | reach string | r | Both in stage | 9 | 8 | 0.1295 | 0.2817 | -0.2439 | 0.0000 |
| C | wrap around psm | r | PSM in stage | 9 | 8 | 0.1670 | 0.4688 | 1.0000 | -0.1707 |
| C | wrap around psm | r | Both in stage | 9 | 8 | 0.1389 | 0.4673 | 0.0000 | -0.3171 |

### 3.3 论文限定阶段的逐录制峰值分层均值

以下固定 `variant=primary`、`signal=position`、`mode=PSM in stage`、`weighting=observer_equal`、`metric=peak_r`。先求每条录制曲线的峰值，再两级算术平均。`delta=passive_value−active_value`；`low/high` 为该差值的探索性 2.5–97.5 百分位范围，不能改称组峰差或正式置信区间。

| task | stage | side | n_pairs | n_observers | passive_value | active_value | delta | low | high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | place ring | l | 7 | 7 | 0.1617 | 0.1803 | -0.0186 | -0.0640 | 0.0296 |
| A | reach ring | l | 7 | 7 | 0.1489 | 0.2456 | -0.0967 | -0.1159 | -0.0784 |
| A | transfer ring | l | 7 | 7 | 0.2027 | 0.3266 | -0.1239 | -0.2214 | -0.0553 |
| A | place ring | r | 7 | 7 | 0.1918 | 0.2061 | -0.0143 | -0.1276 | 0.1047 |
| A | reach ring | r | 7 | 7 | 0.2965 | 0.3650 | -0.0686 | -0.1544 | 0.0163 |
| A | transfer ring | r | 7 | 7 | 0.3127 | 0.4324 | -0.1197 | -0.2233 | -0.0246 |
| C | reach string | l | 9 | 8 | 0.1514 | 0.3792 | -0.2278 | -0.3312 | -0.1064 |
| C | wrap around psm | l | 9 | 8 | 0.0983 | 0.2183 | -0.1200 | -0.2266 | -0.0311 |
| C | reach string | r | 9 | 8 | 0.1857 | 0.3629 | -0.1772 | -0.2798 | -0.0506 |
| C | wrap around psm | r | 9 | 8 | 0.2027 | 0.4836 | -0.2809 | -0.4164 | -0.1666 |

例如 C / wrap around psm / r 的逐录制峰值分层均值为 passive 0.2027、active 0.4836，`delta`=−0.2809，范围 [−0.4164, −0.1666]。同一条件 passive 单域组曲线峰为 0.1670；数值不同来自统计层级不同，并非字段计算冲突。

零 lag 空间差异并不一致：A / reach ring / l 的 `distance_median_px` 差为 −55.7 px，范围 [−81.1, −27.6]；r 侧为 +78.4 px，范围 [28.4, 128.0]。C / wrap around psm / r 的距离差为 +48.9 px，范围 [12.4, 88.6]，`proximity_50px` 差为 −14.32 个百分点，范围 [−24.99, −4.28] 个百分点。以上均为逐录制指标的观看者分层均值及探索性范围，不能概括为所有阶段的统一空间效应。

## 4. 指标说明与统一命名

| 字段/符号 | 统一含义与适用层级 |
| --- | --- |
| `task / stage / side / signal / mode / variant` | 原始条件键；阶段保持原英文名称，`l/r` 始终指固定物理器械侧 |
| `recording_id / content_id` | passive 冻结表中同值的唯一观看录制键；不是来源视频身份 |
| `source_content_id / active_id` | 对应来源操作视频的录制身份；匹配清单中的 `passive_id` 对应观看录制键 |
| `observer_user / source_user` | 观看者/来源操作者；passive 表的 `user` 等于 `observer_user` |
| `source=gt` | 冻结 passive 表中的人工 gaze 来源标签；不据此将该行误认为 active 数据 |
| `τ / lag_s / peak_lag_s` | `τ` 或 `lag_s` 为秒单位 lag；`peak_lag_s` 是指定曲线有符号最大相关的位置 |
| `peak_r / r_zero` | 指定曲线的最大有符号相关/零 lag 相关，不是空间误差；同表声明录制或组曲线层级 |
| `passive_peak_r / active_peak_r` | `group_peak_comparison.csv` 中为两域匹配组曲线的峰值；`paired_metrics.csv` 中为各自逐录制峰值，必须随文件层级解释 |
| `passive_peak_lag_s / active_peak_lag_s` | `group_peak_comparison.csv` 中为匹配组曲线峰位；`paired_metrics.csv` 中为逐录制峰位，单位均为 s，必须随文件层级解释 |
| `delta_peak_lag_s / delta_group_peak_lag_s` | 前者为 `paired_metrics.csv` 中逐录制峰位的 passive − active 差值；后者为匹配组曲线峰位差，二者不互换，也不等于前者的分层均值 |
| `metric / passive_value / active_value / delta` | `paired_group_summary.csv` 中，先算逐录制 `metric`，再在观看者内/间两级算术平均；`delta=passive_value−active_value` |
| `peak_lag_resample_low / peak_lag_resample_high` | 单域组曲线峰位重采样百分位端点，单位 s；沿用 GT84 命名，不另改为“CI” |
| `low / high` | 配对 `delta` 的探索性重采样端点，单位由 `metric` 决定；不是前一行的单域峰位范围 |
| `n_stable_users / n_complete_records` | 单域全 lag 完整支持的观看者数/录制数；不等于峰位可定位人数/曲线数 |
| `n_observers / n_operators / n_pairs` | 配对集合中的观看者数/来源操作者数/记录对数；分母随条件支持变化 |
| `near_peak_*` | 距峰顶相关值 0.01/0.05/0.10 以内的区间、跨度、连续宽度与占比；不表示秒单位的相关容差 |
| `peak_at_boundary / bootstrap_boundary_frequency` | 原曲线峰触及搜索边界/重采样峰触及边界的比例 |
| `lag_identifiable / lag_interpretation` | 逐录制可定位规则/组峰方向描述；两者不可互换 |
| `distance_median_px / proximity_50px` | 逐录制配对点欧氏距离中位数/距离 ≤50 px 的比例，再按观看者分层平均；不是合并全体点后的中位数/比例 |
| `at=zero / at=group_peak` | 零 lag/该单域组峰 lag 的空间指标；直接 active/passive 空间比较使用零 lag |

相关、比例为无量纲；正文以百分数展示 `proximity_50px` 时乘 100，其差值须称“百分点”。表内如未标百分号则仍为 0–1 比例。数学含义相同的变量沿用同一基本名称；域差异使用 `passive_`/`active_`，统计层级差异明确写出，不将不同估计量合并命名。

## 5. 主图和敏感性图

### 5.1 Passive 单域分析

以下沿用原报告的 13 组 PNG/PDF，保持完整预定阶段、左右侧、两种信号及 mode 的展示。

### A 位置相关曲线

![A 位置相关曲线](gt_passive16/figures/primary_position_curves_A.png)

[PDF](gt_passive16/figures/primary_position_curves_A.pdf)

### C 位置相关曲线

![C 位置相关曲线](gt_passive16/figures/primary_position_curves_C.png)

[PDF](gt_passive16/figures/primary_position_curves_C.pdf)

### 位置组峰位及支持

![位置组峰位及支持](gt_passive16/figures/primary_position_peak_lags.png)

[PDF](gt_passive16/figures/primary_position_peak_lags.pdf)

### A 速度相关曲线

![A 速度相关曲线](gt_passive16/figures/primary_velocity_curves_A.png)

[PDF](gt_passive16/figures/primary_velocity_curves_A.pdf)

### C 速度相关曲线

![C 速度相关曲线](gt_passive16/figures/primary_velocity_curves_C.png)

[PDF](gt_passive16/figures/primary_velocity_curves_C.pdf)

### 速度组峰位及支持

![速度组峰位及支持](gt_passive16/figures/primary_velocity_peak_lags.png)

[PDF](gt_passive16/figures/primary_velocity_peak_lags.pdf)

### A 位置观看者重采样

![A 位置观看者重采样](gt_passive16/figures/group_bootstrap_position_A.png)

[PDF](gt_passive16/figures/group_bootstrap_position_A.pdf)

### C 位置观看者重采样

![C 位置观看者重采样](gt_passive16/figures/group_bootstrap_position_C.png)

[PDF](gt_passive16/figures/group_bootstrap_position_C.pdf)

### A 速度观看者重采样

![A 速度观看者重采样](gt_passive16/figures/group_bootstrap_velocity_A.png)

[PDF](gt_passive16/figures/group_bootstrap_velocity_A.pdf)

### C 速度观看者重采样

![C 速度观看者重采样](gt_passive16/figures/group_bootstrap_velocity_C.png)

[PDF](gt_passive16/figures/group_bootstrap_velocity_C.pdf)

### 位置处理与窗口敏感性

![位置处理与窗口敏感性](gt_passive16/figures/position_sensitivity_lag_change.png)

[PDF](gt_passive16/figures/position_sensitivity_lag_change.pdf)

### 速度滤波敏感性

![速度滤波敏感性](gt_passive16/figures/velocity_sensitivity_lag_change.png)

[PDF](gt_passive16/figures/velocity_sensitivity_lag_change.pdf)

### 零 lag 与组峰位移位后的空间接近度

![零 lag 与组峰位移位后的空间接近度](gt_passive16/figures/spatial_zero_vs_peak.png)

[PDF](gt_passive16/figures/spatial_zero_vs_peak.pdf)

### 5.2 Passive 与匹配 Active GT 比较

蓝色为来源操作者 active，橙色为观看者 passive。差值统一为 passive − active；曲线阴影和指标横线均为探索性百分位范围。

### Task A：位置相关曲线对照

![Task A：位置相关曲线对照](gt_passive16/active_passive_comparison/figures/curves_position_A.png)

蓝色为来源操作者 active，橙色为对应观看者 passive。两条曲线采用同一配对集合和同一权重；少于 3 名观看者时不绘制组曲线。

[PDF](gt_passive16/active_passive_comparison/figures/curves_position_A.pdf)

### Task A：位置相关差值曲线

![Task A：位置相关差值曲线](gt_passive16/active_passive_comparison/figures/delta_position_A.png)

每个条件固定共同完整支持的记录对，按观看者等权汇总。阴影为按观看者同步重采样的逐 lag 探索性百分位范围，不是同时置信带。

[PDF](gt_passive16/active_passive_comparison/figures/delta_position_A.pdf)

### Task C：位置相关曲线对照

![Task C：位置相关曲线对照](gt_passive16/active_passive_comparison/figures/curves_position_C.png)

蓝色为来源操作者 active，橙色为对应观看者 passive。两条曲线采用同一配对集合和同一权重；少于 3 名观看者时不绘制组曲线。

[PDF](gt_passive16/active_passive_comparison/figures/curves_position_C.pdf)

### Task C：位置相关差值曲线

![Task C：位置相关差值曲线](gt_passive16/active_passive_comparison/figures/delta_position_C.png)

每个条件固定共同完整支持的记录对，按观看者等权汇总。阴影为按观看者同步重采样的逐 lag 探索性百分位范围，不是同时置信带。

[PDF](gt_passive16/active_passive_comparison/figures/delta_position_C.pdf)

### Task A：速度相关曲线对照

![Task A：速度相关曲线对照](gt_passive16/active_passive_comparison/figures/curves_velocity_A.png)

蓝色为来源操作者 active，橙色为对应观看者 passive。两条曲线采用同一配对集合和同一权重；少于 3 名观看者时不绘制组曲线。

[PDF](gt_passive16/active_passive_comparison/figures/curves_velocity_A.pdf)

### Task A：速度相关差值曲线

![Task A：速度相关差值曲线](gt_passive16/active_passive_comparison/figures/delta_velocity_A.png)

每个条件固定共同完整支持的记录对，按观看者等权汇总。阴影为按观看者同步重采样的逐 lag 探索性百分位范围，不是同时置信带。

[PDF](gt_passive16/active_passive_comparison/figures/delta_velocity_A.pdf)

### Task C：速度相关曲线对照

![Task C：速度相关曲线对照](gt_passive16/active_passive_comparison/figures/curves_velocity_C.png)

蓝色为来源操作者 active，橙色为对应观看者 passive。两条曲线采用同一配对集合和同一权重；少于 3 名观看者时不绘制组曲线。

[PDF](gt_passive16/active_passive_comparison/figures/curves_velocity_C.pdf)

### Task C：速度相关差值曲线

![Task C：速度相关差值曲线](gt_passive16/active_passive_comparison/figures/delta_velocity_C.png)

每个条件固定共同完整支持的记录对，按观看者等权汇总。阴影为按观看者同步重采样的逐 lag 探索性百分位范围，不是同时置信带。

[PDF](gt_passive16/active_passive_comparison/figures/delta_velocity_C.pdf)

### 零 lag 相关的逐记录配对比较

![零 lag 相关的逐记录配对比较](gt_passive16/active_passive_comparison/figures/paired_r_zero.png)

每个点代表同一来源视频、同阶段、同侧、同信号和同 mode 的一对结果。虚线为两者相等；点不代表彼此独立的受试者。

[PDF](gt_passive16/active_passive_comparison/figures/paired_r_zero.pdf)

### 峰值相关的逐记录配对比较

![峰值相关的逐记录配对比较](gt_passive16/active_passive_comparison/figures/paired_peak_r.png)

每个点代表同一来源视频、同阶段、同侧、同信号和同 mode 的一对结果。虚线为两者相等；点不代表彼此独立的受试者。

[PDF](gt_passive16/active_passive_comparison/figures/paired_peak_r.pdf)

### 位置指标的配对差异汇总

![位置指标的配对差异汇总](gt_passive16/active_passive_comparison/figures/group_differences_position.png)

点为观看者分层算术平均的配对差异，横线为 5000 次观看者同步重采样的探索性 2.5–97.5 百分位范围，不作正式显著性判定。P 为 PSM in stage，B 为 Both in stage；l/r 为固定左/右器械。

[PDF](gt_passive16/active_passive_comparison/figures/group_differences_position.pdf)

### 速度指标的配对差异汇总

![速度指标的配对差异汇总](gt_passive16/active_passive_comparison/figures/group_differences_velocity.png)

点为观看者分层算术平均的配对差异，横线为 5000 次观看者同步重采样的探索性 2.5–97.5 百分位范围，不作正式显著性判定。P 为 PSM in stage，B 为 Both in stage；l/r 为固定左/右器械。

[PDF](gt_passive16/active_passive_comparison/figures/group_differences_velocity.pdf)

### 零 lag 空间接近度

![零 lag 空间接近度](gt_passive16/active_passive_comparison/figures/spatial_zero.png)

距离越小、接近比例越高，表示 gaze 越接近器械。这里只比较共同零 lag 条件，未混用原来两组各自的组峰位。

[PDF](gt_passive16/active_passive_comparison/figures/spatial_zero.pdf)

### 峰值 lag 与定位可靠性

![峰值 lag 与定位可靠性](gt_passive16/active_passive_comparison/figures/lag_reliability.png)

峰位图仅作描述。边界峰及未通过可靠性规则的峰值不能解释为精确的生理领先／滞后时间。重合点可能覆盖。

[PDF](gt_passive16/active_passive_comparison/figures/lag_reliability.pdf)

### 预处理配置敏感性

![预处理配置敏感性](gt_passive16/active_passive_comparison/figures/sensitivity_variants.png)

颜色表示零 lag 相关的配对组差异。各配置按自身双方完整支持的配对集合汇总；覆盖差异应结合 sensitivity_summary.csv 查看。P 为 PSM in stage，B 为 Both in stage。

[PDF](gt_passive16/active_passive_comparison/figures/sensitivity_variants.pdf)

### 汇总权重敏感性

![汇总权重敏感性](gt_passive16/active_passive_comparison/figures/sensitivity_weights.png)

观看者等权为主分析，另外比较记录对等权和来源操作者等权。每种权重内部，active/passive 使用相同配对和权重。

[PDF](gt_passive16/active_passive_comparison/figures/sensitivity_weights.pdf)

### 附录：共同可用坐标样本的相关指标敏感性

![附录：共同可用坐标样本的相关指标敏感性](gt_passive16/active_passive_comparison/figures/common_time_correlation.png)

每点为同一组条件的 Passive − Active：横轴复用原结果，纵轴将两域过滤后的 gaze 有限有效坐标掩码取交集后重算。虚线表示差异不变。此项仅为可选的坐标可用性敏感性分析，不改变已按来源映射和 NPY 行号确定的配对，不新增时间戳对齐条件。

[PDF](gt_passive16/active_passive_comparison/figures/common_time_correlation.pdf)

### 附录：共同可用坐标样本的空间指标敏感性

![附录：共同可用坐标样本的空间指标敏感性](gt_passive16/active_passive_comparison/figures/common_time_spatial.png)

每点为同一组条件的 Passive − Active：横轴复用原结果，纵轴将两域过滤后的 gaze 有限有效坐标掩码取交集后重算。虚线表示差异不变。此项仅为可选的坐标可用性敏感性分析，不改变已按来源映射和 NPY 行号确定的配对，不新增时间戳对齐条件。

[PDF](gt_passive16/active_passive_comparison/figures/common_time_spatial.pdf)


## 6. 数据来源、验证和可比较性

passive 的全部 440 条 primary 曲线曾由独立显式配对核复算，有限掩码一致，最大绝对相关误差 8.1324×10⁻¹⁴；GT84 代表性 A/C 的 72 条回归曲线最大误差 2.2204×10⁻¹⁶，原生 NPY 到冻结 gaze 网格误差为 0 px。配对独立验证检查全部 2420 条逐录制配对指标、364 条组曲线及组峰，组曲线最大绝对误差为 0，并复核同步重采样数组。它们验证实现与冻结数据的一致性，不验证实际播放时间或生理时延。

passive 两套非配对参考各运行每组 5000 次尝试；主参考 56 个观察支持充分条件的有效参考数为 114–5000。donor 为同 task、不同观看者，规则允许共享来源视频，本次 16 个来源视频互异，实际同源 donor 配对为 0。尾部比例及其 BH 调整是参考峰高的描述性比较，不能当作 lag 方向的正式 p 值或 FDR 控制。该非配对参考与本报告的 active/passive 来源视频配对比较是不同分析。

本报告保留全部已观测阶段和缺失预定阶段；论文摘要仅取 A 的 reach ring、transfer ring、place ring 与 C 的 reach string、wrap around psm。完整 GT84 与 matched active 的样本、支持和权重不同，不能混用其组峰或范围。2026-11-05 的历史 CSV-GT 结果也不能当作本次 NPY-GT 基准。

本样本只包含 novice 观看者与 novice 来源操作者，未作专家/新手受控比较。相关强度、峰 lag 与空间接近度回答不同问题；观看/操作状态、样本组成、输入构造、有效支持及交叉身份依赖均限制解释。报告不据此推断技能、模型架构机制、因果效应或普遍预视时长。

[输入来源清单](gt_passive16/input_manifest.json) · [纳入记录](gt_passive16/analysis_record_inventory.csv) · [方法一致性](gt_passive16/methods_consistency.md) · [覆盖](gt_passive16/stage_coverage.csv) · [逐录制结果](gt_passive16/record_summary.csv) · [单域组峰](gt_passive16/group_summary.csv) · [空间结果](gt_passive16/spatial_summary.csv)

[来源配对清单](gt_passive16/active_passive_comparison/paired_record_mapping.csv) · [配对覆盖](gt_passive16/active_passive_comparison/paired_condition_coverage.csv) · [逐录制配对](gt_passive16/active_passive_comparison/paired_metrics.csv) · [指标分层均值](gt_passive16/active_passive_comparison/paired_group_summary.csv) · [匹配组曲线](gt_passive16/active_passive_comparison/paired_group_curves.csv) · [配对组峰](gt_passive16/active_passive_comparison/group_peak_comparison.csv)

[非配对参考](gt_passive16/unpaired_reference.csv) · [Passive 独立验证](gt_passive16/passive_independent_validation.json) · [配对独立验证](gt_passive16/active_passive_comparison/independent_validation.json) · [单域交互报告](gt_passive16/report.html) · [配对交互报告](gt_passive16/active_passive_comparison/report.html) · [论文摘要](gt_passive_paper_summary.md)
