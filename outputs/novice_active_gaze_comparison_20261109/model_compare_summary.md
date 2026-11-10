# 五种模型对 novice active user gaze 的复现程度：中文分析与总结

分析日期：2026-11-09。数据口径：当前 NPY 为 GT；测试集、novice、active；逐帧距离覆盖全部 15 个视频，stage JSD 仅覆盖 A/C。

## 1. 核心结论

**在当前测试样本与既定计算方法下，all_ABCD 与 single_task 的逐帧位置表现基本持平；single_task 在多数 stage 的空间分布上更接近 GT；spd_finetune 优于 spd_zero_shot；leave_out 的总体距离误差最大。**

主要证据如下。

1. **逐帧位置：all_ABCD 与 single_task 几乎持平。** 二者平均误差分别为画面对角线的 **9.6342%** 和 **9.6304%**，仅相差 **0.0038 个百分点**。配对差值 all_ABCD−single_task 的 95% 用户聚类 bootstrap 区间为 **[−0.4351, 0.2875]** 个百分点，包含 0。因此不能依据这一极小数值差距，声称 single_task 的逐帧位置准确性明确优于 all_ABCD。
2. **阶段分布：single_task 在 9 个 stage 中有 6 个 JSD 数值最低。** 这包括 A 的 reach ring、grab ring，以及 C 的全部 4 个 stage。all_ABCD 在 A 的 transfer ring、place ring 最低。该计数仅是逐行描述，不是按 stage 等权构造的新综合指标，也不是显著性检验。
3. **finetune 相对 zero-shot 的改善较一致。** 平均距离从 **11.7377%** 降至 **9.8244%**，降低 **1.9133 个百分点**，相对降低 **16.30%**。finetune 在 **13/15 个视频**的距离上更低，在 **8/9 个 stage**的 JSD 上更低。
4. **leave_out 的总体距离为 12.6042%，五类模型中最高。** all_ABCD 在 **14/15 个视频**上具有更低的距离误差。不过，leave_out 并非每个 stage 都最差，任务与阶段差异需要分别看待。

**对于“是否复现了 novice active gaze”，本报告给出的是复现程度和模型之间的相对证据，不能给出无条件的通过／不通过判定。** 当前分析没有人类之间的一致性参照、预先设定的可接受误差阈值，或与简单注视基线的比较。不能仅凭模型排名断言某模型已经达到人类 gaze 的复现水平。

## 2. 数据范围、有效帧与匹配

### 2.1 数据支持

| 项目 | 本次实际纳入情况 |
| --- | --- |
| 筛选条件 | `domain="active"`、`meta.split="test"`、`meta.src_skill="novice"` |
| 测试视频／用户 | 15 个视频／10 位用户 |
| 任务分布 | A：5 个；B：4 个；C：5 个；D：1 个视频 |
| GT 总帧数 | 154,296 |
| 无效坐标或超出范围的 GT 帧数 | 6,842 |
| 五模型共同有效距离评估帧 | 147,454 |
| 有效 GT 因模型覆盖缺失而额外剔除的帧 | 0 |
| stage 分析范围 | A/C 的 10 个视频、9 个语义 stage |
| 有效且具有语义 stage 标注的帧数 | 104,966 |
| 视频分辨率 | 全部为 1280 × 1024 |
| prediction 文件数 | 105：all_ABCD 15、single_task 15、leave_out 15、finetune 45、zero-shot 15 |

来源：[audit.json](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/audit.json)、[audit_clips.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/audit_clips.csv)、[input_manifest.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/input_manifest.csv)。距离评估包含 A/C 中 stage 为 `none` 或空字符串的有效 gaze 帧；这些帧不进入 stage JSD。B/D 缺少 stage 字段，本次不推断或补造其阶段标签。

### 2.2 当前 GT 与模型运行版本

用 prediction 的 `frame_index` 作为当前 NPY 的零基行号，并核验 prediction 的原始 `frame_number` 与 NPY 的 `Frame_Number` 一致。所有输入通过这一帧身份核验。

四个 A 视频——`User7_1_1`、`User9_7_1`、`User11_6_1`、`User12_3_1`——的当前 GT 坐标与 prediction 内缓存的 `target` 不同。本次距离及 JSD 均读取当前 NPY，不复用缓存 target。`User1_1_1` 的当前时间戳与缓存不一致，因此时间戳不作为匹配键。这些差异不说明其发生原因，具体核查记录见 [audit_predictions.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/audit_predictions.csv)。

| 模型类别 | 本次使用的运行 |
| --- | --- |
| all_ABCD | seed 43 |
| single_task | A：seed 43；B：42；C：44；D：43 |
| leave_out | A：seed 44；B：44；C：42；D：43 |
| spd_finetune | seed 42、43、44；先分别算指标，再平均指标 |
| spd_zero_shot | seed 43 |

single_task 与 leave_out 分别按测试视频的任务匹配对应子模型。finetune 不是三个 seed 的概率图集成。其他模型现有预测的 seed 数不足以评价训练随机性的影响，因此本报告结论限定于这些已保存的运行；不据此单独归因于训练任务数量或预训练策略。

## 3. 指标如何计算

### 3.1 逐帧距离：主要指标

对每帧完整 `32×40` 概率图取概率加权均值：

$$
\hat x_t=\sum_{r=0}^{31}\sum_{c=0}^{39}P_t(r,c)\frac{c}{39},\qquad
\hat y_t=\sum_{r=0}^{31}\sum_{c=0}^{39}P_t(r,c)\frac{r}{31}.
$$

将归一化坐标差换算为原始画面上的欧氏距离，再换算为画面对角线百分比：

$$
d_t=\sqrt{[1279(\hat x_t-x_t^{GT})]^2+[1023(\hat y_t-y_t^{GT})]^2},\qquad
e_t=100\frac{d_t}{\sqrt{1279^2+1023^2}}.
$$

先对每个视频内的共同有效帧取均值，再对 15 个视频等权平均。**报告的总体均值不是将全部帧混合后的帧加权均值，也不是各用户等权的均值。** 表中的像素与百分比表示同一距离指标的不同单位。

95% 区间采用 10,000 次用户聚类 bootstrap，随机种子固定为 20261109；每次抽取用户并保留其全部视频，再计算视频等权均值。模型间差值使用同一组用户重采样，保留配对关系。区间反映用户样本的不确定性，不代表训练 seed 的不确定性。

### 3.2 Stage JSD：阶段内空间关注分布

每个“视频 × stage”内，对同一批有效帧的预测概率图取平均得到 P。对每帧当前 GT 坐标，在相同 `32×40` 网格上生成 σ=1 网格单位的归一化高斯图，再取平均得到 Q。令 M=(P+Q)/2：

$$
JSD(P,Q)=\frac12\sum_j P_j\log_2\frac{P_j}{M_j}+
\frac12\sum_j Q_j\log_2\frac{Q_j}{M_j}.
$$

JSD 范围为 0–1，越小表示越相似。先逐视频、stage、seed 计算，再在同一任务和 stage 内对视频等权平均；不存在的 stage 不填 0。预测使用完整概率图，不将均值坐标重新绘制为预测热图。

两个指标相互补充：距离检验同一帧是否看向相近位置；stage JSD 检验该阶段的整体空间关注分布。**较低的 stage JSD 不等于阶段内部的注视时间顺序也被复现。** 两个指标的尺度和目标不同，不合并成一个分数。

## 4. 全部 15 个视频的逐帧距离结果

### 4.1 五模型总体对比

| 模型 | 平均距离（对角线 %） | 95% 用户聚类 bootstrap 区间 | 平均距离（像素） |
| --- | --- | --- | --- |
| all_ABCD | 9.6342 | [8.1786, 11.4532] | 157.79 |
| single_task | 9.6304 | [8.0286, 11.6357] | 157.73 |
| leave_out | 12.6042 | [10.9427, 14.7085] | 206.43 |
| spd_finetune | 9.8244 | [8.3164, 11.7055] | 160.90 |
| spd_zero_shot | 11.7377 | [10.3870, 13.4614] | 192.24 |

数据来源：[model_comparison.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/model_comparison.csv)。均值均基于相同的 147,454 个有效帧、15 个视频。

![五模型总体逐帧距离对比](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/distance_comparison.png)

### 4.2 配对比较及其解释

下表差值为模型 A 减模型 B，单位为**画面对角线百分比的百分点**。负数表示 A 的距离误差更低。区间为未作多重比较校正的 95% 配对 bootstrap 区间；本报告不把这 10 个比较当作经过校正的显著性检验。

| 模型 A | 模型 B | 差值 A−B（百分点） | 95% 配对区间 |
| --- | --- | --- | --- |
| all_ABCD | single_task | +0.0038 | [-0.4351, +0.2875] |
| all_ABCD | leave_out | -2.9700 | [-3.5631, -2.2226] |
| all_ABCD | spd_finetune | -0.1901 | [-0.3754, -0.0730] |
| all_ABCD | spd_zero_shot | -2.1035 | [-2.7370, -1.2129] |
| single_task | leave_out | -2.9738 | [-3.4923, -2.2804] |
| single_task | spd_finetune | -0.1940 | [-0.3801, +0.0767] |
| single_task | spd_zero_shot | -2.1073 | [-2.8806, -0.8864] |
| leave_out | spd_finetune | +2.7798 | [+2.0654, +3.3218] |
| leave_out | spd_zero_shot | +0.8665 | [+0.1724, +1.7916] |
| spd_finetune | spd_zero_shot | -1.9133 | [-2.6218, -0.8800] |

来源：[pairwise_distance.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/pairwise_distance.csv)。

- **all_ABCD 与 single_task：** 平均差仅 +0.0038 个百分点，区间跨 0。虽然 single_task 在 10/15 个视频上数值更低，但每个视频的改善幅度不同，因此胜出视频数不能替代平均误差。两者总体接近的判断比“single_task 最好”的单纯排序更符合证据。
- **finetune 与 zero-shot：** 差值为 −1.9133 个百分点，区间 [−2.6218, −0.8800]。finetune 在 13/15 个视频上更低，支持其在当前评估样本上有较一致的改善。例外是 `User5_3_2` 和 `User10_3_1`；其误差分别从 zero-shot 的 11.9470%、11.6317% 增至 finetune 的 12.6942%、13.7442%。
- **all_ABCD 与 finetune：** all_ABCD 低 0.1901 个百分点，区间 [−0.3754, −0.0730]。这描述当前已保存运行的差异，幅度远小于 finetune 相对 zero-shot 的改善，不能外推为 all_ABCD 在所有训练 seed 上均更优。
- **single_task 与 finetune：** single_task 数值低 0.1940 个百分点，但区间 [−0.3801, 0.0767] 跨 0。因此不能将两个模型的数值排序解释成已确定的总体优劣。
- **leave_out：** 相比 all_ABCD，距离高 2.9700 个百分点；相比 zero-shot，高 0.8665 个百分点。前者区间完全为正（将表中差值反向），且 all_ABCD 在 14/15 个视频更低，说明当前 leave_out 设置的逐帧位置泛化表现较弱。数据本身不能分离任务覆盖、具体训练运行和其他配置对这一差异的贡献。

### 4.3 任务差异解释了总体近似并列

下表在每个任务内部对视频等权平均，仍为同一个距离指标。

| 任务 | 视频数 | 有效帧数 | all_ABCD | single_task | leave_out | spd_finetune | spd_zero_shot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | 61,798 | 12.5609 | 12.3884 | 16.3004 | 12.6805 | 16.4386 |
| B | 4 | 31,664 | 8.6468 | 9.2625 | 12.2123 | 9.1215 | 9.3064 |
| C | 5 | 45,626 | 7.9906 | 7.8361 | 10.1633 | 8.0836 | 9.4602 |
| D | 1 | 8,366 | 7.1692 | 6.2839 | 7.8954 | 7.0594 | 9.3462 |

由 [video_distance.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/video_distance.csv) 按任务汇总。全体 15 视频均值不是对 A/B/C/D 四个任务再等权平均，任务权重由视频数 5、4、5、1 决定。

single_task 在 A、C、D 的均值低于 all_ABCD，但在 B 高于 all_ABCD（9.2625% 对 8.6468%）。尤其 `User10_3_1` 的误差为 single_task 14.7984%、all_ABCD 12.6191%，抵消了部分其他视频上的优势。由此可以解释：single_task 虽在更多视频上数值较好，整体均值却与 all_ABCD 几乎一致。

A 任务中五模型的平均误差均高于各自的 B/C/D 均值；但不同任务的视频、用户和 gaze 分布不同，这只是当前样本的描述，不能据此认定 A 在所有条件下更难。D 只有一个视频，不足以支持任务层面的普遍结论。

### 4.4 finetune 的 seed 结果

| spd_finetune seed | 15 视频平均距离（对角线 %） |
| --- | --- |
| 42 | 9.9977 |
| 43 | 9.6937 |
| 44 | 9.7817 |

来源：[video_distance_by_seed.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/video_distance_by_seed.csv)。三个 seed 的总体均值处于 9.6937%–9.9977%，均低于当前 zero-shot 的 11.7377%；按计划对三个 seed 的指标平均得到 9.8244%。这不是匹配多个 zero-shot seed 的训练随机性实验，不能将其解读为完整的跨 seed 因果对照。

## 5. A/C 的 stage 注视分布结果

以下均为视频级 JSD 的等权平均。每行粗体表示未四舍五入数值的最小值；最低值的微小差异不自动意味着具有统计或实际意义的优势。stage JSD 本次未计算置信区间或执行显著性检验。

### 5.1 任务 A

| Stage | 视频数 | 有效帧数 | all_ABCD | single_task | leave_out | spd_finetune | spd_zero_shot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reach ring | 5 | 19,738 | 0.1675 | **0.1629** | 0.3525 | 0.1726 | 0.2629 |
| grab ring | 5 | 8,022 | 0.2754 | **0.2475** | 0.5625 | 0.2751 | 0.4620 |
| transfer ring | 5 | 20,589 | **0.1627** | 0.1629 | 0.1922 | 0.1741 | 0.2478 |
| place ring | 5 | 11,486 | **0.2450** | 0.2530 | 0.2454 | 0.2610 | 0.3207 |
| drop ring | 1 | 30 | 0.6621 | 0.6158 | 0.6130 | 0.6439 | **0.5651** |

数据来源：[stage_jsd_comparison.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/stage_jsd_comparison.csv)；视频级数据见 [stage_jsd_video.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/stage_jsd_video.csv)。

- **reach ring、grab ring：** single_task 的 JSD 分别为 0.1629、0.2475，低于 all_ABCD 的 0.1675、0.2754。leave_out 分别达到 0.3525、0.5625，偏差明显更大。特别在 grab ring，当前 leave_out 的空间分布与 GT 的差异较为突出。
- **transfer ring：** all_ABCD 为 0.1627、single_task 为 0.1629，几乎一致。不能把这一行的最小值标记当作强优势证据。
- **place ring：** all_ABCD 为 0.2450，leave_out 为 0.2454，两者很接近；single_task 为 0.2530。说明 leave_out 的较差总体距离并不意味着它在每个阶段的空间分布都最差。
- **drop ring：** zero-shot 的 0.5651 为本行最低，但只有 **1 个视频、30 个有效帧**。该结果只能描述这一小段样本，不据此判断 zero-shot 更能复现掉环行为，也不因其余模型 JSD 高于其他 stage 就作总体能力结论。

### 5.2 任务 C

| Stage | 视频数 | 有效帧数 | all_ABCD | single_task | leave_out | spd_finetune | spd_zero_shot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reach string | 5 | 14,446 | 0.2413 | **0.2249** | 0.3466 | 0.2552 | 0.3223 |
| threading string through tower | 3 | 830 | 0.3330 | **0.3179** | 0.4025 | 0.3335 | 0.4382 |
| wrap around psm | 5 | 21,433 | 0.1756 | **0.1669** | 0.3298 | 0.1848 | 0.2515 |
| pull through and cinch | 4 | 8,392 | 0.2005 | **0.1797** | 0.3820 | 0.2122 | 0.3063 |

single_task 在 C 的四个 stage 均取得最低 JSD：reach string 0.2249、threading string through tower 0.3179、wrap around psm 0.1669、pull through and cinch 0.1797。相较 all_ABCD，对应降低约 0.0163、0.0151、0.0087 和 0.0208，表明这一优势不只来自单个阶段；其统计稳定性仍需另行评估。

finetune 在 C 的四个阶段均低于 zero-shot。例如 pull through and cinch 从 0.3063 降至 0.2122，wrap around psm 从 0.2515 降至 0.1848。这与逐帧距离中 finetune 优于 zero-shot 的方向一致。

threading string through tower 只有 **3 个视频、830 帧**；pull through and cinch 有 **4 个视频、8,392 帧**，其余 C 阶段各有 5 个视频。各 stage 覆盖量不同，不把阶段间的差值直接解释为固有难度差异，也不对九行简单平均形成额外的总体 JSD 排名。

![任务 A/C 的 stage JSD 五模型对比](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/stage_jsd_comparison.png)

## 6. 综合判断与适用边界

| 模型 | 当前证据支持的判断 | 不宜作出的推断 |
| --- | --- | --- |
| all_ABCD | 总体位置误差与 single_task 接近；A 的 transfer/place 分布数值最低 | 不能因个别 stage 微小领先就判定整体严格优于 single_task |
| single_task | 总体位置与 all_ABCD 接近；多数 stage，尤其 C 的全部阶段，分布更接近 GT | 不能把 6/9 最低值当作经过显著性检验的全局胜出 |
| leave_out | 当前运行的总体位置误差最高；A 的 reach/grab 分布差异较大 | 不代表所有阶段均最差，也不能将差异完全归因于训练任务缺失 |
| spd_finetune | 相比当前 zero-shot，多数视频的位置和多数 stage 的分布均有改善 | 三个 seed 的指标平均不等于概率集成，也不是全模型匹配 seed 的对照 |
| spd_zero_shot | 总体距离低于 leave_out，但高于其余三类；部分视频和稀少 stage 有局部优势 | drop ring 的单视频 30 帧优势不能泛化到该行为 |

本次结果最稳妥的表述是：**all_ABCD 与 single_task 在当前 novice active 测试集上具有相近的逐帧位置复现程度，single_task 的阶段空间分布表现更有优势；finetune 相比 zero-shot 改善了当前数据上的位置与分布匹配；leave_out 的总体位置匹配较弱。**

这些结论限定于本次 10 位用户、15 个视频和已保存模型运行。虽然逐帧样本数较大，相邻帧存在时间相关性，不能将 147,454 帧当作同等数量的独立受试者。GT 在部分 A 视频中发生过版本差异，因此本报告不能与基于旧缓存 target 的结果直接混用。

若后续需要把“相对接近”提升为“达到可接受的复现水平”的判断，需要另外定义可接受误差或加入人类一致性、简单 gaze 基线等参照；这些未纳入本次结果，报告不预设其结果。

## 附录 A：15 个视频的完整距离证据

单位：画面对角线百分比；所有模型在同一视频内使用完全相同的有效帧。finetune 一列为三个 seed 指标的均值。

| 任务 | 视频 | 有效帧数 | all_ABCD | single_task | leave_out | spd_finetune | spd_zero_shot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | User11_6_1 | 11,039 | 12.6973 | 12.7287 | 16.1374 | 12.6156 | 16.6699 |
| A | User12_3_1 | 12,808 | 11.9404 | 11.7594 | 14.0796 | 12.1665 | 14.5365 |
| A | User1_1_1 | 12,418 | 11.3311 | 11.2570 | 15.1749 | 11.6389 | 14.7897 |
| A | User7_1_1 | 18,286 | 8.3563 | 7.9131 | 12.4621 | 8.4428 | 13.2305 |
| A | User9_7_1 | 7,247 | 18.4793 | 18.2840 | 23.6480 | 18.5387 | 22.9662 |
| B | User10_3_1 | 9,220 | 12.6191 | 14.7984 | 16.6123 | 13.7442 | 11.6317 |
| B | User1_9_1 | 5,354 | 6.3089 | 5.9819 | 9.4728 | 6.3618 | 7.4996 |
| B | User8_5_1 | 7,423 | 7.0259 | 7.6273 | 11.0998 | 7.3975 | 8.9525 |
| B | User9_3_1 | 9,667 | 8.6331 | 8.6423 | 11.6645 | 8.9824 | 9.1419 |
| C | User12_5_2 | 11,521 | 6.4477 | 5.9824 | 9.1810 | 6.4148 | 7.0591 |
| C | User1_5_2 | 3,550 | 7.3809 | 7.6847 | 9.5810 | 7.4529 | 11.0843 |
| C | User5_3_2 | 4,720 | 12.5252 | 12.3036 | 12.5220 | 12.6942 | 11.9470 |
| C | User6_1_2 | 13,089 | 8.5332 | 8.4651 | 12.1371 | 8.8004 | 10.5955 |
| C | User7_4_1 | 12,746 | 5.0660 | 4.7445 | 7.3955 | 5.0557 | 6.6152 |
| D | User2_10_1 | 8,366 | 7.1692 | 6.2839 | 7.8954 | 7.0594 | 9.3462 |

来源：[video_distance.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/video_distance.csv)。

![15 个视频的五模型逐帧距离明细](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/distance_by_video.png)

## 附录 B：支持文件与复核记录

| 文件 | 支持内容 |
| --- | --- |
| [model_comparison.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/model_comparison.csv) | 总体距离、像素换算、置信区间 |
| [pairwise_distance.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/pairwise_distance.csv) | 全部 10 个模型配对差值及区间 |
| [video_distance.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/video_distance.csv) | 任务聚合、胜出视频数及完整视频级距离 |
| [video_distance_by_seed.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/video_distance_by_seed.csv) | finetune 各 seed 的视频级指标 |
| [stage_jsd_comparison.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/stage_jsd_comparison.csv) | A/C 九个 stage 的五模型 JSD 对照 |
| [stage_jsd_video.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/stage_jsd_video.csv) | 每个视频、每个 stage 的 JSD 与帧数 |
| [stage_jsd_by_seed.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/stage_jsd_by_seed.csv) | stage JSD 的 seed 级原始汇总 |
| [stage_coverage.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/stage_coverage.csv) | 包含零帧缺失阶段的覆盖情况 |
| [audit.json](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/audit.json)、[audit_clips.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/audit_clips.csv)、[audit_predictions.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/audit_predictions.csv) | 样本量、有效帧、帧号对齐、缓存与当前 GT 差异 |
| [input_manifest.csv](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/input_manifest.csv) | 142 个实际输入文件的路径、大小、修改时间；当前 GT 的 SHA256 |
| [comparison_tables.xlsx](D:/GitHub_Project/co-tracker/outputs/novice_active_gaze_comparison_20261109/comparison_tables.xlsx) | 距离、stage 和视频级阶段明细的 Excel 表 |

本报告直接引用上述已完成的分析结果。任务级均值、胜出次数和相对改善幅度由已保存的视频级／stage 表重新汇总，未改变原始计算口径。

既有结果已进行独立复核：从源概率图重算 `User7_1_1` 的全部共同帧，坐标最大差异约 1.33×10⁻¹⁵，距离最大差异约 7.64×10⁻¹⁴ 个百分点；独立重算其 reach ring 的 JSD 为 0.12301782109030733，与保存结果差异约 3.1×10⁻¹³。五模型总体均值和 finetune 的全部 seed 聚合亦核验一致。
