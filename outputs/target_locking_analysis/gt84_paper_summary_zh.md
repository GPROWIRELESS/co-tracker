# GT84：目标锁定能否代表高质量注视？

**总论：目标锁定描述的是任务和阶段相关的注视分配，目前不能作为“越高越好”的通用技能或专家注视指标。Sea Spikes呈明确目标偏好；Knot-tying的reach string仅表现出相对面积基线的目标富集。练习中成绩提高可与目标偏好下降并存。**

**与大纲一致的分析范围。** 依据[Paper Outline，第3、5–6页](<C:/Users/GT502/Desktop/JMRR2027/Paper Outline.pptx>)，Task A纳入reach/grab/transfer/place ring四阶段；Task C主文只分析reach string。大纲明确要求剔除低质量cinch与threading标签，并提出wrap可能排除；本版将wrap留在补充范围。因此本文的任务标签“C”均指reach string（与偏好指标C区分），不能与旧长报告的Task C全阶段均值混用。以下仅使用v2结果。

GT84包含84段录像、11名参与者：A 33段，C 51段；不是84名独立被试。C reach string的联合有效帧覆盖为83.6%，其占用可用51段，成绩相关可用50段。

**统计口径。** target为有效标注槽位均值周围50 px圆盘；PSM为左右DLC腕部50 px邻域并集。定义T/P/O为目标独占、腕部独占、重叠：目标占用L_target=L_T+L_O，腕部占用L_gripper=L_P+L_O；C=L_T/(L_T+L_P)，分母为0时不可估。先按原生时间计算录像指标，再在参与者内、参与者间等权平均；方括号为10,000次参与者bootstrap的95% CI，pp为百分点。区域代表空间邻域，不能称夹爪尖端或完整物体mask。

本报告另使用C_U（均匀画面落点下的独占目标面积比例）、C_excess=C−C_U、B=L_target−L_gripper，以及参与者指标—SimNow任务总分的Spearman相关ρ。面积修正控制空间机会，不是技能校正。

## 结论一：目标偏好依赖任务与动作，不能用同一阈值定义高质量注视

| 指标 | A四阶段整体，n=11 | C reach string，n=11 |
|---|---|---|
| 目标占用 % | 24.98 [21.23, 28.60] | 23.38 [19.46, 27.34] |
| 腕部占用 % | 9.46 [6.34, 13.15] | 19.31 [15.60, 22.92] |
| 目标偏好C % | 76.79 [70.29, 82.87] | 54.86 [46.99, 63.31] |
| 目标−腕部占用，pp | 15.52 [11.21, 19.68] | 4.08 [-0.10, 8.92] |
| 超出面积基线的偏好，pp | 43.43 [37.24, 49.27] | 22.52 [14.81, 30.85] |

A的目标−腕部占用差为+15.52 pp，CI不跨0，支持明确的目标邻域偏好。A各阶段target占用从reach的12.42%到place的42.98%，表明阶段差异不能被整体均值替代；这里未进行阶段间配对显著性检验。

C reach string的C=54.86%，CI跨50%；原始target−腕部差为+4.08 pp，CI跨0，**不能确认其原始占用更偏目标**。但几何机会基线C_U仅32.34%，C_excess为+22.52 pp [14.81,30.85]，因此**相对均匀空间机会，仍存在明确目标富集**。这两个结论回答不同问题，并不矛盾。

![图1：论文纳入阶段的占用、目标偏好与面积参照](paper_summary_figures/G1_task_and_phase_dependence.png)

**图1图注。** 左图为目标与腕部占用；右图为C及面积基线，50%虚线表示目标/腕部独占占用相等。点与误差线为11名参与者均值及95% CI；A whole仅作四阶段汇总，C仅保留reach string。A各阶段的空间分配不同；C接近50%不代表没有超过面积机会水平的目标富集。两任务未进行任务间配对差检验。[矢量PDF](paper_summary_figures/G1_task_and_phase_dependence.pdf)

## 结论二：目标锁定尚未被验证为表现越好数值越高的指标

A整体C—成绩ρ=0.036 [−0.702,0.785]，target占用—成绩ρ=0.082 [−0.528,0.735]；C reach string分别为0.009 [−0.687,0.715]和0.358 [−0.334,0.848]。**两项主分析都没有确定方向的目标锁定—表现关系，不能据此把高C或高target占用定义为专家式注视。** 区间宽也意味着不能证明二者绝对无关。

相反，C reach string腕部占用与成绩呈探索性正相关：ρ=0.624 [0.100,0.906]，逐人留出范围0.553–0.736。这提示有效策略可能包含对器械邻域的信息采集，而非单纯减少腕部注视；它仍是11人、多指标探索中的关联，不能解释为因果机制或专家策略。

## 结论三：练习中的成绩提升不要求目标锁定同步增强

| 第3轮−第1轮，同人配对 | A四阶段整体 | C reach string |
|---|---|---|
| 目标偏好C变化，pp | 1.79 [-7.65, 10.04] | -24.38 [-38.38, -9.71] |
| 目标占用变化，pp | -1.29 [-6.73, 3.79] | -11.36 [-17.36, -4.31] |
| 腕部占用变化，pp | -1.52 [-5.04, 2.64] | 8.32 [0.31, 15.85] |
| SimNow任务总分变化，分 | 38.45 [24.82, 51.27] | 22.18 [8.14, 39.96] |

A成绩提高38.45分，目标偏好C变化的CI跨0；C任务成绩提高22.18分，同时reach string的C下降24.38 pp、target下降11.36 pp、腕部上升8.32 pp，后三项CI均不跨0。**因此，表现提升与目标锁定增强并非同义；将C最大化作为高质量注视模型的目标缺乏当前数据支持。** 这一结论在剔除其他C阶段后仍成立，不依赖低质量cinch/threading标签。

![图2：表现相关与练习轮次的变化](paper_summary_figures/G2_quality_proxy_and_practice.png)

**图2图注。** 左：A整体、C reach string的指标—任务成绩ρ及95% CI；中：同人第3−第1轮C变化；右：对应任务总分变化，右图不是阶段评分。各轮次差为11人配对。A前后均11段；C注视指标18→15段、成绩18→14段，录像支持不完全相同。图示的是表现与注视分配变化并存，不能推出“减少目标注视导致成绩提高”。[矢量PDF](paper_summary_figures/G2_quality_proxy_and_practice.pdf)

## 对Research Questions的明确回答

| 大纲问题 | GT84可以回答什么 |
|---|---|
| RQ1：适合动态、多模态注视的模型形式？ | **提供阶段相关的行为评价依据，不能选择架构。** 阶段占用不同不等于已证明时序输入必要；本目录没有memory消融或gaze领先PSM的证据。 |
| RQ2：如何获得higher-quality gaze，需不需要专家数据？ | **先否定未经验证的替代标准：不能以高target-locking直接定义高质量。** 当前数据说明其方向随任务、练习变化；没有专家参照或受控数据来源实验，仍不能判断大量新手能否替代专家监督。 |
| RQ3：跨任务是否提高低数据泛化/效率？ | **GT84本身不回答训练策略优劣。** 它提供行为参照；需在同一测试支持上比较模型，相关结论见模型精简版。 |

### 可用于论文的英文结果与结论段落

Observed target-directed gaze depended on task and phase. In Sea Spikes, target preference among exclusive regions was 76.79% (95% CI 70.29–82.87%). In the reach-string phase of Knot-tying, preference was 54.86% (46.99–63.31%), but remained above the geometric-area baseline by 22.52 percentage points (14.81–30.85). Neither analysis provided a clearly directed association between target preference and task performance. Across practice rounds, Knot-tying task scores increased by 22.18 points (8.14–39.96), while reach-string preference decreased by 24.38 percentage points (9.71–38.38). Thus, target locking should be treated as a context-dependent description of gaze allocation, rather than an established monotonic indicator of skill or expert-like gaze. Expert-referenced comparisons remain necessary to evaluate whether novice-trained models approximate higher-quality gaze.

**证据与限制。** 11人、小样本CI、多重探索未校正；50 px为预设容差而非经独立校准验证的准确度。来源：[占用](20261105T063658759189Z_v2_full/main_probability/group_stage_summary.csv)、[成绩相关](20261105T063658759189Z_v2_full/main_probability/performance_correlations.csv)、[轮次配对差](20261105T063658759189Z_v2_full/main_probability/repetition_contrasts.csv)。完整指标/敏感性见[详细版](gt84_analysis_zh.md)，模型回答见[模型精简版](models10_vs_gt_paper_summary_zh.md)。
