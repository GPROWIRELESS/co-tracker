# 模型与GT：面向论文的精简结论

**总论：现有模型尚未充分重现真实目标锁定。保留概率分布能降低区域事件预测损失，但不能据此证明时序模型有效；当前跨任务训练未显示优于单任务训练的明确优势，任务域微调则改善了目标事件预测。**

**与大纲一致的分析范围。** 依据[Paper Outline，第3、5–6页](<C:/Users/GT502/Desktop/JMRR2027/Paper Outline.pptx>)，Task A纳入reach/grab/transfer/place ring四阶段；Task C主文只分析reach string。大纲明确要求剔除低质量cinch与threading标签，并提出wrap可能排除；本版将wrap留在补充范围。因此本文的任务标签“C”均指reach string（与偏好指标C区分），不能与旧长报告的Task C全阶段均值混用。以下仅使用v2结果。

样本为10段测试录像、五种模型共50份预测；每任务5名参与者，跨任务共7人。GT与模型使用相同共同有效帧。GT84全量均值不作为本报告的模型对照。

**统计口径。** target为有效标注槽位均值周围50 px圆盘；PSM为左右DLC腕部50 px邻域并集。定义T/P/O为目标独占、腕部独占、重叠：目标占用L_target=L_T+L_O，腕部占用L_gripper=L_P+L_O；C=L_T/(L_T+L_P)，分母为0时不可估。先按原生时间计算录像指标，再在参与者内、参与者间等权平均；方括号为10,000次参与者bootstrap的95% CI，pp为百分点。区域代表空间邻域，不能称夹爪尖端或完整物体mask。

模型概率q来自每帧64个保存样本的区域命中比例。Brier=时间加权平均(q−y)²，y为真实gaze命中事件，越低越好；它评价区域事件，不能单独证明整张注视分布或多模态结构正确。

## 结论一：模型存在一致的目标概率不足，不能作为已验证的真实或专家注视替代

真实目标占用为A的22.68%与C的24.82%，五模型仅为4.94%–6.82%与5.01%–8.84%。**全部10个“模型×分析范围”的目标占用差CI均低于0**（图1）。例如single_task的模型−GT差值为A -15.86 [-20.17, -11.14] pp、C -15.98 [-33.59, -2.01] pp。这是方向一致的空间分配偏差，而非仅有某个模型排序稍差。

| 来源 | A目标占用 % | C目标占用 % | A目标Brier | C目标Brier |
|---|---:|---:|---:|---:|
| 配对GT | 22.68 | 24.82 | — | — |
| all_ABCD | 6.44 | 8.53 | 0.1927 | 0.1843 |
| single_task | 6.82 | 8.84 | 0.1920 | 0.1825 |
| leave_out | 4.94 | 5.15 | 0.1997 | 0.2046 |
| SPD fine-tune | 6.52 | 8.15 | 0.1933 | 0.1885 |
| SPD zero-shot | 5.18 | 5.01 | 0.2032 | 0.2098 |

![图1：模型目标占用的配对偏差及95%CI](paper_summary_figures/M1_target_underestimation.png)

**图1图注。** 五模型的目标概率占用减去相同帧GT占用；点为参与者均值，横线为95% CI，虚线0表示无偏差。A纳入四阶段，C仅reach string，各n=5。所有区间位于0左侧，表明模型一致低估目标区域概率质量；C区间较宽，提示个体间不确定性更大。[矢量PDF](paper_summary_figures/M1_target_underestimation.pdf)

概率预测也未稳定超过强基线：A五模型相对训练事件频率的目标BSS均为负，且CI全低于0；C五模型的该BSS区间均跨0。BSS=1−模型平均Brier/参照平均Brier，正值才表示优于参照。**优于固定中心或均匀面积基线，不等于已经捕捉到超出训练事件频率的目标锁定规律。**

## 结论二：概率输出适合评价区域事件，时序必要性仍未被检验

A与C各五模型、目标/腕部两事件，共20个条件；保留区域概率相较期望点、argmax、独立随机点，各有20/20个Brier点估计更低。例如single_task目标Brier：A为0.1920，三种点解码为0.2432/0.2806/0.2519；C为0.1825，对应0.2401/0.2496/0.2534。

![图2：概率输出与三种点解码的目标Brier](paper_summary_figures/M2_probability_and_decoding.png)

**图2图注。** 相同测试支持下的目标Brier，蓝色为区域概率，其他曲线为同模型的点解码，绿色虚线为训练事件频率参照。横轴五种模型为离散类别，连线仅辅助阅读。概率分支点估计均低于点解码，支持保留分布进行事件评价；图中不展示分支差CI，不能宣称全部差异显著。[矢量PDF](paper_summary_figures/M2_probability_and_decoding.pdf)

这是**输出表示**的对照，不是Transformer对其他架构、视频对单帧或有无memory的消融。Brier本身是概率评分，硬化预测会丢失不确定性；保存样本与概率图也有有限采样差异。因此它支持RQ1的“概率输出有用”，不能证明“时序信息必需”或“模型已学会正确多峰注视”。

## 结论三：没有明确跨任务优势；目标任务数据与域适配仍有价值

下表为目标Brier的配对差，负值有利于左模型。先匹配录像再按参与者bootstrap，不能用两组独立CI是否重叠替代该比较。

| 对照（左−右） | A差值 [95% CI] | C差值 [95% CI] |
|---|---|---|
| single_task − all_ABCD | -0.0008 [-0.0015, 0.0002] | -0.0018 [-0.0071, 0.0038] |
| single_task − leave_out | -0.0078 [-0.0117, -0.0038] | -0.0221 [-0.0451, -0.0002] |
| SPD fine-tune − zero-shot | -0.0099 [-0.0151, -0.0051] | -0.0213 [-0.0363, -0.0071] |

single_task的目标Brier点估计最低，但与all_ABCD的差在两项主分析中均跨0，故**没有观察到跨任务优于单任务，也不能反过来宣布单任务确定更优**。相较leave_out，single_task的目标Brier更低，两个差值CI均为负，说明去掉目标任务训练数据伴随更差的目标事件预测。

SPD微调相对零样本的目标Brier在两项主分析中均下降；腕部差值CI则均跨0，收益具有事件特异性。现有输出没有等训练量学习曲线、完整多seed对照及充分checkpoint溯源，不能将差异完全归因于数据效率或微调本身，也不能把SPD名称直接等同于被动观看数据。

## 对Research Questions的明确回答

| 大纲问题 | 本分析给出的回答 |
|---|---|
| RQ1：什么模型形式适合动态、多模态注视？ | **部分支持概率输出；时序必要性未检验。** 需补单帧/视频、无memory/有memory消融，才能回答架构问题。 |
| RQ2：需要专家数据，还是大量新手/被动数据即可？ | **当前不能判定。** 五来源没有形成专家、better-novice、普通新手、passive的受控比较；现有模型仍有明显目标概率不足，不能宣称新手聚合已近似专家。SPD结果仅支持本次域适配对目标事件的局部收益。 |
| RQ3：低数据条件下跨任务更好、更省数据吗？ | **当前未观察到跨任务优势；数据效率未被直接检验。** 目标任务数据相关的收益明确，但还需要等样本量、多数据预算实验与独立参与者测试。 |

### 可用于论文的英文结果与结论段落

Across held-out recordings, all five models underestimated target-region occupancy in Sea Spikes and in the reach-string phase of Knot-tying. The paired 95% confidence intervals were below zero for every model in both analysis scopes. Region-probability predictions achieved lower Brier point estimates than expectation, argmax, or independently sampled point decodings. This supports retaining predictive uncertainty for region-event evaluation, but does not establish the necessity of temporal context. Single-task and all-task training did not differ clearly in target Brier score, whereas single-task training outperformed leave-out training. SPD fine-tuning also reduced target Brier score relative to zero-shot transfer. These findings indicate incomplete recovery of target-directed gaze and no demonstrated cross-task advantage under the present evaluation; they do not establish expert-like gaze or data efficiency.

**证据与限制。** n=5/任务，留出录像不等于留出参与者；CI未校正多重比较。来源：[占用](20261105T063658759189Z_v2_full/main_probability/group_stage_summary.csv)、[事件误差](20261105T063658759189Z_v2_full/main_probability/model_error_summary.csv)、[模型配对差](20261105T063658759189Z_v2_full/main_probability/model_paired_contrasts.csv)。图2另用三个解码分支同名误差表。完整定义与全部旧图保留在[详细版](models10_vs_gt_analysis_zh.md)；GT对高质量指标的检验见[GT84精简版](gt84_paper_summary_zh.md)。
