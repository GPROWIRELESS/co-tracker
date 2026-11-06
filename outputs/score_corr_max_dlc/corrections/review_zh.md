# 附件修正意见逐条复核与本次修订

复核日期：2026-11-05。对象：`tools/`、`outputs/score_corr_max_dlc/`，另只读核查 `Index/Task_A_Score.csv`、`Task_C_Score.csv`。结论：附件大部分审计方向合理，但不能照单全收；其中对 critical error 的归因、完整数据条数、检验总数、以及“噪声”“全局无效应”的推论需要改正。

本次保留原始成绩、匹配键、选侧结果和原推断值，修正报告与图示，补充可追溯诊断和单独的探索性分析。没有根据猜测重配成绩或删除零分。

## 对附件“一、正确的部分”的核对

1. **匹配与审计：基本同意。** 成绩按 UserID、TaskNum、Attempt 匹配，并核对索引信息；83/84 条Score可用，NotAttempted没有填零。但键一致只证明文件之间自洽，不能证明源成绩录入正确。哈希不变只证明运行期间未改文件，不能证明输入是真实正确的。
2. **推断实现：数值复核通过，但不能因此笼统评为统计设计完全正确。** 原五种结局759个成功检验，罚分两种结局另240个，合计999个。本次按原场景用SciPy及全哑变量加权回归复核相关系数、参与者聚合、CR1标准误/p和BH，全部通过。它不等于重新验证每个bootstrap区间的覆盖率、置换的科学交换性假设或评分效度。
3. **选侧元数据：同意一致性检查。** 原脚本核对获胜侧峰值与选侧表相同；但一致性不消除max选择带来的估计目标变化。原图CSV校验也不能代替对所有图的逐幅视觉检查；本次重新生成所有overview并检查关键零变异图。
4. **原有限定措辞：同意且保留。** selected PSM不等于独立确定的操作器械，Score属于整次尝试，全部结果仍为探索性。

## 对附件“二、需要修正的部分”的逐条判断

### 1. Score地板效应：问题成立；critical归因和建议替换分析不成立

- C有效Score有50条，35条为零，包含16条Failed和19条Completed。completed_only的reach string仍有19/34零分，确实没有消除地板效应。
- **19条Completed零分中，只有15条PenaltySum≤−100，不是19条。** 另外3条大于−100，1条分数组成项缺失。因此“零分基本就是有没有critical error”缺乏依据。
- 总罚分≤−100也不等于发生某个严重事件：13条明确有Detached Towers的−100项；User6_1_2的−100来自50次器械碰撞累计；User8_6_1缺少该分项。事件名称本身的严重性还需评分规范确认。
- `max(0,min(100,Time+Eco)+PenaltySum)`只在**66条组成项完整的记录**上可检验，误差−0.7至+0.8。不能说验证了84条。
- C的**position/psm_stage/threading**为15条录制、7人且全零，确应显示“no score variation”；velocity该阶段有18条、8人、3种Score，不能一概称无变异。
- **处理：**修正图示，保留completed_only和原主分析；另加“已知PenaltySum>−100”敏感性。这个子集不是“无critical error”，且会排除失败或罚分缺失记录，改变目标群体。二元事件和非零条件分析可以另做，但不能把删除零分当成对原问题的无偏修复。

### 2. Time/Eco和Raw镜像：基本同意，但应保留审计记录

任务内近似变换属实：A为Time=60−TimeRaw/6、Eco=80−EcoRaw/10；C为Time=102−TimeRaw/5、Eco=75−EcoRaw/4。最大残差0.04–0.05，是舍入精度内近似，不能叫严格精确。

Time/TimeRaw各120个成功结果最大反号偏差0.01457，Eco/EcoRaw为0.00839。它们不构成独立重复证据。报告已明确镜像性质；保留已有表以便追溯。Score是复合结局，不能进一步把Score与Time严格说成同一个统计假设。

### 3. C时长异常和重复字段：应核查；尚不能认定复制错误或更改成绩

A的33条完整数据ρ(TimeRaw,duration)=0.99791，比值0.949–1.006；C的33条ρ=0.57525，比值0.653–5.154。以偏离1超过20%为**描述性审计阈值**，A标记0/33、C标记22/33；其余缺失记为unavailable，不当作通过。

User6_1_2与User8_6_1确实六项汇总成绩完全重复，但**原始整行不同**，后者有缺失的碰撞分项。TimeRaw与其中一个录制时长更接近，也不足以确定谁填错。需要源系统导出、计时定义和尝试日志才能解决。

**处理：**在matched_scores和Score对照表追加时长比、20%标志、重复字段对应ID、罚分代理与具体事件字段标志；原始字段值保持一致。没有用录制时长覆盖TimeRaw，也没有据此排除或重新匹配。

### 4. max向上选择：同意；“velocity接近噪声”尚无检验支持

附件中位数0.394/0.230/0.330、平均max−min=0.192以及7.9%，对应的是**position/psm_stage的291条**，不能不加限定地代表全部1164条。

A transfer ring velocity的逐录制peak_r为0.0293–0.1555，参与者Fisher均值0.0462–0.1092，确实较弱。但“弱”“不显著”“噪声”不是同义词；还需要保留时序自相关并覆盖lag/side搜索的零模型。按峰值选侧没有使用Score，也不能直接推出Score关联是伪相关。已在报告中加入限制。

### 5. lag混合和换侧：解释风险成立；“主要是噪声”过强

position/psm_stage左右lag差中位0.915秒、Q75=1.841秒。A transfer ring velocity的11人中10人曾换侧，属实。因而它估计的是“各次尝试选中器械的lag”，不能直接解释为固定器械时序。

固定器械需要独立任务信息；按左右lag接近筛选也改变样本与估计目标。若只留该velocity条件中不换侧且有重复观测者，仅剩**1人**，不能完成最低6人的推断。本次披露这一限制，不制造不可估计的敏感性结果。

### 6. order调整：应重点补充；不稳健不等于已证明练习混杂

Score / A / transfer ring / velocity / peak_r：

| 方法 | 关联 | p | 原family q |
|---|---:|---:|---:|
| 参与者间 | 0.636 | 0.0389 | 0.5058 |
| 参与者内 | 0.5789 | 0.0001611 | 0.003383 |
| 参与者内，调整order与attempt | 0.3884 | 0.07388 | 0.5172 |

position的对应参与者内关联仅0.0881。Task A全部33条Attempt=1，所以Attempt控制没有新增信息，但TaskNum控制仍有效。

附件的0.81/0.47不是普通Spearman：它们是**参与者内中心化的TaskNum与中心化全样本秩Score/peak之间的相关**；普通合并Spearman是0.5398/0.2976。报告改为准确表述“调整顺序后证据减弱，不能稳定支持独立于order的关联”，不宣称练习已被证明是原因。

### 7. 多重比较：披露建议合理；计数和全局无效应结论要纠正

合并罚分分析后有1377行，其中**999个有效检验、51个BH family**，其余为排除或不可估计行。3条q<0.05均来自同一阶段/velocity/参与者内条件，结局为Score、Time、TimeRaw。Time与TimeRaw是镜像；可称一个条件下的相关证据，不能称三次独立验证，也不能严格折算成一个假设。

不能从发现少推出“全局无效应”；不能在看过结果后追认一个family为预先主假设。BH控制的是预期假发现比例，不能简单用0.05×检验数估算BH结果的假阳性数量。

**新增透明性分析：**将999个有限p一起做探索性BH，3条原显著行的整体q均为0.15864；去除Raw镜像后759个检验，Score为0.12227、Time为0.15576。原family q不变，新增列放在独立附表；这同样不能证明无效应，也不替代一项事先指定的检验方案。

### 8. 小簇CR1、bootstrap与置换：风险合理；直接置换和“退化”结论不充分

10–11个簇的CR1近似应谨慎，但不能单凭簇数认定必然偏自由。参与者内保留原始秩的bootstrap是特定近似，也应明示。

有顺序趋势时，每人内直接任意置换Score未必满足交换性；不能把它作为无条件更正确的p值替换。需要先确定零假设、干扰变量处理和允许的置换区组。本次没有添加未经论证的置换p。

C wrap/Time/position/参与者间CI为[0.20755,1]，**10000/10000个bootstrap均有定义**。边界值得提示，但不足以叫整个bootstrap“退化”或判定区间无效。报告已说明边界与小样本限制。

### 9. 最后循环变量和罚分：代码修复同意；事件解释需纠正

main末尾原本使用最后一次EcoRaw循环的config及recordings，当前数值碰巧一致。现已显式保存并使用Score那一轮的数据，防止将来扩展产生错误。

罚分解释沿用第1点：15个大额总罚分不等于15次critical事件。新增审计同时区分总分阈值、特定事件字段及unknown，不把缺失值当成“无事件”。

## 本次新增分析的结果与范围

“已知PenaltySum>−100”子集在阶段筛选前保留A33条、C18条；C reach string为18条、9人。使用原方法、10000次bootstrap、19999次参与者间置换、原最低人数标准，另成family。唯一family q<0.05的条目仍是A transfer ring velocity未经order调整的Score结果；没有新的C显著发现。该结果不解决C计时来源不明的问题。

修订文件入口：

- [逐录制诊断](matched_scores_audited.csv)：比值、阈值、重复值及罚分标志。
- [不可估计原因](availability_diagnostics.csv)：避免把零变异误标为low N。
- [关键方法比较](key_order_comparison.csv)、[原显著行](significant_results.csv)。
- [整体BH敏感性](global_bh_sensitivity.csv)：保留原q，并追加整体范围q。
- [已知罚分子集分析](penalty_subset_associations.csv)及[参与者汇总](penalty_subset_participants.csv)。
- [数值验证](validation.json)、[修订前哈希](original_file_sha256.json)。

原历史validation/provenance保留其历史含义；当前修订的哈希与文件覆盖清单单独记录。需要原系统证据才能完成的两项事实核查是：Task C的TimeRaw计时定义，以及两条重复六元组的来源。当前资料不足以替它们填入“正确成绩”。

## 方法依据

- [Benjamini与Hochberg原文](https://www.stat.cmu.edu/~ryantibs/journalclub/benjamini_1995.pdf)：FDR是预期假发现比例；不能用发现数量证明全局零假设。
- [Winkler等，Permutation inference for the general linear model](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)：含干扰变量和重复测量时，必须考虑交换性与区组。
- [Pustejovsky与Tipton，小样本聚类稳健推断](https://jepusto.com/files/Pustejovsky-Tipton-201601.pdf)：小簇近似的误差需要针对设计评估，不能仅凭实现自检排除。
