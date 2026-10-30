# Gaze delay 图片可读性检查与结论图选择

2026-10-30。对象：`outputs/gaze_psm_xcorr` 的修复后完整 84 条记录。此次从保存的结果重画图片，没有重新选择参数或改算统计量。

## 1. 这次修改解决了什么

原图主要存在三类问题：相关曲线与记录峰值散点统一使用 `[-1,1]`，把只有约 0.02～0.05 的速度相关压在零附近；空间接近比例使用 0～100%，而实际只有约 1%～12%；图例、状态文字和个体曲线叠在组估计上。另有个体端点峰值拉大范围，以及 bootstrap 横轴留下大量空白的问题。

现在采用“**需要直接比较的内容共享合理尺度，细节与完整范围同时保留**”的规则。不是让每个子图随意放大，否则相似的视觉高度可能代表不同的相关强度。

| 图类 | 现在的尺度与布局 | 可以直接比较什么 |
|---|---|---|
| 整任务 xcorr | 同一种信号的四个面板共享数据范围，包含完整组 CI、异人参考和零值；位置和速度分开定尺度 | 同一种信号下的 A/C 与左右器械 |
| 阶段 xcorr | 每一行同一阶段的左右器械及两种模式共享 y 轴；不同阶段行可用不同范围 | 同一阶段左右，以及主阶段与同片段模式；跨阶段强度必须读刻度 |
| 空间绝对值 | 四个面板共享范围；接近比例改成百分比；距离仍用 px | A/C、左右的实际距离/接近水平 |
| 空间变化量（新增） | 减去各自零 lag 值；同列共享 y 轴，叠加 L/R | 不被绝对距离基线掩盖的微小改善/恶化 |
| 记录峰值散点 | 同信号共享数据驱动的 x/y 范围；保留端点空心标记 | 记录的分散程度；不把记录当独立参与者 |
| 个人 A/C 配对峰值 | 上排显示全部数据，下排左右都放大到 -0.8～+0.8 s；注明超出细节窗的点数 | 个人变化方向、主体分布，同时保留异常和端点 |
| bootstrap 峰值分布 | 同信号共享横轴、纵轴、一个采样格宽的分箱；纵轴改百分比；所有尾部保留 | 峰值方向、离散程度、多峰和集中程度 |
| 个人曲线（新增） | 单独诊断图，保留全部个人范围并叠加组均值 | 个体差异；组 CI 在整任务主图中看 |

实际新范围：整任务位置相关约 `[-0.105,0.656]`；速度相关约 `[-0.0196,0.0594]`；空间接近比例约 `0～13.73%`。速度图相较原先跨度 2 的纵轴，现跨度约 0.079，约放大 25 倍；**相关系数本身完全没有变大**。

主图中个体曲线已移到 `novice_curves_*.png`。图例集中放在图外，样本数和峰值文字放在面板上方。阶段图补充同片段模式的逐点 CI；固定记录敏感性使用紫色线，与橙色同片段主曲线区分。数据不足的面板明确显示无组估计，并保持完整 ±2 s 横轴。

## 2. 哪些图适合用来写结论

### 第一选择：`conclusion_position.png`

[打开 PNG](../outputs/gaze_psm_xcorr/figures/conclusion_position.png) · [打开矢量 PDF](../outputs/gaze_psm_xcorr/figures/conclusion_position.pdf)

**最适合论文、报告或汇报中的总体结论页。** 一张图同时呈现：

- A：四个整任务位置峰值及参与者 bootstrap 95% 区间。
- B：11 人配对的 C−A 峰值差及区间；这个统计量不同于个人峰值差的平均。
- C：零 lag 与组峰值 lag 下的 50 px 空间接近比例。

可以支持的文字：

> 视线与器械质心轨迹的时间对应具有任务和器械侧别差异。Task A 右器械的位置轨迹峰值约为 gaze 领先 341 ms，Task C 右器械约为 gaze 滞后 171 ms，配对任务差约 -512 ms。左器械的两个任务区间均跨零。空间接近比例较低，且移至相关峰值后的改善有限。

C 面板的零 lag 与峰 lag 两种标记接近重叠，反映真实空间改善很小；不是遗漏了一组点。它们没有空间指标的置信区间，只能作描述性比较。

这张图不支持“每个人都提前同样时间”“fixation 先到达同一点”或“因果引导”的结论。

### 第二选择：`whole_task_position.png`

[打开 PNG](../outputs/gaze_psm_xcorr/figures/whole_task_position.png) · [打开矢量 PDF](../outputs/gaze_psm_xcorr/figures/whole_task_position.pdf)

**适合支撑总体结果的完整曲线证据，与结论汇总图搭配。** 可以展示真实配对与异人参考的差别、位置曲线峰的宽度，以及零 lag 附近的整体形状。

蓝色阴影是固定 lag 下 r 的逐点 CI，不是峰值 lag 的 CI。领先/滞后的方向判断应读上一张图的横向峰值区间。也不要只根据曲线最高点的横坐标断言方向已经确定。

如果正文只能放两张图，建议优先使用**这张和 `conclusion_position.png`**。

### 阶段结论选择：`stage_peak_lags_position.png`

[打开 PNG](../outputs/gaze_psm_xcorr/figures/stage_peak_lags_position.png) · [打开矢量 PDF](../outputs/gaze_psm_xcorr/figures/stage_peak_lags_position.pdf)

**适合汇总阶段依赖及分析模式敏感性，应明确标注为探索性。** 蓝色是 PSM 时间锚定的主阶段分析，橙色是仅同一连续片段内部配对；每行给出峰值区间和各模式人数，三角标记为搜索端点峰值。

可以支持：A transfer 的正 lag 在主阶段分析中明显，在同片段模式下靠近零；C reach 右侧的主分析为正，但同片段区间跨零；C pull/cinch 和部分 threading 条件存在端点或很宽的区间。

它不支持一个统一的阶段提前时间。阶段比较未经多重比较校正，不宜把所有区间不跨零的条目都写成确认性发现。

### 空间结论选择：`spatial_proximity.png`

[打开 PNG](../outputs/gaze_psm_xcorr/figures/spatial_proximity.png) · [打开矢量 PDF](../outputs/gaze_psm_xcorr/figures/spatial_proximity.pdf)

**适合作为结论限制或空间对应证据。** 四面板共用约 0～13.73% 的范围，清楚显示接近比例水平有限。它支持“轨迹相关并不等同于空间重合”，不能单凭低比例认定坐标注册一定错误。

结论汇总图 C 已展示核心空间信息；篇幅紧张时，这张可以放补充材料。50 px 是探索性阈值，图注应保留这一点。

## 3. 哪些更适合补充材料或诊断

| 图片 | 建议用途 | 不宜独立承担的结论 |
|---|---|---|
| `bootstrap_peak_lags_position.png` | 高优先级补充；展示 A/C 左侧的宽分布/多峰，以及右侧的方向 | 单个柱子的高度不是领先显著性 |
| `stages_A_position.png`、`stages_C_position.png` | 阶段完整曲线、同片段与固定记录敏感性 | 跨行看曲线高度时不能忽略不同 y 刻度 |
| `spatial_distance.png` | 补充各条件绝对距离 | 质心距离不等于器械尖端或目标距离 |
| `spatial_changes.png` | 查看峰值移位相对零 lag 的距离/比例变化 | 没有变化量 CI，不能单独声称显著改善 |
| `whole_task_velocity.png` | 辅助结果；说明速度关联较弱，与位置结果并不相同 | 不宜用来概括主要 gaze 提前时间 |
| `stages_A_velocity.png`、`stages_C_velocity.png` | 辅助/探索性阶段速度模式 | 不宜只摘某个局部尖峰作为主要结论 |
| `bootstrap_peak_lags_velocity.png` | 检查弱速度关联下的峰值不确定性 | 峰值集中不代表相关强度大 |
| `record_peaks_position.png`、`record_peaks_velocity.png` | 记录质量、离群点和端点诊断 | 不能把所有记录点当作独立的人来推断 |
| `paired_novices_position.png`、`paired_novices_velocity.png` | 看个人 A/C 变化与异常点 | 不等同于组曲线峰值的配对检验 |
| `novice_curves_position.png`、`novice_curves_velocity.png` | 个体差异与主曲线的关系 | 个人曲线集合不直接代替组 CI |

更清楚的速度图仍显示 r 只有约 0.02～0.05。放大坐标使峰形可见，并没有增加效应量或证据强度。

另外检查了 `interpretation/lag_summary.png`：它的修复后数值可以保留，但没有同时呈现同片段敏感性和配对任务差。正式选图优先用新增的两张峰值区间汇总图；这张旧摘要可作简短辅助，历史目录中的旧图保留为历史对照。

## 4. 建议的正文选图组合

**总体问题为主：** `conclusion_position.png` + `whole_task_position.png`；补充 `bootstrap_peak_lags_position.png`。

**阶段差异为主：** 在上述组合中增加 `stage_peak_lags_position.png`，用 `stages_A_position.png`、`stages_C_position.png` 放补充说明敏感性。

**强调是否先看同一目标：** 必须同时展示空间证据。可使用汇总图 C，或额外放 `spatial_proximity.png`，并说明当前输入是 raw gaze、器械位置是整体 mask 质心。现有分析还不足以回答事件级 fixation 首次到达时间。

## 5. 保存、复现与验证

- 新图在 [`outputs/gaze_psm_xcorr/figures/`](../outputs/gaze_psm_xcorr/figures/)：共 **19 张 PNG 与 19 份矢量 PDF**。
- 原来的 14 张 PNG 保存在 [`figures_before_readability_20261030/`](../outputs/gaze_psm_xcorr/figures_before_readability_20261030/)。重复重画不会向原版备份中混入新图。
- 各图坐标范围与比较分组记录在 [`figures/axes.json`](../outputs/gaze_psm_xcorr/figures/axes.json)。
- 绘图前后文件哈希及代码来源记录在 [`figure_readability_20261030.json`](../outputs/gaze_psm_xcorr/figure_readability_20261030.json)；原始数值文件、NPZ 缓存与原分析代码快照保留。
- 44 项测试通过，其中 5 项专门验证比较尺度一致、CI 未被截断、缺失阶段不变成假零结果、bootstrap 尾部保留和百分比/变化量尺度。
- 绘图代码调整后再次从缓存复算记录、个人/组、bootstrap、132 项比较及全部 CSV，见 [`audit_after_figure_refresh_20261030.json`](../outputs/gaze_psm_xcorr/audit_after_figure_refresh_20261030.json)。这次不再重放异人配对；100 次异人完整重放的原审计证据继续保留。

从仓库根目录仅重画，不需要 NAS 原始数据或重新进行 xcorr/bootstrap：

```powershell
python tools/replot_gaze_psm_outputs.py --output outputs/gaze_psm_xcorr
```

实现位于 [`tools/gaze_psm_figures.py`](../tools/gaze_psm_figures.py)；[`gaze_psm_report.py`](../tools/gaze_psm_report.py) 的数值导出函数不变，绘图部分委托给该模块。历史 `code_snapshot` 保留的是原始分析时的代码，不覆盖成新的绘图代码。

相关说明：[计算与字段详解](gaze_psm_code_and_formulas_zh.md)；[数值结果复核报告](gaze_psm_results_audit_zh.md)。
