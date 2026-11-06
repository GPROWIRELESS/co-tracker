# Target Locking Analysis：新版详细结果

本报告将概率占用、点解码和时间行为分开，所有统计数值来自同一运行目录。NA保留为不可估，不填0；CI是近似的参与者bootstrap区间。

## 当前数据的主要发现

**Task A（Sea Spikes）**：真实全量 target占用（%）为 25.0 [21.2, 28.6]；n=11，腕部占用（%）为 9.5 [6.3, 13.2]；n=11，C（%）为 76.8 [70.3, 82.9]；n=11。占用的分母是实际有效观察时间，C的分母是两类exclusive占用之和；模型误差另用配对测试集合。

该任务五模型中的可估 target占用差为 -17.7 至 -15.9 个百分点；5 个模型的差值CI全为负，0 个全为正，0 个跨0或区间不可估。

target Brier 的最低点估计来自 single_task：0.192 [0.157, 0.222]；n=5。这是点估计排序；是否支持优于另一模型需阅读第9节的配对差CI。

真实全量 C与SimNow成绩的ρ=0.036 [-0.702, 0.785]；n=11；区间包含0。

真实全量 target占用与SimNow成绩的ρ=0.082 [-0.528, 0.735]；n=11；区间包含0。

真实全量 腕部占用与SimNow成绩的ρ=-0.109 [-0.748, 0.533]；n=11；区间包含0。

**Task C（Knot-tying）**：真实全量 target占用（%）为 24.7 [21.2, 27.8]；n=11，腕部占用（%）为 23.4 [19.9, 26.7]；n=11，C（%）为 51.4 [44.2, 59.3]；n=11。占用的分母是实际有效观察时间，C的分母是两类exclusive占用之和；模型误差另用配对测试集合。

该任务五模型中的可估 target占用差为 -19.7 至 -13.9 个百分点；5 个模型的差值CI全为负，0 个全为正，0 个跨0或区间不可估。

target Brier 的最低点估计来自 single_task：0.188 [0.102, 0.285]；n=5。这是点估计排序；是否支持优于另一模型需阅读第9节的配对差CI。

真实全量 C与SimNow成绩的ρ=-0.174 [-0.706, 0.575]；n=11；区间包含0。

真实全量 target占用与SimNow成绩的ρ=0.156 [-0.561, 0.833]；n=11；区间包含0。

真实全量 腕部占用与SimNow成绩的ρ=0.220 [-0.518, 0.773]；n=11；区间包含0。

以下表格保留相同任务的阶段、概率/点解码、覆盖和敏感性差异。高C或更强成绩相关都不能单独替代预测准确性检验。

## 1. 分析对象、实际覆盖和计量单位

本报告只读取本次 v2 目录中的结果。主分析是保存的 K 个未平滑预测样本的区域概率估计；期望点、argmax 和逐帧抽样分别是独立的点解码分支。两类结果不能混称。

运行目录：[20261105T063658759189Z_v2_full](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full>)。统计 bootstrap 配置为 10000 次，seed=42。

| 任务 | 读取录像 | 读取人数 | 真实有效录像 | 真实有效人数 | 真实有效帧 | 真实有效秒 | 配对录像 | 配对人数 | 配对帧 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A Sea Spikes | 33 | 11 | 33 | 11 | 367203 | 8614.7 | 5 | 5 | 58649 |
| C Knot-tying | 51 | 11 | 51 | 11 | 336886 | 8022.8 | 5 | 5 | 36186 |

`real_all` 用本次选择范围内所有可用真实 gaze；`paired_test` 只用有五模型预测、且五模型及三种解码共同有效的测试录像/帧。若本次是 subset，上表仅代表该 subset。不能把真实全量人数当成模型测试人数，也不能把两个集合的均值直接作模型误差。

每录像/阶段先按原生时间间隔加权；同一参与者的有效录像再等权平均，组水平对参与者等权平均。`whole` 是纳入阶段的时间合并结果，不是四个阶段均值。表中比例通常按百分比显示，差值用百分点；Brier、BSS 和 Spearman ρ 保持原值。方括号是参与者 bootstrap 95% CI，n 是该具体指标的实际人数。

| 任务 | 阶段 | 录像 | 有有效帧人数 | 阶段帧 | 有target % | 有gaze % | 双侧DLC坐标 % | 共同有效 % | 配对有效帧 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | reach ring | 33 | 11 | 122724 | 98.6 | 94.3 | 100.0 | 93.1 | 19049 |
| A | grab ring | 33 | 11 | 64648 | 99.9 | 95.0 | 100.0 | 94.9 | 7944 |
| A | transfer ring | 33 | 11 | 130674 | 99.8 | 94.1 | 100.0 | 94.0 | 20303 |
| A | place ring | 33 | 11 | 72190 | 99.9 | 95.5 | 100.0 | 95.4 | 11353 |
| C | reach string | 51 | 11 | 156116 | 88.5 | 94.6 | 100.0 | 83.6 | 12342 |
| C | wrap around psm | 51 | 11 | 211102 | 99.1 | 88.5 | 100.0 | 87.7 | 21392 |
| C | pull through and cinch | 51 | 6 | 93752 | 12.5 | 94.6 | 100.0 | 12.1 | 1622 |
| C | threading string through tower | 51 | 8 | 10250 | 100.0 | 97.3 | 100.0 | 97.3 | 830 |

这些覆盖比例分别是边际比例，不能相乘得到交集。DLC 坐标可用表示坐标通过几何检查，不等于其语义是 jaw tip，也不等于已验证定位精度。所有列出的任务阶段都进入统计框架；目标或 gaze 缺失的帧仍不能计算。

Cinch 的当前共同有效帧覆盖为 **12.1%**。覆盖偏低，其估计描述可观测部分；不能把缺失时段视作零占用，也不能据此概括完整 cinch 行为。

源数据中额外 `drop ring` 标签累计 0.716 s；本次纳入 0.000 s。它不属于本次预定的八类阶段，避免把额外标签无声并入 RR/GR/TR/PR。

## 2. 主指标的准确含义

主半径保持 target=50 px、PSM=50 px。当前 target 是已有有效标注槽位的算术均值周围的一个圆盘；它不是所有槽位圆盘的并集。该选择沿用当前明确的分析配置，并未通过本报告改写目标语义。PSM 区域为左右 DLC 点周围圆盘的并集，交叠只计一次。

四种互斥状态为 T=target-only、P=PSM-only、O=两者重叠、N=other。`L_target=L_T+L_O`，`L_gripper=L_P+L_O`，`D=L_T+L_P`，`C=L_T/(L_T+L_P)`。C 描述可分辨区域内偏向 target 的比例；D 描述有多少时间/概率落入可分辨区域。D=0 时 C 为 NA，绝不能填成0；高 C 也不能单独等同于高 target 占用。

真实 gaze 每帧是单点事件。主模型每帧 q 是 K 个保存样本中落入相应区域的比例，再按原生时间加权求占用。这是有限 K 的 Monte Carlo 概率估计，不是把热图期望点硬分类，也不是本次重新生成的平滑热图。有限 K 会带来采样误差；此处参与者 CI 不单独量化模型重新采样的不确定性。

面积基线 a 是区域在画面中的面积比例。E_target=L_target−a_target，E_gripper=L_gripper−a_gripper；C_U=a_T/(a_T+a_P)，C_excess=C−C_U。面积修正控制的是 ROI 大小，不会自动修正器械遮挡、目标语义或追踪误差。

## 3. 有效码、时间轴和区域语义审计

本次读取 874630 个保留视频帧；其中 874629 个可在规定近邻窗口内核对原始 Left Status，874629 个与 NPY valid 相等。该核对是状态来源证据，不是独立的眼动精度校验。

| 任务 | 保留帧 | 2/3且画面内 | 严格3且画面内 | 1/2/3且画面内 | 旧规则额外纳入 |
| --- | --- | --- | --- | --- | --- |
| A | 397453 | 375943 | 120927 | 378156 | 2213 |
| C | 477177 | 438705 | 184727 | 440578 | 1873 |

Varjo 单眼状态 1 表示可见但跟踪不可靠，2 表示跟踪但质量受影响，3 表示跟踪；主规则使用2/3，严格3为敏感性，1/2/3仅作为旧规则敏感性。原始 combined `Status=2` 是另一套枚举，不能直接替代 NPY 单眼 valid。来源：[Varjo eye-tracking documentation](https://developer.varjo.com/docs/native/eye-tracking)。

预测样本采用40×32含端点网格，1280×1024画面上的节点间距约为32.82×33.03 px。半径25 px已接近/小于单个节点间距，其结果需要结合空间量化解释；敏感性分析不会补出模型未保存的亚网格信息。训练平均热图等网格基线按像素中心Voronoi单元的覆盖比例积分；模型MC则直接测试保存样本坐标是否落入圆盘。50 px在此保留为分析设定，本报告不补称其已由逐参与者校准误差验证。

候选校准重投影在 81/84 份录像中更支持左眼坐标来源。该比较搜索现存校准矩阵，最终历史 NPY 生成脚本及每次校准选用链仍未完整确认；未确定的录像没有被自动改成 combined gaze。

| 录像 | NPY时间不同帧 | 最大绝对差 s | 备份与视频时间匹配帧 |
| --- | --- | --- | --- |
| User1_1_1 | 12595 | 2915.013247 | 12595 |
| User1_6_1 | 7327 | 742.064749 | 7327 |

分析时间轴取精确 `video_ts.csv` 并按原始 Frame_Number 对齐。录制源码在视频 WriteSample 后写入主机 system_clock，gaze 使用 Varjo captureTime 转 Unix；因此它是记录时间轴，不是已校准的曝光/显示硬件时间。模型 timestamp 一致能证明共享帧索引和记录时钟一致，不能独立排除固定延迟或时钟漂移。NPY 时间精度损坏和真实采集同步错误是不同问题。

Mask 推导的 distal/jaw-tip 候选 **未通过主分析替换验证**：32 个侧别候选中 19 个通过启发式 QC，但其中至少 4 个在抽样视觉复核中明显选到近端。该复核不是专家真值误差率，不能把通过启发式 QC 的点称作已验证 jaw tip。原 DLC 点保持不变，主结果必须称为 **腕部/铰接部附近50 px区域的注视**，不能称 jaw-tip attention。

| 阶段 | occurrence数 | 有target帧 | 槽位半径>50px帧 | 槽位身份集合变化次数 |
| --- | --- | --- | --- | --- |
| reach ring | 423 | 121034 | 58763 | 296 |
| grab ring | 423 | 64583 | 28133 | 75 |
| transfer ring | 418 | 130476 | 1479 | 162 |
| place ring | 407 | 72083 | 268 | 88 |
| reach string | 346 | 138153 | 11912 | 75 |
| wrap around psm | 196 | 209286 | 130851 | 71 |
| pull through and cinch | 112 | 11681 | 0 | 46 |
| threading string through tower | 43 | 10250 | 3896 | 8 |

这里的身份集合变化来自有效槽位 bitmask，不等同于独立物体身份切换。几何 flags 只用于标注质量说明，不能据此自动把 TR 的环改成 peg、把 wrap 的目标改成器械、或排除整个阶段。目标语义仍需标注协议确认；目标/腕部重合时 C 只能作空间偏好解释。

## 4. 概率主分析：任务整体

所有单元格是参与者均值 [95% CI]；比例单位为%。同一行各指标可能因 C 的零分母而有不同 n，故逐格保留实际人数。

### 真实 gaze 全量

| 任务 | 来源 | target占用 | 腕部占用 | C | D | 重叠O |
| --- | --- | --- | --- | --- | --- | --- |
| A | 真实 gaze | 25.0 [21.2, 28.6]；n=11 | 9.5 [6.3, 13.2]；n=11 | 76.8 [70.3, 82.9]；n=11 | 29.0 [24.8, 33.2]；n=11 | 2.7 [1.5, 4.2]；n=11 |
| C | 真实 gaze | 24.7 [21.2, 27.8]；n=11 | 23.4 [19.9, 26.7]；n=11 | 51.4 [44.2, 59.3]；n=11 | 24.3 [21.6, 26.9]；n=11 | 11.8 [9.5, 14.5]；n=11 |

### held-out共同帧

| 任务 | 来源 | target占用 | 腕部占用 | C | D | 重叠O |
| --- | --- | --- | --- | --- | --- | --- |
| A | 真实 gaze | 22.7 [18.2, 26.5]；n=5 | 9.8 [4.3, 16.5]；n=5 | 75.0 [65.4, 84.1]；n=5 | 28.2 [21.9, 33.6]；n=5 | 2.1 [0.3, 4.5]；n=5 |
| A | all_ABCD | 6.4 [5.8, 7.3]；n=5 | 7.1 [6.2, 8.4]；n=5 | 47.2 [41.4, 52.6]；n=5 | 12.3 [11.5, 13.2]；n=5 | 0.6 [0.2, 1.0]；n=5 |
| A | single_task | 6.8 [6.2, 7.7]；n=5 | 7.3 [6.0, 8.9]；n=5 | 48.1 [41.7, 54.5]；n=5 | 12.9 [12.0, 13.8]；n=5 | 0.6 [0.2, 1.1]；n=5 |
| A | leave_out | 4.9 [4.6, 5.3]；n=5 | 4.9 [3.8, 6.1]；n=5 | 50.6 [44.0, 56.6]；n=5 | 8.9 [8.0, 9.7]；n=5 | 0.5 [0.2, 0.8]；n=5 |
| A | SPD fine-tune | 6.5 [6.0, 7.3]；n=5 | 7.1 [6.1, 8.4]；n=5 | 47.9 [41.6, 53.4]；n=5 | 12.4 [11.5, 13.2]；n=5 | 0.6 [0.2, 1.0]；n=5 |
| A | SPD zero-shot | 5.2 [4.6, 6.0]；n=5 | 3.9 [2.9, 4.9]；n=5 | 58.1 [49.8, 65.0]；n=5 | 8.3 [7.2, 9.4]；n=5 | 0.4 [0.1, 0.7]；n=5 |
| C | 真实 gaze | 25.9 [12.6, 39.9]；n=5 | 21.3 [14.1, 28.6]；n=5 | 57.2 [34.7, 71.3]；n=5 | 24.2 [16.8, 31.7]；n=5 | 11.5 [5.4, 17.7]；n=5 |
| C | all_ABCD | 11.8 [10.3, 13.3]；n=5 | 14.7 [12.4, 16.3]；n=5 | 41.2 [32.4, 50.1]；n=5 | 15.4 [13.7, 17.3]；n=5 | 5.5 [4.9, 6.4]；n=5 |
| C | single_task | 12.1 [10.1, 14.3]；n=5 | 14.7 [11.9, 16.9]；n=5 | 42.0 [33.3, 51.3]；n=5 | 15.4 [13.4, 17.6]；n=5 | 5.7 [4.7, 7.0]；n=5 |
| C | leave_out | 6.3 [5.1, 7.3]；n=5 | 9.5 [6.8, 11.6]；n=5 | 34.9 [29.9, 40.8]；n=5 | 9.9 [7.6, 11.8]；n=5 | 2.9 [2.1, 3.9]；n=5 |
| C | SPD fine-tune | 11.1 [9.7, 12.5]；n=5 | 13.6 [11.7, 15.3]；n=5 | 41.6 [33.3, 50.2]；n=5 | 14.5 [12.7, 16.5]；n=5 | 5.1 [4.6, 5.9]；n=5 |
| C | SPD zero-shot | 8.7 [7.1, 10.1]；n=5 | 13.8 [11.3, 15.9]；n=5 | 32.2 [27.8, 35.9]；n=5 | 14.3 [12.2, 15.8]；n=5 | 4.1 [3.0, 5.1]；n=5 |

## 5. 概率主分析：逐阶段

以下模型和真实 gaze 均来自 paired_test 的同一有效帧集合。阶段结果用于定位模型偏差；八个阶段、多模型、多指标会产生大量比较，不应仅挑选几个区间不跨0的结果作为确认性结论。

### Task A：Sea Spikes

| 阶段 | 来源 | target占用 | 腕部占用 | C | D | 重叠O |
| --- | --- | --- | --- | --- | --- | --- |
| reach ring | 真实 gaze | 11.0 [6.4, 15.4]；n=5 | 10.4 [4.0, 17.5]；n=5 | 55.1 [40.1, 71.5]；n=5 | 20.0 [12.1, 28.5]；n=5 | 0.7 [0.0, 1.9]；n=5 |
| reach ring | all_ABCD | 3.7 [2.4, 5.4]；n=5 | 6.3 [5.1, 7.4]；n=5 | 35.6 [26.9, 47.9]；n=5 | 9.6 [8.0, 11.1]；n=5 | 0.2 [0.0, 0.4]；n=5 |
| reach ring | single_task | 4.4 [3.1, 6.1]；n=5 | 6.6 [5.3, 8.0]；n=5 | 38.2 [31.4, 49.5]；n=5 | 10.5 [8.8, 12.1]；n=5 | 0.3 [0.0, 0.5]；n=5 |
| reach ring | leave_out | 1.0 [0.5, 1.5]；n=5 | 3.7 [2.1, 5.2]；n=5 | 21.1 [10.8, 34.0]；n=5 | 4.4 [2.8, 5.9]；n=5 | 0.1 [0.0, 0.3]；n=5 |
| reach ring | SPD fine-tune | 3.8 [2.4, 5.5]；n=5 | 6.2 [5.2, 7.4]；n=5 | 36.3 [27.2, 47.8]；n=5 | 9.6 [8.0, 11.2]；n=5 | 0.2 [0.0, 0.4]；n=5 |
| reach ring | SPD zero-shot | 1.9 [1.2, 2.7]；n=5 | 3.6 [2.5, 4.8]；n=5 | 33.6 [24.0, 41.6]；n=5 | 5.2 [3.6, 6.6]；n=5 | 0.2 [0.0, 0.3]；n=5 |
| grab ring | 真实 gaze | 27.9 [16.9, 41.4]；n=5 | 11.5 [3.5, 20.2]；n=5 | 76.0 [61.9, 87.6]；n=5 | 34.8 [22.2, 48.5]；n=5 | 2.3 [0.3, 4.3]；n=5 |
| grab ring | all_ABCD | 9.7 [8.4, 11.0]；n=5 | 7.4 [5.3, 9.9]；n=5 | 58.0 [50.4, 65.7]；n=5 | 15.9 [13.5, 18.3]；n=5 | 0.7 [0.1, 1.2]；n=5 |
| grab ring | single_task | 10.8 [9.1, 12.4]；n=5 | 8.6 [6.6, 11.1]；n=5 | 56.2 [49.8, 62.8]；n=5 | 18.0 [14.9, 20.8]；n=5 | 0.7 [0.1, 1.3]；n=5 |
| grab ring | leave_out | 2.8 [1.9, 3.8]；n=5 | 3.1 [1.8, 4.4]；n=5 | 49.5 [35.3, 63.6]；n=5 | 5.4 [3.9, 6.6]；n=5 | 0.3 [0.0, 0.6]；n=5 |
| grab ring | SPD fine-tune | 10.0 [8.8, 11.2]；n=5 | 7.0 [5.0, 9.4]；n=5 | 60.2 [53.1, 67.4]；n=5 | 15.8 [13.7, 18.1]；n=5 | 0.6 [0.1, 1.1]；n=5 |
| grab ring | SPD zero-shot | 5.1 [3.3, 6.8]；n=5 | 3.8 [2.2, 5.3]；n=5 | 58.7 [51.3, 70.3]；n=5 | 8.0 [5.3, 11.1]；n=5 | 0.4 [0.0, 0.8]；n=5 |
| transfer ring | 真实 gaze | 22.2 [20.1, 24.7]；n=5 | 5.4 [2.0, 9.0]；n=5 | 84.5 [76.5, 92.7]；n=5 | 25.2 [21.7, 29.2]；n=5 | 1.2 [0.1, 2.4]；n=5 |
| transfer ring | all_ABCD | 5.2 [4.6, 5.8]；n=5 | 5.9 [4.8, 6.9]；n=5 | 46.9 [40.7, 53.2]；n=5 | 10.3 [9.5, 10.9]；n=5 | 0.4 [0.2, 0.6]；n=5 |
| transfer ring | single_task | 5.4 [4.8, 6.0]；n=5 | 5.9 [4.6, 7.3]；n=5 | 48.2 [41.4, 55.7]；n=5 | 10.5 [9.4, 11.5]；n=5 | 0.4 [0.2, 0.6]；n=5 |
| transfer ring | leave_out | 5.6 [5.1, 6.1]；n=5 | 4.6 [3.5, 5.8]；n=5 | 55.9 [49.0, 62.5]；n=5 | 9.5 [8.5, 10.6]；n=5 | 0.4 [0.2, 0.6]；n=5 |
| transfer ring | SPD fine-tune | 5.2 [4.6, 5.8]；n=5 | 5.9 [4.7, 7.0]；n=5 | 47.2 [40.6, 53.9]；n=5 | 10.2 [9.5, 10.9]；n=5 | 0.4 [0.2, 0.6]；n=5 |
| transfer ring | SPD zero-shot | 5.6 [4.6, 6.7]；n=5 | 3.3 [2.3, 4.3]；n=5 | 63.3 [54.0, 74.4]；n=5 | 8.3 [7.5, 9.0]；n=5 | 0.3 [0.1, 0.4]；n=5 |
| place ring | 真实 gaze | 38.7 [23.2, 52.2]；n=5 | 15.7 [5.3, 26.4]；n=5 | 75.4 [52.2, 93.5]；n=5 | 41.5 [34.9, 48.7]；n=5 | 6.4 [1.0, 13.9]；n=5 |
| place ring | all_ABCD | 11.2 [10.3, 12.1]；n=5 | 10.7 [9.1, 12.5]；n=5 | 51.2 [44.9, 57.3]；n=5 | 18.4 [16.1, 20.1]；n=5 | 1.7 [0.7, 2.8]；n=5 |
| place ring | single_task | 11.0 [10.4, 11.5]；n=5 | 10.2 [8.1, 12.6]；n=5 | 52.1 [45.7, 59.2]；n=5 | 17.7 [15.8, 19.0]；n=5 | 1.7 [0.6, 2.8]；n=5 |
| place ring | leave_out | 11.5 [10.8, 12.2]；n=5 | 8.8 [7.0, 10.7]；n=5 | 58.1 [52.9, 63.3]；n=5 | 17.1 [15.1, 19.1]；n=5 | 1.6 [0.6, 2.6]；n=5 |
| place ring | SPD fine-tune | 11.3 [10.2, 12.3]；n=5 | 10.9 [9.4, 12.6]；n=5 | 50.8 [44.7, 55.8]；n=5 | 18.7 [16.6, 20.7]；n=5 | 1.7 [0.7, 2.8]；n=5 |
| place ring | SPD zero-shot | 9.8 [8.5, 11.4]；n=5 | 5.3 [4.2, 6.6]；n=5 | 66.8 [59.8, 72.5]；n=5 | 13.0 [11.1, 15.0]；n=5 | 1.1 [0.4, 1.8]；n=5 |

### Task C：Knot-tying

| 阶段 | 来源 | target占用 | 腕部占用 | C | D | 重叠O |
| --- | --- | --- | --- | --- | --- | --- |
| reach string | 真实 gaze | 24.8 [7.8, 44.9]；n=5 | 20.0 [9.6, 30.8]；n=5 | 53.7 [27.3, 70.0]；n=5 | 25.7 [17.7, 34.8]；n=5 | 9.6 [0.6, 20.2]；n=5 |
| reach string | all_ABCD | 8.5 [5.4, 11.7]；n=5 | 13.4 [9.5, 16.8]；n=5 | 35.1 [29.8, 39.5]；n=5 | 17.0 [12.6, 20.7]；n=5 | 2.5 [1.0, 4.0]；n=5 |
| reach string | single_task | 8.8 [5.5, 12.1]；n=5 | 13.2 [8.6, 16.9]；n=5 | 36.7 [30.9, 41.2]；n=5 | 16.8 [12.1, 20.8]；n=5 | 2.6 [1.0, 4.1]；n=5 |
| reach string | leave_out | 5.1 [2.8, 7.5]；n=5 | 9.1 [6.3, 10.9]；n=5 | 31.3 [24.9, 38.3]；n=5 | 11.1 [8.0, 13.5]；n=5 | 1.5 [0.6, 2.4]；n=5 |
| reach string | SPD fine-tune | 8.2 [5.2, 11.1]；n=5 | 12.3 [8.6, 15.9]；n=5 | 36.5 [31.2, 41.1]；n=5 | 16.0 [12.0, 19.6]；n=5 | 2.2 [0.9, 3.7]；n=5 |
| reach string | SPD zero-shot | 5.0 [2.5, 7.5]；n=5 | 13.0 [10.5, 15.2]；n=5 | 21.4 [15.0, 27.7]；n=5 | 14.6 [12.1, 17.1]；n=5 | 1.7 [0.7, 2.8]；n=5 |
| wrap around psm | 真实 gaze | 30.1 [17.4, 42.7]；n=5 | 25.7 [17.5, 32.2]；n=5 | 57.0 [38.1, 71.8]；n=5 | 26.7 [18.7, 35.1]；n=5 | 14.5 [9.3, 19.7]；n=5 |
| wrap around psm | all_ABCD | 14.9 [13.2, 17.2]；n=5 | 17.5 [15.7, 19.4]；n=5 | 42.0 [33.6, 50.0]；n=5 | 16.2 [14.3, 18.0]；n=5 | 8.1 [7.0, 9.2]；n=5 |
| wrap around psm | single_task | 14.9 [13.1, 16.8]；n=5 | 17.6 [16.0, 19.1]；n=5 | 41.7 [34.3, 48.9]；n=5 | 16.2 [14.1, 18.0]；n=5 | 8.2 [7.1, 9.2]；n=5 |
| wrap around psm | leave_out | 7.7 [6.1, 10.2]；n=5 | 10.9 [8.3, 13.6]；n=5 | 35.3 [30.5, 41.5]；n=5 | 10.1 [8.1, 12.1]；n=5 | 4.2 [3.0, 6.2]；n=5 |
| wrap around psm | SPD fine-tune | 13.9 [12.0, 16.4]；n=5 | 16.3 [14.4, 18.0]；n=5 | 42.0 [34.4, 49.7]；n=5 | 15.1 [12.9, 17.0]；n=5 | 7.6 [6.4, 8.7]；n=5 |
| wrap around psm | SPD zero-shot | 11.5 [10.3, 13.0]；n=5 | 15.6 [13.2, 18.5]；n=5 | 36.6 [32.6, 40.7]；n=5 | 15.1 [14.1, 16.1]；n=5 | 6.0 [4.6, 7.7]；n=5 |
| pull through and cinch | 真实 gaze | 10.5 [8.1, 12.9]；n=2 | 0.9 [0.0, 1.7]；n=2 | 91.3 [82.7, 100.0]；n=2 | 11.3 [9.8, 12.9]；n=2 | 0.0 [0.0, 0.0]；n=2 |
| pull through and cinch | all_ABCD | 12.5 [6.1, 18.8]；n=2 | 1.8 [0.1, 3.6]；n=2 | 81.6 [63.6, 99.5]；n=2 | 14.2 [9.4, 18.9]；n=2 | 0.1 [0.0, 0.2]；n=2 |
| pull through and cinch | single_task | 13.7 [6.8, 20.6]；n=2 | 1.7 [0.0, 3.4]；n=2 | 83.6 [67.5, 99.8]；n=2 | 15.2 [9.7, 20.7]；n=2 | 0.1 [0.0, 0.2]；n=2 |
| pull through and cinch | leave_out | 5.0 [3.8, 6.2]；n=2 | 1.2 [0.3, 2.1]；n=2 | 80.2 [64.9, 95.5]；n=2 | 6.1 [5.8, 6.5]；n=2 | 0.0 [0.0, 0.1]；n=2 |
| pull through and cinch | SPD fine-tune | 12.6 [5.8, 19.4]；n=2 | 1.9 [0.1, 3.8]；n=2 | 80.3 [60.9, 99.7]；n=2 | 14.4 [9.3, 19.4]；n=2 | 0.1 [0.0, 0.2]；n=2 |
| pull through and cinch | SPD zero-shot | 6.4 [5.7, 7.1]；n=2 | 3.1 [0.7, 5.6]；n=2 | 70.7 [50.4, 91.0]；n=2 | 9.3 [7.8, 10.8]；n=2 | 0.1 [0.0, 0.2]；n=2 |
| threading string through tower | 真实 gaze | 12.0 [6.8, 21.4]；n=3 | 12.3 [0.0, 29.1]；n=3 | 66.8 [33.5, 100.0]；n=2 | 10.1 [0.0, 23.4]；n=3 | 7.1 [0.0, 13.5]；n=3 |
| threading string through tower | all_ABCD | 6.1 [2.7, 11.2]；n=3 | 6.9 [1.9, 12.7]；n=3 | 39.6 [15.5, 58.3]；n=3 | 7.3 [2.4, 14.9]；n=3 | 2.8 [0.0, 4.5]；n=3 |
| threading string through tower | single_task | 5.5 [3.5, 9.3]；n=3 | 5.1 [1.2, 10.1]；n=3 | 49.0 [23.4, 76.6]；n=3 | 6.4 [1.3, 12.9]；n=3 | 2.1 [0.0, 3.2]；n=3 |
| threading string through tower | leave_out | 5.8 [0.5, 11.6]；n=3 | 7.4 [1.4, 13.1]；n=3 | 27.4 [11.5, 45.4]；n=3 | 6.9 [1.9, 15.7]；n=3 | 3.1 [0.0, 4.8]；n=3 |
| threading string through tower | SPD fine-tune | 6.0 [3.1, 11.1]；n=3 | 6.7 [2.9, 12.1]；n=3 | 38.1 [15.8, 51.9]；n=3 | 7.5 [2.2, 14.3]；n=3 | 2.6 [0.0, 4.4]；n=3 |
| threading string through tower | SPD zero-shot | 2.0 [0.4, 4.2]；n=3 | 6.1 [3.9, 8.9]；n=3 | 12.2 [2.6, 25.4]；n=3 | 6.0 [4.0, 9.7]；n=3 | 1.0 [0.0, 1.7]；n=3 |

## 6. 均匀面积基线与超额占用

整体超额占用和 C_excess 如下；单位为百分点。正值表示超过该帧几何区域对应的均匀位置基线，不是对真实 gaze 的预测改善。

| 任务 | 来源 | E_target | E_腕部 | C_excess |
| --- | --- | --- | --- | --- |
| A | 真实 gaze | 22.1 [17.6, 25.9]；n=5 | 8.6 [3.1, 15.4]；n=5 | 42.0 [32.8, 50.5]；n=5 |
| A | all_ABCD | 5.8 [5.2, 6.7]；n=5 | 6.0 [5.0, 7.2]；n=5 | 14.1 [8.9, 19.7]；n=5 |
| A | single_task | 6.2 [5.6, 7.1]；n=5 | 6.1 [4.9, 7.7]；n=5 | 15.0 [9.2, 21.2]；n=5 |
| A | leave_out | 4.3 [4.0, 4.7]；n=5 | 3.8 [2.7, 4.9]；n=5 | 17.5 [12.0, 23.0]；n=5 |
| A | SPD fine-tune | 5.9 [5.4, 6.7]；n=5 | 5.9 [4.9, 7.2]；n=5 | 14.8 [9.0, 20.5]；n=5 |
| A | SPD zero-shot | 4.6 [4.0, 5.4]；n=5 | 2.7 [1.7, 3.7]；n=5 | 25.1 [17.5, 31.0]；n=5 |
| C | 真实 gaze | 25.3 [12.0, 39.3]；n=5 | 20.1 [12.9, 27.4]；n=5 | 29.4 [7.5, 43.6]；n=5 |
| C | all_ABCD | 11.3 [9.7, 12.7]；n=5 | 13.5 [11.2, 15.1]；n=5 | 13.4 [6.3, 20.3]；n=5 |
| C | single_task | 11.5 [9.5, 13.7]；n=5 | 13.5 [10.7, 15.7]；n=5 | 14.2 [7.0, 21.9]；n=5 |
| C | leave_out | 5.7 [4.5, 6.7]；n=5 | 8.3 [5.7, 10.4]；n=5 | 7.1 [3.9, 10.8]；n=5 |
| C | SPD fine-tune | 10.5 [9.1, 11.9]；n=5 | 12.5 [10.5, 14.1]；n=5 | 13.7 [6.8, 20.6]；n=5 |
| C | SPD zero-shot | 8.1 [6.5, 9.5]；n=5 | 12.7 [10.2, 14.8]；n=5 | 4.3 [1.5, 7.1]；n=5 |

| 任务 | 阶段 | target-only面积 % | PSM-only面积 % | 重叠面积 % | 均匀C_U % |
| --- | --- | --- | --- | --- | --- |
| A | whole | 0.6 [0.5, 0.6]；n=5 | 1.1 [1.1, 1.2]；n=5 | 0.0 [0.0, 0.1]；n=5 | 33.1 [32.3, 34.0]；n=5 |
| A | reach ring | 0.6 [0.6, 0.6]；n=5 | 1.2 [1.1, 1.2]；n=5 | 0.0 [0.0, 0.0]；n=5 | 33.4 [32.7, 34.2]；n=5 |
| A | grab ring | 0.6 [0.5, 0.6]；n=5 | 1.1 [1.1, 1.2]；n=5 | 0.0 [0.0, 0.1]；n=5 | 32.9 [32.1, 33.7]；n=5 |
| A | transfer ring | 0.6 [0.6, 0.6]；n=5 | 1.2 [1.1, 1.2]；n=5 | 0.0 [0.0, 0.0]；n=5 | 33.3 [32.8, 34.0]；n=5 |
| A | place ring | 0.5 [0.5, 0.6]；n=5 | 1.1 [1.0, 1.1]；n=5 | 0.1 [0.0, 0.1]；n=5 | 32.1 [30.3, 33.9]；n=5 |
| C | whole | 0.4 [0.3, 0.4]；n=5 | 0.9 [0.9, 1.0]；n=5 | 0.2 [0.2, 0.3]；n=5 | 27.8 [25.7, 30.0]；n=5 |
| C | reach string | 0.5 [0.4, 0.5]；n=5 | 1.1 [1.0, 1.1]；n=5 | 0.1 [0.1, 0.2]；n=5 | 31.4 [30.3, 32.7]；n=5 |
| C | wrap around psm | 0.3 [0.2, 0.3]；n=5 | 0.9 [0.8, 0.9]；n=5 | 0.3 [0.3, 0.4]；n=5 | 24.1 [22.1, 26.1]；n=5 |
| C | pull through and cinch | 0.6 [0.6, 0.6]；n=2 | 0.9 [0.8, 1.1]；n=2 | 0.0 [0.0, 0.0]；n=2 | 38.6 [34.9, 42.3]；n=2 |
| C | threading string through tower | 0.4 [0.1, 0.6]；n=3 | 0.9 [0.6, 1.1]；n=3 | 0.2 [0.0, 0.5]；n=3 | 25.1 [10.6, 34.4]；n=3 |

## 7. 概率预测与真实事件的误差

Δ=模型−真实。ΔC、Δtarget、Δ腕部单位为百分点；绝对值越小才表示占用更接近。Brier 是每帧 (q−h)² 的原生时间加权值，先试次再参与者汇总，越低越好。它同时受到概率校准、区分度和事件频率影响。

| 任务 | 模型 | ΔC pp | Δtarget pp | Δ腕部 pp | Brier target | Brier 腕部 |
| --- | --- | --- | --- | --- | --- | --- |
| A | all_ABCD | -27.9 [-37.9, -17.5]；n=5 | -16.2 [-20.5, -11.6]；n=5 | -2.6 [-9.3, 2.4]；n=5 | 0.193 [0.157, 0.223]；n=5 | 0.087 [0.041, 0.142]；n=5 |
| A | single_task | -26.9 [-36.6, -17.3]；n=5 | -15.9 [-20.2, -11.1]；n=5 | -2.5 [-8.7, 2.1]；n=5 | 0.192 [0.157, 0.222]；n=5 | 0.087 [0.042, 0.141]；n=5 |
| A | leave_out | -24.4 [-32.3, -16.6]；n=5 | -17.7 [-21.7, -13.2]；n=5 | -4.8 [-11.2, -0.1]；n=5 | 0.200 [0.164, 0.229]；n=5 | 0.088 [0.042, 0.145]；n=5 |
| A | SPD fine-tune | -27.2 [-37.2, -16.5]；n=5 | -16.2 [-20.5, -11.5]；n=5 | -2.7 [-9.4, 2.3]；n=5 | 0.193 [0.158, 0.223]；n=5 | 0.087 [0.041, 0.142]；n=5 |
| A | SPD zero-shot | -16.9 [-25.0, -8.7]；n=5 | -17.5 [-21.9, -12.7]；n=5 | -5.9 [-12.3, -1.3]；n=5 | 0.203 [0.166, 0.236]；n=5 | 0.091 [0.042, 0.152]；n=5 |
| C | all_ABCD | -16.0 [-30.0, 0.9]；n=5 | -14.1 [-27.3, -1.9]；n=5 | -6.6 [-12.3, -0.3]；n=5 | 0.193 [0.103, 0.289]；n=5 | 0.154 [0.111, 0.197]；n=5 |
| C | single_task | -15.2 [-29.6, 2.2]；n=5 | -13.9 [-26.8, -2.1]；n=5 | -6.6 [-11.9, -0.4]；n=5 | 0.188 [0.102, 0.285]；n=5 | 0.152 [0.110, 0.194]；n=5 |
| C | leave_out | -22.3 [-37.8, -2.3]；n=5 | -19.7 [-33.0, -6.3]；n=5 | -11.8 [-17.7, -5.6]；n=5 | 0.224 [0.113, 0.336]；n=5 | 0.173 [0.122, 0.224]；n=5 |
| C | SPD fine-tune | -15.7 [-30.0, 1.8]；n=5 | -14.8 [-28.0, -2.7]；n=5 | -7.7 [-13.3, -1.6]；n=5 | 0.196 [0.104, 0.295]；n=5 | 0.156 [0.112, 0.199]；n=5 |
| C | SPD zero-shot | -25.1 [-37.5, -5.6]；n=5 | -17.2 [-30.5, -4.4]；n=5 | -7.5 [-13.9, -0.8]；n=5 | 0.214 [0.113, 0.321]；n=5 | 0.164 [0.118, 0.209]；n=5 |

| 任务 | 阶段 | 模型 | ΔC pp | Brier target | Brier 腕部 |
| --- | --- | --- | --- | --- | --- |
| A | reach ring | all_ABCD | -19.6 [-35.3, -7.9]；n=5 | 0.097 [0.060, 0.134]；n=5 | 0.092 [0.038, 0.150]；n=5 |
| A | reach ring | single_task | -16.9 [-29.8, -6.7]；n=5 | 0.097 [0.060, 0.131]；n=5 | 0.092 [0.038, 0.149]；n=5 |
| A | reach ring | leave_out | -34.0 [-51.5, -21.3]；n=5 | 0.106 [0.064, 0.145]；n=5 | 0.094 [0.040, 0.153]；n=5 |
| A | reach ring | SPD fine-tune | -18.8 [-35.2, -6.1]；n=5 | 0.098 [0.060, 0.132]；n=5 | 0.092 [0.037, 0.151]；n=5 |
| A | reach ring | SPD zero-shot | -21.6 [-41.1, -5.2]；n=5 | 0.104 [0.062, 0.144]；n=5 | 0.097 [0.039, 0.163]；n=5 |
| A | grab ring | all_ABCD | -18.0 [-32.5, -3.2]；n=5 | 0.228 [0.144, 0.327]；n=5 | 0.103 [0.036, 0.180]；n=5 |
| A | grab ring | single_task | -19.8 [-31.8, -7.3]；n=5 | 0.227 [0.141, 0.329]；n=5 | 0.104 [0.038, 0.178]；n=5 |
| A | grab ring | leave_out | -26.5 [-47.8, -5.3]；n=5 | 0.265 [0.162, 0.396]；n=5 | 0.103 [0.035, 0.180]；n=5 |
| A | grab ring | SPD fine-tune | -15.8 [-29.6, -1.9]；n=5 | 0.230 [0.146, 0.331]；n=5 | 0.102 [0.035, 0.176]；n=5 |
| A | grab ring | SPD zero-shot | -17.3 [-33.4, 0.1]；n=5 | 0.257 [0.155, 0.383]；n=5 | 0.106 [0.036, 0.184]；n=5 |
| A | transfer ring | all_ABCD | -37.5 [-45.8, -29.2]；n=5 | 0.195 [0.177, 0.217]；n=5 | 0.049 [0.022, 0.078]；n=5 |
| A | transfer ring | single_task | -36.2 [-44.1, -28.4]；n=5 | 0.195 [0.176, 0.217]；n=5 | 0.049 [0.022, 0.078]；n=5 |
| A | transfer ring | leave_out | -28.5 [-33.5, -23.6]；n=5 | 0.195 [0.177, 0.215]；n=5 | 0.051 [0.021, 0.082]；n=5 |
| A | transfer ring | SPD fine-tune | -37.2 [-45.4, -29.0]；n=5 | 0.195 [0.177, 0.217]；n=5 | 0.050 [0.022, 0.079]；n=5 |
| A | transfer ring | SPD zero-shot | -21.1 [-28.0, -14.5]；n=5 | 0.198 [0.180, 0.221]；n=5 | 0.051 [0.021, 0.084]；n=5 |
| A | place ring | all_ABCD | -24.2 [-45.2, -1.9]；n=5 | 0.316 [0.195, 0.421]；n=5 | 0.135 [0.055, 0.215]；n=5 |
| A | place ring | single_task | -23.4 [-44.5, -0.3]；n=5 | 0.316 [0.196, 0.419]；n=5 | 0.136 [0.055, 0.218]；n=5 |
| A | place ring | leave_out | -17.3 [-35.0, 2.3]；n=5 | 0.317 [0.198, 0.419]；n=5 | 0.138 [0.055, 0.220]；n=5 |
| A | place ring | SPD fine-tune | -24.6 [-45.4, -1.5]；n=5 | 0.318 [0.197, 0.422]；n=5 | 0.136 [0.056, 0.215]；n=5 |
| A | place ring | SPD zero-shot | -8.6 [-30.5, 14.7]；n=5 | 0.336 [0.213, 0.446]；n=5 | 0.145 [0.053, 0.238]；n=5 |
| C | reach string | all_ABCD | -18.6 [-34.5, 3.5]；n=5 | 0.184 [0.066, 0.327]；n=5 | 0.140 [0.076, 0.204]；n=5 |
| C | reach string | single_task | -17.1 [-32.5, 4.6]；n=5 | 0.183 [0.066, 0.323]；n=5 | 0.139 [0.078, 0.201]；n=5 |
| C | reach string | leave_out | -22.5 [-36.3, -2.1]；n=5 | 0.205 [0.069, 0.365]；n=5 | 0.157 [0.080, 0.233]；n=5 |
| C | reach string | SPD fine-tune | -17.2 [-33.4, 5.0]；n=5 | 0.188 [0.067, 0.333]；n=5 | 0.141 [0.076, 0.205]；n=5 |
| C | reach string | SPD zero-shot | -32.4 [-50.2, -6.9]；n=5 | 0.210 [0.074, 0.369]；n=5 | 0.148 [0.077, 0.219]；n=5 |
| C | wrap around psm | all_ABCD | -15.0 [-29.4, -0.7]；n=5 | 0.222 [0.140, 0.304]；n=5 | 0.189 [0.139, 0.228]；n=5 |
| C | wrap around psm | single_task | -15.3 [-29.4, 0.2]；n=5 | 0.216 [0.137, 0.295]；n=5 | 0.185 [0.136, 0.226]；n=5 |
| C | wrap around psm | leave_out | -21.7 [-41.3, -0.7]；n=5 | 0.264 [0.158, 0.370]；n=5 | 0.211 [0.151, 0.252]；n=5 |
| C | wrap around psm | SPD fine-tune | -15.0 [-29.7, 0.2]；n=5 | 0.224 [0.140, 0.309]；n=5 | 0.191 [0.140, 0.230]；n=5 |
| C | wrap around psm | SPD zero-shot | -20.4 [-35.3, -4.1]；n=5 | 0.247 [0.153, 0.340]；n=5 | 0.201 [0.148, 0.239]；n=5 |
| C | pull through and cinch | all_ABCD | -9.8 [-19.1, -0.5]；n=2 | 0.094 [0.068, 0.121]；n=2 | 0.007 [0.000, 0.014]；n=2 |
| C | pull through and cinch | single_task | -7.7 [-15.2, -0.2]；n=2 | 0.093 [0.068, 0.118]；n=2 | 0.007 [0.000, 0.014]；n=2 |
| C | pull through and cinch | leave_out | -11.1 [-17.8, -4.5]；n=2 | 0.097 [0.074, 0.119]；n=2 | 0.008 [0.000, 0.016]；n=2 |
| C | pull through and cinch | SPD fine-tune | -11.0 [-21.8, -0.3]；n=2 | 0.094 [0.070, 0.118]；n=2 | 0.007 [0.000, 0.013]；n=2 |
| C | pull through and cinch | SPD zero-shot | -20.6 [-32.3, -9.0]；n=2 | 0.097 [0.070, 0.124]；n=2 | 0.008 [0.000, 0.016]；n=2 |
| C | threading string through tower | all_ABCD | -15.1 [-41.7, 11.5]；n=2 | 0.090 [0.043, 0.165]；n=3 | 0.085 [0.001, 0.209]；n=3 |
| C | threading string through tower | single_task | -5.0 [-23.4, 13.4]；n=2 | 0.092 [0.042, 0.172]；n=3 | 0.084 [0.000, 0.208]；n=3 |
| C | threading string through tower | leave_out | -31.4 [-74.8, 11.9]；n=2 | 0.092 [0.048, 0.162]；n=3 | 0.088 [0.000, 0.215]；n=3 |
| C | threading string through tower | SPD fine-tune | -17.5 [-48.1, 13.1]；n=2 | 0.095 [0.052, 0.170]；n=3 | 0.092 [0.001, 0.221]；n=3 |
| C | threading string through tower | SPD zero-shot | -49.8 [-91.4, -8.1]；n=2 | 0.111 [0.067, 0.195]；n=3 | 0.101 [0.003, 0.236]；n=3 |

## 8. 四种 Brier 参照及 GT-smoothed 诊断

BSS=1−平均模型BS/平均参照BS，采用匹配参与者/试次后重算比值；它不是逐帧 skill 的均值。BSS>0 表示优于该指定参照，<0 表示更差，CI跨0表示当前数据不足以确定方向。四个参照各自检验不同问题：

- 画面中心点：固定位置产生的0/1事件预测，衡量相对中心偏好的改进。
- 均匀面积：在整幅画面均匀取点的软事件概率，随ROI面积变化。
- 训练平均热图：训练记录的完整空间密度积分，没有把热图缩成一个均值点。
- 训练事件频率：只用训练数据估计的任务/阶段事件常数概率。

训练参照在任务、场景和阶段内按有效原生时间拟合，不使用测试事件频率。这里采用共同 task-training 参照比较所有模型，因此 leave-out/zero-shot 的参照并不等同于该模型实际训练分布。它们是共同评价基准；需避免把 BSS 当成各模型相对自身训练分布的绝对能力。

### target 事件的 BSS

| 任务 | 模型 | 训练平均热图 | 画面中心点 | 均匀面积 | 训练事件频率 |
| --- | --- | --- | --- | --- | --- |
| A | all_ABCD | 0.124 [0.108, 0.134]；n=5 | 0.152 [0.138, 0.161]；n=5 | 0.140 [0.126, 0.150]；n=5 | -0.096 [-0.142, -0.023]；n=5 |
| A | single_task | 0.127 [0.108, 0.139]；n=5 | 0.156 [0.139, 0.166]；n=5 | 0.143 [0.126, 0.155]；n=5 | -0.092 [-0.137, -0.024]；n=5 |
| A | leave_out | 0.092 [0.065, 0.116]；n=5 | 0.121 [0.098, 0.143]；n=5 | 0.109 [0.081, 0.132]；n=5 | -0.136 [-0.184, -0.073]；n=5 |
| A | SPD fine-tune | 0.121 [0.102, 0.133]；n=5 | 0.150 [0.133, 0.160]；n=5 | 0.137 [0.121, 0.149]；n=5 | -0.100 [-0.143, -0.032]；n=5 |
| A | SPD zero-shot | 0.076 [0.054, 0.099]；n=5 | 0.106 [0.086, 0.127]；n=5 | 0.093 [0.071, 0.116]；n=5 | -0.156 [-0.210, -0.083]；n=5 |
| C | all_ABCD | 0.208 [0.134, 0.235]；n=5 | 0.239 [0.169, 0.274]；n=5 | 0.247 [0.167, 0.276]；n=5 | -0.003 [-0.085, 0.157]；n=5 |
| C | single_task | 0.227 [0.144, 0.264]；n=5 | 0.258 [0.179, 0.286]；n=5 | 0.265 [0.176, 0.305]；n=5 | 0.022 [-0.065, 0.174]；n=5 |
| C | leave_out | 0.082 [0.051, 0.109]；n=5 | 0.118 [0.059, 0.165]；n=5 | 0.126 [0.086, 0.148]；n=5 | -0.163 [-0.275, 0.077]；n=5 |
| C | SPD fine-tune | 0.197 [0.130, 0.221]；n=5 | 0.228 [0.162, 0.268]；n=5 | 0.236 [0.163, 0.262]；n=5 | -0.017 [-0.102, 0.160]；n=5 |
| C | SPD zero-shot | 0.122 [0.046, 0.152]；n=5 | 0.156 [0.096, 0.184]；n=5 | 0.165 [0.079, 0.200]；n=5 | -0.112 [-0.206, 0.079]；n=5 |

### 腕部 事件的 BSS

| 任务 | 模型 | 训练平均热图 | 画面中心点 | 均匀面积 | 训练事件频率 |
| --- | --- | --- | --- | --- | --- |
| A | all_ABCD | 0.076 [-0.018, 0.110]；n=5 | 0.264 [0.184, 0.310]；n=5 | 0.092 [-0.011, 0.128]；n=5 | 0.016 [-0.009, 0.055]；n=5 |
| A | single_task | 0.074 [-0.024, 0.114]；n=5 | 0.262 [0.171, 0.300]；n=5 | 0.089 [-0.013, 0.131]；n=5 | 0.013 [-0.013, 0.039]；n=5 |
| A | leave_out | 0.059 [-0.020, 0.088]；n=5 | 0.250 [0.165, 0.302]；n=5 | 0.074 [-0.011, 0.105]；n=5 | -0.003 [-0.016, 0.033]；n=5 |
| A | SPD fine-tune | 0.075 [-0.014, 0.111]；n=5 | 0.263 [0.185, 0.307]；n=5 | 0.091 [-0.006, 0.127]；n=5 | 0.015 [-0.014, 0.057]；n=5 |
| A | SPD zero-shot | 0.030 [-0.015, 0.046]；n=5 | 0.227 [0.147, 0.300]；n=5 | 0.046 [-0.009, 0.065]；n=5 | -0.033 [-0.067, 0.048]；n=5 |
| C | all_ABCD | 0.200 [0.118, 0.265]；n=5 | 0.283 [0.179, 0.338]；n=5 | 0.259 [0.184, 0.313]；n=5 | 0.080 [0.029, 0.145]；n=5 |
| C | single_task | 0.213 [0.126, 0.277]；n=5 | 0.294 [0.199, 0.348]；n=5 | 0.271 [0.197, 0.322]；n=5 | 0.095 [0.040, 0.164]；n=5 |
| C | leave_out | 0.104 [0.048, 0.161]；n=5 | 0.196 [0.114, 0.245]；n=5 | 0.169 [0.115, 0.216]；n=5 | -0.031 [-0.093, 0.054]；n=5 |
| C | SPD fine-tune | 0.194 [0.106, 0.269]；n=5 | 0.277 [0.173, 0.340]；n=5 | 0.252 [0.175, 0.314]；n=5 | 0.072 [0.011, 0.145]；n=5 |
| C | SPD zero-shot | 0.151 [0.069, 0.212]；n=5 | 0.239 [0.134, 0.295]；n=5 | 0.213 [0.136, 0.265]；n=5 | 0.023 [-0.022, 0.078]；n=5 |

| 任务 | 事件 | 参照 | 参照自身 Brier |
| --- | --- | --- | --- |
| A | target | 训练平均热图 | 0.220 [0.176, 0.258]；n=5 |
| A | target | 画面中心点 | 0.227 [0.183, 0.266]；n=5 |
| A | target | 均匀面积 | 0.224 [0.180, 0.262]；n=5 |
| A | target | 训练事件频率 | 0.176 [0.153, 0.195]；n=5 |
| A | target | GT-smoothed（诊断） | 0.065 [0.052, 0.077]；n=5 |
| A | gripper | 训练平均热图 | 0.094 [0.042, 0.158]；n=5 |
| A | gripper | 画面中心点 | 0.118 [0.051, 0.196]；n=5 |
| A | gripper | 均匀面积 | 0.096 [0.042, 0.162]；n=5 |
| A | gripper | 训练事件频率 | 0.088 [0.044, 0.143]；n=5 |
| A | gripper | GT-smoothed（诊断） | 0.034 [0.017, 0.054]；n=5 |
| C | target | 训练平均热图 | 0.244 [0.119, 0.375]；n=5 |
| C | target | 画面中心点 | 0.254 [0.124, 0.390]；n=5 |
| C | target | 均匀面积 | 0.256 [0.124, 0.394]；n=5 |
| C | target | 训练事件频率 | 0.193 [0.122, 0.266]；n=5 |
| C | target | GT-smoothed（诊断） | 0.075 [0.041, 0.110]；n=5 |
| C | gripper | 训练平均热图 | 0.193 [0.128, 0.258]；n=5 |
| C | gripper | 画面中心点 | 0.215 [0.138, 0.292]；n=5 |
| C | gripper | 均匀面积 | 0.208 [0.137, 0.279]；n=5 |
| C | gripper | 训练事件频率 | 0.168 [0.126, 0.209]；n=5 |
| C | gripper | GT-smoothed（诊断） | 0.068 [0.048, 0.087]；n=5 |

GT-smoothed 使用测试真实 gaze 为中心、σ=1网格的平滑分布，属于对观测进行空间模糊化的诊断。它不是真实预测模型，不进入 BSS 分母，也不是数学上的模型上界；例如硬事件 h 自身的 Brier 可为0，而平滑后的概率通常不为0。

| 任务 | 参与者 | GT-smoothed target % | GT-smoothed腕部 % | GT-smoothed C % |
| --- | --- | --- | --- | --- |
| A | 5 | 16.4 | 9.0 | 68.1 |
| C | 5 | 17.7 | 17.9 | 48.4 |

本行诊断占用为试次→参与者→组的均值，仅描述当前平滑设置；未在报告阶段另行计算CI。

## 9. 模型之间的配对证据

下表取整体 Brier 的 A−B 配对差。负值倾向 A，正值倾向 B；只有区间完全位于0的一侧才支持该方向。比较先按同一录像对齐，再平均到参与者；不比较互不相同样本的独立CI。

| 任务 | 指标 | 配对模型 | A−B [CI] | 区间判断 |
| --- | --- | --- | --- | --- |
| A | BS_target | all_ABCD − leave_out | -0.007 [-0.010, -0.003]；n=5 | 区间全为负 |
| A | BS_gripper | all_ABCD − leave_out | -0.002 [-0.004, 0.000]；n=5 | 区间包含0 |
| A | BS_target | all_ABCD − single_task | 0.001 [-0.000, 0.001]；n=5 | 区间包含0 |
| A | BS_gripper | all_ABCD − single_task | -0.000 [-0.001, 0.001]；n=5 | 区间包含0 |
| A | BS_target | all_ABCD − spd_finetune | -0.001 [-0.002, 0.001]；n=5 | 区间包含0 |
| A | BS_gripper | all_ABCD − spd_finetune | -0.000 [-0.001, 0.000]；n=5 | 区间包含0 |
| A | BS_target | all_ABCD − spd_zero_shot | -0.010 [-0.015, -0.006]；n=5 | 区间全为负 |
| A | BS_gripper | all_ABCD − spd_zero_shot | -0.004 [-0.011, 0.001]；n=5 | 区间包含0 |
| A | BS_target | leave_out − single_task | 0.008 [0.004, 0.012]；n=5 | 区间全为正 |
| A | BS_gripper | leave_out − single_task | 0.001 [-0.000, 0.004]；n=5 | 区间包含0 |
| A | BS_target | leave_out − spd_finetune | 0.006 [0.002, 0.010]；n=5 | 区间全为正 |
| A | BS_gripper | leave_out − spd_finetune | 0.002 [-0.000, 0.004]；n=5 | 区间包含0 |
| A | BS_target | leave_out − spd_zero_shot | -0.003 [-0.006, -0.001]；n=5 | 区间全为负 |
| A | BS_gripper | leave_out − spd_zero_shot | -0.003 [-0.008, 0.001]；n=5 | 区间包含0 |
| A | BS_target | single_task − spd_finetune | -0.001 [-0.002, 0.000]；n=5 | 区间包含0 |
| A | BS_gripper | single_task − spd_finetune | 0.000 [-0.001, 0.001]；n=5 | 区间包含0 |
| A | BS_target | single_task − spd_zero_shot | -0.011 [-0.016, -0.007]；n=5 | 区间全为负 |
| A | BS_gripper | single_task − spd_zero_shot | -0.004 [-0.012, 0.001]；n=5 | 区间包含0 |
| A | BS_target | spd_finetune − spd_zero_shot | -0.010 [-0.015, -0.005]；n=5 | 区间全为负 |
| A | BS_gripper | spd_finetune − spd_zero_shot | -0.004 [-0.012, 0.001]；n=5 | 区间包含0 |
| C | BS_target | all_ABCD − leave_out | -0.031 [-0.056, -0.009]；n=5 | 区间全为负 |
| C | BS_gripper | all_ABCD − leave_out | -0.019 [-0.027, -0.009]；n=5 | 区间全为负 |
| C | BS_target | all_ABCD − single_task | 0.005 [0.001, 0.011]；n=5 | 区间全为正 |
| C | BS_gripper | all_ABCD − single_task | 0.003 [-0.000, 0.005]；n=5 | 区间包含0 |
| C | BS_target | all_ABCD − spd_finetune | -0.003 [-0.006, 0.000]；n=5 | 区间包含0 |
| C | BS_gripper | all_ABCD − spd_finetune | -0.001 [-0.004, 0.001]；n=5 | 区间包含0 |
| C | BS_target | all_ABCD − spd_zero_shot | -0.021 [-0.033, -0.009]；n=5 | 区间全为负 |
| C | BS_gripper | all_ABCD − spd_zero_shot | -0.009 [-0.013, -0.005]；n=5 | 区间全为负 |
| C | BS_target | leave_out − single_task | 0.036 [0.011, 0.065]；n=5 | 区间全为正 |
| C | BS_gripper | leave_out − single_task | 0.021 [0.010, 0.032]；n=5 | 区间全为正 |
| C | BS_target | leave_out − spd_finetune | 0.028 [0.009, 0.049]；n=5 | 区间全为正 |
| C | BS_gripper | leave_out − spd_finetune | 0.017 [0.007, 0.027]；n=5 | 区间全为正 |
| C | BS_target | leave_out − spd_zero_shot | 0.010 [-0.002, 0.026]；n=5 | 区间包含0 |
| C | BS_gripper | leave_out − spd_zero_shot | 0.009 [0.003, 0.015]；n=5 | 区间全为正 |
| C | BS_target | single_task − spd_finetune | -0.007 [-0.016, -0.002]；n=5 | 区间全为负 |
| C | BS_gripper | single_task − spd_finetune | -0.004 [-0.009, -0.001]；n=5 | 区间全为负 |
| C | BS_target | single_task − spd_zero_shot | -0.026 [-0.041, -0.010]；n=5 | 区间全为负 |
| C | BS_gripper | single_task − spd_zero_shot | -0.012 [-0.018, -0.006]；n=5 | 区间全为负 |
| C | BS_target | spd_finetune − spd_zero_shot | -0.018 [-0.028, -0.008]；n=5 | 区间全为负 |
| C | BS_gripper | spd_finetune − spd_zero_shot | -0.008 [-0.013, -0.003]；n=5 | 区间全为负 |

逐阶段共有 107 个上述误差比较的未校正95% CI不跨0，完整列于下表。它们是探索性方向证据，不作多重比较后的显著性声明。

| 任务 | 阶段 | 误差指标 | A−B | 差值 [CI] | 误差较小方向 |
| --- | --- | --- | --- | --- | --- |
| A | grab ring | BS_target | all_ABCD − leave_out | -0.037 [-0.070, -0.014]；n=5 | all_ABCD |
| A | grab ring | abs_delta_C | all_ABCD − spd_finetune | 1.9 [0.3, 4.0]；n=5 | spd_finetune |
| A | grab ring | BS_target | all_ABCD − spd_finetune | -0.002 [-0.005, -0.000]；n=5 | all_ABCD |
| A | grab ring | BS_target | all_ABCD − spd_zero_shot | -0.029 [-0.056, -0.012]；n=5 | all_ABCD |
| A | grab ring | BS_target | leave_out − single_task | 0.038 [0.014, 0.069]；n=5 | single_task |
| A | grab ring | abs_delta_C | leave_out − spd_finetune | 9.1 [0.1, 18.1]；n=5 | spd_finetune |
| A | grab ring | BS_target | leave_out − spd_finetune | 0.035 [0.013, 0.065]；n=5 | spd_finetune |
| A | grab ring | BS_target | single_task − spd_zero_shot | -0.030 [-0.054, -0.014]；n=5 | single_task |
| A | grab ring | BS_target | spd_finetune − spd_zero_shot | -0.027 [-0.052, -0.010]；n=5 | spd_finetune |
| A | place ring | abs_delta_C | all_ABCD − leave_out | 7.0 [2.7, 10.7]；n=5 | leave_out |
| A | place ring | BS_gripper | all_ABCD − spd_finetune | -0.001 [-0.001, -0.001]；n=5 | all_ABCD |
| A | place ring | BS_target | all_ABCD − spd_zero_shot | -0.020 [-0.031, -0.013]；n=5 | all_ABCD |
| A | place ring | abs_delta_C | leave_out − single_task | -6.8 [-11.0, -2.5]；n=5 | leave_out |
| A | place ring | abs_delta_C | leave_out − spd_finetune | -7.3 [-10.5, -3.5]；n=5 | leave_out |
| A | place ring | BS_target | leave_out − spd_zero_shot | -0.019 [-0.028, -0.012]；n=5 | leave_out |
| A | place ring | BS_target | single_task − spd_zero_shot | -0.021 [-0.030, -0.014]；n=5 | single_task |
| A | place ring | BS_target | spd_finetune − spd_zero_shot | -0.019 [-0.031, -0.011]；n=5 | spd_finetune |
| A | reach ring | abs_delta_C | all_ABCD − leave_out | -14.4 [-16.2, -13.4]；n=5 | all_ABCD |
| A | reach ring | BS_target | all_ABCD − leave_out | -0.008 [-0.013, -0.003]；n=5 | all_ABCD |
| A | reach ring | abs_delta_C | all_ABCD − single_task | 2.7 [1.0, 5.5]；n=5 | single_task |
| A | reach ring | BS_target | all_ABCD − spd_zero_shot | -0.007 [-0.012, -0.002]；n=5 | all_ABCD |
| A | reach ring | abs_delta_C | leave_out − single_task | 17.1 [14.5, 21.6]；n=5 | single_task |
| A | reach ring | BS_target | leave_out − single_task | 0.009 [0.003, 0.015]；n=5 | single_task |
| A | reach ring | abs_delta_C | leave_out − spd_finetune | 15.2 [13.5, 16.9]；n=5 | spd_finetune |
| A | reach ring | BS_target | leave_out − spd_finetune | 0.008 [0.003, 0.013]；n=5 | spd_finetune |
| A | reach ring | abs_delta_C | leave_out − spd_zero_shot | 12.4 [7.1, 16.6]；n=5 | spd_zero_shot |
| A | reach ring | BS_target | leave_out − spd_zero_shot | 0.001 [0.001, 0.002]；n=5 | spd_zero_shot |
| A | reach ring | BS_target | single_task − spd_zero_shot | -0.008 [-0.013, -0.002]；n=5 | single_task |
| A | reach ring | BS_target | spd_finetune − spd_zero_shot | -0.007 [-0.012, -0.002]；n=5 | spd_finetune |
| A | transfer ring | abs_delta_C | all_ABCD − leave_out | 9.0 [5.3, 13.1]；n=5 | leave_out |
| A | transfer ring | abs_delta_C | all_ABCD − spd_zero_shot | 16.4 [11.1, 22.9]；n=5 | spd_zero_shot |
| A | transfer ring | abs_delta_C | leave_out − single_task | -7.7 [-11.8, -4.3]；n=5 | leave_out |
| A | transfer ring | abs_delta_C | leave_out − spd_finetune | -8.7 [-12.3, -5.1]；n=5 | leave_out |
| A | transfer ring | abs_delta_C | leave_out − spd_zero_shot | 7.4 [3.7, 12.6]；n=5 | spd_zero_shot |
| A | transfer ring | abs_delta_C | single_task − spd_zero_shot | 15.1 [10.5, 20.4]；n=5 | spd_zero_shot |
| A | transfer ring | abs_delta_C | spd_finetune − spd_zero_shot | 16.1 [11.2, 22.2]；n=5 | spd_zero_shot |
| C | pull through and cinch | BS_gripper | all_ABCD − leave_out | -0.001 [-0.002, -0.000]；n=2 | all_ABCD |
| C | pull through and cinch | abs_delta_C | all_ABCD − single_task | 2.0 [0.3, 3.8]；n=2 | single_task |
| C | pull through and cinch | BS_gripper | all_ABCD − spd_finetune | 0.000 [0.000, 0.001]；n=2 | spd_finetune |
| C | pull through and cinch | abs_delta_C | all_ABCD − spd_zero_shot | -10.9 [-13.2, -8.5]；n=2 | all_ABCD |
| C | pull through and cinch | BS_target | all_ABCD − spd_zero_shot | -0.002 [-0.003, -0.002]；n=2 | all_ABCD |
| C | pull through and cinch | BS_gripper | all_ABCD − spd_zero_shot | -0.001 [-0.003, -0.000]；n=2 | all_ABCD |
| C | pull through and cinch | abs_delta_C | leave_out − single_task | 3.4 [2.6, 4.3]；n=2 | single_task |
| C | pull through and cinch | BS_target | leave_out − single_task | 0.004 [0.001, 0.007]；n=2 | single_task |
| C | pull through and cinch | BS_gripper | leave_out − single_task | 0.001 [0.000, 0.002]；n=2 | single_task |
| C | pull through and cinch | BS_target | leave_out − spd_finetune | 0.002 [0.001, 0.004]；n=2 | spd_finetune |
| C | pull through and cinch | BS_gripper | leave_out − spd_finetune | 0.001 [0.000, 0.002]；n=2 | spd_finetune |
| C | pull through and cinch | abs_delta_C | leave_out − spd_zero_shot | -9.5 [-14.5, -4.5]；n=2 | leave_out |
| C | pull through and cinch | BS_gripper | leave_out − spd_zero_shot | -0.001 [-0.001, -0.000]；n=2 | leave_out |
| C | pull through and cinch | abs_delta_C | single_task − spd_finetune | -3.3 [-6.5, -0.1]；n=2 | single_task |
| C | pull through and cinch | BS_target | single_task − spd_finetune | -0.001 [-0.002, -0.000]；n=2 | single_task |
| C | pull through and cinch | abs_delta_C | single_task − spd_zero_shot | -12.9 [-17.1, -8.7]；n=2 | single_task |
| C | pull through and cinch | BS_target | single_task − spd_zero_shot | -0.004 [-0.006, -0.002]；n=2 | single_task |
| C | pull through and cinch | BS_gripper | single_task − spd_zero_shot | -0.001 [-0.002, -0.000]；n=2 | single_task |
| C | pull through and cinch | abs_delta_C | spd_finetune − spd_zero_shot | -9.6 [-10.5, -8.7]；n=2 | spd_finetune |
| C | pull through and cinch | BS_gripper | spd_finetune − spd_zero_shot | -0.002 [-0.003, -0.000]；n=2 | spd_finetune |
| C | reach string | BS_target | all_ABCD − leave_out | -0.020 [-0.038, -0.002]；n=5 | all_ABCD |
| C | reach string | BS_target | all_ABCD − spd_finetune | -0.004 [-0.008, -0.001]；n=5 | all_ABCD |
| C | reach string | abs_delta_C | all_ABCD − spd_zero_shot | -11.7 [-21.3, -2.1]；n=5 | all_ABCD |
| C | reach string | BS_target | all_ABCD − spd_zero_shot | -0.025 [-0.043, -0.008]；n=5 | all_ABCD |
| C | reach string | BS_target | leave_out − single_task | 0.022 [0.000, 0.045]；n=5 | single_task |
| C | reach string | BS_target | leave_out − spd_finetune | 0.016 [0.000, 0.032]；n=5 | spd_finetune |
| C | reach string | abs_delta_C | leave_out − spd_zero_shot | -9.8 [-14.0, -4.4]；n=5 | leave_out |
| C | reach string | BS_target | leave_out − spd_zero_shot | -0.005 [-0.011, -0.000]；n=5 | leave_out |
| C | reach string | abs_delta_C | single_task − spd_zero_shot | -12.8 [-23.6, -2.0]；n=5 | single_task |
| C | reach string | BS_target | single_task − spd_zero_shot | -0.027 [-0.047, -0.008]；n=5 | single_task |
| C | reach string | abs_delta_C | spd_finetune − spd_zero_shot | -12.6 [-23.2, -2.0]；n=5 | spd_finetune |
| C | reach string | BS_target | spd_finetune − spd_zero_shot | -0.021 [-0.036, -0.007]；n=5 | spd_finetune |
| C | threading string through tower | abs_delta_C | all_ABCD − leave_out | -16.7 [-33.1, -0.4]；n=2 | all_ABCD |
| C | threading string through tower | BS_gripper | all_ABCD − single_task | 0.001 [0.000, 0.002]；n=3 | single_task |
| C | threading string through tower | abs_delta_C | all_ABCD − spd_finetune | -4.0 [-6.4, -1.6]；n=2 | all_ABCD |
| C | threading string through tower | BS_gripper | all_ABCD − spd_finetune | -0.007 [-0.013, -0.001]；n=3 | all_ABCD |
| C | threading string through tower | BS_target | all_ABCD − spd_zero_shot | -0.020 [-0.031, -0.004]；n=3 | all_ABCD |
| C | threading string through tower | BS_gripper | all_ABCD − spd_zero_shot | -0.016 [-0.028, -0.003]；n=3 | all_ABCD |
| C | threading string through tower | BS_gripper | leave_out − single_task | 0.004 [0.000, 0.007]；n=3 | single_task |
| C | threading string through tower | BS_gripper | leave_out − spd_finetune | -0.004 [-0.006, -0.001]；n=3 | leave_out |
| C | threading string through tower | BS_gripper | leave_out − spd_zero_shot | -0.013 [-0.021, -0.003]；n=3 | leave_out |
| C | threading string through tower | BS_gripper | single_task − spd_finetune | -0.008 [-0.013, -0.001]；n=3 | single_task |
| C | threading string through tower | BS_target | single_task − spd_zero_shot | -0.019 [-0.028, -0.005]；n=3 | single_task |
| C | threading string through tower | BS_gripper | single_task − spd_zero_shot | -0.017 [-0.028, -0.003]；n=3 | single_task |
| C | threading string through tower | BS_target | spd_finetune − spd_zero_shot | -0.016 [-0.026, -0.004]；n=3 | spd_finetune |
| C | threading string through tower | BS_gripper | spd_finetune − spd_zero_shot | -0.009 [-0.015, -0.002]；n=3 | spd_finetune |
| C | wrap around psm | abs_delta_C | all_ABCD − leave_out | -9.1 [-12.4, -5.3]；n=5 | all_ABCD |
| C | wrap around psm | BS_target | all_ABCD − leave_out | -0.042 [-0.066, -0.018]；n=5 | all_ABCD |
| C | wrap around psm | BS_gripper | all_ABCD − leave_out | -0.022 [-0.030, -0.011]；n=5 | all_ABCD |
| C | wrap around psm | BS_target | all_ABCD − single_task | 0.006 [0.002, 0.013]；n=5 | single_task |
| C | wrap around psm | BS_gripper | all_ABCD − single_task | 0.004 [0.001, 0.007]；n=5 | single_task |
| C | wrap around psm | abs_delta_C | all_ABCD − spd_finetune | -0.5 [-1.0, -0.0]；n=5 | all_ABCD |
| C | wrap around psm | BS_gripper | all_ABCD − spd_finetune | -0.002 [-0.003, -0.000]；n=5 | all_ABCD |
| C | wrap around psm | abs_delta_C | all_ABCD − spd_zero_shot | -5.1 [-7.1, -3.2]；n=5 | all_ABCD |
| C | wrap around psm | BS_target | all_ABCD − spd_zero_shot | -0.025 [-0.040, -0.010]；n=5 | all_ABCD |
| C | wrap around psm | BS_gripper | all_ABCD − spd_zero_shot | -0.012 [-0.019, -0.005]；n=5 | all_ABCD |
| C | wrap around psm | abs_delta_C | leave_out − single_task | 9.0 [5.8, 12.6]；n=5 | single_task |
| C | wrap around psm | BS_target | leave_out − single_task | 0.048 [0.021, 0.079]；n=5 | single_task |
| C | wrap around psm | BS_gripper | leave_out − single_task | 0.026 [0.014, 0.037]；n=5 | single_task |
| C | wrap around psm | abs_delta_C | leave_out − spd_finetune | 8.6 [5.1, 12.1]；n=5 | spd_finetune |
| C | wrap around psm | BS_target | leave_out − spd_finetune | 0.040 [0.018, 0.062]；n=5 | spd_finetune |
| C | wrap around psm | BS_gripper | leave_out − spd_finetune | 0.020 [0.010, 0.028]；n=5 | spd_finetune |
| C | wrap around psm | BS_gripper | leave_out − spd_zero_shot | 0.010 [0.002, 0.018]；n=5 | spd_zero_shot |
| C | wrap around psm | BS_target | single_task − spd_finetune | -0.008 [-0.018, -0.002]；n=5 | single_task |
| C | wrap around psm | BS_gripper | single_task − spd_finetune | -0.006 [-0.010, -0.002]；n=5 | single_task |
| C | wrap around psm | abs_delta_C | single_task − spd_zero_shot | -5.1 [-7.4, -2.6]；n=5 | single_task |
| C | wrap around psm | BS_target | single_task − spd_zero_shot | -0.031 [-0.047, -0.012]；n=5 | single_task |
| C | wrap around psm | BS_gripper | single_task − spd_zero_shot | -0.016 [-0.025, -0.006]；n=5 | single_task |
| C | wrap around psm | abs_delta_C | spd_finetune − spd_zero_shot | -4.7 [-6.9, -2.4]；n=5 | spd_finetune |
| C | wrap around psm | BS_target | spd_finetune − spd_zero_shot | -0.022 [-0.036, -0.009]；n=5 | spd_finetune |
| C | wrap around psm | BS_gripper | spd_finetune − spd_zero_shot | -0.010 [-0.019, -0.003]；n=5 | spd_finetune |

## 10. 与 SimNow overall score 的关系

每个相关系数在任务内计算：先在同一指标可计算、且有有效成绩的试次上，分别求每位参与者的平均占用与平均SimNow overall score，再计算 Spearman ρ。Completed 与 Failed 中存在的有效成绩可参与；无有效成绩或指标为NA的试次不参与。每格列出实际 n 和不同平均分数的个数；后者小于 n 表示成绩存在ties。

相关性不等于因果，也不能说明模型占用准确。真实全量和测试配对集合必须分别阅读；阶段相关都是探索性分析。bootstrap 重抽参与者并在每次重抽内重新排名；小样本、ties、常量变量会让部分复制无定义或区间很宽。

| 任务 | 集合 | 来源 | C–score ρ [CI] | target–score ρ [CI] | 腕部–score ρ [CI] |
| --- | --- | --- | --- | --- | --- |
| A | real_all | 真实 gaze | 0.036 [-0.702, 0.785]；n=11；分数值=11 | 0.082 [-0.528, 0.735]；n=11；分数值=11 | -0.109 [-0.748, 0.533]；n=11；分数值=11 |
| A | paired_test | 真实 gaze | 0.900 [0.111, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 | -0.800 [-1.000, -0.111]；n=5；分数值=5 |
| A | paired_test | all_ABCD | 0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 | -0.200 [-1.000, 1.000]；n=5；分数值=5 |
| A | paired_test | single_task | 0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 |
| A | paired_test | leave_out | 0.500 [-1.000, 1.000]；n=5；分数值=5 | -0.800 [-1.000, -0.111]；n=5；分数值=5 | -0.700 [-1.000, 0.875]；n=5；分数值=5 |
| A | paired_test | SPD fine-tune | 0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| A | paired_test | SPD zero-shot | 0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 |
| C | real_all | 真实 gaze | -0.174 [-0.706, 0.575]；n=11；分数值=9 | 0.156 [-0.561, 0.833]；n=11；分数值=9 | 0.220 [-0.518, 0.773]；n=11；分数值=9 |
| C | paired_test | 真实 gaze | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 |
| C | paired_test | all_ABCD | 0.000 [-1.000, 1.000]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.000 [-1.000, 1.000]；n=5；分数值=2 |
| C | paired_test | single_task | 0.000 [-1.000, 1.000]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 |
| C | paired_test | leave_out | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.000 [-1.000, 1.000]；n=5；分数值=2 |
| C | paired_test | SPD fine-tune | 0.000 [-1.000, 1.000]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 |
| C | paired_test | SPD zero-shot | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 | -0.354 [-1.000, 0.395]；n=5；分数值=2 |

**Task C测试集合的ties与退化复制**：以真实gaze整体target相关的参与者表为准，平均成绩分布为 0分×4人、50分×1人。该相关的bootstrap有 3210/10000 次无定义，现有CI只由 6790 次可定义复制计算；不可把丢弃的复制理解成零相关。

移除User12后，剩余4人的平均成绩均为0，相关因此不可估。target指标的leave-one-out仅 4/5 次可估。表中的最小/最大范围只覆盖这些可估删除；即使显示相同的窄范围，也不能据此宣称对全部参与者删除都稳健。

### Task A 逐阶段相关

| 阶段 | 集合 | 来源 | C ρ [CI] | target ρ [CI] | 腕部 ρ [CI] |
| --- | --- | --- | --- | --- | --- |
| reach ring | real_all | 真实 gaze | 0.027 [-0.640, 0.769]；n=11；分数值=11 | 0.218 [-0.408, 0.725]；n=11；分数值=11 | 0.000 [-0.632, 0.621]；n=11；分数值=11 |
| reach ring | paired_test | 真实 gaze | -0.500 [-1.000, 1.000]；n=5；分数值=5 | -1.000 [-1.000, -1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| reach ring | paired_test | all_ABCD | -0.200 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| reach ring | paired_test | single_task | -0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.200 [-1.000, 1.000]；n=5；分数值=5 |
| reach ring | paired_test | leave_out | -0.200 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| reach ring | paired_test | SPD fine-tune | -0.200 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| reach ring | paired_test | SPD zero-shot | -0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| grab ring | real_all | 真实 gaze | -0.173 [-0.787, 0.557]；n=11；分数值=11 | 0.600 [0.000, 0.887]；n=11；分数值=11 | 0.045 [-0.636, 0.684]；n=11；分数值=11 |
| grab ring | paired_test | 真实 gaze | 0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 |
| grab ring | paired_test | all_ABCD | -0.200 [-1.000, 1.000]；n=5；分数值=5 | 0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.200 [-1.000, 1.000]；n=5；分数值=5 |
| grab ring | paired_test | single_task | 0.200 [-1.000, 1.000]；n=5；分数值=5 | 0.000 [-1.000, 1.000]；n=5；分数值=5 | -0.200 [-1.000, 1.000]；n=5；分数值=5 |
| grab ring | paired_test | leave_out | 0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.900 [-1.000, -0.111]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| grab ring | paired_test | SPD fine-tune | -0.100 [-1.000, 1.000]；n=5；分数值=5 | 0.200 [-1.000, 1.000]；n=5；分数值=5 | -0.200 [-1.000, 1.000]；n=5；分数值=5 |
| grab ring | paired_test | SPD zero-shot | 0.200 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.700 [-1.000, 0.875]；n=5；分数值=5 |
| transfer ring | real_all | 真实 gaze | 0.209 [-0.651, 0.822]；n=11；分数值=11 | 0.091 [-0.564, 0.659]；n=11；分数值=11 | -0.009 [-0.593, 0.690]；n=11；分数值=11 |
| transfer ring | paired_test | 真实 gaze | 0.800 [0.111, 1.000]；n=5；分数值=5 | -0.400 [-1.000, 1.000]；n=5；分数值=5 | -0.800 [-1.000, -0.111]；n=5；分数值=5 |
| transfer ring | paired_test | all_ABCD | -0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.900 [-1.000, -0.111]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 |
| transfer ring | paired_test | single_task | 0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.900 [-1.000, -0.111]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 |
| transfer ring | paired_test | leave_out | 0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.700 [-1.000, 0.667]；n=5；分数值=5 | -0.700 [-1.000, 0.875]；n=5；分数值=5 |
| transfer ring | paired_test | SPD fine-tune | 0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 |
| transfer ring | paired_test | SPD zero-shot | 0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 |
| place ring | real_all | 真实 gaze | 0.100 [-0.674, 0.769]；n=11；分数值=11 | -0.064 [-0.747, 0.659]；n=11；分数值=11 | 0.073 [-0.583, 0.703]；n=11；分数值=11 |
| place ring | paired_test | 真实 gaze | 0.800 [0.111, 1.000]；n=5；分数值=5 | 0.300 [-1.000, 1.000]；n=5；分数值=5 | -1.000 [-1.000, -1.000]；n=5；分数值=5 |
| place ring | paired_test | all_ABCD | 0.600 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| place ring | paired_test | single_task | 0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.600 [-1.000, 1.000]；n=5；分数值=5 |
| place ring | paired_test | leave_out | 0.500 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.700 [-1.000, 0.875]；n=5；分数值=5 |
| place ring | paired_test | SPD fine-tune | 0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.100 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |
| place ring | paired_test | SPD zero-shot | -0.200 [-1.000, 1.000]；n=5；分数值=5 | -0.300 [-1.000, 1.000]；n=5；分数值=5 | -0.500 [-1.000, 1.000]；n=5；分数值=5 |

### Task C 逐阶段相关

| 阶段 | 集合 | 来源 | C ρ [CI] | target ρ [CI] | 腕部 ρ [CI] |
| --- | --- | --- | --- | --- | --- |
| reach string | real_all | 真实 gaze | 0.009 [-0.687, 0.715]；n=11；分数值=9 | 0.358 [-0.334, 0.848]；n=11；分数值=9 | 0.624 [0.100, 0.906]；n=11；分数值=9 |
| reach string | paired_test | 真实 gaze | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 |
| reach string | paired_test | all_ABCD | -0.354 [-1.000, 0.395]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 |
| reach string | paired_test | single_task | -0.354 [-1.000, 0.395]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 |
| reach string | paired_test | leave_out | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 |
| reach string | paired_test | SPD fine-tune | -0.354 [-1.000, 0.395]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 |
| reach string | paired_test | SPD zero-shot | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 |
| wrap around psm | real_all | 真实 gaze | -0.266 [-0.769, 0.423]；n=11；分数值=9 | 0.193 [-0.574, 0.751]；n=11；分数值=9 | 0.294 [-0.437, 0.850]；n=11；分数值=9 |
| wrap around psm | paired_test | 真实 gaze | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 | 0.707 [0.707, 1.000]；n=5；分数值=2 |
| wrap around psm | paired_test | all_ABCD | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.000 [-1.000, 1.000]；n=5；分数值=2 | -0.354 [-1.000, 0.395]；n=5；分数值=2 |
| wrap around psm | paired_test | single_task | 0.000 [-1.000, 1.000]；n=5；分数值=2 | 0.000 [-1.000, 1.000]；n=5；分数值=2 | 0.000 [-1.000, 1.000]；n=5；分数值=2 |
| wrap around psm | paired_test | leave_out | -0.354 [-1.000, 0.395]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.354 [-0.395, 1.000]；n=5；分数值=2 |
| wrap around psm | paired_test | SPD fine-tune | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.000 [-1.000, 1.000]；n=5；分数值=2 | 0.000 [-1.000, 1.000]；n=5；分数值=2 |
| wrap around psm | paired_test | SPD zero-shot | 0.354 [-0.395, 1.000]；n=5；分数值=2 | 0.000 [-1.000, 1.000]；n=5；分数值=2 | -0.354 [-1.000, 0.395]；n=5；分数值=2 |
| pull through and cinch | real_all | 真实 gaze | -0.203 [-1.000, 1.000]；n=6；分数值=5 | -0.464 [-1.000, 0.636]；n=6；分数值=5 | -0.145 [-1.000, 0.894]；n=6；分数值=5 |
| pull through and cinch | paired_test | 真实 gaze | NA；n=2；分数值=2 | NA；n=2；分数值=2 | NA；n=2；分数值=2 |
| pull through and cinch | paired_test | all_ABCD | NA；n=2；分数值=2 | NA；n=2；分数值=2 | NA；n=2；分数值=2 |
| pull through and cinch | paired_test | single_task | NA；n=2；分数值=2 | NA；n=2；分数值=2 | NA；n=2；分数值=2 |
| pull through and cinch | paired_test | leave_out | NA；n=2；分数值=2 | NA；n=2；分数值=2 | NA；n=2；分数值=2 |
| pull through and cinch | paired_test | SPD fine-tune | NA；n=2；分数值=2 | NA；n=2；分数值=2 | NA；n=2；分数值=2 |
| pull through and cinch | paired_test | SPD zero-shot | NA；n=2；分数值=2 | NA；n=2；分数值=2 | NA；n=2；分数值=2 |
| threading string through tower | real_all | 真实 gaze | -0.405 [-0.941, 0.182]；n=8；分数值=3 | -0.343 [-0.813, 0.282]；n=8；分数值=3 | 0.078 [-0.779, 0.784]；n=8；分数值=3 |
| threading string through tower | paired_test | 真实 gaze | NA；n=2；分数值=1 | NA；n=3；分数值=1 | NA；n=3；分数值=1 |
| threading string through tower | paired_test | all_ABCD | NA；n=3；分数值=1 | NA；n=3；分数值=1 | NA；n=3；分数值=1 |
| threading string through tower | paired_test | single_task | NA；n=3；分数值=1 | NA；n=3；分数值=1 | NA；n=3；分数值=1 |
| threading string through tower | paired_test | leave_out | NA；n=3；分数值=1 | NA；n=3；分数值=1 | NA；n=3；分数值=1 |
| threading string through tower | paired_test | SPD fine-tune | NA；n=3；分数值=1 | NA；n=3；分数值=1 | NA；n=3；分数值=1 |
| threading string through tower | paired_test | SPD zero-shot | NA；n=3；分数值=1 | NA；n=3；分数值=1 | NA；n=3；分数值=1 |

逐一去掉参与者后的整体相关范围如下。它用于识别单个参与者驱动的关联，不是另一组95% CI；NA表示保留人数/变异不足。范围仅使用可估的删除，必须同时阅读可估次数/全部删除次数。

| 任务 | 集合 | 来源 | 指标 | 有效bootstrap | 无定义bootstrap | LOO可估/全部 | 可估LOO的ρ范围 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | paired_test | all_ABCD | C | 9987 | 13 | 5/5 | -0.400 至 0.600 |
| A | paired_test | all_ABCD | L_target | 9987 | 13 | 5/5 | -0.800 至 0.000 |
| A | paired_test | all_ABCD | L_gripper | 9987 | 13 | 5/5 | -0.800 至 0.200 |
| A | paired_test | 真实 gaze | C | 9987 | 13 | 5/5 | 0.800 至 1.000 |
| A | paired_test | 真实 gaze | L_target | 9987 | 13 | 5/5 | -0.800 至 0.000 |
| A | paired_test | 真实 gaze | L_gripper | 9987 | 13 | 5/5 | -0.800 至 -0.600 |
| A | paired_test | leave_out | C | 9987 | 13 | 5/5 | 0.000 至 0.800 |
| A | paired_test | leave_out | L_target | 9987 | 13 | 5/5 | -0.800 至 -0.600 |
| A | paired_test | leave_out | L_gripper | 9987 | 13 | 5/5 | -1.000 至 -0.400 |
| A | paired_test | single_task | C | 9987 | 13 | 5/5 | -0.400 至 0.600 |
| A | paired_test | single_task | L_target | 9987 | 13 | 5/5 | -0.800 至 0.400 |
| A | paired_test | single_task | L_gripper | 9987 | 13 | 5/5 | -0.600 至 0.000 |
| A | paired_test | SPD fine-tune | C | 9987 | 13 | 5/5 | -0.400 至 0.600 |
| A | paired_test | SPD fine-tune | L_target | 9987 | 13 | 5/5 | -0.800 至 0.000 |
| A | paired_test | SPD fine-tune | L_gripper | 9987 | 13 | 5/5 | -0.800 至 0.000 |
| A | paired_test | SPD zero-shot | C | 9987 | 13 | 5/5 | 0.400 至 0.800 |
| A | paired_test | SPD zero-shot | L_target | 9987 | 13 | 5/5 | -0.800 至 0.400 |
| A | paired_test | SPD zero-shot | L_gripper | 9987 | 13 | 5/5 | -0.800 至 -0.400 |
| C | paired_test | all_ABCD | C | 6790 | 3210 | 4/5 | -0.258 至 0.258 |
| C | paired_test | all_ABCD | L_target | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | all_ABCD | L_gripper | 6790 | 3210 | 4/5 | -0.258 至 0.258 |
| C | paired_test | 真实 gaze | C | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | 真实 gaze | L_target | 6790 | 3210 | 4/5 | 0.775 至 0.775 |
| C | paired_test | 真实 gaze | L_gripper | 6790 | 3210 | 4/5 | 0.775 至 0.775 |
| C | paired_test | leave_out | C | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | leave_out | L_target | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | leave_out | L_gripper | 6790 | 3210 | 4/5 | -0.258 至 0.258 |
| C | paired_test | single_task | C | 6790 | 3210 | 4/5 | -0.258 至 0.258 |
| C | paired_test | single_task | L_target | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | single_task | L_gripper | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | SPD fine-tune | C | 6790 | 3210 | 4/5 | -0.258 至 0.258 |
| C | paired_test | SPD fine-tune | L_target | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | SPD fine-tune | L_gripper | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | SPD zero-shot | C | 6790 | 3210 | 4/5 | 0.775 至 0.775 |
| C | paired_test | SPD zero-shot | L_target | 6790 | 3210 | 4/5 | 0.258 至 0.775 |
| C | paired_test | SPD zero-shot | L_gripper | 6790 | 3210 | 4/5 | -0.775 至 -0.258 |
| A | real_all | 真实 gaze | C | 10000 | 0 | 11/11 | -0.176 至 0.285 |
| A | real_all | 真实 gaze | L_target | 10000 | 0 | 11/11 | -0.224 至 0.248 |
| A | real_all | 真实 gaze | L_gripper | 10000 | 0 | 11/11 | -0.345 至 0.030 |
| C | real_all | 真实 gaze | C | 10000 | 0 | 11/11 | -0.387 至 0.031 |
| C | real_all | 真实 gaze | L_target | 10000 | 0 | 11/11 | 0.036 至 0.450 |
| C | real_all | 真实 gaze | L_gripper | 10000 | 0 | 11/11 | -0.043 至 0.444 |

## 11. 三种点解码的共同帧比较

概率主分析不产生唯一 gaze 轨迹。三个分支来自同一模型的导出文件，但使用的表示不同：期望点和argmax读取保存的 `probability`；MC占用和逐帧抽样读取原始 `samples`。保存的概率图与有限K样本的经验分布相关但不等同，不能把分支差异全部解释为对完全相同分布的点解码规则差异。期望点可能落在两个峰之间，argmax选概率最大的网格位置，逐帧抽样用固定seed从K个样本中独立选一个。三者共同有效帧规则在生成阶段统一，以下从各分支本次统计表读取各自均值和CI，没有重算一轮bootstrap。

当前可查 `dvss_pytorch/model/cvae.py` 的sample函数先对各latent解码的softmax概率求均值得到probability，另用multinomial产生样本；`scripts/predictor.py`分别保存两个字段。该代码没有证明保存probability是对样本作σ=1高斯核平滑；σ=1的gaze_heatmap用于标签/GT诊断。旧预测导出时的代码revision缺失，当前源码不能补足历史版本溯源。

| 分支对 | 配对试次阶段行 | 有效帧数相等 | 有效秒数相等 |
| --- | --- | --- | --- |
| 期望点 / argmax | 300 | True | True |
| 期望点 / 逐帧抽样 | 300 | True | True |
| argmax / 逐帧抽样 | 300 | True | True |

相同帧支持不保证 C 或驻留时长在三个分支中都有值；因此仍显示逐指标实际人数。不同分支CI相互重叠/不重叠不能替代专门的解码器配对差检验。

| 任务 | 模型 | 解码器 | target % | 腕部 % | C % | D % | O % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | all_ABCD | 期望点 | 8.2 [6.3, 9.9]；n=5 | 8.2 [4.9, 11.6]；n=5 | 51.4 [35.2, 68.7]；n=5 | 14.0 [12.8, 15.0]；n=5 | 1.2 [0.3, 2.0]；n=5 |
| A | all_ABCD | argmax | 20.7 [17.7, 24.4]；n=5 | 17.4 [9.7, 26.4]；n=5 | 57.4 [42.4, 71.1]；n=5 | 30.6 [26.5, 35.4]；n=5 | 3.8 [1.1, 6.4]；n=5 |
| A | all_ABCD | 逐帧抽样 | 6.5 [6.0, 7.3]；n=5 | 7.1 [6.0, 8.4]；n=5 | 47.9 [42.0, 54.1]；n=5 | 12.4 [11.6, 13.1]；n=5 | 0.6 [0.2, 1.0]；n=5 |
| A | single_task | 期望点 | 11.5 [7.8, 14.7]；n=5 | 7.1 [3.7, 10.6]；n=5 | 63.0 [44.3, 81.7]；n=5 | 15.3 [14.0, 16.5]；n=5 | 1.7 [0.3, 3.5]；n=5 |
| A | single_task | argmax | 21.0 [18.9, 24.0]；n=5 | 17.0 [8.9, 26.3]；n=5 | 58.7 [43.3, 74.1]；n=5 | 31.1 [27.8, 35.5]；n=5 | 3.4 [1.1, 5.8]；n=5 |
| A | single_task | 逐帧抽样 | 6.9 [6.2, 7.9]；n=5 | 7.2 [5.7, 9.0]；n=5 | 48.9 [41.2, 56.6]；n=5 | 12.9 [12.1, 13.8]；n=5 | 0.6 [0.2, 1.1]；n=5 |
| A | leave_out | 期望点 | 6.7 [4.5, 9.0]；n=5 | 4.5 [1.6, 7.4]；n=5 | 64.8 [41.1, 85.7]；n=5 | 9.7 [7.3, 12.2]；n=5 | 0.8 [0.2, 1.4]；n=5 |
| A | leave_out | argmax | 13.4 [11.0, 15.5]；n=5 | 9.0 [5.1, 12.3]；n=5 | 64.4 [56.8, 74.8]；n=5 | 17.9 [14.0, 20.4]；n=5 | 2.2 [0.6, 4.1]；n=5 |
| A | leave_out | 逐帧抽样 | 5.0 [4.7, 5.2]；n=5 | 5.0 [3.9, 6.2]；n=5 | 50.5 [44.7, 56.0]；n=5 | 9.0 [8.1, 9.8]；n=5 | 0.5 [0.2, 0.8]；n=5 |
| A | SPD fine-tune | 期望点 | 10.1 [7.8, 12.1]；n=5 | 7.3 [4.3, 10.3]；n=5 | 59.4 [46.6, 73.8]；n=5 | 13.8 [12.6, 15.5]；n=5 | 1.8 [0.2, 3.7]；n=5 |
| A | SPD fine-tune | argmax | 20.9 [17.4, 24.5]；n=5 | 17.3 [10.4, 26.7]；n=5 | 57.9 [41.5, 70.4]；n=5 | 30.3 [27.2, 34.6]；n=5 | 3.9 [1.3, 6.6]；n=5 |
| A | SPD fine-tune | 逐帧抽样 | 6.4 [5.8, 7.1]；n=5 | 7.1 [5.9, 8.7]；n=5 | 47.4 [40.7, 53.3]；n=5 | 12.2 [11.4, 13.3]；n=5 | 0.6 [0.2, 1.0]；n=5 |
| A | SPD zero-shot | 期望点 | 4.3 [2.7, 6.8]；n=5 | 1.1 [0.5, 1.7]；n=5 | 78.4 [65.3, 91.5]；n=5 | 5.3 [3.5, 7.6]；n=5 | 0.0 [0.0, 0.1]；n=5 |
| A | SPD zero-shot | argmax | 14.4 [13.2, 15.8]；n=5 | 5.0 [3.3, 7.0]；n=5 | 77.5 [72.5, 82.8]；n=5 | 17.4 [15.8, 19.6]；n=5 | 1.0 [0.2, 1.8]；n=5 |
| A | SPD zero-shot | 逐帧抽样 | 5.1 [4.4, 5.9]；n=5 | 3.7 [2.8, 4.7]；n=5 | 58.6 [49.9, 66.0]；n=5 | 8.2 [7.2, 9.1]；n=5 | 0.3 [0.1, 0.6]；n=5 |
| C | all_ABCD | 期望点 | 19.9 [12.4, 28.1]；n=5 | 20.7 [14.7, 27.5]；n=5 | 46.6 [30.4, 66.6]；n=5 | 20.0 [17.3, 22.2]；n=5 | 10.3 [5.9, 16.3]；n=5 |
| C | all_ABCD | argmax | 35.2 [27.2, 42.0]；n=5 | 29.9 [23.6, 36.1]；n=5 | 58.6 [44.1, 73.2]；n=5 | 31.5 [29.4, 33.7]；n=5 | 16.8 [12.6, 21.2]；n=5 |
| C | all_ABCD | 逐帧抽样 | 11.7 [10.1, 13.3]；n=5 | 14.4 [12.0, 16.2]；n=5 | 41.4 [32.5, 50.1]；n=5 | 15.1 [13.5, 16.9]；n=5 | 5.5 [4.8, 6.5]；n=5 |
| C | single_task | 期望点 | 24.3 [16.7, 34.0]；n=5 | 22.7 [14.5, 30.9]；n=5 | 53.1 [39.5, 71.0]；n=5 | 21.6 [18.6, 24.3]；n=5 | 12.7 [6.9, 20.7]；n=5 |
| C | single_task | argmax | 35.4 [26.2, 45.4]；n=5 | 31.5 [22.1, 40.5]；n=5 | 56.5 [42.2, 72.0]；n=5 | 30.4 [28.4, 32.3]；n=5 | 18.2 [11.7, 26.3]；n=5 |
| C | single_task | 逐帧抽样 | 12.4 [10.3, 14.4]；n=5 | 14.8 [12.1, 17.0]；n=5 | 42.6 [33.4, 53.0]；n=5 | 15.6 [14.1, 17.5]；n=5 | 5.8 [4.7, 7.2]；n=5 |
| C | leave_out | 期望点 | 8.7 [6.6, 11.3]；n=5 | 11.5 [5.9, 16.6]；n=5 | 46.8 [25.0, 73.7]；n=5 | 13.3 [9.2, 17.4]；n=5 | 3.5 [1.7, 5.2]；n=5 |
| C | leave_out | argmax | 16.1 [13.4, 20.7]；n=5 | 20.5 [14.2, 27.2]；n=5 | 42.8 [31.3, 58.8]；n=5 | 20.5 [15.7, 23.8]；n=5 | 8.1 [5.3, 12.2]；n=5 |
| C | leave_out | 逐帧抽样 | 6.1 [5.0, 7.2]；n=5 | 9.2 [6.7, 11.1]；n=5 | 35.3 [29.8, 41.2]；n=5 | 9.7 [7.4, 11.7]；n=5 | 2.8 [2.0, 3.7]；n=5 |
| C | SPD fine-tune | 期望点 | 20.1 [11.9, 28.5]；n=5 | 17.7 [11.5, 24.4]；n=5 | 51.6 [35.7, 71.4]；n=5 | 20.1 [15.8, 24.7]；n=5 | 8.8 [4.9, 14.2]；n=5 |
| C | SPD fine-tune | argmax | 33.5 [26.9, 39.8]；n=5 | 28.4 [20.6, 34.7]；n=5 | 58.3 [43.3, 72.3]；n=5 | 31.4 [29.6, 33.6]；n=5 | 15.3 [11.4, 19.3]；n=5 |
| C | SPD fine-tune | 逐帧抽样 | 11.1 [9.5, 12.7]；n=5 | 13.4 [11.2, 15.1]；n=5 | 42.5 [33.4, 52.3]；n=5 | 14.1 [12.2, 16.1]；n=5 | 5.2 [4.7, 5.8]；n=5 |
| C | SPD zero-shot | 期望点 | 18.0 [15.0, 21.4]；n=5 | 15.1 [11.4, 17.8]；n=5 | 57.8 [49.2, 66.4]；n=5 | 18.2 [15.5, 20.9]；n=5 | 7.4 [5.4, 9.2]；n=5 |
| C | SPD zero-shot | argmax | 21.2 [15.7, 25.8]；n=5 | 24.7 [20.3, 30.1]；n=5 | 43.2 [37.2, 50.1]；n=5 | 25.9 [22.9, 29.4]；n=5 | 10.0 [6.4, 12.8]；n=5 |
| C | SPD zero-shot | 逐帧抽样 | 8.5 [6.9, 9.9]；n=5 | 13.7 [11.0, 16.2]；n=5 | 31.8 [26.5, 36.7]；n=5 | 14.1 [12.2, 15.6]；n=5 | 4.1 [2.8, 5.2]；n=5 |

在每个指标都有值的共同录像上重新配对、再平均至参与者，得到以下描述性解码差（百分点）。此表没有新增CI，不据其单独判断显著差异。

| 任务 | 模型 | A−B | Δtarget pp | Δ腕部 pp | ΔC pp |
| --- | --- | --- | --- | --- | --- |
| A | all_ABCD | 期望点 − argmax | -12.5；n=5 | -9.2；n=5 | -6.0；n=5 |
| A | leave_out | 期望点 − argmax | -6.7；n=5 | -4.5；n=5 | 0.4；n=5 |
| A | single_task | 期望点 − argmax | -9.5；n=5 | -9.8；n=5 | 4.3；n=5 |
| A | SPD fine-tune | 期望点 − argmax | -10.9；n=5 | -10.0；n=5 | 1.5；n=5 |
| A | SPD zero-shot | 期望点 − argmax | -10.2；n=5 | -3.9；n=5 | 0.9；n=5 |
| C | all_ABCD | 期望点 − argmax | -15.3；n=5 | -9.2；n=5 | -11.9；n=5 |
| C | leave_out | 期望点 − argmax | -7.4；n=5 | -9.0；n=5 | 4.0；n=5 |
| C | single_task | 期望点 − argmax | -11.1；n=5 | -8.7；n=5 | -3.5；n=5 |
| C | SPD fine-tune | 期望点 − argmax | -13.4；n=5 | -10.7；n=5 | -6.7；n=5 |
| C | SPD zero-shot | 期望点 − argmax | -3.2；n=5 | -9.6；n=5 | 14.6；n=5 |
| A | all_ABCD | 期望点 − 逐帧抽样 | 1.7；n=5 | 1.2；n=5 | 3.5；n=5 |
| A | leave_out | 期望点 − 逐帧抽样 | 1.7；n=5 | -0.5；n=5 | 14.3；n=5 |
| A | single_task | 期望点 − 逐帧抽样 | 4.6；n=5 | -0.1；n=5 | 14.2；n=5 |
| A | SPD fine-tune | 期望点 − 逐帧抽样 | 3.7；n=5 | 0.3；n=5 | 11.9；n=5 |
| A | SPD zero-shot | 期望点 − 逐帧抽样 | -0.8；n=5 | -2.6；n=5 | 19.8；n=5 |
| C | all_ABCD | 期望点 − 逐帧抽样 | 8.2；n=5 | 6.3；n=5 | 5.3；n=5 |
| C | leave_out | 期望点 − 逐帧抽样 | 2.6；n=5 | 2.4；n=5 | 11.5；n=5 |
| C | single_task | 期望点 − 逐帧抽样 | 11.9；n=5 | 7.9；n=5 | 10.5；n=5 |
| C | SPD fine-tune | 期望点 − 逐帧抽样 | 9.0；n=5 | 4.4；n=5 | 9.2；n=5 |
| C | SPD zero-shot | 期望点 − 逐帧抽样 | 9.5；n=5 | 1.4；n=5 | 26.0；n=5 |
| A | all_ABCD | argmax − 逐帧抽样 | 14.2；n=5 | 10.3；n=5 | 9.5；n=5 |
| A | leave_out | argmax − 逐帧抽样 | 8.4；n=5 | 4.0；n=5 | 13.9；n=5 |
| A | single_task | argmax − 逐帧抽样 | 14.1；n=5 | 9.7；n=5 | 9.8；n=5 |
| A | SPD fine-tune | argmax − 逐帧抽样 | 14.6；n=5 | 10.2；n=5 | 10.5；n=5 |
| A | SPD zero-shot | argmax − 逐帧抽样 | 9.3；n=5 | 1.3；n=5 | 19.0；n=5 |
| C | all_ABCD | argmax − 逐帧抽样 | 23.5；n=5 | 15.6；n=5 | 17.2；n=5 |
| C | leave_out | argmax − 逐帧抽样 | 10.0；n=5 | 11.3；n=5 | 7.5；n=5 |
| C | single_task | argmax − 逐帧抽样 | 23.0；n=5 | 16.6；n=5 | 14.0；n=5 |
| C | SPD fine-tune | argmax − 逐帧抽样 | 22.4；n=5 | 15.1；n=5 | 15.8；n=5 |
| C | SPD zero-shot | argmax − 逐帧抽样 | 12.7；n=5 | 11.0；n=5 | 11.4；n=5 |

## 12. 时间维度：只解释点解码后的观察序列

驻留为同一AOI连续观察段，保留时长≥120 ms的段。每录像/阶段先求这些段的中位时长，再在参与者内平均试次，最后组均值及参与者CI；不是把全体事件池化后的总体中位数。missing帧、原始帧断裂、>100 ms时间缺口和occurrence边界切断序列。触及观察边界的截断段达到120 ms仍计入，并单报截断数量，所以这些结果不是删失校正后的完整 fixation 寿命。

直接切换连接相邻且各≥120 ms的T/P段；经重叠切换连接T→O→P或反向，两个端点段各≥120 ms，中间O段不要求120 ms。other不跳过。两个切换率都用有效观察分钟作分母，单位次/min；两者分别报告，没有把经O的切换漏作直接切换。

| 任务 | 来源 | 解码器 | T驻留 ms | P驻留 ms | O驻留 ms | 直接 次/min | 经O 次/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 真实 gaze | 真实观察 | 327.0 [303.5, 357.4]；n=5 | 272.7 [226.1, 318.6]；n=5 | 259.7 [202.4, 314.9]；n=5 | 0.34 [0.05, 0.77]；n=5 | 0.32 [0.05, 0.69]；n=5 |
| A | all_ABCD | 期望点 | 257.9 [231.6, 300.1]；n=5 | 228.8 [191.8, 272.6]；n=5 | 287.5 [166.5, 462.6]；n=3 | 0.03 [0.00, 0.09]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | all_ABCD | argmax | 245.2 [207.1, 280.6]；n=5 | 222.5 [215.9, 229.1]；n=5 | 214.4 [170.9, 249.9]；n=5 | 0.06 [0.00, 0.18]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | all_ABCD | 逐帧抽样 | NA；n=0 | NA；n=0 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | single_task | 期望点 | 241.6 [226.0, 256.2]；n=5 | 239.1 [218.6, 271.0]；n=5 | 246.8 [159.4, 402.9]；n=4 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | single_task | argmax | 252.4 [227.4, 282.2]；n=5 | 228.4 [190.2, 270.7]；n=5 | 181.0 [160.0, 198.3]；n=5 | 0.17 [0.05, 0.30]；n=5 | 0.06 [0.00, 0.18]；n=5 |
| A | single_task | 逐帧抽样 | NA；n=0 | 125.5 [CI不可估]；n=1 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | leave_out | 期望点 | 266.4 [223.3, 309.6]；n=5 | 233.7 [199.6, 292.8]；n=4 | 214.3 [191.1, 245.0]；n=3 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | leave_out | argmax | 248.3 [236.8, 260.3]；n=5 | 210.5 [178.6, 237.7]；n=5 | 207.9 [171.3, 241.3]；n=5 | 0.10 [0.00, 0.23]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | leave_out | 逐帧抽样 | NA；n=0 | NA；n=0 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | SPD fine-tune | 期望点 | 269.2 [260.3, 278.2]；n=5 | 276.4 [237.9, 315.9]；n=5 | 202.5 [176.4, 232.0]；n=3 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | SPD fine-tune | argmax | 248.0 [221.2, 274.3]；n=5 | 229.0 [185.6, 264.3]；n=5 | 204.7 [171.4, 240.8]；n=5 | 0.12 [0.00, 0.36]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | SPD fine-tune | 逐帧抽样 | NA；n=0 | NA；n=0 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | SPD zero-shot | 期望点 | 311.2 [190.7, 451.8]；n=5 | 176.0 [138.0, 203.0]；n=4 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | SPD zero-shot | argmax | 268.0 [217.0, 345.0]；n=5 | 222.1 [185.6, 286.3]；n=5 | 259.5 [199.5, 319.5]；n=2 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| A | SPD zero-shot | 逐帧抽样 | NA；n=0 | NA；n=0 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | 真实 gaze | 真实观察 | 347.1 [280.9, 389.8]；n=5 | 334.5 [275.6, 419.1]；n=5 | 482.6 [290.3, 811.6]；n=5 | 0.54 [0.10, 0.98]；n=5 | 0.85 [0.25, 1.55]；n=5 |
| C | all_ABCD | 期望点 | 254.0 [222.6, 281.1]；n=5 | 225.2 [199.7, 245.8]；n=5 | 274.2 [208.9, 329.5]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | all_ABCD | argmax | 239.9 [221.4, 259.6]；n=5 | 245.2 [227.9, 263.9]；n=5 | 262.6 [225.8, 293.6]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.12 [0.00, 0.26]；n=5 |
| C | all_ABCD | 逐帧抽样 | NA；n=0 | 134.9 [CI不可估]；n=1 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | single_task | 期望点 | 282.9 [224.2, 341.6]；n=5 | 228.1 [208.2, 247.9]；n=5 | 323.6 [267.1, 378.9]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | single_task | argmax | 229.8 [203.2, 256.3]；n=5 | 247.4 [218.8, 284.7]；n=5 | 282.6 [240.8, 326.5]；n=5 | 0.23 [0.05, 0.42]；n=5 | 0.11 [0.00, 0.34]；n=5 |
| C | single_task | 逐帧抽样 | NA；n=0 | 129.0 [CI不可估]；n=1 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | leave_out | 期望点 | 244.8 [163.9, 370.0]；n=5 | 261.7 [223.3, 306.1]；n=4 | 279.7 [259.7, 299.7]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | leave_out | argmax | 222.0 [188.1, 264.0]；n=5 | 240.8 [201.3, 281.1]；n=5 | 224.3 [196.6, 265.0]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.11 [0.00, 0.34]；n=5 |
| C | leave_out | 逐帧抽样 | NA；n=0 | 137.0 [CI不可估]；n=1 | 150.5 [CI不可估]；n=1 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | SPD fine-tune | 期望点 | 278.4 [218.9, 345.4]；n=5 | 223.7 [208.5, 242.0]；n=5 | 250.8 [215.7, 287.4]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | SPD fine-tune | argmax | 250.0 [225.4, 277.3]；n=5 | 239.7 [233.3, 245.7]；n=5 | 259.0 [226.3, 306.7]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.16 [0.00, 0.39]；n=5 |
| C | SPD fine-tune | 逐帧抽样 | NA；n=0 | NA；n=0 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | SPD zero-shot | 期望点 | 262.7 [210.6, 341.9]；n=5 | 252.8 [197.2, 298.4]；n=5 | 267.2 [223.7, 305.0]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.11 [0.00, 0.34]；n=5 |
| C | SPD zero-shot | argmax | 215.5 [193.0, 235.2]；n=5 | 221.5 [206.3, 236.7]；n=5 | 208.4 [182.9, 232.9]；n=5 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |
| C | SPD zero-shot | 逐帧抽样 | NA；n=0 | 138.4 [137.4, 139.4]；n=2 | NA；n=0 | 0.00 [0.00, 0.00]；n=5 | 0.00 [0.00, 0.00]；n=5 |

逐帧抽样的随机变化会直接影响驻留和切换。K列来自逐帧独立潜变量/类别抽样，没有跨时间的联合轨迹身份；固定某一K列也不能将其变成联合scanpath。短驻留或高切换可能是解码随机性，不可直接归因于模型学到了人的扫视行为。概率MC分支不报告驻留/切换。

| 任务 | 来源 | 解码器 | T截断/保留段 | P截断/保留段 | O截断/保留段 |
| --- | --- | --- | --- | --- | --- |
| A | all_ABCD | 期望点 | 30 / 154 | 43 / 178 | 3 / 25 |
| A | 真实 gaze | 真实观察 | 110 / 453 | 49 / 218 | 11 / 66 |
| A | leave_out | 期望点 | 32 / 124 | 24 / 100 | 1 / 25 |
| A | single_task | 期望点 | 54 / 249 | 19 / 120 | 2 / 47 |
| A | SPD fine-tune | 期望点 | 29 / 168 | 34 / 127 | 4 / 50 |
| A | SPD zero-shot | 期望点 | 25 / 85 | 5 / 26 | 0 / 0 |
| C | all_ABCD | 期望点 | 11 / 126 | 16 / 152 | 21 / 154 |
| C | 真实 gaze | 真实观察 | 10 / 190 | 31 / 169 | 27 / 168 |
| C | leave_out | 期望点 | 10 / 85 | 10 / 137 | 8 / 51 |
| C | single_task | 期望点 | 24 / 132 | 21 / 145 | 27 / 161 |
| C | SPD fine-tune | 期望点 | 13 / 138 | 13 / 149 | 22 / 146 |
| C | SPD zero-shot | 期望点 | 12 / 178 | 15 / 111 | 12 / 120 |
| A | all_ABCD | argmax | 106 / 383 | 60 / 310 | 17 / 99 |
| A | leave_out | argmax | 66 / 236 | 35 / 174 | 7 / 57 |
| A | single_task | argmax | 99 / 385 | 61 / 325 | 10 / 83 |
| A | SPD fine-tune | argmax | 92 / 402 | 63 / 302 | 13 / 111 |
| A | SPD zero-shot | argmax | 78 / 279 | 12 / 93 | 4 / 21 |
| C | all_ABCD | argmax | 22 / 275 | 26 / 225 | 31 / 265 |
| C | leave_out | argmax | 9 / 127 | 17 / 188 | 26 / 137 |
| C | single_task | argmax | 22 / 253 | 30 / 211 | 38 / 263 |
| C | SPD fine-tune | argmax | 29 / 271 | 23 / 219 | 28 / 247 |
| C | SPD zero-shot | argmax | 12 / 188 | 28 / 256 | 19 / 173 |
| A | all_ABCD | 逐帧抽样 | 0 / 0 | 0 / 0 | 0 / 0 |
| A | leave_out | 逐帧抽样 | 0 / 0 | 0 / 0 | 0 / 0 |
| A | single_task | 逐帧抽样 | 0 / 0 | 0 / 1 | 0 / 0 |
| A | SPD fine-tune | 逐帧抽样 | 0 / 0 | 0 / 0 | 0 / 0 |
| A | SPD zero-shot | 逐帧抽样 | 0 / 0 | 0 / 0 | 0 / 0 |
| C | all_ABCD | 逐帧抽样 | 0 / 0 | 0 / 1 | 0 / 0 |
| C | leave_out | 逐帧抽样 | 0 / 0 | 0 / 1 | 0 / 1 |
| C | single_task | 逐帧抽样 | 0 / 0 | 0 / 1 | 0 / 0 |
| C | SPD fine-tune | 逐帧抽样 | 0 / 0 | 0 / 0 | 0 / 0 |
| C | SPD zero-shot | 逐帧抽样 | 0 / 0 | 0 / 4 | 0 / 0 |

## 13. 七种预设条件的敏感性

主定义保持50 px；25、100 px及单独扩大PSM用于判断几何依赖。严格3/旧123改变眼动质量支持，平滑分支改变位置与可填补的短缺口，因此它们不仅改变数值，也可能改变进入比较的帧和实际人数。没有把敏感性条件当作新的最优参数搜索。

| 任务 | 集合 | 来源 | 主分析 50/50, code 2/3 C% | 半径 25/25 C% | 半径 100/100 C% | 半径 50/100 C% | 严格 code 3 C% | 旧规则 code 1/2/3 C% | 插值及三样本平滑 C% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | real_all | 真实 gaze | 76.8 [70.3, 82.9]；n=11 | 79.3 [72.4, 86.1]；n=11 | 70.4 [64.1, 76.1]；n=11 | 40.4 [31.7, 49.1]；n=11 | 76.8 [69.1, 84.1]；n=11 | 76.7 [70.2, 82.8]；n=11 | 76.7 [70.2, 82.7]；n=11 |
| A | paired_test | 真实 gaze | 75.0 [65.4, 84.1]；n=5 | 81.1 [69.1, 91.2]；n=5 | 67.0 [58.8, 74.5]；n=5 | 36.8 [27.0, 46.6]；n=5 | 71.8 [54.9, 88.8]；n=5 | 75.0 [65.4, 84.1]；n=5 | 74.8 [65.1, 83.8]；n=5 |
| A | paired_test | all_ABCD | 47.2 [41.4, 52.6]；n=5 | 47.1 [42.0, 53.1]；n=5 | 43.7 [35.8, 50.9]；n=5 | 15.4 [10.6, 20.0]；n=5 | 49.0 [34.8, 64.9]；n=5 | 47.1 [41.4, 52.6]；n=5 | 47.1 [41.4, 52.6]；n=5 |
| A | paired_test | single_task | 48.1 [41.7, 54.5]；n=5 | 47.9 [41.5, 55.2]；n=5 | 44.6 [36.0, 52.8]；n=5 | 16.1 [10.8, 21.5]；n=5 | 45.9 [37.2, 55.0]；n=5 | 48.1 [41.7, 54.5]；n=5 | 48.1 [41.7, 54.5]；n=5 |
| A | paired_test | leave_out | 50.6 [44.0, 56.6]；n=5 | 49.6 [44.2, 55.0]；n=5 | 48.0 [38.5, 56.6]；n=5 | 16.8 [10.9, 22.1]；n=5 | 46.7 [28.0, 69.6]；n=5 | 50.6 [44.0, 56.5]；n=5 | 50.6 [44.0, 56.6]；n=5 |
| A | paired_test | SPD fine-tune | 47.9 [41.6, 53.4]；n=5 | 47.8 [42.1, 53.9]；n=5 | 44.4 [36.0, 51.8]；n=5 | 15.8 [10.6, 20.4]；n=5 | 48.0 [37.3, 58.8]；n=5 | 47.8 [41.6, 53.4]；n=5 | 47.8 [41.6, 53.4]；n=5 |
| A | paired_test | SPD zero-shot | 58.1 [49.8, 65.0]；n=5 | 57.8 [51.0, 63.8]；n=5 | 54.8 [44.1, 63.6]；n=5 | 21.9 [14.0, 28.9]；n=5 | 53.5 [36.4, 73.9]；n=5 | 58.1 [49.8, 65.0]；n=5 | 58.1 [49.8, 65.0]；n=5 |
| C | real_all | 真实 gaze | 51.4 [44.2, 59.3]；n=11 | 53.2 [46.7, 60.1]；n=11 | 43.8 [35.9, 51.6]；n=11 | 11.4 [7.8, 15.1]；n=11 | 52.2 [41.9, 62.8]；n=11 | 51.4 [44.2, 59.3]；n=11 | 51.3 [43.9, 59.3]；n=11 |
| C | paired_test | 真实 gaze | 57.2 [34.7, 71.3]；n=5 | 52.2 [38.5, 63.3]；n=5 | 43.8 [26.3, 59.8]；n=5 | 10.9 [3.4, 18.8]；n=5 | 54.9 [27.2, 74.6]；n=5 | 57.2 [34.7, 71.3]；n=5 | 57.1 [34.5, 71.1]；n=5 |
| C | paired_test | all_ABCD | 41.2 [32.4, 50.1]；n=5 | 44.8 [37.2, 52.1]；n=5 | 34.2 [23.2, 45.6]；n=5 | 5.4 [2.6, 9.0]；n=5 | 36.5 [26.0, 46.0]；n=5 | 41.2 [32.3, 50.1]；n=5 | 41.2 [32.4, 50.1]；n=5 |
| C | paired_test | single_task | 42.0 [33.3, 51.3]；n=5 | 45.3 [37.7, 52.7]；n=5 | 36.2 [25.1, 48.7]；n=5 | 5.9 [2.7, 10.2]；n=5 | 37.6 [26.1, 48.9]；n=5 | 42.0 [33.3, 51.3]；n=5 | 42.0 [33.3, 51.3]；n=5 |
| C | paired_test | leave_out | 34.9 [29.9, 40.8]；n=5 | 39.5 [35.6, 44.2]；n=5 | 28.7 [21.2, 36.5]；n=5 | 4.1 [1.9, 6.5]；n=5 | 28.1 [21.1, 35.0]；n=5 | 34.9 [29.9, 40.8]；n=5 | 34.9 [29.9, 40.8]；n=5 |
| C | paired_test | SPD fine-tune | 41.6 [33.3, 50.2]；n=5 | 45.2 [37.8, 52.4]；n=5 | 34.5 [24.2, 45.7]；n=5 | 5.5 [2.8, 9.1]；n=5 | 37.0 [26.2, 47.1]；n=5 | 41.5 [33.3, 50.2]；n=5 | 41.5 [33.3, 50.2]；n=5 |
| C | paired_test | SPD zero-shot | 32.2 [27.8, 35.9]；n=5 | 37.4 [33.7, 40.5]；n=5 | 23.3 [17.7, 28.6]；n=5 | 3.0 [1.3, 4.7]；n=5 | 26.2 [20.5, 31.5]；n=5 | 32.2 [27.8, 35.9]；n=5 | 32.2 [27.8, 35.9]；n=5 |

| 任务 | 条件 | 真实有效帧 | 真实有效秒 | 配对有效帧 | 配对有效秒 | 真实有效人数 |
| --- | --- | --- | --- | --- | --- | --- |
| A | main | 367203 | 8614.7 | 58649 | 1250.9 | 11 |
| A | r25 | 367203 | 8614.7 | 58649 | 1250.9 | 11 |
| A | r100 | 367203 | 8614.7 | 58649 | 1250.9 | 11 |
| A | gripper100 | 367203 | 8614.7 | 58649 | 1250.9 | 11 |
| A | gaze3 | 118838 | 2675.1 | 20308 | 455.3 | 11 |
| A | legacy123 | 369366 | 8668.1 | 58893 | 1256.0 | 11 |
| A | smooth | 369982 | 8682.7 | 59005 | 1258.7 | 11 |
| C | main | 336886 | 8022.8 | 36186 | 835.8 | 11 |
| C | r25 | 336886 | 8022.8 | 36186 | 835.8 | 11 |
| C | r100 | 336886 | 8022.8 | 36186 | 835.8 | 11 |
| C | gripper100 | 336886 | 8022.8 | 36186 | 835.8 | 11 |
| C | gaze3 | 134502 | 3116.5 | 16406 | 358.0 | 11 |
| C | legacy123 | 338323 | 8057.6 | 36267 | 837.8 | 11 |
| C | smooth | 338698 | 8066.1 | 36278 | 837.8 | 11 |

相关性方向敏感性：当前找到 55 个任务/阶段/来源/指标组合在至少一个敏感性条件下与主分析ρ异号。以下只标记方向不稳定，不把数值接近0的正负翻转解释成实质逆转。

| 集合 | 任务 | 阶段 | 来源 | 指标 | 异号条件 | 主ρ | 敏感性n范围 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paired_test | A | grab ring | all_ABCD | C | gaze3、gripper100、r100 | -0.200 | 5–5 |
| paired_test | A | grab ring | all_ABCD | L_target | r100、r25 | 0.100 | 5–5 |
| paired_test | A | grab ring | gaze | C | gaze3 | 0.100 | 4–4 |
| paired_test | A | grab ring | leave_out | C | r25 | 0.100 | 5–5 |
| paired_test | A | grab ring | single_task | C | gaze3 | 0.200 | 5–5 |
| paired_test | A | grab ring | single_task | L_gripper | gaze3、r25 | -0.200 | 5–5 |
| paired_test | A | grab ring | spd_finetune | C | gaze3、gripper100、r100 | -0.100 | 5–5 |
| paired_test | A | grab ring | spd_finetune | L_gripper | r25 | -0.200 | 5–5 |
| paired_test | A | grab ring | spd_zero_shot | L_target | gaze3 | -0.600 | 5–5 |
| paired_test | A | place ring | all_ABCD | L_target | gaze3 | -0.300 | 4–4 |
| paired_test | A | place ring | gaze | L_target | gaze3、r100 | 0.300 | 4–5 |
| paired_test | A | place ring | leave_out | C | gaze3 | 0.500 | 4–4 |
| paired_test | A | place ring | leave_out | L_gripper | gaze3 | -0.700 | 4–4 |
| paired_test | A | place ring | leave_out | L_target | gaze3 | -0.300 | 4–4 |
| paired_test | A | place ring | single_task | L_target | gaze3、r25 | -0.100 | 4–5 |
| paired_test | A | place ring | spd_finetune | L_gripper | gaze3 | -0.500 | 4–4 |
| paired_test | A | place ring | spd_finetune | L_target | gaze3 | -0.100 | 4–4 |
| paired_test | A | place ring | spd_zero_shot | C | gripper100 | -0.200 | 5–5 |
| paired_test | A | place ring | spd_zero_shot | L_gripper | gaze3 | -0.500 | 4–4 |
| paired_test | A | place ring | spd_zero_shot | L_target | gaze3、r100 | -0.300 | 4–5 |
| paired_test | A | reach ring | all_ABCD | C | gaze3、gripper100 | -0.200 | 5–5 |
| paired_test | A | reach ring | gaze | C | gaze3、gripper100、r100 | -0.500 | 4–5 |
| paired_test | A | reach ring | single_task | C | gripper100、r100 | -0.100 | 5–5 |
| paired_test | A | reach ring | spd_zero_shot | C | gaze3 | -0.600 | 5–5 |
| paired_test | A | reach ring | spd_zero_shot | L_target | gaze3 | -0.500 | 5–5 |
| paired_test | A | transfer ring | all_ABCD | C | gaze3、gripper100 | -0.100 | 5–5 |
| paired_test | A | transfer ring | gaze | L_target | r25 | -0.400 | 5–5 |
| paired_test | A | whole | gaze | L_target | r25 | -0.500 | 5–5 |
| paired_test | A | whole | single_task | C | gaze3 | 0.100 | 5–5 |
| paired_test | A | whole | spd_finetune | C | gaze3 | 0.100 | 5–5 |
| paired_test | C | reach string | gaze | C | r25 | 0.707 | 5–5 |
| paired_test | C | reach string | single_task | C | gaze3 | -0.354 | 5–5 |
| paired_test | C | reach string | spd_finetune | C | gaze3 | -0.354 | 5–5 |
| paired_test | C | wrap around psm | all_ABCD | L_gripper | gaze3、gripper100、r100 | -0.354 | 5–5 |
| paired_test | C | wrap around psm | leave_out | C | gripper100 | -0.354 | 5–5 |
| paired_test | C | wrap around psm | spd_finetune | C | r25 | 0.354 | 5–5 |
| real_all | A | grab ring | gaze | C | gripper100、r25 | -0.173 | 11–11 |
| real_all | A | grab ring | gaze | L_gripper | gaze3 | 0.045 | 11–11 |
| real_all | A | grab ring | gaze | L_target | gaze3 | 0.600 | 11–11 |
| real_all | A | place ring | gaze | C | gaze3 | 0.100 | 11–11 |
| real_all | A | place ring | gaze | L_gripper | gaze3、r25、smooth | 0.073 | 11–11 |
| real_all | A | place ring | gaze | L_target | r100 | -0.064 | 11–11 |
| real_all | A | reach ring | gaze | C | gaze3 | 0.027 | 11–11 |
| real_all | C | pull through and cinch | gaze | C | r25、smooth | -0.203 | 6–6 |
| real_all | C | pull through and cinch | gaze | L_gripper | gripper100、r100 | -0.145 | 6–6 |
| real_all | C | reach string | gaze | C | gripper100、r100、smooth | 0.009 | 11–11 |
| real_all | C | reach string | gaze | L_target | gaze3 | 0.358 | 11–11 |
| real_all | C | threading string through tower | gaze | C | r100 | -0.405 | 8–8 |
| real_all | C | threading string through tower | gaze | L_gripper | gripper100、r100 | 0.078 | 8–8 |
| real_all | C | whole | gaze | C | gaze3 | -0.174 | 11–11 |
| real_all | C | whole | gaze | L_gripper | gaze3 | 0.220 | 11–11 |
| real_all | C | whole | gaze | L_target | gaze3 | 0.156 | 11–11 |
| real_all | C | wrap around psm | gaze | C | gaze3、gripper100 | -0.266 | 11–11 |
| real_all | C | wrap around psm | gaze | L_gripper | gaze3 | 0.294 | 11–11 |
| real_all | C | wrap around psm | gaze | L_target | gaze3 | 0.193 | 11–11 |

## 14. repetition 1→3 的探索性变化

round 按参与者/任务内不同 TaskNum 的先后排序，随后才应用内容筛选；它不是 Attempt 数，也不能保证所有人都有第3轮。表内为第3轮−第1轮，在两轮指标均可用的同一参与者间配对，比例差单位为百分点。练习相关变化仍可能混有任务时长、阶段构成和质量覆盖变化。

| 任务 | 集合 | 来源 | ΔC pp | Δtarget pp | Δ腕部 pp | ΔO pp | Δscore |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | real_all | 真实 gaze | 1.8 [-7.7, 10.0]；n=11；录像11→11 | -1.3 [-6.7, 3.8]；n=11；录像11→11 | -1.5 [-5.0, 2.6]；n=11；录像11→11 | -0.5 [-2.2, 1.0]；n=11；录像11→11 | 38.5 [24.8, 51.3]；n=11；录像11→11 |
| A | paired_test | 真实 gaze | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| A | paired_test | all_ABCD | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| A | paired_test | single_task | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| A | paired_test | leave_out | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| A | paired_test | SPD fine-tune | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| A | paired_test | SPD zero-shot | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| C | real_all | 真实 gaze | -22.7 [-34.0, -11.4]；n=11；录像18→15 | -8.0 [-14.2, -2.5]；n=11；录像18→15 | 5.4 [-3.8, 14.8]；n=11；录像18→15 | -2.0 [-8.0, 3.4]；n=11；录像18→15 | 22.2 [8.1, 40.0]；n=11；录像18→14 |
| C | paired_test | 真实 gaze | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| C | paired_test | all_ABCD | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| C | paired_test | single_task | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| C | paired_test | leave_out | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| C | paired_test | SPD fine-tune | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |
| C | paired_test | SPD zero-shot | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 | NA；n=0；录像0→0 |

Task C真实全量1→3：target占用变化使用 11 人、两端 18→15 份录像；score变化使用 11 人、两端 18→14 份录像。占用与成绩的试次支持不同；有可用占用但无有效成绩的录像能进入占用变化，不能进入score变化。即使参与者数相同，也不能称为同trial支持。

## 15. 可复核性、图表和结论边界

本次读取 50 个预测文件，K取值=[64]，点抽样seed=[42]；无效概率图共 0 个，无效样本点共 0 个。预测文件实际SHA在每个新进程内至少验证一次；旧期望点只在预测SHA、帧号、时间和分辨率匹配后复用，argmax/采样独立解码。

仍有 40/50 个预测文件的导出侧 checkpoint SHA未记录。文件hash可固定现有预测，但不能补造当时缺失的checkpoint/代码版本溯源。

| 本次验证项 | 记录值 |
| --- | --- |
| occupancy_conservation | True |
| area_conservation | True |
| NA_denominator_check | True |
| paired_frame_support_identical_across_branches | True |
| whole_equals_stage_exposures | True |
| recordings_read | 84 |
| predictions_aligned | 50 |
| recording_issues | 0 |
| drop_ring_included_duration_s | 0.0 |
| geometry_self_tests | pixel enumeration and Monte Carlo toy cases passed |

输入问题记录数：0。完整审计见 [input_issues.json](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/input_issues.json>)。

解释本次结果时应同时满足三层条件：有可用帧与足够参与者；该指标分母和CI确实可估；目标与腕部区域的语义足以支持所讨论的行为问题。不能用预测概率更集中、C更高、或某一阶段ρ更大替代这三层检查。当前最稳妥的行为表述是“已标注目标邻域与DLC腕部邻域之间的空间分配”，而非完整器械注视、jaw-tip fixation或专家策略的直接证明。

### 现有图表入口

- [Task A 主占用](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_A_regions.png>)
- [Task A 真实–模型C](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_A_agreement_C.png>)
- [Task A C–score](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_A_performance_C.png>)
- [Task A target–score](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_A_performance_L_target.png>)
- [Task A 腕部–score](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_A_performance_L_gripper.png>)
- [Task A Brier/BSS](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_A_brier_skill.png>)
- [Task A 敏感性](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_A_sensitivity.png>)
- [Task A 期望点驻留/切换](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/decoded_expectation/figures/paired_test/task_A_dwell_switch.png>)
- [Task A argmax驻留/切换](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/decoded_argmax/figures/paired_test/task_A_dwell_switch.png>)
- [Task A 逐帧抽样驻留/切换](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/decoded_sample/figures/paired_test/task_A_dwell_switch.png>)
- [Task C 主占用](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_C_regions.png>)
- [Task C 真实–模型C](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_C_agreement_C.png>)
- [Task C C–score](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_C_performance_C.png>)
- [Task C target–score](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_C_performance_L_target.png>)
- [Task C 腕部–score](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_C_performance_L_gripper.png>)
- [Task C Brier/BSS](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_C_brier_skill.png>)
- [Task C 敏感性](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/figures/paired_test/task_C_sensitivity.png>)
- [Task C 期望点驻留/切换](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/decoded_expectation/figures/paired_test/task_C_dwell_switch.png>)
- [Task C argmax驻留/切换](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/decoded_argmax/figures/paired_test/task_C_dwell_switch.png>)
- [Task C 逐帧抽样驻留/切换](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/decoded_sample/figures/paired_test/task_C_dwell_switch.png>)

### 原始表入口

- [coverage.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/coverage.csv>)
- [recording_audit.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/recording_audit.csv>)
- [prediction_audit.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/prediction_audit.csv>)
- [main_probability/group_stage_summary.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/group_stage_summary.csv>)
- [main_probability/model_error_summary.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/model_error_summary.csv>)
- [main_probability/model_paired_contrasts.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/model_paired_contrasts.csv>)
- [main_probability/performance_correlations.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/performance_correlations.csv>)
- [main_probability/performance_participants.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/performance_participants.csv>)
- [main_probability/repetition_contrasts.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/main_probability/repetition_contrasts.csv>)
- [baseline_input_audit.json](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/baseline_input_audit.json>)
- [region_review/occurrence_geometry_flags.csv](<D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T063658759189Z_v2_full/region_review/occurrence_geometry_flags.csv>)
