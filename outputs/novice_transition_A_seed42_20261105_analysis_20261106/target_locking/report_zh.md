# Task A：新模型与GT的目标锁定比较

本目录已按用户要求原位重算，不备份旧结果。GT采用坐标有效规则，valid仅作诊断。主分析为**共同锚点时间**上的热图区域概率；未来时间匹配作为补充。样本为5段验证录像、4名参与者，User5贡献2段。该验证集参与checkpoint选择，因此结果属于探索性验证，不能写作未见测试集表现。

主分析共同有效锚点共 **24,120** 个，时间权重总计 **1234.02秒**。每录像先按真实时间加权，同一参与者录像等权，再对参与者等权；95% CI来自10,000次参与者bootstrap。

|模型|GT目标占用 %|模型目标占用 %|模型−GT pp [95% CI]|目标Brier|PSM Brier|
|---|---:|---:|---:|---:|---:|
|current_transition|26.47|6.78|-19.70 [-24.08, -12.88]|0.2227|0.0690|
|future500_transition|26.47|7.73|-18.75 [-22.96, -11.99]|0.2186|0.0687|
|future1000_transition|26.47|5.44|-21.04 [-25.24, -14.20]|0.2319|0.0706|
|future1500_transition|26.47|4.96|-21.51 [-25.92, -14.41]|0.2342|0.0709|

区域定义：target为有效标注槽位均值周围50 px圆盘，PSM为左右DLC腕部/铰接点50 px圆盘并集。T/P/O/N分别表示目标独占、PSM独占、重叠、其他。L_target=T+O；L_psm=P+O；C=T/(T+P)，分母0时NA；D=T+P。C表示独占区域之间的分配，须结合D与绝对占用阅读。

热图积分采用原tools的`map_region_probabilities`：保存的32×40网格值视为格内概率质量，在网格对应像素区域内均匀分布，按圆盘交集像素面积积分。没有生成额外随机样本。模型概率与GT二元区域事件的时间加权Brier越小越好；它不直接衡量完整注视分布是否正确。

GT仅保留两个归一化坐标有限且位于闭区间[0,1]的点，不依据valid状态筛选；25/100 px半径为敏感性。原gaze3状态子集已移除，不再报告状态筛选结果。主PSM只要求双侧DLC坐标有限且位于画面内，不添加与旧target-locking不一致的隐含likelihood阈值。drop ring、none、空阶段排除。

期望点/argmax轨迹用于补充区域占用、Brier与≥120 ms驻留/切换。概率主分支不定义唯一轨迹，不能据此计算驻留。驻留按采样观测时长，阶段、缺失、>100 ms间隔或非正常stride2间隔断开；接触观测边界的run计为删失并记录。

原生`times()`要求连续原始帧，直接输入stride2会得到零时长；本次保留其相邻时间中点思想，显式允许source anchor差2，并禁止跨stage occurrence与缺帧间隔。未来时间轴按源时间加horizon计时，以35 ms容差内最近原生帧匹配GT/target/PSM；没有尾部钳位和motion registration。

对未来模型，锚点分析比较“当前可展示提示”与当前GT/target/PSM；未来时间分析比较训练horizon对应的GT/target/PSM。后者每个horizon有自己的有效支持，因此不把其不同支持的占用均值当作模型直接配对比较。

训练事件频率/训练平均热图没有在本次拟合。表中保留几何面积与中心点Brier作为诊断参照，不能解释为超过强训练基线。

统计表：trial_metrics.csv、trial_errors.csv、participant_metrics.csv、group_metrics.csv、participant_errors.csv、group_errors.csv、leave_one_participant_out.csv、model_paired_contrasts.csv。逐帧可复核结果在frames/；覆盖与删失在coverage.csv及trial_metrics.csv；独立验证见validation.json。

图件：figures/tl_anchor_occupancy、tl_anchor_paired_differences、tl_stage_heatmap、tl_decoding_brier、tl_target_time_secondary（PNG与PDF）。

限制：本分析空间区域不是完整物体或夹爪尖端mask；target槽位均值保留原标注语义。四参与者的小样本不支持强泛化；提供逐人点与逐人剔除表。新热图积分结果不能直接替换旧MC样本占用数值。硬件曝光与显示延迟没有独立校准。
