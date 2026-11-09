# Passive–Active 来源视频配对比较

打开 [report.html](report.html) 查看中文离线报告。所有图形在 figures/ 中，同时提供 PNG 与 PDF。

按用户指定规则，以 passive_gaze_links.json 的来源映射及原始 active/passive gaze NPY 行数相等确定配对，第 i 行对应第 i 行。时间戳不作为配对条件，本次不新增时间戳重对齐。原始行数见 row_correspondence_audit.csv。

以 gt_passive16 的 16 条记录为起点，从 gt84 提取对应 active 单记录结果。配对单位是来源视频，不是同一人的两种实验状态。主分析直接复用原有逐行对应结果，观看者等权，active/passive 使用同一完整支持配对集合。差值统一为 Passive − Active。

41 Hz 仅沿用原分析数值定义；主窗口 ±2 秒、固定左右器械、两种信号及两种阶段模式保持不变。5000 次观看者同步重采样的百分位范围仅为探索性描述，未处理全部参与者交叉依赖；不报告正式 p/FDR 结论。

附录共同可用坐标样本敏感性仅为可选描述，文件保留 common_time_primary 技术名称。它处理坐标有限有效性，不改变来源映射和逐行对应关系。

- [16 对记录映射](paired_record_mapping.csv)
- [原始 NPY 行数与逐行对应核查](row_correspondence_audit.csv)
- [预定条件完整覆盖](paired_condition_coverage.csv)
- [逐记录配对指标](paired_metrics.csv)
- [主分析及坐标可用性补充汇总](paired_group_summary.csv)
- [组曲线与差值](paired_group_curves.csv)
- [组曲线峰位比较](group_peak_comparison.csv)
- [零 lag 空间配对](paired_spatial_metrics.csv)
- [配置和权重敏感性](sensitivity_summary.csv)
- [逐身份剔除敏感性](loo_sensitivity.csv)
- [附录：共同可用坐标补充指标](common_time_record_summary.csv)
- [附录：共同可用坐标配对指标](common_time_paired_metrics.csv)
- [附录：共同可用坐标空间配对](common_time_paired_spatial_metrics.csv)
- [原网格与坐标有效性审计](support_audit.csv)
- [坐标有效性审计说明](support_audit.json)
- [重采样分布](bootstrap_distributions.npz)
- [运行与输入清单](comparison_manifest.json)
- [验证结果](validation.json)
- [独立验证](independent_validation.json)
- [复现说明](REPRODUCE.md)
- [重采样数组索引](bootstrap_index.json)
- [提取的 Active 单记录结果](active_matched_record_summary.csv)
- [提取的 Active 完整曲线](active_matched_record_curves.jsonl)
