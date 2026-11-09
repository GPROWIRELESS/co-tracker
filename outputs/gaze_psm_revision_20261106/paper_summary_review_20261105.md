# 论文摘要复核记录

本文件名保留原版 `paper_summary_review_20261105.md`，内容更新于 2026-11-06。

本次 GT 坐标来自 `D:/Dataset/dvss/Active_Data/User*/npy/<record>_gaze.npy`；仅保留两个坐标均有限且均在归一化闭区间 `[0,1]` 内的样本，完全不依据 `valid` 或原始眼动 CSV 的 `Status` 筛选。按视频帧号使用同一 User 的 `csv/<record>_video_ts.csv` 对齐时间，该 CSV 仅提供时间，不提供 GT gaze 坐标。归一化坐标按原视频尺寸换算为像素后，以原规则重建 41 Hz 网格。2026-11-05 目录属于历史 CSV-GT 口径结果，不能继续作为本次 GT 基准。

已将两篇论文摘要、详细报告、数据与图表全部切换到新一轮结果。完整阶段与论文限定五阶段分别汇总，模型统计不沿用旧 p 值。

独立复核 60 个参与者等权均值，最大差 2.78e-17；140 个精确检验的 p 值、效应和 Holm 校正最大差分别为 0、2.08e-17、0。

原版各文件名和图表名保留，以便既有引用继续使用；文件名日期不代表内容仍为旧值。旧输出目录保留，修正输入后数值变化是预期现象，不作为旧值一致性回归的失败。未将宽峰判定、方向判定或研究阶段改成其他任务的新规则。

[复核明细](paper_summary_data/independent_summary_audit.json) · [来源哈希](paper_summary_data/reanalysis_manifest.json)
