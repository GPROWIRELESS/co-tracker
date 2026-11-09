# Gaze/PSM 固定左右器械修正结果

打开 [report.html](report.html) 阅读中文离线报告。图位于 `figures/`，每张均含 PNG 与 PDF。

本目录：84 个录制、11 个 user；fixed_left / fixed_right 为主分析。GT 选择侧为补充，不命名为操作侧。

## 已确认规则

- DLC 全部有限坐标保留；likelihood、图像外点和跳变只标记，不删除、不加权。
- position：3 点居中移动平均；velocity：从未平滑输入独立做二阶 SG5 后差分。SG7/11 等为敏感性。
- GT 来自各 User 的 npy/*_gaze.npy 中 gaze_x / gaze_y；仅接受两坐标有限且均在 [0,1]（包含边界）的点，不根据 valid 筛选。原始 NPY、视频时间戳和接受点数见 run_manifest.json 的 input_policy。
- 41 Hz、最大插值间隙 0.1 秒；不外推、不跨长缺口滤波。本轮 gaze_raw 为按上述新 GT 规则重新生成的缓存。
- 全 lag 有效录制固定；录制在 user 内等权、user 在总体等权。共同支持 mode 对照单独保存。
- 重抽样范围为探索性百分位范围；参考尾部比例比较相关峰高，不是 lag 正向或因果检验。

## 主要文件

- [group_summary.csv](group_summary.csv)：各分支组峰位、峰高、范围、覆盖
- [group_curves.csv](group_curves.csv)：完整组曲线和逐 lag 重抽样范围
- [record_summary.csv](record_summary.csv)：逐录制峰形与 occurrence 重抽样诊断
- [record_curves.jsonl](record_curves.jsonl)：逐录制完整曲线
- [preprocessing_qc.csv](preprocessing_qc.csv)：原生 QC、插补、滤波边缘汇总
- [support_audit.csv](support_audit.csv)：时间轴和原生重建审计
- [side_selection.csv](side_selection.csv)：GT 选择侧补充规则及赢家
- [legacy_curve_regression.csv](legacy_curve_regression.csv)：旧 CSV-GT 与新 NPY-GT 单录制曲线差异审计（不要求相等）
- [legacy_max_group_regression.csv](legacy_max_group_regression.csv)：沿用旧 max 规则和旧聚合后的 GT 来源差异审计（不要求相等）
- [aggregation_support_comparison.csv](aggregation_support_comparison.csv)：相同信号下仅改变录制支持规则的对照
- [stage_contrasts.csv](stage_contrasts.csv)：共同 user 的阶段差值诊断
- [group_bootstrap_index.json](group_bootstrap_index.json)：组峰位重抽样数组索引
- [group_bootstrap_peak_lags.npz](group_bootstrap_peak_lags.npz)：完整组峰位重抽样分布
- [run_manifest.json](run_manifest.json)：参数、软件版本与代码指纹
- [input_manifest.json](input_manifest.json)：输入文件指纹
- [validation.json](validation.json)：运行验证

- `matched_mode_*`：共同录制支持下的两种 mode 比较。
- `spatial_*`：零 lag / 组峰位处的距离、25/50/100 px 接近比例及跨边界配对比例。
- `unpaired_reference.csv`：若运行参考分支，记录请求/有效/失败次数、失败原因和参考尾部比例。
- `point_provenance/<variant>/<content_id>.npz`：逐点原生 QC、插补、滤波边缘及时间轴。
- `occurrence_statistics/`、`bootstrap_peaks/`：逐记录充分统计量和重抽样分布。
- `code_snapshot/`：本次执行源码；指纹见 run_manifest。

## 验证

输入未改变：True；同一新输入的独立计算核验证：True；独立核最大绝对误差：0.0。

历史 CSV-GT 对照相等：False；对照曲线数：2328；最大绝对相关差值：0.7008216126078326。此次 GT 来源修正会导致差异；历史数值相等不是通过条件。

与 20261105 交付保留相同录像、分析参数、stage、支持规则和输出文件结构；此次仅改变 GT 来源和有效性条件。与更早旧聚合的差异仍在 aggregation_support_comparison.csv 单独保存。
