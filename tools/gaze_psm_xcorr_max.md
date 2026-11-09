# 按 stage 选择最大 xcorr 的 PSM

在仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/gaze_psm_xcorr_max.py --workers 8
```

默认读取 `outputs/gaze_psm_xcorr` 的 `results.json`、`config.json`、
`quality.json` 和轨迹缓存，写入独立的 `outputs/gaze_psm_xcorr_max`。
不修改源目录，不重新解码视频，也不需要访问原始数据盘。

每个 `content_id / task / stage / signal / mode` 比较左右曲线在完整 lag
窗口内的最大**有符号**相关系数。选中一侧后，保留其整条曲线及其空间指标。
不会在不同 lag 上切换左右侧，也不会先汇总 user 再选侧。
position、velocity、PSM 时间锚定的 stage 和 Both in stage 分别选侧。
whole-task 不纳入这份分析，也不拼接不同 stage。

只有一侧有有限相关值时选择该侧；两侧均无有限值时保留缺失结果，
`selected_side` 为空；峰值完全相等时选择左侧并标记 `tie_left`。
缺失结果使用左侧数据作为确定性的占位模板，不认定左侧获胜。
`side=max` 是供统计分组使用的合并条件，实际获胜器械见 `selected_side`。

录制曲线在各 user 内按录制等权 Fisher-z 平均，再对 user 等权平均。
总体峰值从总体曲线寻找，不平均各录制的峰值或 lag。保留原有完整 lag
支持度和最少 user 数规则；不足支持的结果保持缺失。
默认进行 1000 次 user bootstrap，以及 100 次同 task、不同 user 的
unpaired reference。每次随机对照重新计算左右曲线并重新选侧，再进行相同的
分层汇总；不沿用真实数据的获胜侧或旧对照统计。

主要输出：

- `side_selection.csv`：每条录制/stage 的左右峰值、对应 lag、获胜侧和选择状态。
- `record_*`、`novice_*`、`group_*` CSV：选侧后的录制、user 和总体统计。
- `bootstrap_peaks.csv`、`unpaired_peaks.csv`：重新计算的 bootstrap 和对照结果。
- `contrasts.csv`、`bootstrap_contrasts.csv`：同 task 的 stage 间比较。
- `results.json`：完整数值；`config.json`：分析参数和源目录。
- `figures/stages_<task>_<signal>.png/pdf`：总体曲线、置信带、随机对照和峰值。
- `figures/stage_peak_lags_<task>_<signal>.png/pdf`：stage 峰值 lag 和 95% 区间。
- `figures/figure_data.json`：图中的峰值、区间、user 数和状态，供核对。
- `validation.json`：选侧数量和源文件 SHA-256，确认源目录文件未修改。

正 lag 表示 gaze 领先 PSM。选侧结果表示相关性更强的一侧，不能直接认定为
实际操作侧。stage 置信区间及 stage 对比沿用原分析的探索性解释。

可通过 `--source`、`--output` 指定其他独立目录；`--bootstrap`、`--shuffles`
可调整抽样次数，`--no-plots` 只生成数值。`--shuffles 0` 时不读取缓存，
不计算对照；这类诊断运行请使用另外的输出目录，避免覆盖完整结果。

验证：

```bash
python -m unittest tests.test_gaze_psm_xcorr_max tests.test_gaze_psm_group -q
```
