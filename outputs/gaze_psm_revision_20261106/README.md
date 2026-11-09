# Gaze PSM 重算结果 2026 11 06

从 [index.html](index.html) 进入报告。旧 `gaze_psm_revision_20261105` 结果保留，本目录为独立重算版本。

本次 GT 坐标来自 `D:/Dataset/dvss/Active_Data/User*/npy/<record>_gaze.npy`；仅保留两个坐标均有限且均在归一化闭区间 `[0,1]` 内的样本，完全不依据 `valid` 或原始眼动 CSV 的 `Status` 筛选。按视频帧号使用同一 User 的 `csv/<record>_video_ts.csv` 对齐时间，该 CSV 仅提供时间，不提供 GT gaze 坐标。归一化坐标按原视频尺寸换算为像素后，以原规则重建 41 Hz 网格。2026-11-05 目录属于历史 CSV-GT 口径结果，不能继续作为本次 GT 基准。

GT84 原生 874,630 点，坐标规则保留 818,734 点、排除 55,896 点；保留点包含 stored valid=1 的 4,086 点、valid=2 的 508,994 点及 valid=3 的 305,654 点。valid 只做诊断，不影响保留。

## 数据范围

- [GT84 中文报告](gt84_analysis_report.md)：84 条录制、11 名参与者，A=33、C=51。
- [Models10 中文报告](models10_vs_gt_report.md)：10 条相同录制的 GT 与五模型，7 名参与者，A/C 各5人。
- [GT 论文摘要](gt84_paper_summary.md) 和 [模型论文摘要](models10_vs_gt_paper_summary.md)：沿用原论文五阶段范围。
- [逐录制 GT 模型对照](models10_vs_gt10.html)；[GT84 完整报告](gt84/report.html)；[Models10 完整报告](models10/report.html)。

## 本次变更与可复现性

只更换 GT gaze 源及有效性筛选；录制、PSM、阶段、模型预测、滤波、时窗、汇总与随机参数沿用原版。GT 原始坐标、视频时间及输入哈希分别记录于各分支输入清单与 verification。原始数据和旧输出均不覆盖。

主分析固定左右物理器械，使用 41 Hz 网格、最大 0.1 s 插值间隙、±2 s 搜索窗（165 格）。不外推或跨长缺口插值。位置由未平滑坐标做 3 点居中平均；速度独立做二阶 SG5 平滑后相邻差分，时间取中点。原有不平滑/5 点位置、SG7/SG11/旧 box3 速度、裁边及 ±4 s 敏感性分支均保留。

配对为 `gaze(t−τ)` 和 `PSM(t)`，因此正 τ 表示较早的 gaze 对应当前 PSM。每个 occurrence、每个 lag 分别去均值，再汇总二维协方差与平方和求相关；不是 x/y Pearson 系数的平均。总有效配对至少 82 点，双方中心化二维 RMS 至少 1，且分母非退化。PSM 全部有限坐标保留；likelihood、跳变及图外点仅作质量标记。

`PSM in stage` 仅要求 PSM 属于该阶段 occurrence；`Both in stage` 要求两端均属于同一次 occurrence。组曲线仅纳入整个 lag 窗完整的录制，先在参与者内等权平均 Fisher z，再在参与者间等权，最后反变换。组峰位是组曲线最大值的位置，不是录制峰位的平均。分组至少 3 名参与者。

原版峰形、近峰宽度、方向及可定位规则保持不变。`positive_peak` 指正相关峰，不代表正 lag；非零峰也不自动满足可定位规则。重采样范围是探索性百分位范围，尚未校准为生理时延的置信区间。边界峰、宽峰和不稳定峰应结合完整曲线解读；正峰位不构成因果或固定预视时长的证明。

算法和数值参数沿用原版，运行库环境不同：旧 Python 3.11.5/NumPy 1.24.3，本次 Python 3.12.12/NumPy 1.26.4；依赖版本按本次实际环境记录于各分支 requirements 文件，不能将其称为运行环境完全一致。

模型缓存 `qc_json.gaze_source_npy` 是同录像 GT 参照路径，**模型 gaze 本身仍来自原预测 NPZ**，实际来源见 `prediction_manifest.csv` 与工作目录 `input_policy_audit.json` 的 `models` 记录。输入缓存目录 `config.json` 保留旧独立最大侧、bootstrap 等历史缓存生成参数，仅用于来源/历史回归审计，不驱动本轮计算；本轮真实参数以 [GT运行清单](gt84/run_manifest.json) 和 [模型运行清单](models10/run_manifest.json) 为准。缓存中的旧 `results.json` 同样只用于历史差异审计，不能当作本次结果。

模型主表及 380/370 配对统计仅对应 primary + position + PSM in stage + 固定左右侧，并保留全部已观测阶段；论文模型摘要在相同筛选下进一步限定五阶段，为250/250配对。GT84主分支2328曲线同时包含 position/velocity 与 PSM in stage/Both in stage，不能把这几个数量视为同一范围。

### 输入重建和只读审计

在交付根目录、启用本次 sam3 Python 环境后执行以下命令。默认模式只读审计现有输入，不重建或改写输入缓存；核验结果会更新到 verification 中。

```text
python verification/audit_preprocess_revision.py
```

需要从每个 User 的 NPY gaze、原帧 video_ts 时间和既有 PSM/模型来源重建独立输入缓存，再执行同一核验时，使用：

```text
python verification/audit_preprocess_revision.py --rebuild-inputs
```

重建输出位于脚本记录的独立 `gaze_psm_revision_20261106_work/inputs/gt84` 和 `inputs/models10`，不覆盖原始数据或旧结果目录。脚本所用原始输入、缓存位置及数值运行路径见各分支 [GT输入清单](gt84/input_manifest.json)、[GT运行清单](gt84/run_manifest.json)、[模型输入清单](models10/input_manifest.json) 和 [模型运行清单](models10/run_manifest.json)；原始数据和记录的 work 路径需保持可访问。该命令重建并核验输入，不会自动重新生成全部分析结果。

## 验证文件

[最终独立验证](verification/final_validation.json) · [预处理核验](verification/preprocess_cache_audit.json) · [GT 与模型输入对齐](verification/gt_model_input_alignment.json) · [论文汇总独立审计](paper_summary_data/independent_summary_audit.json) · [交付核验](delivery_validation.json)。`code_snapshot` 保存本次使用的计算代码；不能对原先错误 GT 结果宣称数值回归一致。
