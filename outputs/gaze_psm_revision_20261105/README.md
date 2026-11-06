# Gaze / PSM 修正结果 · 2026-11-05

从 [index.html](index.html) 进入两份离线报告。

## 数据范围

- [84 录制人工 gaze/PSM](gt84/report.html)：84 个录制、11 个参与者，A=33、C=51。
- [10 录制 GT 与五模型](models10/report.html)：共有10个录制、7名参与者，每task5人；不是84个录制的模型推断。

## 本次最终确认的规则

- 主分析 fixed_left / fixed_right 分别计算同一物理器械。
- 补充 gt_selected_side：每个录制×stage×signal×mode由GT选一侧，所有模型使用这一侧。
- independent_max_side_legacy：GT和模型分别选最大侧，只作旧版对照，保留same_side标记。
- DLC全部有限坐标保留；likelihood、图像外点、孤立跳变候选只作QC，不删除、不加权。0.2/0.5/0.8仅为QC标记，不是本次过滤分支。
- Position从未平滑网格数据做3点居中平均；1/5点作为对照。
- Velocity从未平滑网格数据独立做二阶SG5后差分；SG7/11及旧box3为对照。
- 最大插值间隙0.1s，41Hz；逐点记录原生/重采样/缺口/滤波边缘及时间轴。
- gaze_raw本来就是41Hz插值缓存，本轮未更改高频原始眼动Status筛选或原始插值。
- 组汇总为完整lag录制在user内等权，再user等权；没有把多次录制当独立参与者。
- 主分析固定侧的user bootstrap和unpaired reference各5000次；无效参考不补抽，数量和原因独立记录。
- 选侧补充仅作描述，未为选择统计量生成新的参考p值；不沿用固定侧p值。
- 区间和参考尾部比例是探索性描述，不构成因果或精确生理时延证明。

## 验证

- GT84旧单录制曲线回归：2328条，最大误差1.74e-13。
- 旧max组曲线回归：36组，全部通过=True。
- GT84输入未改变：True；全部原生重建审计通过：True。
- Model10验证详情：[validation.json](models10/validation.json)。
- [最终独立核验](verification/final_validation.json)；[84缓存预处理核验](verification/preprocess_cache_audit.json)。

## 复现

每个子目录的code_snapshot保存数值计算代码和参数指纹。原输入绝对路径记录在input_manifest.json；交付目录搬迁不改变原始来源记录。

GT84在有NumPy、SciPy、Matplotlib的Python环境执行：

```text
python gt84/code_snapshot/run_gt84_revision.py --source D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_dlc --output NEW_EMPTY_DIRECTORY --bootstrap 5000 --null-count 5000 --workers 4 --no-report
python gt84/report_code/report_gt84_revision.py NEW_EMPTY_DIRECTORY
```

Model10使用其code_snapshot/analyze_model_xcorr_dlc_v2.py，参数见run_manifest.json。必须指定新空输出目录。

原结果与原模型预测均未覆盖。artifact_manifest.json覆盖交付文件，delivery_validation.json另记录复制后核验。
