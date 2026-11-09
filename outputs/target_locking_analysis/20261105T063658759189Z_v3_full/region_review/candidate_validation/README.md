# 区域定位审计：腕部主定义与 jawtip 候选

本轮没有完成可靠的自动 jawtip 修正。主分析应明确使用 **DLC 腕部点为中心、半径 50 px 的区域**。现有腕点加 PSM mask 可以产生候选，但候选未通过代表帧目视验收，不能把它称为已验证的 jawtip，也不能悄悄替换所有区域。

## 已核验的文件与范围

- 原记录来自 `D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T033217114891Z_full/frames` 的全部 84 个记录。
- `D:/Dataset/dvss/Active_Data/<User>/sam/<content_id>/mask/` 下左右 mask `obj_psm_l.mp4`、`obj_psm_r.mp4` 均存在；84/84 记录的左右 mask 帧数和尺寸元数据与保存的 native 数组一致。
- `obj_psm_tip.mp4` 也在 84/84 记录存在且元数据一致，但抽查图像显示它是两侧远端金属部分的组合分割，包含腕部和夹爪，并非 jawtip 点坐标。
- 元数据一致不等于逐帧解码成功或语义、时序全部正确。本轮没有解码全部 84 个记录的每一帧；多阶段样例用源视频叠加左右 mask 检查了对应关系。
- 原来的 User1 任务 A、C 各四个阶段的代表帧，加上每阶段另一个记录，共 16 帧、32 侧候选。选择规则是阶段内主支持有效帧的中间帧，再选择字母顺序第一个另有该阶段的记录；这是便利抽查，不是随机验证集。

## 候选规则与验收结论

候选首先选取靠近 DLC 腕点的足够大连通分量，使用主轴、腕点相对分量中心的方向、端部形态及距离进行检查，再取窄端帽的平均位置。方向不明确、形状不够细长、端帽太宽或分叉、长度异常、腕点附近没有分量时拒绝。它不是简单取距腕点最远的像素，但这些几何规则仍不能处理所有关节姿态、图像边界和遮挡。

32 侧中有 **19 侧通过几何规则，13 侧被拒绝**，只有 5/16 帧左右都通过。19 个放行结果中，目视发现 4 个明显取了近端位置，4 个仍有开放夹爪、交叉或遮挡歧义，11 个与可见远端大致相符。这里的目视记录由助手完成，没有独立人工精确 jawtip 标注，**不能把这些数量当作正式定位误差率、像素误差或泛化准确率**；大致相符也不是精确验收合格。

| 样例 | 侧 | 明显失败现象 |
|---|---|---|
| 03，User10_1_1，grab ring，帧 9191 | 左 | 候选落在左图像边缘近端，开放夹爪指向右侧 |
| 08，User1_5_2，reach string，帧 2299 | 左 | 候选落在腕部下缘，可见夹爪指向右上 |
| 10，User1_5_2，wrap，帧 3020 | 左 | 候选落在近端下缘，可见夹爪指向右上 |
| 13，User10_2_1，cinch，帧 9161 | 右 | 候选落在右边缘近端，可见夹爪指向左上 |

此外，样例 12 的模拟器提示工具离开视野，DLC 点位于底部界面；坐标在图像内本身不保证真实工具可见。样例 05、07 的右侧可见工具附近缺少对应分割，说明“文件存在”不能代替逐帧 mask 质量审核。

因此候选保留为 **探索性诊断实现**，不接入新版主区域，也不生成伪装成有效 jawtip 的完整轨迹。没有用若干失败例子重新调参后宣称验收通过。时间稳定性和独立坐标误差也尚未验证。若继续推进，需要先定义开放夹爪时的规范点（单侧端点还是两尖端中点），在覆盖各阶段、关节角度、边界、遮挡及不可见状态的独立帧上标注，并同时验证误差与可用率；通过后训练与测试必须使用同一规则。

## Target 保留与语义标记

所有 target 坐标保持当前有效点的平均中心，未做坐标 override。`occurrence_geometry_flags.csv` 覆盖 2,368 个非空、非 drop ring 的 occurrence。它报告有效 target 帧数、多点相对平均中心的分散程度、有效 slot 变化、中心距腕点小于 50 px 的比例及原始 label mode。

这些是几何描述，**不能独自证明漂移、目标对象身份或行为意义**。`tracked_inherited` 等 mode 不用于推断 target 是环、peg、线或工具。没有来源明确的对象身份协议时，`target_role` 保持 `unverified_from_annotation_protocol`；标记不自动排除 occurrence，也不改变 target 坐标。`count_changes` 为有效 slot 位掩码变化次数，并非仅仅目标点数量变化。

## 可程序消费的接口

`tl_regions.py` 的 `propose_regions(d, data_root, repo=None, audit_dir=None)` 接收完整 native 顺序的 `content_id`、`frames`、`width`、`height`、`left`、`right`。

返回 `left_xy` / `right_xy`（N×2）、`valid`（N，左右都通过几何规则）、`confidence`（N×2，未校准几何分数）、`reason`（N×2）、`source`、`raw_left_xy` / `raw_right_xy`，以及 `eligible_as_primary=False` 和明确失败的 `validation_status`。被拒绝的位置保留 NaN，**从不自动回退到 wrist**。这里 `valid` 仅表示候选规则通过，不表示定位已被人工验证。训练和测试不存在不同分支。

`inventory()` 兼容 `slotmask` / `target_slotmask`、`label_mode` / `target_label_mode`；缺少 mode 时明确记为 unavailable，不猜测。脚本默认输出到 `<repo>/outputs/target_locking_region_review/<UTC_run>/`，可用 `--out` 指定审计目录。模块的六项软件自检通过，覆盖空 mask、无效腕点、方向不明、合成镜像、缺失 mask 无回退等；这些自检不构成 jawtip 准确性验证。

## 文件说明

- `mask_inventory.csv`：84 记录的文件、尺寸、帧数和元数据状态。
- `inventory_summary.json`：库存概要及尚未全帧解码的限制。
- `occurrence_geometry_flags.csv`：逐 occurrence 的可审计几何、来源及未确认语义标记。
- `candidate_samples.csv`：32 个侧别候选的坐标、几何指标和拒绝原因。
- `visual_review.csv`：每个侧别的目视判断、依据及无参考真值声明。
- `validation_summary.json`：机器可读验收失败结论及 19/32 的准确计数。
- `sample_00_*.jpg` 至 `sample_15_*.jpg`：原帧、mask 和候选叠加；绿色为腕点，紫色为放行候选，红色为拒绝候选，黄色为保留的 target。
- `contact_0.jpg` 至 `contact_3.jpg`：16 个代表帧的缩略概览；`mask_format_preview.jpg` 展示左右与组合 mask 的格式。

上述审计全部写在当前 workspace，未修改 D: 源数据或既有分析结果。主分析保留腕部定义是当前自动定位未通过验证的限制，不能表述为已经完成 tip 修正。
