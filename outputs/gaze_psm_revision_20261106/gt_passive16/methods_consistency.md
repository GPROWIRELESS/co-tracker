# GT84 与 passive 方法一致性核对

数值核采用冻结版本 gt84-fixed-v2.1-npy-20261106。输入按用户最终指定的逐行对应规则处理，保留原始 VID_FRAME 差异和历史构造审计。

| 方法项 | GT84规则 | passive实现 | 代码依据 | 验证结果 | 必要差异 |
|---|---|---|---|---|---|
| 冻结版本与入口 | gt84-fixed-v2.1-npy-20261106 的实际代码与 manifest | 仅输入、身份、路径、历史不适用文件和报告适配 | run_gt84_revision.py; baseline_audit.json | 通过 | outputs/tools 不存在，使用冻结 code_snapshot |
| 分析单位 | active recording，按 user 汇总和重抽样 | 每次 passive 观看唯一 recording_id，user=observer_user | run_gt84_revision.record_job; records.json | 通过 | 身份改为真实观看者 |
| 样本与任务 | A/C，预定 A5+C4 阶段 | novice–novice A7+C9，共16观看、8观看者、16源视频 | input_record_inventory.csv; stage_coverage.csv | 通过 | B/D不套用A/C阶段，User2_1_1查实为B |
| 画面与坐标 | 归一化 gaze 乘对应源图像尺寸 | 用户确认视频归一化1280×1024，左右DLC均在同一源图像 | user_input_confirmations.json; decoded_evidence.json | 通过 | 显示坐标定义由采集者确认 |
| 帧与行对应 | 原始 Frame_Number 对齐精确video_ts，DLC按保留视频局部行号 | 按用户指定row i对应video frame i和DLC row i，且逐项帧号核对一致 | input_mapping/*.npz; npy_video_frame_count_audit.csv | 按用户指定通过 | 原VID_FRAME差异保留，用户明确采用按行对应 |
| 时间轴 | 精确video_ts原始帧键，非NPY近似时间或名义FPS | 精确passive/source video_ts逐值一致，按源采集时钟秒数分析 | prepare_passive_inputs.timeline; input_mapping | 通过 | 不把结果称为已校准的实际播放生理时延 |
| gaze有效性 | 两个坐标有限且均在闭区间[0,1]，valid/Status仅审计 | 从passive NPY读取同规则，保留原生valid和历史clamp来源 | prepare_passive_inputs.py; point_provenance | 通过 | 历史NPY已有clamp/最近邻填充，本轮直接使用所给NPY |
| PSM有效性 | 保留全部有限二维点，likelihood/图外/跳变只标记 | 质量规则删除数0，不加权 | gaze_psm_preprocess_revision.prepare_signals | 通过 | 无 |
| 网格与插值 | 41Hz，最大间隙0.1秒，不外推，稳定排序重复时间取首 | 原样复用冻结interpolate_valid与网格规则 | gaze_psm_preprocess_revision.interpolate_valid | 通过 | 无 |
| Position滤波 | 有限连续段内3点居中boxcar，原边缘/短段规则 | 冻结filter_valid逐字相同 | gaze_psm_preprocess_revision.filter_valid | 通过 | 无 |
| Velocity滤波 | 未经position平滑的输入独立SG5二阶后差分×41，有符号二维速度 | 原样复用，速度中点赋stage/occurrence | prepare_signals; _velocity_labels | 通过 | 无 |
| 配对与lag方向 | gaze(t−τ)对PSM(t)，主±2秒165点 | 原样复用，正τ表示gaze轨迹对应较早 | gaze_psm_reliability_v2.occurrence_statistics | 通过 | 无 |
| 阶段与occurrence | PSM in stage只限制PSM，Both in stage两端同一occurrence | source真实连续occurrence按原帧映射，within跨界0 | occurrence_statistics; independent validation | 通过 | 缺失drop ring保留覆盖状态 |
| 二维相关估计 | 每lag每occurrence按有效配对中心化，汇总协方差及平方和后归一化 | 冻结FFT充分统计量与聚合，独立显式配对核复核全部440primary | occurrence_statistics; aggregate_statistics | 通过 | 独立最大误差8.13e-14 |
| 支持阈值 | occurrence至少2对，总至少82对，双方二维RMS≥1，分母非退化 | 实际min_overlap=2秒，min_rms=1，未使用底层其他默认值 | run_gt84_revision.BASE; _from_totals | 通过 | 无 |
| 峰与可靠性 | signed maximum，原并列处理，0.01/0.05/0.10近峰与0.05/0.10/0.20方向容差 | 原样复用全部峰形/LOO/半段/边界及稳定性诊断 | peak_features; reliability | 通过 | 无 |
| 8个variant | primary、position1/5、velocitybox3/SG7/SG11、trim_edges、window4 | 配置与冻结manifest逐值一致，window4为329lag | run_manifest.variants | 通过 | 无 |
| 组聚合 | 全lag完整录制固定，录制在user内等权Fisher-z，user间等权，裁剪±(1−1e−7)，至少3人 | user为观看者，所有384主/选择侧组独立核对 | gaze_psm_statistics_v2.group_curves | 通过 | 无 |
| 组bootstrap | 5000次整user有放回，SHA256派生seed42 | 逐lag范围、边界频率及全部variant峰分布保存并复核 | group_curves; group_bootstrap_full_validation.json | 通过 | 补存全部variant分布，数值与既有表相同 |
| occurrence bootstrap | primary500/敏感性200，非循环相邻块长度2，precision0.4秒，登记/贡献至少4 | 全部2420条录制条件核对 | reliability; block_weights | 通过 | 无 |
| GT选择侧 | primary逐录制stage/signal/mode选左右signed peak，平局左，保留完整赢家曲线 | 原样运行gt_selected_side | run_gt84_revision.selected_curves | 通过 | 无 |
| 共同mode支持 | 两种mode共同全lag完整录制重算 | 原样运行matched_mode_* | run_gt84_revision.matched_modes | 通过 | 无 |
| 阶段差值 | 共同user，同步5000bootstrap，B−A，至少3人，保留边界 | 观看者共同支持，72个可用差值分布独立复核 | gt_supplement.supplement | 通过 | 无 |
| 聚合支持对照 | 相同信号固定器械，对照available/complete记录支持 | 原样运行 | gt_supplement.supplement; gaze_psm_group.aggregate | 通过 | 无 |
| 空间分析 | primary position完整录制，零lag及组峰lag，距离中位数与25/50/100px，录制/观看者等权 | 原样运行，Both in stage跨occurrence配对为0 | spatial_and_boundaries; independent validation | 通过 | 无 |
| 非配对参考 | 同task不同user，先等概率user再录制，elapsed不伸缩不循环不外推 | 不同观看者，source相同仍可选，固定5000尝试失败不补抽 | gt_reference.donor_choices; run_reference | 通过 | 纳入16源视频互异，实际同source donor为0 |
| duration子集与BH | 子集重算观察值，seed31415，A5+C4 family含缺失，尾部(1+计数)/(1+有效次数) | 两套参考、128条件与16个BH family独立重建 | gt_reference._bh_descriptive; run_reference | 通过 | 无 |
| legacy回归 | Active历史CSV对照文件存在 | 两个同名文件明确不适用，无passive历史基线 | run_gt84_revision.regression | 不适用 | 未读取Active历史结果作为passive回归 |
| 报告图表 | 13种图，每种PNG/PDF | 同样13图，观看者支持、全预定阶段、缺失/边界标记 | report_code/report_passive_revision.py | 通过，13组PNG/PDF和离线链接已核对 | 报告文字按passive实际数值重写 |
