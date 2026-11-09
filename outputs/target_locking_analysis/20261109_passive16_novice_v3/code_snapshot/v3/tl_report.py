"""Build the detailed Chinese narrative solely from one result directory.

No model fitting, inference, new participant bootstrap or legacy-result numbers
are performed here. Published intervals come from this run's statistics tables.
"""
from __future__ import annotations

import ast
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = {"A": "Sea Spikes", "C": "Knot-tying"}
STAGES = {
    "A": ["reach ring", "grab ring", "transfer ring", "place ring"],
    "C": ["reach string", "wrap around psm", "pull through and cinch", "threading string through tower"],
}
LABELS = {"gaze": "真实 gaze", "all_ABCD": "all_ABCD", "single_task": "single_task",
          "leave_out": "leave_out", "spd_finetune": "SPD fine-tune", "spd_zero_shot": "SPD zero-shot"}
DECODERS = {"decoded_expectation": "期望点", "decoded_argmax": "argmax", "decoded_sample": "逐帧抽样"}
SCENARIOS = {"main": "主分析 50/50，valid不限", "r25": "半径 25/25", "r100": "半径 100/100",
             "gripper100": "半径 50/100",
             "smooth": "插值及三样本平滑"}
REFS = {"train_heatmap": "训练平均热图", "center_point": "画面中心点", "uniform_area": "均匀面积", "train_rate": "训练事件频率"}


def _read(path):
    if not path.is_file() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else ({} if default is None else default)


def _finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except (ValueError, TypeError):
        return False


def _num(x, digits=2):
    return f"{float(x):.{digits}f}" if _finite(x) else "NA"


def _int(x):
    return str(int(x)) if _finite(x) else "NA"


def _select(data, **filters):
    if data.empty:
        return data
    mask = np.ones(len(data), dtype=bool)
    for key, value in filters.items():
        if key not in data:
            return data.iloc[:0]
        if value is None:
            mask &= data[key].isna()
        elif isinstance(value, (list, tuple, set)):
            mask &= data[key].isin(value)
        else:
            mask &= data[key].eq(value)
    return data.loc[mask]


def _cell(row, column="mean", scale=100, digits=1, show_n=True, n_column="n_participants"):
    if row is None:
        return "未提供"
    n = f"；n={_int(row.get(n_column))}" if show_n else ""
    if not _finite(row.get(column)):
        return "NA"+n
    value = _num(float(row[column])*scale,digits)
    if _finite(row.get("ci_low")) and _finite(row.get("ci_high")):
        return f"{value} [{_num(float(row['ci_low'])*scale,digits)}, {_num(float(row['ci_high'])*scale,digits)}]"+n
    return value+" [CI不可估]"+n


def _direction(row):
    if row is None or not _finite(row.get("ci_low")) or not _finite(row.get("ci_high")):
        return "区间不可估"
    if row["ci_low"] > 0:
        return "区间全为正"
    if row["ci_high"] < 0:
        return "区间全为负"
    return "区间包含0"


def _table(headers, rows):
    rows = list(rows)
    if not rows:
        return "当前结果中没有可报告的行。\n"
    def escape(v):
        return str(v).replace("|", "／").replace("\n", " ")
    lines = ["| "+" | ".join(map(escape,headers))+" |", "| "+" | ".join(["---"]*len(headers))+" |"]
    lines.extend("| "+" | ".join(map(escape,row))+" |" for row in rows)
    return "\n".join(lines)+"\n"


def _link(path, label=None):
    return f"[{label or path.name}](<{path.resolve().as_posix()}>)"


class Run:
    def __init__(self,out):
        self.out = Path(out).resolve()
        self.config = _json(self.out/"run_config.json")
        self.validation = _json(self.out/"validation.json")
        self.tables = {}
        self.indices = {}

    def table(self,name,branch=None):
        key = (branch,name)
        if key not in self.tables:
            self.tables[key] = _read(self.out/(branch or "")/name)
        return self.tables[key]

    def row(self,name,task,stage,source,metric,*,scope="paired_test",scenario="main",branch="main_probability",reference=None):
        key = (branch,name)
        if key not in self.indices:
            index = {}
            for row in self.table(name,branch).to_dict("records"):
                k = tuple(row.get(c) for c in ("scope","scenario","task","stage","source","metric"))
                ref = row.get("reference")
                k += (None if pd.isna(ref) else ref,)
                index[k] = row
            self.indices[key] = index
        return self.indices[key].get((scope,scenario,task,stage,source,metric,reference))

    def occ(self,task,stage,source,metric,**kwargs):
        return self.row("group_stage_summary.csv",task,stage,source,metric,**kwargs)

    def error(self,task,stage,source,metric,**kwargs):
        return self.row("model_error_summary.csv",task,stage,source,metric,**kwargs)


def _findings(run,lines,models):
    """A few mechanically supported findings, without inferring a winning model."""
    lines += ["## 当前数据的主要发现", ""]
    for task in TASKS:
        gt=run.occ(task,"whole","gaze","L_target",scope="real_all")
        gp=run.occ(task,"whole","gaze","L_gripper",scope="real_all")
        gc=run.occ(task,"whole","gaze","C",scope="real_all")
        if gt and gp and gc:
            lines += [f"**Task {task}（{TASKS[task]}）**：真实全量 target占用（%）为 {_cell(gt)}，腕部占用（%）为 {_cell(gp)}，C（%）为 {_cell(gc)}。占用的分母是实际有效观察时间，C的分母是两类exclusive占用之和；模型误差另用配对测试集合。",""]
        delta=[run.error(task,"whole",m,"delta_L_target") for m in models]
        delta=[r for r in delta if r and _finite(r.get("estimate"))]
        losses=[(m,run.error(task,"whole",m,"BS_target")) for m in models]
        losses=[(m,r) for m,r in losses if r and _finite(r.get("estimate"))]
        if delta:
            lower=sum(_direction(r)=="区间全为负" for r in delta)
            upper=sum(_direction(r)=="区间全为正" for r in delta)
            remaining=len(delta)-lower-upper
            lines += [f"该任务五模型中的可估 target占用差为 {_num(100*min(r['estimate'] for r in delta),1)} 至 {_num(100*max(r['estimate'] for r in delta),1)} 个百分点；{lower} 个模型的差值CI全为负，{upper} 个全为正，{remaining} 个跨0或区间不可估。",""]
        if losses:
            model,row=min(losses,key=lambda x:x[1]["estimate"])
            lines += [f"target Brier 的最低点估计来自 {LABELS.get(model,model)}：{_cell(row,'estimate',1,3)}。这是点估计排序；是否支持优于另一模型需阅读第9节的配对差CI。",""]
        for metric,label in [("C","C"),("L_target","target占用"),("L_gripper","腕部占用")]:
            row=run.row("performance_correlations.csv",task,"whole","gaze",metric,scope="real_all")
            if row:
                lines += [f"真实全量 {label}与SimNow成绩的ρ={_cell(row,'rho',1,3)}；{_direction(row)}。",""]
    lines += ["以下表格保留相同任务的阶段、概率/点解码、覆盖和敏感性差异。高C或更强成绩相关都不能单独替代预测准确性检验。",""]


def _coverage_section(run, lines):
    c = _select(run.table("coverage.csv"),scenario="main")
    a = run.table("recording_audit.csv")
    lines += ["## 1. 分析对象、实际覆盖和计量单位", "",
              f"本报告只读取本次 v{_int(run.config.get('version'))} 目录中的结果。主分析是保存的 K 个未平滑预测样本的区域概率估计；期望点、argmax 和逐帧抽样分别是独立的点解码分支。两类结果不能混称。",
              "", f"运行目录：{_link(run.out,run.out.name)}。统计 bootstrap 配置为 {_int(run.config.get('bootstrap_n'))} 次，seed={_int(run.config.get('bootstrap_seed'))}。", ""]
    rows=[]
    for task in TASKS:
        raw = _select(a,task=task)
        whole = _select(c,task=task,stage="whole")
        if whole.empty:
            continue
        good=whole[whole.valid_n.gt(0)]
        paired=whole[whole.paired_n.gt(0)]
        rows.append([f"{task} {TASKS[task]}",len(raw),raw.participant.nunique() if len(raw) else 0,
                     len(good),good.participant.nunique(),_int(good.valid_n.sum()),_num(good.valid_s.sum(),1),
                     len(paired),paired.participant.nunique(),_int(paired.paired_n.sum())])
    lines += [_table(["任务","读取录像","读取人数","真实有效录像","真实有效人数","真实有效帧","真实有效秒","配对录像","配对人数","配对帧"],rows),
              "`real_all` 用本次选择范围内所有可用真实 gaze；`paired_test` 只用有五模型预测、且五模型及三种解码共同有效的测试录像/帧。若本次是 subset，上表仅代表该 subset。不能把真实全量人数当成模型测试人数，也不能把两个集合的均值直接作模型误差。",
              "", "每录像/阶段先按原生时间间隔加权；同一参与者的有效录像再等权平均，组水平对参与者等权平均。`whole` 是纳入阶段的时间合并结果，不是四个阶段均值。表中比例通常按百分比显示，差值用百分点；Brier、BSS 和 Spearman ρ 保持原值。方括号是参与者 bootstrap 95% CI，n 是该具体指标的实际人数。", ""]
    rows=[]
    for task in TASKS:
        for stage in STAGES[task]:
            g=_select(c,task=task,stage=stage)
            if g.empty:continue
            total=float(g.total_n.sum()); good=g[g.valid_n.gt(0)]
            rows.append([task,stage,len(g),good.participant.nunique(),_int(total),
                         _num(100*g.target_n.sum()/total if total else np.nan,1),
                         _num(100*g.gaze_n.sum()/total if total else np.nan,1),
                         _num(100*g.dlc_geometry_n.sum()/total if total else np.nan,1),
                         _num(100*g.valid_n.sum()/total if total else np.nan,1),_int(g.paired_n.sum())])
    lines += [_table(["任务","阶段","录像","有有效帧人数","阶段帧","有target %","有gaze %","双侧DLC坐标 %","共同有效 %","配对有效帧"],rows),
              "这些覆盖比例分别是边际比例，不能相乘得到交集。DLC 坐标可用表示坐标通过几何检查，不等于其语义是 jaw tip，也不等于已验证定位精度。所有列出的任务阶段都进入统计框架；目标或 gaze 缺失的帧仍不能计算。", ""]
    cinch=_select(c,task="C",stage="pull through and cinch")
    if len(cinch) and cinch.total_n.sum()>0:
        frac=100*cinch.valid_n.sum()/cinch.total_n.sum()
        lines += [f"Cinch 的当前共同有效帧覆盖为 **{frac:.1f}%**。{'覆盖偏低，' if frac<50 else ''}其估计描述可观测部分；不能把缺失时段视作零占用，也不能据此概括完整 cinch 行为。",""]
    if len(a) and "included_drop_ring_s" in a:
        lines += [f"源数据中额外 `drop ring` 标签累计 {_num(a.get('native_drop_ring_s',pd.Series(dtype=float)).sum(),3)} s；本次纳入 {_num(a.included_drop_ring_s.sum(),3)} s。它不属于本次预定的八类阶段，避免把额外标签无声并入 RR/GR/TR/PR。",""]


def _definitions_and_quality(run,lines):
    cfg=run.config
    lines += ["## 2. 主指标的准确含义", "",
              f"主半径保持 target={_int(cfg.get('target_radius_px',50))} px、PSM={_int(cfg.get('psm_radius_px',50))} px。当前 target 是已有有效标注槽位的算术均值周围的一个圆盘；它不是所有槽位圆盘的并集。该选择沿用当前明确的分析配置，并未通过本报告改写目标语义。PSM 区域为左右 DLC 点周围圆盘的并集，交叠只计一次。",
              "", "四种互斥状态为 T=target-only、P=PSM-only、O=两者重叠、N=other。`L_target=L_T+L_O`，`L_gripper=L_P+L_O`，`D=L_T+L_P`，`C=L_T/(L_T+L_P)`。C 描述可分辨区域内偏向 target 的比例；D 描述有多少时间/概率落入可分辨区域。D=0 时 C 为 NA，绝不能填成0；高 C 也不能单独等同于高 target 占用。",
              "", "真实 gaze 每帧是单点事件。主模型每帧 q 是 K 个保存样本中落入相应区域的比例，再按原生时间加权求占用。这是有限 K 的 Monte Carlo 概率估计，不是把热图期望点硬分类，也不是本次重新生成的平滑热图。有限 K 会带来采样误差；此处参与者 CI 不单独量化模型重新采样的不确定性。",
              "", "面积基线 a 是区域在画面中的面积比例。E_target=L_target−a_target，E_gripper=L_gripper−a_gripper；C_U=a_T/(a_T+a_P)，C_excess=C−C_U。面积修正控制的是 ROI 大小，不会自动修正器械遮挡、目标语义或追踪误差。", "",
              "## 3. 坐标有效性、时间轴和区域语义审计", "",
              "真实 gaze 直接读取 `UserN/npy/<ID>_gaze.npy` 的结构化字段 `gaze_x`、`gaze_y`，按原视频 `Frame_Number` 匹配。原始 gaze 的可用条件仅为两个坐标均有限且均位于闭区间 [0,1]；端点0和1保留，NaN、无穷值及越界坐标排除。`valid` 不设任何取值范围，也不参与纳入判定。坐标合格后仍须满足既定阶段、target和双侧DLC条件，模型比较还须满足五模型共同帧规则。", ""]
    a=run.table("recording_audit.csv")
    if len(a):
        compared=a.get("raw_left_status_compared_n",pd.Series(dtype=float)).sum()
        matched=a.get("raw_left_status_match_n",pd.Series(dtype=float)).sum()
        native=a.retained_frames.sum()
        lines += [f"本次读取 {_int(native)} 个保留视频帧；其中 {_int(compared)} 个可在规定近邻窗口内核对原始 Left Status，{_int(matched)} 个与 NPY valid 相等。该核对是状态来源证据，不是独立的眼动精度校验。", ""]
        rows=[]
        for task in TASKS:
            q=_select(a,task=task)
            if len(q):
                rows.append([task,_int(q.retained_frames.sum()),_int(q.gaze_coordinate_valid_n.sum()),
                             _int((q.retained_frames-q.gaze_coordinate_valid_n).sum()),
                             _int(q.gaze_valid_n.sum())])
        lines += [_table(["任务","保留帧","坐标有限且在[0,1]内","坐标不合格","进入后续条件筛选的gaze帧"],rows)]
    lines += ["上述gaze帧数尚未与阶段、目标和DLC条件取交集，因此不等于第1节最终有效帧数。NPY `valid` 及原始设备状态仅保留用于来源审计；状态码的解释、分布或匹配结果均不会重新筛掉坐标合格的帧。旧版按状态码筛选的条件不属于本次分析分支。", ""]
    if len(a):
        grid=[]
        for width,height in a[["width","height"]].drop_duplicates().itertuples(index=False,name=None):
            grid.append(f"{int(width)}×{int(height)}画面上的节点间距约为{width/39:.2f}×{height/31:.2f} px")
        lines += ["预测样本采用40×32含端点网格，"+"；".join(grid)+"。半径25 px已接近/小于单个节点间距，其结果需要结合空间量化解释；敏感性分析不会补出模型未保存的亚网格信息。训练平均热图等网格基线按像素中心Voronoi单元的覆盖比例积分；模型MC则直接测试保存样本坐标是否落入圆盘。50 px在此保留为分析设定，本报告不补称其已由逐参与者校准误差验证。",""]
    if len(a):
        support=a.gaze_coordinate_lineage.fillna("").str.contains("favors left").sum() if "gaze_coordinate_lineage" in a else 0
        lines += [f"候选校准重投影在 {_int(support)}/{len(a)} 份录像中更支持左眼坐标来源。该比较搜索现存校准矩阵，最终历史 NPY 生成脚本及每次校准选用链仍未完整确认；未确定的录像没有被自动改成 combined gaze。", ""]
        damaged=a[a.npy_video_timestamp_differences.gt(0)]
        if len(damaged):
            rows=[[r.content_id,_int(r.npy_video_timestamp_differences),_num(r.npy_video_timestamp_max_abs_ns/1e9,6),
                   _int(getattr(r,"backup_video_timestamp_match_n",np.nan))] for r in damaged.itertuples()]
            lines += [_table(["录像","NPY时间不同帧","最大绝对差 s","备份与视频时间匹配帧"],rows)]
    lines += ["分析时间轴取精确 `video_ts.csv` 并按原始 Frame_Number 对齐。录制源码在视频 WriteSample 后写入主机 system_clock，gaze 使用 Varjo captureTime 转 Unix；因此它是记录时间轴，不是已校准的曝光/显示硬件时间。模型 timestamp 一致能证明共享帧索引和记录时钟一致，不能独立排除固定延迟或时钟漂移。NPY 时间精度损坏和真实采集同步错误是不同问题。", ""]
    review=_json(run.out/"region_review/candidate_validation/validation_summary.json")
    if not review:
        review=_json(run.out/"region_review/validation_summary.json")
    if review:
        lines += [f"Mask 推导的 distal/jaw-tip 候选 **未通过主分析替换验证**：{_int(review.get('side_proposals'))} 个侧别候选中 {_int(review.get('heuristic_accepted'))} 个通过启发式 QC，但其中至少 {_int(review.get('obvious_wrong_proximal_among_accepted'))} 个在抽样视觉复核中明显选到近端。该复核不是专家真值误差率，不能把通过启发式 QC 的点称作已验证 jaw tip。原 DLC 点保持不变，主结果必须称为 **腕部/铰接部附近50 px区域的注视**，不能称 jaw-tip attention。", ""]
    else:
        lines += ["本目录尚未提供 mask-tip 候选验证摘要；主定义仍是现有 DLC 腕部/铰接部附近圆盘，不应称为 jaw-tip。", ""]
    flags=run.table("region_review/occurrence_geometry_flags.csv")
    if len(flags) and len(a) and "content_id" in flags:
        flags=flags[flags.content_id.isin(a.content_id)]
    if len(flags):
        rows=[]
        for stage,g in flags.groupby("stage",sort=False):
            nvalid=int(g.get("target_frames",pd.Series(dtype=float)).sum())
            wide=int(g.get("frames_point_radius_gt50",pd.Series(dtype=float)).sum())
            count_changes=int(g.get("count_changes",pd.Series(dtype=float)).sum())
            rows.append([stage,len(g),nvalid,wide,count_changes])
        lines += [_table(["阶段","occurrence数","有target帧","槽位半径>50px帧","槽位身份集合变化次数"],rows),
                  "这里的身份集合变化来自有效槽位 bitmask，不等同于独立物体身份切换。几何 flags 只用于标注质量说明，不能据此自动把 TR 的环改成 peg、把 wrap 的目标改成器械、或排除整个阶段。目标语义仍需标注协议确认；目标/腕部重合时 C 只能作空间偏好解释。", ""]


def _occupancy(run,lines,models):
    lines += ["## 4. 概率主分析：任务整体", "", "所有单元格是参与者均值 [95% CI]；比例单位为%。同一行各指标可能因 C 的零分母而有不同 n，故逐格保留实际人数。", ""]
    metrics=[("L_target","target占用"),("L_gripper","腕部占用"),("C","C"),("D","D"),("L_O","重叠O")]
    for scope,title,sources in [("real_all","真实 gaze 全量",["gaze"]),("paired_test","held-out共同帧",["gaze",*models])]:
        lines += [f"### {title}",""]
        rows=[]
        for task in TASKS:
            for source in sources:
                row=run.occ(task,"whole",source,"L_target",scope=scope)
                if row:
                    rows.append([task,LABELS.get(source,source),*[_cell(run.occ(task,"whole",source,m,scope=scope)) for m,_ in metrics]])
        lines += [_table(["任务","来源",*[label for _,label in metrics]],rows)]
    lines += ["## 5. 概率主分析：逐阶段", "", "以下模型和真实 gaze 均来自 paired_test 的同一有效帧集合。阶段结果用于定位模型偏差；八个阶段、多模型、多指标会产生大量比较，不应仅挑选几个区间不跨0的结果作为确认性结论。", ""]
    for task in TASKS:
        lines += [f"### Task {task}：{TASKS[task]}",""]
        rows=[]
        for stage in STAGES[task]:
            for source in ["gaze",*models]:
                if run.occ(task,stage,source,"C"):
                    rows.append([stage,LABELS.get(source,source),*[_cell(run.occ(task,stage,source,m)) for m,_ in metrics]])
        lines += [_table(["阶段","来源",*[label for _,label in metrics]],rows)]
    lines += ["## 6. 均匀面积基线与超额占用", "", "整体超额占用和 C_excess 如下；单位为百分点。正值表示超过该帧几何区域对应的均匀位置基线，不是对真实 gaze 的预测改善。", ""]
    rows=[]
    for task in TASKS:
        for source in ["gaze",*models]:
            if run.occ(task,"whole",source,"E_target"):
                rows.append([task,LABELS.get(source,source),*[_cell(run.occ(task,"whole",source,m)) for m in ("E_target","E_gripper","C_excess")]])
    lines += [_table(["任务","来源","E_target","E_腕部","C_excess"],rows)]
    rows=[]
    for task in TASKS:
        for stage in ["whole",*STAGES[task]]:
            cells=[run.occ(task,stage,"gaze",m) for m in ("area_T","area_P","area_O","C_U")]
            if any(cells):rows.append([task,stage,*[_cell(r) for r in cells]])
    lines += [_table(["任务","阶段","target-only面积 %","PSM-only面积 %","重叠面积 %","均匀C_U %"],rows)]


def _errors(run,lines,models):
    lines += ["## 7. 概率预测与真实事件的误差", "", "Δ=模型−真实。ΔC、Δtarget、Δ腕部单位为百分点；绝对值越小才表示占用更接近。Brier 是每帧 (q−h)² 的原生时间加权值，先试次再参与者汇总，越低越好。它同时受到概率校准、区分度和事件频率影响。", ""]
    rows=[]
    for task in TASKS:
        for source in models:
            if run.error(task,"whole",source,"BS_target"):
                rows.append([task,LABELS.get(source,source),
                    *[_cell(run.error(task,"whole",source,m),"estimate") for m in ("delta_C","delta_L_target","delta_L_gripper")],
                    *[_cell(run.error(task,"whole",source,m),"estimate",1,3) for m in ("BS_target","BS_gripper")]])
    lines += [_table(["任务","模型","ΔC pp","Δtarget pp","Δ腕部 pp","Brier target","Brier 腕部"],rows)]
    rows=[]
    for task in TASKS:
        for stage in STAGES[task]:
            for source in models:
                if run.error(task,stage,source,"BS_target"):
                    rows.append([task,stage,LABELS.get(source,source),_cell(run.error(task,stage,source,"delta_C"),"estimate"),
                                 _cell(run.error(task,stage,source,"BS_target"),"estimate",1,3),
                                 _cell(run.error(task,stage,source,"BS_gripper"),"estimate",1,3)])
    lines += [_table(["任务","阶段","模型","ΔC pp","Brier target","Brier 腕部"],rows)]
    lines += ["## 8. 四种 Brier 参照及 GT-smoothed 诊断", "",
              "BSS=1−平均模型BS/平均参照BS，采用匹配参与者/试次后重算比值；它不是逐帧 skill 的均值。BSS>0 表示优于该指定参照，<0 表示更差，CI跨0表示当前数据不足以确定方向。四个参照各自检验不同问题：", "",
              "- 画面中心点：固定位置产生的0/1事件预测，衡量相对中心偏好的改进。\n- 均匀面积：在整幅画面均匀取点的软事件概率，随ROI面积变化。\n- 训练平均热图：训练记录的完整空间密度积分，没有把热图缩成一个均值点。\n- 训练事件频率：只用训练数据估计的任务/阶段事件常数概率。", "",
              "训练参照在任务、场景和阶段内按有效原生时间拟合，不使用测试事件频率。这里采用共同 task-training 参照比较所有模型，因此 leave-out/zero-shot 的参照并不等同于该模型实际训练分布。它们是共同评价基准；需避免把 BSS 当成各模型相对自身训练分布的绝对能力。", ""]
    for event,label in [("target","target"),("gripper","腕部")]:
        rows=[]
        for task in TASKS:
            for source in models:
                if run.error(task,"whole",source,"BSS_"+event,reference="train_heatmap"):
                    rows.append([task,LABELS.get(source,source),*[_cell(run.error(task,"whole",source,"BSS_"+event,reference=ref),"estimate",1,3) for ref in REFS]])
        lines += [f"### {label} 事件的 BSS", "", _table(["任务","模型",*REFS.values()],rows)]
    baseline=run.table("baseline_reference_summary.csv","main_probability")
    diag=run.table("GT_smoothed_diagnostic_summary.csv","main_probability")
    rows=[]
    for task in TASKS:
        for event in ("target","gripper"):
            for ref in [*REFS,"GT_smoothed"]:
                source=diag if ref=="GT_smoothed" else baseline
                part=_select(source,scenario="main",task=task,stage="whole",event=event,reference=ref)
                if len(part): rows.append([task,event,REFS.get(ref,"GT-smoothed（诊断）"),_cell(part.iloc[0],"estimate",1,3)])
    lines += [_table(["任务","事件","参照","参照自身 Brier"],rows),
              "GT-smoothed 使用测试真实 gaze 为中心、σ=1网格的平滑分布，属于对观测进行空间模糊化的诊断。它不是真实预测模型，不进入 BSS 分母，也不是数学上的模型上界；例如硬事件 h 自身的 Brier 可为0，而平滑后的概率通常不为0。", ""]
    gt=_select(run.table("gt_smoothed_occupancy.csv"),scenario="main",stage="whole")
    rows=[]
    if len(gt):
        for task,g in gt.groupby("task"):
            p=g.groupby("participant")[["L_target","L_gripper","C"]].mean()
            rows.append([task,len(p),*[_num(100*p[m].mean(),1) for m in ("L_target","L_gripper","C")]])
    lines += [_table(["任务","参与者","GT-smoothed target %","GT-smoothed腕部 %","GT-smoothed C %"],rows),
              "本行诊断占用为试次→参与者→组的均值，仅描述当前平滑设置；未在报告阶段另行计算CI。", ""]


def _contrasts(run,lines):
    data=_select(run.table("model_paired_contrasts.csv","main_probability"),scenario="main")
    lines += ["## 9. 模型之间的配对证据", "", "下表取整体 Brier 的 A−B 配对差。负值倾向 A，正值倾向 B；只有区间完全位于0的一侧才支持该方向。比较先按同一录像对齐，再平均到参与者；不比较互不相同样本的独立CI。", ""]
    rows=[]
    whole=_select(data,stage="whole",metric=["BS_target","BS_gripper"])
    for r in whole.to_dict("records"):
        rows.append([r["task"],r["metric"],r["model_A"]+" − "+r["model_B"],
                     _cell(r,"estimate_A_minus_B",1,3),_direction(r)])
    lines += [_table(["任务","指标","配对模型","A−B [CI]","区间判断"],rows)]
    stage=_select(data,metric=["BS_target","BS_gripper","abs_delta_C"])
    if len(stage):
        stage=stage[stage.stage.ne("whole") & ((stage.ci_low>0)|(stage.ci_high<0))]
    rows=[]
    for r in stage.to_dict("records"):
        scale=100 if r["metric"]=="abs_delta_C" else 1
        winner=r["model_A"] if r["ci_high"]<0 else r["model_B"]
        rows.append([r["task"],r["stage"],r["metric"],r["model_A"]+" − "+r["model_B"],
                     _cell(r,"estimate_A_minus_B",scale,1 if scale==100 else 3),winner])
    lines += [f"逐阶段共有 {len(rows)} 个上述误差比较的未校正95% CI不跨0，完整列于下表。它们是探索性方向证据，不作多重比较后的显著性声明。", "",
              _table(["任务","阶段","误差指标","A−B","差值 [CI]","误差较小方向"],rows)]


def _performance(run,lines,models):
    data=run.table("performance_correlations.csv","main_probability")
    def pc(task,stage,source,metric,scope):
        part=_select(data,scenario="main",task=task,stage=stage,source=source,metric=metric,scope=scope)
        if part.empty:return "未提供"
        r=part.iloc[0]
        return _cell(r,"rho",1,3)+f"；分数值={_int(r.distinct_scores)}"
    lines += ["## 10. 与 SimNow overall score 的关系", "",
              "每个相关系数在任务内计算：先在同一指标可计算、且有有效成绩的试次上，分别求每位参与者的平均占用与平均SimNow overall score，再计算 Spearman ρ。Completed 与 Failed 中存在的有效成绩可参与；无有效成绩或指标为NA的试次不参与。每格列出实际 n 和不同平均分数的个数；后者小于 n 表示成绩存在ties。", "",
              "相关性不等于因果，也不能说明模型占用准确。真实全量和测试配对集合必须分别阅读；阶段相关都是探索性分析。bootstrap 重抽参与者并在每次重抽内重新排名；小样本、ties、常量变量会让部分复制无定义或区间很宽。", ""]
    rows=[]
    for task in TASKS:
        for scope,sources in [("real_all",["gaze"]),("paired_test",["gaze",*models])]:
            for source in sources:
                rows.append([task,scope,LABELS.get(source,source),*[pc(task,"whole",source,m,scope) for m in ("C","L_target","L_gripper")]])
    lines += [_table(["任务","集合","来源","C–score ρ [CI]","target–score ρ [CI]","腕部–score ρ [CI]"],rows)]
    points=_select(run.table("performance_participants.csv","main_probability"),scenario="main",task="C",stage="whole",
                   source="gaze",scope="paired_test",metric="L_target")
    c_target=_select(data,scenario="main",task="C",stage="whole",source="gaze",scope="paired_test",metric="L_target")
    if len(points) and len(c_target):
        frequencies=points.mean_score.value_counts().sort_index()
        score_text="、".join(f"{score:g}分×{count}人" for score,count in frequencies.items())
        r=c_target.iloc[0]
        lines += [f"**Task C测试集合的ties与退化复制**：以真实gaze整体target相关的参与者表为准，平均成绩分布为 {score_text}。该相关的bootstrap有 {_int(r.bootstrap_undefined)}/{_int(r.bootstrap_defined+r.bootstrap_undefined)} 次无定义，现有CI只由 {_int(r.bootstrap_defined)} 次可定义复制计算；不可把丢弃的复制理解成零相关。", ""]
        loo=_select(run.table("performance_leave_one_out.csv","main_probability"),scenario="main",task="C",stage="whole",
                    source="gaze",scope="paired_test",metric="L_target")
        constant=[]
        for row in loo.itertuples():
            remaining=points[points.participant.ne(row.held_out)].mean_score.dropna()
            if len(remaining) and remaining.nunique()==1:
                constant.append(f"移除{row.held_out}后，剩余{len(remaining)}人的平均成绩均为{remaining.iloc[0]:g}")
        if constant:
            lines += ["；".join(constant)+f"，相关因此不可估。target指标的leave-one-out仅 {_int(r.loo_defined)}/{_int(r.n_participants)} 次可估。表中的最小/最大范围只覆盖这些可估删除；即使显示相同的窄范围，也不能据此宣称对全部参与者删除都稳健。", ""]
    for task in TASKS:
        lines += [f"### Task {task} 逐阶段相关",""]
        rows=[]
        for stage in STAGES[task]:
            for scope,sources in [("real_all",["gaze"]),("paired_test",["gaze",*models])]:
                for source in sources:
                    rows.append([stage,scope,LABELS.get(source,source),*[pc(task,stage,source,m,scope) for m in ("C","L_target","L_gripper")]])
        lines += [_table(["阶段","集合","来源","C ρ [CI]","target ρ [CI]","腕部 ρ [CI]"],rows)]
    whole=_select(data,scenario="main",stage="whole",metric=["C","L_target","L_gripper"])
    rows=[]
    for r in whole.to_dict("records"):
        rows.append([r["task"],r["scope"],LABELS.get(r["source"],r["source"]),r["metric"],
                     _int(r["bootstrap_defined"]),_int(r["bootstrap_undefined"]),
                     f"{_int(r['loo_defined'])}/{_int(r['n_participants'])}",
                     f"{_num(r['loo_min'],3)} 至 {_num(r['loo_max'],3)}"])
    lines += ["逐一去掉参与者后的整体相关范围如下。它用于识别单个参与者驱动的关联，不是另一组95% CI；NA表示保留人数/变异不足。范围仅使用可估的删除，必须同时阅读可估次数/全部删除次数。", "",
              _table(["任务","集合","来源","指标","有效bootstrap","无定义bootstrap","LOO可估/全部","可估LOO的ρ范围"],rows)]


def _decoders(run,lines,models):
    lines += ["## 11. 三种点解码的共同帧比较", "",
              "概率主分析不产生唯一 gaze 轨迹。三个分支来自同一模型的导出文件，但使用的表示不同：期望点和argmax读取保存的 `probability`；MC占用和逐帧抽样读取原始 `samples`。保存的概率图与有限K样本的经验分布相关但不等同，不能把分支差异全部解释为对完全相同分布的点解码规则差异。期望点可能落在两个峰之间，argmax选概率最大的网格位置，逐帧抽样用固定seed从K个样本中独立选一个。三者共同有效帧规则在生成阶段统一，以下从各分支本次统计表读取各自均值和CI，没有重算一轮bootstrap。", "",
              "当前可查 `dvss_pytorch/model/cvae.py` 的sample函数先对各latent解码的softmax概率求均值得到probability，另用multinomial产生样本；`scripts/predictor.py`分别保存两个字段。该代码没有证明保存probability是对样本作σ=1高斯核平滑；σ=1的gaze_heatmap用于标签/GT诊断。旧预测导出时的代码revision缺失，当前源码不能补足历史版本溯源。", ""]
    branches={b:run.table("trial_stage.csv",b) for b in DECODERS}
    ids=["scope","scenario","task","stage","source","participant","content_id"]
    matches=[]
    for a,b in combinations(DECODERS,2):
        x=_select(branches[a],scenario="main",scope="paired_test")
        y=_select(branches[b],scenario="main",scope="paired_test")
        if x.empty or y.empty:
            matches.append([DECODERS[a]+" / "+DECODERS[b],"缺少分支","NA","NA"]);continue
        z=x.merge(y,on=ids,suffixes=("_a","_b"),validate="one_to_one")
        match_n=np.array_equal(z.valid_n_a.to_numpy(),z.valid_n_b.to_numpy())
        match_s=np.allclose(z.valid_s_a,z.valid_s_b,atol=1e-9,rtol=1e-12)
        matches.append([DECODERS[a]+" / "+DECODERS[b],len(z),str(match_n),str(match_s)])
    lines += [_table(["分支对","配对试次阶段行","有效帧数相等","有效秒数相等"],matches),
              "相同帧支持不保证 C 或驻留时长在三个分支中都有值；因此仍显示逐指标实际人数。不同分支CI相互重叠/不重叠不能替代专门的解码器配对差检验。", ""]
    rows=[]
    for task in TASKS:
        for source in models:
            for branch,label in DECODERS.items():
                if run.occ(task,"whole",source,"C",branch=branch):
                    rows.append([task,LABELS.get(source,source),label,*[_cell(run.occ(task,"whole",source,m,branch=branch)) for m in ("L_target","L_gripper","C","D","L_O")]])
    lines += [_table(["任务","模型","解码器","target %","腕部 %","C %","D %","O %"],rows)]
    # Re-align recordings to compute purely descriptive decoder differences.
    rows=[]
    for a,b in combinations(DECODERS,2):
        x=_select(branches[a],scenario="main",scope="paired_test",stage="whole")
        y=_select(branches[b],scenario="main",scope="paired_test",stage="whole")
        if x.empty or y.empty:continue
        z=x.merge(y,on=ids,suffixes=("_a","_b"),validate="one_to_one")
        z=z[z.source.ne("gaze")]
        for (task,source),g in z.groupby(["task","source"]):
            vals=[]
            for metric in ("L_target","L_gripper","C"):
                ok=np.isfinite(g[metric+"_a"]) & np.isfinite(g[metric+"_b"])
                h=g.loc[ok,["participant"]].copy()
                h["diff"]=g.loc[ok,metric+"_a"]-g.loc[ok,metric+"_b"]
                d=h.groupby("participant")["diff"].mean()
                vals.append(f"{_num(100*d.mean(),1)}；n={len(d)}")
            rows.append([task,LABELS.get(source,source),DECODERS[a]+" − "+DECODERS[b],*vals])
    lines += ["在每个指标都有值的共同录像上重新配对、再平均至参与者，得到以下描述性解码差（百分点）。此表没有新增CI，不据其单独判断显著差异。", "",
              _table(["任务","模型","A−B","Δtarget pp","Δ腕部 pp","ΔC pp"],rows),
              "## 12. 时间维度：只解释点解码后的观察序列", "",
              "驻留为同一AOI连续观察段，保留时长≥120 ms的段。每录像/阶段先求这些段的中位时长，再在参与者内平均试次，最后组均值及参与者CI；不是把全体事件池化后的总体中位数。missing帧、原始帧断裂、>100 ms时间缺口和occurrence边界切断序列。触及观察边界的截断段达到120 ms仍计入，并单报截断数量，所以这些结果不是删失校正后的完整 fixation 寿命。", "",
              "直接切换连接相邻且各≥120 ms的T/P段；经重叠切换连接T→O→P或反向，两个端点段各≥120 ms，中间O段不要求120 ms。other不跳过。两个切换率都用有效观察分钟作分母，单位次/min；两者分别报告，没有把经O的切换漏作直接切换。", ""]
    rows=[]
    for task in TASKS:
        for source in ["gaze",*models]:
            for branch,label in DECODERS.items():
                if source=="gaze" and branch!="decoded_expectation":continue
                if run.occ(task,"whole",source,"C",branch=branch):
                    ms=[("dwell_T_ms",1,1),("dwell_P_ms",1,1),("dwell_O_ms",1,1),("direct_rate",1,2),("via_rate",1,2)]
                    cells=[_cell(run.occ(task,"whole",source,branch+"_"+m,branch=branch),scale=scale,digits=digits) for m,scale,digits in ms]
                    rows.append([task,LABELS.get(source,source),"真实观察" if source=="gaze" else label,*cells])
    lines += [_table(["任务","来源","解码器","T驻留 ms","P驻留 ms","O驻留 ms","直接 次/min","经O 次/min"],rows),
              "逐帧抽样的随机变化会直接影响驻留和切换。K列来自逐帧独立潜变量/类别抽样，没有跨时间的联合轨迹身份；固定某一K列也不能将其变成联合scanpath。短驻留或高切换可能是解码随机性，不可直接归因于模型学到了人的扫视行为。概率MC分支不报告驻留/切换。", ""]
    rows=[]
    for branch,label in DECODERS.items():
        d=_select(branches[branch],scenario="main",scope="paired_test",stage="whole")
        for (task,source),g in d.groupby(["task","source"]) if len(d) else []:
            if source=="gaze" and branch!="decoded_expectation":continue
            cells=[]
            for long in ("target_only","psm_only","overlap"):
                count=branch+"_dwell_"+long+"_n"; cens=branch+"_dwell_"+long+"_n_censored"
                n=g[count].sum() if count in g else np.nan; c=g[cens].sum() if cens in g else np.nan
                cells.append(f"{_int(c)} / {_int(n)}")
            rows.append([task,LABELS.get(source,source),"真实观察" if source=="gaze" else label,*cells])
    lines += [_table(["任务","来源","解码器","T截断/保留段","P截断/保留段","O截断/保留段"],rows)]


def _sensitivity_repetition(run,lines,models):
    scenarios=[s for s in SCENARIOS if s in run.config.get("scenarios",SCENARIOS)]
    lines += [f"## 13. {len(scenarios)}种预设条件的敏感性", "",
              "主定义保持50 px；25、100 px及单独扩大PSM用于判断几何依赖。全部条件均取消valid取值限制，以坐标有限且在归一化范围[0,1]内的原始gaze为输入。平滑分支在该输入上执行原定的短缺口插值及三样本平滑，会改变位置和可用帧，因此仍可能改变实际人数；它是单独报告的敏感性条件，不会把插值帧计入未平滑主分析。没有把敏感性条件当作新的最优参数搜索。", ""]
    rows=[]
    for task in TASKS:
        for scope,sources in [("real_all",["gaze"]),("paired_test",["gaze",*models])]:
            for source in sources:
                rows.append([task,scope,LABELS.get(source,source),*[_cell(run.occ(task,"whole",source,"C",scope=scope,scenario=s)) for s in scenarios]])
    lines += [_table(["任务","集合","来源",*[SCENARIOS[s]+" C%" for s in scenarios]],rows)]
    c=run.table("coverage.csv"); rows=[]
    for task in TASKS:
        for scenario in scenarios:
            g=_select(c,task=task,scenario=scenario,stage="whole")
            if g.empty:continue
            rows.append([task,scenario,_int(g.valid_n.sum()),_num(g.valid_s.sum(),1),
                         _int(g.paired_n.sum()),_num(g.paired_s.sum(),1),g[g.valid_n.gt(0)].participant.nunique()])
    lines += [_table(["任务","条件","真实有效帧","真实有效秒","配对有效帧","配对有效秒","真实有效人数"],rows)]
    perf=run.table("performance_correlations.csv","main_probability")
    main=_select(perf,scenario="main",metric=["C","L_target","L_gripper"])
    alts=_select(perf,scenario=[s for s in scenarios if s!="main"],metric=["C","L_target","L_gripper"])
    keys=["scope","task","stage","source","metric"]
    flips=[]
    if len(main) and len(alts):
        z=alts.merge(main[keys+["rho","n_participants"]],on=keys,suffixes=("","_main"),validate="many_to_one")
        z=z[np.isfinite(z.rho)&np.isfinite(z.rho_main)&(z.rho*z.rho_main<0)]
        for key,g in z.groupby(keys,sort=True):
            flips.append([*key,"、".join(g.scenario.astype(str)),_num(g.rho_main.iloc[0],3),
                          f"{g.n_participants.min()}–{g.n_participants.max()}"])
    lines += [f"相关性方向敏感性：当前找到 {len(flips)} 个任务/阶段/来源/指标组合在至少一个敏感性条件下与主分析ρ异号。以下只标记方向不稳定，不把数值接近0的正负翻转解释成实质逆转。", "",
              _table(["集合","任务","阶段","来源","指标","异号条件","主ρ","敏感性n范围"],flips),
              "## 14. repetition 1→3 的探索性变化", "",
              "round 按参与者/任务内不同 TaskNum 的先后排序，随后才应用内容筛选；它不是 Attempt 数，也不能保证所有人都有第3轮。表内为第3轮−第1轮，在两轮指标均可用的同一参与者间配对，比例差单位为百分点。练习相关变化仍可能混有任务时长、阶段构成和质量覆盖变化。", ""]
    d=_select(run.table("repetition_contrasts.csv","main_probability"),scenario="main",stage="whole",round_from=1,round_to=3)
    rows=[]
    for task in TASKS:
        for scope,sources in [("real_all",["gaze"]),("paired_test",["gaze",*models])]:
            for source in sources:
                values=[]
                for metric in ("C","L_target","L_gripper","L_O","score"):
                    g=_select(d,task=task,scope=scope,source=source,metric=metric)
                    value=_cell(g.iloc[0] if len(g) else None,"mean_paired_change",1 if metric=="score" else 100,
                                1,n_column="n_matched_participants")
                    if len(g):
                        value+=f"；录像{_int(g.iloc[0].n_recordings_from)}→{_int(g.iloc[0].n_recordings_to)}"
                    values.append(value)
                rows.append([task,scope,LABELS.get(source,source),*values])
    lines += [_table(["任务","集合","来源","ΔC pp","Δtarget pp","Δ腕部 pp","ΔO pp","Δscore"],rows)]
    c_occ=_select(d,task="C",scope="real_all",source="gaze",metric="L_target")
    c_score=_select(d,task="C",scope="real_all",source="gaze",metric="score")
    if len(c_occ) and len(c_score):
        a,b=c_occ.iloc[0],c_score.iloc[0]
        same=(a.n_recordings_from==b.n_recordings_from and a.n_recordings_to==b.n_recordings_to)
        lines += [f"Task C真实全量1→3：target占用变化使用 {_int(a.n_matched_participants)} 人、两端 {_int(a.n_recordings_from)}→{_int(a.n_recordings_to)} 份录像；score变化使用 {_int(b.n_matched_participants)} 人、两端 {_int(b.n_recordings_from)}→{_int(b.n_recordings_to)} 份录像。"+
                  ("两端录像数相同仍不自动证明每个指标使用了完全相同的录像。" if same else "占用与成绩的试次支持不同；有可用占用但无有效成绩的录像能进入占用变化，不能进入score变化。即使参与者数相同，也不能称为同trial支持。"),""]


def _closing(run,lines):
    prediction=run.table("prediction_audit.csv")
    issues=_json(run.out/"input_issues.json",[])
    lines += ["## 15. 可复核性、图表和结论边界", ""]
    if len(prediction):
        kval=sorted(prediction.num_samples.dropna().astype(int).unique().tolist())
        seeds=sorted(prediction.sampled_trajectory_seed.dropna().astype(int).unique().tolist()) if "sampled_trajectory_seed" in prediction else []
        lines += [f"本次读取 {len(prediction)} 个预测文件，K取值={kval}，点抽样seed={seeds}；无效概率图共 {_int(prediction.invalid_maps.sum())} 个，无效样本点共 {_int(prediction.invalid_sample_points.sum())} 个。预测文件实际SHA在每个新进程内至少验证一次；旧期望点只在预测SHA、帧号、时间和分辨率匹配后复用，argmax/采样独立解码。", ""]
        missing=int(prediction.checkpoint_sha256.isna().sum()) if "checkpoint_sha256" in prediction else len(prediction)
        lines += [f"仍有 {missing}/{len(prediction)} 个预测文件的导出侧 checkpoint SHA未记录。文件hash可固定现有预测，但不能补造当时缺失的checkpoint/代码版本溯源。", ""]
    checks=[[k,str(v)] for k,v in run.validation.items() if isinstance(v,(bool,int,float,str))]
    lines += [_table(["本次验证项","记录值"],checks),f"输入问题记录数：{len(issues) if isinstance(issues,list) else '见文件'}。完整审计见 {_link(run.out/'input_issues.json')}。", ""]
    lines += ["解释本次结果时应同时满足三层条件：有可用帧与足够参与者；该指标分母和CI确实可估；目标与腕部区域的语义足以支持所讨论的行为问题。不能用预测概率更集中、C更高、或某一阶段ρ更大替代这三层检查。当前最稳妥的行为表述是“已标注目标邻域与DLC腕部邻域之间的空间分配”，而非完整器械注视、jaw-tip fixation或专家策略的直接证明。", "",
              "### 现有图表入口", ""]
    links=[]
    for task in TASKS:
        for filename,label in [(f"task_{task}_regions.png",f"Task {task} 主占用"),
                               (f"task_{task}_agreement_C.png",f"Task {task} 真实–模型C"),
                               (f"task_{task}_performance_C.png",f"Task {task} C–score"),
                               (f"task_{task}_performance_L_target.png",f"Task {task} target–score"),
                               (f"task_{task}_performance_L_gripper.png",f"Task {task} 腕部–score"),
                               (f"task_{task}_brier_skill.png",f"Task {task} Brier/BSS"),
                               (f"task_{task}_sensitivity.png",f"Task {task} 敏感性")]:
            path=run.out/"main_probability/figures/paired_test"/filename
            if path.is_file():links.append("- "+_link(path,label))
        for branch,label in DECODERS.items():
            path=run.out/branch/"figures/paired_test"/f"task_{task}_dwell_switch.png"
            if path.is_file():links.append("- "+_link(path,f"Task {task} {label}驻留/切换"))
    lines += ["\n".join(links) or "当前目录尚未生成图。", "",
              "### 原始表入口", "", "\n".join("- "+_link(run.out/name,name) for name in [
                  "coverage.csv","recording_audit.csv","prediction_audit.csv","main_probability/group_stage_summary.csv",
                  "main_probability/model_error_summary.csv","main_probability/model_paired_contrasts.csv",
                  "main_probability/performance_correlations.csv","main_probability/performance_participants.csv",
                  "main_probability/repetition_contrasts.csv","baseline_input_audit.json",
                  "region_review/occurrence_geometry_flags.csv"] if (run.out/name).is_file()),""]


def build_detailed_report(out, destination=None):
    """Write detailed_report_zh.md for a completed/available run; return Path."""
    run=Run(out)
    if run.table("group_stage_summary.csv","main_probability").empty:
        raise ValueError("main_probability/group_stage_summary.csv must exist before detailed report")
    models=run.config.get("models",["all_ABCD","single_task","leave_out","spd_finetune","spd_zero_shot"])
    lines=["# Target Locking Analysis：新版详细结果", "",
           "本报告将概率占用、点解码和时间行为分开，所有统计数值来自同一运行目录。NA保留为不可估，不填0；CI是近似的参与者bootstrap区间。", ""]
    _findings(run,lines,models)
    _coverage_section(run,lines)
    _definitions_and_quality(run,lines)
    _occupancy(run,lines,models)
    _errors(run,lines,models)
    _contrasts(run,lines)
    _performance(run,lines,models)
    _decoders(run,lines,models)
    _sensitivity_repetition(run,lines,models)
    _closing(run,lines)
    path=Path(destination) if destination is not None else run.out/"detailed_report_zh.md"
    path.write_text("\n".join(lines),encoding="utf-8")
    return path


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    print(build_detailed_report(parser.parse_args().out))
