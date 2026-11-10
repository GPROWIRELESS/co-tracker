"""Render the agreed gaze comparison tables and plots from audited result CSVs."""
from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["all_ABCD", "single_task", "leave_out", "spd_finetune", "spd_zero_shot"]
COLORS = ["#2563eb", "#0d9488", "#8b5cf6", "#ea580c", "#64748b"]
STAGES = {
    "A": ["reach ring", "grab ring", "transfer ring", "place ring", "drop ring"],
    "C": ["reach string", "threading string through tower", "wrap around psm", "pull through and cinch"],
}


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.edgecolor": "#cbd5e1",
        "text.color": "#172033", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#334155",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })


def distance_plot(summary, video, output):
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    rng = np.random.default_rng(1909)
    maxval = max(video["mean_distance_pct_diagonal"].max(), summary["ci95_high"].max())
    for i, model in enumerate(MODELS):
        row = summary.loc[summary.model.eq(model)].iloc[0]
        vals = video.loc[video.model.eq(model), "mean_distance_pct_diagonal"].to_numpy()
        ys = i + rng.uniform(-0.14, 0.14, len(vals))
        ax.scatter(vals, ys, s=30, color=COLORS[i], alpha=0.28, linewidths=0, zorder=2)
        mean, lo, hi = row["mean_distance_pct_diagonal"], row["ci95_low"], row["ci95_high"]
        ax.plot([lo, hi], [i, i], color=COLORS[i], lw=3, solid_capstyle="round", zorder=3)
        ax.scatter([mean], [i], marker="D", s=65, color=COLORS[i], edgecolors="white", linewidth=0.8, zorder=4)
        ax.text(maxval * 1.055, i, f"{mean:.2f}%  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=10, color="#334155")
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.invert_yaxis()
    ax.set_xlim(0, maxval * 1.43)
    ax.set_ylim(len(MODELS) - 0.45, -0.7)
    ax.set_xlabel("Mean framewise gaze distance (% of frame diagonal)  |  Lower is better")
    ax.set_title("Novice active gaze: framewise distance", loc="left", pad=30)
    ax.text(0, 1.04, "15 videos / 10 users. Diamonds: equal-video means; bars: 95% user-cluster bootstrap CI.",
            transform=ax.transAxes, fontsize=10, color="#64748b")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.125, 0.005, "Faint dots: individual videos. SPD finetune averages metrics across seeds 42, 43 and 44; it is not an ensemble.",
             fontsize=9, color="#64748b")
    save_figure(fig, output, "distance_comparison")


def heatmap_numbers(ax, mat, vmin, vmax, decimals):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                frac = (val - vmin) / max(vmax - vmin, 1e-12)
                ax.text(j, i, f"{val:.{decimals}f}", ha="center", va="center",
                        fontsize=10, color="white" if frac > 0.55 else "#172033")
            else:
                ax.text(j, i, "n.a.", ha="center", va="center", color="#64748b", fontsize=10)


def video_heatmap(video, output):
    order = video[["video_id", "task"]].drop_duplicates().sort_values(["task", "video_id"])
    pivot = video.pivot(index="video_id", columns="model", values="mean_distance_pct_diagonal")
    pivot = pivot.reindex(index=order.video_id, columns=MODELS)
    mat = pivot.to_numpy()
    vmax = np.ceil(np.nanmax(mat) / 2) * 2
    fig, ax = plt.subplots(figsize=(10.3, 10.4))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(5), MODELS, rotation=20, ha="right")
    ax.set_yticks(range(len(order)), [f"{r.task}   {r.video_id}" for r in order.itertuples()])
    ax.set_title("Framewise distance for each video", loc="left", pad=17)
    ax.tick_params(length=0)
    heatmap_numbers(ax, mat, 0, vmax, 2)
    tasks = order.task.to_list()
    for i in range(1, len(tasks)):
        if tasks[i] != tasks[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label("Distance (% of frame diagonal); lower is better")
    fig.text(0.12, 0.012, "All models use the same valid frames within each video. Current NPY gaze is the ground truth.",
             fontsize=9, color="#64748b")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output, "distance_by_video")


def stage_heatmap(stage, output):
    vmax = min(1.0, np.ceil(float(stage.mean_jsd.max()) / 0.05) * 0.05)
    vmax = max(vmax, 0.05)
    fig, axes = plt.subplots(2, 1, figsize=(11.9, 9.5), gridspec_kw={"height_ratios": [5, 4]})
    fig.subplots_adjust(left=0.36, right=0.88, top=0.88, bottom=0.11, hspace=0.28)
    im = None
    for ax, task in zip(axes, ["A", "C"]):
        rows = stage.loc[stage.task.eq(task)]
        names = [s for s in STAGES[task] if s in set(rows.stage)]
        pivot = rows.pivot(index="stage", columns="model", values="mean_jsd").reindex(index=names, columns=MODELS)
        counts = rows.groupby("stage").n_videos.first()
        mat = pivot.to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        labels = [f"{textwrap.fill(s, 31)}  (n={int(counts[s])})" for s in names]
        ax.set_yticks(range(len(names)), labels)
        ax.set_xticks(range(len(MODELS)), MODELS if task == "C" else [""] * 5,
                      rotation=20, ha="right")
        ax.set_title(f"Task {task}", loc="left", fontsize=13, pad=11)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        heatmap_numbers(ax, mat, 0, vmax, 3)
    fig.suptitle("Stage-specific gaze distribution similarity", x=0.05, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.05, 0.926, "Jensen-Shannon divergence (log2): 0 = identical, 1 = maximally different. Lower is better.",
             fontsize=10, color="#64748b")
    cax = fig.add_axes([0.91, 0.20, 0.022, 0.61])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("JSD (shared scale)")
    fig.text(0.05, 0.018, "Each cell averages video-level stage JSD. n = videos containing the stage.\n"
             "Unannotated frames are excluded. Task A 'drop ring' has one video; Tasks B/D have no stage labels.",
             fontsize=9, color="#64748b")
    save_figure(fig, output, "stage_jsd_comparison")


def escaped(value):
    return html.escape(str(value), quote=True)


def summary_table(summary):
    rows = []
    ranked = summary.sort_values("mean_distance_pct_diagonal")
    for rank, row in enumerate(ranked.itertuples(), 1):
        rows.append(f"<tr><td>{rank}</td><td>{escaped(row.model)}</td>"
                    f"<td class='num'>{row.mean_distance_pct_diagonal:.3f}%</td>"
                    f"<td class='num'>[{row.ci95_low:.3f}, {row.ci95_high:.3f}]</td>"
                    f"<td class='num'>{row.mean_distance_px:.2f}</td>"
                    f"<td class='num'>{int(row.n_videos)}</td></tr>")
    return "<table><thead><tr><th>排序</th><th>模型</th><th>距离 / 对角线</th><th>95% 置信区间（%）</th><th>距离（像素）</th><th>视频数</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def stage_table(stage, task):
    sub = stage.loc[stage.task.eq(task)]
    pivot = sub.pivot(index="stage", columns="model", values="mean_jsd").reindex(columns=MODELS)
    rows = []
    for name in STAGES[task]:
        if name not in pivot.index:
            continue
        values = pivot.loc[name].to_numpy()
        minval = np.nanmin(values)
        info = sub.loc[sub.stage.eq(name)].iloc[0]
        cells = "".join(f"<td class='num{' best' if np.isclose(v, minval, atol=1e-12, rtol=0) else ''}'>{v:.4f}</td>" for v in values)
        rows.append(f"<tr><td>{escaped(name)}</td><td class='num'>{int(info.n_videos)}</td><td class='num'>{int(info.n_frames):,}</td>{cells}</tr>")
    header = "<th>stage</th><th>视频数</th><th>有效帧</th>" + "".join(f"<th>{escaped(m)}</th>" for m in MODELS)
    return "<div class='table-scroll'><table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def build_report(summary, video, stage, output):
    ranked = summary.sort_values("mean_distance_pct_diagonal")
    first, last = ranked.iloc[0], ranked.iloc[-1]
    nframes = int(summary.n_frames.iloc[0])
    nvideos = int(summary.n_videos.iloc[0])
    nusers = int(summary.n_users.iloc[0])
    stage_videos = int(video.loc[video.task.isin(["A", "C"]), "video_id"].nunique())
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8")) if (output / "audit.json").exists() else {}
    pairwise = pd.read_csv(output / "pairwise_distance.csv") if (output / "pairwise_distance.csv").exists() else pd.DataFrame()
    near_tie = ""
    if not pairwise.empty:
        row = pairwise.loc[pairwise.model_a.eq("all_ABCD") & pairwise.model_b.eq("single_task")]
        if len(row):
            r = row.iloc[0]
            near_tie = (f"<p><b>all_ABCD 与 single_task 基本持平：</b>两者平均误差仅相差 {abs(r.difference_a_minus_b):.4f} 个百分点。"
                        f"all_ABCD − single_task 的配对差值 95% 置信区间为 [{r.ci95_low:.3f}, {r.ci95_high:.3f}] 个百分点，"
                        "包含 0，当前结果不支持明确区分这两个模型。</p>")
    wins = stage.loc[stage.groupby(["task", "stage"]).mean_jsd.idxmin()].model.value_counts()
    wintext = "；".join(f"{escaped(m)} 在 {int(n)} 个 stage 上的平均 JSD 最低" for m, n in wins.items())
    available = []
    for filename, label in [
        ("model_comparison.csv", "总体距离对比表"),
        ("video_distance.csv", "视频级距离明细"),
        ("pairwise_distance.csv", "模型配对距离差值"),
        ("stage_jsd_summary.csv", "stage JSD 对比表"),
        ("stage_jsd_video.csv", "视频级 stage JSD 明细"),
        ("comparison_tables.xlsx", "Excel 对比表"),
        ("input_manifest.csv", "输入文件清单"),
        ("audit.json", "数据核验记录"),
    ]:
        if (output / filename).exists():
            available.append(f"<a href='{escaped(filename)}'>{escaped(label)}</a>")
    downloads = " · ".join(available)
    text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Novice active gaze：五模型对比</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#e2e8f0;--blue:#2563eb;}}
*{{box-sizing:border-box}}body{{font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;color:var(--ink);margin:0;background:#f3f6fa;line-height:1.7}}
main{{max-width:1240px;margin:32px auto;padding:36px 44px;background:#fff;border:1px solid var(--line);border-radius:12px}}
h1{{font-size:30px;line-height:1.3;margin:0 0 12px}}h2{{font-size:22px;margin:36px 0 12px}}h3{{font-size:18px;margin-top:24px}}p{{margin:10px 0}}
.eyebrow{{color:var(--blue);font-size:12px;letter-spacing:1.2px;margin-bottom:12px}}.meta,.note{{color:var(--muted);font-size:14px}}.lead{{font-size:17px}}
.facts{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:25px 0}}.fact{{padding:18px;background:#f8fafc;border:1px solid var(--line);border-radius:8px}}.fact b{{display:block;font-size:26px}}.fact span{{color:var(--muted);font-size:13px}}
figure{{margin:22px 0 28px}}img{{max-width:100%;height:auto;display:block}}figcaption{{font-size:13px;color:var(--muted);margin-top:8px}}a{{color:var(--blue);text-decoration:none}}a:hover{{text-decoration:underline}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}}th{{text-align:left;background:#f1f5f9;border-bottom:2px solid #cbd5e1;padding:11px 12px;white-space:nowrap}}td{{border-bottom:1px solid var(--line);padding:10px 12px}}.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}.best{{font-weight:700;color:#1d4ed8;background:#eff6ff}}.table-scroll{{overflow:auto}}code{{background:#f1f5f9;padding:2px 5px;border-radius:4px;font-size:.95em}}
.method{{background:#f8fafc;border-left:3px solid #94a3b8;padding:15px 20px;margin:16px 0}}li{{margin:8px 0}}details{{border-top:1px solid var(--line);padding-top:15px;margin-top:25px}}summary{{cursor:pointer;font-weight:600}}footer{{margin-top:30px;border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted)}}
@media(max-width:760px){{main{{margin:0;border:0;padding:25px 18px}}.facts{{grid-template-columns:1fr}}h1{{font-size:25px}}}}
@media print{{body{{background:white}}main{{margin:0;padding:0;border:0;max-width:none}}figure,table{{break-inside:avoid}}details{{display:block}}}}
</style></head><body><main>
<div class="eyebrow">GAZE MODEL COMPARISON · 2026-11-09</div>
<h1>五种模型对 novice active gaze 的复现程度</h1>
<p class="meta">当前 NPY 为 GT。逐帧距离覆盖全部 15 个测试视频；stage JSD 仅分析具有 stage 标注的 A、C 任务。</p>
<div class="facts"><div class="fact"><b>{nvideos} 个视频 / {nusers} 位用户</b><span>test · novice · active</span></div><div class="fact"><b>{nframes:,} 帧</b><span>全部模型共同有效的距离评估帧</span></div><div class="fact"><b>{stage_videos} 个 stage 视频</b><span>A、C 任务；{int(audit.get('stage_annotated_common_frames', 0)):,} 帧参与 stage JSD</span></div></div>
<p class="lead">在本次共同样本上，<b>{escaped(first.model)}</b> 的平均逐帧距离最小，为画面对角线的 <b>{first.mean_distance_pct_diagonal:.3f}%</b>（{first.mean_distance_px:.2f} 像素）；{escaped(last.model)} 的误差最大，为 {last.mean_distance_pct_diagonal:.3f}%。</p>
{near_tie}
<p>{wintext}。stage JSD 反映整体关注位置的分布，不包含阶段内部的注视时序是否一致。</p>
<p class="note">这些结果比较模型接近真实 gaze 的程度。未设定“成功复现”的绝对通过阈值；模型排序本身不能证明已达到人类一致性。</p>

<h2>逐帧位置：全部 15 个视频</h2>
<p>每帧先计算预测位置与当前 GT 位置的欧氏距离，再在每个视频内取均值，最后对视频等权平均。百分比与像素是同一个距离指标的不同单位；全部视频均为 1280 × 1024。</p>
{summary_table(summary)}
<p class="note">置信区间：以用户为单位的聚类 bootstrap（10,000 次），保留同一用户的视频关联。SPD finetune 先分别计算 seed 42、43、44 的指标，再平均；其余模型使用现有的单个 seed 预测。置信区间不表示跨训练 seed 的不确定性。</p>
<figure><img src="distance_comparison.png" alt="五模型逐帧 gaze 距离对比，包含用户聚类 bootstrap 置信区间和视频点"><figcaption>菱形为视频等权均值，横线为 95% 置信区间，浅色点为单个视频。<a href="distance_comparison.svg">SVG 矢量图</a></figcaption></figure>
<figure><img src="distance_by_video.png" alt="十五个视频五种模型的距离误差热图"><figcaption>同一行中五种模型使用完全相同的有效帧。数值越小越好。<a href="distance_by_video.svg">SVG 矢量图</a></figcaption></figure>

<h2>阶段关注分布：A、C 任务</h2>
<p>对“视频 × stage”分别计算 JSD，再在相同任务与 stage 内对视频等权平均。JSD 使用以 2 为底的对数，范围为 0–1，0 表示分布一致。表中蓝色粗体是该行最低值。</p>
<h3>Task A</h3>{stage_table(stage, "A")}
<h3>Task C</h3>{stage_table(stage, "C")}
<p class="note">视频数与有效帧数均来自当前参与计算的数据。A 的 drop ring 只有一个视频、30 个有效帧，结果仅描述该视频。不同 stage 的样本规模不同，不把这些 stage 简单合并成额外的总体 JSD 排名。</p>
<figure><img src="stage_jsd_comparison.png" alt="A 与 C 各 stage 的五模型 JSD 热图"><figcaption>两面板使用共同色标；n 为含有该 stage 的有效视频数。<a href="stage_jsd_comparison.svg">SVG 矢量图</a></figcaption></figure>

<h2>计算口径与数据匹配</h2>
<div class="method"><p><b>坐标：</b>对每帧保存的 32 × 40 概率图 P，使用概率加权均值：x̂ = ΣP(r,c)·c/39，ŷ = ΣP(r,c)·r/31。不是 argmax，也不从 64 个随机样本中选择最接近 GT 的一个。</p>
<p><b>距离：</b>d = √[(1279·(x̂−xGT))² + (1023·(ŷ−yGT))²]，报告 100·d/√(1279²+1023²)。使用当前 NPY 中有限且位于 [0,1] 的 gaze，并要求所有模型均有有效预测。</p>
<p><b>stage 分布：</b>预测分布是该 stage 共同有效帧的完整概率图均值。GT 每帧在相同 32 × 40 网格上生成 σ = 1 网格单位的归一化高斯图，再平均。保持模型的完整分布，不把均值坐标重新画成模型热图。</p>
<p><b>JSD：</b>M=(P+Q)/2，JSD=½ΣP·log₂(P/M)+½ΣQ·log₂(Q/M)。每个视频、stage、seed 独立计算后，再按约定汇总；零概率项贡献记为 0。</p></div>
<ul><li>数据筛选为 <code>domain=active</code>、<code>meta.split=test</code>、<code>meta.src_skill=novice</code>。视频与模型文件通过 content_id 和任务对应。</li>
<li>使用 prediction 的 <code>frame_index</code> 索引 NPY 行，并核验原始 <code>Frame_Number</code>。原始帧号不直接作为行号。当前部分时间戳异常，因此不用于匹配。</li>
<li>直接读取原始 NPY 以保留 <code>stage</code>。A、C 的空 stage 和 <code>none</code> 不参与 stage JSD，但有效 gaze 帧仍进入整体距离。B、D 没有 stage 字段，因此本次不计算其 stage JSD。</li>
<li>当前 <code>User7_1_1</code>、<code>User9_7_1</code>、<code>User11_6_1</code>、<code>User12_3_1</code> 的 GT 坐标与 prediction 缓存 target 不同。本次以当前 NPY 为准，不复用旧日志中的指标。</li>
<li>single_task 与 leave_out 按测试任务选择 A/B/C/D 对应模型；spd_finetune 的三个 seed 先算指标后平均，没有平均概率图来构建集成模型。</li></ul>

<h2>下载数据表</h2><p>{downloads}</p>
<footer>分析范围：novice active gaze。未分析 passive、skill/performance 分层或训练 seed 的普遍效应。</footer>
</main></body></html>"""
    (output / "report.html").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).parent
    default_output = script_dir if (script_dir / "model_comparison.csv").exists() else script_dir / "deliverable"
    parser.add_argument("output", nargs="?", type=Path, default=default_output)
    parser.add_argument("--report-only", action="store_true", help="Refresh report links/text without regenerating verified figures")
    args = parser.parse_args()
    output = args.output.resolve()
    summary = pd.read_csv(output / "model_comparison.csv")
    video = pd.read_csv(output / "video_distance.csv")
    stage = pd.read_csv(output / "stage_jsd_summary.csv")
    assert set(summary.model) == set(MODELS)
    assert video.video_id.nunique() == 15
    assert set(stage.task) == {"A", "C"}
    assert summary["n_frames"].nunique() == 1
    assert stage.mean_jsd.between(0, 1).all()
    if not args.report_only:
        apply_style()
        distance_plot(summary, video, output)
        video_heatmap(video, output)
        stage_heatmap(stage, output)
    build_report(summary, video, stage, output)
    print(json.dumps({"report": str(output / "report.html"), "figures": ["distance_comparison", "distance_by_video", "stage_jsd_comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
