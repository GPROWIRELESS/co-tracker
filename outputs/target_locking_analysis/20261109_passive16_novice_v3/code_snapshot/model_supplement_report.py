"""Paired model confusion and GT-smoothed diagnostics for passive16.

Statistics retain the same content -> source participant -> group hierarchy.
All figures show active and passive on identical axes and color scales.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report_passive16 import aggregate_pairs, groups, ParticipantBootstrap, COLORS, STAGES

CONTEXT = ["analysis_branch", "model", "task", "scenario", "stage"]
REGIONS = ["T", "P", "O", "N"]
REGION_LABELS = ["Target only", "Wrist only", "Overlap", "Other"]
DOMAINS = ["active", "passive"]


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NA")


def confusion_summary(df):
    df = df.copy()
    if "reference_region" not in df:
        df["reference_region"] = df["active_region"]
    keys = CONTEXT + ["reference_region", "model_region"]
    content_rows = []
    for key, part in groups(df, keys + ["participant", "content_id"]):
        row = dict(key, n_pairs_total=part.pair_id.nunique())
        for metric in ("joint_share", "conditional_share"):
            cols = [d+"_"+metric for d in DOMAINS]
            use = part.loc[np.isfinite(part[cols].to_numpy(float)).all(axis=1)]
            row["n_pairs_"+metric] = len(use)
            for d in DOMAINS:
                row[d+"_"+metric] = use[d+"_"+metric].mean()
            row["delta_"+metric] = (use["passive_"+metric]-use["active_"+metric]).mean()
        content_rows.append(row)
    content = pd.DataFrame(content_rows)
    people = []
    for key, part in groups(content, keys + ["participant"]):
        row = dict(key)
        for metric in ("joint_share", "conditional_share"):
            cols = [d+"_"+metric for d in DOMAINS]
            use = part.loc[np.isfinite(part[cols].to_numpy(float)).all(axis=1)]
            row["n_contents_"+metric] = len(use)
            row["n_pairs_"+metric] = int(use["n_pairs_"+metric].sum())
            for d in [*DOMAINS, "delta"]:
                row[d+"_"+metric] = use[d+"_"+metric].mean()
        people.append(row)
    people = pd.DataFrame(people)
    summary = []
    for key, part in groups(people, keys):
        row = dict(key)
        for metric in ("joint_share", "conditional_share"):
            cols = [d+"_"+metric for d in DOMAINS]
            use = part.loc[np.isfinite(part[cols].to_numpy(float)).all(axis=1)]
            row["n_participants_"+metric] = len(use)
            row["n_contents_"+metric] = int(use["n_contents_"+metric].sum())
            row["n_pairs_"+metric] = int(use["n_pairs_"+metric].sum())
            for d in [*DOMAINS, "delta"]:
                row[d+"_"+metric] = use[d+"_"+metric].mean()
        summary.append(row)
    summary = pd.DataFrame(summary)
    for frame in (content, people, summary):
        frame["active_region"] = frame["reference_region"]  # compatibility alias
    return content, people, summary


def plot_matrix(group, key, out, index):
    branch, model, task = key["analysis_branch"], key["model"], key["task"]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    for row, metric in enumerate(["joint_share", "conditional_share"]):
        for col, d in enumerate(DOMAINS):
            ax = axs[row, col]
            matrix = group.pivot(index="reference_region", columns="model_region", values=d+"_"+metric).reindex(index=REGIONS, columns=REGIONS).to_numpy(float)
            im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues")
            for i in range(4):
                for j in range(4):
                    value = matrix[i, j]
                    ax.text(j, i, f"{value*100:.1f}%" if np.isfinite(value) else "NA",
                            ha="center", va="center", fontsize=10,
                            color="white" if value>.55 else "#252525")
            ax.set_xticks(range(4), REGION_LABELS, rotation=20, ha="right")
            ax.set_yticks(range(4), REGION_LABELS)
            ax.set_xlabel("Model-decoded region")
            ax.set_ylabel(f"{d.capitalize()} gaze region")
            n = int(group["n_participants_"+metric].max())
            label = "Joint time share" if row == 0 else "Conditional on gaze region"
            ax.set_title(f"{d.capitalize()} reference | {label}\n{n} source operator(s)")
            fig.colorbar(im, ax=ax, fraction=.043, pad=.025)
    fig.suptitle(f"Task {task} | {model} | {branch}\nMain 50 px radii, whole included stages; paired frame support", fontsize=13)
    rel = Path("figures/model_supplement") / branch / f"{model}_task_{task}_confusion.png"
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    index.append(dict(path=rel.as_posix(), task=task, analysis_branch=branch, model=model,
                      description="Active/passive reference joint and conditional model-region matrices"))


def plot_gt(long, task, out, index):
    metrics = ["L_target", "L_gripper", "C", "C_excess"]
    labels = ["Target occupancy", "Wrist-region occupancy", "Exclusive target preference C", "Area-corrected preference"]
    stages = ["whole", *STAGES[task]]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for ax, metric, label in zip(axs.flat, metrics, labels):
        for j, d in enumerate(DOMAINS):
            part = long[long.task.eq(task) & long.scenario.eq("main") & long.metric.eq(metric) & long.domain.eq(d)].set_index("stage").reindex(stages)
            x = np.arange(len(stages)) + (j-.5)*.14
            vals = part["mean"].to_numpy(float)
            ax.plot(x, vals, "o-", color=COLORS[d], label=d.capitalize(), markersize=5, linewidth=1)
            lo, hi = part.ci_low.to_numpy(float), part.ci_high.to_numpy(float)
            good = np.isfinite(vals) & np.isfinite(lo) & np.isfinite(hi)
            ax.vlines(x[good], lo[good], hi[good], color=COLORS[d], lw=1.2)
            for pos, value, n in zip(x, vals, part.n_participants):
                if np.isfinite(value):
                    ax.annotate(f"n={int(n)}", (pos, value), xytext=(0, 8 if j else -14), textcoords="offset points", ha="center", fontsize=6, color=COLORS[d])
        unavailable = long[long.task.eq(task) & long.scenario.eq("main") & long.metric.eq(metric) & long.domain.eq("active")].set_index("stage").reindex(stages)["mean"].isna()
        for pos in np.flatnonzero(unavailable.to_numpy()):
            ax.text(pos, .08, "NA", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#777777")
        ax.margins(y=.18)
        ax.set_xticks(np.arange(len(stages)), [s.replace(" ", "\n") for s in stages], fontsize=8)
        ax.set_xlim(-.4, len(stages)-.6)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.15)
    axs.flat[0].legend()
    fig.suptitle(f"Task {task} | GT-smoothed diagnostic, active and passive\nGaze-centered sigma=1 grid kernels; not an independent predictor or upper bound", fontsize=13)
    rel = Path("figures/model_supplement") / f"task_{task}_gt_smoothed.png"
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / rel, dpi=160, facecolor="white")
    plt.close(fig)
    index.append(dict(path=rel.as_posix(), task=task, analysis_branch="GT_smoothed_diagnostic", model="",
                      description="Paired active/passive GT-smoothed occupancy diagnostic; no BSS baseline"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE.parent / "output")
    args = parser.parse_args()
    out = args.output
    conf = pd.read_csv(out / "model_confusions.csv")
    content, person, group = confusion_summary(conf)
    save_csv(content, out / "model_confusion_content_summary.csv")
    save_csv(person, out / "model_confusion_participant_summary.csv")
    save_csv(group, out / "model_confusion_group_summary.csv")
    gt = pd.read_csv(out / "model_gt_smoothed.csv")
    gc, gp, gg, gl, _ = aggregate_pairs(gt, ParticipantBootstrap(42, 10000))
    for label, frame in (("content", gc), ("participant", gp), ("group", gg), ("group_long", gl)):
        save_csv(frame, out / f"model_gt_smoothed_{label}_summary.csv")
    index = []
    whole = group[group.scenario.eq("main") & group.stage.eq("whole")]
    for key, part in groups(whole, ["analysis_branch", "model", "task"]):
        plot_matrix(part, key, out, index)
    for task in sorted(gt.task.unique()):
        plot_gt(gl, task, out, index)
    save_csv(pd.DataFrame(index), out / "model_supplement_figure_index.csv")
    lines = ["# 模型补充：区域对应与 GT 平滑诊断", "",
        "全部结果限定为本次16对中的原模型测试交集：Task A 2对/2位源操作者，Task C 1对/1位源操作者。",
        "Task C 只有一个独立源操作者，均值区间为NA。每项数据均提供 active/passive 对应结果。", "",
        "## 区域对应矩阵", "",
        "每个配对先按共同有效时间归一化，再按内容内observer、源操作者内内容、操作者依次等权。"
        "条件矩阵先对每个配对的实测类别归一化；只有两类gaze对该行均有支持的配对才进入该条件对照，零分母保留NA。"
        "矩阵行是实测gaze区域，列是模型解码区域；上排显示joint share，下排显示条件比例，左为active、右为passive。", "",
        "表中 `reference_region` 是两类参照共用的区域代码索引：`active_*` 列按 active gaze 的该区域统计，"
        "`passive_*` 列按 passive gaze 的该区域统计。`active_region` 仅为兼容旧读取代码保留的同值别名，"
        "不表示 passive 列使用 active gaze 的分类。", "",
        "[内容汇总](model_confusion_content_summary.csv) · [操作者汇总](model_confusion_participant_summary.csv) · [组汇总](model_confusion_group_summary.csv)", "",
        "## GT-smoothed诊断", "",
        "使用原v3的sigma=1网格核，分别由active/passive实测gaze构造，仅作标签平滑诊断。"
        "它不是独立预测器、严格上界或BSS参照；不加入模型排名。", "",
        "[内容汇总](model_gt_smoothed_content_summary.csv) · [操作者汇总](model_gt_smoothed_participant_summary.csv) · [组汇总](model_gt_smoothed_group_summary.csv)", "",
        "## 图件", ""]
    for row in index:
        lines += [f"### Task {row['task']} · {row['analysis_branch']} · {row['model']}", "", f"![{row['description']}]({row['path']})", ""]
    (out / "model_supplement_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Completed supplement: {len(group)} confusion summary cells, {len(index)} paired figures.", flush=True)


if __name__ == "__main__":
    main()
