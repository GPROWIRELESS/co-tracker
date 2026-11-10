#!/usr/bin/env python3
"""Draw the GT84 score analysis from saved tables; no inference is recomputed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

STAGES = {
    "A": ["reach ring", "grab ring", "transfer ring", "place ring", "drop ring"],
    "C": ["reach string", "threading string through tower", "wrap around psm", "pull through and cinch"],
}
SHORT = {"threading string through tower": "Threading string", "pull through and cinch": "Pull through / cinch", "wrap around psm": "Wrap around PSM"}
METHODS = ["between_participants", "within_participant", "within_adjusted_order"]
METHOD_LABELS = {"between_participants": "Between participants", "within_participant": "Within participant", "within_adjusted_order": "Within, order adjusted"}
SCENARIOS = ["primary", "completed_only", "both_in_stage", "velocity", "exclude_boundary"]
SCENARIO_LABELS = {"primary": "Primary position", "completed_only": "Completed only", "both_in_stage": "Both in stage", "velocity": "Velocity", "exclude_boundary": "Exclude boundary peaks"}
PRED_LABELS = {"peak_r": "Peak gaze-PSM correlation", "r_zero": "Zero-lag gaze-PSM correlation"}
BLUE = "#236A92"
ORANGE = "#C76A26"
GRAY = "#687582"
OUTPUTS = []


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def truth(v):
    return str(v).lower() in ("true", "1", "1.0")


def label(stage):
    return SHORT.get(stage, stage.title())


def conditions(task=None):
    return [(t, s, side) for t in ([task] if task else ["A", "C"]) for s in STAGES[t] for side in ["l", "r"]]


def cond_key(row):
    return row["task"], row["stage"], row["side"]


def assoc_key(row):
    return row["scenario"], row["task"], row["stage"], row["side"], row["predictor"], row["method"]


def fmt(value, digits=3):
    x = num(value)
    return f"{x:.{digits}f}" if np.isfinite(x) else "NA"


def qfmt(value):
    q = num(value)
    return "NA" if not np.isfinite(q) else ("<0.001" if q < .001 else f"{q:.3f}")


def get_assoc(lookup, scenario, condition, predictor, method):
    return lookup.get((scenario, *condition, predictor, method), {})


def association_text(row):
    n = int(num(row.get("n_users", 0))) if np.isfinite(num(row.get("n_users", 0))) else 0
    if row.get("status") != "ok" or not np.isfinite(num(row.get("association"))):
        return f"Not estimable | participants={n}"
    qualifier = "*" if not truth(row.get("ci_reliable", True)) else ""
    return f"r={fmt(row['association'])} | 95% range [{fmt(row.get('ci_low'))}, {fmt(row.get('ci_high'))}]{qualifier}\nq={qfmt(row.get('q_bh'))} | participants={n}"


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#DFE5E9", linewidth=.5, alpha=.8)
    ax.set_axisbelow(True)


def save(fig, root, all_pdf, stem, title, meaning, inputs):
    paths = []
    for ext in ("png", "pdf"):
        p = root / "figures" / f"{stem}.{ext}"
        fig.savefig(p, dpi=180, facecolor="white", bbox_inches="tight")
        paths.append(str(p.relative_to(root)).replace("\\", "/"))
    all_pdf.savefig(fig, facecolor="white", bbox_inches="tight")
    OUTPUTS.append({"stem": stem, "title": title, "meaning": meaning, "inputs": inputs, "files": paths})
    plt.close(fig)


def draw_scores(rows, root, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), gridspec_kw={"height_ratios": [1, 1.35]})
    states = {"Completed": (BLUE, "o"), "Failed": (ORANGE, "x")}
    for col, task in enumerate(["A", "C"]):
        rr = [r for r in rows if r["task"] == task]
        valid = [r for r in rr if np.isfinite(num(r.get("Score")))]
        values = np.array([num(r["Score"]) for r in valid])
        ax = axes[0, col]
        upper = max(100, np.nanmax(values) if len(values) else 100)
        bins = np.linspace(0, upper, 11)
        hist_data = [[num(r["Score"]) for r in valid if r["AttemptStatus"] == status] for status in states]
        ax.hist(hist_data, bins=bins, stacked=True, color=[BLUE, ORANGE], label=list(states), edgecolor="white")
        ax.set_title(f"Task {task}: {len(valid)}/{len(rr)} scores available", loc="left", weight="bold")
        ax.text(.97, .96, f"Zero scores: {np.sum(values == 0)}\nMissing: {len(rr)-len(valid)}", transform=ax.transAxes, ha="right", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
        ax.set(xlabel="Score (source values)", ylabel="Recordings")
        ax.legend(frameon=False, loc="upper left")
        style_ax(ax)
        ax = axes[1, col]
        users = sorted({r["user"] for r in rr}, key=lambda x: int(x.removeprefix("User")))
        for j, user in enumerate(users):
            ur = [r for r in valid if r["user"] == user]
            jitter = np.linspace(-.19, .19, len(ur)) if len(ur) > 1 else [0]
            for d, r in zip(jitter, ur):
                c, marker = states.get(r["AttemptStatus"], (GRAY, "s"))
                ax.scatter(j+d, num(r["Score"]), c=c, marker=marker, s=32, linewidths=1, alpha=.85)
        ax.set_xticks(range(len(users)), [u.replace("User", "") for u in users])
        ax.set(xlabel="Participant ID", ylabel="Score", ylim=(-5, upper+5))
        ax.set_title(f"Task {task}: all scored recordings", loc="left")
        style_ax(ax)
    fig.suptitle("GT84 / scoring coverage and score distributions", x=.06, ha="left", weight="bold", fontsize=15)
    fig.text(.06, .012, "One point = one recording. Genuine zero scores are retained; missing scores are not set to zero. Before stage-specific support filtering.", fontsize=9)
    fig.tight_layout(rect=[.02, .035, 1, .95], h_pad=2)
    save(fig, root, pdf, "01_score_distribution", "Score distribution and completion status", "Counts and score distributions before stage filtering; preserves genuine zero scores and distinguishes missing values.", ["tables/matched_scores.csv"])


def draw_coverage(rows, root, pdf):
    lookup = {cond_key(r): r for r in rows if r["scenario"] == "primary" and r["predictor"] == "peak_r"}
    cc = conditions()
    values = np.array([[num(lookup.get(c, {}).get(k, 0)) for k in ("n_total_records", "n_score_available", "n_eligible_records", "n_users")] for c in cc])
    fig, ax = plt.subplots(figsize=(9, 10.3))
    normalized = values / np.array([51, 51, 51, 11])
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(cc)), [f"{t} / {label(s)} / {side.upper()}" for t,s,side in cc])
    ax.set_xticks(range(4), ["Available stage\nrecords", "With usable\nScore", "Eligible\nrecords", "Eligible\nparticipants"])
    for (i,j), val in np.ndenumerate(values):
        ax.text(j, i, str(int(val)) if np.isfinite(val) else "NA", ha="center", va="center", color="white" if normalized[i,j]>.55 else "#22303D", fontsize=10)
    ax.axhline(9.5, color="white", lw=3)
    ax.set_title("Primary position / peak-r analysis coverage", loc="left", weight="bold", fontsize=14, pad=18)
    fig.text(.04, .016, "Counts shown explicitly. Color uses separate record (0–51) and participant (0–11) scales.\nFixed left/right sides; complete lag support and usable Score are required. Zero coverage remains visible.", fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1])
    save(fig, root, pdf, "02_primary_coverage", "Primary condition coverage", "All predefined task/stage/side conditions with record and participant counts; zero-support stages are retained.", ["tables/coverage.csv"])


def user_palette(rows):
    users = sorted({r["user"] for r in rows}, key=lambda x: int(x.removeprefix("User")))
    return {u: plt.get_cmap("tab20")(i) for i,u in enumerate(users)}


def panel_grid(task, title):
    fig, axes = plt.subplots(len(STAGES[task]), 2, figsize=(11.8, 2.8*len(STAGES[task])), squeeze=False)
    fig.suptitle(title, x=.065, ha="left", fontsize=15, weight="bold", y=.995)
    return fig, axes


def draw_scatter(points, lookup, root, pdf):
    palette = user_palette(points)
    for task in ["A", "C"]:
        for predictor in ["peak_r", "r_zero"]:
            rr = [r for r in points if r["scenario"]=="primary" and r["task"]==task and r["predictor"]==predictor]
            title = f"Task {task} / {PRED_LABELS[predictor]} vs Score"
            fig, axes = panel_grid(task, title)
            for i, stage in enumerate(STAGES[task]):
                for j, side in enumerate(["l", "r"]):
                    ax = axes[i,j]
                    pp = [r for r in rr if r["stage"]==stage and r["side"]==side]
                    result = get_assoc(lookup, "primary", (task,stage,side), predictor, "between_participants")
                    for r in pp:
                        x,y = num(r["x"]),num(r["Score"])
                        if np.isfinite(x+y):
                            ax.scatter(x,y,color=palette[r["user"]],s=34,zorder=3,edgecolors="white",linewidths=.35)
                            ax.annotate(r["user"].replace("User", ""),(x,y),xytext=(3,3),textcoords="offset points",fontsize=7.5)
                    ax.set_title(f"{label(stage)} / {side.upper()}",loc="left",weight="bold",fontsize=10)
                    ax.text(.02,.98,association_text(result),transform=ax.transAxes,ha="left",va="top",fontsize=8,bbox={"facecolor":"white","edgecolor":"none","alpha":.82})
                    if not pp:
                        ax.set_facecolor("#F3F5F6")
                        ax.text(.5,.42,"No eligible participant data",transform=ax.transAxes,ha="center",color=GRAY)
                    ax.set_ylabel("Mean Score")
                    ax.set_xlabel("Fisher-z mean peak r" if predictor=="peak_r" else "Fisher-z mean zero-lag r")
                    ax.margins(x=.15,y=.4)
                    style_ax(ax)
            fig.text(.065,.008,"One point = one participant; labels are participant IDs. Correlation and Score use the same eligible recordings.\nSaved Spearman inference; q is family-specific. * Fewer than 97.5% valid bootstrap draws: interval unreliable.",fontsize=9)
            fig.tight_layout(rect=[.01,.035,1,.97],h_pad=1.8,w_pad=2)
            save(fig,root,pdf,f"03_scatter_{task}_{predictor}",title,"Participant-level primary scatter plots for every predefined stage and fixed side. No regression confidence bands are inferred from the display.",["tables/participant_summaries.csv","tables/associations.csv"])


def draw_within(points, lookup, root, pdf):
    palette=user_palette(points)
    for task in ["A","C"]:
        for predictor in ["peak_r","r_zero"]:
            for method in METHODS[1:]:
                rr=[r for r in points if r["scenario"]=="primary" and r["task"]==task and r["predictor"]==predictor and r["method"]==method]
                title=f"Task {task} / {predictor} / {METHOD_LABELS[method]}"
                fig,axes=panel_grid(task,title)
                for i,stage in enumerate(STAGES[task]):
                    for j,side in enumerate(["l","r"]):
                        ax=axes[i,j]
                        pp=[r for r in rr if r["stage"]==stage and r["side"]==side]
                        result=get_assoc(lookup,"primary",(task,stage,side),predictor,method)
                        for r in pp:
                            x,y=num(r["x_residual"]),num(r["y_residual"])
                            if np.isfinite(x+y):
                                ax.scatter(x,y,s=20+110*num(r.get("weight",0)),c=[palette[r["user"]]],alpha=.68,edgecolors="white",linewidths=.3)
                        ax.axhline(0,c=GRAY,lw=.7)
                        ax.axvline(0,c=GRAY,lw=.7)
                        ax.set_title(f"{label(stage)} / {side.upper()}",loc="left",weight="bold",fontsize=10)
                        ax.text(.02,.98,association_text(result),transform=ax.transAxes,ha="left",va="top",fontsize=8,bbox={"facecolor":"white","edgecolor":"none","alpha":.86})
                        if not pp:
                            ax.set_facecolor("#F3F5F6")
                            ax.text(.5,.42,"No residual data",transform=ax.transAxes,ha="center",color=GRAY)
                        ax.set(xlabel="Correlation rank residual",ylabel="Score rank residual")
                        ax.margins(x=.12,y=.42)
                        style_ax(ax)
                fig.text(.065,.008,"One point = one recording; color = participant. Analysis gives each participant equal total weight.\nRanks are residualized for participant effects"+(" and task order / attempt." if method=="within_adjusted_order" else ".")+" q is family-specific; small-cluster inference is exploratory.",fontsize=9)
                fig.tight_layout(rect=[.01,.035,1,.97],h_pad=1.8,w_pad=2)
                save(fig,root,pdf,f"04_within_{task}_{predictor}_{method}",title,"Recording-level residual plots; association estimates and uncertainty are read from saved fixed-effects results.",["tables/within_points.csv","tables/associations.csv"])


def point_interval(ax,row,y,color):
    a,lo,hi=(num(row.get(k)) for k in ["association","ci_low","ci_high"])
    if row.get("status")!="ok" or not np.isfinite(a):
        return False
    reliable=truth(row.get("ci_reliable",True))
    if np.isfinite(lo+hi):
        ax.plot([lo,hi],[y,y],color=color,lw=1.2,linestyle="-" if reliable else ":",alpha=.8)
    significant=np.isfinite(num(row.get("q_bh"))) and num(row.get("q_bh"))<.05
    ax.scatter(a,y,s=33,marker="D" if significant else "o",color=color,edgecolors="white",linewidths=.4,zorder=4)
    return True


def draw_forest(lookup,root,pdf):
    cc=conditions()
    for predictor in ["peak_r","r_zero"]:
        fig,axes=plt.subplots(1,3,figsize=(15.5,10.6),sharey=True)
        for j,method in enumerate(METHODS):
            ax=axes[j]
            for i,c in enumerate(cc):
                r=get_assoc(lookup,"primary",c,predictor,method)
                if not point_interval(ax,r,i,BLUE if j==0 else ORANGE):
                    ax.text(0,i,"NA",ha="center",va="center",color=GRAY,fontsize=8)
                else:
                    ax.text(1.08,i,qfmt(r.get("q_bh")),ha="left",va="center",fontsize=7.8)
            ax.axvline(0,color=GRAY,lw=.8)
            ax.axhline(9.5,color="#BDC7CE",lw=1)
            ax.set(xlim=(-1.06,1.28),xticks=[-1,-.5,0,.5,1],xlabel="Association coefficient",title=METHOD_LABELS[method])
            ax.text(1.08,-1,"q",ha="left",va="bottom",weight="bold",fontsize=8)
            ax.set_yticks(range(len(cc)),[f"{t} / {label(s)} / {side.upper()}" for t,s,side in cc])
            ax.set_ylim(len(cc)-.4,-1.25)
            style_ax(ax)
        fig.suptitle(f"Primary position / {PRED_LABELS[predictor]} and Score",x=.035,ha="left",fontsize=15,weight="bold")
        fig.text(.035,.018,"Bars: 95% participant-bootstrap percentile intervals; dotted bars are unreliable (<97.5% valid draws). Diamonds: family q < 0.05.\nNA means not estimable, not zero association. Each method/predictor has a separate correction family across tasks, stages and sides.",fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.96],w_pad=2)
        save(fig,root,pdf,f"05_forest_primary_{predictor}",f"Primary association forest: {predictor}","All predefined fixed-side conditions and the three analysis methods; includes null/negative estimates and non-estimable conditions.",["tables/associations.csv"])


def draw_sensitivity(lookup,root,pdf):
    colors=["#153E59","#3A869C","#C68139","#8776A5","#64734A"]
    for task in ["A","C"]:
        cc=conditions(task)
        for predictor in ["peak_r","r_zero"]:
            for method in ["between_participants","within_adjusted_order"]:
                fig,ax=plt.subplots(figsize=(12,1.0*len(cc)+2.2))
                for i,c in enumerate(cc):
                    if i%2==0:ax.axhspan(i-.49,i+.49,color="#F3F5F6",zorder=0)
                    for j,scenario in enumerate(SCENARIOS):
                        y=i+(j-2)*.16
                        r=get_assoc(lookup,scenario,c,predictor,method)
                        if not point_interval(ax,r,y,colors[j]):
                            ax.text(1.08,y,"NA",color=colors[j],fontsize=6.5,va="center")
                for c,s in zip(colors,SCENARIOS):ax.plot([],[],color=c,marker="o",label=SCENARIO_LABELS[s])
                ax.axvline(0,color=GRAY,lw=.8)
                ax.set(yticks=range(len(cc)),yticklabels=[f"{label(s)} / {side.upper()}" for _,s,side in cc],xlim=(-1.06,1.22),xticks=[-1,-.5,0,.5,1],ylim=(len(cc)-.4,-.6),xlabel="Association coefficient")
                ax.set_title(f"Task {task} / {predictor} / {METHOD_LABELS[method]}\nPrimary and sensitivity analyses",loc="left",fontsize=14,weight="bold",pad=14)
                ax.legend(loc="upper center",bbox_to_anchor=(.45,-.07),ncol=3,frameon=False,fontsize=9)
                style_ax(ax)
                fig.text(.04,.012,"Participant-bootstrap intervals. Dotted intervals: unreliable. Diamonds: q < 0.05 in that scenario's family.\nScenario samples may differ; this display is a robustness comparison, not a formal test of scenario differences.",fontsize=9)
                fig.tight_layout(rect=[0,.07,1,1])
                save(fig,root,pdf,f"06_sensitivity_{task}_{predictor}_{method}",f"Task {task} sensitivity: {predictor}, {method}","Predefined primary and sensitivity estimates plotted together without pooling their inference or correction families.",["tables/associations.csv"])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.root.resolve()
    (root/"figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"pdf.fonttype":42,"ps.fonttype":42})
    inputs={name:root/"tables"/f"{name}.csv" for name in ["associations","participant_summaries","matched_scores","coverage","within_points"]}
    data={k:read(v) for k,v in inputs.items()}
    lookup={assoc_key(r):r for r in data["associations"]}
    assert len(lookup)==len(data["associations"]),"Duplicate association keys"
    with PdfPages(root/"figures"/"all_figures.pdf") as pdf:
        draw_scores(data["matched_scores"],root,pdf)
        draw_coverage(data["coverage"],root,pdf)
        draw_scatter(data["participant_summaries"],lookup,root,pdf)
        draw_within(data["within_points"],lookup,root,pdf)
        draw_forest(lookup,root,pdf)
        draw_sensitivity(lookup,root,pdf)
    manifest={"generator":"code/plot_results.py","inference_recomputed":False,"input_sha256":{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.values()},"n_figure_pages":len(OUTPUTS),"combined_pdf":"figures/all_figures.pdf","figures":OUTPUTS,"software":{"matplotlib":matplotlib.__version__,"numpy":np.__version__}}
    (root/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"figures":len(OUTPUTS),"combined_pdf":manifest["combined_pdf"]}))


if __name__=="__main__":
    main()
