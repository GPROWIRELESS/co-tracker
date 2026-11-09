"""Paired active/passive reporting, using the frozen v3 participant statistics.

Run after measurement: python report_passive16.py --output-dir ../output
The measurement kernel is deliberately not reimplemented here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "v3"))
from tl_stats import ParticipantBootstrap, corr
from tl_common import STAGES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOMAINS = ("active", "passive")
COLORS = {"active": "#2563a6", "passive": "#cf6239", "delta": "#74499a", "model": "#319183"}
CORE = ["L_target", "L_gripper", "C", "D"]
AREA_METRICS = ["C_excess", "E_target", "E_gripper", "B", "B_excess", "C_U"]
TEMPORAL = ["dwell_T_ms", "dwell_P_ms", "dwell_O_ms", "direct_rate", "via_rate"]
LABELS = {"L_target": "Target occupancy", "L_gripper": "Wrist-region occupancy",
          "C": "Exclusive target preference C", "D": "Exclusive occupancy D",
          "C_excess": "Area-corrected preference", "E_target": "Excess target occupancy",
          "E_gripper": "Excess wrist occupancy", "B": "Exclusive balance B",
          "B_excess": "Area-corrected balance", "C_U": "Area-only preference C_U",
          "dwell_T_ms": "Target-only dwell (ms)", "dwell_P_ms": "Wrist-only dwell (ms)",
          "dwell_O_ms": "Overlap dwell (ms)", "direct_rate": "Direct switches / min",
          "via_rate": "Switches via overlap / min"}
REGIONS = ["T", "P", "O", "N"]
REGION_NAMES = ["Target only", "Wrist only", "Overlap", "Other"]
REGION_COLORS = ["#238975", "#c6733c", "#826cad", "#c9ced0"]
SCENARIOS = ["main", "r25", "r100", "gripper100", "smooth"]
CONTEXT = ["task", "scenario", "stage"]
SUMMARY_METRICS = {"L_T", "L_P", "L_O", "L_N", "L_target", "L_gripper", "D", "C",
                   "C_U", "C_excess", "E_target", "E_gripper", "B", "B_excess",
                   "area_T", "area_P", "area_O", "area_N", *TEMPORAL}


class CachedParticipantBootstrap(ParticipantBootstrap):
    """Memoize identical requests without changing the frozen v3 calculation."""
    def __init__(self, seed, nboot):
        super().__init__(seed, nboot)
        self.mean_cache = {}

    def mean(self, values):
        v = np.asarray(values, float)
        key = tuple(v[np.isfinite(v)])
        if key not in self.mean_cache:
            self.mean_cache[key] = super().mean(v)
        return dict(self.mean_cache[key])


def read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NA")


def groups(frame, keys):
    if frame.empty:
        return
    for key, part in frame.groupby(keys, sort=True, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        yield dict(zip(keys, key)), part


def finite_pair(frame, metric, model=False):
    columns = [f"active_{metric}", f"passive_{metric}"]
    if model and f"model_{metric}" in frame:
        columns.append(f"model_{metric}")
    return np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)


def paired_metrics(frame):
    result = []
    for col in frame:
        if not col.startswith("active_"):
            continue
        metric = col[len("active_"):]
        if f"passive_{metric}" not in frame or metric.endswith("reason"):
            continue
        for prefix in ("active", "passive", "model"):
            c = f"{prefix}_{metric}"
            if c in frame:
                frame[c] = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        result.append(metric)
    return result


def metadata(part):
    out = {}
    for col in ("round", "score", "status"):
        if col in part:
            vals = part[col].dropna().unique()
            if len(vals) > 1:
                raise ValueError(f"Inconsistent source-content {col}: {vals}")
            out[col] = vals[0] if len(vals) else np.nan
    return out


def aggregate_pairs(frame, boot, extra=(), model=False):
    """Equal observers within content, equal content within operator, equal operators.

    Each metric selects finite A/P pairs before either member is averaged. Thus
    the active mean uses exactly the observer-specific frame masks of passive.
    """
    frame = frame.copy()
    metrics = [m for m in paired_metrics(frame) if m in SUMMARY_METRICS or
               (model and m.startswith(("BS_", "delta_", "abs_delta_", "FPR_", "FNR_")))]
    context = list(extra) + CONTEXT
    values, observed = {}, {}
    for metric in metrics:
        ok = finite_pair(frame, metric, model)
        domains = [*DOMAINS] + (["model"] if model and f"model_{metric}" in frame else [])
        for domain in domains:
            values[f"{domain}_{metric}"] = frame[f"{domain}_{metric}"].where(ok)
        values[f"delta_{metric}"] = (frame[f"passive_{metric}"] - frame[f"active_{metric}"]).where(ok)
        observed[f"n_observers_{metric}"] = frame.observer_id.where(ok)
    value_columns = list(values)
    val = pd.DataFrame(values, index=frame.index)
    work = pd.concat([frame[context + ["participant", "content_id", "pair_id", "observer_id"]], val], axis=1)
    content_keys = context + ["participant", "content_id"]
    grouped = work.groupby(content_keys, sort=True, dropna=False, observed=True)
    means = grouped[value_columns].mean()
    counts = grouped[[f"active_{m}" for m in metrics]].count().rename(columns={f"active_{m}": f"n_pairs_{m}" for m in metrics})
    obs = pd.concat([frame[content_keys], pd.DataFrame(observed, index=frame.index)], axis=1)
    observers = obs.groupby(content_keys, sort=True, dropna=False)[list(observed)].nunique()
    totals = grouped.agg(n_pairs_total=("pair_id", "nunique"), n_observers_total=("observer_id", "nunique"))
    content = means.join(counts).join(observers).join(totals).reset_index()
    meta_cols = [c for c in ("round", "score", "status") if c in frame]
    if meta_cols:
        mg = frame.groupby(content_keys, sort=True, dropna=False)[meta_cols]
        if (mg.nunique(dropna=True) > 1).any().any():
            raise ValueError("Source-content metadata is not unique")
        content = content.merge(mg.first().reset_index(), on=content_keys, validate="one_to_one")
    person_keys = context + ["participant"]
    pg = content.groupby(person_keys, sort=True, dropna=False)
    persons = pg[value_columns].mean()
    pc = pg[[f"active_{m}" for m in metrics]].count().rename(columns={f"active_{m}": f"n_contents_{m}" for m in metrics})
    ps = pg[[f"n_pairs_{m}" for m in metrics]].sum()
    pt = pg.agg(n_contents_total=("content_id", "nunique"), n_pairs_total=("n_pairs_total", "sum"))
    persons = persons.join(pc).join(ps).join(pt).reset_index()
    totals = work.groupby(context, sort=True, dropna=False).agg(n_contents_total=("content_id", "nunique"),
            n_pairs_total=("pair_id", "nunique"), n_observers_total=("observer_id", "nunique"))
    observer_counts = obs.groupby(context, sort=True, dropna=False)[list(observed)].nunique()
    summary_rows, long_rows = [], []
    for key, part in groups(persons, context):
        ix = tuple(key[c] for c in context)
        row = dict(key, n_participants_total=part.participant.nunique(), **totals.loc[ix].to_dict())
        matrix = part[value_columns].to_numpy(float)
        column_index = {c: i for i, c in enumerate(value_columns)}
        for metric in metrics:
            ok = np.isfinite(matrix[:, column_index[f"active_{metric}"]]) & np.isfinite(matrix[:, column_index[f"passive_{metric}"]])
            support = dict(n_participants=int(ok.sum()), n_contents=int(part[f"n_contents_{metric}"].to_numpy()[ok].sum()),
                           n_pairs=int(part[f"n_pairs_{metric}"].to_numpy()[ok].sum()),
                           n_observers=int(observer_counts.loc[ix, f"n_observers_{metric}"]))
            for name, value in support.items():
                row[f"{name}_{metric}"] = value
            domains = [*DOMAINS, "delta"] + (["model"] if model and f"model_{metric}" in column_index else [])
            for domain in domains:
                arr = matrix[ok, column_index[f"{domain}_{metric}"]]
                ci = boot.mean(arr)
                avg = float(arr.mean()) if len(arr) else np.nan
                row[f"{domain}_{metric}"] = avg
                for name, value in ci.items():
                    row[f"{domain}_{metric}_{name}"] = value
                long_rows.append(dict(key, metric=metric, domain=domain, mean=avg, **ci, **support))
        summary_rows.append(row)
    return content, persons, pd.DataFrame(summary_rows), pd.DataFrame(long_rows), metrics


def observer_summary(frame, metrics):
    rows = []
    for key, part in groups(frame, CONTEXT + ["observer_id"]):
        row = dict(key, n_contents_total=part.content_id.nunique(), n_pairs_total=part.pair_id.nunique())
        for metric in metrics:
            use = part.loc[finite_pair(part, metric)]
            row[f"n_contents_{metric}"] = use.content_id.nunique()
            for domain in DOMAINS:
                row[f"{domain}_{metric}"] = use[f"{domain}_{metric}"].mean()
            row[f"delta_{metric}"] = (use[f"passive_{metric}"] - use[f"active_{metric}"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def performance(content, boot, metrics):
    rows, points, loo = [], [], []
    if "score" not in content:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for key, part in groups(content, CONTEXT):
        for metric in metrics:
            use = part.loc[finite_pair(part, metric) & np.isfinite(pd.to_numeric(part.score, errors="coerce"))]
            if "status" in use:
                use = use[use.status.astype(str).str.lower().isin(["completed", "failed"])]
            p = use.groupby("participant", sort=True).agg(
                score=("score", "mean"), active=(f"active_{metric}", "mean"),
                passive=(f"passive_{metric}", "mean"), n_contents=("content_id", "nunique")).reset_index()
            for rec in p.to_dict("records"):
                points.append(dict(key, metric=metric, **rec))
            n = len(p)
            row = dict(key, metric=metric, n_participants=n, n_contents=len(use), distinct_scores=p.score.nunique())
            for domain in DOMAINS:
                x, y = p.score.to_numpy(float), p[domain].to_numpy(float)
                rho = corr(x, y) if n >= 3 else np.nan
                ci = boot.spearman(x, y) if n >= 3 else boot.unavailable()
                row[f"{domain}_rho"] = rho
                row[f"{domain}_status"] = ("fewer_than_3_participants" if n < 3 else "constant_score"
                    if np.ptp(x) == 0 else "constant_metric" if np.ptp(y) == 0 else "defined")
                row.update({f"{domain}_{name}": value for name, value in ci.items()})
            row["delta_rho"] = row["passive_rho"] - row["active_rho"]
            for i, person in enumerate(p.participant):
                keep = np.arange(n) != i
                entry = dict(key, metric=metric, held_out=person, n_retained=int(keep.sum()))
                for domain in DOMAINS:
                    entry[f"{domain}_rho"] = corr(p.score.to_numpy()[keep], p[domain].to_numpy()[keep]) if keep.sum() >= 3 else np.nan
                loo.append(entry)
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(points), pd.DataFrame(loo)


def round_analysis(content, boot, metrics):
    rows, changes = [], []
    if "round" not in content:
        return pd.DataFrame(), pd.DataFrame()
    d = content.copy()
    d["round"] = pd.to_numeric(d["round"], errors="coerce")
    d = d[d["round"].isin([1, 2, 3])]
    for key, part in groups(d, CONTEXT + ["participant", "round"]):
        row = dict(key, n_contents_total=len(part))
        for metric in metrics:
            use = part.loc[finite_pair(part, metric)]
            row[f"n_contents_{metric}"] = len(use)
            for domain in DOMAINS:
                row[f"{domain}_{metric}"] = use[f"{domain}_{metric}"].mean()
            row[f"delta_{metric}"] = row[f"passive_{metric}"] - row[f"active_{metric}"]
        rows.append(row)
    rounds = pd.DataFrame(rows)
    for key, part in groups(rounds, CONTEXT):
        for before, after in [(1, 3), (1, 2), (2, 3)]:
            left = part[part["round"].eq(before)].set_index("participant")
            right = part[part["round"].eq(after)].set_index("participant")
            for metric in metrics:
                ids = left.index.intersection(right.index)
                a, b = left.loc[ids], right.loc[ids]
                valid = finite_pair(a, metric) & finite_pair(b, metric)
                a, b = a.loc[valid], b.loc[valid]
                row = dict(key, metric=metric, round_from=before, round_to=after,
                           n_participants=len(a), n_contents_from=int(a[f"n_contents_{metric}"].sum()),
                           n_contents_to=int(b[f"n_contents_{metric}"].sum()))
                for domain in (*DOMAINS, "delta"):
                    delta = b[f"{domain}_{metric}"].to_numpy() - a[f"{domain}_{metric}"].to_numpy()
                    row[f"{domain}_change"] = delta.mean() if len(delta) else np.nan
                    row.update({f"{domain}_{name}": value for name, value in boot.mean(delta).items()})
                changes.append(row)
    return rounds, pd.DataFrame(changes)


def joint_summary(joint):
    if joint.empty:
        return pd.DataFrame()
    measures = [c for c in ("joint_share", "conditional_share") if c in joint]
    keys = CONTEXT + ["active_region", "passive_region"]
    by_content = joint.groupby(keys + ["participant", "content_id"], dropna=False)[measures].mean().reset_index()
    by_person = by_content.groupby(keys + ["participant"], dropna=False)[measures].mean().reset_index()
    result = by_person.groupby(keys, dropna=False)[measures].mean()
    counts = by_person.groupby(keys, dropna=False)[measures].count().add_prefix("n_participants_")
    return result.join(counts).reset_index()


def model_bss(content, boot):
    rows = []
    refs = ["train_heatmap", "center_point", "uniform_area", "train_rate"]
    for key, part in groups(content, ["analysis_branch", "model"] + CONTEXT):
        for event in ("target", "gripper"):
            for ref in refs:
                needed = [f"{d}_BS_{event}{suffix}" for d in DOMAINS for suffix in ("", f"_{ref}")]
                if not set(needed).issubset(part):
                    continue
                use = part[np.isfinite(part[needed].to_numpy(float)).all(axis=1)]
                p = use.groupby("participant", sort=True)[needed].mean()
                row = dict(key, event=event, reference=ref, n_participants=len(p), n_contents=len(use))
                for domain in DOMAINS:
                    loss, base = p[f"{domain}_BS_{event}"].to_numpy(), p[f"{domain}_BS_{event}_{ref}"].to_numpy()
                    row[f"{domain}_BSS"] = 1 - loss.mean()/base.mean() if len(base) and base.mean() > 0 else np.nan
                    row.update({f"{domain}_{name}": value for name, value in boot.bss(loss, base).items()})
                row["delta_BSS"] = row["passive_BSS"] - row["active_BSS"]
                rows.append(row)
    return pd.DataFrame(rows)


def model_error_columns(models):
    """Frozen v3 hard-error definitions; retain missing denominators as NA."""
    additions = {}
    for domain in DOMAINS:
        for event in ("target", "gripper"):
            for name, numerator, other in [("FPR", "FP", "TN"), ("FNR", "FN", "TP")]:
                a, b = f"{domain}_{numerator}_{event}_s", f"{domain}_{other}_{event}_s"
                if a in models and b in models:
                    denominator = models[a] + models[b]
                    additions[f"{domain}_{name}_{event}"] = models[a] / denominator.where(denominator > 0)
        for metric in CORE[:3]:
            c = f"{domain}_delta_{metric}"
            if c in models:
                additions[f"{domain}_abs_delta_{metric}"] = models[c].abs()
    missing = {name: value for name, value in additions.items() if name not in models}
    return pd.concat([models, pd.DataFrame(missing, index=models.index)], axis=1)


class Plots:
    def __init__(self, out):
        self.out = out
        self.index = []
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                             "figure.dpi": 110, "savefig.facecolor": "white"})

    def save(self, fig, rel, description):
        path = self.out / "figures" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        if rel.startswith("models/"):
            data_source = "model_brier_skill.csv" if rel.endswith("_bss.png") else "model_group_stage_summary_long.csv"
        else:
            data_source = next((csv for name, csv in [("agreement", "participant_stage_summary.csv"),
                ("performance", "performance_participants.csv"), ("rounds", "participant_round_summary.csv"),
                ("region_correspondence", "region_joint_summary.csv"), ("coverage", "coverage.csv")]
                if rel.endswith(name+".png")), "group_stage_summary_long.csv")
        self.index.append(dict(path=path.relative_to(self.out).as_posix(), description=description, data_source=data_source))

    @staticmethod
    def axes(n, width=3.7, height=3.1):
        cols = min(3, max(1, n))
        rows = (n + cols - 1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*width, rows*height+.65), squeeze=False,
                                 constrained_layout=True)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        return fig, axes.ravel()

    @staticmethod
    def stages(task):
        return ["whole", *STAGES.get(task, [])]

    @staticmethod
    def title(fig, task, title, n=None):
        fig.suptitle(f"Task {task} | {title}\nMatched active/passive gaze; novice source and observer" +
                     (f" | n={n} source operators" if n is not None else ""), fontsize=12, fontweight="bold")

    @staticmethod
    def point_ci(ax, x, mean, lo, hi, color, label=None, marker="o"):
        good = np.isfinite(np.asarray(mean, float))
        xx = np.asarray(x)[good]
        yy = np.asarray(mean)[good]
        ax.scatter(xx, yy, color=color, s=25, label=label, marker=marker, zorder=4)
        a, b = np.asarray(lo)[good], np.asarray(hi)[good]
        ci = np.isfinite(a) & np.isfinite(b)
        ax.vlines(xx[ci], a[ci], b[ci], color=color, lw=1.4, zorder=3)

    def metric_panel(self, task, long, metrics, stem, title, model=None):
        select = long[long.task.eq(task) & long.scenario.eq("main")]
        metrics = [m for m in metrics if m in set(select.metric)]
        if not metrics:
            return
        fig, axes = self.axes(len(metrics), width=4.4, height=3.8)
        stages = self.stages(task)
        for ax, metric in zip(axes, metrics):
            g = select[select.metric.eq(metric)]
            domains = [*DOMAINS] + (["model"] if model else [])
            for i, domain in enumerate(domains):
                z = g[g.domain.eq(domain)].set_index("stage").reindex(stages)
                x = np.arange(len(stages)) + (i-(len(domains)-1)/2)*.14
                self.point_ci(ax, x, z["mean"], z.ci_low, z.ci_high, COLORS[domain], domain,
                              marker="D" if domain == "model" else "o")
            counts = g[g.domain.eq("active")].set_index("stage").reindex(stages).n_participants.fillna(0).astype(int)
            ax.set_xticks(range(len(stages)), [s.replace(" ", "\n") + f"\n(n={n})" for s, n in zip(stages, counts)], fontsize=7)
            ax.set_title(LABELS.get(metric, metric))
            ax.grid(axis="y", alpha=.2)
            if metric in AREA_METRICS or metric.startswith("delta"):
                ax.axhline(0, color="gray", ls="--", lw=.8)
        axes[0].legend(fontsize=8)
        self.title(fig, task, title)
        self.save(fig, stem, title)

    def all_human(self, long, persons, content, perf_points, rounds, joint, coverage):
        for task in sorted(long.task.unique()):
            main = long[long.task.eq(task) & long.scenario.eq("main")]
            stages = self.stages(task)
            self.metric_panel(task, long, CORE, f"task_{task}/core_metrics.png", "Occupancy and preference (95% participant CI)")
            self.metric_panel(task, long, AREA_METRICS, f"task_{task}/area_corrected.png", "Area-corrected metrics (95% participant CI)")
            self.metric_panel(task, long, TEMPORAL, f"task_{task}/dwell_switch.png", "Observed dwell and switching (95% participant CI)")
            fig, axs = self.axes(len(stages))
            for ax, stage in zip(axs, stages):
                base = np.zeros(2)
                for region, color, name in zip(REGIONS, REGION_COLORS, REGION_NAMES):
                    z = main[main.stage.eq(stage) & main.metric.eq("L_"+region)].set_index("domain")
                    values = np.array([z.loc[d, "mean"] if d in z.index else np.nan for d in DOMAINS])
                    ax.bar(range(2), values, bottom=base, color=color, label=name)
                    base += np.nan_to_num(values)
                z = main[main.stage.eq(stage) & main.metric.eq("L_T") & main.domain.eq("active")]
                n = int(z.n_participants.iloc[0]) if len(z) else 0
                ax.set(xticks=[0, 1], xticklabels=["Active", "Passive"], ylim=(0, 1.02), title=f"{stage}\n(n={n} operators)")
            axs[0].set_ylabel("Fraction of common valid time")
            axs[-1 if len(stages) == len(axs) else len(stages)-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
            self.title(fig, task, "Four mutually exclusive regions")
            self.save(fig, f"task_{task}/regions.png", "四区域占用：Active与Passive并排")

            fig, axs = plt.subplots(2, 3, figsize=(13, 7.8), constrained_layout=True)
            for j, metric in enumerate(CORE[:3]):
                top, ax = axs[0, j], axs[1, j]
                for i, domain in enumerate(DOMAINS):
                    absolute = main[main.metric.eq(metric) & main.domain.eq(domain)].set_index("stage").reindex(stages)
                    self.point_ci(top, np.arange(len(stages))+(i-.5)*.13, 100*absolute["mean"],
                                  100*absolute.ci_low, 100*absolute.ci_high, COLORS[domain], domain)
                top.set_title(LABELS[metric]); top.set_ylabel("Active / passive (%)")
                top.set_xticks(range(len(stages)), [s.replace(" ", "\n") for s in stages], fontsize=7)
                top.grid(axis="y", alpha=.2)
                g = main[main.metric.eq(metric) & main.domain.eq("delta")].set_index("stage").reindex(stages)
                x = np.arange(len(stages))
                self.point_ci(ax, x, 100*g["mean"], 100*g.ci_low, 100*g.ci_high, COLORS["delta"])
                ax.axhline(0, color="gray", ls="--", lw=.8)
                ax.set_xticks(x, [s.replace(" ", "\n") + f"\n(n={int(n)})" for s, n in zip(stages, g.n_participants.fillna(0))], fontsize=7)
                ax.set_title(LABELS[metric]); ax.set_ylabel("Passive minus active (percentage points)")
                ax.grid(axis="y", alpha=.2)
            axs[0, 0].legend(fontsize=8)
            self.title(fig, task, "Paired differences (95% participant CI)")
            self.save(fig, f"task_{task}/paired_differences.png", "配对差值：Passive−Active")

            fig, axs = self.axes(3)
            p = persons[persons.task.eq(task) & persons.scenario.eq("main") & persons.stage.eq("whole")]
            for ax, metric in zip(axs, CORE[:3]):
                ok = finite_pair(p, metric)
                a, b = p.loc[ok, f"active_{metric}"], p.loc[ok, f"passive_{metric}"]
                ax.scatter(a, b, color=COLORS["passive"], s=35)
                for _, row in p.loc[ok].iterrows():
                    ax.annotate(str(row.participant), (row[f"active_{metric}"], row[f"passive_{metric}"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
                ax.plot([0, 1], [0, 1], "--", color="gray", lw=.8)
                ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="Active", ylabel="Passive", title=f"{LABELS[metric]}\n(n={int(ok.sum())} operators)")
                ax.set_aspect("equal")
            self.title(fig, task, "Whole-task agreement by source operator")
            self.save(fig, f"task_{task}/agreement.png", "全任务Active–Passive一致性散点")

            fig, axs = self.axes(3)
            for ax, metric in zip(axs, CORE[:3]):
                g = long[long.task.eq(task) & long.stage.eq("whole") & long.metric.eq(metric)]
                scenarios = [s for s in SCENARIOS if s in set(g.scenario)]
                for i, domain in enumerate(DOMAINS):
                    z = g[g.domain.eq(domain)].set_index("scenario").reindex(scenarios)
                    x = np.arange(len(scenarios)) + (i-.5)*.08
                    ax.plot(x, z["mean"], color=COLORS[domain], lw=.9)
                    self.point_ci(ax, x, z["mean"], z.ci_low, z.ci_high, COLORS[domain], domain)
                counts = g[g.domain.eq("active")].set_index("scenario").reindex(scenarios).n_participants.fillna(0).astype(int)
                ax.set_xticks(range(len(scenarios)), [f"{s}\n(n={n})" for s, n in zip(scenarios, counts)], rotation=30, ha="right", fontsize=8)
                ax.set_title(LABELS[metric]); ax.grid(axis="y", alpha=.2)
            axs[0].legend(); self.title(fig, task, "Whole-task sensitivity (95% participant CI)")
            self.save(fig, f"task_{task}/sensitivity.png", "五种条件敏感性：两种gaze与CI")

            self.performance_plot(task, perf_points)
            self.round_plot(task, rounds)
            self.joint_plot(task, joint)
            self.coverage_plot(task, coverage)

    def performance_plot(self, task, points):
        if points.empty:
            return
        p = points[points.task.eq(task) & points.scenario.eq("main") & points.stage.eq("whole")]
        fig, axs = self.axes(3)
        for ax, metric in zip(axs, CORE[:3]):
            g = p[p.metric.eq(metric)]
            for domain in DOMAINS:
                rho = corr(g.score, g[domain]) if len(g) >= 3 else np.nan
                label = f"{domain}, rho={rho:.2f}" if np.isfinite(rho) else f"{domain}, rho=NA"
                ax.scatter(g.score, g[domain], color=COLORS[domain], label=label, s=28,
                           marker="o" if domain == "active" else "^")
            for _, row in g.iterrows():
                ax.plot([row.score, row.score], [row.active, row.passive], color="#aaaaaa", lw=.7, zorder=0)
            ax.set(xlabel="Source operator's mean score", ylabel=LABELS[metric], title=f"{LABELS[metric]}\n(n={len(g)} operators)")
            ax.legend(fontsize=7); ax.grid(alpha=.2)
        self.title(fig, task, "Association with source operator performance")
        self.save(fig, f"task_{task}/performance.png", "源操作者成绩关联：同内容Active与Passive")

    def round_plot(self, task, rounds):
        if rounds.empty:
            return
        p = rounds[rounds.task.eq(task) & rounds.scenario.eq("main") & rounds.stage.eq("whole")]
        fig, axs = self.axes(3)
        for ax, metric in zip(axs, CORE[:3]):
            for domain in DOMAINS:
                for _, g in p.groupby("participant"):
                    g = g.sort_values("round")
                    ax.plot(g["round"], g[f"{domain}_{metric}"], "o-", color=COLORS[domain], alpha=.32, lw=.8, markersize=3)
                means = p.groupby("round")[f"{domain}_{metric}"].mean()
                ax.plot(means.index, means.values, "o-", color=COLORS[domain], lw=2, label=domain)
            counts = p.groupby("round")[f"active_{metric}"].count()
            ax.set_xticks([1, 2, 3], [f"{r}\n(n={counts.get(r, 0)})" for r in [1, 2, 3]])
            ax.set(xlabel="Source recording practice round", ylabel=LABELS[metric], title=LABELS[metric])
            ax.grid(alpha=.2)
        axs[0].legend(); self.title(fig, task, "Source practice rounds (descriptive trajectories)")
        self.save(fig, f"task_{task}/rounds.png", "源录像练习轮次：配对gaze轨迹")

    def joint_plot(self, task, joint):
        if joint.empty:
            return
        p = joint[joint.task.eq(task) & joint.scenario.eq("main") & joint.stage.eq("whole")]
        fig, axs = self.axes(2, width=4.4, height=3.8)
        for ax, metric, title in zip(axs, ["joint_share", "conditional_share"], ["Joint occupancy", "Conditional on active region"]):
            m = p.pivot(index="active_region", columns="passive_region", values=metric).reindex(index=REGIONS, columns=REGIONS)
            n = p.pivot(index="active_region", columns="passive_region", values="n_participants_"+metric).reindex(index=REGIONS, columns=REGIONS)
            im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
            for i in range(4):
                for j in range(4):
                    val = m.iloc[i, j]
                    cell = (f"{val:.2f}" if np.isfinite(val) else "NA") + f"\n(n={int(n.iloc[i,j])})"
                    ax.text(j, i, cell, ha="center", va="center", fontsize=8,
                            color="white" if val > .6 else "black")
            ax.set(xticks=range(4), xticklabels=REGIONS, yticks=range(4), yticklabels=REGIONS,
                   xlabel="Passive region", ylabel="Active region", title=title)
            fig.colorbar(im, ax=ax, shrink=.8)
        self.title(fig, task, "Active-passive region correspondence")
        self.save(fig, f"task_{task}/region_correspondence.png", "Active×Passive区域联合及条件矩阵")

    def coverage_plot(self, task, coverage):
        needed = {"active_valid_n", "passive_valid_n", "common_valid_n", "pair_id"}
        if coverage.empty or not needed.issubset(coverage):
            return
        p = coverage[coverage.task.eq(task) & coverage.scenario.eq("main") & coverage.stage.eq("whole")].sort_values("pair_id")
        fig, ax = plt.subplots(figsize=(max(7, len(p)*.8), 4), constrained_layout=True)
        for i, (name, color) in enumerate([("active", COLORS["active"]), ("passive", COLORS["passive"]), ("common", "#777777")]):
            ax.bar(np.arange(len(p)) + (i-1)*.25, p[f"{name}_valid_n"], width=.24, color=color, label=name)
        ax.set_xticks(range(len(p)), p.pair_id, rotation=50, ha="right", fontsize=6)
        ax.set_ylabel("Valid frames"); ax.legend()
        self.title(fig, task, f"Main-scenario frame coverage ({len(p)} pairs)")
        self.save(fig, f"task_{task}/coverage.png", "每对Active、Passive及共同有效帧")

    def model_plots(self, long, bss, bss_only=False):
        if long.empty:
            return
        for key, part in groups(long, ["analysis_branch", "model", "task"]):
            branch, model, task = key["analysis_branch"], key["model"], key["task"]
            stem = f"models/{branch}/{model}/task_{task}"
            if not bss_only:
                self.metric_panel(task, part, CORE[:3], stem+"_metrics.png", f"{model}: {branch}", model=model)
                if branch != "probability_mc":
                    self.metric_panel(task, part, TEMPORAL, stem+"_temporal.png", f"{model}: {branch} temporal diagnostic", model=model)
                self.metric_panel(task, part, ["BS_target", "BS_gripper"], stem+"_brier.png", f"{model}: Brier against active / passive")
            q = bss
            for col, value in key.items():
                if col in q:
                    q = q[q[col].eq(value)]
            if q.empty:
                continue
            q = q[q.scenario.eq("main") & q.stage.eq("whole")]
            fig, axs = self.axes(2, width=4.5, height=3.7)
            for ax, event in zip(axs, ["target", "gripper"]):
                g = q[q.event.eq(event)]
                refs = list(g.reference.unique())
                for i, domain in enumerate(DOMAINS):
                    z = g.set_index("reference").reindex(refs)
                    self.point_ci(ax, np.arange(len(refs))+(i-.5)*.12, z[f"{domain}_BSS"], z[f"{domain}_ci_low"], z[f"{domain}_ci_high"], COLORS[domain], f"{domain} reference")
                ax.axhline(0, color="gray", ls="--", lw=.8)
                ax.set_xticks(range(len(refs)), refs, rotation=35, ha="right", fontsize=7)
                n = int(g.n_participants.max()) if len(g) else 0
                ax.set_title(f"{event} event (n={n} operators)"); ax.set_ylabel("Brier skill score")
            fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8)
            self.title(fig, task, f"{model}: {branch} Brier skill")
            self.save(fig, stem+"_bss.png", f"模型BSS：{branch}/{model}，两类gaze参照")


def num(value, scale=1, digits=3):
    return f"{float(value)*scale:.{digits}f}" if pd.notna(value) and np.isfinite(float(value)) else "NA"


def report(out, pairs, content, persons, long, perf, rounds, bss, plots, coverage):
    inventory = pairs.drop_duplicates("pair_id")
    lines = ["# Passive gaze 与配对 Active gaze：novice–novice 16对分析", "",
             "本报告仅纳入 source skill 与 observer skill 均为 novice 的指定配对。"
             "每个 passive 的指标、图表与对应 active 一起呈现；所有数值来自本次逐行配对重新测量。", "",
             "## 数据与共同支持", "", "|任务|配对数|源录像数|源操作者数|观察者数|", "|---|---:|---:|---:|---:|"]
    for task, g in inventory.groupby("task"):
        lines.append(f"|{task}|{len(g)}|{g.content_id.nunique()}|{g.participant.nunique()}|{g.observer_id.nunique()}|")
    lines += ["", "匹配规则为对应 active/passive NPY 行数相等，即第 i 行对应同一视频帧；不要求两份时间戳数值相等，不另做时间重配准。"
              "阶段、target 槽位、双侧 DLC 腕部位置、原始视频时间与尺寸均继承对应 active 内容。"
              "每对在两者共同有效帧上同时计算两种 gaze；有效性只要求坐标有限且归一化坐标在 [0,1]，不以 valid 状态码筛选。", "",
              "[配对逐阶段原始指标](pair_metrics.csv) · [覆盖审计](coverage.csv) · [运行配置](run_config.json)", "",
              "## 完全复用的 v3 指标", "",
              "直接调用冻结的 v3 classify、areas、summarize、temporal 与时间支持代码。主条件 target 半径与双侧腕部邻域半径均为 50 px。"
              "Target 中心仍是现有有效槽位的算术平均；器械区域仍是 DLC 腕部/铰接邻域，未改称器械尖端。", "",
              "令 T/P/O/N 分别表示仅 target、仅腕部、两者重叠及其他区域。每对/阶段以原始视频时间戳计算的帧支持时长 dt 加权得到 L_T、L_P、L_O、L_N；"
              "L_target=L_T+L_O，L_gripper=L_P+L_O，D=L_T+L_P，C=L_T/(L_T+L_P)。分母为零时保留 NA。"
              "面积参照及 C_excess、E_target、E_gripper、B、B_excess 同样调用 v3。whole 按纳入阶段的时间合并，不平均阶段比例。", "",
              "驻留与切换沿用 v3 的连续段、时间断点、最短驻留与端点删失规则。敏感性保留 main、r25、r100、gripper100、smooth；"
              "smooth 为原三样本平滑及短缺口规则，两类 gaze 分别平滑后重建共同支持。", "",
              "## 汇总与统计", "",
              "逐指标先筛选同一 pair 中 active/passive 均可估的配对，再在同一 content 内对观察者等权平均；"
              "随后同一源操作者内录像等权，最后源操作者等权。因共同有效帧随观察者变化，active 也保留各配对重新测量的结果，"
              "在 content 内与 passive 对称平均；不把重复 active 当作新增独立样本。Δ 一律为 Passive−Active。", "",
              "95% CI 直接复用 v3 ParticipantBootstrap，seed=42、10,000次、参与者重抽及百分位区间。"
              "此处参与者是源录像操作者；CI 条件于本次观察者集合，不表示对新观察者总体的双重抽样推断。"
              "每个指标分别记录有效配对/录像/参与者数。少于2个源操作者不计算均值CI；不新增显著性检验。", "",
              "[内容层汇总](content_stage_summary.csv) · [源操作者层汇总](participant_stage_summary.csv) · "
              "[组汇总（宽表）](group_stage_summary.csv) · [组汇总（长表）](group_stage_summary_long.csv) · "
              "[观察者描述表](observer_stage_summary.csv)", "",
              "## 主条件全任务结果", "",
              "占用与 C 以百分比显示；差值为百分点，方括号为95%CI。", "",
              "|任务|指标|Active [95%CI]|Passive [95%CI]|Passive−Active [95%CI]|源操作者/录像/配对|",
              "|---|---|---:|---:|---:|---:|"]
    report_tasks = sorted(long.task.unique())
    for task in report_tasks:
        if task != report_tasks[0]:
            lines += ["", "|任务|指标|Active [95%CI]|Passive [95%CI]|Passive−Active [95%CI]|源操作者/录像/配对|",
                      "|---|---|---:|---:|---:|---:|"]
        for metric in CORE[:3]:
            z = long[long.task.eq(task) & long.scenario.eq("main") & long.stage.eq("whole") & long.metric.eq(metric)].set_index("domain")
            cells = []
            for domain in (*DOMAINS, "delta"):
                r = z.loc[domain]
                cells.append(f"{num(r['mean'],100,2)} [{num(r.ci_low,100,2)}, {num(r.ci_high,100,2)}]")
            r = z.loc["active"]
            lines.append(f"|{task}|{metric}|"+"|".join(cells)+f"|{int(r.n_participants)}/{int(r.n_contents)}/{int(r.n_pairs)}|")
        lines += ["", f"### Task {task} 图表", ""]
        for f in plots.index:
            if f["path"].startswith(f"figures/task_{task}/"):
                lines += [f"**{f['description']}**", "", f"![{f['description']}]({f['path']})", ""]
    lines += ["## 源操作者成绩与练习轮次", "",
              "成绩关联使用源操作者成绩，不是 passive observer 自身表现。每个指标先在分数及两种 gaze 同时有效的同一组源录像内形成操作者均值，"
              "再算 Spearman ρ、参与者bootstrap CI及逐人留一法。少于3人、常量成绩或常量指标保留NA；ties与无定义bootstrap数量保留在CSV。", "",
              "|任务|指标|n|Active ρ [95%CI]|Passive ρ [95%CI]|", "|---|---|---:|---:|---:|"]
    if not perf.empty:
        p = perf[perf.scenario.eq("main") & perf.stage.eq("whole") & perf.metric.isin(CORE[:3])]
        for _, r in p.iterrows():
            cells = [f"{num(r[d+'_rho'])} [{num(r[d+'_ci_low'])}, {num(r[d+'_ci_high'])}]" for d in DOMAINS]
            lines.append(f"|{r.task}|{r.metric}|{int(r.n_participants)}|"+"|".join(cells)+"|")
    lines += ["", "round 表示源操作者/任务内的练习轮次，不是观察者观看顺序或 Attempt。"
              "轮次差只在同一操作者两轮及两类 gaze 均有效时计算；可用样本不足时保留NA，描述曲线不表示纵向因果效应。", "",
              "[成绩相关](performance_correlations.csv) · [相关用参与者数据](performance_participants.csv) · "
              "[逐人留一法](performance_leave_one_out.csv) · [轮次描述](participant_round_summary.csv) · "
              "[轮次配对差](repetition_contrasts.csv)", "",
              "## 模型子集与原四分析分支", ""]
    model_pairs = read_csv(out / "model_pair_metrics.csv")
    if model_pairs.empty:
        lines += ["本次模型子集无可用输出，模型分析保留为空，不以其他内容补齐。", ""]
    else:
        lines += [f"模型比较使用本次配对与原v3测试录像的交集：{model_pairs.pair_id.nunique()}对、"
                  f"{model_pairs.content_id.nunique()}个源内容、{model_pairs.participant.nunique()}个源操作者。"
                  "仍使用原五模型、原保存的64 samples概率质量主分支及expectation/argmax/sample解码分支；"
                  "每个分支同时附active、passive实测参考。", "",
                  "Brier 对同一个模型分别与 active/passive 的二元事件计算；BSS=1−参与者等权平均模型Brier/参与者等权平均参照Brier。"
                  "训练参照冻结为原训练平均热图、中心点、均匀面积及训练事件频率。GT_smoothed只是诊断，不进入BSS分母。"
                  "独立逐帧sample解码的驻留/切换只描述所抽路径，不表示概率分布的时间连贯性。", "",
                  "[模型逐配对指标](model_pair_metrics.csv) · [模型内容汇总](model_content_stage_summary.csv) · "
                  "[模型参与者汇总](model_participant_stage_summary.csv) · [模型组汇总](model_group_stage_summary.csv) · "
                  "[模型BSS](model_brier_skill.csv)", ""]
        for f in plots.index:
            if f["path"].startswith("figures/models/"):
                lines.append(f"- [{f['description']}]({f['path']})")
        supplements = [f for f in plots.index if f["path"].startswith("figures/model_supplement/")]
        if supplements:
            lines += ["", "### 解码区域矩阵与GT平滑诊断", "",
                      "下列图均在同一张图内并列active、passive两种观测参照。GT平滑仅为观测平滑诊断，不作模型上限或训练参照。", ""]
            lines += ["[补充报告及数据索引](model_supplement_report.md) · [区域矩阵组汇总](model_confusion_group_summary.csv) · "
                      "[补充图索引](model_supplement_figure_index.csv)", ""]
            for f in supplements:
                lines.append(f"- [{f.get('description', f['path'])}]({f['path']})")
    examples = read_csv(out / "example_index.csv")
    if not examples.empty:
        lines += ["", "## 每对视频帧与时间序列", "",
                  "每个配对均提供同帧active/passive坐标叠加及双轨迹时间序列。示例帧按主条件共同有效帧的中位索引确定，未按差异大小筛选。", "",
                  "[示例帧选择及坐标索引](example_index.csv)", "", "|配对|任务|视频帧对照|时间序列|", "|---|---|---|---|"]
        for _, r in examples.iterrows():
            lines.append(f"|{r.pair_id}|{r.task}|[同帧叠加]({r.overlay})|[双轨迹]({r.timeline})|")
    lines += ["", "## 解读范围与输出检查", "",
              "- 本次仅16个novice–novice配对，Task A 7对、Task C 9对；结论不外推到其他技能组。",
              "- 每个passive仅与其对应active比较，图上active基线来自本次相同配对、共同帧及相同层级汇总。",
              "- 区域矩阵描述两种人类gaze的对应关系，active不是passive应复制的唯一真值。",
              "- 同一内容的观察者重复及源操作者重复均保留身份，未把每行配对当作独立参与者。",
              "- 平滑和半径敏感性单独报告；缺失值保留NA，不以零补值。",
              "", "[区域联合逐对数据](region_joint.csv) · [区域联合汇总](region_joint_summary.csv) · "
              "[图表索引](figure_index.csv) · [统计配置](statistics_manifest.json)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def include_external_figures(out, plots):
    supplements = read_csv(out / "model_supplement_figure_index.csv")
    if not supplements.empty:
        for row in supplements.to_dict("records"):
            path = row.get("path", row.get("figure_path"))
            if path:
                plots.index.append(dict(path=path, description=row.get("description", path),
                                        data_source=row.get("data_source", "model_confusions.csv;model_gt_smoothed.csv")))
    examples = read_csv(out / "example_index.csv")
    if not examples.empty:
        for _, row in examples.iterrows():
            for kind in ("overlay", "timeline"):
                plots.index.append(dict(path=row[kind], description=f"{row.pair_id}: {kind} (active/passive)",
                                        data_source=f"example_index.csv;frames/{row.pair_id}.npz"))
    plots.index = list({x["path"]: x for x in plots.index}.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "output")
    parser.add_argument("--refresh-report", action="store_true", help="Refresh report and indexes after external figures arrive")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    pair = read_csv(out / "pair_metrics.csv")
    if pair.empty:
        raise ValueError("pair_metrics.csv is missing or empty; run measurement first")
    if args.refresh_report:
        plots = Plots(out)
        plots.index = read_csv(out / "figure_index.csv").to_dict("records")
        include_external_figures(out, plots)
        write_csv(pd.DataFrame(plots.index), out / "figure_index.csv")
        report(out, pair, read_csv(out / "content_stage_summary.csv"), read_csv(out / "participant_stage_summary.csv"),
               read_csv(out / "group_stage_summary_long.csv"), read_csv(out / "performance_correlations.csv"),
               read_csv(out / "participant_round_summary.csv"), read_csv(out / "model_brier_skill.csv"), plots,
               read_csv(out / "coverage.csv"))
        path = out / "statistics_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(n_figures=len(plots.index), figures=plots.index)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(dict(status="report_refreshed", figures=len(plots.index))))
        return
    boot = CachedParticipantBootstrap(seed=42, nboot=10000)
    print("Aggregating human pairs", flush=True)
    content, persons, summary, long, metrics = aggregate_pairs(pair, boot)
    observer = observer_summary(pair, metrics)
    outcomes = [m for m in [*CORE[:3], "C_excess", "E_target", "E_gripper", "B", "B_excess"] if m in metrics]
    perf, points, loo = performance(content, boot, outcomes)
    rounds, contrasts = round_analysis(content, boot, [m for m in [*outcomes, "L_O", *TEMPORAL] if m in metrics])
    joint = joint_summary(read_csv(out / "region_joint.csv"))
    outputs = {"content_stage_summary": content, "participant_stage_summary": persons,
               "group_stage_summary": summary, "group_stage_summary_long": long,
               "observer_stage_summary": observer, "performance_correlations": perf,
               "performance_participants": points, "performance_leave_one_out": loo,
               "participant_round_summary": rounds, "repetition_contrasts": contrasts,
               "region_joint_summary": joint}
    for name, frame in outputs.items():
        write_csv(frame, out / (name + ".csv"))
    print("Human tables saved", flush=True)
    models = read_csv(out / "model_pair_metrics.csv")
    model_long, bss = pd.DataFrame(), pd.DataFrame()
    if not models.empty:
        print("Aggregating model comparisons", flush=True)
        models = model_error_columns(models)
        mc, mp, mg, model_long, mm = aggregate_pairs(models, boot, extra=["analysis_branch", "model"], model=True)
        bss = model_bss(mc, boot)
        outputs.update(model_content_stage_summary=mc, model_participant_stage_summary=mp,
                       model_group_stage_summary=mg, model_group_stage_summary_long=model_long, model_brier_skill=bss)
    for name, frame in outputs.items():
        write_csv(frame, out / (name + ".csv"))
    print("All tables saved; rendering figures", flush=True)
    coverage = read_csv(out / "coverage.csv")
    plots = Plots(out)
    plots.all_human(long, persons, content, points, rounds, joint, coverage)
    plots.model_plots(model_long, bss)
    include_external_figures(out, plots)
    write_csv(pd.DataFrame(plots.index), out / "figure_index.csv")
    manifest = dict(bootstrap_seed=42, bootstrap_replicates=10000,
                    bootstrap_unit="source_operator", confidence_interval="percentile_2.5_97.5",
                    hierarchy="pairwise_finite -> observer_equal_within_content -> content_equal_within_source_operator -> source_operator_equal",
                    delta="passive_minus_active", counts={name: len(frame) for name, frame in outputs.items()},
                    n_figures=len(plots.index), figures=plots.index,
                    frozen_stats_source=str(HERE / "v3" / "tl_stats.py"))
    (out / "statistics_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report(out, pair, content, persons, long, perf, rounds, bss, plots, coverage)
    print(json.dumps(dict(status="complete", tables=len(outputs), figures=len(plots.index), output=str(out))))


if __name__ == "__main__":
    main()
