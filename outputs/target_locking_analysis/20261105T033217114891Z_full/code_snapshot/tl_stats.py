"""Trial-balanced, participant-level summaries for the target-locking analysis."""

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata


GROUP = ["scope", "scenario", "task", "stage", "source"]
PAIR = ["scope", "scenario", "task", "stage"]
PERSON = GROUP + ["participant"]
OCC = ["L_T", "L_P", "L_O", "L_N", "L_target", "L_gripper", "D", "C",
       "C_U", "C_excess", "E_target", "E_gripper", "B", "B_excess"]
AREA = ["area_T", "area_P", "area_O", "area_N"]
TIME = ["direct_rate", "via_rate", "dwell_T_ms", "dwell_P_ms", "dwell_O_ms"]
OUTCOMES = ["C", "L_target", "L_gripper", "C_excess", "E_target", "E_gripper", "B", "B_excess"]
EVENTS = ["target", "gripper"]
REFS = ["center", "train_center", "train_rate", "uniform"]
PARTS = ["T", "P", "O", "N"]


def present(columns, df):
    return [column for column in columns if column in df]


def group_items(df, keys):
    for key, part in df.groupby(keys, sort=True, observed=True, dropna=False):
        yield dict(zip(keys, key if isinstance(key, tuple) else (key,))), part


def finite(df, columns):
    return df.loc[np.isfinite(df[columns].to_numpy(dtype=float)).all(axis=1)]


def percentiles(values):
    valid = np.asarray(values, float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return np.nan, np.nan
    return tuple(np.percentile(valid, [2.5, 97.5]))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    a, b = rankdata(x), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


class ParticipantBootstrap:
    """Use common participant-index draws across estimates of the same size."""

    def __init__(self, seed, nboot):
        self.seed = int(seed)
        self.nboot = int(nboot)
        if self.nboot < 1 or self.seed < 0:
            raise ValueError("seed must be nonnegative and nboot must be positive")
        self.weights = {}
        self.rho_cache = {}

    def draws(self, n):
        if n not in self.weights:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, n]))
            weights = rng.multinomial(n, np.full(n, 1.0 / n), size=self.nboot)
            # With five participants, only 126 distinct bootstrap weight vectors exist.
            if n <= 8:
                compact, freq = np.unique(weights, axis=0, return_counts=True)
            else:
                compact, freq = weights, None
            self.weights[n] = compact.astype(float), freq
        return self.weights[n]

    def interval(self, stats, freq):
        if freq is not None:
            stats = np.repeat(stats, freq)
        lo, hi = percentiles(stats)
        n_ok = int(np.isfinite(stats).sum())
        return {"ci_low": lo, "ci_high": hi, "bootstrap_defined": n_ok,
                "bootstrap_undefined": self.nboot - n_ok}

    def unavailable(self):
        return {"ci_low": np.nan, "ci_high": np.nan, "bootstrap_defined": 0,
                "bootstrap_undefined": 0}

    def mean(self, values):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            return self.unavailable()
        w, freq = self.draws(len(v))
        estimates = np.einsum("ij,j->i", w, v) / len(v)
        return self.interval(estimates, freq)

    def bss(self, loss, reference):
        if len(loss) < 2 or np.mean(reference) <= 0:
            return self.unavailable()
        w, freq = self.draws(len(loss))
        a = np.einsum("ij,j->i", w, loss)
        b = np.einsum("ij,j->i", w, reference)
        out = np.full(len(a), np.nan)
        np.divide(a, b, out=out, where=b > 0)
        return self.interval(1 - out, freq)

    def spearman(self, x, y):
        n = len(x)
        if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return self.unavailable()
        rx, ry = rankdata(x), rankdata(y)
        order = np.lexsort((ry, rx))
        rx, ry = rx[order], ry[order]
        key = (tuple(rx), tuple(ry))
        if key in self.rho_cache:
            return self.rho_cache[key]
        w, freq = self.draws(n)
        # Each drawn participant appears w_i times. Re-rank both variables
        # inside each replicate, including ties introduced by resampling.
        def centered_ranks(r):
            order_weights = (r[:, None] < r[None, :]).astype(float)
            order_weights += 0.5 * (r[:, None] == r[None, :])
            return np.einsum("ij,jk->ik", w, order_weights) - n / 2
        a, b = centered_ranks(rx), centered_ranks(ry)
        cov = np.einsum("ij,ij,ij->i", w, a, b)
        var = np.sqrt(np.einsum("ij,ij,ij->i", w, a, a)
                      * np.einsum("ij,ij,ij->i", w, b, b))
        estimates = np.full(len(cov), np.nan)
        np.divide(cov, var, out=estimates, where=var > 0)
        result = self.interval(estimates, freq)
        self.rho_cache[key] = result
        return result


def participant_summary(trials, boot):
    metrics = present(OCC + AREA + TIME, trials)
    # A missing preference denominator does not invalidate the observed
    # occupancy or area for that trial. Keep per-metric support counts.
    grouped = trials.groupby(PERSON, dropna=False, sort=True, observed=True)
    means = grouped[metrics].mean()
    valid_rows = trials.copy()
    if "valid_n" in trials:
        valid_rows = valid_rows[valid_rows.valid_n.fillna(0) > 0]
    n_valid = valid_rows.groupby(PERSON, dropna=False, observed=True).content_id.nunique()
    means["trials_total"] = grouped.content_id.nunique()
    means["trials_valid"] = n_valid.reindex(means.index, fill_value=0)
    if "valid_n" in trials:
        means["valid_n"] = grouped.valid_n.sum()
    if "valid_s" in trials:
        means["valid_s"] = grouped.valid_s.sum()
    means = means.join(grouped[metrics].count().add_prefix("n_trials_")).reset_index()
    rows = []
    for key, g in group_items(means, GROUP):
        for m in metrics:
            v = g[m].dropna().to_numpy(float)
            rows.append({**key, "metric": m, "n_participants": len(v),
                         "n_trials": int(g["n_trials_" + m].sum()),
                         "mean": np.mean(v) if len(v) else np.nan, **boot.mean(v)})
    return means, pd.DataFrame(rows)


def performance(trials, boot):
    metrics = present(OUTCOMES, trials)
    participants, overlaps, correlations = [], [], []
    for key, g in group_items(trials, GROUP):
        for m in metrics:
            use = finite(g, ["score", m])
            if "status" in use:
                use = use[use.status.astype(str).str.lower().isin(["completed", "failed"])]
            cols = use.groupby("participant", sort=True).agg(
                mean_score=("score", "mean"), mean_metric=(m, "mean"),
                n_trials=("content_id", "nunique")).reset_index()
            for rec in cols.to_dict("records"):
                participants.append({**key, "metric": m, **rec})
            x, y = cols.mean_score.to_numpy(float), cols.mean_metric.to_numpy(float)
            n = len(x)
            rho = corr(x, y) if n >= 3 else np.nan
            loo = []
            for i, person in enumerate(cols.participant):
                take = np.arange(n) != i
                estimate = corr(x[take], y[take]) if take.sum() >= 3 else np.nan
                overlaps.append({**key, "metric": m, "held_out": person,
                                 "n_retained": n - 1, "rho": estimate})
                if np.isfinite(estimate):
                    loo.append(estimate)
            if n < 3:
                flag = "fewer_than_3_participants"
            elif np.ptp(x) == 0:
                flag = "constant_score"
            elif np.ptp(y) == 0:
                flag = "constant_metric"
            else:
                flag = "defined"
            intervals = boot.spearman(x, y) if n >= 3 else boot.unavailable()
            correlations.append({**key, "metric": m, "n_participants": n,
                                 "n_trials": int(cols.n_trials.sum()),
                                 "distinct_scores": len(np.unique(x)), "rho": rho,
                                 "status": flag, **intervals,
                                 "loo_defined": len(loo),
                                 "loo_min": min(loo) if loo else np.nan,
                                 "loo_max": max(loo) if loo else np.nan})
    return pd.DataFrame(correlations), pd.DataFrame(participants), pd.DataFrame(overlaps)


def model_statistics(errors, trials, boot):
    e = errors.copy()
    if e.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "scope" not in e:
        e["scope"] = "paired_test"
    # Each temporal difference uses the two estimates for the *same trial*.
    # The dwell difference is missing when either source lacks a qualifying run.
    keys = PAIR + ["participant", "content_id"]
    temporal = present(TIME, trials)
    gm = trials[trials.scope.eq("paired_test")]
    observed = gm[gm.source.eq("gaze")][keys + temporal]
    predicted = gm[~gm.source.eq("gaze")][keys + ["source"] + temporal]
    tm = predicted.merge(observed, on=keys, validate="many_to_one",
                         suffixes=("_model", "_gaze"))
    for m in temporal:
        tm["delta_" + m] = tm[m + "_model"] - tm[m + "_gaze"]
    e = e.merge(tm[keys + ["source"] + ["delta_" + m for m in temporal]],
                on=keys + ["source"], how="left", validate="one_to_one")
    for event in EVENTS:
        fp, fn, tn, tp = [f"{p}_{event}_s" for p in ("FP", "FN", "TN", "TP")]
        if {fp, fn, tn, tp}.issubset(e.columns):
            neg = e[tn] + e[fp]
            pos = e[tp] + e[fn]
            e["FPR_" + event] = e[fp] / neg.where(neg > 0)
            e["FNR_" + event] = e[fn] / pos.where(pos > 0)
    for m in present(["delta_C", "delta_L_target", "delta_L_gripper"], e):
        e["abs_" + m] = e[m].abs()
    metric_cols = [c for c in e if c.startswith(("delta_", "abs_delta_", "BS_", "FPR_", "FNR_"))]
    model_cols = [c for c in metric_cols if c not in
                  {f"BS_{event}_{ref}" for event in EVENTS for ref in REFS}]
    persons, summaries, contrasts = [], [], []
    for key, g in group_items(e, GROUP):
        by = g.groupby("participant", sort=True)[metric_cols].mean()
        for person, row in by.iterrows():
            persons.append({**key, "participant": person, **row.to_dict()})
        for m in metric_cols:
            v = by[m].dropna().to_numpy(float)
            summaries.append({**key, "metric": m, "reference": "", "estimate":
                              v.mean() if len(v) else np.nan, "n_participants": len(v),
                              "n_trials": int(g[m].notna().sum()), **boot.mean(v)})
        for event in EVENTS:
            pred = f"BS_{event}"
            for ref in REFS:
                base = f"BS_{event}_{ref}"
                if pred not in g or base not in g:
                    continue
                matched = finite(g, [pred, base])
                v = matched.groupby("participant")[[pred, base]].mean()
                a, b = v[pred].to_numpy(), v[base].to_numpy()
                estimate = 1 - a.mean() / b.mean() if len(b) and b.mean() > 0 else np.nan
                summaries.append({**key, "metric": f"BSS_{event}", "reference": ref,
                                  "estimate": estimate, "n_participants": len(v),
                                  "n_trials": len(matched), **boot.bss(a, b)})
    for key, g in group_items(e, PAIR):
        idx = ["participant", "content_id"]
        models = {name: part.set_index(idx)[model_cols] for name, part in g.groupby("source")}
        for a, b in combinations(sorted(models), 2):
            # Alignment is on recording first, then average within participant.
            diff = models[a] - models[b]
            by = diff.groupby(level="participant").mean()
            for m in model_cols:
                v = by[m].dropna().to_numpy(float)
                contrasts.append({**key, "model_A": a, "model_B": b,
                                  "metric": m, "estimate_A_minus_B": v.mean() if len(v) else np.nan,
                                  "n_participants": len(v), "n_paired_trials": int(diff[m].count()),
                                  **boot.mean(v)})
    return pd.DataFrame(persons), pd.DataFrame(summaries), pd.DataFrame(contrasts)


def round_statistics(trials, boot):
    if "round" not in trials:
        return pd.DataFrame(), pd.DataFrame()
    metrics = present(OUTCOMES + ["L_O", "score"] + TIME, trials)
    d = trials.copy()
    d["round"] = pd.to_numeric(d["round"], errors="coerce")
    d = d[d["round"].isin([1, 2, 3])]
    if "score" in metrics:
        has_score = d.valid_n.gt(0) & np.isfinite(d.score)
        if "status" in d:
            has_score &= d.status.astype(str).str.lower().isin(["completed", "failed"])
        d.loc[~has_score, "score"] = np.nan
    grouped = d.groupby(PERSON + ["round"], sort=True, observed=True)
    rounds = grouped[metrics].mean()
    rounds["n_recordings"] = grouped.content_id.nunique()
    rounds = rounds.join(grouped[metrics].count().add_prefix("n_recordings_"))
    rounds = rounds.reset_index()
    rows = []
    for key, g in group_items(rounds, GROUP):
        for r1, r2 in [(1, 3), (1, 2), (2, 3)]:
            a = g[g["round"] == r1].set_index("participant")
            b = g[g["round"] == r2].set_index("participant")
            delta = b[metrics] - a[metrics]
            for m in metrics:
                use = delta[m].dropna()
                v = use.to_numpy(float)
                rows.append({**key, "metric": m, "round_from": r1, "round_to": r2,
                             "n_matched_participants": len(v),
                             "n_recordings_from": int(a.loc[use.index, "n_recordings_" + m].sum()),
                             "n_recordings_to": int(b.loc[use.index, "n_recordings_" + m].sum()),
                             "mean_paired_change": v.mean() if len(v) else np.nan, **boot.mean(v)})
    return rounds, pd.DataFrame(rows)


def confusion_statistics(confusions):
    if confusions.empty:
        return pd.DataFrame()
    c = confusions.copy()
    if "scope" not in c:
        c["scope"] = "paired_test"
    idx = GROUP + ["participant", "content_id"]
    amount = "seconds" if "seconds" in c else "frames"
    combos = pd.MultiIndex.from_product([PARTS, PARTS], names=["actual", "predicted"])
    z = c.pivot_table(index=idx, columns=["actual", "predicted"], values=amount,
                      aggfunc="sum", observed=True).reindex(columns=combos, fill_value=0).fillna(0)
    tot = z.sum(axis=1)
    shares = z.div(tot.where(tot > 0), axis=0)
    correct_label = {}
    for actual in PARTS:
        actual_total = z[actual].sum(axis=1)
        correct_label[actual] = z[actual].div(actual_total.where(actual_total > 0), axis=0)
    norm = pd.concat(correct_label, axis=1)
    norm.columns = combos
    def aggregate(frame):
        person = frame.groupby(level=PERSON, observed=True).mean()
        group = person.groupby(level=GROUP, observed=True)
        return group.mean(), group.count()
    s, sn = aggregate(shares)
    q, qn = aggregate(norm)
    # Long output retains every class pair, including unobserved classes.
    # Construct the fixed 4 x 4 layout directly across pandas versions.
    cells = []
    for a, p in combos:
        cell = pd.concat([s[(a, p)].rename("joint_share"),
                          sn[(a, p)].rename("n_joint"),
                          q[(a, p)].rename("conditional_given_actual"),
                          qn[(a, p)].rename("n_conditional")], axis=1)
        cells.append(cell.reset_index().assign(actual=a, predicted=p))
    return pd.concat(cells, ignore_index=True)


def agreement_pairs(trials):
    d = trials[trials.scope.eq("paired_test")]
    if d.empty:
        return pd.DataFrame()
    ids = PAIR + ["content_id", "participant"]
    metrics = present(["C", "L_target", "L_gripper", "B"], d)
    gaze = d[d.source.eq("gaze")][ids + metrics]
    models = d[~d.source.eq("gaze")][ids + metrics + ["source"]]
    merged = models.merge(gaze, on=ids, how="inner", validate="many_to_one",
                          suffixes=("_model", "_gaze"))
    tables = []
    for m in metrics:
        cols = [m + "_gaze", m + "_model"]
        match = finite(merged, cols)
        by = match.groupby(PERSON, sort=True)[cols].mean()
        by.columns = ["gaze", "model"]
        by["n_paired_trials"] = match.groupby(PERSON).content_id.nunique()
        tables.append(by.assign(metric=m).reset_index())
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def figures(out, participants, pair_table, score_points, groups, rounds, errors, confusion):
    """Render selected summaries at the primary radius, plus a sensitivity panel."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figures_dir = out / "figures"
    palette = list(plt.get_cmap("tab10").colors)
    sources = sorted(set(participants.source) - {"gaze"})
    colors = {"gaze": "#242424", **{s: palette[i % len(palette)] for i, s in enumerate(sources)}}
    modes = {s: ["o", "s", "^", "D", "v", "P", "X"][i % 7]
             for i, s in enumerate(["gaze"] + sources)}
    paths = []
    labels = {"C": "Relative target preference", "L_target": "Target occupancy",
              "L_gripper": "Gripper occupancy", "dwell_T_ms": "Target-only dwell (ms)",
              "dwell_P_ms": "Gripper-only dwell (ms)", "dwell_O_ms": "Overlap dwell (ms)",
              "direct_rate": "Direct switch rate", "via_rate": "Switch rate via overlap",
              "B": "B", "C_excess": "Excess target preference"}

    def save(fig, scope, task, stem):
        path = figures_dir / scope / f"task_{task}_{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path.relative_to(out).as_posix())

    def stages(data):
        return sorted(data.stage.dropna().unique(), key=lambda s: (s != "whole", s))

    def stage_title(stage):
        return "\n".join(textwrap.wrap(stage, width=24))

    def axes_for(names, width=3.2, height=2.6):
        ncol = min(3, len(names))
        nrow = (len(names) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(width*ncol, height*nrow + .85),
                                 squeeze=False, constrained_layout=True)
        for ax in axes.ravel()[len(names):]:
            ax.set_visible(False)
        return fig, axes.ravel()

    def no_points(ax, label="No estimable observations"):
        ax.text(.5, .5, label, ha="center", va="center", fontsize=8, color="#5f6671",
                transform=ax.transAxes)

    def title(fig, scope, task, text):
        data_name = "all available real gaze" if scope == "real_all" else "matched model test recordings"
        fig.suptitle(f"Task {task}  |  {text}\n{data_name}", fontsize=12, fontweight="bold")

    def legend(ax, names, text=None, heading=None):
        handles = [Line2D([], [], marker=modes[s], color=colors[s], linestyle="none",
                          label=(text or {}).get(s, s), markersize=5) for s in names]
        if handles:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                      fontsize=7, frameon=False, title=heading, title_fontsize=7)

    def tidy(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e6e8", lw=.5)
        ax.set_axisbelow(True)

    def condition(df, scope, task, main=True):
        if df.empty:
            return df
        select = df.scope.eq(scope) & df.task.eq(task)
        if main:
            select &= df.scenario.eq("main")
        return df[select]

    select = participants[participants.scenario.eq("main")]
    for (scope, task), p in select.groupby(["scope", "task"], sort=True):
        stage_names = stages(p)
        source_names = sorted(set(p.source), key=lambda s: (s != "gaze", s))
        # Composition is plotted from the disjoint areas, preserving overlap exactly once.
        fractions = present(["L_T", "L_P", "L_O", "L_N"], p)
        if len(fractions) == 4:
            fig, axs = axes_for(stage_names)
            series_colors = ["#238975", "#c6733c", "#826cad", "#c9ced0"]
            part_labels = ["Target only", "Gripper only", "Overlap", "Other"]
            for ax, stage in zip(axs, stage_names):
                grouped = p[p.stage == stage].groupby("source")[fractions].mean().reindex(source_names)
                base = np.zeros(len(grouped))
                for j, col in enumerate(fractions):
                    value = grouped[col].to_numpy(float)
                    ax.bar(np.arange(len(grouped)), np.nan_to_num(value), bottom=base,
                           color=series_colors[j], width=.7)
                    base += np.nan_to_num(value)
                ax.set_xticks(range(len(grouped)), source_names, rotation=32, ha="right", fontsize=7)
                ax.set_ylim(0, 1.02)
                ax.set_title(stage_title(stage), fontsize=9)
                if not np.isfinite(grouped.to_numpy()).any():
                    no_points(ax)
                tidy(ax)
            axs[0].set_ylabel("Fraction of valid occupancy")
            axs[len(stage_names)-1].legend(handles=[Patch(color=c, label=s)
                for c, s in zip(series_colors, part_labels)], loc="upper left",
                bbox_to_anchor=(1.01, 1), frameon=False, fontsize=7)
            title(fig, scope, task, "Occupancy by region")
            save(fig, scope, task, "regions")

        for metric in ["C", "L_target", "L_gripper"]:
            for kind, frame in [("performance", score_points), ("agreement", pair_table)]:
                if kind == "agreement" and scope != "paired_test":
                    continue
                selected = condition(frame, scope, task)
                if selected.empty:
                    continue
                selected = selected[selected.metric.eq(metric)]
                if selected.empty:
                    continue
                fig, axs = axes_for(stage_names)
                for ax, stage in zip(axs, stage_names):
                    g = selected[selected.stage == stage]
                    for s, items in g.groupby("source", sort=True):
                        x, y = (("mean_score", "mean_metric") if kind == "performance" else
                                ("gaze", "model"))
                        ax.scatter(items[x], items[y], facecolors="none", edgecolors=colors[s],
                                   marker=modes[s], s=34, linewidths=1, alpha=.85)
                    ax.set_ylim(-.035, 1.035)
                    if kind == "agreement":
                        ax.plot([0, 1], [0, 1], "--", color="#899097", lw=.8)
                        ax.set_xlim(-.035, 1.035)
                        ax.set_aspect("equal", adjustable="box")
                    if g.empty:
                        no_points(ax)
                    names = source_names if kind == "performance" else [s for s in source_names if s != "gaze"]
                    counts = g.groupby("source").participant.nunique().reindex(names, fill_value=0)
                    support = ("0" if counts.empty else str(counts.min()) if counts.min() == counts.max()
                               else f"{counts.min()}–{counts.max()} per source")
                    ax.set_title(f"{stage_title(stage)}\n(n={support})", fontsize=9)
                    tidy(ax)
                axs[0].set_ylabel(labels[metric] if kind == "performance" else f"Model {labels[metric].lower()}")
                fig.supxlabel("Participant mean SimNow overall score" if kind == "performance" else
                              f"Observed {labels[metric].lower()}", fontsize=9)
                keylabels = {}
                if kind == "performance":
                    for s, part in selected[selected.stage.eq("whole")].groupby("source"):
                        r = corr(part.mean_score, part.mean_metric) if len(part) >= 3 else np.nan
                        value = f"{r:.2f}" if np.isfinite(r) else "NA"
                        keylabels[s] = f"{s}  ρ={value}, n={len(part)}"
                legend(axs[len(stage_names)-1], sorted(selected.source.unique()), keylabels,
                       "Whole-task Spearman" if kind == "performance" else None)
                title(fig, scope, task, f"{labels[metric]} and " +
                      ("performance" if kind == "performance" else "real–model agreement"))
                save(fig, scope, task, f"{kind}_{metric}")

        if scope == "real_all":
            cols = present(TIME, p)
            if cols:
                fig, axs = axes_for(cols)
                gaze = p[p.source.eq("gaze")]
                for ax, m in zip(axs, cols):
                    for i, stage in enumerate(stage_names):
                        v = gaze[gaze.stage == stage][m].dropna()
                        if len(v):
                            ax.scatter(np.linspace(i-.09, i+.09, len(v)), v, color="#267d71",
                                       alpha=.5, s=15)
                            ax.plot([i-.22, i+.22], [v.mean()]*2, color="#263538", lw=1.6)
                    ax.set_xticks(range(len(stage_names)), stage_names, rotation=35, ha="right", fontsize=7)
                    ax.set_title(labels[m], fontsize=9)
                    tidy(ax)
                title(fig, scope, task, "Observed dwell and switching")
                save(fig, scope, task, "dwell_switch")

        if scope == "paired_test":
            cols = present(TIME, p)
            if cols:
                fig, axs = axes_for(cols)
                for ax, m in zip(axs, cols):
                    for j, s in enumerate(source_names):
                        v = p[p.source == s].groupby("stage")[m].mean().reindex(stage_names)
                        x = np.arange(len(stage_names)) + (j - (len(source_names)-1)/2)*.09
                        ax.scatter(x, v, color=colors[s], marker=modes[s], s=19)
                    ax.set_xticks(range(len(stage_names)), stage_names, rotation=35, ha="right", fontsize=7)
                    ax.set_title(labels[m], fontsize=9)
                    tidy(ax)
                legend(axs[len(cols)-1], source_names)
                title(fig, scope, task, "Dwell and switching on matched recordings")
                save(fig, scope, task, "dwell_switch")

        curves = condition(rounds, scope, task)
        if not curves.empty and "C" in curves:
            fig, axs = axes_for(stage_names)
            for ax, stage in zip(axs, stage_names):
                g = curves[(curves.stage == stage) & curves.source.eq("gaze")].dropna(subset=["C"])
                for _, v in g.groupby("participant"):
                    v = v.sort_values("round")
                    ax.plot(v["round"], v.C, "-o", color="#5a8e88", lw=.7, markersize=3, alpha=.55)
                if g.empty:
                    no_points(ax)
                ax.set_xticks([1, 2, 3])
                ax.set_ylim(-.04, 1.04)
                ax.set_title(stage_title(stage), fontsize=9)
                tidy(ax)
            axs[0].set_ylabel("Observed target preference")
            fig.supxlabel("Round (attempts averaged within each round)", fontsize=9)
            title(fig, scope, task, "Participant repetition trajectories")
            save(fig, scope, task, "repetition")

        sensitivities = condition(groups, scope, task, main=False)
        if not sensitivities.empty:
            sensitivities = sensitivities[(sensitivities.stage == "whole") &
                                          sensitivities.metric.isin(["C", "L_target", "L_gripper"])]
            scen = sorted(sensitivities.scenario.unique(), key=lambda x: (x != "main", x))
            if len(scen) > 1:
                fig, axs = axes_for(["C", "L_target", "L_gripper"])
                for ax, m in zip(axs, ["C", "L_target", "L_gripper"]):
                    z = sensitivities[sensitivities.metric == m]
                    for s, each in z.groupby("source"):
                        line = each.set_index("scenario").reindex(scen)
                        ax.plot(range(len(scen)), line["mean"], color=colors[s], marker=modes[s],
                                markersize=4, lw=.8)
                    ax.set_xticks(range(len(scen)), scen, rotation=35, ha="right", fontsize=7)
                    ax.set_ylim(-.025, 1.025)
                    ax.set_title(labels[m], fontsize=9)
                    tidy(ax)
                legend(axs[-1], source_names)
                title(fig, scope, task, "Whole-task sensitivity")
                save(fig, scope, task, "sensitivity")

        if scope == "paired_test":
            summary = condition(errors, scope, task)
            if not summary.empty:
                summary = summary[summary.stage.eq("whole")]
                fig, axs = axes_for(["target", "gripper"])
                for ax, event in zip(axs, EVENTS):
                    z = summary[summary.metric.eq(f"BSS_{event}")]
                    refs = [r for r in REFS if r in set(z.reference)]
                    models = sorted(set(z.source))
                    for i, s in enumerate(models):
                        v = z[z.source.eq(s)].set_index("reference").reindex(refs)
                        x = np.arange(len(refs)) + (i-(len(models)-1)/2)*.09
                        ax.vlines(x, v.ci_low, v.ci_high, color=colors[s], lw=.8)
                        ax.scatter(x, v.estimate, color=colors[s], s=18, marker=modes[s])
                    ax.axhline(0, lw=.8, ls="--", color="#5c6063")
                    ax.set_xticks(range(len(refs)), refs, rotation=35, ha="right", fontsize=8)
                    ax.set_title(event.capitalize() + " event", fontsize=9)
                    tidy(ax)
                axs[0].set_ylabel("Brier skill relative to reference")
                legend(axs[-1], sorted(set(summary.source)))
                title(fig, scope, task, "Whole-task Brier skill with participant intervals")
                save(fig, scope, task, "brier_skill")
            c = condition(confusion, scope, task)
            if not c.empty:
                c = c[c.stage.eq("whole")]
                names = sorted(set(c.source))
                if names:
                    fig, axs = axes_for(names)
                    for ax, s in zip(axs, names):
                        m = c[c.source.eq(s)].pivot(index="actual", columns="predicted",
                            values="conditional_given_actual").reindex(index=PARTS, columns=PARTS)
                        im = ax.imshow(m.to_numpy(float), vmin=0, vmax=1, cmap="Blues")
                        for i in range(4):
                            for j in range(4):
                                val = m.iloc[i, j]
                                ax.text(j, i, f"{val:.2f}" if np.isfinite(val) else "—",
                                        ha="center", va="center", fontsize=8,
                                        color="white" if val > .6 else "#223245")
                        ax.set_xticks(range(4), PARTS)
                        ax.set_yticks(range(4), PARTS)
                        ax.set_title(s, fontsize=9)
                    axs[0].set_ylabel("Observed class")
                    fig.supxlabel("Predicted class (T target only, P gripper only, O overlap, N other)", fontsize=9)
                    title(fig, scope, task, "Whole-task confusion given the observed class")
                    save(fig, scope, task, "confusion")
    return paths


def finish(out: Path, trials: pd.DataFrame, errors: pd.DataFrame,
           confusions: pd.DataFrame, seed=42, nboot=10000) -> dict:
    """Write participant-balanced outputs; caller owns frame and trial measurements."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if trials.empty:
        raise ValueError("The statistics stage requires a trial inventory")
    essential = PERSON + ["content_id", "score", "valid_n"]
    missing = [c for c in essential if c not in trials]
    if missing:
        raise ValueError(f"Missing trial columns: {missing}")
    d = trials.copy()
    idcols = PERSON + ["content_id"]
    if d.duplicated(idcols).any():
        raise ValueError("Duplicate trial/source/stage/scenario key")
    measurements = present(OCC + AREA + TIME, d)
    for m in measurements + ["score", "valid_n"]:
        d[m] = pd.to_numeric(d[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
    d.loc[d.valid_n.fillna(0).le(0), measurements] = np.nan
    boot = ParticipantBootstrap(seed, nboot)
    participants, groups = participant_summary(d, boot)
    correlations, score_points, loo = performance(d, boot)
    error_persons, model_summary, contrasts = model_statistics(errors, d, boot)
    rounds, changes = round_statistics(d, boot)
    confusion = confusion_statistics(confusions)
    paired = agreement_pairs(d)
    outputs = {
        "participant_stage_summary": participants,
        "group_stage_summary": groups,
        "participant_model_agreement": paired,
        "performance_correlations": correlations,
        "performance_participants": score_points,
        "performance_leave_one_out": loo,
        "participant_model_errors": error_persons,
        "model_error_summary": model_summary,
        "model_paired_contrasts": contrasts,
        "participant_round_summary": rounds,
        "repetition_contrasts": changes,
        "confusion_summary": confusion,
    }
    for name, frame in outputs.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast",
                                    category=RuntimeWarning)
            frame.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig", na_rep="")
    generated = figures(out, participants, paired, score_points, groups, rounds,
                        model_summary, confusion)
    return {"rows": {k: len(v) for k, v in outputs.items()}, "figure_files": generated,
            "bootstrap_replicates": int(nboot), "bootstrap_seed": int(seed)}
