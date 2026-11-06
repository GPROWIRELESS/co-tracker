"""Trial-balanced, participant-level summaries for the target-locking analysis."""

from itertools import combinations
from pathlib import Path
import json
import re
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
REFS = ["train_heatmap", "center_point", "uniform_area", "train_rate"]
DIAGNOSTIC = "GT_smoothed"
PARTS = ["T", "P", "O", "N"]
BRANCHES = {"probability_mc": "main_probability",
            "decoded_expectation": "decoded_expectation",
            "decoded_argmax": "decoded_argmax",
            "decoded_sample": "decoded_sample"}


def is_temporal(name):
    return bool(re.search(r"(^|_)(dwell|direct|via)_|(^|_)all_.*_runs($|_)", name))


def is_hard_error(name):
    return bool(re.search(r"(^|_)(FP|FN|FPR|FNR|TP|TN)(_|$)", name))


def probability_columns(frame):
    """Reject inapplicable measurements even when mixed-branch frames share a schema."""
    return frame.drop(columns=[c for c in frame if is_temporal(c) or is_hard_error(c)])


def decoded_names(frame, branch, restore=False):
    """Make the trajectory decoder explicit in every exported temporal measurement."""
    if frame.empty and not len(frame.columns):
        return frame
    def name(value):
        if not isinstance(value, str):
            return value
        prefix = branch + "_"
        if restore and value.startswith(prefix) and is_temporal(value):
            return value[len(prefix):]
        if not restore and is_temporal(value) and not value.startswith(prefix):
            return prefix + value
        return value
    result = frame.rename(columns=name)
    if "metric" in result:
        result = result.copy()
        result["metric"] = result.metric.map(name)
    return result


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
    # The smoothed observation is a reference diagnostic, kept outside model
    # rankings and outside Brier skill denominators.
    e = e.drop(columns=[c for c in e if c.endswith("_" + DIAGNOSTIC)])
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
    e = e.drop(columns=present(["delta_" + m for m in temporal], e))
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


def reference_statistics(errors, boot):
    """Deduplicate identical reference losses across sources before aggregation."""
    ids = PAIR + ["participant", "content_id"]
    columns = present([f"BS_{event}_{ref}" for event in EVENTS
                       for ref in REFS + [DIAGNOSTIC]], errors)
    if errors.empty or not columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    e = errors.copy()
    if "scope" not in e:
        e["scope"] = "paired_test"
    records = []
    for key, g in group_items(e, ids):
        row = dict(key)
        for col in columns:
            values = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            good = np.isfinite(values)
            if good.any() and (not good.all() or
                              not np.allclose(values, values[0], rtol=1e-10, atol=1e-12)):
                raise ValueError(f"Reference loss differs between models for {key}, {col}")
            row[col] = values[0] if good.any() else np.nan
        records.append(row)
    trials = pd.DataFrame(records)
    summaries = []
    for key, g in group_items(trials, PAIR):
        by = g.groupby("participant", sort=True)[columns].mean()
        for event in EVENTS:
            for ref in REFS + [DIAGNOSTIC]:
                col = f"BS_{event}_{ref}"
                if col not in by:
                    continue
                v = by[col].dropna().to_numpy(float)
                summaries.append({**key, "reference": ref, "event": event,
                                  "reference_role": "smoothed_observation_diagnostic"
                                  if ref == DIAGNOSTIC else "evaluation_baseline",
                                  "metric": col, "estimate": v.mean() if len(v) else np.nan,
                                  "n_participants": len(v), "n_trials": int(g[col].count()),
                                  **boot.mean(v)})
    summary = pd.DataFrame(summaries)
    return (trials, summary[summary.reference.ne(DIAGNOSTIC)].copy(),
            summary[summary.reference.eq(DIAGNOSTIC)].copy())


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


def figures(out, participants, pair_table, score_points, groups, rounds, errors, confusion,
            analysis_branch=None):
    """Render selected summaries at the primary radius, plus a sensitivity panel."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if analysis_branch is None and "analysis_branch" in participants:
        unique = participants.analysis_branch.dropna().unique()
        if len(unique) != 1:
            raise ValueError("Each figure set requires exactly one analysis branch")
        analysis_branch = unique[0]
    if analysis_branch not in BRANCHES:
        raise ValueError("A recognized analysis_branch is required for figure labels")
    if analysis_branch.startswith("decoded_"):
        participants, pair_table, score_points, groups, rounds, errors, confusion = [
            decoded_names(frame, analysis_branch, restore=True)
            for frame in [participants, pair_table, score_points, groups, rounds, errors, confusion]]

    figures_dir = out / "figures"
    palette = list(plt.get_cmap("tab10").colors)
    sources = sorted(set(participants.source) - {"gaze"})
    colors = {"gaze": "#242424", **{s: palette[i % len(palette)] for i, s in enumerate(sources)}}
    modes = {s: ["o", "s", "^", "D", "v", "P", "X"][i % 7]
             for i, s in enumerate(["gaze"] + sources)}
    paths = []
    labels = {"C": "Relative target preference", "L_target": "Target occupancy",
              "L_gripper": "DLC wrist-region occupancy", "dwell_T_ms": "Target-only dwell (ms)",
              "dwell_P_ms": "Wrist-only dwell (ms)", "dwell_O_ms": "Overlap dwell (ms)",
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
        names = {"probability_mc": "Probability mass",
                 "decoded_expectation": "Decoded expectation",
                 "decoded_argmax": "Decoded argmax",
                 "decoded_sample": "Independent per-frame sample decoding"}
        detail = f"{data_name} | {names[analysis_branch]}"
        if analysis_branch == "decoded_sample" and "switch" in text.lower():
            detail += "\nTemporal summaries describe the sampled path, not distribution continuity"
        fig.suptitle(f"Task {task}  |  {text}\n{detail}", fontsize=12, fontweight="bold")

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
            part_labels = ["Target only", "Wrist only", "Overlap", "Other"]
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
                    ax.set_title("Target event" if event == "target" else "DLC wrist-region event", fontsize=9)
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
                    fig.supxlabel("Predicted class (T target only, P wrist only, O overlap, N other)", fontsize=9)
                    title(fig, scope, task, "Whole-task confusion given the observed class")
                    save(fig, scope, task, "confusion")
    return paths


def finish_branch(out: Path, trials: pd.DataFrame, errors: pd.DataFrame,
                  confusions: pd.DataFrame, analysis_branch, seed=42, nboot=10000) -> dict:
    """Write one isolated branch with a fixed hierarchy and matched trial support."""
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
    references, baseline_summary, diagnostic_summary = reference_statistics(errors, boot)
    rounds, changes = round_statistics(d, boot)
    confusion = confusion_statistics(confusions)
    paired = agreement_pairs(d)
    outputs = {
        "trial_stage": d,
        "model_trial_errors": errors,
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
        "baseline_reference_bytrial": references,
        "baseline_reference_summary": baseline_summary,
        "GT_smoothed_diagnostic_summary": diagnostic_summary,
    }
    if analysis_branch.startswith("decoded_"):
        outputs["trial_confusions"] = confusions
        outputs["confusion_summary"] = confusion
    for name, frame in outputs.items():
        frame = frame.copy()
        if analysis_branch.startswith("decoded_"):
            frame = decoded_names(frame, analysis_branch)
        frame["analysis_branch"] = analysis_branch
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast",
                                    category=RuntimeWarning)
            frame.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig", na_rep="")
    generated = figures(out, participants, paired, score_points, groups, rounds,
                        model_summary, confusion, analysis_branch=analysis_branch)
    details = {"analysis_branch": analysis_branch,
               "rows": {k: len(v) for k, v in outputs.items()}, "figure_files": generated,
               "bootstrap_replicates": int(nboot), "bootstrap_seed": int(seed),
               "references_for_bss": REFS,
               "GT_smoothed_role": "smoothed observation diagnostic; no BSS denominator"}
    write_branch_report(out, details)
    (out / "statistics_manifest.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return details


def write_branch_report(out, details):
    """Document the estimand next to its tables so branches cannot be conflated."""
    branch = details["analysis_branch"]
    lines = ["# Target Locking 分支结果", "", f"分析分支为 {branch}。", "",
             "L_gripper / gripper 事件表示双侧 DLC 腕部或铰接点邻域占用（DLC wrist-region occupancy），"
             "并非经过验证的器械尖端。L_P 是 wrist-only 独占区域，L_O 是 target 与该腕部邻域的重叠。", ""]
    if branch == "probability_mc":
        lines.extend([
            "模型占用由逐帧区域概率质量计算。target 与 DLC wrist-region 事件概率包含 overlap；"
            "C 使用独占 target 概率的时长加权和除以两个独占区域概率的时长加权和。"
            "先在每个试次内计算，再在参与者内对试次等权平均，参与者之间等权。",
            "",
            "事件 Brier 使用模型的区域概率与真实 gaze 的二元命中比较。"
            "本分支只输出概率占用、面积校正、事件损失及其统计关系。",
        ])
    else:
        rule = branch.removeprefix("decoded_")
        lines.extend([
            f"本分支使用 {rule} 解码得到的模型位置，计算点占用、硬事件错误、"
            "混淆矩阵以及解码轨迹的驻留和切换。gaze 来源保留实际观测轨迹。",
            "",
            f"所有时间指标在导出列名及 metric 值中带有 {branch}_ 前缀。"
            "前缀标识本次比较的解码分支；gaze 行仍表示真实 gaze 的时间指标。",
        ])
        if branch == "decoded_sample":
            lines.extend([
                "",
                "**随机解码轨迹指标，非分布时间连贯性。** 每帧以固定随机方案独立选择一个 sample。"
                "跨帧样本没有经过联合轨迹验证，驻留和切换只能描述此次随机解码得到的路径。",
            ])
    lines.extend([
        "",
        f"统计使用 {details['bootstrap_replicates']:,} 次参与者 bootstrap，"
        f"随机种子为 {details['bootstrap_seed']}。模型差值按共同试次配对后在参与者内平均。"
        "BSS 使用参与者等权平均 Brier 的比值。相关分析先在指标与分数均有效的同一组试次内"
        "形成参与者均值；相关及阶段、轮次结果为描述和探索性结果。",
        "",
        "BSS 的参照为 train_heatmap、center_point、uniform_area 和 train_rate。"
        "各参照的逐试次 Brier 单独保存在 baseline_reference_bytrial.csv，"
        "其中 GT_smoothed 列是平滑真实 gaze 的诊断参考。"
        "GT_smoothed 的独立汇总位于 GT_smoothed_diagnostic_summary.csv，"
        "不作为训练基线或模型上限，也不进入 BSS 分母和模型参照优劣图。",
        "",
        "## 输出索引", "",
    ])
    lines.extend(f"- [{name}.csv]({name}.csv)，{count:,} 行"
                 for name, count in details["rows"].items())
    lines.extend(["", "## 图表", ""])
    lines.extend(f"- [{name}]({name})" for name in details["figure_files"])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finish(out: Path, trials: pd.DataFrame, errors: pd.DataFrame,
           confusions: pd.DataFrame, seed=42, nboot=10000) -> dict:
    """Keep probabilistic forecasts and every trajectory decoder in separate outputs."""
    if trials.empty:
        raise ValueError("The statistics stage requires a trial inventory")
    if "analysis_branch" not in trials:
        raise ValueError("trial_stage must identify analysis_branch explicitly")
    branches = list(trials.analysis_branch.dropna().unique())
    unknown = set(branches) - set(BRANCHES)
    if unknown or trials.analysis_branch.isna().any():
        raise ValueError(f"Unknown or missing analysis branches: {sorted(unknown)}")
    for label, frame in [("errors", errors), ("confusions", confusions)]:
        if not frame.empty:
            if "analysis_branch" not in frame:
                raise ValueError(f"{label} must identify analysis_branch explicitly")
            if frame.analysis_branch.isna().any() or not set(frame.analysis_branch).issubset(branches):
                raise ValueError(f"{label} contain unknown or missing analysis branches")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for branch, directory in BRANCHES.items():
        if branch not in branches:
            continue
        t = trials[trials.analysis_branch.eq(branch)].copy()
        e = errors[errors.analysis_branch.eq(branch)].copy() if not errors.empty else errors.copy()
        c = confusions[confusions.analysis_branch.eq(branch)].copy() if not confusions.empty else confusions.copy()
        if branch == "probability_mc":
            t, e = probability_columns(t), probability_columns(e)
            c = pd.DataFrame()
        else:
            t, e = decoded_names(t, branch, restore=True), decoded_names(e, branch, restore=True)
        results[branch] = finish_branch(out / directory, t, e, c, branch, seed, nboot)
    manifest = {
        "branches": results, "bootstrap_replicates": int(nboot), "bootstrap_seed": int(seed),
        "rows": {f"{BRANCHES[b]}/{name}": n for b, r in results.items() for name, n in r["rows"].items()},
        "figure_files": [f"{BRANCHES[b]}/{name}" for b, r in results.items() for name in r["figure_files"]],
    }
    (out / "statistics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
