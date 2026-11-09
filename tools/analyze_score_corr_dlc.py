#!/usr/bin/env python3
"""Relate valid recording-level gaze/PSM peaks to matched performance scores.

Reads saved results only. NumPy, SciPy and Matplotlib are sufficient; no NAS,
Pandas, or changes to the original correlation analysis are required.
"""
from __future__ import annotations
from gaze_psm_modes import mode_rows, normalize_modes, seed_identity

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/score_corr_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats

GROUP = ("task", "side", "signal", "stage", "mode")
PREDICTORS = ("signed_lag_s", "absolute_lag_s", "peak_r")
METHODS = ("between_participants", "within_participant", "within_adjusted_order")
PHASE_ORDER = {
    "__whole__": 0, "reach ring": 1, "grab ring": 2, "transfer ring": 3,
    "place ring": 4, "drop ring": 5, "reach string": 1,
    "threading string through tower": 2, "wrap around psm": 3,
    "pull through and cinch": 4,
}


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(mode_rows(csv.DictReader(stream)))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple(row[k] for k in GROUP)


def record_key(row):
    return (row["content_id"], *key(row))


def rng_for(seed, *parts):
    digest = hashlib.sha256(repr(seed_identity(parts)).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([seed, int.from_bytes(digest[:4], "little")]))


def row_corr(x, y):
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    den = np.sqrt(np.sum(x*x, axis=-1)*np.sum(y*y, axis=-1))
    return np.divide(np.sum(x*y, axis=-1), den,
                     out=np.full(np.shape(den), np.nan), where=den > 1e-12)


def spearman(x, y):
    return float(row_corr(stats.rankdata(x), stats.rankdata(y)))


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return (*np.percentile(finite, [2.5, 97.5]), len(finite)) if len(finite) else (np.nan, np.nan, 0)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    q = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    if len(order):
        q[order] = np.minimum(1, np.minimum.accumulate(
            (p[order]*len(order)/np.arange(1, len(order)+1))[::-1])[::-1])
    return q


def between_test(x, y, rng, bootstrap, permutations, minimum):
    out = {"n_users": len(x), "n_informative_users": len(x)}
    if len(x) < minimum or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {**out, "status": "insufficient_users_or_variation"}
    rho = spearman(x, y)
    xr, yr = stats.rankdata(x), stats.rankdata(y)
    permutation_indices = np.argsort(rng.random((permutations, len(y))), axis=1)
    null = row_corr(np.broadcast_to(xr, permutation_indices.shape), yr[permutation_indices])
    p = (1 + np.count_nonzero(np.abs(null) >= abs(rho)-1e-12))/(permutations+1)
    indices = rng.integers(len(x), size=(bootstrap, len(x)))
    boot = row_corr(stats.rankdata(x[indices], axis=1), stats.rankdata(y[indices], axis=1))
    low, high, nb = interval(boot)
    loo = [spearman(np.delete(x, i), np.delete(y, i)) for i in range(len(x))]
    return {**out, "status": "ok", "association": rho, "ci_low": low, "ci_high": high,
            "p": p, "bootstrap_valid": nb, "loo_min": float(np.nanmin(loo)),
            "loo_max": float(np.nanmax(loo)), "pearson_descriptive": float(row_corr(x, y)),
            "p_method": "two_sided_participant_score_permutation"}


def partial_from_gram(gram, adjusted):
    """Partial x/y correlation from weighted within-participant crossproducts."""
    vxx, vyy, vxy = gram[..., 0, 0], gram[..., 1, 1], gram[..., 0, 1]
    if adjusted:
        inv = np.linalg.pinv(gram[..., 2:, 2:], rcond=1e-10, hermitian=True)
        gx, gy = gram[..., 0, 2:], gram[..., 1, 2:]
        vxx = vxx - np.einsum("...i,...ij,...j->...", gx, inv, gx)
        vyy = vyy - np.einsum("...i,...ij,...j->...", gy, inv, gy)
        vxy = vxy - np.einsum("...i,...ij,...j->...", gx, inv, gy)
    den = np.sqrt(np.maximum(vxx, 0)*np.maximum(vyy, 0))
    return np.divide(vxy, den, out=np.full(np.shape(den), np.nan), where=den > 1e-10)


def within_test(records, predictor, adjusted, rng, bootstrap, minimum, outcome="Score"):
    """Rank-linear fixed effects association, equal total weight per participant.

    P values use a participant-cluster sandwich with CR1 correction and t(G-1).
    Bootstrap resamples whole participants, retaining the original rank scores.
    These small-cluster tests are exploratory approximations.
    """
    users = sorted({r["user"] for r in records})
    user_array = np.array([r["user"] for r in records])
    x = stats.rankdata([r[predictor] for r in records])
    y = stats.rankdata([r[outcome] for r in records])
    raw = np.column_stack((x, y, [r["TaskNum"] for r in records],
                           [r["Attempt"] for r in records]))
    grams, pieces, weights = [], [], []
    informative = 0
    for user in users:
        block = raw[user_array == user].copy()
        block -= block.mean(axis=0)
        informative += bool(np.ptp(block[:, 0]) > 0 and np.ptp(block[:, 1]) > 0)
        weight = np.full(len(block), 1/len(block))
        grams.append(block.T @ (block*weight[:, None]))
        pieces.append(block)
        weights.append(weight)
    out = {"n_users": len(users), "n_informative_users": informative}
    if informative < minimum:
        return {**out, "status": "insufficient_users_with_within_variation"}
    grams = np.array(grams)
    total = grams.sum(axis=0)
    rho = float(partial_from_gram(total, adjusted))
    if not np.isfinite(rho):
        return {**out, "status": "insufficient_residual_variation"}
    block = np.vstack(pieces)
    w = np.concatenate(weights)
    design = block[:, [0, 2, 3]] if adjusted else block[:, [0]]
    response = block[:, 1]
    bread = np.linalg.pinv(design.T @ (design*w[:, None]), rcond=1e-10, hermitian=True)
    beta = bread @ (design.T @ (w*response))
    residual = response - design @ beta
    cluster_scores = []
    offset = 0
    for piece, weight in zip(pieces, weights):
        end = offset + len(piece)
        cluster_scores.append(design[offset:end].T @ (weight*residual[offset:end]))
        offset = end
    cluster_scores = np.array(cluster_scores)
    n, g = len(records), len(users)
    # Absorbed participant intercepts count in the finite-sample correction.
    rank = np.linalg.matrix_rank(design)
    df_resid = n-g-rank
    if df_resid <= 0 or design[:, 0] @ (w*design[:, 0]) < 1e-10:
        return {**out, "status": "insufficient_residual_degrees_of_freedom"}
    covariance = bread @ (cluster_scores.T @ cluster_scores) @ bread
    covariance *= g/(g-1)*(n-1)/df_resid
    se = np.sqrt(max(covariance[0, 0], 0))
    p = float(2*stats.t.sf(abs(beta[0]/se), g-1)) if se > 1e-12 else np.nan
    indices = rng.integers(g, size=(bootstrap, g))
    boot = partial_from_gram(grams[indices].sum(axis=1), adjusted)
    low, high, nb = interval(boot)
    loo = partial_from_gram(total[None, :, :]-grams, adjusted)
    return {**out, "status": "ok", "association": rho, "ci_low": low, "ci_high": high,
            "p": p, "bootstrap_valid": nb, "loo_min": float(np.nanmin(loo)),
            "loo_max": float(np.nanmax(loo)), "rank_slope": float(beta[0]),
            "rank_slope_cluster_se": float(se), "cluster_df": g-1,
            "p_method": "participant_cluster_CR1_t_approximation"}


def prepare(args, outcome="Score"):
    source, index = args.input, args.index
    config = normalize_modes(json.loads((source/"config.json").read_text()))
    if config.get("partial_record"):
        raise ValueError("Refusing partial-record correlation results")
    summary = read_csv(source/"record_summary.csv")
    groups = {key(r): r for r in read_csv(source/"group_summary.csv")}
    if len({record_key(r) for r in summary}) != len(summary):
        raise ValueError("Duplicate correlation summary key")
    score_rows, scores = [], {}
    for task in ("A", "C"):
        for row in read_csv(index/f"Task_{task}_Score.csv"):
            cid = f"User{int(row['UserID'])}_{int(row['TaskNum'])}_{int(row['Attempt'])}"
            if cid in scores:
                raise ValueError(f"Duplicate score key: {cid}")
            row = {**row, "task": task, "content_id": cid}
            scores[cid] = row
            score_rows.append(row)
    indexed = {r["content_id"]: r for r in read_csv(index/"content_index.csv")}
    quality = {r["content_id"]: r for r in read_csv(source/"quality.csv")}
    # Retain only the few diagnostics needed rather than loading results.json.
    curves = defaultdict(list)
    with (source/"record_curves.csv").open(newline="") as stream:
        for row in mode_rows(csv.DictReader(stream)):
            curves[record_key(row)].append((number(row["lag_s"]), number(row["r"])))
    expected = np.arange(-round(config["max_lag"]*config["hz"]),
                         round(config["max_lag"]*config["hz"])+1)/config["hz"]
    curve_info = {}
    for k, values in curves.items():
        lag, r = np.array(values).T
        if len(lag) != len(expected) or not np.allclose(lag, expected, atol=1e-12):
            raise ValueError(f"Unexpected lag grid: {k}")
        finite = np.isfinite(r)
        peak = np.nanargmax(r) if finite.any() else None
        curve_info[k] = {
            "full_lag_support": bool(finite.all()), "finite_lags": int(finite.sum()),
            "r_zero": float(r[np.argmin(abs(lag))]),
            "curve_peak_lag_s": float(lag[peak]) if peak is not None else np.nan,
            "curve_peak_r": float(r[peak]) if peak is not None else np.nan,
            "near_peak_width_0p01_s": float(np.ptp(lag[finite & (r >= r[peak]-.01)]))
            if peak is not None else np.nan,
        }
    joined, record_audit = {}, []
    for row in summary:
        cid = row["content_id"]
        if cid not in joined:
            s = scores.get(cid)
            ix, qc = indexed[cid], quality[cid]
            assert ix["Task Name"] == row["task"]
            assert str(int(ix["User Num"])) == row["user"].removeprefix("User")
            if s:
                assert s["task"] == row["task"]
                assert str(int(s["TaskNum"])) == str(int(ix["Task Num"]))
                assert str(int(s["Attempt"])) == str(int(ix["Attempt Num"]))
                assert s["UserType"].lower() == ix["src_skill"].lower()
            reasons = []
            if s is None or not np.isfinite(number(s[outcome])):
                reasons.append("missing_score" if outcome == "Score" else "missing_"+outcome)
            if s and s["AttemptStatus"] not in ("Completed", "Failed"):
                reasons.append("unscored_attempt_status")
            if qc["status"] != "ok" or qc["partial_record"].lower() == "true":
                reasons.append("record_quality_or_partial")
            joined[cid] = {
                "content_id": cid, "user": row["user"], "task": row["task"],
                "TaskNum": int(ix["Task Num"]), "Attempt": int(ix["Attempt Num"]),
                "AttemptStatus": s["AttemptStatus"] if s else "missing",
                "Score": number(s["Score"]) if s else np.nan,
                **{k: number(s.get(k)) if s else np.nan for k in
                   ("EcoRaw", "Eco", "TimeRaw", "Time", "PenaltyRawSum", "PenaltySum")},
                "score_usable": not reasons, "score_exclusion": ";".join(reasons),
                "duration_s": number(qc["duration_s"]),
            }
        g = groups[key(row)]
        info = curve_info[record_key(row)]
        lag, peak_r = number(row["peak_lag_s"]), number(row["peak_r"])
        assert np.allclose([lag, peak_r], [info["curve_peak_lag_s"], info["curve_peak_r"]],
                           atol=1e-10, equal_nan=True)
        group_reasons = []
        if g["status"] != "ok":
            group_reasons.append("group_"+g["status"])
        if not np.isfinite(number(g["peak_r"])) or number(g["peak_r"]) <= 0:
            group_reasons.append("group_no_positive_peak")
        if int(g["n_stable_users"]) < 3:
            group_reasons.append("group_fewer_than_3_users")
        reasons = list(group_reasons)
        if not joined[cid]["score_usable"]:
            reasons.append(joined[cid]["score_exclusion"])
        if row["status"] != "ok":
            reasons.append("record_"+row["status"])
        if not np.isfinite(peak_r) or peak_r <= 0:
            reasons.append("record_no_positive_peak")
        if not info["full_lag_support"]:
            reasons.append("incomplete_lag_support")
        record_audit.append({**joined[cid], **{k: row[k] for k in GROUP},
                             "signed_lag_s": lag, "absolute_lag_s": abs(lag),
                             "peak_r": peak_r, "record_peak_status": row["status"],
                             **info, "peak_gain_over_zero": peak_r-info["r_zero"],
                             "pairs_at_peak": number(row["pairs_at_peak"]),
                             "group_status": g["status"],
                             "group_usable": not group_reasons,
                             "eligible": not reasons, "exclusion_reasons": ";".join(reasons)})
    phases = []
    for k, g in groups.items():
        rr = [r for r in record_audit if key(r) == k]
        eligible = [r for r in rr if r["eligible"]]
        phases.append({**{f: g[f] for f in GROUP}, "group_status": g["status"],
                       "group_usable": rr[0]["group_usable"],
                       "group_peak_lag_s": number(g["peak_lag_s"]),
                       "group_peak_r": number(g["peak_r"]),
                       "group_peak_ci_low_s": number(g["peak_lag_ci_low"]),
                       "group_peak_ci_high_s": number(g["peak_lag_ci_high"]),
                       "group_bootstrap_boundary_frequency": number(g["boundary_frequency"]),
                       "n_group_users": int(g["n_stable_users"]), "n_recordings": len(rr),
                       "n_eligible_recordings": len(eligible),
                       "n_eligible_users": len({r["user"] for r in eligible}),
                       "n_record_boundary": sum(r["record_peak_status"] == "boundary_peak" for r in rr),
                       "n_incomplete_lag_support": sum(not r["full_lag_support"] for r in rr),
                       "median_peak_gain_over_zero": float(np.median([r["peak_gain_over_zero"] for r in eligible]))
                       if eligible else np.nan,
                       "median_near_peak_width_0p01_s": float(np.median([r["near_peak_width_0p01_s"] for r in eligible]))
                       if eligible else np.nan})
    for row in score_rows:
        cid = row["content_id"]
        row["has_correlation_recording"] = cid in joined
        row["score_used"] = bool(cid in joined and joined[cid]["score_usable"])
        row["inventory_reason"] = (joined[cid]["score_exclusion"] if cid in joined
                                   else "no_recording_in_novice_correlation_run")
    return config, score_rows, list(joined.values()), record_audit, phases


def scenario_rows(records, scenario):
    selected = [r for r in records if r["eligible"]]
    if scenario == "Both in stage":
        return [r for r in selected if r["signal"] == "position" and r["mode"] == "Both in stage"]
    signal = "velocity" if scenario == "velocity_secondary" else "position"
    selected = [r for r in selected if r["signal"] == signal and r["mode"] in ("whole", "PSM in stage")]
    if scenario == "completed_only":
        selected = [r for r in selected if r["AttemptStatus"] == "Completed"]
    if scenario == "stronger_peaks":
        selected = [r for r in selected if r["peak_r"] >= .2]
    return selected


def aggregate_participants(records, outcome="Score"):
    groups = defaultdict(list)
    for row in records:
        groups[row["user"]].append(row)
    return [{"user": user, "n_recordings": len(rr),
             "content_ids": ";".join(r["content_id"] for r in rr),
             outcome: float(np.mean([r[outcome] for r in rr])),
             "signed_lag_s": float(np.median([r["signed_lag_s"] for r in rr])),
             "absolute_lag_s": float(np.median([r["absolute_lag_s"] for r in rr])),
             "peak_r": float(np.tanh(np.mean(np.arctanh(np.clip([r["peak_r"] for r in rr], -1+1e-7, 1-1e-7)))))}
            for user, rr in sorted(groups.items())]


def analyze(args, records, phases, outcome="Score", scenarios=None):
    output, participant_rows = [], []
    if scenarios is None:
        scenarios = ("primary", "Both in stage", "completed_only", "stronger_peaks", "velocity_secondary")
    for scenario in scenarios:
        print(f"Analyzing {scenario}", flush=True)
        chosen = scenario_rows(records, scenario)
        by_group = defaultdict(list)
        for row in chosen:
            by_group[key(row)].append(row)
        # Include empty groups in result tables so exclusions remain visible.
        for phase in phases:
            signal = "velocity" if scenario == "velocity_secondary" else "position"
            modes = ("Both in stage",) if scenario == "Both in stage" else ("whole", "PSM in stage")
            if phase["signal"] != signal or phase["mode"] not in modes:
                continue
            k = key(phase)
            rr = by_group[k]
            people = aggregate_participants(rr, outcome)
            meta = {"scenario": scenario, **dict(zip(GROUP, k))}
            if outcome != "Score":
                meta["outcome"] = outcome
            participant_rows.extend({**meta, **person} for person in people)
            for predictor in PREDICTORS:
                for method in METHODS:
                    row = {**meta, "predictor": predictor, "method": method,
                           "n_recordings": len(rr), "n_users": len(people),
                           "group_status": phase["group_status"],
                           "family": f"{scenario}/{method}/all_three_predictors"}
                    if outcome != "Score":
                        row["family"] = outcome+"/"+row["family"]
                    rng = (rng_for(args.seed, scenario, k, predictor, method) if outcome == "Score" else
                           rng_for(args.seed, outcome, scenario, k, predictor, method))
                    if not phase["group_usable"]:
                        result = {"status": "excluded_group_"+phase["group_status"]}
                    elif method == "between_participants":
                        result = between_test(np.array([p[predictor] for p in people]),
                                              np.array([p[outcome] for p in people]), rng,
                                              args.bootstrap, args.permutations, args.min_users)
                    else:
                        result = within_test(rr, predictor, method == "within_adjusted_order", rng,
                                             args.bootstrap, args.min_users, outcome)
                    output.append({**row, **result})
    families = defaultdict(list)
    for row in output:
        families[row["family"]].append(row)
    for family in families.values():
        q = bh_adjust([r.get("p", np.nan) for r in family])
        size = sum(np.isfinite(r.get("p", np.nan)) for r in family)
        for row, qq in zip(family, q):
            row.update(q_bh=float(qq), family_size=size)
    return output, participant_rows


def phase_label(row):
    name = {"__whole__": "Whole task", "threading string through tower": "Threading",
            "pull through and cinch": "Pull/cinch", "wrap around psm": "Wrap around PSM"}.get(row["stage"], row["stage"].title())
    return f"{row['task']}: {name} ({row['side'].upper()})"


def group_order(row):
    return (row["task"], PHASE_ORDER[row["stage"]], row["side"])


def make_association_overview(results, phases, outcome="Score", coverage=None):
    """Shared layout for overall-score and component association figures."""
    primary_groups = sorted([p for p in phases if p["signal"] == "position" and
                             p["mode"] in ("whole", "PSM in stage")], key=group_order)
    lookup = {(r["method"], key(r), r["predictor"]): r for r in results if r["scenario"] == "primary"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 12), sharey=True)
    titles = ("Between participants", "Within participant",
              "Within participant\nadjusted for task order + attempt")
    for ax, method, title in zip(axes, METHODS, titles):
        matrix = np.full((len(primary_groups), 3), np.nan)
        for i, g in enumerate(primary_groups):
            for j, predictor in enumerate(PREDICTORS):
                result = lookup[(method, key(g), predictor)]
                matrix[i, j] = result.get("association", np.nan)
                if np.isfinite(matrix[i, j]):
                    label = f"{matrix[i,j]:+.2f}"
                    if result.get("q_bh", 1) < .05:
                        label += " *"
                    ax.text(j, i, label, ha="center", va="center", fontsize=9,
                            color="white" if abs(matrix[i,j]) > .65 else "black")
                else:
                    ax.text(j, i, "excluded" if not g["group_usable"] else "low N",
                            ha="center", va="center", fontsize=7, color="dimgray")
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(3), ("Signed lag", "Absolute lag", "Peak r"))
        ax.set_yticks(range(len(primary_groups)), [phase_label(p) for p in primary_groups])
        ax.set_title(title, fontsize=12)
    heading = "Performance score" if outcome == "Score" else outcome
    fig.suptitle(heading+" vs gaze/PSM peak features (position)\n"
                 "Positive signed lag = gaze leads; * BH q < 0.05 within each analysis panel", fontsize=14)
    fig.subplots_adjust(left=.23, right=.88, top=.9, wspace=.08)
    color_ax = fig.add_axes([.91, .22, .016, .55])
    fig.colorbar(im, cax=color_ax, label="Rank association")
    if outcome != "Score":
        if outcome in ("Time", "Eco"):
            footer = (f"Supplied {outcome} component score used directly. "
                      f"Positive = higher {outcome} score; negative = lower {outcome} score.")
        else:
            interpretation = ("Positive = longer duration; negative = shorter duration." if outcome == "TimeRaw" else
                              "Positive = larger raw motion measure; negative = smaller raw motion measure.")
            footer = f"{outcome} used directly, without sign reversal. "+interpretation
        if coverage:
            footer += "\n"+coverage+" Missing outcome values are excluded; phase-specific counts are in the CSV."
        fig.text(.5, .025, footer, ha="center", va="bottom", fontsize=9)
    return fig


def make_figures(args, results, participant_rows, recordings, phases, record_phases):
    dest = args.output/"figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    saved = []
    with PdfPages(args.output/"score_corr_figures.pdf") as book:
        def save(fig, name):
            fig.savefig(dest/f"{name}.png", dpi=180, bbox_inches="tight")
            fig.savefig(dest/f"{name}.pdf", bbox_inches="tight")
            book.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved.append(name)

        primary_groups = sorted([p for p in phases if p["signal"] == "position" and p["mode"] in ("whole", "PSM in stage")], key=group_order)
        lookup = {(r["scenario"], r["method"], key(r), r["predictor"]): r for r in results}
        fig = make_association_overview(results, phases)
        save(fig, "association_overview")

        users = sorted({r["user"] for r in recordings}, key=lambda u: int(u[4:]))
        colors = {u: plt.get_cmap("tab20")(i) for i, u in enumerate(users)}
        # All admissible phases, including null and weak results.
        admitted = [p for p in primary_groups if p["group_usable"] and p["stage"] != "__whole__"]
        fig, axes = plt.subplots(4, 4, figsize=(17, 15), squeeze=False)
        for ax, g in zip(axes.flat, admitted):
            pp = [p for p in participant_rows if p["scenario"] == "primary" and key(p) == key(g)]
            for p in pp:
                ax.scatter(1000*p["signed_lag_s"], p["Score"], color=colors[p["user"]], s=38)
                ax.annotate(p["user"][4:], (1000*p["signed_lag_s"], p["Score"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
            r = lookup[("primary", "between_participants", key(g), "signed_lag_s")]
            ax.axvline(0, color="gray", lw=.7)
            ax.set_ylim(-4, 104)
            ax.set_title(phase_label(g)+f"\nrho={r.get('association', np.nan):+.2f}; q={r.get('q_bh', np.nan):.3f}; n={len(pp)}", fontsize=10)
            ax.set_xlabel("Participant median signed lag (ms)")
            ax.set_ylabel("Mean matched performance score")
        for ax in list(axes.flat)[len(admitted):]:
            ax.set_visible(False)
        fig.suptitle("Valid phases: timing vs performance between participants\nOne point per participant; labels are participant IDs. Task A and C remain separate.", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .95))
        save(fig, "phase_signed_lag_scatter")

        whole = [p for p in primary_groups if p["stage"] == "__whole__"]
        fig, axes = plt.subplots(3, 4, figsize=(17, 11))
        for j, g in enumerate(whole):
            pp = [p for p in participant_rows if p["scenario"] == "primary" and key(p) == key(g)]
            for i, predictor in enumerate(PREDICTORS):
                ax = axes[i, j]
                scale = 1000 if "lag" in predictor else 1
                for p in pp:
                    ax.scatter(scale*p[predictor], p["Score"], color=colors[p["user"]], s=38)
                    ax.annotate(p["user"][4:], (scale*p[predictor], p["Score"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
                r = lookup[("primary", "between_participants", key(g), predictor)]
                ax.set_title(phase_label(g)+f"\nrho={r.get('association', np.nan):+.2f}; q={r.get('q_bh', np.nan):.3f}")
                ax.set_xlabel({"signed_lag_s": "Median signed lag (ms)", "absolute_lag_s": "Median absolute lag (ms)", "peak_r": "Fisher-z mean recording peak r"}[predictor])
                ax.set_ylabel("Mean matched score")
                ax.set_ylim(-4, 104)
        fig.suptitle("Whole-task peak features vs performance: participant summaries", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .96))
        save(fig, "whole_task_scatter")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        usable = [r for r in recordings if r["score_usable"]]
        for ax, task in zip(axes, ("A", "C")):
            rr = [r for r in usable if r["task"] == task]
            for j, u in enumerate(users):
                rows = sorted([r for r in rr if r["user"] == u], key=lambda r: (r["TaskNum"], r["Attempt"]))
                jitter = np.linspace(-.16, .16, len(rows))
                for delta, r in zip(jitter, rows):
                    ax.scatter(j+delta, r["Score"], c=[colors[u]], marker="x" if r["AttemptStatus"] == "Failed" else "o", s=40)
            ax.set_xticks(range(len(users)), [u[4:] for u in users])
            ax.set_xlabel("Participant ID; each marker is a recording")
            ax.set_ylabel("Performance score")
            ax.set_ylim(-4, 104)
            zeros = sum(r["Score"] == 0 for r in rr)
            ax.set_title(f"Task {task}: {len(rr)} scores; {zeros} zeros\nx = failed attempt, circle = completed")
        fig.suptitle("Score coverage and floor effects")
        fig.tight_layout()
        save(fig, "score_distribution")

        # Forest plot of timing results with every sensitivity on the same row.
        scenarios = ("primary", "completed_only", "stronger_peaks", "Both in stage")
        fig, axes = plt.subplots(1, 2, figsize=(15, 13), sharey=True)
        palette = ("#2463a0", "#b56020", "#267443", "#8b4f9c")
        for ax, method, title in zip(axes, ("between_participants", "within_adjusted_order"), ("Between participants", "Within participant, order adjusted")):
            for s, (scenario, color) in enumerate(zip(scenarios, palette)):
                label_added = False
                for i, g in enumerate(primary_groups):
                    gg = dict(g)
                    if scenario == "Both in stage":
                        if g["mode"] == "whole":
                            continue
                        gg["mode"] = "Both in stage"
                    r = lookup.get((scenario, method, key(gg), "signed_lag_s"), {})
                    if r.get("status") != "ok":
                        continue
                    y = i+(s-1.5)*.15
                    ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=color, alpha=.55, lw=1)
                    ax.plot(r["association"], y, "o", color=color, ms=4, label=scenario if not label_added else None)
                    label_added = True
            ax.axvline(0, color="gray", lw=1)
            ax.set_xlim(-1.05, 1.05)
            ax.set_yticks(range(len(primary_groups)), [phase_label(p) for p in primary_groups])
            ax.invert_yaxis() if not ax.yaxis_inverted() else None
            ax.set_title(title)
            ax.set_xlabel("Signed-lag rank association with score [participant bootstrap 95% CI]")
        axes[1].legend(loc="lower left", fontsize=9)
        fig.suptitle("Timing association sensitivity\nMissing estimates indicate an excluded group or fewer than six informative participants.")
        fig.tight_layout(rect=(0, 0, 1, .95))
        save(fig, "signed_lag_sensitivity")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for j, side in enumerate(("l", "r")):
            group_key = ("A", side, "velocity", "grab ring", "PSM in stage")
            pp = [p for p in participant_rows if p["scenario"] == "velocity_secondary" and key(p) == group_key]
            between = lookup[("velocity_secondary", "between_participants", group_key, "peak_r")]
            unadjusted = lookup[("velocity_secondary", "within_participant", group_key, "peak_r")]
            adjusted = lookup[("velocity_secondary", "within_adjusted_order", group_key, "peak_r")]
            ax = axes[0, j]
            for p in pp:
                ax.scatter(p["peak_r"], p["Score"], color=colors[p["user"]], s=48)
                ax.annotate(p["user"][4:], (p["peak_r"], p["Score"]), xytext=(4, 3), textcoords="offset points", fontsize=9)
            ax.set_title(f"Task A grab ring ({side.upper()}), between participants\nSpearman rho={between['association']:+.3f}; q={between['q_bh']:.3f}")
            ax.set_xlabel("Fisher-z mean recording velocity peak r")
            ax.set_ylabel("Mean matched score")
            ax.set_ylim(-4, 104)
            ax = axes[1, j]
            rr = [r for r in scenario_rows(record_phases, "velocity_secondary") if key(r) == group_key]
            xr = stats.rankdata([number(r["peak_r"]) for r in rr])
            yr = stats.rankdata([number(r["Score"]) for r in rr])
            user_array = np.array([r["user"] for r in rr])
            for user in sorted(set(user_array)):
                keep = user_array == user
                ax.scatter(xr[keep]-xr[keep].mean(), yr[keep]-yr[keep].mean(), color=colors[user], s=40, alpha=.8)
            ax.axvline(0, color="gray", lw=.7)
            ax.axhline(0, color="gray", lw=.7)
            ax.set_xlabel("Velocity peak-r rank minus participant mean")
            ax.set_ylabel("Score rank minus participant mean")
            ax.set_title(f"Within participant: rank r={unadjusted['association']:+.3f}; q={unadjusted['q_bh']:.3f}\nOrder adjusted: r={adjusted['association']:+.3f}; q={adjusted['q_bh']:.3f}")
        fig.suptitle("Secondary velocity result: correlation strength during Task A grab ring\nThese associations describe strength, not peak lead/lag; order adjustment weakens the evidence.", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94))
        save(fig, "velocity_grab_ring_strength")
    return saved


def format_num(x, digits=3):
    return f"{x:.{digits}f}" if np.isfinite(number(x)) else "NA"


def format_p(x):
    return f"{x:.2g}" if np.isfinite(number(x)) and 0 < x < .001 else format_num(x)


def result_table(rows):
    lines = ["| Condition | Predictor | N recordings / users | Association [95% CI] | p | BH q |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {phase_label(r)} | {r['predictor']} | {r['n_recordings']} / {r['n_users']} | "
                     f"{format_num(r.get('association'))} [{format_num(r.get('ci_low'))}, {format_num(r.get('ci_high'))}] | "
                     f"{format_p(r.get('p'))} | {format_p(r.get('q_bh'))} |")
    return "\n".join(lines)


def make_report(args, recordings, records, phases, results, provenance):
    main = [r for r in results if r["scenario"] == "primary" and r["status"] == "ok"]
    between = [r for r in main if r["method"] == "between_participants"]
    within = [r for r in main if r["method"] == "within_adjusted_order"]
    signed = sorted([r for r in between if r["predictor"] == "signed_lag_s"], key=lambda r: r["p"])
    significant = [r for r in main if r.get("q_bh", 1) < .05]
    usable = [r for r in recordings if r["score_usable"]]
    invalid = [r for r in recordings if not r["score_usable"]]
    primary_phases = sorted([p for p in phases if p["signal"] == "position" and p["mode"] == "PSM in stage"], key=group_order)
    timing_hits = [r for r in significant if r["predictor"] == "signed_lag_s"]
    velocity = [r for r in results if r["scenario"] == "velocity_secondary" and r["status"] == "ok"]
    velocity_grab = sorted([r for r in velocity if r["task"] == "A" and r["stage"] == "grab ring" and r["predictor"] == "peak_r"],
                           key=lambda r: (r["side"], METHODS.index(r["method"])))
    def reference(scenario, method, task, side, stage, predictor="signed_lag_s"):
        return next((r for r in results if r["scenario"] == scenario and r["method"] == method and
                     r["task"] == task and r["side"] == side and r["stage"] == stage and r["predictor"] == predictor), {})

    def describe(row):
        return (f"r = {format_num(row.get('association'))}, 95% CI [{format_num(row.get('ci_low'))}, "
                f"{format_num(row.get('ci_high'))}], p = {format_p(row.get('p'))}, q = {format_num(row.get('q_bh'), 4)}")

    right_transfer = reference("primary", "between_participants", "A", "r", "transfer ring")
    left_transfer = reference("primary", "between_participants", "A", "l", "transfer ring")
    wrap_strength = reference("primary", "between_participants", "C", "r", "wrap around psm", "peak_r")
    whole_strength = reference("primary", "between_participants", "A", "l", "__whole__", "peak_r")
    velocity_right = reference("velocity_secondary", "between_participants", "A", "r", "grab ring", "peak_r")
    velocity_left = reference("velocity_secondary", "within_participant", "A", "l", "grab ring", "peak_r")
    velocity_left_adjusted = reference("velocity_secondary", "within_adjusted_order", "A", "l", "grab ring", "peak_r")
    whole_signed = [r["association"] for r in signed if r["stage"] == "__whole__"]
    conclusion = ("No signed peak lead/lag association with performance survives the specified multiple-comparison correction. "
                  "The data do not establish that a larger gaze lead improves performance.") if not timing_hits else (
                  "Some signed peak lead/lag associations survive the specified correction; inspect their within-participant and sensitivity results before interpreting them as general timing effects.")
    lines = ["# Peak gaze/PSM timing and performance scores", "",
             "Analysis date: "+provenance["created_at"]+". All analyses are exploratory.", "",
             "## Main finding", "", conclusion, "",
             f"Across the three primary analysis methods and all three predictors, {len(significant)} estimates have BH q < 0.05. "
             "Correction is performed separately for each method, pooling its valid whole-task and phase tests and all three predictors; methods and sensitivity families are not jointly corrected.", "",
             f"Of {len(signed)} primary between-participant signed-lag tests, "
             f"{sum(r['p'] < .05 for r in signed)} reach uncorrected p < 0.05 "
             f"(smallest p = {format_num(min((r['p'] for r in signed), default=np.nan))}). "
             f"Task A transfer/right: {describe(right_transfer)}; transfer/left has r = {format_num(left_transfer.get('association'))}. "
             f"Whole-task signed-lag correlations range from {format_num(min(whole_signed, default=np.nan))} to {format_num(max(whole_signed, default=np.nan))}.", "",
             f"Selected correlation-strength estimates: in the position analysis, Task C wrap/right has {describe(wrap_strength)}, "
             f"and Task A whole-task/left has {describe(whole_strength)}. "
             f"In the separate secondary velocity analysis, Task A grab/right strength between participants has {describe(velocity_right)}; "
             f"grab/left strength within participants has {describe(velocity_left)}, and after order adjustment {describe(velocity_left_adjusted)}. "
             "These are strength results and do not establish a beneficial lead/lag.", "",
             "## Coverage and matching", "",
             f"The saved correlation run contains {len(recordings)} recordings from {len({r['user'] for r in recordings})} novices. "
             f"Exact UserID + TaskNum + Attempt matching yields {len(usable)} usable scores. Task letter, participant type and index metadata are also checked. "
             "Expert scores have no corresponding curves in this novice-only run and are not analyzed.", "",
             "| Task | Recordings | Usable scores | Score = 0 | Failed, scored zero | Completed |",
             "|---|---:|---:|---:|---:|---:|"]
    for task in ("A", "C"):
        rr = [r for r in usable if r["task"] == task]
        lines.append(f"| {task} | {sum(r['task']==task for r in recordings)} | {len(rr)} | {sum(r['Score']==0 for r in rr)} | "
                     f"{sum(r['AttemptStatus']=='Failed' for r in rr)} | {sum(r['AttemptStatus']=='Completed' for r in rr)} |")
    lines += ["", *[f"- Excluded `{r['content_id']}`: score status `{r['AttemptStatus']}`; {r['score_exclusion']}. Missing scores are never replaced with zero." for r in invalid], "",
              "The supplied failed-attempt zeros are retained in the primary analysis. Completed-only sensitivity removes these failures. "
              "`User6_1_1` has a supplied Completed score of zero but missing score components; its total score is retained as supplied. "
              "The score is for an entire attempt, not a phase-specific performance rating.", "",
              "## Validity rules", "",
              "1. Use the existing full-record correlation run (41 Hz, ±2 s, 80 ms smoothing). Position is primary; velocity is a separate secondary analysis. "
              "Positive lag means gaze precedes PSM; negative means gaze follows PSM. Lag is a time shift, not a frequency phase angle.",
              "2. Admit a task/side/phase/mode condition only if its saved group status is `ok`, its peak correlation is positive, and at least three stable participants support it. "
              "Drop-ring conditions and group peaks at either search limit are excluded before looking at score associations.",
              "3. A recording must have a usable score, quality status ok, an interior positive correlation peak, and finite correlation across the entire lag grid. "
              "This avoids estimating a peak from a partially supported search window. Selection may itself depend on skill, so exclusion counts are provided.",
              f"4. Require at least {args.min_users} participants for between-participant tests, and that many participants with both predictor and score variation for within-participant tests. "
              "A valid computational peak does not imply a precise individual delay: broad curves, weak correlations and near-boundary bootstrap intervals remain limitations.", "",
              "| Phase / side | Group status | Eligible recordings / users | Median peak gain over zero |",
              "|---|---|---:|---:|"]
    for p in primary_phases:
        lines.append(f"| {phase_label(p)} | {p['group_status']} | {p['n_eligible_recordings']} / {p['n_eligible_users']} | {format_num(p['median_peak_gain_over_zero'], 4)} |")
    lines += ["", "## How the association was estimated", "",
              "- **Between participants:** for each task/side/phase, take each participant's median recording peak lag, median absolute lag, and Fisher-z mean recording peak r. "
              "Relate each feature to the mean score of those same eligible recordings using Spearman correlation. This asks whether higher-scoring participants have different timing. "
              "These medians summarize recording peaks; they are not peaks of participant-averaged correlation curves from `novice_summary.csv`.",
              "- **Within participant:** globally rank recording features and scores within each condition (average ranks for ties), subtract each participant's mean ranks, and compute a weighted correlation. "
              "Each participant has total weight one. This asks whether a participant's better attempts have different timing. "
              "The adjusted version additionally removes linear effects of TaskNum (session order in the index) and Attempt number from both rank variables, using the same weights. "
              "It is a rank-linear partial correlation, not ordinary Spearman correlation; the order adjustment is a sensitivity check, not full control of learning.",
              f"- **Uncertainty:** {args.bootstrap:,} whole-participant bootstrap resamples; percentile 95% intervals. Between-participant p values use {args.permutations:,} random score permutations across participants, two-sided with the +1 correction. "
              "Within-participant p values use a participant-cluster CR1 sandwich and t(G−1), with absorbed participant intercepts counted in the finite-sample correction. "
              "With only 6–11 contributing participants these p values and bootstrap intervals are approximate and can disagree. Whole-participant resampling preserves the repeated-attempt structure; "
              "within-participant bootstrap retains the original rank scores. No individual time samples are treated as independent observations.",
              "- **Multiple comparisons:** BH correction includes signed lag, absolute lag and strength, both sides, both tasks, and all admitted whole-task/phase conditions, "
              "separately for each scenario and method. Tests with insufficient variation are omitted from that family's denominator and explicitly labeled in the CSV. "
              "The primary phase analysis is anchored to PSM time; strict same-occurrence pairing is a sensitivity analysis. "
              "`stronger_peaks` additionally requires recording peak r ≥ 0.2; this exploratory cutoff is not a validated reliability threshold.", "",
              "## Signed timing: participant-level results", "",
              result_table(signed), "",
              "Positive association means higher score accompanies a more positive lag (larger gaze lead or smaller gaze lag); "
              "negative association means higher score accompanies a more negative lag. For absolute lag, a negative association means higher performance nearer synchrony.", "",
              "## Strongest primary associations across features", "",
              "The following are sorted by uncorrected p to make candidate effects inspectable. Conclusions should use q and sensitivity, not the ranking alone.", "",
              "Between participants:", "", result_table(sorted(between, key=lambda r: r["p"])[:8]), "",
              "Within participant, after order adjustment:", "", result_table(sorted(within, key=lambda r: r.get("p", 1))[:8]), "",
              "## Timing sensitivity for the strongest between-participant candidates", "",
              "Each row below is a separate analysis. The strict Both in stage mode can differ in phase admissibility and retained recordings; a missing result is not a zero effect.", ""]
    candidates = list(signed[:3])
    strict_timing = sorted([r for r in results if r["scenario"] == "Both in stage" and r["method"] == "between_participants"
                            and r["predictor"] == "signed_lag_s" and r["status"] == "ok"], key=lambda r: r["p"])
    if strict_timing:
        strict = strict_timing[0]
        counterpart = next((r for r in signed if all(r[k] == strict[k] for k in ("task", "stage", "side"))), None)
        if counterpart is not None and counterpart not in candidates:
            candidates.append(counterpart)
    for candidate in candidates:
        lines += ["### "+phase_label(candidate), "",
                  "| Scenario / method | N recordings / users | Association [95% CI] | p | q |",
                  "|---|---:|---:|---:|---:|"]
        for scenario in ("primary", "completed_only", "stronger_peaks", "Both in stage"):
            for method in ("between_participants", "within_participant", "within_adjusted_order"):
                matches = [r for r in results if r["scenario"] == scenario and r["method"] == method and
                           r["predictor"] == "signed_lag_s" and all(r[k] == candidate[k] for k in ("task", "side", "stage"))]
                if matches:
                    r = matches[0]
                    lines.append(f"| {scenario} / {method} | {r['n_recordings']} / {r['n_users']} | "
                                 f"{format_num(r.get('association'))} [{format_num(r.get('ci_low'))}, {format_num(r.get('ci_high'))}] | "
                                 f"{format_num(r.get('p'))} | {format_num(r.get('q_bh'))} |")
        lines.append("")
    lines += ["## Secondary velocity correlation strength", "",
              "Task A grab-ring velocity estimates are shown below. The complete results for all phases are in the association tables.", "",
              "| Side / method | N recordings / users | Rank association [95% CI] | p | BH q |",
              "|---|---:|---:|---:|---:|"]
    for r in velocity_grab:
        lines.append(f"| {r['side'].upper()} / {r['method']} | {r['n_recordings']} / {r['n_users']} | "
                     f"{r['association']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p']:.5f} | {r['q_bh']:.4f} |")
    velocity_rr = [r["peak_r"] for r in records if r["eligible"] and key(r) == ("A", "r", "velocity", "grab ring", "PSM in stage")]
    velocity_ll = [r["peak_r"] for r in records if r["eligible"] and key(r) == ("A", "l", "velocity", "grab ring", "PSM in stage")]
    lines += ["", f"The right-side between-participant strength correlation has a leave-one-participant-out range of "
              f"{format_num(velocity_right.get('loo_min'))} to {format_num(velocity_right.get('loo_max'))}. "
              "Compare between-participant estimates with within-participant and order-adjusted estimates before interpreting performance differences. "
              f"Recording peak velocity correlations in this phase have median L = {format_num(np.median(velocity_ll) if velocity_ll else np.nan)}, "
              f"R = {format_num(np.median(velocity_rr) if velocity_rr else np.nan)}. "
              "A high correlation between that strength metric and scores does not make gaze–PSM velocity correlation itself large. "
              f"Signed velocity timing estimates with BH q < 0.05: {sum(r['predictor'] == 'signed_lag_s' and r.get('q_bh', 1) < .05 for r in velocity)}.", "",
              f"The strict Both in stage position estimate for A place/right is "
              f"{describe(reference('Both in stage', 'between_participants', 'A', 'r', 'place ring'))}. "
              f"Its main PSM in stage estimate is {describe(reference('primary', 'between_participants', 'A', 'r', 'place ring'))}, "
              f"and its within-participant order-adjusted strict-stage association is {describe(reference('Both in stage', 'within_adjusted_order', 'A', 'r', 'place ring'))}. "
              "These estimates use different stage-support rules and should be compared with their uncertainty.", "",
              "## Interpretation and limitations", "",
              "- A/C are analyzed separately because their score distributions and task behaviors differ. Task C's many zeros create heavy ties and some participants have no score variation. "
              "A null result is not evidence that timing never matters, and these data cannot resolve small effects reliably.",
              "- Individual timing is often weakly identified even when a peak is interior. `phase_validity.csv` records group bootstrap boundary frequencies, "
              "gain relative to zero lag and descriptive widths within 0.01 of the maximum. That width is a shape diagnostic, not a confidence interval. "
              "C threading has fewer supported participants and broad group peak intervals; its surviving computational peaks require particular caution.",
              "- The median over only eligible attempts can use a different subset for different phases or sides. Exclusion may be related to performance. "
              "Participant leave-one-out ranges are supplied to expose single-participant sensitivity; no per-recording peak uncertainty is propagated into the score association.",
              "- Between-participant differences can reflect general skill; within-participant changes can reflect practice, task difficulty, failed attempts or score floor effects. "
              "Even an association surviving correction would not establish that changing gaze timing causes better performance. "
              "The measurements are raw gaze and DLC left/right PSM coordinates, not validated fixation arrivals or tool-tip positions.", "",
              "## Outputs and reproduction", "",
              "- [Figure booklet](score_corr_figures.pdf), [association overview](figures/association_overview.png), "
              "[all valid phase timing scatters](figures/phase_signed_lag_scatter.png), [whole-task feature scatters](figures/whole_task_scatter.png), "
              "[timing sensitivity](figures/signed_lag_sensitivity.png), [score coverage](figures/score_distribution.png), "
              "[secondary velocity strength](figures/velocity_grab_ring_strength.png). PNG and vector PDF versions are provided.",
              "- `associations.csv`: all primary, completed-only, stronger-peak, strict Both in stage and secondary velocity estimates, including excluded conditions and BH family sizes.",
              "- `participant_summaries.csv`: exact participant inputs to between-participant analysis, with retained recording IDs.",
              "- `matched_scores.csv`, `score_inventory.csv`, `record_phase_audit.csv`, `phase_validity.csv`: joins, unused score rows, per-record exclusions and phase diagnostics.",
              "- `provenance.json`, `validation.json`, and `code_snapshot/analyze_score_corr_dlc.py`: settings, input hashes, numerical checks and the analysis source.", "",
              "From the repository root:", "", "```bash",
              f"python tools/analyze_score_corr_dlc.py --input {args.input} --index {args.index} --output {args.output} "
              f"--bootstrap {args.bootstrap} --permutations {args.permutations} --seed {args.seed} --min-users {args.min_users}",
              "```", "", "The original correlation outputs and input scores are read only. Statistical associations were computed from saved numerical curves, without access to the NAS source videos.", ""]
    (args.output/"report.md").write_text("\n".join(lines))


def numerical_checks():
    # Mathematical checks of inference, beyond a successful script execution.
    x = np.array([2., 2., 4., 9., 8., 1.])
    y = np.array([1., 3., 3., 7., 9., 0.])
    assert np.isclose(spearman(x, y), stats.spearmanr(x, y).statistic)
    assert np.allclose(bh_adjust([.01, .04, .03, np.nan]), [.03, .04, .04, np.nan], equal_nan=True)
    rng = np.random.default_rng(321)
    values = rng.normal(size=(70, 4))
    gram = values.T @ values
    residuals = values[:, :2] - values[:, 2:] @ np.linalg.lstsq(values[:, 2:], values[:, :2], rcond=None)[0]
    expected = residuals[:, 0] @ residuals[:, 1]/np.sqrt(np.sum(residuals[:, 0]**2)*np.sum(residuals[:, 1]**2))
    assert np.isclose(partial_from_gram(gram, True), expected)
    assert np.allclose(partial_from_gram(np.stack([gram, gram]), True), expected)
    return {"spearman_tie_handling": True, "BH_known_values": True,
            "partial_correlation_direct_residual_check": True}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/gaze_psm_xcorr_dlc"))
    parser.add_argument("--index", type=Path, default=Path("Index"))
    parser.add_argument("--output", type=Path, default=Path("outputs/score_corr_dlc"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=19999)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--min-users", type=int, default=6)
    args = parser.parse_args()
    if args.output.resolve() in (args.input.resolve(), args.index.resolve()):
        parser.error("Use a separate output directory")
    if args.bootstrap < 100 or args.permutations < 100 or args.min_users < 3:
        parser.error("Require bootstrap/permutations >=100 and min-users >=3")
    args.output.mkdir(parents=True, exist_ok=True)
    checks = numerical_checks()
    input_paths = [args.input/name for name in ("config.json", "record_summary.csv", "record_curves.csv", "group_summary.csv", "quality.csv")]
    input_paths += [args.index/name for name in ("content_index.csv", "Task_A_Score.csv", "Task_C_Score.csv")]
    hashes_before = {str(p): sha256(p) for p in input_paths}
    config, inventory, recordings, records, phases = prepare(args)
    print(f"Matched {sum(r['score_usable'] for r in recordings)}/{len(recordings)} recordings", flush=True)
    results, people = analyze(args, records, phases)
    for name, rows in (("matched_scores", recordings), ("score_inventory", inventory),
                       ("record_phase_audit", records), ("phase_validity", phases),
                       ("associations", results), ("participant_summaries", people)):
        write_csv(args.output/f"{name}.csv", rows)
    figures = make_figures(args, results, people, recordings, phases, records)
    import scipy
    provenance = {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "correlation_config": config, "input_sha256": hashes_before,
        "source_sha256": sha256(Path(__file__)),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "figures": figures,
    }
    make_report(args, recordings, records, phases, results, provenance)
    snapshot = args.output/"code_snapshot"
    snapshot.mkdir(exist_ok=True)
    (snapshot/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    unchanged = all(sha256(Path(p)) == h for p, h in hashes_before.items())
    assert unchanged
    successful = [r for r in results if r["status"] == "ok"]
    assert all(-1-1e-8 <= r["association"] <= 1+1e-8 for r in successful)
    assert all(0 <= r["p"] <= r["q_bh"]+1e-12 <= 1+1e-12 for r in successful if np.isfinite(r["p"]))
    checks.update(input_hashes_unchanged=unchanged, all_record_peaks_recomputed_from_curves=True,
                  score_keys_unique_and_index_verified=True,
                  n_recordings=len(recordings), n_scored=sum(r["score_usable"] for r in recordings),
                  n_record_phase_rows=len(records), n_group_conditions=len(phases),
                  n_analysis_rows=len(results), n_successful_tests=len(successful),
                  n_figures=len(figures), all_checks_passed=True)
    (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (args.output/"validation.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
