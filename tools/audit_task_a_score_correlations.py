#!/usr/bin/env python3
"""Independently audit Task A whole-task strength/timing versus performance.

Recalculates every position-correlation lag directly from cached trajectories,
joins the original score CSV by exact recording ID, and uses scalar Pearson
and independently ranked Spearman calculations. Existing results are read only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/task_a_score_audit_matplotlib")
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats

from tools.gaze_psm_core import analyze_curve
from tools.plot_peak_correlation_vs_score import separate_labels


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pearson(x, y):
    mx, my = math.fsum(x)/len(x), math.fsum(y)/len(y)
    xx, yy = [v-mx for v in x], [v-my for v in y]
    den = math.sqrt(math.fsum(v*v for v in xx)*math.fsum(v*v for v in yy))
    return math.fsum(a*b for a, b in zip(xx, yy))/den if den else math.nan


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, start = [0.]*len(values), 0
    while start < len(order):
        end = start+1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start+1+end)/2
        start = end
    return result


def rho(x, y):
    return pearson(ranks(x), ranks(y))


def direct_curve(gaze, psm, steps, config):
    """Use explicit time slices and overlap-centered sums; no FFT or correlate."""
    values, pairs = [], []
    n = len(gaze)
    for step in steps:
        if step >= 0:
            g, p = gaze[:n-step], psm[step:]
        else:
            g, p = gaze[-step:], psm[:n+step]
        valid = np.isfinite(g).all(axis=1) & np.isfinite(p).all(axis=1)
        g, p = g[valid], p[valid]
        count = len(g)
        pairs.append(count)
        value = np.nan
        if count >= max(2, math.ceil(config["min_overlap"]*config["hz"])):
            g, p = g-g.mean(axis=0), p-p.mean(axis=0)
            vg, vp = float(np.sum(g*g)), float(np.sum(p*p))
            denominator = math.sqrt(vg*vp)
            if (math.sqrt(vg/count) >= config["min_gaze_rms"]
                    and math.sqrt(vp/count) >= config["min_psm_rms"]
                    and denominator > np.finfo(float).eps):
                value = np.clip(np.sum(g*p)/denominator, -1., 1.)
        values.append(value)
    return np.asarray(values), np.asarray(pairs)


def aggregate(rows, outcome):
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user"]].append(row)
    return [{"user": user, "outcome": outcome, "score": float(np.mean([r[outcome] for r in rr])),
             "peak_r": float(np.tanh(np.mean(np.arctanh(np.clip([r["peak_r"] for r in rr], -.9999999, .9999999))))),
             "signed_lag_s": float(np.median([r["peak_lag_s"] for r in rr])),
             "absolute_lag_s": float(np.median([abs(r["peak_lag_s"]) for r in rr])),
             "n_recordings": len(rr), "n_positive_peak_lags": sum(r["peak_lag_s"] > 0 for r in rr),
             "content_ids": ";".join(r["content_id"] for r in rr)}
            for user, rr in sorted(by_user.items(), key=lambda item: int(item[0][4:]))]


def figures(output, people, associations):
    users = sorted({p["user"] for p in people}, key=lambda u: int(u[4:]))
    colors = {user: plt.get_cmap("tab20")(i) for i, user in enumerate(users)}
    lookup = {(r["outcome"], r["side"], r["predictor"]): r for r in associations}
    with PdfPages(output/"strength_and_timing.pdf") as book, plt.rc_context({"pdf.fonttype": 42}):
        for outcome in ("Time", "Eco", "Score"):
            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            for row_index, predictor in enumerate(("peak_r", "signed_lag_s")):
                for col, side in enumerate(("l", "r")):
                    ax = axes[row_index, col]
                    pp = [p for p in people if p["outcome"] == outcome and p["side"] == side]
                    scale = 1 if predictor == "peak_r" else 1000
                    x = np.array([p["score"] for p in pp])
                    y = np.array([p[predictor]*scale for p in pp])
                    fit = np.polyfit(x, y, 1)
                    line_x = np.array([min(x), max(x)])
                    ax.plot(line_x, np.polyval(fit, line_x), "--", color=".45", lw=1.2)
                    for p in pp:
                        ax.scatter(p["score"], p[predictor]*scale, color=colors[p["user"]], s=55, zorder=3)
                        ax.annotate(p["user"][4:], (p["score"], p[predictor]*scale), xytext=(6, 5),
                                    textcoords="offset points", fontsize=9)
                    result = lookup[(outcome, side, predictor)]
                    ax.set_title(f"{'Left' if side == 'l' else 'Right'} PSM / "
                                 f"{'strength' if predictor == 'peak_r' else 'timing'}\n"
                                 f"Spearman \u03c1={result['spearman_rho']:+.3f}; "
                                 f"p={result['saved_permutation_p']:.3f}; q={result['saved_q_bh']:.3f}", fontsize=12)
                    ax.set_xlabel(f"{outcome} (mean of matched eligible attempts)")
                    ax.grid(alpha=.18)
                    if predictor == "peak_r":
                        ax.set_ylim(0, 1)
                        ax.set_ylabel("Peak gaze-PSM correlation (Fisher-z mean)")
                    else:
                        ax.set_ylim(-800, 800)
                        ax.axhline(0, color="black", lw=.8)
                        ax.axhspan(0, 800, color="#def1e4", alpha=.4)
                        ax.axhspan(-800, 0, color="#fae7dc", alpha=.4)
                        ax.set_ylabel("Median peak lag (ms); positive = gaze leads")
                    ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"Task A whole task: {outcome} versus correlation strength and gaze lead/lag\n"
                         "One point per participant; labels are participant IDs", fontsize=15, y=.985)
            fig.text(.5, .018, "Top: how strongly trajectories correspond. Bottom: their typical time shift; green = gaze leads, orange = PSM leads.\n"
                     "p/q are the existing Spearman participant-permutation results; q retains the original all-feature family.\n"
                     "Side-specific eligibility is retained (L: 28 attempts, R: 32); the common-recording sensitivity uses 27 attempts in sensitivity.csv.\n"
                     "A whole-task peak lag does not measure the fraction of time gaze leads.", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .11, 1, .93), h_pad=2.5, w_pad=2.5)
            separate_labels(fig, axes)
            fig.savefig(output/f"strength_and_timing_{outcome}.png", dpi=180, bbox_inches="tight")
            fig.savefig(output/f"strength_and_timing_{outcome}.pdf", bbox_inches="tight")
            book.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/gaze_psm_xcorr"))
    parser.add_argument("--scores", type=Path, default=Path("outputs/score_corr"))
    parser.add_argument("--index", type=Path, default=Path("Index"))
    parser.add_argument("--output", type=Path, default=Path("outputs/score_corr/task_A_audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def track(path):
        hashes[str(path)] = sha(path)
        return path

    config = json.loads(track(args.input/"config.json").read_text())
    summary = [r for r in read(track(args.input/"record_summary.csv"))
               if r["task"] == "A" and r["signal"] == "position" and r["mode"] == "whole"]
    original_scores = read(track(args.index/"Task_A_Score.csv"))
    score_lookup = {f"User{int(r['UserID'])}_{int(r['TaskNum'])}_{int(r['Attempt'])}": r for r in original_scores}
    assert len(score_lookup) == len(original_scores)
    indexed = {r["content_id"]: r for r in read(track(args.index/"content_index.csv"))}
    quality = {r["content_id"]: r for r in read(track(args.input/"quality.csv"))}
    curves = defaultdict(list)
    with track(args.input/"record_curves.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            if r["task"] == "A" and r["signal"] == "position" and r["mode"] == "whole":
                curves[(r["content_id"], r["side"])].append((float(r["lag_s"]), float(r["r"]), int(r["pairs"])))
    limit = round(config["max_lag"]*config["hz"])
    steps = np.arange(-limit, limit+1)
    lags = steps/config["hz"]
    records, curve_audit = [], []
    cached = {}
    for saved in summary:
        cid, side = saved["content_id"], saved["side"]
        if cid not in cached:
            path = track(args.input/"cache"/f"{cid}.npz")
            with np.load(path, allow_pickle=False) as cache:
                cached[cid] = {name: cache[name] for name in ("gaze_l", "psm_l", "gaze_r", "psm_r", "qc_json")}
        cache = cached[cid]
        qc = json.loads(str(cache["qc_json"]))
        assert qc["content_id"] == cid and qc["task"] == "A"
        original = score_lookup[cid]
        ix = indexed[cid]
        assert str(int(original["UserID"])) == saved["user"][4:] == str(int(ix["User Num"]))
        assert int(original["TaskNum"]) == int(ix["Task Num"])
        assert int(original["Attempt"]) == int(ix["Attempt Num"])
        assert original["UserType"].lower() == ix["src_skill"].lower()
        r, pairs = direct_curve(cache[f"gaze_{side}"], cache[f"psm_{side}"], steps, config)
        expected = np.array(curves[(cid, side)]).T
        np.testing.assert_allclose(lags, expected[0], atol=1e-12)
        np.testing.assert_allclose(r, expected[1], atol=1e-9, rtol=1e-9, equal_nan=True)
        np.testing.assert_array_equal(pairs, expected[2])
        peak = int(np.nanargmax(r))
        np.testing.assert_allclose([lags[peak], r[peak]], [float(saved["peak_lag_s"]), float(saved["peak_r"])], atol=1e-9)
        peak_ok = peak not in (0, len(r)-1) and r[peak] > 0 and np.isfinite(r).all()
        row = {"content_id": cid, "user": saved["user"], "side": side, "peak_r": float(r[peak]),
               "peak_lag_s": float(lags[peak]), "r_zero": float(r[limit]),
               "eligible_peak": bool(peak_ok and quality[cid]["status"] == "ok"),
               **{name: float(original[name]) if original[name] else np.nan for name in ("Score", "Time", "Eco")}}
        records.append(row)
        curve_audit.append({"content_id": cid, "side": side, "max_abs_curve_error": float(np.nanmax(abs(r-expected[1]))),
                            "peak_r": row["peak_r"], "peak_lag_s": row["peak_lag_s"], "eligible_peak": row["eligible_peak"]})
    print(f"Direct trajectory check passed: {len(summary)} curves / {len(cached)} recordings", flush=True)
    sign_checks = {}
    rng = np.random.default_rng(123)
    gaze = rng.normal(size=(1000, 2))*10
    for name, shift in (("gaze_leads", 12), ("psm_leads", -12)):
        psm = np.full_like(gaze, np.nan)
        if shift > 0:
            psm[shift:] = gaze[:-shift]
        else:
            psm[:shift] = gaze[-shift:]
        result = analyze_curve(gaze, psm, config["hz"], compute_spatial=False)
        lag = float(result["lag_s"][np.nanargmax(result["r"])])
        assert np.isclose(lag, shift/config["hz"])
        sign_checks[name] = lag
    people, associations, sensitivities = [], [], []
    for outcome in ("Score", "Time", "Eco"):
        parent = args.scores if outcome == "Score" else args.scores/"component_metrics"
        suffix = "" if outcome == "Score" else f"_{outcome}"
        saved_people = {(r["user"], r["side"]): r for r in read(track(parent/f"participant_summaries{suffix}.csv"))
                        if r["scenario"] == "primary" and r["task"] == "A" and r["mode"] == "whole"}
        all_results = read(track(parent/f"associations{suffix}.csv"))
        saved_results = {(r["side"], r["predictor"]): r for r in all_results if r["scenario"] == "primary"
                         and r["task"] == "A" and r["mode"] == "whole" and r["method"] == "between_participants"}
        family = [r for r in all_results if r["scenario"] == "primary" and r["method"] == "between_participants" and r["status"] == "ok"]
        np.testing.assert_allclose(stats.false_discovery_control([float(r["p"]) for r in family]),
                                   [float(r["q_bh"]) for r in family], atol=1e-12)
        eligible = [r for r in records if r["eligible_peak"] and np.isfinite(r[outcome])]
        common = {r["content_id"] for r in eligible if r["side"] == "l"} & {r["content_id"] for r in eligible if r["side"] == "r"}
        for side in ("l", "r"):
            pp = aggregate([r for r in eligible if r["side"] == side], outcome)
            for person in pp:
                saved = saved_people[(person["user"], side)]
                assert set(person["content_ids"].split(";")) == set(saved["content_ids"].split(";"))
                np.testing.assert_allclose([person["score"], person["peak_r"], person["signed_lag_s"]],
                                           [float(saved[outcome]), float(saved["peak_r"]), float(saved["signed_lag_s"])], atol=1e-9)
                people.append({"side": side, **person})
            for predictor in ("peak_r", "signed_lag_s", "absolute_lag_s"):
                x, y = [p["score"] for p in pp], [p[predictor] for p in pp]
                saved = saved_results[(side, predictor)]
                direct_r, direct_rho = pearson(x, y), rho(x, y)
                np.testing.assert_allclose([direct_r, direct_rho],
                                           [float(saved["pearson_descriptive"]), float(saved["association"])], atol=1e-10)
                associations.append({"outcome": outcome, "side": side, "predictor": predictor,
                                     "n_users": len(pp), "pearson_r": direct_r, "spearman_rho": direct_rho,
                                     "saved_permutation_p": float(saved["p"]), "saved_q_bh": float(saved["q_bh"])})
            common_people = aggregate([r for r in eligible if r["side"] == side and r["content_id"] in common], outcome)
            for predictor in ("peak_r", "signed_lag_s"):
                sensitivities.append({"outcome": outcome, "side": side, "predictor": predictor,
                                      "cohort": "same_27_recordings_for_both_sides", "n_recordings": len(common),
                                      "n_users": len(common_people), "spearman_rho": rho([p["score"] for p in common_people],
                                                                                          [p[predictor] for p in common_people])})
    write(args.output/"curve_checks.csv", curve_audit)
    write(args.output/"participant_values.csv", people)
    write(args.output/"associations.csv", associations)
    write(args.output/"sensitivity.csv", sensitivities)
    figures(args.output, people, associations)
    assert all(sha(Path(path)) == expected for path, expected in hashes.items())
    validation = {"all_checks_passed": True, "recordings": len(cached), "direct_curves": len(summary),
                  "lag_values_per_curve": len(lags), "max_abs_curve_error": max(r["max_abs_curve_error"] for r in curve_audit),
                  "original_score_and_index_joins_verified": True, "manual_rank_and_pearson_checks": len(associations),
                  "BH_families_checked_against_scipy": 3, "synthetic_sign_checks": sign_checks,
                  "inputs_unchanged": True, "input_sha256": hashes, "script_sha256": sha(Path(__file__)),
                  "scope": "Saved trajectory caches through score association; source video extraction not independently revalidated"}
    (args.output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({k: v for k, v in validation.items() if k != "input_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
