#!/usr/bin/env python3
"""Plot saved peak correlations against Score, Time/Eco or TimeRaw/EcoRaw.

Reads outputs/score_corr without rerunning or changing the original analysis.
Spearman inference is copied from the existing participant permutation tests;
Pearson coefficients and plotted points are independently checked with SciPy.
"""
from __future__ import annotations
from gaze_psm_modes import mode_rows, normalize_modes

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/peak_correlation_vs_score_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Bbox
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats

GROUP = ("task", "side", "signal", "stage", "mode")
PHASES = {
    "A": ["reach ring", "grab ring", "transfer ring", "place ring", "drop ring"],
    "C": ["reach string", "threading string through tower", "wrap around psm", "pull through and cinch"],
}
LABELS = {"__whole__": "Whole task", "threading string through tower": "Threading",
          "wrap around psm": "Wrap around PSM", "pull through and cinch": "Pull/cinch"}
CI_MIN_VALID_FRACTION = .975


def spearman_ci_metadata(result, bootstrap_requested):
    """Screen saved percentile intervals before displaying them as 95% CIs.

    The defined-draw cutoff is a conservative display rule, not a guarantee of
    confidence-interval coverage. Original bounds are retained in the tables.
    """
    valid = int(result["bootstrap_valid"])
    total = int(bootstrap_requested)
    if total <= 0 or not 0 <= valid <= total:
        raise ValueError("Invalid bootstrap draw counts")
    low, high = float(result["ci_low"]), float(result["ci_high"])
    fraction = valid/total
    reason = ""
    if not (np.isfinite(low) and np.isfinite(high) and -1-1e-12 <= low <= high <= 1+1e-12):
        reason = "missing_or_invalid_interval_bounds"
    elif fraction < CI_MIN_VALID_FRACTION:
        reason = "too_many_undefined_bootstrap_draws"
    return {"spearman_bootstrap_valid": valid, "spearman_bootstrap_requested": total,
            "spearman_bootstrap_valid_fraction": fraction,
            "spearman_ci_status": "unreliable" if reason else "supported",
            "spearman_ci_reason": reason}


def spearman_ci_label(result):
    if result["spearman_ci_status"] != "supported":
        fraction = result["spearman_bootstrap_valid_fraction"]
        return f"Spearman CI unreliable ({fraction:.1%} valid draws)"
    return (f"Spearman 95% CI: [{result['spearman_ci_low']:+.2f}, "
            f"{result['spearman_ci_high']:+.2f}]")


def ci_display_summary(summary):
    return {"confidence_level": .95, "method": "paired_participant_percentile_bootstrap",
            "exploratory": True, "multiple_comparison_adjusted": False,
            "minimum_defined_bootstrap_fraction": CI_MIN_VALID_FRACTION,
            "rule_scope": "conservative display screen; does not guarantee coverage",
            "supported_conditions": sum(r["spearman_ci_status"] == "supported" for r in summary),
            "unreliable_conditions": sum(r["spearman_ci_status"] != "supported" for r in summary)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(mode_rows(csv.DictReader(stream)))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group(row):
    return tuple(row[name] for name in GROUP)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def separate_labels(fig, axes):
    """Keep participant IDs readable where several scores are tied at zero."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        annotations = list(ax.texts)
        point_boxes = []
        for annotation in annotations:
            if hasattr(annotation, "xy"):
                x, y = ax.transData.transform(annotation.xy)
                point_boxes.append(Bbox.from_bounds(x-5, y-5, 10, 10))
        occupied = []
        for annotation in annotations:
            if not hasattr(annotation, "xy"):
                continue
            best, best_collisions = None, float("inf")
            for dx in (7, -7, 18, -18):
                for dy in (6, -8, 17, -19, 28, -30, 39, -41):
                    annotation.set_position((dx, dy))
                    annotation.set_ha("left" if dx > 0 else "right")
                    annotation.set_va("center")
                    box = annotation.get_window_extent(renderer).expanded(1.1, 1.2)
                    collisions = sum(box.overlaps(other) for other in occupied+point_boxes)
                    collisions += 100*int(not (ax.bbox.contains(box.x0, box.y0)
                                              and ax.bbox.contains(box.x1, box.y1)))
                    if collisions < best_collisions:
                        best, best_collisions = (dx, dy), collisions
                    if collisions == 0:
                        break
                if best_collisions == 0:
                    break
            annotation.set_position(best)
            annotation.set_ha("left" if best[0] > 0 else "right")
            occupied.append(annotation.get_window_extent(renderer).expanded(1.1, 1.2))


def analysis_table_paths(source, signal, outcome):
    """Route velocity to its own statistics, preserving legacy overall Score."""
    legacy_score = outcome == "Score"
    if outcome in ('PenaltySum', 'PenaltyRawSum'):
        preferred = ('velocity_metrics' if signal == 'velocity' else
                     'component_metrics' if outcome == 'PenaltySum' else 'raw_metrics')
        if (source/preferred/f'associations_{outcome}.csv').exists():
            filenames = [f'{preferred}/{name}_{outcome}.csv' for name in
                         ('participant_summaries', 'associations', 'record_phase_audit')]
            return filenames, f'{preferred}/provenance_{outcome}.json'
        metric_directory = f'penalty_metrics/{signal}'
    elif signal == "velocity":
        metric_directory = "velocity_metrics"
        legacy_score = outcome == "Score" and not (source/metric_directory/"associations_Score.csv").exists()
    else:
        metric_directory = "component_metrics" if outcome in ("Time", "Eco") else "raw_metrics"
    filenames = [(f"{name}.csv" if legacy_score else f"{metric_directory}/{name}_{outcome}.csv")
                 for name in ("participant_summaries", "associations", "record_phase_audit")]
    provenance_name = "provenance.json" if legacy_score else f"{metric_directory}/provenance.json"
    return filenames, provenance_name


def load_and_check(source, signal, outcome="Score", predictor="peak_r"):
    if predictor not in ("peak_r", "signed_lag_s", "absolute_lag_s"):
        raise ValueError(f"Unsupported predictor: {predictor}")
    scenario = "primary" if signal == "position" else "velocity_secondary"
    filenames, provenance_name = analysis_table_paths(source, signal, outcome)
    x_column = "score_x" if outcome == "Score" else f"{outcome}_x"
    hashes = {name: digest(source/name) for name in [*filenames, provenance_name]}
    bootstrap_requested = normalize_modes(json.loads((source/provenance_name).read_text()))["settings"]["bootstrap"]
    people = [row for row in read_csv(source/filenames[0]) if row["scenario"] == scenario]
    results = [row for row in read_csv(source/filenames[1]) if row["scenario"] == scenario
               and row["method"] == "between_participants" and row["predictor"] == predictor]
    recordings = defaultdict(list)
    for row in read_csv(source/filenames[2]):
        if row["eligible"] == "True" and row["signal"] == signal and row["mode"] in ("whole", "PSM in stage"):
            recordings[(group(row), row["user"])].append(row)
    if len({(group(row), row["user"]) for row in people}) != len(people):
        raise ValueError("Duplicate participant/condition summaries")
    if len({group(row) for row in results}) != len(results):
        raise ValueError("Duplicate between-participant results")

    by_group = defaultdict(list)
    for row in people:
        by_group[group(row)].append(row)
    summary, points = [], []
    for result in results:
        if result["status"] != "ok":
            continue
        k = group(result)
        pp = sorted(by_group[k], key=lambda row: int(row["user"][4:]))
        if len(pp) != int(result["n_users"]):
            raise ValueError(f"Participant count mismatch: {k}")
        n_recordings = 0
        for person in pp:
            rr = recordings[(k, person["user"])]
            ids = {r["content_id"] for r in rr}
            if ids != set(person["content_ids"].split(";")) or len(ids) != len(rr):
                raise ValueError(f"Recording membership mismatch: {k}, {person['user']}")
            expected_score = np.mean([float(r[outcome]) for r in rr])
            peaks = np.array([float(r["peak_r"]) for r in rr])
            expected_peak = np.tanh(np.mean(np.arctanh(np.clip(peaks, -1+1e-7, 1-1e-7))))
            expected_lag = np.median([float(r["signed_lag_s"]) for r in rr])
            np.testing.assert_allclose(float(person['absolute_lag_s']),
                                       np.median([float(r['absolute_lag_s']) for r in rr]))
            np.testing.assert_allclose([float(person[outcome]), float(person["peak_r"]), float(person["signed_lag_s"])],
                                       [expected_score, expected_peak, expected_lag], rtol=1e-10, atol=1e-12)
            if len(rr) != int(person["n_recordings"]):
                raise ValueError(f"Recording count mismatch: {k}, {person['user']}")
            n_recordings += len(rr)
            points.append({**{name: person[name] for name in GROUP}, "user": person["user"],
                           **({"outcome": outcome} if outcome != "Score" else {}),
                           x_column: float(person[outcome]),
                           **({"peak_correlation_y": float(person["peak_r"])} if predictor == "peak_r"
                              else {("median_peak_lag_ms_y" if predictor == 'signed_lag_s'
                                     else "median_absolute_peak_lag_ms_y"): float(person[predictor])*1000}),
                           "n_recordings": len(rr), "content_ids": person["content_ids"]})
        x = np.array([float(person[outcome]) for person in pp])
        y = np.array([float(person[predictor]) for person in pp])
        pearson = float(stats.pearsonr(x, y).statistic)
        spearman = float(stats.spearmanr(x, y).statistic)
        np.testing.assert_allclose([pearson, spearman],
                                   [float(result["pearson_descriptive"]), float(result["association"])],
                                   atol=1e-10, rtol=1e-10)
        if n_recordings != int(result["n_recordings"]):
            raise ValueError(f"Total recording count mismatch: {k}")
        summary.append({**{name: result[name] for name in GROUP}, "scenario": scenario, "outcome": outcome,
                        "predictor": predictor,
                        "n_users": len(pp), "n_recordings": n_recordings,
                        "pearson_r_descriptive": pearson, "spearman_rho": spearman,
                        "spearman_ci_low": float(result["ci_low"]),
                        "spearman_ci_high": float(result["ci_high"]),
                        **spearman_ci_metadata(result, bootstrap_requested),
                        "spearman_permutation_p": float(result["p"]),
                        "spearman_q_bh_existing_family": float(result["q_bh"]),
                        "existing_family": result["family"], "existing_family_size": result["family_size"]})
    if not summary:
        raise ValueError(f"No supported {signal} {predictor} results in {source}")
    return summary, points, hashes


def data_limits(values, *, include_zero=False, correlation=False):
    """Padded limits that retain every observation without a global outlier scale."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Axis values must be nonempty and finite")
    low, high = float(values.min()), float(values.max())
    if correlation:
        if low < 0 or high > 1+1e-10:
            raise ValueError("Expected positive peak correlations in [0, 1]")
        target = min(1.02, max(.02, high*1.12))
        return (0., min(1.02, float(np.ceil(target*100-1e-12)/100)))
    if include_zero:
        low, high = min(low, 0.), max(high, 0.)
    span = high-low
    pad = .08*span if span > 0 else max(abs(low)*.08, 1.)
    return (low-pad, high+pad)


def comparison_limits(points, outcome, predictor):
    x_column = "score_x" if outcome == "Score" else f"{outcome}_x"
    y_column = {"signed_lag_s": "median_peak_lag_ms_y", "absolute_lag_s": "median_absolute_peak_lag_ms_y",
                "peak_r": "peak_correlation_y"}[predictor]
    x = [p[x_column] for p in points]
    y = [p[y_column] for p in points]
    # Include positive descriptive fit endpoints so a tight upper bound does
    # not cut off the fitted line above the observed points. Correlation zero
    # remains the baseline; negative fitted predictions are outside that view.
    groups = defaultdict(list)
    for point in points:
        groups[group(point)].append(point)
    for pp in groups.values():
        xx = np.array([p[x_column] for p in pp])
        yy = np.array([p[y_column] for p in pp])
        if len(xx) >= 2 and np.ptp(xx) > 0:
            slope, intercept = np.polyfit(xx, yy, 1)
            fitted = slope*np.array([xx.min(), xx.max()])+intercept
            y.extend((np.clip(fitted, 0, 1) if predictor == "peak_r" else fitted).tolist())
    return {"x": data_limits(x),
            "y": data_limits(y, include_zero=predictor == "signed_lag_s", correlation=predictor == "peak_r")}


def figure_row_limits(rows, by_group, signal, outcome, predictor, whole_task=False):
    def selected(specs):
        return [p for task, stage, side in specs
                for p in by_group.get((task, side, signal, stage, "whole" if stage == "__whole__" else "PSM in stage"), [])]
    if whole_task:
        bounds = comparison_limits(selected([spec for row in rows for spec in row]), outcome, predictor)
        return [bounds for _ in rows]
    return [comparison_limits(selected(row), outcome, predictor) for row in rows]


def make_figures(dest, signal, summary, points, outcome="Score", predictor="peak_r", combined_book=None):
    lookup = {group(row): row for row in summary}
    by_group = defaultdict(list)
    for point in points:
        by_group[group(point)].append(point)
    users = sorted({row["user"] for row in points}, key=lambda user: int(user[4:]))
    colors = {user: plt.get_cmap("tab20")(i) for i, user in enumerate(users)}
    is_lag = predictor != "peak_r"
    absolute = predictor == "absolute_lag_s"
    y_column = ("median_absolute_peak_lag_ms_y" if absolute else "median_peak_lag_ms_y") if is_lag else "peak_correlation_y"
    x_column = "score_x" if outcome == "Score" else f"{outcome}_x"
    x_label = "Performance score" if outcome == "Score" else outcome
    file_label = "score" if outcome == "Score" else outcome
    measurement = ("absolute peak lag" if absolute else "peak lag") if is_lag else "peak correlation"
    pdf_prefix = ("absolute_peak_lag" if absolute else "peak_lag") if is_lag else "peak_correlation"
    sides = sorted({row['side'] for row in summary})
    merged = sides == ['max']

    def panel(ax, task, stage, side, bounds):
        k = (task, side, signal, stage, "whole" if stage == "__whole__" else "PSM in stage")
        result, pp = lookup.get(k), by_group[k]
        name = LABELS.get(stage, stage.title())
        instrument = 'Selected PSM' if side == 'max' else ('Left PSM' if side == 'l' else 'Right PSM')
        title = f"Task {task}: {name} / {instrument}"
        ax.set_xlabel(f"{x_label} (participant mean)")
        ax.set_ylabel(("Absolute peak gaze-PSM lag (participant median, ms)" if absolute
                       else "Peak gaze-PSM lag (participant median, ms)") if is_lag
                      else "Peak gaze-PSM correlation (Fisher-z mean)")
        ax.set_xlim(bounds["x"])
        ax.set_ylim(bounds["y"])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        if is_lag:
            ax.axhline(0, color=".4", lw=.9, zorder=0)
        ax.grid(alpha=.17)
        if result is None:
            ax.set_title(title)
            ax.text(.5, .5, "Insufficient support for association", transform=ax.transAxes, ha="center")
            return
        x = np.array([p[x_column] for p in pp])
        y = np.array([p[y_column] for p in pp])
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, slope*xx+intercept, "--", color=".45", lw=1.3, zorder=1)
        for point in pp:
            ax.scatter(point[x_column], point[y_column], s=58, color=colors[point["user"]],
                       edgecolors="white", linewidths=.7, zorder=3)
            ax.annotate(point["user"][4:], (point[x_column], point[y_column]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8, color=".18")
        ax.set_title(title+f"\nPearson r = {result['pearson_r_descriptive']:+.2f}; "
                     f"Spearman \u03c1 = {result['spearman_rho']:+.2f}"
                     f"\n{spearman_ci_label(result)}"
                     f"\np = {result['spearman_permutation_p']:.3g}; "
                     f"BH q = {result['spearman_q_bh_existing_family']:.3g}; n = {result['n_users']} participants",
                     fontsize=11, pad=12)

    plans = []
    if any(row['stage'] == '__whole__' for row in summary):
        plans.append(("whole_task", f"Whole-task {measurement} versus {x_label}",
                      [[(task, "__whole__", side) for side in sides] for task in ("A", "C")]))
    for task, phases in PHASES.items():
        admitted = [stage for stage in phases if any((task, side, signal, stage, "PSM in stage") in lookup
                                                    for side in sides)]
        if admitted:
            plans.append((f"task_{task}_phases", f"Task {task}: valid-phase {measurement} versus {x_label}",
                          [[(task, stage, side) for side in sides] for stage in admitted]))
    saved, axes_metadata = [], []
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}), PdfPages(dest/f"{pdf_prefix}_vs_{file_label}.pdf") as book:
        for name, title, rows in plans:
            fig, axes = plt.subplots(len(rows), len(sides),
                                     figsize=(8 if merged else 13, 4.7*len(rows)+1.2), squeeze=False)
            bounds_by_row = figure_row_limits(rows, by_group, signal, outcome, predictor, name == "whole_task")
            for row_index, (axes_row, groups, bounds) in enumerate(zip(axes, rows, bounds_by_row)):
                for ax, spec in zip(axes_row, groups):
                    panel(ax, *spec, bounds)
                    task, stage, side = spec
                    pp = by_group[(task, side, signal, stage, "whole" if stage == "__whole__" else "PSM in stage")]
                    assert all(bounds["x"][0] <= p[x_column] <= bounds["x"][1] and
                               bounds["y"][0] <= p[y_column] <= bounds["y"][1] for p in pp)
                    axes_metadata.append({"figure": name, "row": row_index, "task": task, "stage": stage,
                                          "side": side, "signal": signal, "outcome": outcome, "predictor": predictor,
                                          "x_limits": list(bounds["x"]), "y_limits": list(bounds["y"]),
                                          "x_data_min": min((p[x_column] for p in pp), default=None),
                                          "x_data_max": max((p[x_column] for p in pp), default=None),
                                          "y_data_min": min((p[y_column] for p in pp), default=None),
                                          "y_data_max": max((p[y_column] for p in pp), default=None),
                                          "n_points": len(pp), "all_points_inside_limits": True,
                                          "shared_group": "whole_task" if name == "whole_task" else f"{task}/{stage}"})
            kind = "Position (primary analysis)" if signal == "position" else "Velocity (secondary analysis)"
            fig.suptitle(title+"\n"+kind, fontsize=17, y=.987)
            footer = (f"One point per participant; labels show participant IDs. Each point uses the same eligible attempts for {x_label} and {measurement}.\n"
                     "Spearman 95% CIs: exploratory participant percentile bootstrap, unadjusted. p: permutation; q: BH; Pearson r/fit: descriptive.\n"
                     "CI unreliable: <97.5% defined bootstrap draws. BH family spans phases and features; phase scores are whole-attempt.")
            if is_lag and not absolute:
                footer += f"\nPositive lag = gaze leads PSM; negative lag = PSM leads gaze. {x_label} values are used directly; missing values are excluded."
            elif outcome != "Score":
                footer += f"\n{outcome} uses the supplied column values directly; attempts with missing {outcome} values are excluded."
            if merged:
                footer += "\nOne PSM selected per recording/stage by signed xcorr peak, before score matching.\nSelection identifies stronger correlation, not the operated instrument. Axes vary by stage."
            else:
                footer += "\nAxes shared within each L/R phase row; ranges can vary between rows. Whole-task panels share axes."
            fig.text(.5, .015, footer,
                     ha="center", va="bottom", fontsize=9, linespacing=1.6)
            bottom = (.35+.20*len(footer.splitlines()))/fig.get_figheight()
            fig.tight_layout(rect=(0, bottom, 1, .93 if len(rows) == 2 else .95), h_pad=2.4, w_pad=2.6)
            separate_labels(fig, axes)
            fig.savefig(dest/f"{name}.png", dpi=180, bbox_inches="tight")
            fig.savefig(dest/f"{name}.pdf", bbox_inches="tight")
            book.savefig(fig, bbox_inches="tight")
            if combined_book is not None:
                combined_book.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved.append(name)
    (dest/"axes.json").write_text(json.dumps({
        "policy": ("One selected-PSM panel per stage; all data retained" if merged else
                   "Whole-task panels share x/y; each phase row shares x/y between L/R; all data retained"),
        "x_padding_fraction": .08, "correlation_zero_baseline": True,
        "lag_zero_retained": True, "lag_symmetry_required": False,
        "panels": axes_metadata, "all_checks_passed": True}, indent=2)+"\n")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/score_corr"))
    parser.add_argument("--output", type=Path, help="Default: INPUT/peak_correlation_vs_OUTCOME/SIGNAL")
    parser.add_argument("--signal", choices=("position", "velocity"), default="position")
    parser.add_argument("--outcome", choices=("Score", "Time", "Eco", "TimeRaw", "EcoRaw", "PenaltySum", "PenaltyRawSum"), default="Score")
    args = parser.parse_args()
    file_label = "score" if args.outcome == "Score" else args.outcome
    dest = args.output or args.input/f"peak_correlation_vs_{file_label}"/args.signal
    if dest.resolve() == args.input.resolve():
        parser.error("Use a separate output directory for these figures")
    summary, points, hashes = load_and_check(args.input, args.signal, args.outcome)
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest/"associations.csv", summary)
    write_csv(dest/"plotted_points.csv", points)
    figures = make_figures(dest, args.signal, summary, points, args.outcome)
    unchanged = all(digest(args.input/name) == expected for name, expected in hashes.items())
    if not unchanged:
        raise RuntimeError("Source analysis changed while plotting")
    validation = {"signal": args.signal, "outcome": args.outcome,
                  "conditions": len(summary), "plotted_points": len(points),
                  "figures": figures, "participant_inputs_match_eligible_recordings": True,
                  "pearson_and_spearman_match_saved_results_and_scipy": True,
                  "spearman_ci_display": ci_display_summary(summary),
                  "source_files_unchanged": unchanged, "input_sha256": hashes,
                  "script_sha256": digest(Path(__file__)), "all_checks_passed": True}
    (dest/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({"output": str(dest), **validation}, indent=2))


if __name__ == "__main__":
    main()
