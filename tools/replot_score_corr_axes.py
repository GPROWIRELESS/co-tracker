#!/usr/bin/env python3
"""Audit and tighten all active position/velocity peak/lag scatter axes.

Refreshes figures only, preserving every numerical CSV and archiving the
previous PNG/PDF figures for comparison. No bootstrap/permutation rerun.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tools import plot_peak_correlation_vs_score as plots


def old_limits(points, outcome, predictor):
    """The previous all-task/all-phase shared limits, for the change audit."""
    lag = predictor == "signed_lag_s"
    yc = "median_peak_lag_ms_y" if lag else "peak_correlation_y"
    maximum = max(abs(p[yc]) for p in points) if lag else max(p[yc] for p in points)
    top = (max(100., np.ceil((maximum+max(100., maximum*.06))/100)*100) if lag else
           min(1.02, max(.1, np.ceil((maximum+.04)*10)/10)))
    xc = "score_x" if outcome == "Score" else f"{outcome}_x"
    low, high = min(p[xc] for p in points), max(p[xc] for p in points)
    if outcome == "Score":
        low, high, pad = min(0., low), max(100., high), 3.
    else:
        pad = max((high-low)*.06, abs(high)*.01, 1.)
    return {"x": [low-pad, high+pad], "y": [-top if lag else 0., top]}


def check_saved_inputs(folder, summary, points, outcome, predictor):
    """Ensure the fresh display inputs match the existing saved numerical data."""
    stored_points = plots.read_csv(folder/"plotted_points.csv")
    stored_results = plots.read_csv(folder/"associations.csv")
    old_people = {(*plots.group(r), r["user"]): r for r in stored_points}
    old_results = {plots.group(r): r for r in stored_results}
    assert len(old_people) == len(points) and len(old_results) == len(summary)
    xc = "score_x" if outcome == "Score" else f"{outcome}_x"
    yc = "median_peak_lag_ms_y" if predictor == "signed_lag_s" else "peak_correlation_y"
    for person in points:
        original = old_people[(*plots.group(person), person["user"])]
        np.testing.assert_allclose([float(original[xc]), float(original[yc])],
                                   [person[xc], person[yc]], atol=1e-10)
        assert set(original["content_ids"].split(";")) == set(person["content_ids"].split(";"))
    for result in summary:
        original = old_results[plots.group(result)]
        for name in ("pearson_r_descriptive", "spearman_rho", "spearman_ci_low", "spearman_ci_high",
                     "spearman_permutation_p", "spearman_q_bh_existing_family"):
            np.testing.assert_allclose(float(original[name]), result[name], atol=1e-10, equal_nan=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/score_corr"))
    args = parser.parse_args()
    root = args.source
    audit = root/"axis_audit"
    audit.mkdir(exist_ok=True)
    numeric_hashes = {str(p): plots.digest(p) for p in root.rglob("*.csv") if audit not in p.parents}
    plans = []
    for prefix, predictor in (("peak_correlation_vs_", "peak_r"), ("lag_correlation_vs_", "signed_lag_s")):
        for parent in sorted(root.glob(prefix+"*")):
            outcome = parent.name.removeprefix(prefix)
            outcome = "Score" if outcome.lower() == "score" else outcome
            if outcome not in ("Score", "Time", "Eco", "TimeRaw", "EcoRaw"):
                continue
            for signal in ("position", "velocity"):
                folder = parent/signal
                if (folder/"plotted_points.csv").exists():
                    plans.append((folder, signal, outcome, predictor))
    if not plans:
        raise ValueError("No active peak/lag figure folders found")
    records, archived, refreshes = [], [], []
    for folder, signal, outcome, predictor in plans:
        summary, points, source_hashes = plots.load_and_check(root, signal, outcome, predictor)
        check_saved_inputs(folder, summary, points, outcome, predictor)
        previous = (json.loads((folder/"axes.json").read_text()) if (folder/"axes.json").exists() else None)
        previous_lookup = ({(p["figure"], p["task"], p["stage"], p["side"]): p for p in previous["panels"]}
                           if previous else {})
        legacy = old_limits(points, outcome, predictor)
        for path in sorted(folder.iterdir()):
            if path.suffix not in (".png", ".pdf"):
                continue
            dest = audit/"before"/path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(path, dest)
            archived.append({"original_path": str(path), "backup_path": str(dest),
                             "backup_sha256": plots.digest(dest)})
        figures = plots.make_figures(folder, signal, summary, points, outcome, predictor)
        axes = json.loads((folder/"axes.json").read_text())
        groups = {}
        for panel in axes["panels"]:
            k = (panel["figure"], panel["task"], panel["stage"], panel["side"])
            old = previous_lookup.get(k, {"x_limits": legacy["x"], "y_limits": legacy["y"]})
            group = (panel["figure"], panel["shared_group"])
            bounds = (panel["x_limits"], panel["y_limits"])
            if group in groups:
                assert groups[group] == bounds, (folder, group)
            groups[group] = bounds
            assert panel["all_points_inside_limits"]
            if panel["n_points"]:
                assert panel["x_limits"][0] <= panel["x_data_min"] <= panel["x_data_max"] <= panel["x_limits"][1]
                assert panel["y_limits"][0] <= panel["y_data_min"] <= panel["y_data_max"] <= panel["y_limits"][1]
            assert panel["y_limits"][0] <= 0 <= panel["y_limits"][1]
            records.append({"folder": str(folder), **{k: panel[k] for k in
                           ("figure", "task", "stage", "side", "signal", "outcome", "predictor", "n_points")},
                            "old_x_low": old["x_limits"][0], "old_x_high": old["x_limits"][1],
                            "new_x_low": panel["x_limits"][0], "new_x_high": panel["x_limits"][1],
                            "old_y_low": old["y_limits"][0], "old_y_high": old["y_limits"][1],
                            "new_y_low": panel["y_limits"][0], "new_y_high": panel["y_limits"][1],
                            **{k: panel[k] for k in ("x_data_min", "x_data_max", "y_data_min", "y_data_max")},
                            "all_points_inside_limits": True})
        assert all(plots.digest(root/name) == digest for name, digest in source_hashes.items())
        refreshes.append({"folder": str(folder), "figures": figures, "n_panels": len(axes["panels"]),
                          "source_analysis_sha256": source_hashes,
                          "figure_sha256": {p.name: plots.digest(p) for p in folder.iterdir() if p.suffix in (".png", ".pdf")}})
        print(f"Refreshed {folder}: {len(figures)} figures, {len(axes['panels'])} panels", flush=True)
    assert all(plots.digest(Path(path)) == digest for path, digest in numeric_hashes.items())
    plots.write_csv(audit/"ranges.csv", records)
    sources = (Path(__file__), Path(plots.__file__))
    snapshot = audit/"code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in sources:
        (snapshot/path.name).write_bytes(path.read_bytes())
    metadata = {"created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                "n_folders": len(plans), "n_figures": sum(len(r["figures"]) for r in refreshes),
                "n_panels": len(records), "n_original_csv_files_verified_unchanged": len(numeric_hashes),
                "all_points_visible": True, "comparison_group_axes_shared": True,
                "zero_retained_on_y_axes": True, "statistical_CSV_sha256": numeric_hashes,
                "refreshes": refreshes, "archived_figures": archived,
                "source_sha256": {str(p): plots.digest(p) for p in sources}, "all_checks_passed": True}
    (audit/"validation.json").write_text(json.dumps(metadata, indent=2)+"\n")
    notes = ["# Peak/lag versus score axis-range audit", "",
             f"Checked and refreshed {metadata['n_figures']} active scatter figures ({len(records)} panels) across "
             f"{len(plans)} position/velocity result folders. All matching booklets were regenerated. "
             "This is a display-only update; numerical inputs, point membership, correlations, intervals and p/q values were verified unchanged.", "",
             "The old plotting function borrowed limits from every whole-task and phase point. Large threading values therefore expanded unrelated plots. "
             "For velocity whole-task peak correlations, the old upper limit was 0.6 despite a largest point near 0.084; the new whole-task range is 0–0.10.", "",
             "- Whole-task panels share x/y limits across tasks and instrument sides.",
             "- Phase panels share x/y limits within each left/right row. Different phase rows can use different ranges; read the ticks when comparing rows.",
             "- X limits follow the displayed participant outcomes, with 8% padding; negative component scores and all raw values remain visible.",
             "- Peak-correlation y axes retain zero, with modest headroom above the comparison group's data and positive fitted-line endpoints.",
             "- Signed-lag y axes retain zero and all observations, without requiring equal positive and negative limits.",
             "- Categorical heatmaps and correlation coefficient color scales retain their meaningful −1 to +1 range.", "",
             "[All before/after ranges](ranges.csv) and [validation](validation.json) record every panel. "
             "Previous images/booklets are retained under `before/`; refreshed files use the original paths.", "",
             "Examples:", "",
             "- [Velocity: Score, whole task](../peak_correlation_vs_score/velocity/whole_task.png)",
             "- [Velocity: Score, Task A phases](../peak_correlation_vs_score/velocity/task_A_phases.png)",
             "- [Velocity: Score, Task C phases](../peak_correlation_vs_score/velocity/task_C_phases.png)",
             "- [Velocity: Score, signed lag](../lag_correlation_vs_score/velocity/whole_task.png)", "",
             "Reproduce from the repository root:", "", "```bash",
             f"python tools/replot_score_corr_axes.py --source {root}", "```", ""]
    (audit/"README.md").write_text("\n".join(notes))
    print(json.dumps({k: metadata[k] for k in ("n_folders", "n_figures", "n_panels",
                                              "n_original_csv_files_verified_unchanged", "all_checks_passed")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
