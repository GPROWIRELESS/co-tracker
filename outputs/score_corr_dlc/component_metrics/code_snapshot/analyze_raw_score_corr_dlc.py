#!/usr/bin/env python3
"""Add TimeRaw and EcoRaw association overviews to outputs/score_corr_dlc."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import analyze_score_corr_dlc as base
import numpy as np
import scipy
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages


def validate_results(results, people, records, outcome):
    """Check all estimates using independent full-dummy regressions/SciPy."""
    counts = defaultdict(int)
    for result in results:
        if result["status"] != "ok":
            continue
        if result["method"] == "between_participants":
            pp = [p for p in people if base.key(p) == base.key(result)]
            expected = stats.spearmanr([p[result["predictor"]] for p in pp],
                                       [p[outcome] for p in pp]).statistic
            assert np.isclose(expected, result["association"], atol=1e-10)
            for person in pp:
                rr = [r for r in records if r["eligible"] and base.key(r) == base.key(result)
                      and r["user"] == person["user"]]
                assert np.isclose(person[outcome], np.mean([r[outcome] for r in rr]))
            counts["between_vs_scipy"] += 1
        else:
            rr = [r for r in base.scenario_rows(records, result["scenario"]) if base.key(r) == base.key(result)]
            users = sorted({r["user"] for r in rr})
            dummy = np.array([[r["user"] == u for u in users] for r in rr], dtype=float)
            x = stats.rankdata([r[result["predictor"]] for r in rr])
            y = stats.rankdata([r[outcome] for r in rr])
            weights = 1/(dummy @ dummy.sum(axis=0))
            root = np.sqrt(weights)
            nuisance = dummy
            if result["method"] == "within_adjusted_order":
                nuisance = np.column_stack((dummy, [r["TaskNum"] for r in rr], [r["Attempt"] for r in rr]))
            residuals = [v-nuisance @ np.linalg.lstsq(nuisance*root[:, None], v*root, rcond=None)[0]
                         for v in (x, y)]
            expected = np.sum(weights*residuals[0]*residuals[1])/np.sqrt(
                np.sum(weights*residuals[0]**2)*np.sum(weights*residuals[1]**2))
            assert np.isclose(expected, result["association"], atol=1e-8)
            full = np.column_stack((x, nuisance))
            beta = np.linalg.lstsq(full*root[:, None], y*root, rcond=None)[0]
            error = y-full @ beta
            bread = np.linalg.pinv(full.T @ (weights[:, None]*full))
            scores = np.array([full[dummy[:, i] == 1].T @
                               (weights[dummy[:, i] == 1]*error[dummy[:, i] == 1])
                               for i in range(len(users))])
            n, g = len(rr), len(users)
            covariance = bread @ (scores.T @ scores) @ bread*g/(g-1)*(n-1)/(n-np.linalg.matrix_rank(full))
            se = np.sqrt(covariance[0, 0])
            p = float(2*stats.t.sf(abs(beta[0]/se), g-1))
            assert np.isclose(p, result["p"], atol=1e-7)
            assert np.isclose(se, result["rank_slope_cluster_se"], atol=1e-7)
            counts["within_vs_full_dummy_regression"] += 1
    for family in {r["family"] for r in results}:
        rr = [r for r in results if r["family"] == family and r["status"] == "ok"]
        if rr:
            expected = stats.false_discovery_control([r["p"] for r in rr], method="bh")
            np.testing.assert_allclose(expected, [r["q_bh"] for r in rr], atol=1e-10)
            counts["BH_families_vs_scipy"] += 1
    return {**counts, "all_checks_passed": True}


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
    if args.bootstrap < 100 or args.permutations < 100 or args.min_users < 3:
        parser.error("Require bootstrap/permutations >=100 and min-users >=3")
    if args.output.resolve() in (args.input.resolve(), args.index.resolve()):
        parser.error("Use a separate output directory")
    existing = {str(p): base.sha256(p) for p in args.output.rglob("*") if p.is_file()
                and p.name not in ("association_overview_TimeRaw.png", "association_overview_TimeRaw.pdf",
                                   "association_overview_EcoRaw.png", "association_overview_EcoRaw.pdf",
                                   "raw_metric_overviews.pdf") and "raw_metrics" not in p.parts}
    inputs = [args.input/name for name in ("config.json", "record_summary.csv", "record_curves.csv", "group_summary.csv", "quality.csv")]
    inputs += [args.index/name for name in ("content_index.csv", "Task_A_Score.csv", "Task_C_Score.csv")]
    hashes = {str(p): base.sha256(p) for p in inputs}
    output = args.output/"raw_metrics"
    output.mkdir(parents=True, exist_ok=True)
    figures = args.output/"figures"
    figures.mkdir(exist_ok=True)
    base.plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    metadata = {}
    with PdfPages(args.output/"raw_metric_overviews.pdf") as book:
        for outcome in ("TimeRaw", "EcoRaw"):
            _, _, recordings, records, phases = base.prepare(args, outcome)
            usable = [r for r in recordings if r["score_usable"]]
            assert all(np.isfinite(r[outcome]) for r in usable)
            counts = {task: sum(r["task"] == task for r in usable) for task in ("A", "C")}
            print(f"{outcome}: {len(usable)}/{len(recordings)} raw outcomes; tasks {counts}", flush=True)
            results, people = base.analyze(args, records, phases, outcome, scenarios=("primary",))
            validation = validate_results(results, people, records, outcome)
            for name, rows in (("associations", results), ("participant_summaries", people),
                               ("matched_outcomes", recordings), ("record_phase_audit", records),
                               ("phase_validity", phases)):
                base.write_csv(output/f"{name}_{outcome}.csv", rows)
            coverage = f"Available {outcome}: A {counts['A']}, C {counts['C']}; {len({r['user'] for r in usable})} participants."
            fig = base.make_association_overview(results, phases, outcome, coverage)
            name = f"association_overview_{outcome}"
            fig.savefig(figures/f"{name}.png", dpi=180, bbox_inches="tight")
            fig.savefig(figures/f"{name}.pdf", bbox_inches="tight")
            book.savefig(fig, bbox_inches="tight")
            base.plt.close(fig)
            significant = [r for r in results if r["status"] == "ok" and r.get("q_bh", 1) < .05]
            metadata[outcome] = {"n_available": len(usable), "task_counts": counts,
                                 "n_missing": len(recordings)-len(usable),
                                 "n_tests": sum(r["status"] == "ok" for r in results),
                                 "significant_results": significant, "validation": validation}
    assert all(base.sha256(Path(p)) == h for p, h in hashes.items())
    assert all(base.sha256(Path(p)) == h for p, h in existing.items())
    snapshot = output/"code_snapshot"
    snapshot.mkdir(exist_ok=True)
    scripts = (Path(__file__), Path(base.__file__))
    for path in scripts:
        (snapshot/path.name).write_bytes(path.read_bytes())
    provenance = {"created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                  "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "input_sha256": hashes, "existing_output_sha256": existing,
                  "analysis_source_sha256": {str(p): base.sha256(p) for p in scripts},
                  "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                               "matplotlib": base.matplotlib.__version__},
                  "outcomes": metadata, "input_and_existing_outputs_unchanged": True}
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    notes = ["# TimeRaw and EcoRaw association overviews", "",
             "These add two figures matching the original position-association overview: same rows, three analysis panels, three peak features and a −1 to +1 color scale.", "",
             "- [TimeRaw PDF](../figures/association_overview_TimeRaw.pdf) / [PNG](../figures/association_overview_TimeRaw.png)",
             "- [EcoRaw PDF](../figures/association_overview_EcoRaw.pdf) / [PNG](../figures/association_overview_EcoRaw.png)",
             "- [Both figures in one PDF](../raw_metric_overviews.pdf)", "",
             "The outcome is the named raw CSV field, used directly with no sign reversal. Positive coefficients mean a larger raw outcome, and negative coefficients mean a smaller raw outcome. "
             "For TimeRaw, positive means longer duration and negative means shorter duration. EcoRaw is labeled as the raw economy-of-motion measure; no unit conversion is applied.", "",
             "Peak signed lag, absolute lag, peak r, participant aggregation, rank transformations, equal participant weighting, order/attempt adjustment, "
             f"{args.bootstrap:,} participant bootstrap replicates and {args.permutations:,} participant permutations use the same methods as the original analysis. "
             "Between-participant outcomes are means of the exact same eligible recordings used for the peak features. "
             "Each within-participant outcome is the recording's own raw value. Missing raw values are excluded and never imputed. "
             "The `score_usable` column in these component-specific audit files indicates availability of that file's named raw outcome.", "",
             "BH correction is separate for each outcome and analysis method, pooling all three peak features and all admitted whole-task/phase conditions. "
             "It is not jointly corrected across overall Score, TimeRaw and EcoRaw. Asterisks mark q < 0.05 in that outcome/method family. "
             "Only the primary position analysis is shown; no additional sensitivity or velocity analyses were run for these two outcomes.", "",
             "| Outcome | Available A / C | Missing raw outcomes | Successful tests | q < 0.05 |",
             "|---|---:|---:|---:|---:|"]
    for outcome, info in metadata.items():
        notes.append(f"| {outcome} | {info['task_counts']['A']} / {info['task_counts']['C']} | {info['n_missing']} | {info['n_tests']} | {len(info['significant_results'])} |")
    notes += ["", "All available raw outcomes in this run come from completed attempts. The raw-outcome sample differs from the overall-Score analysis, which included scored failures. "
              "Differences across figures can therefore reflect both the outcome definition and the retained attempts. Phase-specific exclusions can reduce these counts further.", "",
              "The CSV files contain estimates, uncertainty, p/q values, participant counts and exact matched recording IDs. "
              "Every successful correlation and cluster-adjusted p value was checked independently using SciPy and full weighted regressions with participant dummy variables. "
              "All BH values were checked against SciPy. Input and original overall-Score output hashes remained unchanged.", "",
              "Reproduce from the repository root:", "", "```bash",
              f"python tools/analyze_raw_score_corr_dlc.py --input {args.input} --index {args.index} --output {args.output} "
              f"--bootstrap {args.bootstrap} --permutations {args.permutations} --seed {args.seed} --min-users {args.min_users}",
              "```", ""]
    (output/"README.md").write_text("\n".join(notes))
    print(json.dumps({k: {f: v for f, v in info.items() if f != "significant_results"}
                      for k, info in metadata.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
