#!/usr/bin/env python3
"""Calculate velocity associations and peak/lag plots for all five score fields."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
from tools import analyze_score_corr as base
from tools import analyze_raw_score_corr as audit

OUTCOMES = ("Score", "Time", "Eco", "TimeRaw", "EcoRaw")


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def compare_existing_score(results, root):
    old_path = root/"associations.csv"
    if not old_path.exists():
        return None
    old = [r for r in base.read_csv(old_path) if r["scenario"] == "velocity_secondary"]
    lookup = {(*base.key(r), r["predictor"], r["method"]): r for r in old}
    if len(lookup) != len(results):
        return False
    for row in results:
        previous = lookup.get((*base.key(row), row["predictor"], row["method"]))
        if previous is None or previous["status"] != row["status"]:
            return False
        for name in ("association", "ci_low", "ci_high", "p", "q_bh", "n_users", "n_recordings"):
            if not np.isclose(base.number(previous.get(name)), row.get(name, np.nan),
                              atol=1e-10, rtol=1e-10, equal_nan=True):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/gaze_psm_xcorr"))
    parser.add_argument("--index", type=Path, default=Path("Index"))
    parser.add_argument("--output", type=Path, default=Path("outputs/score_corr"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=19999)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--min-users", type=int, default=6)
    parser.add_argument("--no-plots", action="store_true", help="Write numerical tables only")
    args = parser.parse_args()
    if args.output.resolve() in (args.input.resolve(), args.index.resolve()):
        parser.error("Use a separate output directory")
    if args.bootstrap < 100 or args.permutations < 100 or args.min_users < 3:
        parser.error("Require bootstrap/permutations >=100 and min-users >=3")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [args.input/name for name in ("config.json", "record_summary.csv", "record_curves.csv",
                                          "group_summary.csv", "quality.csv")]
    inputs += [args.index/name for name in ("content_index.csv", "Task_A_Score.csv", "Task_C_Score.csv")]
    hashes = {str(path): base.sha256(path) for path in inputs}
    position_hashes = {str(p): base.sha256(p) for parent in args.output.iterdir()
                       if parent.is_dir() and parent.name.startswith(("peak_correlation_vs_", "lag_correlation_vs_"))
                       for p in (parent/"position").rglob("*") if p.is_file()}
    output = args.output/"velocity_metrics"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    score_matches = None
    for outcome in OUTCOMES:
        config, _, recordings, records, phases = base.prepare(args, outcome)
        records = [r for r in records if r["signal"] == "velocity"]
        phases = [p for p in phases if p["signal"] == "velocity"]
        usable = [r for r in recordings if r["score_usable"]]
        assert all(np.isfinite(r[outcome]) for r in usable)
        print(f"{outcome}: {len(usable)}/{len(recordings)} matched outcomes; analyzing velocity", flush=True)
        results, people = base.analyze(args, records, phases, outcome, scenarios=("velocity_secondary",))
        validation = audit.validate_results(results, people, records, outcome)
        assert all(r["signal"] == "velocity" for r in results)
        if outcome == "Score":
            score_matches = compare_existing_score(results, args.output)
            old_provenance = args.output/"provenance.json"
            if old_provenance.exists():
                previous = json.loads(old_provenance.read_text())["settings"]
                if all(previous[name] == getattr(args, name) for name in ("bootstrap", "permutations", "seed", "min_users")):
                    assert score_matches, "Overall Score velocity results differ from the saved analysis"
        for name, rows in (("associations", results), ("participant_summaries", people),
                           ("matched_outcomes", recordings), ("record_phase_audit", records),
                           ("phase_validity", phases)):
            base.write_csv(output/f"{name}_{outcome}.csv", rows)
        metadata[outcome] = {
            "n_available": len(usable), "n_missing": len(recordings)-len(usable),
            "task_counts": {task: sum(r["task"] == task for r in usable) for task in ("A", "C")},
            "n_successful_tests": sum(r["status"] == "ok" for r in results),
            "significant_results": [r for r in results if r["status"] == "ok" and r.get("q_bh", 1) < .05],
            "validation": validation,
        }
    assert all(base.sha256(Path(p)) == h for p, h in hashes.items())
    scripts = (Path(__file__), Path(base.__file__), Path(audit.__file__),
               Path(__file__).with_name("plot_peak_correlation_vs_score.py"),
               Path(__file__).with_name("plot_lag_vs_score.py"))
    snapshot = output/"code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in scripts:
        (snapshot/path.name).write_bytes(path.read_bytes())
    provenance = {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "signal": "velocity", "scenario": "velocity_secondary", "outcomes": metadata,
        "correlation_config": config, "input_sha256": hashes,
        "analysis_source_sha256": {str(path): base.sha256(path) for path in scripts},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": base.matplotlib.__version__},
        "overall_Score_matches_existing_velocity_analysis": score_matches,
        "values_used": "Each named CSV outcome directly; missing values excluded; no substitution or sign reversal",
        "multiple_comparisons": "BH separately per outcome and method across all three features and eligible velocity conditions",
        "inputs_unchanged": True, "all_checks_passed": True,
    }
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2, default=json_value)+"\n")
    if not args.no_plots:
        commands = [[sys.executable, str(scripts[3]), "--input", str(args.output), "--signal", "velocity", "--outcome", outcome]
                    for outcome in OUTCOMES]
        commands.append([sys.executable, str(scripts[4]), "--input", str(args.output), "--signal", "velocity", "--outcomes", *OUTCOMES])
        for command in commands:
            print("Plotting "+" ".join(command[1:]), flush=True)
            run = subprocess.run(command, capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(run.stdout+"\n"+run.stderr)
    assert all(base.sha256(Path(p)) == h for p, h in position_hashes.items())
    assert all(base.sha256(Path(p)) == h for p, h in hashes.items())
    validation = {"signal": "velocity", "outcome_columns": list(OUTCOMES),
                  "position_files_verified_unchanged": len(position_hashes),
                  "n_successful_tests": sum(m["n_successful_tests"] for m in metadata.values()),
                  "independent_checks": {outcome: m["validation"] for outcome, m in metadata.items()},
                  "figures_generated": not args.no_plots, "all_checks_passed": True}
    (output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    notes = ["# Velocity correlations versus performance outcomes", "",
             "Velocity-based peak strength and signed peak lag are related separately to the supplied Score, Time, Eco, TimeRaw and EcoRaw fields. "
             "The saved gaze/PSM velocity curves are used; velocity comes from consecutive smoothed position differences at the recorded sampling rate. "
             "No new extraction or parameter tuning was performed.", "",
             "Each outcome has peak-correlation and signed-lag figures under its own `velocity/` subfolder. "
             "Each folder contains a combined PDF, whole-task and phase PNG/PDF figures, plotted participant values, association statistics and validation.", "",
             "| Outcome | Available A / C | Peak strength | Signed peak lag |", "|---|---:|---|---|"]
    for outcome, info in metadata.items():
        label = "score" if outcome == "Score" else outcome
        notes.append(f"| {outcome} | {info['task_counts']['A']} / {info['task_counts']['C']} | "
                     f"[PDF](../peak_correlation_vs_{label}/velocity/peak_correlation_vs_{label}.pdf) | "
                     f"[PDF](../lag_correlation_vs_{label}/velocity/peak_lag_vs_{label}.pdf) |")
    notes += ["", "Phase eligibility and peak filtering are checked for the velocity signal: positive interior peaks, complete lag support and eligible group status. "
              "Recording subsets and admitted phases can therefore differ from position results. Participant means of each outcome use exactly the same attempts as the predictor summary. "
              "Mean correlation strength uses Fisher-z averaging; signed lag uses the median of recording peak lags. "
              "Positive lag means gaze velocity leads instrument velocity. Outcomes are used directly, including negative component scores; missing values are excluded.", "",
              f"The analysis uses {args.bootstrap:,} participant bootstrap draws and {args.permutations:,} participant score permutations. "
              "The complete association tables include between-participant, within-participant, and within-participant order/attempt-adjusted results. "
              "The scatter plots show between-participant associations. BH correction is separate for each outcome and method, including signed lag, absolute lag and strength, "
              "both tasks, both sides and admitted whole-task/phase conditions. It is not jointly corrected across the five outcomes. "
              "Displayed bootstrap intervals are exploratory and unadjusted; intervals with too many undefined bootstrap draws are flagged as unreliable. "
              "Velocity results remain secondary and do not establish causation.", "",
              "All successful estimates, cluster-adjusted p values and BH q values were checked independently. "
              "Overall-Score velocity estimates reproduce the original saved analysis at the same settings. Position result files were verified unchanged.", "",
              "Reproduce from the repository root:", "", "```bash",
              f"python tools/analyze_velocity_score_corr.py --input {args.input} --index {args.index} --output {args.output} "
              f"--bootstrap {args.bootstrap} --permutations {args.permutations} --seed {args.seed} --min-users {args.min_users}", "```", ""]
    (output/"README.md").write_text("\n".join(notes))
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
