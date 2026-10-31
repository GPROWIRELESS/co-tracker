#!/usr/bin/env python3
"""Compute Time/Eco score-component associations using the established analysis.

Uses the supplied Time and Eco columns, with the same recording eligibility,
participant summaries, resampling settings and correction families as the
TimeRaw/EcoRaw analysis. Writes component tables for the scatter-plot script.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools import analyze_score_corr as base
from tools.analyze_raw_score_corr import validate_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/gaze_psm_xcorr"))
    parser.add_argument("--index", type=Path, default=Path("Index"))
    parser.add_argument("--output", type=Path, default=Path("outputs/score_corr/component_metrics"))
    parser.add_argument("--outcomes", nargs="+", choices=("Time", "Eco"), default=("Time", "Eco"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=19999)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--min-users", type=int, default=6)
    args = parser.parse_args()
    if args.output.resolve() in (args.input.resolve(), args.index.resolve()):
        parser.error("Use a separate output directory")
    if args.bootstrap < 100 or args.permutations < 100 or args.min_users < 3:
        parser.error("Require bootstrap/permutations >=100 and min-users >=3")
    inputs = [args.input/name for name in ("config.json", "record_summary.csv", "record_curves.csv",
                                          "group_summary.csv", "quality.csv")]
    inputs += [args.index/name for name in ("content_index.csv", "Task_A_Score.csv", "Task_C_Score.csv")]
    hashes = {str(path): base.sha256(path) for path in inputs}
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for outcome in dict.fromkeys(args.outcomes):
        config, _, recordings, records, phases = base.prepare(args, outcome)
        usable = [row for row in recordings if row["score_usable"]]
        assert all(np.isfinite(row[outcome]) for row in usable)
        print(f"{outcome}: {len(usable)}/{len(recordings)} supplied component values", flush=True)
        results, people = base.analyze(args, records, phases, outcome, scenarios=("primary",))
        validation = validate_results(results, people, records, outcome)
        for name, rows in (("associations", results), ("participant_summaries", people),
                           ("matched_outcomes", recordings), ("record_phase_audit", records),
                           ("phase_validity", phases)):
            base.write_csv(args.output/f"{name}_{outcome}.csv", rows)
        metadata[outcome] = {
            "n_available": len(usable), "n_missing": len(recordings)-len(usable),
            "task_counts": {task: sum(row["task"] == task for row in usable) for task in ("A", "C")},
            "n_successful_tests": sum(row["status"] == "ok" for row in results),
            "n_significant_peak_strength_between_participants": sum(
                row["status"] == "ok" and row["method"] == "between_participants"
                and row["predictor"] == "peak_r" and row.get("q_bh", 1) < .05 for row in results),
            "validation": validation,
        }
    unchanged = all(base.sha256(Path(path)) == expected for path, expected in hashes.items())
    assert unchanged
    snapshot = args.output/"code_snapshot"
    snapshot.mkdir(exist_ok=True)
    sources = (Path(__file__), Path(base.__file__), Path(sys.modules[validate_results.__module__].__file__))
    for path in sources:
        (snapshot/path.name).write_bytes(path.read_bytes())
    provenance = {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
        "settings": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "outcomes": metadata, "correlation_config": config, "input_sha256": hashes,
        "analysis_source_sha256": {str(path): base.sha256(path) for path in sources},
        "values_used": "Named Time/Eco columns as supplied, without sign reversal or substitution from raw values",
        "multiple_comparisons": "BH separately by outcome and method, including all three peak predictors",
        "inputs_unchanged": unchanged, "all_checks_passed": True,
    }
    (args.output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
