#!/usr/bin/env python3
"""Plot signed peak lag versus Score, Eco, EcoRaw, Time and TimeRaw.

Reads validated participant summaries and Spearman permutation results, keeping
the established eligibility, participant weighting and correction families.
Each score's figures and provenance go in lag_correlation_vs_<score>/<signal>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import plot_peak_correlation_vs_score as plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/score_corr"))
    parser.add_argument("--output", type=Path,
                        help="Parent of lag_correlation_vs_<score>/<signal>/ (default: INPUT)")
    parser.add_argument("--signal", choices=("position", "velocity"), default="position")
    parser.add_argument("--outcomes", nargs="+", choices=("Score", "Eco", "EcoRaw", "Time", "TimeRaw", "PenaltySum", "PenaltyRawSum"),
                        default=("Eco", "EcoRaw", "Time", "TimeRaw"))
    args = parser.parse_args()
    output = args.output or args.input
    sources = (Path(__file__), Path(plots.__file__))
    source_hashes = {str(path): plots.digest(path) for path in sources}
    metadata, hashes = {}, {}
    for outcome in dict.fromkeys(args.outcomes):
        summary, points, input_hashes = plots.load_and_check(args.input, args.signal, outcome, "signed_lag_s")
        file_label = "score" if outcome == "Score" else outcome
        dest = output/f"lag_correlation_vs_{file_label}"/args.signal
        dest.mkdir(parents=True, exist_ok=True)
        plots.write_csv(dest/"associations.csv", summary)
        plots.write_csv(dest/"plotted_points.csv", points)
        figures = plots.make_figures(dest, args.signal, summary, points, outcome, predictor="signed_lag_s")
        if not all(plots.digest(args.input/name) == expected for name, expected in input_hashes.items()):
            raise RuntimeError("Source analysis changed while plotting")
        hashes.update(input_hashes)
        validation = {
            "outcome": outcome, "signal": args.signal, "predictor": "signed_lag_s",
            "y_units": "milliseconds", "positive_lag": "gaze leads PSM",
            "participant_summary": "median signed recording peak lag; mean matched outcome",
            "conditions": len(summary), "plotted_points": len(points), "figures": figures,
            "participant_medians_and_mean_outcomes_match_eligible_recordings": True,
            "pearson_and_spearman_match_saved_results_and_scipy": True,
            "spearman_ci_display": plots.ci_display_summary(summary),
            "input_sha256": input_hashes, "all_checks_passed": True,
        }
        (dest/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
        snapshot = dest/"code_snapshot"
        snapshot.mkdir(exist_ok=True)
        for path in sources:
            (snapshot/path.name).write_bytes(path.read_bytes())
        provenance = {
            "outcome": outcome, "signal": args.signal, "input_root": str(args.input.resolve()),
            "booklet": f"peak_lag_vs_{file_label}.pdf", "input_sha256": input_hashes,
            "source_sha256": source_hashes, "source_files_unchanged": True, "all_checks_passed": True,
            "spearman_ci_display": plots.ci_display_summary(summary),
        }
        (dest/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
        metadata[outcome] = {"output": str(dest), **{name: validation[name] for name in ("conditions", "plotted_points", "figures")}}
        print(f"{outcome}: {len(summary)} conditions, {len(points)} participant points", flush=True)
    if not all(plots.digest(args.input/name) == expected for name, expected in hashes.items()):
        raise RuntimeError("Source analysis changed while plotting")
    print(json.dumps({"output": str(output), "outcomes": metadata, "all_checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
