#!/usr/bin/env python3
"""Replace raw-component overview figures with supplied Time/Eco score figures.

First run tools/analyze_component_score_corr.py to calculate the component tables.
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

from tools import analyze_score_corr as base
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def archive_raw_figures(root):
    """Retire only the five superseded figure files; retain numerical raw tables."""
    paths = [root/"figures"/f"association_overview_{outcome}.{extension}"
             for outcome in ("TimeRaw", "EcoRaw") for extension in ("png", "pdf")]
    paths.append(root/"raw_metric_overviews.pdf")
    archive = root/"figures"/"superseded_raw"
    moved = []
    for path in paths:
        if not path.exists():
            continue
        destination = archive/path.name
        archive.mkdir(exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        digest = base.sha256(path)
        path.rename(destination)
        assert base.sha256(destination) == digest
        moved.append({"from": str(path), "to": str(destination), "sha256": digest})
    notes = root/"raw_metrics"/"README.md"
    if notes.exists():
        text = notes.read_text()
        for outcome in ("TimeRaw", "EcoRaw"):
            text = text.replace(f"../figures/association_overview_{outcome}.",
                                f"../figures/superseded_raw/association_overview_{outcome}.")
        text = text.replace("../raw_metric_overviews.pdf", "../figures/superseded_raw/raw_metric_overviews.pdf")
        notice = ("These raw-column figures have been superseded by the "
                  "[Time/Eco score-component figures](../component_metrics/overview_figures.md).\n\n")
        if notice not in text:
            text = notice+text
        notes.write_text(text)
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/score_corr"))
    args = parser.parse_args()
    root = args.source
    tables = root/"component_metrics"
    analysis = json.loads((tables/"provenance.json").read_text())
    assert all(outcome in analysis["outcomes"] for outcome in ("Time", "Eco"))
    assert analysis["all_checks_passed"]
    assert all(base.sha256(Path(path)) == digest for path, digest in analysis["input_sha256"].items())
    original_paths = [root/name for name in ("associations.csv", "participant_summaries.csv", "report.md",
                                            "score_corr_figures.pdf", "figures/association_overview.pdf",
                                            "figures/association_overview.png")]
    original = {str(p): base.sha256(p) for p in original_paths if p.exists()}
    input_hashes = {}
    output_hashes = {}
    coverage_rows = []
    base.plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with PdfPages(root/"component_score_overviews.pdf") as book:
        for outcome in ("Time", "Eco"):
            paths = {name: tables/f"{name}_{outcome}.csv"
                     for name in ("associations", "phase_validity", "matched_outcomes")}
            input_hashes.update({str(p): base.sha256(p) for p in paths.values()})
            results = base.read_csv(paths["associations"])
            phases = base.read_csv(paths["phase_validity"])
            matched = base.read_csv(paths["matched_outcomes"])
            for row in results:
                assert row["outcome"] == outcome
                for field in ("association", "q_bh"):
                    row[field] = base.number(row[field])
            for row in phases:
                row["group_usable"] = row["group_usable"] == "True"
            usable = [row for row in matched if row["score_usable"] == "True"]
            assert all(np.isfinite(base.number(row[outcome])) for row in usable)
            counts = {task: sum(row["task"] == task for row in usable) for task in ("A", "C")}
            coverage = (f"Available {outcome}: A {counts['A']}, C {counts['C']}; "
                        f"{len({row['user'] for row in usable})} participants.")
            fig = base.make_association_overview(results, phases, outcome, coverage)
            labels = [text.get_text() for text in fig.texts]
            assert any(f"{outcome} vs gaze/PSM" in label for label in labels)
            assert all("TimeRaw" not in label and "EcoRaw" not in label for label in labels)
            for extension in ("png", "pdf"):
                path = root/"figures"/f"association_overview_{outcome}.{extension}"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                output_hashes[str(path)] = base.sha256(path)
            book.savefig(fig, bbox_inches="tight")
            base.plt.close(fig)
            hits = sum(row["status"] == "ok" and row["q_bh"] < .05 for row in results)
            coverage_rows.append(f"| {outcome} | {counts['A']} / {counts['C']} | {len(matched)-len(usable)} | {hits} |")
    output_hashes[str(root/"component_score_overviews.pdf")] = base.sha256(root/"component_score_overviews.pdf")
    archived = archive_raw_figures(root)
    assert all(base.sha256(Path(path)) == digest for path, digest in original.items())
    assert all(base.sha256(Path(path)) == digest for path, digest in input_hashes.items())
    provenance = {"created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                  "outcome_columns": ["Time", "Eco"], "input_tables_sha256": input_hashes,
                  "output_figures_sha256": output_hashes, "archived_raw_figures": archived,
                  "overall_score_outputs_unchanged": True,
                  "source_sha256": {str(p): base.sha256(p) for p in (Path(__file__), Path(base.__file__))},
                  "all_checks_passed": True}
    (tables/"overview_figures.json").write_text(json.dumps(provenance, indent=2)+"\n")
    notes = ["# Time and Eco score-component overview figures", "",
             "These replace the TimeRaw/EcoRaw overview figures using the actual `Time` and `Eco` columns supplied in the score CSVs. "
             "They retain the original layout, phase ordering, three analysis methods, three peak features and −1 to +1 color scale.", "",
             "- [Time PDF](../figures/association_overview_Time.pdf) / [PNG](../figures/association_overview_Time.png)",
             "- [Eco PDF](../figures/association_overview_Eco.pdf) / [PNG](../figures/association_overview_Eco.png)",
             "- [Both figures in one PDF](../component_score_overviews.pdf)", "",
             "The supplied component-score values are used directly. Positive coefficients mean higher `Time`/`Eco` scores; negative coefficients mean lower component scores. "
             "Missing scores are excluded, and supplied negative scores are retained. No values are substituted from TimeRaw/EcoRaw.", "",
             "The established participant aggregation, rank-based comparisons, order/attempt adjustment and participant resampling are unchanged. "
             "BH correction is performed separately for each outcome and method across all three peak features and admitted whole-task/phase conditions. "
             "Asterisks indicate q < 0.05 within that family. These figures use the primary position analysis.", "",
             "| Outcome | Available A / C | Missing values | q < 0.05 |", "|---|---:|---:|---:|",
             *coverage_rows, "",
             "Phase-specific exclusions can reduce these counts. The component sample contains completed attempts and differs from the overall-Score sample, which included scored failures. "
             "The component CSV tables and provenance record exact inputs, estimates, uncertainty, p/q values, participant counts and independent numerical checks.", "",
             "The previous raw-column figures are retained under `figures/superseded_raw/`. Numerical raw tables remain available for existing analyses. "
             "The original overall-Score figures and tables were verified unchanged.", "",
             "Reproduce from the repository root:", "", "```bash",
             "python tools/analyze_component_score_corr.py",
             f"python tools/plot_component_score_corr_overviews.py --source {root}", "```", ""]
    (tables/"overview_figures.md").write_text("\n".join(notes))
    snapshot = tables/"code_snapshot"
    snapshot.mkdir(exist_ok=True)
    (snapshot/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"outcome_columns": ["Time", "Eco"], "new_figures": list(output_hashes),
                      "archived_raw_figures": len(archived), "all_checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
