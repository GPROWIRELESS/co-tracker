"""Summarize occurrence peaks by record + stage, with left/right columns.

Input is stage_level_xcorr.csv; this script does not search the source index
or reload trajectories. Each observed stage has one output row per record.
Outputs duration-weighted arithmetic r, duration-weighted Fisher-z r and
duration-weighted arithmetic lag. Weights are original occurrence rows/60;
unsupported peaks are excluded separately for each side. All index meta
columns and dvss-relative source paths are carried through unchanged.
"""
import argparse
import csv
from pathlib import Path

from stage_xcorr_common import aggregate_stage_rows
from xcorr_common import write_csv

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "stage_level_xcorr.csv")
    parser.add_argument("--output", type=Path, default=HERE / "record_level_stage_xcorr.csv")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames or [], list(reader)
    summaries, fields = aggregate_stage_rows(rows, fields)
    write_csv(args.output, summaries, fields)
    print(f"Saved {len(summaries)} record/stage summaries to {args.output.resolve()}")


if __name__ == "__main__":
    main()
