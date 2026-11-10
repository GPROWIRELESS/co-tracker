"""Compute one row per continuous stage occurrence, with left/right peaks.

Run prepare_xcorr_path_list.py separately to read the source index and check
NPY/DLC pairs. This script consumes that matched JSON list and its metadata.
Stage names are audited before computation. Unknown labels or globally
missing requested stages stop the run and are reported for a user decision.
Known ignored labels: drop ring, none, empty. transport ring = transfer ring.
Positive lag means gaze leads. Search is +/-2 seconds; no 500-pair cutoff.
"""
import argparse
from pathlib import Path

from stage_xcorr_common import (
    AUDIT_FIELDS, compute_stage_records, format_stage_output, label_audit,
)
from xcorr_common import DEFAULT_PAIRS, read_path_list, write_csv

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=HERE / "stage_level_xcorr.csv")
    parser.add_argument("--label-audit", type=Path, default=HERE / "stage_label_validation.csv")
    args = parser.parse_args()
    entries = read_path_list(args.pairs)
    audit, unexpected, missing = label_audit(entries)
    write_csv(args.label_audit, audit, AUDIT_FIELDS)
    if unexpected or missing:
        parser.error(f"Stage label validation failed. Unexpected={unexpected!r}; "
                     f"missing={missing!r}. Review {args.label_audit.resolve()} before proceeding.")
    rows, fields = format_stage_output(entries, compute_stage_records(entries))
    write_csv(args.output, rows, fields)
    print(f"Saved {len(rows)} continuous stage occurrences to {args.output.resolve()}")


if __name__ == "__main__":
    main()
