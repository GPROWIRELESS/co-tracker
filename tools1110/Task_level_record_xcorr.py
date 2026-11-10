"""Compute Task A/C summaries directly from the matched source-path list.

Uses the same record computation as record_level_xcorr.py, without reading
its CSV or searching the source index. Weight = original row count / 60.
r = tanh(sum(duration * atanh(r)) / sum(duration)).
lag = sum(duration * peak_lag_seconds) / sum(duration).
Only successful records contribute; left/right are aggregated separately.
"""
import argparse
from pathlib import Path

import numpy as np

from xcorr_common import (
    DEFAULT_PAIRS, HZ, LIKELIHOOD_MIN, GAZE_SMOOTH_POINTS, MIN_PAIRS,
    MAX_LAG_FRAMES, compute_records, read_path_list, write_csv,
)

TASK_FIELDS = [
    "task_name", "psm_side", "records_total", "records_included", "records_excluded",
    "total_duration_seconds", "included_duration_seconds", "fisher_z_weighted_r",
    "duration_weighted_lag_seconds", "duration_weighted_lag_frames", "fisher_r_clip_count",
    "included_record_ids", "excluded_record_ids", "status", "hz", "max_lag_seconds",
    "min_pairs", "likelihood_min", "gaze_smooth_points",
]


def aggregate_tasks(record_rows):
    results = []
    for task in ("A", "C"):
        for side in ("left", "right"):
            group = [r for r in record_rows if r["task_name"] == task and r["psm_side"] == side]
            included = [r for r in group if r["status"] == "ok"]
            excluded = [r for r in group if r["status"] != "ok"]
            result = dict(
                task_name=task, psm_side=side, records_total=len(group),
                records_included=len(included), records_excluded=len(excluded),
                total_duration_seconds=sum(float(r["duration_seconds"]) for r in group
                                           if r["duration_seconds"] != ""),
                included_duration_seconds=0.0, fisher_z_weighted_r="",
                duration_weighted_lag_seconds="", duration_weighted_lag_frames="",
                fisher_r_clip_count=0,
                included_record_ids=";".join(r["record_id"] for r in included),
                excluded_record_ids=";".join(r["record_id"] for r in excluded),
                status="no_valid_records", hz=HZ, max_lag_seconds=MAX_LAG_FRAMES / HZ,
                min_pairs=MIN_PAIRS, likelihood_min=LIKELIHOOD_MIN,
                gaze_smooth_points=GAZE_SMOOTH_POINTS,
            )
            if included:
                weights = np.array([r["duration_seconds"] for r in included], dtype=float)
                correlations = np.array([r["peak_r"] for r in included], dtype=float)
                lags = np.array([r["peak_lag_seconds"] for r in included], dtype=float)
                # atanh(+-1) is infinite: clip only for Fisher transformation.
                limit = 1 - 1e-12
                clipped = np.clip(correlations, -limit, limit)
                result.update(
                    included_duration_seconds=float(weights.sum()),
                    fisher_z_weighted_r=float(np.tanh(np.average(np.arctanh(clipped), weights=weights))),
                    duration_weighted_lag_seconds=float(np.average(lags, weights=weights)),
                    duration_weighted_lag_frames=float(np.average(lags, weights=weights) * HZ),
                    fisher_r_clip_count=int(np.sum(clipped != correlations)), status="ok",
                )
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "Task_level_record_xcorr.csv")
    args = parser.parse_args()
    entries = [e for e in read_path_list(args.pairs) if e["task_name"] in {"A", "C"}]
    rows = aggregate_tasks(compute_records(entries))
    write_csv(args.output, rows, TASK_FIELDS)
    print(f"Saved Task A/C left/right summaries to {args.output.resolve()}")


if __name__ == "__main__":
    main()
