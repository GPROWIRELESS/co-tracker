"""CSV and figure exports for the gaze/PSM cross-correlation pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


META = (
    "content_id", "user", "task", "side", "signal", "stage", "mode",
    "n_stable_users", "n_fixed_record_users", "n_stable_records", "record_support_changes",
)
LAG_FIELDS = (
    "lag_s", "r", "r_available", "r_ci_low", "r_ci_high", "null_r_low",
    "null_r_high", "pairs", "occurrences", "gaze_rms", "psm_rms",
    "distance_median_px", "proximity", "n_records", "n_users",
    "r_fixed_records", "r_primary_fixed_record_users",
)


def _value(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), allow_nan=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, allow_nan=True)
    return value


def _write(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _value(value) for key, value in row.items()} for row in rows)


def _long_rows(curves):
    for curve in curves:
        lags = np.asarray(curve.get("lag_s", []))
        arrays = {key: np.asarray(curve[key]) for key in LAG_FIELDS
                  if key in curve and np.ndim(curve[key]) == 1
                  and len(curve[key]) == len(lags)}
        metadata = {key: curve[key] for key in META if key in curve}
        for index in range(len(lags)):
            yield {**metadata, **{key: value[index] for key, value in arrays.items()}}


def _summary(curve):
    # Preserve explicit group/user estimates; derive peaks for record curves.
    result = {key: value for key, value in curve.items()
              if not isinstance(value, np.ndarray) and key not in LAG_FIELDS}
    if "peak_lag_s" not in result:
        r = np.asarray(curve.get("r", []), dtype=float)
        lags = np.asarray(curve.get("lag_s", []), dtype=float)
        if np.isfinite(r).any():
            index = int(np.nanargmax(r))
            result.update(peak_lag_s=lags[index], peak_r=r[index],
                          peak_at_boundary=index in (0, len(r)-1),
                          status="boundary_peak" if index in (0, len(r)-1) else "ok")
            for source, destination in (("pairs", "pairs_at_peak"),
                                        ("occurrences", "occurrences_at_peak")):
                if source in curve:
                    result[destination] = curve[source][index]
        else:
            result.update(peak_lag_s=np.nan, peak_r=np.nan,
                          peak_at_boundary=False, status="insufficient_support_or_variation")
    return result


# Plotting is isolated from numerical CSV export; preserve the historical
# private entry points for callers that imported report helpers.
from tools.gaze_psm_figures import (
    _matches, _finite, _task_order, _decorate, _curve_panel, _save,
    _whole_figures, _spatial_figures, _stage_figures, _record_figures,
    _paired_figures, _bootstrap_figures, generate_figures,
)


def generate_outputs(output, record_curves, user_curves, group_curves, contrasts, qc, make_plots=True):
    """Write auditable CSVs and optional figures, including unsupported results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for label, curves in (("record", record_curves), ("novice", user_curves), ("group", group_curves)):
        _write(output/f"{label}_curves.csv", _long_rows(curves),
               [key for key in META if any(key in c for c in curves)] +
               [key for key in LAG_FIELDS if any(key in c for c in curves)])
        _write(output/f"{label}_summary.csv", (_summary(c) for c in curves))
    _write(output/"contrasts.csv", ({k: v for k, v in c.items() if k != "bootstrap_difference_s"} for c in contrasts))
    _write(output/"quality.csv", qc)
    bootstrap_rows = []
    for group in group_curves:
        meta = {key: group[key] for key in META if key in group}
        for index, value in enumerate(group.get("bootstrap_peak_lag_s", [])):
            bootstrap_rows.append({**meta, "replicate": index, "peak_lag_s": value})
    _write(output/"bootstrap_peaks.csv", bootstrap_rows)
    null_rows = []
    for group in group_curves:
        meta = {key: group[key] for key in META if key in group}
        for index, value in enumerate(group.get("null_peak_r", [])):
            null_rows.append({**meta, "replicate": index, "peak_r": value})
    _write(output/"unpaired_peaks.csv", null_rows)
    contrast_rows = []
    for contrast in contrasts:
        meta = {key: value for key, value in contrast.items()
                if key not in ("bootstrap_difference_s", "stable_users") and not isinstance(value, np.ndarray)}
        for index, value in enumerate(contrast.get("bootstrap_difference_s", [])):
            contrast_rows.append({**meta, "replicate": index, "difference_s": value})
    _write(output/"bootstrap_contrasts.csv", contrast_rows)
    if make_plots:
        generate_figures(output, record_curves, user_curves, group_curves, contrasts)
