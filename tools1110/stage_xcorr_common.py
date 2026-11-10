"""Stage occurrence correlation and duration-weighted record/stage summaries.

Uses the existing record preprocessing: strict screen interior, likelihood
>=0.2, gaze-only 3-point smoothing, no interpolation or time compression.
The old record analysis keeps its 500-pair threshold; stage analysis has
no duration threshold. Two pairs and nonzero variance are mathematically
necessary for a centered correlation. Lag +k pairs gaze[t-k] with PSM[t].
"""
from collections import Counter, defaultdict
import json

import numpy as np

from record_level_xcorr import dataset_relative_path
from xcorr_common import (
    HZ, MAX_LAG_FRAMES, LIKELIHOOD_MIN, GAZE_SMOOTH_POINTS,
    correlation_curve, filter_sources, load_dlc, load_gaze,
)

STAGES = (
    "grab ring", "transfer ring", "place ring", "reach ring",
    "reach string", "wrap around psm", "pull through and cinch",
    "threading string through tower",
)
ALIASES = {"transport ring": "transfer ring"}
IGNORED_LABELS = {"", "none", "drop ring"}
SIDES = ("l_psm", "r_psm")
PEAK_FIELDS = ("status", "error", "peak_r", "peak_lag_frames", "peak_lag_seconds",
               "peak_valid_pairs", "peak_at_boundary")
SETTINGS = dict(hz=HZ, max_lag_seconds=MAX_LAG_FRAMES/HZ,
                gaze_padding_frames=MAX_LAG_FRAMES, min_pairs_mathematical=2,
                likelihood_min=LIKELIHOOD_MIN, gaze_smooth_points=GAZE_SMOOTH_POINTS)
STAGE_FIELDS = [
    "record_id", "task_name", "stage", "occurrence_index", "start_row", "end_row_exclusive",
    "stage_rows", "duration_seconds", "record_rows", "gaze_window_start_row",
    "gaze_window_end_row_exclusive", "gaze_npy_path", "dlc_csv_path",
] + [f"{side}_{key}" for side in SIDES for key in PEAK_FIELDS] + list(SETTINGS)
SUMMARY_METRICS = (
    "status", "occurrences_included", "occurrences_excluded", "included_duration_seconds",
    "duration_weighted_r", "fisher_z_weighted_r", "duration_weighted_lag_seconds",
    "duration_weighted_lag_frames", "fisher_r_clip_count", "boundary_peak_count",
)
SUMMARY_FIELDS = [
    "record_id", "task_name", "stage", "occurrences_total", "total_stage_rows",
    "total_duration_seconds", "record_rows", "gaze_npy_path", "dlc_csv_path",
] + [f"{side}_{key}" for side in SIDES for key in SUMMARY_METRICS] + list(SETTINGS)
AUDIT_FIELDS = ["raw_stage", "normalized_stage", "canonical_stage", "handling",
                "record_count", "row_count", "continuous_runs"]


def normalize_label(label):
    return " ".join(str(label).lower().split())


def canonical_stage(label):
    label = normalize_label(label)
    label = ALIASES.get(label, label)
    return label if label in STAGES else ""


def load_stage_labels(path):
    source = np.load(path, allow_pickle=False, mmap_mode="r")
    if source.ndim != 1 or "stage" not in (source.dtype.names or ()):
        raise ValueError(f"Gaze NPY needs a 1-D structured stage field: {path}")
    return np.asarray(source["stage"]).astype(str)


def label_audit(entries):
    """List exact labels, and return blockers for new/missing labels.

    Known ignored labels are explicit. No automatic grab/reach substitution:
    reach string is the user-confirmed stage; a new grab string is a blocker.
    """
    stats = defaultdict(lambda: dict(records=set(), rows=0, runs=0))
    for entry in entries:
        labels = load_stage_labels(entry["gaze_npy_path"])
        for label, count in Counter(labels).items():
            stats[str(label)]["records"].add(entry["record_id"])
            stats[str(label)]["rows"] += count
        for i in np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1] if len(labels) else []:
            stats[str(labels[i])]["runs"] += 1
    rows, unexpected, found = [], [], set()
    for raw, stat in sorted(stats.items()):
        normalized, stage = normalize_label(raw), canonical_stage(raw)
        handling = "include" if stage else "ignore" if normalized in IGNORED_LABELS else "unrecognized"
        if handling == "unrecognized":
            unexpected.append(raw)
        if stage:
            found.add(stage)
        rows.append(dict(raw_stage=raw, normalized_stage=normalized, canonical_stage=stage,
                         handling=handling, record_count=len(stat["records"]),
                         row_count=stat["rows"], continuous_runs=stat["runs"]))
    return rows, unexpected, sorted(set(STAGES)-found)


def stage_runs(labels):
    """Yield (canonical stage, occurrence number, start, end-exclusive).

    Run boundaries are defined before filtering, so invalid samples do not
    split a stage, and ignored stages still separate repeated occurrences.
    Adjacent transfer/transport aliases belong to the same continuous run.
    """
    labels = np.asarray([canonical_stage(label) for label in labels])
    if not len(labels):
        return
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1]
    ends = np.r_[starts[1:], len(labels)]
    counts = Counter()
    for start, end in zip(starts, ends):
        stage = str(labels[start])
        if stage:
            counts[stage] += 1
            yield stage, counts[stage], int(start), int(end)


def occurrence_curve(gaze, psm, start, end):
    """PSM only inside [start,end); gaze may be outside this occurrence.

    Both arrays retain source row positions. Their common window is clipped
    to the record, never padded or wrapped. Only PSM outside the occurrence
    is masked; gaze has no stage-membership requirement at shifted times.
    """
    if not 0 <= start < end <= len(gaze) or psm.shape != gaze.shape:
        raise ValueError("Invalid occurrence bounds or unmatched array shapes")
    low, high = max(0, start-MAX_LAG_FRAMES), min(len(gaze), end+MAX_LAG_FRAMES)
    masked_psm = np.full((high-low, 2), np.nan)
    masked_psm[start-low:end-low] = psm[start:end]
    lags, correlations, pairs = correlation_curve(
        gaze[low:high], masked_psm, max_lag_frames=MAX_LAG_FRAMES, min_pairs=2)
    # With very few pairs, subtracting FFT sufficient statistics can lose
    # precision. Recalculate those overlaps by directly centering the data.
    # This is a numerical safeguard, not a minimum-support exclusion rule.
    for index in np.flatnonzero((pairs >= 2) & (pairs <= 16)):
        p_indices = np.arange(start, end)
        g_indices = p_indices-lags[index]
        inside = (g_indices >= 0) & (g_indices < len(gaze))
        g, p = gaze[g_indices[inside]], psm[p_indices[inside]]
        valid = np.isfinite(g).all(axis=1) & np.isfinite(p).all(axis=1)
        g, p = g[valid], p[valid]
        g, p = g-g.mean(axis=0), p-p.mean(axis=0)
        denominator = np.sqrt(np.sum(g*g)*np.sum(p*p))
        correlations[index] = (np.clip(np.sum(g*p)/denominator, -1, 1)
                               if denominator > 0 else np.nan)
    return lags, correlations, pairs


def occurrence_peak(gaze, psm, start, end):
    lags, correlations, pairs = occurrence_curve(gaze, psm, start, end)
    supported = np.flatnonzero(np.isfinite(correlations))
    if not len(supported):
        return dict(status="no_valid_peak", error="No lag has >=2 valid pairs and nonzero variance",
                    peak_r="", peak_lag_frames="", peak_lag_seconds="", peak_valid_pairs="",
                    peak_at_boundary="")
    best = min(supported, key=lambda i: (-correlations[i], abs(lags[i]), lags[i]))
    return dict(status="ok", error="", peak_r=float(correlations[best]),
                peak_lag_frames=int(lags[best]), peak_lag_seconds=float(lags[best]/HZ),
                peak_valid_pairs=int(pairs[best]), peak_at_boundary=bool(abs(lags[best]) == MAX_LAG_FRAMES))


def compute_stage_record(entry):
    labels = load_stage_labels(entry["gaze_npy_path"])
    runs = list(stage_runs(labels))
    base = {key: entry[key] for key in ("record_id", "task_name", "gaze_npy_path", "dlc_csv_path")}
    base.update(SETTINGS, record_rows=len(labels))
    source_error = ""
    try:
        gaze = load_gaze(entry["gaze_npy_path"])
        left, ll, right, rl = load_dlc(entry["dlc_csv_path"])
        if len(gaze) != len(left):
            raise ValueError(f"Row mismatch: gaze={len(gaze)}, DLC={len(left)}; no truncation")
        if "n_rows" in entry and entry["n_rows"] != len(gaze):
            raise ValueError(f"Stale path list: n_rows={entry['n_rows']}, current={len(gaze)}")
        gaze, left, right = filter_sources(gaze, left, ll, right, rl)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        source_error = f"{type(exc).__name__}: {exc}"
    rows = []
    for stage, occurrence, start, end in runs:
        row = dict(base, stage=stage, occurrence_index=occurrence, start_row=start,
                   end_row_exclusive=end, stage_rows=end-start, duration_seconds=(end-start)/HZ,
                   gaze_window_start_row=max(0, start-MAX_LAG_FRAMES),
                   gaze_window_end_row_exclusive=min(len(labels), end+MAX_LAG_FRAMES))
        for side in SIDES:
            if source_error:
                peak = dict.fromkeys(PEAK_FIELDS, "")
                peak.update(status="source_error", error=source_error)
            else:
                peak = occurrence_peak(gaze, left if side == "l_psm" else right, start, end)
            row.update({f"{side}_{key}": value for key, value in peak.items()})
        rows.append(row)
    return rows


def compute_stage_records(entries):
    rows = []
    for index, entry in enumerate(entries, 1):
        current = compute_stage_record(entry)
        rows.extend(current)
        valid = sum(row[f"{side}_status"] == "ok" for row in current for side in SIDES)
        print(f"[{index}/{len(entries)}] {entry['record_id']}: occurrences={len(current)}, "
              f"valid sides={valid}/{2*len(current)}", flush=True)
    return rows


def format_stage_output(entries, rows):
    by_id = {entry["record_id"]: entry for entry in entries}
    columns = {}
    for entry in entries:
        meta = entry.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"meta must be an object: {entry['record_id']}")
        if "task_name" in meta and meta["task_name"] != entry["task_name"]:
            raise ValueError(f"Conflicting meta task_name: {entry['record_id']}")
        for key in meta:
            column = ("meta_"+key if key in set(STAGE_FIELDS+SUMMARY_FIELDS) and key != "task_name" else key)
            columns[key] = column
    if len(set(columns.values())) != len(columns):
        raise ValueError("Metadata column names collide")
    fields = STAGE_FIELDS + [column for column in columns.values() if column not in STAGE_FIELDS]
    output = []
    for row in rows:
        row = row.copy()
        for key in ("gaze_npy_path", "dlc_csv_path"):
            row[key] = dataset_relative_path(row[key])
        for key, value in by_id[row["record_id"]].get("meta", {}).items():
            row[columns[key]] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        output.append(row)
    return output, fields


def aggregate_stage_rows(rows, input_fields):
    """Aggregate occurrence peaks, independently by side, preserving metadata.

    Duration weight is the ORIGINAL occurrence length / 60, regardless of
    filtering or lag. Unsupported occurrences are excluded per side. Fisher
    z is applied only to r, and correlations are clipped at 1-1e-12 solely
    for that transform. Lag is an arithmetic duration-weighted mean.
    """
    missing = set(STAGE_FIELDS)-set(input_fields)
    if missing:
        raise ValueError(f"Input occurrence CSV missing columns: {sorted(missing)}")
    meta_fields = [key for key in input_fields if key not in STAGE_FIELDS]
    fields = SUMMARY_FIELDS + meta_fields
    groups = defaultdict(list)
    seen = set()
    for source in rows:
        row = source.copy()
        key = (row["record_id"], row["stage"])
        if row["stage"] not in STAGES:
            raise ValueError(f"Unrecognized input stage: {row['stage']}")
        identity = (*key, int(row["occurrence_index"]))
        if identity in seen:
            raise ValueError(f"Duplicate occurrence: {identity}")
        seen.add(identity)
        nrows, duration = int(row["stage_rows"]), float(row["duration_seconds"])
        if nrows <= 0 or not np.isclose(duration, nrows/HZ, rtol=0, atol=1e-10):
            raise ValueError(f"Invalid original occurrence duration: {identity}")
        row["duration_seconds"] = nrows/HZ
        row["stage_rows"] = nrows
        for setting, expected in SETTINGS.items():
            if float(row[setting]) != expected:
                raise ValueError(f"Unexpected {setting} for {identity}")
        groups[key].append(row)
    results = []
    for (record_id, stage), group in groups.items():
        first = group[0]
        invariants = ["task_name", "record_rows", "gaze_npy_path", "dlc_csv_path"] + meta_fields
        if any(row[key] != first[key] for row in group for key in invariants):
            raise ValueError(f"Inconsistent source/metadata for {record_id}, {stage}")
        summary = {key: first[key] for key in invariants}
        summary.update(SETTINGS, record_id=record_id, stage=stage, occurrences_total=len(group),
                       total_stage_rows=sum(r["stage_rows"] for r in group),
                       total_duration_seconds=sum(r["duration_seconds"] for r in group))
        for side in SIDES:
            good = [r for r in group if r[f"{side}_status"] == "ok"]
            metrics = dict(status="no_valid_occurrences", occurrences_included=len(good),
                           occurrences_excluded=len(group)-len(good), included_duration_seconds=0.0,
                           duration_weighted_r="", fisher_z_weighted_r="",
                           duration_weighted_lag_seconds="", duration_weighted_lag_frames="",
                           fisher_r_clip_count=0, boundary_peak_count=0)
            if good:
                weights = np.array([r["duration_seconds"] for r in good])
                correlations = np.array([r[f"{side}_peak_r"] for r in good], dtype=float)
                lags = np.array([r[f"{side}_peak_lag_seconds"] for r in good], dtype=float)
                frames = np.array([r[f"{side}_peak_lag_frames"] for r in good], dtype=float)
                pairs = np.array([r[f"{side}_peak_valid_pairs"] for r in good], dtype=int)
                if (not np.isfinite(correlations).all() or np.any(np.abs(correlations) > 1)
                        or not np.isfinite(lags).all() or np.any(np.abs(lags) > MAX_LAG_FRAMES/HZ)
                        or not np.allclose(lags*HZ, frames, rtol=0, atol=1e-10)
                        or np.any(pairs < 2)):
                    raise ValueError(f"Invalid peak values for {record_id}, {stage}, {side}")
                clipped = np.clip(correlations, -1+1e-12, 1-1e-12)
                lag = float(np.average(lags, weights=weights))
                metrics.update(status="ok", included_duration_seconds=float(weights.sum()),
                               duration_weighted_r=float(np.average(correlations, weights=weights)),
                               fisher_z_weighted_r=float(np.tanh(np.average(np.arctanh(clipped), weights=weights))),
                               duration_weighted_lag_seconds=lag, duration_weighted_lag_frames=lag*HZ,
                               fisher_r_clip_count=int(np.sum(clipped != correlations)),
                               boundary_peak_count=int(np.sum(np.abs(frames) == MAX_LAG_FRAMES)))
            summary.update({f"{side}_{key}": value for key, value in metrics.items()})
        results.append(summary)
    return results, fields
