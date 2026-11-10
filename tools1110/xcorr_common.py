"""Shared, row-aligned gaze/PSM position correlation. Requires numpy/scipy.

The computation scripts consume only a JSON list of matched source paths.
Positive lag k compares gaze[t] with PSM[t+k] (gaze leads).
Missing samples retain their original positions; no interpolation/resampling.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate

HZ = 60
WIDTH, HEIGHT = 1280, 1024
MAX_LAG_FRAMES = 120
MIN_PAIRS = 500
LIKELIHOOD_MIN = 0.2
GAZE_SMOOTH_POINTS = 3
DEFAULT_PAIRS = Path(__file__).resolve().parent / "matched_npy_dlc_paths.json"


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_path_list(path):
    """Read an explicit JSON list; never search the index or dataset folders."""
    path = Path(path).resolve()
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list):
        raise ValueError("Path list must be a JSON array of record objects")
    seen = set()
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each path-list entry must be an object")
        entry = entry.copy()
        for key in ("record_id", "task_name", "gaze_npy_path", "dlc_csv_path"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"Missing/non-string {key}: {entry}")
        if entry["record_id"] in seen:
            raise ValueError(f"Duplicate record_id: {entry['record_id']}")
        seen.add(entry["record_id"])
        for key in ("gaze_npy_path", "dlc_csv_path"):
            source = Path(entry[key])
            entry[key] = str((source if source.is_absolute() else path.parent / source).resolve())
        result.append(entry)
    return result


def load_gaze(path):
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    if data.ndim != 1 or not {"gaze_x", "gaze_y"}.issubset(data.dtype.names or ()):
        raise ValueError("Gaze NPY must be a 1-D structured array with gaze_x/gaze_y")
    # Do not use Frame_Number, timestamps, valid, stage or other annotations.
    return np.column_stack((data["gaze_x"], data["gaze_y"])).astype(float)


def load_dlc(path):
    """First x/y/likelihood triple is left; second is right."""
    values = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = [next(reader, None) for _ in range(3)]
        if any(row is None or len(row) != 7 for row in headers):
            raise ValueError("DLC CSV needs three 7-column header rows")
        if [r[0].strip() for r in headers] != ["scorer", "bodyparts", "coords"]:
            raise ValueError("Expected scorer/bodyparts/coords DLC headers")
        if [s.strip() for s in headers[2][1:]] != ["x", "y", "likelihood"] * 2:
            raise ValueError("Expected x,y,likelihood,x,y,likelihood DLC columns")
        bodyparts = [s.strip() for s in headers[1][1:]]
        if bodyparts != ["left_psm"] * 3 + ["right_psm"] * 3:
            raise ValueError("Expected left_psm triple followed by right_psm triple")
        for line, row in enumerate(reader, start=4):
            if not row:
                raise ValueError(f"Blank DLC data row at CSV line {line}")
            if len(row) != 7:
                raise ValueError(f"Expected 7 columns at CSV line {line}")
            # The first column is intentionally ignored: alignment is by row.
            values.append([float(s) if s.strip() else np.nan for s in row[1:]])
    data = np.asarray(values, dtype=float).reshape(-1, 6)
    return data[:, :2], data[:, 2], data[:, 3:5], data[:, 5]


def smooth_gaze(xy):
    """3-point centered mean within valid runs; endpoints use available points."""
    xy = np.asarray(xy, dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    # A point and its neighbors must be in the same valid run. Since the
    # window has radius one, testing adjacent validity is sufficient.
    total = np.where(valid[:, None], xy, 0.0)
    count = valid.astype(float)
    adjacent = valid[:-1] & valid[1:]
    total[1:] += np.where(adjacent[:, None], xy[:-1], 0.0)
    total[:-1] += np.where(adjacent[:, None], xy[1:], 0.0)
    count[1:] += adjacent
    count[:-1] += adjacent
    out = np.full_like(xy, np.nan)
    out[valid] = total[valid] / count[valid, None]
    return out


def filter_sources(gaze, left, left_likelihood, right, right_likelihood):
    gaze = np.asarray(gaze, dtype=float).copy()
    gaze_valid = np.isfinite(gaze).all(axis=1) & ((gaze > 0) & (gaze < 1)).all(axis=1)
    gaze[~gaze_valid] = np.nan
    gaze *= [WIDTH, HEIGHT]
    gaze = smooth_gaze(gaze)
    outputs = []
    for xy, likelihood in ((left, left_likelihood), (right, right_likelihood)):
        xy = np.asarray(xy, dtype=float).copy()
        valid = (np.isfinite(xy).all(axis=1) & np.isfinite(likelihood)
                 & (likelihood >= LIKELIHOOD_MIN)
                 & (xy[:, 0] > 0) & (xy[:, 0] < WIDTH)
                 & (xy[:, 1] > 0) & (xy[:, 1] < HEIGHT))
        xy[~valid] = np.nan
        outputs.append(xy)
    return gaze, outputs[0], outputs[1]


def correlation_curve(gaze, psm, max_lag_frames=MAX_LAG_FRAMES, min_pairs=MIN_PAIRS):
    """Per-lag, overlap-centered 2-D correlation, computed using FFT sums.

    r = sum((gx-mean(gx))*(px-mean(px)) + (gy-mean(gy))*(py-mean(py)))
        / sqrt(sum(||g-mean(g)||**2) * sum(||p-mean(p)||**2)).
    All means/sums use only valid pairs at the current lag. These are pixel
    coordinates, so x and y retain their physical pixel scales.
    """
    gaze, psm = np.asarray(gaze, dtype=float), np.asarray(psm, dtype=float)
    if gaze.ndim != 2 or gaze.shape[1:] != (2,) or psm.shape != gaze.shape:
        raise ValueError("Gaze and PSM must have identical (N, 2) shapes")
    lags = np.arange(-max_lag_frames, max_lag_frames + 1, dtype=int)
    r = np.full(len(lags), np.nan)
    count = np.zeros(len(lags), dtype=np.int64)
    n = len(gaze)
    if n == 0:
        return lags, r, count
    gm, pm = np.isfinite(gaze).all(axis=1), np.isfinite(psm).all(axis=1)
    if not gm.any() or not pm.any():
        return lags, r, count
    # Pre-center by constants to reduce FFT cancellation; the per-lag means
    # are still subtracted below, so the mathematical definition is unchanged.
    g = np.where(gm[:, None], gaze - gaze[gm].mean(axis=0), 0.0)
    p = np.where(pm[:, None], psm - psm[pm].mean(axis=0), 0.0)
    indices = n - 1 + lags
    inside = (indices >= 0) & (indices < 2 * n - 1)

    def cross(a, b):
        out = np.zeros(len(lags))
        out[inside] = correlate(a, b, mode="full", method="fft")[indices[inside]]
        return out

    gf, pf = gm.astype(float), pm.astype(float)
    count = np.rint(cross(pf, gf)).clip(0).astype(np.int64)
    divisor = np.maximum(count, 1)
    covariance, gv, pv = np.zeros((3, len(lags)))
    for axis in range(2):
        gs, ps = cross(pf, g[:, axis]), cross(p[:, axis], gf)
        covariance += cross(p[:, axis], g[:, axis]) - gs * ps / divisor
        gv += cross(pf, g[:, axis] ** 2) - gs ** 2 / divisor
        pv += cross(p[:, axis] ** 2, gf) - ps ** 2 / divisor
    gv, pv = np.maximum(gv, 0), np.maximum(pv, 0)
    # Reject variance indistinguishable from floating-point FFT roundoff.
    eps = 64 * np.finfo(float).eps
    gv_tol = eps * max(1.0, float(np.sum(g * g)))
    pv_tol = eps * max(1.0, float(np.sum(p * p)))
    usable = (count >= min_pairs) & (gv > gv_tol) & (pv > pv_tol)
    r[usable] = np.clip(covariance[usable] / np.sqrt(gv[usable] * pv[usable]), -1, 1)
    return lags, r, count


def peak_correlation(gaze, psm):
    lags, r, count = correlation_curve(gaze, psm)
    supported = np.flatnonzero(np.isfinite(r))
    if not len(supported):
        return {"status": "no_valid_peak", "peak_r": "", "peak_lag_frames": "",
                "peak_lag_seconds": "", "peak_valid_pairs": "", "peak_at_boundary": ""}
    # Maximum signed r. Exact ties: smallest |lag|, then negative before positive.
    best = min(supported, key=lambda i: (-r[i], abs(lags[i]), lags[i]))
    return {"status": "ok", "peak_r": float(r[best]), "peak_lag_frames": int(lags[best]),
            "peak_lag_seconds": float(lags[best] / HZ), "peak_valid_pairs": int(count[best]),
            "peak_at_boundary": bool(abs(lags[best]) == MAX_LAG_FRAMES)}


RECORD_FIELDS = [
    "record_id", "task_name", "psm_side", "gaze_npy_path", "dlc_csv_path",
    "gaze_rows", "dlc_rows", "duration_seconds", "gaze_valid_rows", "psm_valid_rows",
    "peak_r", "peak_lag_frames", "peak_lag_seconds", "peak_valid_pairs", "peak_at_boundary",
    "status", "error", "hz", "max_lag_seconds", "min_pairs", "likelihood_min",
    "gaze_smooth_points",
]


def compute_record(entry):
    """Return one row per side, including explicit source/computation failures."""
    base = {key: entry[key] for key in ("record_id", "task_name", "gaze_npy_path", "dlc_csv_path")}
    base.update({"gaze_rows": "", "dlc_rows": "", "duration_seconds": "",
                 "gaze_valid_rows": "", "psm_valid_rows": "", "peak_r": "",
                 "peak_lag_frames": "", "peak_lag_seconds": "", "peak_valid_pairs": "",
                 "peak_at_boundary": "", "status": "error", "error": "", "hz": HZ,
                 "max_lag_seconds": MAX_LAG_FRAMES / HZ, "min_pairs": MIN_PAIRS,
                 "likelihood_min": LIKELIHOOD_MIN, "gaze_smooth_points": GAZE_SMOOTH_POINTS})
    try:
        gaze = load_gaze(entry["gaze_npy_path"])
        base["gaze_rows"] = len(gaze)
        left, ll, right, rl = load_dlc(entry["dlc_csv_path"])
        base["dlc_rows"] = len(left)
        if len(gaze) != len(left):
            raise ValueError(f"Row mismatch: gaze={len(gaze)}, DLC={len(left)}; no truncation")
        if "n_rows" in entry and entry["n_rows"] != len(gaze):
            raise ValueError(f"Stale path list: n_rows={entry['n_rows']}, current={len(gaze)}")
        base["duration_seconds"] = len(gaze) / HZ
        gaze, left, right = filter_sources(gaze, left, ll, right, rl)
        base["gaze_valid_rows"] = int(np.isfinite(gaze).all(axis=1).sum())
        rows = []
        for side, psm in (("left", left), ("right", right)):
            row = dict(base, psm_side=side, psm_valid_rows=int(np.isfinite(psm).all(axis=1).sum()))
            row.update(peak_correlation(gaze, psm))
            if row["status"] != "ok":
                row["error"] = "No lag has >=500 valid pairs and nonzero 2-D variance in both signals"
            rows.append(row)
        return rows
    except (OSError, ValueError, TypeError, EOFError) as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return [dict(base, psm_side=side) for side in ("left", "right")]


def compute_records(entries):
    rows = []
    for i, entry in enumerate(entries, start=1):
        result = compute_record(entry)
        rows.extend(result)
        print(f"[{i}/{len(entries)}] {entry['record_id']}: "
              + ", ".join(f"{r['psm_side']}={r['status']}" for r in result), flush=True)
    return rows
