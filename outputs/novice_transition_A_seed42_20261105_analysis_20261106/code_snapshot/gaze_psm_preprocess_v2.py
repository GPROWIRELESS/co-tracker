"""Signal preparation for the fixed-side gaze/PSM v2 analysis.

DLC policy, set by the user's visual inspection: ALL finite coordinates are
retained. Likelihood, out-of-frame coordinates, and jumps are diagnostics only;
none of them causes automatic deletion or confidence weighting. No quality-hole
filling is performed because no quality rejection is performed. Ordinary finite
sample interpolation remains limited to ``max_gap`` and never extrapolates.

Cached gaze_raw_* are already on the 41 Hz analysis grid, not original high-rate
eye-tracker data. This module preserves the upstream eye-tracker input policy.
Gaze masks are common across sources, but remain separate from each side's PSM
mask: shifted pairs must be formed in the correlation kernel, not here.
"""

from __future__ import annotations

import json
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


def finite_rows(xy):
    return np.isfinite(xy).all(axis=1)


def interpolate_valid(times, xy, grid, max_gap=.1):
    """Legacy-compatible interpolation of finite 2D rows, without extrapolation."""
    times = np.asarray(times, dtype=float)
    xy = np.asarray(xy, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if xy.shape != (len(times), 2):
        raise ValueError("xy must have shape (len(times), 2)")
    if not np.isfinite(max_gap) or max_gap < 0:
        raise ValueError("max_gap must be finite and nonnegative")
    out = np.full((len(grid), 2), np.nan)
    valid = finite_rows(xy) & np.isfinite(times)
    t, a = times[valid], xy[valid]
    if len(t) < 2:
        return out
    order = np.argsort(t, kind="stable")
    t, a = t[order], a[order]
    t, ix = np.unique(t, return_index=True)
    a = a[ix]
    if len(t) < 2:
        return out
    right = np.searchsorted(t, grid, side="left").clip(0, len(t)-1)
    exact = np.isclose(grid, t[right], atol=1e-9, rtol=0)
    out[exact] = a[right[exact]]
    left = np.maximum(right-1, 0)
    good = (~exact) & (grid >= t[0]) & (grid <= t[-1]) & ((t[right]-t[left]) <= max_gap)
    for axis in range(2):
        out[good, axis] = np.interp(grid[good], t, a[:, axis])
    return out


def filter_valid(xy, window=3, method="boxcar", trim_edges=False):
    """Centered filter within finite runs; return values and removed-edge mask.

    Both filters use nearest extension only within their own finite run. The
    optional trimming sensitivity removes half a window at both ends of each
    run, so no retained sample then depends on padded values. Short runs can
    disappear entirely under trimming. Filtering never spans a NaN gap.
    """
    window = int(window)
    if window < 1 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer")
    if method not in ("boxcar", "savgol"):
        raise ValueError("method must be boxcar or savgol")
    if method == "savgol" and window < 3:
        raise ValueError("quadratic Savitzky-Golay requires window >= 3")
    a = np.asarray(xy, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("xy must be an N by 2 array")
    out = a.copy()
    good = finite_rows(a)
    out[~good] = np.nan
    edges = np.zeros(len(a), dtype=bool)
    cuts = np.flatnonzero(np.diff(np.r_[False, good, False]))
    for start, stop in cuts.reshape(-1, 2):
        if window > 1:
            if method == "boxcar":
                out[start:stop] = uniform_filter1d(a[start:stop], size=window, axis=0, mode="nearest")
            else:
                out[start:stop] = savgol_filter(a[start:stop], window, 2, axis=0, mode="nearest")
        if trim_edges and window > 1:
            half = window // 2
            edges[start:min(start+half, stop)] = True
            edges[max(start, stop-half):stop] = True
    out[edges] = np.nan
    return out, edges


def _common_gazes(caches, side):
    arrays = {s: np.asarray(z[f"gaze_raw_{side}"], dtype=float).copy() for s, z in caches.items()}
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all gaze sources must have identical shapes")
    mask = np.logical_and.reduce([finite_rows(a) for a in arrays.values()])
    for a in arrays.values():
        a[~mask] = np.nan
    return arrays, mask


def _filter_gazes(gazes, window, method, trim_edges):
    results = {s: filter_valid(a, window, method, trim_edges) for s, a in gazes.items()}
    mask = np.logical_and.reduce([finite_rows(a) for a, _ in results.values()])
    arrays = {}
    for s, (a, _) in results.items():
        a[~mask] = np.nan
        arrays[s] = a
    return arrays, mask, {s: edge for s, (_, edge) in results.items()}


def _velocity_labels(base, grid, max_gap):
    midpoint = (grid[:-1] + grid[1:]) / 2
    native_t = np.asarray(base["native_time_s"], dtype=float)
    index = np.searchsorted(native_t, midpoint, side="right") - 1
    index = index.clip(0, len(native_t)-1)
    labels = np.asarray(base["native_stage"]).astype("U128")[index].copy()
    ids = np.asarray(base["native_occurrence"], dtype=np.int64)[index].copy()
    stale = (midpoint-native_t[index] > max_gap) | (midpoint < native_t[0])
    labels[stale], ids[stale] = "none", -1
    return midpoint, labels, ids


def prepare_signals(caches, config):
    """Prepare common-support gaze and separate fixed-side PSM signals.

    Parameters use sample counts: position_window=3, velocity_window=5,
    velocity_filter='savgol', trim_edges=False, hz=41, max_gap=.1. For exact
    legacy smoothing, set velocity_filter='boxcar', velocity_window=3.

    Returns signals[signal][side] with gazes[source], psm, stage, occurrence,
    time_s and independent gaze_valid/psm_valid masks; scalar qc/audit rows;
    diagnostics[side] contains native QC flags (never deletion instructions).
    """
    if "gt" not in caches or not caches:
        raise ValueError("caches must contain gt")
    base = caches["gt"]
    hz, max_gap = float(config.get("hz", 41.)), float(config.get("max_gap", .1))
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be positive")
    grid = np.asarray(base["time_s"], dtype=float)
    if len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("time_s must be increasing with at least two samples")
    if not np.allclose(np.diff(grid), 1/hz, rtol=1e-7, atol=1e-9):
        raise ValueError("time_s is not on the requested uniform grid")
    native_t = np.asarray(base["native_time_s"], dtype=float)
    if not len(native_t) or not np.isfinite(native_t).all() or not np.all(np.diff(native_t) > 0):
        raise ValueError("native_time_s must be finite and strictly increasing")
    meta = json.loads(str(base["qc_json"])) if "qc_json" in base else {}
    identity = {k: meta.get(k, "") for k in ("content_id", "user", "task")}
    pw = int(config.get("position_window", 3))
    vw = int(config.get("velocity_window", 5))
    vf = config.get("velocity_filter", "savgol")
    trim = bool(config.get("trim_edges", False))
    signals = {"position": {}, "velocity": {}}
    qc, audit, diagnostics = [], [], {}
    for source, z in caches.items():
        for field in ("time_s", "native_time_s", "frame_number", "stage", "occurrence", "native_stage", "native_occurrence"):
            if field in base and field in z and not np.array_equal(base[field], z[field]):
                raise ValueError(f"source {source} has mismatched {field}")
        audit.append({**identity, "source": source, "check": "cached_time_frame_stage_match", "passed": True,
                      "gaze_input": "cached_41hz_interpolated_gaze_raw_not_high_rate_raw"})
    midpoint, vel_stage, vel_ids = _velocity_labels(base, grid, max_gap)
    for side in ("l", "r"):
        native = np.asarray(base[f"native_psm_{side}"], dtype=float)
        lik = np.asarray(base[f"likelihood_{side}"], dtype=float)
        psm_raw = interpolate_valid(native_t, native, grid, max_gap)
        cached_psm = np.asarray(base[f"psm_raw_{side}"], dtype=float)
        same_mask = np.array_equal(finite_rows(psm_raw), finite_rows(cached_psm))
        same_values = bool(np.allclose(psm_raw, cached_psm, atol=1e-10, rtol=1e-10, equal_nan=True))
        audit.append({**identity, "source": "psm", "side": side, "check": "native_reconstruction_matches_cached_psm",
                      "passed": same_mask and same_values, "max_gap_s": max_gap})
        if abs(max_gap-.1) < 1e-12 and not (same_mask and same_values):
            raise ValueError(f"native PSM reconstruction disagrees with original cache: {identity}/{side}")
        resolution = np.asarray(base.get(f"resolution_{side}", [np.inf, np.inf]), dtype=float)
        native_good = finite_rows(native)
        outside = native_good & ((native < 0).any(axis=1) | (native > resolution).any(axis=1))
        displacement = np.linalg.norm(np.diff(native, axis=0), axis=1)
        jump = np.r_[False, displacement > 100.]
        native_speed = displacement / np.diff(native_t)
        diagnostics[side] = {"low_likelihood": np.isfinite(lik) & (lik < .5), "jump_over_100px": jump,
                             "outside_frame": outside, "native_nonfinite": ~native_good,
                             "quality_rejected": np.zeros(len(native), dtype=bool)}
        gazes, common = _common_gazes(caches, side)
        for signal, window, method in (("position", pw, "boxcar"), ("velocity", vw, vf)):
            gs, gmask, gedges = _filter_gazes(gazes, window, method, trim)
            ps, pedge = filter_valid(psm_raw, window, method, trim)
            if signal == "velocity":
                gs = {s: np.diff(a, axis=0)*hz for s, a in gs.items()}
                ps = np.diff(ps, axis=0)*hz
                gmask = np.logical_and.reduce([finite_rows(a) for a in gs.values()])
                stage, ids, times = vel_stage, vel_ids, midpoint
            else:
                stage, ids, times = np.asarray(base["stage"]), np.asarray(base["occurrence"]), grid
            for a in gs.values():
                a[~gmask] = np.nan
            pmask = finite_rows(ps)
            signals[signal][side] = {"gazes": gs, "psm": ps, "stage": stage.copy(),
                                     "occurrence": ids.copy(), "time_s": times.copy(),
                                     "gaze_valid": gmask.copy(), "psm_valid": pmask.copy()}
            common_info = {**identity, "side": side, "signal": signal, "filter": method,
                           "window_samples": window, "trim_edges": trim, "n_samples": len(ps)}
            qc.append({**common_info, "source": "psm", "valid_samples": int(pmask.sum()),
                       "valid_fraction": float(pmask.mean()), "filter_edge_points": int(pedge.sum()),
                       "native_samples": len(native), "native_nonfinite": int((~native_good).sum()),
                       "low_likelihood_count": int(diagnostics[side]["low_likelihood"].sum()),
                       "low_likelihood_fraction": float(diagnostics[side]["low_likelihood"].mean()),
                       "jump_over_100px_count": int(jump.sum()), "outside_frame_count": int(outside.sum()),
                       "native_speed_p99_px_s": float(np.nanpercentile(native_speed, 99)) if np.isfinite(native_speed).any() else np.nan,
                       "quality_rejected_count": 0, "likelihood_policy": "QC_only_no_filter_no_weighting",
                       "raw_grid_nonfinite": int((~finite_rows(psm_raw)).sum())})
            for source, a in gs.items():
                qc.append({**common_info, "source": source, "valid_samples": int(finite_rows(a).sum()),
                           "valid_fraction": float(finite_rows(a).mean()), "filter_edge_points": int(gedges[source].sum()),
                           "common_gaze_masked_raw_points": int((finite_rows(caches[source][f"gaze_raw_{side}"]) & ~common).sum()),
                           "quality_rejected_count": 0, "likelihood_policy": "not_applicable"})
    return {"signals": signals, "qc": qc, "audit": audit, "diagnostics": diagnostics}
