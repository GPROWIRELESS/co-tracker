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


def interpolation_provenance(times, xy, grid, max_gap=.1):
    """Describe native-to-grid interpolation without changing the legacy values.

    Every finite grid point is either an exact native observation, ordinary
    between-frame resampling, or interpolation across nonfinite native rows.
    No QC rejection occurs, so quality-hole filling is identically false.
    """
    times, xy, grid = np.asarray(times), np.asarray(xy), np.asarray(grid)
    flags = {name: np.zeros(len(grid), dtype=bool) for name in
             ('observed_exact', 'ordinary_resample', 'filled_nonfinite_hole',
              'filled_quality_hole', 'long_gap', 'outside_retained_support',
              'no_retained_support')}
    native_indices = np.flatnonzero(finite_rows(xy) & np.isfinite(times))
    if len(native_indices) < 2:
        flags['no_retained_support'][:] = True
        return flags
    t = times[native_indices]
    right = np.searchsorted(t, grid, side='left').clip(0, len(t)-1)
    left = np.maximum(right-1, 0)
    exact = np.isclose(grid, t[right], atol=1e-9, rtol=0)
    inside = (grid >= t[0]) & (grid <= t[-1])
    gap = t[right]-t[left]
    interpolated = (~exact) & inside & (gap <= max_gap)
    hole = native_indices[right]-native_indices[left] > 1
    flags['observed_exact'] = exact
    flags['ordinary_resample'] = interpolated & ~hole
    flags['filled_nonfinite_hole'] = interpolated & hole
    flags['long_gap'] = (~exact) & inside & (gap > max_gap)
    flags['outside_retained_support'] = (~exact) & ~inside
    return flags


def _longest_run_duration(mask, hz):
    cuts = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, dtype=bool), False]))
    return float(np.max(np.diff(cuts.reshape(-1, 2), axis=1))/hz) if len(cuts) else 0.


def native_diagnostics(times, xy, likelihood, resolution, max_gap=.1):
    """Uncalibrated QC flags only; none is a deletion instruction.

    An isolated-jump candidate has two opposite large velocities and returns
    near its neighbors. Sustained fast motion is separately flagged and is
    never labeled invalid merely because its speed is high.
    """
    times, xy, likelihood = np.asarray(times), np.asarray(xy), np.asarray(likelihood)
    good = finite_rows(xy)
    dt = np.diff(times)
    displacement = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = displacement/dt
    finite_speed = speed[np.isfinite(speed)]
    typical = float(np.median(finite_speed)) if len(finite_speed) else 0.
    candidate_speed = max(1000., 6*typical)
    isolated = np.zeros(len(xy), dtype=bool)
    if len(xy) > 2:
        excursion = displacement[:-1]+displacement[1:]
        return_distance = np.linalg.norm(xy[2:]-xy[:-2], axis=1)
        isolated[1:-1] = (good[:-2] & good[1:-1] & good[2:]
            & (speed[:-1] > candidate_speed) & (speed[1:] > candidate_speed)
            & (return_distance <= .2*excursion)
            & (dt[:-1] <= max_gap) & (dt[1:] <= max_gap))
    outside = good & ((xy < 0).any(axis=1) | (xy > resolution).any(axis=1))
    result = {
        'native_nonfinite': ~good,
        'outside_frame': outside,
        'low_likelihood': np.isfinite(likelihood) & (likelihood < .5),
        'low_likelihood_02': np.isfinite(likelihood) & (likelihood < .2),
        'low_likelihood_05': np.isfinite(likelihood) & (likelihood < .5),
        'low_likelihood_08': np.isfinite(likelihood) & (likelihood < .8),
        'likelihood_nonfinite': ~np.isfinite(likelihood),
        'jump_over_100px': np.r_[False, displacement > 100.],
        'native_speed_px_s': np.r_[np.nan, speed],
        'high_speed_candidate': np.r_[False, speed > candidate_speed],
        'isolated_jump_candidate': isolated,
        'native_long_gap_before': np.r_[False, dt > max_gap],
        'quality_rejected': np.zeros(len(xy), dtype=bool),
    }
    return result


def filter_valid(xy, window=3, method="boxcar", trim_edges=False):
    """Centered filter within finite runs; return values and support-edge mask.

    Both filters use nearest extension only within their own finite run. The
    optional trimming sensitivity removes half a window at both ends of each
    run, so no retained sample then depends on padded values. Short runs can
    disappear entirely under trimming. Filtering never spans a NaN gap. The
    returned edge mask is populated even when trimming is disabled.
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
        if window > 1:
            half = window // 2
            edges[start:min(start+half, stop)] = True
            edges[max(start, stop-half):stop] = True
    if trim_edges:
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
    diagnostics[side] contains native/grid/filter QC flags and their time axes
    (never deletion instructions). Keys beginning grid_ have grid length;
    position_/velocity_ keys have the corresponding signal length. Other QC
    flags have native length. provenance carries the same data grouped by role.
    """
    if "gt" not in caches or not caches:
        raise ValueError("caches must contain gt")
    # This revision implements the explicit user-confirmed model-v2 policy.
    # Reject unsupported settings rather than silently ignoring a requested cut.
    if config.get('dlc_likelihood_filter') is not None or config.get('jump_filter', False) or config.get('coordinate_range_filter', False):
        raise ValueError('This run retains all finite DLC coordinates; QC filtering is disabled')
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
    qc, audit, diagnostics, provenance = [], [], {}, {}
    for source, z in caches.items():
        for field in ("time_s", "native_time_s", "frame_number", "stage", "occurrence", "native_stage", "native_occurrence"):
            if field in base and field in z and not np.array_equal(base[field], z[field]):
                raise ValueError(f"source {source} has mismatched {field}")
        audit.append({**identity, "source": source, "check": "cached_time_frame_stage_match", "passed": True,
                      "gaze_input": "new_native_NPY_coordinate_valid_gaze_resampled_41hz_models_original_predictions"})
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
        if native.shape != (len(native_t), 2) or lik.shape != (len(native_t),):
            raise ValueError(f'Invalid native DLC shape for {side}')
        native_good = finite_rows(native)
        diagnostics[side] = native_diagnostics(native_t, native, lik, resolution, max_gap)
        diagnostics[side]['native_time_s'] = native_t.copy()
        diagnostics[side]['grid_time_s'] = grid.copy()
        outside = diagnostics[side]['outside_frame']
        jump = diagnostics[side]['jump_over_100px']
        native_speed = diagnostics[side]['native_speed_px_s']
        interpolation = interpolation_provenance(native_t, native, grid, max_gap)
        for flag, mask in interpolation.items():
            diagnostics[side]['grid_'+flag] = mask
        provenance[side] = {'native_time_s': native_t.copy(), 'grid_time_s': grid.copy(),
                            'interpolation': interpolation, 'signals': {}}
        gazes, common = _common_gazes(caches, side)
        for signal, window, method in (("position", pw, "boxcar"), ("velocity", vw, vf)):
            gs, gmask, gedges = _filter_gazes(gazes, window, method, trim)
            ps, pedge = filter_valid(psm_raw, window, method, trim)
            # Edges are always identified, even when the primary keeps padded
            # estimates. For velocity the union of both endpoint flags applies.
            p_signal_edge = pedge[:-1] | pedge[1:] if signal == 'velocity' else pedge.copy()
            g_signal_edges = {s: e[:-1] | e[1:] if signal == 'velocity' else e.copy()
                              for s, e in gedges.items()}
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
                                     "gaze_valid": gmask.copy(), "psm_valid": pmask.copy(),
                                     "psm_filter_edge": p_signal_edge,
                                     "gaze_filter_edges": g_signal_edges}
            provenance[side]['signals'][signal] = {'psm_filter_edge': p_signal_edge,
                'gaze_filter_edges': g_signal_edges, 'gaze_valid': gmask.copy(),
                'psm_valid': pmask.copy(), 'time_s': times.copy()}
            diagnostics[side][signal+'_psm_filter_edge'] = p_signal_edge
            diagnostics[side][signal+'_psm_valid'] = pmask.copy()
            diagnostics[side][signal+'_time_s'] = times.copy()
            for source, edge in g_signal_edges.items():
                diagnostics[side][signal+'_'+source+'_filter_edge'] = edge
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
                       "raw_grid_nonfinite": int((~finite_rows(psm_raw)).sum()),
                       "signal_filter_edge_points": int(p_signal_edge.sum()),
                       "filter_edge_removed_points": int(p_signal_edge.sum()) if trim else 0,
                       "isolated_jump_candidate_count": int(diagnostics[side]['isolated_jump_candidate'].sum()),
                       "low_likelihood_02_fraction": float(diagnostics[side]['low_likelihood_02'].mean()),
                       "low_likelihood_08_fraction": float(diagnostics[side]['low_likelihood_08'].mean()),
                       "ordinary_resample_fraction": float(interpolation['ordinary_resample'].mean()),
                       "filled_nonfinite_hole_fraction": float(interpolation['filled_nonfinite_hole'].mean()),
                       "filled_quality_hole_fraction": 0.,
                       "long_gap_grid_fraction": float(interpolation['long_gap'].mean()),
                       "longest_grid_missing_s": _longest_run_duration(~finite_rows(psm_raw), hz),
                       "jump_candidate_policy": 'uncalibrated_QC_only_actual_dt_return_geometry'})
            for source, a in gs.items():
                qc.append({**common_info, "source": source, "valid_samples": int(finite_rows(a).sum()),
                           "valid_fraction": float(finite_rows(a).mean()), "filter_edge_points": int(gedges[source].sum()),
                           "common_gaze_masked_raw_points": int((finite_rows(caches[source][f"gaze_raw_{side}"]) & ~common).sum()),
                           "quality_rejected_count": 0, "likelihood_policy": "not_applicable",
                           "signal_filter_edge_points": int(g_signal_edges[source].sum()),
                           "filter_edge_removed_points": int(g_signal_edges[source].sum()) if trim else 0,
                           "gaze_input": 'cached_41hz_interpolated_not_original_high_rate'})
    return {"signals": signals, "qc": qc, "audit": audit, "diagnostics": diagnostics,
            "provenance": provenance}


prepare = prepare_signals
