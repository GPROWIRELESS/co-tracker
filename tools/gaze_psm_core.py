"""Position/velocity cross-correlation with PSM-anchored stage pooling.

Positive lag compares gaze at ``u - lag`` with PSM at ``u``.  Stage
occurrences select PSM samples; the gaze sample can precede the occurrence.
Each occurrence is centered on its own valid overlap at every lag.  Centered
sufficient statistics, rather than occurrence correlations, are then pooled.
"""

import numpy as np
from scipy.signal import correlate


def _fft_statistics(gaze, psm, gaze_valid, psm_valid, lags):
    """Return count, centered covariance, and two centered variances."""
    # Removing a convenient constant before the FFT reduces cancellation in
    # sum(x*x) - sum(x)**2/n.  The subsequent overlap centering is unchanged.
    g_offset = gaze[gaze_valid].mean(axis=0) if gaze_valid.any() else np.zeros(2)
    p_offset = psm[psm_valid].mean(axis=0) if psm_valid.any() else np.zeros(2)
    g = np.where(gaze_valid[:, None], gaze-g_offset, 0.)
    p = np.where(psm_valid[:, None], psm-p_offset, 0.)
    gm, pm = gaze_valid.astype(float), psm_valid.astype(float)
    indices = len(gaze)-1+lags
    keep = (indices >= 0) & (indices < 2*len(gaze)-1)

    def corr(a, b):
        result = np.zeros(len(lags), dtype=float)
        full = correlate(a, b, mode="full", method="fft")
        result[keep] = full[indices[keep]]
        return result

    count = np.rint(corr(pm, gm)).clip(0).astype(np.int64)
    divisor = np.maximum(count, 1)
    covariance, gaze_variance, psm_variance = (np.zeros(len(lags)) for _ in range(3))
    for axis in range(2):
        gs, ps = corr(pm, g[:, axis]), corr(p[:, axis], gm)
        covariance += corr(p[:, axis], g[:, axis])-gs*ps/divisor
        gaze_variance += corr(pm, g[:, axis]**2)-gs**2/divisor
        psm_variance += corr(p[:, axis]**2, gm)-ps**2/divisor
    return count, covariance, np.maximum(gaze_variance, 0), np.maximum(psm_variance, 0)


def analyze_curve(gaze, psm, hz, max_lag=2., min_overlap=5.,
                  occurrence_ids=None, within_stage=False,
                  min_gaze_rms=1., min_psm_rms=1., spatial_tolerance=50.,
                  compute_spatial=True):
    """Calculate a fixed-grid, overlap-normalized 2-D xcorr curve.

    Parameters
    ----------
    gaze, psm : arrays, shape (N, 2)
        Synchronized positions, or signed x/y velocities.  A nonfinite value
        in either coordinate makes that sample missing.
    hz : float
        Uniform sampling rate.
    occurrence_ids : integer array, shape (N,), optional
        Unique nonnegative IDs of occurrences of the requested stage.  -1
        excludes samples.  Membership refers to PSM time.  None selects the
        complete recording as one occurrence.
    within_stage : bool
        Require gaze and PSM samples to belong to the same occurrence.  False
        allows gaze outside the occurrence, including the preceding stage.
    min_overlap : float
        Minimum *pooled* overlap duration in seconds.  Individual occurrences
        need only two valid pairs to contribute centered sufficient statistics.
    min_gaze_rms, min_psm_rms : float
        Minimum pooled within-occurrence RMS coordinate variation.  Units are
        pixels for position, pixels/second for velocity.  These are variation
        thresholds, not mean-speed or fixation thresholds.
    spatial_tolerance : float
        Maximum raw-coordinate Euclidean distance for proximity fraction.
    compute_spatial : bool
        Disable for velocities, where distances are not spatial separation.

    Returns
    -------
    dict of arrays
        lag_s, r, pairs, occurrences, gaze_rms, psm_rms,
        distance_median_px, proximity.  All arrays use the same lag grid even
        for short or empty inputs.  ``pairs`` counts pairs from occurrences
        contributing at least two valid pairs at that lag.  Spatial metrics
        use those same pairs, in original coordinates without centering.
        Metrics can be present where r is unsupported; inspect pairs/r before
        interpreting them.  No peak or significance decision is made here.
    """
    gaze, psm = np.asarray(gaze, dtype=float), np.asarray(psm, dtype=float)
    if gaze.ndim != 2 or gaze.shape[1:] != (2,) or psm.shape != gaze.shape:
        raise ValueError("gaze and psm must have identical (N, 2) shapes")
    scalars = [hz, max_lag, min_overlap, min_gaze_rms, min_psm_rms, spatial_tolerance]
    if not np.isfinite(scalars).all() or hz <= 0 or any(v < 0 for v in scalars[1:]):
        raise ValueError("hz must be positive; lag, overlap, RMS and tolerance must be nonnegative")
    n = len(gaze)
    if occurrence_ids is None:
        occurrence_ids = np.zeros(n, dtype=np.int64)
    else:
        ids = np.asarray(occurrence_ids)
        if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("occurrence_ids must be a length-N integer array")
        occurrence_ids = ids.astype(np.int64, copy=False)
    maximum = int(round(max_lag*hz))
    lags = np.arange(-maximum, maximum+1, dtype=np.int64)
    count = np.zeros(len(lags), dtype=np.int64)
    occurrences = np.zeros(len(lags), dtype=np.int64)
    covariance, gaze_variance, psm_variance = (np.zeros(len(lags)) for _ in range(3))
    distance_lists = [[] for _ in lags]
    gaze_finite, psm_finite = np.isfinite(gaze).all(axis=1), np.isfinite(psm).all(axis=1)

    for identity in np.unique(occurrence_ids[occurrence_ids >= 0]):
        members = np.flatnonzero(occurrence_ids == identity)
        if len(members) < 2:
            continue
        # Include surrounding gaze so positive lags can reach before the stage.
        start, end = max(0, members[0]-maximum), min(n, members[-1]+maximum+1)
        local_ids = occurrence_ids[start:end]
        pm = psm_finite[start:end] & (local_ids == identity)
        gm = gaze_finite[start:end].copy()
        if within_stage:
            gm &= local_ids == identity
        if pm.sum() < 2 or gm.sum() < 2:
            continue
        terms = _fft_statistics(gaze[start:end], psm[start:end], gm, pm, lags)
        usable = terms[0] >= 2
        count[usable] += terms[0][usable]
        occurrences[usable] += 1
        covariance[usable] += terms[1][usable]
        gaze_variance[usable] += terms[2][usable]
        psm_variance[usable] += terms[3][usable]
        if compute_spatial:
            for index in np.flatnonzero(usable):
                shifted = members-lags[index]
                inside = (shifted >= 0) & (shifted < n)
                u, t = members[inside], shifted[inside]
                valid = psm_finite[u] & gaze_finite[t]
                if within_stage:
                    valid &= occurrence_ids[t] == identity
                if valid.sum() >= 2:
                    distance_lists[index].append(np.linalg.norm(psm[u[valid]]-gaze[t[valid]], axis=1))

    safe_count = np.maximum(count, 1)
    gaze_rms = np.sqrt(gaze_variance/safe_count)
    psm_rms = np.sqrt(psm_variance/safe_count)
    gaze_rms[count == 0] = np.nan
    psm_rms[count == 0] = np.nan
    denominator = np.sqrt(gaze_variance*psm_variance)
    supported = (count >= max(2, int(np.ceil(min_overlap*hz))))
    supported &= (gaze_rms >= min_gaze_rms) & (psm_rms >= min_psm_rms)
    supported &= denominator > np.finfo(float).eps
    r = np.full(len(lags), np.nan)
    r[supported] = np.clip(covariance[supported]/denominator[supported], -1., 1.)
    median_distance, proximity = np.full(len(lags), np.nan), np.full(len(lags), np.nan)
    if compute_spatial:
        for index, arrays in enumerate(distance_lists):
            if arrays:
                distances = np.concatenate(arrays)
                median_distance[index] = np.median(distances)
                proximity[index] = np.mean(distances <= spatial_tolerance)
    return {"lag_s": lags/hz, "r": r, "pairs": count, "occurrences": occurrences,
            "gaze_rms": gaze_rms, "psm_rms": psm_rms,
            "distance_median_px": median_distance, "proximity": proximity}
