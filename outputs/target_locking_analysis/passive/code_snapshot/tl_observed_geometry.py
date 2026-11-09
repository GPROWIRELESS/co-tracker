"""Unchanged v3 real-gaze geometry, native exposure, smoothing and temporal algorithms.

Only observation functions are retained. Target and PSM centers use original
continuous pixels; gaze accepts the closed image boundary. Pixel areas are
counted at (column+0.5,row+0.5). No probability or prediction code is included.
"""
import numpy as np

NAMES = ("other", "target_only", "psm_only", "overlap")
HZ = 41.0
MAX_GAP_S = 0.100
MIN_DWELL_S = 0.120

def _size(W, H):
    if int(W) != W or int(H) != H or W <= 0 or H <= 0:
        raise ValueError("W and H must be positive integer dimensions")
    return int(W), int(H)


def _radii(rT, rP):
    if not all(np.isscalar(r) and np.isfinite(r) and r >= 0 for r in (rT, rP)):
        raise ValueError("rT and rP must be finite, nonnegative pixel radii")


def _points(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Expected [N,2] coordinates")
    return x


def _vec(x, n, dtype=None):
    x = np.asarray(x, dtype=dtype)
    if x.shape != (n,):
        raise ValueError(f"Expected a vector of length {n}, got {x.shape}")
    return x


def target_center(points, W, H):
    """Mean of finite, in-frame coordinate pairs; uint64 bit i = slot i+1 used."""
    W, H = _size(W, H)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 2 or points.shape[1] > 64:
        raise ValueError("Expected [N,K,2] coordinates with at most 64 slots")
    used = ((points >= 0) & (points < [W, H])).all(axis=2)
    n = used.sum(axis=1)
    total = np.where(used[:, :, None], points, 0.0).sum(axis=1)
    xy = np.full((len(points), 2), np.nan)
    np.divide(total, n[:, None], out=xy, where=n[:, None] > 0)
    bits = np.left_shift(np.uint64(1), np.arange(points.shape[1], dtype=np.uint64))
    slotmask = (used * bits).sum(axis=1, dtype=np.uint64)
    return xy, n, slotmask


def _geometry(z, left, right, valid):
    z, left, right = map(_points, (z, left, right))
    if not (len(left) == len(right) == len(z)):
        raise ValueError("Target and gripper tracks must have the same length")
    ok = _vec(valid, len(z), dtype=bool) & np.isfinite(z).all(axis=1)
    ok &= np.isfinite(left).all(axis=1) | np.isfinite(right).all(axis=1)
    return z, left, right, ok


def classify(xy, z, left, right, rT, rP, valid):
    """Point labels; either gripper can define P, but they never double count."""
    _radii(rT, rP)
    z, left, right, eligible = _geometry(z, left, right, valid)
    xy = _points(xy)
    if len(xy) != len(z):
        raise ValueError("Gaze and region tracks must have the same length")
    eligible &= np.isfinite(xy).all(axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
        T = np.square(xy-z).sum(axis=1) <= rT*rT
        P = ((np.square(xy-left).sum(axis=1) <= rP*rP) |
             (np.square(xy-right).sum(axis=1) <= rP*rP))
    return np.where(eligible, T.astype(np.int8) + P.astype(np.int8)*2, -1).astype(np.int8)


def _scanlines(xy, r, H):
    # At most floor(2r)+1 different pixel-center rows lie inside a diameter.
    first = np.maximum(0, np.ceil(xy[:, 1] - r - .5))
    count = min(H, int(np.floor(2*r)) + 1)
    return first[:, None] + .5 + np.arange(count)


def _interval(xy, present, r, y, W, H):
    """Inclusive integer column indices covered on a batch of pixel-center rows."""
    dx2 = r*r - (y - xy[:, 1, None])**2
    dx = np.sqrt(np.maximum(0, dx2))
    lo = np.maximum(0, np.ceil(xy[:, 0, None] - dx - .5))
    hi = np.minimum(W - 1, np.floor(xy[:, 0, None] + dx - .5))
    hi = np.where(present[:, None] & (dx2 >= 0) & (y < H), hi, -1)
    return lo, hi


def _intersection(a, b):
    return np.maximum(a[0], b[0]), np.minimum(a[1], b[1])


def _count(interval):
    return np.maximum(0, interval[1] - interval[0] + 1).sum(axis=1)


def areas(z, left, right, W, H, rT, rP, valid):
    """Exact fractions on the pixel-center lattice, in codes (other,T,P,O).

    Counts horizontal disk intervals and their intersections, clipping to the
    image. Runtime depends on disk diameter, not the distance between grippers.
    NaN rows represent unavailable geometry. This is a uniform random point
    reference; it is distinct from classifying the mean of a uniform heatmap.
    """
    W, H = _size(W, H)
    _radii(rT, rP)
    z, left, right, eligible = _geometry(z, left, right, valid)
    out = np.full((len(z), 4), np.nan)
    ids = np.flatnonzero(eligible)
    total = float(W*H)
    chunk_size = max(1, min(512, 100000 // min(H, int(2*max(rT,rP))+1)))
    for start in range(0, len(ids), chunk_size):
        idx = ids[start:start+chunk_size]
        t, l, r = z[idx], left[idx], right[idx]
        lt, rt = np.isfinite(l).all(axis=1), np.isfinite(r).all(axis=1)
        l = np.where(lt[:, None], l, 0.0)
        r = np.where(rt[:, None], r, 0.0)
        visible = np.ones(len(idx), dtype=bool)
        yt = _scanlines(t, rT, H)
        ti = _interval(t, visible, rT, yt, W, H)
        tl = _intersection(ti, _interval(l, lt, rP, yt, W, H))
        tr = _intersection(ti, _interval(r, rt, rP, yt, W, H))
        nT = _count(ti)
        nO = _count(tl) + _count(tr) - _count(_intersection(tl, tr))
        yl = _scanlines(l, rP, H)
        li = _interval(l, lt, rP, yl, W, H)
        lr = _intersection(li, _interval(r, rt, rP, yl, W, H))
        ri = _interval(r, rt, rP, _scanlines(r, rP, H), W, H)
        nP = _count(li) + _count(ri) - _count(lr)
        out[idx] = np.column_stack((total - nT - nP + nO, nT - nO, nP - nO, nO)) / total
    return out


def times(ts_ns, frames):
    """Native dwell exposure (seconds) from adjacent timestamp midpoints.

    Each good interval contributes half to each adjacent frame. There is no
    extrapolation before the first sample, after the last, over a missing
    frame number, or over a nonpositive or >100 ms gap. An isolated frame
    contributes zero seconds. The caller chooses how to aggregate classifications.
    """
    t = np.asarray(ts_ns)
    if t.ndim != 1:
        raise ValueError("Expected native timestamps [N]")
    frames = _vec(frames, len(t))
    dt = np.zeros(len(t))
    breaks = np.ones(len(t), dtype=bool)
    if len(t) > 1:
        good_times = np.isfinite(t) & (t >= 0)
        # Subtract native int64 values before converting to seconds.
        edges = np.diff(t).astype(np.float64)/1e9
        good = (good_times[1:] & good_times[:-1] & (np.diff(frames) == 1)
                & (edges > 0) & (edges <= MAX_GAP_S))
        breaks[1:] = ~good
        half = np.where(good, edges * .5, 0)
        dt[:-1] += half
        dt[1:] += half
    return dt, breaks


def _blocks(occ, breaks):
    n = len(occ)
    if not n:
        return
    cuts = np.flatnonzero(breaks[1:] | (occ[1:] != occ[:-1])) + 1
    ends = np.r_[0, cuts, n]
    yield from zip(ends[:-1], ends[1:])


def smooth_gaze(ts_ns, xy, occ, break_before):
    """Sensitivity preprocessing: linear interpolation, 41 Hz, centered 3 samples.

    Each contiguous occurrence is processed separately. Within it, interpolated
    intervals require valid bracketing native samples <=100 ms apart. The uniform
    41 Hz grid starts at the first valid sample of each such block; endpoint values
    are held for the final fractional grid interval and for the centered boxcar
    boundary. Smoothed values are mapped back only within the native support.
    The flag marks filled native positions, not all original points moved by smoothing.
    """
    xy = _points(xy)
    n = len(xy)
    t = _vec(ts_ns, n)
    occ = _vec(occ, n)
    br = _vec(break_before, n, dtype=bool).copy()
    good_t = np.isfinite(t) & (t >= 0)
    if n > 1:
        d = np.diff(t)
        br[1:] |= ~good_t[1:] | ~good_t[:-1] | (d <= 0) | (d > 1e9*MAX_GAP_S)
    observed = np.isfinite(xy).all(axis=1) & good_t
    out = np.full_like(xy, np.nan)
    for start, stop in _blocks(occ, br):
        ids = np.flatnonzero(observed[start:stop]) + start
        if not len(ids):
            continue
        cuts = np.flatnonzero(np.diff(t[ids]) > 1e9*MAX_GAP_S) + 1
        for src in np.split(ids, cuts):
            first, last = src[0], src[-1]
            if len(src) == 1:
                out[first] = xy[first]
                continue
            native = (t[first:last+1] - t[first]).astype(np.float64)/1e9
            grid = np.arange(int(np.ceil(native[-1]*HZ)) + 1) / HZ
            for axis in range(2):
                values = np.interp(grid, native[src-first], xy[src, axis])
                pad = np.pad(values, (1,1), mode="edge")
                smoothed = (pad[:-2]+pad[1:-1]+pad[2:]) / 3
                out[first:last+1, axis] = np.interp(native, grid, smoothed)
    return out, np.isfinite(out).all(axis=1) & ~observed


def temporal(labels, dt, occ, break_before):
    """Median observed AOI runs >=120 ms and switches per valid minute.

    Stage/occurrence boundaries, frame gaps and missing classifications cut
    runs and transitions. A run touching an observation boundary is flagged
    as censored; its observed duration is included only if it reaches 120 ms.
    Direct switches join consecutive qualifying T/P runs. Via-overlap switches
    join qualifying T/P runs separated by one contiguous overlap run. Other
    intervals are never skipped. Both rates use the same valid-time denominator.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (-1,0,1,2,3)).all():
        raise ValueError("Expected -1/0/1/2/3 classification labels")
    n = len(labels)
    dt = _vec(dt, n, dtype=float)
    if np.any(~np.isfinite(dt) | (dt < 0)):
        raise ValueError("Invalid dwell exposures")
    occ = _vec(occ, n)
    br = _vec(break_before, n, dtype=bool)
    durations = [[] for _ in NAMES]
    censor = np.zeros(4, dtype=int)
    nall = np.zeros(4, dtype=int)
    cens_all = np.zeros(4, dtype=int)
    direct = via = 0
    for start, stop in _blocks(occ, br):
        labels_seg = labels[start:stop]
        boundaries = np.r_[start, start+1+np.flatnonzero(labels_seg[1:] != labels_seg[:-1]), stop]
        state = labels[boundaries[:-1]]
        duration = np.add.reduceat(dt[start:stop], boundaries[:-1]-start)
        qualifying = duration >= MIN_DWELL_S - 1e-12
        for j, (k, secs) in enumerate(zip(state, duration)):
            if k < 0:
                continue
            censored = j == 0 or j == len(state)-1 or state[j-1] < 0 or state[j+1] < 0
            nall[k] += 1
            cens_all[k] += int(censored)
            if qualifying[j]:
                durations[k].append(secs)
                censor[k] += int(censored)
            if j == 0 or not qualifying[j] or k not in (1,2):
                continue
            opposite = 3 - k
            if state[j-1] == opposite and qualifying[j-1]:
                direct += 1
            elif j >= 2 and state[j-1] == 3 and state[j-2] == opposite and qualifying[j-2]:
                via += 1
    valid_s = float(dt[labels >= 0].sum())
    result = {"valid_seconds": valid_s, "direct_count": direct, "via_count": via,
              "direct_rate": direct*60/valid_s if valid_s > 0 else np.nan,
              "via_rate": via*60/valid_s if valid_s > 0 else np.nan}
    for k, name in enumerate(NAMES):
        ds = durations[k]
        result[f"dwell_{name}_median_s"] = float(np.median(ds)) if ds else np.nan
        result[f"dwell_{name}_n"] = len(ds)
        result[f"dwell_{name}_n_censored"] = int(censor[k])
        result[f"all_{name}_runs"] = int(nall[k])
        result[f"all_{name}_runs_censored"] = int(cens_all[k])
    for short, name in zip("TPO", NAMES[1:]):
        result[f"dwell_{short}_ms"] = 1000*result[f"dwell_{name}_median_s"]
    return result

