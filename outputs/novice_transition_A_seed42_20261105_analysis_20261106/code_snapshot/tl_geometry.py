"""Frame-level geometry and temporal measurements for target locking.

Coordinate convention: normalized gaze/model coordinates span [0,W] × [0,H].
Raw target/gripper annotations are pixel coordinates in [0,W) × [0,H).
Uniform area is evaluated at W*H pixel centers (col+0.5, row+0.5).
Region codes throughout: -1 unavailable, 0 other, 1 target-only, 2 PSM-only,
3 overlap. PSM is the union of the available gripper disks; pass the agreed
frame/side quality mask when an analysis requires both grippers.
"""

import numpy as np


NAMES = ("other", "target_only", "psm_only", "overlap")
HZ = 41.0
MAX_GAP_S = 0.100
MIN_DWELL_S = 0.120


def sample_region_probabilities(samples_xy, z, left, right, rT, rP, valid):
    """Monte Carlo mass of four disjoint regions, retaining all K model draws.

    A row is available only when every draw is finite. The input loader checks
    the normalized coordinate domain. Draws are points, not Gaussian kernels.
    """
    s = np.asarray(samples_xy, dtype=float)
    if s.ndim != 3 or s.shape[2] != 2 or s.shape[1] < 1:
        raise ValueError('samples_xy must have shape [N,K,2] with K > 0')
    z, left, right, eligible = _geometry(z, left, right, valid)
    eligible &= np.isfinite(s).all(axis=(1, 2))
    q = np.full((len(z), 4), np.nan)
    for start in range(0, len(z), 1024):
        ix = np.flatnonzero(eligible[start:start+1024]) + start
        if not len(ix):
            continue
        points = s[ix]
        t = np.square(points-z[ix, None]).sum(axis=2) <= rT*rT
        p = ((np.square(points-left[ix, None]).sum(axis=2) <= rP*rP) |
             (np.square(points-right[ix, None]).sum(axis=2) <= rP*rP))
        state = t.astype(np.int8) + 2*p.astype(np.int8)
        q[ix] = np.stack([(state == k).mean(axis=1) for k in range(4)], axis=1)
    return q


def gaussian_maps(xy, W, H, shape=(32, 40), sigma=1.0):
    """Normalized sigma-in-grid-units GT kernels at continuous gaze positions.

    These are smoothing diagnostics, not an attainable-error bound or an
    independent predictor: the evaluation gaze itself centers each kernel.
    """
    points = _points(xy)
    h, w = shape
    px = np.exp(-.5*((np.arange(w)[None] - points[:, 0, None]*(w-1)/W)/sigma)**2)
    py = np.exp(-.5*((np.arange(h)[None] - points[:, 1, None]*(h-1)/H)/sigma)**2)
    px /= px.sum(axis=1, keepdims=True)
    py /= py.sum(axis=1, keepdims=True)
    return py[:, :, None]*px[:, None, :]


def _grid_cells(W, H, h, w):
    """Pixel-center Voronoi cells of inclusive endpoint grid nodes.

    Nodes are j*W/(w-1), i*H/(h-1), matching exported heatmaps. Endpoint cells
    are half-width. A heatmap value is cell probability MASS, spread uniformly
    over the image pixels assigned to that node, not a density sample.
    """
    xb = np.floor((np.arange(W)+.5)*(w-1)/W + .5).astype(int)
    yb = np.floor((np.arange(H)+.5)*(h-1)/H + .5).astype(int)
    nx, ny = np.bincount(xb, minlength=w), np.bincount(yb, minlength=h)
    if (nx == 0).any() or (ny == 0).any():
        raise ValueError('Heatmap grid must not exceed the pixel-center resolution')
    xedge = np.r_[0, np.cumsum(nx)]
    return xb, yb, nx, ny, xedge


def map_region_probabilities(probability, z, left, right, W, H, rT, rP, valid):
    """Exact pixel-area weighted integration of grid-cell probability masses.

    Uses the same integer disk scanlines as areas(). It does not form a full
    image or round centers. Input may be a common [h,w] map or [N,h,w] maps.
    """
    W, H = _size(W, H)
    z, left, right, eligible = _geometry(z, left, right, valid)
    p = np.asarray(probability, dtype=float)
    shared = p.ndim == 2
    if p.ndim not in (2, 3) or (not shared and len(p) != len(z)):
        raise ValueError('Expected [h,w] or [N,h,w] probability maps')
    h, w = p.shape[-2:]
    xb, yb, nx, ny, xedge = _grid_cells(W, H, h, w)
    q = np.full((len(z), 4), np.nan)
    ids = np.flatnonzero(eligible)
    for start in range(0, len(ids), 256):
        ix = ids[start:start+256]
        batch = np.broadcast_to(p, (len(ix), h, w)) if shared else p[ix]
        mass = batch.sum(axis=(1, 2))
        good = np.isfinite(batch).all(axis=(1, 2)) & (batch >= 0).all(axis=(1, 2)) & (mass > 0)
        ix = ix[good]
        if not len(ix):
            continue
        batch = batch[good] / mass[good, None, None]
        density = batch / (ny[None, :, None]*nx[None, None, :])
        prefix = np.concatenate([np.zeros((len(ix), h, 1)),
                                 np.cumsum(batch/ny[None, :, None], axis=2)], axis=2)
        visible = np.ones(len(ix), dtype=bool)
        t, l, r = z[ix], left[ix], right[ix]
        lt, rt = np.isfinite(l).all(axis=1), np.isfinite(r).all(axis=1)
        l, r = np.nan_to_num(l), np.nan_to_num(r)

        def weight(interval, y):
            lo, hi = interval
            ok = (hi >= lo) & (y >= .5) & (y < H)
            row = np.clip(np.floor(y).astype(int), 0, H-1)
            rowbin = yb[row]
            b = np.arange(len(ix))[:, None]

            def cdf(x):
                x = np.clip(x, 0, W).astype(int)
                cell = xb[np.minimum(x, W-1)]
                return prefix[b, rowbin, cell] + density[b, rowbin, cell]*(x-xedge[cell])

            value = cdf(hi+1)-cdf(lo)
            return np.where(ok, value, 0).sum(axis=1)

        yt = _scanlines(t, rT, H)
        ti = _interval(t, visible, rT, yt, W, H)
        tl = _intersection(ti, _interval(l, lt, rP, yt, W, H))
        tr = _intersection(ti, _interval(r, rt, rP, yt, W, H))
        qt = weight(ti, yt)
        qo = weight(tl, yt)+weight(tr, yt)-weight(_intersection(tl, tr), yt)
        yl, yr = _scanlines(l, rP, H), _scanlines(r, rP, H)
        li = _interval(l, lt, rP, yl, W, H)
        lr = _intersection(li, _interval(r, rt, rP, yl, W, H))
        ri = _interval(r, rt, rP, yr, W, H)
        qp = weight(li, yl)+weight(ri, yr)-weight(lr, yl)
        out = np.column_stack((1-qt-qp+qo, qt-qo, qp-qo, qo))
        if out.min() < -1e-10 or out.max() > 1+1e-10:
            raise ValueError('Integrated probability outside [0,1]')
        out = np.clip(out, 0, 1)
        out /= out.sum(axis=1, keepdims=True)
        q[ix] = out
    return q


def probability_self_test():
    """Independent enumeration covers boundaries, overlaps and nonuniform maps."""
    rng = np.random.default_rng(42)
    W, H, h, w = 31, 27, 5, 6
    n = 15
    z, l, r = [rng.uniform([0, 0], [W, H], (n, 2)) for _ in range(3)]
    r[0] = l[0]
    z[1] = l[1]
    p = rng.random((n, h, w))
    p /= p.sum(axis=(1, 2), keepdims=True)
    xb, yb, nx, ny, _ = _grid_cells(W, H, h, w)
    yy, xx = np.mgrid[:H, :W]
    pixel = np.column_stack((xx.ravel()+.5, yy.ravel()+.5))
    for radii in [(0, 0), (3, 5), (15, 8)]:
        actual = map_region_probabilities(p, z, l, r, W, H, *radii, np.ones(n, bool))
        for i in range(n):
            state = classify(pixel, np.broadcast_to(z[i], pixel.shape),
                             np.broadcast_to(l[i], pixel.shape), np.broadcast_to(r[i], pixel.shape),
                             *radii, np.ones(len(pixel), bool))
            density = p[i]/(ny[:, None]*nx[None, :])
            pixelmass = density[yb[:, None], xb[None, :]].ravel()
            exact = np.bincount(state, weights=pixelmass, minlength=4)
            np.testing.assert_allclose(actual[i], exact, rtol=0, atol=1e-12)
        uniform = ny[:, None]*nx[None, :]/(W*H)
        q = map_region_probabilities(uniform, z, l, r, W, H, *radii, np.ones(n, bool))
        np.testing.assert_allclose(q, areas(z, l, r, W, H, *radii, np.ones(n, bool)), atol=1e-12)
    s = np.array([[[0., 0.], [10., 10.], [20., 20.], [10., 10.]]])
    q = sample_region_probabilities(s, np.array([[10., 10.]]),
                                  np.array([[0., 0.]]), np.array([[30., 30.]]), 1, 1, [True])
    np.testing.assert_allclose(q, [[.25, .5, .25, 0]])
    h_event = np.array([1., 0.])
    prob = np.array([.75, .25])
    assert np.mean((prob-h_event)**2) == .0625
    print('Probability geometry self-check passed.')


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


def probability_xy(prob, W, H):
    """Normalize each finite nonnegative map; return its continuous pixel mean.

    The model uses inclusive normalized grid positions j/(w-1), i/(h-1).
    Map rows correspond to y; columns correspond to x. No clipping or rounding.
    A map containing a negative or nonfinite entry, or zero mass, yields NaN.
    """
    W, H = _size(W, H)
    p = np.asarray(prob)
    if p.ndim != 3 or min(p.shape[1:]) < 2:
        raise ValueError("Expected probability maps [N,h,w] with h,w >= 2")
    n, h, w = p.shape
    out = np.full((n, 2), np.nan)
    xx = np.linspace(0, W, w)
    yy = np.linspace(0, H, h)
    for start in range(0, n, 512):
        chunk = p[start:start+512]
        with np.errstate(over="ignore", invalid="ignore"):
            px = chunk.sum(axis=1, dtype=np.float64)
            py = chunk.sum(axis=2, dtype=np.float64)
            mass = px.sum(axis=1)
            ok = (chunk.min(axis=(1, 2)) >= 0) & np.isfinite(mass) & (mass > 0)
            # Scale after normalization to avoid overflow in mass*pixel products.
            block = out[start:start+len(chunk)]
            block[ok, 0] = np.einsum("ij,j->i", px[ok] / mass[ok, None], xx)
            block[ok, 1] = np.einsum("ij,j->i", py[ok] / mass[ok, None], yy)
    return out


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


def self_test():
    """Independent pixel enumeration and known synthetic runs; no file I/O."""
    xy, n, bits = target_center(np.array([[[2,4],[4,6],[np.nan,9]],
                                          [[-1,0],[12,0],[np.inf,1]]]), 12, 10)
    assert np.array_equal(xy[0], [3,5]) and np.isnan(xy[1]).all()
    assert np.array_equal(n, [2,0]) and np.array_equal(bits, [3,0])
    prob = np.zeros((6, 3, 5))
    prob[0, 0, 4] = 7; prob[1, :, :] = .5; prob[2, 1, 2] = -1
    prob[3, 1, 3] = np.nan; prob[5, 2, 4] = np.inf
    pp = probability_xy(prob, 100, 60)
    np.testing.assert_allclose(pp[:2], [[100,0],[50,30]])
    assert np.isnan(pp[2:]).all()
    rng = np.random.default_rng(712)
    w, h, m = 11, 9, 40
    t, l, r = rng.uniform([-1,-1], [w+1,h+1], size=(3,m,2))
    # Include exact tangency, coincident grippers, one-sided and missing rows.
    t[:4] = [[0,.5],[w,h],[3.5,4.5],[4,0]]
    l[:4] = [[0,.5],[w-1,h-1],[3.5,4.5],[5,0]]
    r[:4] = l[:4]
    l[4] = np.nan; r[5:7] = np.nan; l[6] = np.nan
    valid = np.ones(m, bool); valid[7] = False
    yy, xx = np.mgrid[:h,:w] + .5
    raster = np.stack((xx.ravel(), yy.ravel()), axis=1)
    for rt, rp in [(0,0),(.5,1),(1.75,2.5),(4,3.75),(12,15)]:
        a = areas(t,l,r,w,h,rt,rp,valid)
        assert np.isnan(a[[6,7]]).all()
        for i in np.flatnonzero(np.isfinite(a[:,0])):
            hit_t = ((raster-t[i])**2).sum(axis=1) <= rt**2
            hit_p = (((raster-l[i])**2).sum(axis=1) <= rp**2) | (((raster-r[i])**2).sum(axis=1) <= rp**2)
            labels = hit_t.astype(int) + hit_p.astype(int)*2
            np.testing.assert_allclose(a[i], np.bincount(labels, minlength=4)/(w*h))
            coords = [np.broadcast_to(v[i], raster.shape) for v in (t,l,r)]
            assert np.array_equal(labels, classify(raster,*coords,rt,rp,np.ones(w*h,bool)))
    # A perfectly overlapping target and PSM has no exclusive-area denominator.
    a = areas(t[2:3],t[2:3],t[2:3],w,h,2,2,[True])
    assert a[0,1] + a[0,2] == 0 and a[0,3] > 0
    t0 = np.int64(1750430000000000000)
    tns = t0 + np.array([0,20,40,150,170,190],np.int64)*1000000
    dt, br = times(tns, [1,2,3,4,6,7])
    np.testing.assert_allclose(dt,[.01,.02,.01,0,.01,.01])
    assert np.array_equal(br,[1,0,0,1,1,0])
    dur = np.repeat(.05, 17); occ = np.zeros(17,int)
    state = np.array([1,1,1,2,2,2,3,1,1,1,-1,2,2,2,1,1,1])
    br = np.zeros(17,bool); br[0] = True
    out = temporal(state,dur,occ,br)
    assert out["direct_count"] == 2 and out["via_count"] == 1
    np.testing.assert_allclose([out["dwell_T_ms"],out["valid_seconds"]],[150,.8])
    assert out["dwell_target_only_n_censored"] == 3
    # Splitting end-runs at a boundary cannot induce a transition.
    occ[14:] = 1
    assert temporal(state,dur,occ,br)["direct_count"] == 1
    ts = t0 + np.arange(10,dtype=np.int64)*25000000
    raw = np.repeat(np.arange(10,dtype=float)[:,None],2,axis=1)
    raw[[1,3,4,5,6],:] = np.nan
    ob = np.zeros(10,int); br = np.zeros(10,bool); br[0] = True
    smooth, filled = smooth_gaze(ts,raw,ob,br)
    assert np.array_equal(np.flatnonzero(filled),[1]) and np.isnan(smooth[3:7]).all()
    ob[2:] = 1
    assert not smooth_gaze(ts,raw,ob,br)[1].any()
    assert np.isnan(temporal([],[],[],[])["direct_rate"])
    assert areas(np.empty((0,2)),np.empty((0,2)),np.empty((0,2)),11,9,1,2,[]).shape == (0,4)
    return True
