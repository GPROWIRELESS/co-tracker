"""Independent passive Target Locking oracles; no analysis-module imports."""
import csv
from decimal import Decimal
from pathlib import Path

import numpy as np

STAGES = {
    'A': ('reach ring', 'grab ring', 'transfer ring', 'place ring'),
    'C': ('reach string', 'wrap around psm', 'pull through and cinch', 'threading string through tower'),
}
ALIASES = {'rr': 'reach ring', 'gr': 'grab ring', 'tr': 'transfer ring', 'pr': 'place ring',
           'wrap': 'wrap around psm', 'cinch': 'pull through and cinch',
           'thread': 'threading string through tower'}
SCENARIOS = {'main': (50, 50), 'r25': (25, 25), 'r100': (100, 100),
             'gripper100': (50, 100), 'smooth': (50, 50)}


def normalize_stage(values):
    cleaned = [' '.join(str(v).strip().lower().split()) for v in values]
    return np.asarray([ALIASES.get(v, v) for v in cleaned])


def csv_records(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def timeline(path):
    raw = csv_records(path)
    retained = [r for r in raw if r.get('IsDelete', '0').strip() in ('', '0', '0.0')]
    f = np.asarray([int(Decimal(r['Frame_Number'])) for r in retained], dtype=np.int64)
    t = np.asarray([int(Decimal(r['Timestamp(ns)'])) for r in retained], dtype=np.int64)
    assert len(np.unique(f)) == len(f), ('duplicate timeline frames', path)
    return f, t


def join_structured(raw, frames):
    ids = np.asarray(raw['Frame_Number'], np.int64)
    assert len(ids) == len(np.unique(ids)), 'duplicate annotation or gaze frame keys'
    index = {int(f): i for i, f in enumerate(ids)}
    found = np.asarray([int(f) in index for f in frames])
    rows = np.asarray([index.get(int(f), -1) for f in frames], np.int64)
    return found, rows


def native_time(timestamps, frames):
    weights = np.zeros(len(frames), float)
    breaks = np.ones(len(frames), bool)
    for i in range(1, len(frames)):
        delta = (int(timestamps[i]) - int(timestamps[i-1])) / 1e9
        good = timestamps[i] >= 0 and timestamps[i-1] >= 0 and frames[i] == frames[i-1] + 1 and 0 < delta <= .1
        if good:
            weights[i-1] += delta / 2
            weights[i] += delta / 2
            breaks[i] = False
    return weights, breaks


def smooth(timestamps, raw_xy, occurrence, breaks):
    """Rebuild native support using explicit contiguous runs and grid samples."""
    n = len(timestamps)
    output = np.full((n, 2), np.nan)
    if not n:
        return output, np.zeros(0, bool)
    valid = np.isfinite(raw_xy).all(1) & np.isfinite(timestamps) & (timestamps >= 0)
    cuts = [0]
    for i in range(1, n):
        delta = int(timestamps[i]) - int(timestamps[i-1])
        if breaks[i] or occurrence[i] != occurrence[i-1] or timestamps[i] < 0 or timestamps[i-1] < 0 or delta <= 0 or delta > 100000000:
            cuts.append(i)
    cuts.append(n)
    for first, stop in zip(cuts[:-1], cuts[1:]):
        donors = np.flatnonzero(valid[first:stop]) + first
        groups = []
        for donor in donors:
            if not groups or int(timestamps[donor])-int(timestamps[groups[-1][-1]]) > 100000000:
                groups.append([donor])
            else:
                groups[-1].append(donor)
        for group in groups:
            if len(group) == 1:
                output[group[0]] = raw_xy[group[0]]
                continue
            a, b = group[0], group[-1]
            native_t = (timestamps[a:b+1]-timestamps[a]).astype(float)/1e9
            donor_t = (timestamps[group]-timestamps[a]).astype(float)/1e9
            grid_t = np.arange(int(np.ceil(native_t[-1]*41))+1)/41
            for axis in range(2):
                interpolated = np.interp(grid_t, donor_t, raw_xy[group, axis])
                left = np.r_[interpolated[0], interpolated[:-1]]
                right = np.r_[interpolated[1:], interpolated[-1]]
                averaged = (left + interpolated + right)/3
                output[a:b+1, axis] = np.interp(native_t, grid_t, averaged)
    return output, np.isfinite(output).all(1) & ~valid


def temporal(states, weights, occurrence, breaks):
    """Explicit event list, separate from the vectorized production algorithm."""
    names = ('other', 'target_only', 'psm_only', 'overlap')
    per_state = [[] for _ in range(4)]
    all_runs = np.zeros(4, int)
    censored_all = np.zeros(4, int)
    censored_qualified = np.zeros(4, int)
    blocks, block = [], []
    for i, state in enumerate(states):
        if i and (breaks[i] or occurrence[i] != occurrence[i-1]):
            blocks.append(block)
            block = []
        if not block or block[-1]['state'] != int(state):
            block.append(dict(state=int(state), duration=float(weights[i])))
        else:
            block[-1]['duration'] += float(weights[i])
    if block:
        blocks.append(block)
    direct = via = 0
    for block in blocks:
        for j, run in enumerate(block):
            state, duration = run['state'], run['duration']
            if state < 0:
                continue
            censored = j == 0 or j == len(block)-1 or block[j-1]['state'] < 0 or block[j+1]['state'] < 0
            all_runs[state] += 1
            censored_all[state] += int(censored)
            qualified = duration >= .120-1e-12
            if qualified:
                per_state[state].append(duration)
                censored_qualified[state] += int(censored)
            if not qualified or state not in (1, 2):
                continue
            if j > 0 and block[j-1]['state'] == 3-state and block[j-1]['duration'] >= .120-1e-12:
                direct += 1
            elif j > 1 and block[j-1]['state'] == 3 and block[j-2]['state'] == 3-state and block[j-2]['duration'] >= .120-1e-12:
                via += 1
    seconds = float(weights[np.asarray(states) >= 0].sum())
    out = dict(valid_seconds=seconds, direct_count=direct, via_count=via,
               direct_rate=60*direct/seconds if seconds else np.nan,
               via_rate=60*via/seconds if seconds else np.nan)
    for k, name in enumerate(names):
        out['dwell_'+name+'_median_s'] = float(np.median(per_state[k])) if per_state[k] else np.nan
        out['dwell_'+name+'_n'] = len(per_state[k])
        out['dwell_'+name+'_n_censored'] = int(censored_qualified[k])
        out['all_'+name+'_runs'] = int(all_runs[k])
        out['all_'+name+'_runs_censored'] = int(censored_all[k])
    for short, long in zip('TPO', names[1:]):
        out['dwell_'+short+'_ms'] = 1000*out['dwell_'+long+'_median_s']
    return out


def pixel_area(target, left, right, width, height, rt, rp):
    """Literal pixel-center enumeration inside the bounding box of all disks."""
    centers = np.asarray([target, left, right], float)
    radii = np.asarray([rt, rp, rp])
    lower = np.maximum(np.floor((centers-radii[:, None]).min(0)).astype(int), [0, 0])
    upper = np.minimum(np.ceil((centers+radii[:, None]).max(0)).astype(int), [width, height])
    xx, yy = np.meshgrid(np.arange(lower[0], upper[0])+.5, np.arange(lower[1], upper[1])+.5)
    xy = np.stack([xx, yy], -1)
    hit_t = ((xy-target)**2).sum(-1) <= rt**2
    hit_p = (((xy-left)**2).sum(-1) <= rp**2) | (((xy-right)**2).sum(-1) <= rp**2)
    counts = np.bincount((hit_t.astype(int)+2*hit_p.astype(int)).ravel(), minlength=4).astype(float)
    counts[0] += width*height-xx.size
    return counts/(width*height)


def bootstrap_mean(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return dict(ci_low=np.nan, ci_high=np.nan, bootstrap_defined=0, bootstrap_undefined=0)
    rng = np.random.default_rng(np.random.SeedSequence([42, n]))
    weights = rng.multinomial(n, np.ones(n)/n, size=10000)
    samples = (weights*values).sum(1)/n
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return dict(ci_low=float(lo), ci_high=float(hi), bootstrap_defined=10000, bootstrap_undefined=0)
