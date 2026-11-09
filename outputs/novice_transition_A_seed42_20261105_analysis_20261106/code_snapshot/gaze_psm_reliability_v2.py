"""Occurrence sufficient statistics and exploratory peak reliability.

No trajectories are spliced during resampling.  The numerical kernel is the
versioned sibling ``gaze_psm_core_legacy.py``.  Peak bootstrap ranges are NOT
coverage-calibrated confidence intervals.  Near-peak sets are descriptive.
Adjacent-occurrence moving blocks preserve only local order; they do not make
nonstationary occurrences exchangeable.  This limitation is exported in results.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.signal import find_peaks
from gaze_psm_core_legacy import _fft_statistics


def occurrence_statistics(gaze, psm, occurrence_ids=None, hz=41., max_lag=2.,
                          within_stage=False):
    """Return terms[K,4,L] in count/covariance/gaze SS/PSM SS order.

    Every supplied nonnegative occurrence ID remains in the result, even if
    unusable.  This preserves paired resampling identities across gaze sources.
    """
    gaze, psm = np.asarray(gaze, float), np.asarray(psm, float)
    if gaze.ndim != 2 or gaze.shape[1:] != (2,) or psm.shape != gaze.shape:
        raise ValueError('gaze and psm must have identical (N,2) shapes')
    if not np.isfinite([hz, max_lag]).all() or hz <= 0 or max_lag < 0:
        raise ValueError('invalid sampling rate or lag window')
    n = len(gaze)
    ids = np.zeros(n, dtype=np.int64) if occurrence_ids is None else np.asarray(occurrence_ids)
    if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('occurrence_ids must be a length-N integer array')
    identities, first = np.unique(ids[ids >= 0], return_index=True)
    # Input IDs are normally increasing; temporal ordering also handles relabels.
    identities = identities[np.argsort(first)]
    maximum = int(round(max_lag * hz))
    lags = np.arange(-maximum, maximum + 1, dtype=np.int64)
    terms = np.zeros((len(identities), 4, len(lags)), dtype=float)
    gf, pf = np.isfinite(gaze).all(1), np.isfinite(psm).all(1)
    for j, identity in enumerate(identities):
        members = np.flatnonzero(ids == identity)
        if len(members) < 2:
            continue
        start, end = max(0, members[0] - maximum), min(n, members[-1] + maximum + 1)
        local_ids = ids[start:end]
        pm = pf[start:end] & (local_ids == identity)
        gm = gf[start:end].copy()
        if within_stage:
            gm &= local_ids == identity
        if pm.sum() < 2 or gm.sum() < 2:
            continue
        raw = np.asarray(_fft_statistics(gaze[start:end], psm[start:end], gm, pm, lags))
        usable = raw[0] >= 2
        terms[j, :, usable] = raw[:, usable].T
    return {'lag_s': lags / hz, 'ids': identities.astype(np.int64),
            'terms': terms, 'hz': float(hz), 'within_stage': bool(within_stage)}


def _from_totals(total, occurrence_counts, lags, hz, min_overlap, min_rms):
    counts, covariance, vg, vp = total
    safe = np.maximum(counts, 1.)
    vg, vp = np.maximum(vg, 0.), np.maximum(vp, 0.)
    gr, pr = np.sqrt(vg / safe), np.sqrt(vp / safe)
    gr = np.where(counts > 0, gr, np.nan)
    pr = np.where(counts > 0, pr, np.nan)
    denominator = np.sqrt(vg * vp)
    supported = counts >= max(2, int(np.ceil(min_overlap * hz)))
    supported &= (gr >= min_rms) & (pr >= min_rms)
    supported &= denominator > np.finfo(float).eps
    r = np.full(counts.shape, np.nan)
    np.divide(covariance, denominator, out=r, where=supported)
    r = np.clip(r, -1., 1.)
    return {'lag_s': np.asarray(lags), 'r': r, 'pairs': counts,
            'occurrences': occurrence_counts, 'gaze_rms': gr, 'psm_rms': pr}


def aggregate_statistics(stats, weights=None, min_overlap=2., min_rms=1.):
    """Pool centered sums, reproducing the legacy estimator exactly."""
    terms = np.asarray(stats['terms'], float)
    if terms.ndim != 3 or terms.shape[1] != 4:
        raise ValueError('terms must have shape (K,4,L)')
    weights = np.ones(len(terms)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(terms),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('weights must be finite nonnegative occurrence weights')
    if min_overlap < 0 or min_rms < 0:
        raise ValueError('support thresholds must be nonnegative')
    total = np.einsum('k,kvl->vl', weights, terms)
    occurrences = weights @ (terms[:, 0] >= 2)
    return _from_totals(total, occurrences, stats['lag_s'], stats['hz'], min_overlap, min_rms)


def _number(value):
    return float(value) if np.isfinite(value) else None


def _intervals(mask, lags):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    return [[float(lags[a]), float(lags[b - 1])] for a, b in bounds]


def peak_features(curve):
    """Scalar features and JSON strings, with strict-JSON None for missing values."""
    r, lags = np.asarray(curve['r'], float), np.asarray(curve['lag_s'], float)
    if r.ndim != 1 or lags.shape != r.shape or not len(r):
        raise ValueError('r and lag_s must be matching nonempty vectors')
    good = np.isfinite(r)
    out = {'full_support': bool(good.all()), 'full_lag_support': bool(good.all()),
           'n_supported_lags': int(good.sum()), 'peak_lag_s': None, 'peak_r': None,
           'r_zero': _number(r[np.argmin(abs(lags))]), 'peak_gain_over_zero': None,
           'peak_at_boundary': False, 'peak_at_support_edge': False,
           'positive_peak': False, 'tied_maxima_count': 0, 'tied_maxima_span_s': None,
           'n_local_maxima': 0, 'second_peak_lag_s': None, 'second_peak_r': None,
           'second_peak_gap_r': None, 'second_peak_distance_s': None}
    for tag in ('0p01', '0p05', '0p10'):
        out.update({f'near_peak_intervals_{tag}': '[]', f'near_peak_components_{tag}': 0,
                    f'near_peak_span_{tag}_s': None, f'near_peak_contiguous_width_{tag}_s': None,
                    f'near_peak_fraction_{tag}': None})
    if not good.any():
        return out
    i = int(np.nanargmax(r))
    ties = good & np.isclose(r, r[i], atol=1e-12, rtol=0.)
    tied_ids = np.flatnonzero(ties)
    local = find_peaks(np.where(good, r, -np.inf))[0].tolist()
    if good[0] and (len(r) == 1 or not good[1] or r[0] >= r[1]):
        local.append(0)
    if len(r) > 1 and good[-1] and (not good[-2] or r[-1] >= r[-2]):
        local.append(len(r) - 1)
    local = sorted(set(local))
    competitors = [j for j in local if abs(lags[j] - lags[i]) >= .2 - 1e-12]
    second = max(competitors, key=lambda j: r[j]) if competitors else None
    out.update(peak_lag_s=float(lags[i]), peak_r=float(r[i]), positive_peak=bool(r[i] > 0),
               peak_gain_over_zero=_number(r[i] - r[np.argmin(abs(lags))]),
               peak_at_boundary=bool(i in (0, len(r) - 1)),
               peak_at_support_edge=bool(i == np.flatnonzero(good)[0] or i == np.flatnonzero(good)[-1]),
               tied_maxima_count=int(ties.sum()), tied_maxima_span_s=float(lags[tied_ids[-1]] - lags[tied_ids[0]]),
               n_local_maxima=len(local))
    if second is not None:
        out.update(second_peak_lag_s=float(lags[second]), second_peak_r=float(r[second]),
                   second_peak_gap_r=float(r[i] - r[second]),
                   second_peak_distance_s=float(abs(lags[i] - lags[second])))
    for epsilon, tag in ((.01, '0p01'), (.05, '0p05'), (.1, '0p10')):
        near = good & (r >= r[i] - epsilon)
        intervals = _intervals(near, lags)
        segment = next(x for x in intervals if x[0] <= lags[i] <= x[1])
        out.update({f'near_peak_intervals_{tag}': json.dumps(intervals, separators=(',', ':')),
                    f'near_peak_components_{tag}': len(intervals),
                    f'near_peak_span_{tag}_s': float(intervals[-1][1] - intervals[0][0]),
                    f'near_peak_contiguous_width_{tag}_s': float(segment[1] - segment[0]),
                    f'near_peak_fraction_{tag}': float(near.sum() / good.sum())})
    return out


def block_weights(k, count, seed=42, block_length=2):
    """Moving non-circular occurrence blocks; exactly k draws per replicate.

    Occurrence *statistics* are reweighted, never raw trajectories concatenated.
    Boundary occurrences have the usual non-circular moving-block edge effects.
    """
    if k < 1 or count < 0 or block_length < 1:
        raise ValueError('invalid block sampling settings')
    block_length = min(k, int(block_length))
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(k / block_length))
    starts = rng.integers(0, k - block_length + 1, size=(count, nblocks))
    indices = (starts[..., None] + np.arange(block_length)).reshape(count, -1)[:, :k]
    weights = np.zeros((count, k), dtype=np.int64)
    np.add.at(weights, (np.repeat(np.arange(count), k), indices.ravel()), 1)
    return weights


def _sample_curves(stats, weights, min_overlap, min_rms):
    terms = np.asarray(stats['terms'], float)
    total = np.einsum('bk,kvl->bvl', weights, terms)
    # _from_totals expects the variable dimension first.
    return _from_totals(total.transpose(1, 0, 2), weights @ (terms[:, 0] >= 2),
                        stats['lag_s'], stats['hz'], min_overlap, min_rms)['r']


def _peaks(r, lags, require_full=True):
    good = np.isfinite(r)
    valid = good.all(1) if require_full else good.any(1)
    indices = np.argmax(np.where(good, r, -np.inf), axis=1)
    return np.where(valid, np.asarray(lags)[indices], np.nan), valid, indices


def reliability(stats, bootstrap=500, seed=42, block_length=2, precision_s=.4,
                min_overlap=2., min_rms=1.):
    """Exploratory within-record diagnostics, not a calibrated inference test.

    At least four registered AND contributing occurrences are required for block
    bootstrap.  This is an operational reporting minimum, not a sufficiency
    guarantee.  Reliability requires full lag support, an interior positive-
    or negative-valued localizable peak, no close competing near-peak component,
    and a resampling range no wider than precision_s.  Peak positivity remains
    a separate coupling diagnostic, never an input eligibility criterion.
    """
    curve = aggregate_statistics(stats, min_overlap=min_overlap, min_rms=min_rms)
    f = peak_features(curve)
    k = len(stats['ids'])
    contributing = int(np.any(np.asarray(stats['terms'])[:, 0] >= 2, axis=1).sum())
    s = dict(f)
    s.update(n_registered_occurrences=k, n_contributing_occurrences=contributing,
             bootstrap_requested=int(bootstrap), bootstrap_valid_replicates=0,
             bootstrap_valid_fraction=None, bootstrap_range_low_s=None, bootstrap_range_high_s=None,
             bootstrap_range_width_s=None, bootstrap_boundary_fraction=None,
             bootstrap_peak_distribution='not_available', uncertainty_calibrated=False,
             uncertainty_method='noncircular_adjacent_occurrence_block_bootstrap_percentiles',
             block_length=int(block_length), operational_precision_s=float(precision_s),
             uncertainty_limit='Exploratory resampling range; not a calibrated 95% confidence interval; occurrence stationarity is not established.',
             loo_valid_replicates=0, loo_max_abs_peak_change_s=None, loo_peak_lags_s='[]',
             first_half_peak_lag_s=None, second_half_peak_lag_s=None, half_peak_difference_s=None)
    lags = np.asarray(stats['lag_s'])
    samples = np.array([], float)
    reasons = []
    if not f['full_support']:
        reasons.append('insufficient_support')
    if f['peak_at_boundary']:
        reasons.append('boundary_limited')
    if f['tied_maxima_count'] > 1 and (f['tied_maxima_span_s'] or 0) > 1.5 / stats['hz']:
        reasons.append('tied_maxima')
    if f['near_peak_components_0p01'] > 1 and f['second_peak_gap_r'] is not None and f['second_peak_gap_r'] <= .01 + 1e-12:
        reasons.append('competing_peaks')
    if f['near_peak_contiguous_width_0p01_s'] is not None and f['near_peak_contiguous_width_0p01_s'] > precision_s:
        reasons.append('broad_near_peak')
    if k >= 2:
        loo_weights = np.ones((k, k)) - np.eye(k)
        loo, loo_good, _ = _peaks(_sample_curves(stats, loo_weights, min_overlap, min_rms), lags)
        s['loo_valid_replicates'] = int(loo_good.sum())
        s['loo_peak_lags_s'] = json.dumps([_number(x) for x in loo], separators=(',', ':'))
        if loo_good.any() and f['peak_lag_s'] is not None:
            s['loo_max_abs_peak_change_s'] = float(np.max(abs(loo[loo_good] - f['peak_lag_s'])))
        split = k // 2
        half_weights = np.zeros((2, k)); half_weights[0, :split] = 1; half_weights[1, split:] = 1
        halves, _, _ = _peaks(_sample_curves(stats, half_weights, min_overlap, min_rms), lags)
        s['first_half_peak_lag_s'], s['second_half_peak_lag_s'] = map(_number, halves)
        s['half_peak_difference_s'] = _number(abs(halves[1] - halves[0]))
    if k < 4 or contributing < 4:
        reasons.append('insufficient_repeats')
    elif bootstrap > 0:
        weights = block_weights(k, bootstrap, seed, block_length)
        samples, valid, indices = _peaks(_sample_curves(stats, weights, min_overlap, min_rms), lags)
        s['bootstrap_valid_replicates'] = int(valid.sum())
        s['bootstrap_valid_fraction'] = float(valid.mean())
        if valid.any():
            low, high = np.percentile(samples[valid], [2.5, 97.5])
            s.update(bootstrap_range_low_s=float(low), bootstrap_range_high_s=float(high),
                     bootstrap_range_width_s=float(high - low),
                     bootstrap_boundary_fraction=float(np.mean((indices[valid] == 0) | (indices[valid] == len(lags) - 1))))
            unique, counts = np.unique(samples[valid], return_counts=True)
            s['bootstrap_peak_distribution'] = json.dumps([[float(x), int(n)] for x, n in zip(unique, counts)], separators=(',', ':'))
            if high - low > precision_s:
                reasons.append('broad_uncertainty')
            if s['bootstrap_boundary_fraction'] > .05:
                reasons.append('bootstrap_boundary_instability')
        if valid.mean() < .95:
            reasons.append('incomplete_bootstrap_support')
    else:
        reasons.append('bootstrap_not_run')
    # Split/LOO instability is shown separately and used as a conservative flag.
    if s['loo_max_abs_peak_change_s'] is not None and s['loo_max_abs_peak_change_s'] > precision_s:
        reasons.append('leave_one_occurrence_instability')
    if s['half_peak_difference_s'] is not None and s['half_peak_difference_s'] > precision_s:
        reasons.append('temporal_half_instability')
    s['reliability_reasons'] = ';'.join(reasons)
    s['lag_identifiable'] = not reasons
    s['lag_status'] = reasons[0] if reasons else 'interior_stable_exploratory'
    for delta, tag in ((.05, '0p05'), (.1, '0p10'), (.2, '0p20')):
        peak = f['peak_lag_s']
        s[f'point_direction_{tag}'] = ('undefined' if peak is None else 'lead' if peak > delta else 'lag' if peak < -delta else 'synchronous')
        low, high = s['bootstrap_range_low_s'], s['bootstrap_range_high_s']
        direction = 'undetermined'
        if s['lag_identifiable'] and f['positive_peak'] and low is not None:
            direction = 'lead' if low > delta else 'lag' if high < -delta else 'synchronous' if low >= -delta and high <= delta else 'undetermined'
        s[f'direction_{tag}'] = direction
    return {'summary': s, 'peak_samples': samples}


def run_validation_simulations(count=200, seed=914):
    """Small reproducible diagnostic simulation, not empirical-data calibration.

    Tests separated stationary occurrences with smooth colored signal, missing
    samples, broad ramps, repeated periodic peaks and independent colored noise.
    Coverage is exported with Monte Carlo SE and never asserted to equal 95%.
    """
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(seed)
    hz, maximum, n = 20., 1., 100
    rows = []
    scenarios = [('identified_delay', 6, 6), ('zero_delay', 0, 6), ('independent_noise', None, 6),
                 ('broad_ramp', 0, 6), ('periodic_multipeak', 0, 6),
                 ('identified_delay_n3', 6, 3), ('noisy_delay_n5', 6, 5), ('boundary_delay', 24, 5)]
    for scenario, delay, k in scenarios:
        errors, covered, identifiable, lead, boundary = [], [], [], [], []
        for _ in range(count):
            length = k * (n + 40)
            g = gaussian_filter1d(rng.normal(size=(length, 2)), 2, axis=0) * 50
            if scenario == 'broad_ramp':
                t = np.arange(length) / hz
                g = np.column_stack([t, 2 * t]) * 10
            elif scenario == 'periodic_multipeak':
                t = np.arange(length) / hz
                g = np.column_stack([np.sin(2*np.pi*t/.5), np.cos(2*np.pi*t/.5)]) * 50
            if delay is None:
                p = gaussian_filter1d(rng.normal(size=g.shape), 2, axis=0) * 50
            else:
                noise_scale = 1 if scenario == 'broad_ramp' else 25 if scenario == 'noisy_delay_n5' else 5
                p = np.roll(g, delay, axis=0) + rng.normal(size=g.shape) * noise_scale
                if delay > 0:
                    p[:delay] = np.nan
            ids = np.full(length, -1, dtype=np.int64)
            for j in range(k):
                ids[j*(n+40)+20:j*(n+40)+20+n] = j
            missing = rng.random(length) < (.1 if scenario == 'noisy_delay_n5' else .015)
            g[missing] = np.nan
            stats = occurrence_statistics(g, p, ids, hz, maximum)
            out = reliability(stats, bootstrap=100, seed=int(rng.integers(2**31)), min_overlap=1)
            s = out['summary']
            identifiable.append(s['lag_identifiable'])
            lead.append(s['direction_0p10'] == 'lead')
            boundary.append(s['peak_at_boundary'])
            if delay is not None and s['peak_lag_s'] is not None:
                truth = delay / hz
                errors.append(abs(s['peak_lag_s'] - truth))
                if s['bootstrap_range_low_s'] is not None:
                    covered.append(s['bootstrap_range_low_s'] <= truth <= s['bootstrap_range_high_s'])
        coverage = float(np.mean(covered)) if covered else None
        # Wilson interval avoids reporting zero Monte Carlo uncertainty when
        # observed coverage is exactly zero or one.
        wilson_low = wilson_high = None
        if covered:
            nn = len(covered); zz = 1.959963984540054 ** 2
            center = (coverage + zz/(2*nn))/(1+zz/nn)
            radius = 1.959963984540054*np.sqrt(coverage*(1-coverage)/nn+zz/(4*nn*nn))/(1+zz/nn)
            wilson_low, wilson_high = float(max(0, center-radius)), float(min(1, center+radius))
        rows.append({'scenario': scenario, 'replicates': int(count), 'registered_occurrences': k,
                     'true_delay_s': None if delay is None else delay/hz,
                     'mean_abs_peak_error_s': float(np.mean(errors)) if errors else None,
                     'exploratory_range_coverage': coverage, 'coverage_replicates': len(covered),
                     'coverage_mc_se': float(np.sqrt(coverage*(1-coverage)/len(covered))) if covered else None,
                     'coverage_wilson_low': wilson_low, 'coverage_wilson_high': wilson_high,
                     'operational_identifiable_fraction': float(np.mean(identifiable)),
                     'lead_classification_fraction': float(np.mean(lead)),
                     'boundary_fraction': float(np.mean(boundary)),
                     'claim': 'diagnostic_only_not_calibration_of_empirical_confidence_intervals'})
    return rows
