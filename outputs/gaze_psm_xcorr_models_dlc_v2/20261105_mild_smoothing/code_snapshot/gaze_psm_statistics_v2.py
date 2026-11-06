"""Participant-level summaries for exploratory fixed-instrument gaze/PSM v2.

No peak-sign, boundary, or localizability filtering is used to define the
technical comparison set. Resampling ranges are explicitly uncalibrated for
small participant counts. All permutations act on complete participant blocks.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from collections import defaultdict
from typing import Iterable
import numpy as np

GROUP = ('variant', 'task', 'side', 'signal', 'stage', 'mode')
OBSERVED = '__observed_stage_average__'
METRICS = ('curve_mae', 'delta_lag_s', 'abs_lag_difference_s')


def _truth(value):
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value) if value is not None else False


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def _mean(values):
    x = np.asarray(list(values), float)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else float('nan')


def _stable_seed(seed, key):
    return int.from_bytes(hashlib.sha256(repr((int(seed), key)).encode()).digest()[:8], 'little')


def _eligible(row):
    return _truth(row.get('common_input', row.get('input_eligible', False)))


def _key(row, fields=GROUP):
    return tuple(row.get(f, 'primary' if f == 'variant' else '') for f in fields)


def _fisher(arrays):
    arrays = np.asarray(arrays, float)
    with np.errstate(invalid='ignore'):
        z = np.arctanh(np.clip(arrays, -1 + 1e-7, 1 - 1e-7))
    count = np.isfinite(z).sum(axis=0)
    return np.tanh(np.divide(np.nansum(z, axis=0), count,
                           out=np.full(z.shape[1:], np.nan), where=count > 0))


def _person_means(rows, metric):
    """Record equal within stage; observed stages equal within participant."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if _eligible(r):
            by[str(r['user'])][str(r['stage'])].append(_number(r.get(metric)))
    return {u: _mean(_mean(v) for v in stages.values()) for u, stages in by.items()}


def _resample_mean(values, count, rng):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or count <= 0:
        return float('nan'), float('nan')
    ix = rng.integers(0, len(values), (count, len(values)))
    return tuple(float(v) for v in np.percentile(values[ix].mean(axis=1), [2.5, 97.5]))


def summarize_pairs(rows, bootstrap=2000, seed=42):
    """Summarize technical common support, retaining all lag result states.

    Returns stage-specific and explicitly observed-stage-average summaries.
    Ranges resample participant means; they do not claim validated 95% coverage.
    """
    groups = defaultdict(list)
    for row in rows:
        key = (*_key(row), row['model'])
        groups[key].append(row)
        global_key = list(key)
        global_key[4] = OBSERVED
        groups[tuple(global_key)].append(row)
    out = []
    for key, all_rows in sorted(groups.items()):
        good = [r for r in all_rows if _eligible(r)]
        result = dict(zip((*GROUP, 'model'), key))
        total_users = sorted({str(r['user']) for r in all_rows})
        users = sorted({str(r['user']) for r in good})
        result.update(n_total_users=len(total_users), n_users=len(users),
                      n_total_conditions=len(all_rows), n_conditions=len(good),
                      input_coverage=len(good) / len(all_rows) if all_rows else np.nan,
                      n_gt_localizable=sum(_truth(r.get('gt_localizable')) for r in good),
                      n_model_localizable=sum(_truth(r.get('model_localizable')) for r in good),
                      n_both_localizable=sum(_truth(r.get('gt_localizable')) and _truth(r.get('model_localizable')) for r in good),
                      range_method='participant_percentile_resampling_uncalibrated',
                      resamples=int(bootstrap),
                      weighting='record_equal_within_stage;observed_stage_equal_within_user;user_equal',
                      participants_json=json.dumps(users),
                      stage_coverage_json=json.dumps({u: sorted({r['stage'] for r in good if str(r['user']) == u}) for u in users}))
        result['gt_localizable_rate'] = result['n_gt_localizable'] / len(good) if good else np.nan
        result['model_localizable_rate'] = result['n_model_localizable'] / len(good) if good else np.nan
        result['both_localizable_rate'] = result['n_both_localizable'] / len(good) if good else np.nan
        for which in ('gt', 'model'):
            result[which + '_boundary_rate'] = _mean(
                _truth(r.get(which + '_boundary', r.get(which + '_peak_at_boundary', 'boundary' in str(r.get(which + '_status', ''))))) for r in good)
        for metric in METRICS:
            means = _person_means(good, metric)
            vals = np.array([means[u] for u in users], float)
            result['participant_equal_' + metric] = _mean(vals)
            result[metric + '_n_users'] = int(np.isfinite(vals).sum())
            rng = np.random.default_rng(_stable_seed(seed, (key[:-1], tuple(users))))
            lo, hi = _resample_mean(vals, bootstrap, rng)
            result[metric + '_resample_low'] = lo
            result[metric + '_resample_high'] = hi
            result[metric + '_user_values_json'] = json.dumps(means)
            finite = vals[np.isfinite(vals)]
            if len(finite) >= 2:
                loo = [(finite.sum()-v)/(len(finite)-1) for v in finite]
                result[metric + '_leave_one_user_min'] = float(min(loo))
                result[metric + '_leave_one_user_max'] = float(max(loo))
            else:
                result[metric + '_leave_one_user_min'] = np.nan
                result[metric + '_leave_one_user_max'] = np.nan
        local = [r for r in good if _truth(r.get('gt_localizable')) and _truth(r.get('model_localizable'))]
        result['conditional_localizable_abs_lag_difference_s'] = _mean(_person_means(local, 'abs_lag_difference_s').values())
        result['conditional_localizable_n_users'] = len({r['user'] for r in local})
        result['summary_status'] = 'exploratory_small_n' if len(users) >= 3 else 'insufficient_users_for_group_inference'
        out.append(result)
    return out


def group_curves(curves, bootstrap=1000, seed=42, min_users=3):
    """Fisher-mean curves with fixed complete-record and participant support."""
    grouped = defaultdict(list)
    for curve in curves:
        grouped[(*_key(curve), curve['source'])].append(curve)
    summaries, long = [], []
    for key, records in sorted(grouped.items()):
        meta = dict(zip((*GROUP, 'source'), key))
        lag = np.asarray(records[0]['lag_s'], float)
        by_user = defaultdict(list)
        for rec in records:
            if not np.array_equal(np.asarray(rec['lag_s']), lag):
                raise ValueError('Group has differing lag grids')
            arr = np.asarray(rec['r'], float)
            if np.isfinite(arr).all():
                by_user[str(rec['user'])].append(arr)
        users = sorted(by_user)
        aa = np.array([_fisher(by_user[u]) for u in users])
        n = len(users)
        r = _fisher(aa) if n >= min_users else np.full(len(lag), np.nan)
        low = high = np.full(len(lag), np.nan)
        peak_lo = peak_hi = boundary_rate = np.nan
        if n >= min_users and bootstrap > 0:
            rng = np.random.default_rng(_stable_seed(seed, (key[:-1], tuple(users))))
            indices = rng.integers(0, n, (bootstrap, n))
            zz = np.arctanh(np.clip(aa, -1+1e-7, 1-1e-7))
            reps = np.tanh(zz[indices].mean(axis=1))
            low, high = np.percentile(reps, [2.5, 97.5], axis=0)
            pi = reps.argmax(axis=1)
            peak_lo, peak_hi = np.percentile(lag[pi], [2.5, 97.5])
            boundary_rate = float(np.mean((pi == 0) | (pi == len(lag)-1)))
        good = np.isfinite(r)
        i = int(np.nanargmax(r)) if good.any() else None
        z = int(np.argmin(abs(lag)))
        summaries.append(dict(meta, n_total_users=len({r['user'] for r in records}),
            n_stable_users=n, n_complete_records=sum(len(x) for x in by_user.values()),
            participants_json=json.dumps(users),
            peak_lag_s=float(lag[i]) if i is not None else np.nan,
            peak_r=float(r[i]) if i is not None else np.nan,
            r_zero=float(r[z]),
            peak_at_boundary=bool(i in (0, len(lag)-1)) if i is not None else False,
            peak_lag_resample_low=float(peak_lo), peak_lag_resample_high=float(peak_hi),
            bootstrap_boundary_frequency=boundary_rate,
            status='exploratory_small_n' if n >= min_users else 'insufficient_users',
            range_method='participant_percentile_resampling_uncalibrated', n_bootstrap=bootstrap))
        for j, t in enumerate(lag):
            long.append(dict(meta, lag_s=float(t), r=float(r[j]),
                r_resample_low=float(low[j]), r_resample_high=float(high[j]), n_stable_users=n))
    return summaries, long


def _holm(rows, field='p_exact'):
    valid = [(i, _number(r.get(field))) for i, r in enumerate(rows)]
    valid = [(i, p) for i, p in valid if np.isfinite(p)]
    previous = 0.
    for rank, (i, p) in enumerate(sorted(valid, key=lambda x: x[1])):
        previous = max(previous, min(1., p * (len(valid)-rank)))
        rows[i]['p_holm'] = previous
    for r in rows:
        r.setdefault('p_holm', np.nan)


def exact_model_comparisons(paired_rows, primary_variant='primary'):
    """Exact two-sided participant-block model label swaps on curve MAE.

    Holm families include all stages + observed-stage aggregate and all model
    pairs within one task/side. Conditional exchangeability is an assumption.
    """
    grouped = defaultdict(list)
    for r in paired_rows:
        if r.get('variant', 'primary') != primary_variant or r.get('signal') != 'position' or r.get('mode') != 'psm_stage':
            continue
        key = _key(r)
        grouped[key].append(r)
        q = list(key); q[4] = OBSERVED
        grouped[tuple(q)].append(r)
    out = []
    for key, data in sorted(grouped.items()):
        models = sorted({r['model'] for r in data})
        lookup = {m: {(r['user'], r['content_id'], r['stage']): r for r in data if r['model'] == m and _eligible(r) and np.isfinite(_number(r.get('curve_mae')))} for m in models}
        for a, b in itertools.combinations(models, 2):
            common = sorted(set(lookup[a]) & set(lookup[b]))
            aa, bb = [lookup[a][k] for k in common], [lookup[b][k] for k in common]
            am, bm = _person_means(aa, 'curve_mae'), _person_means(bb, 'curve_mae')
            users = sorted(set(am) & set(bm))
            diff = np.array([bm[u] - am[u] for u in users])
            n = len(diff)
            if n > 20:
                raise ValueError('Exact participant sign enumeration limited to 20 people')
            observed = float(diff.mean()) if n else np.nan
            signs = np.array(list(itertools.product((-1., 1.), repeat=n))) if n else np.empty((0, 0))
            vals = (signs * diff).mean(axis=1) if n else np.array([])
            p = float(np.mean(np.abs(vals) >= abs(observed)-1e-12)) if n else np.nan
            out.append(dict(zip(GROUP, key), model_a=a, model_b=b, n_users=n,
                n_common_conditions=len(common), mean_curve_mae_b_minus_a=observed,
                p_exact=p, n_permutations=2**n if n else 0,
                minimum_two_sided_p=2/(2**n) if n else np.nan,
                participants_json=json.dumps(users),
                assumption='participant-block model-label exchangeability under null',
                status='exploratory_small_n' if n >= 3 else 'insufficient_users',
                family=f'{key[1]}/{key[2]}/all_stages_and_observed_average/all_model_pairs'))
    families = defaultdict(list)
    for r in out:
        families[r['family']].append(r)
    for rows in families.values():
        _holm(rows)
    return out


def identity_reference(curves, primary_variant='primary'):
    """Whole-person matching reference preserving task, side, and stage.

    Uses only stage profiles complete for every retained participant and all
    sources. Every participant is mapped as one block across those stages;
    permutation maps are identical across models. All n! permutations include
    identity and fixed points. This is an exchangeability-conditional reference,
    not evidence for causal leading or generic independence.
    """
    grouped = defaultdict(list)
    for c in curves:
        if c.get('variant', 'primary') == primary_variant and c.get('signal') == 'position' and c.get('mode') == 'psm_stage':
            grouped[(c['task'], c['side'])].append(c)
    out = []
    for (task, side), data in sorted(grouped.items()):
        source_names = sorted({c['source'] for c in data})
        models = [m for m in source_names if m != 'gt']
        people = sorted({str(c['user']) for c in data})
        stages = sorted({c['stage'] for c in data})
        cell = defaultdict(list)
        lag = np.asarray(data[0]['lag_s'], float)
        for c in data:
            r = np.asarray(c['r'], float)
            if np.array_equal(lag, c['lag_s']) and np.isfinite(r).all():
                cell[(str(c['user']), c['stage'], c['source'])].append(r)
        common_stages = [s for s in stages if all((u, s, src) in cell for u in people for src in source_names)]
        n = len(people)
        if not common_stages or n < 3:
            for model in models:
                out.append(dict(variant=primary_variant, task=task, side=side, model=model,
                    n_users=n, n_common_stages=len(common_stages), stages_json=json.dumps(common_stages),
                    status='insufficient_common_profile', n_permutations=0,
                    observed_curve_mae=np.nan, null_mean_curve_mae=np.nan,
                    empirical_lower_tail_rank=np.nan, observed_abs_lag_difference_s=np.nan,
                    null_mean_abs_lag_difference_s=np.nan,lag_empirical_lower_tail_rank=np.nan))
            continue
        if n > 8:
            raise ValueError('Exact identity reference capped at 8 people; use predeclared Monte Carlo beyond this')
        arrays = {src: np.array([[_fisher(cell[(u, s, src)]) for s in common_stages] for u in people]) for src in source_names}
        perms = list(itertools.permutations(range(n)))
        for model in models:
            # stage means and lag means equally weighted, then participant equal.
            cost = np.mean(np.abs(arrays['gt'][:, None] - arrays[model][None, :]), axis=(2, 3))
            observed = float(np.diag(cost).mean())
            gt_peaks=lag[arrays['gt'].argmax(axis=2)]
            model_peaks=lag[arrays[model].argmax(axis=2)]
            lag_cost=np.abs(gt_peaks[:,None,:]-model_peaks[None,:,:]).mean(axis=2)
            observed_lag=float(np.diag(lag_cost).mean())
            null_lag=np.array([lag_cost[np.arange(n),p].mean() for p in perms])
            null = np.array([cost[np.arange(n), p].mean() for p in perms])
            out.append(dict(variant=primary_variant, task=task, side=side, model=model,
                n_users=n, n_common_stages=len(common_stages), stages_json=json.dumps(common_stages),
                participants_json=json.dumps(people), n_permutations=len(perms),
                observed_curve_mae=observed, null_mean_curve_mae=float(null.mean()),
                null_q025=float(np.percentile(null, 2.5)), null_q975=float(np.percentile(null, 97.5)),
                empirical_lower_tail_rank=float(np.mean(null <= observed+1e-12)),
                observed_abs_lag_difference_s=observed_lag,
                null_mean_abs_lag_difference_s=float(null_lag.mean()),
                null_abs_lag_q025=float(np.percentile(null_lag,2.5)),
                null_abs_lag_q975=float(np.percentile(null_lag,97.5)),
                lag_empirical_lower_tail_rank=float(np.mean(null_lag<=observed_lag+1e-12)),
                lag_reference_definition='descriptive_statistical_peak_matching;no_peak_reliability_filter;not_true_latency_null',
                status='exploratory_exchangeability_conditional_reference',
                assumption='participant whole-profile labels exchangeable under null; stages and side fixed',
                scope='complete_stage_profile_for_all_sources_and_all_task_participants',
                permutation_definition='all_n_factorial_maps_including_identity_same_maps_for_each_model'))
    return out


def paired_group_differences(curves, bootstrap=1000, seed=42):
    """Group-curve peak differences on shared complete-lag participants.

    Different model/GT pairs can have different shared support. This estimand
    is distinct from mean absolute record peak differences. Result-based peak
    filtering is never applied. Bootstrap ranges are exploratory, uncalibrated.
    """
    grouped = defaultdict(list)
    for c in curves:
        grouped[_key(c)].append(c)
    out = []
    for key, data in sorted(grouped.items()):
        lag = np.asarray(data[0]['lag_s'], float)
        cells = defaultdict(list)
        for c in data:
            if not np.array_equal(lag, c['lag_s']):
                raise ValueError('Paired group lag grid mismatch')
            rr = np.asarray(c['r'], float)
            if np.isfinite(rr).all():
                cells[(c['source'], str(c['user']))].append(rr)
        sources = sorted({c['source'] for c in data} - {'gt'})
        for model in sources:
            users = sorted({u for src, u in cells if src == 'gt'} & {u for src, u in cells if src == model})
            row = dict(zip(GROUP, key), model=model, n_shared_users=len(users),
                shared_users_json=json.dumps(users), n_bootstrap=int(bootstrap),
                range_method='paired_participant_percentile_resampling_uncalibrated',
                estimand='difference_of_group_curve_peak_locations_model_minus_gt',
                support_definition='pair_specific_shared_complete_record_participants')
            fields = ('gt_peak_lag_s','model_peak_lag_s','delta_group_peak_lag_s',
                'delta_group_peak_r','delta_group_r_zero','delta_group_peak_lag_resample_low',
                'delta_group_peak_lag_resample_high','delta_group_peak_r_resample_low',
                'delta_group_peak_r_resample_high','gt_bootstrap_boundary_frequency',
                'model_bootstrap_boundary_frequency')
            row.update({f:np.nan for f in fields})
            row.update(gt_boundary=False, model_boundary=False)
            if len(users) < 3:
                row['status']='insufficient_shared_users'
                out.append(row);continue
            aa=np.array([_fisher(cells[('gt',u)]) for u in users])
            bb=np.array([_fisher(cells[(model,u)]) for u in users])
            ga,gb=_fisher(aa),_fisher(bb)
            ia,ib=int(ga.argmax()),int(gb.argmax());z=int(np.argmin(abs(lag)))
            row.update(gt_peak_lag_s=float(lag[ia]),model_peak_lag_s=float(lag[ib]),
                delta_group_peak_lag_s=float(lag[ib]-lag[ia]),
                delta_group_peak_r=float(gb[ib]-ga[ia]),delta_group_r_zero=float(gb[z]-ga[z]),
                gt_boundary=ia in (0,len(lag)-1),model_boundary=ib in (0,len(lag)-1),
                status='exploratory_small_n')
            if bootstrap > 0:
                rng=np.random.default_rng(_stable_seed(seed,(key,tuple(users))))
                ii=rng.integers(0,len(users),(bootstrap,len(users)))
                za=np.arctanh(np.clip(aa,-1+1e-7,1-1e-7));zb=np.arctanh(np.clip(bb,-1+1e-7,1-1e-7))
                ar=np.tanh(za[ii].mean(1));br=np.tanh(zb[ii].mean(1))
                pa,pb=ar.argmax(1),br.argmax(1)
                dlo,dhi=np.percentile(lag[pb]-lag[pa],[2.5,97.5])
                rlo,rhi=np.percentile(br.max(1)-ar.max(1),[2.5,97.5])
                row.update(delta_group_peak_lag_resample_low=float(dlo),
                    delta_group_peak_lag_resample_high=float(dhi),
                    delta_group_peak_r_resample_low=float(rlo),delta_group_peak_r_resample_high=float(rhi),
                    gt_bootstrap_boundary_frequency=float(np.mean((pa==0)|(pa==len(lag)-1))),
                    model_bootstrap_boundary_frequency=float(np.mean((pb==0)|(pb==len(lag)-1))))
            out.append(row)
    return out
