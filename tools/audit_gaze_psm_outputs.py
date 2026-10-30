"""Read-only replay of saved gaze/PSM outputs without the original NAS data.

Run from the repo root. Only the explicitly named audit JSON is written.
This verifies cached trajectories onward; it cannot authenticate source CSVs
or re-decode original masks when those files are unavailable.
"""
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import gaze_psm_pipeline as pipeline
from tools.gaze_psm_group import aggregate, fisher_mean, GROUP_FIELDS
from tools.gaze_psm_report import _long_rows, _summary


def _initialize(directory, config, records):
    global _DIRECTORY, _ARGS, _RECORDS, _DATA
    _DIRECTORY, _RECORDS, _DATA = Path(directory), records, {}
    _ARGS = pipeline.parse_args([])
    for name, value in config.items():
        setattr(_ARGS, name, value)


def _data(cid):
    if cid not in _DATA:
        with np.load(_DIRECTORY/'cache'/f'{cid}.npz', allow_pickle=False) as z:
            _DATA[cid] = dict(z)
    return _DATA[cid]


def _observed(cid):
    return pipeline.calculate(_RECORDS[cid], _data(cid), _ARGS)


def _null(pairings):
    curves = []
    for cid, donor in pairings:
        for curve in pipeline.calculate(_RECORDS[cid], _data(cid), _ARGS, donor=_data(donor)):
            curves.append({k: v for k, v in curve.items()
                           if not isinstance(v, np.ndarray) or k in {'lag_s', 'r'}})
    return curves


def _same(actual, expected, label):
    if isinstance(actual, (np.ndarray, float, np.floating)):
        a = np.asarray(actual, dtype=float)
        b = np.asarray(expected, dtype=float)
        np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-9, equal_nan=True,
                                   err_msg=label)
    else:
        assert actual == expected, (label, actual, expected)


def _report_numerical_signature(source):
    """Allow revised plots only if the historical CSV computations are intact."""
    names = {'_value', '_write', '_long_rows', '_summary', 'META', 'LAG_FIELDS'}
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                                  for t in node.targets):
            nodes.append(node)
    assert len(nodes) == len(names)
    return [ast.dump(node, include_attributes=False) for node in nodes]


def _direct(gaze, psm, ids, lag, within, config, minimum):
    """Independent pair enumeration; no FFT or production sufficient statistics."""
    covariance = gv = pv = 0.
    count = occurrences = 0
    distances = []
    for identity in np.unique(ids[ids >= 0]):
        u = np.flatnonzero(ids == identity)
        t = u-lag
        inside = (t >= 0) & (t < len(gaze))
        u, t = u[inside], t[inside]
        valid = np.isfinite(gaze[t]).all(axis=1) & np.isfinite(psm[u]).all(axis=1)
        if within:
            valid &= ids[t] == identity
        a, b = gaze[t[valid]], psm[u[valid]]
        if len(a) < 2:
            continue
        ac, bc = a-a.mean(axis=0), b-b.mean(axis=0)
        covariance += np.sum(ac*bc)
        gv += np.sum(ac*ac)
        pv += np.sum(bc*bc)
        count += len(a)
        occurrences += 1
        distances.extend(np.linalg.norm(a-b, axis=1))
    gr = np.sqrt(gv/count) if count else np.nan
    pr = np.sqrt(pv/count) if count else np.nan
    supported = (count >= max(2, int(np.ceil(minimum*config['hz'])))
                 and gr >= config['min_gaze_rms'] and pr >= config['min_psm_rms']
                 and np.sqrt(gv*pv) > np.finfo(float).eps)
    return {'r': covariance/np.sqrt(gv*pv) if supported else np.nan,
            'pairs': count, 'occurrences': occurrences, 'gaze_rms': gr, 'psm_rms': pr,
            'distance_median_px': np.median(distances) if distances else np.nan,
            'proximity': np.mean(np.asarray(distances) <= config['spatial_tolerance'])
                         if distances else np.nan}


def _csv_check(path, expected):
    """Compare every exported cell to its JSON-derived value, including NaNs."""
    count = 0
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            item = next(expected)
            for key, text in row.items():
                value = item.get(key)
                if value is None:
                    assert text.lower() in {'', 'nan', 'none'}, (path.name, key, text)
                elif isinstance(value, (list, dict, tuple)):
                    assert json.loads(text) == value, (path.name, key)
                elif isinstance(value, (bool, str)):
                    assert text == str(value), (path.name, key, text, value)
                else:
                    actual = float(text)
                    assert (math.isnan(actual) and math.isnan(value)) or math.isclose(
                        actual, value, rel_tol=1e-9, abs_tol=1e-8), (path.name, key, actual, value)
            count += 1
    assert next(expected, None) is None, f'{path.name}: missing rows'
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/gaze_psm_xcorr'))
    parser.add_argument('--audit-json', type=Path,
                        default=Path('outputs/gaze_psm_xcorr/audit_20261030.json'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--null-replicates', type=int, default=0,
                        help='Replay this many initial unpaired replicates; 100 for the full run')
    opts = parser.parse_args()
    directory = opts.output
    config = json.loads((directory/'config.json').read_text())
    saved = json.loads((directory/'results.json').read_text())
    qc = json.loads((directory/'quality.json').read_text())
    records = {r['content_id']: {'content_id': r['content_id'], 'Task Name': r['task']}
               for r in qc if r['status'] == 'ok'}
    assert len(records) == len(qc), 'Extraction errors require individual review'
    assert 0 <= opts.null_replicates <= config['shuffles']
    arrays = {'lag_s', 'r', 'pairs', 'occurrences', 'gaze_rms', 'psm_rms',
              'distance_median_px', 'proximity', 'n_records', 'r_fixed_records',
              'r_primary_fixed_record_users', 'r_available', 'n_users',
              'r_ci_low', 'r_ci_high', 'bootstrap_peak_lag_s', 'null_r_low',
              'null_r_high', 'null_peak_r', 'bootstrap_difference_s'}
    for rows in saved.values():
        for row in rows:
            for key in arrays & row.keys():
                row[key] = np.asarray(row[key], dtype=float)
    provenance = json.loads((directory/'rerun_provenance.json').read_text())
    hashes = {}
    plotting_changes = {}
    for name, expected in provenance['code_sha256'].items():
        current, snapshot = (ROOT/'tools'/name).read_bytes(), (directory/'code_snapshot'/name).read_bytes()
        digest = hashlib.sha256(current.replace(b'\r\n', b'\n')).hexdigest()
        assert hashlib.sha256(snapshot.replace(b'\r\n', b'\n')).hexdigest() == expected, name
        if name == 'gaze_psm_report.py' and digest != expected:
            assert _report_numerical_signature(current.decode('utf-8')) == _report_numerical_signature(snapshot.decode('utf-8'))
            plotting_changes[name] = 'CSV functions and fields match snapshot AST; plotting may differ'
        else:
            assert digest == expected, name
        hashes[name] = digest
    _initialize(directory, config, records)
    direct_checks = 0
    for cid, record in records.items():
        data = _data(cid)
        quality = json.loads(str(data['qc_json']))
        assert quality == next(q for q in qc if q['content_id'] == cid)
        np.testing.assert_allclose(np.diff(data['time_s']), 1/config['hz'], atol=1e-12)
        for side in ('l', 'r'):
            native = data[f'native_psm_{side}']
            assert np.array_equal(np.isfinite(native).all(axis=1),
                                  data[f'area_{side}'] >= config['min_area'])
            raw = pipeline.interpolate_valid(data['native_time_s'], native,
                                             data['time_s'], config['max_gap'])
            _same(raw, data[f'psm_raw_{side}'], f'{cid}: interpolated PSM')
            for signal in ('gaze', 'psm'):
                key = f'{signal}_{side}'
                value = pipeline.smooth_valid(data[f'{signal}_raw_{side}'],
                                              config['hz'], config['smooth_ms'])
                _same(value, data[key], f'{cid}: smoothed {key}')
                fraction = np.isfinite(data[f'{signal}_raw_{side}']).all(axis=1).mean()
                _same(float(fraction), quality[f'{side}_{signal}_valid_fraction'], f'{cid}: coverage')
        # One full recording from each task: every position condition, all lags.
        if cid == next(c for c in records if records[c]['Task Name'] == record['Task Name']):
            for curve in saved['record_curves']:
                if curve['content_id'] != cid or curve['signal'] != 'position':
                    continue
                ids = (np.zeros(len(data['time_s']), dtype=int) if curve['mode'] == 'whole'
                       else np.where(data['stage'] == curve['stage'], data['occurrence'], -1))
                for index, lag in enumerate(np.rint(curve['lag_s']*config['hz']).astype(int)):
                    result = _direct(data[f"gaze_{curve['side']}"], data[f"psm_{curve['side']}"],
                                     ids, lag, curve['mode'] == 'within_stage', config,
                                     config['min_overlap'] if curve['mode'] == 'whole'
                                     else config['stage_min_overlap'])
                    for key, value in result.items():
                        _same(float(value), curve[key][index], f'{cid}:{curve["stage"]}:{key}')
                    direct_checks += 1
    print(f'Cache checks passed: {len(records)}; independent lag checks: {direct_checks}', flush=True)
    by_record = defaultdict(list)
    for curve in saved['record_curves']:
        by_record[curve['content_id']].append(curve)
    with ProcessPoolExecutor(max_workers=opts.workers, initializer=_initialize,
                             initargs=(directory, config, records)) as pool:
        for number, (cid, replay) in enumerate(zip(records, pool.map(_observed, records)), 1):
            expected = by_record[cid]
            assert len(replay) == len(expected)
            for actual, stored in zip(replay, expected):
                for key, value in actual.items():
                    _same(value, stored[key], f'{cid}:{key}')
            if number % 10 == 0 or number == len(records):
                print(f'Observed records replayed: {number}/{len(records)}', flush=True)
        rng = np.random.default_rng(config['seed'])
        pairings = []
        for _ in range(opts.null_replicates):
            replicate = []
            for cid in records:
                by_user = defaultdict(list)
                for other in records:
                    if other.split('_')[0] != cid.split('_')[0] and records[other]['Task Name'] == records[cid]['Task Name']:
                        by_user[other.split('_')[0]].append(other)
                user = rng.choice(sorted(by_user))
                replicate.append((cid, rng.choice(by_user[user])))
            pairings.append(replicate)
        # Keep replayed reference curves for an independent check of their maxima/envelopes.
        null_curves = defaultdict(list)
        group_lookup = {tuple(g[k] for k in GROUP_FIELDS): g for g in saved['group_curves']}
        if pairings:
            from tools.gaze_psm_group import _participant_curves
            for number, replicate in enumerate(pool.map(_null, pairings), 1):
                people = _participant_curves(replicate)
                lookup = {(tuple(p[k] for k in GROUP_FIELDS), p['user']): p for p in people}
                for key, group in group_lookup.items():
                    if group['n_stable_users'] < config['min_users']:
                        continue
                    selected = [lookup.get((key, u)) for u in group['stable_users']]
                    if any(p is None for p in selected):
                        continue
                    values = np.stack([p['r'] for p in selected])
                    if np.isfinite(values).all():
                        null_curves[key].append(fisher_mean(values))
                print(f'Unpaired replicates replayed: {number}/{len(pairings)}', flush=True)
    users, groups, contrasts = aggregate(saved['record_curves'], bootstrap=config['bootstrap'],
                                         seed=config['seed'], min_users=config['min_users'])
    for label, replay in [('user_curves', users), ('group_curves', groups), ('contrasts', contrasts)]:
        assert len(replay) == len(saved[label])
        for actual, expected in zip(replay, saved[label]):
            for key, value in actual.items():
                if not key.startswith('null_') and key != 'n_null_replicates':
                    _same(value, expected[key], f'{label}:{key}')
    for group in saved['group_curves']:
        values = group['null_peak_r']
        assert len(values) == group['n_null_replicates']
        if len(values):
            _same(float(np.percentile(values, 95)), group['null_peak_r_95'], 'null 95%')
            _same(float((1+np.sum(values >= group['peak_r']))/(1+len(values))), group['null_peak_p'], 'null tail')
        key = tuple(group[k] for k in GROUP_FIELDS)
        replay = np.asarray(null_curves[key])
        if replay.size:
            _same(replay.max(axis=1), values[:len(replay)], 'replayed null maxima')
        if opts.null_replicates == config['shuffles']:
            assert len(replay) == len(values), (key, len(replay), len(values))
            if replay.size:
                low, high = np.percentile(replay, [2.5, 97.5], axis=0)
                _same(low, group['null_r_low'], 'null lower envelope')
                _same(high, group['null_r_high'], 'null upper envelope')
    print('Hierarchical aggregation, bootstraps, contrasts and reference checks passed', flush=True)
    csv_rows = {}
    for label, source in [('record', 'record_curves'), ('novice', 'user_curves'), ('group', 'group_curves')]:
        csv_rows[f'{label}_curves.csv'] = _csv_check(directory/f'{label}_curves.csv', iter(_long_rows(saved[source])))
        csv_rows[f'{label}_summary.csv'] = _csv_check(directory/f'{label}_summary.csv', iter(_summary(c) for c in saved[source]))
    csv_rows['contrasts.csv'] = _csv_check(directory/'contrasts.csv', iter(saved['contrasts']))
    csv_rows['quality.csv'] = _csv_check(directory/'quality.csv', iter(qc))
    for filename, source, array, field in [
        ('bootstrap_peaks.csv', 'group_curves', 'bootstrap_peak_lag_s', 'peak_lag_s'),
        ('unpaired_peaks.csv', 'group_curves', 'null_peak_r', 'peak_r'),
        ('bootstrap_contrasts.csv', 'contrasts', 'bootstrap_difference_s', 'difference_s')]:
        expected = ({**row, 'replicate': i, field: value} for row in saved[source]
                    for i, value in enumerate(row[array]))
        csv_rows[filename] = _csv_check(directory/filename, iter(expected))
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'all_checks_passed': True,
              'scope': 'cached trajectories onward; original NAS source verification unavailable',
              'code_sha256_after_lf_normalization': hashes, 'cache_records_checked': len(records),
              'plotting_changes_with_numerical_exports_verified': plotting_changes,
              'record_curves_replayed': len(saved['record_curves']), 'independent_direct_lag_checks': direct_checks,
              'novice_curves_replayed': len(users), 'group_curves_replayed': len(groups),
              'contrasts_replayed': len(contrasts), 'bootstrap_per_supported_group': config['bootstrap'],
              'unpaired_replicates_replayed': opts.null_replicates, 'csv_rows_checked': csv_rows,
              'task_record_counts': dict(Counter(q['task'] for q in qc)),
              'group_status_counts': dict(Counter(g['status'] for g in saved['group_curves']))}
    pipeline.save_json(opts.audit_json, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
