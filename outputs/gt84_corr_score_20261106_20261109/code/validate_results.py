"""Independent validation of GT84 fixed-side corr-score results.

Uses source CSVs and full-dummy WLS rather than the analysis implementation.
Does not modify source data or historical outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

META = ('scenario', 'task', 'side', 'signal', 'stage', 'mode', 'predictor')
SOURCE_KEY = ('content_id', 'task', 'side', 'signal', 'stage', 'mode')


def csvread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return math.nan


def truth(x):
    return str(x).strip().lower() in ('true', '1', '1.0')


def key(row, fields):
    return tuple(row.get(k, '') for k in fields)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def independent_within(records, adjusted):
    """Full participant-dummy WLS with participant-cluster CR1 covariance."""
    users = sorted({r['user'] for r in records})
    n, g = len(records), len(users)
    labels = np.array([r['user'] for r in records])
    counts = Counter(labels)
    w = np.array([1 / counts[u] for u in labels])
    x = stats.rankdata([num(r['metric_value']) for r in records])
    y = stats.rankdata([num(r['Score']) for r in records])
    dummy = np.column_stack([labels == u for u in users]).astype(float)
    if adjusted:
        covariates = np.column_stack(([num(r['TaskNum']) for r in records],
                                     [num(r['Attempt']) for r in records]))
        nuisance = np.column_stack((dummy, covariates))
    else:
        nuisance = dummy
    design = np.column_stack((nuisance, x))
    sw = np.sqrt(w)
    xw = design * sw[:, None]
    yw = y * sw
    beta = np.linalg.lstsq(xw, yw, rcond=1e-10)[0]
    residual = y - design @ beta
    bread = np.linalg.pinv(xw.T @ xw, rcond=1e-10, hermitian=True)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for user in users:
        mask = labels == user
        score = design[mask].T @ (w[mask] * residual[mask])
        meat += np.outer(score, score)
    rank = np.linalg.matrix_rank(xw)
    covariance = bread @ meat @ bread
    covariance *= g / (g-1) * (n-1) / (n-rank)
    se = float(np.sqrt(max(covariance[-1, -1], 0)))
    slope = float(beta[-1])
    p = float(2 * stats.t.sf(abs(slope/se), g-1)) if se > 1e-12 else math.nan
    nw = nuisance * sw[:, None]
    xr = x - nuisance @ np.linalg.lstsq(nw, x*sw, rcond=1e-10)[0]
    yr = y - nuisance @ np.linalg.lstsq(nw, y*sw, rcond=1e-10)[0]
    rho = float(np.dot(w*xr, yr) / np.sqrt(np.dot(w*xr, xr)*np.dot(w*yr, yr)))
    informative = sum(np.ptp(x[labels == u]) > 0 and np.ptp(y[labels == u]) > 0 for u in users)
    return dict(association=rho, rank_slope=slope, rank_slope_cluster_se=se,
                p=p, cluster_df=g-1, n_users=g, n_informative_users=informative)


def validate(root, repo):
    source = repo / 'outputs/gaze_psm_revision_20261106/gt84'
    tables = root / 'tables'
    errors, checks, maxima = [], Counter(), defaultdict(float)

    def check(condition, message, category):
        checks[category] += 1
        if not condition:
            errors.append(message)

    def close(actual, expected, message, category, atol=1e-7):
        a, b = num(actual), num(expected)
        ok = (math.isnan(a) and math.isnan(b)) or (math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol)
        if math.isfinite(a) and math.isfinite(b):
            maxima[category] = max(maxima[category], abs(a-b))
        check(ok, f'{message}: actual={a}, independently_expected={b}', category)

    protected = json.loads((root / 'validation/protected_hashes_before.json').read_text(encoding='utf-8-sig'))
    protected_after = []
    for record in protected:
        current = sha(record['path'])
        protected_after.append({**record, 'sha256_after': current, 'unchanged': current == record['sha256']})
        check(current == record['sha256'], f'Protected file changed: {record["path"]}', 'protected_files')

    source_rows = csvread(source / 'record_summary.csv')
    primary = [r for r in source_rows if r['variant'] == 'primary']
    lookup = {key(r, SOURCE_KEY): r for r in primary}
    check(len(lookup) == len(primary), 'Duplicate primary source keys', 'source_identity')
    manifest = json.loads((source / 'run_manifest.json').read_text(encoding='utf-8-sig'))
    ids = {r['content_id'] for r in manifest['records']}
    check(len(ids) == 84, f'Manifest expected 84 IDs, got {len(ids)}', 'source_identity')
    check({r['content_id'] for r in primary} == ids, 'Manifest/primary ID mismatch', 'source_identity')

    scores = {}
    for task in ('A', 'C'):
        rows = csvread(repo / f'Index/Task_{task}_Score.csv')
        for row in rows:
            cid = f"User{int(row['UserID'])}_{int(row['TaskNum'])}_{int(row['Attempt'])}"
            check(cid not in scores, f'Duplicate score key {cid}', 'score_source_keys')
            scores[cid] = {**row, 'task': task}
    indexes = {r['content_id']: r for r in csvread(repo / 'Index/content_index.csv')}

    matched = csvread(tables / 'matched_scores.csv')
    check(len(matched) == 84, f'matched_scores length {len(matched)} != 84', 'score_matching')
    check({r['content_id'] for r in matched} == ids, 'Matched ID set mismatch', 'score_matching')
    for row in matched:
        cid = row['content_id']
        s, ix = scores[cid], indexes[cid]
        check(s['task'] == row['task'] == ix['Task Name'], f'Task mismatch {cid}', 'score_matching')
        close(row['Score'], s['Score'], f'Score altered {cid}', 'score_matching', 0)
        check(row['AttemptStatus'] == s['AttemptStatus'], f'AttemptStatus altered {cid}', 'score_matching')
        expected_usable = math.isfinite(num(s['Score'])) and s['AttemptStatus'] in ('Completed', 'Failed')
        check(truth(row['score_usable']) == expected_usable, f'Wrong score eligibility {cid}', 'score_matching')
    for task, n, nonmissing, zeros in [('A', 33, 33, 7), ('C', 51, 50, 35)]:
        rr = [r for r in matched if r['task'] == task]
        check(len(rr) == n, f'Task {task} record count', 'score_totals')
        check(sum(math.isfinite(num(r['Score'])) for r in rr) == nonmissing, f'Task {task} available score count', 'score_totals')
        check(sum(num(r['Score']) == 0 for r in rr) == zeros, f'Task {task} zero score count', 'score_totals')

    candidates = csvread(tables / 'matched_record_metrics.csv')
    ckeys = [key(r, META)+ (r['content_id'],) for r in candidates]
    check(len(ckeys) == len(set(ckeys)), 'Duplicate candidate recording/condition keys', 'candidate_identity')
    scenarios = {r['scenario'] for r in candidates}
    required_scenarios = {'primary', 'completed_only', 'both_in_stage', 'velocity', 'exclude_boundary'}
    check(required_scenarios <= scenarios, f'Missing planned scenarios {sorted(required_scenarios-scenarios)}', 'planned_scope')
    for scenario in required_scenarios:
        expected_signal = 'velocity' if scenario == 'velocity' else 'position'
        expected_mode = 'Both in stage' if scenario == 'both_in_stage' else 'PSM in stage'
        src = [r for r in primary if r['signal'] == expected_signal and r['mode'] == expected_mode]
        for predictor in ('peak_r', 'r_zero'):
            actual = {key(r, SOURCE_KEY) for r in candidates if r['scenario'] == scenario and r['predictor'] == predictor}
            expected = {key(r, SOURCE_KEY) for r in src}
            check(actual == expected, f'Incomplete candidate coverage {scenario}/{predictor}: missing {len(expected-actual)}, extra {len(actual-expected)}', 'planned_scope')
    groups = defaultdict(list)
    all_groups = defaultdict(list)
    unexpected_scenarios = set()
    for row in candidates:
        source_row = lookup.get(key(row, SOURCE_KEY))
        check(source_row is not None, f'Candidate absent from primary source {key(row, SOURCE_KEY)}', 'candidate_identity')
        if source_row is None:
            continue
        pred = row['predictor']
        source_pred = {'signed_lag_s': 'peak_lag_s', 'absolute_lag_s': 'peak_lag_s'}.get(pred, pred)
        x = num(source_row[source_pred])
        if pred == 'absolute_lag_s':
            x = abs(x)
        close(row['metric_value'], x, f'Metric altered {key(row, META)} {row["content_id"]}', 'source_metric', 1e-12)
        s = scores[row['content_id']]
        close(row['Score'], s['Score'], f'Candidate Score altered {row["content_id"]}', 'candidate_scores', 0)
        full = truth(source_row['full_lag_support'])
        eligible = full and math.isfinite(x) and math.isfinite(num(s['Score'])) and s['AttemptStatus'] in ('Completed', 'Failed')
        scenario = row['scenario']
        if scenario == 'completed_only':
            eligible = eligible and s['AttemptStatus'] == 'Completed'
        elif scenario == 'exclude_boundary':
            eligible = eligible and not truth(source_row['peak_at_boundary'])
        elif scenario in ('lag_identifiable', 'identifiable_lag', 'lag_exploratory', 'lag_identifiable_exploratory'):
            eligible = eligible and truth(source_row['lag_identifiable'])
        elif scenario not in ('primary', 'both_in_stage', 'velocity'):
            unexpected_scenarios.add(scenario)
        check(truth(row['eligible']) == eligible, f'Wrong metric eligibility {key(row, META)} {row["content_id"]}', 'candidate_eligibility')
        expected_signal = 'velocity' if scenario == 'velocity' else 'position'
        expected_mode = 'Both in stage' if scenario == 'both_in_stage' else 'PSM in stage'
        check(row['signal'] == expected_signal and row['mode'] == expected_mode, f'Wrong scenario signal/mode {key(row, META)}', 'scenario_definition')
        all_groups[key(row, META)].append(row)
        if truth(row['eligible']):
            groups[key(row, META)].append(row)
    check(not unexpected_scenarios, f'Unknown scenarios: {sorted(unexpected_scenarios)}', 'scenario_definition')

    people = csvread(tables / 'participant_summaries.csv')
    participant_lookup = {}
    for row in people:
        k = key(row, META)
        rr = [r for r in groups[k] if r['user'] == row['user']]
        check(bool(rr), f'Participant point has no eligible rows {k} {row["user"]}', 'participant_aggregation')
        if not rr:
            continue
        raw = np.array([num(r['metric_value']) for r in rr])
        if row['predictor'] in ('peak_r', 'r_zero'):
            x = np.tanh(np.mean(np.arctanh(np.clip(raw, -1+1e-7, 1-1e-7))))
        else:
            x = np.median(raw)
        close(row['x'], x, f'Participant x {k} {row["user"]}', 'participant_aggregation')
        close(row['Score'], np.mean([num(r['Score']) for r in rr]), f'Participant Score {k} {row["user"]}', 'participant_aggregation')
        close(row['n_recordings'], len(rr), f'Participant n {k} {row["user"]}', 'participant_aggregation', 0)
        check(set(row['content_ids'].split(';')) == {r['content_id'] for r in rr}, f'Participant recording mismatch {k} {row["user"]}', 'participant_aggregation')
        pk = k + (row['user'],)
        check(pk not in participant_lookup, f'Duplicate participant point {pk}', 'participant_aggregation')
        participant_lookup[pk] = row

    associations = csvread(tables / 'associations.csv')
    association_keys = [key(r, META) + (r['method'],) for r in associations]
    check(len(set(association_keys)) == len(association_keys), 'Duplicate association keys', 'association_identity')
    expected_associations = {k + (method,) for k in all_groups for method in ('between_participants', 'within_participant', 'within_adjusted_order')}
    check(set(association_keys) == expected_associations, f'Incomplete association grid: missing {len(expected_associations-set(association_keys))}, extra {len(set(association_keys)-expected_associations)}', 'planned_scope')
    for row in associations:
        k = key(row, META)
        rr = groups[k]
        close(row['n_recordings'], len(rr), f'Association n_recordings {k}', 'association_counts', 0)
        close(row['n_users'], len({r['user'] for r in rr}), f'Association n_users {k}', 'association_counts', 0)
        if row['status'] != 'ok':
            check(not math.isfinite(num(row.get('p'))), f'Unavailable association has finite p {k}', 'unavailable_results')
            continue
        if row['method'] == 'between_participants':
            pp = [participant_lookup[k + (u,)] for u in sorted({r['user'] for r in rr})]
            rho = stats.spearmanr([num(p['x']) for p in pp], [num(p['Score']) for p in pp]).statistic
            close(row['association'], rho, f'Spearman {k}', 'between_spearman')
        else:
            calc = independent_within(rr, row['method'] == 'within_adjusted_order')
            for metric, val in calc.items():
                close(row[metric], val, f'Full dummy WLS {k} {row["method"]} {metric}', 'within_full_dummy')
        check(-1-1e-9 <= num(row['association']) <= 1+1e-9, f'Association out of bounds {k}', 'estimate_bounds')
        p = num(row['p'])
        check(not math.isfinite(p) or 0 <= p <= 1, f'p out of bounds {k}', 'estimate_bounds')

    families = defaultdict(list)
    for row in associations:
        families[row['family']].append(row)
    for family, rr in families.items():
        vals = [r for r in rr if math.isfinite(num(r.get('p')))]
        expected = stats.false_discovery_control([num(r['p']) for r in vals], method='bh') if vals else []
        for row, q in zip(vals, expected):
            close(row['q_bh'], q, f'BH {family} {key(row, META)}', 'bh_adjustment')
        for row in rr:
            close(row['family_size'], len(vals), f'BH family_size {family}', 'bh_adjustment', 0)
            if not math.isfinite(num(row.get('p'))):
                check(not math.isfinite(num(row.get('q_bh'))), f'Unavailable p with finite q {family}', 'bh_adjustment')
        # Every family must contain only one scenario/predictor/method, with tasks, stages, sides pooled.
        check(len({(r['scenario'], r['predictor'], r['method']) for r in rr}) == 1,
              f'Unexpected family scope {family}', 'bh_family_scope')

    coverage = csvread(tables / 'coverage.csv')
    for row in coverage:
        k = key(row, META)
        rr, aa = groups[k], all_groups[k]
        close(row['n_total_records'], len(aa), f'Coverage total {k}', 'coverage_counts', 0)
        close(row['n_score_available'], sum(math.isfinite(num(r['Score'])) for r in aa), f'Coverage available {k}', 'coverage_counts', 0)
        close(row['n_eligible_records'], len(rr), f'Coverage eligible {k}', 'coverage_counts', 0)
        close(row['n_users'], len({r['user'] for r in rr}), f'Coverage users {k}', 'coverage_counts', 0)

    plot_groups = defaultdict(list)
    for row in csvread(tables / 'within_points.csv'):
        plot_groups[key(row, META) + (row['method'],)].append(row)
    for pk, points in plot_groups.items():
        k, method = pk[:-1], pk[-1]
        rr = groups[k]
        check({r['content_id'] for r in rr} == {r['content_id'] for r in points}, f'Within plot record mismatch {pk}', 'within_plot_data')
        labels = np.array([r['user'] for r in rr])
        users = sorted(set(labels))
        w = np.array([1/sum(labels == u) for u in labels])
        sw = np.sqrt(w)
        x = stats.rankdata([num(r['metric_value']) for r in rr])
        y = stats.rankdata([num(r['Score']) for r in rr])
        nuisance = np.column_stack([labels == u for u in users]).astype(float)
        if method == 'within_adjusted_order':
            nuisance = np.column_stack((nuisance, [num(r['TaskNum']) for r in rr], [num(r['Attempt']) for r in rr]))
        nw = nuisance*sw[:,None]
        xr = x-nuisance @ np.linalg.lstsq(nw, x*sw, rcond=1e-10)[0]
        yr = y-nuisance @ np.linalg.lstsq(nw, y*sw, rcond=1e-10)[0]
        point_lookup = {r['content_id']: r for r in points}
        for i, row in enumerate(rr):
            point = point_lookup[row['content_id']]
            close(point['x_residual'], xr[i], f'Within plot x {pk} {row["content_id"]}', 'within_plot_data')
            close(point['y_residual'], yr[i], f'Within plot y {pk} {row["content_id"]}', 'within_plot_data')
            close(point['weight'], w[i], f'Within plot weight {pk} {row["content_id"]}', 'within_plot_data')

    for row in csvread(tables / 'leave_one_user_out.csv'):
        k, method, omitted = key(row, META), row['method'], row['omitted_user']
        rr = [r for r in groups[k] if r['user'] != omitted]
        if method == 'between_participants':
            pp = [participant_lookup[k+(u,)] for u in sorted({r['user'] for r in rr})]
            expected = stats.spearmanr([num(p['x']) for p in pp], [num(p['Score']) for p in pp]).statistic
        elif math.isfinite(num(row['association'])):
            expected = independent_within(rr, method == 'within_adjusted_order')['association']
        else:
            continue
        close(row['association'], expected, f'Leave one user out {k} {method} {omitted}', 'leave_one_user_out')

    summary = {
        'passed': not errors,
        'independent_implementation': 'Source CSV matching; scipy Spearman/BH; full participant-dummy weighted least squares with cluster CR1 covariance',
        'limitations': 'Does not independently regenerate all Monte Carlo bootstrap draws or permutation sequences; independently verifies matching, eligibility, participant aggregation, association coefficients, within-participant p values and BH correction.',
        'checks': dict(checks), 'max_absolute_errors': dict(maxima),
        'rows': {'associations': len(associations), 'candidate_metrics': len(candidates), 'participant_points': len(people)},
        'protected_files': protected_after, 'errors': errors,
    }
    out = root / 'validation/independent_validation.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': summary['passed'], 'checks': dict(checks), 'max_errors': dict(maxima), 'errors': errors[:12]}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--repo', type=Path, default=Path('D:/GitHub_Project/co-tracker'))
    args = parser.parse_args()
    raise SystemExit(validate(args.root, args.repo))
