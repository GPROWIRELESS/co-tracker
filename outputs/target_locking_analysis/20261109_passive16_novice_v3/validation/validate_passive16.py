"""Independent read-only checks of passive16 target-locking outputs.

Only writes its own JSON report. Source pairing, coordinates, native time weights,
geometry labels and occupancy formulas are reconstructed independently; temporal
and optional smoothing checks deliberately call the hash-frozen v3 functions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
OUTPUT = WORK if (WORK/'pair_metrics.csv').exists() else WORK/'output'
FROZEN = WORK/'code'/'v3' if (WORK/'code'/'v3').is_dir() else WORK/'code_snapshot'/'v3'
REFERENCE = Path(r'D:\GitHub_Project\co-tracker\outputs\target_locking_analysis\20261105T063658759189Z_v3_full')
LINKS = Path(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json')
SCENARIOS = {'main': (50, 50), 'r25': (25, 25), 'r100': (100, 100),
             'gripper100': (50, 100), 'smooth': (50, 50)}
STAGES = {'A': ['reach ring', 'grab ring', 'transfer ring', 'place ring'],
          'C': ['reach string', 'wrap around psm', 'pull through and cinch',
                'threading string through tower']}
CANONICAL = {'rr': 'reach ring', 'gr': 'grab ring', 'tr': 'transfer ring',
             'pr': 'place ring', 'wrap': 'wrap around psm',
             'cinch': 'pull through and cinch', 'thread': 'threading string through tower'}
CHECKS = []


def check(name, condition, details=None):
    CHECKS.append({'check': name, 'passed': bool(condition), 'details': details})
    if not condition:
        print('FAIL:', name, details, flush=True)


def same(name, a, b, atol=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        check(name, False, {'actual_shape': list(a.shape), 'expected_shape': list(b.shape)})
        return
    if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
        check(name, np.array_equal(a, b))
    elif a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
        ok = np.allclose(a, b, atol=atol, rtol=1e-10, equal_nan=True)
        finite = np.isfinite(a) & np.isfinite(b)
        delta = float(np.max(np.abs(a[finite].astype(float)-b[finite].astype(float)))) if finite.any() else None
        check(name, ok, {'max_abs_error': delta})
    else:
        check(name, np.array_equal(a, b))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(z, *keys):
    for key in keys:
        if key in z:
            return np.asarray(z[key])
    raise KeyError('Missing required frame key; expected one of '+repr(keys))


def scenario_get(z, scenario, kind):
    aliases = {
        'active_mask': [f'active_valid_{scenario}', f'valid_active_{scenario}'],
        'passive_mask': [f'passive_valid_{scenario}', f'valid_passive_{scenario}'],
        'common_mask': [f'common_valid_{scenario}', f'valid_common_{scenario}', f'common_{scenario}', f'valid_{scenario}'],
        'active_state': [f'active_state_{scenario}', f'state_active_{scenario}'],
        'passive_state': [f'passive_state_{scenario}', f'state_passive_{scenario}'],
        'area': [f'area_{scenario}', f'areas_{scenario}'],
    }
    return get(z, *aliases[kind])


def native_clock(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if str(r.get('IsDelete', '0')).strip() in ('', '0', '0.0')]
    frames = np.array([int(r['Frame_Number']) for r in rows], dtype=np.int64)
    stamps = np.array([int(Decimal(r['Timestamp(ns)'])) for r in rows], dtype=np.int64)
    dt = np.zeros(len(stamps))
    breaks = np.ones(len(stamps), dtype=bool)
    for i in range(1, len(stamps)):
        delta = int(stamps[i])-int(stamps[i-1])
        if frames[i]-frames[i-1] == 1 and 0 < delta <= 100_000_000 and min(stamps[i], stamps[i-1]) >= 0:
            dt[i-1] += delta/2e9
            dt[i] += delta/2e9
            breaks[i] = False
    return frames, stamps, dt, breaks


def point_labels(xy, target, left, right, rt, rp, mask):
    # Deliberately independent of classify and its internal helper functions.
    t = ((xy[:, 0]-target[:, 0])**2+(xy[:, 1]-target[:, 1])**2) <= rt**2
    l = ((xy[:, 0]-left[:, 0])**2+(xy[:, 1]-left[:, 1])**2) <= rp**2
    r = ((xy[:, 0]-right[:, 0])**2+(xy[:, 1]-right[:, 1])**2) <= rp**2
    return np.where(mask, t.astype(np.int8)+2*(l | r).astype(np.int8), -1)


def pixel_area(target, left, right, width, height, rt, rp):
    yy, xx = np.ogrid[.5:height:1., .5:width:1.]
    t = (xx-target[0])**2+(yy-target[1])**2 <= rt**2
    p = ((xx-left[0])**2+(yy-left[1])**2 <= rp**2) | ((xx-right[0])**2+(yy-right[1])**2 <= rp**2)
    code = t.astype(np.int8)+2*p.astype(np.int8)
    return np.bincount(code.ravel(), minlength=4)/(width*height)


def divide(num, den):
    return num/den if den > 0 else np.nan


def independent_metrics(labels, mask, dt, area):
    mask = mask & (labels >= 0)
    seconds = float(dt[mask].sum())
    occ = np.array([dt[mask & (labels == k)].sum() for k in range(4)])
    occ = occ/seconds if seconds > 0 else np.full(4, np.nan)
    a = np.sum(area[mask]*dt[mask, None], axis=0)/seconds if seconds > 0 else np.full(4, np.nan)
    n, t, p, o = occ
    an, at, ap, ao = a
    c, cu = divide(t, t+p), divide(at, at+ap)
    return dict(valid_n=int(mask.sum()), valid_s=seconds, L_N=n, L_T=t, L_P=p, L_O=o,
                L_target=t+o, L_gripper=p+o, D=t+p, C=c, B=t-p,
                area_N=an, area_T=at, area_P=ap, area_O=ao, C_U=cu,
                C_excess=c-cu, E_target=t+o-at-ao, E_gripper=p+o-ap-ao,
                B_excess=t-p-at+ap)


def check_frame_pair(pair_id, entry, root, out, metrics, coverage, joints, geo, old_trials):
    meta = entry['meta']
    cid, task = meta['content_id'], meta['task_name']
    tag = pair_id+'/'
    active_path = root / entry['target_locking_label_path']
    passive_path = root / entry['gaze_npy_path']
    active = np.load(active_path, mmap_mode='r', allow_pickle=False)
    passive = np.load(passive_path, mmap_mode='r', allow_pickle=False)
    check(tag+'raw_row_count_equal', len(active) == len(passive), {'active': len(active), 'passive': len(passive)})
    z = dict(np.load(out/'frames'/f'{pair_id}.npz', allow_pickle=False))
    width, height = map(int, get(z, 'resolution', 'image_size'))
    frames, stamps, dt, breaks = native_clock(root/'Active_Data'/f"User{meta['user']}"/'csv'/f'{cid}_video_ts.csv')
    row_index = get(z, 'row_index', 'row_indices', 'source_row_index', 'active_row_index').astype(int)
    source_lookup = {int(n): i for i, n in enumerate(active['Frame_Number'])}
    expected_rows = np.array([source_lookup[int(f)] for f in frames])
    same(tag+'retained_original_row_indices', row_index, expected_rows)
    same(tag+'active_original_frame_number', get(z, 'frame_number', 'frames'), frames)
    same(tag+'original_video_timestamp', get(z, 'timestamp_ns'), stamps, atol=0)
    same(tag+'independent_midpoint_dt', get(z, 'dt_s', 'dt'), dt)
    src = active[row_index]
    raw_active = np.column_stack([src['gaze_x'], src['gaze_y']]).astype(float)
    raw_passive = np.column_stack([passive['gaze_x'][row_index], passive['gaze_y'][row_index]]).astype(float)
    coordinate_ok = {}
    raw_xy = {}
    for domain, raw in [('active', raw_active), ('passive', raw_passive)]:
        coordinate_ok[domain] = np.isfinite(raw).all(1) & (raw >= 0).all(1) & (raw <= 1).all(1)
        xy = raw*np.array([width, height])
        if domain+'_xy_raw' in z:
            same(tag+domain+'_raw_source_npy_coordinates', z[domain+'_xy_raw'], xy)
        xy[~coordinate_ok[domain]] = np.nan
        raw_xy[domain] = xy
        same(tag+domain+'_source_npy_coordinates', get(z, domain+'_xy', domain+'_gaze_xy', domain+'_gaze'), xy)
    slots = np.full((len(src), 5, 2), np.nan)
    for j in range(5):
        if f'point_{j+1}_x' in src.dtype.names:
            slots[:, j, 0], slots[:, j, 1] = src[f'point_{j+1}_x'], src[f'point_{j+1}_y']
    valid_slot = np.isfinite(slots).all(2) & (slots >= 0).all(2) & (slots < [width, height]).all(2)
    counts = valid_slot.sum(1)
    target = np.divide(np.where(valid_slot[:, :, None], slots, 0).sum(1), counts[:, None],
                       out=np.full((len(src), 2), np.nan), where=counts[:, None] > 0)
    same(tag+'source_target_slots', get(z, 'target_slots', 'slots'), slots)
    same(tag+'independent_slot_mean', get(z, 'target_xy', 'z'), target)
    left, right = get(z, 'left_xy', 'left'), get(z, 'right_xy', 'right')
    dlc = pd.read_csv(root/entry['psm_dlc_csv_path'], header=[0, 1, 2], index_col=0)
    for side, body, actual in [('left', 'left_psm', left), ('right', 'right_psm', right)]:
        expected = np.column_stack([dlc.loc[:, [c for c in dlc.columns if c[1:] == (body, axis)]].iloc[:, 0] for axis in ('x', 'y')]).astype(float)
        expected[~np.isfinite(expected).all(1)] = np.nan
        same(tag+'source_DLC_'+side, actual, expected)
    stage = get(z, 'stage').astype(str)
    source_stage = src['stage' if 'stage' in src.dtype.names else 'event'].astype(str)
    source_stage = [' '.join(s.strip().lower().split()) for s in source_stage]
    source_stage = np.array([CANONICAL.get(s, s) for s in source_stage])
    same(tag+'source_normalized_stage', stage, source_stage)
    occurrence = np.r_[0, np.cumsum(stage[1:] != stage[:-1])].astype(int)
    same(tag+'occurrence', get(z, 'occurrence'), occurrence)
    included = np.isin(stage, STAGES[task])
    identity_key = next((k for k in ('target_id', 'target_identity', 'point_1_id') if k in src.dtype.names), None)
    identity_change = np.zeros(len(src), dtype=bool)
    if identity_key:
        identity_change[1:] = src[identity_key][1:] != src[identity_key][:-1]
    breaks |= identity_change | ~included
    breaks[1:] |= (occurrence[1:] != occurrence[:-1]) | ~included[:-1]
    same(tag+'stage_and_frame_breaks', get(z, 'break_before', 'breaks'), breaks)
    geom_ok = included & (counts > 0)
    for side in (left, right):
        geom_ok &= np.isfinite(side).all(1) & (side >= 0).all(1) & (side < [width, height]).all(1)
    block = metrics[metrics.pair_id == pair_id]
    check(tag+'25_stage_scenario_rows', len(block) == 25)
    scenarios_checked = []
    for scenario, (rt, rp) in SCENARIOS.items():
        label = tag+scenario+'/'
        xy = {d: raw_xy[d].copy() for d in ('active', 'passive')}
        if scenario == 'smooth':
            for domain in xy:
                xy[domain], _ = geo.smooth_gaze(stamps, xy[domain], occurrence, breaks)
                if domain+'_smooth_xy' in z:
                    same(label+domain+'_v3_smoothing', z[domain+'_smooth_xy'], xy[domain])
        masks = {d: geom_ok & np.isfinite(x).all(1) & (x >= 0).all(1) & (x <= [width, height]).all(1) for d, x in xy.items()}
        common = masks['active'] & masks['passive']
        for domain in masks:
            same(label+domain+'_mask', scenario_get(z, scenario, domain+'_mask'), masks[domain])
        same(label+'paired_common_mask', scenario_get(z, scenario, 'common_mask'), common)
        states = {d: point_labels(x, target, left, right, rt, rp, common) for d, x in xy.items()}
        own_active_state = point_labels(xy['active'], target, left, right, rt, rp, masks['active'])
        for domain in states:
            # Stored states may retain each domain's own support. On paired
            # support all labels must agree and missing common rows are masked.
            stored = scenario_get(z, scenario, domain+'_state')
            same(label+domain+'_independent_geometry', stored[common], states[domain][common])
        try:
            area = scenario_get(z, scenario, 'area')
        except KeyError:
            area = geo.areas(target, left, right, width, height, rt, rp, geom_ok)
        ids = np.flatnonzero(geom_ok)
        if len(ids):
            for i in np.unique(ids[[0, len(ids)//2, -1]]):
                same(label+f'brute_pixel_area_frame_{i}', area[i], pixel_area(target[i], left[i], right[i], width, height, rt, rp))
        for st in ['whole']+STAGES[task]:
            prefix = label+st+'/'
            st_mask = included if st == 'whole' else stage == st
            selected = common & st_mask
            r = block[(block.scenario == scenario) & (block.stage == st)]
            check(prefix+'one_row', len(r) == 1)
            if len(r) != 1:
                continue
            r = r.iloc[0]
            result = {}
            for domain in states:
                m = independent_metrics(states[domain], selected, dt, area)
                m.update(geo.temporal(np.where(selected, states[domain], -1), dt, occurrence, breaks))
                result[domain] = m
                for name, expected in m.items():
                    col = domain+'_'+name
                    check(prefix+col+'_present', col in r.index)
                    if col in r.index:
                        same(prefix+col, float(r[col]), expected, atol=1e-8)
            for name in result['active']:
                col = 'delta_'+name
                check(prefix+col+'_present', col in r.index)
                if col in r.index:
                    same(prefix+col, float(r[col]), result['passive'][name]-result['active'][name], atol=1e-8)
            # The original active value is only expected on original active
            # support, not on the newly intersected active/passive support.
            own_mask = masks['active'] & st_mask
            original = old_trials.loc[(cid, scenario, st)]
            own_result = independent_metrics(own_active_state, own_mask, dt, area)
            own_result.update(geo.temporal(np.where(own_mask, own_active_state, -1), dt, occurrence, breaks))
            for metric, expected in own_result.items():
                same(prefix+'independent_active_reference_'+metric, original[metric], expected, atol=1e-8)
            same(prefix+'common_denominator_n', r.active_valid_n, r.passive_valid_n)
            same(prefix+'common_denominator_s', r.active_valid_s, r.passive_valid_s)
            if joints is not None:
                j = joints[(joints.pair_id == pair_id) & (joints.scenario == scenario) & (joints.stage == st)]
                check(prefix+'sixteen_joint_cells', len(j) == 16)
                if len(j) == 16:
                    a_col = next((x for x in ['active_state', 'active_region', 'active'] if x in j), None)
                    p_col = next((x for x in ['passive_state', 'passive_region', 'passive'] if x in j), None)
                    s_col = next((x for x in ['seconds', 'joint_seconds', 'duration_s'] if x in j), None)
                    if a_col and p_col and s_col:
                        for _, cell in j.iterrows():
                            codes = {'N': 0, 'T': 1, 'P': 2, 'O': 3, 'other': 0, 'target_only': 1, 'psm_only': 2, 'overlap': 3}
                            a = codes.get(str(cell[a_col]), cell[a_col])
                            p = codes.get(str(cell[p_col]), cell[p_col])
                            mask = selected & (states['active'] == int(a)) & (states['passive'] == int(p))
                            same(prefix+f'joint_{a}_{p}_seconds', cell[s_col], dt[mask].sum())
            if coverage is not None:
                c = coverage[(coverage.pair_id == pair_id) & (coverage.scenario == scenario) & (coverage.stage == st)]
                check(prefix+'coverage_row', len(c) == 1)
                if len(c) == 1:
                    c = c.iloc[0]
                    for name, mask in [('active', masks['active']), ('passive', masks['passive']), ('common', common)]:
                        mask = mask & st_mask
                        for col in (name+'_valid_n', name+'_n'):
                            if col in c: same(prefix+'coverage_'+col, c[col], mask.sum())
                        for col in (name+'_valid_s', name+'_s'):
                            if col in c: same(prefix+'coverage_'+col, c[col], dt[mask].sum())
        scenarios_checked.append(scenario)
    return {'pair_id': pair_id, 'content_id': cid, 'task': task, 'raw_rows': len(active),
            'retained_rows': len(src), 'scenarios': scenarios_checked,
            'active_sha256': sha(active_path), 'passive_sha256': sha(passive_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--report', type=Path, default=WORK/'validation'/'independent_validation.json')
    parser.add_argument('--links', type=Path, default=LINKS)
    args = parser.parse_args()
    report = {'validator': str(Path(__file__).resolve()), 'output': str(args.output), 'pairs': []}
    try:
        sys.path.insert(0, str(FROZEN))
        import tl_geometry as geo
        for path in sorted(FROZEN.glob('*.py')):
            ref = REFERENCE/'code_snapshot'/path.name
            check('frozen_code/'+path.name, ref.exists() and sha(path) == sha(ref), {'sha256': sha(path)})
        check('temporal_constants', geo.HZ == 41 and geo.MAX_GAP_S == .1 and geo.MIN_DWELL_S == .12)
        links = json.loads(args.links.read_text(encoding='utf-8-sig'))
        expected = {}
        for key, entry in links['clips'].items():
            m = entry.get('meta', {})
            if m.get('src_skill') == 'novice' and m.get('observer_skill') == 'novice' and m.get('task_name') in STAGES:
                expected[Path(key).name] = entry
        check('strict_novice_novice_selection_16', len(expected) == 16)
        check('task_counts_A7_C9', Counter(v['meta']['task_name'] for v in expected.values()) == {'A': 7, 'C': 9})
        metrics = pd.read_csv(args.output/'pair_metrics.csv')
        coverage = pd.read_csv(args.output/'coverage.csv')
        joints = pd.read_csv(args.output/'region_joint.csv')
        manifest = pd.read_csv(args.output/'pair_manifest.csv')
        old_trials = pd.read_csv(REFERENCE/'trial_stage.csv')
        old_trials = old_trials[(old_trials.source == 'gaze') & (old_trials.scope == 'real_all') &
                                (old_trials.analysis_branch == 'decoded_expectation')].set_index(['content_id', 'scenario', 'stage'])
        check('manifest_exact_pair_set', set(manifest.pair_id) == set(expected))
        check('manifest_unique_pairs', not manifest.pair_id.duplicated().any())
        check('manifest_16_unique_active_contents', manifest.content_id.nunique() == 16)
        for _, row in manifest.iterrows():
            e = expected[row.pair_id]
            check('manifest_metadata/'+row.pair_id,
                  row.content_id == e['meta']['content_id'] and row.task == e['meta']['task_name'] and
                  int(row.observer_id) == e['meta']['observer'] and row.participant == 'User'+str(e['meta']['user']))
            check('manifest_passive_path/'+row.pair_id,
                  Path(row.passive_npy_path).resolve() == (Path(links['data_root'])/e['gaze_npy_path']).resolve())
            check('manifest_active_path/'+row.pair_id,
                  Path(row.active_npy_path).resolve() == (Path(links['data_root'])/e['target_locking_label_path']).resolve())
        check('pair_metric_exact_pair_set', set(metrics.pair_id) == set(expected))
        check('frame_exact_pair_set', {p.stem for p in (args.output/'frames').glob('*.npz')} == set(expected))
        check('scenario_exact_set', set(metrics.scenario) == set(SCENARIOS))
        check('pair_metric_unique_rows', not metrics.duplicated(['pair_id', 'scenario', 'stage']).any())
        for pair_id, entry in expected.items():
            try:
                report['pairs'].append(check_frame_pair(pair_id, entry, Path(links['data_root']), args.output, metrics, coverage, joints, geo, old_trials))
            except Exception as exc:
                check('pair_exception/'+pair_id, False, str(exc))
                report.setdefault('exceptions', []).append({'pair_id': pair_id, 'traceback': traceback.format_exc()})
        active_cols = [c for c in metrics if c.startswith('active_') and pd.api.types.is_numeric_dtype(metrics[c]) and c != 'active_C_reason']
        check('all_active_metrics_have_passive_and_delta', all('passive_'+c[7:] in metrics and 'delta_'+c[7:] in metrics for c in active_cols))
        check('active_reproduction_present', (args.output/'active_reproduction.csv').is_file())
        reproduction = pd.read_csv(args.output/'active_reproduction.csv')
        check('active_reproduction_all_match', reproduction['match'].all())
        same('active_reproduction_values', reproduction.recomputed_value, reproduction.reference_value, atol=1e-8)
    except Exception as exc:
        check('validation_exception', False, str(exc))
        report['traceback'] = traceback.format_exc()
    report['check_count'] = len(CHECKS)
    report['failed_checks'] = [c for c in CHECKS if not c['passed']]
    report['successful_check_count'] = len(CHECKS)-len(report['failed_checks'])
    report['check_storage'] = 'All failures retained; successful assertions counted. Reproduce details with the validator.'
    report['status'] = 'passed' if CHECKS and not report['failed_checks'] else 'failed'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'check_count': len(CHECKS), 'failures': len(report['failed_checks']), 'report': str(args.report)}, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
