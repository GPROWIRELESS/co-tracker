"""Apply the exact max-DLC pipeline to paired A/C test-model gaze outputs.

Independent record/stage signed-peak side selection for every gaze source;
1,000 participant bootstraps and 100 newly drawn unpaired references.
"""
import model_xcorr_helpers as H
import ast
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict, Counter
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np

PROJECT = H.PROJECT
SOURCE = PROJECT/'outputs/gaze_psm_xcorr_dlc'
REFERENCE = PROJECT/'outputs/gaze_psm_xcorr_max_dlc'
sys.path.insert(0, str(PROJECT))


def extra_legacy():
    path = PROJECT/'tools/gaze_psm_xcorr_max_dlc.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    env = dict(np=np, defaultdict=defaultdict, Path=Path, save_json=H.LEG.save_json)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'_pairings', 'generate_figures'}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return SimpleNamespace(**env)


EXTRA = extra_legacy()


def compare_curves(curves, reference):
    lookup = {H.curve_key(c): c for c in reference}
    differences = []
    for c in curves:
        saved = lookup[H.curve_key(c)]
        for field in ('r', 'pairs', 'occurrences', 'gaze_rms', 'psm_rms', 'distance_median_px', 'proximity'):
            a, b = np.asarray(c[field], float), np.asarray(saved[field], float)
            assert np.array_equal(np.isfinite(a), np.isfinite(b)), (H.curve_key(c), field)
            assert np.allclose(a, b, atol=1e-8, rtol=1e-8, equal_nan=True), (H.curve_key(c), field)
            differences.append(float(np.max(abs(a[np.isfinite(a)]-b[np.isfinite(b)]))) if np.isfinite(a).any() else 0.)
        if 'selected_side' in saved:
            assert c['selected_side'] == saved['selected_side'], H.curve_key(c)
            assert c['selection_status'] == saved['selection_status'], H.curve_key(c)
    return dict(curves=len(curves), checks=len(differences), max_abs_difference=max(differences))


def load_cache(output, name):
    data = {}
    for task, ids in H.TASKS.items():
        for cid in ids:
            with np.load(output/'cache'/name/f'{cid}.npz', allow_pickle=False) as z:
                data[cid] = {k: z[k] for k in z.files}
    return data


def null_pair_job(payload):
    cid, donor_id, datasets, records, config = payload
    curves = H.LEG.calculate(records[cid], datasets[cid], SimpleNamespace(**config), donor=datasets[donor_id])
    # This is the exact _null_job output compaction; retain only r/lag arrays.
    compact = [{k: v for k, v in c.items() if not isinstance(v, np.ndarray) or k in {'lag_s', 'r'}} for c in curves]
    return (cid, donor_id), H.LEG.select_sides(compact)[0]


def save_source(output, name, curves, details, quality, config, references):
    users, groups, contrasts = H.GROUP.aggregate(curves, bootstrap=config['bootstrap'], seed=config['seed'],
                                               min_users=config['min_users'], baseline_replicates=references)
    result = dict(record_curves=curves, user_curves=users, group_curves=groups, contrasts=contrasts)
    dest = output/name
    dest.mkdir(parents=True, exist_ok=True)
    H.LEG.save_json(dest/'config.json', dict(config, gaze_source_id=name, output=str(dest)))
    H.LEG.save_json(dest/'results.json', result)
    qc = [q for q in quality if q['source'] == name]
    H.LEG.save_json(dest/'quality.json', qc)
    H.LEG.generate_outputs(dest, curves, users, groups, contrasts, qc, make_plots=False)
    H.LEG._write(dest/'record_summary.csv', [dict(H.LEG._summary(c), **H.feature(c),
                                               selected_side=c['selected_side'], selection_status=c['selection_status']) for c in curves])
    H.LEG._write(dest/'side_selection.csv', details)
    EXTRA.generate_figures(dest, users, groups)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--reuse-prepared', action='store_true')
    p.add_argument('--reuse-computed', action='store_true')
    p.add_argument('--reuse-null', action='store_true')
    args = p.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((REFERENCE/'config.json').read_text())
    config.update(output=str(output), source=str(SOURCE), method_reference=str(REFERENCE),
                  records=[c for ids in H.TASKS.values() for c in ids], workers=args.workers,
                  score_correlation=False, n_predictions=50, model_gaze='probability expectation',
                  gaze_support='intersection of raw GT validity and all five model interpolation supports before identical smoothing',
                  side_comparison='each gaze source independently selects largest signed record/stage peak, exactly as max_dlc',
                  group_comparison='paired bootstrap on shared complete-lag participants',
                  synchronous_band_s=1/config['hz'],
                  unpaired_reference='100 new same-task different-user draws restricted to the same 10 test videos; independently reselect side for each donor pairing')
    assert config['shuffles'] == 100 and config['bootstrap'] == 1000
    assert config['psm_source'] == 'dlc_left_right_xy' and config['dlc_likelihood_filter'] is None
    H.LEG.save_json(output/'config.json', config)
    snapshot = output/'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    files = [PROJECT/'tools'/f for f in ('gaze_psm_core.py', 'gaze_psm_group.py', 'gaze_psm_pipeline_dlc.py',
             'gaze_psm_xcorr_max_dlc.py', 'gaze_psm_report.py', 'gaze_psm_figures.py')]
    files += [Path(__file__), Path(H.__file__)]
    hashes = {str(f): H.sha(f) for f in files}
    for f in files:
        shutil.copy2(f, snapshot/f.name)
    for f in (REFERENCE/'config.json', REFERENCE/'results.json', SOURCE/'config.json', SOURCE/'results.json'):
        hashes[str(f)] = H.sha(f)
    if args.reuse_prepared or args.reuse_computed:
        quality = json.loads((output/'quality.json').read_text())
        with (output/'alignment_audit.csv').open(newline='') as f:
            checks = list(csv.DictReader(f))
    else:
        quality, checks = H.prepare(output, SOURCE, config)
    assert all(int(c['extra_masked_gt_l']) == int(c['extra_masked_gt_r']) == 0 for c in checks), 'Common support changes legacy GT'
    for row in checks:
        hashes[row['cache_path']] = row['cache_sha256']
    computed = output/'left_right_intermediate'
    computed.mkdir(exist_ok=True)
    names = ['gt_legacy', *H.SOURCE_NAMES]
    by_source = {name: [] for name in names}
    if args.reuse_computed:
        for name in names:
            by_source[name] = [H.as_arrays(c) for c in json.loads((computed/f'{name}.json').read_text())]
    else:
        jobs = [(str(output), name, cid, task, config) for name in names for task, ids in H.TASKS.items() for cid in ids]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (name, cid, curves) in enumerate(pool.map(H.compute_job, jobs), 1):
                by_source[name].extend(curves)
                print(f'Observed {i}/{len(jobs)}: {name}/{cid}', flush=True)
        for name, curves in by_source.items():
            H.LEG.save_json(computed/f'{name}.json', curves)
    old_raw = json.loads((SOURCE/'results.json').read_text())['record_curves']
    old_max = json.loads((REFERENCE/'results.json').read_text())['record_curves']
    raw_regression = compare_curves(by_source['gt_legacy'], old_raw)
    selected, selection = {}, {}
    for name in names:
        selected[name], selection[name] = H.LEG.select_sides(by_source[name])
    max_regression = compare_curves(selected['gt_legacy'], old_max)
    compare_curves(selected['gt'], selected['gt_legacy'])
    print(f'Legacy DLC regression: {raw_regression}; selected-max regression: {max_regression}', flush=True)
    H.LEG.save_json(output/'legacy_regression.json', dict(left_right=raw_regression, selected_max=max_regression))
    # Preserve reference record order before drawing controls, exactly as _load_caches.
    records = {}
    for c in old_raw:
        if c['content_id'] in config['records']:
            records.setdefault(c['content_id'], {'content_id': c['content_id'], 'Task Name': c['task']})
    pairings = list(EXTRA._pairings(records, config['shuffles'], config['seed']))
    unique_pairs = sorted(set(pair for replicate in pairings for pair in replicate))
    assert len(unique_pairs) == 40 and len(pairings) == 100
    H.LEG._write(output/'unpaired_pairings.csv', [dict(replicate=i, target=cid, donor=donor)
                                                for i, rep in enumerate(pairings) for cid, donor in rep])
    null_root = output/'unpaired_cache'
    null_root.mkdir(exist_ok=True)
    models = {}
    equivalence = []
    for name in H.SOURCE_NAMES:
        cache_file = null_root/f'{name}.json'
        datasets = load_cache(output, name)
        if args.reuse_null and cache_file.exists():
            null = {(item['target'], item['donor']): [H.as_arrays(c) for c in item['curves']]
                    for item in json.loads(cache_file.read_text())}
        else:
            null = {}
            payloads = [(cid, donor, datasets, records, config) for cid, donor in unique_pairs]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i, (pair, curves) in enumerate(pool.map(null_pair_job, payloads), 1):
                    null[pair] = curves
                    if i % 10 == 0:
                        print(f'Unpaired {name}: {i}/{len(unique_pairs)} unique donor pairs', flush=True)
            H.LEG.save_json(cache_file, [dict(target=pair[0], donor=pair[1], curves=curves) for pair, curves in null.items()])
        # Repeated donor pairs produce deterministic curves: verify one pairing freshly.
        first = pairings[0][0]
        _, fresh = null_pair_job((*first, datasets, records, config))
        saved = null[first]
        assert len(saved) == len(fresh)
        for a, b in zip(fresh, saved):
            assert H.curve_key(a) == H.curve_key(b) and a['selected_side'] == b['selected_side']
            assert np.allclose(a['r'], b['r'], atol=0, rtol=0, equal_nan=True)
        equivalence.append(dict(source=name, target=first[0], donor=first[1], exact_match=True))
        def references():
            for replicate in pairings:
                # select_sides returns sorted keys, as the original _selected_null_job.
                curves = [c for pair in replicate for c in null[pair]]
                yield sorted(curves, key=lambda c: tuple(c[k] for k in ('content_id', 'user', 'task', 'signal', 'stage', 'mode')))
        models[name] = save_source(output, name, selected[name], selection[name], quality, config, references())
        print(f'Aggregated {name}: bootstrap=1000, unpaired=100, figures saved', flush=True)
    paired, groups, boot = H.paired_analysis(models, config)
    truths = {H.curve_key(c): c for c in selected['gt']}
    lookups = {name: {H.curve_key(c): c for c in selected[name]} for name in H.SOURCE_NAMES}
    for row in paired:
        truth = truths[H.curve_key(row)]
        pred = lookups[row['model']][H.curve_key(row)]
        row.update(gt_selected_side=truth['selected_side'], model_selected_side=pred['selected_side'],
                   gt_selection_status=truth['selection_status'], model_selection_status=pred['selection_status'],
                   same_selected_side=bool(truth['selected_side'] and truth['selected_side'] == pred['selected_side']),
                   comparison_definition='independently-selected maximum-correlation PSM statistics')
    H.LEG._write(output/'paired_record_summary.csv', paired)
    H.LEG._write(output/'paired_group_summary.csv', groups)
    H.LEG._write(output/'paired_bootstrap.csv', boot)
    details = []
    for name in H.SOURCE_NAMES:
        for d in selection[name]:
            cid_key = (d['content_id'], d['task'], 'max', d['signal'], d['stage'], d['mode'])
            gt_side = truths[cid_key]['selected_side']
            details.append(dict(d, source=name, gt_selected_side=gt_side,
                                same_side_as_gt=bool(gt_side and d['selected_side'] == gt_side)))
    H.LEG._write(output/'side_selection.csv', details)
    eligibility = []
    for name in H.SOURCE_NAMES:
        group_lookup = {H.group_key(g): g for g in models[name]['group_curves']}
        for c in selected[name]:
            f = H.feature(c)
            gr = group_lookup[H.group_key(c)]
            eligibility.append(dict(source=name, **{k: c[k] for k in ('content_id', 'user', *H.FIELDS)},
                                    **f, selected_side=c['selected_side'], selection_status=c['selection_status'],
                                    group_status=gr['status'], group_usable=H.supported_group(gr),
                                    group_n_users=gr['n_stable_users'], eligible=f['record_eligible'] and H.supported_group(gr)))
    H.LEG._write(output/'phase_validity.csv', eligibility)
    H.LEG.save_json(output/'comparison_results.json', dict(config=config, models=models, paired_records=paired, paired_groups=groups))
    rng = np.random.default_rng(9)
    gaze = rng.normal(size=(1000, 2))
    psm = np.full_like(gaze, np.nan)
    psm[5:] = gaze[:-5]
    known = H.LEG.analyze_curve(gaze, psm, config['hz'], max_lag=.5, min_overlap=1,
                               min_gaze_rms=0, min_psm_rms=0, compute_spatial=False)
    recovered = float(known['lag_s'][np.nanargmax(known['r'])])
    assert abs(recovered - 5/config['hz']) < 1e-12
    assert all(H.sha(path) == digest for path, digest in hashes.items())
    validation = dict(n_prediction_files=50, n_test_videos=10, n_participants=7,
                      left_right_gt_regression=raw_regression, max_dlc_gt_regression=max_regression,
                      gt_shared_support_unchanged=True, known_delay_sign_test_s=recovered,
                      shuffles=100, bootstrap=1000, unique_donor_pairs_per_source=40,
                      cached_null_equivalence=equivalence, source_sha256=hashes, source_files_unchanged=True,
                      side_selection='record_stage_signed_peak independently for each gaze source and null pairing',
                      curve_counts={name: len(curves) for name, curves in selected.items()},
                      paired_record_count=len(paired), eligible_record_pairs=sum(r['eligible'] for r in paired),
                      group_counts={name: len(result['group_curves']) for name, result in models.items()},
                      score_correlation_performed=False, all_assertions_passed=True)
    H.LEG.save_json(output/'validation.json', validation)
    print(json.dumps({k: v for k, v in validation.items() if k not in {'source_sha256', 'cached_null_equivalence'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
