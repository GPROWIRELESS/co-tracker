"""Reproducible novice-active gaze comparison against the current NPY GT."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

MODELS = ['all_ABCD', 'single_task', 'leave_out', 'spd_finetune', 'spd_zero_shot']
STAGES = {
    'A': ['reach ring', 'grab ring', 'transfer ring', 'place ring', 'drop ring'],
    'C': ['reach string', 'threading string through tower', 'wrap around psm', 'pull through and cinch'],
}
SINGLE = {'A': 43, 'B': 42, 'C': 44, 'D': 43}
LEAVE = {'A': 44, 'B': 44, 'C': 42, 'D': 43}


def jsd(p, q):
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    a, b = p > 0, q > 0
    value = .5 * ((p[a] * np.log2(p[a] / m[a])).sum() + (q[b] * np.log2(q[b] / m[b])).sum())
    assert -1e-12 <= value <= 1 + 1e-12
    return float(np.clip(value, 0, 1))


def specs(repo, task, content_id):
    logs = repo / 'logs'
    base = logs / 'novice_active_temporal_film_60epochs_20261027/models'
    items = [
        ('all_ABCD', 43, base / 'all_ABCD/seed_43/inference_test_k64'),
        ('single_task', SINGLE[task], base / f'single_{task}/seed_{SINGLE[task]}/inference_test_k64'),
        ('leave_out', LEAVE[task], base / f'leave_out_{task}/seed_{LEAVE[task]}/inference_test_k64'),
    ]
    items += [('spd_finetune', s, logs / f'dvss_finetune_spd_full_20261029/seed_{s}/inference_test_k64') for s in [42, 43, 44]]
    items += [('spd_zero_shot', 43, logs / 'spd_fls1_pretrain/full_stride30_val10_100epochs_3seeds/seed_43/dvss_zero_shot/inference_test_k64')]
    return [(model, seed, folder / f'{content_id}_predictions.npz') for model, seed, folder in items]


def write_csv(out, name, rows):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(out / name, index=False, encoding='utf-8-sig', float_format='%.12g')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(r'C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch'))
    parser.add_argument('--data-root', type=Path, default=Path(r'D:\Dataset\dvss'))
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'deliverable')
    parser.add_argument('--ffprobe', default=r'C:\ProgramData\anaconda3\envs\torch\Library\bin\ffprobe.exe')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'per_frame').mkdir(exist_ok=True)
    (out / 'stage_distributions').mkdir(exist_ok=True)
    assert abs(jsd([1, 0], [0, 1]) - 1) < 1e-12
    assert jsd([.25, .75], [.25, .75]) == 0
    index_path = args.repo / 'dvss_pytorch/Index/index.json'
    catalog = json.loads(index_path.read_text(encoding='utf-8'))
    clips = [v for v in catalog['clips'].values() if v['domain'] == 'active' and v['meta']['split'] == 'test' and v['meta']['src_skill'] == 'novice']
    assert len(clips) == 15
    assert len({v['meta']['user'] for v in clips}) == 10
    clip_audit, run_audit, manifest, distance_runs, stage_runs = [], [], [], [], []
    stage_coverage = []
    def source(path, kind, **extra):
        stat = path.stat()
        manifest.append(dict(kind=kind, path=str(path), size_bytes=stat.st_size,
                             mtime_utc=datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat(),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest() if kind in ['index', 'gt', 'metadata'] else '', **extra))
    source(index_path, 'index')
    for seed in [42, 43, 44]:
        for rel in ['config.json', 'inference_test_k64/provenance.json']:
            source(args.repo / f'logs/dvss_finetune_spd_full_20261029/seed_{seed}' / rel, 'metadata', model='spd_finetune', seed=seed)
    grid_y, grid_x = np.indices((32, 40), dtype=np.float64)
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    unit_x, unit_y = grid_x / 39, grid_y / 31
    assert unit_x[0] == 0 and unit_y[0] == 0 and unit_x[-1] == 1 and unit_y[-1] == 1
    for clip_no, item in enumerate(clips, 1):
        meta = item['meta']
        cid, task, user = meta['content_id'], meta['task_name'], meta['user']
        gt_path, video_path = args.data_root / item['gaze_npy_path'], args.data_root / item['video_path']
        print(f'[{clip_no}/{len(clips)}] {cid} task {task}: reading predictions', flush=True)
        source(gt_path, 'gt', video_id=cid, task=task)
        source(video_path, 'video_header', video_id=cid, task=task)
        gt = np.load(gt_path, allow_pickle=False)
        xy = np.column_stack([gt['gaze_x'], gt['gaze_y']]).astype(np.float64)
        gt_finite = np.isfinite(xy).all(axis=1)
        gt_good = gt_finite & ((xy >= 0) & (xy <= 1)).all(axis=1)
        has_stage = 'stage' in gt.dtype.names
        stages = np.char.lower(np.char.strip(gt['stage'].astype(str))) if has_stage else np.full(len(gt), '', dtype='U1')
        if has_stage:
            unknown = set(np.unique(stages)) - set(STAGES.get(task, [])) - {'', 'none'}
            assert not unknown, (cid, unknown)
        metadata = json.loads(subprocess.check_output([args.ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(video_path)], text=True))['streams'][0]
        width, height = int(metadata['width']), int(metadata['height'])
        assert int(metadata['nb_frames']) == len(gt), cid
        assert (width, height) == (1280, 1024)
        diagonal = np.hypot(width - 1, height - 1)
        loaded = []
        common = np.flatnonzero(gt_good)
        for model, seed, path in specs(args.repo, task, cid):
            source(path, 'prediction', model=model, seed=seed, video_id=cid, task=task)
            with np.load(path, allow_pickle=False) as z:
                p = z['probability']
                ids = z['frame_index'].astype(np.int64)
                number, target, valid = z['frame_number'], z['target'], z['valid'].astype(bool)
                timestamps, codes = z['timestamp_ns'], z['valid_code']
            assert p.shape == (len(ids), 32, 40)
            assert len(np.unique(ids)) == len(ids) and np.all((ids >= 0) & (ids < len(gt)))
            assert np.array_equal(number, gt['Frame_Number'][ids]), f'Frame identity mismatch: {path}'
            mass = p.sum(axis=(1, 2), dtype=np.float64)
            p_good = np.isfinite(p).all(axis=(1, 2)) & (p >= 0).all(axis=(1, 2)) & (mass > 0)
            common = np.intersect1d(common, ids[p_good & valid], assume_unique=True)
            delta = np.max(np.abs(target.astype(np.float64) - xy[ids]), axis=1)
            finite_delta = delta[np.isfinite(delta)]
            run_audit.append(dict(video_id=cid, task=task, model=model, seed=seed,
                                  prediction_rows=len(ids), invalid_probability_rows=int((~p_good).sum()),
                                  max_probability_mass_deviation=float(np.max(np.abs(mass[p_good] - 1))),
                                  cached_target_changed_rows=int((delta > 1e-6).sum()),
                                  max_cached_target_delta=float(finite_delta.max()) if len(finite_delta) else None,
                                  timestamp_mismatch_rows=int((timestamps != gt['Timestamp(ns)'][ids]).sum()),
                                  valid_code_mismatch_rows=int((codes != gt['valid'][ids]).sum()),
                                  saved_valid_vs_current_finite_mismatch=int((valid != gt_finite[ids]).sum())))
            row_lookup = np.full(len(gt), -1, dtype=np.int64)
            row_lookup[ids] = np.arange(len(ids))
            loaded.append((model, seed, p, row_lookup))
        assert len(common) > 0
        xy_common, stage_common = xy[common], stages[common]
        expected_stages = STAGES.get(task, []) if has_stage else []
        present_stages = [s for s in expected_stages if np.any(stage_common == s)]
        for s in expected_stages:
            stage_coverage.append(dict(video_id=cid, task=task, user=user, stage=s, n_frames=int((stage_common == s).sum())))
        q_maps = {}
        for s in present_stages:
            coords = xy_common[stage_common == s]
            total = np.zeros(1280, dtype=np.float64)
            for start in range(0, len(coords), 512):
                points = coords[start:start + 512]
                log_g = -.5 * ((grid_x[None, :] - 39 * points[:, 0, None]) ** 2 + (grid_y[None, :] - 31 * points[:, 1, None]) ** 2)
                log_g -= log_g.max(axis=1, keepdims=True)
                g = np.exp(log_g)
                g /= g.sum(axis=1, keepdims=True)
                total += g.sum(axis=0)
            q_maps[s] = total / len(coords)
        run_names, run_coords, run_errors, stage_predictions = [], [], [], []
        for model, seed, p, row_lookup in loaded:
            coords = np.empty((len(common), 2), dtype=np.float64)
            p_sums = {s: np.zeros(1280, dtype=np.float64) for s in present_stages}
            for start in range(0, len(common), 512):
                stop = min(start + 512, len(common))
                pp = p[row_lookup[common[start:stop]]].astype(np.float64).reshape(stop-start, -1)
                pp /= pp.sum(axis=1, keepdims=True)
                coords[start:stop, 0] = pp @ unit_x
                coords[start:stop, 1] = pp @ unit_y
                for s in present_stages:
                    p_sums[s] += pp[stage_common[start:stop] == s].sum(axis=0)
            errors_px = np.linalg.norm((coords - xy_common) * [width - 1, height - 1], axis=1)
            errors_pct = errors_px * 100 / diagonal
            distance_runs.append(dict(model=model, seed=seed, video_id=cid, task=task, user=user, n_frames=len(common), mean_distance_pct_diagonal=float(errors_pct.mean()), mean_distance_px=float(errors_px.mean())))
            run_names.append(f'{model}_seed{seed}')
            run_coords.append(coords)
            run_errors.append(errors_pct)
            model_maps = []
            for s in present_stages:
                n = int((stage_common == s).sum())
                p_stage = p_sums[s] / n
                stage_runs.append(dict(model=model, seed=seed, video_id=cid, task=task, user=user, stage=s, n_frames=n, jsd=jsd(p_stage, q_maps[s])))
                model_maps.append(p_stage.reshape(32, 40))
            if present_stages:
                stage_predictions.append(model_maps)
        model_errors = np.stack([np.mean([e for name, e in zip(run_names, run_errors) if name.startswith(model + '_seed')], axis=0) for model in MODELS])
        np.savez_compressed(out / 'per_frame' / f'{cid}_distances.npz',
                            frame_index=common, frame_number=gt['Frame_Number'][common], gt_xy=xy_common,
                            stage=stage_common, model_names=np.array(MODELS), distance_pct_diagonal=model_errors,
                            run_names=np.array(run_names), run_pred_xy=np.stack(run_coords), run_distance_pct_diagonal=np.stack(run_errors),
                            width=width, height=height)
        if present_stages:
            np.savez_compressed(out / 'stage_distributions' / f'{cid}_stage_distributions.npz',
                                stage_names=np.array(present_stages), run_names=np.array(run_names),
                                gt_probability=np.stack([q_maps[s].reshape(32, 40) for s in present_stages]),
                                pred_probability=np.array(stage_predictions),
                                n_frames=np.array([int((stage_common == s).sum()) for s in present_stages]))
        clip_audit.append(dict(video_id=cid, task=task, user=user, gt_rows=len(gt), gt_finite_rows=int(gt_finite.sum()),
                               gt_in_bounds_rows=int(gt_good.sum()), common_valid_frames=len(common),
                               valid_gt_not_in_common=int(gt_good.sum())-len(common), has_stage=has_stage,
                               stage_annotated_common_frames=int(np.isin(stage_common, expected_stages).sum()),
                               width=width, height=height))
        print(f'  complete: {len(common):,} common frames; {len(present_stages)} stages', flush=True)
        del loaded, p
    dr = write_csv(out, 'video_distance_by_seed.csv', distance_runs)
    sr = write_csv(out, 'stage_jsd_by_seed.csv', stage_runs)
    group_cols = ['model', 'video_id', 'task', 'user', 'n_frames']
    vd = dr.groupby(group_cols, sort=False, as_index=False)[['mean_distance_pct_diagonal', 'mean_distance_px']].mean()
    vd['n_seeds'] = vd['model'].map(lambda x: 3 if x == 'spd_finetune' else 1)
    write_csv(out, 'video_distance.csv', vd)
    sv = sr.groupby(group_cols + ['stage'], sort=False, as_index=False)['jsd'].mean()
    sv['n_seeds'] = sv['model'].map(lambda x: 3 if x == 'spd_finetune' else 1)
    write_csv(out, 'stage_jsd_video.csv', sv)
    summary_rows = []
    for (task, stage, model), g in sv.groupby(['task', 'stage', 'model'], sort=False):
        summary_rows.append(dict(task=task, stage=stage, model=model, mean_jsd=float(g.jsd.mean()),
                                 n_videos=len(g), n_users=g.user.nunique(), n_frames=int(g.n_frames.sum())))
    ss = write_csv(out, 'stage_jsd_summary.csv', summary_rows)
    write_csv(out, 'stage_coverage.csv', stage_coverage)
    audit_clips = write_csv(out, 'audit_clips.csv', clip_audit)
    audit_runs = write_csv(out, 'audit_predictions.csv', run_audit)
    write_csv(out, 'input_manifest.csv', manifest)
    # User-cluster resampling preserves the equal-video point estimator.
    pivot = vd.pivot(index='video_id', columns='model', values='mean_distance_pct_diagonal').reindex(columns=MODELS)
    user_by_video = vd.drop_duplicates('video_id').set_index('video_id').loc[pivot.index, 'user'].to_numpy()
    users = np.unique(user_by_video)
    rng = np.random.default_rng(20261109)
    sampled = rng.integers(0, len(users), size=(10000, len(users)))
    cluster_count = np.stack([(sampled == i).sum(axis=1) for i in range(len(users))], axis=1)
    video_count = cluster_count[:, np.searchsorted(users, user_by_video)]
    boot = video_count @ pivot.to_numpy() / video_count.sum(axis=1, keepdims=True)
    summary = []
    for i, model in enumerate(MODELS):
        g = vd[vd.model == model]
        lo, hi = np.quantile(boot[:, i], [.025, .975])
        summary.append(dict(model=model, mean_distance_pct_diagonal=float(g.mean_distance_pct_diagonal.mean()),
                            ci95_low=float(lo), ci95_high=float(hi), mean_distance_px=float(g.mean_distance_px.mean()),
                            n_videos=len(g), n_users=g.user.nunique(), n_frames=int(g.n_frames.sum())))
    comparison = write_csv(out, 'model_comparison.csv', summary)
    pairwise = []
    for i in range(len(MODELS)):
        for j in range(i+1, len(MODELS)):
            lo, hi = np.quantile(boot[:, i]-boot[:, j], [.025, .975])
            pairwise.append(dict(model_a=MODELS[i], model_b=MODELS[j], difference_a_minus_b=float((pivot.iloc[:, i]-pivot.iloc[:, j]).mean()), ci95_low=float(lo), ci95_high=float(hi)))
    write_csv(out, 'pairwise_distance.csv', pairwise)
    stage_wide = ss.pivot(index=['task', 'stage', 'n_videos', 'n_users', 'n_frames'], columns='model', values='mean_jsd').reindex(columns=MODELS)
    order = [(t, s) for t, stages in STAGES.items() for s in stages]
    stage_wide = stage_wide.reset_index()
    stage_wide['_order'] = [(order.index((t, s))) for t, s in zip(stage_wide.task, stage_wide.stage)]
    stage_wide = stage_wide.sort_values('_order').drop(columns='_order')
    write_csv(out, 'stage_jsd_comparison.csv', stage_wide)
    assert len(vd) == 75 and len(dr) == 105 and len(ss) == 45
    assert comparison.n_frames.nunique() == 1
    assert np.isfinite(comparison.select_dtypes('number')).all().all()
    changed = sorted(audit_runs.loc[audit_runs.cached_target_changed_rows > 0, 'video_id'].unique().tolist())
    audit = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 filter=dict(domain='active', split='test', src_skill='novice'),
                 n_videos=15, n_users=10, n_prediction_files=105, n_models=5, n_input_files=len(manifest),
                 total_gt_frames=int(audit_clips.gt_rows.sum()), common_valid_frames=int(audit_clips.common_valid_frames.sum()),
                 gt_invalid_or_out_of_bounds_frames=int((audit_clips.gt_rows-audit_clips.gt_in_bounds_rows).sum()),
                 valid_gt_frames_missing_from_common=int(audit_clips.valid_gt_not_in_common.sum()),
                 n_stage_videos=int(audit_clips.has_stage.sum()), n_stages=9,
                 stage_annotated_common_frames=int(audit_clips.stage_annotated_common_frames.sum()),
                 cached_target_changed_videos=changed,
                 timestamp_mismatch_videos=sorted(audit_runs.loc[audit_runs.timestamp_mismatch_rows > 0, 'video_id'].unique().tolist()),
                 frame_identity_check='All prediction frame_number values equal current NPY Frame_Number at frame_index.',
                 probability_invalid_rows=int(audit_runs.invalid_probability_rows.sum()),
                 max_probability_mass_deviation=float(audit_runs.max_probability_mass_deviation.max()),
                 gt_version='Current NPY files; SHA256 hashes in input_manifest.csv.',
                 bootstrap_replicates=10000, bootstrap_seed=20261109,
                 bootstrap_method='Paired user-cluster bootstrap, retaining every video within each resampled user; video-equal estimator.',
                 seed_aggregation='Average seed-level metrics within each video/stage; no probability ensemble.',
                 coordinate_method='Expected normalized coordinate from full probability grid; x=c/39, y=r/31.',
                 distance_method='Euclidean pixels scaled by (W-1,H-1); 100*d/hypot(W-1,H-1). Equal-video aggregate.',
                 jsd_method='Mean predicted probability vs mean frame-normalized Gaussian GT map, sigma=1 grid cell, 32x40 grid, log2 JSD.',
                 missing_stage_policy='Only A/C; none/empty omitted from stage JSD, retained for valid-frame distance.',
                 stage_order=STAGES)
    (out/'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    if Path(__file__).resolve() != (out/'reproduce_analysis.py').resolve():
        shutil.copy2(__file__, out/'reproduce_analysis.py')
    print(comparison.to_string(index=False), flush=True)
    print(stage_wide.to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
