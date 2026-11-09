"""Independent v2/v3 checks, including raw-NPY coordinate-only support in v3."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def raw_gaze_support(f, row, cfg):
    """Reconstruct selection from the source; do not call analysis input helpers."""
    path = Path(row.npy_path)
    assert sha256(path) == row.npy_sha256, f'Source NPY changed: {path}'
    source = np.load(path, allow_pickle=False, mmap_mode='r')
    ids = np.asarray(source['Frame_Number'], dtype=np.int64)
    assert len(np.unique(ids)) == len(ids), f'Duplicate source frames: {path}'
    lookup = {int(frame): i for i, frame in enumerate(ids)}
    source = source[[lookup[int(frame)] for frame in f['frame_number']]]
    np.testing.assert_array_equal(source['Frame_Number'], f['frame_number'])
    norm = np.column_stack([source['gaze_x'], source['gaze_y']]).astype(float)
    coordinate_ok = np.isfinite(norm).all(axis=1) & (norm >= 0).all(axis=1) & (norm <= 1).all(axis=1)
    if 'gaze_coordinate_valid' in f.files:
        np.testing.assert_array_equal(coordinate_ok, f['gaze_coordinate_valid'])
    pixels = norm * f['resolution']
    pixels[~coordinate_ok] = np.nan
    np.testing.assert_allclose(f['gaze_xy'], pixels, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(f['gaze_unmasked'], pixels, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(f['valid_code'], source['valid'])
    assert int(coordinate_ok.sum()) == int(row.gaze_valid_n) == int(row.gaze_coordinate_valid_n)

    aliases = {'rr':'reach ring', 'gr':'grab ring', 'tr':'transfer ring', 'pr':'place ring',
               'wrap':'wrap around psm', 'cinch':'pull through and cinch',
               'thread':'threading string through tower'}
    stage_key = 'stage' if 'stage' in source.dtype.names else 'event'
    normalized = [' '.join(str(value).strip().lower().split()) for value in source[stage_key]]
    stages = np.asarray([aliases.get(value, value) for value in normalized])
    np.testing.assert_array_equal(stages, f['stage'])
    included = np.isin(stages, cfg['stages'][row.task])
    np.testing.assert_array_equal(included, f['included'])
    slots = np.full((len(source), 5, 2), np.nan)
    for i in range(5):
        x, y = f'point_{i+1}_x', f'point_{i+1}_y'
        if x in source.dtype.names:
            slots[:, i] = np.column_stack([source[x], source[y]])
    np.testing.assert_allclose(slots, f['target_slots'], rtol=0, atol=0, equal_nan=True)
    used = np.isfinite(slots).all(axis=2) & (slots >= 0).all(axis=2) & (slots < f['resolution']).all(axis=2)
    count = used.sum(axis=1)
    np.testing.assert_array_equal(count, f['target_count'])
    centers = np.full((len(source),2),np.nan)
    np.divide(np.where(used[:,:,None],slots,0).sum(axis=1),count[:,None],out=centers,where=count[:,None]>0)
    np.testing.assert_allclose(centers,f['target_xy'],rtol=0,atol=0,equal_nan=True)
    gripper_ok = np.ones(len(source), dtype=bool)
    for key in ('left_xy', 'right_xy'):
        xy = f[key]
        gripper_ok &= np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1) & (xy < f['resolution']).all(axis=1)
    geometry = included & (count > 0) & gripper_ok
    for scenario, (_, _, codes, smooth) in cfg['scenarios'].items():
        assert codes is None, f'v3 scenario restricts status codes: {scenario}'
        if smooth:
            xy = f['smooth_xy']
            ok = np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1) & (xy <= f['resolution']).all(axis=1)
        else:
            ok = coordinate_ok
        np.testing.assert_array_equal(f['valid_real_' + scenario], geometry & ok)
    return int(coordinate_ok.sum())


def validate_gaze_rows(f, rows, coverage, cfg):
    """Verify every GT row against native classifications, including unpaired GT84."""
    checked = 0
    for scenario,(rt,rp,_,smooth) in cfg['scenarios'].items():
        xy = f['smooth_xy'] if smooth else f['gaze_xy']
        target = np.sum((xy-f['target_xy'])**2,axis=1)<=rt*rt
        left = np.sum((xy-f['left_xy'])**2,axis=1)<=rp*rp
        right = np.sum((xy-f['right_xy'])**2,axis=1)<=rp*rp
        states = target.astype(int)+2*(left|right).astype(int)
        real = f['valid_real_'+scenario]
        np.testing.assert_array_equal(f['state_gaze_'+scenario], np.where(real,states,-1))
        for (scope,stage),group in rows[rows.scenario==scenario].groupby(['scope','stage']):
            native = np.ones(len(real),bool) if stage=='whole' else f['stage']==stage
            mask = native & f['valid_'+('real' if scope=='real_all' else 'paired')+'_'+scenario]
            weights = f['dt_s'][mask]
            seconds = weights.sum()
            np.testing.assert_array_equal(group.valid_n.to_numpy(),np.full(len(group),mask.sum()))
            np.testing.assert_allclose(group.valid_s,seconds,rtol=1e-12,atol=1e-9)
            if seconds>0:
                share=np.bincount(states[mask],weights=weights,minlength=4)/seconds
                np.testing.assert_allclose(group[['L_N','L_T','L_P','L_O']].to_numpy(),np.tile(share,(len(group),1)),atol=1e-10)
            checked += len(group)
        for _,row in coverage[coverage.scenario==scenario].iterrows():
            native = f['included'] if row.stage=='whole' else f['stage']==row.stage
            mask = native & real
            assert int(row.valid_n)==int(mask.sum())
            np.testing.assert_allclose(row.valid_s,f['dt_s'][mask].sum(),rtol=1e-12,atol=1e-9)
    return checked


def compare_v3(run, old, save=False):
    """Attribute changes only when recording inputs are byte-identical."""
    new_audit = pd.read_csv(run / 'recording_audit.csv').set_index('content_id')
    old_audit = pd.read_csv(old / 'recording_audit.csv').set_index('content_id')
    assert set(new_audit.index) == set(old_audit.index)
    frames = []
    changed = []
    for cid, row in new_audit.iterrows():
        before = old_audit.loc[cid]
        differences = [key for key in ('npy_sha256', 'video_ts_sha256', 'dlc_sha256') if row[key] != before[key]]
        if differences:
            changed.append(dict(content_id=cid, changed_input_hashes=differences))
        with np.load(run/'frames'/f'{cid}.npz', allow_pickle=False) as new, np.load(old/'frames'/f'{cid}.npz', allow_pickle=False) as prev:
            np.testing.assert_array_equal(new['frame_number'], prev['frame_number'])
            np.testing.assert_array_equal(new['timestamp_ns'], prev['timestamp_ns'])
            np.testing.assert_allclose(new['dt_s'], prev['dt_s'], rtol=0, atol=0)
            for scope in ('real', 'paired'):
                key = 'valid_' + scope + '_main'
                if key not in new.files:
                    continue
                a, b = prev[key], new[key]
                if not differences:
                    assert not np.any(a & ~b), f'v3 loses v2 main support: {cid}/{scope}'
                added, removed = b & ~a, a & ~b
                frames.append(dict(content_id=cid, task=row.task, scope=scope,
                    identical_recording_inputs=not differences,
                    coordinate_valid_old=int(before.gaze_coordinate_valid_n),
                    coordinate_valid_new=int(row.gaze_coordinate_valid_n),
                    gaze_valid_old=int(before.gaze_valid_n), gaze_valid_new=int(row.gaze_valid_n),
                    valid_n_old=int(a.sum()), valid_n_new=int(b.sum()),
                    added_n=int(added.sum()), removed_n=int(removed.sum()),
                    valid_s_old=float(new['dt_s'][a].sum()), valid_s_new=float(new['dt_s'][b].sum()),
                    added_s=float(new['dt_s'][added].sum()), removed_s=float(new['dt_s'][removed].sum())))
    delta = pd.DataFrame(frames)
    summaries = []
    for name, keys in [('trial_stage', ['content_id','analysis_branch','scenario','scope','stage','source']),
                       ('model_trial_errors', ['content_id','analysis_branch','scenario','stage','source'])]:
        a, b = (pd.read_csv(folder/(name+'.csv')) for folder in (old, run))
        a, b = a[a.scenario == 'main'], b[b.scenario == 'main']
        assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
        metrics = [col for col in b if col not in keys and col in a and
                   pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]) and
                   col not in ('task_num','attempt','round','score')]
        joined = a[keys+metrics].merge(b[keys+metrics], on=keys, suffixes=('_old','_new'), validate='one_to_one', how='outer', indicator=True)
        assert (joined['_merge']=='both').all(), f'Comparison row keys differ: {name}'
        joined = joined.drop(columns='_merge')
        for metric in metrics:
            joined[metric+'_delta'] = joined[metric+'_new'] - joined[metric+'_old']
        summaries.append(dict(table=name, rows=len(joined), compared_metrics=metrics))
        if save:
            joined.to_csv(run/(name+'_v2_v3_changes.csv'), index=False, encoding='utf-8-sig', na_rep='NA')
    old_pred = pd.read_csv(old/'prediction_audit.csv').set_index(['content_id','model'])
    new_pred = pd.read_csv(run/'prediction_audit.csv').set_index(['content_id','model'])
    assert set(old_pred.index) == set(new_pred.index)
    same_predictions = bool((new_pred.observed_sha256 == old_pred.loc[new_pred.index].observed_sha256).all())
    summary = dict(old_run=str(old), new_run=str(run), recording_count=len(new_audit),
        identical_recording_inputs=not changed, changed_recording_inputs=changed,
        identical_prediction_inputs=same_predictions, comparison_tables=summaries,
        gaze_valid_old=int(old_audit.gaze_valid_n.sum()), gaze_valid_new=int(new_audit.gaze_valid_n.sum()),
        per_scope={scope: dict(recordings=len(q), added_frames=int(q.added_n.sum()),
            removed_frames=int(q.removed_n.sum()), added_seconds=float(q.added_s.sum()),
            removed_seconds=float(q.removed_s.sum())) for scope,q in delta.groupby('scope')},
        old_config_sha256=sha256(old/'run_config.json'), new_config_sha256=sha256(run/'run_config.json'))
    if save:
        delta.to_csv(run/'frame_support_v2_v3_changes.csv', index=False, encoding='utf-8-sig')
        (run/'v2_v3_comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return summary


def validate_run(run,old=None,save_comparison=False):
    cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8-sig'))
    is_v3=int(cfg.get('version',2))>=3
    t=pd.read_csv(run/'trial_stage.csv')
    e=pd.read_csv(run/'model_trial_errors.csv')
    checked=0
    raw_checked=0
    audit=pd.read_csv(run/'prediction_audit.csv')
    recordings=pd.read_csv(run/'recording_audit.csv').set_index('content_id')
    coverage=pd.read_csv(run/'coverage.csv')
    raw_coordinate_checked=0
    gaze_rows_checked=0
    pair=t[t.scope=='paired_test'].groupby(['content_id','scenario','stage'])
    assert (pair.size()==24).all()
    assert (pair.valid_n.nunique()==1).all() and (pair.valid_s.nunique()==1).all()
    for fp in sorted((run/'frames').glob('*.npz')):
        with np.load(fp,allow_pickle=False) as f:
            stages=f['stage']; dt=f['dt_s']
            if is_v3:
                raw_coordinate_checked+=raw_gaze_support(f,recordings.loc[fp.stem],cfg)
                gaze_rows_checked+=validate_gaze_rows(f,t[(t.content_id==fp.stem)&(t.source=='gaze')],coverage[coverage.content_id==fp.stem],cfg)
            model_ok=np.ones(len(stages),dtype=bool)
            for _,prediction in audit[audit.content_id==fp.stem].iterrows():
                source=prediction['model']
                with np.load(prediction['prediction_path'],allow_pickle=False) as original:
                    samples=np.asarray(original['samples'],float)*f['resolution']
                    np.testing.assert_array_equal(original['frame_number'],f['frame_number'])
                    np.testing.assert_array_equal(original['timestamp_ns'],f['timestamp_ns'])
                model_ok &= np.isfinite(samples).all(axis=(1,2)) & (samples>=0).all(axis=(1,2)) & (samples<=f['resolution']).all(axis=(1,2))
                for decoder in ('expectation','argmax','sample'):
                    model_ok &= np.isfinite(f[f'xy_{decoder}_{source}']).all(axis=1)
                rng=np.random.default_rng(42)
                for scenario,rt,rp in [('main',50,50),('r25',25,25),('r100',100,100),('gripper100',50,100)]:
                    valid=f['valid_paired_'+scenario]
                    selected=[]
                    for stage in np.unique(stages[valid]):
                        ix=np.flatnonzero(valid&(stages==stage))
                        selected.extend(rng.choice(ix,size=min(64,len(ix)),replace=False))
                    ix=np.asarray(selected,dtype=int)
                    if not len(ix):continue
                    s=samples[ix]
                    target=np.sum((s-f['target_xy'][ix,None])**2,axis=2)<=rt*rt
                    left=np.sum((s-f['left_xy'][ix,None])**2,axis=2)<=rp*rp
                    right=np.sum((s-f['right_xy'][ix,None])**2,axis=2)<=rp*rp
                    labels=target.astype(int)+2*(left|right).astype(int)
                    counts=np.column_stack([(labels==k).mean(axis=1) for k in range(4)])
                    np.testing.assert_allclose(counts,f[f'q_{source}_{scenario}'][ix],rtol=0,atol=1e-8)
                    raw_checked+=len(ix)
            if is_v3 and len(audit[audit.content_id==fp.stem]):
                if 'model_common_valid' in f.files:
                    np.testing.assert_array_equal(f['model_common_valid'],model_ok)
                for scenario in cfg['scenarios']:
                    np.testing.assert_array_equal(f['valid_paired_'+scenario],f['valid_real_'+scenario]&model_ok)
            for key in f.files:
                if not key.startswith('valid_real_') and not key.startswith('valid_paired_'):continue
                assert not np.any(f[key] & np.isin(stages,['drop ring','none','']))
            for key in f.files:
                if not key.startswith('q_'):continue
                # Resolve model/scenario by a trial row, so names may contain underscores.
                rows=t[(t.content_id==fp.stem)&(t.scope=='paired_test')&(t.analysis_branch=='probability_mc')]
                matches=rows[["source","scenario"]].drop_duplicates()
                chosen=[(m,s) for m,s in matches.itertuples(index=False,name=None) if key==f'q_{m}_{s}']
                assert len(chosen)==1,key
                source,scenario=chosen[0]
                valid=f['valid_paired_'+scenario]
                q=f[key].astype(float)
                assert np.isfinite(q[valid]).all() and np.isnan(q[~valid]).all()
                np.testing.assert_allclose(q[valid].sum(axis=1),1,atol=1e-7)
                np.testing.assert_allclose(q[valid]*64,np.rint(q[valid]*64),atol=1e-6)
                state=f['state_gaze_'+scenario]
                for stage in rows.stage.unique():
                    mask=valid & ((stages==stage) if stage!='whole' else True)
                    if not mask.any() or dt[mask].sum()==0:continue
                    w=dt[mask]; mean=w@q[mask]/w.sum()
                    row=rows[(rows.source==source)&(rows.scenario==scenario)&(rows.stage==stage)].iloc[0]
                    np.testing.assert_allclose(mean,row[['L_N','L_T','L_P','L_O']].to_numpy(float),atol=1e-8)
                    err=e[(e.content_id==fp.stem)&(e.source==source)&(e.scenario==scenario)&
                          (e.stage==stage)&(e.analysis_branch=='probability_mc')].iloc[0]
                    for bit,event in [(1,'target'),(2,'gripper')]:
                        h=((state[mask]&bit)>0).astype(float)
                        p=q[mask,bit]+q[mask,3]
                        bs=np.sum(w*(p-h)**2)/w.sum()
                        np.testing.assert_allclose(bs,err['BS_'+event],atol=1e-8)
                    checked+=1
    primary=run/'main_probability'
    for path in primary.glob('*.csv'):
        frame=pd.read_csv(path)
        forbidden=[c for c in frame.columns if any(s in c for s in ('dwell','direct_rate','via_rate','FPR','FNR','FP_target','FN_target'))]
        assert not forbidden,(path,forbidden)
        if 'metric' in frame:
            values=frame.metric.dropna().astype(str)
            assert not values.str.contains(r'dwell|direct_rate|via_rate|FPR|FNR').any(),path
    assert not (primary/'confusion_summary.csv').exists()
    legacy_rows=0
    comparison=None
    if old and is_v3:
        comparison=compare_v3(run,old,save_comparison)
    elif old:
        original=pd.read_csv(old/'trial_stage.csv')
        a=original[original.scenario=='main']
        b=t[(t.scenario=='legacy123')&(t.analysis_branch=='decoded_expectation')]
        m=a.merge(b,on=['content_id','scope','stage','source'],suffixes=('_old','_new'))
        cols=['valid_n','valid_s','L_T','L_P','L_O','L_N','C','D','direct_rate','via_rate','dwell_T_ms']
        assert len(m)==len(a)==len(b)
        for c in cols:np.testing.assert_allclose(m[c+'_old'],m[c+'_new'],rtol=0,atol=1e-10,equal_nan=True)
        legacy_rows=len(m)
    return dict(independent_probability_trial_checks=checked,raw_sample_frame_regions_checked=raw_checked,
                common_support_across_24_rows=True,
                probability_output_has_no_decoded_metrics=True,drop_ring_contributes_zero=True,
                saved_probabilities_sum_to_one=True,MC_probabilities_are_64_sample_counts=True,
                legacy_expectation_rows_reproduced=legacy_rows,
                raw_npy_coordinate_only_valid_frames_checked=raw_coordinate_checked,
                native_frame_gaze_trial_rows_checked=gaze_rows_checked,
                all_real_and_paired_masks_reconstructed_from_coordinates=is_v3,
                v2_v3_comparison=comparison)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--old',type=Path);p.add_argument('--save',action='store_true')
    args=p.parse_args();result=validate_run(args.run,args.old,save_comparison=args.save)
    print(json.dumps(result,indent=2))
    if args.save:(args.run/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
