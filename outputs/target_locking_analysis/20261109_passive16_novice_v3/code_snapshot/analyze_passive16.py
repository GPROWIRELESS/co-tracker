"""Paired novice/novice gaze analysis using unmodified, frozen target-locking v3."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REPO = Path('D:/GitHub_Project/co-tracker')
REFERENCE = REPO/'outputs/target_locking_analysis/20261105T063658759189Z_v3_full'
LINKS = Path('C:/Users/GT502/Desktop/JMRR2027/passive_gaze_links.json')
sys.path[:0] = [str(HERE/'v3'), str(REPO/'tools'), str(REPO)]
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import shutil
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tl_geometry as geo
import tl_inputs as inputs
import analyze_target_locking as v3
from tl_common import stage_masks, summarize


def sha(path):
    return inputs._sha256(path)


def dump(path, obj):
    def default(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding='utf-8')


def csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig', na_rep='NA')


def paired_values(a, p):
    row = {}
    for key in a:
        row['active_'+key] = a[key]
        row['passive_'+key] = p[key]
        if isinstance(a[key], (int, float, np.number)):
            row['delta_'+key] = p[key]-a[key]
    return row


def original_indices(raw, frames):
    # Match only the ACTIVE original row IDs, then use these positions for BOTH domains.
    ids = raw['Frame_Number'].astype(np.int64)
    if np.array_equal(ids, frames): return np.arange(len(raw))
    order = np.argsort(ids)
    pos = np.searchsorted(ids[order], frames)
    assert len(np.unique(ids)) == len(ids) and np.all(pos < len(ids))
    assert np.array_equal(ids[order[pos]], frames)
    return order[pos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=WORK/'output')
    args = ap.parse_args()
    out = args.output
    for folder in ['frames', 'cache', 'figures/examples', 'figure_data', 'code_snapshot/v3', 'code_snapshot/dependencies']:
        (out/folder).mkdir(parents=True, exist_ok=True)
    manifest = json.loads(LINKS.read_text(encoding='utf-8-sig'))
    data_root = Path(manifest['data_root'])
    reference_cfg = json.loads((REFERENCE/'run_config.json').read_text(encoding='utf-8-sig'))
    old_audit = pd.read_csv(REFERENCE/'recording_audit.csv').set_index('content_id')
    old_trials = pd.read_csv(REFERENCE/'trial_stage.csv')
    old_trials = old_trials[(old_trials.source == 'gaze') & (old_trials.scope == 'real_all') &
                            (old_trials.analysis_branch == 'decoded_expectation')].set_index(['content_id','scenario','stage'])
    records = {r['content_id']: r for r in inputs.records(REPO)}
    selected = []
    selection_audit = []
    for key, entry in manifest['clips'].items():
        m = entry['meta']
        novice = m['src_skill'] == m['observer_skill'] == 'novice'
        eligible = novice and m['task_name'] in ['A','C'] and m['content_id'] in reference_cfg['selected']
        selection_audit.append(dict(clip_key=key, content_id=m['content_id'], observer_id=m['observer'],
            task=m['task_name'], src_skill=m['src_skill'], observer_skill=m['observer_skill'], selected=eligible,
            reason='selected' if eligible else ('skill_filter' if not novice else 'outside_v3_A_C_scope')))
        if eligible: selected.append((key, entry))
    assert len(selected) == 16, f'Expected approved 16 records, found {len(selected)}'
    assert Counter(e['meta']['task_name'] for _,e in selected) == {'A':7, 'C':9}
    csv(selection_audit, out/'selection_audit.csv')
    shutil.copy2(LINKS, out/'passive_gaze_links.input.json')
    code_hashes = {}
    for path in (HERE/'v3').glob('*.py'):
        assert sha(path) == sha(REFERENCE/'code_snapshot'/path.name)
        code_hashes[path.name] = sha(path)
        shutil.copy2(path, out/'code_snapshot/v3'/path.name)
    dependencies = ['gaze_psm_pipeline_dlc.py','gaze_psm_core.py','gaze_psm_group.py','gaze_psm_modes.py']
    dependency_hashes = {}
    for name in dependencies:
        path = REPO/'tools'/name
        dependency_hashes[str(path)] = sha(path)
        shutil.copy2(path, out/'code_snapshot/dependencies'/name)
    source_hashes = {str(LINKS):sha(LINKS)}
    for name in ['content_index.csv','Task_A_Score.csv','Task_C_Score.csv']:
        path = REPO/'Index'/name
        source_hashes[str(path)] = sha(path)
        shutil.copy2(path, out/('input_'+name))
    cfg = dict(status='measuring', reference_run=str(REFERENCE), links_path=str(LINKS), data_root=str(data_root),
        started_utc=datetime.now(timezone.utc).isoformat(), scope='novice source AND novice observer; v3 Task A/C',
        pairing_rule='equal original NPY row counts => positional correspondence; no passive timestamp/frame-ID gate',
        filter={'src_skill':'novice','observer_skill':'novice','tasks':['A','C']},
        scenarios={k:list(v) for k,v in v3.SCENARIOS.items()}, stages=v3.STAGES,
        bootstrap_seed=42, bootstrap_n=10000,
        metric_code_sha256=code_hashes, dependencies_sha256=dependency_hashes,
        frame_policy='same active video timeline, geometry, and pairwise common coordinate-valid support',
        gaze_quality_policy='finite normalized xy in inclusive [0,1]; valid code ignored',
        aggregation='pairwise finite metric -> content observer-equal -> active participant content-equal -> participant-equal',
        inference='active-operator bootstrap conditional on the observed passive observer set',
        delta_definition='passive minus active', source_hashes=source_hashes,
        counts=dict(pairs=16, task_A_pairs=7, task_C_pairs=9,
                    contents=len({e['meta']['content_id'] for _,e in selected}),
                    observers=len({e['meta']['observer'] for _,e in selected}),
                    active_participants=len({e['meta']['user'] for _,e in selected})))
    dump(out/'run_config.json', cfg)
    readargs = SimpleNamespace(data_root=data_root, repo=REPO, cache_dir=out/'cache')
    pair_rows, metrics, coverage, joint, reproduction, source_audits = [], [], [], [], [], []
    active_cache = {}
    for number, (clip_key, entry) in enumerate(selected,1):
        m = entry['meta']; cid = m['content_id']; rec = records[cid]
        pair_id = Path(clip_key).name
        print(f'[{number}/16] {pair_id}', flush=True)
        active_path = data_root/'Active_Data'/rec['participant']/'npy'/f'{cid}_gaze.npy'
        passive_path = data_root/entry['gaze_npy_path']
        raw_a = np.load(active_path, mmap_mode='r', allow_pickle=False)
        raw_p = np.load(passive_path, mmap_mode='r', allow_pickle=False)
        assert len(raw_a) == len(raw_p), f'Row count mismatch: {pair_id}'
        assert rec['task'] == m['task_name']
        for key in ['stage_path','target_locking_label_path']:
            assert (data_root/entry[key]).resolve() == active_path.resolve()
        if cid not in active_cache:
            d = v3.prepare(rec, readargs)
            ref = old_audit.loc[cid]
            for key in ['npy_sha256','video_ts_sha256','dlc_sha256']:
                assert d['audit'][key] == ref[key], f'Changed reference input {cid}/{key}'
            for key in ['task','round','status']:
                assert str(rec[key]) == str(ref[key])
            assert np.isclose(float(rec['score']),float(ref['score']), equal_nan=True)
            assert Path(d['audit']['dlc_path']).resolve() == (data_root/entry['psm_dlc_csv_path']).resolve()
            source_audits.append(dict(d['audit']))
            areas = {}
            geometry = d['included'] & (d['count']>0) & d['gripper_ok']
            for scenario,(rt,rp,_,smooth) in v3.SCENARIOS.items():
                if (rt,rp) not in areas:
                    areas[rt,rp] = geo.areas(d['z'], d['left'], d['right'], d['width'], d['height'], rt, rp, geometry)
                xy,valid = v3.support(d,scenario)
                state = geo.classify(xy,d['z'],d['left'],d['right'],rt,rp,valid)
                for stage, st in stage_masks(d):
                    result = summarize(state, valid&st, areas[rt,rp], d)
                    expected = old_trials.loc[(cid,scenario,stage)]
                    for key,value in result.items():
                        if isinstance(value,(int,float,np.number)):
                            previous = float(expected[key])
                            match = bool(np.isclose(value,previous,rtol=1e-10,atol=1e-10,equal_nan=True))
                            reproduction.append(dict(content_id=cid,scenario=scenario,stage=stage,metric=key,
                                reference_value=previous,recomputed_value=value,abs_difference=abs(value-previous),match=match))
                            if not match:
                                raise AssertionError(f'Active reproduction failed: {cid}/{scenario}/{stage}/{key}: {value} != {previous}')
            active_cache[cid] = (d,areas)
        d,areas = active_cache[cid]
        ix = original_indices(raw_a,d['frames'])
        p = raw_p[ix]
        xy_raw = np.column_stack([p['gaze_x'],p['gaze_y']]).astype(float)
        coord_ok = np.isfinite(xy_raw).all(axis=1) & (xy_raw>=0).all(axis=1) & (xy_raw<=1).all(axis=1)
        xy_raw *= [d['width'],d['height']]
        xy = xy_raw.copy(); xy[~coord_ok] = np.nan
        smooth_xy, interp = geo.smooth_gaze(d['timestamp_ns'],xy,d['occurrence'],d['breaks'])
        pdict = dict(d, gaze=xy, gaze_unmasked=xy, smooth_xy=smooth_xy, interpolated=interp)
        meta = dict(pair_id=pair_id, content_id=cid, observer_id=m['observer'], seq=m['seq'],
            src_skill=m['src_skill'], observer_skill=m['observer_skill'], split=m['split'], **{k:rec[k] for k in v3.META if k!='content_id'})
        pair_rows.append(dict(meta,clip_key=clip_key,active_npy_path=str(active_path), passive_npy_path=str(passive_path),
            active_npy_rows=len(raw_a),passive_npy_rows=len(raw_p),rows_equal=True, retained_rows=len(ix),
            active_npy_sha256=sha(active_path),passive_npy_sha256=sha(passive_path),
            source_video_path=d['video'],video_ts_path=d['audit']['video_ts_path'],dlc_path=d['audit']['dlc_path'],
            matching_rule='equal original row counts; identical positional index selection',
            timestamp_equality_required=False,frame_number_equality_required=False))
        araw = np.column_stack([raw_a[ix]['gaze_x'],raw_a[ix]['gaze_y']]).astype(float)*[d['width'],d['height']]
        marks = dict(row_index=ix,frame_number=d['frames'],timestamp_ns=d['timestamp_ns'],time_s=d['time'],dt_s=d['dt'],
            stage=d['stage'],occurrence=d['occurrence'],break_before=d['breaks'],included=d['included'],
            geometry_valid=d['included']&(d['count']>0)&d['gripper_ok'],resolution=np.array([d['width'],d['height']]),
            target_slots=d['slots'],target_xy=d['z'],target_count=d['count'],target_slotmask=d['slotmask'],
            left_xy=d['left'],right_xy=d['right'],likelihood=d['likelihood'],identity_change=d['identity_change'],
            active_xy=d['gaze_unmasked'],passive_xy=xy,active_xy_raw=araw,passive_xy_raw=xy_raw,
            active_smooth_xy=d['smooth_xy'],passive_smooth_xy=smooth_xy,
            active_interpolated=d['interpolated'],passive_interpolated=interp,
            active_valid_code=d['valid_code'],passive_valid_code=p['valid'],
            passive_frame_number=p['Frame_Number'],passive_timestamp_ns=p['Timestamp(ns)'])
        for scenario,(rt,rp,_,smooth) in v3.SCENARIOS.items():
            ax,av = v3.support(d,scenario); px,pv = v3.support(pdict,scenario)
            common = av & pv
            ast = geo.classify(ax,d['z'],d['left'],d['right'],rt,rp,common)
            pst = geo.classify(px,d['z'],d['left'],d['right'],rt,rp,common)
            area = areas[rt,rp]
            marks.update({f'active_valid_{scenario}':av,f'passive_valid_{scenario}':pv,f'valid_{scenario}':common,
                f'active_state_{scenario}':ast,f'passive_state_{scenario}':pst,f'area_{scenario}':area})
            for stage, st in stage_masks(d):
                owner = dict(meta,scenario=scenario,stage=stage)
                mask = common & st
                a = summarize(ast,mask,area,d); pstats = summarize(pst,mask,area,d)
                assert a['valid_n'] == pstats['valid_n'] and a['valid_s'] == pstats['valid_s']
                metrics.append(dict(owner,common_valid_n=int(mask.sum()),common_valid_s=float(d['dt'][mask].sum()),**paired_values(a,pstats)))
                coverage.append(dict(owner,total_n=int(st.sum()),total_s=float(d['dt'][st].sum()),
                    geometry_valid_n=int((st&marks['geometry_valid']).sum()),
                    active_valid_n=int((st&av).sum()),passive_valid_n=int((st&pv).sum()),
                    active_valid_s=float(d['dt'][st&av].sum()),passive_valid_s=float(d['dt'][st&pv].sum()),
                    common_valid_n=int(mask.sum()),common_valid_s=float(d['dt'][mask].sum()),
                    active_excluded_by_pair_n=int((st&av&~pv).sum()),passive_excluded_by_pair_n=int((st&pv&~av).sum()),
                    active_interpolated_n=int((mask&d['interpolated']).sum()) if smooth else 0,
                    passive_interpolated_n=int((mask&interp).sum()) if smooth else 0))
                cells = ast[mask]*4+pst[mask]
                secs = np.bincount(cells,weights=d['dt'][mask],minlength=16).reshape(4,4)
                nums = np.bincount(cells,minlength=16).reshape(4,4)
                total = secs.sum()
                for u,aname in enumerate(['N','T','P','O']):
                    for v,pname in enumerate(['N','T','P','O']):
                        joint.append(dict(owner,active_region=aname,passive_region=pname,seconds=secs[u,v],frames=nums[u,v],
                            joint_share=secs[u,v]/total if total else np.nan,
                            conditional_share=secs[u,v]/secs[u].sum() if secs[u].sum() else np.nan))
        np.savez_compressed(out/'frames'/f'{pair_id}.npz',**marks)
        # Incremental tables make progress reviewable and allow independent checks.
        csv(pair_rows,out/'pair_manifest.csv')
        csv(metrics,out/'pair_metrics.csv')
        csv(coverage,out/'coverage.csv')
        csv(joint,out/'region_joint.csv')
        csv(reproduction,out/'active_reproduction.csv')
        csv(source_audits,out/'recording_audit.csv')
    cfg.update(status='measurements_complete',measurement_completed_utc=datetime.now(timezone.utc).isoformat(),
        active_reproduction=dict(checks=len(reproduction),all_match=all(r['match'] for r in reproduction)),
        counts={**cfg['counts'],'metric_rows':len(metrics),'region_joint_rows':len(joint)})
    dump(out/'run_config.json',cfg)
    shutil.copy2(__file__,out/'code_snapshot'/Path(__file__).name)
    print(json.dumps(cfg['counts']),flush=True)


if __name__ == '__main__':
    main()
