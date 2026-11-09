"""Versioned fixed-instrument gaze/PSM analysis; confidence values are QC only.

All numerical dependencies live beside this file and are copied into each run.
No raw data, predictions or previous results are modified. This is an exploratory
reanalysis, not a prospectively registered confirmation study.
"""
from gaze_psm_modes import mode_rows, normalize_modes, seed_identity
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import itertools
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

SOURCES = ['gt', 'all_ABCD', 'single_task', 'leave_out', 'spd_finetune', 'spd_zero_shot']
TASKS = {'A': ['User1_1_1','User7_1_1','User9_7_1','User11_6_1','User12_3_1'],
         'C': ['User1_5_2','User5_3_2','User6_1_2','User7_4_1','User12_5_2']}
META = ['variant','content_id','user','task','side','signal','stage','mode']
BASE = dict(hz=41., max_gap=.1, max_lag=2., min_overlap=2., min_rms=1.,
            position_window=3, velocity_window=5, velocity_filter='savgol',
            trim_edges=False, dlc_likelihood_filter=None, jump_filter=False,
            coordinate_range_filter=False, direction_bands_s=[.05,.10,.20],
            precision_width_s=.4, occurrence_block_length=2,
            excluded_stages=['drop ring'])
VARIANTS = {
    'primary': {},
    'position_unsmoothed': dict(position_window=1, signals=['position']),
    'position_smooth5': dict(position_window=5, signals=['position']),
    'trim_edges': dict(trim_edges=True),
    'window4': dict(max_lag=4.),
    'velocity_box3': dict(velocity_window=3,velocity_filter='boxcar',signals=['velocity']),
    'velocity_sg7': dict(velocity_window=7,signals=['velocity']),
}

def clean(v):
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v

def save_json(path, obj):
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(path, rows):
    rows=list(rows)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader()
        for row in rows:
            safe={k:clean(v) for k,v in row.items()}
            safe={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in safe.items()}
            writer.writerow(safe)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(repr(seed_identity(parts)).encode()).digest()[:4],'little')

def load_caches(source, cid):
    result={}
    for name in SOURCES:
        with np.load(Path(source)/'cache'/name/(cid+'.npz'),allow_pickle=False) as z:
            result[name]={k:z[k] for k in z.files}
    return result

def audit_inputs(source, output, records, verify_predictions=True):
    files=[]; rows=[]
    for rel in ['config.json','prediction_manifest.csv','validation.json',*[name+'/results.json' for name in SOURCES]]:
        p=Path(source)/rel
        files.append(dict(path=str(p),sha256=sha(p),kind='legacy_reference'))
    for cid in records:
        data=load_caches(source,cid); base=data['gt']
        for name,d in data.items():
            path=Path(source)/'cache'/name/(cid+'.npz')
            assert np.array_equal(d['frame_number'],base['frame_number'])
            assert np.array_equal(d['native_time_s'],base['native_time_s'])
            assert np.array_equal(d['time_s'],base['time_s'])
            files.append(dict(path=str(path),sha256=sha(path),kind='input_cache'))
            rows.append(dict(content_id=cid,source=name,kind='cache',frame_match=True,time_match=True,
                             n_frames=len(d['frame_number']),gaze_raw_is_resampled=True))
    if verify_predictions:
        with (Path(source)/'prediction_manifest.csv').open(encoding='utf-8-sig') as f: preds=list(mode_rows(csv.DictReader(f)))
        for row in preds:
            if row['content_id'] not in records: continue
            path=Path(row['prediction_path']); digest=sha(path)
            assert digest==row['prediction_sha256'], f'Prediction changed: {path}'
            with np.load(path,allow_pickle=False) as z:
                frame=z['frame_number']; ts=z['timestamp_ns']; idx=z['frame_index']
            with np.load(Path(source)/'cache/gt'/(row['content_id']+'.npz')) as z:
                assert np.array_equal(frame,z['frame_number'])
                assert np.array_equal((ts-ts[0])/1e9,z['native_time_s'])
            assert np.array_equal(idx,np.arange(len(idx)))
            files.append(dict(path=str(path),sha256=digest,kind='prediction'))
            rows.append(dict(content_id=row['content_id'],source=row['model'],kind='prediction',
                             frame_match=True,time_match=True,hash_matches_legacy=True,n_frames=len(frame)))
    inference_root=Path(r'C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch\dvss_pytorch')
    evidence={
        'data/temporal.py':'causal window ending at anchor',
        'data/dataset.py':'anchor = final clip index',
        'model/film.py':'target_index is -1',
        'scripts/predictor.py':'saved frame index and timestamp use anchor',
        'utils/spatial.py':'endpoint normalized probability coordinates'}
    provenance=[]
    dest=Path(output)/'inference_code_evidence';dest.mkdir(exist_ok=True)
    for rel,meaning in evidence.items():
        p=inference_root/rel
        if p.exists():
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            provenance.append(dict(path=str(p),sha256=sha(p),interpretation=meaning,
                                   historical_execution_version_proven=False))
    write_csv(Path(output)/'alignment_audit.csv',rows)
    save_json(Path(output)/'input_manifest.json',files)
    save_json(Path(output)/'temporal_semantics_audit.json',dict(evidence=provenance,
        conclusion='Current inspectable implementation predicts final clip frame; no demonstrated center/end mismatch. Historical execution version is not proven by current code. No temporal calibration applied.'))
    return files

def analyze_record(job):
    from gaze_psm_preprocess_v2 import prepare_signals
    from gaze_psm_reliability_v2 import occurrence_statistics,aggregate_statistics,peak_features,reliability
    source,outdir,cid,task,variant,config,reps=job
    caches=load_caches(source,cid); prepared=prepare_signals(caches,config)
    curves=[];rows=[];arrays={};draws={};index=[];counter=0
    for signal in config.get('signals',['position','velocity']):
        for side in ['l','r']:
            d=prepared['signals'][signal][side]
            excluded=set(config.get('excluded_stages',[]))
            stages=sorted(s for s in np.unique(d['stage']) if str(s).lower() not in ('','none','nan','null') and str(s) not in excluded)
            for stage in stages:
                ids=np.where(d['stage']==stage,d['occurrence'],-1)
                for mode in ['PSM in stage','Both in stage']:
                    common_seed=seed_for(42,cid,signal,side,str(stage),mode,variant)
                    for name in SOURCES:
                        meta=dict(variant=variant,content_id=cid,user=cid.split('_')[0],task=task,
                                  side=side,signal=signal,stage=str(stage),mode=mode,source=name)
                        stats=occurrence_statistics(d['gazes'][name],d['psm'],ids,config['hz'],config['max_lag'],mode=='Both in stage')
                        curve=aggregate_statistics(stats,min_overlap=config['min_overlap'],min_rms=config['min_rms'])
                        feat=peak_features(curve)
                        rel=reliability(stats,bootstrap=reps,seed=common_seed,
                            block_length=config['occurrence_block_length'],precision_s=config['precision_width_s'],
                            min_overlap=config['min_overlap'],min_rms=config['min_rms'])
                        row={**meta,**feat,**rel['summary']}
                        token=f'c{counter:04d}';counter+=1
                        arrays[token+'_terms']=stats['terms'];arrays[token+'_ids']=stats['ids'];arrays[token+'_lag_s']=stats['lag_s']
                        draws[token]=rel['peak_samples']
                        row['statistics_key']=token;row['statistics_file']=f'occurrence_statistics/{variant}/{cid}.npz'
                        index.append(dict(meta,statistics_key=token));rows.append(row)
                        curves.append(dict(meta,**curve))
    dest=Path(outdir)/'occurrence_statistics'/variant;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/(cid+'.npz'),**arrays)
    dest=Path(outdir)/'bootstrap_peaks'/variant;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/(cid+'.npz'),**draws)
    if variant=='primary':
        dest=Path(outdir)/'native_qc_flags';dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/(cid+'.npz'),**{side+'_'+k:v for side,d in prepared.get('diagnostics',{}).items() for k,v in d.items()})
    qc=[dict(x,variant=variant,content_id=cid,task=task) for x in prepared.get('qc',[])]
    audit=[dict(x,variant=variant,content_id=cid,task=task) for x in prepared.get('audit',[])]
    return curves,rows,qc,audit

def pair_records(curves,rows):
    lookup={tuple(c[k] for k in META)+(c['source'],):c for c in curves}
    features={tuple(c[k] for k in META)+(c['source'],):c for c in rows}
    keys=sorted(set(k[:-1] for k in lookup)); output=[]
    for key in keys:
        gt=lookup.get(key+('gt',));gf=features.get(key+('gt',))
        if gt is None: continue
        full={name:(key+(name,) in lookup and np.isfinite(lookup[key+(name,)]['r']).all()) for name in SOURCES}
        common=all(full.values())
        for name in SOURCES[1:]:
            model=lookup[key+(name,)]; mf=features[key+(name,)]; good=np.isfinite(gt['r'])&np.isfinite(model['r'])
            both=gf.get('peak_lag_s') is not None and mf.get('peak_lag_s') is not None
            dl=float(mf['peak_lag_s']-gf['peak_lag_s']) if both else None
            row=dict(zip(META,key));row.update(model=name,same_side=True,input_eligible=bool(good.all()),
                common_input=common,n_common_lags=int(good.sum()),n_required_lags=len(good),
                curve_mae=float(np.mean(np.abs(model['r']-gt['r']))) if good.all() else None,
                curve_mae_partial=float(np.mean(np.abs(model['r'][good]-gt['r'][good]))) if good.any() else None,
                delta_lag_s=dl,abs_lag_difference_s=abs(dl) if dl is not None else None,
                gt_peak_lag_s=gf.get('peak_lag_s'),model_peak_lag_s=mf.get('peak_lag_s'),
                gt_peak_r=gf.get('peak_r'),model_peak_r=mf.get('peak_r'),
                delta_peak_r=(mf['peak_r']-gf['peak_r']) if both else None,
                gt_status=gf.get('lag_status'),model_status=mf.get('lag_status'),
                gt_localizable=bool(gf.get('lag_identifiable',False)),model_localizable=bool(mf.get('lag_identifiable',False)),
                gt_boundary=bool(gf.get('peak_at_boundary',False)),model_boundary=bool(mf.get('peak_at_boundary',False)),
                gt_peak_at_boundary=bool(gf.get('peak_at_boundary',False)),model_peak_at_boundary=bool(mf.get('peak_at_boundary',False)),
                comparison='fixed physical instrument; no peak-based exclusion')
            for band in [.05,.1,.2]:
                label=f'{band:.2f}'.replace('.','p')
                direction=lambda x:'undefined' if x is None else 'lead' if x>band else 'lag' if x< -band else 'synchronous'
                a,b=direction(gf.get('peak_lag_s')),direction(mf.get('peak_lag_s'))
                row['point_gt_direction_'+label]=a;row['point_model_direction_'+label]=b
                row['point_direction_agreement_'+label]=(a==b) if both else None
            output.append(row)
    return output

def selected_side_sensitivity(curves,rows):
    features={tuple(r[k] for k in META)+(r['source'],):r for r in rows}
    groups={}
    for c in curves:
        key=tuple(c[k] for k in META if k!='side')
        groups.setdefault(key,{})[(c['source'],c['side'])]=c
    result=[]
    for key,cur in groups.items():
        def winner(source):
            a=[]
            for side in ['l','r']:
                c=cur.get((source,side))
                if c is not None and np.isfinite(c['r']).any(): a.append((float(np.nanmax(c['r'])),side,c))
            return max(a,key=lambda x:(x[0],x[1]=='l')) if a else None
        gt=winner('gt')
        if gt is None: continue
        for name in SOURCES[1:]:
            mw=winner(name)
            for rule,mc in [('gt_selected_side',cur.get((name,gt[1]))),('independent_max_side_legacy',mw[2] if mw else None)]:
                if mc is None: continue
                gc=gt[2]; gf=features[tuple(gc[k] for k in META)+('gt',)];mf=features[tuple(mc[k] for k in META)+(name,)]
                a,b=gf.get('peak_lag_s'),mf.get('peak_lag_s')
                dl=b-a if a is not None and b is not None else None
                result.append(dict(zip([k for k in META if k!='side'],key),model=name,selection_rule=rule,
                    gt_side=gt[1],model_side=mc['side'],same_side=gt[1]==mc['side'],delta_lag_s=dl,
                    abs_lag_difference_s=abs(dl) if dl is not None else None,
                    interpretation='selected statistic difference; GT selection is data-dependent'))
    return result

def direct_delays(source,records,output):
    from gaze_psm_preprocess_v2 import prepare_signals
    from gaze_psm_reliability_v2 import occurrence_statistics,aggregate_statistics,peak_features
    rows=[]; curves=[]
    for cid in records:
        d=prepare_signals(load_caches(source,cid),BASE)['signals']['position']['l']
        for name in SOURCES[1:]:
            for segment,mask in [('whole',np.ones(len(d['time_s']),bool)),
                                 ('first_half',np.arange(len(d['time_s']))<len(d['time_s'])//2),
                                 ('second_half',np.arange(len(d['time_s']))>=len(d['time_s'])//2)]:
                # Within segment avoids using gaze from the other half.
                ids=np.where(mask,0,-1)
                st=occurrence_statistics(d['gazes']['gt'],d['gazes'][name],ids,41,2,True)
                c=aggregate_statistics(st,min_overlap=2,min_rms=1)
                meta=dict(content_id=cid,user=cid.split('_')[0],task='A' if cid in TASKS['A'] else 'C',model=name,segment=segment)
                rows.append(dict(meta,**peak_features(c),positive_lag_means='model gaze follows GT gaze',calibration_applied=False))
                curves.append(dict(meta,**c))
    write_csv(Path(output)/'direct_model_delay.csv',rows)
    with (Path(output)/'direct_model_delay_curves.jsonl').open('w',encoding='utf8') as f:
        for c in curves:f.write(json.dumps(clean(c),ensure_ascii=False)+'\n')
    return rows

def legacy_regression(source,curves,output):
    # Position primary and boxcar3 velocity reproduce the old numerical inputs.
    lookup={(c['content_id'],c['source'],c['side'],c['signal'],c['stage'],c['mode']):c
            for c in curves if (c['variant']=='primary' and c['signal']=='position') or c['variant']=='velocity_box3'}
    checks=[]
    for name in SOURCES:
        path=Path(source)/name/'results.json'
        saved=normalize_modes(json.loads(path.read_text(encoding='utf8')))['record_curves']
        for old in saved:
            side=old.get('selected_side')
            if side not in ('l','r'):continue
            k=(old['content_id'],name,side,old['signal'],old['stage'],old['mode'])
            if k not in lookup:continue
            new=lookup[k];a=np.asarray(old['r'],float);b=np.asarray(new['r'],float)
            support=bool(np.array_equal(np.isfinite(a),np.isfinite(b)))
            finite=np.isfinite(a)&np.isfinite(b)
            diff=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
            checks.append(dict(content_id=k[0],source=name,side=side,signal=k[3],stage=k[4],mode=k[5],support_equal=support,max_abs_difference=diff,passed=support and diff<1e-8))
    write_csv(Path(output)/'legacy_regression.csv',checks)
    return dict(n_curves=len(checks),passed=bool(checks) and all(x['passed'] for x in checks),
                max_abs_difference=max((x['max_abs_difference'] for x in checks),default=None))

def donor_reference(source,records,curves,output,count=1000):
    """Elapsed-time unpaired reference, fixed observed recipients per group.

    This is a descriptive reference, not a universal independence/leading test.
    All observed complete-lag recipients must remain supported in a replicate.
    """
    from gaze_psm_preprocess_v2 import prepare_signals
    from gaze_psm_reliability_v2 import occurrence_statistics,aggregate_statistics
    pre={cid:prepare_signals(load_caches(source,cid),BASE)['signals']['position'] for cid in records}
    observed={(c['content_id'],c['source'],c['side'],c['stage']):c for c in curves
              if c['variant']=='primary' and c['signal']=='position' and c['mode']=='PSM in stage'}
    pair_cache={};pairings=[];group_rows=[];null_draws=[]
    for task in ['A','C']:
        ids=[c for c in TASKS[task] if c in pre]
        if len(ids)<2:continue
        rng=np.random.default_rng(seed_for('unpaired',task,42))
        donors={cid:[x for x in ids if x.split('_')[0]!=cid.split('_')[0]] for cid in ids}
        choices={cid:rng.choice(donors[cid],size=count).tolist() for cid in ids}
        for b in range(count):
            for cid in ids:pairings.append(dict(task=task,replicate=b,target=cid,donor=choices[cid][b]))
        for cid in ids:
            for donor in donors[cid]:
                for side in ['l','r']:
                    d=pre[cid][side];q=pre[donor][side]
                    assert np.allclose(d['time_s'][:min(len(d['time_s']),len(q['time_s']))],q['time_s'][:min(len(d['time_s']),len(q['time_s']))])
                    for name in SOURCES:
                        g=np.full_like(d['psm'],np.nan);n=min(len(g),len(q['gazes'][name]));g[:n]=q['gazes'][name][:n]
                        for stage in sorted(set(d['stage'])-{'','none','nan','null'}-set(BASE.get('excluded_stages',[]))):
                            occ=np.where(d['stage']==stage,d['occurrence'],-1)
                            st=occurrence_statistics(g,d['psm'],occ,41,2,False)
                            pair_cache[(cid,donor,name,side,str(stage))]=aggregate_statistics(st)['r']
        stages=sorted(set(k[3] for k in observed if k[0] in ids))
        for side,stage,name in itertools.product(['l','r'],stages,SOURCES):
            targets=[cid for cid in ids if (cid,name,side,stage) in observed and np.isfinite(observed[cid,name,side,stage]['r']).all()]
            meta=dict(task=task,side=side,stage=stage,source=name,signal='position',mode='PSM in stage',variant='primary',
                      n_target_users=len(targets),target_users=targets,n_requested=count,
                      support_policy='all observed complete-lag recipients retained',reference='different-user elapsed-time gaze')
            if len(targets)<3:
                group_rows.append(dict(meta,n_valid=0,status='insufficient_observed_users'));continue
            obs=np.tanh(np.mean([np.arctanh(np.clip(observed[c,name,side,stage]['r'],-1+1e-7,1-1e-7)) for c in targets],axis=0)).max()
            values=[]
            for b in range(count):
                aa=[pair_cache.get((cid,choices[cid][b],name,side,stage)) for cid in targets]
                if any(a is None or not np.isfinite(a).all() for a in aa):continue
                peak=float(np.tanh(np.mean(np.arctanh(np.clip(aa,-1+1e-7,1-1e-7)),axis=0)).max())
                values.append(peak);null_draws.append(dict(task=task,side=side,stage=stage,source=name,replicate=b,peak_r=peak))
            p=(1+sum(x>=obs for x in values))/(len(values)+1) if values else None
            group_rows.append(dict(meta,n_valid=len(values),observed_peak_r=float(obs),reference_peak_r_95=float(np.percentile(values,95)) if values else None,
                exceedance_fraction_corrected=p,status='descriptive_reference' if values else 'no_complete_support_draws',formal_leading_p_value=False))
        print(f'Unpaired reference completed task {task}',flush=True)
    write_csv(Path(output)/'unpaired_pairings.csv',pairings)
    write_csv(Path(output)/'unpaired_reference.csv',group_rows)
    write_csv(Path(output)/'unpaired_reference_draws.csv',null_draws)
    return group_rows

def sensitivity_summary(rows):
    keys=['content_id','task','side','signal','stage','mode','model']
    primary={tuple(r[k] for k in keys):r for r in rows if r['variant']=='primary'}
    output=[]
    for r in rows:
        if r['variant']=='primary':continue
        p=primary.get(tuple(r[k] for k in keys))
        if p is None:continue
        same=bool(p['common_input'] and r['common_input'])
        output.append({**{k:r[k] for k in keys},'variant':r['variant'],'primary_common_input':p['common_input'],
            'variant_common_input':r['common_input'],'shared_condition_input':same,
            'primary_curve_mae':p['curve_mae'],'variant_curve_mae':r['curve_mae'],
            'delta_curve_mae':r['curve_mae']-p['curve_mae'] if same else None,
            'primary_delta_lag_s':p['delta_lag_s'],'variant_delta_lag_s':r['delta_lag_s'],
            'primary_model_boundary':p['model_boundary'],'variant_model_boundary':r['model_boundary'],
            'note':'window4 curve distances use a different lag domain; compare restricted-window table for same-domain effects' if r['variant']=='window4' else ''})
    return output

def common_condition_summaries(pairs,summary_function,bootstrap):
    """Reweight primary and each branch on exactly the same condition keys.

    This holds condition composition fixed, not necessarily temporal masks.
    The restricted window4 branch holds the lag domain fixed as well.
    """
    keys=['content_id','task','side','signal','stage','mode','model']
    primary={tuple(r[k] for k in keys):r for r in pairs if r['variant']=='primary'}
    result=[]
    for variant in sorted({r['variant'] for r in pairs}-{'primary'}):
        branches=[]
        for row in pairs:
            if row['variant']!=variant:continue
            old=primary.get(tuple(row[k] for k in keys))
            if old is None:continue
            good=bool(row['common_input'] and old['common_input'])
            branches.append(dict(old,variant='primary_matched_to_'+variant,common_input=good))
            branches.append(dict(row,common_input=good))
        for r in summary_function(branches,bootstrap=bootstrap,seed=42):
            result.append(dict(r,sensitivity_branch=variant,support_policy='identical condition keys; temporal masks may differ for edge trimming'))
    return result

def restricted_window_pairs(curves,rows):
    from gaze_psm_reliability_v2 import peak_features
    cc=[];rr=[]
    for c in curves:
        if c['variant']!='window4':continue
        keep=abs(np.asarray(c['lag_s']))<=2+1e-12
        new={k:(v[keep] if isinstance(v,np.ndarray) and v.shape==keep.shape else v) for k,v in c.items()}
        new['variant']='window4_restricted2';cc.append(new)
        # Reliability was calculated over +-4s and cannot be relabeled as +-2s.
        rr.append({**{k:new[k] for k in META},'source':new['source'],**peak_features(new),
                   'lag_status':'restricted_curve_diagnostic_no_resampling','lag_identifiable':False})
    return pair_records(cc,rr)

def direction_sensitivity(pairs):
    """Descriptive participant-equal point classifications; chance is not a test."""
    from collections import defaultdict
    groups=defaultdict(list)
    for r in pairs:
        if r['variant']!='primary' or r['signal']!='position' or r['mode']!='PSM in stage':continue
        for stage in [r['stage'],'__observed_stage_average__']:
            groups[(r['task'],r['side'],stage,r['model'])].append(r)
    result=[]
    for key,rows in sorted(groups.items()):
        good=[r for r in rows if r['common_input'] and r['delta_lag_s'] is not None]
        users=sorted({r['user'] for r in good})
        for delta in [.05,.1,.2]:
            tag=f'{delta:.2f}'.replace('.','p');states=['lead','lag','synchronous']
            peruser=[]
            for user in users:
                pp=[r for r in good if r['user']==user]
                gg=np.array([r['point_gt_direction_'+tag] for r in pp]);mm=np.array([r['point_model_direction_'+tag] for r in pp])
                peruser.append((float(np.mean(gg==mm)),[float(np.mean(gg==s)) for s in states],[float(np.mean(mm==s)) for s in states]))
            agreement=float(np.mean([x[0] for x in peruser])) if peruser else None
            gm=np.mean([x[1] for x in peruser],axis=0) if peruser else np.full(3,np.nan)
            mm=np.mean([x[2] for x in peruser],axis=0) if peruser else np.full(3,np.nan)
            result.append(dict(zip(['task','side','stage','model'],key),band_s=delta,n_users=len(users),n_conditions=len(good),
                participant_equal_point_agreement=agreement,marginal_independence_chance=float(gm@mm),
                gt_marginals=dict(zip(states,gm)),model_marginals=dict(zip(states,mm)),
                interpretation='point classification sensitivity; marginal chance is descriptive, not a permutation p-value; repeats are not independent'))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_xcorr_models_max_dlc'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--variants',nargs='+',default=list(VARIANTS));p.add_argument('--records',nargs='+')
    p.add_argument('--occurrence-bootstrap',type=int,default=500);p.add_argument('--sensitivity-bootstrap',type=int,default=200)
    p.add_argument('--group-bootstrap',type=int,default=1000);p.add_argument('--summary-bootstrap',type=int,default=2000)
    p.add_argument('--null-replicates',type=int,default=1000)
    p.add_argument('--skip-prediction-hash',action='store_true');p.add_argument('--skip-null',action='store_true')
    args=p.parse_args(); out=args.output
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output must be new or empty; never overwrite an analysis run.')
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    snapshot=out/'code_snapshot';snapshot.mkdir()
    for src in Path(__file__).parent.glob('*.py'):shutil.copy2(src,snapshot/src.name)
    # Numerical modules execute from this run's immutable snapshot, not live tools.
    sys.path.insert(0,str(snapshot))
    configs={v:dict(BASE,**VARIANTS[v]) for v in args.variants}
    records=args.records or [c for ids in TASKS.values() for c in ids]
    manifest=dict(version='2.1.0',analysis_type='exploratory reanalysis',excluded_stages=BASE['excluded_stages'],exclusion_basis='user-requested stage exclusion, applied before record curves and donor reference',user_dlc_policy='retain finite low-likelihood and jumping coordinates; mild smoothing only; quality flags are diagnostic',
        source=str(args.source.resolve()),output=str(out.resolve()),parameters=vars(args)|{'source':str(args.source),'output':str(out)},
        variants=configs,records=records,code_sha256={f.name:sha(f) for f in snapshot.glob('*.py')},
        software=dict(python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__,pandas=__import__('pandas').__version__),calibration_applied=False,
        operational_reliability_policy=dict(min_occurrences=4,near_peak_epsilon=.01,max_width_s=.4,
            max_bootstrap_boundary_fraction=.05,min_bootstrap_support_fraction=.95,
            leave_one_and_half_max_change_s=.4,biologically_validated=False),
        unresolved_external_validation=['independent biological timing tolerance','additional participants','historical inference provenance','nominal confidence coverage not established'])
    manifest['parameters']['records']=records
    save_json(out/'run_manifest.json',manifest)
    print('Auditing cached arrays and prediction hashes...',flush=True)
    inputs=audit_inputs(args.source,out,records,not args.skip_prediction_hash)
    jobs=[(str(args.source),str(out),cid,task,v,configs[v],args.occurrence_bootstrap if v=='primary' else args.sensitivity_bootstrap)
          for v in args.variants for task,ids in TASKS.items() for cid in ids if cid in records]
    curves=[];record_rows=[];qc=[];support=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job,result in zip(jobs,pool.map(analyze_record,jobs)):
            cc,rr,qq,aa=result;curves.extend(cc);record_rows.extend(rr);qc.extend(qq);support.extend(aa)
            print(f'Computed {job[4]} / {job[2]}: {len(rr)} curves',flush=True)
    write_csv(out/'record_summary.csv',record_rows);write_csv(out/'condition_inventory.csv',record_rows)
    write_csv(out/'preprocessing_qc.csv',qc);write_csv(out/'support_audit.csv',support)
    with (out/'record_curves.jsonl').open('w',encoding='utf8') as f:
        for c in curves:f.write(json.dumps(clean(c),ensure_ascii=False)+'\n')
    pairs=pair_records(curves,record_rows);write_csv(out/'paired_metrics.csv',pairs)
    write_csv(out/'direction_sensitivity.csv',direction_sensitivity(pairs))
    write_csv(out/'side_selection_sensitivity.csv',selected_side_sensitivity(curves,record_rows))
    write_csv(out/'sensitivity_conditions.csv',sensitivity_summary(pairs))
    print('Computing participant-level summaries and group uncertainty...',flush=True)
    from gaze_psm_statistics_v2 import summarize_pairs,group_curves,exact_model_comparisons,identity_reference
    write_csv(out/'paired_summary.csv',summarize_pairs(pairs,bootstrap=args.summary_bootstrap,seed=42))
    restricted=restricted_window_pairs(curves,record_rows)
    write_csv(out/'window4_restricted2_pairs.csv',restricted)
    write_csv(out/'sensitivity_summary.csv',common_condition_summaries(pairs+restricted,summarize_pairs,args.summary_bootstrap))
    gr,gc=group_curves(curves,bootstrap=args.group_bootstrap,seed=42,min_users=3)
    write_csv(out/'group_summary.csv',gr);write_csv(out/'group_curves.csv',gc)
    from gaze_psm_statistics_v2 import paired_group_differences
    write_csv(out/'paired_group_summary.csv',paired_group_differences(curves,bootstrap=args.group_bootstrap,seed=42))
    write_csv(out/'model_comparisons.csv',exact_model_comparisons(pairs))
    write_csv(out/'identity_reference.csv',identity_reference(curves))
    direct_delays(args.source,records,out)
    regression=legacy_regression(args.source,curves,out)
    if not args.skip_null:donor_reference(args.source,records,curves,out,args.null_replicates)
    # Check caches again, never mutate original input/prediction files.
    changed=[x['path'] for x in inputs if sha(x['path'])!=x['sha256']]
    changed_code=[name for name,digest in manifest['code_sha256'].items() if sha(snapshot/name)!=digest]
    if changed or changed_code:raise RuntimeError(f'Input or frozen code changed during analysis: {changed+changed_code}')
    validation=dict(legacy_regression=regression,n_record_curves=len(curves),n_pairs=len(pairs),
                    n_participants=len(set(c.split('_')[0] for c in records)),input_caches_unchanged=not changed,
                    prediction_hashes_rechecked=not args.skip_prediction_hash,code_snapshot_unchanged=not changed_code,
                    dlc_quality_filter_applied=False,elapsed_seconds=time.time()-start,
                    inference='exploratory; resampling intervals uncalibrated; no causal or confirmatory model ranking')
    save_json(out/'validation.json',validation)
    from report_model_xcorr_dlc_v2 import generate_report
    generate_report(out)
    save_json(out/'artifact_manifest.json',[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=sha(f))
              for f in sorted(out.rglob('*')) if f.is_file() and f.name!='artifact_manifest.json'])
    print(json.dumps(clean(validation),ensure_ascii=False),flush=True)
    if not regression['passed']:raise RuntimeError('Legacy regression failed; inspect before delivery')

if __name__=='__main__':main()
