"""Reproducible fixed-instrument GT84 reanalysis, retaining all finite DLC points."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
import argparse, csv, json, hashlib, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np

BASE=dict(hz=41.,max_gap=.1,max_lag=2.,min_overlap=2.,min_rms=1.,position_window=3,
          velocity_window=5,velocity_filter='savgol',trim_edges=False)
VARIANTS={'primary':{},'position_unsmoothed':dict(position_window=1,signals=['position']),
          'position_smooth5':dict(position_window=5,signals=['position']),
          'velocity_box3':dict(velocity_window=3,velocity_filter='boxcar',signals=['velocity']),
          'velocity_sg7':dict(velocity_window=7,signals=['velocity']),
          'velocity_sg11':dict(velocity_window=11,signals=['velocity']),
          'trim_edges':dict(trim_edges=True),'window4':dict(max_lag=4.)}
FIELDS=('variant','content_id','user','task','side','signal','stage','mode','source')

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def save(path,obj): Path(path).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2),encoding='utf8')
def csvout(path,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            q=clean(r);w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in q.items()})
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def seed(*x):return int.from_bytes(hashlib.sha256(repr(x).encode()).digest()[:4],'little')
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def key(c):return tuple(c[k] for k in FIELDS)

def record_job(job):
    from gaze_psm_preprocess_revision import prepare_signals
    from gaze_psm_reliability_v2 import occurrence_statistics,aggregate_statistics,peak_features,reliability
    cid,task,src,out,variants=job
    cache=load(Path(src)/'cache'/(cid+'.npz'));curves=[];rows=[];qc=[];audits=[];primary=None
    for variant in variants:
        cfg=dict(BASE,**VARIANTS[variant]);p=prepare_signals({'gt':cache},cfg)
        if variant=='primary':primary=p
        qc.extend(dict(x,variant=variant) for x in p['qc']);audits.extend(dict(x,variant=variant) for x in p['audit'])
        arrays={};peaks={};index=[]
        for signal in cfg.get('signals',['position','velocity']):
            for side,d in p['signals'][signal].items():
                stages=sorted(s for s in np.unique(d['stage']) if str(s).lower() not in ('','none','nan','null'))
                for stage in stages:
                    ids=np.where(d['stage']==stage,d['occurrence'],-1)
                    for mode in ('psm_stage','within_stage'):
                        meta=dict(variant=variant,content_id=cid,user=cid.split('_')[0],task=task,side=side,
                                  signal=signal,stage=str(stage),mode=mode,source='gt',analysis='fixed_left' if side=='l' else 'fixed_right')
                        stats=occurrence_statistics(d['gazes']['gt'],d['psm'],ids,cfg['hz'],cfg['max_lag'],mode=='within_stage')
                        c=aggregate_statistics(stats,min_overlap=cfg['min_overlap'],min_rms=cfg['min_rms'])
                        rel=reliability(stats,bootstrap=500 if variant=='primary' else 200,seed=seed(cid,signal,side,str(stage),mode,variant),
                                        block_length=2,precision_s=.4,min_overlap=cfg['min_overlap'],min_rms=cfg['min_rms'])
                        token=f'c{len(index):04d}'
                        index.append(dict(meta,statistics_key=token))
                        arrays[token+'_terms']=stats['terms'];arrays[token+'_ids']=stats['ids'];arrays[token+'_lag_s']=stats['lag_s']
                        peaks[token]=rel['peak_samples']
                        rows.append(dict(meta,**rel['summary'],statistics_key=token))
                        curves.append(dict(meta,**c))
        for folder,data in [('occurrence_statistics',arrays),('bootstrap_peaks',peaks)]:
            dest=Path(out)/folder/variant;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/(cid+'.npz'),**data)
        save(Path(out)/'occurrence_statistics'/variant/(cid+'.json'),index)
        # Native, interpolation, and filter support arrays remain point-addressable.
        if 'provenance' in p:
            flat={}
            def flatten(value,prefix=''):
                for name,a in value.items():
                    full=prefix+name
                    if isinstance(a,dict):flatten(a,full+'__')
                    elif isinstance(a,np.ndarray):flat[full]=a
            flatten(p['provenance'])
            for side,dd in p['diagnostics'].items():
                for name,a in dd.items():flat['diagnostic__'+side+'__'+name]=a
            flat['native_time_s']=cache['native_time_s'];flat['grid_time_s']=cache['time_s'];flat['frame_number']=cache['frame_number']
            dest=Path(out)/'point_provenance'/variant;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/(cid+'.npz'),**flat)
    return curves,rows,qc,audits,primary

def selected_curves(curves):
    groups=defaultdict(list)
    for c in curves:
        if c['variant']=='primary':groups[tuple(c[k] for k in FIELDS if k!='side')].append(c)
    selected=[];details=[]
    for kk,cc in sorted(groups.items()):
        candidates=[c for c in cc if np.isfinite(c['r']).any()]
        win=max(candidates,key=lambda c:(float(np.nanmax(c['r'])),c['side']=='l')) if candidates else None
        row=dict(win or cc[0]);row.update(side='gt_selected_side',physical_side=win['side'] if win else None,analysis='gt_selected_side')
        selected.append(row)
        details.append({**{k:row[k] for k in FIELDS if k!='side'},'selected_side':row['physical_side'],
                        'rule':'GT signed maximum within each recording/stage/signal/mode; tie left',
                        **{c['side']+'_peak_r':float(np.nanmax(c['r'])) if np.isfinite(c['r']).any() else None for c in cc}})
    return selected,details

def augment_group(rows,long):
    from gaze_psm_reliability_v2 import peak_features
    ix=defaultdict(list)
    fields=('variant','task','side','signal','stage','mode','source')
    for r in long:ix[tuple(r[k] for k in fields)].append(r)
    for r in rows:
        c=ix[tuple(r[k] for k in fields)]
        feat=peak_features({'lag_s':np.array([x['lag_s'] for x in c]),'r':np.array([x['r'] for x in c])})
        r.update(feat);lo,hi=r['peak_lag_resample_low'],r['peak_lag_resample_high']
        lim=max(abs(x['lag_s']) for x in c)
        if not np.isfinite(r['peak_lag_s'] if r['peak_lag_s'] is not None else np.nan):status='insufficient_support'
        elif r['peak_at_boundary'] or (np.isfinite(lo) and (lo<=-lim or hi>=lim)):status='window_limited'
        elif np.isfinite(lo) and lo>0:status='positive_peak_resampling_range'
        elif np.isfinite(hi) and hi<0:status='negative_peak_resampling_range'
        else:status='direction_uncertain'
        r.update(lag_interpretation=status,formal_leading_test=False,
                 analysis='fixed_left' if r['side']=='l' else 'fixed_right' if r['side']=='r' else r['side'])

def matched_modes(curves,bootstrap):
    from gaze_psm_statistics_v2 import group_curves
    groups=defaultdict(dict)
    for c in curves:
        if c['variant']=='primary':groups[(c['content_id'],c['side'],c['signal'],c['stage'])][c['mode']]=c
    out=[];coverage=[]
    for k,dd in groups.items():
        good=len(dd)==2 and all(np.isfinite(c['r']).all() for c in dd.values())
        coverage.append(dict(content_id=k[0],side=k[1],signal=k[2],stage=k[3],shared_complete_record=good))
        if good:out.extend(dict(c,variant='common_records_between_modes') for c in dd.values())
    a,b=group_curves(out,bootstrap=bootstrap);augment_group(a,b)
    return a,b,coverage

def spatial_and_boundaries(prepared,records,curves,groups):
    lookup={(r['task'],r['side'],r['signal'],r['stage'],r['mode']):r for r in groups if r['variant']=='primary'}
    rows=[]
    for c in curves:
        if c['variant']!='primary' or c['signal']!='position' or not np.isfinite(c['r']).all():continue
        gg=lookup[(c['task'],c['side'],c['signal'],c['stage'],c['mode'])]
        d=prepared[c['content_id']]['signals']['position'][c['side']]
        g,p=d['gazes']['gt'],d['psm'];ids=np.where(d['stage']==c['stage'],d['occurrence'],-1)
        for label,tau in [('zero',0.),('group_peak',gg['peak_lag_s'])]:
            if tau is None or not np.isfinite(tau):continue
            shift=int(round(tau*41));dist=[];cross=total=0
            for identity in np.unique(ids[ids>=0]):
                u=np.flatnonzero(ids==identity);t=u-shift;inside=(t>=0)&(t<len(g));u,t=u[inside],t[inside]
                good=np.isfinite(p[u]).all(1)&np.isfinite(g[t]).all(1)
                if c['mode']=='within_stage':good &= ids[t]==identity
                if good.sum()<2:continue
                u,t=u[good],t[good];dist.extend(np.linalg.norm(p[u]-g[t],axis=1));total+=len(u);cross+=int((ids[t]!=identity).sum())
            rows.append({**{k:c[k] for k in FIELDS},'at':label,'lag_s':tau,'pairs':total,'cross_boundary_pairs':cross,
                         'cross_boundary_fraction':cross/total if total else None,'distance_median_px':float(np.median(dist)) if dist else None,
                         **{f'proximity_{tol}px':float(np.mean(np.array(dist)<=tol)) if dist else None for tol in (25,50,100)}})
    agg=defaultdict(lambda:defaultdict(list))
    for r in rows:agg[(r['task'],r['side'],r['stage'],r['mode'],r['at'])][r['user']].append(r)
    summary=[]
    for k,people in agg.items():
        rr=dict(zip(('task','side','stage','mode','at'),k));rr['n_users']=len(people)
        for field in ('distance_median_px','proximity_25px','proximity_50px','proximity_100px','cross_boundary_fraction'):
            vals=[np.mean([x[field] for x in xs if x[field] is not None]) for xs in people.values() if any(x[field] is not None for x in xs)]
            rr[field]=float(np.mean(vals)) if vals else None
        summary.append(rr)
    return rows,summary

def regression(source,curves,output):
    lookup={(c['content_id'],c['side'],c['signal'],c['stage'],c['mode']):c for c in curves
            if (c['variant']=='primary' and c['signal']=='position') or c['variant']=='velocity_box3'}
    src=json.loads((Path(source)/'results.json').read_text(encoding='utf8'));checks=[]
    for old in src['record_curves']:
        k=tuple(old[x] for x in ('content_id','side','signal','stage','mode'));new=lookup.get(k)
        if new is None:continue
        a,b=np.asarray(old['r'],float),new['r'];same=np.array_equal(np.isfinite(a),np.isfinite(b));good=np.isfinite(a)&np.isfinite(b)
        err=float(np.max(abs(a[good]-b[good]))) if good.any() else 0.
        checks.append(dict(zip(('content_id','side','signal','stage','mode'),k),same_mask=same,max_abs_error=err,passed=same and err<1e-9))
    csvout(Path(output)/'legacy_curve_regression.csv',checks)
    summary=dict(n=len(checks),passed=bool(checks) and all(x['passed'] for x in checks),maximum_error=max(x['max_abs_error'] for x in checks))
    maxpath=Path(source).parent/'gaze_psm_xcorr_max_dlc'/'results.json'
    if maxpath.exists() and len({c['content_id'] for c in curves})==84:
        from gaze_psm_group import aggregate
        oldmax=json.loads(maxpath.read_text(encoding='utf8'))
        candidates=[dict(c,variant='primary') for c in lookup.values()]
        selected,_=selected_curves(candidates)
        selected=[dict(c,side='max') for c in selected]
        _,regroups,_=aggregate(selected,bootstrap=0,seed=42,min_users=3)
        newgroups={tuple(g[k] for k in ('task','side','signal','stage','mode')):g for g in regroups}
        groupchecks=[]
        for old in oldmax['group_curves']:
            k=tuple(old[x] for x in ('task','side','signal','stage','mode'));new=newgroups[k]
            a,b=np.asarray(old['r'],float),new['r'];good=np.isfinite(a)&np.isfinite(b)
            error=float(np.max(abs(a[good]-b[good]))) if good.any() else 0.
            groupchecks.append(dict(zip(('task','side','signal','stage','mode'),k),max_abs_error=error,
                                    passed=np.array_equal(np.isfinite(a),np.isfinite(b)) and error<1e-9))
        csvout(Path(output)/'legacy_max_group_regression.csv',groupchecks)
        summary.update(legacy_max_groups=len(groupchecks),legacy_max_groups_passed=all(r['passed'] for r in groupchecks))
        summary['passed'] &= summary['legacy_max_groups_passed']
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--null-count',type=int,default=5000);p.add_argument('--records',nargs='+');p.add_argument('--variants',nargs='+',default=list(VARIANTS))
    p.add_argument('--no-report',action='store_true',help='Generate numerical artifacts; report may be built separately from a versioned report script')
    args=p.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refuse to overwrite output')
    out.mkdir(parents=True,exist_ok=True);start=time.time();snap=out/'code_snapshot';snap.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,snap/f.name)
    sys.path.insert(0,str(snap))
    with (args.source/'quality.csv').open(encoding='utf-8-sig') as f:records=list(csv.DictReader(f))
    records=[dict(content_id=r['content_id'],task=r['task'],user=r['user']) for r in records if not args.records or r['content_id'] in args.records]
    manifest=[dict(path=str(args.source/'cache'/(r['content_id']+'.npz')),sha256=sha(args.source/'cache'/(r['content_id']+'.npz'))) for r in records]
    manifest.extend(dict(path=str(args.source/name),sha256=sha(args.source/name)) for name in ('results.json','config.json','quality.csv'))
    oldmax=args.source.parent/'gaze_psm_xcorr_max_dlc'/'results.json'
    if oldmax.exists():manifest.append(dict(path=str(oldmax),sha256=sha(oldmax)))
    save(out/'input_manifest.json',manifest)
    save(out/'run_manifest.json',dict(version='gt84-fixed-v2.1',source=str(args.source),records=records,
         variants={v:dict(BASE,**VARIANTS[v]) for v in args.variants},bootstrap=args.bootstrap,null_attempts=args.null_count,
         dlc_policy='retain all finite coordinates; likelihood/outside/jumps QC only',
         aggregation='complete-lag records; equal records within user; equal users',
         selection='fixed_left/fixed_right primary; GT-selected supplemental per record/stage/signal/mode',
         gaze_origin='cached 41Hz resampled gaze_raw; upstream high-frequency gaze processing unchanged',
         software=dict(python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__),
         code_sha256={f.name:sha(f) for f in snap.glob('*.py')}))
    curves=[];rows=[];qc=[];audit=[];prepared={}
    jobs=[(r['content_id'],r['task'],str(args.source),str(out),args.variants) for r in records]
    with ThreadPoolExecutor(args.workers) as pool:
        for job,result in zip(jobs,pool.map(record_job,jobs)):
            cc,rr,qq,aa,pp=result;curves.extend(cc);rows.extend(rr);qc.extend(qq);audit.extend(aa);prepared[job[0]]=pp
            print(f'Record {job[0]}: {len(cc)} curves',flush=True)
    csvout(out/'record_summary.csv',rows);csvout(out/'preprocessing_qc.csv',qc);csvout(out/'support_audit.csv',audit)
    with (out/'record_curves.jsonl').open('w',encoding='utf8') as f:
        for c in curves:f.write(json.dumps(clean(c),ensure_ascii=False)+'\n')
    from gaze_psm_statistics_v2 import group_curves
    gr,gc=group_curves(curves,bootstrap=args.bootstrap);augment_group(gr,gc)
    csvout(out/'group_summary.csv',gr);csvout(out/'group_curves.csv',gc)
    selected,sd=selected_curves(curves);sg,sc=group_curves(selected,bootstrap=args.bootstrap);augment_group(sg,sc)
    csvout(out/'side_selection.csv',sd);csvout(out/'gt_selected_group_summary.csv',sg);csvout(out/'gt_selected_group_curves.csv',sc)
    from gt_supplement import supplement
    supplement(curves,gr,selected,out,count=args.bootstrap)
    mg,mc,coverage=matched_modes(curves,args.bootstrap)
    csvout(out/'matched_mode_summary.csv',mg);csvout(out/'matched_mode_curves.csv',mc);csvout(out/'matched_mode_coverage.csv',coverage)
    sr,ss=spatial_and_boundaries(prepared,records,curves,gr)
    csvout(out/'spatial_records.csv',sr);csvout(out/'spatial_summary.csv',ss)
    reg=regression(args.source,curves,out)
    print('Observed curves and legacy regression complete',reg,flush=True)
    if args.null_count:
        from gt_reference import run_reference
        run_reference(prepared,records,curves,gr,out,count=args.null_count,workers=args.workers)
    changed=[x['path'] for x in manifest if sha(x['path'])!=x['sha256']]
    val=dict(n_records=len(records),n_users=len({r['user'] for r in records}),n_curves=len(curves),n_groups=len(gr),
             legacy_regression=reg,input_unchanged=not changed,changed_inputs=changed,dlc_quality_deleted=0,
             elapsed_seconds=time.time()-start,all_support_audits_pass=all(x.get('passed',True) for x in audit))
    save(out/'validation.json',val)
    if not args.no_report:
        from report_gt84_revision import generate_report
        generate_report(out)
    print(json.dumps(val),flush=True)
    if changed or not reg['passed']:raise RuntimeError('Validation failed')

if __name__=='__main__':main()
