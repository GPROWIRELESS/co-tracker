"""Audit paired input alignment and compute matched-time primary sensitivity.

Original primary preprocessing is performed separately for each domain. The
post-filter finite gaze masks are then intersected: values and filters remain
unchanged, and every shifted comparison uses the same gaze/PSM time pairs.
RMS support thresholds can still differ because gaze values differ.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys
sys.dont_write_bytecode = True
import numpy as np

BASE = Path(r'D:\GitHub_Project\co-tracker\outputs\gaze_psm_revision_20261106')
WORK = Path(__file__).resolve().parent
OUT = WORK / 'output'
KERNELS=WORK/'frozen_kernels'
if not KERNELS.is_dir():KERNELS=BASE/'gt_passive16'/'code_snapshot'
sys.path.insert(0, str(KERNELS))
from gaze_psm_preprocess_revision import prepare_signals
from gaze_psm_reliability_v2 import occurrence_statistics, aggregate_statistics, peak_features

CFG = dict(hz=41., max_gap=.1, max_lag=2., min_overlap=2., min_rms=1.,
           position_window=3, velocity_window=5, velocity_filter='savgol', trim_edges=False)

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def dumpcsv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clean(rows))

def equal(a,b):
    return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()

def explicit_curve(gaze, psm, ids, both):
    """Independent direct-pair oracle: no FFT or frozen statistics functions."""
    rr=[]; nn=[]
    for shift in range(-82,83):
        count=0; cov=vg=vp=0.
        for identity in np.unique(ids[ids>=0]):
            u=np.flatnonzero(ids==identity); t=u-shift
            inside=(t>=0)&(t<len(gaze));u,t=u[inside],t[inside]
            valid=np.isfinite(gaze[t]).all(1)&np.isfinite(psm[u]).all(1)
            if both:valid&=ids[t]==identity
            u,t=u[valid],t[valid]
            if len(u)<2:continue
            g=gaze[t]-gaze[t].mean(0);p=psm[u]-psm[u].mean(0)
            count+=len(u);cov+=float((g*p).sum());vg+=float((g*g).sum());vp+=float((p*p).sum())
        good=count>=82 and vg/count>=1 and vp/count>=1 and np.sqrt(vg*vp)>np.finfo(float).eps
        rr.append(float(np.clip(cov/np.sqrt(vg*vp),-1,1)) if good else np.nan)
        nn.append(count)
    return np.array(rr),np.array(nn)

def zero_spatial(gaze,psm,ids):
    dist=[]
    for identity in np.unique(ids[ids>=0]):
        u=np.flatnonzero(ids==identity)
        good=np.isfinite(gaze[u]).all(1)&np.isfinite(psm[u]).all(1)
        if good.sum()<2:continue
        u=u[good];dist.extend(np.linalg.norm(psm[u]-gaze[u],axis=1))
    return dict(at='zero',lag_s=0.,pairs=len(dist),cross_boundary_pairs=0,
        cross_boundary_fraction=0. if dist else None,
        distance_median_px=float(np.median(dist)) if dist else None,
        **{f'proximity_{tol}px':float(np.mean(np.array(dist)<=tol)) if dist else None for tol in (25,50,100)})

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output
    OUT.mkdir(parents=True,exist_ok=True)
    pdir=BASE/'gt_passive16'; adir=BASE/'gt84'
    with open(pdir/'analysis_record_inventory.csv',encoding='utf-8-sig') as f: inv=list(csv.DictReader(f))
    with open(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json',encoding='utf-8') as f:
        links={k.split('/')[-1]:v for k,v in json.load(f)['clips'].items()}
    active_source=Path(json.load(open(adir/'run_manifest.json'))['source'])
    pcurves={}
    acurves={}
    fields=('task','side','signal','stage','mode')
    for target,d in ((pcurves,pdir),(acurves,adir)):
        with open(d/'record_curves.jsonl',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['variant']=='primary': target[(r['content_id'],)+tuple(r[k] for k in fields)]=r
    audit=[]; summary=[]; spatial=[]; hashes=[]; verify_errors=[]; max_error=0.; pair_count_error=0.
    explicit_checks=0; explicit_max_error=0.; explicit_pair_count_error=0.
    with open(OUT/'common_time_record_curves.jsonl','w',encoding='utf-8') as f:
        for row in inv:
            pid,aid=row['recording_id'],row['source_content_id']
            assert links[pid]['meta']['content_id']==aid
            pp=pdir/'reproduction_inputs/cache'/f'{pid}.npz'
            ap=active_source/'cache'/f'{aid}.npz'
            pc,ac=load(pp),load(ap)
            hashes.extend([dict(domain='passive',id=pid,path=str(pp),sha256=sha(pp)),dict(domain='active',id=aid,path=str(ap),sha256=sha(ap))])
            alignment={k:equal(pc[k],ac[k]) for k in ('time_s','native_time_s','frame_number','stage','occurrence','native_stage','native_occurrence','native_psm_l','native_psm_r','psm_raw_l','psm_raw_r')}
            assert all(alignment.values()),(pid,alignment)
            ps=prepare_signals({'gt':pc},CFG)['signals']; ass=prepare_signals({'gt':ac},CFG)['signals']
            for signal in ('position','velocity'):
                for side in ('l','r'):
                    pd,ad=ps[signal][side],ass[signal][side]
                    assert all(equal(pd[k],ad[k]) for k in ('psm','time_s','stage','occurrence'))
                    pg,ag=pd['gazes']['gt'].copy(),ad['gazes']['gt'].copy()
                    pm=np.isfinite(pg).all(1);am=np.isfinite(ag).all(1);common=pm&am
                    meta=dict(passive_id=pid,active_id=aid,observer_user=row['observer_user'],source_user=row['source_user'],task=row['task'],side=side,signal=signal)
                    ar=dict(meta,grid_points=len(pm),passive_valid_points=int(pm.sum()),active_valid_points=int(am.sum()),common_valid_points=int(common.sum()),passive_valid_fraction=pm.mean(),active_valid_fraction=am.mean(),common_valid_fraction=common.mean(),mask_disagreement_fraction=(pm!=am).mean(),all_alignment_checks_passed=True,**{k+'_equal':v for k,v in alignment.items()})
                    stage_errors=[]
                    stages=sorted(s for s in np.unique(pd['stage']) if str(s).lower() not in ('','none','nan','null'))
                    for stage in stages:
                        ids=np.where(pd['stage']==stage,pd['occurrence'],-1)
                        for mode in ('PSM in stage','Both in stage'):
                            original=[]; results=[]; domain_spatial=[]
                            for domain,gaze,lookup,cid in (('passive',pg,pcurves,pid),('active',ag,acurves,aid)):
                                stat=occurrence_statistics(gaze,pd['psm'],ids,41.,2.,mode=='Both in stage')
                                orig=aggregate_statistics(stat,min_overlap=2.,min_rms=1.)
                                stored=lookup[(cid,row['task'],side,signal,str(stage),mode)]
                                r1,r2=np.array(orig['r'],float),np.array(stored['r'],float)
                                assert np.array_equal(np.isfinite(r1),np.isfinite(r2)),(pid,domain,stage,mode)
                                err=float(np.nanmax(np.abs(r1-r2))) if np.isfinite(r1).any() else 0.
                                max_error=max(max_error,err);stage_errors.append(err)
                                assert err<1e-10,(pid,domain,err)
                                cg=gaze.copy();cg[~common]=np.nan
                                st=occurrence_statistics(cg,pd['psm'],ids,41.,2.,mode=='Both in stage')
                                c=aggregate_statistics(st,min_overlap=2.,min_rms=1.)
                                feat=peak_features(c)
                                m=dict(meta,stage=str(stage),mode=mode,variant='common_time_primary',domain=domain,content_id=cid,source='gt',support_policy='intersection of post-filter finite gaze masks; original domain-specific filtered values retained')
                                if pid in ('User1_4_User11_2_1','User1_6_User9_5_2','User9_2_User8_6_1') and side=='l' and stage==stages[0]:
                                    direct,counts=explicit_curve(cg,pd['psm'],ids,mode=='Both in stage')
                                    assert np.array_equal(np.isfinite(direct),np.isfinite(c['r']))
                                    err=float(np.nanmax(abs(direct-c['r']))) if np.isfinite(direct).any() else 0.
                                    ce=float(np.max(abs(counts-c['pairs'])))
                                    assert err<1e-10 and ce<1e-6,(pid,domain,signal,stage,mode,err,ce)
                                    explicit_max_error=max(explicit_max_error,err);explicit_pair_count_error=max(explicit_pair_count_error,ce);explicit_checks+=1
                                if signal=='position':
                                    domain_spatial.append(dict(m,**zero_spatial(cg,pd['psm'],ids),full_lag_support=feat['full_lag_support']))
                                f.write(json.dumps(clean(dict(m,**c)),ensure_ascii=False,allow_nan=False)+'\n')
                                summary.append(dict(m,**feat))
                                results.append(c)
                            perr=float(np.max(abs(results[0]['pairs']-results[1]['pairs'])))
                            pair_count_error=max(pair_count_error,perr)
                            assert perr<1e-6,(pid,stage,mode,perr)
                            if signal=='position' and all(x['full_lag_support'] for x in domain_spatial):
                                assert domain_spatial[0]['pairs']==domain_spatial[1]['pairs']
                                spatial.extend(dict(x,both_full_lag_support=True) for x in domain_spatial)
                    ar['original_curve_max_abs_error']=max(stage_errors,default=0.)
                    audit.append(ar)
            print('Audited',pid,'->',aid,flush=True)
    dumpcsv(OUT/'support_audit.csv',audit)
    dumpcsv(OUT/'common_time_record_summary.csv',summary)
    dumpcsv(OUT/'common_time_spatial_records.csv',spatial)
    summary_json=dict(passed=True,record_pairs=len(inv),signal_side_audit_rows=len(audit),all_time_stage_frame_psm_equal=True,
        original_primary_reconstruction_max_abs_error=max_error,common_pair_counts_max_abs_error=pair_count_error,
        common_time_summary_rows=len(summary),mask_disagreement_fraction_min=min(x['mask_disagreement_fraction'] for x in audit),
        common_time_spatial_rows=len(spatial),explicit_common_time_curves_checked=explicit_checks,
        explicit_common_time_max_correlation_error=explicit_max_error,explicit_common_time_max_pair_count_error=explicit_pair_count_error,
        mask_disagreement_fraction_max=max(x['mask_disagreement_fraction'] for x in audit),
        common_valid_fraction_min=min(x['common_valid_fraction'] for x in audit),
        common_valid_fraction_max=max(x['common_valid_fraction'] for x in audit),
        common_time_sensitivity_needed=any(x['mask_disagreement_fraction']>.01 for x in audit),
        method='Apply original preprocessing separately; intersect post-filter gaze finite masks; retain original gaze values, original PSM, stages, occurrences and frozen correlation kernel.',
        interpretation='Controls time-pair availability, not domain-specific upstream data quality or historical passive gaze imputation. No recalibration of playback timing.',
        frozen_kernel_directory=str(KERNELS),frozen_kernel_sha256={x.name:sha(x) for x in KERNELS.glob('*.py') if x.name in ('gaze_psm_preprocess_revision.py','gaze_psm_reliability_v2.py','gaze_psm_core_legacy.py')},
        input_caches=hashes,errors=verify_errors)
    (OUT/'support_audit.json').write_text(json.dumps(clean(summary_json),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary_json.items() if k!='input_caches'},indent=2))

if __name__=='__main__':main()
