"""Paired model/GT gaze-PSM analysis using the project's exact legacy kernels.

Cached inputs only. No video decoder, score data, or training operations are used.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).parent/'mpl_cache'))
import sys
sys.dont_write_bytecode=True
import ast, argparse, csv, hashlib, importlib.util, json, shutil
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
from scipy.ndimage import uniform_filter1d

PROJECT=Path(r'D:\GitHub_Project\co-tracker')
PRED=Path(r'C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch\logs')
TASKS={'A':['User1_1_1','User7_1_1','User9_7_1','User11_6_1','User12_3_1'],
       'C':['User1_5_2','User5_3_2','User6_1_2','User7_4_1','User12_5_2']}
SOURCE_NAMES=['gt','all_ABCD','single_task','leave_out','spd_finetune','spd_zero_shot']
FIELDS=('task','side','signal','stage','mode')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def load_legacy():
    core=module('legacy_core',PROJECT/'tools/gaze_psm_core.py')
    group=module('legacy_group',PROJECT/'tools/gaze_psm_group.py')
    # Compile unchanged numerical functions; omit video-decoder imports because
    # only cached arrays are used. Full original source is retained in snapshot.
    env=dict(np=np,json=json,csv=csv,Path=Path,uniform_filter1d=uniform_filter1d,
             analyze_curve=core.analyze_curve,EMPTY_STAGES={'','none','nan','null'})
    for filename,names in [
        ('gaze_psm_pipeline_dlc.py',{'interpolate_valid','smooth_valid','curve_inputs','calculate','json_ready','save_json'}),
        ('gaze_psm_report.py',{'_value','_write','_long_rows','_summary','generate_outputs'}),
        ('gaze_psm_xcorr_max_dlc.py',{'select_sides'})]:
        tree=ast.parse((PROJECT/'tools'/filename).read_text(encoding='utf-8'))
        nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
        if filename=='gaze_psm_report.py':
            nodes=[node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'META','LAG_FIELDS'} for t in node.targets)]+nodes
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PROJECT/'tools'/filename),'exec'),env)
    from collections import defaultdict
    env.update(defaultdict=defaultdict,SELECTION_FIELDS=('content_id','user','task','signal','stage','mode'),STAGE_MODES={'psm_stage','within_stage'})
    return SimpleNamespace(**env),group

LEG,GROUP=load_legacy()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def path_for(source,task,cid):
    stem='novice_active_temporal_film_60epochs_20261027/models/'
    if source=='all_ABCD': rel=stem+'all_ABCD/seed_43'
    elif source=='single_task': rel=stem+f'single_{task}/seed_'+('43' if task=='A' else '44')
    elif source=='leave_out': rel=stem+f'leave_out_{task}/seed_'+('44' if task=='A' else '42')
    elif source=='spd_finetune':rel='dvss_finetune_spd_full_20261029/seed_42'
    else:rel='spd_fls1_pretrain/full_stride30_val10_100epochs_3seeds/seed_43/dvss_zero_shot'
    return PRED/rel/'inference_test_k64'/f'{cid}_predictions.npz'

def curve_key(curve):return (curve['content_id'],*(curve[k] for k in FIELDS))
def group_key(curve):return tuple(curve[k] for k in FIELDS)

def prepare(output,source_dir,config):
    quality=[];manifest=[];stage_inventory=[];cache_checks=[]
    for task,cids in TASKS.items():
        for cid in cids:
            cache=source_dir/'cache'/f'{cid}.npz'
            with np.load(cache,allow_pickle=False) as z:base={k:z[k] for k in z.files}
            qc=json.loads(str(base['qc_json']));means={};raws={};pred_meta=[]
            assert qc['task']==task and qc['hz']==config['hz']
            for name in SOURCE_NAMES[1:]:
                path=path_for(name,task,cid)
                with np.load(path,allow_pickle=False) as z:
                    frames=z['frame_number'];ts=z['timestamp_ns'];idx=z['frame_index'];prob=z['probability']
                assert np.array_equal(frames,base['frame_number']),f'{name}/{cid}: frame mismatch'
                assert np.array_equal(idx,np.arange(len(idx)))
                times=(ts-ts[0])/1e9
                assert np.array_equal(times,base['native_time_s']),f'{name}/{cid}: time mismatch'
                assert np.isfinite(prob).all() and np.min(prob)>=0
                h,w=prob.shape[-2:];mx=prob.sum(axis=1,dtype=np.float64)@np.linspace(0,1,w)
                my=prob.sum(axis=2,dtype=np.float64)@np.linspace(0,1,h)
                total=prob.sum((-2,-1),dtype=np.float64)
                assert np.max(abs(total-1))<1e-5
                means[name]=np.column_stack([mx,my])/total[:,None]
                raws[name]={side:LEG.interpolate_valid(times,means[name]*base[f'resolution_{side}'],base['time_s'],config['max_gap']) for side in ('l','r')}
                meta=dict(model=name,task=task,content_id=cid,prediction_path=str(path),prediction_sha256=sha(path),prediction_frames=len(idx),frame_match=True,time_match=True)
                manifest.append(meta);pred_meta.append(meta)
                del prob
            common={side:np.isfinite(base[f'gaze_raw_{side}']).all(1)&np.logical_and.reduce([np.isfinite(raws[name][side]).all(1) for name in SOURCE_NAMES[1:]]) for side in ('l','r')}
            for stage in sorted(set(base['stage'])-LEG.EMPTY_STAGES):
                mask=base['stage']==stage
                stage_inventory.append(dict(content_id=cid,user=qc['user'],task=task,stage=stage,occurrences=len(np.unique(base['occurrence'][mask])),grid_samples=int(mask.sum()),duration_s=float(mask.sum()/config['hz']),common_gaze_samples_l=int((mask&common['l']).sum()),common_gaze_samples_r=int((mask&common['r']).sum())))
            cache_checks.append(dict(content_id=cid,task=task,cache_path=str(cache),cache_sha256=sha(cache),raw_gaze_origin='cached high-frequency raw gaze, Status=2',native_frames=len(base['frame_number']),grid_samples=len(base['time_s']),extra_masked_gt_l=int((np.isfinite(base['gaze_raw_l']).all(1)&~common['l']).sum()),extra_masked_gt_r=int((np.isfinite(base['gaze_raw_r']).all(1)&~common['r']).sum())))
            for name in ['gt_legacy',*SOURCE_NAMES]:
                data=dict(base)
                if name!='gt_legacy':
                    for side in ('l','r'):
                        raw=(base[f'gaze_raw_{side}'] if name=='gt' else raws[name][side]).copy()
                        raw[~common[side]]=np.nan
                        data[f'gaze_raw_{side}']=raw
                        data[f'gaze_{side}']=LEG.smooth_valid(raw,config['hz'],config['smooth_ms'])
                    if name!='gt': data['model_native_mean_norm']=means[name]
                newqc={**qc,'source':name,'psm_source':config['psm_source'],'common_support':name!='gt_legacy'}
                for side in ('l','r'):newqc[f'{side}_gaze_valid_fraction']=float(np.isfinite(data[f'gaze_{side}']).all(1).mean())
                data['qc_json']=np.asarray(json.dumps(newqc))
                dest=output/'cache'/name/f'{cid}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,**data)
                quality.append(newqc)
            print(f'Prepared {cid}: 5 models, exact frame/time match',flush=True)
    LEG._write(output/'prediction_manifest.csv',manifest);LEG.save_json(output/'prediction_manifest.json',manifest)
    LEG._write(output/'stage_inventory.csv',stage_inventory)
    LEG._write(output/'alignment_audit.csv',cache_checks)
    LEG.save_json(output/'quality.json',quality)
    return quality,cache_checks

def compute_job(payload):
    root,name,cid,task,config=payload
    with np.load(Path(root)/'cache'/name/f'{cid}.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    curves=LEG.calculate({'content_id':cid,'Task Name':task},data,SimpleNamespace(**config))
    return name,cid,curves

def as_arrays(curve):
    return {k:np.asarray(v,dtype=float) if k in LEG.LAG_FIELDS and isinstance(v,list) else v for k,v in curve.items()}

def feature(c):
    r=np.asarray(c['r'],float);lags=np.asarray(c['lag_s'],float);good=np.isfinite(r)
    z=int(np.argmin(abs(lags)));out=dict(r_zero=float(r[z]),full_lag_support=bool(good.all()),n_supported_lags=int(good.sum()))
    if not good.any():return {**out,'peak_r':np.nan,'peak_lag_s':np.nan,'peak_at_boundary':False,'record_eligible':False,'near_peak_width_0p01_s':np.nan,'peak_gain_over_zero':np.nan}
    i=int(np.nanargmax(r));near=np.flatnonzero(good&(r>=r[i]-.01))
    return dict(out,peak_r=float(r[i]),peak_lag_s=float(lags[i]),peak_at_boundary=i in (0,len(r)-1),record_eligible=bool(good.all() and r[i]>0 and i not in (0,len(r)-1)),near_peak_width_0p01_s=float(lags[near[-1]]-lags[near[0]]),peak_gain_over_zero=float(r[i]-r[z]))

def supported_group(g):return g['status']=='ok' and g['peak_r']>0 and g['n_stable_users']>=3

def paired_analysis(models,config):
    baseline={curve_key(c):c for c in models['gt']['record_curves']}
    gtgroups={group_key(c):c for c in models['gt']['group_curves']}
    records=[];groups=[];bootstrap=[]
    for name in SOURCE_NAMES[1:]:
        mgroups={group_key(c):c for c in models[name]['group_curves']}
        for c in models[name]['record_curves']:
            truth=baseline[curve_key(c)];g=feature(truth);m=feature(c)
            assert np.array_equal(np.asarray(c['lag_s']),np.asarray(truth['lag_s']))
            reasons=[]
            for label,f in [('gt',g),('model',m)]:
                if not f['full_lag_support']:reasons.append(label+'_incomplete_lag_support')
                if not np.isfinite(f['peak_r']) or f['peak_r']<=0:reasons.append(label+'_nonpositive_or_missing_peak')
                if f['peak_at_boundary']:reasons.append(label+'_boundary_peak')
            for label,lookup in [('gt',gtgroups),('model',mgroups)]:
                if not supported_group(lookup[group_key(c)]):reasons.append(label+'_group_not_eligible')
            row={k:c[k] for k in ('content_id','user',*FIELDS)}
            row.update(model=name,**{'gt_'+k:v for k,v in g.items()},**{'model_'+k:v for k,v in m.items()},
                       delta_r_zero=m['r_zero']-g['r_zero'],delta_peak_r=m['peak_r']-g['peak_r'],delta_lag_s=m['peak_lag_s']-g['peak_lag_s'],abs_lag_error_s=abs(m['peak_lag_s']-g['peak_lag_s']),eligible=not reasons,record_pair_eligible=g['record_eligible'] and m['record_eligible'],exclusion_reasons=';'.join(reasons))
            band=1/config['hz']
            direction=lambda lag:'undefined' if not np.isfinite(lag) else 'lead' if lag>band else 'lag' if lag<-band else 'synchronous'
            row.update(gt_direction=direction(g['peak_lag_s']),model_direction=direction(m['peak_lag_s']),direction_agreement=direction(g['peak_lag_s'])==direction(m['peak_lag_s']) if row['eligible'] else None)
            records.append(row)
        gp={ (group_key(c),c['user']):c for c in models['gt']['user_curves']}
        mp={ (group_key(c),c['user']):c for c in models[name]['user_curves']}
        for key in sorted(mgroups):
            people=sorted(u for (k,u) in gp if k==key and (key,u) in mp and np.isfinite(gp[key,u]['r']).all() and np.isfinite(mp[key,u]['r']).all())
            row=dict(zip(FIELDS,key));row.update(model=name,n_shared_users=len(people),shared_users=people,status='insufficient_users')
            if len(people)>=config['min_users']:
                aa=np.array([gp[key,u]['r'] for u in people]);bb=np.array([mp[key,u]['r'] for u in people]);lags=np.array(gp[key,people[0]]['lag_s']);z=int(np.argmin(abs(lags)))
                ar=GROUP.fisher_mean(aa);br=GROUP.fisher_mean(bb);af=feature(dict(r=ar,lag_s=lags));bf=feature(dict(r=br,lag_s=lags))
                seed=int.from_bytes(hashlib.sha256(repr((name,key,config['seed'])).encode()).digest()[:4],'little')
                rng=np.random.default_rng(seed);ii=rng.integers(0,len(people),(config['bootstrap'],len(people)))
                za=np.arctanh(np.clip(aa,-1+1e-7,1-1e-7));zb=np.arctanh(np.clip(bb,-1+1e-7,1-1e-7))
                ac=np.tanh(za[ii].mean(1));bc=np.tanh(zb[ii].mean(1));ia=ac.argmax(1);ib=bc.argmax(1)
                dl=lags[ib]-lags[ia];dr=bc.max(1)-ac.max(1);dz=bc[:,z]-ac[:,z]
                row.update(status='ok' if af['record_eligible'] and bf['record_eligible'] else 'nonpositive_or_boundary_peak',gt_peak_r=af['peak_r'],model_peak_r=bf['peak_r'],gt_peak_lag_s=af['peak_lag_s'],model_peak_lag_s=bf['peak_lag_s'],delta_peak_r=bf['peak_r']-af['peak_r'],delta_lag_s=bf['peak_lag_s']-af['peak_lag_s'],delta_r_zero=bf['r_zero']-af['r_zero'],gt_boundary_frequency=float(np.mean((ia==0)|(ia==len(lags)-1))),model_boundary_frequency=float(np.mean((ib==0)|(ib==len(lags)-1))))
                for key2,vals in [('delta_lag_s',dl),('delta_peak_r',dr),('delta_r_zero',dz)]:row[key2+'_ci_low'],row[key2+'_ci_high']=np.percentile(vals,[2.5,97.5])
                for i in range(config['bootstrap']):bootstrap.append(dict(model=name,**dict(zip(FIELDS,key)),replicate=i,delta_lag_s=dl[i],delta_peak_r=dr[i],delta_r_zero=dz[i]))
            groups.append(row)
    return records,groups,bootstrap

def save_source(root,name,curves,quality,config,bootstrap=None):
    users,groups,contrasts=GROUP.aggregate(curves,bootstrap=config['bootstrap'] if bootstrap is None else bootstrap,seed=config['seed'],min_users=config['min_users'])
    result=dict(record_curves=curves,user_curves=users,group_curves=groups,contrasts=contrasts)
    dest=root/name;dest.mkdir(parents=True,exist_ok=True)
    LEG.save_json(dest/'config.json',dict(config,source=name))
    LEG.save_json(dest/'results.json',result)
    LEG.generate_outputs(dest,curves,users,groups,contrasts,[q for q in quality if q['source']==name],make_plots=False)
    # Keep legacy names, adding zero-lag and explicit support/reliability fields.
    LEG._write(dest/'record_summary.csv',[dict(LEG._summary(c),**feature(c)) for c in curves])
    return result
