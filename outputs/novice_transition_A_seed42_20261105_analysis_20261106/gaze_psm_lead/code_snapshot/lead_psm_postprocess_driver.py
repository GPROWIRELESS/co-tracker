"""Task-A transition-model gaze/PSM analysis, reusing the established numerical core."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys, json, csv, hashlib, shutil, argparse, time, itertools
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_ROOT=Path('D:\\GitHub_Project\\co-tracker\\outputs\\novice_transition_A_seed42_20261105_analysis_20261106')
TOOLS=Path(__file__).resolve().parent
HZ=41.; GAP=.1
VARIANTS={
    'primary':{}, 'position_unsmoothed':{'position_window':1,'signals':['position']},
    'position_smooth5':{'position_window':5,'signals':['position']},
    'trim_edges':{'trim_edges':True}, 'window4':{'max_lag':4.},
    'velocity_box3':{'velocity_window':3,'velocity_filter':'boxcar','signals':['velocity']},
    'velocity_sg7':{'velocity_window':7,'signals':['velocity']},
    'velocity_sg11':{'velocity_window':11,'signals':['velocity']}}
BASE={'hz':HZ,'max_gap':GAP,'max_lag':2.,'min_overlap':2.,'min_rms':1.,
      'position_window':3,'velocity_window':5,'velocity_filter':'savgol','trim_edges':False,
      'signals':['position','velocity']}
KEY=('variant','content_id','user','task','side','signal','stage','mode')
COLORS=['#111827','#2563eb','#f59e0b','#16a34a','#db2777']

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def jsave(p,x): Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf8')
def csvsave(p,rows):
    rows=list(rows); keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:
            d=clean(r); w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in d.items()})
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def seed(*x):return int.from_bytes(hashlib.sha256(repr(x).encode()).digest()[:4],'little')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def label_times(n,t):
    idx=np.searchsorted(n['time'],t,side='right')-1
    idx=np.clip(idx,0,len(n['time'])-1)
    stage=np.asarray(n['stage']).astype('U80')[idx].copy();occ=np.asarray(n['occurrence'],np.int64)[idx].copy()
    bad=(t<n['time'][0])|(t>n['time'][-1])|(t-n['time'][idx]>GAP)
    stage[bad]='none';occ[bad]=-1
    return stage,occ
def sources_for(view,models):return ['gt']+list(models) if view=='anchor' else ['gt',view.removeprefix('target_')]
def model_label(m):
    if m=='gt': return 'GT'
    if m.startswith('current'): return 'Current (0 ms)'
    for h in ('500','1000','1500'):
        if 'future'+h in m:return 'Future '+h+' ms'
    return m

def prepare(n,preds,models,view,config):
    from gaze_psm_preprocess_v2 import interpolate_valid,filter_valid,finite_rows
    grid=np.arange(int(np.floor(float(n['time'][-1])*HZ))+1)/HZ
    used=sources_for(view,models)
    # Every anchor comparison uses the same model-exported frame identities.
    shared=np.array(sorted(set.intersection(*[set(np.asarray(preds[m]['native_index']).tolist()) for m in models])),np.int64)
    if len(shared)<2:raise ValueError('Fewer than two common native anchor identities')
    lookup={m:{int(k):i for i,k in enumerate(preds[m]['native_index'])} for m in models}
    if view=='anchor':
        t=n['time'][shared]; vals={'gt':n['gaze'][shared].copy()}
        masks=[]
        for m in models:
            ix=np.array([lookup[m][int(k)] for k in shared]);p=preds[m]
            if not np.allclose(p['time'][ix],t,atol=1e-8,rtol=0):raise ValueError('Prediction anchor time mismatch')
            vals[m]=np.asarray(p['expectedxy'][ix],float).copy(); masks.append(np.asarray(p['valid_maps'][ix],bool))
        common=np.logical_and.reduce([finite_rows(v) for v in vals.values()]+masks)
        horizon=0
    else:
        m=used[1];p=preds[m];ix=np.array([lookup[m][int(k)] for k in shared]);t=np.asarray(p['target_time'][ix],float)
        gt_at_target=n['gaze'][shared].copy() if float(models[m])==0 else interpolate_valid(n['time'],n['gaze'],t,GAP)
        vals={'gt':gt_at_target,m:np.asarray(p['expectedxy'][ix],float).copy()}
        common=np.logical_and.reduce([finite_rows(v) for v in vals.values()]+[np.asarray(p['valid_maps'][ix],bool),np.asarray(p['target_time_valid'][ix],bool)])
        horizon=float(models[m])
    assert all(v.shape==(len(t),2) for v in vals.values())
    for v in vals.values():v[~common]=np.nan
    raw={s:interpolate_valid(t,v,grid,GAP) for s,v in vals.items()}
    mask=np.logical_and.reduce([finite_rows(a) for a in raw.values()])
    for v in raw.values():v[~mask]=np.nan
    psm={side:interpolate_valid(n['time'],n[field],grid,GAP) for side,field in [('l','left'),('r','right')]}
    signals={};qc=[]
    stage,occ=label_times(n,grid)
    for signal in config['signals']:
        win=config['position_window'] if signal=='position' else config['velocity_window']
        method='boxcar' if signal=='position' else config['velocity_filter']
        gazes={s:filter_valid(v,win,method,config['trim_edges'])[0] for s,v in raw.items()}
        ps={s:filter_valid(v,win,method,config['trim_edges'])[0] for s,v in psm.items()}
        times=grid;ss=stage;oo=occ
        if signal=='velocity':
            gazes={s:np.diff(v,axis=0)*HZ for s,v in gazes.items()};ps={s:np.diff(v,axis=0)*HZ for s,v in ps.items()}
            times=(grid[1:]+grid[:-1])/2;ss,oo=label_times(n,times)
        mask2=np.logical_and.reduce([finite_rows(v) for v in gazes.values()])
        for v in gazes.values():v[~mask2]=np.nan
        assert all(np.array_equal(finite_rows(v),mask2) for v in gazes.values())
        signals[signal]={'gazes':gazes,'psm':ps,'stage':ss,'occurrence':oo,'time_s':times}
        for s,v in gazes.items():qc.append({'source':s,'signal':signal,'n_grid':len(v),'n_valid':int(finite_rows(v).sum()),'common_support_exact':True,'filter':method,'window':win})
    info={'view':view,'horizon_ms':horizon,'shared_anchor_count':len(shared),'usable_common_anchors':int(common.sum()),
          'anchor_gap_median_s':float(np.median(np.diff(t))),'anchor_gaps_over_0p1_s':int(np.sum(np.diff(t)>GAP)),
          'native_gaze_valid_count':int(np.isfinite(n['gaze']).all(1).sum()),'n_grid':len(grid),'grid_shared_valid_count':int(mask.sum()),
          'target_gt_support':'same target_time with <=0.1s finite interpolation' if view!='anchor' else 'same exported native anchors',
          'interpolation_no_extrapolation':True,'psm_quality_filter_applied':False}
    return signals,info,qc

def analyze_job(job):
    from gaze_psm_reliability_v2 import occurrence_statistics,aggregate_statistics,reliability
    rec,n,preds,models,view,vname,out=job
    config=BASE|VARIANTS[vname]; variant=view+'__'+vname
    d,info,qc=prepare(n,preds,models,view,config)
    cid=rec['content_id'];user=rec['participant'];curves=[];rows=[];arrays={};boot={};tokens=[];counter=0
    for signal,s in d.items():
        phases=sorted(x for x in np.unique(s['stage']) if str(x).lower() not in ('','none','nan','null','drop ring'))
        for side in ['l','r']:
            for stage in ['whole']+phases:
                # Whole task excludes drop-ring PSM samples; stage main keeps legacy PSM-time anchoring.
                valid_stage=~np.isin(s['stage'],['drop ring','none','','nan','null'])
                ids=np.where(valid_stage,0,-1) if stage=='whole' else np.where(s['stage']==stage,s['occurrence'],-1)
                modes=['psm_stage'] if stage=='whole' else ['psm_stage','within_stage']
                for mode in modes:
                    for source in sources_for(view,models):
                        meta={'variant':variant,'view':view,'preprocessing':vname,'content_id':cid,'user':user,'task':'A','side':side,'signal':signal,'stage':str(stage),'mode':mode,'source':source}
                        stats=occurrence_statistics(s['gazes'][source],s['psm'][side],ids,HZ,config['max_lag'],mode=='within_stage')
                        minimum=5. if stage=='whole' else config['min_overlap']
                        curve=aggregate_statistics(stats,min_overlap=minimum,min_rms=config['min_rms'])
                        rel=reliability(stats,bootstrap=500 if vname=='primary' else 200,seed=seed(cid,variant,signal,side,stage,mode),block_length=2,precision_s=.4,min_overlap=minimum,min_rms=1.)
                        token=f'c{counter:04d}';counter+=1
                        row=meta|rel['summary']|{'statistics_key':token,'min_overlap_s':minimum}
                        if stage=='whole':row['whole_reliability_note']='one whole-task occurrence; within-record occurrence uncertainty unavailable; participant group range reported separately'
                        rows.append(row);curves.append(meta|curve);tokens.append(meta|{'statistics_key':token})
                        arrays[token+'_terms']=stats['terms'];arrays[token+'_ids']=stats['ids'];arrays[token+'_lag_s']=stats['lag_s'];boot[token]=rel['peak_samples']
    folder=Path(out)/'statistics'/variant;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/(cid+'.npz'),**arrays)
    bfolder=Path(out)/'bootstrap_peaks'/variant;bfolder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(bfolder/(cid+'.npz'),**boot)
    return curves,rows,[dict(r,variant=variant,content_id=cid) for r in qc],info|{'variant':variant,'content_id':cid},tokens

def pair_curves(curves,rows):
    features={tuple(r[k] for k in KEY)+(r['source'],):r for r in rows}
    grouped=defaultdict(dict)
    for c in curves:grouped[tuple(c[k] for k in KEY)][c['source']]=c
    result=[]
    for key,cc in grouped.items():
        gt=cc['gt'];gf=features[key+('gt',)];common=all(np.isfinite(c['r']).all() for c in cc.values())
        for model,c in cc.items():
            if model=='gt':continue
            f=features[key+(model,)];g=np.isfinite(gt['r'])&np.isfinite(c['r']);ok=bool(g.all())
            a,b=gf['peak_lag_s'],f['peak_lag_s'];dl=b-a if a is not None and b is not None else None
            r=dict(zip(KEY,key))|{'model':model,'view':c['view'],'preprocessing':c['preprocessing'],'input_eligible':ok,'common_input':bool(common),
                'n_common_lags':int(g.sum()),'n_required_lags':len(g),'curve_mae':float(np.mean(np.abs(c['r']-gt['r']))) if ok else None,
                'delta_lag_s':dl,'abs_lag_difference_s':abs(dl) if dl is not None else None,'gt_peak_lag_s':a,'model_peak_lag_s':b,
                'gt_peak_r':gf['peak_r'],'model_peak_r':f['peak_r'],'delta_peak_r':f['peak_r']-gf['peak_r'] if dl is not None else None,
                'gt_localizable':gf['lag_identifiable'],'model_localizable':f['lag_identifiable'],'gt_status':gf['lag_status'],'model_status':f['lag_status'],
                'gt_boundary':gf['peak_at_boundary'],'model_boundary':f['peak_at_boundary'],'gt_direction':gf['direction_0p10'],'model_direction':f['direction_0p10']}
            for band in ['0p05','0p10','0p20']:
                r['gt_direction_'+band]=gf['direction_'+band];r['model_direction_'+band]=f['direction_'+band]
                r['point_direction_agreement_'+band]=gf['point_direction_'+band]==f['point_direction_'+band] if dl is not None else None
            result.append(r)
    return result

def shared_group_curves(curves):
    """Use identical complete-lag recordings across GT/models within each view."""
    support=defaultdict(list)
    for c in curves:support[tuple(c[k] for k in KEY)].append(bool(np.isfinite(c['r']).all()))
    return [c if all(support[tuple(c[k] for k in KEY)]) else dict(c,r=np.full_like(c['r'],np.nan)) for c in curves]

def annotate_groups(summary,long):
    from gaze_psm_reliability_v2 import peak_features
    data=defaultdict(list)
    kfields=('variant','task','side','signal','stage','mode','source')
    for r in long:data[tuple(r[k] for k in kfields)].append(r)
    for r in summary:
        rr=sorted(data[tuple(r[k] for k in kfields)],key=lambda x:x['lag_s'])
        feat=peak_features({'lag_s':np.array([x['lag_s'] for x in rr]),'r':np.array([x['r'] for x in rr])})
        r.update({k:v for k,v in feat.items() if k not in r});reason=[]
        if r['n_stable_users']<3:reason.append('fewer_than_3_users')
        if not feat['full_support']:reason.append('incomplete_lag_support')
        if feat['peak_at_boundary']:reason.append('boundary_peak')
        if feat['near_peak_contiguous_width_0p01_s'] is not None and feat['near_peak_contiguous_width_0p01_s']>.4:reason.append('broad_near_peak')
        if feat['near_peak_components_0p01']>1:reason.append('competing_near_peaks')
        lo,hi=r['peak_lag_resample_low'],r['peak_lag_resample_high']
        if not np.isfinite([lo,hi]).all():reason.append('missing_resampling_range')
        elif hi-lo>.4:reason.append('broad_resampling_range')
        if r['bootstrap_boundary_frequency']>.05:reason.append('bootstrap_boundary_instability')
        r['group_localizable']=not reason;r['group_reliability_reasons']=';'.join(reason)
        for delta,tag in [(.05,'0p05'),(.1,'0p10'),(.2,'0p20')]:
            direction='undetermined'
            if not reason and feat['positive_peak']:
                direction='lead' if lo>delta else 'lag' if hi < -delta else 'synchronous' if lo>=-delta and hi<=delta else 'undetermined'
            r['direction_'+tag]=direction
        r['direction_scope']='exploratory participant-bootstrap plus peak-shape diagnostic; not calibrated biological inference'

def direct_delay(native,predictions,models,records,out):
    from gaze_psm_reliability_v2 import occurrence_statistics,aggregate_statistics,peak_features
    rows=[]
    for rec in records:
        cid=rec['content_id'];s,_,_=prepare(native[cid],predictions[cid],models,'anchor',BASE);d=s['position']
        for model in models:
            for seg in ['whole','first_half','second_half']:
                ix=np.arange(len(d['time_s']));mask=np.ones(len(ix),bool)
                if seg=='first_half':mask=ix<len(ix)//2
                elif seg=='second_half':mask=ix>=len(ix)//2
                ids=np.where(mask,0,-1)
                st=occurrence_statistics(d['gazes']['gt'],d['gazes'][model],ids,HZ,2.,True)
                c=aggregate_statistics(st,min_overlap=2.,min_rms=1.)
                rows.append({'content_id':cid,'user':rec['participant'],'model':model,'segment':seg,'positive_lag_means':'model gaze follows GT gaze','calibration_applied':False}|peak_features(c))
    csvsave(out/'direct_model_gt_delay.csv',rows)

def references(native,predictions,models,records,group_summary,out,reps=5000,common_curves=None):
    from gaze_psm_reliability_v2 import occurrence_statistics,aggregate_statistics
    # Different-user elapsed-time reference is descriptive, and cannot test causal leading.
    prepared={r['content_id']:prepare(native[r['content_id']],predictions[r['content_id']],models,'anchor',BASE)[0]['position'] for r in records}
    users={r['content_id']:r['participant'] for r in records};cids=list(prepared);edge={};rows=[]
    observed=[r for r in group_summary if r['variant']=='anchor__primary' and r['signal']=='position' and r['mode']=='psm_stage']
    for side,stage,source in sorted(set((r['side'],r['stage'],r['source']) for r in observed)):
        eligible=[r for r in observed if (r['side'],r['stage'],r['source'])==(side,stage,source)][0]
        people=json.loads(eligible['participants_json']);recipients=[cid for cid in cids if users[cid] in people]
        if common_curves is not None:
            exact={c['content_id'] for c in common_curves if c['variant']=='anchor__primary' and c['signal']=='position' and c['mode']=='psm_stage' and c['side']==side and c['stage']==stage and c['source']==source and np.isfinite(c['r']).all()}
            recipients=[cid for cid in recipients if cid in exact]
        if len(people)<3 or len(recipients)<3 or not np.isfinite(eligible['peak_r']):
            rows.append({'variant':'anchor__primary','side':side,'stage':stage,'source':source,'signal':'position','mode':'psm_stage',
                'planned_draws':reps,'valid_draws':0,'invalid_support_draws':0,'not_attempted_draws':reps,'n_users':len(people),
                'n_observed_recordings':len(recipients),'observed_peak_r':eligible['peak_r'],'status':'insufficient_observed_group_support',
                'reference_mean_peak_r':None,'corrected_exceedance_fraction':None,'formal_leading_p_value':False})
            continue
        for recipient in recipients:
            d=prepared[recipient];valid=~np.isin(d['stage'],['drop ring','none','','nan','null'])
            ids=np.where(valid,0,-1) if stage=='whole' else np.where(d['stage']==stage,d['occurrence'],-1)
            for donor in cids:
                if users[donor]==users[recipient]:continue
                g=np.full_like(d['psm'][side],np.nan);src=prepared[donor]['gazes'][source];length=min(len(src),len(g));g[:length]=src[:length]
                st=occurrence_statistics(g,d['psm'][side],ids,HZ,2.,False)
                c=aggregate_statistics(st,min_overlap=5. if stage=='whole' else 2.,min_rms=1.)
                edge[recipient,donor]=c['r']
        rng=np.random.default_rng(seed('reference',side,stage));values=[];lags=[];invalid=0
        for _ in range(reps):
            donor_user_map={person:rng.choice(sorted({users[d] for d in cids if users[d]!=person})) for person in people}
            donor_people=[donor_user_map[users[r]] for r in recipients]
            donors=[rng.choice([d for d in cids if users[d]==person]) for person in donor_people]
            a=np.array([edge[r,d] for r,d in zip(recipients,donors)])
            if not np.isfinite(a).all():invalid+=1;continue
            z=np.arctanh(np.clip(a,-1+1e-7,1-1e-7));u=[]
            for person in people:u.append(z[[users[r]==person for r in recipients]].mean(0))
            curve=np.tanh(np.mean(u,axis=0));pi=int(curve.argmax());values.append(float(curve[pi]));lags.append((pi-82)/HZ)
        obs=eligible['peak_r'];r={'variant':'anchor__primary','side':side,'stage':stage,'source':source,'signal':'position','mode':'psm_stage',
            'planned_draws':reps,'valid_draws':len(values),'invalid_support_draws':invalid,'n_users':len(people),'n_observed_recordings':len(recipients),'observed_peak_r':obs,
            'not_attempted_draws':0,'status':'descriptive_reference' if values else 'no_valid_complete_support_draws',
            'reference_definition':'different-user elapsed-time gaze; recipient-user block shares donor user, selected uniformly; donor recording uniform; no replacement of invalid draws',
            'formal_leading_p_value':False,'reference_mean_peak_r':float(np.mean(values)) if values else None,
            'corrected_exceedance_fraction':float((1+np.sum(np.array(values)>=obs))/(len(values)+1)) if values else None}
        rows.append(r)
    csvsave(out/'unpaired_reference.csv',rows)
    return rows

def tests(out):
    from gaze_psm_reliability_v2 import occurrence_statistics,aggregate_statistics,peak_features
    from gaze_psm_preprocess_v2 import interpolate_valid
    rng=np.random.default_rng(174);g=rng.normal(size=(2000,2))*20;p=np.full_like(g,np.nan);delay=12;p[delay:]=g[:-delay]
    c=aggregate_statistics(occurrence_statistics(g,p,np.zeros(len(g),np.int64),HZ,2.,False),min_overlap=2.,min_rms=1.)
    lag=peak_features(c)['peak_lag_s'];assert abs(lag-delay/HZ)<1e-12
    t=np.array([0.,.05,.5,.55]);xy=np.stack([t,t],1);a=interpolate_valid(t,xy,np.array([.025,.25,.525,-.01,.6]),GAP)
    assert np.isfinite(a[[0,2]]).all() and not np.isfinite(a[[1,3,4]]).any()
    jsave(out/'independent_tests.json',{'lag_sign_test':True,'known_delay_s':delay/HZ,'recovered_delay_s':lag,'large_gap_not_filled':True,'no_extrapolation':True})

def figures(groups,long,paired,out,models):
    figdir=out.parent/'figures';figdir.mkdir(exist_ok=True)
    observed={r['stage'] for r in groups if r['stage']!='whole'}
    stages=['whole']+[s for s in ['reach ring','grab ring','transfer ring','place ring'] if s in observed]+sorted(observed-{'reach ring','grab ring','transfer ring','place ring'})
    sources=['gt']+list(models);paths=[]
    for signal in ['position','velocity']:
        fig,axes=plt.subplots(2,len(stages),figsize=(4*len(stages),7),squeeze=False,sharex=True,sharey=True)
        for row,side in enumerate(['l','r']):
            for col,stage in enumerate(stages):
                ax=axes[row,col]
                for j,source in enumerate(sources):
                    d=[r for r in long if r['variant']=='anchor__primary' and r['signal']==signal and r['mode']=='psm_stage' and r['stage']==stage and r['side']==side and r['source']==source]
                    if not d:continue
                    d=sorted(d,key=lambda r:r['lag_s']);x=np.array([r['lag_s'] for r in d]);y=np.array([r['r'] for r in d])
                    ax.plot(x,y,color=COLORS[j],label=model_label(source),lw=2 if source=='gt' else 1.5)
                    if source=='gt':ax.fill_between(x,[r['r_resample_low'] for r in d],[r['r_resample_high'] for r in d],color=COLORS[j],alpha=.1)
                ax.axvline(0,color='#94a3b8',ls='--',lw=.7);ax.grid(alpha=.15);ax.set_title(stage);ax.set_ylim(-.4,1.)
                if col==0:ax.set_ylabel(('Left' if side=='l' else 'Right')+' PSM correlation')
                if row==1:ax.set_xlabel('Lag (s); + = gaze leads PSM')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,-.015))
        fig.suptitle('Task A: anchor-time '+signal+' correlations (shared sparse anchors)',fontsize=15)
        fig.tight_layout(rect=[0,.035,1,.95]);path=figdir/('lead_anchor_'+signal+'_curves.png');fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig);paths.append(str(path))
    # Paired peak differences on shared participants, showing sparse conditions honestly.
    select=[r for r in paired if r['variant']=='anchor__primary' and r['signal']=='position' and r['mode']=='psm_stage']
    fig,axes=plt.subplots(1,2,figsize=(15,8),sharey=True)
    labels=[model_label(m)+' | '+s for s in stages for m in models]
    for ax,side in zip(axes,['l','r']):
        for y,(stage,m) in enumerate(itertools.product(stages,models)):
            d=[r for r in select if r['side']==side and r['stage']==stage and r['model']==m]
            if not d:continue
            r=d[0];v=r['delta_group_peak_lag_s'];lo=r['delta_group_peak_lag_resample_low'];hi=r['delta_group_peak_lag_resample_high']
            if np.isfinite(v):ax.plot([lo,hi],[y,y],color=COLORS[list(models).index(m)+1],lw=2);ax.scatter([v],[y],color=COLORS[list(models).index(m)+1],s=25)
        ax.axvline(0,color='#64748b',ls='--');ax.grid(axis='x',alpha=.2);ax.set_title(('Left' if side=='l' else 'Right')+' PSM');ax.set_xlabel('Model − GT group peak lag (s)')
        ax.set_yticks(range(len(labels)),labels,fontsize=8)
    fig.suptitle('Paired group peak differences; exploratory participant resampling');fig.tight_layout();path=figdir/'lead_paired_peak_differences.png';fig.savefig(path,dpi=180);plt.close(fig);paths.append(str(path))
    fig,axes=plt.subplots(2,len(models),figsize=(16,7),sharex=True,sharey=True,squeeze=False)
    for col,model in enumerate(models):
        for row,side in enumerate(['l','r']):
            ax=axes[row,col]
            for source in ['gt',model]:
                d=[r for r in long if r['variant']=='target_'+model+'__primary' and r['signal']=='position' and r['stage']=='whole' and r['side']==side and r['source']==source]
                if d:
                    d=sorted(d,key=lambda r:r['lag_s']);ax.plot([r['lag_s'] for r in d],[r['r'] for r in d],label=model_label(source),color='#111827' if source=='gt' else COLORS[col+1])
            ax.set_title(model_label(model));ax.axvline(0,color='#94a3b8',ls='--');ax.grid(alpha=.15);ax.legend(fontsize=8)
            if col==0:ax.set_ylabel(('Left' if side=='l' else 'Right')+' PSM correlation')
            if row==1:ax.set_xlabel('Lag (s); + = gaze leads PSM')
    fig.suptitle('Target-time whole-task position correlations; matched GT per horizon');fig.tight_layout(rect=[0,0,1,.96]);path=figdir/'lead_target_time_whole_curves.png';fig.savefig(path,dpi=180);plt.close(fig);paths.append(str(path))
    return paths

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--skip-references',action='store_true');args=parser.parse_args()
    start=time.time();out=args.root/'gaze_psm_lead';out.mkdir(parents=True,exist_ok=True);snapshot=out/'code_snapshot';snapshot.mkdir(exist_ok=True)
    modules=['gaze_psm_core_legacy.py','gaze_psm_preprocess_v2.py','gaze_psm_reliability_v2.py','gaze_psm_statistics_v2.py']
    for name in modules:
        source_path,dest_path=TOOLS/name,snapshot/name
        if source_path.resolve()!=dest_path.resolve():shutil.copy2(source_path,dest_path)
    if Path(__file__).resolve()!=(snapshot/Path(__file__).name).resolve():shutil.copy2(__file__,snapshot/Path(__file__).name)
    sys.path.insert(0,str(snapshot));tests(out)
    meta=json.loads((args.root/'input_audit/metadata.json').read_text(encoding='utf-8-sig'));models=meta['models'];records=meta['records']
    native={r['content_id']:load(args.root/'input_audit/native'/(r['content_id']+'.npz')) for r in records}
    predictions={r['content_id']:{m:load(args.root/'input_audit/predictions'/m/(r['content_id']+'.npz')) for m in models} for r in records}
    input_hashes={str(p):sha(p) for p in sorted((args.root/'input_audit/native').glob('*.npz'))}
    for r in records:
        n=native[r['content_id']]
        xy=np.asarray(n['gaze_unmasked'],float)
        expected=np.isfinite(xy).all(1)&(xy>=0).all(1)&(xy<=np.array([r['width'],r['height']])).all(1)
        if not np.array_equal(np.isfinite(n['gaze']).all(1),expected):raise ValueError('GT input must use coordinate-only validity: '+r['content_id'])
        np.testing.assert_array_equal(n['gaze'][expected],xy[expected])
    views=['anchor']+['target_'+m for m in models]
    jsave(out/'run_manifest.json',{'models':models,'records':records,'base_config':BASE,'variants':VARIANTS,'views':views,'excluded_stages':['drop ring'],
        'occurrence_bootstrap_primary':500,'occurrence_bootstrap_sensitivity':200,'participant_bootstrap':5000,'reference_draws':5000,
        'positive_lag':'gaze(t) versus PSM(t+lag); positive=gaze leads PSM','target_semantics':'forecast placed at nominal target time; actual runtime latency not measured',
        'gaze_expected_coordinate':'probability expectation provided by shared input builder','common_support':'all model/GT native anchor identities and finite gaze; target pair own contemporaneous GT support',
        'gt_gaze_policy':'native NPY gaze_x/gaze_y; finite normalized coordinates in inclusive [0,1]; valid ignored',
        'gt_valid_column_used_for_selection':False,'native_input_sha256':input_hashes,
        'no_lag_calibration':True,'code_hashes':{p.name:sha(p) for p in snapshot.glob('*.py')}})
    jobs=[(r,native[r['content_id']],predictions[r['content_id']],models,view,v,out) for view in views for v in VARIANTS for r in records]
    curves=[];rows=[];qc=[];audits=[];tokens=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,(cc,rr,qq,aa,tt) in enumerate(pool.map(analyze_job,jobs),1):
            curves+=cc;rows+=rr;qc+=qq;audits.append(aa);tokens+=tt
            if i%5==0:print(f'Analyzed {i}/{len(jobs)} record/view/config jobs, {len(curves)} curves, elapsed {time.time()-start:.1f}s',flush=True)
    csvsave(out/'record_summary.csv',rows);csvsave(out/'preprocessing_qc.csv',qc);csvsave(out/'alignment_support_audit.csv',audits);csvsave(out/'statistics_index.csv',tokens)
    with (out/'record_curves.jsonl').open('w',encoding='utf8') as f:
        for c in curves:f.write(json.dumps(clean(c),ensure_ascii=False)+'\n')
    pairs=pair_curves(curves,rows);csvsave(out/'paired_metrics.csv',pairs)
    from gaze_psm_statistics_v2 import group_curves,paired_group_differences,summarize_pairs
    print('Building participant-level summaries and 5000-draw group curves...',flush=True)
    common_curves=shared_group_curves(curves)
    groups,long=group_curves(common_curves,bootstrap=5000,seed=42,min_users=3);annotate_groups(groups,long)
    pg=paired_group_differences(common_curves,bootstrap=5000,seed=42);summary=summarize_pairs(pairs,bootstrap=5000,seed=42)
    csvsave(out/'group_summary.csv',groups);csvsave(out/'group_curves.csv',long);csvsave(out/'paired_group_summary.csv',pg);csvsave(out/'paired_summary.csv',summary)
    print('Checking leave-one-user-out primary results...',flush=True)
    loo=[]
    for user in sorted({r['participant'] for r in records}):
        cc=[c for c in common_curves if c['variant']=='anchor__primary' and c['user']!=user]
        ss,ll=group_curves(cc,bootstrap=5000,seed=42,min_users=3);annotate_groups(ss,ll)
        loo.extend(dict(r,excluded_user=user) for r in ss)
    csvsave(out/'leave_one_user_group_summary.csv',loo)
    direct_delay(native,predictions,models,records,out)
    print('Computing different-user gaze references...',flush=True)
    refs=[] if args.skip_references else references(native,predictions,models,records,groups,out)
    print('Rendering figures...',flush=True);figs=figures(groups,long,pg,out,models)
    primary=[r for r in groups if r['variant']=='anchor__primary' and r['signal']=='position' and r['mode']=='psm_stage']
    direction_counts={s:dict(Counter(r['direction_0p10'] for r in primary if r['source']==s)) for s in ['gt']+list(models)}
    findings={'n_record_curves':len(curves),'n_pairs':len(pairs),'n_recordings':len(records),'n_participants':len({r['participant'] for r in records}),
        'n_group_rows':len(groups),'direction_counts_primary_position':direction_counts,'primary_position_groups':primary,
        'primary_velocity_groups':[r for r in groups if r['variant']=='anchor__primary' and r['signal']=='velocity' and r['mode']=='psm_stage'],
        'leave_user1_out_groups':[r for r in loo if r['excluded_user']=='User1' and r['mode']=='psm_stage'],
        'primary_position_paired_groups':[r for r in pg if r['variant']=='anchor__primary' and r['signal']=='position' and r['mode']=='psm_stage'],
        'reference_valid_draw_range':[min((r['valid_draws'] for r in refs),default=0),max((r['valid_draws'] for r in refs),default=0)],'figures':figs,'elapsed_seconds':time.time()-start}
    jsave(out/'findings.json',findings)
    assert all(sha(Path(p))==h for p,h in input_hashes.items()),'Inputs changed during analysis'
    jsave(out/'validation.json',{'complete':True,'gt_valid_column_used_for_selection':False,'native_input_hashes_unchanged':True,'shared_support_all_preprocessing_checks':all(r['common_support_exact'] for r in qc),
        'lag_sign_synthetic_test':True,'drop_ring_stage_rows':sum(r['stage']=='drop ring' for r in rows),'input_masks_not_quality_filtered':True,
        'bootstrap_intervals':'exploratory resampling ranges, not calibrated 95% confidence intervals','n_curves':len(curves),'n_pairs':len(pairs),'elapsed_seconds':time.time()-start})
    print(json.dumps(clean({k:findings[k] for k in ['n_record_curves','n_pairs','n_recordings','n_participants','direction_counts_primary_position','reference_valid_draw_range','elapsed_seconds']}),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
