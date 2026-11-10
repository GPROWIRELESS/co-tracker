"""Read-only GT84 20261106 association analysis; all writes go to --output."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
import scipy

STAGES = {'A': ['reach ring','grab ring','transfer ring','place ring','drop ring'],
          'C': ['reach string','threading string through tower','wrap around psm','pull through and cinch']}
METHODS = ['between_participants','within_participant','within_adjusted_order']
SCENARIOS = {
 'primary': ('position','PSM in stage'),
 'completed_only': ('position','PSM in stage'),
 'both_in_stage': ('position','Both in stage'),
 'velocity': ('velocity','PSM in stage'),
 'exclude_boundary': ('position','PSM in stage'),
 'lag_identifiable_exploratory': ('position','PSM in stage')}
META = ['scenario','task','stage','side','signal','mode','predictor']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def clean_json(obj):
    if isinstance(obj,dict): return {str(k):clean_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean_json(v) for v in obj]
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,(float,np.floating)): return float(obj) if np.isfinite(obj) else None
    return obj

def write_json(path,obj):
    path.write_text(json.dumps(clean_json(obj),ensure_ascii=False,indent=2),encoding='utf-8')

def write_table(path,rows,columns=None):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    df.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.15g')

def corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    x=x-x.mean(axis=-1,keepdims=True); y=y-y.mean(axis=-1,keepdims=True)
    den=np.sqrt((x*x).sum(axis=-1)*(y*y).sum(axis=-1))
    return np.divide((x*y).sum(axis=-1),den,out=np.full(np.shape(den),np.nan),where=den>1e-12)

def spearman(x,y):
    return float(corr(stats.rankdata(x),stats.rankdata(y)))

def rng_for(seed,*keys):
    digest=hashlib.sha256(json.dumps(keys,ensure_ascii=False).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([seed,int.from_bytes(digest[:8],'little')]))

def interval(vals,requested):
    vals=np.asarray(vals); valid=vals[np.isfinite(vals)]
    lo,hi=np.percentile(valid,[2.5,97.5]) if len(valid) else (np.nan,np.nan)
    return dict(ci_low=lo,ci_high=hi,bootstrap_valid=len(valid),bootstrap_valid_fraction=len(valid)/requested,
                ci_reliable=len(valid)/requested>=.975,
                ci_method='participant_percentile_bootstrap_exploratory')

def bh(p):
    p=np.asarray(p,float); out=np.full(len(p),np.nan)
    idx=np.flatnonzero(np.isfinite(p)); idx=idx[np.argsort(p[idx])]
    if len(idx): out[idx]=np.minimum(1,np.minimum.accumulate((p[idx]*len(idx)/np.arange(1,len(idx)+1))[::-1])[::-1])
    return out

def between(people,rng,B,P,minimum):
    x=np.array([r['x'] for r in people]); y=np.array([r['Score'] for r in people])
    base=dict(n_users=len(x),n_informative_users=len(x))
    if len(x)<minimum:return dict(base,status='insufficient_users')
    if np.ptp(x)==0:return dict(base,status='no_predictor_variation')
    if np.ptp(y)==0:return dict(base,status='no_score_variation')
    rho=spearman(x,y); xr=stats.rankdata(x); yr=stats.rankdata(y)
    perm=np.argsort(rng.random((P,len(x))),axis=1)
    null=corr(np.broadcast_to(xr,perm.shape),yr[perm])
    p=(1+np.count_nonzero(np.abs(null)>=abs(rho)-1e-12))/(P+1)
    idx=rng.integers(len(x),size=(B,len(x)))
    boot=corr(stats.rankdata(x[idx],axis=1),stats.rankdata(y[idx],axis=1))
    loo=[dict(omitted_user=person['user'],association=spearman(np.delete(x,i),np.delete(y,i))) for i,person in enumerate(people)]
    vals=[r['association'] for r in loo if np.isfinite(r['association'])]
    return dict(base,status='ok',association=rho,p=p,p_method='two_sided_participant_score_permutation',
                pearson_descriptive=float(corr(x,y)),loo_min=min(vals) if vals else np.nan,
                loo_max=max(vals) if vals else np.nan,loo=loo,**interval(boot,B))

def partial(grams,adjusted):
    xx,yy,xy=grams[...,0,0],grams[...,1,1],grams[...,0,1]
    if adjusted:
        inv=np.linalg.pinv(grams[...,2:,2:],rcond=1e-10,hermitian=True)
        gx,gy=grams[...,0,2:],grams[...,1,2:]
        xx=xx-np.einsum('...i,...ij,...j->...',gx,inv,gx)
        yy=yy-np.einsum('...i,...ij,...j->...',gy,inv,gy)
        xy=xy-np.einsum('...i,...ij,...j->...',gx,inv,gy)
    den=np.sqrt(np.maximum(xx,0)*np.maximum(yy,0))
    return np.clip(np.divide(xy,den,out=np.full(np.shape(den),np.nan),where=den>1e-10),-1,1)

def within_core(records,adjusted):
    """Equal total user weights; rank x/y globally within the fitted condition."""
    records=sorted(records,key=lambda r:(r['user'],r['TaskNum'],r['Attempt'],r['content_id']))
    users=sorted({r['user'] for r in records}); n=len(records); g=len(users)
    if not n:return dict(status='no_eligible_records',n_users=0,n_informative_users=0)
    raw=np.column_stack((stats.rankdata([r['metric_value'] for r in records]),
                         stats.rankdata([r['Score'] for r in records]),
                         [r['TaskNum'] for r in records],[r['Attempt'] for r in records]))
    ua=np.array([r['user'] for r in records]); centered=np.zeros_like(raw); w=np.zeros(n); grams=[]; informative=0
    for u in users:
        m=ua==u; block=raw[m]-raw[m].mean(axis=0); centered[m]=block; w[m]=1/m.sum()
        informative+=int(np.ptp(block[:,0])>0 and np.ptp(block[:,1])>0)
        grams.append(block.T@(block/m.sum()))
    grams=np.array(grams); rho=float(partial(grams.sum(axis=0),adjusted))
    base=dict(n_users=g,n_informative_users=informative)
    if not np.isfinite(rho):return dict(base,status='insufficient_residual_variation')
    design=centered[:,[0,2,3]] if adjusted else centered[:,[0]]
    y=centered[:,1]; bread=np.linalg.pinv(design.T@(w[:,None]*design),rcond=1e-10,hermitian=True)
    beta=bread@(design.T@(w*y)); residual=y-design@beta
    rank=int(np.linalg.matrix_rank(design)); df=n-g-rank
    if df<=0 or g<2:return dict(base,status='insufficient_residual_degrees_of_freedom')
    scores=np.array([design[ua==u].T@(w[ua==u]*residual[ua==u]) for u in users])
    cov=bread@(scores.T@scores)@bread*(g/(g-1))*((n-1)/df)
    se=np.sqrt(max(cov[0,0],0)); p=float(2*stats.t.sf(abs(beta[0]/se),g-1)) if se>1e-12 else np.nan
    xp=centered[:,0].copy(); yp=centered[:,1].copy()
    if adjusted:
        z=centered[:,2:]; iz=np.linalg.pinv(z.T@(w[:,None]*z),rcond=1e-10,hermitian=True)
        xp-=z@(iz@(z.T@(w*xp))); yp-=z@(iz@(z.T@(w*yp)))
    points=[dict(content_id=r['content_id'],user=r['user'],x_residual=xp[i],y_residual=yp[i],weight=w[i]) for i,r in enumerate(records)]
    return dict(base,status='ok' if np.isfinite(p) else 'degenerate_cluster_standard_error',association=rho,p=p,
                rank_slope=float(beta[0]),rank_slope_cluster_se=float(se),cluster_df=g-1,
                residual_df=df,design_rank=rank,attempt_covariate_informative=bool(adjusted and np.ptp(centered[:,3])>0),
                grams=grams,points=points)

def within(records,adjusted,rng,B,minimum):
    result=within_core(records,adjusted)
    if result.get('n_informative_users',0)<minimum:
        return dict(n_users=result.get('n_users',0),n_informative_users=result.get('n_informative_users',0),
                    status='insufficient_users_with_within_variation')
    if result['status']!='ok':return {k:v for k,v in result.items() if k not in ('grams','points')}
    grams=result.pop('grams'); g=len(grams)
    idx=rng.integers(g,size=(B,g)); boot=partial(grams[idx].sum(axis=1),adjusted)
    loo=[]
    for u in sorted({r['user'] for r in records}):
        fitted=within_core([r for r in records if r['user']!=u],adjusted)
        loo.append(dict(omitted_user=u,association=fitted.get('association',np.nan),status=fitted['status']))
    vals=[r['association'] for r in loo if np.isfinite(r['association'])]
    return dict(result,**interval(boot,B),p_method='participant_cluster_CR1_t_approximation',
                bootstrap_rank_policy='fixed_full_sample_ranks',loo_rank_policy='rerank_and_refit',loo=loo,
                loo_min=min(vals) if vals else np.nan,loo_max=max(vals) if vals else np.nan)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path('D:/GitHub_Project/co-tracker'))
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--permutations',type=int,default=19999)
    parser.add_argument('--seed',type=int,default=20261109)
    args=parser.parse_args(); out=args.output.resolve(); repo=args.repo.resolve()
    source=repo/'outputs/gaze_psm_revision_20261106/gt84'; old=repo/'outputs/score_corr_max_dlc'
    assert out!=source and out!=old and source not in out.parents and old not in out.parents
    for sub in ['tables','inputs','figures','code']: (out/sub).mkdir(parents=True,exist_ok=True)
    protocol=dict(analysis_date='2026-11-09',source_version='20261106 NPY GT84',source=str(source),
        stages=STAGES,scenarios=SCENARIOS,primary_predictor='peak_r',secondary_predictor='r_zero',outcome='Score',
        primary_inference='primary/peak_r/between_participants',methods=METHODS,
        bootstrap=args.bootstrap,permutations=args.permutations,seed=args.seed,min_users=6,
        eligible='primary variant; gt; fixed left/right; finite metric and Score; Completed or Failed; full_lag_support',
        keep_nonpositive_correlations=True,keep_zero_scores=True,lag_identifiability_required_for_strength=False,
        family='scenario/predictor/method pooled over A/C, all stages and left/right; finite p only',
        bootstrap_within='whole participant, original full-sample ranks retained; exploratory percentile interval',
        sensitivity_interpretation='separate populations/estimands; no overall FDR across families',
        lag='only lag_identifiable recordings; descriptive/exploratory; do not require positive lag',
        quality='respect saved GT84 finite/support rules; no additional likelihood or jump exclusion')
    write_json(out/'analysis_protocol.json',protocol)
    files=[source/f for f in ['record_summary.csv','run_manifest.json','input_manifest.json','preprocessing_qc.csv','README.md']]
    files += [repo/'Index'/f for f in ['Task_A_Score.csv','Task_C_Score.csv','content_index.csv']]
    protected=[old/f for f in ['all_associations.csv','associations.csv','matched_scores.csv','record_phase_audit.csv','report.md','provenance.json','corrections/review_zh.md']]
    hashes={str(p):sha(p) for p in files+protected}
    # Snapshot consumed files without touching source data or historical outputs.
    for p in files:
        destination=out/'inputs'/p.name
        if destination.exists():assert sha(destination)==hashes[str(p)],'Input snapshot differs; use a new output directory.'
        else:destination.write_bytes(p.read_bytes())
    df=pd.read_csv(source/'record_summary.csv',keep_default_na=True)
    manifest=json.loads((source/'run_manifest.json').read_text(encoding='utf-8'))
    manifest_records=pd.DataFrame(manifest['records'])[['content_id','user','task']].drop_duplicates()
    assert len(manifest_records)==84 and manifest_records.content_id.is_unique
    assert set(df.content_id)==set(manifest_records.content_id)
    keys=['variant','content_id','task','stage','side','signal','mode','source']
    assert not df.duplicated(keys).any(),'Duplicate metric keys'
    scores=[]
    for task in ['A','C']:
        s=pd.read_csv(repo/'Index'/f'Task_{task}_Score.csv')
        s['content_id']=['User%d_%d_%d'%tuple(v) for v in s[['UserID','TaskNum','Attempt']].to_numpy()]
        s['score_task']=task; scores.append(s)
    scores=pd.concat(scores,ignore_index=True)
    assert scores.content_id.is_unique,'Duplicate score keys'
    ix=pd.read_csv(repo/'Index/content_index.csv'); assert ix.content_id.is_unique
    matched=manifest_records.merge(scores,on='content_id',how='left',validate='one_to_one',indicator=True)
    assert (matched['_merge']=='both').all(); matched=matched.drop(columns='_merge')
    indexed=ix.set_index('content_id')
    for r in matched.to_dict('records'):
        q=indexed.loc[r['content_id']]
        assert r['task']==r['score_task']==q['Task Name']
        assert r['user']=='User'+str(int(r['UserID']))
        assert int(r['UserID'])==int(q['User Num']) and int(r['TaskNum'])==int(q['Task Num']) and int(r['Attempt'])==int(q['Attempt Num'])
        assert str(r['UserType']).lower()==str(q['src_skill']).lower()
    matched['score_usable']=np.isfinite(matched.Score)&matched.AttemptStatus.isin(['Completed','Failed'])
    matched['score_exclusion']=np.where(~np.isfinite(matched.Score),'missing_score',np.where(~matched.AttemptStatus.isin(['Completed','Failed']),'unscored_status',''))
    matched['score_source']=matched.task.map({t:str(repo/'Index'/f'Task_{t}_Score.csv') for t in ['A','C']})
    oldmatch=pd.read_csv(old/'matched_scores.csv')
    assert set(oldmatch.content_id)==set(matched.content_id)
    compare=matched[['content_id','Score','AttemptStatus']].merge(oldmatch[['content_id','Score','AttemptStatus']],on='content_id',suffixes=('_new','_old'),validate='one_to_one')
    assert np.allclose(compare.Score_new,compare.Score_old,equal_nan=True)
    assert (compare.AttemptStatus_new==compare.AttemptStatus_old).all()
    # Preserve prior score-source diagnostic flags as provenance, not exclusions.
    prior_flags=[c for c in ['duration_review_status','duration_flag_is_matching_error','duplicate_component_content_ids','severe_total_penalty_proxy','suture_or_tower_minus100_flag'] if c in oldmatch]
    matched=matched.merge(oldmatch[['content_id']+prior_flags],on='content_id',validate='one_to_one')
    write_table(out/'tables/matched_scores.csv',matched)
    qc=pd.read_csv(source/'preprocessing_qc.csv'); write_json(out/'tables/qc_summary.json',dict(rows=len(qc),records=qc.content_id.nunique(),policy='QC read for provenance; no new exclusions beyond saved support rules'))
    score_lookup=matched.set_index('content_id').to_dict('index')
    data=df[(df.variant=='primary')&(df.source=='gt')&df.side.isin(['l','r'])].copy()
    assert data.analysis.isin(['fixed_left','fixed_right']).all()
    for c in ['full_lag_support','peak_at_boundary','lag_identifiable']:
        assert data[c].dropna().isin([True,False]).all(),f'Invalid boolean {c}'
    results=[]; participants=[]; audits=[]; cover=[]; loo_rows=[]; within_points=[]
    for scenario,(signal,mode) in SCENARIOS.items():
        print('Analyzing',scenario,flush=True)
        predictors=['signed_lag_s','absolute_lag_s'] if scenario=='lag_identifiable_exploratory' else ['peak_r','r_zero']
        selected=data[(data.signal==signal)&(data['mode']==mode)]
        for task,stages in STAGES.items():
            for stage in stages:
                for side in ['l','r']:
                    group=selected[(selected.task==task)&(selected.stage==stage)&(selected.side==side)]
                    assert group.content_id.is_unique
                    for predictor in predictors:
                        meta=dict(scenario=scenario,task=task,stage=stage,side=side,signal=signal,mode=mode,predictor=predictor)
                        rr=[]; candidate=[]
                        for r in group.to_dict('records'):
                            sc=score_lookup[r['content_id']]
                            value=r['peak_lag_s'] if predictor=='signed_lag_s' else (abs(r['peak_lag_s']) if predictor=='absolute_lag_s' else r[predictor])
                            reasons=[]
                            if not sc['score_usable']:reasons.append(sc['score_exclusion'])
                            if not bool(r['full_lag_support']):reasons.append('incomplete_lag_support')
                            if not np.isfinite(value):reasons.append('nonfinite_metric')
                            if scenario=='completed_only' and sc['AttemptStatus']!='Completed':reasons.append('not_completed')
                            if scenario=='exclude_boundary' and bool(r['peak_at_boundary']):reasons.append('boundary_peak')
                            if scenario=='lag_identifiable_exploratory' and not bool(r['lag_identifiable']):reasons.append('lag_not_identifiable')
                            row=dict(meta,content_id=r['content_id'],user=r['user'],TaskNum=int(sc['TaskNum']),Attempt=int(sc['Attempt']),
                                Score=sc['Score'],AttemptStatus=sc['AttemptStatus'],score_usable=sc['score_usable'],metric_value=value,
                                peak_r=r['peak_r'],r_zero=r['r_zero'],peak_lag_s=r['peak_lag_s'],full_lag_support=r['full_lag_support'],
                                peak_at_boundary=r['peak_at_boundary'],lag_identifiable=r['lag_identifiable'],
                                eligible=not reasons,exclusion_reasons=';'.join(reasons))
                            candidate.append(row);audits.append(row)
                            if not reasons:rr.append(row)
                        people=[]
                        for user in sorted({r['user'] for r in rr}):
                            sub=[r for r in rr if r['user']==user]; values=np.array([r['metric_value'] for r in sub])
                            x=float(np.tanh(np.mean(np.arctanh(np.clip(values,-1+1e-7,1-1e-7))))) if predictor in ['peak_r','r_zero'] else float(np.median(values))
                            person=dict(meta,user=user,x=x,Score=float(np.mean([r['Score'] for r in sub])),n_recordings=len(sub),content_ids=';'.join(r['content_id'] for r in sub))
                            people.append(person);participants.append(person)
                        cover.append(dict(meta,n_total_records=len(candidate),n_score_available=sum(r['score_usable'] for r in candidate),
                            n_eligible_records=len(rr),n_users=len(people),n_nonpositive_metric=sum(r['metric_value']<=0 for r in rr),
                            n_zero_scores=sum(r['Score']==0 for r in rr),n_failed=sum(r['AttemptStatus']=='Failed' for r in rr),
                            n_boundary=sum(bool(r['peak_at_boundary']) for r in rr),n_lag_identifiable=sum(bool(r['lag_identifiable']) for r in rr)))
                        for method in METHODS:
                            random=rng_for(args.seed,*[meta[k] for k in META],method)
                            fit=between(people,random,args.bootstrap,args.permutations,6) if method=='between_participants' else within(rr,method=='within_adjusted_order',random,args.bootstrap,6)
                            for r in fit.pop('loo',[]):loo_rows.append(dict(meta,method=method,**r))
                            for p in fit.pop('points',[]):
                                if scenario=='primary':within_points.append(dict(meta,method=method,**p))
                            results.append(dict(meta,method=method,outcome='Score',n_recordings=len(rr),family=f'{scenario}/{predictor}/{method}',**fit))
    assoc=pd.DataFrame(results)
    assoc['q_bh']=np.nan;assoc['family_size']=0
    for _,ids in assoc.groupby('family').groups.items():
        assoc.loc[ids,'q_bh']=bh(assoc.loc[ids,'p'].to_numpy())
        assoc.loc[ids,'family_size']=int(np.isfinite(assoc.loc[ids,'p']).sum())
    assoc['analysis_role']=np.where((assoc.scenario=='primary')&(assoc.predictor=='peak_r')&(assoc.method=='between_participants'),'primary',
                         np.where(assoc.scenario=='lag_identifiable_exploratory','lag_exploratory','secondary_or_sensitivity'))
    write_table(out/'tables/associations.csv',assoc)
    write_table(out/'tables/primary_peak_r_associations.csv',assoc[assoc.analysis_role=='primary'])
    write_table(out/'tables/lag_exploratory_associations.csv',assoc[assoc.scenario=='lag_identifiable_exploratory'])
    write_table(out/'tables/participant_summaries.csv',participants)
    write_table(out/'tables/matched_record_metrics.csv',audits)
    write_table(out/'tables/coverage.csv',cover)
    write_table(out/'tables/leave_one_user_out.csv',loo_rows)
    write_table(out/'tables/within_points.csv',within_points)
    write_table(out/'tables/family_summary.csv',assoc.groupby('family',sort=False).agg(planned_conditions=('status','size'),estimable_tests=('p','count'),min_p=('p','min'),min_q=('q_bh','min')).reset_index())
    significant=assoc[(assoc.status=='ok')&(assoc.q_bh<.05)]
    write_table(out/'tables/family_significant_results.csv',significant)
    unchanged={p:sha(Path(p))==h for p,h in hashes.items()}; assert all(unchanged.values())
    counts=[]
    for task,s in matched.groupby('task'):
        counts.append(dict(task=task,n_records=len(s),n_users=s.user.nunique(),n_score_usable=int(s.score_usable.sum()),
            n_zero_score=int((s.Score==0).sum()),n_failed=int((s.AttemptStatus=='Failed').sum()),
            missing_score_ids=s.loc[~s.score_usable,'content_id'].tolist()))
    summary=dict(record_counts=counts,n_association_rows=len(assoc),n_estimable=int(assoc.p.notna().sum()),
        n_significant_rows=len(significant),primary_significant=assoc[(assoc.analysis_role=='primary')&(assoc.q_bh<.05)].to_dict('records'),
        significant_rows=significant.to_dict('records'),source_unchanged=all(unchanged.values()))
    write_json(out/'analysis_summary.json',summary)
    write_json(out/'provenance.json',dict(protocol=protocol,executed_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
        runtime_note='Bundled runtime lacks scipy/matplotlib; existing Anaconda provides required capabilities.',
        input_sha256=hashes,input_unchanged=unchanged,analysis_code_sha256=sha(Path(__file__)),
        previous_max_results_used_for='identity/score cross-check and source issue flags only; no old coefficients substituted'))
    print(json.dumps(clean_json({k:v for k,v in summary.items() if k not in ['significant_rows','primary_significant']}),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
