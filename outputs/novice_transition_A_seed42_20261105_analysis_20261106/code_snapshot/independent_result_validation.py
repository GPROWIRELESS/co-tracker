"""Independent outcome verification, without importing either analysis driver."""
import os,json,sys
from pathlib import Path
from collections import defaultdict
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
ROOT=Path('D:\\GitHub_Project\\co-tracker\\outputs\\novice_transition_A_seed42_20261105_analysis_20261106')
MODELS=['current_transition','future500_transition','future1000_transition','future1500_transition']
STAGES=['reach ring','grab ring','transfer ring','place ring']
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def eq(a,b,tol=1e-9):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def circle_state(xy,z,l,r,radius):
    t=((xy-z)**2).sum(1)<=radius**2
    p=(((xy-l)**2).sum(1)<=radius**2)|(((xy-r)**2).sum(1)<=radius**2)
    return t.astype(int)+2*p.astype(int)
def main():
    checks=[];details={};tl=ROOT/'target_locking';lead=ROOT/'gaze_psm_lead'
    t=pd.read_csv(tl/'trial_metrics.csv');e=pd.read_csv(tl/'trial_errors.csv')
    assert len(t)==2*len(e)
    pk=['content_id','participant','axis','model','scenario','stage','branch']
    gt=t[t.source=='GT'].set_index(pk);md=t[t.source=='model'].set_index(pk);er=e.set_index(pk)
    assert gt.index.equals(md.index) and gt.index.equals(er.index)
    eq(gt.valid_n,md.valid_n);eq(gt.valid_s,md.valid_s)
    for metric in ['L_N','L_T','L_P','L_O','L_target','L_psm','D','C','C_excess','E_target','E_psm']:
        eq(er['delta_'+metric],md[metric]-gt[metric])
    nonempty=t.valid_s>0;eq(t.loc[nonempty,['L_N','L_T','L_P','L_O']].sum(axis=1),1,1e-8)
    eq(t.L_target,t.L_T+t.L_O);eq(t.L_psm,t.L_P+t.L_O);eq(t.D,t.L_T+t.L_P)
    eq(t.C,t.L_T/t.D);eq(t.C_excess,t.C-t.C_U)
    assert t.loc[nonempty,['L_N','L_T','L_P','L_O']].min().min()>=-1e-9
    assert t.loc[nonempty,['L_N','L_T','L_P','L_O']].max().max()<=1+1e-9
    assert e.BS_target.dropna().between(0,1).all() and e.BS_psm.dropna().between(0,1).all()
    assert e.loc[e.valid_s>0,'BS_target'].notna().all() and e.loc[e.valid_s>0,'BS_psm'].notna().all()
    checks.append({'name':'TL_pairing_probability_conservation_differences','passed':True,'trial_rows':len(t)})
    anchor_sets=defaultdict(list);manual_frames=0;metric_checks=0
    scenarios={'main':50,'r25':25,'r100':100}
    assert set(t.scenario)==set(scenarios), 'Removed status-code scenario must not remain'
    assert set(e.scenario)==set(scenarios)
    for p in sorted((tl/'frames').glob('*.npz')):
        f=load(p);cid=str(f['content_id']);axis=str(f['axis']);model=str(f['model'])
        d=load(ROOT/'input_audit/native'/f'{cid}.npz');ix=f['geometry_native_index'];n=len(ix)
        for field,saved in [('time','timestamp_ns'),('z','target_xy'),('left','left_xy'),('right','right_xy'),('gaze_unmasked','gaze_xy')]:
            if field=='time':continue
            eq(f[saved],d[field][ix])
        np.testing.assert_array_equal(f['stage'],d['stage'][ix])
        # Independently form midpoint exposure using adjacent source anchors.
        dt=np.zeros(n);sec=np.diff(f['timestamp_ns']).astype(float)/1e9
        good=(np.diff(f['source_anchor'])==2)&(sec>0)&(sec<=.1)&(np.diff(f['occurrence'])==0)
        good&=f['target_time_valid'][:-1]&f['target_time_valid'][1:]
        for j in np.flatnonzero(good):dt[j]+=sec[j]/2;dt[j+1]+=sec[j]/2
        eq(f['dt_s'],dt,1e-12)
        for scenario,radius in scenarios.items():
            mask=f['valid_'+scenario]
            xy=f['gaze_xy'];l=f['left_xy'];r=f['right_xy'];z=f['target_xy']
            expected=np.isin(f['stage'],STAGES)&(d['count'][ix]>0)&f['target_time_valid']
            for v in [l,r]:expected&=np.isfinite(v).all(1)&(v>=0).all(1)&(v<[d['width'],d['height']]).all(1)
            expected&=np.isfinite(xy).all(1)&(xy>=0).all(1)&(xy<=[d['width'],d['height']]).all(1)
            np.testing.assert_array_equal(mask,expected)
            state=circle_state(xy,z,l,r,radius);np.testing.assert_array_equal(f['gt_'+scenario],np.where(mask,state,-1))
            for branch,name in [('decoded_expectation','expectedxy'),('decoded_argmax','argmaxxy')]:
                s=circle_state(f[name],z,l,r,radius);q=f['q_'+branch+'_'+scenario]
                eq(q[mask],np.eye(4)[s[mask]],1e-12)
            q=f['q_probability_map_'+scenario];eq(q[mask].sum(1),1,2e-7)
            for stage in ['whole']+STAGES:
                m=mask if stage=='whole' else mask&(f['stage']==stage)
                w=dt[m];den=w.sum()
                if den<=0:continue
                rec=t[(t.content_id==cid)&(t.axis==axis)&(t.model==model)&(t.scenario==scenario)&(t.stage==stage)&(t.branch=='probability_map')]
                gm=rec[rec.source=='GT'].iloc[0];mm=rec[rec.source=='model'].iloc[0]
                eq(gm.valid_s,den);eq(gm.valid_n,m.sum());eq(mm.valid_s,den)
                eq(mm[['L_N','L_T','L_P','L_O']].to_numpy(float),w@q[m]/den,3e-8)
                obs=(state[m]&1)>0;pred=q[m,1]+q[m,3]
                ee=e[(e.content_id==cid)&(e.axis==axis)&(e.model==model)&(e.scenario==scenario)&(e.stage==stage)&(e.branch=='probability_map')].iloc[0]
                eq(ee.BS_target,w@((pred-obs)**2)/den,3e-8);metric_checks+=1
            if axis=='anchor':anchor_sets[(cid,scenario)].append(mask)
        manual_frames+=n
    for k,masks in anchor_sets.items():
        assert len(masks)==4
        for m in masks[1:]:np.testing.assert_array_equal(m,masks[0])
    checks.append({'name':'TL_independent_native_mask_geometry_weights_and_Brier','passed':True,'frame_rows':manual_frames,'aggregate_checks':metric_checks})
    details['tl_anchor_common_main']=t[(t.axis=='anchor')&(t.model==MODELS[0])&(t.branch=='probability_map')&(t.source=='GT')&(t.scenario=='main')&(t.stage=='whole')][['content_id','participant','valid_n','valid_s']].to_dict('records')
    if (tl/'group_metrics.csv').exists():
        group=pd.read_csv(tl/'group_metrics.csv');person=pd.read_csv(tl/'participant_metrics.csv')
        fields=['axis','model','branch','scenario','stage','source'];metrics=['L_N','L_T','L_P','L_O','L_target','L_psm','D','C','C_excess','E_target','E_psm']
        wanted=t.groupby(fields+['participant'])[metrics].mean().sort_index()
        got=person.set_index(fields+['participant']).sort_index()[metrics]
        eq(wanted.to_numpy(),got.to_numpy())
        expected=wanted.groupby(level=list(range(len(fields)))).mean()
        for row in group.itertuples():
            if row.metric not in metrics:continue
            key=tuple(getattr(row,k) for k in fields)
            eq(row.estimate,expected.loc[key,row.metric])
        checks.append({'name':'TL_User5_equal_record_then_equal_participant_aggregation','passed':True})
    if (lead/'group_summary.csv').exists():
        g=pd.read_csv(lead/'group_summary.csv');p=pd.read_csv(lead/'paired_metrics.csv');r=pd.read_csv(lead/'record_summary.csv')
        fields=['variant','task','side','signal','stage','mode']
        for key,part in g.groupby(fields):
            assert part.n_complete_records.nunique()==1,(key,'group recording support differs')
            assert part.n_stable_users.nunique()==1,(key,'group participant support differs')
            assert part.participants_json.nunique()==1,(key,'group identities differ')
        assert not r.stage.eq('drop ring').any()
        eq(p.delta_lag_s,p.model_peak_lag_s-p.gt_peak_lag_s);eq(p.abs_lag_difference_s,abs(p.delta_lag_s));eq(p.delta_peak_r,p.model_peak_r-p.gt_peak_r)
        checks.append({'name':'Lead_common_group_record_and_participant_support_and_paired_differences','passed':True})
        # Reconstruct all primary group Fisher means independently from saved curves.
        grouped=defaultdict(list);records=defaultdict(dict);zero=defaultdict(dict)
        with (lead/'record_curves.jsonl').open(encoding='utf-8') as f:
            for line in f:
                c=json.loads(line)
                if c['source'] in ('gt','current_transition') and (c['variant'].startswith('anchor__') or c['variant'].startswith('target_current_transition__')):
                    view,variant=c['variant'].split('__',1)
                    zk=(variant,)+tuple(c[k] for k in ['content_id','user','task','side','signal','stage','mode','source'])
                    zero[zk][view]=c
                if c['variant']!='anchor__primary':continue
                key=tuple(c[k] for k in ['variant','content_id','user','task','side','signal','stage','mode'])
                records[key][c['source']]=c
        for key,src in records.items():
            if not all(np.isfinite(np.asarray(c['r'],float)).all() for c in src.values()):continue
            for source,c in src.items():
                kk=tuple(c[k] for k in fields)+(source,)
                grouped[kk].append(c)
        curves=pd.read_csv(lead/'group_curves.csv');nchecks=0;maxerror=0
        for key,rr in grouped.items():
            by=defaultdict(list)
            for c in rr:by[c['user']].append(np.arctanh(np.clip(np.asarray(c['r'],float),-1+1e-7,1-1e-7)))
            if len(by)<3:continue
            expected=np.tanh(np.mean([np.mean(v,axis=0) for v in by.values()],axis=0))
            select=np.ones(len(curves),bool)
            for field,value in zip(fields+['source'],key):select&=curves[field].eq(value).to_numpy()
            got=curves[select].sort_values('lag_s').r.to_numpy()
            eq(got,expected,1e-10);maxerror=max(maxerror,float(np.max(abs(got-expected))));nchecks+=1
        checks.append({'name':'Lead_independent_Fisher_equal_record_within_user_then_user_group_curve','passed':True,'curves':nchecks,'max_abs_error':maxerror})
        for key,pair in zero.items():
            assert set(pair)=={'anchor','target_current_transition'},('zero_horizon_missing_pair',key)
            for field in ['lag_s','r','pairs','occurrences','gaze_rms','psm_rms']:
                eq(np.asarray(pair['anchor'][field],float),np.asarray(pair['target_current_transition'][field],float),1e-10)
        checks.append({'name':'Lead_zero_horizon_record_curves_and_support_identical','passed':True,'paired_curves':len(zero)})
        details['lead_primary_group_rows']=int(g.variant.eq('anchor__primary').sum())
        if (lead/'supplement_validation.json').exists():
            summary=pd.read_csv(lead/'paired_summary.csv')
            observed=summary[summary.stage=='__observed_stage_average__']
            for row in observed.itertuples():
                coverage=json.loads(row.stage_coverage_json)
                assert all('whole' not in stages for stages in coverage.values())
            refs=pd.read_csv(lead/'unpaired_reference.csv')
            not_attempted=refs.get('not_attempted_draws',pd.Series(0,index=refs.index)).fillna(0)
            eq(refs.valid_draws+refs.invalid_support_draws+not_attempted,refs.planned_draws)
            assert (refs.planned_draws==5000).all()
            for row in refs.itertuples():
                select=g[(g.variant=='anchor__primary')&(g.signal=='position')&(g['mode']=='psm_stage')&(g.side==row.side)&(g.stage==row.stage)&(g.source==row.source)]
                assert len(select)==1
                match=select.iloc[0]
                assert row.n_observed_recordings==match.n_complete_records
                assert row.n_users==match.n_stable_users
            checks.append({'name':'Lead_final_phase_average_and_reference_exact_observed_support','passed':True,'reference_rows':len(refs)})
    result={'passed':all(c['passed'] for c in checks),'complete':(tl/'findings.json').exists() and (lead/'findings.json').exists() and (lead/'supplement_validation.json').exists() and (lead/'zero_horizon_validation.json').exists(),'checks':checks,'details':details}
    (ROOT/'verification/independent_result_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
