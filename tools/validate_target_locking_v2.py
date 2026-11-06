"""Independent checks of saved v2 measurements and legacy reproducibility."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_run(run,old=None):
    t=pd.read_csv(run/'trial_stage.csv')
    e=pd.read_csv(run/'model_trial_errors.csv')
    checked=0
    raw_checked=0
    audit=pd.read_csv(run/'prediction_audit.csv')
    pair=t[t.scope=='paired_test'].groupby(['content_id','scenario','stage'])
    assert (pair.size()==24).all()
    assert (pair.valid_n.nunique()==1).all() and (pair.valid_s.nunique()==1).all()
    for fp in sorted((run/'frames').glob('*.npz')):
        with np.load(fp,allow_pickle=False) as f:
            stages=f['stage']; dt=f['dt_s']
            for _,prediction in audit[audit.content_id==fp.stem].iterrows():
                source=prediction['model']
                with np.load(prediction['prediction_path'],allow_pickle=False) as original:
                    samples=np.asarray(original['samples'],float)*f['resolution']
                    np.testing.assert_array_equal(original['frame_number'],f['frame_number'])
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
    if old:
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
                legacy_expectation_rows_reproduced=legacy_rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--old',type=Path);p.add_argument('--save',action='store_true')
    args=p.parse_args();result=validate_run(args.run,args.old)
    print(json.dumps(result,indent=2))
    if args.save:(args.run/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
