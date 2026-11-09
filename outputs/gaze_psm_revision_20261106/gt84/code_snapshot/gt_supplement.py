"""Transparent support, participant resampling and paired-stage diagnostics."""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from gaze_psm_statistics_v2 import _fisher, _stable_seed

def supplement(curves,groups,selected,output,count=5000):
    from run_gt84_revision import csvout,save
    out=Path(output);gg=defaultdict(lambda:defaultdict(list));index=[];draws={};people={}
    fields=('variant','task','side','signal','stage','mode','source')
    for c in list(curves)+list(selected):
        if c['variant']=='primary' and np.isfinite(c['r']).all():
            gg[tuple(c[k] for k in fields)][c['user']].append(c)
    for key,users in sorted(gg.items()):
        ids=sorted(users);lag=np.asarray(next(iter(users.values()))[0]['lag_s'])
        aa=np.stack([_fisher([c['r'] for c in users[u]]) for u in ids]);people[key]=(ids,aa,lag)
        if len(ids)<3:continue
        rng=np.random.default_rng(_stable_seed(42,(key[:-1],tuple(ids))))
        ix=rng.integers(0,len(ids),(count,len(ids)));z=np.arctanh(np.clip(aa,-1+1e-7,1-1e-7))
        rep=np.tanh(z[ix].mean(1));peak=lag[np.argmax(rep,axis=1)];token=f'g{len(index):03d}'
        draws[token]=peak;index.append(dict(zip(fields,key),token=token,users=ids))
    np.savez_compressed(out/'group_bootstrap_peak_lags.npz',**draws);save(out/'group_bootstrap_index.json',index)
    contrasts=[];contrastdraws={}
    for ka,kb in combinations(sorted(people),2):
        if ka[2] not in ('l','r') or any(ka[i]!=kb[i] for i in (0,1,2,3,5,6)) or ka[4]==kb[4]:continue
        ua,aa,lag=people[ka];ub,bb,lagb=people[kb];users=sorted(set(ua)&set(ub))
        row=dict(task=ka[1],side=ka[2],signal=ka[3],mode=ka[5],condition_a=ka[4],condition_b=kb[4],n_users=len(users),
                 users=users,difference_definition='condition_b_minus_condition_a',inference='exploratory paired-user range; no multiplicity-adjusted lag test')
        if len(users)<3:
            contrasts.append(dict(row,status='insufficient_users'));continue
        a=np.stack([aa[ua.index(u)] for u in users]);b=np.stack([bb[ub.index(u)] for u in users])
        za,zb=np.arctanh(np.clip(a,-1+1e-7,1-1e-7)),np.arctanh(np.clip(b,-1+1e-7,1-1e-7))
        pa,pb=np.argmax(za.mean(0)),np.argmax(zb.mean(0))
        rng=np.random.default_rng(_stable_seed(42,('contrast',ka,kb,tuple(users))));ix=rng.integers(0,len(users),(count,len(users)))
        ia,ib=za[ix].mean(1).argmax(1),zb[ix].mean(1).argmax(1)
        dif=lag[ib]-lag[ia];token=f'c{len(contrasts):03d}';contrastdraws[token]=dif
        boundary=(pa in (0,len(lag)-1) or pb in (0,len(lag)-1));clamp=((ia==0)|(ia==len(lag)-1)|(ib==0)|(ib==len(lag)-1))
        row.update(peak_lag_a_s=lag[pa],peak_lag_b_s=lag[pb],difference_s=lag[pb]-lag[pa],
                   resample_low=np.percentile(dif,2.5),resample_high=np.percentile(dif,97.5),
                   boundary_frequency=clamp.mean(),status='boundary_peak' if boundary else 'exploratory',
                   boundary_screen_passed=not boundary and clamp.mean()<.025,
                   precise_delay_difference_allowed=False,draw_key=token)
        contrasts.append(row)
    csvout(out/'stage_contrasts.csv',contrasts);np.savez_compressed(out/'stage_contrast_bootstrap.npz',**contrastdraws)
    from gaze_psm_group import aggregate
    primary=[c for c in curves if c['variant']=='primary']
    _,old,_=aggregate(primary,bootstrap=0,seed=42,min_users=3)
    current={(r['task'],r['side'],r['signal'],r['stage'],r['mode']):r for r in groups if r['variant']=='primary'}
    comparisons=[]
    for r in old:
        key=tuple(r[x] for x in ('task','side','signal','stage','mode'));new=current[key]
        comparisons.append(dict(zip(('task','side','signal','stage','mode'),key),
            available_record_peak_lag_s=r['peak_lag_s'],complete_record_peak_lag_s=new['peak_lag_s'],
            available_record_peak_r=r['peak_r'],complete_record_peak_r=new['peak_r'],
            available_record_n_users=r['n_stable_users'],complete_record_n_users=new['n_stable_users'],
            record_support_changes=r['record_support_changes'],
            interpretation='Same signals and fixed instrument; only available-record versus complete-record support differs'))
    csvout(out/'aggregation_support_comparison.csv',comparisons)
