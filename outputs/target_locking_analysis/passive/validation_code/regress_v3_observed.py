"""Regress observation-only extraction against pinned v3 and its raw inputs.

This does not load prediction arrays or evaluate any model. The pinned table's
real_all gaze rows are used solely as historical observed-measurement targets.
"""
from pathlib import Path
from decimal import Decimal
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'code'))
import tl_observed_geometry as geo
import tl_observed_core as core
from tl_observed_bootstrap import ParticipantBootstrap
from tl_observed_statistics import participant_summary

BASE = Path(r'D:\GitHub_Project\co-tracker\outputs\target_locking_analysis\20261105T063658759189Z_v3_full')
CANONICAL = {'rr':'reach ring','gr':'grab ring','tr':'transfer ring','pr':'place ring',
             'wrap':'wrap around psm','cinch':'pull through and cinch','thread':'threading string through tower'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):h.update(block)
    return h.hexdigest()

def independent_temporal(labels, dt, occ, breaks):
    """Simple run-list implementation, independent of vectorized v3 code."""
    chunks=[]; chunk=[]
    for i,k in enumerate(labels):
        if i and (breaks[i] or occ[i]!=occ[i-1]):
            chunks.append(chunk);chunk=[]
        if chunk and chunk[-1]['label']==k:
            chunk[-1]['seconds']+=float(dt[i])
        else:chunk.append({'label':int(k),'seconds':float(dt[i])})
    if chunk:chunks.append(chunk)
    durations=[[] for _ in range(4)];cens=np.zeros(4,int);alln=np.zeros(4,int);allc=np.zeros(4,int)
    direct=via=0
    for runs in chunks:
        for j,r in enumerate(runs):
            k,s=r['label'],r['seconds']
            if k<0:continue
            c=j==0 or j==len(runs)-1 or runs[j-1]['label']<0 or runs[j+1]['label']<0
            alln[k]+=1;allc[k]+=int(c)
            if s>=.120-1e-12:durations[k].append(s);cens[k]+=int(c)
            if k not in (1,2) or s<.120-1e-12:continue
            if j and runs[j-1]['label']==3-k and runs[j-1]['seconds']>=.120-1e-12:direct+=1
            elif j>=2 and runs[j-1]['label']==3 and runs[j-2]['label']==3-k and runs[j-2]['seconds']>=.120-1e-12:via+=1
    seconds=float(sum(float(t) for k,t in zip(labels,dt) if k>=0))
    out=dict(valid_seconds=seconds,direct_count=direct,via_count=via,
             direct_rate=direct*60/seconds if seconds else np.nan,via_rate=via*60/seconds if seconds else np.nan)
    for k,n in enumerate(geo.NAMES):
        out.update({f'dwell_{n}_median_s':float(np.median(durations[k])) if durations[k] else np.nan,
                    f'dwell_{n}_n':len(durations[k]),f'dwell_{n}_n_censored':int(cens[k]),
                    f'all_{n}_runs':int(alln[k]),f'all_{n}_runs_censored':int(allc[k])})
    for k,n in zip('TPO',geo.NAMES[1:]):out[f'dwell_{k}_ms']=out[f'dwell_{n}_median_s']*1000
    return out

def scalar_close(a,b,name):
    if isinstance(a,str):
        assert a==('' if pd.isna(b) else b),(name,a,b)
    else:np.testing.assert_allclose(float(a),float(b),rtol=1e-11,atol=1e-10,equal_nan=True,err_msg=name)

def load_raw(row):
    paths={k:Path(row[k+'_path']) for k in ('npy','video_ts','dlc')}
    for k,p in paths.items():assert sha(p)==row[k+'_sha256'],f'baseline input changed: {p}'
    with paths['video_ts'].open(encoding='utf-8-sig',newline='') as f:
        ts=[r for r in csv.DictReader(f) if r.get('IsDelete','0').strip() in ('','0','0.0')]
    frames=np.array([int(r['Frame_Number']) for r in ts],np.int64)
    ns=np.array([int(Decimal(r['Timestamp(ns)'])) for r in ts],np.int64)
    raw=np.load(paths['npy'],allow_pickle=False)
    ids={int(k):i for i,k in enumerate(raw['Frame_Number'])}
    assert len(ids)==len(raw)
    a=raw[[ids[int(k)] for k in frames]]
    with paths['dlc'].open(encoding='utf-8-sig',newline='') as f:
        r=csv.reader(f);headers=[next(r) for _ in range(3)];rows=list(r)
    assert len(rows)==len(frames) and [int(r[0]) for r in rows]==list(range(len(frames)))
    points={}
    for side,bp in [('left','left_psm'),('right','right_psm')]:
        columns=[next(i for i in range(len(headers[1])) if headers[1][i]==bp and headers[2][i]==ax) for ax in ['x','y']]
        xy=np.array([[float(r[c]) if r[c].strip() else np.nan for c in columns] for r in rows])
        xy[~np.isfinite(xy).all(axis=1)]=np.nan;points[side]=xy
    W,H=int(row.width),int(row.height)
    xy=np.column_stack([a['gaze_x'],a['gaze_y']]).astype(float)
    good=np.isfinite(xy).all(axis=1)&(xy>=0).all(axis=1)&(xy<=1).all(axis=1)
    xy*= [W,H];xy[~good]=np.nan
    names=set(a.dtype.names)
    stage=np.array([CANONICAL.get(s,s) for s in [' '.join(str(s).strip().lower().split()) for s in a['stage' if 'stage' in names else 'event']]])
    slots=np.full((len(a),5,2),np.nan)
    for slot in range(1,6):
        if f'point_{slot}_x' in names:slots[:,slot-1]=np.column_stack([a[f'point_{slot}_x'],a[f'point_{slot}_y']])
    identity=np.zeros(len(a),bool)
    field=next((s for s in ('target_id','target_identity','point_1_id') if s in names),None)
    if field:identity[1:]=a[field][1:]!=a[field][:-1]
    d=dict(task=row.task,frames=frames,timestamp_ns=ns,gaze=xy,gaze_unmasked=xy.copy(),
           slots=slots,stage=stage,occurrence=np.r_[0,np.cumsum(stage[1:]!=stage[:-1])].astype(np.int32),
           identity_change=identity,width=W,height=H,**points)
    return core.prepare_arrays(d), dict(input_hashes={str(p):sha(p) for p in paths.values()},
                                       raw_npy_n=len(raw),coordinate_valid_n=int(good.sum()),
                                       valid_code_ignored=True)

def core_identity_checks():
    records=[]
    for new,old in [('tl_observed_geometry.py','tl_geometry.py'),('tl_observed_bootstrap.py','tl_stats.py'),('tl_observed_statistics.py','tl_stats.py')]:
        a=ast.parse((ROOT/'code'/new).read_text());b=ast.parse((BASE/'code_snapshot'/old).read_text(encoding='utf-8-sig'))
        originals={n.name:n for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for n in a.body:
            if not isinstance(n,(ast.FunctionDef,ast.ClassDef)):continue
            original=originals[n.name]
            if isinstance(n,ast.ClassDef):original.body=[m for m in original.body if not isinstance(m,ast.FunctionDef) or m.name!='bss']
            assert ast.dump(n,include_attributes=False)==ast.dump(original,include_attributes=False),n.name
            records.append(new+':'+n.name)
    return records

def toy_checks():
    # Exposure uses adjacent source frames and native int64 differences.
    ns=np.int64(1750000000000000000)+np.array([0,20,40,150,170,190],np.int64)*10**6
    dt,br=geo.times(ns,np.array([1,2,3,4,6,7]))
    np.testing.assert_allclose(dt,[.01,.02,.01,0,.01,.01],atol=0,rtol=0)
    assert br.tolist()==[True,False,False,True,True,False]
    # Closed gaze boundary and open annotation boundary are distinct.
    slots=np.array([[[0,0],[100,20],[50,80],[99.9,79.9],[np.nan,0]]])
    z,n,bits=geo.target_center(slots,100,80)
    np.testing.assert_allclose(z,[[49.95,39.95]]);assert n[0]==2 and bits[0]==9
    assert core.in_image(np.array([[0,0],[100,0],[0,80],[99.9,79.9]]),100,80).tolist()==[True,False,False,True]
    # Independent full pixel enumeration exercises clipped and overlapping disks.
    rng=np.random.default_rng(809);W,H=31,27;n=20
    z,l,r=[rng.uniform([0,0],[W,H],(n,2)) for _ in range(3)]
    l[0]=r[0]=z[0];z[1]=[0,0]
    yy,xx=np.mgrid[:H,:W];pixels=np.column_stack([xx.ravel()+.5,yy.ravel()+.5])
    for rt,rp in [(0,0),(3.5,5),(15,8)]:
        area=geo.areas(z,l,r,W,H,rt,rp,np.ones(n,bool))
        for i in range(n):
            t=((pixels-z[i])**2).sum(axis=1)<=rt**2
            p=(((pixels-l[i])**2).sum(axis=1)<=rp**2)|(((pixels-r[i])**2).sum(axis=1)<=rp**2)
            code=t.astype(int)+2*p.astype(int)
            np.testing.assert_allclose(area[i],np.bincount(code,minlength=4)/(W*H),atol=1e-15)
    # Temporal censoring, overlap, missing and other interruption.
    labels=np.array([1]*3+[2]*3+[3]+[1]*3+[-1]+[2]*3+[0]+[1]*3)
    dt=np.repeat(.05,len(labels));occ=np.zeros(len(labels),int);br=np.zeros(len(labels),bool);br[0]=True
    a=geo.temporal(labels,dt,occ,br);b=independent_temporal(labels,dt,occ,br)
    for k in a:scalar_close(a[k],b[k],k)
    assert a['direct_count']==1 and a['via_count']==1
    # Smooth fills only short bounded gaps, respects occurrences and keeps isolated points.
    ts=np.int64(1750000000000000000)+np.arange(10,dtype=np.int64)*25000000
    raw=np.repeat(np.arange(10,dtype=float)[:,None],2,axis=1);raw[[1,3,4,5,6]]=np.nan
    occ=np.zeros(10,int);br=np.zeros(10,bool);br[0]=True
    smooth,filled=geo.smooth_gaze(ts,raw,occ,br)
    assert np.flatnonzero(filled).tolist()==[1] and np.isnan(smooth[3:7]).all()
    occ[2:]=1;assert not geo.smooth_gaze(ts,raw,occ,br)[1].any()
    # A short isolated observed point contributes zero seconds yet is preserved.
    iso,filled=geo.smooth_gaze(np.array([1000000000]),np.array([[4.,5.]]),[0],[True])
    np.testing.assert_array_equal(iso,[[4.,5.]])
    assert not filled.any() and geo.times(np.array([1000000000]),[2])[0][0]==0
    empty=dict(dt=np.array([],float),occurrence=np.array([],int),breaks=np.array([],bool))
    summary=core.summarize(np.array([],int),np.array([],bool),np.empty((0,4)),empty)
    assert summary['valid_n']==0 and summary['valid_s']==0 and np.isnan(summary['C']) and np.isnan(summary['L_T'])
    assert summary['C_reason']=='no_valid_frames' and np.isnan(summary['direct_rate'])
    only_overlap=dict(dt=np.repeat(.05,4),occurrence=np.zeros(4,int),breaks=np.array([1,0,0,0],bool))
    summary=core.summarize(np.repeat(3,4),np.ones(4,bool),np.tile([.8,0,0,.2],(4,1)),only_overlap)
    assert summary['L_O']==1 and summary['L_target']==1 and summary['D']==0 and np.isnan(summary['C']) and np.isnan(summary['C_U'])
    assert summary['C_reason']=='zero_exclusive_occupancy'
    # Exact seed/size bootstrap draws and <2-person interval behavior.
    boot=ParticipantBootstrap(42,10000)
    for n in [2,3,9]:
        v=np.linspace(.1,.9,n)
        rng=np.random.default_rng(np.random.SeedSequence([42,n]))
        weights=rng.multinomial(n,np.full(n,1/n),size=10000)
        expected=np.percentile(np.einsum('ij,j->i',weights,v)/n,[2.5,97.5])
        actual=boot.mean(v)
        np.testing.assert_allclose([actual['ci_low'],actual['ci_high']],expected,atol=0,rtol=0)
        assert actual['bootstrap_defined']==10000 and actual['bootstrap_undefined']==0
    assert np.isnan(boot.mean([1])['ci_low']) and boot.mean([1])['bootstrap_defined']==0
    common=dict(scope='real_all',scenario='main',task='A',stage='whole',source='gaze')
    trials=pd.DataFrame([dict(common,content_id='watch1',participant='observer1',L_T=.2,C=.25,valid_n=10,valid_s=1),
                         dict(common,content_id='watch2',participant='observer1',L_T=.4,C=np.nan,valid_n=10000,valid_s=1000),
                         dict(common,content_id='watch3',participant='observer2',L_T=.8,C=.5,valid_n=1,valid_s=1)])
    person,group=participant_summary(trials,boot)
    assert len(person)==2
    np.testing.assert_allclose(person.L_T,[.3,.8]);np.testing.assert_allclose(person.C,[.25,.5])
    g=group.set_index('metric')
    np.testing.assert_allclose(g.loc['L_T','mean'],.55);np.testing.assert_allclose(g.loc['C','mean'],.375)
    assert g.loc['L_T','n_trials']==3 and g.loc['C','n_trials']==2
    assert g.loc['L_T','n_participants']==2 and g.loc['C','n_participants']==2
    missing=trials.copy();missing[['L_T','C']]=np.nan;missing[['valid_n','valid_s']]=0
    person,group=participant_summary(missing,boot)
    assert (group.n_participants==0).all() and (group.n_trials==0).all() and group['mean'].isna().all()
    assert group.ci_low.isna().all() and (group.bootstrap_defined==0).all()
    return ['native_dt_edges','closed_gaze_open_annotation_boundaries','pixel_area_enumeration',
            'dwell_switch_missing_other_censoring','smooth_short_gap_occurrence_isolation',
            'NA_empty_and_zero_denominator','isolated_point_zero_exposure',
            'bootstrap_exact_seed_sequence','equal_recording_equal_observer_metric_specific_support']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'baseline_observed_regression.json');args=parser.parse_args()
    exact=core_identity_checks();toys=toy_checks()
    audit=pd.read_csv(BASE/'recording_audit.csv')
    saved=pd.read_csv(BASE/'trial_stage.csv',low_memory=False)
    # The baseline exported duplicate real-gaze time measurements under decoder
    # labels. This selects one observation copy, without reading model fields.
    saved=saved[(saved.source=='gaze')&(saved.scope=='real_all')&(saved.analysis_branch=='decoded_expectation')]
    names=['User1_1_1','User1_3_1','User1_6_1','User12_9_1']
    checks=[];rows=[];scalar_n=0;frame_n=0
    for cid in names:
        row=audit[audit.content_id==cid].iloc[0];d,info=load_raw(row)
        with np.load(BASE/'frames'/f'{cid}.npz',allow_pickle=False) as f:
            fields={'frames':'frame_number','timestamp_ns':'timestamp_ns','gaze':'gaze_xy','slots':'target_slots',
                    'z':'target_xy','count':'target_count','slotmask':'target_slotmask','left':'left_xy','right':'right_xy',
                    'dt':'dt_s','breaks':'break_before','occurrence':'occurrence','included':'included',
                    'smooth_xy':'smooth_xy','interpolated':'interpolated'}
            for new,old in fields.items():
                np.testing.assert_array_equal(d[new],f[old],err_msg=cid+':'+new)
            cache={}
            for scenario,(rt,rp,_,_) in core.SCENARIOS.items():
                xy,valid=core.support(d,scenario)
                states=geo.classify(xy,d['z'],d['left'],d['right'],rt,rp,valid)
                np.testing.assert_array_equal(valid,f['valid_real_'+scenario]);np.testing.assert_array_equal(states,f['state_gaze_'+scenario])
                if (rt,rp) not in cache:cache[rt,rp]=geo.areas(d['z'],d['left'],d['right'],d['width'],d['height'],rt,rp,d['included']&(d['count']>0)&d['gripper_ok'])
                area=cache[rt,rp]
                group=[]
                for stage,mask in core.stage_masks(d):
                    actual=core.summarize(states,valid&mask,area,d)
                    expected=saved[(saved.content_id==cid)&(saved.scenario==scenario)&(saved.stage==stage)]
                    assert len(expected)==1,(cid,scenario,stage)
                    for k,v in actual.items():scalar_close(v,expected.iloc[0][k],f'{cid}/{scenario}/{stage}/{k}');scalar_n+=1
                    ti=independent_temporal(np.where(valid&mask,states,-1),d['dt'],d['occurrence'],d['breaks'])
                    for k,v in ti.items():scalar_close(v,actual[k],f'independent {cid}/{scenario}/{stage}/{k}')
                    rows.append(dict(content_id=cid,scenario=scenario,stage=stage,**actual));group.append(actual)
                np.testing.assert_allclose(group[0]['valid_s'],sum(x['valid_s'] for x in group[1:]),rtol=1e-13)
                frame_n+=len(states)
        info.update(content_id=cid,task=row.task,frame_count=len(d['frames']),
                    source_timestamp_damage_n=int(row.npy_video_timestamp_differences),
                    drop_ring_observed_s=float(d['dt'][d['stage']=='drop ring'].sum()),
                    drop_ring_included_s=float(d['dt'][(d['stage']=='drop ring')&d['included']].sum()))
        checks.append(info);print(cid,'passed',len(d['frames']),'frames x 5 scenarios',flush=True)
    pd.DataFrame(rows).to_csv(args.out.with_name('baseline_observed_regression_rows.csv'),index=False,na_rep='NA',encoding='utf-8-sig')
    report=dict(status='passed',reference_run=str(BASE),extracted_functions_ast_identical=exact,toy_checks=toys,
                recordings=checks,scenarios=list(core.SCENARIOS),trial_stage_rows_checked=len(rows),
                raw_frame_scenario_states_checked=frame_n,scalar_measurements_checked=scalar_n,
                exact_native_dt_smooth_and_interpolated_flags=True,independent_run_list_temporal_check=True,
                whole_equals_included_stage_exposure=True,predictions_loaded=False,
                original_inputs_unmodified=all(sha(p)==h for c in checks for p,h in c['input_hashes'].items()))
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['recordings','extracted_functions_ast_identical']},indent=2))

if __name__=='__main__':main()
