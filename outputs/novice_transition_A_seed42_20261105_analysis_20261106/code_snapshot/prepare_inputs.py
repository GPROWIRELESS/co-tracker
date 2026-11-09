import os, sys, json, hashlib, shutil
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import numpy as np
REPO=Path(r'D:\GitHub_Project\co-tracker')
LOG=Path(r'C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch\logs\novice_active_transition_A_seed42_20261105')
import argparse
_parser=argparse.ArgumentParser()
_parser.add_argument('--output',type=Path,default=Path('D:\\GitHub_Project\\co-tracker\\outputs\\novice_transition_A_seed42_20261105_analysis_20261106'))
OUT=_parser.parse_args().output
sys.path.insert(0,str(REPO/'tools'))
FROZEN=Path(__file__).resolve().parent
sys.path.insert(0,str(FROZEN))
import tl_inputs as inp
import tl_geometry as geo
assert Path(inp.__file__).resolve().parent==FROZEN, "Use the coordinate-only frozen reader"
MODELS={'current_transition':0,'future500_transition':500,'future1000_transition':1000,'future1500_transition':1500}
def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v
def dump(p,v): p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
    return h.hexdigest()
def main():
    dest=OUT/'input_audit'; (dest/'native').mkdir(parents=True,exist_ok=True)
    items=json.loads((LOG/'inference_items.json').read_text())
    overrides=json.loads((LOG/'timestamp_overrides.json').read_text())
    sources=json.loads((LOG/'gaze_source_sha256.json').read_text())
    index={r['content_id']:r for r in inp.records(REPO)}
    audits=[]; recs=[]; native={}; hashes=[]
    for ci,item in enumerate(items):
        cid=item['meta']['content_id']; rec=index[cid]
        print('Read native',cid,flush=True)
        d=inp.read_record(rec,Path(r'D:\Dataset\dvss'),REPO,audit_cache_dir=dest/'reader_cache')
        full=np.load(d['audit']['npy_path'],allow_pickle=False)
        assert len(full)==len(d['frames']), 'Retained row deletion requires explicit anchor video mapping'
        assert np.array_equal(full['Frame_Number'],d['frames'])
        expected=sources[item['gaze_npy_path']]
        if sha(d['audit']['npy_path']) != expected: raise ValueError('Training gaze hash differs: '+cid)
        train_ns=np.asarray(overrides.get(item['video_path'],full['Timestamp(ns)']),dtype=np.int64)
        assert len(train_ns)==len(full) and np.all(np.diff(train_ns)>0)
        training_time=(train_ns-train_ns[0])/1e9
        delta=training_time-d['time']
        d['z'],d['count'],_=geo.target_center(d['slots'],d['width'],d['height'])
        arr={k:v for k,v in d.items() if isinstance(v,np.ndarray)}
        arr.update(width=np.asarray(d['width']),height=np.asarray(d['height']),training_time=training_time)
        np.savez_compressed(dest/'native'/f'{cid}.npz',**arr)
        audit=dict(d['audit'],training_gaze_hash_matches=True,training_clock_override=item['video_path'] in overrides,
                   training_minus_analysis_time_median_ms=np.median(delta)*1000,
                   training_minus_analysis_time_max_abs_ms=np.max(abs(delta))*1000,
                   training_minus_analysis_time_end_ms=delta[-1]*1000,
                   anchor_mapping='source zero-based row to identical full native video row; original frame IDs preserved',
                   stage_counts=dict(zip(*np.unique(d['stage'],return_counts=True))))
        audits.append(audit); native[cid]=arr
        recs.append(dict(rec,width=d['width'],height=d['height'],clip_id=ci,split='validation',native_n=len(full)))
        for k in ['npy_path','video_ts_path','dlc_path']:
            p=Path(d['audit'][k]); hashes.append(dict(path=str(p),sha256=sha(p),kind='raw_input'))
    pred_audit=[]; ref=None
    for model,h in MODELS.items():
        p=LOG/model/'validation_predictions.npz'; hashes.append(dict(path=str(p),sha256=sha(p),kind='prediction'))
        with np.load(p,allow_pickle=False) as z:
            if ref is None: ref={k:z[k].copy() for k in ['anchor','clip_id','future_label_indices','future_valid']}
            else:
                for k,v in ref.items(): assert np.array_equal(z[k],v),(model,k,'mismatch')
            folder=dest/'predictions'/model;folder.mkdir(parents=True,exist_ok=True)
            for rec in recs:
                cid=rec['content_id']; d=native[cid]; take=z['clip_id']==rec['clip_id']; a=z['anchor'][take]
                assert np.array_equal(a,np.arange(0,len(d['time']),2)),(model,cid,'stride')
                prob=z['probability'][take]; mass=prob.sum((1,2),dtype=np.float64)
                assert prob.shape==(len(a),32,40) and np.isfinite(prob).all() and prob.min()>=0
                assert np.max(abs(mass-1))<1e-5
                prob=prob.astype(np.float64)/mass[:,None,None]
                x,y=np.meshgrid(np.linspace(0,rec['width'],40),np.linspace(0,rec['height'],32))
                expectedxy=np.column_stack(((prob*x).sum((1,2)),(prob*y).sum((1,2))))
                imax=prob.reshape(len(a),-1).argmax(1);argmaxxy=np.column_stack((x.ravel()[imax],y.ravel()[imax]))
                time=d['time'][a]; target=time+h/1000
                right=np.searchsorted(d['time'],target).clip(0,len(d['time'])-1);left=(right-1).clip(0)
                ti=np.where(abs(d['time'][left]-target)<abs(d['time'][right]-target),left,right)
                tok=(target<=d['time'][-1])&(abs(d['time'][ti]-target)<=.035)
                idx=z['future_label_indices'][take]; hi=list(MODELS.values()).index(h)
                train_delta=d['time'][idx[:,hi]]-time-h/1000
                v=z['future_valid'][take,hi]
                np.savez_compressed(folder/f'{cid}.npz',anchor=a,native_index=a,time=time,target_time=target,
                    target_native_index=ti,target_time_valid=tok,probability=prob.astype(np.float32),
                    expectedxy=expectedxy,argmaxxy=argmaxxy,valid_maps=np.ones(len(a),bool),
                    future_label_indices=idx,future_valid=z['future_valid'][take],transition=z['transition'][take])
                pred_audit.append(dict(model=model,content_id=cid,horizon_ms=h,n=len(a),anchor_equal_all_models=True,
                    raw_probability_max_mass_error=float(np.max(abs(mass-1))),valid_target_time_n=int(tok.sum()),
                    training_label_actual_horizon_error_p95_ms=float(np.percentile(abs(train_delta[v])*1000,95)) if v.any() else None,
                    training_label_actual_horizon_error_max_ms=float(np.max(abs(train_delta[v])*1000)) if v.any() else None,
                    time_axis='canonical recorded video time; target_time = anchor_time + nominal horizon',
                    model_grid='inclusive endpoints x=j/39,y=i/31; pixel coordinates multiply W,H'))
        print('Prepared',model,flush=True)
    meta=dict(models=MODELS,records=recs,input_root=str(dest),log_root=str(LOG),repo=str(REPO),
              stages=['reach ring','grab ring','transfer ring','place ring'],excluded_stages=['drop ring','none',''],
              split='validation used for checkpoint selection',coordinate_frame='future_screen_coordinates_no_motion_registration',
              gaze_source='User/npy/*_gaze.npy',gaze_validity='finite normalized x/y in inclusive [0,1]',
              valid_column_used_for_selection=False,gaze_allowed_codes=None,
              max_gap_s=.1,seed=42,shared_model_anchors=True,forecast_time_axis='anchor time plus nominal horizon; saved label index timing separately audited')
    dump(dest/'metadata.json',meta);dump(dest/'native_audit.json',audits);dump(dest/'prediction_audit.json',pred_audit)
    dump(dest/'input_manifest.json',hashes)
    snap=OUT/'code_snapshot';snap.mkdir(exist_ok=True)
    for filename in ['tl_inputs.py','tl_geometry.py','tl_common.py','tl_stats.py','gaze_psm_core.py','gaze_psm_core_legacy.py','gaze_psm_preprocess_v2.py','gaze_psm_reliability_v2.py','gaze_psm_statistics_v2.py','gaze_psm_pipeline_dlc.py']:
        origin=FROZEN/filename
        if origin.resolve()!=(snap/filename).resolve():shutil.copy2(origin,snap/filename)
    for filename in ['plan.json','comparison.json','inference_items.json','timestamp_alignment_audit.json','gaze_source_sha256.json']:
        shutil.copy2(LOG/filename,dest/filename)
    dump(dest/'validation.json',dict(passed=True,records=len(recs),participants=len(set(r['participant'] for r in recs)),models=4,
                                    raw_hashes_match_training=True,anchors_identical=True,no_source_writes=True,
                                    gaze_coordinate_only=True,valid_column_used_for_selection=False,
                                    gaze_valid_rows=sum(a['gaze_valid_n'] for a in audits),
                                    gaze_coordinate_valid_rows=sum(a['gaze_coordinate_valid_n'] for a in audits)))
    print(json.dumps(dict(records=len(audits),coordinate_valid_rows=sum(a['gaze_coordinate_valid_n'] for a in audits),valid_column_used_for_selection=False)),flush=True)
if __name__=='__main__': main()
