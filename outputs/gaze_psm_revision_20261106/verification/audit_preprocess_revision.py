"""Rebuild source-compatible caches using native NPY gaze, coordinate-only validity."""
import os,sys,json,csv,hashlib,shutil
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';sys.dont_write_bytecode=True
import numpy as np
REPO=Path(r'D:\GitHub_Project\co-tracker')
DATA=Path(r'D:\Dataset\dvss\Active_Data')
WORK=Path('C:\\Users\\GT502\\Documents\\ChatGPT\\JMRR\\gaze_psm_revision_20261106_work')
OLD=REPO/'outputs/gaze_psm_revision_20261105'
GTSRC=REPO/'outputs/gaze_psm_xcorr_dlc'
MODSRC=REPO/'outputs/gaze_psm_xcorr_models_max_dlc'
GTDEST=WORK/'inputs/gt84';MODDEST=WORK/'inputs/models10'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'tools'))
from tools import gaze_psm_pipeline_dlc as pipe
from gaze_psm_preprocess_v2 import interpolate_valid,filter_valid
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write_cache(p,d):p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**d)
def relabel(d,cid,source,xy,audit):
    meta=json.loads(str(d['qc_json']));meta.update(gaze_source=source,gaze_valid_values=None,gaze_validity='finite normalized x/y within [0,1]; valid column ignored',
        gaze_source_npy=audit['npy_path'],gaze_source_npy_sha256=audit['npy_sha256'],gaze_selection_uses_valid=False)
    for side in ['l','r']:
        px=xy*np.asarray(d['resolution_'+side])
        raw=interpolate_valid(d['native_time_s'],px,d['time_s'],.1)
        d['gaze_raw_'+side]=raw;d['gaze_'+side]=pipe.smooth_valid(raw,41.,80.)
        meta[side+'_gaze_valid_fraction']=float(np.isfinite(raw).all(1).mean())
    d['qc_json']=np.asarray(json.dumps(meta));d['fingerprint']=np.asarray(hashlib.sha256((str(d.get('fingerprint',''))+audit['npy_sha256']+source+'coordinate_only_20261106').encode()).hexdigest())
    return d
def source_shell(src,dst,model=False):
    dst.mkdir(parents=True,exist_ok=True)
    for filename in ['quality.csv','prediction_manifest.csv','validation.json']:
        if (src/filename).exists():shutil.copy2(src/filename,dst/filename)
    config=json.loads((src/'config.json').read_text(encoding='utf-8-sig'))
    config.update(output=str(dst),data_root=str(DATA),gaze_source='native_npy',gaze_x_column='gaze_x',gaze_y_column='gaze_y',gaze_valid_values=None,
        gaze_validity='finite normalized coordinates in inclusive [0,1]; valid is diagnostic only',gaze_selection_uses_valid=False,
        source_policy_revision='2026-11-06 explicit user correction',source_cache_parent=str(src),model_gaze='probability expectation' if model else None)
    dump(dst/'config.json',config)
    paths=[src/'results.json'] if not model else [src/s/'results.json' for s in ['gt','all_ABCD','single_task','leave_out','spd_finetune','spd_zero_shot']]
    for p in paths:
        dest=dst/p.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True)
        obj=json.loads(p.read_text(encoding='utf-8-sig'))
        if isinstance(obj,dict):obj['_historical_reference_notice']='Old raw-CSV-GT run, retained ONLY for changed-source regression auditing; not current coordinate-only results.'
        dump(dest,obj)
def rebuild_inputs():
    source_shell(GTSRC,GTDEST);source_shell(MODSRC,MODDEST,True)
    recs=json.loads((OLD/'gt84/run_manifest.json').read_text(encoding='utf-8'))['records']
    modelrecs=json.loads((OLD/'models10/run_manifest.json').read_text(encoding='utf-8'))['parameters']['records']
    all_audit=[];source_files=[];gtc={}
    for i,rec in enumerate(recs):
        cid=rec['content_id'];base=DATA/cid.split('_')[0];npyp=base/'npy'/(cid+'_gaze.npy');clock=base/'csv'/(cid+'_video_ts.csv')
        d=load(GTSRC/'cache'/(cid+'.npz'));npy=np.load(npyp,allow_pickle=False)
        frames,ns,deleted=pipe.timeline(base,cid)
        assert np.array_equal(frames,d['frame_number']),cid
        assert np.array_equal((ns-ns[0])/1e9,d['native_time_s']),cid
        ids=np.asarray(npy['Frame_Number'],dtype=np.int64);assert len(np.unique(ids))==len(ids)
        mapping={int(f):j for j,f in enumerate(ids)};ix=np.array([mapping[int(f)] for f in frames]);a=npy[ix]
        xy=np.column_stack((a['gaze_x'],a['gaze_y'])).astype(np.float64)
        valid=np.isfinite(xy).all(1)&(xy>=0).all(1)&(xy<=1).all(1)
        codes=a['valid'];included_by_code={str(int(c)):int(np.sum(valid&(codes==c))) for c in np.unique(codes)}
        raw_stage=np.array([pipe.normalized_stage(str(s)) for s in a['stage' if 'stage' in a.dtype.names else 'event']])
        oldstage=d['native_stage'].astype(str)
        assert np.array_equal(raw_stage,oldstage),(cid,'native NPY stage differs from previous cache',np.unique(raw_stage),np.unique(oldstage))
        points,likelihood=pipe.read_dlc(pipe.dlc_path(base,cid),len(frames))
        for side in ['l','r']:
            np.testing.assert_allclose(points[side],d['native_psm_'+side],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_allclose(likelihood[side],d['likelihood_'+side],rtol=0,atol=0,equal_nan=True)
        audit=dict(content_id=cid,user=cid.split('_')[0],task=rec['task'],npy_path=str(npyp),npy_sha256=sha(npyp),
            video_ts_path=str(clock),video_ts_sha256=sha(clock),dlc_path=str(pipe.dlc_path(base,cid)),dlc_sha256=sha(pipe.dlc_path(base,cid)),
            original_cache_path=str(GTSRC/'cache'/(cid+'.npz')),original_cache_sha256=sha(GTSRC/'cache'/(cid+'.npz')),
            native_rows=len(a),coordinate_valid_rows=int(valid.sum()),coordinate_invalid_rows=int((~valid).sum()),accepted_rows_by_stored_valid_code=included_by_code,
            valid_column_used_for_selection=False,timestamps_source='video_ts.csv matched by original Frame_Number',npy_timestamp_nonincreasing=int(np.sum(np.diff(a['Timestamp(ns)'])<=0)),
            frame_clock_stage_psm_equal_previous=True,grid_hz=41.,max_interpolation_gap_s=.1,
            old_grid_valid_n=int(np.isfinite(d['gaze_raw_l']).all(1).sum()))
        xy[~valid]=np.nan
        d=relabel(d,cid,'native_npy_coordinate_only',xy,audit)
        d['native_gaze_norm']=xy;d['native_gaze_coordinate_valid']=valid;d['native_gaze_valid_code_diagnostic']=codes
        audit['new_grid_valid_n']=int(np.isfinite(d['gaze_raw_l']).all(1).sum())
        write_cache(GTDEST/'cache'/(cid+'.npz'),d);gtc[cid]=d;all_audit.append(audit)
        for key in ['npy','video_ts','dlc']:source_files.append(dict(path=audit[key+'_path'],sha256=audit[key+'_sha256'],role=key))
        if i%10==0:print('Prepared GT',i+1,'/',len(recs),flush=True)
    auditby={a['content_id']:a for a in all_audit}
    models=['all_ABCD','single_task','leave_out','spd_finetune','spd_zero_shot']
    with (MODSRC/'prediction_manifest.csv').open(encoding='utf-8-sig',newline='') as f:predrows=list(csv.DictReader(f))
    modelaudit=[]
    for cid in modelrecs:
        write_cache(MODDEST/'cache/gt'/(cid+'.npz'),gtc[cid])
        for model in models:
            d=load(MODSRC/'cache'/model/(cid+'.npz'))
            for key in ['frame_number','native_time_s','time_s','stage','native_stage','occurrence','native_occurrence']:
                assert np.array_equal(d[key],gtc[cid][key]),(cid,model,key)
            row=next(r for r in predrows if r['content_id']==cid and r['model']==model)
            p=Path(row['prediction_path']);assert sha(p)==row['prediction_sha256']
            xy=np.array(d['model_native_mean_norm'],copy=True)
            with np.load(p,allow_pickle=False) as z:
                assert np.array_equal(z['frame_number'],d['frame_number'])
                assert np.array_equal((z['timestamp_ns']-z['timestamp_ns'][0])/1e9,d['native_time_s'])
                prob=z['probability'].astype(np.float64);prob/=prob.sum((1,2))[:,None,None]
                mean=np.column_stack(((prob*np.linspace(0,1,prob.shape[2])[None,None,:]).sum((1,2)),(prob*np.linspace(0,1,prob.shape[1])[None,:,None]).sum((1,2))))
                error=float(np.max(abs(mean-xy)))
                assert error<1e-6,(cid,model,error)
            previous={side:d['gaze_raw_'+side].copy() for side in ['l','r']}
            d=relabel(d,cid,'model_probability_expectation_no_GT_prefilter',xy,auditby[cid])
            write_cache(MODDEST/'cache'/model/(cid+'.npz'),d)
            modelaudit.append(dict(content_id=cid,model=model,prediction_path=str(p),prediction_sha256=row['prediction_sha256'],
                expected_coordinate_max_error=error,native_prediction_unchanged=True,
                previous_raw_cache_identical=all(np.allclose(previous[s],d['gaze_raw_'+s],rtol=0,atol=0,equal_nan=True) for s in ['l','r']),
                old_GT_filter_not_retained=True))
        print('Prepared five models',cid,flush=True)
    policy=dict(gaze_source='User*/npy/*_gaze.npy',fields=['gaze_x','gaze_y'],validity='finite AND normalized coordinates in inclusive [0,1]',
                valid_column_used_for_selection=False,clock='precise original-frame video_ts.csv',records=all_audit,models=modelaudit,source_files=source_files,
                original_model_predictions_unchanged=True,GT84_models10_GT_cache_equal=True)
    dump(GTDEST/'input_policy_audit.json',policy);dump(MODDEST/'input_policy_audit.json',policy);dump(WORK/'input_policy_audit.json',policy)
    # Refresh quality rows describing the new GT cache; legacy results stay explicitly historical.
    with (GTSRC/'quality.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);qs=list(reader);names=reader.fieldnames
    for r in qs:
        cid=r.get('content_id');a=auditby.get(cid)
        if a:
            if 'gaze_source' in r:r['gaze_source']='native_npy_coordinate_only'
            if 'accepted_gaze_samples' in r:r['accepted_gaze_samples']=a['coordinate_valid_rows']
            for side in ['l','r']:
                if side+'_gaze_valid_fraction' in r:r[side+'_gaze_valid_fraction']=a['new_grid_valid_n']/len(gtc[cid]['time_s'])
    with (GTDEST/'quality.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(qs)
    dump(WORK/'input_ready.json',dict(complete=True,gt_records=len(recs),model_records=len(modelrecs),prediction_files=len(modelaudit),
        coordinate_valid_rows=sum(a['coordinate_valid_rows'] for a in all_audit),accepted_stored_valid0_rows=sum(a['accepted_rows_by_stored_valid_code'].get('0',0) for a in all_audit),
        accepted_stored_valid1_rows=sum(a['accepted_rows_by_stored_valid_code'].get('1',0) for a in all_audit)))
    print('INPUTS READY',flush=True)



def independent_grid(t,xy,grid):
    """Separate piecewise linear reconstruction for monotonic video-frame times."""
    good=np.isfinite(xy).all(axis=1)
    t=t[good];xy=xy[good]
    out=np.full((len(grid),2),np.nan)
    if len(t)<2:return out
    assert np.all(np.diff(t)>0)
    # Interpolate each grid point between adjacent accepted native observations.
    left=np.searchsorted(t,grid,side='right')-1
    interior=(left>=0)&(left<len(t)-1)
    idx=np.flatnonzero(interior);l=left[idx];r=l+1
    ok=(t[r]-t[l])<=.1
    idx=idx[ok];l=l[ok];r=r[ok]
    w=(grid[idx]-t[l])/(t[r]-t[l])
    out[idx]=xy[l]+(xy[r]-xy[l])*w[:,None]
    # Production accepts an exact native point within one nanosecond, even at a gap.
    right=np.searchsorted(t,grid,side='left').clip(0,len(t)-1)
    exact=np.abs(grid-t[right])<=1e-9
    out[exact]=xy[right[exact]]
    return out

def audit_inputs():
    import argparse,importlib.util
    from decimal import Decimal
    parser=argparse.ArgumentParser()
    parser.add_argument('--rebuild-inputs',action='store_true')
    args=parser.parse_args()
    if args.rebuild_inputs:rebuild_inputs()
    dest=Path(__file__).parent
    stage=dest.parent
    sys.path.insert(0,str(stage/'gt84/code_snapshot'))
    from gaze_psm_preprocess_revision import prepare_signals
    spec=importlib.util.spec_from_file_location('historical_preprocess',REPO/'tools/gaze_psm_preprocess_v2.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    policy=json.loads((WORK/'input_policy_audit.json').read_text(encoding='utf-8'))
    records=[];max_error=0.;array_checks=0
    for entry in policy['records']:
        cid=entry['content_id'];native=np.load(entry['npy_path'],allow_pickle=False)
        with open(entry['video_ts_path'],encoding='utf-8-sig',newline='') as f:
            clock=[r for r in csv.DictReader(f) if str(r.get('IsDelete','0')).strip() in ('','0','0.0')]
        frames=np.array([int(r['Frame_Number']) for r in clock],dtype=np.int64)
        timestamps=np.array([int(Decimal(r['Timestamp(ns)'])) for r in clock],dtype=np.int64)
        times=(timestamps-timestamps[0])/1e9
        assert len(np.unique(native['Frame_Number']))==len(native)
        order=np.argsort(native['Frame_Number'])
        ix=np.searchsorted(native['Frame_Number'][order],frames)
        selected=native[order[ix]]
        np.testing.assert_array_equal(selected['Frame_Number'],frames)
        xy=np.array([selected['gaze_x'],selected['gaze_y']],dtype=np.float64).T
        valid=np.isfinite(xy).all(axis=1)&np.all((xy>=0)&(xy<=1),axis=1)
        xy[~valid]=np.nan
        cache=load(GTDEST/'cache'/(cid+'.npz'))
        for key,a in [('frame_number',frames),('native_time_s',times),('native_gaze_coordinate_valid',valid),('native_gaze_norm',xy)]:
            np.testing.assert_array_equal(cache[key],a);array_checks+=1
        errors=[]
        for side in ['l','r']:
            expected=independent_grid(times,xy*np.asarray(cache['resolution_'+side]),cache['time_s'])
            got=cache['gaze_raw_'+side]
            np.testing.assert_array_equal(np.isfinite(expected),np.isfinite(got))
            np.testing.assert_allclose(expected,got,rtol=0,atol=1e-9,equal_nan=True)
            errors.append(float(np.nanmax(np.abs(expected-got))))
            array_checks+=2
        revised=prepare_signals({'gt':cache},{})
        old=reference.prepare_signals({'gt':cache},{})
        for signal in ['position','velocity']:
            for side in ['l','r']:
                a,b=revised['signals'][signal][side],old['signals'][signal][side]
                for key in ['psm','time_s','stage','occurrence','gaze_valid','psm_valid']:
                    np.testing.assert_array_equal(a[key],b[key]);array_checks+=1
                np.testing.assert_array_equal(a['gazes']['gt'],b['gazes']['gt']);array_checks+=1
                assert not revised['diagnostics'][side]['quality_rejected'].any()
                assert not revised['diagnostics'][side]['grid_filled_quality_hole'].any()
        accepted={str(int(c)):int(np.sum(valid&(selected['valid']==c))) for c in np.unique(selected['valid'])}
        assert accepted==entry['accepted_rows_by_stored_valid_code']
        assert int(valid.sum())==entry['coordinate_valid_rows']
        records.append(dict(content_id=cid,npy_path=entry['npy_path'],native_rows=len(frames),
            accepted_rows=int(valid.sum()),accepted_rows_by_stored_valid_code=accepted,
            native_frame_clock_xy_mask_equal=True,independent_grid_equal=True,max_abs_grid_error=max(errors),
            old_new_preprocess_equal_on_same_new_input=True,no_quality_rejection=True))
        max_error=max(max_error,*errors)
    source_checks=[]
    for entry in policy['source_files']:
        equal=sha(Path(entry['path']))==entry['sha256']
        source_checks.append(dict(path=entry['path'],role=entry['role'],unchanged=equal))
        assert equal,entry['path']
    # Diagnostic labels are deliberately irrelevant: all accepted valid=1 rows remain.
    code1=sum(r['accepted_rows_by_stored_valid_code'].get('1',0) for r in records)
    assert code1==4086
    model_rebuilt_checks=[]
    for r in policy['models']:
        d=load(MODDEST/'cache'/r['model']/(r['content_id']+'.npz'))
        valid=np.isfinite(d['model_native_mean_norm']).all(1)&np.all((d['model_native_mean_norm']>=0)&(d['model_native_mean_norm']<=1),1)
        assert valid.all()
        for side in ['l','r']:
            expected=independent_grid(d['native_time_s'],d['model_native_mean_norm']*d['resolution_'+side],d['time_s'])
            np.testing.assert_allclose(expected,d['gaze_raw_'+side],rtol=0,atol=1e-9,equal_nan=True)
        assert sha(Path(r['prediction_path']))==r['prediction_sha256']
        model_rebuilt_checks.append(dict(content_id=r['content_id'],model=r['model'],prediction_unchanged=True,grid_independently_verified=True))
    result=dict(status='passed',n_records=len(records),coordinate_only_validity=True,
        valid_column_used_for_selection=False,gaze_source='User*/npy/*_gaze.npy',
        accepted_native_rows=sum(r['accepted_rows'] for r in records),accepted_stored_valid1_rows=code1,
        numerical_array_checks=array_checks,max_abs_grid_error=max_error,
        no_quality_rejection=True,records=records,source_file_checks=source_checks,model_checks=model_rebuilt_checks,
        comparison='independent native NPY/frame-clock/grid reconstruction, and original preprocessing on the same new input',
        provenance_clarifications={
            'model_qc_gaze_source_npy':'This inherited field identifies the same-record GT reference NPY. Model gaze itself comes from original prediction NPZ probability expectations, verified above.',
            'input_config_json':'Contains historical source-cache generation settings. It is provenance/legacy-reference metadata only and does not configure this rerun. Current numerical parameters are in each output branch run_manifest.json.',
            'source_results_json':'Historical raw-CSV-GT result used only for source-change diagnostics; never used as current result arrays.'})
    dump(dest/'preprocess_cache_audit.json',result)
    alignment=[]
    for path in sorted((MODDEST/'cache/gt').glob('*.npz')):
        a,b=load(path),load(GTDEST/'cache'/path.name)
        row={'content_id':path.stem}
        for key in a:
            row[key]=bool(np.array_equal(a[key],b[key],equal_nan=True)) if np.issubdtype(a[key].dtype,np.number) else bool(np.array_equal(a[key],b[key]))
            assert row[key],(path.name,key)
        alignment.append(row)
    dump(dest/'gt_model_input_alignment.json',alignment)
    print(json.dumps({k:result[k] for k in ['status','n_records','accepted_native_rows','accepted_stored_valid1_rows','numerical_array_checks','max_abs_grid_error']}),flush=True)

if __name__=='__main__':audit_inputs()
