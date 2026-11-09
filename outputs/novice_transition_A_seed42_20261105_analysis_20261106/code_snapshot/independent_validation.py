"""Read-only independent checks of prepared model/GT/native inputs and result tables."""
import os,sys,json,hashlib
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import cv2
ROOT=Path('D:\\GitHub_Project\\co-tracker\\outputs\\novice_transition_A_seed42_20261105_analysis_20261106')
INPUT=ROOT/'input_audit'
REPO=Path(r'D:\GitHub_Project\co-tracker')
LOG=Path(r'C:\Users\GT502\Desktop\JMRR2027\DVSS_Pytorch\logs\novice_active_transition_A_seed42_20261105')
sys.path.insert(0,str(REPO/'tools'))
import tl_geometry as geo
from gaze_psm_core import analyze_curve
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
    meta=json.loads((INPUT/'metadata.json').read_text(encoding='utf-8'))
    audit=json.loads((INPUT/'native_audit.json').read_text(encoding='utf-8'))
    a_by_id={a['content_id']:a for a in audit}
    items=json.loads((LOG/'inference_items.json').read_text())
    saved_hashes=json.loads((LOG/'gaze_source_sha256.json').read_text())
    overrides=json.loads((LOG/'timestamp_overrides.json').read_text())
    checks=[];details=[];checks.append({'name':'analysis_records_match_inference_items','passed':[r['content_id'] for r in meta['records']]==[i['meta']['content_id'] for i in items]})
    for ci,r in enumerate(meta['records']):
        cid=r['content_id']; a=a_by_id[cid]; d=load(INPUT/'native'/f'{cid}.npz')
        raw=np.load(a['npy_path'],allow_pickle=False)
        assert sha(a['npy_path'])==saved_hashes[items[ci]['gaze_npy_path']],cid
        assert np.array_equal(d['frames'],raw['Frame_Number'])
        assert len(d['time'])==len(raw) and np.all(np.diff(d['time'])>0)
        assert np.array_equal(d['timestamp_ns']-d['timestamp_ns'][0],np.rint(d['time']*1e9).astype(np.int64))
        gt=np.column_stack((raw['gaze_x'],raw['gaze_y'])).astype(float)
        gt_ok=np.isfinite(gt).all(1)&(gt>=0).all(1)&(gt<=1).all(1)
        np.testing.assert_array_equal(d['gaze_valid'],gt_ok)
        np.testing.assert_array_equal(d['gaze_coordinate_valid'],gt_ok)
        assert a['valid_column_used_for_selection'] is False
        assert a['gaze_valid_n']==a['gaze_coordinate_valid_n']==int(gt_ok.sum())
        np.testing.assert_allclose(d['gaze'][gt_ok],gt[gt_ok]*[r['width'],r['height']],atol=1e-7,rtol=0)
        assert np.isnan(d['gaze'][~gt_ok]).all()
        cap=cv2.VideoCapture(a['video_path']); video_n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        assert video_n==len(raw),(cid,video_n,len(raw))
        t_train=np.asarray(overrides.get(items[ci]['video_path'],raw['Timestamp(ns)']),dtype=np.int64)
        np.testing.assert_allclose(d['training_time'],(t_train-t_train[0])/1e9,atol=0,rtol=0)
        checks.append({'name':'native_alignment_'+cid,'passed':True})
        max_target_error=0; prediction_n=None; horizon_errors=[]
        for model,h in meta['models'].items():
            p=load(INPUT/'predictions'/model/f'{cid}.npz')
            expected_anchor=np.arange(0,len(raw),2)
            np.testing.assert_array_equal(p['anchor'],expected_anchor)
            np.testing.assert_array_equal(p['native_index'],expected_anchor)
            np.testing.assert_allclose(p['time'],d['time'][expected_anchor],atol=0,rtol=0)
            np.testing.assert_allclose(p['target_time'],p['time']+h/1000,atol=0,rtol=0)
            q=p['probability'].astype(float); np.testing.assert_allclose(q.sum((1,2)),1,atol=1e-7,rtol=0)
            assert np.isfinite(q).all() and (q>=0).all()
            xx=np.linspace(0,r['width'],40);yy=np.linspace(0,r['height'],32)
            # Check expectation from exported normalized heatmap with float32 storage tolerance.
            exp=np.column_stack(((q*xx[None,None,:]).sum((1,2)),(q*yy[None,:,None]).sum((1,2))))
            np.testing.assert_allclose(exp,p['expectedxy'],atol=1e-4,rtol=0)
            imax=q.reshape(len(q),-1).argmax(1)
            np.testing.assert_allclose(p['argmaxxy'],np.column_stack((xx[imax%40],yy[imax//40])),atol=0,rtol=0)
            ti=p['target_native_index'];tt=p['target_time'];good=p['target_time_valid']
            assert (tt[good]<=d['time'][-1]).all() and (abs(tt[good]-d['time'][ti[good]])<=.035).all()
            assert not good[tt>d['time'][-1]].any()
            # Independent nearest-neighbor check, without using the producer's searchsorted formula.
            for j in np.linspace(0,len(tt)-1,19).astype(int):
                nearest=int(np.argmin(abs(d['time']-tt[j])))
                assert abs(d['time'][ti[j]]-tt[j])<=abs(d['time'][nearest]-tt[j])+1e-10
            max_target_error=max(max_target_error,float(np.max(abs(tt[good]-d['time'][ti[good]]))*1000))
            hi=list(meta['models'].values()).index(h)
            original_target=p['future_label_indices'][:,hi]
            original_valid=p['future_valid'][:,hi]
            local_error=(d['time'][original_target]-p['time']-h/1000)[original_valid]*1000
            horizon_errors.append(dict(horizon_ms=h,n=int(original_valid.sum()),
                signed_percentiles_ms=dict(zip(['min','p01','median','p95','p99','max'],np.percentile(local_error,[0,1,50,95,99,100]).tolist()))))
            prediction_n=len(p['anchor'])
            checks.append({'name':'prediction_alignment_'+cid+'_'+model,'passed':True})
        details.append(dict(content_id=cid,native_n=len(raw),video_n=video_n,prediction_n=prediction_n,
          original_first_frame=int(d['frames'][0]),original_last_frame=int(d['frames'][-1]),
          training_clock_max_abs_difference_ms=float(np.max(abs(d['training_time']-d['time']))*1000),
          training_clock_signed_percentiles_ms=dict(zip(['min','p01','median','p95','p99','max'],np.percentile((d['training_time']-d['time'])*1000,[0,1,50,95,99,100]).tolist())),
          nearest_target_max_abs_error_ms=max_target_error,saved_label_actual_horizon_errors=horizon_errors))
    # Independent sign test using nonperiodic signal delayed by exactly 0.3 seconds.
    rng=np.random.default_rng(100);g=rng.normal(size=(2000,2))*10;delay=12;p=np.full_like(g,np.nan);p[delay:]=g[:-delay]
    curve=analyze_curve(g,p,40,max_lag=1,min_overlap=2,min_gaze_rms=0,min_psm_rms=0,compute_spatial=False)
    lag=float(curve['lag_s'][np.nanargmax(curve['r'])]);assert abs(lag-.3)<1e-10
    checks.append({'name':'positive_lag_means_gaze_leads_psm','passed':True,'known_delay_s':.3,'estimated_lag_s':lag})
    geo.probability_self_test()
    checks.append({'name':'geometry_exact_probability_self_test','passed':True})
    # A separate one-cell geometry brute force test on a small image.
    W,H=40,32;prob=rng.uniform(size=(1,4,5));prob/=prob.sum()
    z=np.array([[17.,15.]]);left=np.array([[9.,15.]]);right=np.array([[30.,20.]])
    integrated=geo.map_region_probabilities(prob,z,left,right,W,H,7,6,np.array([True]))[0]
    xb,yb,nx,ny,_=geo._grid_cells(W,H,4,5)
    x,y=np.meshgrid(np.arange(W)+.5,np.arange(H)+.5);pix=np.stack((x,y),axis=-1)
    isin_t=np.sum((pix-z[0])**2,axis=-1)<=49
    isin_p=(np.sum((pix-left[0])**2,axis=-1)<=36)|(np.sum((pix-right[0])**2,axis=-1)<=36)
    states=isin_t.astype(int)+2*isin_p.astype(int)
    density=prob[0]/(ny[:,None]*nx[None,:]);weights=density[yb[:,None],xb[None,:]]
    brute=np.array([weights[states==i].sum() for i in range(4)])
    np.testing.assert_allclose(integrated,brute,atol=1e-12,rtol=0)
    checks.append({'name':'independent_heatmap_disk_integration','passed':True,'max_abs_error':float(np.max(abs(integrated-brute)))})
    result={'passed':all(c['passed'] for c in checks),'scope':'prepared inputs and mathematical kernels; result table checks added separately',
            'checks':checks,'record_details':details,'assumptions':['video source frame index equals exported raw gaze row index, verified by count and frame ordering','target-time uses nominal horizon on canonical video clock; finite endpoint support enforced','validation split participated in checkpoint selection']}
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    (out/'independent_input_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'record_details':details},indent=2))
if __name__=='__main__':main()
