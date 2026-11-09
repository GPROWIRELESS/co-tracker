"""Observed-gaze-only v3 runner with explicit input acceptance gates.

The authoritative JSON determines annotations and subjects. Native frame joins
are retained as evidence even when acquisition/playback mapping is unresolved.
Unaccepted records never produce occupancy or temporal estimates.
"""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

import tl_observed_geometry as geo
from tl_observed_core import STAGES, SCENARIOS, prepare_arrays, support, summarize, in_image
from tl_observed_bootstrap import ParticipantBootstrap

FIELDS = ('gaze_npy_path','source_video_path','stage_path','target_locking_label_path','psm_dlc_csv_path')
METRICS = ['L_T','L_P','L_O','L_N','L_target','L_gripper','D','C','B','area_T','area_P','area_O','area_N',
           'C_U','C_excess','E_target','E_gripper','B_excess','direct_rate','via_rate','dwell_T_ms','dwell_P_ms','dwell_O_ms']
ALIASES = {'rr':'reach ring','gr':'grab ring','tr':'transfer ring','pr':'place ring',
           'wrap':'wrap around psm','cinch':'pull through and cinch','thread':'threading string through tower'}

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def save(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def csvout(path,rows,columns=None):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    df.to_csv(path,index=False,encoding='utf-8-sig',na_rep='NA')

def resolve(root,value):
    if value is None or str(value).strip().lower() in ('','nan','null','none'): return None
    p=Path(value)
    return p if p.is_absolute() else root/p

def timeline(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: raw=list(csv.DictReader(f))
    kept=[r for r in raw if r.get('IsDelete','0').strip() in ('','0','0.0')]
    frame=np.array([int(Decimal(r['Frame_Number'])) for r in kept],dtype=np.int64)
    ns=np.array([int(Decimal(r['Timestamp(ns)'])) for r in kept],dtype=np.int64)
    if len(set(frame))!=len(frame) or np.any(np.diff(frame)<=0): raise ValueError('duplicate_or_unsorted_clock_frames')
    return frame,ns

def join_rows(array,frames,role):
    ids=np.asarray(array['Frame_Number'],dtype=np.int64)
    if len(set(ids))!=len(ids): raise ValueError('duplicate_'+role+'_Frame_Number')
    index={int(v):i for i,v in enumerate(ids)}
    rows=np.array([index.get(int(f),-1) for f in frames],dtype=np.int64)
    if (rows<0).any(): raise ValueError('missing_'+role+'_Frame_Number')
    return rows

def dlc(path,expected):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rd=csv.reader(f);headers=[next(rd) for _ in range(3)];rows=list(rd)
    data=np.asarray(rows,dtype=float)
    if len(data)!=expected or not np.array_equal(data[:,0],np.arange(expected)):
        raise ValueError('DLC_local_index_not_verified_against_retained_source_clock')
    result={}
    for side,label in [('left','left_psm'),('right','right_psm')]:
        cols=[]
        for c in ['x','y','likelihood']:
            hit=[i for i,(a,b) in enumerate(zip(headers[1],headers[2])) if a==label and b==c]
            if len(hit)!=1: raise ValueError('DLC_schema_'+side+'_'+c)
            cols.append(hit[0])
        result[side]=data[:,cols[:2]];result[side+'_likelihood']=data[:,cols[2]]
    return result

def load_native(key,item,root,detail):
    paths={k:resolve(root,item.get(k)) for k in FIELDS}
    if not all(p is not None and p.is_file() for p in paths.values()): raise ValueError('missing_authoritative_input')
    a=np.load(paths['gaze_npy_path'],allow_pickle=False)
    s=np.load(paths['stage_path'],allow_pickle=False)
    target=np.load(paths['target_locking_label_path'],allow_pickle=False)
    frames,ns=timeline(detail['passive_clock_path'])
    sf,sn=timeline(detail['source_clock_path'])
    gi=join_rows(a,frames,'passive_gaze');si=join_rows(s,frames,'stage');ti=join_rows(target,frames,'target')
    di=join_rows(np.rec.fromarrays([sf],names=['Frame_Number']),frames,'source_clock')
    cap=cv2.VideoCapture(str(paths['source_video_path']))
    if not cap.isOpened(): raise ValueError('source_video_cannot_open')
    W,H,n=int(cap.get(3)),int(cap.get(4)),int(cap.get(7));cap.release()
    if W<=0 or H<=0: raise ValueError('source_video_invalid_dimensions')
    if n!=len(sf): raise ValueError('source_video_frame_count_mismatch')
    points=dlc(paths['psm_dlc_csv_path'],len(sf))
    norm=np.column_stack((a['gaze_x'][gi],a['gaze_y'][gi])).astype(float)
    good=np.isfinite(norm).all(1)&(norm>=0).all(1)&(norm<=1).all(1)
    xy=norm*[W,H]
    gaze=np.where(good[:,None],xy,np.nan)
    stage_field=next((name for name in ['stage','event'] if name in s.dtype.names),None)
    if stage_field is None: raise ValueError('missing_stage_event_field')
    raw=s[stage_field][si].astype(str)
    normalized=[' '.join(v.strip().lower().split()) for v in raw]
    stage=np.array([ALIASES.get(v,v) for v in normalized])
    occurrence=np.r_[0,np.cumsum(stage[1:]!=stage[:-1])].astype(np.int32)
    slots=np.full((len(frames),5,2),np.nan)
    for slot in range(1,6):
        cols=[f'point_{slot}_x',f'point_{slot}_y']
        if (cols[0] in target.dtype.names)!=(cols[1] in target.dtype.names): raise ValueError('incomplete_target_slot')
        if cols[0] in target.dtype.names: slots[:,slot-1]=np.column_stack([target[c][ti] for c in cols])
    mode=target['point_label_mode'][ti].astype(str) if 'point_label_mode' in target.dtype.names else np.full(len(frames),'unavailable')
    idfield=next((x for x in ['target_id','target_identity','point_1_id'] if x in target.dtype.names),None)
    identity_change=np.zeros(len(frames),bool)
    if idfield:
        values=target[idfield][ti];identity_change[1:]=values[1:]!=values[:-1]
    d=dict(task=item['meta']['task_name'],frames=frames,timestamp_ns=ns,gaze=gaze,gaze_unmasked=gaze.copy(),
           slots=slots,stage=stage,occurrence=occurrence,identity_change=identity_change,
           left=points['left'][di],right=points['right'][di],width=W,height=H,
           label_mode=mode,coordinate_valid=good,gaze_normalized=norm,
           gaze_row=gi,stage_row=si,target_row=ti,dlc_row=di,source_ns=sn[di],
           likelihood=np.column_stack((points['left_likelihood'][di],points['right_likelihood'][di])),
           target_identity_field=idfield or 'unavailable',video=str(paths['source_video_path']))
    # This verifies stored keys, not physical observer playback time.
    if not np.array_equal(ns,d['source_ns']): raise ValueError('passive_source_native_clock_disagreement')
    return d

def metadata(key,item):
    m=item['meta']; rid=Path(key).name+'__'+hashlib.sha256(key.encode()).hexdigest()[:12]
    return dict(analysis_branch='observed_gaze',source='gaze',scope='real_all',
                clip_key=key,recording_id=rid,content_id=key,participant=f"User{m['observer']}",
                task=m['task_name'],observer=m['observer'],observer_skill=m.get('observer_skill'),
                user=m.get('user'),source_content_id=m.get('content_id'),src_skill=m.get('src_skill'),
                seq=m.get('seq'),meta_task=m.get('task'),attempt=m.get('attempt'),split=m.get('split'))

def frame_export(out,meta,d,accepted,reason):
    z,count,bits=geo.target_center(d['slots'],d['width'],d['height'])
    n=len(d['frames'])
    marks=dict(frame_number=d['frames'],mapped_source_timestamp_ns=d['timestamp_ns'],
        gaze_npy_row=d['gaze_row'],stage_npy_row=d['stage_row'],target_npy_row=d['target_row'],
        source_video_local_index=d['dlc_row'],dlc_local_index=d['dlc_row'],
        gaze_normalized=d['gaze_normalized'],gaze_unmasked=d['gaze_unmasked'],gaze_xy=d['gaze'],
        gaze_coordinate_valid=d['coordinate_valid'],target_slots=d['slots'],target_xy=z,target_count=count,
        target_slotmask=bits,label_mode=d['label_mode'],left_xy=d['left'],right_xy=d['right'],
        likelihood=d['likelihood'],stage=d['stage'],occurrence=d['occurrence'],identity_change=d['identity_change'],
        resolution=np.array([d['width'],d['height']]),analysis_accepted=np.array(accepted),
        exclusion_reason=np.asarray(reason),clip_key=np.asarray(meta['clip_key']),
        source_gaze_used=np.asarray(False))
    if accepted:
        marks.update(timestamp_ns=d['timestamp_ns'],dt_s=d['dt'],break_before=d['breaks'],smooth_xy=d['smooth_xy'],
                     interpolated=d['interpolated'],included=d['included'])
    else:
        marks.update(dt_s=np.full(n,np.nan),break_before=np.ones(n,bool),included=np.zeros(n,bool))
    return marks

def write_overlay(out,meta,d,reason):
    # A native-frame overlay checks spatial geometry, not hardware display latency.
    good=d['coordinate_valid'] & (geo.target_center(d['slots'],d['width'],d['height'])[1]>0)
    ids=np.flatnonzero(good)
    if not len(ids): return None
    i=ids[len(ids)//2]
    cap=cv2.VideoCapture(d['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(d['dlc_row'][i]));ok,img=cap.read();cap.release()
    if not ok:return None
    z,_,_=geo.target_center(d['slots'][i:i+1],d['width'],d['height'])
    for point,color,radius in [(z[0],(0,220,255),50),(d['left'][i],(0,200,0),50),(d['right'][i],(0,200,0),50),(d['gaze'][i],(255,170,0),5)]:
        if np.isfinite(point).all():cv2.circle(img,tuple(np.rint(point).astype(int)),radius,color,2)
    panel=np.zeros((90,img.shape[1],3),dtype=np.uint8)
    title='AUDIT ONLY - record mapping unaccepted' if reason else 'Observed gaze - verified native frame mapping; display latency not calibrated'
    lines=[title,meta['recording_id'],
           f"original frame {d['frames'][i]}, source local {d['dlc_row'][i]} | yellow target mean; green wrists; blue stored gaze"]
    for j,line in enumerate(lines):cv2.putText(panel,line,(12,24+26*j),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    path=out/'overlays'/(meta['recording_id']+'_observed_gaze.jpg')
    cv2.imwrite(str(path),np.vstack((panel,img)))
    return str(path.relative_to(out))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mapping',required=True);p.add_argument('--audit-dir',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for folder in ['frames','overlays','region_review','figures','code_snapshot','validation_code','logs']: (out/folder).mkdir(exist_ok=True)
    doc=json.loads(Path(args.mapping).read_text(encoding='utf-8-sig'));root=Path(doc['data_root']);audit=Path(args.audit_dir)
    details={d['clip_key']:d for d in json.loads((audit/'mapping_detail.json').read_text())}
    acceptance_path=audit/'record_acceptance.json'
    acceptance=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
    trials=[];coverage=[];labels=[];recordings=[];overlays=[]
    for key,item in doc['clips'].items():
        m=item['meta'];meta=metadata(key,item)
        if m.get('task_name') not in STAGES or m.get('observer_skill')!='novice' or m.get('src_skill')!='novice': continue
        detail=details[key]
        gate=acceptance.get(key,dict(accepted=False,reason='playback_time_and_display_mapping_not_verified'))
        accepted=bool(gate['accepted']);reason='' if accepted else gate['reason']
        d=load_native(key,item,root,detail)
        if accepted: d=prepare_arrays(d)
        marks=frame_export(out,meta,d,accepted,reason)
        z,count,bits=geo.target_center(d['slots'],d['width'],d['height'])
        included=np.isin(d['stage'],STAGES[d['task']]);gripper=in_image(d['left'],d['width'],d['height'])&in_image(d['right'],d['width'],d['height'])
        record=dict(meta,accepted=accepted,exclusion_reason=reason,frames=len(d['frames']),
                    gaze_coordinate_valid_n=int(d['coordinate_valid'].sum()),target_available_n=int((count>0).sum()),
                    bilateral_dlc_valid_n=int(gripper.sum()),source_gaze_used=False,target_identity_field=d['target_identity_field'],
                    video_ts_path=detail['passive_clock_path'],source_video_ts_path=detail['source_clock_path'])
        for f in FIELDS:record[f]=str(resolve(root,item.get(f)))
        recordings.append(record)
        rawlabel=pd.DataFrame(dict(stage=d['stage'],target_points=count,slot_bitmask=bits,point_label_mode=d['label_mode']))
        for values,part in rawlabel.groupby(['stage','target_points','slot_bitmask','point_label_mode'],dropna=False):
            labels.append(dict(meta,stage=values[0],target_points=int(values[1]),slot_bitmask=int(values[2]),point_label_mode=values[3],
                               frames=len(part),analysis_accepted=accepted,role='source_annotation_inventory'))
        for scenario,(rt,rp,_,sm) in SCENARIOS.items():
            if accepted:
                xy,valid=support(d,scenario);state=geo.classify(xy,d['z'],d['left'],d['right'],rt,rp,valid)
                area=geo.areas(d['z'],d['left'],d['right'],d['width'],d['height'],rt,rp,included&(count>0)&gripper)
                marks['area_'+scenario]=area
            else:
                valid=np.zeros(len(d['frames']),bool);state=np.full(len(valid),-1,np.int8)
            marks['valid_real_'+scenario]=valid;marks['state_gaze_'+scenario]=state
            for stage in ['whole',*STAGES[d['task']]]:
                st=included if stage=='whole' else d['stage']==stage
                cov=dict(meta,scenario=scenario,stage=stage,total_n=int(st.sum()),gaze_coordinate_valid_n=int((st&d['coordinate_valid']).sum()),
                         target_n=int((st&(count>0)).sum()),bilateral_dlc_n=int((st&gripper).sum()),
                         spatial_eligible_before_timing_n=int((st&d['coordinate_valid']&(count>0)&gripper).sum()),
                         valid_n=int((st&valid).sum()),valid_s=float(d['dt'][st&valid].sum()) if accepted else 0.,
                         excluded_for_unverified_record_mapping_n=int(st.sum()) if not accepted else 0,
                         source_annotation_clock_s=float(geo.times(d['timestamp_ns'],d['frames'])[0][st].sum()),
                         source_clock_duration_role='annotation audit only; not passive viewing duration' if not accepted else 'verified native duration',
                         analysis_status='computed' if accepted else 'not_estimable',reason=reason)
                cov['total_s']=float(d['dt'][st].sum()) if accepted else np.nan
                cov['gaze_n']=int((st&np.isfinite(xy).all(1)&(xy>=0).all(1)&(xy<=[d['width'],d['height']]).all(1)).sum()) if accepted else 0
                cov['interpolated_n']=int((st&valid&d['interpolated']).sum()) if accepted and sm else 0
                cov['interpolated_fraction']=cov['interpolated_n']/cov['valid_n'] if cov['valid_n'] else np.nan
                coverage.append(cov)
                summary=summarize(state,valid&st,area,d) if accepted else {**{x:np.nan for x in METRICS},'valid_n':0,'valid_s':0.,'C_reason':reason,'valid_seconds':0.}
                summary['C_U_reason']='' if np.isfinite(summary['C_U']) else (reason if not accepted else ('no_valid_observation_time' if summary['valid_s']==0 else 'zero_exclusive_area'))
                summary['occupancy_reason']='' if summary['valid_s']>0 else (reason if not accepted else 'no_valid_observation_time')
                for short in 'TPO':
                    summary['dwell_'+short+'_reason']='' if np.isfinite(summary['dwell_'+short+'_ms']) else (reason if not accepted else 'no_observed_run_reaches_120ms')
                trials.append(dict(meta,scenario=scenario,stage=stage,analysis_status=cov['analysis_status'],reason=reason,**summary))
        np.savez_compressed(out/'frames'/(meta['recording_id']+'.npz'),**marks)
        overlay=write_overlay(out,meta,d,reason)
        if overlay: overlays.append(dict(clip_key=key,path=overlay,role='native_frame_spatial_review' if accepted else 'unverified_alignment_diagnostic'))
        print(meta['recording_id'], 'accepted' if accepted else reason,flush=True)
    csvout(out/'trial_stage.csv',trials);csvout(out/'coverage.csv',coverage);csvout(out/'recording_audit.csv',recordings);csvout(out/'target_labels.csv',labels)
    csvout(out/'region_review/overlay_index.csv',overlays)
    from tl_observed_statistics import participant_summary
    people,groups=participant_summary(pd.DataFrame(trials),ParticipantBootstrap(42,10000))
    people['analysis_branch']='observed_gaze';groups['analysis_branch']='observed_gaze'
    people['analysis_status']=np.where(people.trials_valid>0,'computed','not_estimable')
    groups['reason']=np.where(groups.n_participants>0,'','no_accepted_observer_records')
    csvout(out/'participant_stage_summary.csv',people);csvout(out/'group_stage_summary.csv',groups)
    save(out/'scenario_execution.json',dict(scenarios=list(SCENARIOS),eligible_recordings=len(recordings),accepted_recordings=sum(r['accepted'] for r in recordings),
          result='computed' if any(r['accepted'] for r in recordings) else 'all_five_conditions_unavailable_after_input_acceptance_gate',
          no_synthetic_estimates=True))

if __name__=='__main__':main()
