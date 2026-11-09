"""Compare annotation content and resulting point-mean centers; read-only."""
import csv,json,collections
from pathlib import Path
import numpy as np
import cv2
OUT=Path(r'C:\Users\GT502\Documents\ChatGPT\JMRR\gaze_pair_audit_work')
base=json.loads((OUT/'independent_gaze_pair_results.json').read_text(encoding='utf-8'))
records=[];totals=collections.Counter();field_totals=collections.defaultdict(collections.Counter)
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def num(s):return float(s) if s.strip() else np.nan
def center(a,w,h):
    valid=((a>=0)&(a<[w,h])).all(2)
    count=valid.sum(1);sm=np.where(valid[:,:,None],a,0).sum(1);xy=np.full((len(a),2),np.nan)
    np.divide(sm,count[:,None],out=xy,where=count[:,None]>0)
    return xy,count
for item in base['records']:
    if item.get('shared_fields_match',True):continue
    p=Path(item['npy']);q=Path(item['csv']);a=np.load(p,allow_pickle=False)
    with q.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    cid=p.stem[:-5];video=p.parents[1]/'stimuli'/(cid+'.mp4')
    cap=cv2.VideoCapture(str(video));w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
    assert w>0 and h>0,str(video)
    data=dict(domain=item['domain'],user=item['user'],record=item['stem'],npy=str(p),csv=str(q),rows=len(a),video_width=w,video_height=h,fields={})
    ns=np.full((len(a),5,2),np.nan);cs=ns.copy()
    for slot in range(1,6):
        for axis,coord in enumerate('xy'):
            key=f'point_{slot}_{coord}';n=a[key];c=np.array([num(r[key]) for r in rows])
            ns[:,slot-1,axis]=n;cs[:,slot-1,axis]=c
            nf,cf=np.isfinite(n),np.isfinite(c)
            counts=dict(npy_populated_csv_missing=int((nf&~cf).sum()),npy_missing_csv_populated=int((~nf&cf).sum()),
                both_finite_changed=int((nf&cf&(n!=c)).sum()),both_finite_equal=int((nf&cf&(n==c)).sum()),both_missing=int((~nf&~cf).sum()))
            data['fields'][key]=counts;field_totals[key].update(counts)
    nl=np.asarray(a['point_label_mode'],str);cl=np.array([r['point_label_mode'] for r in rows])
    lm=nl!=cl
    counts=dict(npy_nonblank_csv_blank=int(((nl!='')&(cl=='')).sum()),npy_blank_csv_nonblank=int(((nl=='')&(cl!='')).sum()),
        both_nonblank_changed=int(((nl!='')&(cl!='')&lm).sum()),mismatched_rows=int(lm.sum()))
    data['point_label_mode_counts']=counts;field_totals['point_label_mode'].update(counts)
    transitions=collections.Counter(zip(nl[lm].tolist(),cl[lm].tolist()))
    data['point_label_mode_transitions']=[dict(npy=n,csv=c,rows=count) for (n,c),count in transitions.most_common()]
    nx,nc=center(ns,w,h);cx,cc=center(cs,w,h);nf=nc>0;cf=cc>0
    delta=np.linalg.norm(nx-cx,axis=1);both_changed=nf&cf&(delta>0)
    cm=(nf!=cf)|both_changed
    center_counts=dict(npy_center_only=int((nf&~cf).sum()),csv_center_only=int((~nf&cf).sum()),both_center_changed=int(both_changed.sum()),
        both_center_identical=int((nf&cf&~both_changed).sum()),neither_center=int((~nf&~cf).sum()),slot_count_changed=int((nc!=cc).sum()),
        differing_center_rows=int(cm.sum()),both_changed_max_distance_px=float(delta[both_changed].max()) if both_changed.any() else 0,
        both_changed_median_distance_px=float(np.median(delta[both_changed])) if both_changed.any() else 0,
        both_center_difference_over50px=int((both_changed&(delta>50)).sum()))
    data['derived_target_center']=center_counts
    for key,value in center_counts.items():
        if key not in ['both_changed_max_distance_px','both_changed_median_distance_px']:totals[key]+=value
    data['center_difference_stage_counts']=dict(collections.Counter(a['stage'][cm].tolist()))
    candidates=np.flatnonzero(cm)
    data['first_three_center_differences']=[dict(row0=int(i),csv_line=int(i)+2,frame_number=int(a['Frame_Number'][i]),stage=str(a['stage'][i]),
        npy_center=nx[i],csv_center=cx[i],npy_valid_target_slots=int(nc[i]),csv_valid_target_slots=int(cc[i]),
        center_distance_px=delta[i],npy_point_label_mode=nl[i],csv_point_label_mode=cl[i]) for i in candidates[:3]]
    changed_indices=np.flatnonzero(both_changed)
    data['first_three_both_finite_center_changes']=[dict(row0=int(i),csv_line=int(i)+2,frame_number=int(a['Frame_Number'][i]),stage=str(a['stage'][i]),
        npy_center=nx[i],csv_center=cx[i],center_distance_px=delta[i]) for i in changed_indices[:3]]
    records.append(data)
    if len(records)%20==0:print('Compared annotations',len(records),flush=True)
result=dict(scope='Only83 file pairs with differing shared annotation fields; all rows, no stage exclusion',
            center_definition='arithmetic mean of finite in-image slot pairs, actual video width/height; no tracking or gaze/PSM analysis',
            pairs=len(records),field_totals={k:dict(v) for k,v in field_totals.items()},center_totals=dict(totals),records=records,
            source_files_modified=False,authority_of_versions='not inferred; neither file rewritten')
(OUT/'independent_annotation_details.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(clean({k:v for k,v in result.items() if k!='records'}),ensure_ascii=False,indent=2))
