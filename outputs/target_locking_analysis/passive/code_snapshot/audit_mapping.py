import csv, hashlib, json
from collections import Counter
from decimal import Decimal
from pathlib import Path
import numpy as np
import cv2

OUT=Path(__file__).resolve().parent
LINK=Path(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def timeline(p):
    rr=rows(p); rr=[r for r in rr if r.get('IsDelete','0').strip() in ('','0','0.0')]
    return np.array([int(r['Frame_Number']) for r in rr]),np.array([int(Decimal(r['Timestamp(ns)'])) for r in rr],dtype=np.int64)
def video(p):
    c=cv2.VideoCapture(str(p)); d=dict(width=int(c.get(3)),height=int(c.get(4)),frames=int(c.get(7)),fps=float(c.get(5)));c.release();return d
def main():
    # Legacy per-record key selected_raw_vid_frame_conflicts means unequal numeric indices only.
    # It is not proof of physical display misalignment and does not reject a record.
    doc=json.loads(LINK.read_text(encoding='utf-8-sig')); root=Path(doc['data_root']); allrows=[]; detailed=[]
    fields=('gaze_npy_path','source_video_path','stage_path','target_locking_label_path','psm_dlc_csv_path')
    for key,item in doc['clips'].items():
        meta=item['meta']; r=dict(clip_key=key,**meta); ps={}
        for f in fields:
            v=item.get(f); miss=v is None or str(v).strip().lower() in ('','nan','none','null'); p=None if miss else Path(v);p=p if p is None or p.is_absolute() else root/p
            ps[f]=p;r[f+'_raw']=v;r[f+'_resolved']=str(p) if p else None;r[f+'_exists']=p.is_file() if p else False
        r['candidate_AC']=meta['task_name'] in ('A','C');r['skill_eligible']=meta.get('observer_skill')==meta.get('src_skill')=='novice'
        r['all_five_paths_exist']=all(r[f+'_exists'] for f in fields)
        gp=ps['gaze_npy_path']; r['gaze_fields']=[]
        if gp and gp.is_file():
            a=np.load(gp,allow_pickle=False);r['gaze_rows']=len(a);r['gaze_fields']=list(a.dtype.names or ());ids=a['Frame_Number'];r['gaze_duplicate_frames']=len(ids)-len(np.unique(ids));r['gaze_nonincreasing_frames']=int((np.diff(ids)<=0).sum())
        if r['candidate_AC'] and r['skill_eligible']:
            detail=dict(clip_key=key,recording_id=gp.stem.removesuffix('_gaze'),meta=meta,all_five_paths_exist=r['all_five_paths_exist'])
            # Clock/video discovery follows historical constructor naming, then original-key and pixel verification.
            rid=detail['recording_id']; pb=gp.parent.parent; pc=pb/'csv'/f'{rid}_video_ts.csv';pv=pb/'stimuli'/f'{rid}.mp4';rawp=pb/'csv'/f'{rid}_gaze.csv'
            detail.update(passive_clock_path=str(pc),passive_clock_exists=pc.is_file(),passive_video_path=str(pv),raw_gaze_path=str(rawp))
            if pc.is_file():
                pf,pn=timeline(pc); detail.update(clock_frames=len(pf),clock_duplicate_frames=len(pf)-len(np.unique(pf)),clock_nonincreasing_frames=int((np.diff(pf)<=0).sum()),clock_nonincreasing_timestamps=int((np.diff(pn)<=0).sum()),clock_frame_gaps=int((np.diff(pf)!=1).sum()),clock_time_gaps_over_100ms=int((np.diff(pn)>100000000).sum()),clock_npy_frame_exact=bool(np.array_equal(pf,a['Frame_Number'])),clock_npy_timestamp_exact=bool(np.array_equal(pn,a['Timestamp(ns)'])),passive_clock_sha256=sha(pc))
                detail['passive_video']=video(pv)
                if r['source_video_path_exists']:
                    sp=ps['source_video_path'];sc=sp.parent.parent/'csv'/f'{sp.stem}_video_ts.csv';sf,sn=timeline(sc)
                    detail.update(source_clock_path=str(sc),source_clock_sha256=sha(sc),passive_source_clock_identical=bool(np.array_equal(pf,sf) and np.array_equal(pn,sn)),source_video=video(sp))
                    ix={int(f):i for i,f in enumerate(sf)};di=np.array([ix.get(int(f),-1) for f in pf]);detail.update(source_frame_join_missing=int((di<0).sum()),dlc_local_first=int(di[0]),dlc_local_last=int(di[-1]))
                    if r['psm_dlc_csv_path_exists']:
                        with ps['psm_dlc_csv_path'].open(newline='',encoding='utf-8-sig') as f:
                            rd=csv.reader(f);headers=[next(rd) for _ in range(3)];dd=list(rd)
                        detail.update(dlc_rows=len(dd),dlc_indices_exact=bool([int(x[0]) for x in dd]==list(range(len(dd)))),dlc_rows_match_source_clock=len(dd)==len(sf),dlc_bodyparts=headers[1],dlc_coords=headers[2])
                    cp=cv2.VideoCapture(str(pv));cs=cv2.VideoCapture(str(sp)); pixel=[]
                    for i in sorted(set([0,len(pf)//2,len(pf)-1])):
                        cp.set(0 if False else cv2.CAP_PROP_POS_FRAMES,i);cs.set(cv2.CAP_PROP_POS_FRAMES,int(di[i]));op,ip=cp.read();os,im=cs.read()
                        q=dict(passive_local_index=i,source_local_index=int(di[i]),decoded=bool(op and os))
                        if op and os and ip.shape==im.shape:q.update(exact=bool(np.array_equal(ip,im)),mae=float(np.abs(ip.astype(float)-im.astype(float)).mean()),correlation=float(np.corrcoef(ip.flatten(),im.flatten())[0,1]))
                        pixel.append(q)
                    cp.release();cs.release();detail['pixel_checks']=pixel
                for f in ('stage_path','target_locking_label_path'):
                    if r[f+'_exists']:
                        b=np.load(ps[f],allow_pickle=False); detail[f+'_fields']=list(b.dtype.names);bf=b['Frame_Number'];detail[f+'_frame_exact']=bool(np.array_equal(pf,bf));detail[f+'_duplicate_frames']=len(bf)-len(np.unique(bf))
                        if f=='stage_path':
                            label=next((n for n in ('stage','event') if n in b.dtype.names),None);detail['stage_field']=label;detail['stage_values']=dict(Counter(str(x) for x in b[label])) if label else {}
                if rawp.is_file():
                    raw=rows(rawp);ticks=np.array([int(Decimal(z['TIMETICK(f=10000000)'])) for z in raw],dtype=np.int64);gn=pn[0]+(ticks-ticks[0])*100;order=np.argsort(gn);s=gn[order];j=np.searchsorted(s,pn);l=(j-1).clip(0,len(raw)-1);h=j.clip(0,len(raw)-1);chosen=order[np.where(abs(s[h]-pn)<abs(s[l]-pn),h,l)]
                    gx=np.array([float(z['BPOGX']) for z in raw],dtype=np.float32);gy=np.array([float(z['BPOGY']) for z in raw],dtype=np.float32);v=np.array([int(Decimal(z['BPOGV'])) for z in raw],dtype=np.uint8)
                    exact=bool(np.array_equal(a['gaze_x'],np.clip(gx[chosen],0,1),equal_nan=True) and np.array_equal(a['gaze_y'],np.clip(gy[chosen],0,1),equal_nan=True) and np.array_equal(a['valid'],v[chosen]))
                    rv=np.array([int(Decimal(z['VID_FRAME'])) for z in raw]);delta=rv[chosen]-np.arange(len(pf));tc=next(n for n in raw[0] if n.startswith('TIME('));rt=np.array([float(z[tc]) for z in raw]);res=rv-rt*detail['passive_video']['fps']
                    detail.update(raw_gaze_sha256=sha(rawp),constructor_exact=exact,raw_media_names=sorted(set(z['MEDIA_NAME'] for z in raw)),raw_time_first_s=float(rt[0]),raw_time_last_s=float(rt[-1]),selected_raw_vid_frame_delta_min=int(delta.min()),selected_raw_vid_frame_delta_median=float(np.median(delta)),selected_raw_vid_frame_delta_max=int(delta.max()),selected_raw_vid_frame_conflicts=int((delta!=0).sum()),selected_raw_vid_frame_matches=int((delta==0).sum()),selected_raw_vid_frame_outside=int(((rv[chosen]<0)|(rv[chosen]>=len(pf))).sum()),raw_vid_frame_minus_TIME_times_fps_min=float(res.min()),raw_vid_frame_minus_TIME_times_fps_max=float(res.max()),historically_clamped_coordinates=int(((gx[chosen]<0)|(gx[chosen]>1)|(gy[chosen]<0)|(gy[chosen]>1)).sum()))
                    np.savez_compressed(OUT/f'{rid}_raw_mapping.npz',original_frame_number=pf,selected_raw_row=chosen,selected_raw_vid_frame=rv[chosen],raw_frame_index_discrepancy=delta!=0,source_ns=pn)
            detailed.append(detail)
        allrows.append(r)
    ac=[r for r in allrows if r['candidate_AC']];el=[r for r in ac if r['skill_eligible']]; summary=dict(total=len(allrows),tasks=dict(Counter(r['task_name'] for r in allrows)),AC=len(ac),AC_observer_skills=dict(Counter(r['observer_skill'] for r in ac)),AC_source_skills=dict(Counter(r['src_skill'] for r in ac)),AC_all_paths_exist=sum(r['all_five_paths_exist'] for r in ac),novice_pair_AC=len(el),novice_pair_AC_tasks=dict(Counter(r['task_name'] for r in el)),novice_pair_AC_all_paths=sum(r['all_five_paths_exist'] for r in el),novice_pair_AC_observers=sorted(set(r['observer'] for r in el)),novice_pair_AC_sources=sorted(set(r['content_id'] for r in el)),json_sha256=sha(LINK),raw_vid_frame_difference_recordings=sum(d.get('selected_raw_vid_frame_conflicts',0)>0 for d in detailed))
    (OUT/'mapping_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(OUT/'mapping_detail.json').write_text(json.dumps(detailed,indent=2),encoding='utf-8');(OUT/'json_inventory.json').write_text(json.dumps(allrows,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
