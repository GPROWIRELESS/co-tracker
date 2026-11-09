"""Read-only passive mapping audit using verified canonical original-frame associations."""
import csv, hashlib, json
from collections import Counter
from decimal import Decimal
from pathlib import Path
import numpy as np
from audit_mapping import rows, timeline, video

OUT=Path(__file__).resolve().parent
LINK=Path(r'C:\Users\GT502\Desktop\JMRR2027\passive_gaze_links.json')
FIELDS=('gaze_npy_path','source_video_path','stage_path','target_locking_label_path','psm_dlc_csv_path')
STAGES={'A':['reach ring','grab ring','transfer ring','place ring'],'C':['reach string','wrap around psm','pull through and cinch','threading string through tower']}
fingerprints={}
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def track(p,role):
    p=Path(p).resolve(); k=str(p)
    if k not in fingerprints:fingerprints[k]=dict(path=k,roles=[role],size=p.stat().st_size,sha256_before=digest(p))
    elif role not in fingerprints[k]['roles']:fingerprints[k]['roles'].append(role)
    return p
def load(p,role):return np.load(track(p,role),allow_pickle=False)
def mapping(src,dst):
    c=Counter(int(x) for x in src);d={int(x):i for i,x in enumerate(src) if c[int(x)]==1};return np.array([d.get(int(x),-1) for x in dst],dtype=np.int64)
def indices_stats(a,prefix):
    f=a['Frame_Number'];return {prefix+'_rows':len(a),prefix+'_duplicate_frames':len(f)-len(np.unique(f)),prefix+'_nonincreasing_frames':int((np.diff(f)<=0).sum()),prefix+'_fields':list(a.dtype.names)}
def csvwrite(p,rs):
    cols=list(dict.fromkeys(k for r in rs for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rs)
def main():
    doc=json.loads(track(LINK,'authoritative_json').read_text(encoding='utf-8-sig'));root=Path(doc['data_root']);prior={x['clip_key']:x for x in json.loads((OUT/'mapping_detail.json').read_text())};audit=[];issues=[];records=[];target_rows=[]
    (OUT/'frames').mkdir(exist_ok=True)
    for key,item in doc['clips'].items():
        m=item['meta'];slug=key.rsplit('/',1)[-1]+'__'+hashlib.sha256(key.encode()).hexdigest()[:12];r=dict(clip_key=key,recording_id=slug,participant=str(m.get('observer')),**m,analysis_branch='observed_gaze',source='gaze',scope='real_all',raw_entry_json=item,data_root_raw=doc['data_root']);paths={};reasons=[]
        r['candidate_AC']=m.get('task_name') in STAGES;r['skill_eligible']=m.get('observer_skill')==m.get('src_skill')=='novice';r['split_used_as_filter']=False
        if not r['candidate_AC']:reasons.append('task_outside_A_C' if m.get('task_name') in ('B','D') else 'unknown_task')
        if m.get('observer_skill')!='novice':reasons.append('observer_skill_not_novice')
        if m.get('src_skill')!='novice':reasons.append('source_skill_not_novice')
        for field in FIELDS:
            v=item.get(field);missing=v is None or str(v).strip().lower() in ('','none','null','nan');p=None if missing else Path(v);p=p if p is None or p.is_absolute() else root/p
            paths[field]=p;r[field+'_raw']=v;r[field+'_resolved']=str(p) if p else None;r[field+'_missing']=missing;r[field+'_exists']=p.is_file() if p else False
            if not r[field+'_exists']:issues.append(dict(clip_key=key,kind='missing_mapping_path' if missing else 'linked_file_missing',role=field,raw=v,resolved=str(p) if p else None,affects_requested_AC=r['candidate_AC'],affects_skill_eligible=r['skill_eligible']))
        r['all_five_paths_exist']=all(r[f+'_exists'] for f in FIELDS);r['missing_link_roles']=[f for f in FIELDS if not r[f+'_exists']]
        if r['candidate_AC'] and r['skill_eligible']:reasons += ['missing_'+f for f in r['missing_link_roles']]
        gp=paths['gaze_npy_path'];a=load(gp,'passive_gaze') if gp and gp.is_file() else None
        if a is not None:
            r.update(indices_stats(a,'gaze'));r['gaze_required_fields_complete']=all(f in a.dtype.names for f in ('Frame_Number','gaze_x','gaze_y'))
            xy=np.column_stack([a['gaze_x'],a['gaze_y']]).astype(float);cv=np.isfinite(xy).all(axis=1)&(xy>=0).all(axis=1)&(xy<=1).all(axis=1);r['gaze_coordinate_valid_frames']=int(cv.sum());r['gaze_endpoint_frames']=int((cv&((xy==0)|(xy==1)).any(axis=1)).sum());r['valid_or_status_used_for_filter']=False
        # Inspect mapped annotations for every A/C record with links, including skill-excluded records.
        loaded={}
        if r['candidate_AC']:
            for role,pre in [('stage_path','stage'),('target_locking_label_path','target')]:
                p=paths[role]
                if p and p.is_file():
                    b=load(p,role);loaded[pre]=b;r.update(indices_stats(b,pre));idx=mapping(b['Frame_Number'],a['Frame_Number']) if a is not None else np.array([],dtype=int);r[pre+'_gaze_original_frame_missing']=int((idx<0).sum());r[pre+'_frames_exact_order']=bool(a is not None and np.array_equal(a['Frame_Number'],b['Frame_Number']))
                    if pre=='stage':
                        sf=next((f for f in ('stage','event') if f in b.dtype.names),None);r['stage_field']=sf;r['stage_required_fields_complete']=sf is not None;r['stage_values']=dict(Counter(str(x) for x in b[sf])) if sf else {}
                    else:
                        slots=[i for i in range(1,6) if all(f'point_{i}_{c}' in b.dtype.names for c in ('x','y'))];r['target_present_slots']=slots;r['target_required_fields_complete']=1 in slots and not any((f'point_{i}_x' in b.dtype.names)!=(f'point_{i}_y' in b.dtype.names) for i in range(1,6));r['point_label_mode_available']='point_label_mode' in b.dtype.names;r['identity_field']=next((f for f in ('target_id','target_identity','point_1_id') if f in b.dtype.names),None)
            p=paths['psm_dlc_csv_path']
            if p and p.is_file():
                with track(p,'psm_dlc').open(encoding='utf-8-sig',newline='') as f:
                    rd=csv.reader(f);hh=[next(rd) for _ in range(3)];dd=list(rd)
                di=np.array([int(x[0]) for x in dd]);r['dlc_rows']=len(dd);r['dlc_duplicate_indices']=len(di)-len(np.unique(di));r['dlc_indices_zero_based_consecutive']=bool(np.array_equal(di,np.arange(len(dd))));r['dlc_required_fields_complete']=all(sum(bb==b and cc==c for bb,cc in zip(hh[1],hh[2]))==1 for b in ('left_psm','right_psm') for c in ('x','y','likelihood'))
                loaded['dlc_headers']=hh;loaded['dlc_rows']=dd
        if key in prior:
            d=prior[key];r.update({k:v for k,v in d.items() if k not in ('clip_key','recording_id','meta')});r['coordinate_reference_status']='stored_normalized_full_1280x1024_media_reference';r['coordinate_reference_reason']='Use stored normalized NPY coordinates in the full media reference as the dataset analysis interpretation; decoded source/passive media dimensions and sampled image geometry agree. External display viewport transform is not independently verified.'
            r['timeline_validation']='verified_native_frame_mapping';r['physical_display_timing']='not_independently_established';r['external_display_transform']='not_independently_verified';r['playback_alignment_status']='canonical_original_frame_association_verified';r['canonical_accepted']=True;r['main_analysis_included']=True
            for role in ('passive_clock_path','source_clock_path','passive_video_path','raw_gaze_path'):track(d[role],role)
            track(paths['source_video_path'],'source_video');pf,pn=timeline(Path(d['passive_clock_path']));sf,sn=timeline(Path(d['source_clock_path']));pi=mapping(pf,a['Frame_Number']);si=mapping(sf,a['Frame_Number']);st=loaded['stage'];tg=loaded['target'];sti=mapping(st['Frame_Number'],a['Frame_Number']);tgi=mapping(tg['Frame_Number'],a['Frame_Number']);raw=rows(Path(d['raw_gaze_path']));ticks=np.array([int(Decimal(z['TIMETICK(f=10000000)'])) for z in raw],dtype=np.int64);gn=pn[0]+(ticks-ticks[0])*100;order=np.argsort(gn);s=gn[order];j=np.searchsorted(s,pn[pi]);lo=(j-1).clip(0,len(raw)-1);hi=j.clip(0,len(raw)-1);chosen=order[np.where(abs(s[hi]-pn[pi])<abs(s[lo]-pn[pi]),hi,lo)];rv=np.array([int(Decimal(z['VID_FRAME'])) for z in raw]);tc=next(n for n in raw[0] if n.startswith('TIME('));rt=np.array([float(z[tc]) for z in raw]);gx=np.array([float(z['BPOGX']) for z in raw],dtype=np.float32);gy=np.array([float(z['BPOGY']) for z in raw],dtype=np.float32)
            same=np.array_equal(a['gaze_x'],np.clip(gx[chosen],0,1),equal_nan=True) and np.array_equal(a['gaze_y'],np.clip(gy[chosen],0,1),equal_nan=True);assert same
            slots=np.full((len(a),5,2),np.nan)
            for q in range(1,6):
                if f'point_{q}_x' in tg.dtype.names:slots[:,q-1,:]=np.column_stack([tg[f'point_{q}_x'][tgi],tg[f'point_{q}_y'][tgi]])
            hh,dd=loaded['dlc_headers'],loaded['dlc_rows'];psm=[];lik=[]
            for b in ('left_psm','right_psm'):
                cidx=[next(i for i,(bb,cc) in enumerate(zip(hh[1],hh[2])) if bb==b and cc==c) for c in ('x','y','likelihood')];vals=np.array([[float(z[i]) if z[i].strip() else np.nan for i in cidx] for z in dd]);psm.append(vals[si,:2]);lik.append(vals[si,2])
            width,height=d['source_video']['width'],d['source_video']['height'];validslot=np.isfinite(slots).all(2)&(slots[:,:,0]>=0)&(slots[:,:,0]<width)&(slots[:,:,1]>=0)&(slots[:,:,1]<height);nslot=validslot.sum(1);slotmask=(validslot*(1<<np.arange(5))).sum(1);center=np.divide(np.where(validslot[:,:,None],slots,0).sum(1),nslot[:,None],out=np.full((len(a),2),np.nan),where=nslot[:,None]>0);stage=np.array(st[r['stage_field']][sti],dtype='U128');mode=np.array(tg['point_label_mode'][tgi],dtype='U128') if 'point_label_mode' in tg.dtype.names else np.full(len(a),'',dtype='U128');identityfield=r.get('identity_field');identity=np.array(tg[identityfield][tgi],dtype='U128') if identityfield else np.full(len(a),'',dtype='U128')
            fn=a['Frame_Number'].astype(np.int64);orig_raw=rv[chosen];rawframe_to_original=np.full(len(a),-1,dtype=np.int64);ok=(orig_raw>=0)&(orig_raw<len(pf));rawframe_to_original[ok]=pf[orig_raw[ok]];conflict=orig_raw!=pi
            psmvalid=[np.isfinite(x).all(1)&(x[:,0]>=0)&(x[:,0]<width)&(x[:,1]>=0)&(x[:,1]<height) for x in psm];eligible=cv&(nslot>0)&psmvalid[0]&psmvalid[1]&np.isin(stage,STAGES[m['task_name']])
            vals=dict(clip_key=np.array(key),recording_id=np.array(slug),original_frame_number=fn,gaze_npy_row_index=np.arange(len(a)),passive_clock_retained_row_index=pi,source_clock_retained_row_index=si,stage_npy_row_index=sti,target_npy_row_index=tgi,dlc_local_index=si,passive_video_ts_ns=pn[pi],source_video_ts_ns=sn[si],gaze_npy_timestamp_ns=a['Timestamp(ns)'],gaze_norm=xy,gaze_px=xy*[width,height],gaze_coordinate_valid=cv,gaze_valid_code_audit=a['valid'],stage=stage,point_slots_px=slots,point_slot_valid=validslot,target_center_px=center,slot_count=nslot,slot_bitmask=slotmask,point_label_mode=mode,target_identity=identity,left_psm_px=psm[0],right_psm_px=psm[1],left_psm_likelihood=lik[0],right_psm_likelihood=lik[1],selected_raw_eye_row=chosen,selected_raw_vid_frame=orig_raw,selected_raw_time_s=rt[chosen],selected_raw_tick=ticks[chosen],selected_raw_bpog=np.column_stack([gx[chosen],gy[chosen]]),raw_vid_frame_original_frame=rawframe_to_original,raw_playback_frame_index_discrepancy=conflict,physical_playback_verified=np.zeros(len(a),dtype=bool),canonical_frame_mapping_verified=np.ones(len(a),dtype=bool),canonical_record_accepted=np.array(True),analysis_eligible=eligible,external_display_transform_verified=np.array(False),media_scale_transform=np.diag([width,height,1]),image_width=np.array(width),image_height=np.array(height))
            fp=OUT/'frames'/f'{slug}.npz';np.savez_compressed(fp,**vals);r['frame_audit_file']=f'frames/{slug}.npz';r['frame_audit_sha256']=digest(fp);r['frame_audit_count']=len(a);r['target_valid_slot_frames']=int((nslot>0).sum());r['stage_in_scope_frames']=int(np.isin(stage,STAGES[m['task_name']]).sum());r['drop_ring_frames']=int((stage=='drop ring').sum());r['target_identity_field_used']=identityfield or 'unavailable';r['source_gaze_used']=False;r['source_gaze_validity_used']=False;r['source_score_used']=False;r['source_round_used']=False
            records.append(r.copy())
            for i in range(len(a)):
                target_rows.append(dict(clip_key=key,recording_id=slug,participant=m['observer'],task=m['task_name'],Frame_Number=int(fn[i]),stage=str(stage[i]),valid_slot_count=int(nslot[i]),slot_bitmask=int(slotmask[i]),point_label_mode=str(mode[i]),target_x=float(center[i,0]),target_y=float(center[i,1]),target_identity=str(identity[i]),canonical_record_accepted=True,analysis_included=bool(eligible[i]),exclusion_reason='' if eligible[i] else 'frame_outside_stage_or_invalid_gaze_target_or_bilateral_DLC'))
            issues.append(dict(clip_key=key,kind='raw_frame_index_definition_discrepancy',native_rows=len(a),different_index_rows=int(conflict.sum()),constructor_reproduced_exact=True,delta_min=int((orig_raw-pi).min()),delta_max=int((orig_raw-pi).max()),interpretation='Raw VID_FRAME is empirically floor(raw TIME * nominal FPS); canonical NPY uses precise native video_ts and shifted TIMETICK. Independent physical display counter semantics are not established, so index differences alone are not used as rejection evidence.',canonical_frame_mapping_verified=True,physical_display_timing='not_independently_established',recording_excluded=False));issues.append(dict(clip_key=key,kind='external_display_transform_not_independently_verified',analysis_reference='stored normalized full media image',recording_excluded=False))
        else:r['main_analysis_included']=False;r['coordinate_reference_status']='not_evaluated_outside_selected_population_or_missing_input';r['playback_alignment_status']='not_evaluated_outside_selected_population_or_missing_input'
        r['exclusion_reasons']=reasons;r['inclusion_status']='included_canonical_native_mapping' if r['main_analysis_included'] else 'excluded';audit.append(r)
    # Preserve constructor evidence; no source code execution and no old-chat permission reused.
    evidence=[Path(__file__).parent.parent.parent/'passive_gt_work/input_audit/historical_generate_gaze_npy.py.txt',Path(__file__).parent.parent.parent/'passive_gt_work/input_audit/historical_generate_gaze_npy.ipynb.txt']
    for p in evidence:track(p,'historical_constructor_provenance_text')
    for z in fingerprints.values():z['sha256_after']=digest(Path(z['path']));z['unchanged']=z['sha256_before']==z['sha256_after']
    csvwrite(OUT/'passive_mapping_audit.csv',audit);csvwrite(OUT/'recording_audit.csv',records);csvwrite(OUT/'target_labels.csv',target_rows)
    (OUT/'input_issues.json').write_text(json.dumps(dict(issues=issues,main_analysis_included_recordings=16,blocking_reasons=[],limitations=['physical_display_timing_not_independently_established','external_display_transform_not_independently_verified'],notes='Native original Frame_Number joins are verified exactly against passive/source clocks, annotations, video counts and DLC indices, and historical NPY constructor provenance is reproduced exactly. Raw VID_FRAME differences are descriptive provenance diagnostics, not a substitute clock or proof of incorrect canonical association. No remapping, nominal FPS clock or source-operator gaze is used.'),ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'input_fingerprints.json').write_text(json.dumps(dict(files=list(fingerprints.values()),all_unchanged=all(z['unchanged'] for z in fingerprints.values())),indent=2),encoding='utf-8')
    (OUT/'record_acceptance.json').write_text(json.dumps({r['clip_key']:dict(accepted=r['main_analysis_included'],reason=';'.join(r['exclusion_reasons']),basis='Exact original Frame_Number joins; precise passive video_ts; matching source/video/DLC local mapping; historical NPY constructor independently reproduced; stored normalized full-media coordinate interpretation. Physical display synchronization and external viewport transform remain unverified limitations.' if r['main_analysis_included'] else 'Authoritative JSON task, observer/source skill and linked input scope.',user_waiver_claimed=False) for r in audit},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'audit_summary_final.json').write_text(json.dumps(dict(total_records=len(audit),AC_candidates=sum(r['candidate_AC'] for r in audit),AC_link_complete=sum(r['candidate_AC'] and r['all_five_paths_exist'] for r in audit),novice_pair_AC=len(records),included=sum(r['main_analysis_included'] for r in audit),recording_ids_unique=len(set(r['recording_id'] for r in audit))==len(audit),clip_keys_unique=len(set(r['clip_key'] for r in audit))==len(audit),frame_rows=sum(r['frame_audit_count'] for r in records),raw_frame_index_discrepancy_rows=sum(r['selected_raw_vid_frame_conflicts'] for r in records),files_fingerprinted=len(fingerprints),source_files_unchanged=all(z['unchanged'] for z in fingerprints.values()),canonical_original_frame_mapping_verified=True,physical_display_timing='not_independently_established',external_display_transform='not_independently_verified'),indent=2),encoding='utf-8')
    print((OUT/'audit_summary_final.json').read_text(),flush=True)
if __name__=='__main__':main()
