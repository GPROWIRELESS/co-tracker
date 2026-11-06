"""Target locking v2: sampled probability mass, with separate point decoders.

Native-frame/time support is identical across all models and decoding branches.
All pre-existing runs are preserved. A successful full run supersedes the
specified legacy run only after writing and validating its own results.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import tl_geometry as geo
import tl_inputs as inputs
from tl_common import STAGES, dump_json, save_rows, in_image, stage_masks, ratio, summarize
from tl_stats import finish

MODELS = ('all_ABCD','single_task','leave_out','spd_finetune','spd_zero_shot')
DECODERS = {'expectation':'expectedxy', 'argmax':'argmaxxy', 'sample':'sampledxy'}
META = ('content_id','participant','task','task_num','attempt','round','score','status')
# Radius stays 50/50 in the primary analysis; changed radii are sensitivities.
SCENARIOS = {
    'main':(50,50,(2,3),False), 'gaze3':(50,50,(3,),False),
    'legacy123':(50,50,(1,2,3),False), 'r25':(25,25,(2,3),False),
    'r100':(100,100,(2,3),False), 'gripper100':(50,100,(2,3),False),
    'smooth':(50,50,(2,3),True),
}
DEFAULT_OLD = Path('D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/20261105T033217114891Z_full')


def prepare(rec,args):
    d=inputs.read_record(rec,args.data_root,args.repo,gaze_allowed_codes=(2,3),audit_cache_dir=args.cache_dir)
    d['z'],d['count'],d['slotmask']=geo.target_center(d['slots'],d['width'],d['height'])
    d['included']=np.isin(d['stage'],STAGES[rec['task']])
    d['gripper_ok']=in_image(d['left'],d['width'],d['height']) & in_image(d['right'],d['width'],d['height'])
    d['dt'],d['breaks']=geo.times(d['timestamp_ns'],d['frames'])
    d['breaks'] |= d['identity_change'] | ~d['included']
    d['breaks'][1:] |= (d['occurrence'][1:]!=d['occurrence'][:-1]) | ~d['included'][:-1]
    d['smooth_xy'],d['interpolated']=geo.smooth_gaze(d['timestamp_ns'],d['gaze'],d['occurrence'],d['breaks'])
    return d


def support(d,scenario):
    _,_,codes,smooth=SCENARIOS[scenario]
    xy=d['smooth_xy'] if smooth else d['gaze_unmasked']
    gaze_ok=np.isfinite(xy).all(axis=1)
    if not smooth:
        gaze_ok &= np.isin(d['valid_code'],codes)
    return xy,d['included'] & (d['count']>0) & d['gripper_ok'] & gaze_ok


def sum_probability(q,ok,area,d):
    idx=ok & np.isfinite(q).all(axis=1)
    weights=d['dt'][idx]
    seconds=weights.sum()
    share=(weights@q[idx])/seconds if seconds>0 else np.full(4,np.nan)
    base=(weights@area[idx])/seconds if seconds>0 else np.full(4,np.nan)
    n,t,p,o=share; an,at,ap,ao=base
    c=ratio(t,t+p); cu=ratio(at,at+ap)
    return dict(valid_n=int(idx.sum()),valid_s=seconds,L_N=n,L_T=t,L_P=p,L_O=o,
        L_target=t+o,L_gripper=p+o,D=t+p,C=c,B=t-p,area_N=an,area_T=at,area_P=ap,
        area_O=ao,C_U=cu,C_excess=c-cu,E_target=t+o-at-ao,E_gripper=p+o-ap-ao,
        B_excess=t-p-at+ap,C_reason='' if np.isfinite(c) else ('zero_exclusive_occupancy' if seconds else 'no_valid_frames'))


def onehot(state):
    out=np.full((len(state),4),np.nan)
    ix=state>=0
    out[ix]=np.eye(4)[state[ix]]
    return out


def fit_refs(records,args,tasks,out):
    """Keep the entire duration-weighted training heatmap, not its mean point."""
    train={t:inputs.train_ids(args.repo,t) for t in tasks}
    sums={}; members=[]; failures=[]
    selected=[r for r in records if r['content_id'] in train.get(r['task'],set())]
    for k,rec in enumerate(selected,1):
        print(f"Reference {k}/{len(selected)} {rec['content_id']}",flush=True)
        try:
            d=prepare(rec,args)
        except (ValueError,OSError,KeyError) as exc:
            failures.append(dict(content_id=rec['content_id'],reason=str(exc)))
            continue
        for scenario,(rt,rp,_,smooth) in SCENARIOS.items():
            xy,ok=support(d,scenario)
            h=geo.classify(xy,d['z'],d['left'],d['right'],rt,rp,ok)
            for stage,st in stage_masks(d):
                ids=np.flatnonzero(ok & st)
                if not len(ids):
                    continue
                w=d['dt'][ids]
                seconds=w.sum()
                key=(rec['task'],scenario,stage)
                a=sums.setdefault(key,dict(seconds=0.,target=0.,gripper=0.,heatmap=np.zeros((32,40))))
                a['seconds']+=seconds
                a['target']+=w[(h[ids]&1)>0].sum()
                a['gripper']+=w[(h[ids]&2)>0].sum()
                for start in range(0,len(ids),1024):
                    chunk=ids[start:start+1024]
                    kernels=geo.gaussian_maps(xy[chunk],d['width'],d['height'])
                    a['heatmap']+=np.einsum('n,nhw->hw',d['dt'][chunk],kernels)
                members.append(dict(task=rec['task'],scenario=scenario,stage=stage,
                    content_id=rec['content_id'],frames=len(ids),seconds=seconds))
    refs={}; tab=[]; maps={}
    for j,(key,a) in enumerate(sums.items()):
        if a['seconds']<=0: continue
        ref=dict(seconds=a['seconds'],rate_target=a['target']/a['seconds'],
                 rate_gripper=a['gripper']/a['seconds'],heatmap=a['heatmap']/a['seconds'])
        refs[key]=ref
        map_key=f'map_{j:03d}'
        maps[map_key]=ref['heatmap']
        tab.append(dict(task=key[0],scenario=key[1],stage=key[2],map_key=map_key,
                        **{k:v for k,v in ref.items() if k!='heatmap'}))
    save_rows(tab,out/'baseline_parameters.csv')
    save_rows(members,out/'baseline_training_members.csv')
    np.savez_compressed(out/'baseline_training_heatmaps.npz',**maps)
    dump_json(out/'baseline_input_audit.json',dict(train_ids={t:sorted(v) for t,v in train.items()},
        unavailable=failures,weight='valid native frame exposure, pooled in task/scenario/stage',
        reference_types={'center_point':'deterministic center location; event probability 0/1',
          'uniform_area':'uniform random image location; soft event area probability',
          'train_heatmap':'full normalized sigma=1 training mean heatmap integrated by pixel area',
          'train_rate':'training-only event probability, no test fitted frequency',
          'GT_smoothed':'test-gaze-centered sigma=1 map; diagnostic only, NOT a bound or BSS baseline'},
        grid='inclusive endpoint nodes, pixel-center Voronoi cells; heatmap cell masses',
        scope='common task-training evaluation reference for every model, including cross-task/zero-shot'))
    return refs,train,failures


def compare(owner,d,ok,g,q,gs,ms,refq,hard_state=None):
    idx=ok & (g>=0) & np.isfinite(q).all(axis=1)
    w=d['dt'][idx]; total=w.sum()
    row=dict(owner,valid_s=total,valid_n=int(idx.sum()))
    for metric in ('C','L_target','L_gripper'):
        row['delta_'+metric]=ms[metric]-gs[metric]
        row['abs_delta_'+metric]=abs(row['delta_'+metric])
    for bit,name in [(1,'target'),(2,'gripper')]:
        gh=(g[idx]&bit)>0
        pred=q[idx,bit]+q[idx,3]
        row['BS_'+name]=ratio(np.sum(w*(pred-gh)**2),total)
        for label,rq in refq.items():
            p=np.asarray(rq)[idx,bit]+np.asarray(rq)[idx,3]
            row[f'BS_{name}_{label}']=ratio(np.sum(w*(p-gh)**2),total)
        if hard_state is not None:
            mh=(hard_state[idx]&bit)>0
            for label,mask in [('FP',mh&~gh),('FN',~mh&gh),('TP',mh&gh),('TN',~mh&~gh)]:
                row[f'{label}_{name}_s']=w[mask].sum()
                row[f'{label}_{name}_n']=int(mask.sum())
    matrix=[]
    if hard_state is not None:
        for metric in ('direct_rate','via_rate','dwell_T_ms','dwell_P_ms','dwell_O_ms'):
            row['delta_'+metric]=ms[metric]-gs[metric]
        codes=g[idx]*4+hard_state[idx]
        counts=np.bincount(codes,minlength=16)
        weights=np.bincount(codes,weights=w,minlength=16)
        names=('N','T','P','O')
        matrix=[dict(owner,actual=names[j//4],predicted=names[j%4],frames=int(counts[j]),seconds=weights[j]) for j in range(16)]
    return row,matrix


def point_overlay(out,rec,d,bundles):
    if rec['content_id'] not in ('User1_1_1','User1_5_2'): return
    folder=out/'overlays';folder.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(d['video'])
    try:
        xy,ok=support(d,'main')
        for stage,st in list(stage_masks(d))[1:]:
            ids=np.flatnonzero(st&ok)
            if not len(ids): continue
            i=ids[len(ids)//2];cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
            found,img=cap.read()
            if not found: continue
            for point in d['slots'][i]:
                if in_image(point[None],d['width'],d['height'])[0]:
                    cv2.circle(img,tuple(np.rint(point).astype(int)),4,(0,220,255),-1)
            for name,color in [('z',(0,220,255)),('left',(0,210,0)),('right',(0,210,0)),('gaze',(255,170,0))]:
                pt=tuple(np.rint(d[name][i]).astype(int))
                if name!='gaze':cv2.circle(img,pt,50,color,2)
                cv2.drawMarker(img,pt,color,cv2.MARKER_CROSS,16,2)
            if 'all_ABCD' in bundles:
                b=bundles['all_ABCD']
                for key,color,marker in [('expectedxy',(230,40,240),cv2.MARKER_DIAMOND),
                    ('argmaxxy',(40,60,255),cv2.MARKER_TRIANGLE_UP),('sampledxy',(255,255,255),cv2.MARKER_TILTED_CROSS)]:
                    if np.isfinite(b[key][i]).all():cv2.drawMarker(img,tuple(np.rint(b[key][i]).astype(int)),color,marker,19,2)
            cv2.rectangle(img,(0,0),(img.shape[1],65),(15,15,15),-1)
            for j,line in enumerate([f'{rec["content_id"]} / {stage} / frame {d["frames"][i]}',
                'Yellow: target mean; Green: DLC wrist neighborhood r=50; Blue: gaze',
                'all_ABCD: magenta mean, red argmax, white independent sample']):
                cv2.putText(img,line,(10,18+20*j),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            cv2.imwrite(str(folder/f'{rec["content_id"]}_{stage.replace(" ","_")}.jpg'),img)
    finally:cap.release()


def analyze(out,args,rec,d,bundles,refs,trials,errors,confusions,coverage,diagnostics):
    meta={k:rec[k] for k in META};paired=set(bundles)==set(MODELS)
    model_ok=np.zeros(len(d['frames']),bool)
    if paired:
        model_ok=np.logical_and.reduce([b['valid_maps'] & b['valid_samples'].all(axis=1) &
            np.logical_and.reduce([np.isfinite(b[key]).all(axis=1) for key in DECODERS.values()]) for b in bundles.values()])
    geometry=d['included']&(d['count']>0)&d['gripper_ok']
    marks=dict(frame_number=d['frames'],timestamp_ns=d['timestamp_ns'],dt_s=d['dt'],stage=d['stage'],
        occurrence=d['occurrence'],valid_code=d['valid_code'],break_before=d['breaks'],
        gaze_xy=d['gaze'],gaze_unmasked=d['gaze_unmasked'],smooth_xy=d['smooth_xy'],
        interpolated=d['interpolated'],target_slots=d['slots'],target_xy=d['z'],target_count=d['count'],
        target_slotmask=d['slotmask'],slotmask=d['slotmask'],label_mode=d['label_mode'],
        identity_change=d['identity_change'],left_xy=d['left'],right_xy=d['right'],likelihood=d['likelihood'],
        included=d['included'],resolution=np.array([d['width'],d['height']]))
    for name,b in bundles.items():
        for rule,key in DECODERS.items():marks[f'xy_{rule}_{name}']=b[key]
        marks[f'sample_index_{name}']=b['sample_index']
    area_cache={};q_cache={}
    for scenario,(rt,rp,codes,smooth) in SCENARIOS.items():
        xy,valid=support(d,scenario);pair=valid&model_ok
        marks['valid_real_'+scenario]=valid
        if paired:marks['valid_paired_'+scenario]=pair
        for stage,st in stage_masks(d):
            v=valid&st
            coverage.append(dict(meta,scenario=scenario,stage=stage,total_n=int(st.sum()),total_s=d['dt'][st].sum(),
                target_n=int((st&(d['count']>0)).sum()),gaze_n=int((st&np.isfinite(xy).all(axis=1)&(np.isin(d['valid_code'],codes) if not smooth else True)).sum()),
                dlc_geometry_n=int((st&d['gripper_ok']).sum()),valid_n=int(v.sum()),valid_s=d['dt'][v].sum(),
                fraction=ratio(v.sum(),st.sum()),paired_n=int((pair&st).sum()),paired_s=d['dt'][pair&st].sum(),
                paired_available=paired,interpolated_fraction=ratio(np.sum(v&d['interpolated']) if smooth else 0,v.sum()),
                psm_definition='bilateral_DLC_wrist_neighborhood',target_definition='mean_existing_target_slots',
                target_coverage_fraction=ratio(np.sum(st&(d['count']>0)),st.sum()),
                semantic_quality_flags=semantic_flags(stage)))
        if args.audit_only:continue
        if (rt,rp) not in area_cache:
            area_cache[rt,rp]=geo.areas(d['z'],d['left'],d['right'],d['width'],d['height'],rt,rp,geometry)
        area=area_cache[rt,rp]
        real=geo.classify(xy,d['z'],d['left'],d['right'],rt,rp,valid)
        realq=onehot(real);marks['state_gaze_'+scenario]=real
        if paired:
            if (rt,rp) not in q_cache:
                q_cache[rt,rp]={name:geo.sample_region_probabilities(b['samples_xy'],d['z'],d['left'],d['right'],rt,rp,geometry&model_ok) for name,b in bundles.items()}
            modelq=q_cache[rt,rp]
            for name,q in modelq.items():marks[f'q_{name}_{scenario}']=np.where(pair[:,None],q,np.nan).astype(np.float32)
            # Evaluation-GT kernels are never used to train or choose a decoder.
            gt=np.full((len(xy),4),np.nan)
            ids=np.flatnonzero(pair)
            for start in range(0,len(ids),512):
                ix=ids[start:start+512]
                maps=geo.gaussian_maps(xy[ix],d['width'],d['height'])
                gt[ix]=geo.map_region_probabilities(maps,d['z'][ix],d['left'][ix],d['right'][ix],d['width'],d['height'],rt,rp,np.ones(len(ix),bool))
            center=geo.classify(np.broadcast_to([d['width']/2,d['height']/2],xy.shape),d['z'],d['left'],d['right'],rt,rp,pair)
            centerq=onehot(center)
        for stage,st in stage_masks(d):
            base=dict(meta,scenario=scenario,stage=stage)
            # The real_all branch is duplicated deliberately so every branch has
            # its independently labeled, directly comparable gaze reference.
            for branch in ['probability_mc']+['decoded_'+s for s in DECODERS]:
                gs=sum_probability(realq,valid&st,area,d) if branch=='probability_mc' else summarize(real,valid&st,area,d)
                trials.append(dict(base,analysis_branch=branch,scope='real_all',source='gaze',**gs))
            if not paired:continue
            ok=pair&st
            ref=refs.get((rec['task'],scenario,stage))
            if ref is None:
                raise ValueError(f'No independent training reference for {rec["task"]}/{scenario}/{stage}')
            trq=geo.map_region_probabilities(ref['heatmap'],d['z'],d['left'],d['right'],d['width'],d['height'],rt,rp,ok)
            # Only the two event marginal probabilities enter train_rate Brier.
            rateq=np.tile([0,ref['rate_target'],ref['rate_gripper'],0],(len(xy),1))
            refq={'center_point':centerq,'uniform_area':area,'train_heatmap':trq,'train_rate':rateq,'GT_smoothed':gt}
            diag=sum_probability(gt,ok,area,d)
            diagnostics.append(dict(base,reference='GT_smoothed',**diag))
            for branch in ['probability_mc']+['decoded_'+s for s in DECODERS]:
                isprob=branch=='probability_mc'
                gs=sum_probability(realq,ok,area,d) if isprob else summarize(real,ok,area,d)
                owner=dict(base,analysis_branch=branch,scope='paired_test')
                trials.append(dict(owner,source='gaze',**gs))
                for name,b in bundles.items():
                    hard=None
                    if isprob:
                        q=modelq[name];ms=sum_probability(q,ok,area,d)
                    else:
                        rule=branch.removeprefix('decoded_')
                        hard=geo.classify(b[DECODERS[rule]],d['z'],d['left'],d['right'],rt,rp,pair)
                        q=onehot(hard);ms=summarize(hard,ok,area,d)
                        marks[f'state_{rule}_{name}_{scenario}']=np.where(pair,hard,-1).astype(np.int8)
                    source=dict(owner,source=name)
                    trials.append(dict(source,**ms))
                    error,matrix=compare(source,d,ok,real,q,gs,ms,refq,hard)
                    errors.append(error);confusions.extend(matrix)
    if not args.audit_only:
        np.savez_compressed(out/'frames'/f'{rec["content_id"]}.npz',**marks)
        point_overlay(out,rec,d,bundles)


def validate(trials,coverage):
    positive=trials[trials.valid_s>0]
    np.testing.assert_allclose(positive[['L_N','L_T','L_P','L_O']].sum(axis=1),1,atol=1e-10)
    np.testing.assert_allclose(positive[['area_N','area_T','area_P','area_O']].sum(axis=1),1,atol=1e-10)
    if ((trials.D.fillna(0)==0)!=trials.C.isna()).any():raise ValueError('C NA denominator mismatch')
    paired=trials[trials.scope=='paired_test']
    by=paired.groupby(['content_id','scenario','stage'])
    if len(paired) and ((by.valid_n.nunique()!=1).any() or (by.valid_s.nunique()!=1).any() or (by.size()!=24).any()):
        raise ValueError('Common support differs across model/decoder/probability branches')
    for metric in ['valid_n','valid_s','total_n','total_s']:
        a=coverage[coverage.stage=='whole'].set_index(['content_id','scenario'])[metric].sort_index()
        b=coverage[coverage.stage!='whole'].groupby(['content_id','scenario'])[metric].sum().sort_index()
        np.testing.assert_allclose(a,b,atol=1e-8)
    return dict(occupancy_conservation=True,area_conservation=True,NA_denominator_check=True,
        paired_frame_support_identical_across_branches=True,whole_equals_stage_exposures=True)


def semantic_flags(stage):
    flags=['DLC_point_is_wrist_neighborhood_not_validated_jaw_tip']
    if stage in ('transfer ring','place ring'):
        flags.append('target_role_not_assumed_ring_or_destination_peg')
    if stage=='wrap around psm':
        flags.append('reviewed_target_on_tool_target_vs_instrument_not_independent')
    if stage=='pull through and cinch':
        flags.append('sparse_target_labels_selection_limited')
    if stage=='threading string through tower':
        flags.append('multi_point_mean_object_semantics_requires_review')
    return ';'.join(flags)


def report(out,cfg,trials,errors,coverage):
    text=['# Target Locking v2：概率质量主分析与独立解码分析','',
        f'运行：`{out.name}`。主半径 target=gripper=50 px；主 gaze 状态为 2/3，3-only 与旧1/2/3规则另作敏感性。',
        '', '## 本次变化及评价边界','',
        '- 主分析直接统计每帧 K=64 个原始模型 samples 的四区域命中比例。它估计模型采样分布的质量，不将热图压成一个点，也不包含导出热图额外的 Gaussian 平滑。',
        '- 期望点、argmax 和固定随机种子的逐帧采样点均在本次相同区域、相同共同有效帧上重算。旧运行数值没有拼入新表。',
        '- K列在不同帧间没有联合轨迹身份。decoded_sample 的驻留/切换只是独立随机解码诊断，不代表模型预测分布的真实时间连贯性。',
        '- 原始DLC点被如实称为腕部/铰接邻域，不能冒称jaw tip或完整器械。target保留人工/跟踪槽位算术平均，不以未经确认的物体身份重新指定坐标。区域候选与语义审计另见附带文件。',
        '- 50 px仍是预设空间容差，当前没有独立眼动校准误差95百分位证据。不能把投影校准矩阵当作定位准确率验证。',
        '- train_heatmap直接保留并积分完整训练平均热图；train_rate仅由训练录像拟合。center_point是确定性点，uniform_area是均匀随机落点的软概率，二者不同。',
        '- GT_smoothed由真实gaze构建，是标签平滑的诊断参照，不是严格上界，不参与BSS排名。',
        '- 所有模型仅按留出录像评价，同人其他录像可在训练集。参与者bootstrap条件于当前模型；模型采样MC误差没有由参与者bootstrap完整覆盖。K=64单帧概率步长1/64，二项抽样标准差理论最大0.0625。',
        '', '## 主概率分析：全任务均值','',
        '| 集合 | 任务 | 来源 | Target | Gripper | Overlap | D | C |','|---|---|---|---:|---:|---:|---:|---:|']
    z=trials[(trials.analysis_branch=='probability_mc')&(trials.scenario=='main')&(trials.stage=='whole')&(trials.valid_s>0)]
    cols=['L_target','L_gripper','L_O','D','C']
    z=z.groupby(['scope','task','source','participant'])[cols].mean().groupby(level=[0,1,2]).mean()
    for key,row in z.iterrows():text.append('| '+' | '.join(key)+' | '+' | '.join(f'{x:.4f}' for x in row)+' |')
    text+=['','## 概率事件误差','', '|任务|模型|ΔTarget|ΔC|Target Brier|Gripper Brier|','|---|---|---:|---:|---:|---:|']
    if len(errors):
        z=errors[(errors.analysis_branch=='probability_mc')&(errors.scenario=='main')&(errors.stage=='whole')]
        cols=['delta_L_target','delta_C','BS_target','BS_gripper']
        z=z.groupby(['task','source','participant'])[cols].mean().groupby(level=[0,1]).mean()
        for key,row in z.iterrows():text.append('| '+' | '.join(key)+' | '+' | '.join(f'{x:.4f}' for x in row)+' |')
    text+=['','## 结果导航','',
        '- [主概率分析](main_probability/)：参与者/阶段/模型比较、表现相关、bootstrap和图件。',
        '- [期望点解码](decoded_expectation/) · [argmax解码](decoded_argmax/) · [随机采样点解码](decoded_sample/)。',
        '- [覆盖](coverage.csv) · [来源审计](recording_audit.csv) · [预测审计](prediction_audit.csv) · [训练参照](baseline_input_audit.json)。',
        '- [GT平滑诊断](gt_smoothed_occupancy.csv) · [验证](validation.json) · [参数和代码哈希](run_config.json)。',
        '', '原始与敏感性结果必须按analysis_branch/scenario分开使用；不得用旧运行均值替代新运行gaze对照。',
        '表现相关继续是小样本描述分析；特别是Task C测试分数仅两个不同取值，不据此主张技能识别泛化。']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def supersede(old,out):
    """Add provenance; preserve exact pre-change metadata/reports in an archive."""
    if not old or not (old/'run_config.json').is_file():return
    archive=old/'pre_supersession_snapshot'
    archive.mkdir(exist_ok=True)
    files=[old/'run_config.json']+list(old.glob('*.md'))
    original_hashes={}
    for file in files:
        backup=archive/file.name
        if not backup.exists():shutil.copy2(file,backup)
        original_hashes[file.name]=hashlib.sha256(backup.read_bytes()).hexdigest()
    cfg=json.loads((old/'run_config.json').read_text(encoding='utf-8-sig'))
    cfg.setdefault('original_status',cfg['status'])
    cfg.update(status='superseded',superseded_by=str(out),superseded_by_run_id=out.name,
        supersession_reason='new common support, probability primary and recomputed decoders',
        pre_supersession_sha256=original_hashes)
    dump_json(old/'run_config.json',cfg)
    banner=f'> **已被新运行替代（superseded）：** [{out.name}](../{out.name}/report.md)。本文件为旧定义/旧帧集结果，不与新版概率结果直接配对。原始文件保存在 `pre_supersession_snapshot/`。\n\n'
    for file in old.glob('*.md'):
        original=(archive/file.name).read_text(encoding='utf-8-sig')
        file.write_text(banner+original,encoding='utf-8-sig')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root',type=Path,default=Path('D:/Dataset/dvss'))
    parser.add_argument('--logs',type=Path,default=Path('C:/Users/GT502/Desktop/JMRR2027/DVSS_Pytorch/logs'))
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--content-id',action='append')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--legacy-cache',type=Path,default=DEFAULT_OLD)
    parser.add_argument('--supersedes',type=Path,default=DEFAULT_OLD)
    parser.add_argument('--no-supersede',action='store_true')
    parser.add_argument('--nboot',type=int,default=10000)
    parser.add_argument('--postprocess-only',type=Path)
    parser.add_argument('--region-review',type=Path,help='versioned mask candidate visual audit to include')
    args=parser.parse_args()
    if args.self_test:
        geo.self_test();geo.probability_self_test();return
    if args.postprocess_only:
        out=args.postprocess_only
        tables=[]
        for name in ('trial_stage.csv','model_trial_errors.csv','trial_confusions.csv'):
            try:tables.append(pd.read_csv(out/name))
            except pd.errors.EmptyDataError:tables.append(pd.DataFrame())
        finish(out,*tables,seed=42,nboot=args.nboot)
        return
    geo.self_test()
    geo.probability_self_test()
    args.repo=args.repo.resolve()
    base=args.output_root or args.repo/'outputs'/'target_locking_analysis'
    mode='audit' if args.audit_only else ('subset' if args.content_id else 'full')
    out=base/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_v2_'+mode)
    out.mkdir(parents=True,exist_ok=False)
    args.cache_dir=out/'cache';args.cache_dir.mkdir()
    if not args.audit_only:(out/'frames').mkdir()
    print(f'Output: {out}',flush=True)
    code=out/'code_snapshot';code.mkdir()
    hashes={}
    for name in ['analyze_target_locking.py','tl_inputs.py','tl_geometry.py','tl_stats.py','tl_common.py','tl_regions.py','tl_report.py']:
        file=Path(__file__).with_name(name)
        if file.is_file():shutil.copy2(file,code/name);hashes[name]=hashlib.sha256(file.read_bytes()).hexdigest()
    records=inputs.records(args.repo)
    selected=[r for r in records if not args.content_id or r['content_id'] in args.content_id]
    if not selected or (args.content_id and set(args.content_id)!={r['content_id'] for r in selected}):raise ValueError('Unknown content id')
    mapping=inputs.models(args.repo,args.logs,selected)
    cfg=dict(version=2,status='running',run_id=out.name,selected=[r['content_id'] for r in selected],
        stages=STAGES,scenarios=SCENARIOS,models=MODELS,bootstrap_seed=42,bootstrap_n=args.nboot,
        sample_decoder_seed=42,primary_analysis='probability_mc_from_64_unsmoothed_samples',
        decoder_rules=DECODERS,common_frame_rule='all five models all three decoders and all K samples valid',
        target_radius_px=50,psm_radius_px=50,psm_definition='DLC wrist/hinge neighborhood; not jaw tip',
        target_definition='arithmetic mean of existing valid slots; no unsupported semantic overrides',
        gaze_allowed_codes=[2,3],gaze_sensitivity_codes=[[3],[1,2,3]],
        clock='precise original-frame video_ts; not claimed hardware exposure time',
        probability_estimator='mean of exact disk membership across saved sample coordinates',
        grid_reference='area fractions on pixel-center Voronoi cells of inclusive endpoint grid nodes',
        supersedes=str(args.supersedes),code_sha256=hashes,repo=str(args.repo),data_root=str(args.data_root))
    dump_json(out/'run_config.json',cfg)
    refs={};train={};problems=[]
    if not args.audit_only:
        refs,train,failures=fit_refs(records,args,{r['task'] for r in selected if mapping.get(r['content_id'])},out)
        problems += [dict(x,source='reference') for x in failures]
    trials=[];errors=[];confusions=[];coverage=[];diagnostics=[];ra=[];ma=[];labels=[]
    for j,rec in enumerate(selected,1):
        cid=rec['content_id'];print(f'Analysis {j}/{len(selected)} {cid}',flush=True)
        try:d=prepare(rec,args)
        except (ValueError,OSError,KeyError) as exc:
            problems.append(dict(content_id=cid,source='recording',reason=str(exc)));continue
        drop=d['stage']=='drop ring'
        ra.append(dict(rec,**{k:v for k,v in d['audit'].items() if k not in rec},native_drop_ring_n=int(drop.sum()),
            native_drop_ring_s=float(d['dt'][drop].sum()),included_drop_ring_s=0.))
        bundles={}
        for model,entry in mapping.get(cid,{}).items():
            try:
                b=inputs.prediction_bundle(entry,d,trajectory_seed=42,cache_dir=args.cache_dir,
                    legacy_cache_dir=args.legacy_cache,include_probability=False)
                bundles[model]=b;ma.append(b['metadata'])
            except (ValueError,OSError,KeyError) as exc:
                problems.append(dict(content_id=cid,source=model,reason=str(exc)))
        if set(bundles)!=set(MODELS):bundles={}
        if bundles and cid in train.get(rec['task'],set()):raise ValueError('Test reference training overlap')
        grouped=pd.DataFrame(dict(stage=d['stage'],target_points=d['count'],label_mode=d['label_mode']))
        for (stage,count,label),n in grouped.groupby(['stage','target_points','label_mode']).size().items():
            labels.append(dict(content_id=cid,task=rec['task'],stage=stage,target_points=int(count),label_mode=label,frames=int(n)))
        analyze(out,args,rec,d,bundles,refs,trials,errors,confusions,coverage,diagnostics)
    for name,rows in [('recording_audit',ra),('prediction_audit',ma),('coverage',coverage),('target_labels',labels)]:save_rows(rows,out/(name+'.csv'))
    dump_json(out/'input_issues.json',problems)
    cfg.update(recordings_ok=len(ra),predictions_ok=len(ma),issues=len(problems))
    if not args.audit_only:
        t,e,c=map(pd.DataFrame,(trials,errors,confusions));cv=pd.DataFrame(coverage)
        validation=validate(t,cv)
        validation.update(recordings_read=len(ra),predictions_aligned=len(ma),recording_issues=len(problems),
            drop_ring_included_duration_s=0.,geometry_self_tests='pixel enumeration and Monte Carlo toy cases passed')
        for name,rows in [('trial_stage',trials),('model_trial_errors',errors),('trial_confusions',confusions),('gt_smoothed_occupancy',diagnostics)]:save_rows(rows,out/(name+'.csv'))
        dump_json(out/'validation.json',validation)
        print('Participant bootstrap and branch-separated figures',flush=True)
        stats=finish(out,t,e,c,seed=42,nboot=args.nboot)
        dump_json(out/'statistics_manifest.json',stats)
        report(out,cfg,t,e,cv)
        import tl_regions
        region_dir=out/'region_review';region_dir.mkdir(exist_ok=True)
        tl_regions.inventory(out/'frames',args.data_root,region_dir)
        # Visual review artifacts are versioned inputs of this code revision.
        reviewed=args.region_review or args.repo/'outputs'/'target_locking_region_review'/'v2_candidate_review_20261105'
        if not reviewed.is_dir():reviewed=Path(__file__).with_name('region_review')
        if reviewed.is_dir():
            shutil.copytree(reviewed,region_dir/'candidate_validation',dirs_exist_ok=True)
        cfg['region_validation']=dict(main='wrist50',mask_tip_enabled=False,
            reason='candidate failed visual acceptance; no unvalidated replacement',
            target_coordinate_overrides=0,semantic_flags='coverage.csv and region_review/occurrence_geometry_flags.csv')
        report_code=Path(__file__).with_name('tl_report.py')
        if report_code.is_file():
            from tl_report import build_detailed_report
            build_detailed_report(out)
            shutil.copy2(report_code,code/report_code.name)
            cfg['reporting_code_sha256']=hashlib.sha256(report_code.read_bytes()).hexdigest()
    cfg['status']='complete' if not problems else 'complete_with_input_issues'
    dump_json(out/'run_config.json',cfg)
    if not args.audit_only and not args.content_id and not problems and len(ra)==84 and len(ma)==50 and not args.no_supersede:
        supersede(args.supersedes,out)
    save_rows([dict(path=str(p.relative_to(out)),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()],out/'output_index.csv')
    print(f'{cfg["status"]}: {out}',flush=True)


if __name__=='__main__':main()
