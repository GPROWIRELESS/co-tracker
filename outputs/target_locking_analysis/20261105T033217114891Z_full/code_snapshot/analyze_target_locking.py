"""Target-versus-gripper analysis on native video frames; run --help for paths."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import tl_geometry as geo
import tl_inputs as inputs
from tl_stats import finish


STAGES = {
    'A': ('reach ring', 'grab ring', 'transfer ring', 'place ring'),
    'C': ('reach string', 'wrap around psm', 'pull through and cinch',
          'threading string through tower'),
}
MODELS = ('all_ABCD', 'single_task', 'leave_out', 'spd_finetune', 'spd_zero_shot')
CODES = ('N', 'T', 'P', 'O')
# Each tuple is target radius, gripper radius, DLC cutoff, smoothed gaze.
SCENARIOS = {'main': (50,50,None,False), 'r25': (25,25,None,False),
             'r100': (100,100,None,False), 'gripper100': (50,100,None,False),
             'dlc90': (50,50,.9,False), 'smooth': (50,50,None,True)}
META = ('content_id','participant','task','task_num','attempt','round','score','status')


def dump_json(path, obj):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(str(type(value)))
    path.write_text(json.dumps(obj, default=convert, ensure_ascii=False, indent=2),
                    encoding='utf-8')


def save_rows(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig', na_rep='NA')


def in_image(xy, width, height):
    return np.isfinite(xy).all(axis=1) & (xy>=0).all(axis=1) & (xy<[width,height]).all(axis=1)


def prepare(record, args):
    d = inputs.read_record(record, args.data_root, args.repo)
    z, count, slots = geo.target_center(d['slots'], d['width'], d['height'])
    d.update(z=z, count=count, slotmask=slots)
    d['included'] = np.isin(d['stage'], STAGES[record['task']])
    d['gripper_ok'] = in_image(d['left'], d['width'], d['height']) & in_image(
        d['right'], d['width'], d['height'])
    d['dt'], d['breaks'] = geo.times(d['timestamp_ns'], d['frames'])
    d['breaks'] |= d['identity_change']
    d['breaks'][1:] |= d['occurrence'][1:] != d['occurrence'][:-1]
    # Even a single excluded-stage frame breaks an episode.
    d['breaks'] |= ~d['included']
    d['breaks'][1:] |= ~d['included'][:-1]
    d['smooth_xy'], d['interpolated'] = geo.smooth_gaze(
        d['timestamp_ns'], d['gaze'], d['occurrence'], d['breaks'])
    return d


def support(d, scenario):
    rt,rp,cut,smooth = SCENARIOS[scenario]
    xy = d['smooth_xy'] if smooth else d['gaze']
    mask = d['included'] & (d['count']>0) & d['gripper_ok'] & np.isfinite(xy).all(axis=1)
    if cut is not None:
        mask &= (d['likelihood']>=cut).all(axis=1)
    return xy,mask


def stage_masks(d):
    yield 'whole', d['included']
    for name in STAGES[d['task']]:
        yield name, d['stage']==name


def gaussian_mean_xy(xy, width, height):
    """Mean of each normalized, truncated sigma=1 grid-cell GT kernel (separable)."""
    ans = np.full_like(xy, np.nan, dtype=float)
    good = np.isfinite(xy).all(axis=1)
    for axis,(length,size) in enumerate(((40,width),(32,height))):
        grid = np.arange(length, dtype=float)
        center = xy[good,axis,None]*(length-1)/size
        p = np.exp(-.5*(grid-center)**2)
        ans[good,axis] = np.sum(p*grid,axis=1)/np.sum(p,axis=1)/(length-1)
    return ans


def fit_refs(records, args, tasks, out):
    """Fit common *evaluation references* only to the matching task's train split."""
    train = {task:inputs.train_ids(args.repo,task) for task in tasks}
    sums, ref_members, issues = {}, [], []
    for rec in records:
        task,cid = rec['task'],rec['content_id']
        if cid not in train.get(task,set()):
            continue
        try:
            d=prepare(rec,args)
        except (ValueError,OSError,KeyError) as exc:
            issues.append({'task':task,'content_id':cid,'reason':str(exc)})
            continue
        kernel = {False:gaussian_mean_xy(d['gaze'],d['width'],d['height']),
                  True:gaussian_mean_xy(d['smooth_xy'],d['width'],d['height'])}
        for scenario,(rt,rp,cut,sm) in SCENARIOS.items():
            xy,ok = support(d,scenario)
            h=geo.classify(xy,d['z'],d['left'],d['right'],rt,rp,ok)
            for stage,st in stage_masks(d):
                idx=ok & st
                if not idx.any():
                    continue
                w=d['dt'][idx]
                hits=h[idx]
                a=sums.setdefault((task,scenario,stage),np.zeros(5))
                a += [w.sum(), w[(hits&1)>0].sum(), w[(hits&2)>0].sum(),
                      np.dot(w,kernel[sm][idx,0]),np.dot(w,kernel[sm][idx,1])]
                ref_members.append(dict(task=task,scenario=scenario,stage=stage,content_id=cid,
                                        frames=int(idx.sum()),seconds=w.sum()))
    refs={}
    for key,a in sums.items():
        refs[key]=dict(seconds=a[0],rate_target=a[1]/a[0],rate_gripper=a[2]/a[0],
                       mean_u=a[3]/a[0],mean_v=a[4]/a[0])
    save_rows(ref_members,out/'baseline_training_members.csv')
    save_rows([dict(task=k[0],scenario=k[1],stage=k[2],**v) for k,v in refs.items()],
              out/'baseline_parameters.csv')
    dump_json(out/'baseline_input_audit.json',{'train_ids':{t:sorted(v) for t,v in train.items()},
              'unavailable':issues,'weight':'native valid frame duration pooled within task/scenario/stage',
              'mean_map':'normalized sigma=1 grid cell Gaussian labels, average then decode expectation',
              'scope':'common DVSS task training comparator, including when evaluating an external zero-shot model'})
    return refs,train,issues


def ratio(num,den):
    return num/den if den>0 else np.nan


def summarize(labels, ok, area, d):
    # Code -1 is never part of the occupancy denominator.
    idx=ok & (labels>=0)
    w=d['dt'][idx]
    seconds=w.sum()
    h=labels[idx]
    share=(np.bincount(h,weights=w,minlength=4)/seconds if seconds>0 else np.full(4,np.nan))
    baseline=((area[idx]*w[:,None]).sum(axis=0)/seconds if seconds>0 else np.full(4,np.nan))
    n,t,p,o=share
    an,at,ap,ao=baseline
    c=ratio(t,t+p)
    cu=ratio(at,at+ap)
    out=dict(valid_n=int(idx.sum()),valid_s=seconds,L_N=n,L_T=t,L_P=p,L_O=o,
             L_target=t+o,L_gripper=p+o,D=t+p,C=c,B=t-p,area_N=an,area_T=at,
             area_P=ap,area_O=ao,C_U=cu,C_excess=c-cu,E_target=t+o-at-ao,
             E_gripper=p+o-ap-ao,B_excess=t-p-at+ap,
             C_reason='' if np.isfinite(c) else ('zero_exclusive_occupancy' if seconds else 'no_valid_frames'))
    out.update(geo.temporal(np.where(idx,labels,-1),d['dt'],d['occurrence'],d['breaks']))
    return out


def compare(key, d, ok, g, m, gs, ms, area, center, train_center, ref):
    idx=ok & (g>=0) & (m>=0)
    w=d['dt'][idx]
    total=w.sum()
    row=dict(key,valid_s=total,valid_n=int(idx.sum()))
    for metric in ('C','L_target','L_gripper'):
        delta=ms[metric]-gs[metric]
        row['delta_'+metric]=delta
        row['abs_delta_'+metric]=abs(delta)
    for bit,name in ((1,'target'),(2,'gripper')):
        gh=(g[idx]&bit)>0
        mh=(m[idx]&bit)>0
        row['BS_'+name]=ratio(w[gh!=mh].sum(),total)
        for label,mask in [('FP',mh & ~gh),('FN',~mh & gh),('TP',mh & gh),('TN',~mh & ~gh)]:
            row[f'{label}_{name}_s']=w[mask].sum()
            row[f'{label}_{name}_n']=int(mask.sum())
        # Bit labels are only evaluated on common valid frames.
        trainp=(np.asarray((train_center[idx]&bit)>0,float) if ref else np.full(len(w),np.nan))
        preds={'center':(center[idx]&bit)>0,'uniform':area[idx,bit]+area[idx,3],
               'train_center':trainp,'train_rate':ref['rate_'+name] if ref else np.nan}
        for label,pred in preds.items():
            row[f'BS_{name}_{label}']=ratio(np.sum(w*(np.asarray(pred,float)-gh)**2),total)
    ids=g[idx].astype(int)*4+m[idx]
    counts=np.bincount(ids,minlength=16)
    weights=np.bincount(ids,weights=w,minlength=16)
    matrix=[dict(key,actual=CODES[j//4],predicted=CODES[j%4],frames=int(counts[j]),seconds=weights[j])
            for j in range(16)]
    return row,matrix


def overlays(out,rec,d,tracks):
    if rec['content_id'] not in ('User1_1_1','User1_5_2'):
        return
    target_dir=out/'overlays'
    target_dir.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(d['video']))
    try:
        for stage,mask in list(stage_masks(d))[1:]:
            ids=np.flatnonzero(mask & support(d,'main')[1])
            if not len(ids):
                continue
            i=int(ids[len(ids)//2])
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            success,img=cap.read()
            if not success:
                raise ValueError(f"cannot read overlay frame {i} from {d['video']}")
            for pos in d['slots'][i]:
                if in_image(pos[None],d['width'],d['height'])[0]:
                    cv2.circle(img,tuple(np.rint(pos).astype(int)),4,(0,230,255),-1)
            for key,col in [('z',(0,220,255)),('left',(0,210,0)),('right',(0,210,0)),('gaze',(255,170,0))]:
                pix=tuple(np.rint(d[key][i]).astype(int))
                if key!='gaze':
                    cv2.circle(img,pix,50,col,2)
                cv2.drawMarker(img,pix,col,cv2.MARKER_CROSS,16,2)
            if 'all_ABCD' in tracks and np.isfinite(tracks['all_ABCD'][i]).all():
                cv2.drawMarker(img,tuple(np.rint(tracks['all_ABCD'][i]).astype(int)),
                               (230,40,240),cv2.MARKER_DIAMOND,19,3)
            cv2.rectangle(img,(0,0),(img.shape[1],55),(15,15,15),-1)
            for j,line in enumerate((f"{rec['content_id']} / {stage} / native frame {d['frames'][i]}",
                                      "Yellow: target slots + mean   Green: grippers   Blue: gaze   Magenta: all_ABCD")):
                cv2.putText(img,line,(10,21+24*j),cv2.FONT_HERSHEY_SIMPLEX,.55,(250,250,250),1,cv2.LINE_AA)
            if not cv2.imwrite(str(target_dir/f"{rec['content_id']}_{stage.replace(' ','_')}.jpg"),img):
                raise OSError('overlay write failed')
    finally:
        cap.release()


def frame_data(d,tracks):
    return dict(frame_number=d['frames'],timestamp_ns=d['timestamp_ns'],dt_s=d['dt'],
                break_before=d['breaks'],stage=d['stage'],occurrence=d['occurrence'],
                identity_change=d['identity_change'],valid_code=d['valid_code'],
                gaze_xy=d['gaze'],smooth_xy=d['smooth_xy'],interpolated=d['interpolated'],
                target_slots=d['slots'],target_xy=d['z'],slotmask=d['slotmask'],
                target_count=d['count'],label_mode=d['label_mode'],left_xy=d['left'],
                right_xy=d['right'],likelihood=d['likelihood'],included=d['included'],
                resolution=np.array([d['width'],d['height']]),**{'xy_'+n:x for n,x in tracks.items()})


def analyze_record(out,args,rec,d,tracks,refs,trial_rows,error_rows,conf_rows,coverage):
    meta={k:rec[k] for k in META}
    marks=frame_data(d,tracks)
    paired=len(tracks)==len(MODELS)
    model_ok=np.logical_and.reduce([np.isfinite(x).all(axis=1) for x in tracks.values()]) if paired else np.zeros(len(d['frames']),bool)
    common_xy=(d['included'] & (d['count']>0) & d['gripper_ok'])
    area_cache={}
    for scenario,(rt,rp,cut,sm) in SCENARIOS.items():
        xy,valid=support(d,scenario)
        pair=valid & model_ok
        if np.any((valid | pair) & ~d['included']):
            raise ValueError('excluded stage contributes to an analysis mask')
        dlc_pass=d['gripper_ok'] if cut is None else d['gripper_ok'] & (d['likelihood']>=cut).all(axis=1)
        marks['valid_real_'+scenario]=valid
        if paired:
            marks['valid_paired_'+scenario]=pair
        for stage,st in stage_masks(d):
            raw=int(st.sum())
            v=valid&st
            coverage.append(dict(meta,scenario=scenario,stage=stage,total_n=raw,total_s=d['dt'][st].sum(),
                target_n=int((st&(d['count']>0)).sum()),gaze_n=int((st&np.isfinite(xy).all(axis=1)).sum()),
                dlc_geometry_n=int((st&d['gripper_ok']).sum()),dlc_n=int((st&dlc_pass).sum()),
                valid_n=int(v.sum()),valid_s=d['dt'][v].sum(),
                fraction=ratio(v.sum(),raw),paired_n=int((st&pair).sum()),
                paired_s=d['dt'][st&pair].sum(),paired_available=paired,
                interpolated_fraction=ratio(np.sum(v&d['interpolated']) if sm else 0,v.sum())))
        if args.audit_only:
            continue
        if (rt,rp) not in area_cache:
            area_cache[rt,rp]=geo.areas(d['z'],d['left'],d['right'],d['width'],d['height'],rt,rp,common_xy)
        area=area_cache[rt,rp]
        def label(points):
            return geo.classify(points,d['z'],d['left'],d['right'],rt,rp,valid)
        real=label(xy)
        marks['state_gaze_'+scenario]=real
        if paired:
            predictions={n:label(p) for n,p in tracks.items()}
            for n,cls in predictions.items():
                marks[f'state_{n}_{scenario}']=np.where(pair,cls,-1).astype(np.int8)
            center=label(np.broadcast_to([d['width']/2,d['height']/2],xy.shape))
        for stage,st in stage_masks(d):
            key=dict(meta,scenario=scenario,stage=stage)
            trial_rows.append(dict(key,scope='real_all',source='gaze',**summarize(real,valid&st,area,d)))
            if not paired:
                continue
            key.update(scope='paired_test')
            ok=pair&st
            gs=summarize(real,ok,area,d)
            trial_rows.append(dict(key,source='gaze',**gs))
            ref=refs.get((rec['task'],scenario,stage))
            tc=label(np.broadcast_to([ref['mean_u']*d['width'],ref['mean_v']*d['height']],xy.shape)) if ref else None
            for name,cls in predictions.items():
                ms=summarize(cls,ok,area,d)
                owner=dict(key,source=name)
                trial_rows.append(dict(owner,**ms))
                err,matrix=compare(owner,d,ok,real,cls,gs,ms,area,center,tc,ref)
                error_rows.append(err)
                conf_rows.extend(matrix)
    if not args.audit_only:
        np.savez_compressed(out/'frames'/f"{rec['content_id']}.npz",**marks)
        overlays(out,rec,d,tracks)


def check_tables(trials, coverage):
    positive=trials.valid_s>0
    shares=trials.loc[positive,['L_N','L_T','L_P','L_O']].to_numpy()
    if not np.allclose(shares.sum(axis=1),1,rtol=0,atol=1e-10):
        raise ValueError('four-way occupancy conservation failed')
    if not np.allclose(trials.loc[positive,['area_N','area_T','area_P','area_O']].sum(axis=1),1,
                       rtol=0,atol=1e-10):
        raise ValueError('four-way area conservation failed')
    if ((trials.D.fillna(0)==0) != trials.C.isna()).any():
        raise ValueError('undefined preference disagrees with exclusive denominator')
    paired=trials[trials.scope=='paired_test']
    groups=paired.groupby(['content_id','scenario','stage'])
    if len(paired) and ((groups.source.nunique()!=6).any() or (groups.valid_n.nunique()!=1).any()
                        or (groups.valid_s.nunique()!=1).any()):
        raise ValueError('paired sources differ in rows or support')
    key=['content_id','scenario']
    for metric in ['valid_n','total_n','valid_s','total_s']:
        whole=coverage[coverage.stage=='whole'].set_index(key)[metric].sort_index()
        parts=coverage[coverage.stage!='whole'].groupby(key)[metric].sum().sort_index()
        np.testing.assert_allclose(whole.to_numpy(),parts.to_numpy(),rtol=0,atol=1e-8)
    return {'occupancy_conservation':True,'area_conservation':True,'NA_denominator_check':True,
            'paired_frame_support_identical':True,'whole_equals_eight_eligible_stages':True,
            'excluded_drop_ring_duration_s':0.0}


def write_report(out,cfg,ct,cv,trials,errors,validation,stats):
    main=trials[(trials.scenario=='main')&(trials.stage=='whole')&(trials.source=='gaze')]
    table=main.groupby(['scope','task']).agg(people=('participant','nunique'),videos=('content_id','size'),
        valid_frames=('valid_n','sum'),valid_seconds=('valid_s','sum')).reset_index()
    text=['# Target Locking 分析结果','',
          f"本次成功读取 {len(ct)} 段录像及 {cfg['predictions_ok']} 份预测。主半径 50 px，独立单位为参与者。",
          '', '| 集合 | 任务 | 人数 | 录像数 | 有效帧 | 有效秒数 |','|---|---|---:|---:|---:|---:|']
    for row in table.itertuples():
        text.append(f'| {row.scope} | {row.task} | {row.people} | {row.videos} | {row.valid_frames} | {row.valid_seconds:.1f} |')
    text += ['', '## 主结果','',
        '表中 occupancy、D、C 均先在参与者内对可用 trial 等权，再对参与者等权。各列分别按可估计数据汇总。',
        '', '| 集合 | 任务 | 来源 | Target occupancy | Gripper occupancy | Overlap | D | C |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    p=trials[(trials.scenario=='main')&(trials.stage=='whole')&(trials.valid_s>0)]
    agg=p.groupby(['scope','task','source','participant'])[['L_target','L_gripper','L_O','D','C']].mean()
    for (scope,task,source),v in agg.groupby(level=[0,1,2]).mean().iterrows():
        text.append(f'| {scope} | {task} | {source} | '+' | '.join(f'{a:.4f}' for a in v)+' |')
    if len(errors):
        text+=['','## 十段测试录像上的模型误差','',
               '有符号误差按模型减真实 gaze 计算。Brier 为时间加权的二值事件不一致率。',
               '', '| 任务 | 模型 | ΔC | Target Brier | Gripper Brier |',
               '|---|---|---:|---:|---:|']
        e=errors[(errors.scenario=='main')&(errors.stage=='whole')]
        g=e.groupby(['task','source','participant'])[['delta_C','BS_target','BS_gripper']].mean()
        for (t,n),v in g.groupby(level=[0,1]).mean().iterrows():
            text.append(f'| {t} | {n} | '+' | '.join(f'{a:.4f}' for a in v)+' |')
    text+=['','## 阅读与复核','',
      '- 全量真人行为与共同测试录像分 scope 保存。模型评估为 held-out recordings，同一参与者的其他录像可在训练集中。',
      '- Target 为 NPY 有效点的算术平均，gripper 为双侧 DLC 跟踪点的并集。输入没有可用的独立目标身份字段时，驻留按连续阶段 occurrence 分段。图上位置重合时，C 仅表示空间偏好。',
      '- 50 px 是预设空间容差。标注来源、点数和覆盖率可核查；本次没有新增人工定位准确率测量。',
      '- 视频原时钟与帧号用于全部时长。超过 100 ms 或原始帧不连续时中断。中点法对相邻有效采样间隔各分配一半。观察到的 120 ms 以上驻留包含截断片段并单报截断计数。',
      '- uniform 面积参照与固定画面中心点是不同预测器。训练参照采用同任务 train 清单，与全部被比较模型使用同一评价标尺。训练空间热图为 sigma=1 网格单位的 gaze 核，按有效帧时长平均。',
      '- 完成及失败且有实际分数的 trials 用于 performance，缺失分数和零方差原因在表中标明。测试集每项任务最多五人，Spearman、bootstrap 区间和模型对比均作样本内描述。CI 条件于现有模型，未校正多重比较。',
      '- 对应 static、better-novice、新参与者留出及模型 repetition 配对的预测不在当前导出清单。真实 gaze 的轮次用相异 TaskNum 排序，先合并重试。',
      '', '## 输出导航','',
      '- [输入审计](recording_audit.csv) · [模型审计](prediction_audit.csv) · [覆盖表](coverage.csv) · [标注点数与来源](target_labels.csv)',
      '- [Trial 与阶段指标](trial_stage.csv) · [配对误差](model_trial_errors.csv) · [训练参照](baseline_parameters.csv)',
      '- [运行配置](run_config.json) · [完整性核验](validation.json) · [汇总与图件目录](output_index.csv)',
      '', '帧级 NPZ 的坐标为像素；坐标列顺序 (x,y)。状态码 -1 无效、0 other、1 target-only、2 gripper-only、3 overlap。valid_real / valid_paired 标记对应情景的分母。']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root',type=Path,default=Path('D:/Dataset/dvss'))
    parser.add_argument('--logs',type=Path,default=Path('C:/Users/GT502/Desktop/JMRR2027/DVSS_Pytorch/logs'))
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--content-id',action='append',help='restrict evaluation to this recording (repeatable)')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        geo.self_test()
        assert np.isnan(ratio(0,0))
        print('Numerical self-check passed.')
        return
    args.repo=args.repo.resolve()
    base=args.output_root or args.repo/'outputs'/'target_locking_analysis'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    mode='audit' if args.audit_only else ('subset' if args.content_id else 'full')
    out=base/f'{stamp}_{mode}'
    out.mkdir(parents=True,exist_ok=False)
    if not args.audit_only:
        (out/'frames').mkdir()
    print(f'Output: {out}',flush=True)
    records=inputs.records(args.repo)
    selected=[r for r in records if not args.content_id or r['content_id'] in args.content_id]
    if not selected or (args.content_id and set(args.content_id)!={r['content_id'] for r in selected}):
        raise ValueError('Unknown or non-novice A/C content id')
    mapping=inputs.models(args.repo,args.logs,selected)
    code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__).resolve().with_name(n) for n in
                  ('analyze_target_locking.py','tl_inputs.py','tl_geometry.py','tl_stats.py')]}
    cfg=dict(mode=mode,run_id=out.name,repo=str(args.repo),data_root=str(args.data_root),logs=str(args.logs),
             selected=[r['content_id'] for r in selected],stages=STAGES,scenarios=SCENARIOS,models=MODELS,
             bootstrap_seed=42,bootstrap_n=10000,dwell_min_ms=120,max_gap_ms=100,smooth_hz=41,
             code_sha256=code_hashes,valid_codes=[1,2,3],frame_clock='non-deleted native video timestamps',
             geometry_codes={str(i):n for i,n in enumerate(CODES)},status='running')
    dump_json(out/'run_config.json',cfg)
    refs={}
    baseline_issues=[]
    if not args.audit_only:
        tasks={r['task'] for r in selected if mapping.get(r['content_id'])}
        print('Preparing independent evaluation references from task training clips',flush=True)
        refs,train,baseline_issues=fit_refs(records,args,tasks,out)
        for r in selected:
            if mapping.get(r['content_id']) and r['content_id'] in train[r['task']]:
                raise ValueError(f"Evaluation recording overlaps reference training: {r['content_id']}")
    trials,errors,confusions,coverage=[],[],[],[]
    ra,ma,labels,problems=[],[],[],[dict(x,source='reference') for x in baseline_issues]
    for j,rec in enumerate(selected,1):
        cid=rec['content_id']
        print(f'{j}/{len(selected)} {cid}',flush=True)
        try:
            d=prepare(rec,args)
        except (ValueError,OSError,KeyError) as exc:
            problems.append(dict(content_id=cid,reason=str(exc),source='recording'))
            continue
        dropped = d['stage']=='drop ring'
        ra.append(dict(rec,**{k:v for k,v in d['audit'].items() if k not in rec},
                       native_drop_ring_n=int(dropped.sum()),native_drop_ring_s=float(d['dt'][dropped].sum()),
                       included_drop_ring_s=float(d['dt'][dropped & d['included']].sum())))
        tracks={}
        for name,entry in mapping.get(cid,{}).items():
            try:
                tracks[name],audit=inputs.model_track(entry,d)
                ma.append(audit)
            except (ValueError,OSError,KeyError) as exc:
                problems.append(dict(content_id=cid,reason=str(exc),source=name))
                ma.append(dict(entry,reason=str(exc),model_frame_match=False))
        # Partial groups never enter the paired denominator.
        if set(tracks)!=set(MODELS):
            tracks={}
        grouped=pd.DataFrame({'stage':d['stage'],'point_count':d['count'],'label_mode':d['label_mode']})
        for (s,n,m),v in grouped.groupby(['stage','point_count','label_mode'],dropna=False).size().items():
            labels.append(dict(content_id=cid,task=rec['task'],participant=rec['participant'],stage=s,
                               target_points=int(n),label_mode=m,frames=int(v),included=s in STAGES[rec['task']]))
        analyze_record(out,args,rec,d,tracks,refs,trials,errors,confusions,coverage)
    save_rows(ra,out/'recording_audit.csv')
    save_rows(ma,out/'prediction_audit.csv')
    save_rows(labels,out/'target_labels.csv')
    save_rows(coverage,out/'coverage.csv')
    dump_json(out/'input_issues.json',problems)
    cfg['predictions_ok']=sum(bool(v.get('model_frame_match')) for v in ma)
    if not args.audit_only:
        trial_df,error_df,conf_df=map(pd.DataFrame,(trials,errors,confusions))
        cv=pd.DataFrame(coverage)
        validation=check_tables(trial_df,cv)
        validation['excluded_drop_ring_duration_s']=float(sum(a['native_drop_ring_s'] for a in ra))
        validation['drop_ring_included_duration_s']=float(sum(a['included_drop_ring_s'] for a in ra))
        assert validation['drop_ring_included_duration_s']==0
        validation.update(recordings_read=len(ra),predictions_aligned=cfg['predictions_ok'],
                          recording_issues=len(problems),frame_files=len(list((out/'frames').glob('*.npz'))))
        save_rows(trials,out/'trial_stage.csv')
        save_rows(errors,out/'model_trial_errors.csv')
        save_rows(confusions,out/'trial_confusions.csv')
        print('Computing participant summaries, bootstrap intervals and figures',flush=True)
        stats=finish(out,trial_df,error_df,conf_df,seed=42,nboot=10000)
        dump_json(out/'statistics_manifest.json',stats)
        dump_json(out/'validation.json',validation)
        write_report(out,cfg,ra,cv,trial_df,error_df,validation,stats)
        save_rows([dict(path=str(p.relative_to(out)),bytes=p.stat().st_size) for p in sorted(out.rglob('*'))
                   if p.is_file()],out/'output_index.csv')
    cfg.update(status='complete' if not problems else 'complete_with_input_issues',
               recordings_ok=len(ra),issues=len(problems))
    dump_json(out/'run_config.json',cfg)
    print(f"{cfg['status']}: {out}",flush=True)


if __name__=='__main__':
    main()
