"""Chinese offline report and scientific figures for observed passive gaze."""
from __future__ import annotations
import argparse, hashlib, html, json, platform, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_passive import STAGES, SCENARIOS, METRICS, save, sha, csvout

COLORS=['#d99725','#2878a7','#8161aa','#c8cdd3']
LABEL={'whole':'Whole','reach ring':'Reach','grab ring':'Grab','transfer ring':'Transfer','place ring':'Place',
       'reach string':'Reach','wrap around psm':'Wrap','pull through and cinch':'Cinch','threading string through tower':'Thread'}
def table(df,cols=None):
    if cols is not None:df=df[cols]
    return df.to_html(index=False,escape=True,na_rep='NA',float_format=lambda v:f'{v:.4f}',border=0)

def figures(out,group,coverage,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    results=[]
    def export(fig,name):
        fig.savefig(out/'figures'/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
        fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
        results.append(f'figures/{name}.png')
    for task in STAGES:
        stages=['whole',*STAGES[task]];labels=[LABEL[s] for s in stages];g=group[(group.task==task)&(group.scenario=='main')]
        def get(metric):return g[g.metric==metric].set_index('stage').reindex(stages)
        fig,ax=plt.subplots(figsize=(10,4.2));bottom=np.zeros(5)
        for m,c in zip(['L_T','L_P','L_O','L_N'],COLORS):
            values=get(m)['mean'].to_numpy()*100;ax.bar(labels,values,bottom=bottom,color=c,label=m);bottom+=np.nan_to_num(values)
        counts=get('L_T').n_participants.to_numpy()
        for i,n in enumerate(counts):ax.text(i,102,f'n={n}',ha='center',fontsize=9)
        ax.set_ylim(0,111);ax.set_ylabel('Equal-observer mean occupancy (%)');ax.set_title(f'Task {task} | observed passive gaze | main r=50/50 px')
        ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.1));export(fig,f'task_{task}_four_regions')
        for name,metrics,unit in [('occupancy',['L_target','L_gripper','C','D'],'Fraction'),
                                  ('area_adjustment',['C_excess','E_target','E_gripper','B_excess'],'Difference in fractions'),
                                  ('temporal',['direct_rate','via_rate','dwell_T_ms','dwell_P_ms','dwell_O_ms'],'')]:
            fig,axes=plt.subplots(1,len(metrics),figsize=(3.2*len(metrics),4),squeeze=False)
            for ax,m in zip(axes[0],metrics):
                q=get(m);x=np.arange(5);v=q['mean'].to_numpy();lo=q.ci_low.to_numpy();hi=q.ci_high.to_numpy()
                ax.errorbar(x,v,yerr=np.vstack((v-lo,hi-v)),fmt='o',color='#245779',capsize=3)
                ax.set_xticks(x,labels,rotation=40,ha='right');ax.set_title(m);ax.grid(axis='y',alpha=.2)
                if name=='area_adjustment':ax.axhline(0,color='gray',lw=.8)
                ax.set_ylabel(('per valid minute' if 'rate' in m else 'ms') if name=='temporal' else unit)
                for i,(v,n) in enumerate(zip(v,q.n_participants)):
                    if np.isfinite(v):ax.annotate(f'n={n}',(i,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=7)
                    else:ax.text(i,.03,'NA',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
            fig.suptitle(f'Task {task} | observed gaze | means and 95% participant bootstrap intervals',y=1.04)
            fig.tight_layout();export(fig,f'task_{task}_{name}')
        metrics=['L_target','L_gripper','C','C_excess','direct_rate','via_rate']
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for ax,m in zip(axes.flat,metrics):
            part=group[(group.task==task)&(group.metric==m)]
            data=part.pivot(index='scenario',columns='stage',values='mean').reindex(index=list(SCENARIOS),columns=stages).to_numpy()
            im=ax.imshow(np.ma.masked_invalid(data),aspect='auto',cmap='viridis')
            ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_yticks(range(5),list(SCENARIOS));ax.set_title(m)
            for i in range(5):
                for j in range(5):ax.text(j,i,'NA' if not np.isfinite(data[i,j]) else f'{data[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.2,edgecolor='none',pad=.6))
            fig.colorbar(im,ax=ax,fraction=.035)
        fig.suptitle(f'Task {task} | all five pre-specified sensitivities | equal-observer means')
        fig.tight_layout();export(fig,f'task_{task}_sensitivity')
    fig,ax=plt.subplots(figsize=(9,4))
    counts=[summary['total'],summary['AC'],summary['AC_all_paths_exist'],summary['novice_pair_AC']]
    ax.barh(['All JSON watches','Task A/C candidates','A/C with all linked files','A/C novice viewer + source'],counts,color=['#b7c5d0','#7ba4be','#4c829e','#245779'])
    for i,v in enumerate(counts):ax.text(v+1,i,str(v),va='center')
    ax.set_xlim(0,max(counts)*1.14);ax.set_xlabel('Watching records (file/metadata coverage, before frame validity)')
    export(fig,'input_coverage')
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--audit-dir',required=True);p.add_argument('--baseline',required=True);p.add_argument('--mapping',required=True);args=p.parse_args()
    out=Path(args.output);audit=Path(args.audit_dir);work=Path(__file__).resolve().parents[1];baseline=Path(args.baseline)
    regression_dir=work/'regression' if (work/'regression').exists() else work
    code_dir=work/'code' if (work/'code').exists() else work/'code_snapshot'
    group=pd.read_csv(out/'group_stage_summary.csv');trial=pd.read_csv(out/'trial_stage.csv');coverage=pd.read_csv(out/'coverage.csv');records=pd.read_csv(out/'recording_audit.csv')
    summary=json.loads((audit/'mapping_summary.json').read_text());details=json.loads((audit/'mapping_detail.json').read_text())
    for name in ['passive_mapping_audit.csv','input_issues.json','input_fingerprints.json','mapping_summary.json','mapping_detail.json','canonical_mapping_note.json','record_acceptance.json','audit_summary_final.json']:
        if (audit/name).exists() and (audit/name).resolve()!=(out/name).resolve():shutil.copy2(audit/name,out/name)
    (out/'mapping_frames').mkdir(exist_ok=True)
    mapping_frame_dir=audit/'mapping_frames' if (audit/'mapping_frames').exists() else audit/'frames'
    for source in mapping_frame_dir.glob('*.npz'):
        if source.resolve()!=(out/'mapping_frames'/source.name).resolve():shutil.copy2(source,out/'mapping_frames'/source.name)
    for name in ['baseline_observed_regression.json','baseline_observed_regression_rows.csv']:
        if (regression_dir/name).resolve()!=(out/name).resolve():shutil.copy2(regression_dir/name,out/name)
    regression_log=regression_dir/'regression_run.log' if (regression_dir/'regression_run.log').exists() else work/'logs/baseline_regression.log'
    if regression_log.resolve()!=(out/'logs/baseline_regression.log').resolve():shutil.copy2(regression_log,out/'logs/baseline_regression.log')
    for source in code_dir.glob('*.py'):
        if source.resolve()!=(out/'code_snapshot'/source.name).resolve():shutil.copy2(source,out/'code_snapshot'/source.name)
    if (code_dir/'core_extraction_provenance.json').resolve()!=(out/'code_snapshot/core_extraction_provenance.json').resolve():
        shutil.copy2(code_dir/'core_extraction_provenance.json',out/'code_snapshot/core_extraction_provenance.json')
    for source in (work/'regression').glob('*.py'):shutil.copy2(source,out/'validation_code'/source.name)
    for source in audit.glob('*.py'):shutil.copy2(source,out/'code_snapshot'/source.name)
    # More detailed original-frame lineage is kept separately from analysis frames.
    region=[]
    for r in records.to_dict('records'):
        with np.load(out/'frames'/(r['recording_id']+'.npz')) as z:
            slots=z['target_slots'];valid=np.isfinite(slots).all(2)&(slots>=0).all(2)&(slots<z['resolution']).all(2)
            dist=np.sqrt(((slots-z['target_xy'][:,None,:])**2).sum(2));spread=np.max(np.where(valid,dist,0),axis=1)
            for occurrence in np.unique(z['occurrence']):
                ids=np.flatnonzero(z['occurrence']==occurrence);mask=z['target_slotmask'][ids]
                region.append(dict(clip_key=r['clip_key'],recording_id=r['recording_id'],task=r['task'],stage=z['stage'][ids[0]],occurrence=int(occurrence),
                  frames=len(ids),target_frames=int((z['target_count'][ids]>0).sum()),slots_spread_over_50px_n=int((spread[ids]>50).sum()),
                  slot_bitmask_changes=int((mask[1:]!=mask[:-1]).sum()),explicit_identity_changes=int(z['identity_change'][ids].sum()),
                  included_s=float(z['dt_s'][ids][z['included'][ids]].sum())))
    csvout(out/'region_review/occurrence_geometry_flags.csv',region)
    figs=figures(out,group,coverage,summary)
    meta_status=dict(performance_correlations=dict(applicable=False,reason='JSON contains no passive observer-owned trial score and completion status'),
        performance_leave_one_out=dict(applicable=False,reason='No eligible observer-owned score-metric pairs'),
        participant_round_summary=dict(applicable=False,reason='No passive observer practice-round metadata; seq is viewing order and task/attempt describe source recording'),
        repetition_contrasts=dict(applicable=False,reason='No verified observer practice rounds 1/2/3'))
    save(out/'supplementary_applicability.json',meta_status)
    excluded=['five-model inputs','64 samples','probability MC','three model decoders','paired_test','model-common-frame restriction','model error and pair comparisons',
              'Brier/BSS','model confusion and ranking','center_point/train_heatmap/train_rate fitted references','GT_smoothed evaluation diagnostic']
    config=dict(version='Target Locking v3 observed-gaze passive adaptation',analysis_branch='observed_gaze',source='gaze',scope='real_all',
      status='complete_for_verified_stored_native_mapping',mapping_json=args.mapping,data_root='D:/Dataset/dvss',reference_run=str(baseline),
      output_dir='D:/GitHub_Project/co-tracker/outputs/target_locking_analysis/passive',stages=STAGES,scenarios=SCENARIOS,
      participant='JSON meta.observer',source_operator='JSON meta.user',eligibility='task_name in A/C and observer_skill=novice and src_skill=novice; all splits',
      native_clock='exact passive video_ts joined by original Frame_Number; verified identical inherited source acquisition timeline',
      physical_display_timing='not independently established; raw VID_FRAME has different semantics and is not a verified display counter',
      coordinate_transform=dict(stored_image_size=[1280,1024],gaze='normalized stored xy multiplied by [1280,1024]',target_and_DLC='native pixels, no scaling',
                                media_transform='single image; decoded passive/source image geometry consistent; no stereo-panel offset or crop applied',
                                external_display_viewport='not independently verified from acquisition setup'),
      gaze_quality_policy='finite normalized xy in inclusive [0,1], endpoints retained; valid/Status ignored',
      annotation_bounds='finite x/y within [0,W) x [0,H)',target_definition='arithmetic mean of existing valid point_1..point_5 slots',
      psm_definition='union of bilateral DLC wrist/hinge disks; both points must be in image; likelihood audit only',
      dt='consecutive original frames and 0<delta<=100ms: half interval to each endpoint before stage split; no extrapolation',
      bootstrap_seed=42,bootstrap_n=10000,aggregation='native dt within recording, equal recordings within observer, equal observers in group',
      removed_modules=excluded,supplementary_applicability=meta_status,recordings=len(records),observers=int(records.participant.nunique()),source_videos=int(records.source_content_id.nunique()),
      execution_environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
      code_sha256={p.name:sha(p) for p in (out/'code_snapshot').glob('*') if p.is_file()},created_utc=datetime.now(timezone.utc).isoformat())
    save(out/'run_config.json',config)
    save(out/'statistics_manifest.json',dict(analysis_branch='observed_gaze',source='gaze',scope='real_all',bootstrap_replicates=10000,bootstrap_seed=42,
      bootstrap_algorithm='unchanged ParticipantBootstrap; SeedSequence([42,n]), participant multinomial with replacement; 2.5/97.5 percentiles',
      missing_mean_interval_rule='n<2: no interval and bootstrap_defined=bootstrap_undefined=0 (not run), matching v3',
      metrics=METRICS,hierarchy=config['aggregation'],figure_files=figs,rows={n:len(pd.read_csv(out/(n+'.csv'))) for n in ['trial_stage','participant_stage_summary','group_stage_summary','coverage','recording_audit','target_labels','passive_mapping_audit']},
      supplementary=meta_status,no_p_values=True,no_added_group_comparisons=True))
    # Internal invariants are separate from independent raw-data reconstruction.
    positive=trial.valid_s>0
    occupancy=np.max(np.abs(trial.loc[positive,['L_T','L_P','L_O','L_N']].sum(axis=1)-1))
    area=np.max(np.abs(trial.loc[positive,['area_T','area_P','area_O','area_N']].sum(axis=1)-1))
    differences=[]
    for _,part in trial.groupby(['clip_key','scenario']):differences.append(abs(float(part.loc[part.stage=='whole','valid_s'].iloc[0])-float(part.loc[part.stage!='whole','valid_s'].sum())))
    rawdrop=sum(r['included_s'] for r in region if r['stage']=='drop ring')
    checks=dict(status='passed_for_stored_native_mapping',occupancy_conservation=bool(occupancy<1e-10),area_conservation=bool(area<1e-10),
       occupancy_max_error=float(occupancy),area_max_error=float(area),whole_equals_stage_exposures=bool(max(differences)<1e-9),whole_max_error_s=max(differences),
       drop_ring_included_duration_s=rawdrop,NA_denominator_check=bool(trial.loc[(trial.L_T+trial.L_P)==0,'C'].isna().all()),
       mapping_unique=bool(records.clip_key.is_unique and records.recording_id.is_unique),all_five_scenarios=set(trial.scenario)==set(SCENARIOS),
       all_analysis_rows_observed_only=bool((trial.analysis_branch=='observed_gaze').all() and (trial.source=='gaze').all() and (trial.scope=='real_all').all()),
       physical_display_clock_validation='not independently established; not claimed passed',external_display_viewport_validation='not independently established; not claimed passed')
    required_checks=['occupancy_conservation','area_conservation','whole_equals_stage_exposures','NA_denominator_check',
                     'mapping_unique','all_five_scenarios','all_analysis_rows_observed_only']
    checks['status']='passed_for_stored_native_mapping' if all(checks[k] for k in required_checks) and rawdrop==0 else 'failed_internal_invariants'
    save(out/'validation.json',checks)
    main=group[(group.scenario=='main') & (group.stage=='whole')]
    def number(task,metric):
        r=main[(main.task==task)&(main.metric==metric)].iloc[0];scale=100 if metric not in ['direct_rate','via_rate','dwell_T_ms','dwell_P_ms','dwell_O_ms'] else 1
        interval='NA' if not np.isfinite(r.ci_low) else f'{r.ci_low*scale:.2f}–{r.ci_high*scale:.2f}'
        return f'{r["mean"]*scale:.2f} [{interval}]，n={int(r.n_participants)}'
    findings=''.join(f'<p><b>Task {task}：</b>target占用（%）{number(task,"L_target")}；腕部邻域占用（%）{number(task,"L_gripper")}；C（%）{number(task,"C")}。直接切换率（次/有效分钟）{number(task,"direct_rate")}，经重叠切换率（次/有效分钟）{number(task,"via_rate")}。区间为参与者bootstrap 95%区间，n为该指标观察者数。</p>' for task in STAGES)
    covmain=coverage[(coverage.scenario=='main') & (coverage.stage!='whole')].groupby(['task','stage'],sort=False).agg(watches=('clip_key','nunique'),observers=('participant','nunique'),stage_frames=('total_n','sum'),target_frames=('target_n','sum'),gaze_frames=('gaze_coordinate_valid_n','sum'),DLC_frames=('bilateral_dlc_n','sum'),valid_frames=('valid_n','sum'),valid_seconds=('valid_s','sum')).reset_index()
    observed_coverage=coverage[(coverage.scenario=='main') & (coverage.stage!='whole') & (coverage.valid_s>0)].groupby(['task','stage'],sort=False).agg(valid_watches=('clip_key','nunique'),valid_observers=('participant','nunique')).reset_index()
    covmain=covmain.merge(observed_coverage,on=['task','stage'],how='left')
    covmain[['valid_watches','valid_observers']]=covmain[['valid_watches','valid_observers']].fillna(0).astype(int)
    covmain=covmain.rename(columns={'watches':'candidate_watches','observers':'candidate_observers'})
    covmain['valid_fraction']=covmain.valid_frames/covmain.stage_frames.replace(0,np.nan)
    csvout(out/'stage_coverage_summary.csv',covmain)
    comparison=pd.DataFrame([dict(recording=d['recording_id'],rows=d['clock_frames'],constructor_exact=d['constructor_exact'],raw_VID_FRAME_different=d['selected_raw_vid_frame_conflicts'],delta_min=d['selected_raw_vid_frame_delta_min'],delta_max=d['selected_raw_vid_frame_delta_max']) for d in details])
    whole=trial[(trial.scenario=='main')&(trial.stage=='whole')].groupby('task').agg(watches=('clip_key','nunique'),observers=('participant','nunique'),source_videos=('source_content_id','nunique'),valid_n=('valid_n','sum'),valid_s=('valid_s','sum')).reset_index()
    links=['trial_stage.csv','participant_stage_summary.csv','group_stage_summary.csv','coverage.csv','recording_audit.csv','target_labels.csv','passive_mapping_audit.csv','input_issues.json','run_config.json','statistics_manifest.json','validation.json','independent_validation.json','final_output_validation.json','output_index.csv','baseline_observed_regression.json','supplementary_applicability.json']
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Passive Target Locking v3 真实gaze分析</title><style>body{{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#1c2b36;max-width:1240px;margin:36px auto;padding:0 24px}}h1,h2{{color:#18465d}}.note{{background:#fff4d8;border-left:5px solid #bd8a20;padding:16px}}table{{border-collapse:collapse;font-size:13px;display:block;overflow:auto}}td,th{{border:1px solid #dce3e8;padding:6px 9px;text-align:right;white-space:nowrap}}th{{background:#edf2f5}}img{{max-width:100%;margin:16px 0}}a{{color:#17628c}}code{{background:#eef2f5}}</style><body>
    <h1>删除模型评测后的 Target Locking v3 真实 gaze 分析：Passive</h1>
    <p>仅使用JSON指定的passive观看者gaze。统一标识为analysis_branch=observed_gaze、source=gaze、scope=real_all。结果针对已经逐原帧键核实的现存NPY关联和其原生video_ts时间轴。</p>
    <div class="note"><b>时间与坐标解释边界：</b>已核实NPY、标注、video_ts、源视频和DLC的规范原帧关联，历史生成过程精确重现；passive video_ts继承源视频采集时间。原始CSV的VID_FRAME采用另一种时间/帧定义，其与NPY局部索引不同，不能据此断言物理播放错配，也不能宣称已验证实际显示时序。媒体均为1280×1024单视图，采用存储归一化坐标乘以图像宽高；实验现场屏幕viewport/播放显示延迟未获独立校准证据。本结果是现存数据规范映射下的分析，不构成硬件显示同步校验。</div>
    <h2>1. 候选、范围和实际覆盖</h2><p>权威JSON共122条观看记录：A35、C38、B33、D14、未知任务2；不纠正或猜测JSON中的任务。A/C候选73条，观看者技能为novice32、intermediate41；源操作者为novice45、intermediate28，二者不混用。45条A/C记录五项链接均存在。双novice筛选得到16条（A7、C9）、8名观看者、16个源视频。所有split均可纳入。A/C另外28条的stage、target及DLC链接缺失，且源技能不符合本次范围；逐项缺失重叠不能重复扣减。</p>
    {table(whole)}<img src="figures/input_coverage.png"><p>上述有效秒按规范原生video_ts计算，并非声称现场实际显示曝光时间。B/D及未知任务不进入分析；drop ring纳入时长严格为0。whole为四个纳入阶段的时间合并。</p>
    <h2>2. 主结果</h2>{findings}<p>高C只表示T/P可分辨部分偏向target，不能直接解释为高target总占用；须同时报告D、L_target和重叠O。这里不进行观察者间、Passive与Active、观看者与源操作者的统计比较。</p>
    <h2>3. 分阶段覆盖与区域定义</h2>{table(covmain)}<p>这些是各条件的边际帧数；有效分析要求阶段、gaze、至少一个target槽位和双侧DLC同时可用。低覆盖阶段只能代表被观测部分，缺失时段不补零。target取有效point_1至point_5的均值，以未取整中心半径50px分类；PSM取左右DLC腕部/铰接点圆盘并集，不代表完整器械或jaw tip。T/P/O/N互斥，不可用帧保持-1。像素面积在像素中心计数、画面边界裁剪并对重叠去重。</p>
    <p><b>低覆盖提醒：</b>Task C的cinch仅3条观看、3名观察者有有效占用；threading仅2条观看、1名观察者，因此该阶段不生成均值置信区间。驻留等指标还可能因没有达到120ms的段而具有更少的可估人数，必须读取逐指标n，不能把候选8人当作有效8人。</p>
    <h2>4. 时间、驻留和敏感性</h2><p>dt在阶段切分前计算：仅原帧连续且0&lt;Δt≤100ms的间隔两端各分一半；不向首尾外推，不跨缺帧补时长。occurrence、明确identity、缺帧/时间缺口和未纳入阶段切断序列。bitmask变化只审计，不视为物体身份切换。观察段≥120ms才纳入驻留中位数，达到阈值的截断段仍纳入并报告截断数。直接切换仅相邻合格T/P段；经重叠切换允许中间O不足120ms；不跨N或missing跳接。先记录内求段中位数，再在观察者内等权。</p>
    <p>主条件Task A共2次合格直接切换、0次经重叠切换；Task C共0次直接、1次经重叠切换，事件很少。A的合格T/P/O驻留段分别331/136/46，其中截断20/7/0；C分别141/441/133，其中截断0/9/0。组驻留中位数的均值约为191–220ms，但各状态覆盖人数不同。零切换是当前有效观察支持下按预定规则检出的计数，不代表不存在其他形式的注意转移。</p>
    <p>五条件全部运行：main 50/50、r25 25/25、r100 100/100、gripper100 50/100，均使用原始有效坐标；smooth为50/50，沿用原短缺口≤100ms插值、41Hz网格、居中3点平均及映回原生支持。主分析不重采样，target/DLC不平滑，smooth插补不进入main。不依据结果挑选半径。</p>
    <p>whole的敏感性显示：A的C在等半径25/50/100px条件为70.24%/72.04%/69.73%，smooth为73.03%；仅扩大PSM至100px后C为28.35%。C任务相应C为24.83%/25.29%/26.92%，smooth为25.66%，gripper100为3.19%。占用本身随区域半径明显变化；C任务C_excess从r25的−6.17个百分点变为r100的+3.82个百分点，不能脱离区域大小解释偏好。smooth的C任务via_rate为约0.176次/有效分钟，高于main的约0.018，稀少切换对预处理敏感，不据此选择最佳条件。</p>
    <h2>5. 图表：占用、面积修正、驻留/切换和敏感性</h2>
    {''.join('<figure><img src="'+f+'"><figcaption>'+html.escape(Path(f).stem)+'</figcaption></figure>' for f in figs if 'input_coverage' not in f)}
    <h2>6. 指标、层级统计与NA</h2><p>L_T+L_P+L_O+L_N=1；L_target=L_T+L_O；L_gripper=L_P+L_O；D=L_T+L_P；C=L_T/D；B=L_T−L_P。面积同样分为a_T/a_P/a_O/a_N；C_U=a_T/(a_T+a_P)；C_excess=C−C_U；E_target=L_target−a_T−a_O；E_gripper=L_gripper−a_P−a_O；B_excess=L_T−L_P−a_T+a_P。零分母为NA并保留原因，C不可估不影响其他指标。</p>
    <p>每观看记录内原生dt加权；观察者内有效记录等权平均；组内观察者等权平均。各指标独立保存人数和有效观看次数。Bootstrap沿用ParticipantBootstrap、10000次、seed=42及SeedSequence([42,n])，参与者有放回抽样，2.5/97.5百分位；不足2名可估观察者不生成均值区间，defined/undefined均为0表示未运行。无新增p值、置换、t检验或BH/FDR。</p>
    <h2>7. 输入、坐标和时间映射审计</h2><p>每个观看以完整clip键和带hash的唯一输出ID表示。stage/target/DLC仅来自JSON指定路径；缺失字串nan/null/none不当文件路径。stage与target可同NPY，读取的只是相应标注字段，从未读取该文件的gaze坐标作观看者gaze，也不使用源gaze有效性。双侧DLC局部索引由保留source video_ts的Frame_Number映射验证。</p>
    <p>真实gaze只从gaze_npy_path取gaze_x/gaze_y，两个坐标有限并在闭区间[0,1]，0/1保留；valid及原始Status不筛选。target和DLC采用半开图像边界。历史构造曾把超界原始坐标裁到端点，本次严格分析现存NPY，不反向剔除或恢复原坐标。已验证媒体尺寸和三个抽样位置的源/passive图像几何一致；不新增左右并排折叠、裁剪或比例变换。</p>
    {table(comparison)}<p>此表的差值为原始VID_FRAME减NPY局部行号，二者不可混作同一计数器。所有原始NPY行和原帧、stage/target行、DLC行及原始眼动样本行索引保存在frames与mapping_frames，保证追溯。overlays显示存储映射下的空间关系，不作为现场显示同步证据。</p>
    <h2>8. 不适用模块、验证和交付</h2><p>JSON未提供passive观看者本人、与基准语义一致的成绩/完成状态及练习轮次；因此成绩相关、leave-one-out、participant_round_summary和1→3/1→2/2→3变化不适用。源操作者成绩、技能和seq均未替代这些信息。</p>
    <p>代表性基准回归先行：4条原始录像、5条件、100阶段行、227015帧状态及4900项标量复现。独立验证重新读取原始passive NPY和链接标注，验证有效帧、占用、smooth、时间序列、层级平均和bootstrap。数值与原帧数据检查的状态，与未独立建立的物理播放/外部屏幕映射明确分开，不将后者标为通过。</p>
    <p>已删除模型预测输入、K样本、概率MC、decoder、paired_test、模型误差/比较/排名、Brier/BSS和混淆矩阵，不拟合center/train参照，不输出GT_smoothed评价诊断；保留uniform area解释和smooth敏感性。本次复现的是“删除模型评测后的Target Locking v3真实gaze分析”。</p>
    <ul>{''.join('<li><a href="'+n+'">'+n+'</a></li>' for n in links)}</ul>
    <p><a href="region_review/occurrence_geometry_flags.csv">Occurrence区域几何审计</a> · <a href="region_review/overlay_index.csv">叠加图索引</a> · <a href="input_fingerprints.json">输入指纹</a> · <a href="baseline_and_mapping_fingerprints_after.json">基准及JSON未改变验证</a></p>
    </body></html>'''
    (out/'detailed_report_zh.html').write_text(report,encoding='utf8')
    print(json.dumps(dict(figures=len(figs),trial_rows=len(trial),valid_seconds=float(trial[(trial.scenario=='main')&(trial.stage=='whole')].valid_s.sum())),ensure_ascii=False))

if __name__=='__main__':main()
