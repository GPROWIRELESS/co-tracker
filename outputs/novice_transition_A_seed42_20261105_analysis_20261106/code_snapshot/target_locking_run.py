"""Task-A transition models: heatmap target locking on audited shared inputs.

Reuses original tl_geometry and participant bootstrap. This coordinate-only revision rewrites the requested output directory in place.
Raw dataset and prediction files are read-only. All analysis paths written are below --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOOLS = Path('D:/GitHub_Project/co-tracker/tools')
sys.path.insert(0, str(TOOLS))
import tl_geometry as geo
from tl_stats import ParticipantBootstrap

STAGES = ('reach ring', 'grab ring', 'transfer ring', 'place ring')
MODELS = ('current_transition','future500_transition','future1000_transition','future1500_transition')
HORIZONS = dict(zip(MODELS, (0, 500, 1000, 1500)))
SCENARIOS = {'main': 50, 'r25': 25, 'r100': 100}
GAZE_VALIDITY = 'finite normalized x/y within inclusive [0,1]; valid code is diagnostic only'
OCC = ['L_N','L_T','L_P','L_O','L_target','L_psm','D','C','C_excess','E_target','E_psm']
TEMP = ['dwell_T_ms','dwell_P_ms','dwell_O_ms','direct_rate','via_rate']
GROUP = ['axis','model','branch','scenario','stage']


def dump(path, value):
    def default(v):
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default)+'\n', encoding='utf-8')


def save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig', na_rep='NA')


def ratio(a, b):
    return float(a/b) if b > 0 else np.nan


def frame_weights(timestamp_ns, anchor, occurrence, eligible=None):
    """Midpoint exposures on the declared stride-2 anchor grid, never across gaps.

    Existing times() demands adjacent original frames and cannot accept stride2.
    This adaptation admits exactly two source rows, no >100 ms interval, no
    occurrence boundary. Missing classifications cut runs, not frame clocks.
    """
    ts = np.asarray(timestamp_ns, dtype=np.int64)
    rows = np.asarray(anchor, dtype=np.int64)
    occ = np.asarray(occurrence)
    dt = np.zeros(len(ts), float)
    br = np.ones(len(ts), bool)
    if len(ts)>1:
        sec = np.diff(ts).astype(float)/1e9
        good = (np.diff(rows)==2)&(sec>0)&(sec<=.100)&(occ[1:]==occ[:-1])
        if eligible is not None:
            good &= np.asarray(eligible)[1:] & np.asarray(eligible)[:-1]
        dt[:-1] += np.where(good,sec*.5,0)
        dt[1:] += np.where(good,sec*.5,0)
        br[1:] = ~good
    return dt, br


def target_time_weights(timestamp_ns, source_anchor, occurrence, valid):
    """Target-time clock: source stride2 continuity and target-time <=100ms.

    Future nearest labels need not advance exactly two target rows. A repeated
    target timestamp breaks rather than acquiring a duplicated exposure.
    """
    return frame_weights(timestamp_ns, source_anchor, occurrence, valid)


def point_state(xy, z, left, right, radius, mask):
    return geo.classify(xy,z,left,right,radius,radius,mask)


def onehot(state):
    q = np.full((len(state),4),np.nan)
    good = np.asarray(state)>=0
    q[good] = np.eye(4)[np.asarray(state)[good]]
    return q


def summarize(q, valid, area, dt, hard=None, occurrence=None, breaks=None):
    mask = np.asarray(valid)&np.isfinite(q).all(axis=1)
    w = dt[mask]
    duration = w.sum()
    masses = w@q[mask]/duration if duration>0 else np.full(4,np.nan)
    areas = w@area[mask]/duration if duration>0 else np.full(4,np.nan)
    n,t,p,o = masses
    an,at,ap,ao = areas
    c, cu = ratio(t,t+p),ratio(at,at+ap)
    row = dict(valid_n=int(mask.sum()),valid_s=float(duration),L_N=n,L_T=t,L_P=p,L_O=o,
               L_target=t+o,L_psm=p+o,D=t+p,C=c,C_U=cu,C_excess=c-cu,
               E_target=t+o-at-ao,E_psm=p+o-ap-ao,
               area_N=an,area_T=at,area_P=ap,area_O=ao)
    if hard is not None:
        time = geo.temporal(np.where(mask,hard,-1),dt,occurrence,breaks)
        row.update({k:time[k] for k in TEMP})
        row.update({k:v for k,v in time.items() if k.endswith('_n') or k.endswith('_n_censored')})
    return row


def event_error(q, gt_state, valid, dt, model_summary, gt_summary, refs):
    mask = np.asarray(valid)&np.isfinite(q).all(axis=1)&(gt_state>=0)
    w=dt[mask]; den=w.sum()
    out=dict(valid_n=int(mask.sum()),valid_s=float(den))
    for metric in OCC+TEMP:
        if metric in model_summary and metric in gt_summary:
            out['delta_'+metric]=model_summary[metric]-gt_summary[metric]
    for bit,event in ((1,'target'),(2,'psm')):
        y=((gt_state[mask]&bit)>0).astype(float)
        prob=q[mask,bit]+q[mask,3]
        out['BS_'+event]=ratio(np.sum(w*(prob-y)**2),den)
        for name,r in refs.items():
            pr=r[mask,bit]+r[mask,3]
            out['BS_'+event+'_'+name]=ratio(np.sum(w*(pr-y)**2),den)
    return out


def paired_statistics(trials,errors,nboot):
    boot=ParticipantBootstrap(42,nboot)
    persons=[]; groups=[]; error_persons=[]; error_groups=[]; loo=[]
    for key,g in trials.groupby(GROUP+['source'],dropna=False,sort=True):
        common=dict(zip(GROUP+['source'],key))
        for person,p in g.groupby('participant'):
            row=dict(common,participant=person,n_recordings=len(p))
            row.update(p[[x for x in OCC+TEMP if x in p]].mean().to_dict())
            persons.append(row)
    person_df=pd.DataFrame(persons)
    for key,g in person_df.groupby(GROUP+['source'],dropna=False,sort=True):
        common=dict(zip(GROUP+['source'],key))
        for metric in OCC+TEMP:
            if metric not in g: continue
            v=g[metric].dropna().to_numpy(float)
            groups.append(dict(common,metric=metric,estimate=float(v.mean()) if len(v) else np.nan,
                               n_participants=len(v),**boot.mean(v)))
    metrics=[c for c in errors if c.startswith(('delta_','BS_'))]
    for key,g in errors.groupby(GROUP,dropna=False,sort=True):
        common=dict(zip(GROUP,key))
        by=g.groupby('participant')[metrics].mean()
        for person,row in by.iterrows(): error_persons.append(dict(common,participant=person,**row.to_dict()))
        for metric in metrics:
            v=by[metric].dropna()
            vals=v.to_numpy(float)
            error_groups.append(dict(common,metric=metric,estimate=float(vals.mean()) if len(vals) else np.nan,
                                    n_participants=len(vals),n_recordings=int(g[metric].notna().sum()),**boot.mean(vals)))
            for person in v.index:
                rest=v.drop(person)
                loo.append(dict(common,metric=metric,held_out=person,n_remaining=len(rest),
                                estimate=float(rest.mean()) if len(rest) else np.nan))
        for event in ('target','psm'):
            for reference in ('uniform_area','center_point','train_rate','train_heatmap'):
                pred,ref='BS_'+event,'BS_'+event+'_'+reference
                if ref not in by: continue
                good=by[[pred,ref]].dropna()
                a,b=good[pred].to_numpy(),good[ref].to_numpy()
                estimate=1-ratio(a.mean(),b.mean()) if len(a) else np.nan
                error_groups.append(dict(common,metric='BSS_'+event+'_'+reference,estimate=estimate,
                    n_participants=len(a),n_recordings=int(g[[pred,ref]].notna().all(axis=1).sum()),**boot.bss(a,b)))
    return person_df,pd.DataFrame(groups),pd.DataFrame(error_persons),pd.DataFrame(error_groups),pd.DataFrame(loo)


def contrasts(errors,nboot):
    """Only compare models directly on the common anchor-time support."""
    boot=ParticipantBootstrap(42,nboot); out=[]
    a=errors[errors.axis.eq('anchor')]
    for keys,g in a.groupby(['branch','scenario','stage']):
        for metric in ('BS_target','BS_psm','delta_L_target','delta_C'):
            pivot=g.pivot(index=['participant','content_id'],columns='model',values=metric)
            for i,m1 in enumerate(MODELS):
                for m2 in MODELS[i+1:]:
                    if m1 not in pivot or m2 not in pivot:continue
                    diff=(pivot[m1]-pivot[m2]).dropna().groupby(level=0).mean()
                    vals=diff.to_numpy()
                    out.append(dict(zip(['branch','scenario','stage'],keys),axis='anchor',metric=metric,
                                    model_A=m1,model_B=m2,n_participants=len(vals),estimate_A_minus_B=vals.mean(),**boot.mean(vals)))
    return out


def figure_save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def draw_figures(out,groups,error_groups,persons):
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'font.weight':'normal','axes.titlesize':10,
        'axes.titleweight':'normal','axes.labelsize':10,'axes.labelweight':'normal',
        'xtick.labelsize':9,'ytick.labelsize':9,'figure.titlesize':12,'figure.titleweight':'normal',
        'legend.fontsize':9})
    colors=['#273c75','#0097a7','#d18d1e','#bd4f6c']
    labels=['Current','Future 500','Future 1000','Future 1500']
    primary=groups[(groups.axis=='anchor')&(groups.branch=='probability_map')&(groups.scenario=='main')]
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,metric,title in zip(axes,['L_target','L_psm','C'],['Target occupancy','PSM occupancy','Exclusive target preference C']):
        for i,m in enumerate(MODELS):
            row=primary[(primary.model==m)&(primary.source=='model')&(primary.stage=='whole')&(primary.metric==metric)].iloc[0]
            ax.errorbar(i,100*row.estimate,yerr=np.array([[row.estimate-row.ci_low],[row.ci_high-row.estimate]])*100,
                        fmt='o',color=colors[i],capsize=4)
            dots=persons[(persons.axis=='anchor')&(persons.model==m)&(persons.source=='model')&
                         (persons.branch=='probability_map')&(persons.scenario=='main')&(persons.stage=='whole')][metric]
            ax.scatter(np.full(len(dots),i)+np.linspace(-.07,.07,len(dots)),100*dots,s=17,color=colors[i],alpha=.4)
        gt=primary[(primary.model==MODELS[0])&(primary.source=='GT')&(primary.stage=='whole')&(primary.metric==metric)].iloc[0]
        ax.axhline(100*gt.estimate,color='black',label='Paired GT mean',lw=1.2,ls='--')
        ax.axhspan(100*gt.ci_low,100*gt.ci_high,color='black',alpha=.08)
        ax.set_xticks(range(4),labels,rotation=25,ha='right'); ax.set_title(title);ax.set_ylabel('%');ax.grid(axis='y',alpha=.18)
    axes[0].legend(fontsize=8);fig.suptitle('Task A | common anchor-time support | 4 participants, 5 recordings\nHeatmap probability integration; participant bootstrap 95% CI')
    figure_save(fig,out,'tl_anchor_occupancy')

    base=error_groups[(error_groups.axis=='anchor')&(error_groups.branch=='probability_map')&(error_groups.scenario=='main')]
    fig,axes=plt.subplots(1,2,figsize=(10.5,5),layout='constrained')
    for ax,metric,title in zip(axes,['delta_L_target','delta_C'],['Model − GT target occupancy','Model − GT preference C']):
        for i,m in enumerate(MODELS):
            row=base[(base.model==m)&(base.stage=='whole')&(base.metric==metric)].iloc[0]
            ax.errorbar(100*row.estimate,i,xerr=np.array([[row.estimate-row.ci_low],[row.ci_high-row.estimate]])*100,
                        fmt='o',color=colors[i],capsize=4)
        ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlabel('Percentage points');ax.set_title(title);ax.grid(axis='x',alpha=.18)
    fig.suptitle('Paired differences | participant bootstrap 95% CI')
    figure_save(fig,out,'tl_anchor_paired_differences')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    branches=['probability_map','decoded_expectation','decoded_argmax']
    for ax,event in zip(axes,['target','psm']):
        for branch,color in zip(branches,['#146c94','#c4702f','#8a4f9d']):
            sel=error_groups[(error_groups.axis=='anchor')&(error_groups.branch==branch)&
                (error_groups.scenario=='main')&(error_groups.stage=='whole')&(error_groups.metric=='BS_'+event)].set_index('model').reindex(MODELS)
            ax.plot(range(4),sel.estimate,'o-',label=branch.replace('decoded_','').replace('probability_map','heatmap'),color=color)
        ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_ylabel('Brier score (lower is better)');ax.set_title(event.capitalize()+' event');ax.grid(alpha=.18)
    axes[0].legend();fig.suptitle('Same paired frames | probability versus point decoding')
    figure_save(fig,out,'tl_decoding_brier')

    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
    for ax,metric,title in zip(axes,['L_target','C'],['Target occupancy','Exclusive target preference C']):
        vals=[];names=['GT']+labels
        for source,m in [('GT',MODELS[0])]+[('model',m) for m in MODELS]:
            vals.append([100*primary[(primary.model==m)&(primary.source==source)&(primary.stage==stage)&(primary.metric==metric)].iloc[0].estimate for stage in STAGES])
        im=ax.imshow(vals,aspect='auto',cmap='YlGnBu',vmin=0,vmax=100)
        for y,row in enumerate(vals):
            for x,v in enumerate(row):ax.text(x,y,f'{v:.1f}',ha='center',va='center',fontsize=9,color='white' if v>65 else 'black')
        ax.set_xticks(range(4),['Reach','Grab','Transfer','Place']);ax.set_yticks(range(5),names);ax.set_title(title+' (%)')
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.02)
    fig.suptitle('Stage-specific participant means | main anchor-time analysis')
    figure_save(fig,out,'tl_stage_heatmap')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,metric,title in zip(axes,['delta_L_target','BS_target'],['Model − paired GT target occupancy','Target Brier']):
        for axis,c in [('anchor','#276a9a'),('target','#b0633e')]:
            sel=error_groups[(error_groups.axis==axis)&(error_groups.branch=='probability_map')&
                    (error_groups.scenario=='main')&(error_groups.stage=='whole')&(error_groups.metric==metric)].set_index('model').reindex(MODELS)
            scale=100 if metric.startswith('delta_') else 1
            ax.plot(range(4),scale*sel.estimate,'o-',label=axis+' time',color=c)
        ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_title(title);ax.set_ylabel('pp' if metric.startswith('delta_') else 'Brier');ax.grid(alpha=.18)
    axes[0].legend();fig.suptitle('Secondary target-time evaluation uses each horizon’s future GT and geometry\nDifferent horizons have different matched target-time support')
    figure_save(fig,out,'tl_target_time_secondary')


def self_checks():
    geo.probability_self_test()
    # Both normalized endpoints are accepted; out-of-range/nonfinite coordinates are absent.
    test=np.array([[0,0],[1280,1024],[640,512],[-1,512],[1281,512],[640,1025],[np.nan,512],[np.inf,0]],float)
    np.testing.assert_array_equal(gaze_coordinate_valid(test,1280,1024),[True,True,True,False,False,False,False,False])
    dt,br=frame_weights(np.array([0,50000000,100000000,150000000]),[0,2,4,6],[0,0,0,0])
    np.testing.assert_allclose(dt,[.025,.050,.050,.025])
    assert not br[1:].any()
    dt,br=frame_weights(np.array([0,50000000,100000000,150000000]),[0,2,4,6],[0,0,1,1])
    np.testing.assert_allclose(dt,[.025,.025,.025,.025])
    dt,br=frame_weights(np.array([0,50000000,200000000]),[0,2,8],[0,0,0])
    np.testing.assert_allclose(dt,[.025,.025,0])


def independent_map_check(probability, z, left, right, width, height, radius, expected):
    """Enumerate image pixels independently of the reused scanline integrator."""
    yy,xx=np.mgrid[:height,:width]
    xx=xx.ravel()+.5;yy=yy.ravel()+.5
    h,w=probability.shape
    ix=np.floor(xx*(w-1)/width+.5).astype(int)
    iy=np.floor(yy*(h-1)/height+.5).astype(int)
    bins=iy*w+ix
    counts=np.bincount(bins,minlength=h*w)
    flat=probability.ravel().astype(float);flat/=flat.sum()
    weight=flat[bins]/counts[bins]
    target=(xx-z[0])**2+(yy-z[1])**2<=radius**2
    psm=((xx-left[0])**2+(yy-left[1])**2<=radius**2)|((xx-right[0])**2+(yy-right[1])**2<=radius**2)
    labels=target.astype(int)+2*psm.astype(int)
    actual=np.bincount(labels,weights=weight,minlength=4)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-9)
    return float(np.max(np.abs(actual-expected)))


def validate_saved(out,trials,errors):
    """Recompute stored trial aggregates and Brier from saved frame results."""
    count=0;pair_counts=[]
    for path in sorted((out/'frames').glob('*.npz')):
        f=dict(np.load(path,allow_pickle=False))
        cid=str(f['content_id']);axis=str(f['axis']);model=str(f['model'])
        pick=trials[(trials.content_id==cid)&(trials.axis==axis)&(trials.model==model)]
        for row in pick.itertuples(index=False):
            valid=f['valid_'+row.scenario].copy()
            if row.stage!='whole':valid&=f['stage']==row.stage
            w=f['dt_s'][valid];den=w.sum()
            assert row.valid_n==valid.sum()
            np.testing.assert_allclose(row.valid_s,den,atol=1e-10)
            if den<=0:continue
            state=f['gt_'+row.scenario]
            q=onehot(state) if row.source=='GT' else f['q_'+row.branch+'_'+row.scenario]
            aggregate=w@q[valid]/den
            np.testing.assert_allclose(aggregate,np.array([row.L_N,row.L_T,row.L_P,row.L_O]),atol=2e-8)
            if row.source=='model':
                e=errors[(errors.content_id==cid)&(errors.axis==axis)&(errors.model==model)&
                         (errors.branch==row.branch)&(errors.scenario==row.scenario)&(errors.stage==row.stage)].iloc[0]
                for bit,event in ((1,'target'),(2,'psm')):
                    obs=((state[valid]&bit)>0).astype(float)
                    pred=q[valid,bit]+q[valid,3]
                    bs=w@((pred-obs)**2)/den
                    np.testing.assert_allclose(bs,e['BS_'+event],atol=2e-8)
            count+=1
        if axis=='anchor':
            pair_counts.append((cid,model,{s:np.packbits(f['valid_'+s]).tobytes() for s in SCENARIOS}))
    for cid in set(x[0] for x in pair_counts):
        members=[x for x in pair_counts if x[0]==cid]
        assert len(members)==4
        assert all(x[2]==members[0][2] for x in members)
    assert set(trials.scenario.unique())==set(SCENARIOS)
    assert set(errors.scenario.unique())==set(SCENARIOS)
    for path in (out/'frames').glob('*.npz'):
        with np.load(path,allow_pickle=False) as z:
            assert not any('gaze3' in key for key in z.files),path
    positive=trials[trials.valid_s>0]
    np.testing.assert_allclose(positive[['L_N','L_T','L_P','L_O']].sum(axis=1),1,atol=2e-8)
    for _,g in trials.groupby(['content_id','axis','model','branch','scenario','source']):
        whole=g[g.stage=='whole'].iloc[0]
        parts=g[g.stage!='whole']
        np.testing.assert_allclose(whole.valid_s,parts.valid_s.sum(),atol=1e-8)
        assert whole.valid_n==parts.valid_n.sum()
    return dict(saved_trial_metric_recomputations=count,anchor_common_support_identical_all_models=True,
                four_region_probability_conservation=True,whole_equals_stage_exposure_sum=True,
                source_weighting_handles_stride2=True)


def report_findings(out,metadata,trials,errors,groups,error_groups,coverage,validation):
    select=lambda df,**kwargs: df.loc[np.logical_and.reduce([df[k].eq(v) for k,v in kwargs.items()])]
    primary=select(error_groups,axis='anchor',branch='probability_map',scenario='main',stage='whole')
    per_model={}
    for model in MODELS:
        per_model[model]={}
        for metric in ('delta_L_target','delta_L_psm','delta_C','BS_target','BS_psm'):
            row=select(primary,model=model,metric=metric).iloc[0]
            per_model[model][metric]={k:row[k] for k in ('estimate','ci_low','ci_high','n_participants')}
        gt=select(groups,axis='anchor',model=model,branch='probability_map',scenario='main',stage='whole',source='GT')
        modelrows=select(groups,axis='anchor',model=model,branch='probability_map',scenario='main',stage='whole',source='model')
        per_model[model]['GT']={r.metric:r.estimate for r in gt.itertuples() if r.metric in OCC}
        per_model[model]['model']={r.metric:r.estimate for r in modelrows.itertuples() if r.metric in OCC}
    cov=select(coverage,axis='anchor',model=MODELS[0],scenario='main',stage='whole')
    summary=dict(scope='Task A validation; 5 recordings, 4 participants; checkpoint-selected validation set',
        models=per_model,common_anchor_n=int(cov.valid_n.sum()),common_anchor_seconds=float(cov.valid_s.sum()),
        actual_participants=sorted(cov.participant.unique()),actual_recordings=sorted(cov.content_id.unique()),
        validation=validation,gaze_validity=GAZE_VALIDITY,valid_column_used_for_selection=False,
        limitations=['Validation was used for checkpoint selection; results are exploratory, not unseen-test generalization.',
          'PSM regions are DLC wrist/hinge neighborhoods, not validated jaw-tip or object masks.',
          'Target is the mean of in-image existing target point slots; target object identity is not re-annotated.',
          'Heatmap probability integration is a different representation from previous saved-sample Monte Carlo occupancy.',
          'Four participants support only coarse uncertainty estimates; leave-one-participant-out results are provided.',
          'Future model heatmaps refer to future-screen coordinates without motion registration; anchor-time comparisons test where an available cue lies now.',
          'Target-time comparisons use each horizon’s future GT and geometry and have different supports across horizons.',
          'Stored timelines do not independently establish hardware exposure/display latency.',
          'Temporal dwell/switch measures apply only to decoded trajectories and are sensitive to stride2 sampling.'])
    dump(out/'findings.json',summary)
    lines=['# Task A：新模型与GT的目标锁定比较','',
           '本目录已按用户要求原位重算，不备份旧结果。GT采用坐标有效规则，valid仅作诊断。主分析为**共同锚点时间**上的热图区域概率；未来时间匹配作为补充。样本为5段验证录像、4名参与者，User5贡献2段。该验证集参与checkpoint选择，因此结果属于探索性验证，不能写作未见测试集表现。','',
           f'主分析共同有效锚点共 **{summary["common_anchor_n"]:,}** 个，时间权重总计 **{summary["common_anchor_seconds"]:.2f}秒**。每录像先按真实时间加权，同一参与者录像等权，再对参与者等权；95% CI来自10,000次参与者bootstrap。','',
           '|模型|GT目标占用 %|模型目标占用 %|模型−GT pp [95% CI]|目标Brier|PSM Brier|',
           '|---|---:|---:|---:|---:|---:|']
    for model,info in per_model.items():
        d=info['delta_L_target']
        lines.append(f'|{model}|{info["GT"]["L_target"]*100:.2f}|{info["model"]["L_target"]*100:.2f}|{d["estimate"]*100:.2f} [{d["ci_low"]*100:.2f}, {d["ci_high"]*100:.2f}]|{info["BS_target"]["estimate"]:.4f}|{info["BS_psm"]["estimate"]:.4f}|')
    lines += ['', '区域定义：target为有效标注槽位均值周围50 px圆盘，PSM为左右DLC腕部/铰接点50 px圆盘并集。T/P/O/N分别表示目标独占、PSM独占、重叠、其他。L_target=T+O；L_psm=P+O；C=T/(T+P)，分母0时NA；D=T+P。C表示独占区域之间的分配，须结合D与绝对占用阅读。', '',
        '热图积分采用原tools的`map_region_probabilities`：保存的32×40网格值视为格内概率质量，在网格对应像素区域内均匀分布，按圆盘交集像素面积积分。没有生成额外随机样本。模型概率与GT二元区域事件的时间加权Brier越小越好；它不直接衡量完整注视分布是否正确。', '',
        'GT仅保留两个归一化坐标有限且位于闭区间[0,1]的点，不依据valid状态筛选；25/100 px半径为敏感性。原gaze3状态子集已移除，不再报告状态筛选结果。主PSM只要求双侧DLC坐标有限且位于画面内，不添加与旧target-locking不一致的隐含likelihood阈值。drop ring、none、空阶段排除。', '',
        '期望点/argmax轨迹用于补充区域占用、Brier与≥120 ms驻留/切换。概率主分支不定义唯一轨迹，不能据此计算驻留。驻留按采样观测时长，阶段、缺失、>100 ms间隔或非正常stride2间隔断开；接触观测边界的run计为删失并记录。', '',
        '原生`times()`要求连续原始帧，直接输入stride2会得到零时长；本次保留其相邻时间中点思想，显式允许source anchor差2，并禁止跨stage occurrence与缺帧间隔。未来时间轴按源时间加horizon计时，以35 ms容差内最近原生帧匹配GT/target/PSM；没有尾部钳位和motion registration。', '',
        '对未来模型，锚点分析比较“当前可展示提示”与当前GT/target/PSM；未来时间分析比较训练horizon对应的GT/target/PSM。后者每个horizon有自己的有效支持，因此不把其不同支持的占用均值当作模型直接配对比较。', '',
        '训练事件频率/训练平均热图没有在本次拟合。表中保留几何面积与中心点Brier作为诊断参照，不能解释为超过强训练基线。', '',
        '统计表：trial_metrics.csv、trial_errors.csv、participant_metrics.csv、group_metrics.csv、participant_errors.csv、group_errors.csv、leave_one_participant_out.csv、model_paired_contrasts.csv。逐帧可复核结果在frames/；覆盖与删失在coverage.csv及trial_metrics.csv；独立验证见validation.json。', '',
        '图件：figures/tl_anchor_occupancy、tl_anchor_paired_differences、tl_stage_heatmap、tl_decoding_brier、tl_target_time_secondary（PNG与PDF）。', '',
        '限制：本分析空间区域不是完整物体或夹爪尖端mask；target槽位均值保留原标注语义。四参与者的小样本不支持强泛化；提供逐人点与逐人剔除表。新热图积分结果不能直接替换旧MC样本占用数值。硬件曝光与显示延迟没有独立校准。']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    meta=json.loads((args.inputs/'metadata.json').read_text(encoding='utf-8-sig'))
    out=args.output
    (out/'frames').mkdir(exist_ok=True)
    trial_rows=[];error_rows=[];cover_rows=[];independent=[]
    for rec in meta['records']:
        cid=rec['content_id'];person=rec['participant']
        print('Target locking:',cid,flush=True)
        native=dict(np.load(args.inputs/'native'/f'{cid}.npz',allow_pickle=False))
        pred={m:dict(np.load(args.inputs/'predictions'/m/f'{cid}.npz',allow_pickle=False)) for m in MODELS}
        width,height=int(native['width']),int(native['height'])
        source=pred[MODELS[0]]['native_index'].astype(int)
        anchors=pred[MODELS[0]]['anchor'].astype(int)
        common_maps=np.ones(len(source),bool)
        for model,p in pred.items():
            np.testing.assert_array_equal(p['native_index'],source)
            np.testing.assert_array_equal(p['anchor'],anchors)
            common_maps &= p['valid_maps']&np.isfinite(p['expectedxy']).all(axis=1)&np.isfinite(p['argmaxxy']).all(axis=1)
        geometry_cache={}
        for axis in ('anchor','target'):
            for model,p in pred.items():
                ix=source if axis=='anchor' else p['target_native_index'].astype(int)
                eligible=np.ones(len(source),bool) if axis=='anchor' else p['target_time_valid'].astype(bool)
                assert np.all((ix>=0)&(ix<len(native['frames'])))
                stage=native['stage'][ix].astype(str);occ=native['occurrence'][ix]
                z=native['z'][ix];left=native['left'][ix];right=native['right'][ix]
                gaze=native['gaze_unmasked'][ix];valid_code=native['valid_code'][ix]
                ts=native['timestamp_ns'][source]+int(HORIZONS[model]*1_000_000 if axis=='target' else 0)
                dt,breaks=frame_weights(ts,anchors,occ,eligible)
                if 'identity_change' in native:breaks|=native['identity_change'][ix].astype(bool)
                included=np.isin(stage,STAGES)
                geometry=in_image(left,width,height)&in_image(right,width,height)&(native['count'][ix]>0)&included&eligible
                maps=common_maps if axis=='anchor' else p['valid_maps'].astype(bool)
                coordinate=gaze_coordinate_valid(gaze,width,height)
                frame=dict(content_id=np.asarray(cid),axis=np.asarray(axis),model=np.asarray(model),
                    source_anchor=anchors,source_native_index=source,geometry_native_index=ix,stage=stage,occurrence=occ,
                    dt_s=dt,timestamp_ns=ts,target_xy=z,left_xy=left,right_xy=right,gaze_xy=gaze,
                    expectedxy=p['expectedxy'],argmaxxy=p['argmaxxy'],valid_code=valid_code,
                    nominal_horizon_ms=np.asarray(HORIZONS[model]),target_time_valid=eligible)
                qcache={};area_cache={}
                for scenario,radius in SCENARIOS.items():
                    valid=geometry&maps&coordinate
                    # Pair every branch on exactly the same support within this model/axis.
                    valid&=np.isfinite(p['expectedxy']).all(axis=1)&np.isfinite(p['argmaxxy']).all(axis=1)
                    if radius not in area_cache:
                        key=(axis,model if axis=='target' else 'shared',radius)
                        if key not in geometry_cache:
                            geometry_cache[key]=geo.areas(z,left,right,width,height,radius,radius,geometry)
                        area_cache[radius]=geometry_cache[key]
                        qcache[radius]=geo.map_region_probabilities(p['probability'],z,left,right,width,height,radius,radius,geometry&maps)
                    area=area_cache[radius]
                    g=point_state(gaze,z,left,right,radius,valid)
                    gtq=onehot(g)
                    center=point_state(np.tile([width/2,height/2],(len(ix),1)),z,left,right,radius,valid)
                    refs={'uniform_area':area,'center_point':onehot(center)}
                    frame['valid_'+scenario]=valid;frame['gt_'+scenario]=g
                    states={'decoded_expectation':point_state(p['expectedxy'],z,left,right,radius,valid),
                            'decoded_argmax':point_state(p['argmaxxy'],z,left,right,radius,valid)}
                    qs={'probability_map':qcache[radius],**{b:onehot(s) for b,s in states.items()}}
                    for branch,q in qs.items():frame['q_'+branch+'_'+scenario]=np.where(valid[:,None],q,np.nan).astype(np.float32)
                    if scenario=='main':
                        good=np.flatnonzero(valid&(dt>0))
                        for sample in good[np.linspace(0,len(good)-1,min(2,len(good))).astype(int)] if len(good) else []:
                            err=independent_map_check(p['probability'][sample],z[sample],left[sample],right[sample],width,height,radius,qcache[radius][sample])
                            independent.append(dict(content_id=cid,axis=axis,model=model,anchor=int(anchors[sample]),max_abs_error=err))
                    for st in ('whole',)+STAGES:
                        stmask=included if st=='whole' else stage==st
                        support=valid&stmask
                        owner=dict(content_id=cid,participant=person,axis=axis,model=model,scenario=scenario,stage=st)
                        cover_rows.append(dict(owner,total_anchor_n=int(stmask.sum()),geometry_n=int((geometry&stmask).sum()),
                            gaze_n=int((coordinate&stmask).sum()),valid_n=int(support.sum()),
                            positive_duration_n=int((support&(dt>0)).sum()),valid_s=float(dt[support].sum()),
                            valid_fraction=ratio(support.sum(),stmask.sum())))
                        for branch,q in qs.items():
                            gs=summarize(gtq,support,area,dt,g if branch!='probability_map' else None,occ,breaks)
                            ms=summarize(q,support,area,dt,states.get(branch),occ,breaks)
                            trial_rows.append(dict(owner,branch=branch,source='GT',**gs))
                            trial_rows.append(dict(owner,branch=branch,source='model',**ms))
                            error_rows.append(dict(owner,branch=branch,**event_error(q,g,support,dt,ms,gs,refs)))
                np.savez_compressed(out/'frames'/f'{cid}__{axis}__{model}.npz',**frame)
    trials=pd.DataFrame(trial_rows);errors=pd.DataFrame(error_rows);coverage=pd.DataFrame(cover_rows)
    for name,table in [('trial_metrics',trials),('trial_errors',errors),('coverage',coverage)]:save(table,out/(name+'.csv'))
    print('Target locking: independent validation and participant statistics',flush=True)
    validation=validate_saved(out,trials,errors)
    validation.update(independent_full_pixel_probability_checks=len(independent),
        independent_max_error=max(x['max_abs_error'] for x in independent),
        native_input_files=len(meta['records']),source_models=len(MODELS),scenario_count=len(SCENARIOS),
        input_arrays_unmodified=True,heatmap_no_fabricated_samples=True,
        gaze_validity=GAZE_VALIDITY,valid_column_used_for_selection=False,removed_obsolete_scenarios=['gaze3'])
    save(independent,out/'independent_pixel_checks.csv');dump(out/'validation.json',validation)
    persons,groups,errorpersons,errorgroups,loo=paired_statistics(trials,errors,args.nboot)
    for name,table in [('participant_metrics',persons),('group_metrics',groups),('participant_errors',errorpersons),
                       ('group_errors',errorgroups),('leave_one_participant_out',loo)]:save(table,out/(name+'.csv'))
    save(contrasts(errors,args.nboot),out/'model_paired_contrasts.csv')
    draw_figures(out/'figures',groups,errorgroups,persons)
    report_findings(out,meta,trials,errors,groups,errorgroups,coverage,validation)
    dump(out/'run_config.json',dict(models=HORIZONS,stages=STAGES,scenarios=SCENARIOS,
        primary_axis='anchor',secondary_axis='target',primary_representation='pixel_area_heatmap_probability',
        bootstrap_seed=42,bootstrap_n=args.nboot,tools=str(TOOLS),input_root=str(args.inputs),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused={name:hashlib.sha256((TOOLS/name).read_bytes()).hexdigest() for name in ('tl_geometry.py','tl_stats.py')},
        output_status='complete',native_input_mutation=False,
        gaze_validity=GAZE_VALIDITY,valid_column_used_for_selection=False,
        removed_obsolete_scenarios=['gaze3'],update_mode='in_place_no_backup'))
    print('Target locking complete:',out,flush=True)


def gaze_coordinate_valid(points,width,height):
    """Inclusive normalized [0,1] eligibility; stored valid codes never enter it."""
    points=np.asarray(points,dtype=float)
    return np.isfinite(points).all(axis=1)&(points>=0).all(axis=1)&(points<=[width,height]).all(axis=1)


def in_image(points,width,height):
    return np.isfinite(points).all(axis=1)&(points>=0).all(axis=1)&(points<[width,height]).all(axis=1)


def gt_native_reference(args):
    """Separate full-native GT context; never substituted for paired model GT."""
    meta=json.loads((args.inputs/'metadata.json').read_text(encoding='utf-8-sig'))
    rows=[]
    for rec in meta['records']:
        cid=rec['content_id'];d=dict(np.load(args.inputs/'native'/f'{cid}.npz',allow_pickle=False))
        w,h=int(d['width']),int(d['height'])
        # Native frame number difference1 is mapped to declared step2 here;
        # timestamps and midpoints are untouched, including native missing gaps.
        dt,breaks=frame_weights(d['timestamp_ns'],d['frames']*2,d['occurrence'])
        included=np.isin(d['stage'],STAGES)
        ok=in_image(d['left'],w,h)&in_image(d['right'],w,h)&(d['count']>0)&included
        valid=ok&gaze_coordinate_valid(d['gaze_unmasked'],w,h)
        state=point_state(d['gaze_unmasked'],d['z'],d['left'],d['right'],50,valid)
        q=onehot(state);area=geo.areas(d['z'],d['left'],d['right'],w,h,50,50,ok)
        for stage in ('whole',)+STAGES:
            mask=valid&(included if stage=='whole' else d['stage']==stage)
            rows.append(dict(content_id=cid,participant=rec['participant'],stage=stage,
                scope='native_GT_reference_not_model_comparison',**summarize(q,mask,area,dt,state,d['occurrence'],breaks)))
    trial=pd.DataFrame(rows);save(trial,args.output/'gt_native_reference_bytrial.csv')
    persons=trial.groupby(['stage','participant'])[OCC+TEMP].mean().reset_index()
    save(persons,args.output/'gt_native_reference_byparticipant.csv')
    boot=ParticipantBootstrap(42,args.nboot);summary=[]
    for stage,g in persons.groupby('stage'):
        for metric in OCC+TEMP:
            v=g[metric].dropna().to_numpy(float)
            summary.append(dict(stage=stage,metric=metric,estimate=v.mean() if len(v) else np.nan,
                n_participants=len(v),scope='native_GT_reference_not_model_comparison',**boot.mean(v)))
    save(summary,args.output/'gt_native_reference.csv')
    print('Full-native GT reference complete.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nboot',type=int,default=10000)
    parser.add_argument('--native-reference-only',action='store_true')
    parser.add_argument('--figures-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.native_reference_only:
        gt_native_reference(args)
        return
    if args.figures_only:
        draw_figures(args.output/'figures',pd.read_csv(args.output/'group_metrics.csv'),
                     pd.read_csv(args.output/'group_errors.csv'),pd.read_csv(args.output/'participant_metrics.csv'))
        config=json.loads((args.output/'run_config.json').read_text(encoding='utf-8'))
        config['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        config['figure_revision']='constrained layout; numerical results unchanged'
        dump(args.output/'run_config.json',config)
        return
    self_checks()
    # Input schema adapter and execution are appended once shared audit is ready.
    run(args)
    gt_native_reference(args)


if __name__=='__main__':
    main()
