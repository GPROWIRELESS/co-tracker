"""Offline Chinese report and standalone scientific figures for the GT84 run."""
from __future__ import annotations
from gaze_psm_modes import mode_rows, normalize_modes
import argparse
import csv
import html
import json
import textwrap
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {'PSM in stage':'#2364a5', 'Both in stage':'#c16a21'}
MODE = {'PSM in stage':'PSM in stage', 'Both in stage':'Both in stage'}
SIDE = {'l':'Left instrument', 'r':'Right instrument'}
STATUS = {'insufficient_support':'支持不足', 'window_limited':'搜索窗限制',
          'positive_peak_resampling_range':'重抽样范围为正',
          'negative_peak_resampling_range':'重抽样范围为负',
          'direction_uncertain':'方向不确定'}
STAGES = {'A':['reach ring','grab ring','transfer ring','place ring','drop ring'],
          'C':['reach string','wrap around psm','threading string through tower','pull through and cinch']}
VARIANT = {'primary':'Primary','position_unsmoothed':'No smoothing',
           'position_smooth5':'Boxcar 5','velocity_box3':'Boxcar 3',
           'velocity_sg7':'SG 7','velocity_sg11':'SG 11',
           'trim_edges':'Trim edges','window4':'Window +/-4 s'}


def read_csv(path):
    if not Path(path).exists(): return []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(mode_rows(csv.DictReader(f)))


def read_json(path):
    return normalize_modes(json.loads(Path(path).read_text(encoding='utf8'))) if Path(path).exists() else {}


def num(value):
    try: return float(value)
    except (TypeError,ValueError): return np.nan


def boolean(value):
    return str(value).lower() in ('true','1','yes')


def fmt(value, digits=3):
    x = num(value)
    return f'{x:.{digits}f}' if np.isfinite(x) else '—'


def esc(value): return html.escape(str(value))


def _key(row):
    return tuple(row.get(k,'') for k in ('task','side','signal','stage','mode'))


def stage_order(rows, task=None, finite_only=False):
    pairs = {(r['task'],r['stage']) for r in rows if task is None or r['task']==task}
    if finite_only:
        pairs = {x for x in pairs if any((r['task'],r['stage'])==x and np.isfinite(num(r.get('peak_lag_s'))) for r in rows)}
    return sorted(pairs,key=lambda x:(x[0],STAGES.get(x[0],[]).index(x[1]) if x[1] in STAGES.get(x[0],[]) else 100,x[1]))


def label(pair):
    short={'threading string through tower':'threading string','pull through and cinch':'pull through / cinch',
           'wrap around psm':'wrap around'}
    return pair[0]+' · '+short.get(pair[1],pair[1])


def _save(fig, output, stem):
    folder=Path(output)/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(stem+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return stem


def plot_curves(output, summaries, curves, signal):
    rows=[r for r in summaries if r.get('variant')=='primary' and r['signal']==signal and r['side'] in SIDE]
    lookup=defaultdict(list)
    for r in curves:
        if r.get('variant')=='primary' and r['signal']==signal and r['side'] in SIDE:
            lookup[_key(r)].append(r)
    summary={_key(r):r for r in rows};made=[]
    for task in sorted({r['task'] for r in rows}):
        stages=stage_order(rows,task,True)
        if not stages: continue
        fig,axes=plt.subplots(len(stages),2,figsize=(12.5,2.55*len(stages)+1),squeeze=False,sharex=True)
        for i,(_,stage) in enumerate(stages):
            limits=[]
            for side in SIDE:
                for mode in MODE:
                    for r in lookup.get((task,side,signal,stage,mode),[]):
                        limits.extend(num(r.get(k)) for k in ('r','r_resample_low','r_resample_high'))
            limits=np.asarray(limits);limits=limits[np.isfinite(limits)]
            span=max(float(np.ptp(limits)),.03) if len(limits) else .1
            ymin,ymax=(float(limits.min())-.08*span,float(limits.max())+.1*span) if len(limits) else (-.1,.1)
            for j,side in enumerate(SIDE):
                ax=axes[i,j];support=[]
                for mode in MODE:
                    k=(task,side,signal,stage,mode);data=sorted(lookup.get(k,[]),key=lambda x:num(x['lag_s']))
                    sr=summary.get(k,{})
                    support.append(f"{MODE[mode]}: n={sr.get('n_stable_users','0')}, records={sr.get('n_complete_records','0')}")
                    if not data:continue
                    x=np.array([num(r['lag_s']) for r in data]); y=np.array([num(r['r']) for r in data])
                    lo=np.array([num(r.get('r_resample_low')) for r in data]);hi=np.array([num(r.get('r_resample_high')) for r in data])
                    if not np.isfinite(y).any():continue
                    ax.plot(x,y,c=COLORS[mode],lw=1.8,label=MODE[mode])
                    ax.fill_between(x,lo,hi,color=COLORS[mode],alpha=.14,lw=0)
                    px,py=num(sr.get('peak_lag_s')),num(sr.get('peak_r'))
                    if np.isfinite(px) and np.isfinite(py):
                        ax.plot(px,py,'D' if boolean(sr.get('peak_at_boundary')) else 'o',color=COLORS[mode],ms=4)
                ax.axvline(0,color='#8491a1',ls=':',lw=.8);ax.axhline(0,color='#bac2cb',lw=.7)
                ax.set_xlim(-2,2);ax.set_ylim(ymin,ymax);ax.grid(alpha=.15)
                ax.set_title(f'{stage}\n{SIDE[side]}',fontsize=10,pad=8)
                ax.text(.02,.04,'\n'.join(support),transform=ax.transAxes,fontsize=7,color='#475569',va='bottom',
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=1.5))
                if j==0:ax.set_ylabel('Correlation r')
                if i==len(stages)-1:ax.set_xlabel('Lag (s); positive = gaze earlier')
        handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=MODE[m]) for m in MODE]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False)
        fig.suptitle(f'Task {task} | {signal.capitalize()} | Fixed physical instruments',fontsize=14,y=.995)
        fig.text(.5,.007,'Shaded bands: participant percentile resampling ranges; exploratory, not calibrated confidence bands.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.025,1,.935),h_pad=1.2)
        made.append(_save(fig,output,f'primary_{signal}_curves_{task}'))
    return made


def plot_lags(output, summaries, signal):
    rows=[r for r in summaries if r.get('variant')=='primary' and r['signal']==signal and r['side'] in SIDE]
    stages=stage_order(rows,finite_only=True)
    if not stages:return None
    lookup={_key(r):r for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(12.5,max(4.8,.66*len(stages)+1.8)),sharey=True)
    for j,side in enumerate(SIDE):
        ax=axes[j]
        for i,(task,stage) in enumerate(stages):
            for mode,offset in [('PSM in stage',-.12),('Both in stage',.12)]:
                row=lookup.get((task,side,signal,stage,mode),{});peak=num(row.get('peak_lag_s'))
                if not np.isfinite(peak):continue
                y=i+offset;lo,hi=num(row.get('peak_lag_resample_low')),num(row.get('peak_lag_resample_high'))
                if np.isfinite(lo) and np.isfinite(hi):ax.plot([lo,hi],[y,y],c=COLORS[mode],lw=2,alpha=.7)
                ax.scatter(peak,y,color=COLORS[mode],marker='D' if boolean(row.get('peak_at_boundary')) else 'o',s=30,zorder=3)
        ax.axvline(0,c='#718096',ls=':',lw=1);ax.set_xlim(-2.12,2.12);ax.grid(axis='x',alpha=.2)
        ax.set_title(SIDE[side]);ax.set_xlabel('Group peak lag (s); positive = gaze earlier')
        ax.set_yticks(range(len(stages)),[label(s) for s in stages]);ax.set_ylim(len(stages)-.5,-.5)
    handles=[Line2D([0],[0],color=COLORS[m],marker='o',label=MODE[m]) for m in MODE]
    handles.append(Line2D([0],[0],color='#526173',marker='D',ls='none',label='Boundary point estimate'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False,fontsize=9)
    fig.suptitle(signal.capitalize()+' peak locations and exploratory resampling ranges',fontsize=14,y=.995)
    fig.text(.5,.01,'Bars: 2.5th–97.5th participant resampling percentiles. A positive correlation is not a test of positive lag.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.88))
    return _save(fig,output,f'primary_{signal}_peak_lags')


def plot_sensitivity(output, summaries, signal):
    primary=[r for r in summaries if r.get('variant')=='primary' and r['signal']==signal and r['mode']=='PSM in stage' and r['side'] in SIDE]
    stages=stage_order(primary,finite_only=True)
    if not stages:return None
    variants=['position_unsmoothed','position_smooth5','trim_edges','window4'] if signal=='position' else ['velocity_box3','velocity_sg7','velocity_sg11','trim_edges','window4']
    variants=[v for v in variants if any(r.get('variant')==v and r['signal']==signal for r in summaries)]
    if not variants:return None
    lookup={(r.get('variant'),*_key(r)):r for r in summaries}
    matrices=[]
    for side in SIDE:
        a=np.full((len(stages),len(variants)),np.nan)
        for i,(task,stage) in enumerate(stages):
            base=num(lookup.get(('primary',task,side,signal,stage,'PSM in stage'),{}).get('peak_lag_s'))
            for j,v in enumerate(variants):
                value=num(lookup.get((v,task,side,signal,stage,'PSM in stage'),{}).get('peak_lag_s'))
                a[i,j]=value-base
        matrices.append(a)
    finite=np.concatenate([a.ravel() for a in matrices]);finite=finite[np.isfinite(finite)]
    bound=max(.2,float(abs(finite).max())) if len(finite) else 1
    fig,axes=plt.subplots(1,2,figsize=(12.5,max(4.8,.52*len(stages)+1.5)),sharey=True)
    for ax,side,a in zip(axes,SIDE,matrices):
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e5e7eb')
        im=ax.imshow(a,cmap=cmap,vmin=-bound,vmax=bound,aspect='auto')
        ax.set_title(SIDE[side]);ax.set_xticks(range(len(variants)),[VARIANT[v] for v in variants],rotation=30,ha='right',fontsize=9)
        ax.set_yticks(range(len(stages)),[label(s) for s in stages])
        for i in range(len(stages)):
            for j in range(len(variants)):
                text=f'{a[i,j]:+.2f}' if np.isfinite(a[i,j]) else 'NA'
                ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if np.isfinite(a[i,j]) and abs(a[i,j])>.6*bound else '#172033')
    fig.suptitle(signal.capitalize()+' sensitivity | peak lag minus primary (s) | PSM in stage',fontsize=13,y=.995)
    fig.tight_layout(rect=(0,.04,.93,.94))
    cax=fig.add_axes([.945,.22,.015,.56]);fig.colorbar(im,cax=cax,label='Change in peak lag (s)')
    fig.text(.5,.01,'Gray cells: unsupported. These branches are descriptive checks, not alternatives selected for better results.',ha='center',fontsize=8)
    return _save(fig,output,f'{signal}_sensitivity_lag_change')


def plot_spatial(output, rows):
    stages=stage_order(rows)
    if not stages:return None
    lookup={(r['task'],r['side'],r['stage'],r['mode'],r['at']):r for r in rows}
    fig,axes=plt.subplots(2,2,figsize=(12.5,max(8,1.15*len(stages)+2.4)),sharey=True)
    for ri,(field,xlab) in enumerate([('distance_median_px','Mean of recording-median distances (px)'),('proximity_50px','Fraction within 50 pixels')]):
        values=np.array([num(r.get(field)) for r in rows]);values=values[np.isfinite(values)]
        right=max(float(values.max())*1.08,.01) if len(values) else 1.
        for j,side in enumerate(SIDE):
            ax=axes[ri,j]
            for i,(task,stage) in enumerate(stages):
                for mode,offset in [('PSM in stage',-.12),('Both in stage',.12)]:
                    a=num(lookup.get((task,side,stage,mode,'zero'),{}).get(field));b=num(lookup.get((task,side,stage,mode,'group_peak'),{}).get(field))
                    if np.isfinite(a) and np.isfinite(b):
                        ax.plot([a,b],[i+offset,i+offset],color=COLORS[mode],lw=1.3)
                        ax.scatter(a,i+offset,c='white',edgecolors=COLORS[mode],s=26,zorder=3)
                        ax.scatter(b,i+offset,c=COLORS[mode],marker='D',s=24,zorder=3)
            ax.set_title(SIDE[side]);ax.set_xlabel(xlab,fontsize=9);ax.grid(axis='x',alpha=.2)
            ax.set_yticks(range(len(stages)),[label(s) for s in stages]);ax.set_ylim(len(stages)-.5,-.5)
            ax.set_xlim(0,right)
    handles=[Line2D([0],[0],color=COLORS[m],label=MODE[m]) for m in MODE]
    handles += [Line2D([0],[0],color='#526173',marker='o',markerfacecolor='white',ls='none',label='Zero lag'),Line2D([0],[0],color='#526173',marker='D',ls='none',label='Group peak lag')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False,fontsize=8)
    fig.suptitle('Spatial correspondence | zero lag versus common group peak',y=.995,fontsize=14)
    fig.tight_layout(rect=(0,.025,1,.925))
    fig.text(.5,.006,'Record-equal within user, user-equal across users. 50 px is an exploratory radius, not a calibrated gaze-on-tool threshold.',ha='center',fontsize=8)
    return _save(fig,output,'spatial_zero_vs_peak')


def plot_group_bootstrap(output, signal):
    index=read_json(Path(output)/'group_bootstrap_index.json')
    path=Path(output)/'group_bootstrap_peak_lags.npz'
    if not isinstance(index,list) or not path.exists():return []
    selected=[r for r in index if r.get('variant')=='primary' and r.get('signal')==signal and r.get('side') in SIDE]
    lookup={_key(r):r for r in selected};made=[]
    with np.load(path,allow_pickle=False) as data:
        for task in sorted({r['task'] for r in selected}):
            stages=stage_order(selected,task)
            if not stages:continue
            fig,axes=plt.subplots(len(stages),2,figsize=(12.5,2.1*len(stages)+1),sharex=True,squeeze=False)
            for i,(_,stage) in enumerate(stages):
                for j,side in enumerate(SIDE):
                    ax=axes[i,j]
                    for mode in MODE:
                        row=lookup.get((task,side,signal,stage,mode))
                        if not row:continue
                        values=np.asarray(data[row['token']],float);values=values[np.isfinite(values)]
                        if not len(values):continue
                        ax.hist(values,bins=np.linspace(-2.05,2.05,42),weights=np.ones(len(values))/len(values),
                                histtype='step',color=COLORS[mode],linewidth=1.5,label=MODE[mode])
                    ax.axvline(0,color='#8491a1',ls=':',lw=.8);ax.set_xlim(-2.05,2.05);ax.grid(alpha=.15)
                    ax.set_title(f'{stage}\n{SIDE[side]}',fontsize=10)
                    if j==0:ax.set_ylabel('Fraction of draws')
                    if i==len(stages)-1:ax.set_xlabel('Group peak lag in participant resampling (s)')
            handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=MODE[m]) for m in MODE]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False)
            fig.suptitle(f'Task {task} | {signal.capitalize()} | Resampled group peak distribution',fontsize=14,y=.995)
            fig.text(.5,.008,'Each outline uses fractions summing to one. Boundary accumulation and separated peaks are retained.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.03,1,.93),h_pad=1.2)
            made.append(_save(fig,output,f'group_bootstrap_{signal}_{task}'))
    return made


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure(stem,caption):
    if not stem:return ''
    return f'<figure><a href="figures/{stem}.png"><img loading="lazy" src="figures/{stem}.png" alt="{esc(caption)}"></a><figcaption>{esc(caption)} · <a href="figures/{stem}.pdf">PDF</a> · <a href="figures/{stem}.png">PNG</a></figcaption></figure>'


def group_table(rows):
    sorted_rows=sorted(rows,key=lambda r:(stage_order(rows).index((r['task'],r['stage'])),r['side'],r['mode']))
    return table(['任务 / 阶段','器械','mode','用户 / 录制','峰值 r','r(0)','峰增益','峰位 / 秒','重抽样 2.5%–97.5%','边界频率','解释'],[
        [r['task']+' / '+r['stage'],SIDE.get(r['side'],r['side']),r['mode'],r.get('n_stable_users','')+' / '+r.get('n_complete_records',''),
         fmt(r.get('peak_r')),fmt(r.get('r_zero')),fmt(r.get('peak_gain_over_zero')),
         fmt(r.get('peak_lag_s')), '['+fmt(r.get('peak_lag_resample_low'))+', '+fmt(r.get('peak_lag_resample_high'))+']',
         fmt(r.get('bootstrap_boundary_frequency')),STATUS.get(r.get('lag_interpretation'),r.get('lag_interpretation',''))] for r in sorted_rows])


def generate_report(output):
    output=Path(output)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    manifest=read_json(output/'run_manifest.json');validation=read_json(output/'validation.json')
    groups=read_csv(output/'group_summary.csv');curves=read_csv(output/'group_curves.csv')
    primary=[r for r in groups if r.get('variant')=='primary' and r['side'] in SIDE]
    matched=read_csv(output/'matched_mode_summary.csv');spatial=read_csv(output/'spatial_summary.csv')
    contrasts=read_csv(output/'stage_contrasts.csv')
    selected=read_csv(output/'gt_selected_group_summary.csv');reference=read_csv(output/'unpaired_reference.csv')
    preprocessing=read_csv(output/'preprocessing_qc.csv')
    records=manifest.get('records',[]);n_records=len(records);n_users=len({r['user'] for r in records})
    finite=[r for r in primary if np.isfinite(num(r.get('peak_lag_s')))];boundary=[r for r in finite if r.get('lag_interpretation')=='window_limited']
    imgs={}
    for signal in ('position','velocity'):
        imgs[signal+'_curves']=plot_curves(output,groups,curves,signal)
        imgs[signal+'_lags']=plot_lags(output,groups,signal)
        imgs[signal+'_sensitivity']=plot_sensitivity(output,groups,signal)
        imgs[signal+'_bootstrap']=plot_group_bootstrap(output,signal)
    imgs['spatial']=plot_spatial(output,spatial)
    parts=[]
    parts.append(f'<header><p class="eyebrow">GAZE / PSM · FIXED-INSTRUMENT REVISION</p><h1>人工 gaze 与 PSM：左右器械分别分析</h1><p>GT 使用各 User 的 npy 眼动，仅保留坐标有限且在 [0,1] 内的点，不根据 valid 筛选；PSM 保留原生 DLC 有限坐标，主结果分别分析左右器械。</p><div class="cards"><div><b>{n_records}</b><span>录制</span></div><div><b>{n_users}</b><span>参与者</span></div><div><b>{len(finite)}</b><span>有可用峰的主分析组</span></div><div><b>{len(boundary)}</b><span>受搜索窗限制的组</span></div></div></header>')
    parts.append('<nav>'+''.join(f'<a href="#{id}">{name}</a>' for id,name in [('methods','方法与范围'),('position','Position 主结果'),('velocity','Velocity 补充'),('sensitivity','敏感性'),('support','共同支持'),('spatial','空间对应'),('reference','随机参考'),('files','文件与验证')])+'</nav>')
    parts.append('<section id="methods"><h2>方法与范围</h2><p>本报告覆盖人工眼动（GT）与左右 PSM 的关系。fixed_left、fixed_right 分别表示同一物理器械；不生成左右最大值的主结果，不把最大相关侧称为操作侧。GT 选择侧仅为补充分析。GT 与五模型的比较位于同次交付的 models10 子目录。</p>')
    parts.append('<div class="note"><b>解释边界：</b>正 lag 表示 gaze 的轨迹对应位置较早出现。相关峰高、正向峰位、空间接近是不同证据。这里的参与者重抽样范围是探索性百分位范围，不宣称经校准的 95% 置信覆盖，也不构成正式的“gaze 领先”检验。</div>')
    cfg=manifest.get('variants',{}).get('primary',{})
    parts.append(table(['部分','本次规则'],[
        ['输入','84 录制全量任务使用既有原生 DLC 坐标、时间戳和 stage；实际数量以上方卡片为准。'],
        ['DLC','保留全部有限坐标；likelihood、图像外点、跳变仅保存 QC 标记，不删点、不加权。'],
        ['Position',f"从未平滑的网格坐标进行 {cfg.get('position_window',3)} 点居中移动平均。"],
        ['Velocity',f"从未平滑坐标独立进行 {cfg.get('velocity_window',5)} 点二阶 Savitzky–Golay，再差分；stage 按区间中点。"],
        ['时间与缺口','41 Hz；仅在连续有效段内滤波；插值最大间隙 0.1 秒，不外推。普通重采样与跨非有限点插补分别标记。'],
        ['GT gaze 来源与有效性','每个 User 的 npy/*_gaze.npy 中 gaze_x / gaze_y；仅要求两坐标有限且均在 [0,1]（包含边界），不根据 valid 筛选。按原视频时间戳映射帧并重新生成 41 Hz 缓存。'],
        ['组曲线','主分析固定全 lag 有效录制；录制在 user 内等权 Fisher-z，user 在组内等权；至少 3 个 user。'],
        ['与 20261105 结果的差异','仅修正 GT gaze 数据源与有效性规则；录像、stage、时间轴、平滑、支持规则及重抽样配置保留。历史 CSV-GT 曲线作为差异审计，数值相等不再是通过条件；独立计算核验证同一新输入。'],
        ['阶段','PSM in stage 只限制 PSM 时间；Both in stage 限制双方属于同一 occurrence。它们的差不是可直接归因的“预视时长”。'],
        ['主搜索范围','±2 秒；±4 秒只作预定敏感性分支，不据结果选择搜索窗。']]))
    qrows=[r for r in preprocessing if r.get('variant')=='primary' and r.get('signal')=='position' and r.get('source')=='psm']
    if qrows:
        n_native=sum(num(r.get('native_samples')) for r in qrows)
        n_grid=sum(num(r.get('n_samples')) for r in qrows)
        qtable=[]
        for title,field in [('原生非有限坐标','native_nonfinite'),('图像外坐标（保留）','outside_frame_count'),
                            ('likelihood < 0.5（保留）','low_likelihood_count'),('孤立跳出返回候选（保留）','isolated_jump_candidate_count')]:
            count=sum(num(r.get(field,0)) for r in qrows)
            qtable.append([title,f'{count:,.0f}',fmt(count/n_native*100,2)+'%','原生点，左右侧分别计数'])
        for title,field in [('普通重采样','ordinary_resample_fraction'),('跨非有限点的短孔洞插补','filled_nonfinite_hole_fraction'),
                            ('质量删除导致的插补','filled_quality_hole_fraction'),('超过最大间隙而保持缺失','long_gap_grid_fraction')]:
            count=sum(num(r.get(field,0))*num(r.get('n_samples')) for r in qrows)
            qtable.append([title,f'{count:,.0f}',fmt(count/n_grid*100,2)+'%','41 Hz 网格，左右侧分别计数'])
        parts.append('<details><summary>查看逐点原因比例汇总</summary><p>标记可重叠，原生标记不应相加作为总删除率；质量标记均未用于剔除。网格统计来自重新构造的 PSM，而不是原始高频 gaze。</p>'+table(['原因 / 来源','点数','对应分母比例','分母'],qtable)+'</details>')
    parts.append('</section>')
    for signal,title in [('position','Position 主结果'),('velocity','Velocity 补充分析')]:
        rows=[r for r in primary if r['signal']==signal]
        parts.append(f'<section id="{signal}"><h2>{title}</h2><p>左右器械分别展示；两种阶段条件采用相同物理器械。圆点为窗口内峰，菱形为点估计触及边界。表格保留无支持阶段，不把缺失写成零相关。</p>')
        parts.append(figure(imgs[signal+'_lags'],'组峰位与参与者重抽样范围；范围跨零时方向不确定，触及窗口时无法充分定位。'))
        for stem in imgs[signal+'_curves']:parts.append(figure(stem,'两种阶段条件的完整相关曲线；阴影为逐 lag 探索性重抽样范围。'))
        if imgs[signal+'_bootstrap']:
            parts.append('<details><summary>查看完整组峰位重抽样分布</summary>')
            for stem in imgs[signal+'_bootstrap']:parts.append(figure(stem,'组峰位分布保留边界堆积和多峰；百分位区间不能替代分布形状。'))
            parts.append('</details>')
        parts.append('<details><summary>查看全部主结果数值</summary>'+group_table(rows)+'</details></section>')
    parts.append('<section id="sensitivity"><h2>平滑、边缘与搜索窗口敏感性</h2><p>Position：无平滑、3 点主分支、5 点。Velocity：3 点移动平均旧分支、SG5 主分支、SG7 和 SG11。另有连续有效段滤波边缘裁除与 ±4 秒窗口。每项都是预先列出的描述性分支，不根据哪个结果更显著挑选配置。图中为各分支峰位减去主分支，样本支持可能随分支变化，应结合 CSV 的 n 查看。</p>')
    for signal in ('position','velocity'):parts.append(figure(imgs[signal+'_sensitivity'],signal+'：PSM 时间锚定分析的峰位变化，单位秒。'))
    parts.append('<p>完整数值：<a href="group_summary.csv">group_summary.csv</a>。GT 选择侧补充结果保存在 <a href="gt_selected_group_summary.csv">gt_selected_group_summary.csv</a>；它仍有选择过程，不代替固定左右侧主分析。</p>')
    if selected:parts.append('<details><summary>GT 选择侧补充：position</summary>'+group_table([r for r in selected if r['signal']=='position'])+'</details>')
    parts.append('</section>')
    lookup={_key(r):r for r in primary};support_rows=[]
    for row in matched:
        original=lookup.get(_key(row),{})
        support_rows.append([row['task']+' / '+row['stage'],SIDE.get(row['side'],row['side']),row['signal'],row['mode'],
          original.get('n_stable_users','—')+' → '+row.get('n_stable_users','—'),original.get('n_complete_records','—')+' → '+row.get('n_complete_records','—'),
          fmt(original.get('peak_lag_s'))+' → '+fmt(row.get('peak_lag_s'))])
    parts.append('<section id="support"><h2>两种 mode 的共同支持对照</h2><p>只使用在两种 mode 的完整 lag 窗中都有效的同一录制集合，再按 user 聚合。这样固定器械、录制和 user 后检查 mode 差异。它是缩小后的条件子集，不能取代主结果；贡献范围减少也不是质量删除。</p>')
    parts.append('<details><summary>共同支持前后的用户、录制及峰位</summary>'+table(['阶段','器械','signal','mode','用户：原 → 共同','录制：原 → 共同','lag / 秒：原 → 共同'],support_rows)+'</details><p><a href="matched_mode_summary.csv">汇总 CSV</a> · <a href="matched_mode_coverage.csv">逐录制覆盖 CSV</a></p></section>')
    if contrasts:
        cr=[[r.get('task',''),SIDE.get(r.get('side'),r.get('side','')),r.get('signal','')+' / '+r.get('mode',''),
             r.get('condition_a','')+' → '+r.get('condition_b',''),r.get('n_users',''),fmt(r.get('difference_s')),
             '['+fmt(r.get('resample_low'))+', '+fmt(r.get('resample_high'))+']',fmt(r.get('boundary_frequency')),r.get('status','')] for r in contrasts]
        parts.append('<section><h2>阶段之间的配对比较</h2><p>各对阶段使用共同 user，双方同步重抽身份；差值为 B−A。边界峰以及重抽样边界频率保留。所有差值均为探索性诊断，不输出“已定位的生理时延差”，也不把跨零与否当作多重比较校正后的检验结论。</p><details><summary>查看阶段差值</summary>'+table(['任务','器械','signal / mode','阶段 A → B','共同用户','差值 / 秒','重抽样范围','边界频率','状态'],cr)+'</details><p><a href="stage_contrasts.csv">完整 CSV</a> · <a href="stage_contrast_bootstrap.npz">差值重抽样分布 NPZ</a></p></section>')
    parts.append('<section id="spatial"><h2>空间对应与跨边界配对</h2><p>距离是各录制中位距离经 user 分层平均；接近比例同样录制内、user 间等权。50 px 为探索性阈值，同时输出 25 和 100 px。指标不证明 gaze 注视器械，也不代替坐标注册核验。空间距离不改善与位置轨迹相关可以同时存在。</p>')
    parts.append(figure(imgs['spatial'],'零 lag 与共同组峰位处的空间距离及 50 px 接近比例。'))
    parts.append('<p><a href="spatial_summary.csv">空间汇总与跨边界比例</a> · <a href="spatial_records.csv">逐录制空间数据</a>。跨边界配对比例描述所用样本，不能直接视为边界造成的相关或领先比例。</p></section>')
    parts.append('<section id="reference"><h2>Unpaired reference：关联强度的描述性参照</h2><p>同 task、不同 user 的 gaze 与目标 PSM 按经过时间配对；固定左右侧分别计算。它比较最大相关强度，不检验 lag 是否为正，不是因果检验。完整录制和 user 集合固定；失败抽样单独记录，不通过补抽隐藏失败。持续时间覆盖子集另算其自身观测曲线和参考，不与全体观测混用。</p>')
    if reference:
        headers=['阶段','侧','signal / mode','参考范围','观测 r','有效 / 请求','尾部比例','描述性 BH','状态']
        rrows=[[r.get('task','')+' / '+r.get('stage',''),r.get('side',''),r.get('signal','')+' / '+r.get('mode',''),r.get('reference_scope',''),fmt(r.get('observed_peak_r')),
                r.get('n_valid','')+' / '+r.get('n_requested',''),fmt(r.get('reference_tail_fraction'),4),fmt(r.get('reference_bh_descriptive'),4),r.get('status','')] for r in reference]
        parts.append('<details><summary>查看参考比较结果及有效次数</summary>'+table(headers,rrows)+'</details><p><a href="unpaired_reference.csv">完整参考结果 CSV</a>。BH 值仅描述该参考下的多重比较调整，不能自动提供正式错误率保证。</p>')
    else:parts.append('<p class="note">本目录未发现 unpaired_reference.csv；本报告不补造参考 p 值。若这是 smoke run，参考分析可能被显式关闭。</p>')
    parts.append('</section><section id="files"><h2>文件、逐点来源与验证</h2>')
    regression=validation.get('legacy_regression',{})
    parts.append(table(['检查','结果'],[['输入文件未改变',str(validation.get('input_unchanged','未记录'))],['DLC 质量规则删除点',str(validation.get('dlc_quality_deleted','未记录'))],
        ['原生重建 / 支持审计全部通过',str(validation.get('all_support_audits_pass','未记录'))],['独立计算核验证（同一新输入）',str(validation.get('same_input_kernel_validation',{}).get('passed','未记录'))],
        ['独立计算核最大绝对误差',str(validation.get('same_input_kernel_validation',{}).get('maximum_error','未记录'))],['与历史 CSV-GT 曲线相等（此次不要求）',str(regression.get('passed','未记录'))],
        ['历史对照曲线数',str(regression.get('n','未记录'))],['历史对照最大绝对相关差值',str(regression.get('maximum_error','未记录'))],
        ['组 bootstrap 次数',str(manifest.get('bootstrap','未记录'))],['参考预定尝试数',str(manifest.get('null_attempts','未记录'))]]))
    files=[('group_summary.csv','各分支组峰位、峰高、范围、覆盖'),('group_curves.csv','完整组曲线和逐 lag 重抽样范围'),('record_summary.csv','逐录制峰形与 occurrence 重抽样诊断'),
           ('record_curves.jsonl','逐录制完整曲线'),('preprocessing_qc.csv','原生 QC、插补、滤波边缘汇总'),('support_audit.csv','时间轴和原生重建审计'),
           ('side_selection.csv','GT 选择侧补充规则及赢家'),('legacy_curve_regression.csv','旧 CSV-GT 与新 NPY-GT 单录制曲线差异审计（不要求相等）'),
           ('legacy_max_group_regression.csv','沿用旧 max 规则和旧聚合后的 GT 来源差异审计（不要求相等）'),('aggregation_support_comparison.csv','相同信号下仅改变录制支持规则的对照'),
           ('stage_contrasts.csv','共同 user 的阶段差值诊断'),('group_bootstrap_index.json','组峰位重抽样数组索引'),('group_bootstrap_peak_lags.npz','完整组峰位重抽样分布'),
           ('run_manifest.json','参数、软件版本与代码指纹'),('input_manifest.json','输入文件指纹'),('validation.json','运行验证')]
    files=[(name,meaning) for name,meaning in files if (output/name).exists()]
    parts.append(table(['文件','内容'],[[name,meaning] for name,meaning in files]))
    parts.append('<ul>'+''.join(f'<li><a href="{name}">{name}</a> — {esc(meaning)}</li>' for name,meaning in files if (output/name).exists())+'</ul>')
    parts.append('<p>逐点原因：<code>point_provenance/&lt;variant&gt;/&lt;content_id&gt;.npz</code>，含原生/网格/速度时间轴、原生 frame number、likelihood 三档标记、实际 dt 速度和孤立跳出返回候选、普通重采样/非有限孔洞插补/长缺口/滤波边缘，以及各 signal 的有效 mask。DLC 质量标记不作为删除指令。GT 已从各 user 的 NPY 按有限且归一化范围内规则重新生成；源文件指纹、接受点数及 valid 分类计数见 run_manifest.json 和 validation.json 的 input_policy，valid 分类只作审计。</p>')
    parts.append('<p>记录内 occurrence 重抽样结果：<code>bootstrap_peaks/</code>；充分统计量及索引：<code>occurrence_statistics/</code>；本次执行代码：<code>code_snapshot/</code>。所有图均有可独立使用的 PNG 和 PDF。</p></section>')
    css='''body{margin:0;background:#f4f7fa;color:#172335;font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:32px}header{background:#112c46;color:white;padding:38px;border-radius:14px}h1{font-size:30px;line-height:1.3;margin:10px 0}.eyebrow{letter-spacing:.15em;color:#add8ee;font-size:12px}.cards{display:flex;gap:16px;flex-wrap:wrap;margin-top:24px}.cards div{background:#20415f;border-radius:8px;padding:14px 20px;min-width:130px}.cards b{font-size:27px;display:block}.cards span{color:#c5d5e2;font-size:12px}nav{display:flex;flex-wrap:wrap;gap:14px;margin:22px 0}a{color:#126091;text-decoration:none}a:hover{text-decoration:underline}section{background:white;border:1px solid #e0e7ed;border-radius:12px;padding:28px;margin:20px 0}h2{font-size:23px;margin:0 0 14px;color:#173d5a}.note{background:#edf5fa;border-left:4px solid #3d8dbd;padding:15px;margin:16px 0}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{text-align:left;padding:10px 12px;border-bottom:1px solid #e1e8ef;vertical-align:top}th{background:#eaf0f5;white-space:nowrap}tr:nth-child(even){background:#f8fafc}figure{margin:24px 0}figure img{width:100%;height:auto;border:1px solid #e5e9ee}figcaption{font-size:12px;color:#57687a;margin-top:7px}summary{cursor:pointer;padding:12px;background:#edf2f7;font-weight:600;margin:12px 0}code{background:#edf2f7;padding:2px 4px;overflow-wrap:anywhere}footer{font-size:12px;color:#63778b;padding:12px}@media(max-width:760px){main{padding:12px}header,section{padding:18px}h1{font-size:24px}.cards div{min-width:100px}table{font-size:11px}}@media print{body{background:white}nav{display:none}main{padding:0}section{break-inside:avoid}details{display:block}}'''
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gaze/PSM 固定器械分析</title><style>'+css+'</style><main>'+''.join(parts)+'<footer>本报告是现有数据的探索性再分析。图表及文本均由本目录结果生成；未根据显著性筛选分析分支。</footer></main></html>'
    (output/'report.html').write_text(document,encoding='utf8')
    readme=f'''# Gaze/PSM 固定左右器械修正结果

打开 [report.html](report.html) 阅读中文离线报告。图位于 `figures/`，每张均含 PNG 与 PDF。

本目录：{n_records} 个录制、{n_users} 个 user；fixed_left / fixed_right 为主分析。GT 选择侧为补充，不命名为操作侧。

## 已确认规则

- DLC 全部有限坐标保留；likelihood、图像外点和跳变只标记，不删除、不加权。
- position：3 点居中移动平均；velocity：从未平滑输入独立做二阶 SG5 后差分。SG7/11 等为敏感性。
- GT 来自各 User 的 npy/*_gaze.npy 中 gaze_x / gaze_y；仅接受两坐标有限且均在 [0,1]（包含边界）的点，不根据 valid 筛选。原始 NPY、视频时间戳和接受点数见 run_manifest.json 的 input_policy。
- 41 Hz、最大插值间隙 0.1 秒；不外推、不跨长缺口滤波。本轮 gaze_raw 为按上述新 GT 规则重新生成的缓存。
- 全 lag 有效录制固定；录制在 user 内等权、user 在总体等权。共同支持 mode 对照单独保存。
- 重抽样范围为探索性百分位范围；参考尾部比例比较相关峰高，不是 lag 正向或因果检验。

## 主要文件

{chr(10).join('- ['+name+']('+name+')：'+meaning for name,meaning in files)}

- `matched_mode_*`：共同录制支持下的两种 mode 比较。
- `spatial_*`：零 lag / 组峰位处的距离、25/50/100 px 接近比例及跨边界配对比例。
- `unpaired_reference.csv`：若运行参考分支，记录请求/有效/失败次数、失败原因和参考尾部比例。
- `point_provenance/<variant>/<content_id>.npz`：逐点原生 QC、插补、滤波边缘及时间轴。
- `occurrence_statistics/`、`bootstrap_peaks/`：逐记录充分统计量和重抽样分布。
- `code_snapshot/`：本次执行源码；指纹见 run_manifest。

## 验证

输入未改变：{validation.get('input_unchanged','未记录')}；同一新输入的独立计算核验证：{validation.get('same_input_kernel_validation',{}).get('passed','未记录')}；独立核最大绝对误差：{validation.get('same_input_kernel_validation',{}).get('maximum_error','未记录')}。

历史 CSV-GT 对照相等：{regression.get('passed','未记录')}；对照曲线数：{regression.get('n','未记录')}；最大绝对相关差值：{regression.get('maximum_error','未记录')}。此次 GT 来源修正会导致差异；历史数值相等不是通过条件。

与 20261105 交付保留相同录像、分析参数、stage、支持规则和输出文件结构；此次仅改变 GT 来源和有效性条件。与更早旧聚合的差异仍在 aggregation_support_comparison.csv 单独保存。
'''
    (output/'README.md').write_text(readme,encoding='utf8')
    return {'report':str(output/'report.html'),'figures':[x.name for x in (output/'figures').glob('*.png')],
            'n_records':n_records,'n_users':n_users,'n_primary_supported_groups':len(finite)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    print(json.dumps(generate_report(parser.parse_args().output),ensure_ascii=False,indent=2))
